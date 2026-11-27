"""Measure shared-channel drift using the frozen MAR development suite."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_dataset import append_rows, digest_config, write_json
from scripts.eval_sdr006 import load_selected
from scripts.perturb_mar001 import directional_jvp
from scripts.run_mar001 import recurrent_step, rms, setup, trajectory
from scripts.train_sdr003 import read_json
from sdr_mamba.crowd_data import decode, tensor_image
from sdr_mamba.mar_metrics import Mass, background, decomposition, measure, retotal
from sdr_mamba.mar_readout import anchor_channel_mean, channel_parts, matched_centered_rms, readout_parts
from sdr_mamba.mar_readout import relative_rms, shapley_counts


def logits(model, feature):
    return model.head(feature).squeeze(-1)


def array(tensor):
    return tensor[0].detach().cpu().numpy().astype(np.float64)


def head_stats(model, mode):
    weight = model.head.weight.detach().double().flatten()
    norm = float(weight.norm())
    return {"mode": mode, "channels": weight.numel(), "bias": float(model.head.bias.item()),
            "weight_sum": float(weight.sum()), "weight_norm": norm,
            "centered_weight_norm": float((weight-weight.mean()).norm()),
            "cos_ones": float(weight.sum()) / (norm * weight.numel()**.5) if norm else None,
            "ln_normalized_shape": "x".join(str(n) for n in model.blocks[0].norm.normalized_shape),
            "direct_softplus_readout": True}


def summarize_logit(field):
    return {"mean": float(field.mean()), "mean_abs": float(np.abs(field).mean()),
            "rms": float(np.sqrt(np.square(field).mean())), "positive": float((field > 0).mean()),
            "negative": float((field < 0).mean()), "zero": float((field == 0).mean())}


def logit_rows(meta, m, z, prev_m, lm, lz, now, base, diff, density, truth, bg, config):
    for label, mask in (("all", np.ones_like(bg)), ("foreground", ~bg), ("background", bg)):
        if not mask.any():
            continue
        cur, ref = now[mask], base[mask]
        negative = cur <= config["softplus"]["negative_cutoff"]
        positive = cur >= config["softplus"]["positive_cutoff"]
        crossing = config["softplus"]["crossing"]
        mean_terms, centered_terms = summarize_logit(lm[mask]), summarize_logit(lz[mask])
        values = m[mask]
        update = (m-prev_m)[mask]
        yield {**meta, "region": label, "cells": int(mask.sum()), "mean_channels": float(values.mean()),
               "rms_mean_channels": float(np.sqrt(np.square(values).mean())), "std_mean_channels": float(values.std()),
               "rms_centered": float(np.sqrt(np.square(z[mask]).mean())),
               "mean_channel_update": float(update.mean()), "abs_channel_update": float(np.abs(update).mean()),
               **{f"mean_logit_delta_{k}": v for k, v in mean_terms.items()},
               **{f"centered_logit_delta_{k}": v for k, v in centered_terms.items()},
               "logit_split_max_abs_error": float(np.abs(diff[mask]).max()),
               "logit_positive_softplus_region": float(positive.mean()),
               "logit_negative_softplus_region": float(negative.mean()),
               "negative_exp_max_relative_error": float(np.max(
                   np.abs(density[mask][negative] / np.exp(cur[negative]) - 1))) if negative.any() else None,
               "positive_linear_max_relative_error": float(np.max(
                   np.abs(density[mask][positive] / cur[positive] - 1))) if positive.any() else None,
               "crossed_up_fraction": float(((ref <= crossing) & (cur > crossing)).mean()),
               "crossed_down_fraction": float(((ref > crossing) & (cur <= crossing)).mean()),
               "density_cell_mae": float(np.abs(density[mask]-truth[mask]).mean())}


def equivariance(model, ref, next_feature, meta, config, output, with_jvp):
    eq = config["equivariance"]
    token = hashlib.sha256(f"{eq['seed']}:{meta['id']}".encode()).digest()
    rng = torch.Generator().manual_seed(int.from_bytes(token[:8], "little"))
    field = torch.randn(ref.shape[:-1] + (1,), generator=rng).to(ref)
    field *= rms(ref)/rms(field)
    norm = model.blocks[0].norm
    ln = norm(ref)
    observations, derivatives = [], []
    for amp in eq["amplitudes"]:
        direction = field * amp
        shifted = ref + direction
        shifted_norm = norm(shifted)
        evolved = recurrent_step(model, shifted)
        normalized_error = relative_rms(shifted_norm-ln, ln)
        observations.append({**meta, "amp": amp, "ln_relative_rms": normalized_error,
                             "recurrent_relative_rms": relative_rms(evolved-next_feature-direction, direction)})
        if with_jvp:
            try:
                tangent, method = directional_jvp(lambda x: recurrent_step(model, x), ref, direction.expand_as(ref).contiguous())
                error = relative_rms(tangent-direction, direction)
                status = "PASS" if error <= config["tolerance"]["fp32_jvp_relative"] else "FAIL"
                derivatives.append({**meta, "amp": amp, "status": status, "method": method,
                                    "relative_rms": error, "reason": ""})
            except (RuntimeError, NotImplementedError) as failure:
                print(f"{meta['mode']} {meta['id']} amplitude {amp} JVP {type(failure).__name__}: {failure}", flush=True)
                derivatives.append({**meta, "amp": amp, "status": "NOT_RUN" if isinstance(failure, NotImplementedError) else "FAIL",
                                    "method": "", "relative_rms": None, "reason": type(failure).__name__})
    append_rows(output / "equivariance.csv", observations)
    if derivatives:
        append_rows(output / "jvp.csv", derivatives)


@torch.no_grad()
def evaluate(model, samples, config, metrics_config, model_config, output, deadline):
    mode = model.audit_mode
    ref_t = config["reference"]
    jvp_ids = sorted(entry["id"] for entry, _, _ in samples)[:config["equivariance"]["jvp_images"]]
    for index, (entry, file, truth) in enumerate(samples):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Registered MAR-001B compute limit reached")
        x = tensor_image(decode(file), model_config).unsqueeze(0).cuda()
        features, original = trajectory(model, x, config["maximum"])
        ref = features[ref_t]
        m4, z4 = channel_parts(ref)
        m0 = channel_parts(features[0])[0]
        l4 = logits(model, ref)
        lm4, lz4 = readout_parts(model.head, ref)
        p4 = Mass.from_cells(array(F.softplus(l4)))
        bg = background(truth.cells, metrics_config["background_radius_cells"])
        labels = {"mode": mode, "seed": metrics_config["seed"], "id": entry["id"]}
        equivariance(model, ref, features[ref_t+1], labels, config, output, entry["id"] in jvp_ids)
        anchored = ref
        prev_m = m0
        diagnostics, logs, counters, anchors = [], [], [], []
        for t, value in enumerate(features[1:], 1):
            meta = {**labels, "t": t, "train_depth": model.rounds, "extrapolation": t > model.rounds}
            m, z = channel_parts(value)
            lm, lz = readout_parts(model.head, value)
            observed_logit = logits(model, value)
            prediction = Mass.from_cells(array(original[t-1]))
            observed = measure(truth, prediction, bg, metrics_config)
            diagnostics.append({**meta, **observed})
            diff = (observed_logit-l4) - ((lm-lm4)+(lz-lz4))
            logs.extend(logit_rows(meta, array(m.squeeze(-1)), array(z), array(prev_m.squeeze(-1)), array(lm-lm4),
                                   array(lz-lz4), array(observed_logit), array(l4), array(diff),
                                   prediction.cells, truth.cells, bg, config))
            b, c = logits(model, z+m4), logits(model, z4+m)
            rms_aligned = matched_centered_rms(z, z4)
            variants = {"A_reference": p4.cells, "B_fixed_mean": array(F.softplus(b)),
                        "C_fixed_centered": array(F.softplus(c)), "D_observed": prediction.cells,
                        "fixed_mean_t0": array(F.softplus(logits(model, z+m0))),
                        "fixed_count": retotal(prediction, p4.count),
                        "matched_mean_and_rms": array(F.softplus(logits(model, rms_aligned+m4))) if rms_aligned is not None else None}
            if t >= ref_t:
                previous = anchored
                if t > ref_t:
                    next_raw = recurrent_step(model, anchored)
                    anchored = anchor_channel_mean(next_raw, m4)
                    norm = model.blocks[0].norm
                    branch_relative = relative_rms(
                        (next_raw-previous)-(value-features[t-1]), value-features[t-1])
                    norm_relative = relative_rms(norm(previous)-norm(features[t-1]), norm(features[t-1]))
                else:
                    branch_relative = norm_relative = 0.0
                anchor_logit = logits(model, anchored)
                variants["trajectory_anchor"] = array(F.softplus(anchor_logit))
                anchor_count = variants["trajectory_anchor"].sum()
                b_count = variants["B_fixed_mean"].sum()
                anchors.append({**meta, "centered_relative_rms": relative_rms(channel_parts(anchored)[1]-z, z),
                                "ln_input_relative_rms": norm_relative, "branch_update_relative_rms": branch_relative,
                                "max_logit_abs": float((anchor_logit-b).abs().max()),
                                "count_relative": float(abs(anchor_count-b_count)/max(b_count, 1e-12))})
            a_count, b_count, c_count, d_count = (
                float(variants[v].sum()) for v in ("A_reference", "B_fixed_mean", "C_fixed_centered", "D_observed"))
            mean, centered = shapley_counts(a_count, b_count, c_count, d_count)
            count_delta = d_count-a_count
            for label, cells in variants.items():
                present = cells is not None
                pred = Mass.from_cells(cells) if present else prediction
                metrics = measure(truth, pred, bg, metrics_config)
                terms = decomposition(truth, pred, metrics_config["epsilon"])
                evidence = {**metrics, **{k: terms[k] for k in ("count_kl", "shape_kl", "total_kl")}}
                counters.append({**meta, "variant": label, "status": "DEFINED" if present else "UNDEFINED",
                                 **(evidence if present else dict.fromkeys(evidence)),
                                 "count_delta": count_delta, "mean_shapley": mean, "centered_shapley": centered,
                                 "shapley_residual": mean+centered-count_delta})
            prev_m = m
        for name, records in (("per_image_depth", diagnostics), ("logit_decomposition", logs),
                              ("counterfactual", counters), ("anchor_equivalence", anchors)):
            append_rows(output / f"{name}.csv", records)
        if (index+1) % 15 == 0:
            print(f"{mode} evaluated {index+1}/{len(samples)} frozen development images", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_json(Path("configs/mar001b.json"))
    base_cfg, model_cfg, samples, selection, checkpoints = setup(args.output, "readout")
    if cfg["reference"] != base_cfg["depths"]["reference"] or cfg["models"] != base_cfg["models"]:
        raise ValueError("MAR-001B conflicts with its inherited dataset/model scope")
    historical = Path("reports/MAR-001")
    for name in ("data_manifest", "checkpoint_manifest"):
        path = args.output / f"{name}.json"
        if read_json(path) != read_json(historical / path.name):
            raise ValueError(f"Historical {name} mismatch")
    provenance = args.output / "readout_environment.json"
    write_json(provenance, {**read_json(provenance), "readout_config_sha256": digest_config(cfg), "base_commit": cfg["base_commit"]})
    deadline = time.perf_counter() + cfg["gpu_seconds_cap"]
    for name in cfg["models"]:
        model = load_selected(checkpoints, name, base_cfg["seed"], model_cfg, selection[name]).requires_grad_(False)
        append_rows(args.output / "head_stats.csv", [head_stats(model, name)])
        torch.cuda.reset_peak_memory_stats()
        start, status = time.perf_counter(), "FAIL"
        try:
            evaluate(model, samples, cfg, base_cfg, model_cfg, args.output, deadline)
            status = "PASS"
        finally:
            append_rows(args.output / "resource_log.csv", [{"mode": name, "status": status, "seconds": time.perf_counter()-start,
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated()}])


if __name__ == "__main__":
    main()
