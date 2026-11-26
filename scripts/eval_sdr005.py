"""Paired validation and locked official-test evaluation for SDR-005."""

import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
from statistics import mean, median, stdev
import time

import torch

from scripts.audit_dataset import append_rows, write_json
from scripts.train_sdr003 import read_json
from scripts.train_sdr005 import checkpoint, image_stats, make_model, reduce_metrics, setup
from sdr_mamba.crowd_data import decode, mass_map, tensor_image

DISTANCE = ((0, 7), (8, 15), (16, 31), (32, 63), (64, None))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def intervention(model, density, frames, image, trace, dataset_meta):
    prev, current = frames[2], frames[3]
    variants = [
        (label, model.step(current.features_in, prev.bank, mode=mode).density[0])
        for label, mode in (("zero_current", "C1"), ("same_direction", "C2"), ("wrong_line", "C4"))
    ]
    variants.append(("full_trajectory_reset", model(image, 4, mode="C1")[-1].density[0]))
    for label, altered in variants:
        delta = density - altered
        trace.append({
            **dataset_meta, "variant": label, "signed_count_delta": delta.sum().item(),
            "abs_count_delta": abs(delta.sum().item()), "density_l1": delta.abs().sum().item(),
        })


def update_diagnostic(store, frames):
    for t in (2, 4):
        for group, shape, probes in frames[t - 1].probes[1]:
            for start, retention, contribution, output in probes:
                steps = torch.arange(start, start + retention.shape[1], device=retention.device)
                for lo, hi in DISTANCE:
                    valid = (steps >= lo) & ((steps <= hi) if hi is not None else torch.ones_like(steps, dtype=torch.bool))
                    if not valid.any():
                        continue
                    group_shape = (len(group), shape[1] * shape[2])
                    states = retention[:, valid].unflatten(0, group_shape)
                    y = output[:, valid].unflatten(0, group_shape)
                    h = None if contribution is None else contribution[:, valid].unflatten(0, group_shape)
                    for idx, direction in enumerate(group):
                        vals = store[t, direction, f"{lo}-{hi}" if hi is not None else f"{lo}+"]
                        vals[0] += states[idx].sum().item()
                        vals[1] += states[idx].numel()
                        vals[2] += 0 if h is None else h[idx].square().sum().item()
                        vals[3] += y[idx].square().sum().item()
                        vals[4] += int((y[idx].square().sum((-1, -2)) == 0).sum())


@torch.no_grad()
def evaluate(mode, seed, phase, samples, config, device, output, ledger):
    stage = "screen" if seed == 0 else "pilot"
    artifact = checkpoint(output, stage, mode, seed)
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != ledger[stage, mode, seed]:
        raise ValueError(f"Checkpoint SHA mismatch {stage} {mode} {seed}")
    model = make_model(mode, config, device, seed).eval()
    model.load_state_dict(torch.load(artifact, map_location=device, weights_only=True))
    rounds = config["evaluation_rounds"] if mode not in ("C0", "C6") else ([1] if mode == "C0" else [1, 2, 4])
    train_depth = 0 if mode == "C0" else config["model"]["rounds"]
    records = defaultdict(list)
    trace, times = [], []
    distance = defaultdict(lambda: [0., 0, 0., 0., 0])
    meta = {"split": phase, "mode": mode, "seed": seed}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for sample in samples:
        image = tensor_image(decode(sample.file), config).unsqueeze(0).to(device)
        target = mass_map(sample.points, sample.height, sample.width, config["stride"])
        sync(device)
        start = time.perf_counter()
        frames = model(image, max(rounds), diagnose=mode == "C3")
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
        for t in rounds:
            density = frames[t - 1].density[0].cpu()
            if not torch.isfinite(density).all():
                raise FloatingPointError(f"Nonfinite {phase} {mode} T{t}")
            records[t].append({**meta, "t": t, "id": sample.identifier, **image_stats(density, target)})
        if mode == "C3":
            intervention(model, frames[3].density[0], frames, image, trace, {**meta, "id": sample.identifier})
            update_diagnostic(distance, frames)
    append_rows(output / "per_image.csv", [row for r in records.values() for row in r])
    results = []
    bins = [*config["count_bins"], None]
    for t, rr in records.items():
        common = {**meta, "t": t, "train_depth": train_depth, "extrapolation": t > train_depth and mode != "C0"}
        results.append({**common, **reduce_metrics(rr)})
        ranges = []
        for lo, hi in zip(bins, bins[1:]):
            group = [r for r in rr if r["gt"] >= lo and (hi is None or r["gt"] < hi)]
            if group:
                ranges.append({**common, "gt_min": lo, "gt_max_exclusive": hi, **reduce_metrics(group)})
        append_rows(output / "density_bins.csv", ranges)
    append_rows(output / "depth_metrics.csv", results)
    append_rows(output / "intervention.csv", trace)
    append_rows(output / "retention.csv", [
        {**meta, "round": t, "direction": d, "token_distance": bin_, "retention_mean": s / n,
         "initial_to_output_l2": (p / q) ** .5 if q else None, "zero_readouts": zero}
        for (t, d, bin_), (s, n, p, q, zero) in distance.items() if n
    ])
    append_rows(output / "latency.csv", [
        {**meta, "profile_T": max(rounds), "median_ms_per_image": median(times),
         "image_count": len(times), "with_retention_instrumentation": mode == "C3",
         "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None},
    ])
    formal_t = 1 if mode == "C0" else 4
    primary = next(r for r in results if r["t"] == formal_t)
    print(f"{phase} {mode} seed={seed} T{formal_t} MAE {primary['mae']:.4f} GAME1 {primary['game1']:.4f}", flush=True)
    return primary, trace


def relative(a, b):
    return (b - a) / b if b else (0. if a == 0 else float("-inf"))


def screen_gate(config, primary, trace, output):
    limits = config["gate"]
    metrics = {r["mode"]: r for r in primary}
    c3 = metrics["C3"]
    deltas = {m: {k: relative(c3[k], metrics[m][k]) for k in ("mae", "game1")}
              for m in ("C1", "C2", "C4", "C5", "C6")}
    curves = read_csv(output / "train_curves.csv")
    converged = {}
    for m in config["train_modes"]:
        vals = [float(r["mean_train_loss"]) for r in curves
                if r["mode"] == m and r["seed"] == "0" and r["mean_train_loss"]]
        converged[m] = (vals[0] - mean(vals[-2:])) / vals[0]
    interventions = {}
    for variant in sorted({r["variant"] for r in trace}):
        subset = [r for r in trace if r["variant"] == variant]
        interventions[variant] = {key: mean(r[key] for r in subset)
                                  for key in ("abs_count_delta", "density_l1")}
    checks = {
        "C1_within_C0_margin": metrics["C1"]["mae"] <= metrics["C0"]["mae"] * (1 + limits["c1_near_c0"]),
        "loss_reduction_all_modes": all(v >= limits["min_loss_reduction"] for v in converged.values()),
        "C3_mae_gain_each_core": all(deltas[m]["mae"] >= limits["screen_rel_mae_gain"] for m in ("C1", "C2", "C4")),
        "C3_game1_margin_each_core": all(deltas[m]["game1"] >= -limits["screen_max_game1_regression"] for m in ("C1", "C2", "C4")),
        "intervention_current_effect": any(interventions[v]["abs_count_delta"] >= limits["intervention_mean_abs_count"]
            and interventions[v]["density_l1"] >= limits["intervention_density_l1"]
            for v in ("zero_current", "same_direction", "wrong_line")),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "official_test": "PENDING" if all(checks.values()) else "NOT_RUN",
            "checks": checks, "paired_relative_improvement": deltas,
            "initial_vs_final_train_loss_reduction": converged, "interventions": interventions,
            "primary": metrics}


def formal_gate(config, primary):
    lookup = {(r["mode"], r["seed"]): r for r in primary}
    checks, contrasts = {}, {}
    limits = config["gate"]
    for mode in ("C1", "C2", "C4", "C5", "C6"):
        paired = {k: [relative(lookup["C3", s][k], lookup[mode, s][k]) for s in config["train_seeds"]]
                  for k in ("mae", "rmse", "game1")}
        contrasts[mode] = {k: {"mean": mean(v), "sample_sd": stdev(v),
                               "seed_values": v, "positive_seeds": sum(x > 0 for x in v)}
                           for k, v in paired.items()}
        if mode in ("C1", "C2", "C4"):
            checks[mode + "_mae"] = (mean(paired["mae"]) >= limits["test_rel_mae_gain"]
                                     and sum(x > 0 for x in paired["mae"]) >= limits["min_seed_wins"])
            checks[mode + "_rmse_game1"] = all(mean(paired[k]) >= -limits["test_max_rmse_or_game1_regression"]
                                               for k in ("rmse", "game1"))
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "comparisons": contrasts}


def bootstrap(config, phase, output):
    data = read_csv(output / "per_image.csv")
    seeds = config["train_seeds"] if phase == "test" else [0]
    joined = defaultdict(dict)
    for r in data:
        if r["split"] == phase and int(r["t"]) == (1 if r["mode"] == "C0" else 4):
            joined[r["mode"], int(r["seed"])][r["id"]] = float(r["absolute"])
    ids = sorted(joined["C3", 0])
    generator = torch.Generator().manual_seed(config["bootstrap_seed"])
    indices = torch.randint(len(ids), (config["bootstrap_draws"], len(ids)), generator=generator)
    rows = []
    for mode in ("C0", "C1", "C2", "C4", "C5", "C6"):
        delta = torch.tensor([[joined["C3", seed][i] - joined[mode, seed][i] for i in ids] for seed in seeds])
        distribution = delta[:, indices].mean((0, 2))
        interval = torch.quantile(distribution, torch.tensor([.025, .975]))
        values = delta.mean(1).tolist()
        rows.append({"split": phase, "comparison": mode, "mae_C3_minus_control": mean(values),
                     "seed_sd": stdev(values) if len(values) > 1 else None,
                     "ci95_low": interval[0].item(), "ci95_high": interval[1].item(),
                     "images": len(ids), "draws": len(indices)})
    append_rows(output / "paired_bootstrap.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.phase == "test" and read_json(args.output / "validation_gate.json")["status"] != "PASS":
        print("Official test NOT_RUN because screening did not pass", flush=True)
        return 0
    marker = args.output / f"{args.phase}_evaluation_started.json"
    if marker.exists():
        raise FileExistsError("Evaluation already attempted; preserve original output")
    cfg, data, prov = setup(args.output, args.device)
    ledger = {(r["stage"], r["mode"], int(r["seed"])): r["checkpoint_sha256"]
              for r in read_csv(args.output / "runs.csv") if r["status"] == "PASS"}
    seeds = cfg["train_seeds"] if args.phase == "test" else [0]
    for seed in seeds:
        for mode in cfg["train_modes"]:
            if ("screen" if seed == 0 else "pilot", mode, seed) not in ledger:
                raise RuntimeError(f"Checkpoint ledger incomplete for {mode} seed {seed}")
    write_json(marker, prov)
    primary, trace = [], []
    for mode in cfg["train_modes"]:
        for seed in seeds:
            main_result, changes = evaluate(mode, seed, args.phase, data[args.phase], cfg, args.device, args.output, ledger)
            primary.append(main_result)
            trace.extend(changes)
    append_rows(args.output / ("validation_metrics.csv" if args.phase == "validation" else "metrics.csv"), primary)
    bootstrap(cfg, args.phase, args.output)
    decision = (screen_gate(cfg, primary, trace, args.output) if args.phase == "validation"
                else formal_gate(cfg, primary))
    write_json(args.output / ("validation_gate.json" if args.phase == "validation" else "test_gate.json"), decision)
    print({k: v for k, v in decision.items() if k != "primary"}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
