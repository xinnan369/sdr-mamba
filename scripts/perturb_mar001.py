"""Same-RMS hidden interventions and local directional derivatives for MAR-001."""

import argparse
import hashlib
import math
from pathlib import Path
import time

import torch

from scripts.audit_dataset import append_rows, write_json
from scripts.eval_sdr005 import read_csv
from scripts.eval_sdr006 import load_selected
from scripts.run_mar001 import density, recurrent_step, rms, setup, trajectory
from sdr_mamba.crowd_data import decode, tensor_image


def stratified_samples(samples, size):
    ranked = sorted(samples, key=lambda r: (r[2].count, r[0]["id"]))
    return [ranked[(2*i+1)*len(ranked)//(2*size)] for i in range(size)]


def directions(feature, update, identifier, spec):
    seed = int.from_bytes(hashlib.sha256(f"{spec['seed']}:{identifier}".encode()).digest()[:8], "little")
    rng = torch.Generator().manual_seed(seed)
    _, height, width, channels = feature.shape
    ph, pw = (max(2, int(n*spec["local_fraction"])) for n in (height, width))
    result = [("radial", 0, feature), ("radial", 1, -feature),
              ("residual", 0, update), ("residual", 1, -update)]
    for index in range(spec["draws"]):
        top = int(torch.randint(height-ph+1, (), generator=rng))
        left = int(torch.randint(width-pw+1, (), generator=rng))
        noise = torch.randn((1, ph, pw, channels), generator=rng, dtype=feature.dtype)
        patch = noise - noise.mean((1, 2), keepdim=True)
        local = torch.zeros(feature.shape, dtype=feature.dtype)
        local[:, top:top+ph, left:left+pw] = patch
        common = torch.randn((1, height, width, 1), generator=rng, dtype=feature.dtype)
        result.extend([("local_zero_spatial_mean", index, local.to(feature.device)),
                       ("common_channel_shift", index, common.expand(feature.shape).to(feature.device))])
    return [(kind, draw, value/rms(value)) for kind, draw, value in result]


def evolve(model, feature, steps):
    for _ in range(steps):
        feature = recurrent_step(model, feature)
    return feature


@torch.no_grad()
def intervention(model, frames, predictions, direction, amp, horizons):
    reference = frames[0]
    isolated = reference + direction * (amp*rms(reference))
    actual = rms(isolated-reference)
    if not math.isfinite(actual) or actual == 0:
        raise FloatingPointError("Hidden perturbation amplitude is not finite and positive")
    result = []
    for k in horizons:
        if k:
            isolated = recurrent_step(model, isolated)
        pred = density(model, isolated)
        delta = pred - predictions[k]
        error = rms(isolated-frames[k])
        if not math.isfinite(error) or not torch.isfinite(pred).all():
            raise FloatingPointError("Non-finite perturbed trajectory")
        result.append({"horizon": k, "rms_initial": actual, "rms_delta": error,
                       "feature_gain": error/actual, "relative_trajectory_offset": error/rms(frames[k]),
                       "signed_count_delta": float(delta.double().sum()), "density_l1": float(delta.double().abs().sum())})
    return result


def directional_jvp(fn, feature, direction):
    try:
        _, tangent = torch.func.jvp(fn, (feature,), (direction,))
        return tangent, "torch.func.jvp"
    except (NotImplementedError, RuntimeError) as error:
        print(f"Forward AD failed ({type(error).__name__}): {error}", flush=True)
        _, tangent = torch.autograd.functional.jvp(fn, feature, direction, strict=True)
        return tangent, "reverse_over_reverse"


def derivative(model, feature, direction, amp, horizon):
    def fn(value):
        changed = evolve(model, value, horizon)
        return changed, density(model, changed)

    tangent, method = directional_jvp(fn, feature, direction)
    continuous, output = tangent
    step = amp*rms(feature)
    with torch.no_grad():
        plus_feature, plus_density = fn(feature + step*direction)
        minus_feature, minus_density = fn(feature - step*direction)
        feature_fd = (plus_feature-minus_feature)/(2*step)
        density_fd = (plus_density-minus_density)/(2*step)
    scale = rms(direction)
    jvp_rms = rms(continuous)
    out_rms = rms(output)
    if not all(math.isfinite(v) for v in (jvp_rms, out_rms)):
        raise FloatingPointError("Nonfinite JVP")
    return {"method": method, "feature_gain": jvp_rms/scale, "count_derivative": float(output.double().sum()),
            "density_derivative_rms": out_rms, "fd_feature_gain": rms(feature_fd)/scale,
            "fd_count_derivative": float(density_fd.double().sum()),
            "fd_feature_relative_error": rms(feature_fd-continuous)/max(jvp_rms, 1e-12),
            "fd_density_relative_error": rms(density_fd-output)/max(out_rms, 1e-12)}


def evaluate(model, samples, config, model_config, output, deadline):
    reference = config["depths"]["reference"]
    spec = config["perturbation"]
    variants = spec["relative_rms"]
    horizons = spec["horizons"]
    mode = model.audit_mode
    for index, (entry, file, _) in enumerate(samples):
        if time.perf_counter() >= deadline:
            raise TimeoutError("MAR-001 GPU budget reached")
        identifier = entry["id"]
        x = tensor_image(decode(file), model_config).unsqueeze(0).cuda()
        with torch.no_grad():
            frames, predictions = trajectory(model, x, reference + max(horizons))
            window = frames[reference:]
            predictions = predictions[reference-1:]
            vectors = directions(window[0], window[1]-window[0], identifier, spec)
        base = {"mode": mode, "seed": config["seed"], "id": identifier, "start_t": reference}
        rows = []
        for kind, draw, vector in vectors:
            for amp in variants:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("MAR-001 GPU budget reached during interventions")
                for row in intervention(model, window, predictions, vector, amp, horizons):
                    rows.append({**base, "direction": kind, "draw": draw, "relative_rms": amp, **row})
        append_rows(output / "perturbation.csv", rows)
        if index < spec["jvp_images"]:
            jvp_rows = []
            for kind, draw, vector in vectors:
                if draw == 0 and kind in ("radial", "local_zero_spatial_mean"):
                    for horizon in (1, 2):
                        if time.perf_counter() >= deadline:
                            raise TimeoutError("MAR-001 GPU budget reached during JVP")
                        try:
                            row = derivative(model, window[0], vector, min(variants), horizon)
                            status, error = "PASS", ""
                        except (NotImplementedError, RuntimeError, FloatingPointError) as exception:
                            print(f"{mode} {identifier} {kind} T+{horizon} JVP: {exception}", flush=True)
                            status = "NOT_RUN" if isinstance(exception, NotImplementedError) else "FAIL"
                            error = type(exception).__name__
                            row = dict.fromkeys(("method", "feature_gain", "count_derivative", "density_derivative_rms",
                                                 "fd_feature_gain", "fd_count_derivative",
                                                 "fd_feature_relative_error", "fd_density_relative_error"))
                        jvp_rows.append({**base, "direction": kind, "draw": draw, "horizon": horizon,
                                         "status": status, "error": error, **row})
            append_rows(output / "jvp.csv", jvp_rows)
        print(f"{mode} completed fixed-state interventions on {index+1}/{len(samples)} images", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, model_config, samples, selected, checkpoint_home = setup(args.output, "perturbation")
    spent = sum(float(r["seconds"]) for r in read_csv(args.output / "resource_log.csv"))
    deadline = time.perf_counter() + config["gpu_seconds_cap"] - spent
    samples = stratified_samples(samples, config["perturbation"]["images"])
    write_json(args.output / "perturbation_manifest.json", {
        "images": [s[0]["id"] for s in samples],
        "jvp_images": [s[0]["id"] for s in samples[:config["perturbation"]["jvp_images"]]],
    })
    for mode in config["models"]:
        status = "FAIL"
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        try:
            model = load_selected(checkpoint_home, mode, config["seed"], model_config, selected[mode])
            model.requires_grad_(False)
            evaluate(model, samples, config, model_config, args.output, deadline)
            status = "PASS"
        finally:
            append_rows(args.output / "resource_log.csv", [{"phase": "perturbation", "model": mode, "status": status,
                        "seconds": time.perf_counter() - begin, "peak_bytes": torch.cuda.max_memory_allocated()}])


if __name__ == "__main__":
    main()
