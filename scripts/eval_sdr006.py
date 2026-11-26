"""Diagnostics and once-only evaluation of the locked internal holdout."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean, median, stdev
import time

import torch

from scripts.audit_dataset import append_rows, write_json
from scripts.audit_sdr006 import raw_points
from scripts.eval_sdr005 import read_csv
from scripts.train_sdr003 import read_json
from scripts.train_sdr005 import image_stats, reduce_metrics
from scripts.train_sdr006 import artifact, make_model, setup
from sdr_mamba.crowd_audit import project_points
from sdr_mamba.crowd_data import decode, in_rect, mass_map, sha_file, tensor_image


def depths(mode):
    return {"F0": (1,), "F2": (1, 2, 4)}.get(mode, (1, 2, 4, 6, 8))


def load_selected(output, mode, seed, config, selected):
    path = artifact(output, mode, seed, int(selected["step"]))
    if sha_file(path) != selected["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint integrity mismatch for {mode} seed {seed}")
    model = make_model(mode, seed, config)
    model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
    return model.eval()


def tile_density(model, x, rounds, size=256):
    rows, columns = x.shape[-2] // size, x.shape[-1] // size
    patches = [x[..., i*size:(i+1)*size, j*size:(j+1)*size]
               for i in range(rows) for j in range(columns)]
    if not rows or not columns or any(d % size for d in x.shape[-2:]):
        raise ValueError("Diagnostics require exact nonoverlapping tiles")
    density = model(torch.cat(patches), rounds)[-1]
    return torch.cat([torch.cat(list(row), dim=1) for row in density.split(columns)], dim=0)


def sensitivity(sample, pred, config):
    raw = raw_points(sample)
    args = (sample.height, sample.width, config["stride"])
    options = [
        ("raw0_project", sample.points),
        ("raw1_project", project_points(raw, sample.width, sample.height, 1)),
        ("raw0_drop", raw[in_rect(raw, sample.width, sample.height)]),
        ("raw1_drop", (raw - 1)[in_rect(raw - 1, sample.width, sample.height)]),
    ]
    return [{"protocol": name, **image_stats(pred, mass_map(points, *args))} for name, points in options]


def matched_profile(model, samples, rounds, config):
    inputs = [tensor_image(decode(s.file), config).unsqueeze(0).cuda() for s in samples[:5]]
    for _ in range(2):
        model(inputs[0], rounds)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall = time.perf_counter()
    times = []
    for x in inputs:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model(x, rounds)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {"median_ms": median(times), "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "images": len(times), "timed_loop_seconds": time.perf_counter() - wall}


@torch.inference_mode()
def evaluate(output, split, mode, seed, selected, samples, config):
    model = load_selected(output, mode, seed, config, selected)
    depth = config["modes"][mode]
    primary_depth = max(1, depth)
    depth_values = depths(mode) if split == "development" else tuple(sorted(set((primary_depth, *depths(mode)))))
    key = {"split": split, "mode": mode, "seed": seed, "checkpoint_step": int(selected["step"])}
    distributed, tiled, protocols = defaultdict(list), [], defaultdict(list)
    for sample in samples:
        x = tensor_image(decode(sample.file), config).unsqueeze(0).cuda()
        target = mass_map(sample.points, sample.height, sample.width, config["stride"])
        frames = model(x, max(depth_values))
        for t in depth_values:
            pred = frames[t-1][0].cpu()
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"Nonfinite diagnostic prediction for {mode} seed {seed} T{t}")
            distributed[t].append({"id": sample.identifier, **image_stats(pred, target)})
        if split == "development":
            primary = frames[primary_depth-1][0].cpu()
            tile = tile_density(model, x, primary_depth).cpu()
            tile_stats = image_stats(tile, target)
            tiled.append({**key, "id": sample.identifier, "gt": tile_stats["gt"], "full_count": primary.sum().item(),
                          "tile_count": tile_stats["count"], "tile_absolute": tile_stats["absolute"],
                          "tile_game1": tile_stats["game1"], "tile_minus_full": (tile - primary).sum().item(),
                          "density_l1": (tile - primary).abs().sum().item()})
            for variant in sensitivity(sample, primary, config):
                protocols[variant["protocol"]].append(variant)
    totals = []
    bins = [*config["count_bins"], math.inf]
    for t, rows in distributed.items():
        meta = {**key, "t": t, "train_depth": depth, "extrapolation": t > depth > 0}
        append_rows(output / "per_image.csv", [{**meta, **r} for r in rows])
        totals.append({**meta, **reduce_metrics(rows)})
        for lo, hi in zip(bins, bins[1:]):
            group = [row for row in rows if lo <= row["gt"] < hi]
            if group:
                append_rows(output / "density_bins.csv", [
                    {**meta, "gt_lo": lo, "gt_hi_exclusive": hi, **reduce_metrics(group)}
                ])
    append_rows(output / "depth_metrics.csv", totals)
    result = next(row for row in totals if row["t"] == primary_depth)
    append_rows(output / "metrics.csv", [result])
    if split == "development":
        append_rows(output / "tiles_per_image.csv", tiled)
        append_rows(output / "count_protocol_sensitivity.csv", [
            {**key, "protocol": p, **reduce_metrics(rows)} for p, rows in protocols.items()
        ])
        append_rows(output / "latency.csv", [
            {**key, "t": primary_depth, **matched_profile(model, sorted(samples, key=lambda s: s.identifier),
                                                         primary_depth, config)}
        ])
    print(result, flush=True)
    return result


def final_gate(primary, config):
    by_key = {(row["mode"], row["seed"]): row for row in primary}
    contrasts, checks = {}, {}
    margin = config["margin"]
    for mode in ("F0", "F2", "F3"):
        variants = {k: [(by_key["F1", s][k] - by_key[mode, s][k]) / max(by_key[mode, s][k], 1e-12)
                        for s in config["seeds"]] for k in ("mae", "rmse", "game1")}
        contrasts[mode] = {k: {"seeds": v, "mean": mean(v), "sd": stdev(v)} for k, v in variants.items()}
        mae = variants["mae"]
        checks[f"{mode}_mae"] = (mean(mae) <= -margin and sum(x < 0 for x in mae) >= 2 if mode == "F0"
                                  else mean(mae) <= margin and (mode != "F2" or sum(x <= margin for x in mae) >= 2))
        checks[f"{mode}_rmse_game1"] = all(mean(variants[k]) <= margin for k in ("rmse", "game1"))
    mix1 = sum(p.numel() for p in make_model("F1", 0, config, "cpu").blocks.parameters())
    mix2 = sum(p.numel() for p in make_model("F2", 0, config, "cpu").blocks.parameters())
    checks["F1_mixer_at_most_half_F2"] = mix1 <= mix2 / 2
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "official_test": "NOT_RUN", "relative_F1_minus_control": contrasts}


def bootstrap(output, split, config):
    joined = defaultdict(dict)
    for row in read_csv(output / "per_image.csv"):
        if row["split"] == split and int(row["t"]) == max(1, config["modes"][row["mode"]]):
            joined[row["mode"], int(row["seed"])][row["id"]] = float(row["absolute"])
    ids = sorted(joined["F1", 0])
    generator = torch.Generator().manual_seed(config["bootstrap_seed"])
    indices = torch.randint(len(ids), (config["bootstrap_draws"], len(ids)), generator=generator)
    seeds = [0] if split == "development" else config["seeds"]
    for mode in ("F0", "F2", "F3", "F4"):
        delta = torch.tensor([[joined["F1", s][i] - joined[mode, s][i] for i in ids] for s in seeds],
                             dtype=torch.float64)
        draws = delta[:, indices].mean((0, 2))
        low, high = torch.quantile(draws, torch.tensor([.025, .975], dtype=draws.dtype)).tolist()
        append_rows(output / "paired_bootstrap.csv", [{
            "split": split, "comparison": mode, "F1_minus_control": delta.mean().item(),
            "seed_sd": delta.mean(1).std().item() if len(seeds) > 1 else None,
            "images": len(ids), "ci95_low": low, "ci95_high": high, "draws": len(indices),
        }])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    output, split = args.output, args.split
    gate = read_json(output / "gate.json")
    if split == "confirmation" and gate["status"] != "PASS":
        print("Internal confirmation NOT_RUN because the development gate failed", flush=True)
        return 0
    config, data, env = setup(output)
    selections = {(r["mode"], int(r["seed"])): r for r in read_csv(output / "checkpoint_selection.csv")}
    seeds = [0] if split == "development" else config["seeds"]
    matrix = [(m, seed) for seed in seeds for m in config["modes"]]
    if any(key not in selections for key in matrix):
        raise ValueError("Required selected checkpoints are incomplete")
    marker = output / f"{split}_started.json"
    if marker.exists():
        raise FileExistsError(f"{split} evaluation was already attempted")
    write_json(marker, env)
    budget = sum(float(r["seconds"]) for r in read_csv(output / "runs.csv"))
    budget += read_json(output / "sanity_gate.json")["seconds"]
    budget += sum(float(r["seconds"]) for r in read_csv(output / "preflight.csv"))
    resources = output / "eval_resources.csv"
    if resources.exists():
        budget += sum(float(r["seconds"]) for r in read_csv(resources))
    begin = time.perf_counter()
    try:
        primaries = []
        for m, seed in matrix:
            if time.perf_counter() - begin + budget >= config["max_total_seconds"]:
                raise TimeoutError("Registered evaluation budget reached")
            primaries.append(evaluate(output, split, m, seed, selections[m, seed], data[split], config))
        bootstrap(output, split, config)
        if split == "confirmation":
            write_json(output / "confirmation_gate.json", final_gate(primaries, config))
    finally:
        append_rows(resources, [{"split": split, "seconds": time.perf_counter() - begin}])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
