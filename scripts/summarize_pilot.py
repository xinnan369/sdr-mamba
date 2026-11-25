"""Validate a complete pilot matrix and print its seed-level summaries."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from scripts.run_experiment import DISTANCE_BINS, confusion, scores
from sdr_mamba.data import generate
from sdr_mamba.state_transport import MODES


def read_csv(directory, name):
    with (directory / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def band(values):
    return f"{mean(values):.6f} ± {stdev(values) if len(values) > 1 else 0.0:.6f}"


def summarize(directory):
    config = json.loads((directory / "effective_config.json").read_text())
    seeds = config["seeds"]
    expected_runs = {f"{mode}-s{seed}" for mode in MODES for seed in seeds}
    runs = read_csv(directory, "runs.csv")
    if len(runs) != len(expected_runs) or {r["run_id"] for r in runs if r["status"] == "PASS"} != expected_runs:
        raise ValueError("Incomplete experiment matrix. See runs.csv and the run log.")
    train_steps = read_csv(directory, "train_steps.csv")
    expected_steps = {(run_id, str(step)) for run_id in expected_runs for step in range(1, config["steps"] + 1)}
    if len(train_steps) != len(expected_steps) or {(r["run_id"], r["step"]) for r in train_steps} != expected_steps:
        raise ValueError("Missing or duplicate training steps")
    rows = read_csv(directory, "metrics.csv")
    metrics = {(r["mode"], int(r["seed"]), r["split"], int(r["test_T"])): r for r in rows}
    expected = {
        (mode, seed, split, turn)
        for mode in MODES for seed in seeds
        for split, rounds in (("validation", [config["train_T"]]), ("test", config["test_T"]))
        for turn in rounds
    }
    if set(metrics) != expected or len(rows) != len(expected):
        raise ValueError("Missing or duplicate metric rows")
    print(f"Verified {len(runs)} runs, {len(train_steps)} optimizer steps and {len(rows)} metric rows")
    print(f"Total run seconds: {sum(float(r['total_wall_time_s']) for r in runs):.3f}")
    print("Mode | T | test IoU mean ± sample SD | test Dice mean ± sample SD | seed IoU")
    for mode in MODES:
        for turn in config["test_T"]:
            values = [metrics[mode, seed, "test", turn] for seed in seeds]
            iou = [float(row["iou"]) for row in values]
            dice = [float(row["dice"]) for row in values]
            print(f"{mode} | {turn} | {band(iou)} | {band(dice)} | "
                  + ", ".join(f"{value:.6f}" for value in iou))
    print("SST minus comparator at train depth (percentage points)")
    turn = config["train_T"]
    for mode in ("reset", "local", "shuffled", "feature_neighbor"):
        deltas = [
            100 * (float(metrics["sst", seed, "test", turn]["iou"])
                   - float(metrics[mode, seed, "test", turn]["iou"]))
            for seed in seeds
        ]
        wins = sum(delta > 0 for delta in deltas)
        passed = len(seeds) == 3 and mean(deltas) >= 2 and wins >= 2
        print(f"{mode}: mean={mean(deltas):+.6f}, wins={wins}/{len(seeds)}, "
              f"gate={'PASS' if passed else 'NOT_MET'}, seeds="
              + ", ".join(f"{delta:+.6f}" for delta in deltas))

    print("T=train_T distance IoU (test, sample mean ± SD over seeds)")
    distances = read_csv(directory, "distance_metrics.csv")
    for mode in MODES:
        values = [r for r in distances if r["split"] == "test" and r["round"] == str(turn)
                  and r["mode"] == mode]
        cells = [band([float(r["iou"]) for r in values if r["bucket"] == f"{low}-{high}"])
                 for low, high in DISTANCE_BINS]
        print(f"{mode}: " + " | ".join(cells))

    splits, manifest = generate(config)
    original_manifest = json.loads((directory / "data_manifest.json").read_text())
    for split, entry in manifest.items():
        if entry["sha256"] != original_manifest[split]["sha256"]:
            raise ValueError(f"Regenerated {split} data hash differs")
    image, target, distances = splits["test"]
    pred, label = image[..., 0].bool(), target.bool()
    print("All-walkable test baseline:", scores(confusion(pred, label, distances >= 0)))
    for low, high in DISTANCE_BINS:
        print(f"  {low}-{high}:", scores(confusion(pred, label, (distances >= low) & (distances <= high))))

    print("SST intervention (T=4, test, mean over seeds)")
    interventions = read_csv(directory, "intervention.csv")
    for replacement in ("zero", "shuffled"):
        for current in ("2", "3", "4"):
            values = [r for r in interventions if r["round"] == current
                      and r["replacement"] == replacement and r["spatial_axis"] == "window_distance"]
            count = sum(int(r["pixels"]) for r in values)
            delta, flips = [
                sum(float(r[key]) * int(r["pixels"]) for r in values) / count
                for key in ("mean_abs_probability_delta", "threshold_flip_rate")
            ]
            print(f"{replacement} T={current} all: probability_delta={delta:.9f}, flip_rate={flips:.9f}")
        for low, high in DISTANCE_BINS:
            bucket = f"{low}-{high}"
            rows = [r for r in interventions if r["round"] == "4" and r["replacement"] == replacement
                    and r["spatial_axis"] == "window_distance" and r["bucket"] == bucket]
            print(f"{replacement} {bucket}: probability_delta="
                  f"{mean(float(r['mean_abs_probability_delta']) for r in rows):.9f}, "
                  f"flip_rate={mean(float(r['threshold_flip_rate']) for r in rows):.9f}")

    print("SST training state gradients (mean RMS over seeds and recorded steps)")
    train_states = [r for r in read_csv(directory, "train_states.csv") if r["run_id"].startswith("sst-")]
    for turn in range(1, config["train_T"] + 1):
        frames = [r for r in train_states if r["round"] == str(turn)]
        summary = []
        for key in ("final_rms", "initial_rms", "final_grad_rms", "initial_grad_rms"):
            values = [float(r[key]) for r in frames if r[key]]
            summary.append(f"{key}={mean(values):.9g}" if values else f"{key}=NA")
        print(f"round={turn}: " + ", ".join(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
