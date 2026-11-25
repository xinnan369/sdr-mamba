"""Audit SDR-004 raw results and retain a concise seed-paired readout."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import torch

from scripts.summarize_sdr003 import export_distances
from scripts.train_sdr003 import read_json
from scripts.train_sdr004 import checkpoint, formal_matrix, screen_matrix
from sdr_mamba.data_v4 import build


def read_rows(root, name):
    with (root / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def band(sequence):
    values = [float(x) for x in sequence if x != ""]
    if not values:
        return "NA"
    return f"{mean(values):.6f} +/- {stdev(values) if len(values) > 1 else 0:.6f}"


def summarize(root, distances=None):
    torch.set_num_threads(2)
    cfg = read_json(root / "effective_config.json")
    data, manifest = build(cfg)
    if json.loads(json.dumps(manifest)) != read_json(root / "data_manifest.json"):
        raise ValueError("Manifest regeneration mismatch")
    if distances:
        export_distances(distances, data, manifest)
    audit = read_json(root / "data_audit.json")
    print("AUDIT", audit["status"])
    for split, info in audit["splits"].items():
        print(split, "maps", info["map_count"], "area", info["component_area_mean"],
              "rank", info["eccentricity_rank_mean"], "range", info["eccentricity_range"],
              "rejected", info["rejection_rate"], "far target px", info["far_target_pixels"],
              "unreachable at T8", info["grid_t8_unreachable_target_pixels"],
              "baseline IoU", info["baselines"]["all_walkable"]["iou"])
    selection = read_json(root / "selection.json")
    gate = read_json(root / "gate.json")
    seeds = cfg["train_seeds"]
    expected = {("screen", mode, ds, 0) for mode, ds in screen_matrix()}
    if selection["status"] == "CONTINUE":
        expected.update(("pilot", mode, ds, seed)
                        for mode, ds in formal_matrix(selection["selected_dstate"]) for seed in seeds)
    runs = read_rows(root, "runs.csv")
    found = {(r["stage"], r["mode"], int(r["dstate"]), int(r["seed"])) for r in runs if r["status"] == "PASS"}
    if found != expected:
        raise ValueError(f"Expected {len(expected)} completed runs, recorded {len(found)}")
    for row in runs:
        if row["status"] != "PASS":
            print("FAILED", row)
            continue
        model = checkpoint(root, row["stage"], row["mode"], int(row["dstate"]), int(row["seed"]))
        if model.exists() and hashlib.sha256(model.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
            raise ValueError(f"Artifact integrity failed for {model.name}")
    print("RUNS", len(found), "UPDATES", sum(int(r["updates"]) for r in runs), "seconds",
          sum(float(r["wall_time_s"]) for r in runs))
    print("SELECTION", selection)
    print("GATE", gate)
    rows = read_rows(root, "metrics.csv")
    tables = defaultdict(list)
    if len(rows) != len(expected) * 3 * len(cfg["eval_T"]):
        raise ValueError("Incomplete evaluation matrix")
    index = set()
    for r in rows:
        tp, fp, fn = (int(r[key]) for key in ("tp", "fp", "fn"))
        if abs(float(r["iou"]) - tp / (tp + fp + fn)) > 1e-12:
            raise ValueError("IoU disagreement with confusion counts")
        key = tuple(r[x] for x in ("stage", "split", "mode", "dstate", "test_T"))
        observation = (*key, r["seed"])
        if observation in index:
            raise ValueError(f"Duplicate aggregate metric {observation}")
        index.add(observation)
        tables[key].append(r)
    print("Verified", len(rows), "metric rows against counts")
    for key, values in tables.items():
        print(*key, "|", " | ".join(f"{column}: {band(r[column] for r in values)}" for column in
              ("iou", "recall", "far_recall", "distractor_coverage", "matched", "swapped")),
              "| IoU seeds", [float(r["iou"]) for r in values])
    pairs = read_rows(root, "counterfactual.csv")
    bounds = [float(r["max_outside_logit_delta"]) for r in pairs if r["max_outside_logit_delta"] != ""]
    peak = max(bounds)
    if peak > 2e-6:
        raise ValueError(f"Outside-bound seed swap delta {peak}")
    print("Maximum bounded-mode outside-bound logit difference", peak)
    for row in read_rows(root, "paired_bootstrap.csv"):
        print("PAIRED", *(row[key] for key in ("stage", "split", "dstate", "comparison", "mean_iou_delta",
              "sample_sd", "seed_deltas", "positive_seeds", "map_bootstrap_ci95_low", "map_bootstrap_ci95_high")))
    latency = read_rows(root, "evaluation_latency.csv")
    for key in dict.fromkeys((r["stage"], r["mode"], r["dstate"]) for r in runs if r["status"] == "PASS"):
        registered = [r for r in runs if (r["stage"], r["mode"], r["dstate"]) == key]
        profiles = [float(r["median_ms_per_image"]) for r in latency if (r["stage"], r["mode"], r["dstate"]) == key]
        print("PROFILE", *key, "params", registered[0]["params"],
              "gpu_peak_bytes", max((int(r["peak_cuda_bytes"]) for r in registered if r["peak_cuda_bytes"]), default=0),
              "forward_ms_per_image_T16", median(profiles) if profiles else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--distance-archive", type=Path)
    args = parser.parse_args()
    summarize(args.output, args.distance_archive)
