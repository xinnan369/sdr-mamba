"""Verify SDR-005 measurements against per-image records and crop ledgers."""

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scripts.eval_sdr005 import read_csv
from scripts.train_sdr003 import read_json
from scripts.train_sdr005 import checkpoint


def summarize(root):
    config = read_json(root / "effective_config.json")
    sanity = read_json(root / "sanity_gate.json")
    print("SANITY", sanity)
    decision = read_json(root / "validation_gate.json")
    print("VALIDATION GATE", {key: value for key, value in decision.items() if key != "primary"})
    runs = read_csv(root / "runs.csv")
    keys = {(r["stage"], r["mode"], int(r["seed"])) for r in runs if r["status"] == "PASS"}
    expect = {("screen", m, 0) for m in config["train_modes"]}
    if decision["status"] == "PASS":
        expect |= {("pilot", m, s) for m in config["train_modes"] for s in config["train_seeds"][1:]}
    if keys != expect or len(runs) != len(expect) or any(int(r["updates"]) != config["train_updates"] for r in runs):
        raise ValueError("Training ledger is incomplete")
    for r in runs:
        path = checkpoint(root, r["stage"], r["mode"], int(r["seed"]))
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != r["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint integrity mismatch {path.name}")
    print("RUNS", len(keys), "UPDATES", sum(int(r["updates"]) for r in runs),
          "aggregated seconds", sum(float(r["wall_time_s"]) for r in runs))
    ledger = read_csv(root / "crop_ledger.csv")
    crops = defaultdict(list)
    fields = ("step", "batch_slot", "id", "top", "left", "height", "width", "flipped",
              "image_points", "crop_points", "excluded_by_crop")
    for r in ledger:
        if int(r["image_points"]) != int(r["crop_points"]) + int(r["excluded_by_crop"]):
            raise ValueError("Crop count mismatch")
        crops[r["seed"], r["mode"]].append(tuple(r[k] for k in fields))
    if set(crops) != {(str(s), m) for _, m, s in expect}:
        raise ValueError("Unexpected or incomplete crop matrix")
    for seed in {s for s, _ in crops}:
        first = crops[seed, config["train_modes"][0]]
        if len(first) != config["batch_size"] * config["train_updates"]:
            raise ValueError("Incomplete crop ledger")
        if any(crops[seed, mode] != first for mode in config["train_modes"]):
            raise ValueError(f"Unmatched crop sequence for seed {seed}")
    print("Crop ledger rows verified", len(ledger))
    data = read_csv(root / "per_image.csv")
    metrics = read_csv(root / "depth_metrics.csv")
    keyed = defaultdict(list)
    for r in data:
        keyed[r["split"], r["mode"], r["seed"], r["t"]].append(r)
    if decision["status"] != "PASS" and any(key[0] == "test" for key in keyed):
        raise ValueError("Official test predictions were recorded without gate approval")
    metric_keys = {(r["split"], r["mode"], r["seed"], r["t"]) for r in metrics}
    if metric_keys != set(keyed) or len(metrics) != len(metric_keys):
        raise ValueError("Depth matrix disagrees with per-image keys")
    split = read_json(root / "split_manifest.json")
    for r in metrics:
        selection = keyed[r["split"], r["mode"], r["seed"], r["t"]]
        expected_ids = {entry["id"] for entry in split[r["split"]]["entries"]}
        if int(r["images"]) != len(selection) or {x["id"] for x in selection} != expected_ids or len(selection) != len(expected_ids):
            raise ValueError("Missing or duplicate image in depth matrix")
        recomputed = {
            "mae": mean(float(x["absolute"]) for x in selection),
            "rmse": math.sqrt(mean(float(x["squared"]) for x in selection)),
            "bias": mean(float(x["signed"]) for x in selection),
            **{key: mean(float(x[key]) for x in selection)
               for key in ("game1", "game2", "game3", "density_cell_mae")},
        }
        if any(abs(value - float(r[key])) > 1e-9 for key, value in recomputed.items()):
            raise ValueError("Metric disagrees with per-image results")
    print(f"Verified {len(metrics)} depth rows and {len(data)} per-image rows")
    fields = ("mae", "rmse", "game1", "game2", "game3", "bias", "density_cell_mae")
    for mode in config["train_modes"]:
        for split in ("validation", "test"):
            for t in config["evaluation_rounds"]:
                rows = [r for r in metrics if r["mode"] == mode and r["split"] == split and int(r["t"]) == t]
                if rows:
                    results = {col: [float(r[col]) for r in rows] for col in fields}
                    print(split, mode, t, {col: (mean(values), stdev(values) if len(values) > 1 else None)
                                           for col, values in results.items()})
    print("PAIRED", read_csv(root / "paired_bootstrap.csv"))
    print("PROFILES", read_csv(root / "latency.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
