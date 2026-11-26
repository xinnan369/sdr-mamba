"""Recompute SDR-006 summaries and verify the published evidence."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean, stdev

from scripts.eval_sdr005 import read_csv
from scripts.eval_sdr006 import depths
from scripts.train_sdr003 import read_json
from scripts.train_sdr006 import artifact, checkpoint_rank
from sdr_mamba.crowd_data import sha_file


def verify(output):
    config = read_json(output / "effective_config.json")
    manifest = read_json(output / "split_manifest.json")
    expected_ids = {split: {r["id"] for r in data["entries"]} for split, data in manifest.items()}
    expected_sizes = {"train": 280, "development": 60, "confirmation": 60, "official_test_static": 316}
    assert {split: len(ids) for split, ids in expected_ids.items()} == expected_sizes
    assert len(set.union(*expected_ids.values())) == sum(expected_sizes.values())
    gate = read_json(output / "gate.json")
    seeds = config["seeds"] if gate["status"] == "PASS" else [0]
    expected_runs = {(m, str(seed)) for m in config["modes"] for seed in seeds}
    run_rows = read_csv(output / "runs.csv")
    assert {(r["mode"], r["seed"]) for r in run_rows} == expected_runs
    assert len(run_rows) == len(expected_runs)
    assert all(r["status"] == "PASS" and int(r["updates"]) == config["updates"] for r in run_rows)
    curves = defaultdict(list)
    for r in read_csv(output / "train_curves.csv"):
        path = artifact(output, r["mode"], int(r["seed"]), int(r["step"]))
        if path.exists():
            assert r["checkpoint_sha256"] == sha_file(path), path.name
        curves[r["mode"], r["seed"]].append(r)
    for r in read_csv(output / "checkpoint_selection.csv"):
        candidates = curves[r["mode"], r["seed"]]
        assert [int(x["step"]) for x in candidates] == list(range(config["eval_period"], config["updates"] + 1,
                                                                 config["eval_period"]))
        assert checkpoint_rank(r) == min(map(checkpoint_rank, candidates))
        tail = [float(x["mae"]) for x in candidates[-3:]]
        assert math.isclose(float(r["tail_relative_sd"]), stdev(tail)/mean(tail), abs_tol=1e-12)
    crop_rows = read_csv(output / "crop_ledger.csv")
    ledger = defaultdict(list)
    fields = ("id", "step", "slot", "top", "left", "height", "width", "flipped", "crop_points", "image_points")
    for r in crop_rows:
        assert r["id"] in expected_ids["train"]
        assert int(r["crop_points"]) + int(r["excluded_by_crop"]) == int(r["image_points"])
        ledger[r["mode"], r["seed"]].append(tuple(r[field] for field in fields))
    assert set(ledger) == expected_runs
    for seed in seeds:
        reference = ledger["F0", str(seed)]
        assert len(reference) == config["updates"] * config["batch_size"]
        assert all(ledger[mode, str(seed)] == reference for mode in config["modes"])
    expected_metrics = {(split, mode, str(seed), str(depth))
                        for split in ("development", "confirmation")
                        for mode in config["modes"] for depth in depths(mode)
                        for seed in ([0] if split == "development" else (seeds if gate["status"] == "PASS" else []))}
    per_image = read_csv(output / "per_image.csv")
    groups = defaultdict(list)
    keys = ("split", "mode", "seed", "t")
    for row in per_image:
        groups[tuple(row[k] for k in keys)].append(row)
        delta = float(row["count"]) - int(row["gt"])
        assert abs(delta - float(row["signed"])) < 1e-9
        assert abs(abs(delta) - float(row["absolute"])) < 1e-9
    metrics = read_csv(output / "depth_metrics.csv")
    assert set(groups) == expected_metrics == {tuple(r[k] for k in keys) for r in metrics}
    assert len(metrics) == len(expected_metrics)
    for metric in metrics:
        group = groups[tuple(metric[k] for k in keys)]
        assert len(group) == len(expected_ids[metric["split"]]) == int(metric["images"])
        assert {r["id"] for r in group} == expected_ids[metric["split"]]
        counts = {
            "mae": mean(float(r["absolute"]) for r in group),
            "rmse": math.sqrt(mean(float(r["signed"])**2 for r in group)),
            "bias": mean(float(r["signed"]) for r in group),
            **{field: mean(float(r[field]) for r in group) for field in ("game1", "game2", "game3", "density_cell_mae")},
        }
        assert all(math.isclose(value, float(metric[k]), abs_tol=1e-10) for k, value in counts.items())
    print(f"Verified {len(metrics)} depth aggregates, {len(per_image)} per-image rows, "
          f"{len(crop_rows)} matched crop rows and {sum(map(len, curves.values()))} checkpoint candidates")
    primary = read_csv(output / "metrics.csv")
    for split in ("development", "confirmation"):
        for mode in config["modes"]:
            rows = [r for r in primary if r["split"] == split and r["mode"] == mode]
            if rows:
                numeric = {k: [float(r[k]) for r in rows]
                           for k in ("mae", "rmse", "game1", "game2", "game3", "bias", "density_cell_mae")}
                print(split, mode, {k: (mean(v), stdev(v) if len(v) > 1 else None) for k, v in numeric.items()})
    train_seconds = sum(float(r["seconds"]) for r in run_rows)
    gpu_seconds = (train_seconds + read_json(output / "sanity_gate.json")["seconds"]
                   + sum(float(r["seconds"]) for name in ("preflight", "eval_resources")
                         for r in read_csv(output / f"{name}.csv")))
    print("Training seconds", train_seconds, "Tracked total experiment seconds", gpu_seconds)
    assert gpu_seconds <= config["max_total_seconds"]
    print("Development gate", gate["status"])
    assert not (output / "test_started.json").exists()
    if gate["status"] != "PASS":
        assert not (output / "confirmation_started.json").exists()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    verify(parser.parse_args().output)
