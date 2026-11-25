"""Audit dataset structure and shortcut ceilings before any model is trained."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import torch

from sdr_mamba.data_v2 import PIXEL_BINS, WINDOW_BINS, bucket, generate


def scores(pred, target):
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    return measurements(tp, fp, fn)


def measurements(tp, fp, fn):
    def ratio(num, den):
        return num / den if den else None

    return {
        "tp": tp, "fp": fp, "fn": fn, "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn), "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def append_rows(path, rows):
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def digest_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit(splits, manifest, config):
    output, occupancy_hashes, sample_hashes = {}, set(), set()
    passed = True
    for split_index, (split, data) in enumerate(splits.items()):
        occupancy, seed = data["image"].bool().unbind(-1)
        target = data["target"]
        occ_triplets = occupancy.unflatten(0, (-1, 3))
        targets = target.unflatten(0, (-1, 3))
        pairs_ok = bool((occ_triplets == occ_triplets[:, :1]).all()
                        and (targets.sum(1) == occ_triplets[:, 0]).all()
                        and (seed.sum((1, 2)) == 1).all() and (seed <= target).all())
        map_rows = manifest[split]["maps"]
        map_hashes = {row["occupancy_sha256"] for row in map_rows}
        samples = [s for row in map_rows for s in row["samples"]]
        hashes = {s["sample_sha256"] for s in samples}
        unique = (len(map_hashes) == len(map_rows) and len(hashes) == len(samples)
                  and not occupancy_hashes.intersection(map_hashes)
                  and not sample_hashes.intersection(hashes))
        occupancy_hashes.update(map_hashes)
        sample_hashes.update(hashes)
        random_seed = config["baselines"]["random_seed"] + split_index
        # A seeded random decision on each walkable pixel, shared across the same-map seeds.
        chance = torch.rand(occ_triplets[:, 0].shape, generator=torch.Generator().manual_seed(random_seed))
        random = (chance < config["baselines"]["random_positive_probability"]).repeat_interleave(3, dim=0)
        baselines = {
            "all_walkable": scores(occupancy, target), "seed_only": scores(seed, target),
            "occupancy_only_micro_iou_ceiling": scores(occupancy, target),
            "random_walkable": scores(random & occupancy, target), "bfs_oracle": scores(target, target),
        }
        window_hist, pixel_hist = Counter(), Counter()
        for sample in samples:
            for d, n in sample["window_graph_pixel_histogram"].items():
                window_hist[bucket(int(d), WINDOW_BINS)] += n
            for d, n in sample["pixel_distance_histogram"].items():
                pixel_hist[bucket(int(d), PIXEL_BINS)] += n
        coverage = window_hist["1-1"] > 0 and window_hist["2-3"] > 0 and window_hist["4-6"] > 0
        ratios = [s["target_ratio"] for s in samples]
        balanced = (all(0.25 <= ratio <= 0.55 for ratio in ratios)
                    and all(len(row["component_areas"]) == 3 and len(set(row["component_areas"])) == 1
                            for row in map_rows))
        structural_pass = (pairs_ok and unique and coverage and balanced
                           and baselines["all_walkable"]["iou"] <= config["validation_gate"]["all_walkable_max"])
        passed &= structural_pass
        output[split] = {
            "status": "PASS" if structural_pass else "FAIL", "count": len(samples),
            "map_count": len(map_rows), "component_area_counts_per_map": dict(sorted(
                Counter(row["component_areas"][0] for row in map_rows).items())),
            "target_ratio_range": [min(ratios), max(ratios)],
            "max_path_length_histogram": dict(sorted(Counter(s["target_eccentricity"] for s in samples).items())),
            "paired_gt_disjoint_and_exhaustive": pairs_ok, "paired_gt_iou": 0.0 if pairs_ok else None,
            "distinct_map_and_sample_hashes": unique, "required_distance_coverage": coverage,
            "window_graph_target_pixel_histogram": dict(window_hist), "pixel_path_histogram": dict(pixel_hist),
            "random_baseline_seed": random_seed, "baselines": baselines,
        }
    return {"status": "PASS" if passed else "FAIL", "config_sha256": digest_config(config), "splits": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sdr002.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    splits, manifest = generate(config)
    result = audit(splits, manifest, config)
    write_json(args.output / "data_manifest.json", manifest)
    write_json(args.output / "data_audit.json", result)
    print(json.dumps(result, indent=2))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
