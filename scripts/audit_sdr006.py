"""Archive ShanghaiTech count protocols and freeze the new internal split."""

import argparse
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from statistics import mean

import numpy as np

from scripts.audit_dataset import append_rows, write_json
from scripts.train_sdr003 import read_json
from sdr_mamba.crowd_audit import project_points
from sdr_mamba.crowd_data import in_rect, load_dataset, mass_map, parse_points


def raw_points(sample):
    return parse_points(sample.file.parent.parent / "ground_truth" / f"GT_{sample.file.stem}.mat")


def select_holdout(samples, seed):
    if len(samples) != 340:
        raise ValueError("SDR-006 requires the 340 prior training records")
    ranked = sorted(samples, key=lambda s: (len(s.points), s.identifier))
    result = set()
    for start in range(0, len(ranked), 17):
        block = ranked[start:start + 17]
        block.sort(key=lambda s: hashlib.sha256(f"{seed}:{s.identifier}".encode()).digest())
        result.update(s.identifier for s in block[:3])
    return result


def describe(values):
    return {key: float(value) for key, value in zip(
        ("min", "p25", "median", "p75", "max", "mean"),
        (*np.quantile(values, (0, .25, .5, .75, 1)), mean(values)),
    )}


def dataset(config):
    historical, manifest, _ = load_dataset(Path(os.environ["SDR006_DATA_ROOT"]),
                                            read_json(Path("configs/sdr005.json")))
    if manifest != read_json(Path("reports/SDR-005/split_manifest.json")):
        raise ValueError("Local dataset does not match the published SDR-005 manifest")
    raw = {s.identifier: raw_points(s) for partition in historical.values() for s in partition}
    holdout = select_holdout([replace(s, points=raw[s.identifier]) for s in historical["train"]],
                             config["holdout_seed"])
    subsets = {
        "train": [s for s in historical["train"] if s.identifier not in holdout],
        "development": historical["validation"],
        "confirmation": [s for s in historical["train"] if s.identifier in holdout],
        "official_test_static": historical["test"],
    }
    audit_rows, partitions, frozen = [], {}, {}
    for name, samples in subsets.items():
        entries, projected = [], []
        for s in samples:
            points = raw[s.identifier]
            if (s.width, s.height) != (1024, 768):
                raise ValueError("Unexpected Part B resolution")
            base0 = int(in_rect(points, s.width, s.height).sum())
            base1 = int(in_rect(points - 1, s.width, s.height).sum())
            p0 = project_points(points, s.width, s.height, config["coordinate_base"])
            grid = mass_map(p0, s.height, s.width, config["stride"])
            if grid.sum().item() != len(points):
                raise ValueError("Projection changed annotation mass")
            xy = np.floor(p0 / config["stride"])
            other = project_points(points, s.width, s.height, 1)
            audit_rows.append({
                "split": name, "id": s.identifier, "mat_number": len(points),
                "base0_in_frame": base0, "base1_in_frame": base1, "projection_mass": int(grid.sum()),
                "base0_over_1px_outside": int((~in_rect(points + 1, s.width + 2, s.height + 2)).sum()),
                "cell_assignments_changed_after_base1_projection": int(
                    (xy != np.floor(other / config["stride"])).any(axis=1).sum()),
                "both_coordinates_fractional": int((points % 1 != 0).all(axis=1).sum()),
                "exact_zero_component": int((points == 0).any(axis=1).sum()),
            })
            projected.append(replace(s, points=p0))
            entries.append({"id": s.identifier, "image_sha256": s.image_sha, "annotation_sha256": s.annotation_sha,
                            "width": s.width, "height": s.height, "full_count": len(points)})
        digest = hashlib.sha256()
        for entry in entries:
            digest.update(f"{entry['id']}:{entry['image_sha256']}:{entry['annotation_sha256']}\n".encode())
        frozen[name] = {"sha256": digest.hexdigest(), "entries": entries}
        if name != "official_test_static":
            partitions[name] = projected
    return partitions, frozen, audit_rows


def write_audit(output, config, manifest, rows):
    output.mkdir(exist_ok=True, parents=True)
    splits = {}
    for name, record in manifest.items():
        group = [r for r in rows if r["split"] == name]
        fields = group[0].keys() - {"split", "id"}
        splits[name] = {"images": len(group), "sha256": record["sha256"],
                        **{key: sum(r[key] for r in group) for key in sorted(fields)},
                        "count_distribution": describe([r["mat_number"] for r in group]),
                        "affected_base0_images": sum(r["mat_number"] != r["base0_in_frame"] for r in group),
                        "affected_base1_images": sum(r["mat_number"] != r["base1_in_frame"] for r in group)}
    write_json(output / "dataset_audit.json", {
        "status": "PASS", "holdout_seed": config["holdout_seed"], "rule": config["boundary"],
        "coordinate_base": config["coordinate_base"], "splits": splits,
    })
    write_json(output / "split_manifest.json", manifest)
    if (output / "annotation_counts.csv").exists():
        raise FileExistsError("Use a new audit output directory")
    append_rows(output / "annotation_counts.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_json(Path("configs/sdr006.json"))
    _, frozen, counts = dataset(cfg)
    write_audit(args.output, cfg, frozen, counts)
    print("Static protocol audited", {name: len(r["entries"]) for name, r in frozen.items()})
