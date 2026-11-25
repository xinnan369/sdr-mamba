"""Verify persisted SDR-003 metrics and report seed-paired pilot results."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

from scripts.train_sdr003 import checkpoint, read_json
from sdr_mamba.data_v3 import build


def read_rows(root, name):
    with (root / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def band(values):
    return f"{mean(values):.6f} +/- {stdev(values) if len(values) > 1 else 0:.6f}"


def values(rows, key):
    return [float(r[key]) for r in rows if r[key] != ""]


def summarize(root):
    torch.set_num_threads(2)
    config = read_json(root / "effective_config.json")
    _, manifest, _ = build(config)
    if json.loads(json.dumps(manifest)) != read_json(root / "data_manifest.json"):
        raise ValueError("Manifest failed full regeneration")
    audit = read_json(root / "data_audit.json")
    for split, entry in audit["splits"].items():
        print(split, manifest[split]["sha256"], entry["grid_target_histogram"],
              f"all-walkable={entry['baselines']['all_walkable']['iou']:.6f}")
    runs = read_rows(root, "runs.csv")
    metrics = read_rows(root, "metrics.csv")
    for row in metrics:
        tp, fp, fn = (int(row[col]) for col in ("tp", "fp", "fn"))
        if abs((tp / (tp + fp + fn)) - float(row["iou"])) > 1e-12:
            raise ValueError("IoU does not match recorded counts")
    print(f"Validated {len(metrics)} IoU rows and all dataset manifests")
    for r in runs:
        artifact = checkpoint(root, r["mode"], int(r["seed"]))
        if artifact.exists() and hashlib.sha256(artifact.read_bytes()).hexdigest() != r["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {r['run_id']}")
        print(r["run_id"], r["status"], "updates", r["updates"], "params", r["params"],
              "tokens", r["scan_tokens"], "seconds", r["wall_time_s"], "peak_bytes", r["peak_cuda_bytes"])
    print(f"Runs={len(runs)}, updates={sum(int(r['updates']) for r in runs)}")
    print("A selection:", read_json(root / "a_selection.json"))
    for phase in ("a", "b"):
        found = root / f"{phase}_gate.json"
        if found.exists():
            print(f"{phase.upper()} gate:", read_json(found))
    groups = defaultdict(list)
    for r in metrics:
        groups[(r["stage"], r["split"], r["mode"], int(r["test_T"]))].append(r)
    for (stage, split, mode, turn), rows in groups.items():
        print(stage, split, mode, turn, "|", "iou", band(values(rows, "iou")), "|",
              "recall", band(values(rows, "recall")), "|", "far recall", band(values(rows, "far_4_7_recall")),
              "|", "distractor recall", band(values(rows, "distractor_recall")), "|", "matched/swapped",
              f"{mean(values(rows, 'paired_matched_iou')):.6f}/{mean(values(rows, 'paired_swapped_iou')):.6f}",
              "| seeds", values(rows, "iou"))
    pairs = read_rows(root, "counterfactual.csv")
    print("Maximum seed-swap logit delta outside both physical bounds:",
          max(values(pairs, "outside_bound_max_logit_delta")))
    drift = read_rows(root, "feature_drift.csv")
    retention = read_rows(root, "retention.csv")
    for mode in dict.fromkeys(("A0", "A1", read_json(root / "a_selection.json")["selected"])):
        for turn in ("4", "8", "12"):
            items = [r for r in drift if r["mode"] == mode and r["round"] == turn and r["split"] == "test"]
            r_items = [r for r in retention if r["mode"] == mode and r["round"] == turn and r["split"] == "test"]
            print(mode, "ID round", turn, "|", " | ".join(
                f"{key} {band(values(items, key))}" for key in ("feature_rms", "update_rms",
                  "anchor_drift_rms", "logit_rms", "initial_to_full_output_l2")
            ), "|", "retention q50 across heads/seeds", band(values(r_items, "q50")))
    for r in read_rows(root, "gate_diagnostics.csv"):
        if r["round"] == "8":
            print("Edge", r["mode"], r["split"], r["seed"], "auc", r["edge_auroc"],
                  "false_open", r["false_open"], "false_close", r["false_close"],
                  "closed_edge_rms", r["closed_edge_payload_rms"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    summarize(parser.parse_args().report)
