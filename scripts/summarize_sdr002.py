"""Check recorded SDR-002 results and summarize completed stages."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import torch

from scripts.audit_dataset import audit
from sdr_mamba.data_v2 import generate
from sdr_mamba.state_transport import MODES


def rows(directory, name):
    with (directory / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def band(values):
    present = [float(v) for v in values if v != ""]
    if not present:
        return "NA"
    return f"{mean(present):.6f} ± {stdev(present) if len(present) > 1 else 0:.6f}"


def grouped_counts(items):
    tp, fp, fn = (sum(int(r[key]) for r in items) for key in ("tp", "fp", "fn"))
    return tp / (tp + fp + fn) if tp + fp + fn else float("nan")


def summarize(directory):
    torch.set_num_threads(2)
    config = read_json(directory, "effective_config.json")
    data, manifest = generate(config)
    stored_manifest = read_json(directory, "data_manifest.json")
    # JSON conversion canonicalizes integer histogram keys.
    if json.loads(json.dumps(manifest)) != stored_manifest:
        raise ValueError("Regeneration did not reproduce the complete saved manifest")
    result = read_json(directory, "data_audit.json")
    if json.loads(json.dumps(audit(data, manifest, config))) != result:
        raise ValueError("Regeneration did not reproduce the saved audit")
    for split, value in result["splits"].items():
        print(f"{split} {value['status']} {manifest[split]['sha256']} "
              f"distances={value['window_graph_target_pixel_histogram']}")
        print("  Baseline micro IoU:", {name: scores["iou"] for name, scores in value["baselines"].items()})
        images, label, graph = (data[split][key] for key in ("image", "target", "window"))
        k = config["model"]["k"]
        grid = torch.arange(config["size"]) // k
        ys, xs = (images[..., 1] == 1).nonzero(as_tuple=True)[1:]
        manhattan = ((grid[None, :, None] - ys[:, None, None] // k).abs()
                     + (grid[None, None, :] - xs[:, None, None] // k).abs())
        print(f"  Graph > 3 and Manhattan <= 3 target pixels: {int((label & (graph > 3) & (manhattan <= 3)).sum())}")
    gate = read_json(directory, "validation_gate.json")
    print("Validation gate:", json.dumps(gate, ensure_ascii=False))
    if gate["status"] != "PASS":
        print("Formal test evaluation and mechanism decision: NOT_RUN")
        return
    runs = rows(directory, "runs.csv")
    metrics = rows(directory, "metrics.csv")
    seeds = config["seeds"]
    expected = {f"{mode}-s{seed}" for mode in MODES for seed in seeds} | {"occupancy-s0"}
    if len(runs) != len(expected) or {r["run_id"] for r in runs if r["status"] == "PASS"} != expected:
        raise ValueError("Training matrix is incomplete; check runs.csv")
    full = {(r["mode"], int(r["seed"]), int(r["test_T"])): r for r in metrics if r["variant"] == "full"}
    expected_metrics = {(m, s, t) for m in MODES for s in seeds for t in config["test_T"]}
    if set(full) != expected_metrics or len(metrics) != len(expected_metrics) + 2:
        raise ValueError("Expected a full metric matrix and the two T=4 seed controls")
    curves = rows(directory, "train_curves.csv")
    expected_curve_rows = len(runs) * (1 + 2 * (config["steps"] // config["period"]))
    if len(curves) != expected_curve_rows:
        raise ValueError("Incomplete train/validation curves")
    print(f"{len(runs)} completed training runs, {sum(int(r['steps']) for r in runs)} updates, "
          f"{len(metrics)} metric rows and {len(curves)} train/validation curve rows")
    print("mode | T | IoU mean ± sample SD | Dice | precision | recall | IoU seeds")
    for mode in MODES:
        for turn in config["test_T"]:
            entries = [full[mode, seed, turn] for seed in seeds]
            summary = [band([r[key] for r in entries]) for key in ("iou", "dice", "precision", "recall")]
            print(f"{mode} | {turn} | " + " | ".join(summary)
                  + " | " + ", ".join(f"{float(r['iou']):.6f}" for r in entries))
    for r in metrics:
        if r["variant"] != "full":
            print(f"Test {r['variant']} at T=4: IoU={float(r['iou']):.6f}, Dice={float(r['dice']):.6f}")
    print("Parameters:", sorted({int(r["params"]) for r in metrics}))
    print("Train directional scans/run:", sorted({int(r["train_scan_count"]) for r in metrics}))
    print("Run time seconds:", min(float(r["wall_time_s"]) for r in runs),
          max(float(r["wall_time_s"]) for r in runs), "sum", sum(float(r["wall_time_s"]) for r in runs))
    peaks = [int(r["peak_cuda_allocated_bytes"]) for r in runs if r["peak_cuda_allocated_bytes"]]
    if peaks:
        print("CUDA peak allocated bytes:", min(peaks), max(peaks))
    print("Paired SST delta in IoU percentage points at train depth")
    turn = config["train_T"]
    for mode in ("reset", "local", "shuffled", "feature_neighbor"):
        deltas = [100 * (float(full["sst", seed, turn]["iou"]) - float(full[mode, seed, turn]["iou"])) for seed in seeds]
        count = sum(d > 0 for d in deltas)
        print(f"{mode}: {band(deltas)}; wins={count}/{len(seeds)}; seeds={deltas}")
    depths = rows(directory, "distance_metrics.csv")
    print("Window-graph IoU at T=4 (0, 1, 2–3, 4–6, 7+)")
    for mode in MODES:
        blocks = [[grouped_counts([r for r in depths if r["run_id"] == f"{mode}-s{s}" and r["test_T"] == str(turn)
                                   and r["bucket"] == region and r["distance_axis"] == "window"])
                   for s in seeds] for region in ("0-0", "1-1", "2-3", "4-6", "7+")]
        print(mode + ": " + " | ".join(band(block) for block in blocks))
    print("Paired SST IoU deltas at window distance 2–3")
    for mode in ("reset", "local", "shuffled", "feature_neighbor"):
        values = []
        for seed in seeds:
            entries = [r for r in depths if r["test_T"] == str(turn) and r["distance_axis"] == "window"
                       and r["bucket"] == "2-3"]
            sst = grouped_counts([r for r in entries if r["run_id"] == f"sst-s{seed}"])
            control = grouped_counts([r for r in entries if r["run_id"] == f"{mode}-s{seed}"])
            values.append(100 * (sst - control))
        print(f"{mode}: {band(values)}; wins={sum(v > 0 for v in values)}/{len(seeds)}; seeds={values}")
    pairs = rows(directory, "counterfactual.csv")
    for mode in (*MODES, "occupancy"):
        selected = [r for r in pairs if r["split"] == "test" and r["test_T"] == str(turn)
                    and r["run_id"] in {f"{mode}-s{s}" for s in seeds}]
        if selected:
            matched = mean((float(r["iou_a"]) + float(r["iou_b"])) / 2 for r in selected)
            swapped = mean((float(r["swapped_iou_a"]) + float(r["swapped_iou_b"])) / 2 for r in selected)
            flip = mean(float(r["prediction_change_rate"]) for r in selected)
            print(f"Counterfactual {mode}: matched={matched:.6f}, swapped={swapped:.6f}, flip={flip:.6f}")
    components = rows(directory, "component_metrics.csv")
    print("T=4 component-wise mean IoU and recall (macro over maps, seeds and components)")
    for mode in MODES:
        ids = {f"{mode}-s{s}" for s in seeds}
        mode_rows = [r for r in components if r["run_id"] in ids and r["test_T"] == str(turn)]
        blocks = {
            role: {metric: mean(float(r[metric]) for r in mode_rows if r["role"] == role)
                   for metric in ("iou", "recall")}
            for role in ("target", "distractor")
        }
        print(f"{mode}: {blocks}")

    print("T=4 SST intervention by window graph distance")
    interventions = rows(directory, "intervention.csv")
    regions = ("0-0", "1-1", "2-3", "4-6", "7+", "outside")
    for variant in ("zero_neighbor", "shuffled_neighbor"):
        for region in regions:
            entries = [r for r in interventions if r["split"] == "test" and r["round"] == str(turn)
                       and r["replacement"] == variant and r["graph_bucket"] == region]
            print(f"{variant} {region}: " + " | ".join(
                band([r[key] for r in entries])
                for key in ("mean_abs_probability_delta", "flip_rate", "neighbor_state_delta_rms")
            ))
    norms = rows(directory, "state_norms.csv")
    print("T=4 packet RMS by window distance (sent, received; pooled over seeds and directions)")
    for mode in ("sst", "feature_neighbor"):
        ids = {f"{mode}-s{s}" for s in seeds}
        for region in regions:
            entries = [r for r in norms if r["run_id"] in ids and r["round"] == str(turn)
                       and r["graph_bucket"] == region and float(r["elements"])]
            if entries:
                count = sum(float(r["elements"]) for r in entries)
                values = [math.sqrt(sum(float(r[key]) ** 2 * float(r["elements"]) for r in entries) / count)
                          for key in ("sent_message_rms", "received_message_rms")]
                print(f"{mode} {region}: sent={values[0]:.6f}, received={values[1]:.6f}")
    print("SST gradient RMS (mean at step 512 across directions and seeds)")
    grad = rows(directory, "gradient_diagnostics.csv")
    for current in range(1, turn + 1):
        selected = [r for r in grad if r["run_id"].startswith("sst-") and r["round"] == str(current)
                    and r["step"] == str(config["steps"])]
        print(f"T={current}: " + " | ".join(band([r[key] for r in selected]) for key in (
            "current_raw_readout_grad_rms", "future_state_transport_grad_rms", "initial_state_grad_rms",
        )))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summarize(args.directory)
