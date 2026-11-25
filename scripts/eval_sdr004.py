"""Fixed-checkpoint metrics, paired map intervals and physical-lane diagnosis."""

import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev
import time

import torch

from scripts.audit_dataset import append_rows, scores, write_json
from scripts.eval_sdr003 import mask_bins
from scripts.run_sdr002 import batches
from scripts.train_sdr003 import read_json
from scripts.train_sdr004 import MODES, TEST, VAL, checkpoint, formal_matrix, screen_matrix, setup, summary
from sdr_mamba.data_v2 import PIXEL_BINS, WINDOW_BINS
from sdr_mamba.model_v3 import boundary_pairs, valid_edges
from sdr_mamba.model_v4 import model_for, neighbors
from sdr_mamba.scan import tile

GRID_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, None))
LANE_MODES = ("L1", "L2", "L3", "L4", "L5")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def confusion(pred, y):
    return torch.stack(((pred & y).sum((1, 2, 3)), (pred & ~y).sum((1, 2, 3)),
                        (~pred & y).sum((1, 2, 3))), dim=-1)


def paired(logits, data, meta, output):
    labels = data["target"].unflatten(0, (-1, 3))
    grid = data["grid"].unflatten(0, (-1, 3))
    raw = logits.unflatten(0, (-1, 3))
    pred = raw >= 0
    rows = []
    for i, j in combinations(range(3), 2):
        diff = (raw[:, i] - raw[:, j]).abs()
        predictions_changed = pred[:, i] ^ pred[:, j]
        outside = grid[:, i].minimum(grid[:, j]) >= meta["test_T"]
        for m, (y, p) in enumerate(zip(labels, pred)):
            safe = outside[m] if meta["mode"] != "L7" else torch.zeros_like(outside[m])
            rows.append({
                **meta, "map_index": m, "seed_a": i, "seed_b": j,
                "matched_a": scores(p[i], y[i])["iou"], "matched_b": scores(p[j], y[j])["iou"],
                "swapped_a": scores(p[i], y[j])["iou"], "swapped_b": scores(p[j], y[i])["iou"],
                "prediction_flip_rate": predictions_changed[m].float().mean().item(),
                "mean_logit_delta": diff[m].mean().item(),
                "difference_iou": scores(predictions_changed[m], y[i] ^ y[j])["iou"],
                "bounded_pixels": int(safe.sum()),
                "max_outside_logit_delta": diff[m][safe].max().item() if safe.any() else None,
                "one_hop_bound_applies": meta["mode"] != "L7",
            })
    append_rows(output / "counterfactual.csv", rows)
    return {
        "matched": mean((r["matched_a"] + r["matched_b"]) / 2 for r in rows),
        "swapped": mean((r["swapped_a"] + r["swapped_b"]) / 2 for r in rows),
        "prediction_flip_rate": mean(r["prediction_flip_rate"] for r in rows),
    }


def distance_tables(pred, data, meta):
    for name, key, bins, positive_only in (
        ("grid", "grid", GRID_BINS, False), ("window_graph", "window", WINDOW_BINS, False),
        ("pixel_bfs", "pixel", PIXEL_BINS, True),
    ):
        for key_b, mask in mask_bins(data[key], bins, outside=name != "grid"):
            result = scores(pred[mask], data["target"][mask])
            if positive_only:
                for metric in ("iou", "dice", "precision"):
                    result[metric] = None
            yield {**meta, "axis": name, "bucket": key_b, "pixels": int(mask.sum()), **result}
    if meta["mode"] != "L7":
        for label, mask in (("reachable", data["grid"] < meta["test_T"]),
                            ("unreachable", data["grid"] >= meta["test_T"])):
            yield {**meta, "axis": "grid_by_depth", "bucket": label, "pixels": int(mask.sum()),
                   **scores(pred[mask], data["target"][mask])}
    rank = data["rank"]
    for label, mask in (("low", rank < .45), ("middle", (rank >= .45) & (rank < .65)), ("high", rank >= .65)):
        yield {**meta, "axis": "root_eccentricity_rank", "bucket": label,
               "pixels": int(mask.sum()) * pred.shape[1] * pred.shape[2],
               **scores(pred[mask], data["target"][mask])}
    for label, mask in (("straight", data["target"] & ~data["bend"].bool()),
                        ("requires_turn", data["bend"].bool())):
        values = scores(pred[mask], data["target"][mask])
        for metric in ("iou", "dice", "precision"):
            values[metric] = None
        yield {**meta, "axis": "target_paths", "bucket": label, "pixels": int(mask.sum()), **values}


class LaneDiagnostics:
    def __init__(self):
        self.retentions = defaultdict(list)
        self.norms = defaultdict(list)
        self.interventions = defaultdict(lambda: [0., 0., 0., 0, 0])

    def add(self, frames, batch, model, turns, intervene):
        windows = tile(batch["image"][..., :1], 4)
        source, target = boundary_pairs(windows)
        opened = (source.bool() & target.bool()).squeeze(-1)
        valid = valid_edges(windows).bool().unsqueeze(-1).expand_as(opened)
        for turn in turns:
            frame = frames[turn - 1]
            prefix = frame.state.shape[:5]
            retained, component, y = frame.diagnostic
            retained = retained.unflatten(0, prefix)
            self.retentions[turn].append(retained.flatten(0, 2).cpu())
            total_y = y.unflatten(0, prefix).square().sum((-1, -2, -3))
            init_y = torch.zeros_like(total_y) if component is None else (
                component.unflatten(0, prefix).square().sum((-1, -2, -3))
            )
            axes = (0, 1, 2)
            sent = frame.sent if frame.sent is not None else frame.state
            bank = sent.square().sum((0, 1, 2, 5, 6, 7))
            incoming = torch.zeros_like(bank) if frame.initial is None else frame.initial.square().sum((0, 1, 2, 5, 6, 7))
            self.norms[turn].append((
                prefix[0] * prefix[1] * prefix[2] * sent.shape[5] * sent.shape[6] * sent.shape[7],
                bank.cpu(), incoming.cpu(), init_y.sum(axes).cpu(), total_y.sum(axes).cpu(),
                int((total_y == 0).sum()),
            ))
        if not intervene:
            return
        frame, old = frames[7], frames[6].state
        baseline = frame.logits
        _, _, raw = frame.diagnostic
        raw = raw.unflatten(0, frame.state.shape[:5])
        for variant in ("no_neighbor", "lane_permutation"):
            initial = neighbors(old, "line", model.gate, model.mixing,
                                permute=variant == "lane_permutation", zero=variant == "no_neighbor")
            changed = model.step(frame.feature_in, initial, diagnose=True)
            logit_change = (baseline - changed.logits).abs()
            prob_change = (baseline.sigmoid() - changed.logits.sigmoid()).abs()
            flip = (baseline >= 0) != (changed.logits >= 0)
            directional = (raw - changed.diagnostic[2].unflatten(0, frame.state.shape[:5])).abs()
            masks = list(mask_bins(batch["grid"], GRID_BINS))
            masks.extend([("straight", batch["target"] & ~batch["bend"].bool()),
                          ("requires_turn", batch["bend"].bool())])
            for label, mask in masks:
                values = self.interventions[variant, "fused_map", label, None, None]
                values[0] += logit_change[mask].sum().item()
                values[1] += prob_change[mask].sum().item()
                values[3] += int(flip[mask].sum())
                values[4] += int(mask.sum())
            for direction in range(4):
                for lane in range(4):
                    signal = directional[..., direction, lane, :, :, :]
                    for label, mask in (("open", opened & valid), ("closed", ~opened & valid),
                                        ("missing_neighbor", ~valid)):
                        values = self.interventions[variant, "directional_scan", label, direction, lane]
                        selected = signal[mask[..., direction, lane]]
                        values[2] += selected.sum().item()
                        values[4] += selected.numel()

    def write(self, output, meta):
        rows = []
        for t, arrays in self.retentions.items():
            q = torch.quantile(torch.cat(arrays), torch.tensor([.05, .5, .95]), dim=0)
            count = sum(x[0] for x in self.norms[t])
            sent, incoming, component, overall = (sum(item[i] for item in self.norms[t]) for i in range(1, 5))
            zero = sum(item[5] for item in self.norms[t])
            for d in range(4):
                for lane in range(4):
                    for head in range(2):
                        rows.append({
                            **meta, "test_T": t, "direction": d, "lane": lane, "head": head,
                            "retention_q05": q[0, d, lane, head].item(), "retention_q50": q[1, d, lane, head].item(),
                            "retention_q95": q[2, d, lane, head].item(),
                            "sent_or_state_rms_all_heads": (sent[d, lane] / count).sqrt().item(),
                            "initial_rms_all_heads": (incoming[d, lane] / count).sqrt().item(),
                            "initial_to_output_l2_all_heads": (component[d, lane] / overall[d, lane]).sqrt().item()
                            if overall[d, lane] else None, "zero_total_output_lanes": zero,
                        })
        append_rows(output / "retention.csv", rows)
        append_rows(output / "intervention.csv", [
            {**meta, "test_T": 8, "replacement": variant, "axis": axis, "bucket": label,
             "direction": direction, "lane": lane, "n": n,
             "mean_abs_logit_delta": logit / n if n and axis == "fused_map" else None,
             "mean_abs_probability_delta": prob / n if n and axis == "fused_map" else None,
             "mean_abs_directional_scan_delta": scan / n if n and axis == "directional_scan" else None,
             "flip_rate": flips / n if n and axis == "fused_map" else None}
            for (variant, axis, label, direction, lane), (logit, prob, scan, flips, n) in self.interventions.items()
        ])


@torch.no_grad()
def evaluate(stage, name, ds, seed, split, partition, cfg, output):
    rev = MODES[name]
    device = partition["image"].device
    model = model_for(rev, ds).to(device)
    model.load_state_dict(torch.load(checkpoint(output, stage, name, ds, seed), map_location=device, weights_only=True))
    model.eval()
    traces = defaultdict(list)
    diagnostic = LaneDiagnostics()
    is_lane = name in LANE_MODES
    times = []
    turns = cfg["eval_T"]
    for batch in batches(partition, cfg["eval_batch_size"]):
        synchronize(device)
        start = time.perf_counter()
        frames = model(batch["image"], max(turns), diagnose=is_lane)
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000 / len(batch["image"]))
        for turn in turns:
            logits = frames[turn - 1].logits
            if not logits.isfinite().all():
                raise FloatingPointError(f"Nonfinite {stage} {name}-d{ds}-s{seed}, {split}, T={turn}")
            traces[turn].append(logits.cpu())
        if is_lane:
            diagnostic.add(frames, batch, model, turns, name == "L3")
    meta = {"stage": stage, "split": split, "family": cfg["data"][split]["family"], "mode": name,
            "operator": rev, "dstate": ds, "seed": seed, "train_T": cfg["train_T"]}
    diagnostic.write(output, meta)
    append_rows(output / "evaluation_latency.csv", [
        {**meta, "depth": max(turns), "median_ms_per_image": median(times),
         "device": str(device), "batches": len(times), "retention_instrumentation": is_lane},
    ])
    result = []
    data = {key: value.cpu() for key, value in partition.items()}
    target = data["target"]
    for t, parts in traces.items():
        logits = torch.cat(parts)
        pred = logits >= 0
        info = {**meta, "test_T": t, "extrapolation": t > cfg["train_T"]}
        totals = confusion(pred.unflatten(0, (-1, 3)), target.unflatten(0, (-1, 3)))
        append_rows(output / "map_metrics.csv", [
            {**info, "map_index": index, "tp": tp, "fp": fp, "fn": fn}
            for index, (tp, fp, fn) in enumerate(totals.tolist())
        ])
        pair = paired(logits, data, info, output)
        row = {**info, **summary(pred, data), **pair, "token_visits_per_image": t * 32 * 32 * 4}
        append_rows(output / "metrics.csv", [row])
        append_rows(output / "distance_metrics.csv", list(distance_tables(pred, data, info)))
        result.append(row)
    t8 = next(r["iou"] for r in result if r["test_T"] == 8)
    print(f"EVAL {stage} {name}-d{ds}-s{seed} {split}: IoU@8={t8:.6f}", flush=True)
    return result


def map_bootstrap(config, output, stage, states):
    with (output / "map_metrics.csv").open(encoding="utf-8", newline="") as stream:
        groups = defaultdict(dict)
        for r in csv.DictReader(stream):
            if r["stage"] == stage and int(r["test_T"]) == 8:
                key = (r["split"], r["mode"], int(r["dstate"]), int(r["seed"]))
                groups[key][int(r["map_index"])] = [int(r[k]) for k in ("tp", "fp", "fn")]
    seeds = config["train_seeds"] if stage == "pilot" else [0]
    rows = []
    for split in TEST if stage == "pilot" else VAL:
        map_count = config["data"][split]["maps"]
        generator = torch.Generator().manual_seed(config["bootstrap_seed"])
        sample = torch.randint(map_count, (config["bootstrap_draws"], map_count), generator=generator)
        for state in states:
            for mode in ("L4", "L5", "L0", "L6", "L7"):
                ds = state if mode in ("L4", "L5") else 4
                def counts(label, size):
                    return torch.tensor([[groups[split, label, size, seed][i] for i in range(map_count)]
                                         for seed in seeds], dtype=torch.float64)
                left, right = counts("L3", state), counts(mode, ds)

                def iou(totals):
                    return totals[..., 0] / totals.sum(-1)

                effect = iou(left.sum(1)) - iou(right.sum(1))
                replicates = (iou(left[:, sample].sum(2)) - iou(right[:, sample].sum(2))).mean(0)
                ci = torch.quantile(replicates, torch.tensor([.025, .975], dtype=torch.float64))
                per_seed = effect.tolist()
                rows.append({
                    "stage": stage, "split": split, "test_T": 8, "dstate": state,
                    "comparison": mode, "comparison_dstate": ds, "mean_iou_delta": mean(per_seed),
                    "sample_sd": stdev(per_seed) if len(per_seed) > 1 else None,
                    "seed_deltas": ",".join(f"{x:.9g}" for x in per_seed),
                    "positive_seeds": sum(x > 0 for x in per_seed), "total_seeds": len(per_seed),
                    "map_bootstrap_ci95_low": ci[0].item(), "map_bootstrap_ci95_high": ci[1].item(),
                    "map_count": map_count, "draws": len(sample),
                })
    append_rows(output / "paired_bootstrap.csv", rows)
    return rows


def decide(config, selection, records, stage, bootstrap):
    chosen = selection["selected_dstate"]
    if stage != "pilot":
        return {"status": "PENDING_PILOT" if chosen is not None else "NOT_RUN",
                "reason": "Positive validation signal triggered the formal pilot" if chosen is not None
                else "No positive aligned-vs-unaligned validation signal; held-out evaluation prohibited"}
    seeds = config["train_seeds"]
    limits = config["gate"]
    results = {(r["split"], r["mode"], r["dstate"], r["seed"]): r
               for r in records if int(r["test_T"]) == 8}
    comparisons = {(r["split"], r["dstate"], r["comparison"]): r for r in bootstrap}
    tests = {}
    for mode in ("L4", "L5"):
        row = comparisons["test", chosen, mode]
        tests[f"id_against_{mode}"] = (row["mean_iou_delta"] >= limits["id_delta"]
                                      and row["positive_seeds"] >= limits["min_wins"])
    tests["id_raster_noninferior"] = comparisons["test", chosen, "L0"]["mean_iou_delta"] >= -limits["raster_tolerance"]
    alternate = 4 if chosen == 1 else 1
    tests["alternate_capacity_direction"] = all(comparisons["test", alternate, name]["mean_iou_delta"] > 0
                                                for name in ("L4", "L5"))
    far_and_distractor = {}
    for split in TEST:
        proposed = [results[split, "L3", chosen, seed] for seed in seeds]
        raster = [results[split, "L0", 4, seed] for seed in seeds]
        far = [a["far_recall"] - b["far_recall"] for a, b in zip(proposed, raster)]
        false = [a["distractor_coverage"] - b["distractor_coverage"] for a, b in zip(proposed, raster)]
        far_and_distractor[split] = {"far_recall_vs_raster": far, "distractor_coverage_vs_raster": false}
        tests[f"{split}_far_recall_gain"] = mean(far) >= limits["far_recall_gain"]
        tests[f"{split}_distractor_margin"] = mean(false) <= limits["max_distractor_increase"]
        if split != "test":
            tests[f"{split}_ood_gain"] = comparisons[split, chosen, "L0"]["mean_iou_delta"] >= limits["ood_delta_each"]
            tests[f"{split}_above_occupancy_baseline"] = mean(a["iou"] for a in proposed) > limits["ood_iou_each"]
    return {"status": "PASS" if all(tests.values()) else "FAIL", "selected_dstate": chosen,
            "checks": tests, "far_and_distractor": far_and_distractor}


def evaluation_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "screen", "pilot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = args.output
    stage = "pilot" if args.phase == "pilot" else "screen"
    marker = output / f"{stage}_evaluation_started.json"
    if marker.exists():
        raise FileExistsError(f"{stage} evaluation already attempted")
    config, data, provenance = setup(output, args.device, args.phase == "smoke")
    selection = read_json(output / "selection.json")
    if stage == "pilot" and selection["status"] == "STOP":
        print("Held-out evaluation NOT_RUN because selection stopped the pilot.", flush=True)
        return 0
    models = formal_matrix(selection["selected_dstate"]) if stage == "pilot" else screen_matrix()
    seeds = config["train_seeds"] if stage == "pilot" else [0]
    splits = TEST if stage == "pilot" else VAL
    with (output / "runs.csv").open(encoding="utf-8", newline="") as stream:
        saved = {(r["mode"], int(r["dstate"]), int(r["seed"])): r["checkpoint_sha256"]
                 for r in csv.DictReader(stream) if r["stage"] == stage and r["status"] == "PASS"}
    for mode, ds in models:
        for seed in seeds:
            key = mode, ds, seed
            digest = hashlib.sha256(checkpoint(output, stage, *key).read_bytes()).hexdigest()
            if digest != saved[key]:
                raise ValueError(f"Checkpoint hash mismatch at {stage} {key}")
    write_json(marker, provenance)
    records = []
    for name, ds in models:
        for seed in seeds:
            for split in splits:
                records.extend(evaluate(stage, name, ds, seed, split, data[split], config, output))
    comparison = map_bootstrap(config, output, stage, (1, 4))
    gate = decide(config, selection, records, stage, comparison)
    if stage == "screen":
        write_json(output / "screen_gate.json", gate)
    if stage == "pilot" or gate["status"] == "NOT_RUN":
        write_json(output / "gate.json", gate)
    print(gate, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluation_main())
