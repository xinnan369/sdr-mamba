"""One-pass held-out diagnostics and pre-registered stage decisions."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
import time

import torch

from scripts.audit_dataset import append_rows, scores, write_json
from scripts.run_sdr002 import batches
from scripts.sdr002_diagnostics import mask_bins
from scripts.train_sdr003 import A, B, checkpoint, read_json, setup
from sdr_mamba.data_v2 import PIXEL_BINS, WINDOW_BINS
from sdr_mamba.model_v3 import LoopV3, hard_gate, transport, valid_edges
from sdr_mamba.scan import tile
from sdr_mamba.state_transport import upstream

GRID_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 11), (12, None))


def ratio(n, d):
    return n / d if d else None


def synctime(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def auc(score, label):
    if not label.any() or label.all():
        return None
    _, groups, counts = score.unique(sorted=True, return_inverse=True, return_counts=True)
    positives = torch.bincount(groups[label], minlength=len(counts)).float()
    negatives = counts - positives
    return ((positives * (negatives.cumsum(0) - negatives * .5)).sum()
            / (positives.sum() * negatives.sum())).item()


class Diagnostics:
    def __init__(self, model, compare_boundaries=False):
        self.model = model
        self.compare_boundaries = compare_boundaries
        self.statistics = defaultdict(list)
        self.retention = defaultdict(list)
        self.edges = defaultdict(list)
        self.intervention = defaultdict(lambda: [0., 0, 0])

    def add(self, frames, batch):
        model = self.model
        image = batch["image"]
        original = frames[0].f_in
        hard = hard_gate(image, model.k).bool()
        legal = valid_edges(tile(image, model.k)).bool()
        for t, frame in enumerate(frames, 1):
            retained, readout, output = frame.diagnostic
            self.retention[t].append(retained.cpu())
            output_sq = output.square().sum((1, 3))
            init_sq = torch.zeros_like(output_sq) if readout is None else readout.square().sum((1, 3))
            values = [
                frame.f_out.square().mean().item(), (frame.f_out - frame.f_in).square().mean().item(),
                (frame.f_out - original).square().mean().item(), frame.logits.square().mean().item(),
                frame.state.square().mean().item(),
                0.0 if frame.initial is None else frame.initial.square().mean().item(),
                init_sq.sum().item(), output_sq.sum().item(), int((output_sq == 0).sum()),
            ]
            self.statistics[t].append((len(image), values))
            if t > 1 and (frame.edge is not None or self.compare_boundaries):
                previous = frames[t - 2]
                edge = frame.edge if frame.edge is not None else legal.to(image.dtype)
                if model.strong is None:
                    payload = previous.state if previous.sent is None else previous.sent
                    sources = torch.stack([upstream(p, d) for d, p in enumerate(payload.unbind(3))], dim=3)
                    channel_power = sources.square().mean((-1, -2, -3)) * (model.gate * model.mixing * edge).square()
                else:
                    channel_power = model.strong(frame.f_in, model.k).square().mean((-1, -2))
                self.edges[t].append((edge[legal].cpu(), hard[legal].cpu(), channel_power[legal].cpu()))
            if t in (4, 8) and frame.initial is not None:
                previous = frames[t - 2].state
                for replacement in ("zero", "shuffled"):
                    changed = transport(previous, model.gate, model.mixing, frame.edge, replacement)
                    other = model.step_v3(original, frame.f_in, changed).logits
                    delta = (frame.logits.sigmoid() - other.sigmoid()).abs()
                    flip = (frame.logits >= 0) != (other >= 0)
                    for axis, bins in (("grid", GRID_BINS), ("graph", WINDOW_BINS)):
                        distance = batch["grid" if axis == "grid" else "window"]
                        for name, mask in mask_bins(distance, bins, outside=axis == "graph"):
                            values = self.intervention[t, replacement, axis, name]
                            values[0] += delta[mask].sum().item()
                            values[1] += int(flip[mask].sum())
                            values[2] += int(mask.sum())

    def log(self, output, meta):
        stat_rows = []
        for t, batches_at_t in self.statistics.items():
            count = sum(n for n, _ in batches_at_t)
            pooled = [sum(n * v[i] for n, v in batches_at_t) / count for i in range(6)]
            numer, denom, zero = (sum(v[i] for _, v in batches_at_t) for i in range(6, 9))
            stat_rows.append({
                **meta, "round": t, **{key: power ** .5 for key, power in zip((
                    "feature_rms", "update_rms", "anchor_drift_rms", "logit_rms",
                    "final_state_rms", "initial_state_rms",
                ), pooled)},
                "initial_to_full_output_l2": (numer / denom) ** .5 if denom else None,
                "zero_output_window_heads": int(zero),
            })
        append_rows(output / "feature_drift.csv", stat_rows)
        retention_rows = []
        for t, batch_list in self.retention.items():
            vals = torch.cat(batch_list)
            quantiles = torch.quantile(vals, torch.tensor([.05, .5, .95]), dim=0).T
            retention_rows.extend(
                {**meta, "round": t, "head": h, "q05": row[0], "q50": row[1], "q95": row[2]}
                for h, row in enumerate(quantiles.tolist())
            )
        append_rows(output / "retention.csv", retention_rows)
        gate_rows = []
        for t, batches_at_t in self.edges.items():
            prob, label, leakage = (torch.cat(parts) for parts in zip(*batches_at_t))
            false_open, false_close = (prob >= .5) & ~label, (prob < .5) & label
            explicit_gate = self.model.strong is None
            gate_rows.append({
                **meta, "round": t, "edge_auroc": auc(prob, label) if explicit_gate else None,
                "positive_edges": int(label.sum()), "negative_edges": int((~label).sum()),
                "gate_mean_positive": prob[label].mean().item() if explicit_gate and label.any() else None,
                "gate_mean_negative": prob[~label].mean().item() if explicit_gate and (~label).any() else None,
                "false_open": ratio(int(false_open.sum()), int((~label).sum())) if explicit_gate else None,
                "false_close": ratio(int(false_close.sum()), int(label.sum())) if explicit_gate else None,
                "closed_edge_payload_rms": leakage[~label].mean().sqrt().item() if (~label).any() else None,
                "open_edge_payload_rms": leakage[label].mean().sqrt().item() if label.any() else None,
            })
        append_rows(output / "gate_diagnostics.csv", gate_rows)
        append_rows(output / "state_intervention.csv", [
            {**meta, "round": t, "replacement": variant, "distance_axis": axis, "bucket": bucket,
             "probability_delta": ratio(delta, n), "flip_rate": ratio(flips, n), "pixels": n}
            for (t, variant, axis, bucket), (delta, flips, n) in self.intervention.items()
        ])


def counterfactual(logits, dataset, meta, output):
    logit = logits.unflatten(0, (-1, 3))
    targets = dataset["target"].cpu().unflatten(0, (-1, 3))
    grids = dataset["grid"].cpu().unflatten(0, (-1, 3))
    pred = logit >= 0
    rows, maps = [], []
    for left, right in combinations(range(3), 2):
        flip = pred[:, left] ^ pred[:, right]
        gt_change = targets[:, left] ^ targets[:, right]
        far = grids[:, left].minimum(grids[:, right]) >= meta["test_T"]
        delta = (logit[:, left] - logit[:, right]).abs()
        for index in range(len(pred)):
            matching = [scores(pred[index, source], targets[index, same])["iou"]
                        for source, same in ((left, left), (right, right), (left, right), (right, left))]
            rows.append({
                **meta, "map": index, "seed_a": left, "seed_b": right,
                "matched_a": matching[0], "matched_b": matching[1],
                "swapped_a": matching[2], "swapped_b": matching[3],
                "flip_rate": flip[index].float().mean().item(),
                "change_iou": scores(flip[index], gt_change[index])["iou"],
                "outside_bound_total": int(far[index].sum()),
                "outside_bound_max_logit_delta": delta[index][far[index]].max().item() if far[index].any() else None,
            })
        d_win = delta.unflatten(1, (8, 4)).unflatten(-1, (8, 4)).mean((0, 2, 4))
        maps.extend({**meta, "seed_a": left, "seed_b": right,
                     "window_y": y, "window_x": x, "mean_abs_logit_delta": d_win[y, x].item()}
                    for y in range(8) for x in range(8))
    append_rows(output / "counterfactual.csv", rows)
    append_rows(output / "seed_influence_maps.csv", maps)
    return (mean((r["matched_a"] + r["matched_b"]) / 2 for r in rows),
            mean((r["swapped_a"] + r["swapped_b"]) / 2 for r in rows), mean(r["flip_rate"] for r in rows))


@torch.no_grad()
def evaluate(group, seed, operator, protocol, config, dataset, split, env, output):
    depth, _, anchored = protocol
    device = torch.device(env["device"])
    model = LoopV3(operator, anchored).to(device)
    model.load_state_dict(torch.load(checkpoint(output, group, seed), map_location=device, weights_only=True))
    model.eval()
    turns = config["test_T"] if split != "validation" else [4, 6, 8]
    traces = defaultdict(list)
    diagnostics = Diagnostics(model, compare_boundaries=group.startswith("B"))
    latency = []
    for batch in batches(dataset, config["eval_batch_size"]):
        image = batch["image"]
        start = synctime(device)
        frames = model(image, max(turns), diagnose=True)
        latency.append((synctime(device) - start) * 1000 / len(image))
        for t, frame in enumerate(frames, 1):
            if not frame.logits.isfinite().all():
                raise FloatingPointError(f"{group}-s{seed} {split} nonfinite logits at round {t}")
            traces[t].append(frame.logits.cpu())
        diagnostics.add(frames, batch)
    meta = {"stage": group[0], "mode": group, "operator": operator, "seed": seed,
            "family": "nested_open_u" if split == "ood" else "fresh_id", "split": split, "train_T": depth}
    append_rows(output / "evaluation_latency.csv", [
        {**meta, "profile_depth": max(turns), "median_instrumented_forward_ms_per_image": median(latency),
         "batches": len(latency), "device": str(device)},
    ])
    diagnostics.log(output, meta)
    t_data = {name: t.cpu() for name, t in dataset.items()}
    labels = t_data["target"]
    distractor = t_data["image"][..., 0].bool() & ~labels
    records = []
    for t, parts in traces.items():
        logit = torch.cat(parts)
        pred = logit >= 0
        record = {**meta, "test_T": t, "extrapolation": t > depth}
        fields = scores(pred, labels)
        far = (t_data["grid"] >= 4) & (t_data["grid"] <= 7) & labels
        reachable = (t_data["grid"] < t) & labels
        reachable_far = reachable & far
        summary = {
            **record, **fields,
            "far_4_7_recall": ratio(int((pred & far).sum()), int(far.sum())),
            "reachable_far_recall": ratio(int((pred & reachable_far).sum()), int(reachable_far.sum())),
            "distractor_recall": ratio(int((pred & distractor).sum()), int(distractor.sum())),
            "reachable_recall": ratio(int((pred & reachable).sum()), int(reachable.sum())),
            "scan_tokens_per_image": t * 32 * 32 * 4,
        }
        append_rows(output / "per_round_metrics.csv", [summary])
        if t not in turns:
            continue
        matched, swapped, flip = counterfactual(logit, dataset, record, output)
        summary.update(paired_matched_iou=matched, paired_swapped_iou=swapped, seed_swap_flip_rate=flip)
        append_rows(output / "metrics.csv", [summary])
        records.append(summary)
        rows = []
        for axis, field, bins in (("grid", "grid", GRID_BINS), ("graph", "window", WINDOW_BINS),
                                  ("pixel", "pixel", PIXEL_BINS)):
            for bucket, mask in mask_bins(t_data[field], bins, outside=axis != "grid"):
                result = scores(pred[mask], labels[mask])
                if axis == "pixel":
                    for key in ("iou", "dice", "precision"):
                        result[key] = None
                rows.append({**record, "axis": axis, "bucket": bucket, "pixels": int(mask.sum()), **result})
        for name, mask in (("pointwise_reachable", t_data["grid"] < t),
                           ("pointwise_unreachable", t_data["grid"] >= t)):
            rows.append({**record, "axis": "grid_dynamic", "bucket": name, "pixels": int(mask.sum()),
                         **scores(pred[mask], labels[mask])})
        append_rows(output / "distance_metrics.csv", rows)
    print(f"EVAL {group}-s{seed} {split} T8={next(r['iou'] for r in records if r['test_T']==8):.5f}", flush=True)
    return records


def stage_a_gate(config, records, selected):
    reference = {(r["mode"], r["seed"], r["test_T"]): r for r in records if r["split"] == "test"}
    seeds = config["seeds"]

    def paired(t, key, candidate=selected, baseline="A1"):
        return [reference[candidate, s, t][key] - reference[baseline, s, t][key] for s in seeds]

    at_eight = paired(8, "iou")
    extra = [(a + b) / 2 for a, b in zip(paired(10, "iou"), paired(12, "iou"))]
    far = paired(8, "far_4_7_recall")
    distractor = paired(8, "distractor_recall")
    limits = config["a_gate"]
    checks = {"t8_tolerance": mean(at_eight) >= -limits["t8_tolerance"],
              "extrapolation_gain": mean(extra) >= limits["extrapolation_gain"],
              "extrapolation_wins": sum(v > 0 for v in extra) >= limits["min_wins"],
              "far_recall": mean(far) >= limits["far_recall_gain"],
              "distractors": mean(distractor) <= limits["max_distractor_increase"]}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL", "selected": selected, "checks": checks,
        "id_paired_deltas": {"t8_iou": at_eight, "mean_t10_t12": extra, "far_recall": far, "distractor": distractor},
        "depth_only_A1_minus_A0": {"t8_iou": paired(8, "iou", "A1", "A0"),
                                  "t8_far_recall": paired(8, "far_4_7_recall", "A1", "A0")},
    }


def stage_b_gate(config, records, formal):
    reference = {(r["mode"], r["seed"], r["split"]): r for r in records if r["test_T"] == 8}
    if not formal:
        return {"status": "NOT_RUN", "reason": "Stage A gate failed; seed-0 validation diagnostic only",
                "validation_t8": {mode: reference[mode, 0, "validation"]["iou"] for mode in B}}
    limits = config["b_gate"]
    comparisons = {}
    of = lambda mode, seed, split, key: reference[mode, seed, split][key]
    for mode in ("B0", "B3", "B5", "B6"):
        fields = {}
        for split in ("test", "ood"):
            for key in ("iou", "far_4_7_recall", "distractor_recall"):
                fields[f"{split}_{key}"] = [of("B1", s, split, key) - of(mode, s, split, key)
                                           for s in config["seeds"]]
        fields["pass"] = (mean(fields["test_iou"]) >= limits["id_gain"]
                          and sum(x > 0 for x in fields["test_iou"]) >= limits["min_wins"]
                          and mean(fields["ood_iou"]) > 0 and mean(fields["test_far_4_7_recall"]) > 0
                          and mean(fields["ood_far_4_7_recall"]) > 0
                          and mean(fields["test_distractor_recall"]) <= limits["max_distractor_increase"]
                          and mean(fields["ood_distractor_recall"]) <= limits["max_distractor_increase"])
        comparisons[mode] = fields
    return {"status": "PASS" if all(v["pass"] for v in comparisons.values()) else "FAIL",
            "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sdr003.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--phase", choices=("a", "a-validation", "b", "smoke"), required=True)
    args = parser.parse_args()
    output = args.output
    phase = args.phase
    if (output / f"{phase}_evaluation_started.json").exists():
        raise FileExistsError("Evaluation already attempted; preserve previous output")
    config, datasets, env = setup(args.config, output, args.device, phase == "smoke")
    chosen = read_json(output / "a_selection.json")["selected"]
    formal = phase == "a" or (phase == "b" and read_json(output / "a_gate.json")["status"] == "PASS")
    if phase == "a-validation":
        modes = dict.fromkeys(A, "sst")
    else:
        modes = {group: "sst" for group in dict.fromkeys(("A0", "A1", chosen))} if phase == "a" else B
    seeds = config["seeds"] if formal else [0]
    missing = [str(checkpoint(output, group, seed)) for group in modes for seed in seeds
               if not checkpoint(output, group, seed).is_file()]
    if missing:
        raise FileNotFoundError(f"Required checkpoints missing: {missing}")
    write_json(output / f"{phase}_evaluation_started.json", env)
    records = []
    for group, operator in modes.items():
        protocol = A[group] if phase in ("a", "a-validation") else A[chosen]
        for seed in seeds:
            for split in ("test", "ood") if formal else ("validation",):
                records.extend(evaluate(group, seed, operator, protocol, config, datasets[split], split, env, output))
    if phase == "a-validation":
        print("Seed-0 validation-only mechanism diagnostics complete", flush=True)
        return 0
    gate = stage_a_gate(config, records, chosen) if phase == "a" else stage_b_gate(config, records, formal)
    write_json(output / f"{phase}_gate.json", gate)
    print(gate, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
