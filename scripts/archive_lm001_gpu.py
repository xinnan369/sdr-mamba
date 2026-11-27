"""Verify and archive completed LM-001 development evidence without data or weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scripts.audit_dataset import append_rows, write_json
from scripts.train_sdr003 import read_json
from sdr_mamba.crowd_data import sha_file


def close(left, right):
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("An aggregate disagrees with its per-image evidence")


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def extract(case, local, config, audit):
    seed, arm = case["seed"], case["arm"]
    root = local / "runs" / str(seed) / arm
    history = read_json(root / "history.json")
    checkpoints = read_json(root / "checkpoint_index.json")
    provenance = read_json(root / "run_manifest.json")
    expected = audit["cases"][str(seed)][arm]
    if (provenance["tracked_changes"] or provenance["data"]["sha256"] != config["reference"]["manifest_sha256"]
            or provenance["weight_initialization"]["sha256"] != config["reference"]["weight_sha256"]):
        raise ValueError("A completed case has unfrozen code or input provenance")
    if len(history) != len(checkpoints) or provenance["initial_state_sha256"] != expected["initial_sha256"]:
        raise ValueError("Checkpoint ledger or initialization mismatch")
    if sha_file(root / "last.pt") != checkpoints[-1]["sha256"]:
        raise ValueError("Last checkpoint hash differs")
    if "checkpoint_sha256" in case and (len(history) != 40 or case["checkpoint_sha256"] != checkpoints[-1]["sha256"]):
        raise ValueError("The completed case does not match its checkpoint")
    if len(history) == 40 and json_digest([rate for row in history for rate in row["learning_rates"]]) != expected["lr_sha256"]:
        raise ValueError("The learning-rate trajectory differs from the frozen audit")
    metrics, training, images = [], [], []
    for epoch, (row, ckpt) in enumerate(zip(history, checkpoints), 1):
        if (row["epoch"],row["global_step"],row["optimizer_updates"],row["amp_skipped_updates"],
            row["ema_step"],ckpt["epoch"],ckpt["step"]) != (epoch,180*epoch,180,0,180*epoch,epoch,180*epoch):
            raise ValueError("Interrupted, skipped or unmatched epoch in the completed ledger")
        if row["crop_order_sha256"] != expected["crop_epoch_sha256"][epoch-1]:
            raise ValueError("Crop hash mismatch in completed epoch")
        path = root / "images" / f"{epoch:02}.json"
        if sha_file(path) != row["image_metrics_sha256"]:
            raise ValueError("Image metrics provenance mismatch")
        group_rows = read_json(path)
        meta = {"seed": seed, "arm": arm, "epoch": epoch}
        train_fields = ("global_step","optimizer_updates","amp_skipped_updates","ema_step",
                        "crop_order_sha256","loss","final_loss","classification_loss","offset_loss",
                        "early_loss","train_seconds","epoch_seconds","peak_train_allocated_bytes",
                        "peak_train_reserved_bytes","gradient_l2_last_step")
        training.append({**meta, **{field: row.get(field) for field in train_fields},
                         "learning_rates_sha256": json_digest(row["learning_rates"]),
                         "checkpoint_sha256": ckpt["sha256"], "image_metrics_sha256": row["image_metrics_sha256"]})
        readouts = {"live": {"final": row["validation"]}, "ema": row["ema_validation"]}
        if "early_validation" in row:
            readouts["live"]["early"] = row["early_validation"]
        for policy, stages in readouts.items():
            for stage, table in stages.items():
                selected = [r for r in group_rows if r["stage"] == stage and r["policy"] == policy]
                if len(selected) != 40 or len({r["image_id"] for r in selected}) != 40:
                    raise ValueError("Missing or duplicate validation image")
                for group, result in {"all": table, **table["density_groups"]}.items():
                    subset = selected if group == "all" else [r for r in selected if r["density_group"] == group]
                    close(len(subset), result["images"])
                    if not subset:
                        continue
                    if any(r["predicted"]-r["gt"] != r["count_error"] or
                           r["local_tp"]+r["local_fp"] != r["predicted"] or
                           r["local_tp"]+r["local_fn"] != r["gt"] for r in subset):
                        raise ValueError("Per-image count or matching identity failed")
                    close(mean(abs(r["count_error"]) for r in subset), result["count"]["mae"])
                    close(math.sqrt(mean(r["count_error"]**2 for r in subset)), result["count"]["rmse"])
                    close(sum(r["count_error"] for r in subset), result["signed_count_error"])
                    metrics.append({**meta, "policy": policy, "stage": stage, "group": group, "images": len(subset),
                                    "mae": result["count"]["mae"], "rmse": result["count"]["rmse"],
                                    "bias": mean(r["count_error"] for r in subset), "nap": result["nap_local_v1"]["nap"],
                                    "fp": sum(r["local_fp"] for r in subset), "fn": sum(r["local_fn"] for r in subset)})
                for entry in selected:
                    public_id = hashlib.sha256(f"lm001:{entry['image_id']}".encode()).hexdigest()[:16]
                    images.append({**meta, "id": public_id, **{k: v for k,v in entry.items() if k != "image_id"}})
    return training, metrics, images


def summarize(output, metrics, config):
    primary = []
    for seed in map(int, config["order"]):
        for arm in config["arms"]:
            for policy in ("live","ema"):
                records = [r for r in metrics if r["arm"] == arm and r["seed"] == seed and r["group"] == "all" and
                           r["stage"] == "final" and r["policy"] == policy and r["epoch"] in config["analysis"]["epochs"]]
                if len(records) != 5:
                    continue
                report = {field: mean(r[field] for r in records) for field in ("mae","rmse","bias","nap","fp","fn")}
                primary.append({"seed": seed, "arm": arm, "policy": policy, **report,
                                "mae_sample_cv": stdev(r["mae"] for r in records)/report["mae"] if report["mae"] else 0})
    contrasts = []
    comparisons = [(arm, "cnn_only") for arm in config["arms"] if arm != "cnn_only"]
    comparisons += [("shared_2_mamba","shared_2_transformer"),("unshared_2_mamba","unshared_2_transformer"),
                    ("shared_2_transformer","unshared_2_transformer"),("shared_2_mamba","unshared_2_mamba")]
    ledger = {(r["seed"],r["arm"]): r for r in primary if r["policy"] == config["analysis"]["primary_policy"]}
    for left, right in comparisons:
        per_seed = [{k: ledger[seed,left][k]-ledger[seed,right][k] for k in ("mae","rmse","bias","nap","fp","fn")}
                    | {"seed": seed} for seed in map(int, config["order"]) if (seed,left) in ledger and (seed,right) in ledger]
        contrasts += [{"arm": left,"control": right,**row} for row in per_seed]
        if len(per_seed) == len(config["order"]):
            contrasts.append({"arm": left, "control": right, "seed": "equal_seed_mean",
                              **{k: mean(r[k] for r in per_seed) for k in ("mae","rmse","bias","nap","fp","fn")}})
    append_rows(output / "late_summary.csv", primary)
    append_rows(output / "paired_contrasts.csv", contrasts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-cpu", type=Path)
    args = parser.parse_args()
    config = read_json(Path("configs/lm001.json"))
    audit = read_json(Path("reports/LM-001/cpu_audit.json"))
    if args.focus_cpu:
        from scripts.run_lm001_a456 import focused_plan
        config, audit = focused_plan(), read_json(args.focus_cpu)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    preflight = read_json(args.local / "preflight.json")
    write_json(output / "gpu_preflight.json", preflight)
    if not (args.local / "queue.json").exists():
        write_json(output / "gpu_decision.json", {"status": "BLOCKED" if preflight["status"] != "PASS" else "NOT_RUN",
                                                 "completed": 0, "optimizer_updates": 0, "official_test": "NOT_RUN"})
        return
    queue = read_json(args.local / "queue.json")
    if queue["status"] == "RUNNING":
        raise ValueError("Finish or stop the live queue before authoring the archive")
    cases = list(queue["completed"])
    if queue["current"]:
        current = queue["current"]
        path = args.local / "runs" / str(current["seed"]) / current["arm"]
        if (path / "history.json").exists():
            cases.append(current)
    metrics, training, images = [], [], []
    for case in cases:
        tr, mt, im = extract(case, args.local, config, audit)
        training.extend(tr)
        metrics.extend(mt)
        images.extend(im)
    for name, records in (("training", training),("metrics", metrics),("per_image", images)):
        append_rows(output / f"gpu_{name}.csv", records)
    summarize(output, metrics, config)
    planned = sum(map(len, config["order"].values()))
    if queue["status"] == "PASS" and (len(cases) != planned or len(training) != planned*40
                                    or queue["updates"]-queue.get("prior_updates",0) != planned*7200):
        raise ValueError("A complete queue must match the frozen size, epochs and updates")
    public = {k: v for k,v in queue.items() if k != "reason"}
    write_json(output / "gpu_decision.json", public | {"verified_epochs": len(training), "image_rows": len(images),
                                                       "metric_rows": len(metrics)})
    phases = args.local / "gpu_phases.jsonl"
    if phases.exists():
        (output / phases.name).write_bytes(phases.read_bytes())
    print(f"Verified {len(training)} epochs, {len(metrics)} grouped metrics and {len(images)} image rows")


if __name__ == "__main__":
    main()
