"""Fixed-budget validation screen and conditional LAST pilot."""

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import platform
from statistics import mean
import time
import traceback

import torch

from scripts.audit_dataset import append_rows, audit, digest_config, measurements, scores, write_json
from scripts.run_sdr002 import batches, objective
from scripts.train_sdr003 import git, read_json
from sdr_mamba.data_v4 import build
from sdr_mamba.model_v4 import model_for

MODES = {"L0": "raster", "L0b": "raster_same", "L1": "reset", "L2": "local", "L3": "line",
         "L4": "line_wrong", "L5": "line_feature", "L6": "strong", "L7": "global"}
VAL = ("validation", "u_validation", "branch_validation")
TEST = ("test", "u_test", "branch_test")


def screen_matrix():
    return [("L0", 4), ("L0b", 4), ("L1", 1), ("L2", 1),
            *((name, state) for state in (1, 4) for name in ("L3", "L4", "L5")),
            ("L6", 4), ("L7", 4)]


def formal_matrix(state):
    return [(name, state if name in ("L1", "L2") else ds) for name, ds in screen_matrix()]


def setup(output, device, smoke=False):
    config = read_json(Path("configs/sdr004.json"))
    if smoke:
        config.update(screen_updates=2, period=2)
        for spec in config["data"].values():
            spec.update(seed=spec["seed"] + 300000, maps=4)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if (output / "effective_config.json").exists() and read_json(output / "effective_config.json") != config:
        raise ValueError("Different configuration for existing output directory")
    data, manifest = build(config)
    audit_config = {"validation_gate": {"all_walkable_max": .55},
                    "baselines": {"random_seed": 604219, "random_positive_probability": 1 / 3}}
    result = audit(data, manifest, audit_config)
    for split, metrics in result["splits"].items():
        rows = manifest[split]["maps"]
        cases = [s for x in rows for s in x["samples"]]
        far = (data[split]["grid"] >= 4) & (data[split]["grid"] <= 7)
        unsupported = data[split]["grid"] >= 8
        actual = [s["eccentricity_rank"] for s in cases]
        ecc = [s["target_eccentricity"] for s in cases]
        sizes = [x["component_areas"][0] for x in rows]
        metrics.update({
            "family": manifest[split]["family"], "component_area_mean": mean(sizes),
            "eccentricity_rank_mean": mean(actual), "eccentricity_range": [min(ecc), max(ecc)],
            "mean_absolute_eccentricity_rank_error": mean(abs(s["requested_rank"] - s["eccentricity_rank"]) for s in cases),
            "far_target_pixels": int((far & data[split]["target"]).sum()),
            "grid_t8_unreachable_target_pixels": int((unsupported & data[split]["target"]).sum()),
            "rejection_rate": sum(manifest[split]["rejections"].values())
            / (len(rows) + sum(manifest[split]["rejections"].values())),
        })
    config_hash = digest_config(config)
    result["config_sha256"] = config_hash
    prov = {
        "code_commit": git("rev-parse", "HEAD"), "config_sha256": config_hash,
        "dataset_sha256": hashlib.sha256("".join(p["sha256"] for p in manifest.values()).encode()).hexdigest(),
        "device": device, "torch": torch.__version__, "cuda_build": torch.version.cuda,
    }
    if not (output / "environment.txt").exists():
        write_json(output / "environment.txt", {
            **prov, "tracked_dirty": git("status", "--porcelain", "--untracked-files=no"),
            "platform": platform.platform(), "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "official_ssd": "NOT_RUN",
        })
    write_json(output / "effective_config.json", config)
    write_json(output / "data_manifest.json", manifest)
    write_json(output / "data_audit.json", result)
    print(f"Data audit {result['status']}: {prov['dataset_sha256']}", flush=True)
    if result["status"] != "PASS":
        raise ValueError("SDR-004 structural audit failed")
    return config, {name: {key: t.to(device) for key, t in part.items()} for name, part in data.items()}, prov


def checkpoint(output, stage, name, ds, seed):
    return output / "checkpoints" / f"{stage}-{name}-d{ds}-s{seed}.pt"


def summary(pred, split):
    target = split["target"]
    distractor = split["image"][..., 0].bool() & ~target
    far = target & (split["grid"] >= 4) & (split["grid"] <= 7)
    hit = int((pred & far).sum())
    return {**scores(pred, target),
            "far_recall": hit / int(far.sum()), "distractor_coverage": (pred & distractor).sum().item() / int(distractor.sum())}


@torch.no_grad()
def validate(model, split, config):
    model.eval()
    guesses, accumulated = [], 0.
    for batch in batches(split, config["eval_batch_size"]):
        logits = model(batch["image"], config["train_T"])[-1].logits
        if not logits.isfinite().all():
            raise FloatingPointError("Nonfinite validation logits")
        accumulated += objective(logits, batch["target"]).item() * len(logits)
        guesses.append(logits >= 0)
    return {"loss": accumulated / len(split["target"]), **summary(torch.cat(guesses), split)}


def fit(stage, name, ds, seed, config, data, prov, output):
    mode = MODES[name]
    depth = config["train_T"]
    budget = config["updates"] if stage == "pilot" else config["screen_updates"]
    artifact = checkpoint(output, stage, name, ds, seed)
    meta = {"stage": stage, "mode": name, "operator": mode, "dstate": ds, "seed": seed, "train_T": depth}
    if artifact.exists():
        raise FileExistsError(artifact)
    torch.manual_seed(seed)
    model = model_for(mode, ds).to(prov["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    shuffle = torch.Generator().manual_seed(config["order_seed"] + seed)
    train = data["train"]
    if prov["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    n = 0
    status, problem, checkpoint_hash = "FAIL", "", ""
    matrix = []
    print(f"START {stage} {name} d={ds} seed={seed} steps={budget}", flush=True)
    try:
        initial = validate(model, data["validation"], config)
        append_rows(output / "train_curves.csv", [{**meta, "step": 0,
                    "split": "validation", "grad_l2": None, **initial}])
        aggregate = torch.zeros(4, device=prov["device"])
        interval = 0
        model.train()
        for step in range(1, budget + 1):
            ids = torch.randint(len(train["target"]), (config["batch_size"],), generator=shuffle).to(prov["device"])
            x, y = train["image"][ids], train["target"][ids]
            logits = model(x, depth)[-1].logits
            loss = objective(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            capture = step % config["period"] == 0 or step == budget
            if capture:
                grad = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm().item()
                if not math.isfinite(loss.item()) or not math.isfinite(grad):
                    raise FloatingPointError(f"Nonfinite loss/gradient at update {step}")
            optimizer.step()
            n = step
            with torch.no_grad():
                pred = logits >= 0
                aggregate += torch.stack((loss, (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()))
            interval += 1
            if capture:
                loss_sum, tp, fp, fn = aggregate.tolist()
                end_val = validate(model, data["validation"], config)
                append_rows(output / "train_curves.csv", [
                    {**meta, "step": step, "split": "train", "grad_l2": grad, "loss": loss_sum / interval,
                     **measurements(int(tp), int(fp), int(fn)), "far_recall": None, "distractor_coverage": None},
                    {**meta, "step": step, "split": "validation", "grad_l2": None, **end_val},
                ])
                print(f"{stage} {name}-d{ds}-s{seed} {step}/{budget} loss={loss_sum/interval:.5f} "
                      f"ID val IoU={end_val['iou']:.5f}", flush=True)
                model.train()
                aggregate.zero_()
                interval = 0
        if stage == "screen":
            for split in VAL:
                result = end_val if split == "validation" else validate(model, data[split], config)
                matrix.append({**meta, "split": split, "test_T": depth, **result})
            append_rows(output / "screen_metrics.csv", matrix)
        torch.save(model.state_dict(), artifact)
        checkpoint_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        status = "PASS"
    except Exception as error:
        traceback.print_exc()
        problem = repr(error)
    peak = torch.cuda.max_memory_allocated() if prov["device"] == "cuda" else None
    append_rows(output / "runs.csv", [{
        **meta, **prov, "updates": n, "status": status, "error": problem,
        "params": sum(p.numel() for p in model.parameters()), "token_visits": n * config["batch_size"] * depth * 32 * 32 * 4,
        "peak_cuda_bytes": peak, "wall_time_s": time.perf_counter() - start, "checkpoint_sha256": checkpoint_hash,
    }])
    print(f"{status} {stage} {name}-d{ds}-s{seed} updates={n}", flush=True)
    return status == "PASS"


def select(config, output):
    with (output / "screen_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = {(r["mode"], int(r["dstate"]), r["split"]): r for r in csv.DictReader(stream)}
    candidates = {}
    for state in (1, 4):
        by_split = {}
        for split in VAL:
            a, b, c = (rows[mode, state, split] for mode in ("L3", "L4", "L5"))
            by_split[split] = {
                "mean_iou": float(a["iou"]), "aligned_minus_unaligned": float(a["iou"]) - float(b["iou"]),
                "aligned_minus_feature": float(a["iou"]) - float(c["iou"]),
                "far_recall": float(a["far_recall"]), "distractor_coverage": float(a["distractor_coverage"]),
            }
        score = sum(weight * mean(row[key] for row in by_split.values())
                    for key, weight in config["selection"]["score_weights"].items())
        eligible = any(row["aligned_minus_unaligned"] > 0 for row in by_split.values())
        candidates[state] = {"score": score, "eligible": eligible, "splits": by_split}
    pool = [state for state in candidates if candidates[state]["eligible"]]
    best = max((candidates[s]["score"] for s in pool), default=None)
    winner = next((s for s in pool if best - candidates[s]["score"] <= config["selection"]["tie_margin"]), None)
    return {"status": "CONTINUE" if winner is not None else "STOP", "selected_dstate": winner,
            "rule": "Any positive L3-vs-L4 validation split; select balanced score with lower storage tie-break",
            "candidates": candidates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "screen", "pilot", "smoke"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    config, data, provenance = setup(args.output, args.device, args.phase == "smoke")
    if args.phase == "audit":
        return 0
    if args.phase in ("smoke", "screen"):
        results = [fit("screen", name, ds, 0, config, data, provenance, args.output)
                   for name, ds in screen_matrix()]
        if not all(results):
            return 1
        decision = select(config, args.output)
        write_json(args.output / "selection.json", decision)
        print(decision, flush=True)
        return 0
    decision = read_json(args.output / "selection.json")
    if decision["status"] == "STOP":
        print("Pilot NOT_RUN because no lane alignment signal was observed; held-out models not evaluated.", flush=True)
        return 0
    results = [fit("pilot", name, ds, seed, config, data, provenance, args.output)
               for seed in config["train_seeds"] for name, ds in formal_matrix(decision["selected_dstate"])]
    return int(not all(results))


if __name__ == "__main__":
    raise SystemExit(main())
