"""Run the validation gate before dispatching the SDR-002 pilot."""

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import torch
from torch.nn import functional as F

from scripts.audit_dataset import append_rows, audit, digest_config, measurements, scores, write_json
from scripts.sdr002_diagnostics import DiagnosticTotals, component_rows, distance_rows, gradients, paired_rows
from sdr_mamba.data_v2 import generate
from sdr_mamba.model_v2 import WindowSSM
from sdr_mamba.state_transport import MODES


def repository_state(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def set_seed_channel(image, variant):
    if variant == "full":
        return image
    return torch.stack((image[..., 0], torch.zeros_like(image[..., 1])), dim=-1)


def objective(logits, target):
    label = target.float()
    p = logits.sigmoid()
    dice = (2 * (p * label).sum((1, 2)) + 1) / (p.sum((1, 2)) + label.sum((1, 2)) + 1)
    return F.binary_cross_entropy_with_logits(logits, label) + (1 - dice).mean()


def batches(data, size):
    for start in range(0, len(data["target"]), size):
        yield {name: t[start:start + size] for name, t in data.items()}


@torch.no_grad()
def validate(model, dataset, mode, variant, config, diagnostic=None):
    predictions, weighted_loss = [], 0
    model.eval()
    for batch in batches(dataset, config["eval_batch_size"]):
        image = set_seed_channel(batch["image"], variant)
        logits, trace = model(image, config["train_T"], mode, return_trace=True)
        predictions.append(logits >= 0)
        weighted_loss += objective(logits, batch["target"]).item() * len(logits)
        if diagnostic is not None:
            diagnostic.add_interventions(model, trace, batch["window"])
    pred = torch.cat(predictions)
    return {"loss": weighted_loss / len(pred), **scores(pred, dataset["target"])}, pred


def train(model, data, config, metadata, output):
    run_id, seed, mode, variant = (metadata[key] for key in ("run_id", "seed", "mode", "variant"))
    generator = torch.Generator().manual_seed(1900 + seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    start = time.perf_counter()
    initial_val, _ = validate(model, data["validation"], mode, variant, config)
    append_rows(output / "train_curves.csv", [
        {"run_id": run_id, "step": 0, "split": "validation", **initial_val},
    ])
    images, targets = data["train"]["image"], data["train"]["target"]
    running_loss, running_tp, running_fp, running_fn = [torch.zeros((), device=images.device) for _ in range(4)]
    interval = 0
    model.train()
    for step in range(1, config["steps"] + 1):
        ids = torch.randint(len(images), (config["batch_size"],), generator=generator).to(images.device)
        image, target = set_seed_channel(images[ids], variant), targets[ids]
        logits, trace = model(image, config["train_T"], mode, return_trace=True)
        loss = objective(logits, target)
        capture = step % config["period"] == 0 or step == config["steps"]
        optimizer.zero_grad(set_to_none=True)
        if capture:
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            append_rows(output / "gradient_diagnostics.csv", gradients(
                model, trace, [objective(frame.logits, target) for frame in trace], loss,
                {"run_id": run_id, "step": step},
            ))
        loss.backward()
        if capture:
            grad_l2 = torch.stack([p.grad.norm() for p in model.parameters()]).norm().item()
            if not math.isfinite(grad_l2):
                raise FloatingPointError(f"Nonfinite parameter gradient at step {step}")
            append_rows(output / "parameter_gradients.csv", [
                {"run_id": run_id, "step": step, "parameter": name, "grad_l2": p.grad.norm().item()}
                for name, p in model.named_parameters()
            ])
        optimizer.step()
        with torch.no_grad():
            pred = logits >= 0
            running_loss += loss.detach()
            running_tp += (pred & target).sum()
            running_fp += (pred & ~target).sum()
            running_fn += (~pred & target).sum()
        interval += 1
        if capture:
            train_result = {"loss": running_loss.item() / interval,
                            **measurements(int(running_tp), int(running_fp), int(running_fn))}
            diagnostic = DiagnosticTotals() if mode == "sst" and variant == "full" else None
            validation_result, _ = validate(model, data["validation"], mode, variant, config, diagnostic)
            if diagnostic is not None:
                append_rows(output / "intervention.csv",
                            diagnostic.interventions({"run_id": run_id, "step": step, "split": "validation"}))
            append_rows(output / "train_curves.csv", [
                {"run_id": run_id, "step": step, "split": "train", **train_result},
                {"run_id": run_id, "step": step, "split": "validation", **validation_result},
            ])
            print(f"{run_id} step {step} train loss={train_result['loss']:.5f} "
                  f"validation loss={validation_result['loss']:.5f} IoU={validation_result['iou']:.5f}", flush=True)
            for stat in (running_loss, running_tp, running_fp, running_fn):
                stat.zero_()
            interval = 0
            model.train()
    return initial_val, validation_result, time.perf_counter() - start


def checkpoint(output, run_id):
    return output / "checkpoints" / f"{run_id}.pt"


def fit(run_id, seed, mode, variant, data, config, provenance, output):
    context = {"run_id": run_id, "seed": seed, "mode": mode, "variant": variant, **provenance}
    torch.manual_seed(seed)
    model = WindowSSM(**config["model"]).to(provenance["device"])
    cuda = provenance["device"] == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print(f"START {run_id}", flush=True)
    try:
        initial, final, wall = train(model, data, config, context, output)
        torch.save(model.state_dict(), checkpoint(output, run_id))
        status = "PASS"
        error = ""
    except Exception as failure:
        traceback.print_exc()
        wall, status, error = time.perf_counter() - start, "FAIL", repr(failure)
        initial = final = None
    append_rows(output / "runs.csv", [{
        **context, "status": status, "wall_time_s": wall, "steps": config["steps"],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None, "error": error,
    }])
    print(f"{status} {run_id} ({wall:.2f}s)", flush=True)
    return initial, final


def get_model(config, output, run_id, device):
    model = WindowSSM(**config["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint(output, run_id), map_location=device, weights_only=True))
    return model.eval()


def validation_gate(data, config, device, output, full, occupancy, all_walkable):
    initial, final = full
    _, occupied = occupancy
    if final is None or occupied is None:
        value = {"status": "FAIL", "reason": "Validation training failed"}
    else:
        model = get_model(config, output, "sst-s0", device)
        val, targets = data["validation"], data["validation"]["target"]
        ablated, _ = validate(model, val, "sst", "seed_ablated", config)
        _, pred = validate(model, val, "sst", "full", config)
        pairs = paired_rows(pred, targets, {"run_id": "sst-s0", "split": "validation", "test_T": config["train_T"]})
        append_rows(output / "counterfactual.csv", pairs)
        flip = sum(row["prediction_change_rate"] for row in pairs) / len(pairs)
        matched = sum((row["iou_a"] + row["iou_b"]) * .5 for row in pairs) / len(pairs)
        swapped = sum((row["swapped_iou_a"] + row["swapped_iou_b"]) * .5 for row in pairs) / len(pairs)
        thresholds = config["validation_gate"]
        checks = {
            "above_all_walkable": final["iou"] - all_walkable >= thresholds["seed_iou_margin"],
            "above_seed_ablated": final["iou"] - ablated["iou"] >= thresholds["seed_iou_margin"],
            "above_trained_occupancy_only": final["iou"] - occupied["iou"] >= thresholds["seed_iou_margin"],
            "seed_flip": flip >= thresholds["min_prediction_flip"],
            "matched_above_swapped": matched - swapped >= thresholds["seed_iou_margin"],
            "validation_loss_decrease": initial["loss"] - final["loss"] >= thresholds["loss_decrease"],
        }
        value = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                 "initial": initial, "full": final, "all_walkable_iou": all_walkable, "seed_ablated": ablated,
                 "occupancy_only": occupied, "paired_flip_rate": flip, "paired_matched_iou": matched,
                 "paired_swapped_iou": swapped}
    write_json(output / "validation_gate.json", value)
    print(json.dumps(value, indent=2), flush=True)
    return value["status"] == "PASS"


@torch.no_grad()
def evaluate(model, config, data, mode, variant, run_id, seed, output):
    rounds = [config["train_T"]] if variant != "full" else config["test_T"]
    predictions = {turn: [] for turn in rounds}
    cases = DiagnosticTotals()
    start = time.perf_counter()
    for batch in batches(data, config["eval_batch_size"]):
        _, trace = model(set_seed_channel(batch["image"], variant), max(rounds), mode, return_trace=True)
        for turn in rounds:
            predictions[turn].append(trace[turn - 1].logits >= 0)
        if variant == "full":
            cases.add_norms(trace, batch["window"], model.k)
            if mode == "sst":
                cases.add_interventions(model, trace, batch["window"])
    info = {"run_id": run_id, "split": "test"}
    append_rows(output / "state_norms.csv", cases.state_norms(info))
    append_rows(output / "intervention.csv", cases.interventions({"run_id": run_id, "step": config["steps"], "split": "test"}))
    windows = (config["size"] // model.k) ** 2
    params = sum(p.numel() for p in model.parameters())
    for turn, tensors in predictions.items():
        pred = torch.cat(tensors)
        meta = {**info, "test_T": turn}
        append_rows(output / "metrics.csv", [{
            **meta, "mode": mode, "variant": variant, "seed": seed, "train_T": config["train_T"],
            "params": params, "steps": config["steps"], "scans_per_image": turn * windows * 4,
            "train_scan_count": config["steps"] * config["batch_size"] * config["train_T"] * windows * 4,
            **scores(pred, data["target"]),
        }])
        append_rows(output / "distance_metrics.csv", distance_rows(pred, data["target"], data, meta))
        append_rows(output / "component_metrics.csv", component_rows(pred.cpu(), data["target"].cpu(), meta))
        append_rows(output / "counterfactual.csv", paired_rows(pred.cpu(), data["target"].cpu(), meta))
    print(f"EVAL {run_id} depth={rounds} {time.perf_counter() - start:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sdr002.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output, device = args.output, args.device
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(seeds=[0], steps=2, period=2)
        for spec in config["data"].values():
            spec["maps"] = 4
            spec["seed"] += 100000
    output.mkdir(exist_ok=True, parents=True)
    if (output / "runs.csv").exists():
        parser.error("Use a new output directory for each attempt")
    (output / "checkpoints").mkdir(exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    data, manifest = generate(config)
    audit_result = audit(data, manifest, config)
    write_json(output / "data_manifest.json", manifest)
    write_json(output / "data_audit.json", audit_result)
    write_json(output / "effective_config.json", config)
    provenance = {
        "code_commit": repository_state("rev-parse", "HEAD"), "config_sha256": digest_config(config),
        "dataset_sha256": hashlib.sha256("".join(manifest[split]["sha256"]
                                                    for split in ("train", "validation", "test")).encode()).hexdigest(),
        "device": device, "torch": torch.__version__, "cuda_build": torch.version.cuda,
    }
    environment = {
        **provenance, "tracked_dirty": repository_state("status", "--porcelain", "--untracked-files=no"),
        "python": platform.python_version(), "platform": platform.platform(), "pytest": metadata.version("pytest"),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "official_ssd_parity": "NOT_RUN",
    }
    write_json(output / "environment.txt", environment)
    print(f"DATA AUDIT {audit_result['status']} ({provenance['dataset_sha256']})", flush=True)
    if audit_result["status"] != "PASS":
        return 2
    data = {split: {key: t.to(device) for key, t in values.items()} for split, values in data.items()}
    full = fit("sst-s0", 0, "sst", "full", data, config, provenance, output)
    occupancy = fit("occupancy-s0", 0, "sst", "occupancy_only", data, config, provenance, output)
    all_walkable = audit_result["splits"]["validation"]["baselines"]["all_walkable"]["iou"]
    gate = validation_gate(data, config, device, output, full, occupancy, all_walkable)
    if not gate and not args.smoke:
        print("VALIDATION GATE FAILED; matrix and test evaluation NOT_RUN", flush=True)
        return 2
    failures = 0
    matrix = [(f"{mode}-s{seed}", seed, mode, "full") for seed in config["seeds"] for mode in MODES]
    for run_id, seed, mode, variant in matrix:
        if run_id != "sst-s0":
            _, final = fit(run_id, seed, mode, variant, data, config, provenance, output)
            failures += int(final is None)
    for run_id, seed, mode, variant in [*matrix, ("occupancy-s0", 0, "sst", "occupancy_only")]:
        if not checkpoint(output, run_id).exists():
            failures += 1
            continue
        try:
            model = get_model(config, output, run_id, device)
            evaluate(model, config, data["test"], mode, variant, run_id, seed, output)
            if run_id == "sst-s0":
                evaluate(model, config, data["test"], mode, "seed_ablated", "sst-s0-ablated", seed, output)
        except Exception as error:
            failures += 1
            traceback.print_exc()
            append_rows(output / "evaluation_failures.csv", [{"run_id": run_id, "error": repr(error)}])
    print(f"FINISHED failures={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
