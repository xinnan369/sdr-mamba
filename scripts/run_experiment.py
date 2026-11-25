"""Run the fixed synthetic segmentation experiment and preserve each attempted run."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

import torch
import torch.nn.functional as F

from sdr_mamba.data import generate
from sdr_mamba.model import LoopedSSM
from sdr_mamba.state_transport import MODES, initial_states

DISTANCE_BINS = ((0, 0), (1, 1), (2, 3), (4, 6))


def save_json(path, value):
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


def subprocess_output(argv):
    result = subprocess.run(argv, capture_output=True, text=True, errors="replace", check=False)
    return result.returncode, (result.stdout + result.stderr).strip()


def environment(path, config):
    _, commit = subprocess_output(["git", "rev-parse", "HEAD"])
    _, dirty = subprocess_output(["git", "status", "--porcelain", "--untracked-files=no"])
    try:
        distribution = importlib.metadata.distribution("mamba-ssm")
        mamba = distribution.version
        origin = distribution.read_text("direct_url.json")
        upstream = (json.loads(origin).get("vcs_info", {}).get("commit_id") if origin else None)
    except importlib.metadata.PackageNotFoundError:
        mamba, upstream = "absent", None
    info = [
        f"started_utc={datetime.now(timezone.utc).isoformat()}",
        f"code_commit={commit}", f"tracked_worktree_status={dirty or 'clean'}",
        f"platform={platform.platform()}", f"processor={platform.processor()}",
        f"logical_cpus={os.cpu_count()}", f"python={platform.python_version()}",
        f"torch={torch.__version__}", f"pytest={importlib.metadata.version('pytest')}",
        f"cuda_build={torch.version.cuda}", f"cuda_available={torch.cuda.is_available()}",
        f"mamba_ssm={mamba}", f"mamba_upstream_commit={upstream or 'NOT_RUN / unavailable'}",
        f"experiment_device={config['device']}", f"torch_threads={config['threads']}",
    ]
    if shutil.which("nvidia-smi"):
        _, gpu = subprocess_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])
        info.append(gpu)
    path.write_text("\n".join(info) + "\n", encoding="utf-8")


def rms(tensor):
    return None if tensor is None else tensor.detach().square().mean().sqrt().item()


def train(model, mode, data, config, seed, run_id, destination):
    generator = torch.Generator().manual_seed(seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    features, target, _ = data
    model.train()
    start = time.perf_counter()
    for step in range(1, config["steps"] + 1):
        ids = torch.randint(len(features), (config["batch_size"],), generator=generator).to(features.device)
        x, y = features[ids], target[ids]
        logits, trace = model(x, config["train_T"], mode, return_trace=True)
        probabilities = logits.sigmoid()
        dice_loss = 1 - ((2 * (probabilities * y).sum((1, 2)) + 1)
                         / (probabilities.sum((1, 2)) + y.sum((1, 2)) + 1)).mean()
        bce = F.binary_cross_entropy_with_logits(logits, y)
        loss = bce + dice_loss
        if not loss.isfinite():
            raise FloatingPointError(f"Nonfinite loss at step {step}: {loss.item()}")
        for frame in trace:
            frame.final.retain_grad()
            if frame.initial is not None:
                frame.initial.retain_grad()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.stack([
            parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None
        ]).norm().item()
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError(f"Nonfinite parameter gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        append_rows(destination / "train_steps.csv", [{
            "run_id": run_id, "step": step, "loss": loss.item(), "bce": bce.item(),
            "soft_dice_loss": dice_loss.item(), "parameter_grad_l2": grad_norm,
        }])
        append_rows(destination / "train_states.csv", [
            {"run_id": run_id, "step": step, "round": turn, "final_rms": rms(frame.final),
             "initial_rms": rms(frame.initial), "final_grad_rms": rms(frame.final.grad),
             "initial_grad_rms": None if frame.initial is None else rms(frame.initial.grad)}
            for turn, frame in enumerate(trace, 1)
        ])
    if features.is_cuda:
        torch.cuda.synchronize(features.device)
    return time.perf_counter() - start


def confusion(pred, label, mask):
    tp = int((pred & label & mask).sum())
    fp = int((pred & ~label & mask).sum())
    fn = int((~pred & label & mask).sum())
    return tp, fp, fn


def scores(counts):
    tp, fp, fn = counts
    union, dice_denom = tp + fp + fn, 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / union if union else None,
            "dice": 2 * tp / dice_denom if dice_denom else None}


def add_stats(store, key, values):
    if key not in store:
        store[key] = values
    else:
        store[key] = tuple(x + y for x, y in zip(store[key], values))


def intervention(model, trace, turn, distance, store):
    frame, previous = trace[turn - 1], trace[turn - 2].final
    for label, history in (("zero", None), ("shuffled", initial_states(previous, "shuffled", model.gate, model.mixing))):
        perturbed, _, _ = model.step(frame.feature_in, history)
        probability_delta = (frame.logits.sigmoid() - perturbed.sigmoid()).abs()
        flips = (frame.logits >= 0) != (perturbed >= 0)
        for low, high in DISTANCE_BINS:
            mask = (distance >= low) & (distance <= high)
            add_stats(store, (turn, label, "window_distance", f"{low}-{high}"),
                      (probability_delta[mask].sum().item(), int(flips[mask].sum()), int(mask.sum())))
        k = model.k
        for y in range(probability_delta.shape[1] // k):
            for x in range(probability_delta.shape[2] // k):
                region = (slice(None), slice(y * k, (y + 1) * k), slice(x * k, (x + 1) * k))
                add_stats(store, (turn, label, "grid_y_x", f"{y},{x}"),
                          (probability_delta[region].sum().item(), int(flips[region].sum()),
                           probability_delta[region].numel()))


@torch.no_grad()
def evaluate(model, mode, data, rounds, batch_size, do_intervention):
    model.eval()
    counts, states, sensitivity = {}, {}, {}
    features, targets, distances = data
    for x, target, distance in zip(features.split(batch_size), targets.split(batch_size),
                                   distances.split(batch_size)):
        _, trace = model(x, max(rounds), mode, return_trace=True)
        label = target.bool()
        for turn, frame in enumerate(trace, 1):
            pred = frame.logits >= 0
            add_stats(counts, (turn, "all"), confusion(pred, label, torch.ones_like(label)))
            for low, high in DISTANCE_BINS:
                mask = (distance >= low) & (distance <= high)
                add_stats(counts, (turn, f"{low}-{high}"), confusion(pred, label, mask))
            add_stats(states, turn, (frame.final.square().sum().item(), frame.final.numel(),
                                     0.0 if frame.initial is None else frame.initial.square().sum().item()))
            if do_intervention and 2 <= turn <= 4:
                intervention(model, trace, turn, distance, sensitivity)
    return counts, states, sensitivity


def run(config, mode, seed, datasets, destination):
    run_id = f"{mode}-s{seed}"
    start = time.perf_counter()
    torch.manual_seed(seed)
    device = torch.device(config["device"])
    model = LoopedSSM(**config["model"]).to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall = train(model, mode, datasets["train"], config, seed, run_id, destination)
    metadata = {"run_id": run_id, "seed": seed, "mode": mode, "train_T": config["train_T"]}
    windows = (config["size"] // model.k) ** 2
    for split in ("validation", "test"):
        rounds = [config["train_T"]] if split == "validation" else config["test_T"]
        eval_start = time.perf_counter()
        counts, states, sensitivity = evaluate(
            model, mode, datasets[split], rounds, config["eval_batch_size"],
            do_intervention=split == "test" and mode == "sst",
        )
        append_rows(destination / "metrics.csv", [
            {**metadata, "test_T": turn, "split": split, **scores(counts[turn, "all"]),
             "params": params, "scan_count": 4 * turn * windows, "steps": config["steps"],
             "train_scan_count": 4 * config["train_T"] * windows * config["batch_size"] * config["steps"],
             "wall_time_s": wall} for turn in rounds
        ])
        append_rows(destination / "distance_metrics.csv", [
            {**metadata, "round": turn, "split": split, "bucket": bucket, **scores(values)}
            for (turn, bucket), values in counts.items() if bucket != "all"
        ])
        append_rows(destination / "round_metrics.csv", [
            {**metadata, "round": turn, "split": split, **scores(counts[turn, "all"]),
             "final_state_rms": (sum_final / elements) ** 0.5,
             "initial_state_rms": (sum_initial / elements) ** 0.5}
            for turn, (sum_final, elements, sum_initial) in states.items()
        ])
        append_rows(destination / "intervention.csv", [
            {**metadata, "round": turn, "replacement": replacement, "spatial_axis": axis,
             "bucket": bucket, "mean_abs_probability_delta": delta / pixels,
             "threshold_flip_rate": flips / pixels, "pixels": pixels}
            for (turn, replacement, axis, bucket), (delta, flips, pixels) in sensitivity.items()
        ])
        print(f"{run_id} {split} {time.perf_counter() - eval_start:.2f}s "
              f"IoU@{config['train_T']}={scores(counts[config['train_T'], 'all'])['iou']:.6f}", flush=True)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return time.perf_counter() - start, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(seeds=[0], steps=2)
        for spec in config["data"].values():
            spec["count"] = 8
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("*.csv")) or (destination / "effective_config.json").exists():
        parser.error("Output directory already contains experiment results")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    save_json(destination / "effective_config.json", config)
    environment(destination / "environment.txt", config)
    splits, manifest = generate(config)
    save_json(destination / "data_manifest.json", manifest)
    datasets = {name: tuple(t.to(config["device"]) for t in split) for name, split in splits.items()}
    failures = 0
    for seed in config["seeds"]:
        for mode in MODES:
            status = {"run_id": f"{mode}-s{seed}", "seed": seed, "mode": mode}
            print(f"START {status['run_id']}", flush=True)
            start = time.perf_counter()
            try:
                wall, peak = run(config, mode, seed, datasets, destination)
                status.update(status="PASS", total_wall_time_s=wall, peak_cuda_allocated_bytes=peak, error="")
            except Exception as error:
                failures += 1
                traceback.print_exc()
                status.update(status="FAIL", total_wall_time_s=time.perf_counter() - start,
                              peak_cuda_allocated_bytes=None, error=repr(error))
            append_rows(destination / "runs.csv", [status])
            print(f"{status['status']} {status['run_id']} {status['total_wall_time_s']:.2f}s", flush=True)
    print(f"Completed {len(config['seeds']) * len(MODES)} attempted runs, failures={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
