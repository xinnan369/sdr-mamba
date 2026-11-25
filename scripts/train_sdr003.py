"""Pre-registered independent training stages for SDR-003."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import torch

from scripts.audit_dataset import append_rows, audit, digest_config, scores, write_json
from scripts.run_sdr002 import batches, objective
from sdr_mamba.data_v3 import build
from sdr_mamba.model_v3 import LoopV3

A = {"A0": (4, "final", False), "A1": (8, "final", False),
     "A2": (8, "deep", False), "A3": (8, "deep", True)}
B = {"B0": "sst", "B1": "learned", "B2": "hard", "B3": "gated_feature",
     "B4": "feature", "B5": "strong_feature", "B6": "local_gate"}


def loss_for(frames, target, supervision):
    if supervision == "final":
        return objective(frames[-1].logits, target)
    return sum(weight * objective(frames[turn - 1].logits, target)
               for turn, weight in ((4, .25), (6, .25), (8, .5)))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(("git", *args), text=True).strip()


def checkpoint(directory, group, seed):
    return directory / "checkpoints" / f"{group}-s{seed}.pt"


def setup(config_path, output, device, smoke=False):
    config = read_json(config_path)
    if smoke:
        config["steps"] = 2
        config["period"] = 2
        config["seeds"] = [0]
        for spec in config["data"].values():
            spec.update(seed=spec["seed"] + 200000, maps=4)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is not available")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    splits, manifest, base = build(config)
    result = audit(splits, manifest, base)
    for name, data in splits.items():
        target = data["target"]
        grid = data["grid"]
        hist = {str(d): int(((grid == d) & target).sum()) for d in range(15)}
        result["splits"][name]["grid_target_histogram"] = hist
        result["splits"][name]["grid_far_4_to_7"] = sum(hist[str(i)] for i in range(4, 8))
        result["splits"][name]["grid_unreachable_at_t8"] = sum(hist[str(i)] for i in range(8, 15))
        if not result["splits"][name]["grid_far_4_to_7"]:
            result["status"] = "FAIL"
    closed = [row["closed_walking_window_edges"] for row in manifest["ood"]["maps"]]
    result["splits"]["ood"]["hard_negative_boundary_range"] = [min(closed), max(closed)]
    result["config_sha256"] = digest_config(config)
    provenance = {
        "code_commit": git("rev-parse", "HEAD"), "config_sha256": digest_config(config),
        "dataset_sha256": hashlib.sha256("".join(r["sha256"] for r in manifest.values()).encode()).hexdigest(),
        "torch": torch.__version__, "cuda_build": torch.version.cuda, "device": device,
    }
    if (output / "effective_config.json").exists() and read_json(output / "effective_config.json") != config:
        raise ValueError("Destination already contains a different effective config")
    write_json(output / "effective_config.json", config)
    write_json(output / "data_manifest.json", manifest)
    write_json(output / "data_audit.json", result)
    env = {**provenance, "tracked_dirty": git("status", "--porcelain", "--untracked-files=no"),
           "python": platform.python_version(), "platform": platform.platform(),
           "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           "official_ssd_parity": "NOT_RUN"}
    if not (output / "environment.txt").exists():
        write_json(output / "environment.txt", env)
    print(f"Data audit: {result['status']} {provenance['dataset_sha256']}", flush=True)
    if result["status"] != "PASS":
        raise RuntimeError("Structural data audit failed")
    moved = {name: {key: value.to(device) for key, value in data.items()} for name, data in splits.items()}
    return config, moved, provenance


@torch.no_grad()
def check(model, data, turns, batch_size):
    model.eval()
    pred = {t: [] for t in turns}
    losses = {t: 0. for t in turns}
    for batch in batches(data, batch_size):
        frames = model(batch["image"], max(turns))
        for t in turns:
            logit = frames[t - 1].logits
            losses[t] += objective(logit, batch["target"]).item() * len(logit)
            pred[t].append(logit >= 0)
    return {t: {**scores(torch.cat(pred[t]), data["target"]), "loss": losses[t] / len(data["target"])}
            for t in turns}


def fit(group, seed, mode, protocol, config, data, environment, output, steps=None):
    depth, supervision, anchored = protocol
    steps = config["steps"] if steps is None else steps
    destination = checkpoint(output, group, seed)
    if destination.exists():
        raise FileExistsError(destination)
    run = f"{group}-s{seed}"
    context = {"run_id": run, "stage": group[0], "mode": group, "seed": seed, "operator": mode,
               "train_T": depth, "supervision": supervision, "anchored": anchored}
    device = environment["device"]
    torch.manual_seed(seed)
    model = LoopV3(mode, anchored).to(device)
    order = torch.Generator().manual_seed(503300 + seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    images, target = data["train"]["image"], data["train"]["target"]
    turns = (4, 6, 8)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"START {run} T={depth} steps={steps} mode={mode}", flush=True)
    start = time.perf_counter()
    final = None
    status, error, digest = "FAIL", "", ""
    updates = 0
    try:
        initial = check(model, data["validation"], turns, config["eval_batch_size"])
        append_rows(output / "train_curves.csv", [
            {**context, "step": 0, "split": "validation", "round": t, **entry} for t, entry in initial.items()
        ])
        model.train()
        totals = torch.zeros(4, device=device)
        interval = 0
        for step in range(1, steps + 1):
            ids = torch.randint(len(images), (config["batch_size"],), generator=order).to(device)
            x, y = images[ids], target[ids]
            frames = model(x, depth)
            loss = loss_for(frames, y, supervision)
            pred = frames[-1].logits.detach() >= 0
            capture = step % config["period"] == 0 or step == steps
            if capture:
                if not math.isfinite(loss.item()):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                nodes = [frame.f_out for frame in frames]
                values = torch.autograd.grad(loss, nodes, retain_graph=True)
                append_rows(output / "training_gradients.csv", [
                    {"run_id": run, "step": step, "round": t, "kind": "feature_grad_l2", "value": v.norm().item()}
                    for t, v in enumerate(values, 1)
                ])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if capture:
                norm = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm().item()
                if not math.isfinite(norm):
                    raise FloatingPointError(f"Non-finite gradient at step {step}")
                append_rows(output / "training_gradients.csv", [
                    {"run_id": run, "step": step, "round": depth, "kind": "parameter_grad_l2", "value": norm},
                ])
            optimizer.step()
            updates = step
            from scripts.audit_dataset import measurements

            with torch.no_grad():
                totals += torch.stack((loss.detach(), (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()))
            interval += 1
            if capture:
                value, tp, fp, fn = totals.tolist()
                record = {**measurements(int(tp), int(fp), int(fn)), "loss": value / interval}
                final = check(model, data["validation"], turns, config["eval_batch_size"])
                append_rows(output / "train_curves.csv", [
                    {**context, "step": step, "split": "train", "round": depth, **record},
                    *[{**context, "step": step, "split": "validation", "round": t, **entry}
                      for t, entry in final.items()],
                ])
                losses = "/".join(f"{final[t]['iou']:.4f}" for t in turns)
                print(f"{run} {step}/{steps} loss={record['loss']:.4f} val IoU T4/6/8={losses}", flush=True)
                model.train()
                totals.zero_()
                interval = 0
        torch.save(model.state_dict(), destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        status = "PASS"
    except Exception as failure:
        traceback.print_exc()
        error = repr(failure)
        final = None
    finally:
        cuda_peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
        append_rows(output / "runs.csv", [{
            **context, **environment, "updates": updates, "params": sum(p.numel() for p in model.parameters()),
            "scan_tokens": updates * config["batch_size"] * depth * 32 * 32 * 4,
            "wall_time_s": time.perf_counter() - start, "peak_cuda_bytes": cuda_peak,
            "status": status, "checkpoint_sha256": digest, "error": error,
        }])
        print(f"{status} {run}, updates={updates}", flush=True)
    return final


def select(config, screen):
    baseline = screen["A1"][8]["iou"]
    limits = config["a_screen"]
    details = {}
    for name in ("A1", "A2", "A3"):
        run = screen.get(name)
        ious = [] if run is None else [run[t]["iou"] for t in (4, 6, 8)]
        if not ious or not all(math.isfinite(v) for v in ious):
            details[name] = {"eligible": False, "reason": "training or validation failed"}
            continue
        details[name] = {"iou": ious, "mean": sum(ious) / 3,
                         "eligible": ious[2] >= baseline - limits["t8_tolerance"]
                         and max(abs(ious[2] - v) for v in ious[:2]) <= limits["max_shallow_gap"]}
    ranked = [name for name, values in details.items() if values["eligible"]]
    if ranked:
        best = max(details[name]["mean"] for name in ranked)
        winner = next(name for name in ranked if best - details[name]["mean"] <= limits["mean_tie_band"])
    else:
        winner = "A1"
    return {"selected": winner, "details": details, "fallback": not bool(ranked)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sdr003.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--phase", choices=("audit", "a-screen", "a-complete", "b", "smoke"), required=True)
    args = parser.parse_args()
    config, data, provenance = setup(args.config, args.output, args.device, args.phase == "smoke")
    output = args.output
    if args.phase == "audit":
        return 0
    if args.phase in ("a-screen", "smoke"):
        result = {group: fit(group, 0, "sst", spec, config, data, provenance, output)
                  for group, spec in A.items()}
        if result["A0"] is None or result["A1"] is None:
            return 1
        selection = select(config, result)
        write_json(output / "a_selection.json", selection)
        print(f"VALIDATION SELECTION {selection}", flush=True)
        if args.phase == "smoke":
            for group, mode in B.items():
                fit(group, 0, mode, A[selection["selected"]], config, data, provenance, output)
        return int(any(result[group] is None for group in A))
    selection = read_json(output / "a_selection.json")
    chosen = selection["selected"]
    failures = 0
    if args.phase == "a-complete":
        for group in dict.fromkeys(("A0", "A1", chosen)):
            for seed in config["seeds"][1:]:
                run = fit(group, seed, "sst", A[group], config, data, provenance, output)
                failures += int(run is None)
    if args.phase == "b":
        decision = read_json(output / "a_gate.json")
        seeds = config["seeds"] if decision["status"] == "PASS" else [0]
        steps = config["steps"] if decision["status"] == "PASS" else 256
        print(f"Stage B Scope: {'formal' if len(seeds) > 1 else 'validation-only diagnostic'}", flush=True)
        for group, mode in B.items():
            for seed in seeds:
                run = fit(group, seed, mode, A[chosen], config, data, provenance, output, steps)
                failures += int(run is None)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
