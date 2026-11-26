"""Locked common-recipe crowd training, development screening and selection."""

import argparse
import math
import os
from pathlib import Path
import platform
import random
from statistics import mean, stdev
import time

import torch

from scripts.audit_dataset import append_rows, digest_config, write_json
from scripts.audit_sdr006 import dataset, describe
from scripts.eval_sdr005 import read_csv
from scripts.train_sdr003 import git, read_json
from scripts.train_sdr005 import image_stats, objective, reduce_metrics
from sdr_mamba.crowd_audit import AuditModel
from sdr_mamba.crowd_data import decode, mass_map, sha_file, tensor_image, training_crop


def setup(output):
    config = read_json(Path("configs/sdr006.json"))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data, manifest, _ = dataset(config)
    if manifest != read_json(Path("reports/SDR-006/split_manifest.json")):
        raise ValueError("Frozen SDR-006 manifest mismatch")
    output.mkdir(exist_ok=True, parents=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    previous = output / "effective_config.json"
    if previous.exists() and read_json(previous) != config:
        raise ValueError("Use a fresh directory after any recipe change")
    write_json(previous, config)
    env = {
        "code_commit": git("rev-parse", "HEAD"), "config_sha256": digest_config(config),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(), "platform": platform.platform(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if not (output / "environment.json").exists():
        write_json(output / "environment.json", env)
    return config, data, env


def make_model(mode, seed, config, device="cuda"):
    torch.manual_seed(seed)
    return AuditModel(mode, config["modes"][mode], **config["model"]).to(device)


def crop(sample, size, config, rng, center=None):
    return training_crop(sample, {**config, "crop": size or max(sample.height, sample.width)}, rng,
                         fixed=center is not None, center=center)


def sample_batch(data, config, rng, step):
    size = config["scales"][(step - 1) % len(config["scales"])]
    return [crop(data[rng.randrange(len(data))], size, config, rng)
            for _ in range(config["batch_size"])]


def rate(step, config):
    warmup, updates = config["warmup"], config["updates"]
    end = config["final_lr_fraction"]
    scale = step / warmup if step <= warmup else end + (1 - end) * (
        1 + math.cos(math.pi * (step - warmup) / (updates - warmup))) / 2
    return config["lr"] * scale


def optimize(model, batch, optimizer, config):
    x, y = (torch.stack(values).cuda() for values in zip(*[(x, y) for x, y, _ in batch]))
    pred = model(x)[-1]
    loss = objective(pred, y, config)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_l2"], error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), norm.item(), (pred.detach().sum((-1, -2)) - y.sum((-1, -2))).abs().mean().item()


@torch.inference_mode()
def full_image(model, sample, config, rounds=None):
    x = tensor_image(decode(sample.file), config).unsqueeze(0).cuda()
    return model(x, rounds)


@torch.inference_mode()
def development(model, samples, config):
    model.eval()
    rows = []
    for s in samples:
        pred = full_image(model, s, config)[-1][0].cpu()
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Non-finite development prediction")
        target = mass_map(s.points, s.height, s.width, config["stride"])
        rows.append({"id": s.identifier, **image_stats(pred, target)})
    return reduce_metrics(rows)


def make_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


def preflight(data, config, output):
    records = []
    for mode in config["modes"]:
        model = make_model(mode, 0, config)
        optimizer = make_optimizer(model, config)
        for size in (256, 512, 0):
            rng = random.Random(706071)
            batch = [crop(s, size, config, rng) for s in data["train"][:config["batch_size"]]]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss, grad, _ = optimize(model, batch, optimizer, config)
            torch.cuda.synchronize()
            row = {"mode": mode, "scale": size, "loss": loss, "gradient_l2": grad,
                   "seconds": time.perf_counter() - start, "peak_bytes": torch.cuda.max_memory_allocated()}
            records.append(row)
            append_rows(output / "preflight.csv", [row])
            print(row, flush=True)
    return records


def overfit(data, config, output):
    samples = sorted((s for s in data["train"] if len(s.points)), key=lambda s: (len(s.points), s.identifier))
    n = config["sanity_images"]
    chosen = [samples[(2*i + 1) * len(samples) // (2*n)] for i in range(n)]
    fixed = [crop(s, 256, config, random.Random(0), center=s.points[len(s.points)//2]) for s in chosen]
    append_rows(output / "sanity_crops.csv", [row for _, _, row in fixed])
    model = make_model("F1", 0, config)
    opt = make_optimizer(model, config)
    xs, ys = (torch.stack(seq).cuda() for seq in zip(*[(x, y) for x, y, _ in fixed]))

    @torch.inference_mode()
    def measure(step):
        pred = model(xs)[-1]
        return {"step": step, "loss": objective(pred, ys, config).item(),
                "mae": (pred.sum((-1, -2)) - ys.sum((-1, -2))).abs().mean().item()}

    start = time.perf_counter()
    first = measure(0)
    append_rows(output / "sanity_curve.csv", [first])
    for step in range(1, config["sanity_updates"] + 1):
        slot = (step - 1) * config["batch_size"]
        batch = [fixed[(slot + j) % n] for j in range(config["batch_size"])]
        optimize(model, batch, opt, config)
        if step % 40 == 0 or step == config["sanity_updates"]:
            final = measure(step)
            append_rows(output / "sanity_curve.csv", [final])
            print("Sanity", final, flush=True)
    zero = ys.sum((-1, -2)).mean().item()
    ratio = config["sanity_relative"]
    passed = final["loss"] <= ratio * first["loss"] and final["mae"] <= ratio * min(first["mae"], zero)
    record = {"status": "PASS" if passed else "FAIL", "initial": first, "last": final, "zero_mae": zero,
              "seconds": time.perf_counter() - start}
    write_json(output / "sanity_gate.json", record)
    return passed


def artifact(output, mode, seed, step):
    return output / "checkpoints" / f"{mode}-s{seed}-{step}.pt"


def checkpoint_rank(row):
    return (float(row["mae"]), float(row["game1"]), int(row["step"]))


def parameter_counts(model):
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "mixer_parameters": sum(p.numel() for p in model.blocks.parameters())}


def train(mode, seed, config, data, output, env, whole_deadline):
    run = {"mode": mode, "seed": seed}
    print("START", run, flush=True)
    rng = random.Random(config["augmentation_seed"] + seed)
    model = make_model(mode, seed, config)
    optimizer = make_optimizer(model, config)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    deadline = min(whole_deadline, start + config["max_run_seconds"])
    curves, losses, ledgers, buffers = [], [], [], []
    updates, cells, status, error = 0, 0, "FAIL", ""
    try:
        for step in range(1, config["updates"] + 1):
            if time.perf_counter() > deadline:
                raise TimeoutError("Registered compute budget reached")
            model.train()
            batch = sample_batch(data["train"], config, rng, step)
            lr = rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses.append(optimize(model, batch, optimizer, config))
            updates = step
            cells += sum(t.numel() for _, t, _ in batch)
            buffers.extend({**run, "step": step, "slot": i, **row} for i, (_, _, row) in enumerate(batch))
            if step % config["eval_period"] == 0:
                append_rows(output / "crop_ledger.csv", buffers)
                ledgers.extend(buffers)
                buffers.clear()
                stats = development(model, data["development"], config)
                if time.perf_counter() > deadline:
                    raise TimeoutError("Registered compute budget reached during development")
                path = artifact(output, mode, seed, step)
                torch.save(model.state_dict(), path)
                loss, grad, mae = map(mean, zip(*losses))
                row = {**run, "step": step, "lr": lr, "train_loss": loss, "gradient_l2": grad,
                       "train_mae": mae, **stats, "checkpoint_sha256": sha_file(path)}
                curves.append(row)
                append_rows(output / "train_curves.csv", [row])
                print(f"{mode} s{seed} step {step} loss {loss:.5f} dev MAE {row['mae']:.4f} "
                      f"GAME1 {row['game1']:.4f} bias {row['bias']:+.4f}", flush=True)
                losses.clear()
        selected = min(curves, key=checkpoint_rank)
        tail = [row["mae"] for row in curves[-3:]]
        tail_bias = [row["bias"] for row in curves[-3:]]
        append_rows(output / "checkpoint_selection.csv", [{
            **selected, "last_mae": tail[-1], "last_game1": curves[-1]["game1"], "last_bias": tail_bias[-1],
            "tail_mean": mean(tail), "tail_sd": stdev(tail), "tail_relative_sd": stdev(tail) / max(mean(tail), 1e-12),
            "tail_bias_mean": mean(tail_bias), "tail_bias_sd": stdev(tail_bias),
            "max_relative_deviation": max(abs(x-mean(tail)) for x in tail) / max(mean(tail), 1e-12),
        }])
        if mode == "F0" and seed == 0:
            write_crop_distribution(output, ledgers, data["train"])
        status = "PASS"
    except BaseException as failure:
        status = "BLOCKED" if isinstance(failure, TimeoutError) else "FAIL"
        error = type(failure).__name__
        raise
    finally:
        if buffers:
            append_rows(output / "crop_ledger.csv", buffers)
        append_rows(output / "runs.csv", [{
            **run, "status": status, "error": error, "updates": updates, **parameter_counts(model),
            "feature_cells": cells, "scan_tokens": cells * 4 * config["modes"][mode] if mode in ("F1", "F2", "F4") else 0,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "seconds": time.perf_counter() - start,
            "config_sha256": env["config_sha256"], "code_commit": env["code_commit"],
        }])


def write_crop_distribution(output, ledger, train_data):
    groups = {"256": [], "512": [], "full": []}
    for row in ledger:
        key = str(row["height"]) if row["height"] in (256, 512) else "full"
        groups[key].append((row["crop_points"], row["height"] * row["width"], row["width"]//8, row["height"]//8))
    groups["unique_train_images"] = [(len(s.points), s.height*s.width, s.width//8, s.height//8) for s in train_data]
    write_json(output / "crop_distribution.json", {
        key: {"samples": len(values), "count": describe([v[0] for v in values]),
              "area_px": values[0][1], "axis_tokens": values[0][2:],
              "heads_per_megapixel": describe([n * 1e6/a for n, a, _, _ in values])}
        for key, values in groups.items()
    })


def development_gate(output, config):
    selection = {r["mode"]: r for r in read_csv(output / "checkpoint_selection.csv") if r["seed"] == "0"}
    if selection.keys() != config["modes"].keys():
        raise ValueError("Incomplete development screen")
    f1 = selection["F1"]
    limit = config["margin"]
    contrasts = {mode: {k: (float(f1[k]) - float(selection[mode][k])) / max(float(selection[mode][k]), 1e-12)
                        for k in ("mae", "game1")} for mode in ("F0", "F2", "F3", "F4")}
    stable = {m: float(r["tail_relative_sd"]) for m, r in selection.items()}
    checks = {
        "F1_MAE_gain_vs_F0": contrasts["F0"]["mae"] <= -limit,
        "F1_MAE_noninferiority_vs_F2": contrasts["F2"]["mae"] <= limit,
        "F1_GAME1_noninferiority": all(contrasts[m]["game1"] <= limit for m in ("F0", "F2")),
        "all_modes_tail_stability": all(math.isfinite(s) and s <= config["max_tail_relative_sd"] for s in stable.values()),
    }
    passed = all(checks.values())
    verdict = {"status": "PASS" if passed else "FAIL", "confirmation": "PENDING" if passed else "NOT_RUN",
               "official_test": "NOT_RUN", "checks": checks, "relative_F1_minus_control": contrasts,
               "last_three_relative_sd": stable}
    write_json(output / "gate.json", verdict)
    print("Development gate", verdict, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "sanity", "screen", "expand"), required=True)
    args = parser.parse_args()
    output = args.output
    if args.phase == "expand" and read_json(output / "gate.json")["status"] != "PASS":
        print("Additional seeds NOT_RUN because the development gate failed")
        return 0
    config, data, env = setup(output)
    if args.phase == "preflight":
        preflight(data, config, output)
        return 0
    if args.phase == "sanity":
        return int(not overfit(data, config, output))
    sanity = read_json(output / "sanity_gate.json")
    if sanity["status"] != "PASS":
        raise RuntimeError("The real-data engineering gate failed")
    ledger = output / "runs.csv"
    previous = read_csv(ledger) if ledger.exists() else []
    spent = sum(float(r["seconds"]) for r in previous) + sanity["seconds"]
    spent += sum(float(r["seconds"]) for r in read_csv(output / "preflight.csv"))
    deadline = time.perf_counter() + config["max_total_seconds"] - spent
    seeds = [0] if args.phase == "screen" else config["seeds"][1:]
    for seed in seeds:
        for mode in config["modes"]:
            if any(r["mode"] == mode and int(r["seed"]) == seed for r in previous):
                raise FileExistsError(f"Run already attempted for {mode} seed {seed}")
            train(mode, seed, config, data, output, env, deadline)
    if args.phase == "screen":
        development_gate(output, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
