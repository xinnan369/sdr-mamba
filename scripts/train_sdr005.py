"""ShanghaiTech fixed-budget validation screen and conditional multi-seed phase."""

import argparse
from collections import defaultdict
import csv
import hashlib
from importlib import metadata
import math
import os
from pathlib import Path
import platform
import random
from statistics import mean
import time
import traceback

import torch
from torch.nn import functional as F

from scripts.audit_dataset import append_rows, digest_config, write_json
from scripts.train_sdr003 import git, read_json
from sdr_mamba.crowd_data import decode, load_dataset, mass_map, tensor_image, training_crop
from sdr_mamba.crowd_ssm import CrowdModel


def setup(output, device):
    cfg = read_json(Path("configs/sdr005.json"))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if "SDR005_DATA_ROOT" not in os.environ:
        raise RuntimeError("Set SDR005_DATA_ROOT to an authorized existing ShanghaiTech directory")
    partitions, manifest, audit = load_dataset(Path(os.environ["SDR005_DATA_ROOT"]), cfg)
    key = hashlib.sha256("".join(s["sha256"] for s in manifest.values()).encode()).hexdigest()
    run_info = {
        "code_commit": git("rev-parse", "HEAD"), "config_sha256": digest_config(cfg), "dataset_sha256": key,
        "device": device, "torch": torch.__version__, "cuda": torch.version.cuda, "backend": cfg["model"]["backend"],
    }
    if (output / "split_manifest.json").exists() and read_json(output / "split_manifest.json") != manifest:
        raise ValueError("Existing output uses a different dataset")
    write_json(output / "split_manifest.json", manifest)
    write_json(output / "data_audit.json", audit)
    write_json(output / "effective_config.json", cfg)
    if not (output / "environment.txt").exists():
        write_json(output / "environment.txt", {
            **run_info, "python": platform.python_version(), "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "scipy": metadata.version("scipy"), "numpy": metadata.version("numpy"),
            "pillow": metadata.version("pillow"), "tracked_dirty": git("status", "--porcelain", "--untracked-files=no"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        })
    print(f"Part {cfg['part']} dataset audit {audit['status']} {key}", flush=True)
    return cfg, partitions, run_info


def smooth_mass(target):
    line = target.new_tensor([1., 2., 1.])
    kernel = (line[:, None] * line[None, :] / 16).view(1, 1, 3, 3)
    unit = torch.ones_like(target[:1, None])
    normalization = F.conv2d(unit, kernel, padding=1)
    return F.conv2d(target[:, None] / normalization, kernel, padding=1).squeeze(1)


def objective(pred, target, cfg):
    target_density = smooth_mass(target)
    weights = 1 + cfg["density_objective"]["mass_weight"] * target_density
    density = cfg["density_objective"]["scale"] * (weights * (pred - target_density).square()).mean()
    count = F.smooth_l1_loss(pred.sum((-2, -1)).log1p(), target.sum((-2, -1)).log1p(), beta=.2)
    return density + cfg["count_loss_weight"] * count


def game(pred, target, level):
    # Cell mass stays inside the exact image-grid region. For Part B all H/W
    # are divisible by 8 stride cells as needed for GAME levels 1–3.
    tiles = 2 ** level
    difference = pred - target
    return sum(tile.sum().abs() for row in difference.tensor_split(tiles, -2)
               for tile in row.tensor_split(tiles, -1)).item()


def image_stats(pred, target):
    err = pred.sum().item() - target.sum().item()
    return {
        "gt": int(target.sum()), "count": pred.sum().item(), "signed": err, "absolute": abs(err),
        "squared": err ** 2, "game1": game(pred, target, 1), "game2": game(pred, target, 2),
        "game3": game(pred, target, 3), "density_cell_mae": (pred - target).abs().mean().item(),
    }


def reduce_metrics(rows):
    vals = {
        "mae": mean(r["absolute"] for r in rows), "rmse": math.sqrt(mean(r["squared"] for r in rows)),
        "bias": mean(r["signed"] for r in rows), "game1": mean(r["game1"] for r in rows),
        "game2": mean(r["game2"] for r in rows), "game3": mean(r["game3"] for r in rows),
        "density_cell_mae": mean(r["density_cell_mae"] for r in rows),
    }
    return {"images": len(rows), **vals}


@torch.no_grad()
def validation(model, samples, cfg, device):
    model.eval()
    records = []
    for sample in samples:
        image = tensor_image(decode(sample.file), cfg).unsqueeze(0).to(device)
        density = model(image)[-1].density[0].cpu()
        target = mass_map(sample.points, sample.height, sample.width, cfg["stride"])
        if not torch.isfinite(density).all():
            raise FloatingPointError("Nonfinite validation density")
        records.append({"id": sample.identifier, **image_stats(density, target)})
    return reduce_metrics(records), records


def checkpoint(output, stage, mode, seed):
    return output / "checkpoints" / f"{stage}-{mode}-s{seed}.pt"


def make_model(mode, cfg, device, seed):
    torch.manual_seed(seed)
    return CrowdModel(mode, **cfg["model"], head_bias=cfg["density_head_initial_bias"]).to(device)


def ready_crops(batch, model, device):
    images, targets = zip(*((im, tgt) for im, tgt, _ in batch))
    shapes = {value.shape for value in images}
    if len(shapes) == 1:
        x, y = torch.stack(images).to(device), torch.stack(targets).to(device)
        return [(model(x)[-1].density, y)]
    return [(model(x.unsqueeze(0).to(device))[-1].density, y.unsqueeze(0).to(device))
            for x, y in zip(images, targets)]


def overfit(cfg, data, output, device):
    pool = [item for sample in sorted(data["train"], key=lambda x: x.identifier) if len(sample.points)
            for item in [training_crop(sample, cfg, random.Random(0), fixed=True,
                                       center=sample.points[len(sample.points) // 2])]
            if item[2]["crop_points"] > 0]
    count = cfg["overfit_images"]
    if len(pool) < count:
        raise ValueError("Not enough nonempty fixed crops for the training sanity check")
    pool.sort(key=lambda item: (item[2]["crop_points"], item[2]["id"]))
    fixed = [pool[min(len(pool) - 1, (2 * i + 1) * len(pool) // (2 * count))] for i in range(count)]
    zero_mae = mean(row["crop_points"] for _, _, row in fixed)
    write_json(output / "sanity_crops.json", [row for _, _, row in fixed])
    model = make_model("C1", cfg, device, 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    @torch.no_grad()
    def measure(step):
        model.eval()
        readings = ready_crops(fixed, model, device)
        loss = mean(objective(p, y, cfg).item() for p, y in readings)
        errors = [abs(p0.sum().item() - y0.sum().item()) for p, y in readings for p0, y0 in zip(p, y)]
        record = {"step": step, "loss": loss, "crop_mae": mean(errors)}
        append_rows(output / "sanity_overfit.csv", [record])
        return record

    first = measure(0)
    for step in range(1, cfg["overfit_updates"] + 1):
        model.train()
        indices = [(step * cfg["batch_size"] + j) % len(fixed) for j in range(cfg["batch_size"])]
        observed = ready_crops([fixed[i] for i in indices], model, device)
        loss = torch.stack([objective(p, y, cfg) for p, y in observed]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite overfit objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_l2"], error_if_nonfinite=True)
        optimizer.step()
        if step % 40 == 0 or step == cfg["overfit_updates"]:
            last = measure(step)
            print(f"Overfit step {step}: {last}", flush=True)
    result = {"status": "PASS" if last["loss"] < first["loss"] * .8
              and last["crop_mae"] < min(first["crop_mae"], zero_mae) * cfg["overfit_gate_relative_zero_baseline"]
              else "FAIL", "initial": first, "final": last, "zero_prediction_mae": zero_mae}
    write_json(output / "sanity_gate.json", result)
    print(result, flush=True)
    return result["status"] == "PASS"


def train(stage, mode, seed, cfg, data, output, prov, deadline):
    device = prov["device"]
    artifact = checkpoint(output, stage, mode, seed)
    if artifact.exists():
        raise FileExistsError(artifact.name)
    run = {"stage": stage, "mode": mode, "seed": seed}
    begin = time.perf_counter()
    allowed = min(deadline, begin + cfg["resource_seconds"]["single_run"])
    rng = random.Random(cfg["augmentation_seed"] + seed)
    model = make_model(mode, cfg, device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    training = data["train"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"START {stage} {mode} seed={seed}", flush=True)
    stage_status, problem, sha = "FAIL", "", ""
    updates, tokens = 0, 0
    try:
        baseline, _ = validation(model, data["validation"], cfg, device)
        append_rows(output / "train_curves.csv", [{**run, "step": 0, "mean_train_loss": None,
                    "mean_train_crop_mae": None, "gradient_l2": None, **baseline}])
        interval_loss, interval_err = [], []
        for step in range(1, cfg["train_updates"] + 1):
            if time.perf_counter() > allowed:
                raise TimeoutError("Pre-registered GPU run budget exhausted")
            model.train()
            batch = [training_crop(training[rng.randrange(len(training))], cfg, rng)
                     for _ in range(cfg["batch_size"])]
            reads = ready_crops(batch, model, device)
            losses = torch.stack([objective(pred, target, cfg) for pred, target in reads])
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["gradient_clip_l2"], error_if_nonfinite=True
            )
            optimizer.step()
            updates = step
            tokens += 0 if mode == "C0" else cfg["model"]["rounds"] * 4 * sum(t.numel() for _, t, _ in batch)
            interval_loss.append(loss.item())
            interval_err.extend(abs(p0.sum().item() - y0.sum().item())
                                for p, y in reads for p0, y0 in zip(p.detach(), y))
            append_rows(output / "crop_ledger.csv",
                        [{**run, "step": step, "batch_slot": i, **ledger} for i, (_, _, ledger) in enumerate(batch)])
            if step % cfg["log_period"] == 0 or step == cfg["train_updates"]:
                grad = grad.item()
                if not math.isfinite(grad):
                    raise FloatingPointError(f"Nonfinite gradient at step {step}")
                current, records = validation(model, data["validation"], cfg, device)
                row = {**run, "step": step, "mean_train_loss": mean(interval_loss),
                       "mean_train_crop_mae": mean(interval_err), "gradient_l2": grad, **current}
                append_rows(output / "train_curves.csv", [row])
                print(f"{mode} s{seed} {step} loss={row['mean_train_loss']:.5f} "
                      f"val MAE={row['mae']:.4f} GAME1={row['game1']:.4f}", flush=True)
                interval_loss.clear()
                interval_err.clear()
        append_rows(output / "validation_per_image.csv", [{**run, **row} for row in records])
        torch.save(model.state_dict(), artifact)
        sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        stage_status = "PASS"
    except Exception as error:
        stage_status = "BLOCKED" if isinstance(error, TimeoutError) else "FAIL"
        problem = type(error).__name__
        # File paths, images and original annotations never enter published evidence.
        traceback.print_exc()
    append_rows(output / "runs.csv", [{
        **run, **prov, "status": stage_status, "error_type": problem, "updates": updates, "token_visits": tokens,
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "wall_time_s": time.perf_counter() - begin, "checkpoint_sha256": sha,
    }])
    print(f"{stage_status} {stage} {mode} s{seed} updates={updates}", flush=True)
    return stage_status == "PASS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--phase", choices=("audit", "overfit", "screen", "pilot"), required=True)
    parser.add_argument("--modes", nargs="+")
    args = parser.parse_args()
    output = args.output
    cfg, data, prov = setup(output, args.device)
    if args.phase == "audit":
        return 0
    if args.phase == "overfit":
        return int(not overfit(cfg, data, output, args.device))
    if read_json(output / "sanity_gate.json")["status"] != "PASS":
        raise RuntimeError("Real-data crop overfit sanity failed")
    seeds = [0]
    if args.phase == "pilot":
        if read_json(output / "validation_gate.json")["status"] != "PASS":
            print("Pilot and official test NOT_RUN due to validation gate", flush=True)
            return 0
        seeds = cfg["train_seeds"][1:]
    modes = cfg["train_modes"] if args.modes is None else args.modes
    if any(m not in cfg["train_modes"] for m in modes):
        parser.error("Modes must be in the pre-registered training matrix")
    spent = 0.
    if (output / "runs.csv").exists():
        with (output / "runs.csv").open(encoding="utf-8", newline="") as stream:
            spent = sum(float(row["wall_time_s"]) for row in csv.DictReader(stream))
    deadline = time.perf_counter() + cfg["resource_seconds"]["whole_round"] - spent
    results = [train(args.phase, mode, seed, cfg, data, output, prov, deadline)
               for seed in seeds for mode in modes]
    return int(not all(results))


if __name__ == "__main__":
    raise SystemExit(main())
