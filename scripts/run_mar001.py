"""Run frozen MAR-001 depth diagnostics on the SDR-006 development set only."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts.audit_dataset import append_rows, digest_config, write_json
from scripts.eval_sdr005 import read_csv
from scripts.eval_sdr006 import load_selected
from scripts.train_sdr003 import git, read_json
from sdr_mamba.crowd_audit import project_points
from sdr_mamba.crowd_data import decode, mass_map, parse_points, sha_file, tensor_image
from sdr_mamba.mar_metrics import Mass, background, decomposition, measure, retotal

HISTORY = Path("reports/SDR-006")


def development_entries(manifest):
    dev = manifest["development"]
    entries = dev["entries"]
    ids = {entry["id"] for entry in entries}
    hashes = {entry["image_sha256"] for entry in entries}
    if len(ids) != 60 or len(hashes) != 60 or len(entries) != 60:
        raise ValueError("Expected exactly 60 unique development records")
    for split, group in manifest.items():
        if split != "development":
            if any(r["id"] in ids or r["image_sha256"] in hashes for r in group["entries"]):
                raise ValueError("Development split overlaps a forbidden or training split")
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['id']}:{entry['image_sha256']}:{entry['annotation_sha256']}\n".encode())
    if digest.hexdigest() != dev["sha256"]:
        raise ValueError("Development manifest hash mismatch")
    return dev


def load_development(root, manifest, model_config):
    dev = development_entries(manifest)
    official_train = root / "train_data"
    paths = {
        hashlib.sha256(f"sdr005:B:train_data:{path.stem}".encode()).hexdigest()[:16]: path
        for path in (official_train / "images").glob("*.jpg")
    }
    samples = []
    for entry in dev["entries"]:
        file = paths[entry["id"]]
        annotation = official_train / "ground_truth" / f"GT_{file.stem}.mat"
        if sha_file(file) != entry["image_sha256"] or sha_file(annotation) != entry["annotation_sha256"]:
            raise ValueError(f"Development data SHA-256 mismatch for {entry['id']}")
        with Image.open(file) as image:
            width, height = image.size
            image.verify()
        if (width, height) != (entry["width"], entry["height"]):
            raise ValueError("Development resolution mismatch")
        points = parse_points(annotation)
        if len(points) != entry["full_count"]:
            raise ValueError("Development count mismatch")
        projected = project_points(points, width, height, model_config["coordinate_base"])
        target = mass_map(projected, height, width, model_config["stride"]).numpy()
        if target.sum() != entry["full_count"]:
            raise ValueError("Projection failed to preserve original ground-truth count")
        samples.append((entry, file, Mass.from_cells(target)))
    return dev, samples


def setup(output, phase):
    config = read_json(Path("configs/mar001.json"))
    model_config = read_json(Path("configs/sdr006.json"))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, samples = load_development(Path(os.environ["MAR001_DATA_ROOT"]),
                                         read_json(HISTORY / "split_manifest.json"), model_config)
    selection = {row["mode"]: row for row in read_csv(HISTORY / "checkpoint_selection.csv")
                 if int(row["seed"]) == config["seed"] and row["mode"] in config["models"]}
    checkpoint_home = Path(os.environ["MAR001_CHECKPOINT_HOME"]) if "MAR001_CHECKPOINT_HOME" in os.environ else HISTORY
    checkpoints = []
    for mode in config["models"]:
        row = selection[mode]
        name = f"{mode}-s{config['seed']}-{row['step']}.pt"
        path = checkpoint_home / "checkpoints" / name
        if sha_file(path) != row["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint hash mismatch for {mode}")
        checkpoints.append({"mode": mode, "seed": config["seed"], "file": name,
                            "bytes": path.stat().st_size, "sha256": row["checkpoint_sha256"],
                            "train_depth": model_config["modes"][mode], "step": int(row["step"])})
    output.mkdir(exist_ok=True, parents=True)
    marker = output / f"{phase}_environment.json"
    if marker.exists():
        raise FileExistsError("Use a fresh output directory for a repeated attempt")
    write_json(output / "data_manifest.json", {"split": config["split"], **manifest})
    write_json(output / "checkpoint_manifest.json", {
        "source_commit": config["base_commit"], "selection_config_sha256": digest_config(model_config),
        "checkpoints": checkpoints,
    })
    write_json(marker, {
        "phase": phase, "code_commit": git("rev-parse", "HEAD"), "config_sha256": digest_config(config),
        "dataset_sha256": manifest["sha256"], "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "python": platform.python_version(), "platform": platform.platform(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    })
    return config, model_config, samples, selection, checkpoint_home


def recurrent_step(model, feature):
    block = model.blocks[0]
    return block(feature) if model.audit_mode == "F3" else block(feature, (None,)*4, (None,)*4)[0]


def density(model, feature):
    return F.softplus(model.head(feature).squeeze(-1))


def rms(tensor):
    return float(tensor.square().mean().sqrt())


def trajectory(model, x, limit):
    features = [model.encoder(x).permute(0, 2, 3, 1)]
    predictions = []
    for _ in range(limit):
        features.append(recurrent_step(model, features[-1]))
        predictions.append(density(model, features[-1]))
    return features, predictions


def round_stats(previous, current):
    step = current - previous
    magnitude = rms(current)
    update = rms(step)
    before = rms(previous)
    return {"feature_rms": magnitude, "centered_feature_rms": rms(current-current.mean((-3,-2,-1), keepdim=True)),
            "channel_centered_feature_rms": rms(current-current.mean(-1, keepdim=True)),
            "update_rms": update, "update_relative_to_input_rms": update / before if before else None,
            "feature_cosine": float(F.cosine_similarity(previous.flatten(), current.flatten(), dim=0))}


def variants(truth, pred, ref):
    return [("observed", pred.cells), ("reference", ref.cells),
            ("fixed_mass", retotal(pred, ref.count)), ("fixed_shape", retotal(ref, pred.count)),
            ("oracle_mass", retotal(pred, truth.count))]


@torch.inference_mode()
def evaluate(mode, model, samples, config, model_config, output, deadline):
    observed, decomposed, counterfactual = [], [], []
    for index, (entry, path, truth) in enumerate(samples):
        if time.perf_counter() >= deadline:
            raise TimeoutError("MAR-001 GPU budget reached")
        x = tensor_image(decode(path), model_config).unsqueeze(0).cuda()
        frames, densities = trajectory(model, x, config["depths"]["maximum"])
        if index == 0:
            original = model(x, 8)
            for a, b in zip(densities, original):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        ref = Mass.from_cells(densities[config["depths"]["reference"] - 1][0].cpu().numpy())
        mask = background(truth.cells, config["background_radius_cells"])
        previous = None
        for depth, predicted in enumerate(densities, 1):
            pred = Mass.from_cells(predicted[0].cpu().numpy())
            meta = {"mode": mode, "seed": config["seed"], "id": entry["id"], "t": depth,
                    "train_depth": model.rounds, "extrapolation": depth > model.rounds}
            metrics = measure(truth, pred, mask, config)
            observed.append({**meta, **metrics, **round_stats(frames[depth-1], frames[depth]),
                             "count_update": pred.count-previous.count if previous is not None else None,
                             "density_update_l1": float(np.abs(pred.cells-previous.cells).sum()) if previous is not None else None})
            decomposed.append({**meta, "gt": truth.count, "count": pred.count,
                               **decomposition(truth, pred, config["epsilon"])})
            for label, values in variants(truth, pred, ref):
                if values is None:
                    metrics_variant = dict.fromkeys(metrics)
                    terms = dict.fromkeys(decomposed[-1].keys() - meta.keys() - {"gt", "count"})
                    metrics_variant["gt"] = truth.count
                else:
                    changed = Mass.from_cells(values)
                    metrics_variant = metrics if label == "observed" else measure(truth, changed, mask, config)
                    terms = {k: v for k, v in decomposition(truth, changed, config["epsilon"]).items() if k in
                             ("count_kl", "shape_kl", "total_kl")}
                counterfactual.append({**meta, "variant": label, "status": "DEFINED" if values is not None else "UNDEFINED",
                                       **metrics_variant,
                                       **{k: terms.get(k) for k in ("count_kl", "shape_kl", "total_kl")}})
            previous = pred
        if index == 0 or (index + 1) % 15 == 0:
            print(f"{mode} complete development trajectories {index+1}/{len(samples)}", flush=True)
        append_rows(output / "per_image_depth.csv", observed)
        append_rows(output / "decomposition.csv", decomposed)
        append_rows(output / "counterfactual.csv", counterfactual)
        observed.clear()
        decomposed.clear()
        counterfactual.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, model_config, samples, selected, checkpoint_home = setup(args.output, "depth")
    start = time.perf_counter()
    for mode in config["models"]:
        status = "FAIL"
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        try:
            model = load_selected(checkpoint_home, mode, config["seed"], model_config, selected[mode])
            evaluate(mode, model, samples, config, model_config, args.output, start + config["gpu_seconds_cap"])
            status = "PASS"
        finally:
            append_rows(args.output / "resource_log.csv", [{"phase": "depth", "model": mode, "status": status,
                        "seconds": time.perf_counter() - begin, "peak_bytes": torch.cuda.max_memory_allocated()}])


if __name__ == "__main__":
    main()
