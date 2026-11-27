"""Read-only CPU audit for the frozen LM-001 platform."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn

from scripts.audit_dataset import digest_config, write_json
from scripts.train_sdr003 import read_json
from sdr_mamba.lm001 import COMMON_PREFIXES, build_model, recipe


def git_read(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


@contextmanager
def source(repo, revision):
    if "loopcrowd" in sys.modules:
        raise RuntimeError("Use a fresh process before importing the pinned reference package")
    payload = git_read(repo, "archive", "--format=zip", revision, "src/loopcrowd")
    with tempfile.TemporaryDirectory(prefix="lm001-source-") as directory, zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(directory)
        src = str(Path(directory) / "src")
        sys.path.insert(0, src)
        try:
            yield
        finally:
            sys.path.remove(src)
            for name in tuple(sys.modules):
                if name == "loopcrowd" or name.startswith("loopcrowd."):
                    del sys.modules[name]


def historical(repo, revision, relative):
    return git_read(repo, "show", f"{revision}:{relative}")


def verify_assets(config, manifest, weights_path, accepted):
    from loopcrowd.accuracy_diagnosis import density
    from loopcrowd.experiment import file_sha
    from loopcrowd.manifest import read_manifest

    reference = config["reference"]
    weight_hash, manifest_hash = file_sha(weights_path), file_sha(manifest)
    if (weight_hash, manifest_hash, weights_path.stat().st_size) != (
        reference["weight_sha256"], reference["manifest_sha256"], reference["weight_bytes"]
    ):
        raise ValueError("Local weight or manifest differs from the frozen LC-007B inputs")
    rows = read_manifest(manifest, splits=("train", "val"))
    train, validation = ([r for r in rows if r.split == split] for split in ("train", "val"))
    sizes = {key: [len(group), sum(len(r.points) for r in group)] for key, group in (("train", train), ("val", validation))}
    if sizes != reference["split_images_heads"] or [r.image_id for r in validation] != accepted["ordered_val_ids"]:
        raise ValueError("Split membership or cleaned label counts differ from LC-007B")
    cuts = np.quantile([density(row) for row in train], (1/3, 2/3)).tolist()
    np.testing.assert_allclose(cuts, reference["train_density_boundaries"], atol=1e-12, rtol=0)
    published = read_json(Path("reports/SDR-006/split_manifest.json"))
    expected_images = {entry["image_sha256"] for split in ("train", "development", "confirmation")
                       for entry in published[split]["entries"]}
    evidence = {}
    all_hashes = set()
    for split, group in (("train", train), ("val", validation)):
        hashed = []
        for row in group:
            image_path = manifest.parent / row.image_path
            image_sha = file_sha(image_path)
            hashed.append((hashlib.sha256(row.image_id.encode()).hexdigest(), image_sha))
            if image_sha in all_hashes:
                raise ValueError("Duplicate JPG bytes across the LM development splits")
            all_hashes.add(image_sha)
            with Image.open(image_path) as im:
                if im.size != (row.image_width, row.image_height):
                    raise ValueError("Frozen dimension differs from source image")
                im.verify()
        evidence[split] = hashlib.sha256(json.dumps(hashed, separators=(",", ":")).encode()).hexdigest()
    if all_hashes != expected_images:
        raise ValueError("The 400 local images differ from the previously audited SHB train population")
    assets = {"weights_sha256": weight_hash, "weights_bytes": weights_path.stat().st_size,
              "manifest_sha256": manifest_hash, "split_images_and_cleaned_heads": sizes,
              "density_boundaries": cuts, "membership_and_image_hashes": evidence,
              "all_image_hashes_match_previously_audited_400_shb_train": True,
              "official_test_inference": "NOT_RUN"}
    return train, assets


def crop_trajectory(train, configurations, manifest_dir):
    from loopcrowd.data import TrainingCrops

    points = [np.asarray(record.points, dtype=float).reshape(-1, 2) for record in train]
    for config in configurations:
        seed = config["seed"]
        for epoch in range(1, config["epochs"]+1):
            dataset = TrainingCrops(train, manifest_dir, config, epoch)
            indices = torch.utils.data.DataLoader(range(len(train)), batch_size=config["batch_size"], shuffle=True,
                                                 generator=torch.Generator().manual_seed(seed+epoch))
            rows = [(i, *dataset.origin(train[i], points[i])) for batch in indices for i in batch.tolist()]
            yield hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def lr_trajectory(model, config):
    from loopcrowd.calibration import json_digest
    from loopcrowd.scheduling import UpdateSchedule
    from loopcrowd.training import optimizer_for

    schedule = UpdateSchedule(optimizer_for(model, config), config, 7200)
    rates = []
    while schedule.global_step < schedule.total_updates:
        rates.append(schedule.apply())
        schedule.global_step += 1
    return {"lr_sha256": json_digest(rates), "warmup_updates": schedule.warmup_updates}


def flops(model, config, arm, size):
    """Dense MAC×2 estimate, matching LC-007G's observable hook convention."""
    counts = dict.fromkeys(("backbone", "decoder", "ffn"), 0)
    handles = []

    def monitor(module, out, key):
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0]*module.kernel_size[1]
            counts[key] += 2*out.numel()*kernel*module.in_channels//module.groups
        if isinstance(module, nn.Linear):
            counts[key] += 2*out.numel()*module.in_features

    def attach(block, key):
        for op in block.modules():
            if type(op) in (nn.Conv2d, nn.Linear):
                handles.append(op.register_forward_hook(lambda layer, inp, out: monitor(layer, out, key)))

    for key in ("p2", "p3", "project", "fuse", "point_logits", "point_offsets"):
        attach(getattr(model, key), "backbone" if key in ("p2", "p3", "project") else "decoder")
    for block in model.blocks:
        attach(block.ffn, "ffn")
    try:
        started = time.perf_counter()
        with torch.inference_mode():
            example = torch.zeros(1, 3, size, size)
            if model.loops:
                model.eval().forward_stages(example)
            else:
                model.eval()(example)
        duration = time.perf_counter()-started
    finally:
        for handle in handles:
            handle.remove()
    g = config["geometry"]
    w, d, inner, s, heads = g["window"], g["width"], g["inner"], g["state"], g["inner"]//g["head_width"]
    side = (size+7)//8
    windows = ((side+w-1)//w)**2
    tokens = windows*w*w
    kind = config["arms"][arm]["mixer"]
    scans = 0
    if kind == "attention":
        per_call = 8*tokens*d*d + 4*tokens*w*w*d
    elif kind == "reference_ssd":
        # Shared projection per cell, then 4*w lines per window with B/C dot
        # products and a separate length-by-length X multiply per head.
        per_call = 2*tokens*(d*(2*inner+2*s+heads)+inner*d) + windows*4*w*2*w*w*(s+inner)
        scans = 4*tokens*model.loops
    else:
        per_call = 0
    decoder_read = counts["decoder"]//(2 if model.loops else 1)
    final = counts["backbone"]+decoder_read+counts["ffn"]+model.loops*per_call
    return {"tile": size, "backbone_flops": counts["backbone"], "one_point_readout_flops": decoder_read,
            "ffn_flops": counts["ffn"], "mixer_flops": model.loops*per_call, "final_only_flops": final,
            "supervised_or_staged_flops": final + (decoder_read if model.loops else 0),
            "scan_token_visits": scans, "cpu_fp32_single_forward_seconds": duration}


def cpu_probe(model, config, boundaries):
    from loopcrowd.ema import ParameterEMA, preserve_training_context
    from loopcrowd.experiment import state_sha
    from loopcrowd.grouped_validation import grouped_validation
    from loopcrowd.manifest import Record
    from loopcrowd.targets import point_targets
    from loopcrowd.training import supervised_point_loss

    device = torch.device("cpu")
    images = torch.rand((2, 3, 80, 96), generator=torch.Generator().manual_seed(1001))
    labels = point_targets([[16., 18.], [20., 19.], [39., 63.]], 96, 80, slots=2)
    batch = {name: torch.stack((value, value)) for name, value in labels.items()}
    model.train()
    ema = ParameterEMA(model, config["ema_decay"])
    before = {k: parameter.detach().clone() for k, parameter in model.named_parameters()}
    diagnostics = {}
    loss = supervised_point_loss(model, images, batch, config, diagnostics)
    loss.backward()
    norms = {name: tensor.grad.detach().norm().item() for name, tensor in model.named_parameters()}
    if not torch.isfinite(loss) or not all(np.isfinite(value) for value in norms.values()):
        raise FloatingPointError("Nonfinite CPU loss or gradient")
    if not all(any(value > 0 for name, value in norms.items() if name.startswith(f"blocks.{i}.attn."))
               for i in range(len(model.blocks))):
        raise ValueError("Missing mixer gradient")
    if not all(torch.equal(before[n], p) for n, p in model.named_parameters()):
        raise ValueError("CPU probe unexpectedly updated weights")
    config_cpu = {**config, "precision": "fp32"}
    with tempfile.TemporaryDirectory(prefix="lm001-smoke-") as tmp:
        path = Path(tmp)
        Image.new("RGB", (640, 544), (139, 93, 68)).save(path / "synthetic.png")
        record = Record("synthetic", "cpu_fixture", "val", "synthetic.png", 640, 544, [[24., 40.], [498., 118.]])
        snapshot, rng = state_sha(model), torch.get_rng_state().clone()
        with preserve_training_context(model, device), ema.use_shadow():
            score, rows = grouped_validation(model, [record], path, config_cpu, device, staged=bool(model.loops),
                                             boundaries=boundaries, check=lambda: None)
        if state_sha(model) != snapshot or not model.training or not torch.equal(torch.get_rng_state(), rng):
            raise ValueError("EMA probe did not restore weights, BN state, mode and RNG")
    return {"loss": float(loss.detach()), "early_supervision": "early_loss" in diagnostics,
            "global_gradient_l2": float(np.linalg.norm(list(norms.values()))),
            "mixer_gradient_l2": float(np.linalg.norm([v for n,v in norms.items() if ".attn." in n])),
            "unique_ema_parameters": len(ema.parameters), "ema_steps": ema.step,
            "optimizer_steps": 0, "synthetic_eval_stages": list(score), "synthetic_eval_rows": len(rows),
            "matching_protocol": {stage: {"threshold": value["nap_score_floor"], "delta": value["local_match_delta"],
                                           "nap_schema": sorted(value["nap_local_v1"])} for stage,value in score.items()}}


def audit(args, config):
    from loopcrowd.experiment import DEFAULT_CONFIG, state_sha

    reference = config["reference"]
    b, g = (reference[f"lc007{key}"] for key in ("b", "g"))
    accepted = json.loads(historical(args.reference, b, "reports/LC-007B/preflight.json"))
    train, assets = verify_assets(config, args.manifest, args.weights, accepted)
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    recipes = {"cnn_only": "cnn", "shared_2": "shared_2_aux_025", "unshared_2": "unshared_2_aux_025"}
    archive_cases = recipes | {"cnn_only": "cnn_only"}
    results, matrix = {}, []
    for seed_str, order in config["order"].items():
        seed = int(seed_str)
        keys, digest_cache = {}, {}
        old_order_bytes = historical(args.reference, b, f"reports/LC-007B/epochs/{seed}/cnn_only/training.csv")
        historical_crop = [row["crop_order_sha256"] for row in csv.DictReader(io.StringIO(old_order_bytes.decode()))]
        for index, arm in enumerate(order, 1):
            print(f"CPU audit {seed}/{arm}", flush=True)
            run = recipe(config, arm, seed)
            old_config = json.loads(historical(args.reference, b, f"configs/lc006_{recipes[run['variant']]}.json"))
            expected = DEFAULT_CONFIG | old_config | {"seed": seed, "ema_decay": .99}
            if run != expected:
                raise ValueError(f"Recipe drift for {arm}: {sorted(k for k in set(run)|set(expected) if run.get(k)!=expected.get(k))}")
            model = build_model(config, arm, seed, weights)
            history = accepted["cases"][f"{seed}/{archive_cases[run['variant']]}"]
            common = state_sha(model, prefixes=COMMON_PREFIXES)
            initial = state_sha(model)
            if common != history["common_sha256"]:
                raise ValueError(f"Historical common initialization differs for {arm}")
            if config["arms"][arm]["mixer"] != "reference_ssd" and initial != history["initial_sha256"]:
                raise ValueError(f"Historical control initial checksum differs for {arm}")
            scheduled = lr_trajectory(model, run)
            if scheduled["lr_sha256"] != history["lr_sha256"] or scheduled["warmup_updates"] != 360:
                raise ValueError("Historical learning-rate trajectory differs")
            geometry = (run["seed"], run["epochs"], run["batch_size"], run["tile_size"], run["crop_mode"])
            if geometry not in digest_cache:
                digest_cache[geometry] = list(crop_trajectory(train, [run], args.manifest.parent))
            if digest_cache[geometry] != historical_crop:
                raise ValueError("Planned crop trajectory differs from LC-007B")
            grad = cpu_probe(model, run, assets["density_boundaries"])
            record = {
                "initial_sha256": initial, "historical_common_sha256": common, "blocks": [state_sha(block) for block in model.blocks],
                "core_parameters": sum(p.numel() for block in model.blocks for p in block.attn.parameters()),
                "non_mixer_block_sha256": [state_sha(block, prefixes=("norm1.", "norm2.", "ffn.")) for block in model.blocks],
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "unique_blocks": len(model.blocks), "calls": model.loops, "shared": model.shared,
                "historical_full_state_exact": initial == history["initial_sha256"],
                "crop_epoch_sha256": digest_cache[geometry],
                **scheduled, **grad,
            }
            if seed == 42:
                record["profile"] = flops(model, config, arm, run["tile_size"])
            keys[arm] = record
            matrix.append({"seed": seed, "order": index, "arm": arm, "mixer": config["arms"][arm]["mixer"],
                           "updates": run["epochs"]*180, "recipe": run, "initial_sha256": initial,
                           "lr_sha256": scheduled["lr_sha256"], "status": "GPU_NOT_AUTHORIZED"})
        for name in ("transformer", "mamba"):
            shared = keys[f"shared_2_{name}"]
            unshared = keys[f"unshared_2_{name}"]
            if len(shared["blocks"]) != 1 or len(unshared["blocks"]) != 2 or (
                shared["blocks"][0] != unshared["blocks"][0] or unshared["blocks"][0] == unshared["blocks"][1]
            ):
                raise ValueError(f"Incorrect tied or untied initialization for {name}")
        for sharing in ("shared", "unshared"):
            if keys[f"{sharing}_2_transformer"]["non_mixer_block_sha256"] != keys[f"{sharing}_2_mamba"]["non_mixer_block_sha256"]:
                raise ValueError("Mismatch outside the replacement mixer")
        results[seed_str] = keys
    phases = list(csv.DictReader(io.StringIO(historical(args.reference, b, "reports/LC-007B/gpu_phases.csv").decode())))
    licensed = bool(git_read(args.reference, "ls-tree", g, "LICENSE", "LICENSE.md", "COPYING"))
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "cpu_audit.json", {
        "status": "PASS", "optimizer_steps": 0, "cuda_visible": torch.cuda.is_available(),
        "code_commit": git_read(Path.cwd(), "rev-parse", "HEAD").decode().strip(),
        "config_sha256": digest_config(config),
        "python": sys.version.split()[0], "torch": torch.__version__, "torchvision": sys.modules["torchvision"].__version__,
        "local_reference_has_top_level_license": licensed, "source_trees": {"LC-007B": b, "LC-007G": g},
        "assets": assets, "cases": results, "historical_007b_gpu_seconds": float(phases[-1]["cumulative_gpu_seconds"]),
        "historical_007b_updates": int(phases[-1]["successful_project_updates"]),
        "geo": {"p3_stride": 8, "tokens_per_window": 64, "scan_length": 8, "scan_directions": 4,
                "interior_per_token_mixer_reach": {"cnn_only": [1, 1], "attention": [64,64], "reference_ssd": [15,64]},
                "cross_window_mixer_paths": False}
    })
    write_json(root / "experiment_matrix.json", {"status": config["status"], "reference": reference,
                                               "geometry": config["geometry"], "runs": matrix,
                                               "budget_proposal": config["budget_proposal"]})
    print(f"Verified {len(matrix)} CPU arm/seed combinations", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the empty string for this CPU-only entry point")
    if torch.cuda.is_available():
        raise RuntimeError("Unexpected CUDA visibility in CPU-only entry point")
    torch.set_num_threads(2)
    config = read_json(Path("configs/lm001.json"))
    with source(args.reference, config["reference"]["lc007g"]):
        audit(args, config)


if __name__ == "__main__":
    main()
