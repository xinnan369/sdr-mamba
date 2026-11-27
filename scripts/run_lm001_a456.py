"""Arm-specific LM-001 update following the A4–A6 scope clarification."""

import argparse
from pathlib import Path
import sys

import torch

from scripts.audit_dataset import digest_config, write_json
from scripts.prepare_lm001 import cpu_probe, git_read, lr_trajectory, source
from scripts.run_lm001_gpu import Budget, assets, paths, preflight, train
from scripts.train_sdr003 import read_json
from sdr_mamba.lm001 import COMMON_PREFIXES, build_model, recipe


def focused_plan():
    scope = read_json(Path("configs/lm001_a456.json"))
    parent = read_json(Path(scope["parent"]))
    arms = {name: parent["arms"][name] if name in parent["arms"] else scope[name] for name in scope["arm_ids"]}
    return parent | {"status": scope["status"], "arms": arms, "order": scope["order"]}


def cpu_audit(args, config, base):
    from loopcrowd.experiment import file_sha, state_sha

    if torch.cuda.is_available():
        raise ValueError("The focused initialization audit requires CUDA hidden")
    if file_sha(args.weights) != config["reference"]["weight_sha256"]:
        raise ValueError("Wrong pretrained backbone")
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    for seed_text, group in base["cases"].items():
        seed = int(seed_text)
        old = group["shared_2_mamba"]
        model = build_model(config, "single_mamba", seed, weights)
        if model.loops != 1 or state_sha(model.blocks[0]) != old["blocks"][0]:
            raise ValueError("A4 must share the exact first-block initialization with A5/A6")
        common = state_sha(model, prefixes=COMMON_PREFIXES)
        if common != old["historical_common_sha256"]:
            raise ValueError("The common architecture drifted")
        run = recipe(config, "single_mamba", seed)
        schedule = lr_trajectory(model, run)
        if schedule["lr_sha256"] != old["lr_sha256"]:
            raise ValueError("The optimizer trajectory drifted")
        one = {"initial_sha256": state_sha(model), "common_sha256": common, "block_sha256": old["blocks"][0],
               "crop_epoch_sha256": old["crop_epoch_sha256"], "parameters": sum(p.numel() for p in model.parameters()),
               **schedule, **cpu_probe(model, run, base["assets"]["density_boundaries"])}
        group["single_mamba"] = one
        print(f"A4 seed {seed}: common/first-block/LR checksum, final-only backward and EMA evaluation PASS", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "cpu_audit.json", {**base, "config_sha256": digest_config(config),
               "focus_code_commit": git_read(Path.cwd(), "rev-parse", "HEAD").decode().strip()})


def focus_preflight(args, config, audit):
    past = read_json(args.prior / "queue.json")
    reusable = read_json(args.prior / "preflight.json")
    if past["status"] != "STOPPED_BY_SCOPE" or reusable["status"] != "PASS":
        raise ValueError("The original queue must be stopped and its preflight passed")
    device, environment = paths(args, config)
    args.output.mkdir(parents=True, exist_ok=False)
    guard = Budget(config["budget_proposal"], device, args.output)
    weights, verified = assets(args, config)
    if verified != audit["assets"]:
        raise ValueError("The focus queue changed assets")
    remaining_preflight = config["budget_proposal"]["gpu_preflight_seconds"] - reusable["elapsed_seconds"]
    if remaining_preflight <= 0:
        raise ValueError("Original GPU preflight budget is exhausted")
    subset = config | {"order": {"42": ["single_mamba"]},
                       "budget_proposal": config["budget_proposal"] | {"gpu_preflight_seconds": remaining_preflight}}
    try:
        passed = preflight(args, subset, weights, audit, device, guard, environment)
    except Exception as failure:
        write_json(args.output / "preflight.json", {"status": "BLOCKED", "error_type": type(failure).__name__,
                                                     "elapsed_seconds": guard.totals()["seconds"], "updates": 0})
        raise
    current = read_json(args.output / "preflight.json")
    charged = past["seconds"] + current["elapsed_seconds"]
    previous = {k:v for k,v in reusable["arms"].items() if k in ("shared_2_mamba", "unshared_2_mamba")}
    all_arms = current["arms"] | previous
    projected = charged + sum(2*r["estimated_seconds_per_run"] for r in all_arms.values())
    caps = config["budget_proposal"]
    checks = current["checks"] | {
        "total_preflight_budget": current["elapsed_seconds"]+past["preflight_seconds"] <= caps["gpu_preflight_seconds"],
        "all_runs_projection": projected <= caps["gpu_seconds"],
        "all_updates_projection": past["updates"]+43200 <= caps["successful_updates"],
    }
    final = passed and all(checks.values())
    write_json(args.output / "preflight.json", {**current, "status": "PASS" if final else "BLOCKED",
               "config_sha256": digest_config(config), "new_preflight_seconds": current["elapsed_seconds"],
               "elapsed_seconds": charged, "prior_updates": past["updates"], "total_projected_seconds": projected,
               "checks": checks, "arms": all_arms, "reused_arms": list(previous)})
    print("A4–A6 gate", "PASS" if final else "BLOCKED", checks, f"total forecast {projected:.1f}s", flush=True)
    return int(not final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu", "preflight", "train"), required=True)
    for key in ("reference","weights","output","cpu-evidence"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    config = focused_plan()
    torch.set_num_threads(2)
    with source(args.reference, config["reference"]["lc007g"]):
        if args.phase == "cpu":
            cpu_audit(args, config, read_json(Path("reports/LM-001/cpu_audit.json")))
            return 0
        if args.manifest is None or args.prior is None:
            parser.error("Preflight and training require --manifest and --prior")
        audit = read_json(args.cpu_evidence)
        if audit["config_sha256"] != digest_config(config):
            raise ValueError("The focused CPU initialization differs")
        if args.phase == "preflight":
            return focus_preflight(args, config, audit)
        return train(args, config, audit, prior_updates=read_json(args.prior / "queue.json")["updates"])


if __name__ == "__main__":
    sys.exit(main())
