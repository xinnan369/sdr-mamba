"""Run the authorized LM-001 gate and the frozen queue using existing local assets."""

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import csv
import io
import json
from pathlib import Path
import statistics
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from scripts.audit_dataset import digest_config, write_json
from scripts.prepare_lm001 import git_read, historical, source, verify_assets
from scripts.train_sdr003 import read_json
from sdr_mamba.lm001 import build_model, recipe


class Budget:
    """Conservative wall time charged to synchronized GPU work and associated I/O."""

    def __init__(self, caps, device, output, spent=0.0):
        self.limits, self.device, self.output = caps, device, output
        self.spent = spent
        self.start = perf_counter()
        self.run_start = self.start
        self.phase_limit = caps["gpu_seconds"]
        self.run_limit = caps["per_run_seconds"]
        self.updates = 0
        self.case = "preflight"
        self.peak_reserved = 0

    def totals(self):
        return {"seconds": self.spent + perf_counter()-self.start, "updates": self.updates,
                "peak_reserved_bytes": self.peak_reserved}

    def check(self, upcoming=None):
        reserve = {None: 0, "train_batch": .50, "validation_image": 2, "checkpoint": 12,
                   "training": 1, "validation_live": 3, "validation_ema": 3, "initialization": 12}[upcoming]
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(self.device))
        if self.peak_reserved > self.limits["peak_reserved_bytes"]:
            raise RuntimeError("LM-001 GPU allocator reserved memory exceeded the 6 GiB ceiling")
        now = perf_counter()
        if self.spent + now-self.start + reserve >= self.phase_limit:
            raise TimeoutError("LM-001 cumulative GPU seconds exhausted")
        if now-self.run_start + reserve >= self.run_limit:
            raise TimeoutError(f"LM-001 elapsed cap exceeded for {self.case}")
        if upcoming == "train_batch" and self.updates >= self.limits["successful_updates"]:
            raise RuntimeError("LM-001 update budget exhausted")

    def record_update(self):
        self.updates += 1
        if self.updates > self.limits["successful_updates"]:
            raise RuntimeError("LM-001 exceeded the successful-update budget")

    @contextmanager
    def phase(self, name):
        self.check(name)
        torch.cuda.synchronize(self.device)
        start = perf_counter()
        status = "FAIL"
        try:
            yield
            torch.cuda.synchronize(self.device)
            self.check()
            status = "PASS"
        finally:
            row = {"case": self.case, "phase": name, "status": status, "seconds": perf_counter()-start,
                   **{k: v for k, v in self.totals().items() if k != "seconds"}}
            with (self.output / "gpu_phases.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")


def paths(args, config):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("Approved LM-001 hardware is exactly one RTX 3090")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 support is required")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free < config["budget_proposal"]["min_device_free_bytes"]:
        raise RuntimeError(f"Free CUDA memory {free} is below approved launch headroom")
    return device, {"free_bytes": free, "total_bytes": total, "device": torch.cuda.get_device_name(device),
                    "torch": torch.__version__, "cuda_runtime": torch.version.cuda}


def assets(args, config):
    b = config["reference"]["lc007b"]
    archived = json.loads(historical(args.reference, b, "reports/LC-007B/preflight.json"))
    _, verified = verify_assets(config, args.manifest, args.weights, archived)
    return torch.load(args.weights, map_location="cpu", weights_only=True), verified


def phase_history(args, config):
    data = historical(args.reference, config["reference"]["lc007b"], "reports/LC-007B/gpu_phases.csv")
    grouped = defaultdict(list)
    for row in csv.DictReader(io.StringIO(data.decode())):
        phase = row["phase"]
        if phase not in ("training", "validation_live", "validation_ema", "checkpoint"):
            continue
        _, arm = row["case"].split("/")
        grouped[arm, phase].append(float(row["seconds"]))
    return grouped


def preflight(args, config, weights, audit, device, guard, environment):
    from loopcrowd.ema import ParameterEMA, preserve_training_context
    from loopcrowd.experiment import state_sha
    from loopcrowd.grouped_validation import grouped_validation
    from loopcrowd.manifest import Record
    from loopcrowd.targets import point_targets
    from loopcrowd.training import supervised_point_loss

    caps = config["budget_proposal"]
    guard.phase_limit = guard.run_limit = caps["gpu_preflight_seconds"]
    history = phase_history(args, config)
    history_names = {"cnn_only": "cnn_only", "shared_1": "shared_1",
                     "shared_2": "shared_2_aux_025", "unshared_2": "unshared_2_aux_025"}
    starts = digest_by_key(audit)
    results = {}
    with tempfile.TemporaryDirectory(prefix="lm001-gpu-preflight-") as directory:
        root = Path(directory)
        rng = np.random.default_rng(170601)
        fixtures = []
        for index in range(4):
            name = f"fixture{index}.png"
            Image.fromarray(rng.integers(0, 256, size=(768, 1024, 3), dtype=np.uint8)).save(root / name)
            points = rng.uniform((0, 0), (1024, 768), size=(80+40*index, 2)).tolist()
            fixtures.append(Record(str(index), "lm001_fixture", "val", name, 1024, 768, points))
        images = torch.rand(2, 3, 512, 512, generator=torch.Generator().manual_seed(867)).to(device)
        labels = point_targets([[19., 21.], [53., 54.], [74., 90.], [476., 483.]], 512, 512)
        targets = {key: torch.stack((value,value)).to(device) for key, value in labels.items()}
        for arm in config["order"]["42"]:
            guard.case = f"preflight/{arm}"
            guard.check("initialization")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            run = recipe(config, arm, 42)
            model = build_model(config, arm, 42, weights)
            if state_sha(model) != starts[42, arm]:
                raise ValueError("CPU-platform checksum changed before moving to CUDA")
            model.to(device).train()
            ema = ParameterEMA(model, run["ema_decay"])
            samples, losses = [], []
            checked = next(model.parameters()).detach().clone()
            for index in range(16):
                guard.check("train_batch")
                torch.cuda.synchronize(device)
                start = perf_counter()
                model.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = supervised_point_loss(model, images, targets, run)
                loss.backward()
                gradients = [p.grad for p in model.parameters()]
                if any(g is None for g in gradients) or not torch.isfinite(
                    torch.stack([loss.detach(), *torch._foreach_norm(gradients)])
                ).all():
                    raise FloatingPointError(f"Nonfinite or missing BF16 gradients in {arm}")
                torch.cuda.synchronize(device)
                elapsed = perf_counter()-start
                if index >= 4:
                    samples.append(elapsed)
                    losses.append(float(loss.detach()))
                guard.check()
            model.zero_grad(set_to_none=True)
            scans = {}
            with preserve_training_context(model, device):
                for policy, swap in (("live", nullcontext()), ("ema", ema.use_shadow())):
                    guard.check("validation_image")
                    start = perf_counter()
                    with swap:
                        grouped_validation(model, fixtures, root, run, device,
                                           staged=run["early_loss_weight"] is not None,
                                           boundaries=audit["assets"]["density_boundaries"],
                                           check=lambda: guard.check("validation_image"))
                        torch.cuda.synchronize(device)
                    scans[policy] = perf_counter()-start
            peak = torch.cuda.max_memory_reserved(device)
            budget_peak = int(peak + 3*sum(p.numel()*p.element_size() for p in model.parameters()) + (512<<20))
            original = history_names[run["variant"]]
            historical_epoch = {p: float(np.quantile(history[original,p], .95)) for p in
                                ("training", "validation_live", "validation_ema", "checkpoint")}
            conservative_training = max(1.50 * float(np.quantile(samples, .90))*caps["updates_per_epoch"]+2,
                                        1.3*historical_epoch["training"])
            conservative_evaluation = sum(max(1.5*10*scans[p],1.3*historical_epoch[f"validation_{p}"])
                                          for p in ("live","ema"))
            conservative_checkpoint = max(1, 2*historical_epoch["checkpoint"])
            estimated_run = run["epochs"]*(conservative_training+conservative_evaluation+conservative_checkpoint)+20
            if not torch.equal(checked, next(model.parameters())) or ema.step:
                raise RuntimeError("The zero-update preflight changed a frozen parameter")
            results[arm] = {"samples_s": samples, "synthetic_loss_mean": statistics.mean(losses), "optimizer_updates": 0,
                            "validation_four_images_seconds": scans, "historical_epoch_p95_seconds": historical_epoch,
                            "observed_reserved_bytes": peak, "projected_reserved_with_margin_bytes": budget_peak,
                            "estimated_seconds_per_run": estimated_run}
            print(f"{guard.case} elapsed {guard.totals()['seconds']:.1f}s, projected {estimated_run:.1f}s/run, "
                  f"GPU reserved with margin {budget_peak/(1024**3):.3f} GiB", flush=True)
            del model, ema, gradients, loss, checked
    guard.check()
    elapsed = guard.totals()["seconds"]
    projected = elapsed + sum(2*r["estimated_seconds_per_run"] for r in results.values())
    checks = {"duration": elapsed <= caps["gpu_preflight_seconds"],
              "training_run_projection": all(r["estimated_seconds_per_run"] <= caps["per_run_seconds"] for r in results.values()),
              "all_runs_projection": projected <= caps["gpu_seconds"],
              "reserved_with_margin": all(r["projected_reserved_with_margin_bytes"] <= caps["peak_reserved_bytes"] for r in results.values()),
              "zero_optimizer_updates": guard.updates == 0}
    write_json(args.output / "preflight.json", {"status": "PASS" if all(checks.values()) else "BLOCKED",
                "code_commit": git_read(Path.cwd(), "rev-parse", "HEAD").decode().strip(),
                "config_sha256": digest_config(config), "environment": environment, "checks": checks,
                "elapsed_seconds": elapsed, "projected_total_seconds": projected, "arms": results})
    return all(checks.values())


def digest_by_key(audit):
    return {(int(seed), arm): row["initial_sha256"] for seed, group in audit["cases"].items() for arm,row in group.items()}


def historical_epoch_pairs(args, config, seed, audit_rows):
    b = config["reference"]["lc007b"]
    data = historical(args.reference, b, f"reports/LC-007B/epochs/{seed}/cnn_only/training.csv")
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    if len(rows) != 40 or any(int(row["optimizer_updates"]) != 180 or int(row["amp_skipped_updates"]) for row in rows):
        raise ValueError("The reference optimizer and crop ledger is incomplete")
    paths = [(r["crop_order_sha256"],r["learning_rates_sha256"]) for r in rows]
    if any([h for h, _ in paths] != r["crop_epoch_sha256"] for r in audit_rows.values()):
        raise ValueError("Historical crop hashes differ from the LM-001 CPU freeze")
    return paths


def train(args, config, audit, prior_updates=0):
    from loopcrowd import training

    approved = read_json(args.output / "preflight.json")
    if approved["status"] != "PASS" or approved["config_sha256"] != digest_config(config):
        raise ValueError("A passed preflight for this exact configuration is required")
    queue = args.output / "queue.json"
    if queue.exists():
        raise FileExistsError("This authorized queue was already started")
    if git_read(Path.cwd(), "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("Formal runs require the committed LM-001 launch code")
    launch_sha = git_read(Path.cwd(), "rev-parse", "HEAD").decode().strip()
    device, environment = paths(args, config)
    guard = Budget(config["budget_proposal"], device, args.output, spent=approved["elapsed_seconds"])
    guard.updates = prior_updates
    planned_runs = sum(map(len, config["order"].values()))
    planned_updates = planned_runs * config["recipe"]["epochs"] * config["budget_proposal"]["updates_per_epoch"]
    if prior_updates + planned_updates > config["budget_proposal"]["successful_updates"]:
        raise ValueError("The focused queue would exceed the originally authorized aggregate update limit")
    state = {"status": "RUNNING", "config_sha256": digest_config(config), "code_commit": launch_sha,
             "environment": environment, "preflight_seconds": approved["elapsed_seconds"],
             "planned_runs": planned_runs, "planned_updates": planned_updates, "prior_updates": prior_updates,
             "completed": [], "current": None, "official_test": "NOT_RUN",
             "external_download_bytes": 0}
    write_json(queue, state)
    try:
        weights, verified = assets(args, config)
        if verified != audit["assets"]:
            raise ValueError("Runtime assets differ from the frozen CPU audit")
        base_weight_provenance = {"requested": "IMAGENET1K_V1", "loaded": True, "sha256": verified["weights_sha256"]}
        starts = digest_by_key(audit)
        counts = config["budget_proposal"]
        for seed_text, arms in config["order"].items():
            seed = int(seed_text)
            trajectory = historical_epoch_pairs(args, config, seed, audit["cases"][seed_text])
            for arm in arms:
                guard.case = f"{seed}/{arm}"
                guard.run_start = perf_counter()
                guard.check("initialization")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                state["current"] = {"seed": seed, "arm": arm, **guard.totals()}
                write_json(queue, state)
                print(f"START {guard.case} {guard.totals()}", flush=True)
                run = recipe(config, arm, seed)
                output = args.output / "runs" / seed_text / arm
                before = guard.updates
                expected = {"initial_sha256": starts[seed,arm], "trajectory": trajectory}

                def make_frozen_model(run_config, initialize):
                    if not initialize or run_config["seed"] != seed or run_config["variant"] != run["variant"]:
                        raise ValueError("Unexpected model initialization outside the frozen case")
                    return build_model(config, arm, seed, weights)

                with patch.object(training, "make_model", make_frozen_model), patch.object(
                        training, "weight_provenance", lambda *unused: base_weight_provenance):
                    result = training.fit(run, args.manifest, output, device, guard=guard, expected=expected)
                consumed = guard.updates - before
                if (result["execution"], result["completed_epochs"], result["global_step"], consumed) != (
                        "completed", run["epochs"], run["epochs"]*counts["updates_per_epoch"],
                        run["epochs"]*counts["updates_per_epoch"]):
                    raise RuntimeError("Upstream runner did not complete the registered successful updates")
                state["completed"].append({"seed": seed, "arm": arm, "seconds": perf_counter()-guard.run_start,
                                           "updates": consumed, "checkpoint_sha256": result["checkpoints"]["last.pt"]["sha256"]})
                state["current"] = None
                write_json(queue, state)
                print(f"COMPLETE {guard.case} {guard.totals()}", flush=True)
        if guard.updates != prior_updates + planned_updates:
            raise RuntimeError("Final successful-update sum differs from the freeze")
        guard.check()
        state["status"] = "PASS"
    except Exception as failure:
        state.update(status="BLOCKED", error_type=type(failure).__name__, reason=str(failure))
        raise
    finally:
        state.update(guard.totals())
        write_json(queue, state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "train"), required=True)
    args = parser.parse_args()
    config = read_json(Path("configs/lm001.json"))
    audit = read_json(Path("reports/LM-001/cpu_audit.json"))
    if audit["config_sha256"] != digest_config(config):
        raise ValueError("Frozen CPU configuration differs")
    args.reference, args.manifest, args.weights = (p.resolve() for p in (args.reference,args.manifest,args.weights))
    if args.phase == "preflight":
        args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with source(args.reference, config["reference"]["lc007g"]):
        if args.phase == "train":
            return train(args, config, audit)
        try:
            device, environment = paths(args, config)
            guard = Budget(config["budget_proposal"], device, args.output)
            weights, verified = assets(args, config)
            if verified != audit["assets"]:
                raise ValueError("Runtime assets differ from the frozen CPU audit")
            return int(not preflight(args, config, weights, audit, device, guard, environment))
        except Exception as error:
            write_json(args.output / "preflight.json", {"status": "BLOCKED", "error_type": type(error).__name__,
                       "elapsed_seconds": guard.totals()["seconds"] if "guard" in locals() else 0,
                       "optimizer_updates": 0})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
