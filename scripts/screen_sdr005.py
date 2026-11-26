"""Run isolated seed-0 mode shards and join their verified evidence."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

from scripts.pilot_sdr004_shards import atomic_csv, rows
from scripts.train_sdr003 import read_json

WORKERS = (("C0", "C1", "C2"), ("C3", "C5"), ("C4", "C6"))
FILES = ("runs.csv", "train_curves.csv", "crop_ledger.csv", "validation_per_image.csv")


def merge(output, shards):
    data = (output / "split_manifest.json").read_bytes()
    table = {file: [] for file in FILES}
    for path in shards:
        if (path / "split_manifest.json").read_bytes() != data:
            raise ValueError("Screen shard dataset mismatch")
        for file in FILES:
            table[file].extend(rows(path / file))
        for record in rows(path / "runs.csv"):
            name = f"screen-{record['mode']}-s0.pt"
            artifact = path / "checkpoints" / name
            if record["status"] != "PASS" or hashlib.sha256(artifact.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
                raise ValueError(f"Screen checkpoint verification failed for {record['mode']}")
            destination = output / "checkpoints" / name
            if destination.exists():
                raise FileExistsError(name)
            shutil.copy2(artifact, destination)
    config = read_json(output / "effective_config.json")
    modes = [r["mode"] for r in table["runs.csv"]]
    if sorted(modes) != sorted(config["train_modes"]):
        raise ValueError("Incomplete or duplicate screen modes")
    if sum(float(r["wall_time_s"]) for r in table["runs.csv"]) > config["resource_seconds"]["whole_round"]:
        raise TimeoutError("Aggregate screen GPU budget exhausted")
    for name, data in table.items():
        atomic_csv(output / name, data)
    print(f"Joined and verified {len(modes)} screen checkpoints", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    output = args.output
    if read_json(output / "sanity_gate.json")["status"] != "PASS":
        raise RuntimeError("Real-data sanity gate failed")
    if (output / "runs.csv").exists():
        raise FileExistsError("The merged screening data already exist")
    paths, workers = [], []
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        for i, modes in enumerate(WORKERS):
            shard = args.shards / f"worker-{i}"
            shard.mkdir(parents=True)
            paths.append(shard)
            shutil.copy2(output / "sanity_gate.json", shard / "sanity_gate.json")
            with (output / f"screen_worker{i}.txt").open("wb") as stream:
                worker = subprocess.Popen([
                    args.python, "-u", "-m", "scripts.train_sdr005", "--phase", "screen",
                    "--device", "cuda", "--output", str(shard), "--modes", *modes,
                ], env=env, stdout=stream, stderr=subprocess.STDOUT)
            workers.append(worker)
            print(f"Screen shard {i}: {','.join(modes)}", flush=True)
        statuses = [worker.wait() for worker in workers]
        if any(statuses):
            raise RuntimeError(f"Screen worker statuses: {statuses}; raw worker logs retained")
        merge(output, paths)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
