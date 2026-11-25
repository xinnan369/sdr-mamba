"""Run original seed recipes in isolated subprocesses and verify joined evidence."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.train_sdr004 import checkpoint, formal_matrix


def rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def key(row):
    return row["stage"], row["mode"], int(row["dstate"]), int(row["seed"])


def atomic_csv(path, values):
    new = path.with_suffix(".tmp")
    with new.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    new.replace(path)


def merge(output, roots, config, state):
    ledger = rows(output / "runs.csv")
    curves = rows(output / "train_curves.csv")
    existing = {key(r) for r in ledger if r["status"] == "PASS"}
    # Running in serial may have been stopped while a trial was in flight.
    # Keep that entire progress journal and publish canonical curves only for completed trials.
    shutil.copy2(output / "train_curves.csv", output / "train_curves_before_sharding.csv")
    curves = [r for r in curves if key(r) in existing]
    manifest = (output / "data_manifest.json").read_bytes()
    for root in roots:
        if (root / "data_manifest.json").read_bytes() != manifest:
            raise RuntimeError(f"Partition regenerated different data: {root}")
        for row in rows(root / "runs.csv"):
            record = key(row)
            source = checkpoint(root, *record)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if row["status"] != "PASS" or digest != row["checkpoint_sha256"] or record in existing:
                raise RuntimeError(f"Unverified or duplicate checkpoint: {source}")
            shutil.copy2(source, checkpoint(output, *record))
            ledger.append(row)
            existing.add(record)
        curves.extend(rows(root / "train_curves.csv"))
    expected = {("pilot", mode, state_size, seed) for mode, state_size in formal_matrix(state)
                for seed in config["train_seeds"]}
    if {record for record in existing if record[0] == "pilot"} != expected:
        raise RuntimeError("Merged pilot matrix is incomplete")
    atomic_csv(output / "runs.csv", ledger)
    atomic_csv(output / "train_curves.csv", curves)
    print(f"Joined {len(expected)} verified held-out pilot checkpoints.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    output = args.output
    config = json.loads((output / "effective_config.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    if selection["status"] != "CONTINUE":
        raise RuntimeError("Validation did not authorize a formal pilot")
    roots, workers = [], []
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        for seed in config["train_seeds"]:
            where = args.shards / f"seed-{seed}"
            where.mkdir(parents=True)
            shutil.copy2(output / "selection.json", where / "selection.json")
            roots.append(where)
            log = output / f"pilot_seed{seed}_log.txt"
            with log.open("wb") as stream:
                worker = subprocess.Popen([
                    args.python, "-u", "-m", "scripts.train_sdr004", "--phase", "pilot", "--device", "cuda",
                    "--output", str(where), "--pilot-seed", str(seed), "--skip-complete-from", str(output),
                ], env=env, stdout=stream, stderr=subprocess.STDOUT)
            workers.append(worker)
            print(f"Started isolated seed {seed} with PID {worker.pid}", flush=True)
        failures = [worker.wait() for worker in workers]
        if any(failures):
            raise RuntimeError(f"Shard exit statuses: {failures}")
        merge(output, roots, config, selection["selected_dstate"])
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
