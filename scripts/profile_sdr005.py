"""Measure the locked seed-0 models at the same depth on validation images."""

import argparse
import hashlib
from pathlib import Path
from statistics import median

import torch

from scripts.audit_dataset import append_rows
from scripts.eval_sdr005 import read_csv
from scripts.train_sdr005 import checkpoint, make_model, setup
from sdr_mamba.crowd_data import decode, tensor_image


@torch.inference_mode()
def profile(output):
    destination = output / "latency_matched_t4.csv"
    if destination.exists():
        raise FileExistsError(destination)
    config, data, _ = setup(output, "cuda")
    # Fix a small validation-only subset before looking at predictions.
    samples = sorted(data["validation"], key=lambda sample: sample.identifier)[:5]
    images = [tensor_image(decode(sample.file), config).unsqueeze(0).cuda() for sample in samples]
    ledger = {row["mode"]: row for row in read_csv(output / "runs.csv") if row["stage"] == "screen"}
    rows = []
    for mode in config["train_modes"]:
        model = make_model(mode, config, "cuda", 0).eval()
        saved = checkpoint(output, "screen", mode, 0)
        if hashlib.sha256(saved.read_bytes()).hexdigest() != ledger[mode]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint integrity mismatch {mode}")
        model.load_state_dict(torch.load(saved, map_location="cuda", weights_only=True))
        depth = 1 if mode == "C0" else config["model"]["rounds"]
        for _ in range(2):
            model(images[0], depth)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(2):
            for image in images:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(image, depth)
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
        rows.append({"mode": mode, "seed": 0, "t": depth, "images": len(images),
                     "timed_passes": len(timings), "median_ms": median(timings),
                     "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
        del model
        print(rows[-1], flush=True)
    append_rows(destination, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    profile(parser.parse_args().output)
