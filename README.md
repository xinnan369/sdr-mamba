# SDR-Mamba

SDR-001 tests whether sending internal state between windows helps seeded connected-component segmentation. The model uses an explicit PyTorch Mamba-2-style recurrence and runs on CPU. Its optional official SSD adapter is covered by a conditional CUDA parity test.

Run from the repository root with Python 3.11+, PyTorch 2.4+ and pytest 8+ already available. The tested versions are recorded with the runs. No dataset or weights are needed.

```powershell
python -m pytest -ra -s
$env:PYTHONPATH = "src"
python -m scripts.run_experiment --smoke --output runs/smoke
python -m scripts.run_experiment --config configs/pilot.json --output runs/pilot
```

Each command that writes results takes a fresh output directory. The smoke does two updates with seed 0 for all five modes. The pilot compares all five modes at fixed training depth and evaluates identical weights at multiple depths. The data generator writes per-split and per-example SHA-256 hashes so a rerun can verify its local data.

See [the SDR-001 report](reports/SDR-001/REPORT.md) for the definitions, results, and parity status.

## SDR-002 balanced seed benchmark

Run the data audit, tests, two-update smoke and gated 512-update pilot with locally available PyTorch and pytest. SDR-002 also supports a locally available Python 3.9 / PyTorch 2.5 CUDA environment when run directly from source. Each run regenerates the synthetic data from its config and uses a fresh output directory.

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
python -m scripts.audit_dataset --output runs/sdr002-audit
python -m pytest -ra -s
python -m scripts.run_sdr002 --smoke --output runs/sdr002-smoke
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python -m scripts.run_sdr002 --device cuda --output runs/sdr002-pilot
python -m scripts.summarize_sdr002 runs/sdr002-pilot
```

The pilot defaults to CPU if `--device cuda` is omitted. For the full pilot, test metrics and the five-mode matrix run only after the validation gate passes. The fixed checkpoint rule is the last update. The short smoke exercises evaluation code and bypasses the learning gate.

See [the SDR-002 report](reports/SDR-002/REPORT.md) and [the SDR-001 report](reports/SDR-001/REPORT.md) for the separate experiment records.
