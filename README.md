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
