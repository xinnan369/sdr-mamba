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

## SDR-003 propagation and boundary diagnostics

The staged scripts keep the existing SDR-002 data generator, draw fresh ID maps, and use a fixed nested-U OOD family. The validation screen selects the depth protocol. The held-out Stage A gate determines whether Stage B runs the full comparison or only a seed-0 validation diagnostic.

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python -m scripts.train_sdr003 --phase audit --output runs/sdr003-audit
python -m pytest -ra -s
python -m scripts.train_sdr003 --phase smoke --output runs/sdr003-smoke
python -m scripts.eval_sdr003 --phase smoke --output runs/sdr003-smoke
$sdrPython = "$env:USERPROFILE\.conda\envs\pytorch\python.exe"
$sdrOut = "runs/sdr003-pilot"
& $sdrPython -m scripts.train_sdr003 --phase a-screen --device cuda --output $sdrOut
& $sdrPython -m scripts.train_sdr003 --phase a-complete --device cuda --output $sdrOut
python -m scripts.eval_sdr003 --phase a-validation --output $sdrOut
& $sdrPython -m scripts.eval_sdr003 --phase a --device cuda --output $sdrOut
& $sdrPython -m scripts.train_sdr003 --phase b --device cuda --output $sdrOut
& $sdrPython -m scripts.eval_sdr003 --phase b --device cuda --output $sdrOut
python -m scripts.summarize_sdr003 $sdrOut --distance-archive runs/sdr003-pilot/distances.jsonl.gz
```

Use an available Python 3.9+ / PyTorch 2.5+ source environment. The CPU commands use the explicit recurrence's dense short-window formulation; the same implementation runs on CUDA. For the actual stage decisions, provenance, and official SSD status, see [the SDR-003 report](reports/SDR-003/REPORT.md).

## SDR-004 line-aligned state transport

The validation screen selects the line-state dimension and determines whether to start the formal pilot. Only the formal evaluation reads trained predictions on the three held-out splits.

```powershell
$env:PYTHONPATH = "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$sdrPython = "$env:USERPROFILE\.conda\envs\pytorch\python.exe"
$sdrOutput = "runs/sdr004-repro"
python -m scripts.train_sdr004 --phase audit --output runs/sdr004-audit
python -m pytest -ra -s
python -m scripts.train_sdr004 --phase smoke --output runs/sdr004-smoke
python -m scripts.eval_sdr004 --phase smoke --output runs/sdr004-smoke
& $sdrPython -m scripts.train_sdr004 --phase screen --device cuda --output $sdrOutput
& $sdrPython -m scripts.train_sdr004 --phase pilot --device cuda --output $sdrOutput
& $sdrPython -m scripts.eval_sdr004 --phase screen --device cuda --output $sdrOutput
& $sdrPython -m scripts.eval_sdr004 --phase pilot --device cuda --output $sdrOutput
python -m scripts.summarize_sdr004 $sdrOutput --distance-archive runs/sdr004-repro/distances.jsonl.gz
```

See the [SDR-004 report](reports/SDR-004/REPORT.md) for the actual sharded run, resource budgets, and stage decisions.

## SDR-005 ShanghaiTech Part B

The real-data screen compares seven crowd-counting models at a common budget. It reads an existing, authorized Part B directory from `SDR005_DATA_ROOT` and uses a validated PyTorch CUDA SSD oracle. The [SDR-005 report](reports/SDR-005/REPORT.md) contains the reproduction commands, validation decision, and official SSD installation record.

## SDR-006 final feasibility audit

The five-arm ShanghaiTech Part B audit uses a common multiscale recipe and a preregistered development checkpoint rule. The [SDR-006 report](reports/SDR-006/REPORT.md) records the annotation protocol, development gate, and reproduction commands. The [project review](reports/PROJECT_REVIEW.md) traces the evidence across all six rounds.

## MAR-001B readout diagnostics

The frozen-weight diagnostic tests whether channel-mean drift explains overcounting beyond the training depth. The [MAR-001B report](reports/MAR-001B/REPORT.md) contains the preregistered decision, matched-image results, and CUDA reproduction commands. It uses the same development data and checkpoints as [MAR-001](reports/MAR-001/REPORT.md).

## LM-001 CPU comparison platform

The five-arm point-counting platform reuses pinned LoopCrowd components and the verified ShanghaiTech Part B development split. See the [CPU audit report](reports/LM-001/REPORT.md) and [frozen protocol and GPU budget proposal](reports/LM-001/PROTOCOL.md). GPU training is pending authorization.
