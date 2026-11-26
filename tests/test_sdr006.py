from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.audit_sdr006 import select_holdout
from scripts.eval_sdr006 import final_gate, tile_density
from scripts.train_sdr006 import checkpoint_rank, make_model, rate
from scripts.train_sdr003 import read_json
from sdr_mamba.crowd_audit import project_points
from sdr_mamba.crowd_data import in_rect, mass_map, transform_points


def config():
    from pathlib import Path

    return read_json(Path("configs/sdr006.json"))


def test_projection_preserves_original_annotation_mass_at_every_edge():
    raw = np.array([[-20., -2.], [30.1, 19.2], [1.2, 19.3], [12.4, 0.2], [3., 5.]])
    for base in (0, 1):
        points = project_points(raw, 16, 16, base)
        assert in_rect(points, 16, 16).all()
        cropped = transform_points(points, 0, 0, 16, 16, flip=True, output_size=(64, 32))
        assert int(mass_map(cropped, 64, 32, 8).sum()) == len(raw)
    assert (raw[0] == [-20., -2.]).all()


def test_holdout_has_three_per_stratum_stable_under_input_reordering():
    samples = [SimpleNamespace(identifier=f"{i:04d}", points=[None] * (i // 17)) for i in range(340)]
    ids = select_holdout(samples, 706019)
    assert len(ids) == 60
    assert ids == select_holdout(samples[::-1], 706019)
    assert [sum(stratum * 17 <= int(i) < (stratum + 1) * 17 for i in ids) for stratum in range(20)] == [3]*20


def test_untied_axis_blocks_start_as_independent_copies_of_shared_block():
    torch.set_num_threads(1)
    cfg = config()
    shared = make_model("F1", 0, cfg, "cpu")
    untied = make_model("F2", 0, cfg, "cpu")
    single = make_model("F4", 0, cfg, "cpu")
    assert len({block.project.weight.data_ptr() for block in untied.blocks}) == 4
    x = torch.randn(1, 3, 32, 48)
    for a, b in zip(shared(x), untied(x)):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(single(x)[0], shared(x)[0])
    with pytest.raises(ValueError, match="independent-block"):
        untied(x, 5)


def test_tile_assembly_count_and_space_are_preserved_for_local_operator():
    def reference(x, rounds):
        return [x.mean(1)]

    y = torch.arange(24).reshape(1, 1, 4, 6).float()
    torch.testing.assert_close(tile_density(reference, y, 1, size=2), y[0, 0])
    with pytest.raises(ValueError, match="nonoverlapping"):
        tile_density(reference, y, 1, size=4)


def test_optimizer_schedule_and_checkpoint_tiebreak():
    cfg = config()
    assert rate(1, cfg) < rate(cfg["warmup"], cfg)
    assert rate(cfg["warmup"], cfg) == pytest.approx(cfg["lr"])
    assert rate(cfg["updates"], cfg) == pytest.approx(cfg["lr"] * cfg["final_lr_fraction"])
    rows = [
        {"step": 200, "mae": 33, "game1": 50}, {"step": 600, "mae": 33, "game1": 48},
        {"step": 400, "mae": 33, "game1": 48}, {"step": 800, "mae": 34, "game1": 1},
    ]
    assert min(rows, key=checkpoint_rank)["step"] == 400


def test_confirmation_gate_requires_consistency_and_each_control():
    cfg = config()
    rows = [{"mode": mode, "seed": seed, "mae": 90 if mode == "F1" else 100, "rmse": 120, "game1": 110}
            for mode in cfg["modes"] for seed in cfg["seeds"]]
    assert final_gate(rows, cfg)["status"] == "PASS"
    for row in rows:
        if row["mode"] == "F3":
            row["game1"] = 100
    result = final_gate(rows, cfg)
    assert result["status"] == "FAIL"
    assert not result["checks"]["F3_rmse_game1"]
