import hashlib
import math

import numpy as np
from PIL import Image
import pytest
import torch

import scripts.run_mar001 as runner
from scripts.perturb_mar001 import directional_jvp, directions
from sdr_mamba.mar_metrics import Mass, aggregate, background, decomposition, measure, retotal, sliced_w1


def test_fp64_kl_matches_cell_formula_and_dtype_inputs():
    q = np.array([[0, 4, 2], [1, 0, 6]])
    d = np.array([[1, 3, 2], [1, 2, 7]])
    computed = []
    for dtype in (np.float32, np.float64):
        y = Mass.from_cells(q.astype(dtype))
        pred = Mass.from_cells(d.astype(dtype))
        actual = decomposition(y, pred, 1e-8)
        valid = q > 0
        assert actual["raw_total_kl"] == pytest.approx(
            (q[valid]*np.log(q[valid]/d[valid])).sum() + d.sum()-q.sum(), abs=1e-13)
        assert actual["total_kl"] == pytest.approx(actual["direct_smoothed_kl"], abs=1e-13)
        assert actual["identity_error"] < 1e-13
        computed.append(actual)
    assert computed[0] == computed[1]


def test_empty_zero_and_structural_zeros_have_distinct_semantics():
    empty = Mass.from_cells(np.zeros((8, 8)))
    truth = Mass.from_cells(np.pad(np.ones((1, 1)), ((0, 7), (0, 7))))
    elsewhere = Mass.from_cells(np.flip(truth.cells, axis=1)*3)
    assert decomposition(empty, elsewhere, 1e-8)["total_kl"] == pytest.approx(3)
    assert decomposition(empty, empty, 1e-8)["total_kl"] == 0
    absent = decomposition(truth, empty, 1e-8)
    assert math.isinf(absent["total_kl"]) and absent["shape_kl"] is None
    sparse = decomposition(truth, elsewhere, 1e-8)
    assert math.isinf(sparse["raw_total_kl"]) and sparse["identity_error"] < 1e-12
    assert sparse["positive_gt_zero_pred_cells"] == 1
    assert retotal(empty, 1) is None
    assert not retotal(empty, 0).any()
    result = measure(empty, elsewhere, background(empty.cells, 1), {"projection_grids": [(2, 2), (4, 4)]})
    assert result["background_fraction"] == 1 and result["tv"] is None


def test_grid_mass_wasserstein_and_retotaling():
    a, b = np.zeros((8, 8)), np.zeros((8, 8))
    a[0, 0], b[-1, -1] = 3, 6
    truth, pred = Mass.from_cells(a), Mass.from_cells(b)
    cfg = {"projection_grids": [(2, 2), (4, 4)]}
    mask = background(a, 1)
    assert not mask[:2, :2].any() and mask.sum() == 60
    assert aggregate(a, 4, 4).sum() == 3
    st = measure(truth, pred, mask, cfg)
    assert st["tv"] == 1 and st["game3"] == 9
    assert st["sw1"] == pytest.approx(.75)
    assert sliced_w1(truth.shape-truth.shape, (4, 4)) == 0
    scaled = Mass.from_cells(retotal(pred, 11))
    assert scaled.count == pytest.approx(11)
    assert measure(truth, scaled, mask, cfg)["sw1"] == st["sw1"]
    assert decomposition(truth, scaled, 1e-8)["shape_kl"] == pytest.approx(decomposition(truth, pred, 1e-8)["shape_kl"])


def test_local_spatial_mean_directions_have_matched_rms_and_support():
    torch.manual_seed(4)
    feature = torch.randn(1, 16, 24, 4)
    spec = {"seed": 0, "local_fraction": .25, "draws": 3}
    result = directions(feature, feature.square(), "anonymous", spec)
    assert len(result) == 10
    for kind, _, direction in result:
        assert runner.rms(direction) == pytest.approx(1, abs=1e-6)
        if kind == "local_zero_spatial_mean":
            torch.testing.assert_close(direction.mean((1, 2)), torch.zeros(1, 4), atol=1e-7, rtol=0)
            assert torch.count_nonzero(direction) <= 4*6*4
        if kind == "common_channel_shift":
            assert torch.equal(direction[..., 0], direction[..., -1])


def test_nonnormal_small_matrix_jvp_is_directional_amplification():
    matrix = torch.tensor([[.5, 4.], [0., .5]], dtype=torch.float64)
    x = torch.tensor([2., 3.], dtype=matrix.dtype)
    v = torch.tensor([0., 1.], dtype=matrix.dtype)
    d, _ = directional_jvp(lambda value: matrix @ value, x, v)
    torch.testing.assert_close(d, matrix @ v)
    assert d.norm() > 1 and torch.linalg.eigvals(matrix).abs().max() < 1


def test_development_loader_never_reads_decoy_splits(tmp_path, monkeypatch):
    train = tmp_path / "train_data"
    (train / "images").mkdir(parents=True)
    (train / "ground_truth").mkdir()
    entries = []
    digest = hashlib.sha256()
    for i in range(60):
        name = f"IMG_{i}"
        path = train / "images" / f"{name}.jpg"
        Image.new("RGB", (16, 16), (i, 0, 0)).save(path, format="PNG")
        annotation = train / "ground_truth" / f"GT_{name}.mat"
        annotation.write_text(name)
        ident = hashlib.sha256(f"sdr005:B:train_data:{name}".encode()).hexdigest()[:16]
        image_hash, ann_hash = runner.sha_file(path), runner.sha_file(annotation)
        digest.update(f"{ident}:{image_hash}:{ann_hash}\n".encode())
        entries.append({"id": ident, "image_sha256": image_hash, "annotation_sha256": ann_hash,
                        "width": 16, "height": 16, "full_count": 1})
    (tmp_path / "test_data").mkdir()
    (tmp_path / "test_data" / "secret.jpg").write_bytes(b"invalid")
    (train / "images" / "CONFIRMATION.jpg").write_bytes(b"invalid")
    other = {"id": "outside", "image_sha256": "other"}
    manifest = {"development": {"entries": entries, "sha256": digest.hexdigest()},
                "confirmation": {"entries": [other]}, "official_test_static": {"entries": [other]}}
    monkeypatch.setattr(runner, "parse_points", lambda _: np.array([[16.2, 3.]]))
    _, records = runner.load_development(tmp_path, manifest, {"coordinate_base": 0, "stride": 8})
    assert len(records) == 60 and all(truth.count == 1 for _, _, truth in records)
    manifest["confirmation"]["entries"].append(entries[0])
    with pytest.raises(ValueError, match="overlaps"):
        runner.load_development(tmp_path, manifest, {"coordinate_base": 0, "stride": 8})
