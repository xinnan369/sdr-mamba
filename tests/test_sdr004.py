import json
from pathlib import Path

import pytest
import torch

from scripts.audit_dataset import audit
from sdr_mamba.data_v4 import build, historical_test_hashes
from sdr_mamba.model_v3 import LoopV3, dense_oracle
from sdr_mamba.model_v4 import LineLoop, inverse_lanes, lanes, neighbors
from sdr_mamba.scan import tile, untile
from sdr_mamba.ssm import oracle


def test_four_physical_lanes_and_exact_inverse_on_non_square_grid():
    image = torch.arange(8 * 12).reshape(1, 8, 12, 1)
    windows = tile(image, 4)
    packed = lanes(windows)
    assert packed.shape == (1, 2, 3, 4, 4, 4, 1)
    assert packed[0, 0, 0, 1, 2, :, 0].tolist() == [27, 26, 25, 24]
    assert packed[0, 0, 0, 3, 2, :, 0].tolist() == [38, 26, 14, 2]
    for plane in inverse_lanes(packed).unbind(3):
        torch.testing.assert_close(untile(plane), image)
    torch.testing.assert_close(lanes(windows.transpose(-3, -2))[:, :, :, 0], packed[:, :, :, 2])
    with pytest.raises(ValueError, match="multiples"):
        LineLoop("line", 1)(torch.zeros(1, 7, 12, 2), 1)


@pytest.mark.parametrize("direction", list(range(4)))
def test_oracle_pulse_reaches_only_the_correct_downstream_physical_lane(direction):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = LineLoop("line", 1)
    dy, dx = ((0, 1), (0, -1), (1, 0), (-1, 0))[direction]
    features = model.encoder(torch.rand(2, 12, 16, 2))
    baseline = model.step(features)
    payload = torch.zeros_like(baseline.state)
    payload[0, 1, 2, direction, 1] = 5
    received = neighbors(payload, "line", mixing=1)
    expected = torch.zeros_like(payload)
    expected[0, 1 + dy, 2 + dx, direction, 1] = 2.5
    torch.testing.assert_close(received, expected, atol=0, rtol=0)
    shift = neighbors(payload, "line", mixing=1, permute=True)
    expected = torch.zeros_like(payload)
    expected[0, 1 + dy, 2 + dx, direction, 0] = 2.5
    torch.testing.assert_close(shift, expected)
    altered = model.step(features, received)
    impact = tile((altered.logits - baseline.logits).unsqueeze(-1), 4).squeeze(-1)
    mask = torch.zeros_like(impact, dtype=torch.bool)
    if direction < 2:
        mask[0, 1 + dy, 2 + dx, 1, :] = True
    else:
        mask[0, 1 + dy, 2 + dx, :, 1] = True
    assert (impact[mask].abs() > 0).all()
    assert (impact[~mask] == 0).all()
    boundary = torch.zeros_like(payload)
    yy = 2 if dy == 1 else 0
    xx = 3 if dx == 1 else 0
    boundary[0, yy, xx, direction, 1] = 5
    assert not neighbors(boundary, "line", mixing=1).any()


def test_reset_equivalence_and_across_round_gradients_no_sample_leakage():
    torch.manual_seed(55)
    image = torch.rand(2, 8, 12, 2)
    model = LineLoop("line", 1)
    tracks = model(image, 3)
    earlier_loss = model.step(tracks[1].feature_in.detach(), tracks[1].initial).logits.square().mean()
    for grad in torch.autograd.grad(earlier_loss, (tracks[0].state, tracks[1].initial)):
        assert grad.isfinite().all() and grad.norm() > 0
    torch.testing.assert_close(model(image[:1], 3)[-1].logits, tracks[-1].logits[:1])
    model.gate = 0
    zero = model(image, 3)[-1].logits
    model.mode = "reset"
    torch.testing.assert_close(model(image, 3)[-1].logits, zero, atol=0, rtol=0)


@pytest.mark.parametrize("length,state", [(4, 1), (4, 4), (16, 4), (9, 4)])
def test_short_window_oracle_fp64_output_final_and_gradients(length, state):
    torch.manual_seed(37)
    t = lambda *shape: torch.randn(shape, dtype=torch.float64, requires_grad=True)
    args = (t(2, length, 2, 8), t(2, length, 2).sigmoid(), -t(2).exp(),
            t(2, length, 1, state), t(2, length, 1, state), t(2), t(2, 2, 8, state))
    y, s = oracle(*args)
    actual = dense_oracle(*args)[:2]
    for got, expected in zip(actual, (y, s)):
        torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-10)
    probes = [torch.randn_like(y), torch.randn_like(s)]
    for got, expected in zip(torch.autograd.grad(actual, args, probes),
                             torch.autograd.grad((y, s), args, probes)):
        torch.testing.assert_close(got, expected, atol=1e-9, rtol=1e-9)


def test_raster_l0_uses_previous_operator_and_global_is_a_rectangular_baseline():
    x = torch.randn(1, 8, 12, 2)
    torch.manual_seed(1974)
    old = LoopV3()
    torch.manual_seed(1974)
    new = LineLoop("raster", 4)
    for a, b in zip(old(x, 3), new(x, 3)):
        torch.testing.assert_close(a.logits, b.logits)
        torch.testing.assert_close(a.state, b.state)
    global_trace = LineLoop("global", 4)(x, 2)
    assert global_trace[-1].logits.shape == (1, 8, 12)
    assert global_trace[-1].initial is None


@pytest.mark.parametrize("mode", ["line", "line_wrong", "line_feature", "local"])
def test_input_seed_swap_is_bounded_to_one_window_per_round(mode):
    model = LineLoop(mode, 1)
    x = torch.ones(2, 16, 20, 2)
    x[..., 1] = 0
    x[0, 0, 0, 1] = x[1, 2, 2, 1] = 1
    with torch.no_grad():
        result = model(x, 3)
    for t, frame in enumerate(result, 1):
        diff = tile((frame.logits[0] - frame.logits[1]).unsqueeze(0).unsqueeze(-1), 4)
        assert diff.abs().max() > 0
        outside = (torch.arange(4)[:, None] + torch.arange(5)[None, :]) >= t
        assert diff[:, outside].abs().max() < 2e-6


def test_seven_fresh_splits_with_rank_matched_three_component_targets():
    config = json.loads((Path(__file__).parents[1] / "configs/sdr004.json").read_text())
    for item in config["data"].values():
        item["maps"] = 2
    datasets, manifest = build(config)
    assert manifest == build(config)[1]
    keys = [m["occupancy_sha256"] for split in manifest.values() for m in split["maps"]]
    assert len(keys) == len(set(keys)) and not set(keys) & historical_test_hashes()
    settings = {"validation_gate": {"all_walkable_max": .55},
                "baselines": {"random_seed": 8, "random_positive_probability": 1 / 3}}
    assert audit(datasets, manifest, settings)["status"] == "PASS"
    for split, partition in datasets.items():
        assert (partition["target"] == (partition["pixel"] >= 0)).all()
        assert all(sorted(s["requested_rank"] for s in m["samples"]) == config["root_percentiles"]
                   for m in manifest[split]["maps"])
