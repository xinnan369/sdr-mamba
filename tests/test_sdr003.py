from itertools import product
import json
from pathlib import Path

import pytest
import torch

from scripts.audit_dataset import audit
from scripts.eval_sdr003 import auc
from sdr_mamba.data_v3 import build
from sdr_mamba.model_v3 import EdgeGate, LoopV3, align_message, dense_oracle, hard_gate, transport
from sdr_mamba.ssm import oracle
from sdr_mamba.state_transport import initial_states


@pytest.mark.parametrize("length", [5, 16])
def test_dense_scan_matches_explicit_forward_and_gradients(length):
    torch.manual_seed(52)
    torch.set_num_threads(1)

    def value(*shape):
        return torch.randn(shape, dtype=torch.float64).requires_grad_()

    args = (value(3, length, 4, 3), value(3, length, 4).sigmoid(),
            -value(4).exp(), value(3, length, 2, 3), value(3, length, 2, 3),
            value(4), value(3, 4, 3, 3))
    actual = dense_oracle(*args)[:2]
    expected = oracle(*args)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)
        incoming = torch.randn_like(got)
        first = torch.autograd.grad(got, args, incoming, retain_graph=True, allow_unused=True)
        second = torch.autograd.grad(want, args, incoming, retain_graph=True, allow_unused=True)
        for g, e in zip(first, second):
            assert (g is None) == (e is None)
            if g is not None:
                torch.testing.assert_close(g, e, rtol=1e-9, atol=1e-9)


def test_edge_masks_non_square_batch_reflect_and_rotate():
    x = torch.zeros(2, 8, 12, 2)
    x[0, 1, 3, 0] = x[0, 2, 4, 0] = 1
    assert not hard_gate(x, 4).any()
    x[0, 1, 4, 0] = 1
    gate = hard_gate(x, 4)
    assert gate.sum() == 2
    assert gate[0, 0, 1, 0] == gate[0, 0, 0, 1] == 1
    torch.testing.assert_close(hard_gate(x.flip(2), 4), gate.flip(2)[..., [1, 0, 2, 3]])
    torch.testing.assert_close(hard_gate(x.transpose(1, 2), 4), gate.transpose(1, 2)[..., [2, 3, 0, 1]])
    torch.testing.assert_close(hard_gate(x.rot90(1, (1, 2)), 4), gate.rot90(1, (1, 2))[..., [2, 3, 1, 0]])


def test_blocked_pulse_and_no_change_with_unity_gate():
    bank = torch.zeros(2, 2, 3, 4, 2, 3, 2)
    bank[0, 0, 0, 0] = 3
    p = torch.ones(2, 2, 3, 4)
    expected = initial_states(bank, "sst", .5, .6)
    torch.testing.assert_close(transport(bank, .5, .6, p), expected)
    p[0, 0, 1, 0] = 0
    blocked = transport(bank, .5, .6, p)
    assert blocked[0, 0, 1, 0].count_nonzero() == 0
    assert blocked[0, 0, 0, 2].count_nonzero() > 0
    assert not blocked[1].any()
    torch.testing.assert_close(transport(bank, .5, .6, torch.zeros_like(p)), transport(bank, .5, .6, variant="zero"))


def test_learnable_gate_and_state_receive_gradients():
    torch.manual_seed(6)
    model = LoopV3("learned")
    assert isinstance(model.edge_model, EdgeGate)
    data = torch.rand(2, 8, 12, 2)
    frames = model(data, 3)
    p, s = torch.autograd.grad(frames[-1].logits.square().mean(), (frames[1].edge, frames[0].state))
    assert p.isfinite().all() and s.isfinite().all() and p.norm() > 0 and s.norm() > 0


@pytest.mark.parametrize("mode", ["sst", "learned", "strong_feature"])
def test_actual_seed_swap_cannot_cross_more_than_one_window_per_round(mode):
    torch.manual_seed(91)
    x = torch.ones(2, 16, 20, 2)
    x[..., 1] = 0
    x[0, 1, 2, 1] = x[1, 3, 3, 1] = 1
    model = LoopV3(mode)
    with torch.no_grad():
        trace = model(x, 3)
    for turn, frame in enumerate(trace, 1):
        delta = (frame.logits[0] - frame.logits[1]).abs()
        assert delta.max() > 0
        for y, x in product(range(4), range(5)):
            if y + x >= turn:
                assert delta[y * 4:(y + 1) * 4, x * 4:(x + 1) * 4].max() < 2e-6


def test_fresh_and_ood_splits_are_deterministic_and_independent():
    config = json.loads((Path(__file__).parents[1] / "configs/sdr003.json").read_text())
    for entry in config["data"].values():
        entry["maps"] = 3
    datasets, manifest, base = build(config)
    assert build(config)[1] == manifest
    result = audit(datasets, manifest, base)
    assert result["status"] == "PASS"
    assert all(split["baselines"]["all_walkable"]["iou"] == 1 / 3 for split in result["splits"].values())
    assert all(m["closed_walking_window_edges"] > 0 for m in manifest["ood"]["maps"])
    for data in datasets.values():
        assert torch.equal(data["grid"][data["image"][..., 1].bool()], torch.zeros_like(data["grid"][:, 0, 0]))
        assert torch.all(data["window"][data["target"]] >= data["grid"][data["target"]])


def test_aligned_strong_packet_enters_physical_edges_and_auc_handles_ties():
    payload = torch.tensor([1., 2., 3., 4.]).reshape(1, 1, 1, 1, 4, 1).expand(1, 1, 1, 4, 4, 1)
    aligned = align_message(payload, 4).reshape(4, 16)
    assert (aligned[:, 1::4] == 0).all() and (aligned[:, 2::4] == 0).all() and (aligned[:, 3::4] == 0).all()
    torch.testing.assert_close(aligned[:, ::4], torch.tensor([
        [1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 3., 4.], [4., 3., 2., 1.],
    ]))
    assert auc(torch.tensor([.1, .4, .4, .9]), torch.tensor([False, False, True, True])) == .875
