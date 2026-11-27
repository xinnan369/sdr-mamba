import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.prepare_lm001 import source
from scripts.train_sdr003 import read_json
from sdr_mamba.lm001 import WindowSSDMixer, build_model, recipe, reference_scan


def sequential(x, dt, a, b, c, skip, valid):
    state = x.new_zeros((*x.shape[:1], *x.shape[2:], b.shape[-1]))
    projections = []
    for t in range(x.shape[1]):
        delta = dt[:, t] * valid[:, t, None]
        inp = x[:, t]
        state = (delta*a).exp()[..., None, None]*state + (
            delta[..., None, None]*inp[..., None]*b[:, t, None, None])
        projected = (state*c[:, t, None, None]).sum(-1) + skip[:, None]*inp
        projections.append(projected*valid[:, t, None, None])
    return torch.stack(projections, 1)


@pytest.mark.parametrize("length", [1, 5, 8])
def test_reference_scan_output_and_gradients_match_independent_recurrence(length):
    torch.manual_seed(5)
    shapes = [(2,length,3,4),(2,length,3),(3,), (2,length,5),(2,length,5),(3,)]
    x, dt, a, b, c, d = (torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in shapes)
    dt, a = dt.sigmoid(), -a.exp()
    valid = torch.stack((torch.arange(length)%3 != 1, torch.zeros(length, dtype=torch.bool)))
    operands = (x,dt,a,b,c,d)
    got = reference_scan(*operands, valid)
    target = sequential(*operands, valid)
    torch.testing.assert_close(got, target, atol=1e-12, rtol=1e-12)
    for lhs, rhs in zip(torch.autograd.grad(got.square().sum(), operands), torch.autograd.grad(target.square().sum(), operands)):
        torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_reference_mixer_window_isolation_reach_and_padded_tokens():
    torch.manual_seed(9)
    mixer = WindowSSDMixer(8,4,8,4,3).double()
    x = torch.randn(2,16,8,dtype=torch.float64,requires_grad=True)

    def step(y, pad=None):
        return mixer(y,y,y,key_padding_mask=pad)[0]

    first = x + step(x)
    second = first + step(first)
    reachable_one = torch.autograd.grad(first[0,0].sum(), x, retain_graph=True)[0].abs().sum(-1) > 0
    reachable_two = torch.autograd.grad(second[0,0].sum(), x)[0].abs().sum(-1) > 0
    assert reachable_one[0].sum().item() == 7
    assert reachable_two[0].sum().item() == 16
    assert not reachable_two[1].any()
    mask = torch.zeros(2,16,dtype=torch.bool)
    mask[:,[1,2,6,10,15]] = True
    poisoned = x.detach().clone()
    poisoned[mask] = 1e6
    np_masked = step(x,mask)
    torch.testing.assert_close(np_masked, step(poisoned,mask), atol=0, rtol=0)
    assert not np_masked[mask].any()


def test_autocast_reference_keeps_finite_output_and_gradients():
    model = WindowSSDMixer(8,4,8,4,3)
    x = torch.randn(2,16,8,requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, _ = model(x,x,x)
        assert out.dtype == torch.bfloat16
        error = out.float().square().mean()
    error.backward()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()


@pytest.fixture(scope="module")
def upstream():
    location, cache = os.environ.get("LM001_REFERENCE"), os.environ.get("LM001_ASSETS")
    if not location or not cache:
        pytest.skip("Set LM001_REFERENCE and LM001_ASSETS to exercise the offline platform")
    config = read_json(Path("configs/lm001.json"))
    with source(Path(location), config["reference"]["lc007g"]):
        file = Path(cache)/"torch/hub/checkpoints/resnet18-f37072fd.pth"
        from loopcrowd.experiment import file_sha
        assert file_sha(file) == config["reference"]["weight_sha256"]
        weights = torch.load(file, map_location="cpu", weights_only=True)
        yield config, weights


@pytest.mark.parametrize("family", ["transformer", "mamba"])
def test_exact_shared_gradient_routing_and_stage_supervision(upstream, family):
    cfg, weights = upstream
    from loopcrowd.targets import point_targets, point_loss
    from loopcrowd.training import supervised_point_loss

    x = torch.rand(1,3,80,96)
    target = {k:v[None] for k,v in point_targets([[18.,22.],[20.,19.]],96,80).items()}
    for sharing in ("shared", "unshared"):
        arm = f"{sharing}_2_{family}"
        model = build_model(cfg, arm, 42, weights).eval()
        calls = []
        handles = [block.register_forward_hook(lambda block, inp, out: calls.append(id(block))) for block in model.blocks]
        stage = model.forward_stages(x)
        for handle in handles:
            handle.remove()
        assert len(calls) == 2 and (calls[0] == calls[1]) == (sharing == "shared")
        first = point_loss(stage["early"], target)
        final = point_loss(stage["final"], target)
        coupled = supervised_point_loss(model, x, target, recipe(cfg, arm, 42))
        torch.testing.assert_close(coupled, final+.25*first)
        first_params = tuple(model.blocks[0].attn.parameters())
        last_params = tuple(model.blocks[-1].attn.parameters())
        for output, params in ((first, first_params), (final, first_params), (final, last_params)):
            assert any(g.count_nonzero() for g in torch.autograd.grad(output, params, retain_graph=True))
        if sharing == "unshared":
            assert not {p.data_ptr() for p in first_params} & {p.data_ptr() for p in last_params}
            assert all(g is None for g in torch.autograd.grad(first, last_params, allow_unused=True))


def test_reference_ema_handles_shared_objects_and_restores_buffers(upstream):
    cfg, weights = upstream
    from loopcrowd.ema import ParameterEMA

    model = build_model(cfg, "shared_2_mamba", 42, weights)
    ema = ParameterEMA(model, .99)
    parameter = model.point_logits.bias
    initial = parameter.detach().clone()
    assert len(ema.parameters) == len({id(p) for p in model.parameters()})
    with torch.no_grad():
        parameter.add_(1)
    ema.update(updated=False)
    assert ema.step == 0
    ema.update(updated=True)
    assert ema.step == 1
    bn = next(module for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    buffers = bn.running_mean.clone()
    with ema.use_shadow():
        torch.testing.assert_close(parameter, initial+.01)
        assert torch.equal(buffers, bn.running_mean)
    torch.testing.assert_close(parameter, initial+1)


def test_fixed_count_threshold_and_ap_pool_use_distinct_candidate_views(upstream):
    from loopcrowd.calibration import count_match_metrics, rank_metrics
    from loopcrowd.inference import TilePredictions
    from loopcrowd.metrics import pooled_point_nap

    points = [[10., 10.], [20., 20.]]
    candidates = np.array([[10.,10.,.84], [20.,20.,.30], [30.,30.,.42]])
    predictions = TilePredictions(candidates, np.array([0,0,1]), np.array([True,True,False]), [])
    count, = count_match_metrics(points, predictions.count_candidates(), (.30,))
    assert (count["predicted"], count["local_tp"], count["local_fp"], count["local_fn"]) == (1,1,0,1)
    scores, matches = rank_metrics(points, predictions)
    assert len(scores) == 3 and np.all(matches.sum(axis=1) == 2)
    nap = pooled_point_nap([scores], [matches], len(points))
    assert len(nap["ap_by_delta"]) == 10 and .5 < nap["nap"] < 1
