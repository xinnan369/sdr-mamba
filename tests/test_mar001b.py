"""Numerical contracts for channel-mean isolation."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.perturb_mar001 import directional_jvp
from scripts.run_mar001 import recurrent_step
from sdr_mamba.crowd_audit import AuditModel
from sdr_mamba.mar_readout import anchor_channel_mean, channel_parts, matched_centered_rms, readout_parts
from sdr_mamba.mar_readout import relative_rms, shapley_counts


@pytest.mark.parametrize("mode", ["F1", "F3"])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 3e-4)])
def test_recurrence_equivariance_neutral_jvp_and_full_anchor(mode, dtype, tolerance):
    torch.set_num_threads(1)
    torch.manual_seed(611)
    model = AuditModel(mode, 4, 8, 4, 2, 4, "chunked", -4.3, 5).to(dtype).eval().requires_grad_(False)
    norm = model.blocks[0].norm
    x = torch.randn(2, 5, 9, 8, dtype=dtype)
    field = torch.randn(2, 5, 9, 1, dtype=dtype)
    expected = recurrent_step(model, x)
    y = x
    errors = {"norm": [], "step": [], "jvp": [], "anchor_centered": [], "anchor_head": []}
    for amplitude in (.1, 1.):
        bias = amplitude*field
        errors["norm"].append(relative_rms(norm(x+bias)-norm(x), norm(x)))
        errors["step"].append(relative_rms(recurrent_step(model, x+bias)-expected-bias, bias))
        tangent, _ = directional_jvp(lambda value: recurrent_step(model, value), x, bias.expand_as(x).contiguous())
        errors["jvp"].append(relative_rms(tangent-bias, bias))
    mean = channel_parts(x)[0]
    for _ in range(4):
        x = recurrent_step(model, x)
        y = anchor_channel_mean(recurrent_step(model, y), mean)
        z = channel_parts(x)[1]
        errors["anchor_centered"].append(relative_rms(channel_parts(y)[1]-z, z))
        observed = model.head(y)
        predicted = model.head(z + mean)
        errors["anchor_head"].append(relative_rms(observed-predicted, predicted))
    solo = recurrent_step(model, x[:1])
    torch.testing.assert_close(solo, recurrent_step(model, x)[:1], atol=tolerance, rtol=tolerance)
    maxima = {name: max(observed) for name, observed in errors.items()}
    print(mode, dtype, "relative RMS maxima", maxima)
    assert all(error < tolerance for error in maxima.values())


def test_readout_shapley_and_zero_sum_weight():
    x = torch.tensor([[[[1., 2., -1.]]]], dtype=torch.float64)
    y = torch.tensor([[[[2., 0., 5.]]]], dtype=torch.float64)
    model = nn.Linear(3, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.5, -.5, .25]]))
    m, z = channel_parts(x)
    my, zy = channel_parts(y)
    mean, centered = readout_parts(model, y)
    torch.testing.assert_close(mean+centered+model.bias, model(y).squeeze(-1))
    a, b, c, d = (F.softplus(model(value)).sum().item() for value in (x, zy+m, z+my, y))
    mc, zc = shapley_counts(a, b, c, d)
    assert mc + zc == pytest.approx(d-a)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., -.5, -.5]]))
    torch.testing.assert_close(F.softplus(model(y+4)), F.softplus(model(y)), atol=1e-12, rtol=1e-12)


def test_centered_rms_control_matches_norm_and_handles_degenerate_cells():
    x = torch.tensor([[[[3., 0., -3.]]]])
    y = torch.tensor([[[[1., 3., -4.]]]])
    actual = matched_centered_rms(x, y)
    torch.testing.assert_close(actual.square().mean(-1), y.square().mean(-1))
    torch.testing.assert_close(actual.mean(-1), torch.zeros(1, 1, 1))
    assert matched_centered_rms(torch.zeros_like(x), y) is None
    assert not matched_centered_rms(torch.zeros_like(x), torch.zeros_like(y)).any()
