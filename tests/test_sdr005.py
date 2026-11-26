import importlib.util

import numpy as np
import pytest
import torch

from scripts.train_sdr005 import image_stats
from sdr_mamba.crowd_data import mass_map, transform_points
from sdr_mamba.crowd_ssm import CrowdModel, axes, chunked_ssd, communicate, restore
from sdr_mamba.model_v3 import dense_oracle
from sdr_mamba.ssm import official_ssd, oracle


def operands(length, device, dtype):
    torch.manual_seed(705071)

    def sample(*shape):
        return (torch.randn(*shape, device=device, dtype=dtype) * .18).requires_grad_()

    return (sample(2, length, 4, 4), sample(2, length, 4).sigmoid() * .3,
            -sample(4).exp(), sample(2, length, 2, 3), sample(2, length, 2, 3),
            sample(4), sample(2, 4, 4, 3))


def compare(reference, actual, args, atol, rtol):
    first_y, first_final = reference
    second_y, second_final = actual
    probes = (torch.randn_like(first_y), torch.randn_like(first_final))
    errs = []
    for a, b in zip((second_y, second_final), (first_y, first_final)):
        torch.testing.assert_close(a, b.to(a.dtype), atol=atol, rtol=rtol)
        errs.append(float((a.detach() - b.detach().to(a.dtype)).abs().max()))
    grad1 = torch.autograd.grad(reference, args, probes, retain_graph=True)
    grad2 = torch.autograd.grad(actual, args, probes, retain_graph=True)
    for a, b in zip(grad1, grad2):
        torch.testing.assert_close(a, b.to(a.dtype), atol=atol, rtol=rtol)
        errs.append(float((a.detach() - b.detach().to(a.dtype)).abs().max()))
    return errs


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float64, 1e-9, 1e-8), (torch.float32, 4e-6, 4e-5)])
@pytest.mark.parametrize("length", [9, 37])
def test_chunked_y_final_grad_and_second_round(length, dtype, atol, rtol):
    torch.set_num_threads(1)
    args = operands(length, "cpu", dtype)
    expect_first = oracle(*args)
    actual_first = chunked_ssd(*args)[:2]
    compare(expect_first, actual_first, args, atol, rtol)
    if length == 9:
        compare(dense_oracle(*args)[:2], actual_first, args, atol, rtol)
    expect2 = oracle(*args[:-1], expect_first[1])
    actual2 = chunked_ssd(*args[:-1], actual_first[1])[:2]
    compare(expect2, actual2, args, atol, rtol)
    passed = torch.autograd.grad(actual2[0].sum(), actual_first[1])[0]
    assert passed.isfinite().all() and passed.norm() > 0


def test_axes_non_square_physical_endpoint_is_the_other_direction_terminal():
    features = torch.arange(30).view(1, 5, 6, 1)
    scans = axes(features)
    for z in restore(scans):
        torch.testing.assert_close(z, features)
    for index, opposite in enumerate((1, 0, 3, 2)):
        torch.testing.assert_close(scans[index][:, :, 0], scans[opposite][:, :, -1])
    assert scans[1][0, 2, :, 0].tolist() == [17, 16, 15, 14, 13, 12]
    assert scans[3][0, 2, :, 0].tolist() == [26, 20, 14, 8, 2]


def test_communication_never_mixes_images_and_zero_gain_resets():
    banks = tuple(torch.arange(12).reshape(2, 6, 1).float() + 20 * i for i in range(4))
    a = communicate(banks, "C3", .5)
    b = communicate(banks, "C4", .5)
    for i, op in enumerate((1, 0, 3, 2)):
        torch.testing.assert_close(a[i], banks[op] * .5)
        torch.testing.assert_close(b[i][:, :-1], banks[op][:, 1:] * .5)
        torch.testing.assert_close(b[i][:, -1], banks[op][:, 0] * .5)
    assert all(t is None for t in communicate(banks, "C3", 0))


def test_real_model_state_gradient_and_same_first_round():
    torch.manual_seed(9)
    model = CrowdModel("C3", width=8, state=3, heads=2, chunk=3)
    x = torch.randn(2, 3, 37, 49)
    frames = model(x, 2)
    for mode in ("C1", "C2", "C4", "C5"):
        torch.testing.assert_close(model(x, 1, mode=mode)[0].density, frames[0].density)
    detached_core = model.step(frames[1].features_in.detach(), frames[0].bank, mode="C3")
    grads = torch.autograd.grad(detached_core.density.square().mean(), frames[0].bank)
    assert all(g.isfinite().all() and g.norm() > 0 for g in grads)
    torch.testing.assert_close(model(x[:1], 2)[-1].density, frames[-1].density[:1])
    model.gain = 0
    torch.testing.assert_close(model(x, 2)[-1].density, model(x, 2, mode="C1")[-1].density)


def test_padding_values_do_not_reach_encoder_or_ssm():
    torch.manual_seed(9)
    model = CrowdModel("C3", width=8, state=3, heads=2, chunk=3).eval()
    x = torch.randn(2, 3, 56, 53)
    actual_sizes = [(33, 53), (38, 41)]
    clean = model.forward_valid(x, actual_sizes, 2)
    x[0, :, 33:] = 100
    x[1, :, 38:] = 100
    x[1, :, :, 41:] = 100
    poisoned = model.forward_valid(x, actual_sizes, 2)
    for a, b in zip(clean, poisoned):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_cropped_flipped_resized_count_mass_and_game():
    pts = np.array([[0, 0], [3.8, 4.1], [9.7, 19.2], [10, 5], [4, 20]])
    x = transform_points(pts, 0, 0, 20, 10, True, (13, 23))
    assert x.shape == (3, 2)
    m = mass_map(x, 13, 23, 8)
    assert m.shape == (2, 3) and m.sum() == 3 and (m >= 0).all()
    assert mass_map(x[:0], 13, 23, 8).sum() == 0
    assert image_stats(m, m)["game3"] == 0
    assert image_stats(torch.zeros_like(m), m)["game1"] == 3


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 8e-4, 5e-3), (torch.bfloat16, .025, .09)])
def test_official_two_round_cuda_adapter(dtype, atol, rtol):
    if not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None:
        pytest.skip("NOT_RUN: official Mamba SSD dependencies unavailable in this CUDA environment")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("NOT_RUN: BF16 unavailable")
    args = operands(37, "cuda", torch.float32)
    special = tuple(t.to(dtype) if i in (0, 3, 4) else t for i, t in enumerate(args))
    first = official_ssd(*special)
    ref = oracle(*args)
    print(dtype, "official first errors", compare(ref, first, args, atol, rtol))
    actual2 = official_ssd(*special[:-1], first[-1])
    ref2 = oracle(*args[:-1], ref[-1])
    print(dtype, "official two-round errors", compare(ref2, actual2, args, atol, rtol))
