from importlib import metadata

import pytest
import torch

from sdr_mamba.ssm import official_ssd, oracle


def inputs(device, dtype, length, heads):
    torch.manual_seed(123)

    def gaussian(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) * 0.2

    groups = 2
    tensors = (
        gaussian(2, length, heads, 4), gaussian(2, length, heads).exp() / 10,
        -gaussian(heads).exp(), gaussian(2, length, groups, 3),
        gaussian(2, length, groups, 3), gaussian(heads),
        gaussian(2, heads, 4, 3),
    )
    return tuple(t.requires_grad_() for t in tensors)


def test_fp64_gradcheck():
    assert torch.autograd.gradcheck(oracle, inputs("cpu", torch.float64, 3, 4),
                                    eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("skip_per_channel", [False, True])
def test_group_layout_and_single_step(skip_per_channel):
    x, dt, a, b, c, d, initial = inputs("cpu", torch.float64, 1, 4)
    if skip_per_channel:
        d = d[:, None] + torch.arange(4, dtype=d.dtype)[None, :]
    y, final = oracle(x, dt, a, b, c, d, initial)
    for head in range(4):
        group = head // 2
        driven = x[:, 0, head, :, None] * b[:, 0, group, None, :]
        expected = torch.exp(dt[:, 0, head] * a[head])[:, None, None] * initial[:, head]
        expected = expected + dt[:, 0, head, None, None] * driven
        torch.testing.assert_close(final[:, head], expected)
        readout = (expected * c[:, 0, group, None, :]).sum(-1)
        torch.testing.assert_close(y[:, 0, head], readout + d[head] * x[:, 0, head])


def test_official_cuda_output_final_and_backward_parity():
    if not torch.cuda.is_available():
        pytest.skip("NOT_RUN: CUDA unavailable to this interpreter")
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except (ImportError, OSError) as error:
        pytest.skip(f"NOT_RUN: official SSD import failed: {error}")
    print(f"mamba_ssm={metadata.version('mamba-ssm')} operator={mamba_chunk_scan_combined.__module__}")
    tensors = inputs("cuda", torch.float32, 33, 4)
    expected = oracle(*tensors)
    actual = official_ssd(*tensors)
    probes = tuple(torch.randn_like(t) for t in expected)
    atol, rtol = 5e-4, 5e-3
    for name, got, ref in zip(("y", "final"), actual, expected):
        print(f"{name}: max_abs={(got - ref).abs().max().item():.8g}, atol={atol}, rtol={rtol}")
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    for component in (0, 1):
        actual_grad = torch.autograd.grad(actual[component], tensors, probes[component],
                                          retain_graph=True, allow_unused=True)
        expected_grad = torch.autograd.grad(expected[component], tensors, probes[component],
                                            retain_graph=True, allow_unused=True)
        for name, got, ref, original in zip("xtABCDI", actual_grad, expected_grad, tensors):
            got = torch.zeros_like(original) if got is None else got
            ref = torch.zeros_like(original) if ref is None else ref
            print(f"branch={component} grad={name} max_abs={(got - ref).abs().max().item():.8g}")
            torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
