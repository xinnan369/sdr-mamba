"""Explicit Mamba-2-style recurrence and optional official SSD adapter."""

import torch


def _check_shapes(x, dt, a, b, c, d, initial):
    batch, length, heads, width = x.shape
    if b.ndim != 4 or b.shape != c.shape or b.shape[:2] != (batch, length):
        raise ValueError("B and C must share shape [batch, length, groups, state]")
    groups, state_size = b.shape[-2:]
    if groups < 1 or heads % groups or length < 1:
        raise ValueError("Sequence must be nonempty and groups must divide heads")
    if dt.shape != (batch, length, heads) or a.shape != (heads,):
        raise ValueError("dt must have shape [batch, length, heads] and A [heads]")
    if d.shape not in ((heads,), (heads, width)):
        raise ValueError("D must have shape [heads] or [heads, width]")
    if initial is not None and initial.shape != (batch, heads, width, state_size):
        raise ValueError("Initial state shape must be [batch, heads, width, state]")
    return groups, state_size


def oracle(x, dt, a, b, c, d, initial=None):
    """FP32/FP64 recurrence using pretransformed positive dt and negative A.

    Each B/C group is broadcast to a contiguous block of heads. The output and
    final state use the same dtype as their inputs.
    """
    groups, state_size = _check_shapes(x, dt, a, b, c, d, initial)
    batch, length, heads, width = x.shape
    state = initial if initial is not None else x.new_zeros(batch, heads, width, state_size)
    decay = torch.exp(dt * a)
    driven = dt.unsqueeze(-1) * x
    b = b.repeat_interleave(heads // groups, dim=2)
    c = c.repeat_interleave(heads // groups, dim=2)
    skip = x * (d.unsqueeze(-1) if d.ndim == 1 else d)
    outputs = []
    for i in range(length):
        state = decay[:, i, :, None, None] * state + (
            driven[:, i].unsqueeze(-1) * b[:, i].unsqueeze(-2)
        )
        outputs.append((state * c[:, i].unsqueeze(-2)).sum(dim=-1))
    return torch.stack(outputs, dim=1) + skip, state


def official_ssd(x, dt, a, b, c, d, initial=None):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    _check_shapes(x, dt, a, b, c, d, initial)
    return mamba_chunk_scan_combined(
        x.contiguous(), dt.contiguous(), a.contiguous(), b.contiguous(), c.contiguous(),
        chunk_size=16, D=d.contiguous(),
        initial_states=None if initial is None else initial.contiguous(),
        dt_softplus=False, return_final_states=True,
    )
