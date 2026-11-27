"""Exact channel-mean and centered feature readout diagnostics."""

import torch


def channel_parts(value):
    mean = value.mean(dim=-1, keepdim=True)
    return mean, value - mean


def anchor_channel_mean(value, target):
    return channel_parts(value)[1] + target


def readout_parts(head, value):
    mean, centered = channel_parts(value)
    weight = head.weight.flatten()
    mean_component = mean.squeeze(-1) * weight.sum()
    centered_component = centered @ weight
    return mean_component, centered_component


def matched_centered_rms(current, reference):
    source = current.square().mean(-1, keepdim=True).sqrt()
    target = reference.square().mean(-1, keepdim=True).sqrt()
    if ((source == 0) & (target > 0)).any():
        return None
    scale = target / torch.where(source > 0, source, 1)
    return current * scale


def shapley_counts(a, b, c, d):
    return ((c - a) + (d - b)) / 2, ((b - a) + (d - c)) / 2


def relative_rms(error, ref):
    numerator = error.double().square().mean().sqrt().item()
    denominator = ref.double().square().mean().sqrt().item()
    return numerator / max(denominator, 1e-12)
