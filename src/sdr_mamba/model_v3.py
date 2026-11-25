"""SDR-003 anchored loop and aligned, direction-conditioned boundary messages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model_v2 import WindowSSM
from .scan import inverse, scan, tile, untile
from .state_transport import upstream

OFFSETS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def dense_oracle(x, dt, a, b, c, d, initial=None, diagnose=False):
    """Evaluate the explicit recurrence by FP32/64 causal convolution on a short window."""
    n, length, heads, width = x.shape
    state_size = b.shape[-1]
    groups = b.shape[-2]
    decay_log = (dt * a).transpose(1, 2).cumsum(-1)
    delta = decay_log.unsqueeze(-1) - decay_log.unsqueeze(-2)
    causal = torch.ones((length, length), device=x.device, dtype=torch.bool).tril()
    kernel = delta.masked_fill(~causal, -torch.inf).exp()
    b = b.repeat_interleave(heads // groups, dim=2)
    c = c.repeat_interleave(heads // groups, dim=2)
    drive = ((dt.unsqueeze(-1) * x).unsqueeze(-1) * b.unsqueeze(-2)).transpose(1, 2)
    state = (kernel @ drive.flatten(-2)).reshape(n, heads, length, width, state_size).transpose(1, 2)
    from_initial = None
    if initial is not None:
        from_initial = decay_log.transpose(1, 2).exp()[..., None, None] * initial.unsqueeze(1)
        state = state + from_initial
    final = state[:, -1]
    earlier = (state[:, :-1] * c[:, :-1].unsqueeze(-2)).sum(-1)
    tail = (final * c[:, -1].unsqueeze(-2)).sum(-1)
    output = torch.cat((earlier, tail.unsqueeze(1)), dim=1) + x * (d.unsqueeze(-1) if d.ndim == 1 else d)
    if not diagnose:
        return output, final, None
    contribution = None if from_initial is None else (from_initial * c.unsqueeze(-2)).sum(-1)
    return output, final, (decay_log[..., -1].exp(), contribution, output)


def boundary_pairs(windows):
    """Aligned pairs in image coordinates, [batch, grid-y, grid-x, direction, k, channels]."""
    enter = (windows[..., :, 0, :], windows[..., :, -1, :],
             windows[..., 0, :, :], windows[..., -1, :, :])
    leave = (windows[..., :, -1, :], windows[..., :, 0, :],
             windows[..., -1, :, :], windows[..., 0, :, :])
    return (
        torch.stack([upstream(edge, direction) for direction, edge in enumerate(leave)], dim=3),
        torch.stack(enter, dim=3),
    )


def valid_edges(windows):
    plane = torch.ones_like(windows[..., 0, 0, 0])
    return torch.stack([upstream(plane, direction) for direction in range(4)], dim=3)


def hard_gate(image, k):
    source, target = boundary_pairs(tile(image[..., :1], k))
    return ((source > 0) & (target > 0)).any(dim=(-1, -2)).to(image.dtype)


class EdgeGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.direction = nn.Embedding(4, 4)
        self.project = nn.Sequential(nn.Linear(6 * width + 4, 8), nn.SiLU(), nn.Linear(8, 1))

    def forward(self, original, features, k, local=False):
        windows = torch.cat((tile(original, k), tile(features, k)), dim=-1)
        source, target = boundary_pairs(windows)
        if local:
            means = windows.mean((3, 4)).unsqueeze(3).unsqueeze(4)
            # Both summaries are within the current window; no boundary positions are observed.
            center = windows[..., 1:-1, 1:-1, :].mean((3, 4)).unsqueeze(3).unsqueeze(4)
            source, target = (value.expand_as(target) for value in (means, center))
        embedding = self.direction.weight.view(1, 1, 1, 4, 1, 4).expand(*target.shape[:-1], 4)
        design = torch.cat((source, target, (source - target).abs(), embedding), dim=-1)
        probability = self.project(design).squeeze(-1).amax(-1).sigmoid()
        return probability * valid_edges(windows)


def transport(previous, gate, mixing, edge=None, variant="true"):
    if previous is None or gate == 0:
        return None
    from .state_transport import shuffle_sources

    directions = previous.unbind(3)
    perpendicular = ((directions[2] + directions[3]) * .5, (directions[0] + directions[1]) * .5)
    result = []
    for index, own in enumerate(directions):
        source = upstream(shuffle_sources(own) if variant == "shuffled" else own, index)
        if variant == "zero":
            source = torch.zeros_like(source)
        if edge is not None:
            weight = edge[..., index].reshape(*source.shape[:3], *((1,) * (source.ndim - 3)))
            source = source * weight
        result.append(gate * (mixing * source + (1 - mixing) * perpendicular[index // 2]))
    return torch.stack(result, dim=3)


class StrongFeature(nn.Module):
    """One-hop pixel-aligned messages via a shared nonlinear edge MLP."""

    def __init__(self, width):
        super().__init__()
        self.direction = nn.Embedding(4, 4)
        self.project = nn.Sequential(nn.Linear(2 * width + 4, 14), nn.SiLU(), nn.Linear(14, width))

    def forward(self, features, k):
        windows = tile(features, k)
        source, target = boundary_pairs(windows)
        embedding = self.direction.weight.view(1, 1, 1, 4, 1, 4).expand(*target.shape[:-1], 4)
        message = .5 * self.project(torch.cat((source, target, embedding), dim=-1))
        return message * valid_edges(windows)[..., None, None]


def align_message(packet, k):
    # Forward row and column scans encounter their incoming boundary at positions 0,k,2k,...
    reversed_packet = packet.flip(-2)
    values = torch.stack((packet[..., 0, :, :], reversed_packet[..., 1, :, :],
                          packet[..., 2, :, :], reversed_packet[..., 3, :, :]), dim=3)
    values = F.pad(values.unsqueeze(-2), (0, 0, 0, k - 1))
    return values.flatten(-3, -2).flatten(0, 3)


@dataclass
class Frame:
    logits: torch.Tensor
    f_in: torch.Tensor
    f_out: torch.Tensor
    raw: torch.Tensor
    state: torch.Tensor
    initial: torch.Tensor | None
    sent: torch.Tensor | None
    edge: torch.Tensor | None
    diagnostic: tuple | None


class LoopV3(WindowSSM):
    def __init__(self, mode="sst", anchored=False):
        super().__init__(k=4, d_model=16, d_state=4, expand=1, headdim=8, groups=1, gate=.5, mixing=.6)
        self.mode = mode
        self.anchored = anchored
        self.edge_model = EdgeGate(16) if mode in ("learned", "gated_feature", "local_gate") else None
        self.strong = StrongFeature(16) if mode == "strong_feature" else None

    def step_v3(self, original, features, initial, packet=None, aligned=None, edge=None, diagnose=False):
        windows = tile(features, self.k)
        b, hg, wg, *_ = windows.shape
        x, b_coef, c_coef, dt = scan(self.project(self.norm(windows))).flatten(0, 3).split(
            (self.width, self.d_state, self.d_state, self.heads), dim=-1,
        )
        if aligned is not None:
            x = x + align_message(aligned, self.k)
        x = x.unflatten(-1, (self.heads, self.headdim))
        if packet is not None:
            x = torch.cat((x[:, :self.d_state] + packet.flatten(0, 3).movedim(-1, 1),
                           x[:, self.d_state:]), dim=1)
        dt = F.softplus(dt).clamp_min(torch.finfo(dt.dtype).tiny)
        output, raw, diag = dense_oracle(
            x, dt, -self.log_a.exp(), b_coef.unsqueeze(-2), c_coef.unsqueeze(-2), self.skip,
            None if initial is None else initial.flatten(0, 3), diagnose,
        )
        sent = None
        if self.mode in ("feature", "gated_feature"):
            sent = output[:, -self.d_state:].movedim(1, -1).unflatten(0, (b, hg, wg, 4))
        directional = output.flatten(-2).unflatten(0, (b, hg, wg, 4))
        residual = self.output(untile(inverse(directional, self.k).mean(3)))
        updated = features + (.1 * (original - features) + .5 * residual if self.anchored else residual)
        return Frame(self.readout(updated).squeeze(-1), features, updated, raw,
                     raw.unflatten(0, (b, hg, wg, 4)), initial, sent, edge, diag)

    def forward(self, image, rounds, diagnose=False):
        original = self.encoder(image)
        features = original
        hard = hard_gate(image, self.k) if self.mode == "hard" else None
        frames = []
        for index in range(rounds):
            edge = hard
            initial = packet = aligned = None
            if index:
                if self.edge_model is not None:
                    edge = self.edge_model(original, features, self.k, local=self.mode == "local_gate")
                prev = frames[-1]
                if prev.sent is not None:
                    packet = transport(prev.sent, self.gate, self.mixing, edge)
                elif self.strong is not None:
                    aligned = self.strong(features, self.k)
                else:
                    initial = transport(prev.state, self.gate, self.mixing, edge)
            current = self.step_v3(original, features, initial, packet, aligned, edge, diagnose)
            features = current.f_out
            frames.append(current)
        return frames
