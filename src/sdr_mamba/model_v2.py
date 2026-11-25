"""Window SSM with matched state and directional feature messages for SDR-002."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .scan import inverse, scan, tile, untile
from .ssm import oracle
from .state_transport import MODES, initial_states, shuffle_sources, upstream


@dataclass
class Frame:
    logits: torch.Tensor
    features: torch.Tensor
    carried: torch.Tensor
    raw_final: torch.Tensor
    state: torch.Tensor
    initial: torch.Tensor | None
    sent_features: torch.Tensor | None
    received_features: torch.Tensor | None
    normalized: torch.Tensor
    tokens: torch.Tensor


class WindowSSM(nn.Module):
    def __init__(self, k, d_model, d_state, expand, headdim, groups, gate, mixing):
        super().__init__()
        width = d_model * expand
        if width % (headdim * groups) or not 1 <= d_state <= k * k:
            raise ValueError("Heads must divide groups evenly and message must fit a window")
        self.k, self.d_state, self.width = k, d_state, width
        self.heads, self.headdim, self.groups = width // headdim, headdim, groups
        self.gate, self.mixing = gate, mixing
        self.encoder = nn.Linear(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.project = nn.Linear(d_model, width + 2 * groups * d_state + self.heads)
        self.log_a = nn.Parameter(torch.zeros(self.heads))
        self.skip = nn.Parameter(torch.ones(self.heads))
        nn.init.constant_(self.project.bias[-self.heads:], math.log(math.expm1(0.1)))
        self.output = nn.Linear(width, d_model)
        self.readout = nn.Linear(d_model, 1)

    def step(self, features, initial=None, message=None, send_features=False):
        start = tile(features, self.k)
        b, hg, wg, *_ = start.shape
        normalized = self.norm(start)
        tokens = scan(self.project(normalized)).flatten(0, 3)
        x, b_coef, c_coef, dt = tokens.split(
            (self.width, self.groups * self.d_state, self.groups * self.d_state, self.heads), dim=-1,
        )
        x = x.unflatten(-1, (self.heads, self.headdim))
        if message is not None:
            prefix = x[:, :self.d_state] + message.flatten(0, 3).movedim(-1, 1)
            x = torch.cat((prefix, x[:, self.d_state:]), dim=1)
        b_coef = b_coef.unflatten(-1, (self.groups, self.d_state))
        c_coef = c_coef.unflatten(-1, (self.groups, self.d_state))
        dt = F.softplus(dt).clamp_min(torch.finfo(dt.dtype).tiny)
        output, raw_final = oracle(
            x, dt, -self.log_a.exp(), b_coef, c_coef, self.skip,
            None if initial is None else initial.flatten(0, 3),
        )
        packet = None
        if send_features:
            packet = output[:, -self.d_state:].movedim(1, -1).unflatten(0, (b, hg, wg, 4))
        directional = output.flatten(-2).unflatten(0, (b, hg, wg, 4))
        carried = features + self.output(untile(inverse(directional, self.k).mean(dim=3)))
        return Frame(
            self.readout(carried).squeeze(-1), features, carried, raw_final,
            raw_final.unflatten(0, (b, hg, wg, 4)), initial, packet, message,
            normalized, tokens,
        )

    def forward(self, image, rounds, mode, return_trace=False):
        if rounds < 1 or mode not in MODES:
            raise ValueError("Positive rounds and a supported mode are required")
        features = self.encoder(image)
        trace, state, packet = [], None, None
        for _ in range(rounds):
            initial = initial_states(state, mode, self.gate, self.mixing)
            message = initial_states(packet, "sst", self.gate, self.mixing)
            frame = self.step(features, initial, message, send_features=mode == "feature_neighbor")
            features, state, packet = frame.carried, frame.state, frame.sent_features
            if return_trace:
                trace.append(frame)
        return (frame.logits, trace) if return_trace else frame.logits


def zero_neighbor(previous, gate, mixing):
    """Remove only the neighbor branch, preserving the perpendicular local term."""
    return initial_states(previous, "local", gate * (1 - mixing), 0)


def transported_sources(previous, shuffled=False):
    return torch.stack([
        upstream(shuffle_sources(own) if shuffled else own, direction)
        for direction, own in enumerate(previous.unbind(3))
    ], dim=3)


def raw_readout_gradient(logits, frame):
    return torch.autograd.grad(logits, frame.raw_final, retain_graph=True)[0]
