"""Shared recurrent window operator and pixelwise encoder/readout."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .scan import inverse, scan, tile, untile
from .ssm import official_ssd, oracle
from .state_transport import MODES, initial_states, neighbor_features


@dataclass
class Round:
    logits: torch.Tensor
    feature_in: torch.Tensor
    initial: torch.Tensor | None
    final: torch.Tensor


class LoopedSSM(nn.Module):
    def __init__(self, k=4, d_model=32, d_state=8, expand=2, headdim=16,
                 groups=1, gate=0.5, mixing=0.6, backend="oracle"):
        super().__init__()
        inner = d_model * expand
        if min(k, d_model, d_state, expand, headdim, groups) < 1 or inner % (headdim * groups):
            raise ValueError("Positive dimensions and groups dividing heads are required")
        if not 0 <= gate <= 1 or not 0 <= mixing <= 1:
            raise ValueError("Gate and mixing must belong to [0, 1]")
        self.k, self.inner, self.d_state = k, inner, d_state
        self.heads, self.headdim, self.groups = inner // headdim, headdim, groups
        self.gate, self.mixing = gate, mixing
        self.encoder = nn.Linear(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, inner + 2 * groups * d_state + self.heads)
        self.log_a = nn.Parameter(torch.zeros(self.heads))
        self.d = nn.Parameter(torch.ones(self.heads))
        nn.init.constant_(self.projection.bias[-self.heads:], math.log(math.expm1(0.1)))
        self.output = nn.Linear(inner, d_model)
        self.head = nn.Linear(d_model, 1)
        self.ssm = {"oracle": oracle, "ssd": official_ssd}[backend]

    def step(self, feature, initial):
        windows = tile(feature, self.k)
        b, hg, wg, *_ = windows.shape
        # Project each pixel once; all directions use exactly the same coefficients.
        sequence = scan(self.projection(self.norm(windows))).flatten(0, 3)
        x, b_coef, c_coef, dt = sequence.split(
            (self.inner, self.groups * self.d_state, self.groups * self.d_state, self.heads),
            dim=-1,
        )
        x = x.unflatten(-1, (self.heads, self.headdim))
        b_coef = b_coef.unflatten(-1, (self.groups, self.d_state))
        c_coef = c_coef.unflatten(-1, (self.groups, self.d_state))
        dt = F.softplus(dt).clamp_min(torch.finfo(dt.dtype).tiny)
        output, final = self.ssm(
            x, dt, -self.log_a.exp(), b_coef, c_coef, self.d,
            None if initial is None else initial.flatten(0, 3),
        )
        directional = output.flatten(-2).unflatten(0, (b, hg, wg, 4))
        updated = feature + self.output(untile(inverse(directional, self.k).mean(dim=3)))
        return self.head(updated).squeeze(-1), updated, final.unflatten(0, (b, hg, wg, 4))

    def forward(self, image, rounds=4, state_mode="sst", return_trace=False):
        if rounds < 1 or state_mode not in MODES:
            raise ValueError("Positive round count and a known state mode are required")
        feature = self.encoder(image)
        previous_state = None
        trace = []
        for turn in range(rounds):
            initial = initial_states(previous_state, state_mode, self.gate, self.mixing)
            feature_in = feature
            if turn and state_mode == "feature_neighbor" and self.gate:
                feature_in = feature + self.gate * untile(neighbor_features(tile(feature, self.k)))
            logits, feature, previous_state = self.step(feature_in, initial)
            if return_trace:
                trace.append(Round(logits, feature_in, initial, previous_state))
        return (logits, trace) if return_trace else logits
