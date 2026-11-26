"""Reset-only models and count-preserving geometry for SDR-006."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .crowd_ssm import AxialBlock, CrowdModel


def project_points(points, width, height, base):
    upper = np.nextafter(np.array([width, height], dtype=np.float64), 0)
    return np.clip(points - base, 0, upper)


class ConvResidual(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.expand = nn.Conv2d(width, hidden, 1)
        self.output = nn.Conv2d(hidden, width, 1)
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        x = self.spatial(self.norm(features).permute(0, 3, 1, 2))
        return features + self.output(F.gelu(self.expand(x))).permute(0, 2, 3, 1)


class AuditModel(nn.Module):
    def __init__(self, mode, rounds, width, state, heads, chunk, backend, head_bias, conv_hidden):
        super().__init__()
        backbone = CrowdModel("C0", width=width, state=state, heads=heads, rounds=rounds, head_bias=head_bias)
        self.encoder, self.head = backbone.encoder, backbone.head
        self.blocks = nn.ModuleList()
        self.audit_mode = mode
        self.rounds = rounds
        if mode in ("F1", "F2", "F4"):
            first = AxialBlock(width, state, heads, chunk, backend)
            self.blocks.append(first)
            if mode == "F2":
                self.blocks.extend(deepcopy(first) for _ in range(rounds - 1))
        if mode == "F3":
            self.blocks.append(ConvResidual(width, conv_hidden))

    def forward(self, images, rounds=None):
        features = self.encoder(images).permute(0, 2, 3, 1)
        if self.audit_mode == "F0":
            return [F.softplus(self.head(features).squeeze(-1))]
        count = self.rounds if rounds is None else rounds
        if self.audit_mode == "F2" and count > self.rounds:
            raise ValueError("The independent-block model has no extrapolation operator")
        result = []
        for step in range(count):
            block = self.blocks[step if self.audit_mode == "F2" else 0]
            if self.audit_mode == "F3":
                features = block(features)
            else:
                features = block(features, (None,) * 4, (None,) * 4)[0]
            result.append(F.softplus(self.head(features).squeeze(-1)))
        return result
