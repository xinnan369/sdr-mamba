"""Matched LoopCrowd controls and a window-local Mamba-2-style reference mixer."""

import math

import torch
from torch import nn
from torch.nn import functional as F


COMMON_PREFIXES = ("p2.", "p3.", "project.", "fuse.", "point_logits.", "point_offsets.")


def reference_scan(x, dt, a, b, c, skip, valid):
    """Exact short causal SSD summation with one shared B/C group and reset state.

    x: [lines, length, heads, head_width]; dt: [lines, length, heads].
    B/C: [lines, length, state]; valid: [lines, length].
    """
    output_dtype = x.dtype
    calc_dtype = torch.float32 if x.dtype in (torch.bfloat16, torch.float16) else x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, dt, a, b, c, skip = (item.to(calc_dtype) for item in (x, dt, a, b, c, skip))
        dt = dt.masked_fill(~valid[..., None], 0)
        b = b.masked_fill(~valid[..., None], 0)
        c = c.masked_fill(~valid[..., None], 0)
        x = x.masked_fill(~valid[..., None, None], 0)
        prefix = (dt * a).transpose(1, 2).cumsum(-1)
        causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).tril()
        decay = (prefix.unsqueeze(-1) - prefix.unsqueeze(-2)).masked_fill(~causal, -torch.inf).exp()
        content = c @ b.transpose(-1, -2)
        kernel = decay * content.unsqueeze(1)
        y = (kernel @ (dt[..., None] * x).transpose(1, 2)).transpose(1, 2)
        y = y + skip[None, None, :, None] * x
        return y.to(output_dtype)


class WindowSSDMixer(nn.Module):
    """Drop-in self-attention replacement for LoopCrowd's fixed windows."""

    def __init__(self, width, window, inner, head_width, state):
        super().__init__()
        if inner % head_width:
            raise ValueError("Mixer inner width must be divisible by head width")
        self.window, self.inner, self.state = window, inner, state
        self.heads, self.head_width = inner // head_width, head_width
        self.input = nn.Linear(width, 2 * inner + 2 * state + self.heads)
        self.output = nn.Linear(inner, width)
        self.log_decay = nn.Parameter(torch.zeros(self.heads))
        self.dt_bias = nn.Parameter(torch.full((self.heads,), math.log(math.expm1(.1))))
        self.skip = nn.Parameter(torch.ones(self.heads))

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        if key is not query or value is not query or need_weights:
            raise ValueError("Reference mixer only supports self-mixing without attention weights")
        n, tokens, _ = query.shape
        w = self.window
        if tokens != w * w:
            raise ValueError("Expected one flattened square window")
        valid = query.new_ones((n, w, w), dtype=torch.bool) if key_padding_mask is None else ~key_padding_mask.reshape(n, w, w)
        projected = self.input(query).reshape(n, w, w, -1)
        gate, packed = projected.split((self.inner, projected.shape[-1]-self.inner), dim=-1)
        columns, column_valid = packed.transpose(1, 2), valid.transpose(1, 2)
        packed = torch.stack((packed, packed.flip(2), columns, columns.flip(2)))
        masks = torch.stack((valid, valid.flip(2), column_valid, column_valid.flip(2)))
        x, b, c, dt = packed.flatten(0, 2).split((self.inner, self.state, self.state, self.heads), dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias.float()) if dt.dtype in (torch.bfloat16, torch.float16) else F.softplus(dt + self.dt_bias)
        scan = reference_scan(x.unflatten(-1, (self.heads, self.head_width)), dt, -self.log_decay.exp(), b, c,
                              self.skip, masks.flatten(0, 2))
        east, west, south, north = scan.flatten(-2).reshape(4, n, w, w, self.inner)
        mixed = (east + west.flip(2) + south.transpose(1, 2) + north.flip(2).transpose(1, 2)) * .25
        update = self.output(F.silu(gate) * mixed).masked_fill(~valid[..., None], 0)
        return update.reshape_as(query), None


def recipe(config, arm, seed):
    base = config["arms"][arm]["base"]
    return config["recipe"] | {"variant": base, "seed": seed,
                               "early_loss_weight": None if base == "cnn_only" else .25}


def initialize_backbone(model, weights):
    """Map a verified ResNet-18 state dict into the unchanged upstream P2/P3."""
    from torchvision.models import resnet18

    with torch.random.fork_rng(devices=[]):
        backbone = resnet18(weights=None)
    # The accepted legacy checkpoint predates per-BN batch counters. Let
    # torchvision's normal strict state loader supply those buffers.
    backbone.load_state_dict(weights)
    p2 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
    model.p2.load_state_dict(p2.state_dict())
    model.p3.load_state_dict(backbone.layer2.state_dict())


def build_model(config, arm, seed, weights):
    from loopcrowd.experiment import seed_all
    from loopcrowd.model import LoopCrowd

    seed_all(seed)
    model = LoopCrowd(config["arms"][arm]["base"], slots=config["recipe"]["slots"], pretrained=False)
    initialize_backbone(model, weights)
    if config["arms"][arm]["mixer"] == "reference_ssd":
        geometry = config["geometry"]
        with torch.random.fork_rng(devices=[]):
            for index, block in enumerate(model.blocks):
                if block.window_size != geometry["window"] or model.shifted:
                    raise ValueError("Fixed-window geometry differs from the frozen plan")
                torch.random.default_generator.manual_seed(seed + 634621 + index)
                block.attn = WindowSSDMixer(*(geometry[key] for key in ("width", "window", "inner", "head_width", "state")))
    return model
