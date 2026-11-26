"""Endpoint-aligned axis scans and an exact short-chunk recurrence."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import torch
from torch import nn
from torch.nn import functional as F

from .model_v3 import dense_oracle
from .ssm import official_ssd

OPPOSITE = (1, 0, 3, 2)


def chunked_ssd(x, dt, a, b, c, d, initial=None, chunk=16, diagnostics=False):
    if chunk < 1 or x.shape[1] < 1:
        raise ValueError("Positive chunk and sequence lengths required")
    outputs, probes = [], []
    state, prefix = initial, None
    for start in range(0, x.shape[1], chunk):
        segment = slice(start, start + chunk)
        y, state, _ = dense_oracle(
            x[:, segment], dt[:, segment], a, b[:, segment], c[:, segment], d, state
        )
        outputs.append(y)
        if diagnostics:
            decay = dt[:, segment] * a
            mask = torch.ones(decay.shape[1], decay.shape[1], device=x.device, dtype=x.dtype).triu()
            running = (decay.transpose(1, 2) @ mask).transpose(1, 2)
            if prefix is not None:
                running = running + prefix
            prefix = running[:, -1:]
            retained = running.exp()
            signal = None
            if initial is not None:
                c_head = c[:, segment].repeat_interleave(x.shape[2] // c.shape[2], dim=2)
                signal = (initial.unsqueeze(1) * c_head.unsqueeze(-2)).sum(-1) * retained.unsqueeze(-1)
            probes.append((start, retained, signal, y))
    return torch.cat(outputs, dim=1), state, probes


def axes(features):
    """[B,H,W,C] to two directions on rows and two on columns."""
    columns = features.transpose(1, 2)
    return features, features.flip(2), columns, columns.flip(2)


def restore(directions):
    a, b, c, d = directions
    return a, b.flip(2), c.transpose(1, 2), d.flip(2).transpose(1, 2)


def communicate(previous, mode, gain):
    if previous is None or gain == 0 or mode in ("C0", "C1", "C6"):
        return (None,) * 4
    if mode not in ("C2", "C3", "C4", "C5"):
        raise ValueError(mode)
    result = []
    for direction in range(4):
        source = previous[direction if mode == "C2" else OPPOSITE[direction]]
        if mode == "C4":
            index = (torch.arange(source.shape[1], device=source.device) + 1) % source.shape[1]
            source = source.index_select(1, index)
        result.append(gain * source)
    return tuple(result)


@dataclass
class CrowdFrame:
    density: torch.Tensor
    features_in: torch.Tensor
    bank: tuple | None
    packet: tuple | None
    probes: tuple


class AxialBlock(nn.Module):
    def __init__(self, width, state, heads, chunk, backend):
        super().__init__()
        self.width, self.state, self.heads = width, state, heads
        self.chunk, self.backend = chunk, backend
        self.norm = nn.LayerNorm(width)
        self.project = nn.Linear(width, width + 2 * state + heads)
        self.log_decay = nn.Parameter(torch.zeros(heads))
        self.skip = nn.Parameter(torch.ones(heads))
        self.output = nn.Linear(width, width)
        nn.init.constant_(self.project.bias[-heads:], -2.252168)

    def forward(self, f, initial, packets, diagnose=False, send=False):
        sequences = axes(self.project(self.norm(f)))
        results, banks, outgoing, probes = [], [], [], []
        directional_groups = [(0, 1, 2, 3)] if f.shape[1] == f.shape[2] else [(0, 1), (2, 3)]
        for group in directional_groups:
            combined = torch.stack([sequences[i] for i in group], dim=0)
            shape = combined.shape[:3]
            x, b, c, dt = combined.flatten(0, 2).split(
                (self.width, self.state, self.state, self.heads), dim=-1
            )
            x = x.unflatten(-1, (self.heads, self.width // self.heads))
            if packets[group[0]] is not None:
                tail = torch.stack([packets[i] for i in group]).flatten(0, 2)
                length = min(x.shape[1], self.state)
                x = torch.cat((x[:, :length] + tail[..., :length].movedim(-1, 1), x[:, length:]), dim=1)
            state = None if initial[group[0]] is None else torch.stack([initial[i] for i in group]).flatten(0, 2)
            dt = F.softplus(dt).clamp_min(torch.finfo(dt.dtype).tiny)
            args = (x, dt, -self.log_decay.exp(), b.unsqueeze(-2), c.unsqueeze(-2), self.skip, state)
            if self.backend == "official":
                out, last = official_ssd(*args)
                diag = ()
            else:
                out, last, diag = chunked_ssd(*args, chunk=self.chunk, diagnostics=diagnose)
            results.extend(out.flatten(-2).unflatten(0, shape).unbind(0))
            banks.extend(last.unflatten(0, shape).unbind(0))
            if send:
                length = min(out.shape[1], self.state)
                # The previous endpoint's closest readout token comes first.
                packet = out[:, -length:].flip(1).movedim(1, -1)
                packet = F.pad(packet, (0, self.state - length))
                outgoing.extend(packet.unflatten(0, shape).unbind(0))
            if diagnose and diag:
                probes.append((group, shape, diag))
        update = self.output(torch.stack(restore(results)).mean(0))
        return f + update, tuple(banks), tuple(outgoing) if send else None, tuple(probes)


class CrowdModel(nn.Module):
    def __init__(self, mode, width=64, state=8, heads=4, rounds=4, gain=.5,
                 chunk=16, backend="chunked", weight_file=None, head_bias=-3.):
        super().__init__()
        if mode not in {f"C{i}" for i in range(7)} or width % heads or state < 1:
            raise ValueError("Unsupported crowd mode or state dimensions")
        self.mode, self.rounds, self.gain = mode, rounds, gain
        if weight_file is not None:
            from torchvision.models import resnet18

            network = resnet18(weights=None)
            network.load_state_dict(torch.load(weight_file, map_location="cpu", weights_only=True))
            self.encoder = nn.Sequential(network.conv1, network.bn1, network.relu,
                                         network.maxpool, network.layer1, network.layer2)
            self.encoder.requires_grad_(False)
            self.frozen_backbone = True
            self.adapter = nn.Conv2d(128, width, 1)
        else:
            self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
                                         nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                         nn.Conv2d(32, width, 3, 2, 1), nn.ReLU())
            self.frozen_backbone = False
            self.adapter = nn.Identity()
        self.head = nn.Linear(width, 1)
        nn.init.constant_(self.head.bias, head_bias)
        self.blocks = nn.ModuleList([
            AxialBlock(width, state, heads, chunk, backend)
            for _ in range(rounds if mode == "C6" else (0 if mode == "C0" else 1))
        ])
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbone:
            self.encoder.eval()
        return self

    def step(self, features, previous=None, packet=None, round_index=0, mode=None, diagnose=False):
        mode = self.mode if mode is None else mode
        if mode == "C0":
            return CrowdFrame(F.softplus(self.head(features).squeeze(-1)), features, None, None, ())
        core = self.blocks[min(round_index, len(self.blocks) - 1)]
        state = (None,) * 4 if mode == "C5" else communicate(previous, mode, self.gain)
        feature_packet = communicate(packet, mode, self.gain) if mode == "C5" else (None,) * 4
        updated, bank, sent, probe = core(features, state, feature_packet, diagnose, send=mode == "C5")
        return CrowdFrame(F.softplus(self.head(updated).squeeze(-1)), features, bank, sent, (updated, probe))

    def forward(self, images, rounds=None, mode=None, diagnose=False):
        features = self.adapter(self.encoder(images)).permute(0, 2, 3, 1)
        count = 1 if self.mode == "C0" else (self.rounds if rounds is None else rounds)
        frames, bank, packet = [], None, None
        for i in range(count):
            frame = self.step(features, bank, packet, i, mode, diagnose)
            if frame.probes:
                features = frame.probes[0]
            frames.append(frame)
            bank, packet = frame.bank, frame.packet
        return frames

    def forward_valid(self, images, sizes, rounds=None):
        """Crop to valid geometry before CNN or state updates, grouped by exact size."""
        result = [None] * len(sizes)
        order = sorted(range(len(sizes)), key=sizes.__getitem__)
        for (height, width), indices in groupby(order, key=sizes.__getitem__):
            ids = list(indices)
            valid = torch.stack([images[i, :, :height, :width] for i in ids])
            output = self(valid, rounds)[-1].density
            for i, predictions in zip(ids, output):
                result[i] = predictions
        return result
