"""Line-aligned state transport with explicit physical lane indices."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model_v3 import LoopV3, dense_oracle
from .scan import inverse, scan, tile, untile
from .state_transport import initial_states, upstream


def lanes(windows):
    column = windows.transpose(-3, -2)
    return torch.stack((windows, windows.flip(-2), column, column.flip(-2)), dim=3)


def inverse_lanes(sequences):
    a, b, c, d = sequences.unbind(3)
    return torch.stack((a, b.flip(-2), c.transpose(-3, -2),
                        d.flip(-2).transpose(-3, -2)), dim=3)


def neighbors(previous, mode, gate=.5, mixing=.6, permute=False, zero=False):
    if previous is None or gate == 0 or mode == "reset":
        return None
    pieces = []
    for direction, local in enumerate(previous.unbind(3)):
        source = local if mode == "local" else upstream(local, direction)
        if permute:
            rotation = (torch.arange(source.shape[3], device=source.device) + 1) % source.shape[3]
            source = source.index_select(3, rotation)
        if zero:
            source = torch.zeros_like(source)
        pieces.append(gate * (mixing * source + (1 - mixing) * local))
    return torch.stack(pieces, dim=3)


@dataclass
class Frame:
    logits: torch.Tensor
    feature_in: torch.Tensor
    feature_out: torch.Tensor
    state: torch.Tensor | None
    raw: torch.Tensor | None
    initial: torch.Tensor | None
    sent: torch.Tensor | None
    diagnostic: tuple | None


class LineLoop(nn.Module):
    def __init__(self, mode, dstate):
        super().__init__()
        super_model = LoopV3()
        self.k, self.mode, self.dstate = 4, mode, dstate
        self.gate, self.mixing = .5, .6
        self.encoder, self.norm = super_model.encoder, super_model.norm
        self.output, self.head = super_model.output, super_model.readout
        self.log_a, self.skip = super_model.log_a, super_model.skip
        if dstate == 4:
            self.project = super_model.project
        else:
            self.project = nn.Linear(16, 18 + 2 * dstate)
            with torch.no_grad():
                # Match channel and dt initialization with dstate4; retain the first B/C channels.
                source = torch.cat((torch.arange(16), torch.arange(16, 16 + dstate),
                                    torch.arange(20, 20 + dstate), torch.arange(24, 26)))
                self.project.weight.copy_(super_model.project.weight[source])
                self.project.bias.copy_(super_model.project.bias[source])

    def recur(self, tokens, initial=None, message=None, diagnose=False):
        x, b, c, dt = tokens.split((16, self.dstate, self.dstate, 2), dim=-1)
        x = x.unflatten(-1, (2, 8))
        if message is not None:
            prefix = x[:, :self.dstate] + message.movedim(-1, 1)
            x = torch.cat((prefix, x[:, self.dstate:]), dim=1)
        dt = F.softplus(dt).clamp_min(torch.finfo(dt.dtype).tiny)
        return dense_oracle(x, dt, -self.log_a.exp(), b.unsqueeze(-2), c.unsqueeze(-2),
                            self.skip, initial, diagnose)

    def global_context(self, projected, diagnose):
        outputs = []
        for along_column in (False, True):
            grids = projected.transpose(1, 2) if along_column else projected
            pairs = torch.stack((grids, grids.flip(-2)), dim=2)
            result, _, _ = self.recur(pairs.flatten(0, 2), diagnose=diagnose)
            row, reverse = result.flatten(-2).unflatten(0, pairs.shape[:3]).unbind(2)
            output = row + reverse.flip(-2)
            outputs.append(output.transpose(1, 2) if along_column else output)
        return (outputs[0] + outputs[1]) / 4

    def step(self, feature, initial=None, packet=None, diagnose=False):
        windows = tile(self.project(self.norm(feature)), self.k)
        if self.mode == "global":
            update = self.output(self.global_context(untile(windows), diagnose=False))
            final = feature + update
            return Frame(self.head(final).squeeze(-1), feature, final, None, None, None, None, None)
        raster = self.mode in ("raster", "raster_same")
        seq = scan(windows) if raster else lanes(windows)
        prefix = seq.shape[:-2]
        output, raw, diagnostic = self.recur(
            seq.flatten(0, len(prefix) - 1),
            None if initial is None else initial.flatten(0, len(prefix) - 1),
            None if packet is None else packet.flatten(0, len(prefix) - 1),
            diagnose,
        )
        sent = None
        if self.mode == "line_feature":
            sent = output[:, -self.dstate:].movedim(1, -1).unflatten(0, prefix)
        directional = output.flatten(-2).unflatten(0, prefix)
        restored = inverse(directional, self.k) if raster else inverse_lanes(directional)
        final = feature + self.output(untile(restored.mean(3)))
        return Frame(self.head(final).squeeze(-1), feature, final, raw.unflatten(0, prefix),
                     raw, initial, sent, diagnostic)

    def forward(self, image, rounds, diagnose=False):
        features = self.encoder(image)
        history, state, feature_packet = [], None, None
        for _ in range(rounds):
            packet = None
            if self.mode == "raster":
                initial = initial_states(state, "sst", self.gate, self.mixing)
            elif self.mode == "line_feature":
                initial = None
                packet = neighbors(feature_packet, "line", self.gate, self.mixing)
            else:
                initial = neighbors(state, self.mode, self.gate, self.mixing, self.mode == "line_wrong")
            turn = self.step(features, initial, packet, diagnose)
            features, state, feature_packet = turn.feature_out, turn.state, turn.sent
            history.append(turn)
        return history


def model_for(mode, dstate):
    return LoopV3("strong_feature") if mode == "strong" else LineLoop(mode, dstate)
