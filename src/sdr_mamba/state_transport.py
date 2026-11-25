"""Synchronous transport over window grids with zero-padded boundaries."""

import torch

MODES = ("reset", "local", "sst", "shuffled", "feature_neighbor")


def upstream(grid: torch.Tensor, direction: int) -> torch.Tensor:
    # Grid axes are batch, window row, window column, then payload axes.
    if direction == 0:
        return torch.cat((torch.zeros_like(grid[:, :, :1]), grid[:, :, :-1]), dim=2)
    if direction == 1:
        return torch.cat((grid[:, :, 1:], torch.zeros_like(grid[:, :, :1])), dim=2)
    if direction == 2:
        return torch.cat((torch.zeros_like(grid[:, :1]), grid[:, :-1]), dim=1)
    if direction == 3:
        return torch.cat((grid[:, 1:], torch.zeros_like(grid[:, :1])), dim=1)
    raise ValueError(f"Unknown direction {direction}")


def shuffle_sources(grid: torch.Tensor) -> torch.Tensor:
    """Fixed within-image derangement for grids with more than one window."""
    _, hg, wg, *_ = grid.shape
    count = hg * wg
    indices = (torch.arange(count, device=grid.device) + max(1, count // 2)) % count
    return grid.flatten(1, 2).index_select(1, indices).unflatten(1, (hg, wg))


def initial_states(previous, mode: str, gate: float, mixing: float):
    if mode not in MODES:
        raise ValueError(f"Unknown state mode {mode}")
    if previous is None or gate == 0 or mode in ("reset", "feature_neighbor"):
        return None
    directions = previous.unbind(dim=3)
    row = (directions[0] + directions[1]) / 2
    column = (directions[2] + directions[3]) / 2
    states = []
    for direction, own in enumerate(directions):
        perpendicular = column if direction < 2 else row
        source = own if mode == "local" else upstream(own, direction)
        if mode == "shuffled":
            # Permute payloads before applying the same destination boundary mask as SST.
            source = upstream(shuffle_sources(own), direction)
        states.append(gate * (mixing * source + (1 - mixing) * perpendicular))
    return torch.stack(states, dim=3)


def neighbor_features(windows: torch.Tensor) -> torch.Tensor:
    return torch.stack([upstream(windows, direction) for direction in range(4)]).mean(0)
