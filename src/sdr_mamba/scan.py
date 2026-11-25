"""Window tiling and four permutations of the pixels inside each window."""

import torch

DIRECTIONS = ("row", "reverse_row", "column", "reverse_column")


def tile(image: torch.Tensor, k: int) -> torch.Tensor:
    b, h, w, c = image.shape
    if k < 1 or h < 1 or w < 1 or h % k or w % k:
        raise ValueError("H and W must be positive multiples of the window size")
    return image.reshape(b, h // k, k, w // k, k, c).transpose(2, 3)


def untile(windows: torch.Tensor) -> torch.Tensor:
    b, hg, wg, h, w, c = windows.shape
    return windows.transpose(2, 3).reshape(b, hg * h, wg * w, c)


def scan(windows: torch.Tensor) -> torch.Tensor:
    row = windows.flatten(-3, -2)
    column = windows.transpose(-3, -2).flatten(-3, -2)
    return torch.stack((row, row.flip(-2), column, column.flip(-2)), dim=-3)


def inverse(directional: torch.Tensor, k: int) -> torch.Tensor:
    row, rev_row, column, rev_column = directional.unbind(-3)

    def grid(sequence):
        return sequence.unflatten(-2, (k, k))

    return torch.stack(
        (grid(row), grid(rev_row.flip(-2)),
         grid(column).transpose(-3, -2), grid(rev_column.flip(-2)).transpose(-3, -2)),
        dim=-4,
    )
