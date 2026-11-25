import json
from pathlib import Path

import torch

from sdr_mamba.data import component, generate


def test_bfs_disallows_diagonals_and_edges_do_not_wrap():
    grid = [1, 0, 1, 0, 1, 1, 0, 1, 0]
    assert component(grid, 0, 3) == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert component(grid, 2, 3) == [0, 0, 1, 0, 1, 1, 0, 1, 0]


def test_fixed_splits_and_independent_targets():
    config = json.loads((Path(__file__).parents[1] / "configs/pilot.json").read_text())
    splits, manifest = generate(config)
    assert generate(config)[1] == manifest
    hashes = [r["map_sha256"] for split in manifest.values() for r in split["samples"]]
    assert len(set(hashes)) == len(hashes)
    for image, target, distance in splits.values():
        foreground, seeds = image.unbind(-1)
        assert torch.all(seeds.sum((1, 2)) == 1)
        assert torch.all(seeds <= target) and torch.all(target <= foreground)
        assert torch.all((foreground - target).sum((1, 2)) >= 8)
        assert torch.all(((distance >= 4) * target).sum((1, 2)) > 0)
        assert torch.all((((distance > 0) & (distance < 4)) * target).sum((1, 2)) > 0)
