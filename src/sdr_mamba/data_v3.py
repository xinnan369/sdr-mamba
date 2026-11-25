"""Fresh v2 draws and a fixed nested-U out-of-distribution family."""

from collections import Counter
import hashlib
import json
import random
from pathlib import Path

import torch

from sdr_mamba.data_v2 import adjacent, components, generate, pixel_bfs, window_graph_distances


def closed_edges(occupancy, k=4):
    """Count adjacent occupied windows with no aligned occupied boundary pair."""
    graph = {(y // k, x // k) for y, x in occupancy}
    connected = set()
    for y, x in occupancy:
        src = y // k, x // k
        for dy, dx in ((0, 1), (1, 0)):
            neighbor = y + dy, x + dx
            dest = neighbor[0] // k, neighbor[1] // k
            if dest != src and neighbor in occupancy:
                connected.add((src, dest))
    return sum(
        other in graph and (one, other) not in connected
        for one in graph for other in ((one[0] + 1, one[1]), (one[0], one[1] + 1))
    )


def nested_u(rng, area):
    transform = rng.randrange(8)
    regions = []
    for left, right, bottom, top in ((1, 30, 30, 2), (6, 25, 25, 3), (11, 20, 20, 3)):
        left += rng.randrange(2)
        right -= rng.randrange(2)
        bottom -= rng.randrange(2)
        top += rng.randrange(3)
        right_top = top + rng.randrange(4)
        regions.append({(y, left) for y in range(top, bottom + 1)}
                       | {(y, right) for y in range(right_top, bottom + 1)}
                       | {(bottom, x) for x in range(left, right + 1)})
    if max(map(len, regions)) > area:
        return None

    def neighbors(p):
        return adjacent(*p, 32, 32)

    for i, region in enumerate(regions):
        guards = set().union(*(other for j, other in enumerate(regions) if j != i))
        forbidden = guards | {q for p in guards for q in neighbors(p)}
        frontier = {q for p in region for q in neighbors(p)} - region - forbidden
        while frontier and len(region) < area:
            p = rng.choice(sorted(frontier))
            frontier.remove(p)
            region.add(p)
            frontier.update(q for q in neighbors(p) if q not in region and q not in forbidden)
        if len(region) != area:
            return None

    def orient(p):
        y, x = p
        y = 31 - y if transform & 1 else y
        x = 31 - x if transform & 2 else x
        return (x, y) if transform & 4 else (y, x)

    return [{orient(p) for p in region} for region in regions]


def ood(config, forbidden):
    spec = config["data"]["ood"]
    rng = random.Random(spec["seed"])
    points = [(y, x) for y in range(32) for x in range(32)]
    images, targets, pixels, windows, records = [], [], [], [], []
    rejects = Counter()
    digest = hashlib.sha256()
    while len(records) < spec["maps"]:
        if sum(rejects.values()) > 80 * spec["maps"]:
            raise ValueError(f"OOD generation exhausted: {dict(rejects)}")
        area = rng.randint(*config["ood_area"])
        regions = nested_u(rng, area)
        if regions is None:
            rejects["growth_or_corridor_area"] += 1
            continue
        occupancy = set().union(*regions)
        groups = components(occupancy, 32)
        if sorted(map(len, groups)) != [area] * 3:
            rejects["component_verification"] += 1
            continue
        gates_closed = closed_edges(occupancy)
        if not gates_closed:
            rejects["no_hard_negative"] += 1
            continue
        occ = [int(p in occupancy) for p in points]
        key = hashlib.sha256(bytes(occ)).hexdigest()
        if key in forbidden:
            rejects["seen_occupancy"] += 1
            continue
        cases = []
        for group in groups:
            candidates = sorted(group)
            rng.shuffle(candidates)
            for seed in candidates:
                pixel = pixel_bfs(occupancy, seed, 32)
                window = window_graph_distances(pixel, seed, 4, 32)
                if any(abs(seed[0] // 4 - p[0] // 4) + abs(seed[1] // 4 - p[1] // 4) >= 4 for p in pixel):
                    cases.append((seed, pixel, window))
                    break
        if len(cases) != 3:
            rejects["no_far_root_in_component"] += 1
            continue
        forbidden.add(key)
        row = {"map_id": len(records), "occupancy_sha256": key, "component_areas": [area] * 3,
               "closed_walking_window_edges": gates_closed, "samples": []}
        for seed, pixel, window in cases:
            marker = [int(p == seed) for p in points]
            label = [int(p in pixel) for p in points]
            encoded = bytes(occ + marker + label)
            digest.update(encoded)
            row["samples"].append({
                "seed_yx": list(seed), "sample_sha256": hashlib.sha256(encoded).hexdigest(),
                "target_ratio": 1 / 3, "target_eccentricity": max(pixel.values()),
                "pixel_distance_histogram": dict(Counter(pixel.values())),
                "window_graph_pixel_histogram": dict(Counter(window[y // 4, x // 4] for y, x in pixel)),
            })
            images.append(list(zip(occ, marker)))
            targets.append(label)
            pixels.append([pixel.get(p, -1) for p in points])
            windows.append([window.get((y // 4, x // 4), -1) for y, x in points])
        records.append(row)
    data = {"image": torch.tensor(images, dtype=torch.float32).reshape(-1, 32, 32, 2),
            "target": torch.tensor(targets, dtype=torch.bool).reshape(-1, 32, 32),
            "pixel": torch.tensor(pixels, dtype=torch.int16).reshape(-1, 32, 32),
            "window": torch.tensor(windows, dtype=torch.int16).reshape(-1, 32, 32)}
    return data, {"seed": spec["seed"], "count": len(images), "sha256": digest.hexdigest(),
                  "rejections": dict(rejects), "maps": records}


def build(config):
    original = json.loads(Path("configs/sdr002.json").read_text())
    previous = json.loads(Path("reports/SDR-002/data_manifest.json").read_text())
    old_test = {m["occupancy_sha256"] for m in previous["test"]["maps"]}
    original["data"] = {name: config["data"][name] for name in ("train", "validation", "test")}
    splits, manifest = generate(original)
    id_hashes = [m["occupancy_sha256"] for split in manifest.values() for m in split["maps"]]
    if len(set(id_hashes)) != len(id_hashes) or set(id_hashes) & old_test:
        raise RuntimeError("Fresh ID occupancy overlap")
    if config["ood_family"] != "nested-open-u-corridors":
        raise ValueError("Unknown OOD family")
    splits["ood"], manifest["ood"] = ood(config, set(id_hashes) | old_test)
    grid = torch.arange(32) // 4
    for split, data in splits.items():
        _, ys, xs = data["image"][..., 1].nonzero(as_tuple=True)
        data["grid"] = ((grid[None, :, None] - ys[:, None, None] // 4).abs()
                        + (grid[None, None, :] - xs[:, None, None] // 4).abs()).to(torch.int16)
        for i, row in enumerate(manifest[split]["maps"]):
            for j, sample in enumerate(row["samples"]):
                idx = i * 3 + j
                values = data["grid"][idx][data["target"][idx]].tolist()
                sample["grid_target_histogram"] = dict(sorted(Counter(values).items()))
    return splits, manifest, original
