"""Paired, balanced components with distances derived from the pixel graph."""

from collections import Counter, deque
import hashlib
import random

import torch

WINDOW_BINS = ((0, 0), (1, 1), (2, 3), (4, 6), (7, None))
PIXEL_BINS = ((0, 0), (1, 3), (4, 7), (8, 15), (16, 31), (32, None))


def adjacent(y, x, height, width):
    for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= yy < height and 0 <= xx < width:
            yield yy, xx


def distances(seed, edges):
    result = {seed: 0}
    queue = deque([seed])
    while queue:
        point = queue.popleft()
        for neighbor in edges(point):
            if neighbor not in result:
                result[neighbor] = result[point] + 1
                queue.append(neighbor)
    return result


def pixel_bfs(occupancy, seed, size):
    if seed not in occupancy:
        raise ValueError("Seed must be walkable")
    return distances(seed, lambda p: (q for q in adjacent(*p, size, size) if q in occupancy))


def window_graph_distances(pixel_distances, seed, k, size):
    def window(p):
        return p[0] // k, p[1] // k

    graph = {window(p): set() for p in pixel_distances}
    for p in pixel_distances:
        source = window(p)
        for q in adjacent(*p, size, size):
            if q in pixel_distances and window(q) != source:
                graph[source].add(window(q))
    return distances(window(seed), graph.__getitem__)


def components(occupancy, size):
    unseen = set(occupancy)
    found = []
    while unseen:
        group = set(pixel_bfs(occupancy, min(unseen), size))
        found.append(group)
        unseen.difference_update(group)
    return found


def make_map(rng, size, area):
    """Grow three equal-area polyominoes around bent corridors in disjoint bands."""
    bands = ((1, 9), (11, 20), (22, 30))
    transform = rng.randrange(8)

    def orient(point):
        y, x = point
        if transform & 1:
            y = size - 1 - y
        if transform & 2:
            x = size - 1 - x
        return (x, y) if transform & 4 else (y, x)

    occupancy, seeds = set(), []
    for low, high in bands:
        y = rng.randint(low + 2, high - 2)
        x_start, x_end = rng.randint(1, 2), rng.randint(29, 30)
        path = []
        for x in range(x_start, x_end + 1):
            path.append((y, x))
            if x % 3 == 0:
                y = max(low + 1, min(high - 1, y + rng.choice((-1, 0, 1))))
                path.append((y, x))
        region = set(path)
        frontier = set()

        def add_frontier(p):
            frontier.update(q for q in adjacent(*p, size, size)
                            if low <= q[0] <= high and q not in region)

        for p in region:
            add_frontier(p)
        while len(region) < area:
            p = rng.choice(sorted(frontier))
            frontier.remove(p)
            region.add(p)
            add_frontier(p)
        # Vary endpoint versus central roots while every map is paired across all components.
        root_x = rng.choice((7, 9, 11, 14, 17, 20, 22, 24))
        root = rng.choice(sorted(p for p in region if p[1] == root_x))
        seeds.append(orient(root))
        occupancy.update(map(orient, region))
    return occupancy, seeds


def bucket(value, bins):
    return next(f"{low}-{high}" if high is not None else f"{low}+"
                for low, high in bins if value >= low and (high is None or value <= high))


def generate(config):
    size, k = config["size"], config["model"]["k"]
    if size != 32 or k != 4:
        raise ValueError("The v2 layout has a fixed 32-pixel grid and k=4")
    points = [(y, x) for y in range(size) for x in range(size)]
    splits, manifest, seen = {}, {}, set()
    low, high = config["area_range"]
    for split, spec in config["data"].items():
        rng = random.Random(spec["seed"])
        digest = hashlib.sha256()
        metadata, arrays = [], []
        rejects = Counter()
        while len(metadata) < spec["maps"]:
            if sum(rejects.values()) >= 50 * spec["maps"]:
                raise RuntimeError(f"Rejection budget exhausted for {split}")
            area = rng.randint(low, high)
            occupancy, seeds = make_map(rng, size, area)
            groups = components(occupancy, size)
            if sorted(map(len, groups)) != [area] * 3:
                rejects["component_area_or_count"] += 1
                continue
            occ = [int(p in occupancy) for p in points]
            map_hash = hashlib.sha256(bytes(occ)).hexdigest()
            if map_hash in seen:
                rejects["duplicate_occupancy"] += 1
                continue
            paths = [pixel_bfs(occupancy, seed, size) for seed in seeds]
            graphs = [window_graph_distances(path, seed, k, size) for path, seed in zip(paths, seeds)]
            if any(len(path) != area for path in paths) or set().union(*paths) != occupancy:
                rejects["seed_component_coverage"] += 1
                continue
            if any(max(graph.values()) < 4 or not {1, 2, 3}.intersection(graph.values()) for graph in graphs):
                rejects["window_distance_coverage"] += 1
                continue
            seen.add(map_hash)
            map_id = len(metadata)
            row = {"map_id": map_id, "occupancy_sha256": map_hash, "component_areas": [area] * 3, "samples": []}
            for seed, path, graph in zip(seeds, paths, graphs):
                seed_mask = [int(p == seed) for p in points]
                pixel_distance = [path.get(p, -1) for p in points]
                graph_distance = [graph.get((p[0] // k, p[1] // k), -1) for p in points]
                target = [int(d >= 0) for d in pixel_distance]
                encoded = bytes(occ + seed_mask + target)
                digest.update(encoded)
                row["samples"].append({
                    "sample_sha256": hashlib.sha256(encoded).hexdigest(), "seed_yx": list(seed),
                    "target_ratio": area / len(occupancy), "target_eccentricity": max(path.values()),
                    "pixel_distance_histogram": dict(sorted(Counter(path.values()).items())),
                    "window_graph_pixel_histogram": dict(sorted(Counter(
                        d for d, present in zip(graph_distance, target) if present
                    ).items())),
                    "window_graph_window_histogram": dict(sorted(Counter(graph.values()).items())),
                })
                arrays.append((list(zip(occ, seed_mask)), target, pixel_distance, graph_distance))
            metadata.append(row)
        x, y, pixel, window = zip(*arrays)
        splits[split] = {
            "image": torch.tensor(x, dtype=torch.float32).reshape(-1, size, size, 2),
            "target": torch.tensor(y, dtype=torch.bool).reshape(-1, size, size),
            "pixel": torch.tensor(pixel, dtype=torch.int16).reshape(-1, size, size),
            "window": torch.tensor(window, dtype=torch.int16).reshape(-1, size, size),
        }
        manifest[split] = {"seed": spec["seed"], "count": len(x), "sha256": digest.hexdigest(),
                           "rejections": dict(rejects), "maps": metadata}
    return splits, manifest
