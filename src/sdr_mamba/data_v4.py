"""Fresh area-matched families with the same eccentricity-rank root rule."""

from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random

import torch

from .data_v2 import components, make_map, window_graph_distances
from .data_v3 import closed_edges, nested_u

SIZE = 32
K = 4
POINTS = [(y, x) for y in range(SIZE) for x in range(SIZE)]


def adjacent(p):
    y, x = p
    return ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))


def branched(rng, area):
    regions = []
    # Separated ragged domains contain branching trees with independently chosen bends.
    domains = [
        {(y, x) for y in range(1, 19) for x in range(1, 20) if y < 14 + x // 3},
        {(y, x) for y in range(3, 30) for x in range(19, 31) if x > 20 + (y // 6) % 2},
        {(y, x) for y in range(13, 31) for x in range(1, 20) if y > 15 + x // 3},
    ]
    tips = [
        ((2, 2), (2, 18), (13, 2), (16, 18)),
        ((3, 29), (8, 22), (20, 29), (28, 22)),
        ((17, 2), (23, 18), (29, 2), (29, 18)),
    ]
    hubs = ((7, 9), (14, 26), (25, 11))
    for legal, ends, hub in zip(domains, tips, hubs):
        region = {hub}
        for tip in ends:
            end = rng.choice(sorted({tip, *adjacent(tip)} & legal))
            path = {end: None}
            queue = deque([end])
            while queue:
                pos = queue.popleft()
                if pos in region:
                    while pos is not None:
                        region.add(pos)
                        pos = path[pos]
                    break
                directions = list(adjacent(pos))
                rng.shuffle(directions)
                for nxt in directions:
                    if nxt in legal and nxt not in path:
                        path[nxt] = pos
                        queue.append(nxt)
        frontier = {q for p in region for q in adjacent(p) if q in legal and q not in region}
        while frontier and len(region) < area:
            p = rng.choice(sorted(frontier))
            frontier.remove(p)
            region.add(p)
            frontier.update(q for q in adjacent(p) if q in legal and q not in region)
        if len(region) != area:
            return None
        regions.append(region)
    transform = rng.randrange(8)

    def orient(p):
        y, x = p
        y = 31 - y if transform & 1 else y
        x = 31 - x if transform & 2 else x
        return (x, y) if transform & 4 else (y, x)

    return [{orient(p) for p in region} for region in regions]


def roots(group, rank, rng):
    adjacency = {p: [q for q in adjacent(p) if q in group] for p in group}
    candidates = []
    eccentricity = []
    for seed in sorted(group):
        queue = [seed]
        distance = {seed: 0}
        for p in queue:
            for q in adjacency[p]:
                if q not in distance:
                    distance[q] = distance[p] + 1
                    queue.append(q)
        eccentricity.append(max(distance.values()))
        ys, xs = seed[0] // K, seed[1] // K
        if any(abs(y // K - ys) + abs(x // K - xs) >= 4 for y, x in group):
            candidates.append((seed, distance, eccentricity[-1]))
    if not candidates:
        return None
    count = Counter(eccentricity)
    percentiles, n = {}, 0
    for value, length in sorted(count.items()):
        percentiles[value] = (n + length / 2) / len(group)
        n += length
    delta = min(abs(percentiles[ecc] - rank) for _, _, ecc in candidates)
    seed, pixels, ecc = rng.choice([triple for triple in candidates
                                  if abs(percentiles[triple[2]] - rank) <= delta + 1e-9])
    return seed, pixels, ecc, percentiles[ecc]


def historical_test_hashes():
    hashes = set()
    for study, names in (("SDR-001", ("test",)), ("SDR-002", ("test",)), ("SDR-003", ("test", "ood"))):
        doc = json.loads(Path(f"reports/{study}/data_manifest.json").read_text(encoding="utf-8"))
        for name in names:
            partition = doc[name]
            rows = partition.get("maps", partition.get("samples"))
            hashes.update(row.get("occupancy_sha256", row.get("map_sha256")) for row in rows)
    return hashes


def build(config):
    forbidden = historical_test_hashes()
    datasets, manifest = {}, {}
    for split, spec in config["data"].items():
        rng = random.Random(spec["seed"])
        images, labels, pixels, graphs, grids, ranks, bends, records = [], [], [], [], [], [], [], []
        rejected = Counter()
        checksum = hashlib.sha256()
        while len(records) < spec["maps"]:
            if sum(rejected.values()) >= 100 * spec["maps"]:
                raise RuntimeError(f"{split} generation exhausted: {dict(rejected)}")
            area = rng.randint(*config["area"])
            family = spec["family"]
            if family == "id":
                occupancy, _ = make_map(rng, SIZE, area)
            else:
                region = nested_u(rng, area) if family == "nested_u" else branched(rng, area)
                if region is None:
                    rejected["construction_area"] += 1
                    continue
                occupancy = set().union(*region)
            regions = components(occupancy, SIZE)
            if sorted(map(len, regions)) != [area] * 3:
                rejected["component_count_or_area"] += 1
                continue
            occ = [int(p in occupancy) for p in POINTS]
            key = hashlib.sha256(bytes(occ)).hexdigest()
            if key in forbidden:
                rejected["occupancy_hash_overlap"] += 1
                continue
            closed = closed_edges(occupancy)
            if family != "id" and not closed:
                rejected["missing_closed_edge"] += 1
                continue
            desired = list(config["root_percentiles"])
            rng.shuffle(desired)
            cases = [roots(region, rank, rng) for region, rank in zip(regions, desired)]
            if any(c is None for c in cases):
                rejected["missing_far_root"] += 1
                continue
            forbidden.add(key)
            row = {"map_id": len(records), "occupancy_sha256": key, "component_areas": [area] * 3,
                   "closed_walking_window_edges": closed, "samples": []}
            for (seed, pixel, ecc, rank), requested in zip(cases, desired):
                window = window_graph_distances(pixel, seed, K, SIZE)
                grid = [abs(y // K - seed[0] // K) + abs(x // K - seed[1] // K) for y, x in POINTS]
                label = [int(p in pixel) for p in POINTS]
                seed_mask = [int(p == seed) for p in POINTS]
                # A turn is required when no single-axis path exists to the pixel.
                straight = {seed}
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    y, x = seed
                    while (y + dy, x + dx) in pixel:
                        y, x = y + dy, x + dx
                        straight.add((y, x))
                encoded = bytes(occ + seed_mask + label)
                checksum.update(encoded)
                row["samples"].append({
                    "seed_yx": list(seed), "target_ratio": 1 / 3, "target_eccentricity": ecc,
                    "requested_rank": requested, "eccentricity_rank": rank,
                    "sample_sha256": hashlib.sha256(encoded).hexdigest(),
                    "grid_target_histogram": dict(sorted(Counter(d for d, v in zip(grid, label) if v).items())),
                    "window_graph_pixel_histogram": dict(Counter(window[y // K, x // K] for y, x in pixel)),
                    "pixel_distance_histogram": dict(sorted(Counter(pixel.values()).items())),
                })
                images.append(list(zip(occ, seed_mask)))
                labels.append(label)
                pixels.append([pixel.get(p, -1) for p in POINTS])
                graphs.append([window.get((y // K, x // K), -1) for y, x in POINTS])
                grids.append(grid)
                ranks.append(rank)
                bends.append([int(p in pixel and p not in straight) for p in POINTS])
            records.append(row)
        datasets[split] = {
            "image": torch.tensor(images, dtype=torch.float32).reshape(-1, SIZE, SIZE, 2),
            "target": torch.tensor(labels, dtype=torch.bool).reshape(-1, SIZE, SIZE),
            **{name: torch.tensor(values, dtype=torch.int16).reshape(-1, SIZE, SIZE)
               for name, values in (("pixel", pixels), ("window", graphs), ("grid", grids), ("bend", bends))},
            "rank": torch.tensor(ranks, dtype=torch.float32),
        }
        manifest[split] = {"seed": spec["seed"], "family": spec["family"], "count": len(images),
                           "sha256": checksum.hexdigest(), "rejections": dict(rejected), "maps": records}
    return datasets, manifest
