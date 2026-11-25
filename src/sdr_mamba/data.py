"""Deterministic synthetic data with independent four-connected BFS labels."""

from collections import Counter, deque
import hashlib
import random

import torch


def component(walkable, seed, size):
    distance = [-1] * (size ** 2)
    if not walkable[seed]:
        raise ValueError("Seed must be on a walkable pixel")
    distance[seed] = 0
    queue = deque([seed])
    while queue:
        index = queue.popleft()
        y, x = divmod(index, size)
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < size and 0 <= xx < size:
                adjacent = yy * size + xx
                if walkable[adjacent] and distance[adjacent] == -1:
                    distance[adjacent] = distance[index] + 1
                    queue.append(adjacent)
    return [int(d >= 0) for d in distance]


def window_distances(seed, size, k):
    y, x = divmod(seed, size)
    return [
        abs(i // size // k - y // k) + abs(i % size // k - x // k)
        for i in range(size ** 2)
    ]


def generate(config):
    size, k = config["size"], config["model"]["k"]
    splits, manifest = {}, {}
    seen = set()
    for split in ("train", "validation", "test"):
        spec = config["data"][split]
        rng = random.Random(spec["seed"])
        inputs, targets, distances, records = [], [], [], []
        histogram = Counter()
        digest = hashlib.sha256()
        attempts = 0
        while len(records) < spec["count"]:
            attempts += 1
            if attempts > spec["count"] * 1000:
                raise RuntimeError(f"Dataset constraints exhausted for {split}")
            p = rng.uniform(0.53, 0.68)
            walkable = [int(rng.random() < p) for _ in range(size ** 2)]
            far, near = False, False
            seed = rng.randrange(size ** 2)
            if not walkable[seed]:
                continue
            target = component(walkable, seed, size)
            if not 12 <= sum(target) <= size ** 2 * 0.70 or sum(walkable) - sum(target) < 8:
                continue
            distance = window_distances(seed, size, k)
            for d, positive in zip(distance, target):
                if positive:
                    near |= 0 < d < 4
                    far |= d >= 4
            if not near or not far:
                continue
            map_hash = hashlib.sha256(bytes(walkable)).hexdigest()
            if map_hash in seen:
                continue
            seen.add(map_hash)
            seed_mask = [int(i == seed) for i in range(size ** 2)]
            payload = bytes(walkable + seed_mask + target)
            digest.update(payload)
            inputs.append(list(zip(walkable, seed_mask)))
            targets.append(target)
            distances.append(distance)
            histogram.update(d for d, positive in zip(distance, target) if positive)
            records.append({"map_sha256": map_hash, "sample_sha256": hashlib.sha256(payload).hexdigest(),
                            "target_pixels": sum(target), "distractor_pixels": sum(walkable) - sum(target)})
        count = len(records)
        splits[split] = (
            torch.tensor(inputs, dtype=torch.float32).reshape(count, size, size, 2),
            torch.tensor(targets, dtype=torch.float32).reshape(count, size, size),
            torch.tensor(distances).reshape(count, size, size),
        )
        manifest[split] = {
            "seed": spec["seed"], "count": count, "attempts": attempts, "sha256": digest.hexdigest(),
            "target_window_distance_histogram": dict(sorted(histogram.items())), "samples": records,
        }
    return splits, manifest
