"""ShanghaiTech image-level split, auditable point transforms and exact mass targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class CrowdSample:
    identifier: str
    file: Path
    width: int
    height: int
    points: np.ndarray
    image_sha: str
    annotation_sha: str
    discarded: int


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_points(path):
    from scipy.io import loadmat

    info = loadmat(path, struct_as_record=False)["image_info"]
    while isinstance(info, np.ndarray) and info.size == 1:
        info = info.item()
    points = np.asarray(info.location)
    if not points.size:
        points = points.reshape(0, 2)
    if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in "iuf":
        raise ValueError("ShanghaiTech annotation must be numeric N×2")
    if np.asarray(info.number).size != 1 or np.asarray(info.number).item() != len(points):
        raise ValueError("ShanghaiTech point and count metadata differ")
    if not np.isfinite(points).all():
        raise ValueError("Non-finite ShanghaiTech coordinates")
    return points.astype(np.float64)


def in_rect(points, width, height):
    return (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)


def list_split(root, directory, expected, part, coordinate_base):
    base = root / directory
    images = {p.stem: p for p in (base / "images").glob("*.jpg")}
    points = {p.stem[3:]: p for p in (base / "ground_truth").glob("GT_*.mat")}
    if len(images) != expected or images.keys() != points.keys():
        raise ValueError(f"{directory}: expected {expected} exact image/annotation pairs; "
                         f"images={len(images)} annotations={len(points)}")
    records = []
    for name, file in sorted(images.items()):
        anonymous = hashlib.sha256(f"sdr005:{part}:{directory}:{name}".encode()).hexdigest()[:16]
        try:
            with Image.open(file) as image:
                width, height = image.size
                image.verify()
            coords = parse_points(points[name]) - coordinate_base
        except Exception as error:
            raise ValueError(f"Invalid image/annotation for anonymous record {anonymous}") from error
        selected = in_rect(coords, width, height)
        records.append(CrowdSample(
            anonymous, file, width, height, coords[selected],
            sha_file(file), sha_file(points[name]), int((~selected).sum()),
        ))
    return records


def load_dataset(root, config):
    part = config["part"]
    sizes = {"A": (300, 182), "B": (400, 316)}[part]
    train = list_split(root, "train_data", sizes[0], part, config["coordinate_base"])
    test = list_split(root, "test_data", sizes[1], part, config["coordinate_base"])
    image_hashes = {s.image_sha for s in (*train, *test)}
    if len(image_hashes) != len(train) + len(test):
        raise ValueError("Duplicate image bytes in the complete official partition")
    length = len(train)
    if config["validation_fraction"] != .15 or length % 20:
        raise ValueError("SDR-005 uses exactly 15% from count-sorted blocks of 20")
    ranked = sorted(train, key=lambda s: (len(s.points), s.identifier))
    val_ids = set()
    for start in range(0, length, 20):
        local = ranked[start:start + 20]
        local.sort(key=lambda s: hashlib.sha256(f"{config['split_seed']}:{s.identifier}".encode()).digest())
        val_ids.update(s.identifier for s in local[:3])
    result = {
        "train": [s for s in train if s.identifier not in val_ids],
        "validation": [s for s in train if s.identifier in val_ids],
        "test": test,
    }
    manifest, details = {}, {}
    for split, samples in result.items():
        entries = [{"id": s.identifier, "image_sha256": s.image_sha, "annotation_sha256": s.annotation_sha,
                    "width": s.width, "height": s.height, "count": len(s.points),
                    "out_of_bounds_dropped": s.discarded} for s in samples]
        digest = hashlib.sha256()
        for entry in entries:
            digest.update(f"{entry['id']}:{entry['image_sha256']}:{entry['annotation_sha256']}\n".encode())
        manifest[split] = {"sha256": digest.hexdigest(), "entries": entries}
        details[split] = {
            "images": len(samples), "head_count": sum(len(s.points) for s in samples),
            "empty_after_filtering": sum(len(s.points) == 0 for s in samples),
            "out_of_bounds_dropped": sum(s.discarded for s in samples),
            "affected_images": sum(s.discarded != 0 for s in samples),
            "dimensions": dict(sorted(Counter(f"{s.width}x{s.height}" for s in samples).items())),
            "on_valid_edge": sum(int(((s.points[:, 0] == 0) | (s.points[:, 1] == 0)
                                     | (s.points[:, 0] == s.width - 1) | (s.points[:, 1] == s.height - 1)).sum())
                                 for s in samples),
        }
    return result, manifest, {"part": part, "coordinate_base": config["coordinate_base"], "status": "PASS",
                               "out_of_bounds_policy": "Drop finite coordinates outside the half-open rectangle",
                               "splits": details}


def transform_points(points, top, left, height, width, flip=False, output_size=None):
    local = points - np.asarray((left, top))
    local = local[in_rect(local, width, height)].copy()
    if flip:
        # Reflect within each image's pixel grid and keep fractional sub-pixel position.
        x = local[:, 0]
        local[:, 0] = width - 1 - np.floor(x) + (x - np.floor(x))
    if output_size is not None:
        out_h, out_w = output_size
        local *= (out_w / width, out_h / height)
        local[:, 0] = np.minimum(local[:, 0], np.nextafter(float(out_w), 0))
        local[:, 1] = np.minimum(local[:, 1], np.nextafter(float(out_h), 0))
    return local


def mass_map(points, height, width, stride):
    if not in_rect(points, width, height).all():
        raise ValueError("Trying to bin out-of-frame points")
    shape = ((height + stride - 1) // stride, (width + stride - 1) // stride)
    indices = np.floor(points / stride).astype(np.int64)
    density = np.bincount(indices[:, 1] * shape[1] + indices[:, 0], minlength=shape[0] * shape[1])
    return torch.from_numpy(density.astype(np.float32).reshape(shape))


@lru_cache(maxsize=400)
def decode(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def tensor_image(array, config):
    values = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float().div_(255)
    mean = torch.tensor(config["normalization"]["mean"])[:, None, None]
    std = torch.tensor(config["normalization"]["std"])[:, None, None]
    return (values - mean) / std


def training_crop(sample, config, rng, fixed=False):
    crop = config["crop"]
    h, w = min(crop, sample.height), min(crop, sample.width)
    top = (sample.height - h) // 2 if fixed else rng.randrange(sample.height - h + 1)
    left = (sample.width - w) // 2 if fixed else rng.randrange(sample.width - w + 1)
    flip = not fixed and rng.random() < .5
    tile = decode(sample.file)[top:top + h, left:left + w]
    if flip:
        tile = tile[:, ::-1]
    points = transform_points(sample.points, top, left, h, w, flip)
    target = mass_map(points, h, w, config["stride"])
    return tensor_image(tile, config), target, {
        "id": sample.identifier, "top": top, "left": left, "height": h, "width": w,
        "flipped": flip, "image_points": len(sample.points), "crop_points": len(points),
        "excluded_by_crop": len(sample.points) - len(points),
    }
