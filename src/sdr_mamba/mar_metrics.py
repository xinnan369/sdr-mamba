"""Mass and spatial diagnostics for nonnegative cell masses."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


def mass_array(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected a finite nonnegative two-dimensional mass map")
    return values


@dataclass
class Mass:
    cells: np.ndarray
    count: float
    shape: np.ndarray | None

    @classmethod
    def from_cells(cls, cells):
        cells = mass_array(cells)
        count = float(cells.sum())
        if not math.isfinite(count):
            raise ValueError("Total mass exceeds FP64 range")
        return cls(cells, count, cells / count if count > 0 else None)


def retotal(prediction, total):
    if prediction.shape is None:
        return np.zeros_like(prediction.cells) if total == 0 else None
    return prediction.shape * total


def aggregate(cells, height, width):
    h, w = cells.shape
    if h % height or w % width:
        raise ValueError("Grid must evenly partition native cells")
    return cells.reshape(height, h // height, width, w // width).sum((1, 3))


@lru_cache(maxsize=16)
def projections(height, width):
    y, x = np.indices((height, width))
    return tuple((axis.ravel(), int(axis.max())) for axis in (x, y, x + y, x - y + height - 1))


def sliced_w1(delta, grid):
    coarse = aggregate(delta, *grid).ravel()
    distances = []
    for index, span in projections(*grid):
        if span:
            signed = np.bincount(index, weights=coarse, minlength=span + 1)
            distances.append(float(np.abs(signed.cumsum()[:-1]).sum() / span))
    return float(np.mean(distances)) if distances else 0.0


def background(target, radius):
    h, w = target.shape
    active = target > 0
    pad = np.pad(active, radius)
    near = np.zeros_like(active)
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            near |= pad[y:y+h, x:x+w]
    return ~near


def decomposition(truth, prediction, epsilon):
    c, n = prediction.count, truth.count
    if c == 0 or n == 0:
        count = c if n == 0 else math.inf
        shape = 0.0 if n == 0 else None
        return {"count_kl": count, "shape_kl": shape, "total_kl": count,
                "raw_shape_kl": shape, "raw_total_kl": count,
                "direct_smoothed_kl": count, "direct_raw_kl": count,
                "identity_error": 0.0 if math.isfinite(count) else None,
                "positive_gt_zero_pred_cells": int(np.count_nonzero(truth.cells))}
    if not 0 <= epsilon < 1:
        raise ValueError("Probability-floor mixture must lie in [0, 1)")
    positive = truth.cells > 0
    zero_hits = int(np.count_nonzero(prediction.shape[positive] == 0))
    q = truth.shape[positive]
    p = prediction.shape[positive]
    smooth = (1 - epsilon) * p + epsilon / prediction.cells.size
    count = (c - n) - n * (math.log(c) - math.log(n))
    raw_shape = math.inf if zero_hits else float(n * (q * (np.log(q) - np.log(p))).sum())
    shape = (float(n * (q * (np.log(q) - np.log(smooth))).sum())
             if np.all(smooth > 0) else math.inf)
    labels = truth.cells[positive]

    def direct(probability):
        if not np.all(probability > 0):
            return math.inf
        return float((labels * (np.log(labels) - math.log(c) - np.log(probability))).sum() + c - n)

    independent = direct(smooth)
    return {"count_kl": count, "shape_kl": shape, "total_kl": count + shape,
            "raw_shape_kl": raw_shape, "raw_total_kl": count + raw_shape,
            "direct_smoothed_kl": independent, "direct_raw_kl": direct(p),
            "identity_error": abs(independent - (count + shape)) if math.isfinite(independent) else None,
            "positive_gt_zero_pred_cells": zero_hits}


def measure(truth, pred, mask, config):
    delta = pred.cells - truth.cells
    error = pred.count - truth.count
    bg = float(pred.cells[mask].sum())
    result = {"gt": truth.count, "count": pred.count,
              "count_ratio": pred.count / truth.count if truth.count else None,
              "signed": error, "absolute": abs(error), "density_cell_mae": float(np.abs(delta).mean()),
              "background_mass": bg, "foreground_mass": pred.count - bg,
              "background_fraction": bg / pred.count if pred.count else None,
              "peak_mass": float(pred.cells.max()), "peak_p": None, "entropy": None,
              "shape_l1": None, "tv": None, "sw1_coarse": None, "sw1_fine": None, "sw1": None,
              "gt_weighted_sw1": None, "pred_weighted_sw1": None,
              "zero_gt": truth.count == 0, "zero_prediction": pred.count == 0}
    for level in (1, 2, 3):
        result[f"game{level}"] = float(np.abs(aggregate(delta, 2**level, 2**level)).sum())
    if pred.shape is not None:
        positive = pred.shape[pred.shape > 0]
        result.update(peak_p=float(pred.shape.max()), entropy=float(-(positive * np.log(positive)).sum()))
    if truth.shape is not None and pred.shape is not None:
        shape_delta = pred.shape - truth.shape
        l1 = float(np.abs(shape_delta).sum())
        coarse, fine = (sliced_w1(shape_delta, grid) for grid in config["projection_grids"])
        sw1 = (coarse + fine) / 2
        result.update(shape_l1=l1, tv=l1/2, sw1_coarse=coarse, sw1_fine=fine, sw1=sw1,
                      gt_weighted_sw1=truth.count*sw1, pred_weighted_sw1=pred.count*sw1)
    return result
