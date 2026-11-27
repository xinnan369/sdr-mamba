"""Verify MAR-001B identities, pair by image, and apply registered gates."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from scripts.audit_dataset import append_rows, digest_config, write_json
from scripts.eval_sdr005 import read_csv
from scripts.train_sdr003 import read_json

VARIANTS = ("A_reference", "B_fixed_mean", "C_fixed_centered", "D_observed", "fixed_mean_t0",
            "fixed_count", "matched_mean_and_rms", "trajectory_anchor")
METRICS = ("count", "signed", "absolute", "game1", "game2", "game3", "tv", "sw1",
           "count_kl", "shape_kl", "background_fraction", "background_mass", "foreground_mass", "density_cell_mae")
CONTRASTS = ((v, "A_reference") for v in VARIANTS if v not in ("A_reference", "trajectory_anchor"))
CONTRASTS = (*CONTRASTS, ("B_fixed_mean", "D_observed"), ("trajectory_anchor", "B_fixed_mean"))


def key(row):
    return row["mode"], row["id"], int(row["t"])


def values(rows, field):
    return np.array([float(r[field]) for r in rows])


def mean_metrics(rows, fields):
    return {k: float(np.mean([float(r[k]) for r in rows if r.get(k, "") != ""]))
            if any(r.get(k, "") != "" for r in rows) else None for k in fields}


def validate(output, config):
    manifest = read_json(output / "data_manifest.json")
    env = read_json(output / "readout_environment.json")
    assert env["dataset_sha256"] == manifest["sha256"] and env["readout_config_sha256"] == digest_config(config)
    ids = [r["id"] for r in manifest["entries"]]
    assert len(set(ids)) == len(ids) == 60
    expected = {(m, i, t) for m in config["models"] for i in ids for t in range(1, config["maximum"]+1)}
    depth = read_csv(output / "per_image_depth.csv")
    frozen = {key(r): float(r["count"]) for r in read_csv(Path("reports/MAR-001/per_image_depth.csv"))}
    assert {key(r) for r in depth} == expected == frozen.keys() and len(depth) == len(expected)
    historical_max = max(abs(float(r["count"])-frozen[key(r)]) for r in depth)
    historical_relative = max(abs(float(r["count"])-frozen[key(r)])/max(frozen[key(r)], 1e-12) for r in depth)
    observed = {key(r): float(r["count"]) for r in depth}
    counterfactual = read_csv(output / "counterfactual.csv")
    cube = {(*key(r), r["variant"]): r for r in counterfactual}
    assert len(cube) == len(counterfactual)
    assert cube.keys() == {(*k, v) for k in expected for v in VARIANTS
                           if v != "trajectory_anchor" or k[2] >= config["reference"]}
    shapley_error = 0.
    for m, i, t in expected:
        a, b, c, d = (cube[m, i, t, v] for v in VARIANTS[:4])
        assert float(d["count"]) == observed[m, i, t]
        ref = observed[m, i, config["reference"]]
        assert float(a["count"]) == ref
        fixed = cube[m, i, t, "fixed_count"]
        assert math.isclose(float(fixed["count"]), ref, abs_tol=1e-10)
        assert math.isclose(float(fixed["tv"]), float(d["tv"]), abs_tol=1e-12)
        av, bv, cv, dv = (float(r["count"]) for r in (a, b, c, d))
        assert math.isclose(float(d["mean_shapley"]), ((cv-av)+(dv-bv))/2, abs_tol=1e-10)
        assert math.isclose(float(d["centered_shapley"]), ((bv-av)+(dv-cv))/2, abs_tol=1e-10)
        residual = float(d["mean_shapley"]) + float(d["centered_shapley"]) - (dv-av)
        shapley_error = max(shapley_error, abs(residual))
    anchors = read_csv(output / "anchor_equivalence.csv")
    logit = read_csv(output / "logit_decomposition.csv")
    assert len(anchors) == len({key(r) for r in anchors}) == sum(k[2] >= config["reference"] for k in expected)
    assert len(logit) == len({(*key(r), r["region"]) for r in logit}) == 3*len(expected)
    eq = read_csv(output / "equivariance.csv")
    jvp = read_csv(output / "jvp.csv")
    spec = config["equivariance"]
    assert len(eq) == len(config["models"])*len(ids)*len(spec["amplitudes"])
    assert len(jvp) == len(config["models"])*spec["jvp_images"]*len(spec["amplitudes"])
    max_relative = {k: max(values(anchors, k)) for k in ("centered_relative_rms", "ln_input_relative_rms", "branch_update_relative_rms")}
    max_relative.update({k: max(values(eq, k)) for k in ("ln_relative_rms", "recurrent_relative_rms")})
    max_absolute = {
        "anchor_logit_error": float(max(values(anchors, "max_logit_abs"))),
        "split_logit_error": float(max(values(logit, "logit_split_max_abs_error"))),
        "anchor_count_relative": float(max(values(anchors, "count_relative"))),
        "jvp_relative": max((float(r["relative_rms"]) for r in jvp if r["relative_rms"]), default=None),
        "historical_count_abs": historical_max, "historical_count_relative": historical_relative,
        "shapley_count_error": shapley_error,
    }
    tol = config["tolerance"]
    heads = read_csv(output / "head_stats.csv")
    checks = {"ln_channel_axis": all(row["channels"] == row["ln_normalized_shape"] for row in heads),
              "historical_forward": historical_relative <= tol["fp32_count_relative"],
              "ln_and_step_equivariance": all(v <= tol["fp32_relative_rms"] for k, v in max_relative.items()),
              "anchored_readout_and_split": max(max_absolute["anchor_logit_error"], max_absolute["split_logit_error"]) <= tol["fp32_logit_absolute"],
              "anchor_count": max_absolute["anchor_count_relative"] <= tol["fp32_count_relative"],
              "neutral_jvp": all(r["status"] == "PASS" for r in jvp)
              and max_absolute["jvp_relative"] <= tol["fp32_jvp_relative"],
              "shapley_identity": shapley_error < tol["shapley_absolute"]}
    resources = read_csv(output / "resource_log.csv")
    assert len(resources) == len(config["models"]) and all(r["status"] == "PASS" for r in resources)
    assert sum(float(r["seconds"]) for r in resources) <= config["gpu_seconds_cap"]
    print(f"Verified {len(depth)} image-depth, {len(counterfactual)} counterfactual, {len(logit)} region and {len(anchors)} anchor rows")
    print("Engineering", checks, {**max_relative, **max_absolute})
    return ids, cube, checks, {**max_relative, **max_absolute}


def analyze(output, config, ids, cube, engineering, errors):
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    indices = rng.integers(len(ids), size=(config["bootstrap"]["replicates"], len(ids)))
    summaries, bootstrap, gates, bins = [], [], [], []
    for mode in config["models"]:
        for t in range(1, config["maximum"]+1):
            base = {"mode": mode, "t": t}
            group = {v: [cube[mode, i, t, v] for i in ids] for v in VARIANTS
                     if v != "trajectory_anchor" or t >= config["reference"]}
            for label, rows in group.items():
                summaries.append({**base, "variant": label, "images": len(rows),
                                  **mean_metrics(rows, METRICS + ("mean_shapley", "centered_shapley"))})
                for lo, hi in zip((0, 25, 50, 100, 200), (25, 50, 100, 200, math.inf)):
                    bin_rows = [x for x in rows if x["gt"] and lo <= float(x["gt"]) < hi]
                    if bin_rows:
                        bins.append({**base, "variant": label, "count_lower": lo, "count_upper": hi,
                                     "images": len(bin_rows), **mean_metrics(bin_rows, METRICS)})
            for name, ref in CONTRASTS:
                if name not in group:
                    continue
                pairs = [(a, b) for a, b in zip(group[name], group[ref]) if a["status"] == b["status"] == "DEFINED"]
                if not pairs:
                    continue
                first, second = zip(*pairs)
                take = indices if len(pairs) == len(ids) else rng.integers(len(pairs), size=(len(indices), len(pairs)))
                for field in ("signed", "absolute", "game1", "game2", "game3", "tv", "sw1", "count_kl", "shape_kl", "background_mass"):
                    delta = values(first, field)-values(second, field)
                    lo, hi = np.quantile(delta[take].mean(1), (.025,.975))
                    bootstrap.append({**base, "variant": name, "reference": ref, "metric": field,
                                      "images": len(pairs), "mean_delta": delta.mean(), "median_delta": np.median(delta),
                                      "ci95_low": lo, "ci95_high": hi})
            if t == config["primary_depth"]:
                a, b, d = (group[v] for v in ("A_reference", "B_fixed_mean", "D_observed"))
                before, after, original = (values(values_for_variant, "absolute") for values_for_variant in (a, b, d))
                scale = original.mean()-before.mean()
                removed = (original.mean()-after.mean())/scale if scale > 0 else None
                count_delta, mean_shapley, centered = (values(d, k) for k in ("count_delta", "mean_shapley", "centered_shapley"))
                positive = count_delta > 0
                share = mean_shapley[positive].sum()/count_delta[positive].sum() if positive.any() else None
                ratios = mean_shapley[positive]/count_delta[positive]
                g1_before = values(a, "game1").mean()
                gain = 1 - values(b, "game1").mean()/g1_before if g1_before else None
                tv_drop = float((values(a, "tv")-values(b, "tv")).mean())
                sw_drop = float((values(a, "sw1")-values(b, "sw1")).mean())
                limits = config["gate"]
                improved = float((after < original-1e-10).mean())
                count_pass = removed is not None and share is not None and (
                    removed >= limits["min_removed_mae_fraction"] and improved >= limits["min_improved_image_fraction"]
                    and share >= limits["min_mean_shapley_on_positive_excess"])
                spatial = gain is not None and gain >= limits["min_game1_gain"] and (
                    tv_drop >= limits["min_tv_gain"] or sw_drop >= limits["min_sw1_gain"])
                gates.append({**base, "mae_reference": float(before.mean()), "mae_anchor": float(after.mean()),
                              "mae_observed": float(original.mean()), "removed_mae_fraction": float(removed) if removed is not None else None,
                              "improved_images_fraction": improved,
                              "mean_share_positive_excess": float(share) if share is not None else None,
                              "positive_excess_images": int(positive.sum()),
                              "positive_mean_contribution_images": int((mean_shapley > 0).sum()),
                              "positive_centered_contribution_images": int((centered > 0).sum()),
                              "mean_share_p25_median_p75": np.quantile(ratios, (.25,.5,.75)).tolist() if len(ratios) else None,
                              "game1_gain": float(gain) if gain is not None else None, "tv_drop": tv_drop, "sw1_drop": sw_drop,
                              "count_gate": bool(count_pass), "spatial_gate": bool(spatial)})
                vectors = {"removed_mae_fraction": (original-after, original-before),
                           "mean_share_positive_excess": (np.where(positive, mean_shapley, 0), np.where(positive, count_delta, 0)),
                           "improved_images_fraction": ((after < original-1e-10).astype(float), np.ones(len(ids)))}
                for metric, (num, den) in vectors.items():
                    denom = den[indices].sum(1)
                    valid = denom > 0
                    ci = np.quantile(num[indices].sum(1)[valid]/denom[valid], (.025,.975)) if valid.any() else (None,None)
                    bootstrap.append({**base, "variant": "B_fixed_mean", "reference": "D_observed", "metric": metric,
                                      "images": len(ids), "mean_delta": num.sum()/den.sum() if den.sum() > 0 else None,
                                      "median_delta": None, "ci95_low": ci[0], "ci95_high": ci[1]})
    for name, rows in (("summary", summaries), ("bootstrap", bootstrap), ("count_bins", bins)):
        append_rows(output / f"{name}.csv", rows)
    primary = next(row for row in gates if row["mode"] == config["primary_mode"])
    engineering_status = all(engineering.values())
    result = {"status": ("PASS" if primary["count_gate"] and primary["spatial_gate"] else "FAIL") if engineering_status else "BLOCKED",
              "engineering": "PASS" if engineering_status else "FAIL", "engineering_checks": engineering, "max_errors": errors,
              "count_mechanism": "PASS" if primary["count_gate"] else "FAIL",
              "spatial_scaling": "PASS" if primary["spatial_gate"] else "FAIL",
              "seed": 0, "train": "NOT_RUN", "confirmation": "NOT_RUN", "official_test": "NOT_RUN", "T8": gates}
    write_json(output / "gate.json", result)
    print("Gate", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "gate.json").exists():
        raise FileExistsError("Analysis already completed")
    config = read_json(Path("configs/mar001b.json"))
    analyze(args.output, config, *validate(args.output, config))


if __name__ == "__main__":
    main()
