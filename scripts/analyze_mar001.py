"""Validate MAR-001 evidence, compute paired intervals and apply frozen gates."""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path

import numpy as np

from scripts.audit_dataset import append_rows, digest_config, write_json
from scripts.eval_sdr005 import read_csv
from scripts.train_sdr003 import read_json


METRICS = ("count_ratio", "signed", "absolute", "game1", "game2", "game3", "density_cell_mae", "tv",
           "sw1_coarse", "sw1_fine", "sw1", "gt_weighted_sw1", "background_mass", "background_fraction",
           "peak_mass", "peak_p", "entropy", "feature_rms", "centered_feature_rms", "channel_centered_feature_rms",
           "update_rms", "update_relative_to_input_rms", "feature_cosine", "count_update", "density_update_l1",
           "count_kl", "shape_kl", "total_kl")


def key(row):
    return row["mode"], row["id"], int(row["t"])


def averages(records, fields):
    result = {}
    for field in fields:
        values = [float(r[field]) for r in records if r.get(field, "") != ""]
        result[field] = float(np.mean(values)) if values else None
    return result


def validate_and_join(output, config):
    manifest = read_json(output / "data_manifest.json")
    ids = {r["id"]: r["full_count"] for r in manifest["entries"]}
    perturbation = read_json(output / "perturbation_manifest.json")
    assert manifest["split"] == config["split"] and len(ids) == 60
    assert set(perturbation["images"]) <= ids.keys()
    for stage in ("depth", "perturbation"):
        env = read_json(output / f"{stage}_environment.json")
        assert env["config_sha256"] == digest_config(config) and env["dataset_sha256"] == manifest["sha256"]
    expected = {(mode, ident, t) for mode in config["models"] for ident in ids for t in range(1, config["depths"]["maximum"]+1)}
    depth = read_csv(output / "per_image_depth.csv")
    decomp = read_csv(output / "decomposition.csv")
    cf = read_csv(output / "counterfactual.csv")
    source, terms, alternatives = {key(r): r for r in depth}, {key(r): r for r in decomp}, {}
    assert len(source) == len(depth) == len(terms) == len(decomp) == len(expected)
    assert source.keys() == terms.keys() == expected
    assert all(float(r["gt"]) == ids[r["id"]] for r in depth)
    for r in cf:
        alternatives[(*key(r), r["variant"])] = r
    variants = ("observed", "reference", "fixed_mass", "fixed_shape", "oracle_mass")
    assert set(alternatives) == {(*k, v) for k in expected for v in variants} and len(cf) == len(alternatives)
    ref = config["depths"]["reference"]
    maximum_absolute_identity, maximum_relative_identity = 0.0, 0.0
    for k, observed in source.items():
        mode, identifier, _ = k
        term = terms[k]
        assert float(observed["count"]) == float(term["count"])
        n, c = float(observed["gt"]), float(observed["count"])
        assert math.isclose(float(observed["signed"]), c-n)
        assert math.isclose(float(observed["foreground_mass"]) + float(observed["background_mass"]), c)
        if term["identity_error"]:
            error = float(term["identity_error"])
            relative = error / max(1.0, float(term["total_kl"]))
            assert relative < 1e-12
            maximum_absolute_identity = max(maximum_absolute_identity, error)
            maximum_relative_identity = max(maximum_relative_identity, relative)
        for label, goal in (("fixed_mass", float(source[mode, identifier, ref]["count"])),
                            ("fixed_shape", c), ("oracle_mass", n)):
            alt = alternatives[(*k, label)]
            if alt["status"] == "DEFINED":
                assert math.isclose(float(alt["count"]), goal, abs_tol=1e-10)
        if observed["tv"]:
            for label in ("fixed_mass", "oracle_mass"):
                alt = alternatives[(*k, label)]
                if alt["tv"]:
                    assert math.isclose(float(alt["tv"]), float(observed["tv"]), abs_tol=1e-13)
        source[k].update({name: term[name] for name in ("count_kl", "shape_kl", "total_kl")})
    print(f"Verified {len(depth)} observations, {len(decomp)} decompositions, {len(cf)} counterfactuals")
    print("Maximum finite FP64 identity error", maximum_absolute_identity, "relative", maximum_relative_identity)
    return source, alternatives, ids


def summarize(output, config, source, alternatives, ids):
    depth_params = config["depths"]
    ref = depth_params["reference"]
    bins = [*config["count_bins"], math.inf]
    depth_rows, by_bin, cf_rows, step_rows = [], [], [], []
    for mode in config["models"]:
        reference = [source[mode, ident, ref] for ident in ids]
        for t in range(1, depth_params["maximum"]+1):
            rows = [source[mode, ident, t] for ident in ids]
            meta = {"mode": mode, "t": t, "images": len(rows)}
            depth_rows.append({**meta, **averages(rows, METRICS)})
            for lo, hi in zip(bins, bins[1:]):
                selected = [r for r in rows if lo <= float(r["gt"]) < hi]
                if selected:
                    by_bin.append({**meta, "images": len(selected), "gt_lower": lo, "gt_upper": hi,
                                   **averages(selected, METRICS)})
            for variant in ("reference", "observed", "fixed_mass", "fixed_shape", "oracle_mass"):
                cf = [alternatives[mode, ident, t, variant] for ident in ids]
                cf_rows.append({**meta, "variant": variant, "defined": sum(r["status"] == "DEFINED" for r in cf),
                                **averages(cf, METRICS[:15] + ("count_kl", "shape_kl", "total_kl"))})
            if t > 1:
                previous = [source[mode, ident, t-1] for ident in ids]
                for field in ("absolute", "game1", "tv", "sw1", "total_kl"):
                    delta = [float(x[field])-float(y[field]) for x, y in zip(rows, previous) if x[field] and y[field]]
                    if delta:
                        step_rows.append({**meta, "metric": field, "paired": len(delta), "mean_delta": float(np.mean(delta)),
                                          "improved": sum(x < -1e-10 for x in delta), "worsened": sum(x > 1e-10 for x in delta)})
        assert len(reference) == len(ids)
    for name, rows in (("depth_summary", depth_rows), ("count_bins", by_bin), ("counterfactual_summary", cf_rows),
                       ("step_changes", step_rows)):
        append_rows(output / f"{name}.csv", rows)


def gate_report(output, config, source, alternatives, ids):
    ref = config["depths"]["reference"]
    scope = config["depths"]["gate"]
    plan = config["gate"]
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    take = rng.integers(len(ids), size=(config["bootstrap"]["replicates"], len(ids)))
    results, intervals = [], []
    for mode in config["models"]:
        for t in scope:
            paired = [(source[mode, ident, t], source[mode, ident, ref]) for ident in ids]
            deltas = {m: np.array([float(a[m])-float(b[m]) for a, b in paired])
                      for m in ("absolute", "total_kl", "count_kl", "shape_kl", "tv", "sw1", "game1")}
            d_fixed = np.array([float(alternatives[mode, ident, t, "fixed_mass"]["game1"])
                                -float(source[mode, ident, ref]["game1"]) for ident in ids])
            deltas["fixed_mass_game1"] = d_fixed
            worsened = deltas["total_kl"] > 0
            dominance = (deltas["count_kl"] > deltas["shape_kl"]) & (deltas["count_kl"] > 0)
            condition = float(dominance[worsened].mean()) if worsened.any() else 0.
            baseline = np.mean([float(b["game1"]) for _, b in paired])
            row = {"mode": mode, "t": t, "count_worsened_fraction": float((deltas["absolute"] > 0).mean()),
                   "deviance_worsened_fraction": float(worsened.mean()), "count_dominance_fraction": condition,
                   "mean_tv_delta": float(deltas["tv"].mean()), "mean_sw1_delta": float(deltas["sw1"].mean()),
                   "relative_fixed_mass_game1_improvement": float(-d_fixed.mean()/baseline)}
            row["count_drift"] = (row["count_worsened_fraction"] >= plan["worsening_image_fraction"]
                                  and row["deviance_worsened_fraction"] >= plan["worsening_image_fraction"]
                                  and condition >= plan["count_dominance_fraction"])
            row["shape_preserved"] = row["mean_tv_delta"] <= plan["max_tv_increase"] and row["mean_sw1_delta"] <= plan["max_sw1_increase"]
            row["spatial_gain"] = (row["relative_fixed_mass_game1_improvement"] >= plan["fixed_mass_game1_gain"]
                                   and (row["mean_tv_delta"] <= -plan["min_tv_gain"]
                                        or row["mean_sw1_delta"] <= -plan["min_sw1_gain"]))
            results.append(row)
            for metric, delta in deltas.items():
                lo, hi = np.quantile(delta[take].mean(1), (.025, .975))
                q25, med, q75 = np.quantile(delta, (.25, .50, .75))
                intervals.append({"mode": mode, "t": t, "metric": metric, "mean_delta": delta.mean(),
                                  "p25": q25, "median": med, "p75": q75,
                                  "ci95_low": lo, "ci95_high": hi, "images": len(ids)})
    primary = [r for r in results if r["mode"] in config["primary_models"]]
    checks = {"cross_backbone_count_drift": all(r["count_drift"] for r in primary),
              "shape_preserved": all(r["shape_preserved"] for r in primary),
              "spatial_gain_T8": all(r["spatial_gain"] for r in primary if r["t"] == 8)}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "comparison": results,
              "scope": config["split"], "training": "NOT_RUN", "confirmation": "NOT_RUN", "official_test": "NOT_RUN"}
    write_json(output / "gate.json", result)
    append_rows(output / "paired_bootstrap.csv", intervals)
    print("Gate", result["status"], checks)


def perturbation_summary(output, config):
    ledger = read_csv(output / "perturbation.csv")
    by_key = defaultdict(list)
    spec = config["perturbation"]
    ref = config["depths"]["reference"]
    for r in ledger:
        by_key[r["mode"], r["direction"], r["relative_rms"], r["horizon"]].append(r)
        assert math.isclose(float(r["feature_gain"]), float(r["rms_delta"])/float(r["rms_initial"]))
        assert int(r["start_t"]) == ref and math.isfinite(float(r["feature_gain"]))
    expected_variants = 4 + 2*spec["draws"]
    assert len(ledger) == len(config["models"])*spec["images"]*expected_variants*len(spec["relative_rms"])*len(spec["horizons"])
    for r in ledger:
        r["abs_count_delta"] = abs(float(r["signed_count_delta"]))
    summary = []
    for (mode, direction, relative, horizon), rows in by_key.items():
        gain = [float(r["feature_gain"]) for r in rows]
        summary.append({"mode": mode, "direction": direction, "relative_rms": relative, "horizon": horizon,
                        "n": len(rows), **averages(rows, ("feature_gain", "relative_trajectory_offset",
                                                         "signed_count_delta", "abs_count_delta", "density_l1")),
                        "median_gain": float(np.median(gain)), "max_gain": max(gain)})
    append_rows(output / "perturbation_summary.csv", summary)
    jvp = read_csv(output / "jvp.csv")
    assert len(jvp) == len(config["models"])*spec["jvp_images"]*2*2
    print("Perturbation rows", len(ledger), "JVP", dict(Counter(r["status"] for r in jvp)))
    resources = read_csv(output / "resource_log.csv")
    assert {(r["phase"], r["model"]) for r in resources} == {(p, m) for p in ("depth", "perturbation") for m in config["models"]}
    assert all(r["status"] == "PASS" for r in resources)
    seconds = sum(float(r["seconds"]) for r in resources)
    assert seconds <= config["gpu_seconds_cap"]
    print("Tracked GPU-stage seconds", seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "gate.json").exists():
        raise FileExistsError("Use a fresh directory for repeated analysis")
    config = read_json(Path("configs/mar001.json"))
    source, alternatives, ids = validate_and_join(args.output, config)
    summarize(args.output, config, source, alternatives, ids)
    perturbation_summary(args.output, config)
    gate_report(args.output, config, source, alternatives, ids)


if __name__ == "__main__":
    main()
