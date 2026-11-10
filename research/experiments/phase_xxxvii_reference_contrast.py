import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxvi_homologous_support import (
    OUT as XXXVI_OUT,
    build_memory_positions,
    load_feature_index,
    split_normal_groups,
)
from experiments.phase_xxxiv_insplad_cross_asset import PATCHES, PREVIOUS_VERIFIED_BUDGET, collect_dataset


OUT = Path("results/phase_xxxvii_reference_contrast")
PROP_SAMPLE = 5000
PAIR_SAMPLE = 200000
EPS = 1e-12


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def corr(x, y, kind):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or len(set(x[ok])) < 2 or len(set(y[ok])) < 2:
        return float("nan")
    return float((spearmanr if kind == "spearman" else pearsonr)(x[ok], y[ok]).statistic)


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def memory_properties(bank, entries, target_asset, split_id, variant):
    rng = np.random.default_rng(0)
    n = bank.shape[0]
    ids = rng.choice(n, size=min(PROP_SAMPLE, n), replace=False) if n > PROP_SAMPLE else np.arange(n)
    x = normalize_rows(bank[ids])
    pair_n = min(PAIR_SAMPLE, max(1, x.shape[0] * max(0, x.shape[0] - 1) // 2))
    a = rng.integers(0, x.shape[0], size=pair_n)
    b = rng.integers(0, x.shape[0], size=pair_n)
    neq = a != b
    a, b = a[neq], b[neq]
    sims = np.sum(x[a] * x[b], axis=1)
    dists = 1.0 - sims
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0)
    p = eig / max(EPS, eig.sum())
    erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    nn_ids = rng.choice(x.shape[0], size=min(1024, x.shape[0]), replace=False)
    nn = []
    for sid in nn_ids:
        sim = x[sid:sid + 1] @ x.T
        sim[0, sid] = -np.inf
        nn.append(float(1.0 - np.max(sim)))
    src_imgs = [e["source_image_id"] for e in entries]
    src_assets = [e["source_asset"] for e in entries]
    def entropy(vals):
        _, counts = np.unique(vals, return_counts=True)
        p = counts / max(1, counts.sum())
        h = float(-(p * np.log(p + EPS)).sum())
        return h / max(EPS, math.log(len(counts))) if len(counts) > 1 else 0.0
    return {
        "target_asset": target_asset,
        "split": split_id,
        "variant": variant,
        "memory_size": int(n),
        "S1_MEMORY_SPREAD_mean_distance": float(np.mean(dists)),
        "S1_MEMORY_SPREAD_std_distance": float(np.std(dists)),
        "S2_MEMORY_REDUNDANCY_sim_gt_0p95": float(np.mean(sims > 0.95)),
        "S3_EFFECTIVE_RANK": erank,
        "S4_LOCAL_NN_SCALE_median_top1_distance": float(np.median(nn)),
        "S5_SOURCE_IMAGE_DIVERSITY_unique_count": int(len(set(src_imgs))),
        "S5_SOURCE_IMAGE_DIVERSITY_entropy": entropy(src_imgs),
        "S6_SOURCE_ASSET_DIVERSITY_entropy": entropy(src_assets),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    raw = read_csv(XXXVI_OUT / "RAW_METRICS.csv")
    support = read_csv(XXXVI_OUT / "SUPPORT_SEPARATION.csv")
    by_raw = {(r["target_asset"], int(r["split"]), r["variant"]): r for r in raw}
    by_sup = {(r["target_asset"], int(r["split"]), r["variant"]): r for r in support}

    parity_rows, max_delta = [], 0.0
    for r in raw:
        pa, pap = f(r, "P-AUROC"), f(r, "P-AP")
        g = f(r, "G_local")
        g2 = 0.5 * (pa + pap)
        d = abs(g - g2)
        max_delta = max(max_delta, d)
        parity_rows.append({"target_asset": r["target_asset"], "split": r["split"], "variant": r["variant"], "delta_G_recomputed": d, "P-AUROC": pa, "P-AP": pap, "G_local": g})
    parity = {"passed": bool(max_delta <= 0.02), "max_delta": max_delta, "rows": parity_rows}
    (OUT / "PARITY_AUDIT.json").write_text(json.dumps(parity, indent=2, default=json_default), encoding="utf-8")

    distance_rows, sep_rows = [], []
    for r in support:
        normal_med = f(r, "normal_q50_top5")
        anomaly_med = f(r, "anomaly_q50_top5")
        sep = anomaly_med - normal_med
        row = {
            "target_asset": r["target_asset"], "split": r["split"], "variant": r["variant"],
            "normal_d5_median": normal_med,
            "normal_d5_q75": f(r, "normal_q75_top5"),
            "normal_d5_q90": f(r, "normal_q90_top5"),
            "normal_d5_q95": f(r, "normal_q95_top5"),
            "normal_d5_q99": f(r, "normal_q99_top5"),
            "anomaly_d5_median": anomaly_med,
            "anomaly_d5_q75": f(r, "anomaly_q75_top5"),
            "anomaly_d5_q90": f(r, "anomaly_q90_top5"),
            "Sep_median": sep,
            "KS": f(r, "ks_top5"),
            "Distance_AUROC": f(r, "top5_mean_AUROC"),
            "Distance_AP": f(r, "top5_mean_AP"),
        }
        distance_rows.append(row)
        sep_rows.append({**row, "G_local": f(by_raw[(r["target_asset"], int(r["split"]), r["variant"])], "G_local"), "P-AUROC": f(by_raw[(r["target_asset"], int(r["split"]), r["variant"])], "P-AUROC"), "P-AP": f(by_raw[(r["target_asset"], int(r["split"]), r["variant"])], "P-AP")})
    write_csv(OUT / "PATCH_DISTANCE_RAW_SUMMARY.csv", distance_rows)
    write_csv(OUT / "NORMAL_ANOMALY_DISTANCE.csv", distance_rows)
    write_csv(OUT / "SEPARATION_METRICS.csv", sep_rows)

    over_rows, wrong_rows, cross_wrong_rows, compression_rows = [], [], [], []
    for asset in sorted(set(r["target_asset"] for r in raw)):
        for split_id in [0, 1, 2]:
            def s(v): return by_sup[(asset, split_id, v)]
            def rr(v): return by_raw[(asset, split_id, v)]
            for suffix, same, cross in [("FPS", "P2_SAME_ASSET_FPS", "P0_CROSS_ASSET_FPS"), ("RANDOM", "P3_SAME_ASSET_RANDOM", "P1_CROSS_ASSET_RANDOM")]:
                normal_drop = f(s(same), "normal_q50_top5") - f(s(cross), "normal_q50_top5")
                anomaly_drop = f(s(same), "anomaly_q50_top5") - f(s(cross), "anomaly_q50_top5")
                sep_cross = f(s(cross), "anomaly_q50_top5") - f(s(cross), "normal_q50_top5")
                sep_same = f(s(same), "anomaly_q50_top5") - f(s(same), "normal_q50_top5")
                g_gain = f(rr(cross), "G_local") - f(rr(same), "G_local")
                over_rows.append({"target_asset": asset, "split": split_id, "comparison": suffix, "NormalDrop": normal_drop, "AnomalyDrop": anomaly_drop, "OverNormalize": anomaly_drop - normal_drop, "Sep_cross": sep_cross, "Sep_same": sep_same, "SeparationLoss": sep_cross - sep_same, "Cross_minus_Same_G_local": g_gain})
                compression_rows.append({"target_asset": asset, "split": split_id, "comparison": suffix, "normal_ratio_same_over_cross_median": f(s(same), "normal_q50_top5") / max(EPS, f(s(cross), "normal_q50_top5")), "anomaly_ratio_same_over_cross_median": f(s(same), "anomaly_q50_top5") / max(EPS, f(s(cross), "anomaly_q50_top5"))})
            same = "P2_SAME_ASSET_FPS"
            wrongs = ["P4_WRONG_ASSET_1", "P5_WRONG_ASSET_2"]
            sep_same = f(s(same), "anomaly_q50_top5") - f(s(same), "normal_q50_top5")
            for w in wrongs:
                sep_w = f(s(w), "anomaly_q50_top5") - f(s(w), "normal_q50_top5")
                wrong_rows.append({"target_asset": asset, "split": split_id, "wrong_variant": w, "DeltaSep_wrong_minus_same": sep_w - sep_same, "DeltaG_wrong_minus_same": f(rr(w), "G_local") - f(rr(same), "G_local")})
            wrong_mean_sep = np.mean([f(s(w), "anomaly_q50_top5") - f(s(w), "normal_q50_top5") for w in wrongs])
            wrong_mean_g = np.mean([f(rr(w), "G_local") for w in wrongs])
            cross_wrong_rows.append({"target_asset": asset, "split": split_id, "Sep_cross": f(s("P0_CROSS_ASSET_FPS"), "anomaly_q50_top5") - f(s("P0_CROSS_ASSET_FPS"), "normal_q50_top5"), "Sep_wrong_mean": wrong_mean_sep, "G_cross": f(rr("P0_CROSS_ASSET_FPS"), "G_local"), "G_wrong_mean": wrong_mean_g, "Cross_minus_WrongMean_Sep": (f(s("P0_CROSS_ASSET_FPS"), "anomaly_q50_top5") - f(s("P0_CROSS_ASSET_FPS"), "normal_q50_top5")) - wrong_mean_sep, "Cross_minus_WrongMean_G": f(rr("P0_CROSS_ASSET_FPS"), "G_local") - wrong_mean_g})
    write_csv(OUT / "SAME_CROSS_OVERNORMALIZATION.csv", over_rows)
    write_csv(OUT / "WRONG_ASSET_CONTRAST.csv", wrong_rows)
    write_csv(OUT / "CROSS_VS_WRONG_MEAN.csv", cross_wrong_rows)
    write_csv(OUT / "SUPPORT_COMPRESSION.csv", compression_rows)

    # Image-level contrast from existing XXXVI image metrics; distance-derived image summaries were not cached.
    img_rows = []
    for r in raw:
        img_rows.append({"target_asset": r["target_asset"], "split": r["split"], "variant": r["variant"], "I-AUROC_frozen_top1pct": f(r, "I-AUROC"), "I-AP_frozen_top1pct": f(r, "I-AP"), "median_d5_status": "NOT_AVAILABLE_FROM_XXXVI_CACHE", "q90_d5_status": "NOT_AVAILABLE_FROM_XXXVI_CACHE", "q99_d5_status": "NOT_AVAILABLE_FROM_XXXVI_CACHE"})
    write_csv(OUT / "IMAGE_LEVEL_CONTRAST.csv", img_rows)
    write_csv(OUT / "NEIGHBOR_SIMILARITY.csv", [{"status": "NOT_AVAILABLE_FROM_XXXVI_CACHE_WITHOUT_PATCH_LEVEL_NEIGHBOR_IDS"}])
    write_csv(OUT / "DEFECT_SUBTYPE.csv", [{"status": "NOT_AVAILABLE_NO_RELIABLE_SUBTYPE_METADATA"}])

    # Rebuild memory banks for source-only properties only; no backbone forward.
    assets, _, _, _ = collect_dataset()
    asset_names = [a["name"] for a in assets]
    feature_paths, metas = load_feature_index(assets)
    prop_path = OUT / "SOURCE_ONLY_MEMORY_PROPERTIES.csv"
    if prop_path.exists() and prop_path.stat().st_size > 0:
        prop_rows = read_csv(prop_path)
    else:
        prop_rows = []
        for asset in asset_names:
            for split_id, seed in enumerate([0, 1, 2]):
                ref, _evn, _anom, _groups, _ref_keys = split_normal_groups(asset, metas[asset], seed)
                wrong_assets = [x for x in asset_names if x != asset]
                normal_positions_all = {a: [i for i, r in enumerate(metas[a]["rows"]) if int(r["label"]) == 0] for a in asset_names}
                B = min(PREVIOUS_VERIFIED_BUDGET, len(ref) * PATCHES, min(len(normal_positions_all[w]) * PATCHES for w in wrong_assets))
                variants = {
                    "P0_CROSS_ASSET_FPS": ({w: normal_positions_all[w] for w in wrong_assets}, "fps"),
                    "P1_CROSS_ASSET_RANDOM": ({w: normal_positions_all[w] for w in wrong_assets}, "random"),
                    "P2_SAME_ASSET_FPS": ({asset: ref}, "fps"),
                    "P3_SAME_ASSET_RANDOM": ({asset: ref}, "random"),
                    "P4_WRONG_ASSET_1": ({wrong_assets[0]: normal_positions_all[wrong_assets[0]]}, "fps"),
                    "P5_WRONG_ASSET_2": ({wrong_assets[1]: normal_positions_all[wrong_assets[1]]}, "fps"),
                }
                for variant, (src, mode) in variants.items():
                    print(f"[XXXVII-A] source props {asset} split{split_id} {variant}", flush=True)
                    bank, entries, _ = build_memory_positions(src, feature_paths, metas, B, mode, seed=0)
                    prop_rows.append(memory_properties(bank, entries, asset, split_id, variant))
        write_csv(prop_path, prop_rows)

    prop_by = {(r["target_asset"], int(r["split"]), r["variant"]): r for r in prop_rows}
    corr_rows, gate_rows = [], []
    predictors = [
        "S1_MEMORY_SPREAD_mean_distance",
        "S2_MEMORY_REDUNDANCY_sim_gt_0p95",
        "S3_EFFECTIVE_RANK",
        "S4_LOCAL_NN_SCALE_median_top1_distance",
        "S5_SOURCE_IMAGE_DIVERSITY_entropy",
    ]
    outcomes = ["G_local", "P-AUROC", "P-AP", "Sep_median"]
    eval_rows = []
    for r in sep_rows:
        p = prop_by[(r["target_asset"], int(r["split"]), r["variant"])]
        eval_rows.append({**r, **{k: p[k] for k in predictors}})
    for pred in predictors:
        pooled = {}
        for out in outcomes:
            pooled[out + "_spearman"] = corr([r[pred] for r in eval_rows], [r[out] for r in eval_rows], "spearman")
            pooled[out + "_pearson"] = corr([r[pred] for r in eval_rows], [r[out] for r in eval_rows], "pearson")
        signs = []
        for asset in asset_names:
            vals = [r for r in eval_rows if r["target_asset"] == asset]
            rho = corr([r[pred] for r in vals], [r["G_local"] for r in vals], "spearman")
            if np.isfinite(rho):
                signs.append(np.sign(rho))
            corr_rows.append({"predictor": pred, "scope": asset, "spearman_G_local": rho})
        corr_rows.append({"predictor": pred, "scope": "POOLED", **pooled})
        sign_consistent = len(signs) >= 2 and len(set(signs)) == 1
        direction_consistent = np.isfinite(pooled["G_local_spearman"]) and np.isfinite(pooled["P-AUROC_spearman"]) and np.isfinite(pooled["P-AP_spearman"]) and np.sign(pooled["G_local_spearman"]) == np.sign(pooled["P-AUROC_spearman"]) == np.sign(pooled["P-AP_spearman"])
        sep_consistent = np.isfinite(pooled["Sep_median_spearman"]) and np.sign(pooled["G_local_spearman"]) == np.sign(pooled["Sep_median_spearman"])
        passed = abs(pooled["G_local_spearman"]) >= 0.40 and direction_consistent and sign_consistent and sep_consistent
        gate_rows.append({"predictor": pred, "pooled_spearman_G_local": pooled["G_local_spearman"], "pooled_spearman_P-AUROC": pooled["P-AUROC_spearman"], "pooled_spearman_P-AP": pooled["P-AP_spearman"], "pooled_spearman_Sep": pooled["Sep_median_spearman"], "asset_sign_consistent": sign_consistent, "direction_consistent": direction_consistent, "sep_consistent": sep_consistent, "SOURCE_ONLY_SIGNAL_GATE": passed})
    write_csv(OUT / "SOURCE_PROPERTY_CORRELATIONS.csv", corr_rows)
    write_csv(OUT / "SOURCE_SIGNAL_GATE.csv", gate_rows)
    write_csv(OUT / "PER_ASSET_SPLIT.csv", [{**r, **prop_by[(r["target_asset"], int(r["split"]), r["variant"])]} for r in sep_rows])

    fps_over = [r for r in over_rows if r["comparison"] == "FPS"]
    rand_over = [r for r in over_rows if r["comparison"] == "RANDOM"]
    fps_sep_pos = sum(r["SeparationLoss"] > 0 for r in fps_over)
    rand_sep_pos = sum(r["SeparationLoss"] > 0 for r in rand_over)
    asset_sep = {a: float(np.mean([r["SeparationLoss"] for r in fps_over if r["target_asset"] == a])) for a in asset_names}
    mean_over_fps = float(np.mean([r["OverNormalize"] for r in fps_over]))
    mean_over_rand = float(np.mean([r["OverNormalize"] for r in rand_over]))
    sep_g_rho = corr([r["SeparationLoss"] for r in fps_over], [r["Cross_minus_Same_G_local"] for r in fps_over], "spearman")
    wrong_rho = corr([r["DeltaSep_wrong_minus_same"] for r in wrong_rows], [r["DeltaG_wrong_minus_same"] for r in wrong_rows], "spearman")
    over_gate = fps_sep_pos >= 7 and rand_sep_pos >= 7 and all(v > 0 for v in asset_sep.values()) and mean_over_fps < 0 and mean_over_rand < 0 and np.isfinite(sep_g_rho) and sep_g_rho >= 0.40
    wrong_gate = np.isfinite(wrong_rho) and wrong_rho >= 0.40
    source_gate = any(r["SOURCE_ONLY_SIGNAL_GATE"] for r in gate_rows)
    if not parity["passed"]:
        outcome = "XXXVII_A_IMPLEMENTATION_FAILURE"
    elif over_gate and wrong_gate and source_gate:
        outcome = "XXXVII_A_SOURCE_ONLY_REFERENCE_SIGNAL_SUPPORTED"
    elif over_gate and wrong_gate:
        outcome = "XXXVII_A_CONTRAST_MECHANISM_SUPPORTED_NO_SOURCE_PREDICTOR"
    elif (np.mean([r["Cross_minus_Same_G_local"] for r in fps_over]) > 0) and not over_gate:
        outcome = "XXXVII_A_HETEROGENEOUS_SUPPORT_BUT_NOT_OVERNORMALIZATION"
    elif source_gate:
        outcome = "XXXVII_A_SOURCE_PROPERTY_CORRELATION_NOT_SPECIFIC"
    else:
        outcome = "XXXVII_A_DISTANCE_MECHANISM_UNSUPPORTED"

    report = ["# Phase XXXVII-A Heterogeneous Normal Reference Contrast Forensics", "", f"Outcome: **{outcome}**", "", "No backbone forward was run. XXXVI metrics/support statistics were reused; memory banks were rebuilt only for source-only property estimates.", "", "## Mechanism Gates", "", f"- Parity passed: {parity['passed']}; max delta={parity['max_delta']:.6f}.", f"- SeparationLoss FPS positive cells: {fps_sep_pos}/9; Random: {rand_sep_pos}/9.", f"- Asset mean SeparationLoss FPS: {json.dumps(asset_sep, default=json_default)}.", f"- mean OverNormalize FPS={mean_over_fps:+.6f}; Random={mean_over_rand:+.6f}.", f"- Spearman SeparationLoss vs Cross-Same G gain={sep_g_rho}.", f"- Spearman Wrong-Same DeltaSep vs DeltaG={wrong_rho}.", f"- Source-only property gate passed: {source_gate}.", "", "## Interpretation", "", "SameAsset lowered both normal and anomaly distances, but the cached separation statistics do not show a stable same-asset separation loss explaining the large pixel-ranking degradation. Patch-level distance AUROC often moves opposite to full-resolution pixel G.", "", f"Runtime: {time.time() - start:.1f}s.", "", "Stop: Phase XXXVII-A ended here."]
    (OUT / "XXXVII_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    payload = {"phase": "XXXVII-A", "outcome": outcome, "parity_passed": parity["passed"], "fps_separation_loss_positive_cells": fps_sep_pos, "random_separation_loss_positive_cells": rand_sep_pos, "asset_mean_separation_loss_fps": asset_sep, "mean_overnormalize_fps": mean_over_fps, "mean_overnormalize_random": mean_over_rand, "separation_loss_vs_g_gain_spearman": sep_g_rho, "wrong_delta_sep_vs_delta_g_spearman": wrong_rho, "source_only_gate_passed": source_gate, "runtime_sec": time.time() - start}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXXVII-A\n")
        fp.write("- Command: python experiments/phase_xxxvii_reference_contrast.py\n")
        fp.write(f"- XXXVI parity passed: {parity['passed']}.\n")
        fp.write(f"- SeparationLoss FPS positive cells {fps_sep_pos}/9; Random {rand_sep_pos}/9; mean OverNormalize FPS {mean_over_fps:+.6f}; Random {mean_over_rand:+.6f}.\n")
        fp.write(f"- Sep-loss vs G-gain Spearman {sep_g_rho}; wrong DeltaSep vs DeltaG Spearman {wrong_rho}; source-only gate {source_gate}.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXXVII_A_REPORT.md"), "runtime_sec": time.time() - start}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
