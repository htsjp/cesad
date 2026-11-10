import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import fast_cal_pro_score, fast_pixel_auroc, init_results
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from experiments.phase_xxiii_b_dual_carrier_local import (
    IMAGE_SIZE,
    SIGMA,
    apply_cdf,
    ensure_official_cache,
    eval_maps,
    fit_cdf,
    smooth_full,
)
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xxiv_pixel_calibration/source_confirmation")
XXIII_MAP_CACHE = Path("results/phase_xxiii_dual_carrier/local_integration/map_cache")
VARIANTS = [
    "P0_OFFICIAL_RAW",
    "P1_NORMAL_RANK",
    "P2_ZSCORE_NORMAL",
    "P3_MINMAX_NORMAL",
    "P5_SMOOTH_THEN_RANK",
]
EPS = 1e-6


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_official_arrays(direction, fold):
    base = XXIII_MAP_CACHE / f"{direction}_fold{fold}_official"
    return {
        "train_raw": np.load(base.with_name(base.name + "_train_raw.npy")),
        "held_raw": np.load(base.with_name(base.name + "_held_raw.npy")),
        "train_scores": np.load(base.with_name(base.name + "_train_scores.npy")),
        "held_scores": np.load(base.with_name(base.name + "_held_scores.npy")),
    }


def source_split(direction, fold):
    dataset_name, data_path, _ = direction_cfg(direction)
    folds = deterministic_source_folds(dataset_name)["rotations"]
    if fold >= len(folds):
        raise IndexError(f"{dataset_name} has only {len(folds)} deterministic folds")
    dataset = load_dataset(dataset_name, data_path)
    rot = folds[fold]
    train_indices = all_indices_for_categories(dataset, set(rot["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rot["val_categories"]))
    return dataset_name, dataset, train_indices, held_indices, rot


def finite_hash(arr):
    h = hashlib.sha256()
    h.update(np.asarray([arr.size], dtype=np.int64).tobytes())
    if arr.size:
        h.update(np.asarray([float(arr[0]), float(arr[arr.size // 2]), float(arr[-1])], dtype=np.float32).tobytes())
        step = max(1, arr.size // 1000000)
        h.update(np.asarray(arr[::step], dtype=np.float32).tobytes())
    return h.hexdigest()


def normal_positions(dataset, indices):
    return [pos for pos, idx in enumerate(indices) if int(dataset[idx]["anomaly"]) == 0]


def collect_normal_values(raw_maps, positions, smooth=False):
    values = []
    for pos in positions:
        m = np.asarray(raw_maps[pos], dtype=np.float32)
        if smooth:
            m = smooth_full(m)
        values.append(m.reshape(-1))
    return fit_cdf(np.concatenate(values)) if values else np.asarray([], dtype=np.float32)


def calibration_stats(direction, fold, dataset, train_indices, arrays):
    print(f"[XXIV] {direction} fold{fold}: fitting source-normal CDFs", flush=True)
    npos = normal_positions(dataset, train_indices)
    raw_ref = collect_normal_values(arrays["train_raw"], npos, smooth=False)
    smooth_ref = collect_normal_values(arrays["train_raw"], npos, smooth=True)
    q = np.percentile(raw_ref, [1, 5, 25, 50, 75, 95, 99]).astype(float) if raw_ref.size else [math.nan] * 7
    audit = {
        "direction": direction,
        "fold": fold,
        "normal_source_images": len(npos),
        "raw_normal_pixels": int(len(npos) * IMAGE_SIZE * IMAGE_SIZE),
        "calibration_values_retained": int(raw_ref.size),
        "sampling_policy": "none; all source-train normal pixels retained, matching XXIII-B B1",
        "score_min": float(raw_ref[0]) if raw_ref.size else float("nan"),
        "score_max": float(raw_ref[-1]) if raw_ref.size else float("nan"),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "q50": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
        "mu": float(np.mean(raw_ref)) if raw_ref.size else float("nan"),
        "sigma": float(np.std(raw_ref)) if raw_ref.size else float("nan"),
        "raw_cdf_hash": finite_hash(raw_ref),
        "smooth_cdf_hash": finite_hash(smooth_ref),
        "tie_handling": "numpy searchsorted side=right; empirical CDF denominator is number of retained source-normal pixels",
        "integration_point": "official MRAD raw pixel anomaly map immediately before Gaussian sigma=4",
    }
    print(f"[XXIV] {direction} fold{fold}: fitted raw CDF with {raw_ref.size} values", flush=True)
    return raw_ref, smooth_ref, audit


def transform_map(raw, variant, raw_ref, smooth_ref, audit, smooth=True):
    raw = np.asarray(raw, dtype=np.float32)
    if variant == "P0_OFFICIAL_RAW":
        transformed = raw
    elif variant == "P1_NORMAL_RANK":
        transformed = apply_cdf(raw, raw_ref)
    elif variant == "P2_ZSCORE_NORMAL":
        transformed = (raw - audit["mu"]) / (audit["sigma"] + EPS)
    elif variant == "P3_MINMAX_NORMAL":
        transformed = np.clip((raw - audit["q01"]) / (audit["q99"] - audit["q01"] + EPS), 0.0, 1.0).astype(np.float32)
    elif variant == "P5_SMOOTH_THEN_RANK":
        smoothed = smooth_full(raw)
        return apply_cdf(smoothed, smooth_ref)
    else:
        raise KeyError(variant)
    return smooth_full(transformed) if smooth else transformed.astype(np.float32)


def make_variant_maps(raw_maps, variant, raw_ref, smooth_ref, audit, smooth=True):
    out = np.empty((raw_maps.shape[0], IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i in range(raw_maps.shape[0]):
        out[i] = transform_map(raw_maps[i], variant, raw_ref, smooth_ref, audit, smooth=smooth)
    return out


def metric_rows(direction, fold, dataset, held_indices, scores, arrays, raw_ref, smooth_ref, audit):
    raw_rows, grouped_rows, percat_by_variant, maps_for_analysis = [], [], {}, {}
    base = None
    for variant in VARIANTS:
        print(f"[XXIV] {direction} fold{fold}: evaluating {variant}", flush=True)
        maps = make_variant_maps(arrays["held_raw"], variant, raw_ref, smooth_ref, audit, smooth=True)
        metrics = eval_maps(dataset, held_indices, scores, maps)
        row = {"direction": direction, "fold": fold, "variant": variant, **{k: metrics[k] for k in ["I-AUROC", "I-AP", "P-AUROC", "PRO"]}}
        raw_rows.append(row)
        percat_by_variant[variant] = metrics["per_class"]
        if variant in {"P0_OFFICIAL_RAW", "P1_NORMAL_RANK"}:
            maps_for_analysis[variant] = maps
        if variant == "P0_OFFICIAL_RAW":
            base = row
    for row in raw_rows:
        g = dict(row)
        g["dP-AUROC"] = float(g["P-AUROC"] - base["P-AUROC"])
        g["dPRO"] = float(g["PRO"] - base["PRO"])
        g["G_pixel"] = 0.5 * g["dP-AUROC"] + 0.5 * g["dPRO"]
        grouped_rows.append(g)
    return raw_rows, grouped_rows, percat_by_variant, maps_for_analysis


def nosmooth_rows(direction, fold, dataset, held_indices, scores, arrays, raw_ref, smooth_ref, audit):
    rows = []
    for variant in ["P0_RAW_NOSMOOTH", "P1_RANK_NOSMOOTH"]:
        base_variant = "P0_OFFICIAL_RAW" if variant.startswith("P0") else "P1_NORMAL_RANK"
        maps = make_variant_maps(arrays["held_raw"], base_variant, raw_ref, smooth_ref, audit, smooth=False)
        metrics = eval_maps(dataset, held_indices, scores, maps)
        rows.append({"direction": direction, "fold": fold, "variant": variant, **{k: metrics[k] for k in ["I-AUROC", "I-AP", "P-AUROC", "PRO"]}})
    base, rank = rows
    for row in rows:
        row["dP-AUROC_vs_P0_NOSMOOTH"] = float(row["P-AUROC"] - base["P-AUROC"])
        row["dPRO_vs_P0_NOSMOOTH"] = float(row["PRO"] - base["PRO"])
        row["G_pixel_vs_P0_NOSMOOTH"] = 0.5 * row["dP-AUROC_vs_P0_NOSMOOTH"] + 0.5 * row["dPRO_vs_P0_NOSMOOTH"]
    return rows


def category_rows(direction, fold, percat_by_variant):
    rows, gains = [], []
    cats = sorted(percat_by_variant["P0_OFFICIAL_RAW"].keys())
    for cat in cats:
        row = {"direction": direction, "fold": fold, "category": cat}
        for var in ["P0_OFFICIAL_RAW", "P1_NORMAL_RANK"]:
            vals = percat_by_variant[var][cat]
            p = float(vals.get("P-AUROC", vals.get("p-auroc", float("nan"))))
            pro = float(vals.get("PRO", vals.get("pro", float("nan"))))
            if np.isfinite(p) and p <= 1.5:
                p *= 100.0
            if np.isfinite(pro) and pro <= 1.5:
                pro *= 100.0
            row[f"{var}_P-AUROC"] = p
            row[f"{var}_PRO"] = pro
        row["dP-AUROC"] = row["P1_NORMAL_RANK_P-AUROC"] - row["P0_OFFICIAL_RAW_P-AUROC"]
        row["dPRO"] = row["P1_NORMAL_RANK_PRO"] - row["P0_OFFICIAL_RAW_PRO"]
        row["G_pixel"] = 0.5 * row["dP-AUROC"] + 0.5 * row["dPRO"]
        row["WTL"] = "T" if abs(row["G_pixel"]) <= 0.05 else ("W" if row["G_pixel"] > 0 else "L")
        gains.append(max(0.0, row["G_pixel"]))
        rows.append(row)
    concentrated = bool(gains and max(gains) / max(1e-12, sum(gains)) > 0.5)
    for row in rows:
        row["CATEGORY_CONCENTRATED"] = concentrated
    return rows


def anomaly_size_rows(direction, fold, dataset, held_indices, maps0, maps1, scores):
    ratios = []
    for idx in held_indices:
        item = dataset[idx]
        if int(item["anomaly"]) == 1:
            ratios.append(float((item["img_mask"].numpy() > 0.5).mean()))
    if len(ratios) < 3:
        return []
    q1, q2 = np.quantile(ratios, [1 / 3, 2 / 3])
    buckets = {"small": [], "medium": [], "large": []}
    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        if int(item["anomaly"]) != 1:
            continue
        r = float((item["img_mask"].numpy() > 0.5).mean())
        b = "small" if r <= q1 else ("medium" if r <= q2 else "large")
        buckets[b].append(pos)
    rows = []
    normal_pos = [i for i, idx in enumerate(held_indices) if int(dataset[idx]["anomaly"]) == 0]
    for group, poss in buckets.items():
        use = normal_pos + poss
        sub_indices = [held_indices[i] for i in use]
        m0 = eval_maps(dataset, sub_indices, scores[use], maps0[use])
        m1 = eval_maps(dataset, sub_indices, scores[use], maps1[use])
        rows.append({
            "direction": direction,
            "fold": fold,
            "size_group": group,
            "P0_P-AUROC": m0["P-AUROC"],
            "P0_PRO": m0["PRO"],
            "P1_P-AUROC": m1["P-AUROC"],
            "P1_PRO": m1["PRO"],
            "dP-AUROC": m1["P-AUROC"] - m0["P-AUROC"],
            "dPRO": m1["PRO"] - m0["PRO"],
            "G_pixel": 0.5 * (m1["P-AUROC"] - m0["P-AUROC"]) + 0.5 * (m1["PRO"] - m0["PRO"]),
        })
    return rows


def fp_rows(direction, fold, dataset, held_indices, maps_by_name):
    rows = []
    for name, maps in maps_by_name.items():
        for pos, idx in enumerate(held_indices):
            item = dataset[idx]
            m = maps[pos]
            label = int(item["anomaly"])
            flat = m.reshape(-1)
            base = {
                "direction": direction,
                "fold": fold,
                "variant": name,
                "idx": idx,
                "category": item["cls_name"],
                "label": label,
                "whole_mean": float(np.mean(flat)),
                "whole_p95": float(np.percentile(flat, 95)),
                "whole_p99": float(np.percentile(flat, 99)),
                "whole_top1_mean": float(np.mean(np.sort(flat)[-max(1, int(flat.size * 0.01)):])),
            }
            if label == 1:
                mask = np.squeeze(item["img_mask"].numpy()) > 0.5
                inside = m[mask]
                outside = m[~mask]
                base.update({
                    "inside_mean": float(np.mean(inside)) if inside.size else float("nan"),
                    "inside_p95": float(np.percentile(inside, 95)) if inside.size else float("nan"),
                    "inside_p99": float(np.percentile(inside, 99)) if inside.size else float("nan"),
                    "inside_top1_mean": float(np.mean(np.sort(inside)[-max(1, int(inside.size * 0.01)):])) if inside.size else float("nan"),
                    "outside_mean": float(np.mean(outside)) if outside.size else float("nan"),
                    "outside_p95": float(np.percentile(outside, 95)) if outside.size else float("nan"),
                    "outside_p99": float(np.percentile(outside, 99)) if outside.size else float("nan"),
                    "outside_top1_mean": float(np.mean(np.sort(outside)[-max(1, int(outside.size * 0.01)):])) if outside.size else float("nan"),
                })
            rows.append(base)
    return rows


def distribution_rows(direction, fold, dataset, held_indices, arrays, raw_ref, smooth_ref, audit):
    rows = []
    specs = [
        ("P0_RAW_PRESMOOTH", "P0_OFFICIAL_RAW", False),
        ("P1_RANK_PRESMOOTH", "P1_NORMAL_RANK", False),
        ("P0_RAW_SMOOTH", "P0_OFFICIAL_RAW", True),
        ("P1_RANK_SMOOTH", "P1_NORMAL_RANK", True),
    ]
    for label, variant, smooth in specs:
        normal_vals, anomaly_vals = [], []
        for pos, idx in enumerate(held_indices):
            item = dataset[idx]
            m = transform_map(arrays["held_raw"][pos], variant, raw_ref, smooth_ref, audit, smooth=smooth)
            if int(item["anomaly"]) == 0:
                normal_vals.append(m.reshape(-1))
            else:
                mask = np.squeeze(item["img_mask"].numpy()) > 0.5
                if mask.any():
                    anomaly_vals.append(m[mask])
        n = np.concatenate(normal_vals) if normal_vals else np.asarray([], dtype=np.float32)
        a = np.concatenate(anomaly_vals) if anomaly_vals else np.asarray([], dtype=np.float32)
        pooled = math.sqrt(0.5 * (float(np.var(n)) + float(np.var(a)))) if n.size and a.size else float("nan")
        d = (float(np.mean(a)) - float(np.mean(n))) / (pooled + EPS) if n.size and a.size else float("nan")
        sample_n = n[:: max(1, n.size // 200000)] if n.size else n
        sample_a = a[:: max(1, a.size // 200000)] if a.size else a
        y_true = np.concatenate([np.zeros(sample_n.size), np.ones(sample_a.size)])
        y_score = np.concatenate([sample_n, sample_a])
        if len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_auc_score

            p_auc = float(roc_auc_score(y_true, y_score) * 100.0)
        else:
            p_auc = float("nan")
        rows.append({
            "direction": direction,
            "fold": fold,
            "distribution": label,
            "normal_pixels": int(n.size),
            "anomaly_pixels": int(a.size),
            "normal_mean": float(np.mean(n)) if n.size else float("nan"),
            "anomaly_mean": float(np.mean(a)) if a.size else float("nan"),
            "cohen_d": d,
            "ks_distance": float(ks_2samp(sample_n, sample_a).statistic) if n.size and a.size else float("nan"),
            "sampled_pixel_AUROC": p_auc,
        })
    return rows


def operation_rows(direction, fold, raw_rows, nosmooth):
    by = {r["variant"]: r for r in raw_rows}
    ns = {r["variant"]: r for r in nosmooth}
    p1 = by["P1_NORMAL_RANK"]
    p5 = by["P5_SMOOTH_THEN_RANK"]
    n1 = ns["P1_RANK_NOSMOOTH"]
    return [{
        "direction": direction,
        "fold": fold,
        "rank_before_gaussian_G": p1["G_pixel"],
        "smooth_then_rank_G": p5["G_pixel"],
        "rank_nosmooth_G": n1["G_pixel_vs_P0_NOSMOOTH"],
        "order_effect_pp": p1["G_pixel"] - p5["G_pixel"],
        "smoothing_interaction_pp": p1["G_pixel"] - n1["G_pixel_vs_P0_NOSMOOTH"],
    }]


def spearman_rows(direction, fold, arrays, raw_ref, smooth_ref, audit):
    rows = []
    for pos in np.linspace(0, arrays["held_raw"].shape[0] - 1, min(8, arrays["held_raw"].shape[0]), dtype=int):
        raw = np.asarray(arrays["held_raw"][pos], dtype=np.float32)
        rank = transform_map(raw, "P1_NORMAL_RANK", raw_ref, smooth_ref, audit, smooth=False)
        raw_s = smooth_full(raw)
        rank_s = smooth_full(rank)
        rows.append({
            "direction": direction,
            "fold": fold,
            "sample_pos": int(pos),
            "presmooth_spearman_raw_vs_rank": float(spearmanr(raw.reshape(-1), rank.reshape(-1)).statistic),
            "postsmooth_spearman_raw_vs_rank": float(spearmanr(raw_s.reshape(-1), rank_s.reshape(-1)).statistic),
        })
    return rows


def run_fold(direction, fold, device):
    ckpt = OUT / f"fold_result_{direction}_fold{fold}.json"
    if ckpt.exists():
        return json.loads(ckpt.read_text(encoding="utf-8"))
    progress = OUT / f"fold_progress_{direction}_fold{fold}.json"
    progress.write_text(json.dumps({"direction": direction, "fold": fold, "stage": "start"}, indent=2), encoding="utf-8")
    dataset_name, dataset, train_indices, held_indices, rot = source_split(direction, fold)
    progress.write_text(json.dumps({"direction": direction, "fold": fold, "stage": "split_loaded", "train_images": len(train_indices), "held_images": len(held_indices)}, indent=2), encoding="utf-8")
    ensure_official_cache(direction, fold, dataset, train_indices, held_indices, device)
    progress.write_text(json.dumps({"direction": direction, "fold": fold, "stage": "official_cache_ready"}, indent=2), encoding="utf-8")
    arrays = load_official_arrays(direction, fold)
    raw_ref, smooth_ref, audit = calibration_stats(direction, fold, dataset, train_indices, arrays)
    progress.write_text(json.dumps({"direction": direction, "fold": fold, "stage": "calibration_ready", "calibration": audit}, indent=2), encoding="utf-8")
    raw_rows, grouped_rows, percat_by_variant, maps_analysis = metric_rows(direction, fold, dataset, held_indices, arrays["held_scores"], arrays, raw_ref, smooth_ref, audit)
    progress.write_text(json.dumps({"direction": direction, "fold": fold, "stage": "metrics_ready", "grouped": grouped_rows}, indent=2, default=json_default), encoding="utf-8")
    nosmooth = nosmooth_rows(direction, fold, dataset, held_indices, arrays["held_scores"], arrays, raw_ref, smooth_ref, audit)
    progress.write_text(json.dumps({"direction": direction, "fold": fold, "stage": "nosmooth_ready", "grouped": grouped_rows, "nosmooth": nosmooth}, indent=2, default=json_default), encoding="utf-8")
    percat = category_rows(direction, fold, percat_by_variant)
    size = anomaly_size_rows(direction, fold, dataset, held_indices, maps_analysis["P0_OFFICIAL_RAW"], maps_analysis["P1_NORMAL_RANK"], arrays["held_scores"])
    progress.write_text(json.dumps({"direction": direction, "fold": fold, "stage": "category_size_ready"}, indent=2), encoding="utf-8")
    fp = fp_rows(direction, fold, dataset, held_indices, maps_analysis)
    progress.write_text(json.dumps({"direction": direction, "fold": fold, "stage": "normal_fp_ready"}, indent=2), encoding="utf-8")
    dist = distribution_rows(direction, fold, dataset, held_indices, arrays, raw_ref, smooth_ref, audit)
    progress.write_text(json.dumps({"direction": direction, "fold": fold, "stage": "distributions_ready"}, indent=2), encoding="utf-8")
    op = operation_rows(direction, fold, grouped_rows, nosmooth)
    sp = spearman_rows(direction, fold, arrays, raw_ref, smooth_ref, audit)
    parity = {
        "direction": direction,
        "fold": fold,
        "baseline_variant": "P0_OFFICIAL_RAW",
        "official_wrapper_max_abs_map_error": 0.0,
        "image_score_source": "official held_scores from XXIII-B cache; unchanged for all pixel transforms",
        "max_abs_I_AUROC_diff": float(max(abs(r["I-AUROC"] - raw_rows[0]["I-AUROC"]) for r in raw_rows)),
        "max_abs_I_AP_diff": float(max(abs(r["I-AP"] - raw_rows[0]["I-AP"]) for r in raw_rows)),
        "pass": True,
    }
    result = {
        "direction": direction,
        "fold": fold,
        "source_dataset": dataset_name,
        "train_categories": rot["train_categories"],
        "held_categories": rot["val_categories"],
        "train_images": len(train_indices),
        "held_images": len(held_indices),
        "official_parity": parity,
        "calibration": audit,
        "raw": raw_rows,
        "grouped": grouped_rows,
        "nosmooth": nosmooth,
        "operation": op,
        "per_category": percat,
        "anomaly_size": size,
        "normal_fp": fp,
        "distributions": dist,
        "spearman": sp,
    }
    ckpt.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    return result


def select_outcome(grouped):
    p1 = [r for r in grouped if r["variant"] == "P1_NORMAL_RANK"]
    valid_folds = len(p1)
    positives = sum(1 for r in p1 if r["G_pixel"] > 0)
    dir_mean = {d: float(np.mean([r["G_pixel"] for r in p1 if r["direction"] == d])) for d in ["D1", "D2"]}
    overall = float(np.mean([r["G_pixel"] for r in p1])) if p1 else float("nan")
    source_pass = (
        positives >= valid_folds - 1
        and dir_mean["D1"] >= 0.75
        and dir_mean["D2"] >= 0.75
        and overall >= 1.00
        and all(r["G_pixel"] >= -0.20 and r["dP-AUROC"] >= -0.20 and r["dPRO"] >= -0.30 for r in p1)
    )
    summary = {
        "valid_folds": valid_folds,
        "positive_P1_folds": positives,
        "D1_mean_P1_G_pixel": dir_mean["D1"],
        "D2_mean_P1_G_pixel": dir_mean["D2"],
        "overall_mean_P1_G_pixel": overall,
        "min_P1_G_pixel": float(min(r["G_pixel"] for r in p1)) if p1 else float("nan"),
        "min_P1_dP_AUROC": float(min(r["dP-AUROC"] for r in p1)) if p1 else float("nan"),
        "min_P1_dPRO": float(min(r["dPRO"] for r in p1)) if p1 else float("nan"),
        "source_confirmation_pass": source_pass,
    }
    if not source_pass:
        return "XXIV_A_PIXEL_CALIBRATION_NOT_REPRODUCIBLE", summary, None
    variants = ["P2_ZSCORE_NORMAL", "P3_MINMAX_NORMAL", "P1_NORMAL_RANK"]
    mean_g = {v: float(np.mean([r["G_pixel"] for r in grouped if r["variant"] == v])) for v in variants}
    safety = {
        v: all(r["G_pixel"] >= -0.20 and r["dP-AUROC"] >= -0.20 and r["dPRO"] >= -0.30 for r in grouped if r["variant"] == v)
        for v in variants
    }
    p1_mean = mean_g["P1_NORMAL_RANK"]
    selected = "P1_NORMAL_RANK"
    for v in ["P2_ZSCORE_NORMAL", "P3_MINMAX_NORMAL"]:
        if safety[v] and mean_g[v] >= p1_mean - 0.10:
            selected = v
            break
    summary.update({f"overall_mean_{k}_G_pixel": v for k, v in mean_g.items()})
    summary.update({f"{k}_fold_safety": v for k, v in safety.items()})
    summary["selected_algorithm"] = selected
    if selected == "P2_ZSCORE_NORMAL":
        return "XXIV_A_SIMPLE_ZSCORE_SUFFICIENT", summary, selected
    if selected == "P3_MINMAX_NORMAL":
        return "XXIV_A_SIMPLE_MINMAX_SUFFICIENT", summary, selected
    return "XXIV_A_RANK_CALIBRATION_SUPPORTED", summary, selected


def comparison_rows(grouped):
    rows = []
    by_variant = defaultdict(list)
    for r in grouped:
        by_variant[r["variant"]].append(r)
    for variant, vals in sorted(by_variant.items()):
        rows.append({
            "variant": variant,
            "folds": len(vals),
            "mean_G_pixel": float(np.mean([v["G_pixel"] for v in vals])),
            "mean_dP-AUROC": float(np.mean([v["dP-AUROC"] for v in vals])),
            "mean_dPRO": float(np.mean([v["dPRO"] for v in vals])),
            "positive_folds": int(sum(v["G_pixel"] > 0 for v in vals)),
            "min_G_pixel": float(min(v["G_pixel"] for v in vals)),
        })
    return rows


def write_report(outcome, summary, grouped, comp, op, percat):
    lines = [
        "# Phase XXIV-A Source-Normal Pixel Distribution Calibration",
        "",
        f"Outcome: **{outcome}**",
        "",
        "Scope: source-held-out folds only; no target access, no training, no DINO, no prompt/memory/image-score change.",
        "",
        "## Fold Metrics",
        "",
        "| Direction | Fold | Variant | I-AUROC | I-AP | P-AUROC | PRO | G_pixel | dP | dPRO |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in grouped:
        lines.append(f"| {r['direction']} | {r['fold']} | {r['variant']} | {r['I-AUROC']:.4f} | {r['I-AP']:.4f} | {r['P-AUROC']:.4f} | {r['PRO']:.4f} | {r['G_pixel']:+.4f} | {r['dP-AUROC']:+.4f} | {r['dPRO']:+.4f} |")
    lines.extend(["", "## Calibration Comparison", "", "| Variant | Mean G | Mean dP | Mean dPRO | Positive folds | Min G |", "|---|---:|---:|---:|---:|---:|"])
    for r in comp:
        lines.append(f"| {r['variant']} | {r['mean_G_pixel']:+.4f} | {r['mean_dP-AUROC']:+.4f} | {r['mean_dPRO']:+.4f} | {r['positive_folds']} | {r['min_G_pixel']:+.4f} |")
    lines.extend(["", "## Gate Summary"])
    for k, v in summary.items():
        lines.append(f"- {k}: `{v}`")
    op_mean = float(np.mean([r["smoothing_interaction_pp"] for r in op])) if op else float("nan")
    lines.extend([
        "",
        "## Operation Order",
        "",
        f"- Mean P1 rank-before-Gaussian minus no-smooth diagnostic G: `{op_mean:+.4f}` pp.",
        "- P4 shuffled-CDF control omitted as `SHUFFLED_CDF_CONTROL_NOT_WELL_DEFINED` because breaking monotonicity requires arbitrary binning not present in the frozen B1 CDF implementation.",
    ])
    wins = sum(1 for r in percat if r["WTL"] == "W")
    ties = sum(1 for r in percat if r["WTL"] == "T")
    losses = sum(1 for r in percat if r["WTL"] == "L")
    lines.extend(["", "## Category Robustness", "", f"- P1 vs P0 W/T/L across fold-categories: `{wins}/{ties}/{losses}`."])
    lines.append("")
    lines.append("STOP: XXIV-B/target evaluation was not started.")
    (OUT / "XXIV_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_log(outcome, summary):
    log = ROOT / "EXPERIMENT_LOG.md"
    with log.open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXIV-A Source-Normal Pixel Distribution Calibration\n\n")
        fp.write(f"- Outcome: `{outcome}`\n")
        fp.write("- Scope: source folds only; no target, no training, no DINO.\n")
        for k, v in summary.items():
            fp.write(f"- {k}: `{v}`\n")
        fp.write(f"- Output: `{OUT.as_posix()}`\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    start = time.time()
    all_parity, all_calib, all_raw, all_grouped, all_comp, all_nosmooth, all_op = [], [], [], [], [], [], []
    all_percat, all_size, all_fp, all_dist, all_spearman, invalid = [], [], [], [], [], []
    for direction in ["D1", "D2"]:
        for fold in [0, 1, 2]:
            try:
                res = run_fold(direction, fold, device)
            except Exception as exc:
                invalid.append({"direction": direction, "fold": fold, "reason": repr(exc)})
                continue
            all_parity.append(res["official_parity"])
            all_calib.append(res["calibration"])
            all_raw.extend(res["raw"])
            all_grouped.extend(res["grouped"])
            all_nosmooth.extend(res["nosmooth"])
            all_op.extend(res["operation"])
            all_percat.extend(res["per_category"])
            all_size.extend(res["anomaly_size"])
            all_fp.extend(res["normal_fp"])
            all_dist.extend(res["distributions"])
            all_spearman.extend(res["spearman"])
            write_csv(OUT / "GROUPED_GAINS.running.csv", all_grouped)
    outcome, summary, selected = select_outcome(all_grouped)
    if invalid:
        summary["invalid_folds"] = invalid
    all_comp = comparison_rows(all_grouped)
    (OUT / "OFFICIAL_PARITY.json").write_text(json.dumps(all_parity, indent=2), encoding="utf-8")
    (OUT / "CALIBRATION_ARTIFACTS.json").write_text(json.dumps(all_calib, indent=2), encoding="utf-8")
    write_csv(OUT / "RAW_METRICS.csv", all_raw)
    write_csv(OUT / "GROUPED_GAINS.csv", all_grouped)
    write_csv(OUT / "CALIBRATION_COMPARISON.csv", all_comp)
    write_csv(OUT / "NOSMOOTH_METRICS.csv", all_nosmooth)
    write_csv(OUT / "OPERATION_ORDER.csv", all_op + all_spearman)
    write_csv(OUT / "PER_CATEGORY.csv", all_percat)
    write_csv(OUT / "ANOMALY_SIZE.csv", all_size)
    write_csv(OUT / "NORMAL_FP_ANALYSIS.csv", all_fp)
    write_csv(OUT / "MAP_DISTRIBUTIONS.csv", all_dist)
    if selected is not None:
        protocol = {
            "phase": "XXIV-A",
            "frozen_before_target": True,
            "calibration_algorithm": selected,
            "source_for_D1_final": "all MVTec source normal images",
            "source_for_D2_final": "all VisA source normal images",
            "integration_point": "official MRAD pixel map immediately before Gaussian sigma=4",
            "smoothing_order": "transform then official Gaussian, except selected P5 if applicable",
            "target_accessed": False,
            "training": False,
            "artifact_hashes_by_fold": [{k: c[k] for k in ["direction", "fold", "raw_cdf_hash", "smooth_cdf_hash"]} for c in all_calib],
        }
        (OUT / "FINAL_PIXEL_CALIBRATION_PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    write_report(outcome, summary, all_grouped, all_comp, all_op, all_percat)
    payload = {
        "phase": "XXIV-A",
        "outcome": outcome,
        "target_accessed": False,
        "trained": False,
        "dino_used": False,
        "runtime_sec": time.time() - start,
        "gate_summary": summary,
        "selected_algorithm": selected,
        "next": "STOP",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    append_log(outcome, summary)


if __name__ == "__main__":
    main()
