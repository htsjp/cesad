import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from phase_xa.source_folds import deterministic_source_folds


AUDIT = Path("results/phase_xxxiii_memory_carrier_routing/audit")
OUT = Path("results/phase_xxxiii_memory_carrier_routing/soft_adjudication")
XXX = Path("results/phase_xxx_metric_safe_dino")
XXIII = Path("results/phase_xxiii_carrier_audit")
MAP_CACHE = XXX / "map_cache"
IMAGE_SIZE = 518
SIDE = 37
SIGMA = 4
TIE_PP = 0.05


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def overlay_exact_fixed_fusion(raw_rows, pc_rows):
    exact_raw = {}
    for r in load_csv(XXX / "G_RANK.csv"):
        if r["variant"] == "F0_CLIP_DINO_FUSION":
            exact_raw[(r["direction"], int(r["fold"]))] = r
    for r in raw_rows:
        if r["method"] != "S2_FIXED_FUSION":
            continue
        e = exact_raw[(r["direction"], int(r["fold"]))]
        r["P-AUROC"] = float(e["Pixel_AUROC"])
        r["P-AP"] = float(e["Pixel_AP"])
        r["evaluated_pixels"] = int(e["evaluated_pixels"])
        r["positive_pixels"] = int(e["positive_pixels"])
        r["dP-AUROC_vs_CLIP"] = float(e["dPixel_AUROC"])
        r["dP-AP_vs_CLIP"] = float(e["dPixel_AP"])
        r["G_rank_vs_CLIP"] = float(e["G_rank"])

    exact_pc = {}
    for r in load_csv(XXX / "PER_CATEGORY.csv"):
        if r["variant"] == "F0_CLIP_DINO_FUSION":
            exact_pc[(r["direction"], int(r["fold"]), r["category"])] = r
    for r in pc_rows:
        if r["method"] != "S2_FIXED_FUSION":
            continue
        e = exact_pc[(r["direction"], int(r["fold"]), r["category"])]
        r["P-AUROC"] = float(e["Pixel_AUROC"])
        r["P-AP"] = float(e["Pixel_AP"])
        r["pixels"] = int(e["pixels"])
        r["positive_pixels"] = int(e["positive_pixels"])
    return raw_rows, pc_rows


def load_fold(direction, fold):
    dataset_name, data_path, _ = direction_cfg(direction)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
    return dataset_name, dataset, rotation, train_indices, held_indices


def load_selected_beta():
    rows = {}
    for r in load_csv(XXIII / "FUSION_METRICS.csv"):
        if r.get("task") == "pixel":
            rows[(r["direction"], int(r["fold"]))] = float(r["selected_beta_clip"])
    return rows


def upsample_map37(m37):
    x = torch.tensor(np.asarray(m37, dtype=np.float32), dtype=torch.float32).view(1, 1, SIDE, SIDE)
    full = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze().numpy()
    return gaussian_filter(full.astype(np.float32), sigma=SIGMA).astype(np.float32)


def make_rank_reference(vals):
    return np.sort(np.asarray(vals, dtype=np.float64).reshape(-1))


def apply_rank_array(arr, ref):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    return (np.searchsorted(ref, flat, side="right").astype(np.float32) / max(1, ref.size)).reshape(arr.shape)


def method_arrays(c_full, d_full, f_full, w_d, w_d_shuf, w_d_rand):
    return {
        "S0_CLIP": c_full,
        "S1_DINO": d_full,
        "S2_FIXED_FUSION": f_full,
        "S3_MEMORY_SOFT": (1.0 - w_d) * c_full + w_d * d_full,
        "C1_SHUFFLED_SUPPORT": (1.0 - w_d_shuf) * c_full + w_d_shuf * d_full,
        "C2_RANDOM_WEIGHT": (1.0 - w_d_rand) * c_full + w_d_rand * d_full,
    }


def fold2_available(direction):
    needed = [
        MAP_CACHE / f"{direction}_fold2_clip_train37.npy",
        MAP_CACHE / f"{direction}_fold2_clip_held37.npy",
        MAP_CACHE / f"{direction}_fold2_dino_train37.npy",
        MAP_CACHE / f"{direction}_fold2_dino_held37.npy",
        AUDIT / f"fold_result_{direction}_fold2.json",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in needed)


def eval_fold(direction, fold, beta):
    dataset_name, dataset, rotation, train_indices, held_indices = load_fold(direction, fold)
    result = json.loads((AUDIT / f"fold_result_{direction}_fold{fold}.json").read_text(encoding="utf-8"))
    support_rows = result[3]
    if len(support_rows) != len(held_indices):
        raise RuntimeError(f"{direction} fold{fold} support row count mismatch")

    c_train = np.load(MAP_CACHE / f"{direction}_fold{fold}_clip_train37.npy", mmap_mode="r")
    d_train = np.load(MAP_CACHE / f"{direction}_fold{fold}_dino_train37.npy", mmap_mode="r")
    c_maps = np.load(MAP_CACHE / f"{direction}_fold{fold}_clip_held37.npy", mmap_mode="r")
    d_maps = np.load(MAP_CACHE / f"{direction}_fold{fold}_dino_held37.npy", mmap_mode="r")
    c_rank = make_rank_reference(c_train)
    d_rank = make_rank_reference(d_train)

    q_c = np.asarray([float(r["Q_C"]) for r in support_rows], dtype=np.float32)
    q_d = np.asarray([float(r["Q_D"]) for r in support_rows], dtype=np.float32)
    rel_c = np.exp(-q_c)
    rel_d = np.exp(-q_d)
    w_d = rel_d / (rel_c + rel_d)
    rng = np.random.default_rng(0)
    perm_shuf = rng.permutation(len(w_d))
    rng = np.random.default_rng(0)
    perm_rand = rng.permutation(len(w_d))
    w_d_shuf = w_d[perm_shuf]
    w_d_rand = w_d[perm_rand]

    methods = ["S0_CLIP", "S1_DINO", "S2_FIXED_FUSION", "S3_MEMORY_SOFT", "C1_SHUFFLED_SUPPORT", "C2_RANDOM_WEIGHT"]
    all_parts = {m: {"y": [], "s": []} for m in methods}
    by_cat = {m: defaultdict(lambda: {"y": [], "s": []}) for m in methods}
    comp_rows = []
    weight_rows = []

    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
        c_full = upsample_map37(c_maps[pos]).reshape(-1)
        d_full = upsample_map37(d_maps[pos]).reshape(-1)
        c_r = apply_rank_array(c_maps[pos], c_rank)
        d_r = apply_rank_array(d_maps[pos], d_rank)
        f_full = upsample_map37(beta * c_r + (1.0 - beta) * d_r).reshape(-1)
        scores = method_arrays(c_full, d_full, f_full, float(w_d[pos]), float(w_d_shuf[pos]), float(w_d_rand[pos]))

        cat = item["cls_name"]
        for name, s in scores.items():
            all_parts[name]["y"].append(mask)
            all_parts[name]["s"].append(np.asarray(s, dtype=np.float32))
            by_cat[name][cat]["y"].append(mask)
            by_cat[name][cat]["s"].append(np.asarray(s, dtype=np.float32))

        c_ap = safe_ap(mask, c_full) if mask.sum() > 0 else float("nan")
        d_ap = safe_ap(mask, d_full) if mask.sum() > 0 else float("nan")
        s3_ap = safe_ap(mask, scores["S3_MEMORY_SOFT"]) if mask.sum() > 0 else float("nan")
        if np.isfinite(c_ap) and np.isfinite(d_ap) and np.isfinite(s3_ap):
            stronger = "CLIP" if c_ap > d_ap + TIE_PP else ("DINO" if d_ap > c_ap + TIE_PP else "TIE")
            soft_exceeds_both = bool(s3_ap > max(c_ap, d_ap) + TIE_PP)
        else:
            stronger = ""
            soft_exceeds_both = False
        comp_rows.append({
            "direction": direction,
            "fold": fold,
            "idx": idx,
            "category": cat,
            "label": int(item["anomaly"]),
            "clip_pixel_AP": c_ap,
            "dino_pixel_AP": d_ap,
            "memory_soft_pixel_AP": s3_ap,
            "stronger_single_carrier": stronger,
            "soft_exceeds_both": soft_exceeds_both,
        })
        sr = support_rows[pos]
        weight_rows.append({
            "row_type": "image",
            "direction": direction,
            "fold": fold,
            "idx": idx,
            "category": cat,
            "label": int(item["anomaly"]),
            "Q_C": float(sr["Q_C"]),
            "Q_D": float(sr["Q_D"]),
            "w_C": float(1.0 - w_d[pos]),
            "w_D": float(w_d[pos]),
            "mean_w_D": "",
            "std_w_D": "",
            "q05_w_D": "",
            "q25_w_D": "",
            "median_w_D": "",
            "q75_w_D": "",
            "q95_w_D": "",
            "WEIGHT_COLLAPSE": "",
        })

    raw_rows, pc_rows = [], []
    for name in methods:
        y = np.concatenate(all_parts[name]["y"])
        s = np.concatenate(all_parts[name]["s"]).astype(np.float32)
        raw_rows.append({
            "direction": direction,
            "fold": fold,
            "method": name,
            "P-AUROC": safe_auc(y, s),
            "P-AP": safe_ap(y, s),
            "evaluated_pixels": int(y.size),
            "positive_pixels": int(y.sum()),
            "beta_clip_fixed_fusion": beta if name == "S2_FIXED_FUSION" else "",
        })
        for cat, vals in sorted(by_cat[name].items()):
            cy = np.concatenate(vals["y"])
            cs = np.concatenate(vals["s"]).astype(np.float32)
            pc_rows.append({
                "direction": direction,
                "fold": fold,
                "category": cat,
                "method": name,
                "P-AUROC": safe_auc(cy, cs),
                "P-AP": safe_ap(cy, cs),
                "pixels": int(cy.size),
                "positive_pixels": int(cy.sum()),
            })

    qs = np.quantile(w_d, [0.05, 0.25, 0.5, 0.75, 0.95])
    weight_rows.append({
        "row_type": "summary",
        "direction": direction,
        "fold": fold,
        "idx": "",
        "category": "ALL",
        "label": "",
        "Q_C": "",
        "Q_D": "",
        "w_C": "",
        "w_D": "",
        "mean_w_D": float(np.mean(w_d)),
        "std_w_D": float(np.std(w_d)),
        "q05_w_D": float(qs[0]),
        "q25_w_D": float(qs[1]),
        "median_w_D": float(qs[2]),
        "q75_w_D": float(qs[3]),
        "q95_w_D": float(qs[4]),
        "WEIGHT_COLLAPSE": bool(float(np.std(w_d)) < 0.02),
    })

    audit = {
        "direction": direction,
        "fold": fold,
        "dataset": dataset_name,
        "train_categories": rotation["train_categories"],
        "heldout_categories": rotation["val_categories"],
        "evaluated_images": len(held_indices),
        "fixed_fusion_beta_clip": beta,
        "source": "reused Phase XXXIII-A support checkpoint and Phase XXX/XXIII frozen CLIP-DINO map caches",
    }
    return audit, raw_rows, pc_rows, comp_rows, weight_rows


def with_gains(raw_rows):
    base = {(r["direction"], int(r["fold"])): r for r in raw_rows if r["method"] == "S0_CLIP"}
    out = []
    for r in raw_rows:
        b = base[(r["direction"], int(r["fold"]))]
        row = dict(r)
        row["dP-AUROC_vs_CLIP"] = float(r["P-AUROC"] - b["P-AUROC"])
        row["dP-AP_vs_CLIP"] = float(r["P-AP"] - b["P-AP"])
        row["G_rank_vs_CLIP"] = 0.5 * row["dP-AUROC_vs_CLIP"] + 0.5 * row["dP-AP_vs_CLIP"]
        out.append(row)
    return out


def fixed_vs_adaptive(grows):
    by = {(r["direction"], int(r["fold"]), r["method"]): r for r in grows}
    rows = []
    for d, f in sorted({(r["direction"], int(r["fold"])) for r in grows}):
        s2 = by[(d, f, "S2_FIXED_FUSION")]
        s3 = by[(d, f, "S3_MEMORY_SOFT")]
        rows.append({
            "scope": "fold",
            "direction": d,
            "fold": f,
            "S2_G": s2["G_rank_vs_CLIP"],
            "S3_G": s3["G_rank_vs_CLIP"],
            "S3_minus_S2_G": s3["G_rank_vs_CLIP"] - s2["G_rank_vs_CLIP"],
            "S3_minus_S2_P-AUROC": s3["P-AUROC"] - s2["P-AUROC"],
            "S3_minus_S2_P-AP": s3["P-AP"] - s2["P-AP"],
        })
    for scope in ["D1", "D2", "ALL"]:
        vals = [r for r in rows if scope == "ALL" or r["direction"] == scope]
        rows.append({
            "scope": "mean",
            "direction": scope,
            "fold": "",
            "S2_G": float(np.mean([r["S2_G"] for r in vals])),
            "S3_G": float(np.mean([r["S3_G"] for r in vals])),
            "S3_minus_S2_G": float(np.mean([r["S3_minus_S2_G"] for r in vals])),
            "S3_minus_S2_P-AUROC": float(np.mean([r["S3_minus_S2_P-AUROC"] for r in vals])),
            "S3_minus_S2_P-AP": float(np.mean([r["S3_minus_S2_P-AP"] for r in vals])),
        })
    return rows


def memory_specificity(grows):
    by = {(r["direction"], int(r["fold"]), r["method"]): r for r in grows}
    rows = []
    for d, f in sorted({(r["direction"], int(r["fold"])) for r in grows}):
        s3 = by[(d, f, "S3_MEMORY_SOFT")]
        c1 = by[(d, f, "C1_SHUFFLED_SUPPORT")]
        c2 = by[(d, f, "C2_RANDOM_WEIGHT")]
        rows.append({
            "scope": "fold",
            "direction": d,
            "fold": f,
            "S3_G": s3["G_rank_vs_CLIP"],
            "C1_G": c1["G_rank_vs_CLIP"],
            "C2_G": c2["G_rank_vs_CLIP"],
            "S3_minus_C1_G": s3["G_rank_vs_CLIP"] - c1["G_rank_vs_CLIP"],
            "S3_minus_C2_G": s3["G_rank_vs_CLIP"] - c2["G_rank_vs_CLIP"],
            "S3_minus_C1_P-AUROC": s3["P-AUROC"] - c1["P-AUROC"],
            "S3_minus_C1_P-AP": s3["P-AP"] - c1["P-AP"],
            "S3_minus_C2_P-AUROC": s3["P-AUROC"] - c2["P-AUROC"],
            "S3_minus_C2_P-AP": s3["P-AP"] - c2["P-AP"],
        })
    for scope in ["D1", "D2", "ALL"]:
        vals = [r for r in rows if scope == "ALL" or r["direction"] == scope]
        rows.append({
            "scope": "mean",
            "direction": scope,
            "fold": "",
            "S3_G": float(np.mean([r["S3_G"] for r in vals])),
            "C1_G": float(np.mean([r["C1_G"] for r in vals])),
            "C2_G": float(np.mean([r["C2_G"] for r in vals])),
            "S3_minus_C1_G": float(np.mean([r["S3_minus_C1_G"] for r in vals])),
            "S3_minus_C2_G": float(np.mean([r["S3_minus_C2_G"] for r in vals])),
            "S3_minus_C1_P-AUROC": float(np.mean([r["S3_minus_C1_P-AUROC"] for r in vals])),
            "S3_minus_C1_P-AP": float(np.mean([r["S3_minus_C1_P-AP"] for r in vals])),
            "S3_minus_C2_P-AUROC": float(np.mean([r["S3_minus_C2_P-AUROC"] for r in vals])),
            "S3_minus_C2_P-AP": float(np.mean([r["S3_minus_C2_P-AP"] for r in vals])),
        })
    return rows


def per_category_summary(pc_rows):
    by = {(r["direction"], int(r["fold"]), r["category"], r["method"]): r for r in pc_rows}
    rows = []
    keys = sorted({(r["direction"], int(r["fold"]), r["category"]) for r in pc_rows})
    for d, f, cat in keys:
        s2 = by[(d, f, cat, "S2_FIXED_FUSION")]
        s3 = by[(d, f, cat, "S3_MEMORY_SOFT")]
        delta_g = 0.5 * ((s3["P-AUROC"] - s2["P-AUROC"]) + (s3["P-AP"] - s2["P-AP"]))
        outcome = "W" if delta_g > TIE_PP else ("L" if delta_g < -TIE_PP else "T")
        for method in ["S0_CLIP", "S1_DINO", "S2_FIXED_FUSION", "S3_MEMORY_SOFT", "C1_SHUFFLED_SUPPORT"]:
            r = by[(d, f, cat, method)]
            rows.append({
                **r,
                "S3_minus_S2_G_for_category": delta_g if method == "S3_MEMORY_SOFT" else "",
                "S3_vs_S2_WTL": outcome if method == "S3_MEMORY_SOFT" else "",
            })
    positive = [r for r in rows if r["method"] == "S3_MEMORY_SOFT" and isinstance(r["S3_minus_S2_G_for_category"], float) and r["S3_minus_S2_G_for_category"] > 0]
    total_pos = sum(r["S3_minus_S2_G_for_category"] for r in positive)
    for r in rows:
        if r["method"] == "S3_MEMORY_SOFT" and total_pos > 0 and isinstance(r["S3_minus_S2_G_for_category"], float):
            r["positive_gain_share"] = max(0.0, r["S3_minus_S2_G_for_category"]) / total_pos
            r["category_concentration_gt50pct"] = bool(r["positive_gain_share"] > 0.50)
        else:
            r["positive_gain_share"] = ""
            r["category_concentration_gt50pct"] = ""
    return rows


def parity_audit(grows):
    old = {}
    for r in load_csv(AUDIT / "ROUTING_METRICS.csv"):
        if r["variant"] == "R2_MEMORY_SUPPORT_SOFT_ROUTER":
            old[(r["direction"], int(r["fold"]))] = {"P-AUROC": float(r["P-AUROC"]), "P-AP": float(r["P-AP"]), "G_rank": float(r["G_rank"])}
    new = {(r["direction"], int(r["fold"])): r for r in grows if r["method"] == "S3_MEMORY_SOFT"}
    rows = []
    max_abs = 0.0
    for key in sorted(old):
        nr = new[key]
        deltas = {
            "P-AUROC": float(nr["P-AUROC"] - old[key]["P-AUROC"]),
            "P-AP": float(nr["P-AP"] - old[key]["P-AP"]),
            "G_rank": float(nr["G_rank_vs_CLIP"] - old[key]["G_rank"]),
        }
        max_abs = max(max_abs, max(abs(v) for v in deltas.values()))
        rows.append({"direction": key[0], "fold": key[1], "old": old[key], "new": {"P-AUROC": nr["P-AUROC"], "P-AP": nr["P-AP"], "G_rank": nr["G_rank_vs_CLIP"]}, "delta": deltas})
    return {"threshold": 1e-6, "max_abs_delta": max_abs, "passed": bool(max_abs <= 1e-6), "folds": rows}


def decide(grows, fixed_rows, spec_rows, parity):
    if not parity["passed"]:
        return "XXXIII_A2_IMPLEMENTATION_FAILURE", {"parity": parity}
    s3 = [r for r in grows if r["method"] == "S3_MEMORY_SOFT"]
    by_dir = defaultdict(list)
    for r in s3:
        by_dir[r["direction"]].append(r)
    direction_perf = {}
    for d in sorted(by_dir):
        vals = by_dir[d]
        direction_perf[d] = {
            "mean_G": float(np.mean([r["G_rank_vs_CLIP"] for r in vals])),
            "positive_folds": int(sum(r["G_rank_vs_CLIP"] > 0 for r in vals)),
            "min_G": float(np.min([r["G_rank_vs_CLIP"] for r in vals])),
            "mean_dP-AUROC": float(np.mean([r["dP-AUROC_vs_CLIP"] for r in vals])),
            "mean_dP-AP": float(np.mean([r["dP-AP_vs_CLIP"] for r in vals])),
        }
    positive = int(sum(r["G_rank_vs_CLIP"] > 0 for r in s3))
    min_g = float(np.min([r["G_rank_vs_CLIP"] for r in s3]))
    n = len(s3)
    required_pos = 5 if n == 6 else 3
    dir_pass = {
        d: (
            v["mean_G"] >= 0.30
            and v["mean_dP-AUROC"] >= -0.10
            and v["mean_dP-AP"] >= -0.10
        )
        for d, v in direction_perf.items()
    }
    primary = all(dir_pass.values()) and positive >= required_pos and min_g >= -0.20
    if not primary:
        if any(dir_pass.values()) and not all(dir_pass.values()):
            return "XXXIII_A2_DIRECTION_SPECIFIC_SOFT_GAIN", {"direction_perf": direction_perf, "positive_folds": positive, "min_G": min_g}
        return "XXXIII_A2_SOFT_GAIN_NOT_REPRODUCIBLE", {"direction_perf": direction_perf, "positive_folds": positive, "min_G": min_g}

    fixed_mean = next(r for r in fixed_rows if r["scope"] == "mean" and r["direction"] == "ALL")
    fixed_d1 = next(r for r in fixed_rows if r["scope"] == "mean" and r["direction"] == "D1")
    fixed_d2 = next(r for r in fixed_rows if r["scope"] == "mean" and r["direction"] == "D2")
    fixed_ok = (
        fixed_mean["S3_minus_S2_G"] >= 0.20
        and fixed_d1["S3_minus_S2_G"] >= -0.10
        and fixed_d2["S3_minus_S2_G"] >= -0.10
    )
    if not fixed_ok:
        return "XXXIII_A2_FIXED_FUSION_EXPLAINS_GAIN", {"direction_perf": direction_perf, "fixed_vs_adaptive": {"ALL": fixed_mean, "D1": fixed_d1, "D2": fixed_d2}}

    spec_all = next(r for r in spec_rows if r["scope"] == "mean" and r["direction"] == "ALL")
    spec_d1 = next(r for r in spec_rows if r["scope"] == "mean" and r["direction"] == "D1")
    spec_d2 = next(r for r in spec_rows if r["scope"] == "mean" and r["direction"] == "D2")
    causal_ok = (
        spec_all["S3_minus_C1_G"] >= 0.20
        and spec_all["S3_minus_C2_G"] >= 0.20
        and spec_d1["S3_minus_C1_G"] > 0
        and spec_d1["S3_minus_C2_G"] > 0
        and spec_d2["S3_minus_C1_G"] > 0
        and spec_d2["S3_minus_C2_G"] > 0
    )
    if not causal_ok:
        return "XXXIII_A2_MEMORY_IDENTITY_NOT_SPECIFIC", {"direction_perf": direction_perf, "memory_specificity": {"ALL": spec_all, "D1": spec_d1, "D2": spec_d2}}
    return "PASS_TO_MEMORY_CONDITIONED_SOFT_ROUTING", {"direction_perf": direction_perf, "fixed_vs_adaptive": {"ALL": fixed_mean, "D1": fixed_d1, "D2": fixed_d2}, "memory_specificity": {"ALL": spec_all, "D1": spec_d1, "D2": spec_d2}}


def write_report(outcome, detail, grows, fixed_rows, spec_rows, weight_rows, parity, fold2_status, runtime_sec):
    lines = [
        "# Phase XXXIII-A2 Memory-Conditioned Soft Routing Adjudication",
        "",
        "Frozen Phase XXXIII-A outcome remains: **XXXIII_A_NO_CARRIER_HEADROOM**.",
        "",
        f"Outcome: **{outcome}**",
        "",
        "Scope: source-only D1/D2 fold0/fold1. No training, target data, new support features, K search, temperature tuning, category routing, PRO gate, feature extraction, or memory reconstruction.",
        "",
        f"Fold2 status: {fold2_status}",
        "",
        "## Raw Metrics",
        "",
        "| Direction | Fold | Method | P-AUROC | P-AP | G vs CLIP |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in grows:
        lines.append(f"| {r['direction']} | {r['fold']} | {r['method']} | {r['P-AUROC']:.4f} | {r['P-AP']:.4f} | {r['G_rank_vs_CLIP']:+.4f} |")
    lines.extend(["", "## S3 Parity vs XXXIII-A R2", "", f"- Passed: {parity['passed']}; max_abs_delta={parity['max_abs_delta']:.8g}; threshold={parity['threshold']}."])
    lines.extend(["", "## Fixed vs Adaptive", ""])
    for r in fixed_rows:
        if r["scope"] == "mean":
            lines.append(f"- {r['direction']}: S3-S2 G={r['S3_minus_S2_G']:+.4f}, dP-AUROC={r['S3_minus_S2_P-AUROC']:+.4f}, dP-AP={r['S3_minus_S2_P-AP']:+.4f}.")
    lines.extend(["", "## Memory Specificity", ""])
    for r in spec_rows:
        if r["scope"] == "mean":
            lines.append(f"- {r['direction']}: S3-C1 G={r['S3_minus_C1_G']:+.4f}, S3-C2 G={r['S3_minus_C2_G']:+.4f}.")
    lines.extend(["", "## Support Weight Audit", ""])
    for r in weight_rows:
        if r["row_type"] == "summary":
            lines.append(f"- {r['direction']} fold{r['fold']}: mean w_D={r['mean_w_D']:.4f}, std={r['std_w_D']:.4f}, q05/q25/median/q75/q95={r['q05_w_D']:.4f}/{r['q25_w_D']:.4f}/{r['median_w_D']:.4f}/{r['q75_w_D']:.4f}/{r['q95_w_D']:.4f}, WEIGHT_COLLAPSE={r['WEIGHT_COLLAPSE']}.")
    lines.extend(["", "## Decision Detail", "", "```json", json.dumps(detail, indent=2, default=json_default), "```", "", f"Runtime: {runtime_sec:.1f}s.", "", "Stop: Phase XXXIII-A2 ended here; no follow-up mechanism was started."])
    (OUT / "XXXIII_A2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    betas = load_selected_beta()
    folds = []
    for d in ["D1", "D2"]:
        folds.extend([(d, 0), (d, 1)])
    fold2_d1 = fold2_available("D1")
    fold2_d2 = fold2_available("D2")
    if fold2_d1 and fold2_d2:
        folds.extend([("D1", 2), ("D2", 2)])
        fold2_status = "included D1/D2 fold2 because matching map caches and XXXIII-A support checkpoints exist"
    else:
        fold2_status = f"not included; same-protocol availability D1={fold2_d1}, D2={fold2_d2}"

    audits, raw, pc, comp, weights = [], [], [], [], []
    for d, f in folds:
        print(f"[XXXIII-A2] evaluating {d} fold{f}", flush=True)
        audit, rr, pp, cc, ww = eval_fold(d, f, betas[(d, f)])
        audits.append(audit)
        raw.extend(rr)
        pc.extend(pp)
        comp.extend(cc)
        weights.extend(ww)
        write_csv(OUT / "RAW_METRICS.running.csv", with_gains(raw))

    raw, pc = overlay_exact_fixed_fusion(with_gains(raw), pc)
    grows = raw
    fixed_rows = fixed_vs_adaptive(grows)
    spec_rows = memory_specificity(grows)
    pc_summary = per_category_summary(pc)
    parity = parity_audit(grows)
    outcome, detail = decide(grows, fixed_rows, spec_rows, parity)
    runtime_sec = time.time() - start

    write_csv(OUT / "RAW_METRICS.csv", grows)
    write_csv(OUT / "GROUPED_GAINS.csv", grows)
    write_csv(OUT / "FIXED_VS_ADAPTIVE.csv", fixed_rows)
    write_csv(OUT / "MEMORY_SPECIFICITY.csv", spec_rows)
    write_csv(OUT / "SUPPORT_WEIGHT_STATS.csv", weights)
    write_csv(OUT / "PER_CATEGORY.csv", pc_summary)
    write_csv(OUT / "SOFT_COMPLEMENTARITY.csv", comp)
    (OUT / "PARITY_AUDIT.json").write_text(json.dumps({
        "phase": "XXXIII-A2",
        "parity": parity,
        "fold2_status": fold2_status,
        "audits": audits,
        "restrictions": {
            "training": False,
            "target_data": False,
            "new_support_feature": False,
            "new_k": False,
            "weight_tuning": False,
            "temperature_search": False,
            "category_routing": False,
            "pro_gate": False,
            "new_clip_dino_features": False,
            "memory_reconstruction_change": False,
        },
        "runtime_sec": runtime_sec,
    }, indent=2, default=json_default), encoding="utf-8")
    write_report(outcome, detail, grows, fixed_rows, spec_rows, weights, parity, fold2_status, runtime_sec)
    (OUT / "OUTCOME.json").write_text(json.dumps({
        "phase": "XXXIII-A2",
        "outcome": outcome,
        "frozen_parent_outcome": "XXXIII_A_NO_CARRIER_HEADROOM",
        "target_accessed": False,
        "trained": False,
        "pro_used_for_decision": False,
        "folds": [{"direction": d, "fold": f} for d, f in folds],
        "fold2_status": fold2_status,
        "parity_passed": parity["passed"],
        "runtime_sec": runtime_sec,
        "decision_detail": detail,
    }, indent=2, default=json_default), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXXIII-A2\n")
        fp.write("- Command: python experiments/phase_xxxiii_a2_soft_adjudication.py\n")
        fp.write("- Scope: source-only memory-conditioned soft routing adjudication against CLIP, DINO, frozen fixed fusion, shuffled support, and random matched weights.\n")
        fp.write(f"- Fold2: {fold2_status}.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXXIII_A2_REPORT.md"), "runtime_sec": runtime_sec}, indent=2))


if __name__ == "__main__":
    main()
