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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xxxiii_memory_carrier_routing/audit")
FEATURE_CACHE = Path("results/phase_xxiii_carrier_audit/feature_cache")
MAP_CACHE = Path("results/phase_xxx_metric_safe_dino/map_cache")
IMAGE_SIZE = 518
SIDE = 37
K = 5
K_PER_IMAGE = 4
SIGMA = 4
REF_PATCHES_PER_NORMAL_IMAGE = 128


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


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fps_indices(tokens, k):
    x = norm(tokens)
    n = x.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int64)
    centroid = norm(x.mean(axis=0, keepdims=True))[0]
    first = int(np.argmin(x @ centroid))
    selected = [first]
    max_sim = x @ x[first]
    for _ in range(1, k):
        idx = int(np.argmin(max_sim))
        selected.append(idx)
        max_sim = np.maximum(max_sim, x @ x[idx])
    return np.asarray(selected, dtype=np.int64)


def load_fold(direction, fold):
    dataset_name, data_path, _ = direction_cfg(direction)
    dataset = load_dataset(dataset_name, data_path)
    rot = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rot["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rot["val_categories"]))
    return dataset_name, dataset, rot, train_indices, held_indices


def selected_memory(dataset, train_indices, clip_train, dino_train):
    clip_mem, dino_mem, meta = [], [], []
    for pos, idx in enumerate(train_indices):
        item = dataset[idx]
        if int(item["anomaly"]) != 0:
            continue
        ids = fps_indices(clip_train[pos].astype(np.float32), K_PER_IMAGE)
        clip_mem.append(norm(clip_train[pos, ids].astype(np.float32)))
        dino_mem.append(norm(dino_train[pos, ids].astype(np.float32)))
        for pid in ids:
            y, x = divmod(int(pid), SIDE)
            meta.append({
                "source_idx": idx,
                "source_category": item["cls_name"],
                "patch_id": int(pid),
                "coord_y": float((y + 0.5) / SIDE),
                "coord_x": float((x + 0.5) / SIDE),
            })
    return np.concatenate(clip_mem, axis=0), np.concatenate(dino_mem, axis=0), meta


def topk_stats_for_patches(patches, memory, meta, refs=None):
    x = norm(patches.reshape(-1, patches.shape[-1]).astype(np.float32))
    m = norm(memory)
    n = x.shape[0]
    top_dist = np.zeros((n, K), dtype=np.float32)
    top_idx = np.zeros((n, K), dtype=np.int32)
    for start in range(0, n, 8192):
        sims = x[start : start + 8192] @ m.T
        idx = np.argpartition(-sims, kth=min(K, sims.shape[1]) - 1, axis=1)[:, :K]
        vals = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        top_dist[start : start + vals.shape[0]] = 1.0 - vals
        top_idx[start : start + vals.shape[0]] = idx
    out = {
        "top1": top_dist[:, 0],
        "top5_mean": top_dist.mean(axis=1),
        "top5_std": top_dist.std(axis=1),
        "margin": top_dist[:, 1] - top_dist[:, 0] if K > 1 else np.zeros(n, dtype=np.float32),
        "top_idx": top_idx,
    }
    if refs is not None:
        for key in ["top1", "top5_mean", "top5_std"]:
            ref = refs[key]
            out[key + "_pct"] = np.searchsorted(ref, out[key], side="right").astype(np.float32) / max(1, ref.size)
    return out


def reference_patch_sample(patches, max_patches=REF_PATCHES_PER_NORMAL_IMAGE):
    if patches.shape[0] <= max_patches:
        return patches
    ids = np.linspace(0, patches.shape[0] - 1, num=max_patches, dtype=np.int64)
    return patches[ids]


def build_support_reference(clip_train_normals, dino_train_normals, clip_mem, dino_mem, meta):
    c_vals = defaultdict(list)
    d_vals = defaultdict(list)
    for cp, dp in zip(clip_train_normals, dino_train_normals):
        cs = topk_stats_for_patches(reference_patch_sample(cp), clip_mem, meta)
        ds = topk_stats_for_patches(reference_patch_sample(dp), dino_mem, meta)
        for k in ["top1", "top5_mean", "top5_std"]:
            c_vals[k].append(cs[k])
            d_vals[k].append(ds[k])
    c_ref = {k: np.sort(np.concatenate(v).astype(np.float32)) for k, v in c_vals.items()}
    d_ref = {k: np.sort(np.concatenate(v).astype(np.float32)) for k, v in d_vals.items()}
    return c_ref, d_ref


def summarize_image_support(cstats, dstats, cmeta, dmeta, image_pos):
    rows = {}
    for name, arr in [
        ("C_top1_pct", cstats["top1_pct"]),
        ("C_top5_mean_pct", cstats["top5_mean_pct"]),
        ("C_top5_std_pct", cstats["top5_std_pct"]),
        ("C_margin", cstats["margin"]),
        ("D_top1_pct", dstats["top1_pct"]),
        ("D_top5_mean_pct", dstats["top5_mean_pct"]),
        ("D_top5_std_pct", dstats["top5_std_pct"]),
        ("D_margin", dstats["margin"]),
        ("X_percentile_gap_DminusC", dstats["top5_mean_pct"] - cstats["top5_mean_pct"]),
        ("X_dispersion_gap_DminusC", dstats["top5_std_pct"] - cstats["top5_std_pct"]),
        ("X_margin_gap_DminusC", dstats["margin"] - cstats["margin"]),
    ]:
        vals = arr.reshape(SIDE * SIDE)
        rows[f"{name}_q25"] = float(np.quantile(vals, 0.25))
        rows[f"{name}_median"] = float(np.quantile(vals, 0.50))
        rows[f"{name}_q75"] = float(np.quantile(vals, 0.75))
        rows[f"{name}_q90"] = float(np.quantile(vals, 0.90))
    phys_j, img_j = [], []
    for a, b in zip(cstats["top_idx"], dstats["top_idx"]):
        c_phys = {(cmeta[int(i)]["source_idx"], cmeta[int(i)]["patch_id"]) for i in a}
        d_phys = {(dmeta[int(i)]["source_idx"], dmeta[int(i)]["patch_id"]) for i in b}
        c_img = {cmeta[int(i)]["source_idx"] for i in a}
        d_img = {dmeta[int(i)]["source_idx"] for i in b}
        phys_j.append(len(c_phys & d_phys) / max(1, len(c_phys | d_phys)))
        img_j.append(len(c_img & d_img) / max(1, len(c_img | d_img)))
    rows["X_physical_patch_jaccard_median"] = float(np.median(phys_j))
    rows["X_source_image_jaccard_median"] = float(np.median(img_j))
    rows["query_image_position"] = int(image_pos)
    rows["Q_C"] = rows["C_top5_mean_pct_median"]
    rows["Q_D"] = rows["D_top5_mean_pct_median"]
    rows["SupportGap_QDminusQC"] = rows["Q_D"] - rows["Q_C"]
    return rows


def upsample_map37(m37):
    x = torch.tensor(m37, dtype=torch.float32).view(1, 1, SIDE, SIDE)
    full = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze().numpy()
    return gaussian_filter(full.astype(np.float32), sigma=SIGMA).astype(np.float32)


def evaluate_maps(dataset, held_indices, maps_by_variant):
    raw_rows, percat_rows = [], []
    for variant, maps in maps_by_variant.items():
        by_cat = defaultdict(lambda: {"y": [], "s": []})
        y_all, s_all = [], []
        for pos, idx in enumerate(held_indices):
            item = dataset[idx]
            y = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
            s = upsample_map37(maps[pos]).reshape(-1)
            y_all.append(y); s_all.append(s)
            by_cat[item["cls_name"]]["y"].append(y); by_cat[item["cls_name"]]["s"].append(s)
        y = np.concatenate(y_all); s = np.concatenate(s_all).astype(np.float32)
        raw_rows.append({"variant": variant, "P-AUROC": safe_auc(y, s), "P-AP": safe_ap(y, s)})
        for cat, vals in sorted(by_cat.items()):
            cy = np.concatenate(vals["y"]); cs = np.concatenate(vals["s"]).astype(np.float32)
            percat_rows.append({"category": cat, "variant": variant, "P-AUROC": safe_auc(cy, cs), "P-AP": safe_ap(cy, cs)})
    return raw_rows, percat_rows


def evaluate_route_maps(dataset, held_indices, c_maps, d_maps, choose_d, soft_w_d, choose_shuf, rand_choose):
    variant_names = ["R0_CLIP", "RD_DINO", "R1_MEMORY_SUPPORT_ROUTER", "R2_MEMORY_SUPPORT_SOFT_ROUTER", "C1_SHUFFLED_SUPPORT", "C2_RANDOM_ROUTER", "ORACLE_IMAGE_GT"]
    all_parts = {v: {"y": [], "s": []} for v in variant_names}
    by_cat = {v: defaultdict(lambda: {"y": [], "s": []}) for v in variant_names}
    for i, idx in enumerate(held_indices):
        item = dataset[idx]
        c = upsample_map37(c_maps[i]).reshape(-1)
        d = upsample_map37(d_maps[i]).reshape(-1)
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
        if mask.sum() > 0:
            oracle_score = d if safe_ap(mask, d) > safe_ap(mask, c) else c
        else:
            oracle_score = d if np.quantile(d, 0.90) < np.quantile(c, 0.90) else c
        scores = {
            "R0_CLIP": c,
            "RD_DINO": d,
            "R1_MEMORY_SUPPORT_ROUTER": d if choose_d[i] else c,
            "R2_MEMORY_SUPPORT_SOFT_ROUTER": (1.0 - soft_w_d[i]) * c + soft_w_d[i] * d,
            "C1_SHUFFLED_SUPPORT": d if choose_shuf[i] else c,
            "C2_RANDOM_ROUTER": d if rand_choose[i] else c,
            "ORACLE_IMAGE_GT": oracle_score,
        }
        cat = item["cls_name"]
        for v, s in scores.items():
            all_parts[v]["y"].append(mask)
            all_parts[v]["s"].append(s.astype(np.float32))
            by_cat[v][cat]["y"].append(mask)
            by_cat[v][cat]["s"].append(s.astype(np.float32))
    raw_rows, percat_rows = [], []
    for v in variant_names:
        y = np.concatenate(all_parts[v]["y"])
        s = np.concatenate(all_parts[v]["s"]).astype(np.float32)
        raw_rows.append({"variant": v, "P-AUROC": safe_auc(y, s), "P-AP": safe_ap(y, s)})
        for cat, vals in sorted(by_cat[v].items()):
            cy = np.concatenate(vals["y"])
            cs = np.concatenate(vals["s"]).astype(np.float32)
            percat_rows.append({"category": cat, "variant": v, "P-AUROC": safe_auc(cy, cs), "P-AP": safe_ap(cy, cs)})
    return raw_rows, percat_rows


def category_preference(percat_rows, support_rows):
    by = {(r["category"], r["variant"]): r for r in percat_rows}
    supp = defaultdict(list)
    for r in support_rows:
        supp[r["category"]].append(float(r["SupportGap_QDminusQC"]))
    rows = []
    for cat in sorted(supp):
        c = by[(cat, "R0_CLIP")]
        d = by[(cat, "RD_DINO")]
        pref = 0.5 * (float(d["P-AUROC"]) - float(c["P-AUROC"])) + 0.5 * (float(d["P-AP"]) - float(c["P-AP"]))
        gap = float(np.median(supp[cat]))
        rows.append({
            "category": cat,
            "P-AUROC_CLIP": c["P-AUROC"],
            "P-AP_CLIP": c["P-AP"],
            "P-AUROC_DINO": d["P-AUROC"],
            "P-AP_DINO": d["P-AP"],
            "Preference_DINO_minus_CLIP": pref,
            "SupportGap_QDminusQC": gap,
            "support_selects_DINO": bool(gap < 0),
            "empirical_better_carrier": "DINO" if pref > 0 else "CLIP",
            "support_correct": bool((gap < 0 and pref > 0) or (gap >= 0 and pref <= 0)),
        })
    return rows


def relation_rows(pref_rows, direction, fold=None, shuffled=False):
    if len(pref_rows) < 2:
        pear = spear = float("nan")
    else:
        x = -np.asarray([r["SupportGap_QDminusQC"] for r in pref_rows], dtype=np.float64)
        y = np.asarray([r["Preference_DINO_minus_CLIP"] for r in pref_rows], dtype=np.float64)
        pear = float(pearsonr(x, y).statistic)
        spear = float(spearmanr(x, y).statistic)
    acc = float(np.mean([r["support_correct"] for r in pref_rows])) if pref_rows else float("nan")
    return {"direction": direction, "fold": fold if fold is not None else "", "shuffled": shuffled, "pearson_negSupportGap_vs_preference": pear, "spearman_negSupportGap_vs_preference": spear, "category_selection_accuracy": acc, "categories": len(pref_rows)}


def run_fold(direction, fold):
    ckpt = OUT / f"fold_result_{direction}_fold{fold}.json"
    if ckpt.exists():
        return json.loads(ckpt.read_text(encoding="utf-8"))
    dataset_name, dataset, rot, train_indices, held_indices = load_fold(direction, fold)
    clip = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    dino = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_dino.npz", allow_pickle=True)
    c_train, d_train = clip["train_patch"], dino["train_patch"]
    c_held, d_held = clip["held_patch"], dino["held_patch"]
    print(f"[{direction} fold{fold}] selecting paired memory", flush=True)
    c_mem, d_mem, meta = selected_memory(dataset, train_indices, c_train, d_train)
    normal_pos = [i for i, idx in enumerate(train_indices) if int(dataset[idx]["anomaly"]) == 0]
    print(f"[{direction} fold{fold}] building support reference from {len(normal_pos)} normal images", flush=True)
    c_ref, d_ref = build_support_reference([c_train[i] for i in normal_pos], [d_train[i] for i in normal_pos], c_mem, d_mem, meta)
    support_rows = []
    agreement_rows = []
    print(f"[{direction} fold{fold}] computing held-out support summaries for {len(held_indices)} images", flush=True)
    for pos, idx in enumerate(held_indices):
        cs = topk_stats_for_patches(c_held[pos], c_mem, meta, c_ref)
        ds = topk_stats_for_patches(d_held[pos], d_mem, meta, d_ref)
        row = summarize_image_support(cs, ds, meta, meta, pos)
        item = dataset[idx]
        row.update({"direction": direction, "fold": fold, "idx": idx, "category": item["cls_name"], "label": int(item["anomaly"])})
        support_rows.append(row)
        agreement_rows.append({
            "direction": direction,
            "fold": fold,
            "idx": idx,
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "topk_physical_patch_jaccard_median": row["X_physical_patch_jaccard_median"],
            "topk_source_image_jaccard_median": row["X_source_image_jaccard_median"],
            "carrier_support_disagreement_abs": abs(row["SupportGap_QDminusQC"]),
        })

    c_maps = np.load(MAP_CACHE / f"{direction}_fold{fold}_clip_held37.npy", mmap_mode="r")
    d_maps = np.load(MAP_CACHE / f"{direction}_fold{fold}_dino_held37.npy", mmap_mode="r")
    q_c = np.asarray([r["Q_C"] for r in support_rows], dtype=np.float32)
    q_d = np.asarray([r["Q_D"] for r in support_rows], dtype=np.float32)
    choose_d = q_d < q_c
    r_c = np.exp(-q_c); r_d = np.exp(-q_d)
    w_d = r_d / (r_c + r_d)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(held_indices))
    choose_shuf = q_d[perm] < q_c[perm]
    freq = float(choose_d.mean())
    rand_choose = rng.random(len(held_indices)) < freq
    print(f"[{direction} fold{fold}] evaluating route maps", flush=True)
    raw, pc = evaluate_route_maps(dataset, held_indices, c_maps, d_maps, choose_d, w_d, choose_shuf, rand_choose)
    for r in raw:
        r.update({"direction": direction, "fold": fold, "dino_selection_frequency": freq if r["variant"] in {"R1_MEMORY_SUPPORT_ROUTER", "C2_RANDOM_ROUTER"} else ""})
    for r in pc:
        r.update({"direction": direction, "fold": fold})
    pref = category_preference([r for r in pc if r["variant"] in {"R0_CLIP", "RD_DINO"}], support_rows)
    for r in pref:
        r.update({"direction": direction, "fold": fold})
    ref_summary = {
        "direction": direction,
        "fold": fold,
        "normal_memory_entries": int(c_mem.shape[0]),
        "clip_dim": int(c_mem.shape[1]),
        "dino_dim": int(d_mem.shape[1]),
        "normal_train_images": len(normal_pos),
        "clip_reference_quantiles": {k: [float(x) for x in np.quantile(v, [0.25, 0.5, 0.75, 0.9, 0.95])] for k, v in c_ref.items()},
        "dino_reference_quantiles": {k: [float(x) for x in np.quantile(v, [0.25, 0.5, 0.75, 0.9, 0.95])] for k, v in d_ref.items()},
    }
    audit = {"direction": direction, "fold": fold, "dataset": dataset_name, "train_categories": rot["train_categories"], "heldout_categories": rot["val_categories"], "alignment": "CLIP and DINO both use 37x37 patch grids; paired by identical source image index and patch_id", "max_coordinate_alignment_error": 0.0}
    align_rows = [{"direction": direction, "fold": fold, "memory_id": i, **m, "alignment_error": 0.0} for i, m in enumerate(meta)]
    result = (audit, align_rows, ref_summary, support_rows, pref, raw, pc, agreement_rows)
    ckpt.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    return result


def grouped(raw_rows):
    base = {(r["direction"], int(r["fold"])): r for r in raw_rows if r["variant"] == "R0_CLIP"}
    out = []
    for r in raw_rows:
        b = base[(r["direction"], int(r["fold"]))]
        row = dict(r)
        row["dP-AUROC"] = r["P-AUROC"] - b["P-AUROC"]
        row["dP-AP"] = r["P-AP"] - b["P-AP"]
        row["G_rank"] = 0.5 * row["dP-AUROC"] + 0.5 * row["dP-AP"]
        out.append(row)
    return out


def relation_all(pref_rows):
    rows = []
    for d in ["D1", "D2"]:
        rows.append(relation_rows([r for r in pref_rows if r["direction"] == d], d))
    rows.append(relation_rows(pref_rows, "ALL"))
    rng = np.random.default_rng(0)
    shuf = []
    gaps = np.asarray([r["SupportGap_QDminusQC"] for r in pref_rows], dtype=np.float64)
    gaps = gaps[rng.permutation(len(gaps))]
    for r, g in zip(pref_rows, gaps):
        rr = dict(r)
        rr["SupportGap_QDminusQC"] = float(g)
        rr["support_correct"] = bool((g < 0 and rr["Preference_DINO_minus_CLIP"] > 0) or (g >= 0 and rr["Preference_DINO_minus_CLIP"] <= 0))
        shuf.append(rr)
    rows.append(relation_rows(shuf, "ALL_SHUFFLED", shuffled=True))
    return rows


def control_rows(grows):
    rows = []
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in grows}
    for d in ["D1", "D2", "ALL"]:
        keys = [(dd, f) for dd in ["D1", "D2"] for f in [0, 1]] if d == "ALL" else [(d, f) for f in [0, 1]]
        r1_c1 = [by[(dd, f, "R1_MEMORY_SUPPORT_ROUTER")]["G_rank"] - by[(dd, f, "C1_SHUFFLED_SUPPORT")]["G_rank"] for dd, f in keys]
        r1_c2 = [by[(dd, f, "R1_MEMORY_SUPPORT_ROUTER")]["G_rank"] - by[(dd, f, "C2_RANDOM_ROUTER")]["G_rank"] for dd, f in keys]
        r2_c1 = [by[(dd, f, "R2_MEMORY_SUPPORT_SOFT_ROUTER")]["G_rank"] - by[(dd, f, "C1_SHUFFLED_SUPPORT")]["G_rank"] for dd, f in keys]
        r2_c2 = [by[(dd, f, "R2_MEMORY_SUPPORT_SOFT_ROUTER")]["G_rank"] - by[(dd, f, "C2_RANDOM_ROUTER")]["G_rank"] for dd, f in keys]
        rows.append({"direction": d, "R1_minus_C1_mean_G": float(np.mean(r1_c1)), "R1_minus_C2_mean_G": float(np.mean(r1_c2)), "R2_minus_C1_mean_G": float(np.mean(r2_c1)), "R2_minus_C2_mean_G": float(np.mean(r2_c2))})
    return rows


def oracle_rows(grows):
    out = []
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in grows}
    for d in ["D1", "D2"]:
        vals = []
        for f in [0, 1]:
            vals.append(by[(d, f, "ORACLE_IMAGE_GT")]["G_rank"])
        out.append({"direction": d, "oracle_image_mean_G": float(np.mean(vals)), "oracle_image_min_G": float(np.min(vals))})
    return out


def decide(grows, rel, controls, oracle):
    oracle_mean = float(np.mean([r["oracle_image_mean_G"] for r in oracle]))
    if oracle_mean < 0.20:
        return "XXXIII_A_NO_CARRIER_HEADROOM", {}
    rel_d1 = next(r for r in rel if r["direction"] == "D1")
    rel_d2 = next(r for r in rel if r["direction"] == "D2")
    rel_all = next(r for r in rel if r["direction"] == "ALL")
    same_sign = np.sign(float(rel_d1["spearman_negSupportGap_vs_preference"])) == np.sign(float(rel_d2["spearman_negSupportGap_vs_preference"]))
    predictive = same_sign and abs(float(rel_all["spearman_negSupportGap_vs_preference"])) >= 0.30 and float(rel_all["category_selection_accuracy"]) >= 0.65
    if not predictive:
        return "XXXIII_A_MEMORY_SUPPORT_NOT_PREDICTIVE", {"same_sign": bool(same_sign), "pooled_spearman": rel_all["spearman_negSupportGap_vs_preference"], "selection_accuracy": rel_all["category_selection_accuracy"]}
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in grows}
    best_variant = None
    best_mean = -1e9
    summary = {}
    for variant in ["R1_MEMORY_SUPPORT_ROUTER", "R2_MEMORY_SUPPORT_SOFT_ROUTER"]:
        passes = {}
        all_vals = []
        for d in ["D1", "D2"]:
            vals = [by[(d, f, variant)] for f in [0, 1]]
            mean_g = float(np.mean([r["G_rank"] for r in vals]))
            passes[d] = {"mean_G": mean_g, "positive_folds": int(sum(r["G_rank"] > 0 for r in vals)), "min_G": float(min(r["G_rank"] for r in vals)), "mean_dP_AUROC": float(np.mean([r["dP-AUROC"] for r in vals])), "mean_dP_AP": float(np.mean([r["dP-AP"] for r in vals]))}
            all_vals.extend([r["G_rank"] for r in vals])
        summary[variant] = passes
        m = float(np.mean(all_vals))
        if m > best_mean:
            best_mean, best_variant = m, variant
    s = summary[best_variant]
    perf = all(s[d]["mean_G"] >= 0.20 and s[d]["min_G"] >= -0.20 and s[d]["mean_dP_AUROC"] >= -0.10 and s[d]["mean_dP_AP"] >= -0.10 for d in ["D1", "D2"])
    perf = perf and sum(by[(d, f, best_variant)]["G_rank"] > 0 for d in ["D1", "D2"] for f in [0, 1]) >= 3
    if not perf:
        if (s["D1"]["mean_G"] >= 0.20) != (s["D2"]["mean_G"] >= 0.20):
            return "XXXIII_A_DIRECTION_SPECIFIC_ROUTING_GAIN", {"best_variant": best_variant, "performance": s}
        return "XXXIII_A_ROUTING_NOT_TASK_POSITIVE", {"best_variant": best_variant, "performance": s}
    c_all = next(r for r in controls if r["direction"] == "ALL")
    if best_variant.startswith("R1"):
        specific = c_all["R1_minus_C1_mean_G"] >= 0.20 and c_all["R1_minus_C2_mean_G"] >= 0.20
    else:
        specific = c_all["R2_minus_C1_mean_G"] >= 0.20 and c_all["R2_minus_C2_mean_G"] >= 0.20
    if not specific:
        return "XXXIII_A_MEMORY_IDENTITY_NOT_SPECIFIC", {"best_variant": best_variant, "performance": s, "controls": c_all}
    return "PASS_TO_PHASE_XXXIII_B", {"best_variant": best_variant, "performance": s, "controls": c_all}


def write_report(outcome, detail, grows, rel, controls, oracle):
    lines = ["# Phase XXXIII-A Memory-Supported Carrier Preference Audit", "", f"Outcome: **{outcome}**", "", "Scope: source-only D1/D2 fold0/fold1. Paired CLIP/DINO source-normal memory is used only as a support estimator; no training, target data, semantic retrieval, new memory optimization, or PRO-based decision.", "", "## Routing Metrics", "", "| Direction | Fold | Variant | P-AUROC | P-AP | G_rank |", "|---|---:|---|---:|---:|---:|"]
    for r in grows:
        if r["variant"] == "ORACLE_IMAGE_GT":
            continue
        lines.append(f"| {r['direction']} | {r['fold']} | {r['variant']} | {r['P-AUROC']:.4f} | {r['P-AP']:.4f} | {r['G_rank']:+.4f} |")
    lines.extend(["", "## Support Preference Relation", ""])
    for r in rel:
        lines.append(f"- {r['direction']}: Spearman={float(r['spearman_negSupportGap_vs_preference']):+.4f}, Pearson={float(r['pearson_negSupportGap_vs_preference']):+.4f}, selection_acc={float(r['category_selection_accuracy']):.4f}.")
    lines.extend(["", "## Controls", ""])
    for r in controls:
        lines.append(f"- {r['direction']}: R1-C1={float(r['R1_minus_C1_mean_G']):+.4f}, R1-C2={float(r['R1_minus_C2_mean_G']):+.4f}, R2-C1={float(r['R2_minus_C1_mean_G']):+.4f}, R2-C2={float(r['R2_minus_C2_mean_G']):+.4f}.")
    lines.extend(["", "## Oracle Headroom", ""])
    for r in oracle:
        lines.append(f"- {r['direction']}: image oracle mean G={float(r['oracle_image_mean_G']):+.4f}, min={float(r['oracle_image_min_G']):+.4f}.")
    lines.extend(["", "## Decision Detail", "", f"`{json.dumps(detail, default=json_default)}`", "", "Mandatory stop: Phase XXXIII-B was not started."])
    (OUT / "XXXIII_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    audits, align, refs, supports, prefs, raw, pc, agree = [], [], [], [], [], [], [], []
    for d in ["D1", "D2"]:
        for f in [0, 1]:
            res = run_fold(d, f)
            a, al, ref, sup, pref, rr, pcr, agr = res
            audits.append(a); align.extend(al); refs.append(ref); supports.extend(sup); prefs.extend(pref); raw.extend(rr); pc.extend(pcr); agree.extend(agr)
            write_csv(OUT / "ROUTING_METRICS.running.csv", raw)
    grows = grouped(raw)
    rel = relation_all(prefs)
    controls = control_rows(grows)
    oracle = oracle_rows(grows)
    outcome, detail = decide(grows, rel, controls, oracle)
    (OUT / "PAIRED_MEMORY_AUDIT.json").write_text(json.dumps({
        "phase": "XXXIII-A",
        "target_accessed": False,
        "trained": False,
        "memory_policy": "CLIP FPS_K4 selected physical patch ids from source-train normal images; identical ids used for CLIP and DINO memories",
        "support_reference_policy": f"source-train normal query patches only; deterministic uniform stride sample up to {REF_PATCHES_PER_NORMAL_IMAGE} patches per normal image for percentile references; held-out query images use all patches for robust summaries",
        "retrieval_k": K,
        "alignment": "same 37x37 grids, identical patch_id correspondence",
        "runtime_sec": time.time() - start,
        "folds": audits,
    }, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "MEMORY_ALIGNMENT.csv", align)
    (OUT / "SUPPORT_REFERENCE.json").write_text(json.dumps(refs, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "QUERY_SUPPORT_FEATURES.csv", supports)
    write_csv(OUT / "CARRIER_PREFERENCE.csv", prefs)
    write_csv(OUT / "SUPPORT_PREFERENCE_RELATION.csv", rel)
    write_csv(OUT / "ROUTING_METRICS.csv", [r for r in grows if r["variant"] not in {"C1_SHUFFLED_SUPPORT", "C2_RANDOM_ROUTER", "ORACLE_IMAGE_GT"}])
    write_csv(OUT / "CONTROL_METRICS.csv", controls + [r for r in grows if r["variant"] in {"C1_SHUFFLED_SUPPORT", "C2_RANDOM_ROUTER"}])
    write_csv(OUT / "ORACLE_HEADROOM.csv", oracle + [r for r in grows if r["variant"] == "ORACLE_IMAGE_GT"])
    write_csv(OUT / "RETRIEVAL_AGREEMENT.csv", agree)
    write_csv(OUT / "PER_CATEGORY.csv", pc)
    write_report(outcome, detail, grows, rel, controls, oracle)
    payload = {"phase": "XXXIII-A", "outcome": outcome, "target_accessed": False, "trained": False, "pro_used_for_decision": False, "phase_xxxiii_b_started": False, "runtime_sec": time.time() - start, "decision_detail": detail}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXXIII-A\n")
        fp.write("- Command: python experiments/phase_xxxiii_memory_carrier_routing.py\n")
        fp.write("- Scope: D1 fold0/fold1 and D2 fold0/fold1 paired CLIP/DINO source-normal memory support carrier-routing audit; no target; no training.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXXIII_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
