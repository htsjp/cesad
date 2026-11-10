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
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xxxii_hard_normal_memory/audit")
FEATURE_CACHE = Path("results/phase_xxiii_carrier_audit/feature_cache")
OFFICIAL_MAP_CACHE = Path("results/phase_xxiii_dual_carrier/local_integration/map_cache")
IMAGE_SIZE = 518
SIDE = 37
SIGMA = 4
TOPK_RATIO = 0.01
FUSION_K = 0.7
K_PER_IMAGE = 4
VARIANTS = [
    "M0_STATIC_FPS",
    "M1_STATIC_CORESET",
    "M2_RANDOM_NORMAL",
    "M3_LOW_NOVELTY_NORMAL",
    "M4_HARD_NORMAL_EVOLUTION",
    "M5_ONE_SHOT_HARD_NORMAL",
]


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


def normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fps_indices(tokens, k):
    x = normalize_np(tokens)
    n = x.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int64)
    centroid = normalize_np(x.mean(axis=0, keepdims=True))[0]
    first = int(np.argmin(x @ centroid))
    selected = [first]
    max_sim = x @ x[first]
    for _ in range(1, k):
        idx = int(np.argmin(max_sim))
        selected.append(idx)
        max_sim = np.maximum(max_sim, x @ x[idx])
    return np.asarray(selected, dtype=np.int64)


def coreset_indices(tokens, k):
    return fps_indices(tokens, k)


def random_indices(tokens, k, seed):
    rng = np.random.default_rng(seed)
    n = tokens.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)


def novelty(tokens, memory):
    x = normalize_np(tokens)
    m = normalize_np(memory)
    best = np.full(x.shape[0], -1.0, dtype=np.float32)
    for start in range(0, m.shape[0], 4096):
        best = np.maximum(best, (x @ m[start : start + 4096].T).max(axis=1))
    return 1.0 - best


def load_fold(direction, fold):
    dataset_name, data_path, _ = direction_cfg(direction)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
    return dataset_name, dataset, rotation, train_indices, held_indices


def resize_mask37(mask):
    x = torch.tensor(mask, dtype=torch.float32).view(1, 1, *mask.shape[-2:])
    return F.interpolate(x, size=(SIDE, SIDE), mode="bilinear", align_corners=False).squeeze().numpy() > 0.5


def build_candidates(dataset, train_indices, train_patch):
    normal_images, anomaly_keys, normal_meta = [], [], []
    for pos, idx in enumerate(train_indices):
        item = dataset[idx]
        patch = normalize_np(train_patch[pos].astype(np.float32))
        if int(item["anomaly"]) == 0:
            normal_images.append({"idx": idx, "category": item["cls_name"], "patch": patch})
            normal_meta.append({"idx": idx, "category": item["cls_name"]})
        else:
            mask = resize_mask37(item["img_mask"].detach().cpu().numpy())
            if mask.any():
                key = normalize_np(patch[mask.reshape(-1)].mean(axis=0, keepdims=True))[0]
                anomaly_keys.append(key)
    if not anomaly_keys:
        anomaly_keys = [np.zeros((train_patch.shape[-1],), dtype=np.float32)]
    return normal_images, np.stack(anomaly_keys).astype(np.float32), normal_meta


def select_static(normal_images, mode):
    parts, build_rows = [], []
    for i, im in enumerate(normal_images):
        if mode == "fps":
            sel = fps_indices(im["patch"], K_PER_IMAGE)
        elif mode == "coreset":
            sel = coreset_indices(im["patch"], K_PER_IMAGE)
        elif mode == "random":
            sel = random_indices(im["patch"], K_PER_IMAGE, seed=i)
        else:
            raise ValueError(mode)
        parts.append(im["patch"][sel])
        build_rows.append({"source_image": im["idx"], "category": im["category"], "selected": len(sel), "mode": mode})
    return normalize_np(np.concatenate(parts, axis=0)), build_rows


def select_evolution(normal_images, hard=True, one_shot=False):
    selected = [fps_indices(im["patch"], 1).tolist() for im in normal_images]
    memory = normalize_np(np.concatenate([im["patch"][sel] for im, sel in zip(normal_images, selected)], axis=0))
    rows = [{"round": 0, "selected_total": int(memory.shape[0]), "mode": "init_fps25"}]
    rounds = [1, 2, 3] if not one_shot else [1]
    add_per_round = 1 if not one_shot else 3
    for rnd in rounds:
        additions = []
        novelty_stats = []
        for i, im in enumerate(normal_images):
            used = set(selected[i])
            remaining = np.asarray([j for j in range(im["patch"].shape[0]) if j not in used], dtype=np.int64)
            if remaining.size == 0:
                continue
            scores = novelty(im["patch"][remaining], memory)
            n_keep = max(add_per_round, int(math.ceil(remaining.size * 0.10)))
            order = np.argsort(-scores if hard else scores, kind="mergesort")[:n_keep]
            pool = remaining[order]
            pool_scores = scores[order]
            take = np.argsort(-pool_scores if hard else pool_scores, kind="mergesort")[:add_per_round]
            chosen = pool[take].tolist()
            additions.extend((i, j) for j in chosen)
            novelty_stats.extend(pool_scores[take].tolist())
        for i, j in additions:
            selected[i].append(int(j))
        memory = normalize_np(np.concatenate([im["patch"][sel] for im, sel in zip(normal_images, selected)], axis=0))
        rows.append({"round": rnd, "selected_total": int(memory.shape[0]), "mean_added_novelty": float(np.mean(novelty_stats)) if novelty_stats else float("nan"), "mode": "hard" if hard else "low"})
    return memory, rows


def project_keys(patch_proj, keys, device):
    outs = []
    with torch.no_grad():
        for start in range(0, keys.shape[0], 4096):
            k = torch.tensor(keys[start : start + 4096], dtype=torch.float32, device=device)
            outs.append(patch_proj(k, 1).detach().cpu())
    return torch.cat(outs, dim=0)


def project_queries(patch_proj, patches, device):
    outs = []
    flat = patches.reshape(-1, patches.shape[-1]).astype(np.float32)
    with torch.no_grad():
        for start in range(0, flat.shape[0], 8192):
            q = torch.tensor(flat[start : start + 8192], dtype=torch.float32, device=device)
            outs.append(patch_proj(q, 0).detach().cpu())
    return torch.cat(outs, dim=0).reshape(patches.shape[0], patches.shape[1], -1)


def official_cls_memory(direction, fold):
    raw = np.load(OFFICIAL_MAP_CACHE / f"{direction}_fold{fold}_official_held_raw.npy", mmap_mode="r")
    scores = np.load(OFFICIAL_MAP_CACHE / f"{direction}_fold{fold}_official_held_scores.npy")
    tops = []
    for i in range(raw.shape[0]):
        m = gaussian_filter(np.asarray(raw[i], dtype=np.float32), sigma=SIGMA).reshape(-1)
        k = max(1, int(m.size * TOPK_RATIO))
        tops.append(float(np.sort(m)[-k:].mean()))
    tops = np.asarray(tops, dtype=np.float32)
    return (scores.astype(np.float32) - (1.0 - FUSION_K) * tops) / FUSION_K


def evaluate_variant(dataset, held_indices, held_patch, q_proj, keys_proj, values, cls_memory, normal_memory):
    all_y, all_s = [], []
    by_cat = defaultdict(lambda: {"y": [], "s": [], "labels": [], "scores": []})
    maps37 = []
    with torch.no_grad():
        k = keys_proj.to(torch.float32)
        vals = torch.tensor(values, dtype=torch.float32)
        for i in range(q_proj.shape[0]):
            q = q_proj[i].to(torch.float32)
            logits = q @ k.T
            prob = torch.softmax(logits, dim=-1) @ vals
            amap37 = prob[:, 1].reshape(SIDE, SIDE).numpy().astype(np.float32)
            maps37.append(amap37)
    image_scores = []
    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        full = F.interpolate(torch.tensor(maps37[pos]).view(1, 1, SIDE, SIDE), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze().numpy()
        full = gaussian_filter(full.astype(np.float32), sigma=SIGMA).astype(np.float32)
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
        score = full.reshape(-1)
        all_y.append(mask)
        all_s.append(score)
        ktop = max(1, int(score.size * TOPK_RATIO))
        img_score = (1.0 - FUSION_K) * float(np.sort(score)[-ktop:].mean()) + FUSION_K * float(cls_memory[pos])
        image_scores.append(img_score)
        cat = item["cls_name"]
        by_cat[cat]["y"].append(mask)
        by_cat[cat]["s"].append(score)
        by_cat[cat]["labels"].append(int(item["anomaly"]))
        by_cat[cat]["scores"].append(img_score)
    percat = {}
    p_aucs, p_aps, i_aucs, i_aps = [], [], [], []
    for cat, v in sorted(by_cat.items()):
        y = np.concatenate(v["y"])
        s = np.concatenate(v["s"]).astype(np.float32)
        pa, pap = safe_auc(y, s), safe_ap(y, s)
        ia, iap = safe_auc(v["labels"], v["scores"]), safe_ap(v["labels"], v["scores"])
        percat[cat] = {"P-AUROC": pa, "P-AP": pap, "I-AUROC": ia, "I-AP": iap}
        if np.isfinite(pa): p_aucs.append(pa)
        if np.isfinite(pap): p_aps.append(pap)
        if np.isfinite(ia): i_aucs.append(ia)
        if np.isfinite(iap): i_aps.append(iap)
    return {
        "I-AUROC": float(np.mean(i_aucs)) if i_aucs else float("nan"),
        "I-AP": float(np.mean(i_aps)) if i_aps else float("nan"),
        "P-AUROC": float(np.mean(p_aucs)) if p_aucs else float("nan"),
        "P-AP": float(np.mean(p_aps)) if p_aps else float("nan"),
        "per_category": percat,
        "maps37": np.stack(maps37),
    }


def coverage_geometry(variant, normal_memory, normal_images):
    sample = np.concatenate([im["patch"][:: max(1, im["patch"].shape[0] // 64)] for im in normal_images], axis=0)
    dist = novelty(sample, normal_memory)
    sims = normal_memory @ normal_memory.T
    tri = sims[np.triu_indices_from(sims, k=1)]
    s = np.linalg.svd(normal_memory - normal_memory.mean(axis=0, keepdims=True), compute_uv=False)
    p = s / max(1e-12, s.sum())
    return {
        "variant": variant,
        "avg_candidate_nearest_distance": float(np.mean(dist)),
        "p95_candidate_nearest_distance": float(np.quantile(dist, 0.95)),
        "max_candidate_nearest_distance": float(np.max(dist)),
        "effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum())),
        "memory_pairwise_cosine_mean": float(np.mean(tri)) if tri.size else float("nan"),
        "memory_pairwise_cosine_std": float(np.std(tri)) if tri.size else float("nan"),
        "memory_redundancy_cos_gt_0p95": float(np.mean(tri > 0.95)) if tri.size else float("nan"),
        "source_image_coverage": len(normal_images),
        "source_category_coverage": len(set(im["category"] for im in normal_images)),
    }


def boundary_and_separation(dataset, held_indices, held_patch, normal_memory):
    normal_d, anomaly_d = [], []
    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        patch = normalize_np(held_patch[pos].astype(np.float32))
        d = novelty(patch, normal_memory)
        if int(item["anomaly"]) == 0:
            normal_d.extend(d.tolist())
        else:
            mask = resize_mask37(item["img_mask"].detach().cpu().numpy()).reshape(-1)
            if mask.any():
                anomaly_d.extend(d[mask].tolist())
            normal_d.extend(d[~mask].tolist())
    normal_d = np.asarray(normal_d, dtype=np.float32)
    anomaly_d = np.asarray(anomaly_d, dtype=np.float32)
    pooled = math.sqrt(0.5 * (float(normal_d.var()) + float(anomaly_d.var()))) if anomaly_d.size else float("nan")
    return (
        {"normal_q90": float(np.quantile(normal_d, 0.90)), "normal_q95": float(np.quantile(normal_d, 0.95)), "normal_q99": float(np.quantile(normal_d, 0.99))},
        {
            "normal_mean": float(normal_d.mean()),
            "normal_std": float(normal_d.std()),
            "anomaly_mean": float(anomaly_d.mean()) if anomaly_d.size else float("nan"),
            "anomaly_std": float(anomaly_d.std()) if anomaly_d.size else float("nan"),
            "cohen_d_anomaly_minus_normal": float((anomaly_d.mean() - normal_d.mean()) / pooled) if anomaly_d.size and pooled > 0 else float("nan"),
            "ks_distance": float(ks_2samp(normal_d, anomaly_d).statistic) if anomaly_d.size else float("nan"),
        },
    )


def run_fold(direction, fold, device):
    start = time.time()
    dataset_name, dataset, rotation, train_indices, held_indices = load_fold(direction, fold)
    _, _, _, patch_proj, _ = load_components(direction_cfg(direction)[2], device)
    feat = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_patch = feat["train_patch"]
    held_patch = feat["held_patch"]
    normal_images, anomaly_keys, _ = build_candidates(dataset, train_indices, train_patch)
    budget = K_PER_IMAGE * len(normal_images)
    cls_mem = official_cls_memory(direction, fold)
    q_proj = project_queries(patch_proj, held_patch, device)

    variants = {}
    rows_build = []
    variants["M0_STATIC_FPS"], r = select_static(normal_images, "fps"); rows_build.extend({**x, "variant": "M0_STATIC_FPS"} for x in r)
    variants["M1_STATIC_CORESET"], r = select_static(normal_images, "coreset"); rows_build.extend({**x, "variant": "M1_STATIC_CORESET"} for x in r)
    variants["M2_RANDOM_NORMAL"], r = select_static(normal_images, "random"); rows_build.extend({**x, "variant": "M2_RANDOM_NORMAL"} for x in r)
    variants["M3_LOW_NOVELTY_NORMAL"], r = select_evolution(normal_images, hard=False); rows_build.extend({**x, "variant": "M3_LOW_NOVELTY_NORMAL"} for x in r)
    variants["M4_HARD_NORMAL_EVOLUTION"], r = select_evolution(normal_images, hard=True); rows_build.extend({**x, "variant": "M4_HARD_NORMAL_EVOLUTION"} for x in r)
    variants["M5_ONE_SHOT_HARD_NORMAL"], r = select_evolution(normal_images, hard=True, one_shot=True); rows_build.extend({**x, "variant": "M5_ONE_SHOT_HARD_NORMAL"} for x in r)

    raw_rows, percat_rows, coverage_rows, boundary_rows, sep_rows, parity_rows = [], [], [], [], [], []
    for variant, normal_mem in variants.items():
        keys = np.concatenate([normal_mem, anomaly_keys], axis=0).astype(np.float32)
        values = np.zeros((keys.shape[0], 2), dtype=np.float32)
        values[: normal_mem.shape[0], 0] = 1.0
        values[normal_mem.shape[0] :, 1] = 1.0
        keys_proj = project_keys(patch_proj, keys, device)
        met = evaluate_variant(dataset, held_indices, held_patch, q_proj, keys_proj, values, cls_mem, normal_mem)
        raw_rows.append({"direction": direction, "fold": fold, "variant": variant, "I-AUROC": met["I-AUROC"], "I-AP": met["I-AP"], "P-AUROC": met["P-AUROC"], "P-AP": met["P-AP"], "normal_memory_entries": normal_mem.shape[0], "anomaly_memory_entries": anomaly_keys.shape[0]})
        for cat, vals in met["per_category"].items():
            percat_rows.append({"direction": direction, "fold": fold, "category": cat, "variant": variant, **vals})
        cov = coverage_geometry(variant, normal_mem, normal_images)
        coverage_rows.append({"direction": direction, "fold": fold, **cov})
        bnd, sep = boundary_and_separation(dataset, held_indices, held_patch, normal_mem)
        boundary_rows.append({"direction": direction, "fold": fold, "variant": variant, **bnd})
        sep_rows.append({"direction": direction, "fold": fold, "variant": variant, **sep})
        parity_rows.append({"direction": direction, "fold": fold, "variant": variant, "normal_memory_entries": normal_mem.shape[0], "expected_normal_budget_B": budget, "memory_size_parity": normal_mem.shape[0] == budget, "bytes_normal_memory": int(normal_mem.nbytes), "relative_inference_query_count": int(len(held_indices) * SIDE * SIDE)})

    return {
        "audit": {"direction": direction, "fold": fold, "dataset": dataset_name, "train_categories": rotation["train_categories"], "heldout_categories": rotation["val_categories"], "train_images": len(train_indices), "heldout_images": len(held_indices), "normal_source_images": len(normal_images), "normal_budget_B": budget, "anomaly_entries_preserved": int(anomaly_keys.shape[0]), "runtime_sec": time.time() - start},
        "budget": {"direction": direction, "fold": fold, "B": budget, "B_definition": "4 normal patch entries per source-train normal image, matching prior FPS_K4 local-memory budget", "initial_budget": budget // 4, "round_additions": budget // 4, "rounds": 3},
        "build": [{"direction": direction, "fold": fold, **r} for r in rows_build],
        "raw": raw_rows,
        "percat": percat_rows,
        "coverage": coverage_rows,
        "boundary": boundary_rows,
        "separation": sep_rows,
        "parity": parity_rows,
    }


def grouped(raw_rows):
    base = {(r["direction"], int(r["fold"])): r for r in raw_rows if r["variant"] == "M0_STATIC_FPS"}
    out = []
    for r in raw_rows:
        b = base[(r["direction"], int(r["fold"]))]
        row = dict(r)
        row["dP-AUROC"] = r["P-AUROC"] - b["P-AUROC"]
        row["dP-AP"] = r["P-AP"] - b["P-AP"]
        row["dI-AUROC"] = r["I-AUROC"] - b["I-AUROC"]
        row["dI-AP"] = r["I-AP"] - b["I-AP"]
        row["G_pixel_rank"] = 0.5 * row["dP-AUROC"] + 0.5 * row["dP-AP"]
        row["G_image_rank"] = 0.5 * row["dI-AUROC"] + 0.5 * row["dI-AP"]
        row["G_primary"] = row["G_pixel_rank"]
        out.append(row)
    return out


def percat_wtl(percat_rows):
    base = {(r["direction"], int(r["fold"]), r["category"]): r for r in percat_rows if r["variant"] == "M0_STATIC_FPS"}
    out = []
    gains = []
    for r in percat_rows:
        row = dict(r)
        b = base[(r["direction"], int(r["fold"]), r["category"])]
        row["dP-AUROC_vs_M0"] = r["P-AUROC"] - b["P-AUROC"]
        row["dP-AP_vs_M0"] = r["P-AP"] - b["P-AP"]
        row["G_pixel_vs_M0"] = 0.5 * row["dP-AUROC_vs_M0"] + 0.5 * row["dP-AP_vs_M0"]
        row["WTL_vs_M0"] = "T" if abs(row["G_pixel_vs_M0"]) <= 0.05 else ("W" if row["G_pixel_vs_M0"] > 0 else "L")
        if r["variant"] == "M4_HARD_NORMAL_EVOLUTION" and row["G_pixel_vs_M0"] > 0:
            gains.append(row["G_pixel_vs_M0"])
        out.append(row)
    concentrated = bool(gains and max(gains) / max(1e-12, sum(gains)) > 0.5)
    for row in out:
        row["CATEGORY_CONCENTRATED"] = concentrated
    return out


def decide(grows, coverage_rows):
    m4 = [r for r in grows if r["variant"] == "M4_HARD_NORMAL_EVOLUTION"]
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in grows}
    direction = {}
    for d in ["D1", "D2"]:
        vals = [by[(d, f, "M4_HARD_NORMAL_EVOLUTION")] for f in [0, 1]]
        mean_g = float(np.mean([r["G_primary"] for r in vals]))
        mean_dp = float(np.mean([r["dP-AUROC"] for r in vals]))
        mean_dap = float(np.mean([r["dP-AP"] for r in vals]))
        direction[d] = {"mean_G_primary": mean_g, "positive_folds": int(sum(r["G_primary"] > 0 for r in vals)), "min_G_primary": float(min(r["G_primary"] for r in vals)), "mean_dP_AUROC": mean_dp, "mean_dP_AP": mean_dap, "passes": bool(mean_g >= 0.20 and min(r["G_primary"] for r in vals) >= -0.20 and mean_dp >= -0.10 and mean_dap >= -0.10)}
    all_positive = sum(r["G_primary"] > 0 for r in m4)
    m3_gap = float(np.mean([by[(d, f, "M4_HARD_NORMAL_EVOLUTION")]["G_primary"] - by[(d, f, "M3_LOW_NOVELTY_NORMAL")]["G_primary"] for d in ["D1", "D2"] for f in [0, 1]]))
    m2_gap = float(np.mean([by[(d, f, "M4_HARD_NORMAL_EVOLUTION")]["G_primary"] - by[(d, f, "M2_RANDOM_NORMAL")]["G_primary"] for d in ["D1", "D2"] for f in [0, 1]]))
    m5_gap = float(np.mean([by[(d, f, "M4_HARD_NORMAL_EVOLUTION")]["G_primary"] - by[(d, f, "M5_ONE_SHOT_HARD_NORMAL")]["G_primary"] for d in ["D1", "D2"] for f in [0, 1]]))
    specificity = {"M4_minus_M3_mean_G": m3_gap, "M4_minus_M2_mean_G": m2_gap, "M4_minus_M5_mean_G": m5_gap}
    if direction["D1"]["passes"] and direction["D2"]["passes"] and all_positive >= 3 and m3_gap >= 0.30 and m2_gap >= 0.20:
        return "PASS_TO_PHASE_XXXII_B", direction, specificity
    if direction["D1"]["passes"] != direction["D2"]["passes"]:
        return "XXXII_A_DIRECTION_SPECIFIC_HARD_NORMAL_GAIN", direction, specificity
    if m3_gap < 0.30 or m2_gap < 0.20:
        return "XXXII_A_HARD_NORMAL_IDENTITY_NOT_SPECIFIC", direction, specificity
    cov_m0 = [r for r in coverage_rows if r["variant"] == "M0_STATIC_FPS"]
    cov_m4 = [r for r in coverage_rows if r["variant"] == "M4_HARD_NORMAL_EVOLUTION"]
    cov_improved = np.mean([float(a["p95_candidate_nearest_distance"]) - float(b["p95_candidate_nearest_distance"]) for a, b in zip(cov_m0, cov_m4)]) > 0
    if cov_improved and any(r["G_primary"] <= 0 for r in m4):
        return "XXXII_A_COVERAGE_NOT_PERFORMANCE_LIMITING", direction, specificity
    return "XXXII_A_STATIC_FPS_ALREADY_SUFFICIENT", direction, specificity


def write_report(outcome, direction, specificity, raw_rows, grows):
    lines = ["# Phase XXXII-A Hard-Normal Memory Coverage Evolution Audit", "", f"Outcome: **{outcome}**", "", "Scope: source-train normal patch memory content only. Image memory was unchanged, anomaly patch entries were preserved, no target, no training, no text, no DINO, no PRO-based decision.", "", "## Metrics", "", "| Direction | Fold | Variant | P-AUROC | P-AP | G_pixel | I-AUROC | I-AP |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    gmap = {(r["direction"], int(r["fold"]), r["variant"]): r for r in grows}
    for r in raw_rows:
        g = gmap[(r["direction"], int(r["fold"]), r["variant"])]
        lines.append(f"| {r['direction']} | {r['fold']} | {r['variant']} | {r['P-AUROC']:.4f} | {r['P-AP']:.4f} | {g['G_primary']:+.4f} | {r['I-AUROC']:.4f} | {r['I-AP']:.4f} |")
    lines.extend(["", "## Gates", ""])
    for d, s in direction.items():
        lines.append(f"- {d}: mean G={s['mean_G_primary']:+.4f}, positive_folds={s['positive_folds']}/2, min={s['min_G_primary']:+.4f}, mean dP-AUROC={s['mean_dP_AUROC']:+.4f}, mean dP-AP={s['mean_dP_AP']:+.4f}, pass={s['passes']}.")
    lines.extend(["", "## Specificity"])
    for k, v in specificity.items():
        lines.append(f"- {k}: `{v:+.4f}`")
    lines.extend(["", "## Answers", "", "- Q1/Q2: answered by M4 vs M0 G_pixel gates above.", "- Q3: answered by M4-M3 specificity.", "- Q4: M5 one-shot hard was included diagnostically.", "- Q5: coverage geometry and task metrics are both reported; outcome follows P-AUROC/P-AP only.", "", "Mandatory stop: Phase XXXII-B was not started."])
    (OUT / "XXXII_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    all_audit, all_budget, all_build, all_raw, all_cov, all_bound, all_sep, all_parity, all_pc = [], [], [], [], [], [], [], [], []
    for d in ["D1", "D2"]:
        for f in [0, 1]:
            res = run_fold(d, f, device)
            all_audit.append(res["audit"]); all_budget.append(res["budget"]); all_build.extend(res["build"]); all_raw.extend(res["raw"])
            all_cov.extend(res["coverage"]); all_bound.extend(res["boundary"]); all_sep.extend(res["separation"]); all_parity.extend(res["parity"]); all_pc.extend(res["percat"])
            write_csv(OUT / "RAW_METRICS.running.csv", all_raw)
    grows = grouped(all_raw)
    pc = percat_wtl(all_pc)
    outcome, direction, specificity = decide(grows, all_cov)
    memory_audit = {
        "phase": "XXXII-A",
        "modified_path": "patch/local normal memory only",
        "official_patch_path": "query patch features are projected with official patch_proj and matched by official softmax over patch keys/values",
        "normal_memory_changed": True,
        "anomaly_memory_preserved": True,
        "image_memory_unchanged": True,
        "cls_memory_reconstruction": "official held final score minus 0.3 official prompt top1%, divided by 0.7",
        "candidate_pool": "all patches from source-train normal images in held-out source protocol",
        "target_accessed": False,
        "trained": False,
        "runtime_sec": time.time() - start,
        "folds": all_audit,
    }
    (OUT / "MEMORY_PATH_AUDIT.json").write_text(json.dumps(memory_audit, indent=2, default=json_default), encoding="utf-8")
    (OUT / "MEMORY_BUDGET.json").write_text(json.dumps(all_budget, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "MEMORY_BUILD_STATS.csv", all_build)
    write_csv(OUT / "RAW_METRICS.csv", all_raw)
    write_csv(OUT / "GROUPED_GAINS.csv", grows)
    write_csv(OUT / "COVERAGE_GEOMETRY.csv", all_cov)
    write_csv(OUT / "BOUNDARY_NORMAL_ANALYSIS.csv", all_bound)
    write_csv(OUT / "ANOMALY_SEPARATION.csv", all_sep)
    write_csv(OUT / "MEMORY_PARITY.csv", all_parity)
    write_csv(OUT / "PER_CATEGORY.csv", pc)
    write_report(outcome, direction, specificity, all_raw, grows)
    payload = {"phase": "XXXII-A", "outcome": outcome, "target_accessed": False, "trained": False, "pro_used_for_decision": False, "phase_xxxii_b_started": False, "runtime_sec": time.time() - start, "direction_gates": direction, "specificity": specificity}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXXII-A\n")
        fp.write("- Command: python experiments/phase_xxxii_hard_normal_memory.py\n")
        fp.write("- Scope: D1 fold0/fold1 and D2 fold0/fold1 source-train normal patch memory content audit; no target; no training.\n")
        fp.write(f"- Outcome: {outcome}.\n")
        for d, s in direction.items():
            fp.write(f"- {d}: M4 mean G_pixel {s['mean_G_primary']:+.4f}, positive folds {s['positive_folds']}/2, pass {s['passes']}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXXII_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
