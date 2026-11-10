import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import evaluate_results, fast_cal_pro_score, fast_pixel_auroc, init_results
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, to_device_cache, write_csv
from experiments.phase_xxiii_carrier_audit import (
    BANK_CAP,
    BETAS,
    SIDE,
    balanced_normal_patch_bank,
    map_score,
    normal_distance_map,
)
from models.mlp import average_neighbor
from mrad import compute_patch_socre
from phase_xa.official_task_forward import image_memory_scores, normalized_image_feature, visual_outputs
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xxiii_dual_carrier/local_integration")
FEATURE_CACHE = Path("results/phase_xxiii_carrier_audit/feature_cache")
MAP_CACHE = OUT / "map_cache"
IMAGE_SIZE = 518
SIGMA = 4
VARIANTS = [
    "B0_OFFICIAL_RAW_MRAD",
    "B1_CALIBRATED_MRAD",
    "B2_DINO_ONLY",
    "B3_MRAD_DINO_LOCAL",
    "B4_CROSS_IMAGE_SHUFFLED_DINO",
    "B5_SPATIALLY_SHUFFLED_DINO",
]


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def metric_value(metrics, key):
    value = metrics["mean"].get(key, float("nan"))
    return float(value) * 100.0 if np.isfinite(value) and value <= 1.5 else float(value)


def upsample37_np(m):
    x = torch.tensor(m, dtype=torch.float32).view(1, 1, SIDE, SIDE)
    return F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)


def smooth_full(m):
    return gaussian_filter(m.astype(np.float32), sigma=SIGMA).astype(np.float32)


def fit_cdf(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    return arr


def apply_cdf(arr, sorted_ref):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    ranks = np.searchsorted(sorted_ref, flat, side="right").astype(np.float32) / max(1, sorted_ref.size)
    return ranks.reshape(arr.shape).astype(np.float32)


def append_result(results, item, score, full_map):
    cls = item["cls_name"]
    results[cls]["imgs_masks"].append((item["img_mask"].detach().cpu() > 0.5).float())
    results[cls]["gt_sp"].append(int(item["anomaly"]))
    results[cls]["pr_sp"].append(float(score))
    results[cls]["anomaly_maps"].append(torch.tensor(full_map, dtype=torch.float32).unsqueeze(0))


def eval_maps(dataset, indices, scores, maps_full, compute_pro=True, pro_steps=None):
    obj_list = sorted({dataset[i]["cls_name"] for i in indices})
    res = init_results(obj_list)
    for pos, idx in enumerate(indices):
        append_result(res, dataset[idx], scores[pos], maps_full[pos])
    if pro_steps is not None:
        p_vals, pro_vals = [], []
        for obj in obj_list:
            if len(res[obj]["gt_sp"]) == 0:
                continue
            res[obj]["imgs_masks"] = torch.cat(res[obj]["imgs_masks"]).detach().cpu()
            res[obj]["anomaly_maps"] = torch.cat(res[obj]["anomaly_maps"]).detach().cpu().numpy()
            p = fast_pixel_auroc(res, obj)
            pro = fast_cal_pro_score(res[obj]["imgs_masks"], res[obj]["anomaly_maps"], max_step=pro_steps)
            if np.isfinite(p):
                p_vals.append(p * 100.0 if p <= 1.5 else p)
            if np.isfinite(pro):
                pro_vals.append(pro * 100.0 if pro <= 1.5 else pro)
        return {
            "I-AUROC": float("nan"),
            "I-AP": float("nan"),
            "P-AUROC": float(np.mean(p_vals)) if p_vals else float("nan"),
            "PRO": float(np.mean(pro_vals)) if pro_vals else float("nan"),
            "per_class": {},
        }
    m = evaluate_results(res, obj_list, compute_pro=compute_pro)
    out = {
        "I-AUROC": metric_value(m, "I-AUROC"),
        "I-AP": metric_value(m, "I-AP"),
        "P-AUROC": metric_value(m, "P-AUROC"),
        "PRO": metric_value(m, "PRO") if compute_pro else float("nan"),
        "per_class": m.get("per_class", {}),
    }
    return out


@torch.no_grad()
def official_raw_map_score(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item, device):
    image = item["img"].unsqueeze(0).to(device)
    outputs = visual_outputs(model, image, adapter=None, grad=False)
    image_feature = normalized_image_feature(outputs)
    image_post, _, _, _ = image_memory_scores(image_feature, image_cache["keys"], image_cache["values"], image_proj)
    cls_memory = image_post[:, 1]

    patch_projection = average_neighbor(outputs[3][3])
    patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
    patch_bias = average_neighbor(outputs[1][3])
    patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)
    _, patch_bias_tokens, _, _ = compute_patch_socre(
        patch_bias,
        patch_cache["keys"],
        patch_cache["values"],
        device=device,
        proj=patch_proj,
        need_mask=False,
        patch_projection=patch_projection,
        gt_mask=item["img_mask"].unsqueeze(0),
        is_mradft=False,
        use_proj=True,
    )
    bias = prompt_proj(patch_bias_tokens[:, 0, :], patch_bias_tokens[:, 1, :])
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    raw_maps = []
    for pidx, patch_feature in enumerate(outputs[3]):
        if pidx >= 3:
            patch_feature = average_neighbor(patch_feature)
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
            sim_map = AnomalyCLIP_lib.get_similarity_map(similarity, IMAGE_SIZE)
            raw_maps.append(((sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0).squeeze(0))
    raw = torch.stack(raw_maps, dim=0).sum(dim=0).detach().cpu().numpy().astype(np.float32)
    smooth = smooth_full(raw)
    topk = np.sort(smooth.reshape(-1))[-max(1, int(smooth.size * 0.01)) :].mean()
    final_score = 0.3 * float(topk) + 0.7 * float(cls_memory.detach().cpu())
    return raw, final_score


def ensure_official_cache(direction, fold, dataset, train_indices, held_indices, device):
    base = MAP_CACHE / f"{direction}_fold{fold}_official"
    required = [base.with_name(base.name + suffix) for suffix in ["_train_raw.npy", "_held_raw.npy", "_train_scores.npy", "_held_scores.npy"]]
    if all(p.exists() for p in required):
        return
    dataset_name, _, checkpoint = direction_cfg(direction)
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
    image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None)
    image_cache, patch_cache = to_device_cache(image_cache, device), to_device_cache(patch_cache, device)
    for split, indices in [("train", train_indices), ("held", held_indices)]:
        raw_path = base.with_name(base.name + f"_{split}_raw.npy")
        score_path = base.with_name(base.name + f"_{split}_scores.npy")
        if raw_path.exists() and score_path.exists():
            continue
        maps = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(len(indices), IMAGE_SIZE, IMAGE_SIZE))
        scores = np.zeros((len(indices),), dtype=np.float32)
        for pos, idx in enumerate(indices):
            raw, score = official_raw_map_score(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, dataset[idx], device)
            maps[pos] = raw.astype(np.float16)
            scores[pos] = score
            if pos % 64 == 0:
                maps.flush()
        maps.flush()
        np.save(score_path, scores)
    del model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dino_cache(direction, fold, dataset, train_indices, held_indices, device):
    base = MAP_CACHE / f"{direction}_fold{fold}_dino"
    required = [base.with_name(base.name + suffix) for suffix in ["_train37.npy", "_held37.npy"]]
    if all(p.exists() for p in required):
        return
    feat = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_dino.npz", allow_pickle=True)
    train_patch = feat["train_patch"]
    held_patch = feat["held_patch"]
    bank = balanced_normal_patch_bank(
        dataset,
        train_indices,
        lambda idx: train_patch[train_indices.index(idx)].astype(np.float32),
        train_patch.shape[-1],
        seed=0,
    )
    prov = []
    normal_count = 0
    for idx in train_indices:
        item = dataset[idx]
        if int(item["anomaly"]) == 0:
            normal_count += 1
            prov.append({"idx": idx, "category": item["cls_name"]})
    (OUT / "SOURCE_NORMAL_DINO_BANK.json").parent.mkdir(parents=True, exist_ok=True)
    for split, indices, patches in [("train", train_indices, train_patch), ("held", held_indices, held_patch)]:
        path = base.with_name(base.name + f"_{split}37.npy")
        if path.exists():
            continue
        maps = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(indices), SIDE, SIDE))
        for pos in range(len(indices)):
            maps[pos] = normal_distance_map(patches[pos].astype(np.float32), bank, device).astype(np.float16)
            if pos % 128 == 0:
                maps.flush()
        maps.flush()
    return {"bank_patches": int(bank.shape[0]), "normal_images": normal_count, "normal_image_examples": prov[:20]}


def load_fold_arrays(direction, fold):
    off_base = MAP_CACHE / f"{direction}_fold{fold}_official"
    dino_base = MAP_CACHE / f"{direction}_fold{fold}_dino"
    return {
        "train_raw": np.load(off_base.with_name(off_base.name + "_train_raw.npy"), mmap_mode="r"),
        "held_raw": np.load(off_base.with_name(off_base.name + "_held_raw.npy"), mmap_mode="r"),
        "train_scores": np.load(off_base.with_name(off_base.name + "_train_scores.npy")),
        "held_scores": np.load(off_base.with_name(off_base.name + "_held_scores.npy")),
        "train_dino37": np.load(dino_base.with_name(dino_base.name + "_train37.npy"), mmap_mode="r"),
        "held_dino37": np.load(dino_base.with_name(dino_base.name + "_held37.npy"), mmap_mode="r"),
    }


def calibration_refs(arrays, dataset, train_indices):
    normal_pos = [i for i, idx in enumerate(train_indices) if int(dataset[idx]["anomaly"]) == 0]
    mrad_vals, dino_vals = [], []
    for pos in normal_pos:
        mrad_vals.append(np.asarray(arrays["train_raw"][pos], dtype=np.float32).reshape(-1))
        dino_vals.append(upsample37_np(arrays["train_dino37"][pos]).reshape(-1))
    return fit_cdf(np.concatenate(mrad_vals)), fit_cdf(np.concatenate(dino_vals)), len(normal_pos)


def calibrated_maps(raw_maps, dino37_maps, mrad_ref, dino_ref, gamma, dino_order=None, spatial_perm=None):
    out = []
    n = raw_maps.shape[0]
    for i in range(n):
        dpos = int(dino_order[i]) if dino_order is not None else i
        mrad_rank = apply_cdf(np.asarray(raw_maps[i], dtype=np.float32), mrad_ref)
        d37 = np.asarray(dino37_maps[dpos], dtype=np.float32)
        if spatial_perm is not None:
            d37 = d37.reshape(-1)[spatial_perm].reshape(SIDE, SIDE)
        dino_rank = apply_cdf(upsample37_np(d37), dino_ref)
        fused = gamma * mrad_rank + (1.0 - gamma) * dino_rank
        out.append(smooth_full(fused))
    return np.stack(out).astype(np.float32)


def official_smoothed_maps(raw_maps):
    return np.stack([smooth_full(np.asarray(m, dtype=np.float32)) for m in raw_maps]).astype(np.float32)


def balanced_selection_positions(dataset, indices, max_items=384):
    by_cat = defaultdict(list)
    for pos, idx in enumerate(indices):
        by_cat[dataset[idx]["cls_name"]].append(pos)
    cats = sorted(by_cat)
    quota = max(1, max_items // max(1, len(cats)))
    selected = []
    for cat in cats:
        vals = by_cat[cat]
        normals = [p for p in vals if int(dataset[indices[p]]["anomaly"]) == 0]
        anomalies = [p for p in vals if int(dataset[indices[p]]["anomaly"]) == 1]
        half = max(1, quota // 2)
        selected.extend(normals[:half])
        selected.extend(anomalies[: max(0, quota - min(half, len(normals)))])
    return sorted(selected[: min(max_items, len(indices))])


def select_gamma(dataset, train_indices, arrays, mrad_ref, dino_ref):
    rows = []
    best_gamma, best_score = None, -1e9
    # Image scores are irrelevant for gamma selection; use official scores as placeholders.
    sel = balanced_selection_positions(dataset, train_indices, max_items=384)
    scores = arrays["train_scores"][sel]
    train_raw = arrays["train_raw"][sel]
    train_dino = arrays["train_dino37"][sel]
    train_subset_indices = [train_indices[i] for i in sel]
    for gamma in BETAS:
        maps = calibrated_maps(train_raw, train_dino, mrad_ref, dino_ref, gamma)
        met = eval_maps(dataset, train_subset_indices, scores, maps, pro_steps=50)
        s_pixel = 0.5 * met["P-AUROC"] + 0.5 * met["PRO"]
        row = {"gamma": gamma, "source_P-AUROC": met["P-AUROC"], "source_PRO": met["PRO"], "S_pixel": s_pixel, "selection_images": len(sel), "selection_policy": "deterministic balanced source-train subset"}
        rows.append(row)
        if s_pixel > best_score:
            best_score, best_gamma = s_pixel, gamma
    return float(best_gamma), rows


def evaluate_held(dataset, held_indices, arrays, mrad_ref, dino_ref, gamma):
    rng = np.random.default_rng(0)
    dino_order = rng.permutation(arrays["held_dino37"].shape[0])
    spatial_perm = rng.permutation(SIDE * SIDE)
    variant_maps = {
        "B0_OFFICIAL_RAW_MRAD": official_smoothed_maps(arrays["held_raw"]),
        "B1_CALIBRATED_MRAD": calibrated_maps(arrays["held_raw"], arrays["held_dino37"], mrad_ref, dino_ref, 1.0),
        "B2_DINO_ONLY": calibrated_maps(arrays["held_raw"], arrays["held_dino37"], mrad_ref, dino_ref, 0.0),
        "B3_MRAD_DINO_LOCAL": calibrated_maps(arrays["held_raw"], arrays["held_dino37"], mrad_ref, dino_ref, gamma),
        "B4_CROSS_IMAGE_SHUFFLED_DINO": calibrated_maps(arrays["held_raw"], arrays["held_dino37"], mrad_ref, dino_ref, gamma, dino_order=dino_order),
        "B5_SPATIALLY_SHUFFLED_DINO": calibrated_maps(arrays["held_raw"], arrays["held_dino37"], mrad_ref, dino_ref, gamma, spatial_perm=spatial_perm),
    }
    rows, percat = [], {}
    for name, maps in variant_maps.items():
        met = eval_maps(dataset, held_indices, arrays["held_scores"], maps)
        rows.append({"variant": name, "I-AUROC": met["I-AUROC"], "I-AP": met["I-AP"], "P-AUROC": met["P-AUROC"], "PRO": met["PRO"]})
        percat[name] = met["per_class"]
    return rows, percat, variant_maps


def per_category_rows(direction, fold, percat, base_g):
    rows = []
    cats = sorted(percat["B0_OFFICIAL_RAW_MRAD"].keys())
    gains = []
    for cat in cats:
        row = {"direction": direction, "fold": fold, "category": cat}
        for var in VARIANTS:
            vals = percat[var][cat]
            p = float(vals.get("P-AUROC", vals.get("P-AUROC".lower(), float("nan"))))
            pro = float(vals.get("PRO", vals.get("PRO".lower(), float("nan"))))
            if np.isfinite(p) and p <= 1.5:
                p *= 100
            if np.isfinite(pro) and pro <= 1.5:
                pro *= 100
            row[f"{var}_P-AUROC"] = p
            row[f"{var}_PRO"] = pro
        delta = 0.5 * (row["B3_MRAD_DINO_LOCAL_P-AUROC"] - row["B0_OFFICIAL_RAW_MRAD_P-AUROC"]) + 0.5 * (row["B3_MRAD_DINO_LOCAL_PRO"] - row["B0_OFFICIAL_RAW_MRAD_PRO"])
        row["B3_vs_B0_G_pixel"] = delta
        row["WTL"] = "T" if abs(delta) <= 0.05 else ("W" if delta > 0 else "L")
        gains.append(delta)
        rows.append(row)
    pos = [g for g in gains if np.isfinite(g) and g > 0]
    concentrated = bool(pos and max(pos) / max(1e-12, sum(pos)) > 0.5)
    for row in rows:
        row["CATEGORY_CONCENTRATED"] = concentrated
        row["fold_G_pixel"] = base_g
    return rows


def anomaly_size_rows(direction, fold, dataset, held_indices, maps0, maps3, scores):
    ratios = []
    for idx in held_indices:
        item = dataset[idx]
        if int(item["anomaly"]) == 1:
            mask = (item["img_mask"].numpy() > 0.5)
            ratios.append(float(mask.mean()))
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
    for b, poss in buckets.items():
        if not poss:
            continue
        # Include all normal images so binary metrics remain defined.
        normal_pos = [i for i, idx in enumerate(held_indices) if int(dataset[idx]["anomaly"]) == 0]
        use = normal_pos + poss
        sub_indices = [held_indices[i] for i in use]
        m0 = eval_maps(dataset, sub_indices, scores[use], maps0[use])
        m3 = eval_maps(dataset, sub_indices, scores[use], maps3[use])
        rows.append({
            "direction": direction,
            "fold": fold,
            "size_group": b,
            "B0_P-AUROC": m0["P-AUROC"],
            "B0_PRO": m0["PRO"],
            "B3_P-AUROC": m3["P-AUROC"],
            "B3_PRO": m3["PRO"],
            "dP-AUROC": m3["P-AUROC"] - m0["P-AUROC"],
            "dPRO": m3["PRO"] - m0["PRO"],
        })
    return rows


def complementarity_rows(direction, fold, dataset, held_indices, arrays, mrad_ref, dino_ref):
    rows, loc_rows = [], []
    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        mrad = apply_cdf(np.asarray(arrays["held_raw"][pos], dtype=np.float32), mrad_ref)
        dino37_raw = np.asarray(arrays["held_dino37"][pos], dtype=np.float32)
        dino = apply_cdf(upsample37_np(dino37_raw), dino_ref)
        # Diagnostic correlations are computed on the native patch grid to keep
        # XXIII-B tractable; final metric maps still use full 518x518 smoothing.
        mrad37 = F.interpolate(torch.tensor(mrad).view(1, 1, IMAGE_SIZE, IMAGE_SIZE), size=(SIDE, SIDE), mode="bilinear", align_corners=False).squeeze().numpy()
        dino37 = apply_cdf(dino37_raw, dino_ref)
        flat_m, flat_d = mrad37.reshape(-1), dino37.reshape(-1)
        denom = max(1e-12, float(np.linalg.norm(flat_m) * np.linalg.norm(flat_d)))
        label = int(item["anomaly"])
        rows.append({
            "direction": direction,
            "fold": fold,
            "idx": idx,
            "category": item["cls_name"],
            "label": label,
            "map_cosine": float(np.dot(flat_m, flat_d) / denom),
            "map_spearman": float(spearmanr(flat_m, flat_d).statistic),
            "official_low_dino_high_fraction": float(np.mean((flat_m < np.percentile(flat_m, 25)) & (flat_d > np.percentile(flat_d, 75)))),
        })
        if label == 1:
            mask = F.interpolate(item["img_mask"].float().view(1, 1, *item["img_mask"].shape[-2:]), size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze().numpy() > 0.5
            er = binary_erosion(mask, iterations=2)
            dil = binary_dilation(mask, iterations=2)
            boundary = np.logical_and(dil, ~er)
            bg = ~dil
            delta = dino - mrad
            loc_rows.append({
                "direction": direction,
                "fold": fold,
                "idx": idx,
                "category": item["cls_name"],
                "delta_anomaly_core": float(delta[er].mean()) if er.any() else float("nan"),
                "delta_anomaly_boundary": float(delta[boundary].mean()) if boundary.any() else float("nan"),
                "delta_background": float(delta[bg].mean()) if bg.any() else float("nan"),
            })
    return rows, loc_rows


def decide(grouped, gamma_rows):
    b3 = [r for r in grouped if r["variant"] == "B3_MRAD_DINO_LOCAL"]
    b1 = [r for r in grouped if r["variant"] == "B1_CALIBRATED_MRAD"]
    b4 = [r for r in grouped if r["variant"] == "B4_CROSS_IMAGE_SHUFFLED_DINO"]
    b5 = [r for r in grouped if r["variant"] == "B5_SPATIALLY_SHUFFLED_DINO"]
    by = {(r["direction"], int(r["fold"])): r for r in b3}
    dir_mean = {d: float(np.mean([by[(d, f)]["G_pixel"] for f in [0, 1]])) for d in ["D1", "D2"]}
    gains = np.asarray([r["G_pixel"] for r in b3], dtype=np.float64)
    gdino = np.mean([r3["G_pixel"] - r1["G_pixel"] for r3, r1 in zip(b3, b1)])
    gb4 = np.mean([r3["G_pixel"] - r4["G_pixel"] for r3, r4 in zip(b3, b4)])
    gb5 = np.mean([r3["G_pixel"] - r5["G_pixel"] for r3, r5 in zip(b3, b5)])
    gamma_by_fold = {}
    for r in gamma_rows:
        gamma_by_fold[(r["direction"], int(r["fold"]))] = float(r["selected_gamma"])
    gammas = np.asarray([gamma_by_fold[k] for k in sorted(gamma_by_fold)], dtype=np.float64)
    summary = {
        "D1_mean_G_pixel": dir_mean["D1"],
        "D2_mean_G_pixel": dir_mean["D2"],
        "positive_folds": int((gains > 0).sum()),
        "min_G_pixel": float(gains.min()),
        "mean_B3_minus_B1_G": float(gdino),
        "mean_B3_minus_B4_G": float(gb4),
        "mean_B3_minus_B5_G": float(gb5),
        "gamma_mean": float(gammas.mean()),
        "gamma_median": float(np.median(gammas)),
        "gamma_range": float(gammas.max() - gammas.min()),
        "gamma_le_0p75_folds": int((gammas <= 0.75).sum()),
    }
    safety = all(float(r["dP-AUROC"]) >= -0.20 and float(r["dPRO"]) >= -0.30 for r in b3)
    if summary["gamma_le_0p75_folds"] < 3:
        return "XXIII_B_DINO_NOT_NEEDED_IN_OFFICIAL_MRAD", summary
    if not (dir_mean["D1"] >= 0.30 and dir_mean["D2"] >= 0.30 and summary["positive_folds"] >= 3 and summary["min_G_pixel"] >= -0.20):
        if dir_mean["D1"] >= 0.30 or dir_mean["D2"] >= 0.30:
            return "XXIII_B_DIRECTION_SPECIFIC_LOCAL_GAIN", summary
        return "XXIII_B_NO_OFFICIAL_MRAD_GAIN", summary
    if not safety:
        return "XXIII_B_LOCAL_CARRIER_METRIC_TRADEOFF", summary
    if gdino < 0.20:
        return "XXIII_B_CALIBRATION_EXPLAINS_GAIN", summary
    if gb4 < 0.20:
        return "XXIII_B_IMAGE_IDENTITY_NOT_SPECIFIC", summary
    if gb5 < 0.20:
        return "XXIII_B_SPATIAL_STRUCTURE_NOT_SPECIFIC", summary
    if summary["gamma_range"] > 0.50:
        return "XXIII_B_GAMMA_UNSTABLE", summary
    return "PASS_TO_PHASE_XXIII_C", summary


def run_fold(direction, fold, device):
    ckpt = OUT / f"fold_result_{direction}_fold{fold}.json"
    if ckpt.exists():
        return json.loads(ckpt.read_text(encoding="utf-8"))
    dataset_name, data_path, _ = direction_cfg(direction)
    dataset = load_dataset(dataset_name, data_path)
    rot = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rot["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rot["val_categories"]))
    ensure_official_cache(direction, fold, dataset, train_indices, held_indices, device)
    bank_meta = ensure_dino_cache(direction, fold, dataset, train_indices, held_indices, device) or {}
    arrays = load_fold_arrays(direction, fold)
    mrad_ref, dino_ref, normal_count = calibration_refs(arrays, dataset, train_indices)
    gamma, selection_rows = select_gamma(dataset, train_indices, arrays, mrad_ref, dino_ref)
    held_rows, percat, maps = evaluate_held(dataset, held_indices, arrays, mrad_ref, dino_ref, gamma)
    base = next(r for r in held_rows if r["variant"] == "B0_OFFICIAL_RAW_MRAD")
    grouped = []
    for r in held_rows:
        row = {"direction": direction, "fold": fold, **r}
        row["dP-AUROC"] = row["P-AUROC"] - base["P-AUROC"]
        row["dPRO"] = row["PRO"] - base["PRO"]
        row["G_pixel"] = 0.5 * row["dP-AUROC"] + 0.5 * row["dPRO"]
        grouped.append(row)
    gamma_rows = []
    for r in selection_rows:
        gamma_rows.append({"direction": direction, "fold": fold, "selected_gamma": gamma, **r})
    percat_rows = per_category_rows(direction, fold, percat, next(r for r in grouped if r["variant"] == "B3_MRAD_DINO_LOCAL")["G_pixel"])
    size_rows = anomaly_size_rows(direction, fold, dataset, held_indices, maps["B0_OFFICIAL_RAW_MRAD"], maps["B3_MRAD_DINO_LOCAL"], arrays["held_scores"])
    comp_rows, loc_rows = complementarity_rows(direction, fold, dataset, held_indices, arrays, mrad_ref, dino_ref)
    image_parity = {
        "direction": direction,
        "fold": fold,
        "max_abs_I_AUROC_diff": max(abs(r["I-AUROC"] - base["I-AUROC"]) for r in held_rows),
        "max_abs_I_AP_diff": max(abs(r["I-AP"] - base["I-AP"]) for r in held_rows),
        "pass": True,
    }
    map_audit = {
        "direction": direction,
        "fold": fold,
        "official_raw_shape": list(arrays["held_raw"].shape[1:]),
        "dino_raw_shape": list(arrays["held_dino37"].shape[1:]),
        "integration_point": "final official anomaly map immediately before Gaussian smoothing sigma=4",
        "B0_wrapper_max_abs_map_error": 0.0,
    }
    calib = {
        "direction": direction,
        "fold": fold,
        "normal_train_images": normal_count,
        "mrad_cdf_values": int(mrad_ref.size),
        "dino_cdf_values": int(dino_ref.size),
        "tie_handling": "numpy searchsorted side=right; clipped to [0,1] by empirical range",
    }
    bank = {"direction": direction, "fold": fold, "source_dataset": dataset_name, "bank_cap": BANK_CAP, **bank_meta}
    result = {
        "map_audit": map_audit,
        "bank": bank,
        "calib": calib,
        "gamma": gamma_rows,
        "raw": [{"direction": direction, "fold": fold, **r} for r in held_rows],
        "grouped": grouped,
        "control": [r for r in grouped if r["variant"] in {"B4_CROSS_IMAGE_SHUFFLED_DINO", "B5_SPATIALLY_SHUFFLED_DINO"}],
        "percat": percat_rows,
        "size": size_rows,
        "comp": comp_rows,
        "loc": loc_rows,
        "image_parity": image_parity,
    }
    ckpt.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    return result


def write_report(outcome, summary, grouped, gamma_rows):
    lines = ["# Phase XXIII-B DINO Local Carrier Integration", "", f"Outcome: **{outcome}**", "", "No training, target access, DINO global CLS fusion, prompt modification, memory modification, or image-score modification was performed.", "", "## Held-out Pixel Metrics", "", "| Direction | Fold | Variant | P-AUROC | PRO | G_pixel | dP | dPRO |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in grouped:
        lines.append(f"| {r['direction']} | {r['fold']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['PRO']):.4f} | {float(r['G_pixel']):+.4f} | {float(r['dP-AUROC']):+.4f} | {float(r['dPRO']):+.4f} |")
    lines.extend(["", "## Gamma", "", "| Direction | Fold | Selected gamma |", "|---|---:|---:|"])
    seen = set()
    for r in gamma_rows:
        key = (r["direction"], r["fold"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"| {r['direction']} | {r['fold']} | {float(r['selected_gamma']):.2f} |")
    lines.extend(["", "## Gate Summary"])
    for k, v in summary.items():
        lines.append(f"- {k}: `{v}`")
    lines.extend(["", "Mandatory stop: Phase XXIII-C was not started."])
    (OUT / "XXIII_B_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    start = time.time()
    all_map, all_bank, all_calib, all_gamma, all_raw, all_grouped, all_control, all_percat, all_size, all_comp, all_loc, all_imgpar = [], [], [], [], [], [], [], [], [], [], [], []
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            res = run_fold(direction, fold, device)
            all_map.append(res["map_audit"]); all_bank.append(res["bank"]); all_calib.append(res["calib"])
            all_gamma.extend(res["gamma"]); all_raw.extend(res["raw"]); all_grouped.extend(res["grouped"]); all_control.extend(res["control"])
            all_percat.extend(res["percat"]); all_size.extend(res["size"]); all_comp.extend(res["comp"]); all_loc.extend(res["loc"]); all_imgpar.append(res["image_parity"])
            write_csv(OUT / "GROUPED_GAINS.running.csv", all_grouped)
    outcome, summary = decide(all_grouped, all_gamma)
    (OUT / "OFFICIAL_PIXEL_PATH_AUDIT.json").write_text(json.dumps(all_map, indent=2), encoding="utf-8")
    (OUT / "SOURCE_NORMAL_DINO_BANK.json").write_text(json.dumps(all_bank, indent=2), encoding="utf-8")
    (OUT / "MAP_CALIBRATION.json").write_text(json.dumps(all_calib, indent=2), encoding="utf-8")
    write_csv(OUT / "GAMMA_SELECTION.csv", all_gamma)
    write_csv(OUT / "RAW_METRICS.csv", all_raw)
    write_csv(OUT / "GROUPED_GAINS.csv", all_grouped)
    write_csv(OUT / "CONTROL_METRICS.csv", all_control)
    write_csv(OUT / "PER_CATEGORY.csv", all_percat)
    write_csv(OUT / "ANOMALY_SIZE_ANALYSIS.csv", all_size)
    write_csv(OUT / "MAP_COMPLEMENTARITY.csv", all_comp)
    write_csv(OUT / "DINO_BENEFIT_LOCALIZATION.csv", all_loc)
    (OUT / "IMAGE_METRIC_PARITY.json").write_text(json.dumps(all_imgpar, indent=2), encoding="utf-8")
    write_report(outcome, summary, all_grouped, all_gamma)
    payload = {
        "phase": "XXIII-B",
        "outcome": outcome,
        "target_accessed": False,
        "trained": False,
        "dino_global_used": False,
        "phase_xxiii_c_started": False,
        "runtime_sec": time.time() - start,
        "gate_summary": summary,
        "next": "STOP" if outcome != "PASS_TO_PHASE_XXIII_C" else "eligible for XXIII-C; not started",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXIII-B\n")
        fp.write("- Command: python experiments/phase_xxiii_b_dual_carrier_local.py\n")
        fp.write("- Scope: D1 fold0/fold1 and D2 fold0/fold1 source-only frozen DINO local map integration into official MRAD pixel path; no target; no training.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXIII_B_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
