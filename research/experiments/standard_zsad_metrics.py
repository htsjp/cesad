import math
from collections import defaultdict

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc


PRO_FPR_LIMIT = 0.30
PRO_STEPS = 200


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s) * 100.0)


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s) * 100.0)


def f1max(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    precision, recall, _thresholds = precision_recall_curve(y, s)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)
    return float(np.nanmax(f1) * 100.0)


def resize_mask_flat(item, side):
    import torch
    import torch.nn.functional as F

    mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.float32)
    x = torch.tensor(mask, dtype=torch.float32).view(1, 1, *mask.shape[-2:])
    return F.interpolate(x, size=(side, side), mode="nearest").view(-1).numpy().astype(np.uint8)


def fast_aupro(mask_list, score_list, fpr_limit=PRO_FPR_LIMIT, steps=PRO_STEPS):
    masks = np.asarray(mask_list)
    scores = np.asarray(score_list, dtype=np.float32)
    if masks.size == 0 or scores.size == 0:
        return float("nan")
    if masks.ndim == 2:
        side = int(round(math.sqrt(masks.shape[1])))
        masks = masks.reshape(-1, side, side)
    if scores.ndim == 2:
        side = int(round(math.sqrt(scores.shape[1])))
        scores = scores.reshape(-1, side, side)
    masks = masks > 0.5
    if not np.any(masks) or not np.any(~masks):
        return float("nan")
    mn, mx = float(np.min(scores)), float(np.max(scores))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return float("nan")

    region_labels, region_areas = [], []
    structure = np.ones((3, 3), dtype=np.int8)
    for m in masks:
        labels, n = ndimage.label(m, structure=structure)
        areas = np.bincount(labels.reshape(-1), minlength=n + 1)[1:].astype(np.float64)
        region_labels.append(labels)
        region_areas.append(areas)
    if sum(len(a) for a in region_areas) == 0:
        return float("nan")

    neg_count = float((~masks).sum())
    thresholds = np.linspace(mn, mx, int(steps), dtype=np.float32)
    pros, fprs = [], []
    for th in thresholds:
        binary = scores > th
        fp = np.logical_and(binary, ~masks).sum()
        fprs.append(float(fp) / max(neg_count, 1.0))
        per_region = []
        for pred, labels, areas in zip(binary, region_labels, region_areas):
            if len(areas) == 0:
                continue
            hits = np.bincount(labels[pred].reshape(-1), minlength=len(areas) + 1)[1:].astype(np.float64)
            per_region.extend((hits / np.maximum(areas, 1.0)).tolist())
        pros.append(float(np.mean(per_region)) if per_region else float("nan"))
    pros = np.asarray(pros, dtype=np.float64)
    fprs = np.asarray(fprs, dtype=np.float64)
    keep = np.isfinite(pros) & np.isfinite(fprs) & (fprs <= fpr_limit)
    if keep.sum() < 2:
        return float("nan")
    f = fprs[keep]
    p = pros[keep]
    order = np.argsort(f)
    f = f[order]
    p = p[order]
    if f[-1] <= f[0]:
        return float("nan")
    return float(auc((f - f[0]) / (f[-1] - f[0]), p) * 100.0)


def summarize_standard(iy, iscore, py, pscore, mask_list=None, score_map_list=None):
    out = {
        "I-AUROC": safe_auc(iy, iscore),
        "I-AP": safe_ap(iy, iscore),
        "I-F1max": f1max(iy, iscore),
        "P-AUROC": safe_auc(py, pscore),
        "P-AP": safe_ap(py, pscore),
        "P-F1max": f1max(py, pscore),
        "PRO": fast_aupro(mask_list if mask_list is not None else [], score_map_list if score_map_list is not None else []),
    }
    out["AUPRO"] = out["PRO"]
    out["G_pixel"] = 0.5 * (float(out["P-AUROC"]) + float(out["P-AP"])) if np.isfinite(out["P-AUROC"]) and np.isfinite(out["P-AP"]) else float("nan")
    return out


def evaluate_zsad_standard(cell_id, target, dataset, idx_or_rows, kind, image_scores, maps, side):
    by_cat = defaultdict(lambda: {"iy": [], "is": [], "py": [], "ps": [], "masks": [], "score_maps": [], "mask_images": 0})
    all_iy, all_is, all_py, all_ps, all_masks, all_score_maps = [], [], [], [], [], []
    for pos, ref in enumerate(idx_or_rows):
        item = dataset[int(ref)] if kind == "source_dataset" else dataset[pos]
        if kind == "source_dataset":
            cat = str(item.get("cls_name", target))
            label = int(item.get("anomaly", 0))
            mask_available = "img_mask" in item
        else:
            row = dict(ref)
            cat = str(row.get("category", item.get("cls_name", target)))
            label = int(row.get("label", item.get("anomaly", 0)))
            mask_available = bool(row.get("mask_available", "img_mask" in item))
        score = float(image_scores[pos])
        d = by_cat[cat]
        d["iy"].append(label)
        d["is"].append(score)
        all_iy.append(label)
        all_is.append(score)
        if mask_available and "img_mask" in item:
            mask = resize_mask_flat(item, side).astype(np.uint8)
            smap = np.asarray(maps[pos], dtype=np.float32).reshape(side, side)
            score_flat = smap.reshape(-1)
            d["py"].append(mask)
            d["ps"].append(score_flat)
            d["masks"].append(mask.reshape(side, side))
            d["score_maps"].append(smap)
            d["mask_images"] += 1
            all_py.append(mask)
            all_ps.append(score_flat)
            all_masks.append(mask.reshape(side, side))
            all_score_maps.append(smap)
    per = {}
    for cat, vals in sorted(by_cat.items()):
        py = np.concatenate(vals["py"]) if vals["py"] else np.asarray([], dtype=np.uint8)
        ps = np.concatenate(vals["ps"]) if vals["ps"] else np.asarray([], dtype=np.float32)
        row = summarize_standard(vals["iy"], vals["is"], py, ps, vals["masks"], vals["score_maps"])
        row["mask_images"] = vals["mask_images"]
        per[cat] = row
    py_all = np.concatenate(all_py) if all_py else np.asarray([], dtype=np.uint8)
    ps_all = np.concatenate(all_ps) if all_ps else np.asarray([], dtype=np.float32)
    metric = summarize_standard(all_iy, all_is, py_all, ps_all, all_masks, all_score_maps)
    metric.update({"cell": cell_id, "target": target, "mask_images": len(all_masks), "pixel_eval_resolution": f"{side}x{side}"})
    return metric, per
