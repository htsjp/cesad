import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAM2_ROOT = ROOT / "_external" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from experiments.phase_isro_instance_structure_reobservation import (
    SAM2_CFG,
    SAM2_CKPT,
    SAM_POINTS_PER_BATCH,
    SAM_POINTS_PER_SIDE,
    ensure_sam2_ready,
)
from experiments.phase_mva_mrad_same_protocol_visa_factorization import load_metas_direct, pair_slug
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    IMAGE_SIZE,
    SIDE,
    fnum,
    mask_np,
    summarize_category,
    torch_resize,
)
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform

OUT = ROOT / "results" / "phase_snts_structure_normalized_two_state" / "formal"
STRUCTURE_CACHE = OUT / "STRUCTURE_CACHE"
MAP_CACHE = OUT / "MAP_CACHE"

DATASETS = ("MVTec", "VisA", "BTAD", "MPDD")
DATASET_LOADERS = {
    "MVTec": ("Full VisA", "MVTec"),
    "VisA": ("Full MVTec", "VisA"),
    "BTAD": ("Full MVTec", "BTAD"),
    "MPDD": ("Full VisA", "MPDD"),
}
PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "MPDD"),
)
FOCUS_VISA_CATEGORIES = {
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "fryum",
    "pipe_fryum",
    "capsules",
    "chewinggum",
    "candle",
}
E0 = "E0_Core_NTR_FixedLRP"
E1 = "E1_RegionMedianResidualRescue"
E2 = "E2_RobustStructureZScoreRescue"
E3 = "E3_HierarchyConsensusRescue"
VARIANTS = (E0, E1, E2, E3)

MIN_REGION_AREA_RATIO = 0.002
MAX_REGION_AREA_RATIO = 0.85
LAMBDA_RESCUE = 0.1
Z_MAX = 3.0
EPS = 1e-6


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite_mean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    n = len(sorted_s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    pos = y.astype(bool)
    n_pos = float(pos.sum())
    n_neg = float(y.size - pos.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1.0) / 2.0) / max(n_pos * n_neg, 1.0) * 100.0)


def load_sam_generator(device):
    build_sam2, SAM2AutomaticMaskGenerator = ensure_sam2_ready()
    sam2 = build_sam2(SAM2_CFG, str(SAM2_CKPT), device=device, apply_postprocessing=False)
    sam2.eval()
    return SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=SAM_POINTS_PER_SIDE,
        points_per_batch=SAM_POINTS_PER_BATCH,
        output_mode="binary_mask",
    )


def load_dataset_metas(name, device, smoke=False):
    source, target = DATASET_LOADERS[name]
    return load_metas_direct(source, target, device, smoke=smoke)


def structure_cache_path(dataset, image_index):
    return STRUCTURE_CACHE / dataset / f"{image_index:06d}.npz"


def load_structure(dataset, image_index):
    p = structure_cache_path(dataset, image_index)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    return {
        "masks": np.asarray(data["masks"], dtype=bool),
        "areas": np.asarray(data["areas"], dtype=np.float32),
        "stability": np.asarray(data["stability"], dtype=np.float32),
        "pred_iou": np.asarray(data["pred_iou"], dtype=np.float32),
    }


def save_structure(dataset, image_index, masks, areas, stability, pred_iou):
    p = structure_cache_path(dataset, image_index)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        masks=np.asarray(masks, dtype=bool),
        areas=np.asarray(areas, dtype=np.float32),
        stability=np.asarray(stability, dtype=np.float32),
        pred_iou=np.asarray(pred_iou, dtype=np.float32),
    )


def retained_sam_masks(anns, h, w):
    masks, areas, stability, pred_iou = [], [], [], []
    image_area = float(h * w)
    for ann in anns:
        mask = np.asarray(ann["segmentation"], dtype=bool)
        if mask.shape != (h, w):
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
        area = float(ann.get("area", mask.sum()))
        ar = area / max(image_area, 1.0)
        if ar < MIN_REGION_AREA_RATIO or ar > MAX_REGION_AREA_RATIO:
            continue
        masks.append(mask)
        areas.append(area)
        stability.append(float(ann.get("stability_score", 0.0)))
        pred_iou.append(float(ann.get("predicted_iou", 0.0)))
    order = np.argsort(-np.asarray(areas, dtype=np.float64)) if areas else []
    return [masks[i] for i in order], [areas[i] for i in order], [stability[i] for i in order], [pred_iou[i] for i in order]


def mask_entropy(areas, image_area):
    vals = np.asarray(areas, dtype=np.float64)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    p = vals / max(float(np.sum(vals)), 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum() / max(np.log(vals.size + 1e-12), 1e-12))


def overlap_and_nesting(masks):
    if len(masks) == 0:
        return 0.0, 0.0, 0.0
    stack = np.stack(masks, axis=0).astype(np.uint8)
    covered = stack.sum(axis=0)
    union = covered > 0
    overlap = float((covered > 1).sum() / max(union.sum(), 1))
    nested = 0
    depths = []
    areas = np.asarray([m.sum() for m in masks], dtype=np.float64)
    order = np.argsort(areas)
    for ii in order:
        mi = masks[int(ii)]
        ai = max(float(mi.sum()), 1.0)
        parents = 0
        for jj in order:
            if areas[int(jj)] <= areas[int(ii)]:
                continue
            mj = masks[int(jj)]
            if float((mi & mj).sum()) / ai >= 0.80:
                parents += 1
        nested += int(parents > 0)
        depths.append(parents)
    return overlap, float(nested / max(len(masks), 1)), float(np.mean(depths)) if depths else 0.0


def edge_density(img):
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag > np.percentile(mag, 75)).mean())


def feature_stats(feat):
    z = np.asarray(feat, dtype=np.float32).reshape(SIDE, SIDE, -1)
    flat = z.reshape(-1, z.shape[-1])
    centroid = flat.mean(axis=0, keepdims=True)
    dispersion = float(np.mean(np.linalg.norm(flat - centroid, axis=1)))
    sims = []
    for y in range(SIDE):
        for x in range(SIDE):
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < SIDE and 0 <= xx < SIDE:
                        neigh.append(z[yy, xx])
            if neigh:
                m = np.mean(neigh, axis=0)
                denom = max(float(np.linalg.norm(z[y, x]) * np.linalg.norm(m)), 1e-12)
                sims.append(float(np.dot(z[y, x], m) / denom))
    sim = np.asarray(sims, dtype=np.float32)
    hist, _ = np.histogram(sim, bins=20, range=(-1.0, 1.0), density=False)
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(20.0))
    return dispersion, entropy


@torch.no_grad()
def clip_patch_feature(clip_model, preprocess, img_path, device):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    out = visual_outputs(clip_model, x, adapter=None, grad=False)
    patch = normalized_patch_feature(out).float().cpu().numpy().reshape(SIDE, SIDE, -1)
    return patch


def gt_morph(mask):
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {"small": False, "thin": False, "has_anomaly": False}
    t = torch.tensor(m.astype(np.float32)).view(1, 1, *m.shape)
    p = F.interpolate(t, size=(SIDE * 2, SIDE * 2), mode="nearest").view(SIDE * 2, SIDE * 2).numpy() > 0.5
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = m & ~er
    return {"small": bool(p.sum() <= 8), "thin": bool(boundary.sum() / max(m.sum(), 1) >= 0.35), "has_anomaly": True}


@torch.no_grad()
def dataset_structure_stats(dataset, metas, generator, clip_model, preprocess, device, smoke=False, force=False):
    rows = []
    n = min(24, len(metas)) if smoke else len(metas)
    for i, meta in enumerate(metas[:n]):
        if i % 25 == 0:
            print(f"[SNTS:structure] {dataset} {i}/{n}", flush=True)
        img = Image.open(meta["img_path"]).convert("RGB")
        w, h = img.size
        cached = None if force else load_structure(dataset, i)
        if cached is None:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                anns = generator.generate(np.asarray(img, dtype=np.uint8).copy())
            masks, areas, stability, pred_iou = retained_sam_masks(anns, h, w)
            save_structure(dataset, i, masks, areas, stability, pred_iou)
        else:
            masks = [np.asarray(m, dtype=bool) for m in cached["masks"]]
            areas = [float(x) for x in cached["areas"]]
            stability = [float(x) for x in cached["stability"]]
            pred_iou = [float(x) for x in cached["pred_iou"]]
        overlap, nested, depth = overlap_and_nesting(masks)
        feat = clip_patch_feature(clip_model, preprocess, meta["img_path"], device)
        dispersion, local_entropy = feature_stats(feat)
        area_ratios = np.asarray(areas, dtype=np.float64) / max(float(h * w), 1.0)
        medium_small = int(np.sum((area_ratios >= 0.002) & (area_ratios <= 0.18)))
        small = int(np.sum((area_ratios >= 0.002) & (area_ratios <= 0.03)))
        morph = gt_morph(meta["mask"])
        rows.append(
            {
                "dataset": dataset,
                "category": meta["category"],
                "image_index": i,
                "label": int(meta["label"]),
                "number_retained_sam_masks": int(len(masks)),
                "mask_area_mean": float(np.mean(area_ratios)) if area_ratios.size else 0.0,
                "mask_area_median": float(np.median(area_ratios)) if area_ratios.size else 0.0,
                "mask_area_entropy": mask_entropy(areas, h * w),
                "mask_overlap_ratio": overlap,
                "nested_mask_ratio": nested,
                "average_hierarchy_depth": depth,
                "foreground_fragmentation_proxy": float(len(masks) * (1.0 + overlap) * (1.0 + nested)),
                "number_medium_small_structural_regions": medium_small,
                "number_small_structural_regions": small,
                "edge_density": edge_density(img),
                "clip_patch_feature_dispersion": dispersion,
                "local_patch_feature_entropy": local_entropy,
                "small_anomaly": morph["small"],
                "thin_anomaly": morph["thin"],
                "has_anomaly": morph["has_anomaly"],
            }
        )
    return rows


def structure_summary(rows):
    summary = []
    keys = [
        "number_retained_sam_masks",
        "mask_area_entropy",
        "mask_overlap_ratio",
        "nested_mask_ratio",
        "average_hierarchy_depth",
        "foreground_fragmentation_proxy",
        "number_medium_small_structural_regions",
        "edge_density",
        "clip_patch_feature_dispersion",
        "local_patch_feature_entropy",
    ]
    for ds in DATASETS:
        dr = [r for r in rows if r["dataset"] == ds]
        out = {"dataset": ds, "n_images": len(dr)}
        for k in keys:
            out[f"{k}_mean"] = finite_mean([fnum(r, k) for r in dr])
            out[f"{k}_median"] = finite_median([fnum(r, k) for r in dr])
        summary.append(out)
    return summary


def structure_hypothesis_supported(summary, rows):
    by = {r["dataset"]: r for r in summary}
    visa = by.get("VisA", {})
    mvtec = by.get("MVTec", {})
    score_terms = []
    for k in (
        "number_retained_sam_masks_mean",
        "mask_area_entropy_mean",
        "nested_mask_ratio_mean",
        "mask_overlap_ratio_mean",
        "foreground_fragmentation_proxy_mean",
        "clip_patch_feature_dispersion_mean",
        "local_patch_feature_entropy_mean",
    ):
        v, m = fnum(visa, k), fnum(mvtec, k)
        if np.isfinite(v) and np.isfinite(m):
            score_terms.append(1.0 if v > m * 1.10 else 0.0)
    focus = [r for r in rows if r["dataset"] == "VisA" and r["category"] in {"macaroni2", "pcb1", "pcb2", "pcb3"}]
    other = [r for r in rows if r["dataset"] == "VisA" and r["category"] not in {"macaroni2", "pcb1", "pcb2", "pcb3"}]
    focus_complex = finite_mean([fnum(r, "foreground_fragmentation_proxy") for r in focus])
    other_complex = finite_mean([fnum(r, "foreground_fragmentation_proxy") for r in other])
    focus_ok = np.isfinite(focus_complex) and np.isfinite(other_complex) and focus_complex > other_complex * 1.05
    votes = int(np.sum(score_terms)) + int(focus_ok)
    return votes >= 3, {"structure_support_votes": votes, "visa_focus_complexity": focus_complex, "visa_other_complexity": other_complex}


def load_e0_maps(source, target, metas):
    p = ROOT / "results" / "phase_fova_field_of_view_amplification_audit" / "formal" / "MAP_CACHE" / f"{pair_slug(source, target)}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing locked LRP map cache: {p}")
    data = np.load(p, allow_pickle=True)
    q = np.asarray(data["q_lrp"], dtype=np.float32)
    return q[: len(metas)]


def resized_region_masks(dataset, image_index, score_shape):
    cached = load_structure(dataset, image_index)
    if cached is None:
        return [], np.asarray([], dtype=np.float32)
    masks = cached["masks"]
    areas = cached["areas"]
    out = []
    for m in masks:
        t = torch.tensor(np.asarray(m, dtype=np.float32)).view(1, 1, *m.shape)
        mm = F.interpolate(t, size=score_shape, mode="nearest").view(*score_shape).numpy() > 0.5
        if mm.any():
            out.append(mm)
    return out, np.asarray(areas[: len(out)], dtype=np.float32)


def assign_smallest_and_parent(masks, areas, shape):
    smallest = -np.ones(shape, dtype=np.int32)
    parent = -np.ones(shape, dtype=np.int32)
    if len(masks) == 0:
        return smallest, parent
    order = np.argsort(np.asarray(areas, dtype=np.float64))
    cover_lists = [[int(idx) for idx in order if masks[int(idx)][y, x]] for y in range(shape[0]) for x in range(shape[1])]
    for flat, cov in enumerate(cover_lists):
        if not cov:
            continue
        y, x = divmod(flat, shape[1])
        smallest[y, x] = cov[0]
        if len(cov) > 1:
            parent[y, x] = cov[1]
    return smallest, parent


def region_medians(q, masks):
    med, mad = {}, {}
    for idx, m in enumerate(masks):
        vals = np.asarray(q[m], dtype=np.float32)
        if vals.size == 0:
            continue
        mi = float(np.median(vals))
        med[idx] = mi
        mad[idx] = float(np.median(np.abs(vals - mi)))
    return med, mad


def apply_snts(q, dataset, image_index):
    q = np.asarray(q, dtype=np.float32)
    masks, areas = resized_region_masks(dataset, image_index, q.shape)
    smallest, parent = assign_smallest_and_parent(masks, areas, q.shape)
    med, mad = region_medians(q, masks)
    e1 = q.copy()
    e2 = q.copy()
    e3 = q.copy()
    for y in range(q.shape[0]):
        for x in range(q.shape[1]):
            rid = int(smallest[y, x])
            if rid < 0 or rid not in med:
                continue
            r_small = q[y, x] - med[rid]
            pos = max(0.0, float(r_small))
            e1[y, x] = q[y, x] + LAMBDA_RESCUE * pos
            z = min(max(float(r_small) / (mad.get(rid, 0.0) + EPS), 0.0), Z_MAX)
            e2[y, x] = q[y, x] + LAMBDA_RESCUE * z
            pid = int(parent[y, x])
            if pid >= 0 and pid in med:
                r_parent = q[y, x] - med[pid]
                e3[y, x] = q[y, x] + LAMBDA_RESCUE * min(pos, max(0.0, float(r_parent)))
    return e1.astype(np.float32), e2.astype(np.float32), e3.astype(np.float32)


def evaluate_maps(source, target, maps, metas, variants):
    raw_rows, cat_rows = [], []
    for variant in variants:
        by_cat = {}
        for i, meta in enumerate(metas[: len(maps[variant])]):
            d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "scores": []})
            score = np.asarray(maps[variant][i], dtype=np.float32)
            d["labels"].append(int(meta["label"]))
            d["image_scores"].append(top1_image_score(score))
            d["masks"].append(np.asarray(meta["mask"], dtype=np.uint8))
            d["scores"].append(score)
        fr_cat = []
        for cat, vals in sorted(by_cat.items()):
            resized = [torch_resize(s, m.shape, align_corners=True) for s, m in zip(vals["scores"], vals["masks"])]
            met = summarize_category(vals["labels"], vals["image_scores"], vals["masks"], resized)
            row = {
                "source": source,
                "dataset": target,
                "category": cat,
                "variant": variant,
                "protocol": "M_FR",
                "aggregation": "CATEGORY",
                "score_resolution": f"{vals['scores'][0].shape[0]}x{vals['scores'][0].shape[1]}",
            }
            for k, v in met.items():
                row[f"M_FR_{k.replace('-', '_')}"] = v
            cat_rows.append(row)
            fr_cat.append(row)
        raw = {
            "source": source,
            "dataset": target,
            "variant": variant,
            "aggregation": "CATEGORY_MACRO",
            "score_resolution": f"{maps[variant][0].shape[0]}x{maps[variant][0].shape[1]}",
        }
        for met in ("I_AUROC", "I_AP", "I_F1max", "P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
            vals = np.asarray([fnum(r, f"M_FR_{met}") for r in fr_cat], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            raw[f"M_FR_{met}"] = float(vals.mean()) if vals.size else float("nan")
        raw_rows.append(raw)
    return raw_rows, cat_rows


def diagnostic_rows(source, target, maps, metas, variants):
    rows = []
    for i, meta in enumerate(metas[: len(maps[E0])]):
        mask = np.asarray(meta["mask"], dtype=bool)
        if mask.size == 0:
            continue
        q0 = torch_resize(maps[E0][i], mask.shape, align_corners=True)
        thr0 = np.percentile(q0.reshape(-1), 95)
        pred0 = q0 >= thr0
        fn0 = mask & (~pred0)
        tn0 = (~mask) & (~pred0)
        tp0 = mask & pred0
        morph = gt_morph(mask)
        for variant in variants:
            if variant == E0:
                continue
            q = torch_resize(maps[variant][i], mask.shape, align_corners=True)
            thr = np.percentile(q.reshape(-1), 95)
            pred = q >= thr
            rows.append(
                {
                    "source": source,
                    "dataset": target,
                    "category": meta["category"],
                    "image_index": i,
                    "variant": variant,
                    "small": morph["small"],
                    "thin": morph["thin"],
                    "mean_score_delta_anomaly": float((q - q0)[mask].mean()) if mask.any() else float("nan"),
                    "mean_score_delta_normal": float((q - q0)[~mask].mean()) if (~mask).any() else float("nan"),
                    "E0_FN_to_SNTS_TP": float((fn0 & pred).sum() / max(fn0.sum(), 1)),
                    "E0_TN_to_SNTS_FP": float((tn0 & pred).sum() / max(tn0.sum(), 1)),
                    "E0_TP_to_SNTS_FN": float((tp0 & (~pred)).sum() / max(tp0.sum(), 1)),
                }
            )
    return rows


def run_pair(source, target, metas):
    cached_path = MAP_CACHE / f"{pair_slug(source, target)}.npz"
    if cached_path.exists():
        data = np.load(cached_path, allow_pickle=True)
        maps = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        if all(k in maps for k in (E0, E1, E2)) and len(maps[E0]) == len(metas):
            return maps
    base = load_e0_maps(source, target, metas)
    maps = {E0: [], E1: [], E2: [], E3: []}
    for i, q in enumerate(base):
        if i % 100 == 0:
            print(f"[SNTS:maps] {source}->{target} {i}/{len(base)}", flush=True)
        e1, e2, e3 = apply_snts(q, target, i)
        maps[E0].append(np.asarray(q, dtype=np.float32))
        maps[E1].append(e1)
        maps[E2].append(e2)
        maps[E3].append(e3)
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached_path, **{k: np.asarray(v, dtype=np.float16) for k, v in maps.items()})
    return {k: np.asarray(v, dtype=np.float32) for k, v in maps.items()}


def visa_category_analysis(cat_rows, structure_rows):
    by = {(r["category"], r["variant"]): r for r in cat_rows if r["source"] == "Full MVTec" and r["dataset"] == "VisA"}
    s_by_cat = {}
    for r in structure_rows:
        if r["dataset"] == "VisA":
            s_by_cat.setdefault(r["category"], []).append(r)
    rows = []
    cats = sorted({c for c, _v in by.keys()})
    for cat in cats:
        base = by.get((cat, E0), {})
        row = {
            "source": "Full MVTec",
            "dataset": "VisA",
            "category": cat,
            "focus_category": cat in FOCUS_VISA_CATEGORIES,
            "E0_P_AP": fnum(base, "M_FR_P_AP"),
        }
        for v in (E1, E2, E3):
            cur = by.get((cat, v), {})
            row[f"{v}_P_AP"] = fnum(cur, "M_FR_P_AP")
            row[f"{v}_Delta_P_AP"] = fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP")
        sr = s_by_cat.get(cat, [])
        for k in ("foreground_fragmentation_proxy", "number_retained_sam_masks", "nested_mask_ratio", "clip_patch_feature_dispersion", "local_patch_feature_entropy"):
            row[f"{k}_mean"] = finite_mean([fnum(x, k) for x in sr])
        rows.append(row)
    return rows


def correlation_analysis(visa_cat):
    rows = []
    factors = ("foreground_fragmentation_proxy_mean", "number_retained_sam_masks_mean", "nested_mask_ratio_mean", "clip_patch_feature_dispersion_mean", "local_patch_feature_entropy_mean")
    metrics = ("E0_P_AP", f"{E1}_Delta_P_AP", f"{E2}_Delta_P_AP", f"{E3}_Delta_P_AP")
    for metric in metrics:
        y = [fnum(r, metric) for r in visa_cat]
        for factor in factors:
            rows.append({"dataset": "VisA", "metric": metric, "factor": factor, "spearman": safe_spearman([fnum(r, factor) for r in visa_cat], y), "n": len(visa_cat), "note": "exploratory category-level n≈12"})
    return rows


def decide(raw_rows, visa_cat):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw_rows}
    best = None
    for v in (E1, E2, E3):
        deltas = []
        for source, target in PAIRS:
            cur = by.get((source, target, v), {})
            base = by.get((source, target, E0), {})
            deltas.append(fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"))
        arr = np.asarray(deltas, dtype=np.float64)
        score = np.nanmean(arr)
        if best is None or score > best["mean_delta"]:
            best = {"variant": v, "deltas": arr, "mean_delta": float(score)}
    if best is None:
        return "STRUCTURE_NORMALIZATION_HAS_NO_COMPETITIVE_VALUE", {}
    visa_delta, btad_delta, mpdd_delta = best["deltas"]
    focus = [r for r in visa_cat if r["focus_category"]]
    focus_gain = finite_mean([fnum(r, f"{best['variant']}_Delta_P_AP") for r in focus])
    all_gain = finite_mean([fnum(r, f"{best['variant']}_Delta_P_AP") for r in visa_cat])
    complex_ok = np.isfinite(focus_gain) and np.isfinite(all_gain) and focus_gain > all_gain
    positive = int(np.sum(best["deltas"] > 0))
    info = {
        "best_variant": best["variant"],
        "visa_delta_pap": float(visa_delta),
        "btad_delta_pap": float(btad_delta),
        "mpdd_delta_pap": float(mpdd_delta),
        "mean_delta_pap": best["mean_delta"],
        "positive_transfers": positive,
        "focus_category_mean_delta": focus_gain,
        "all_visa_category_mean_delta": all_gain,
    }
    if visa_delta >= 1.5 and btad_delta >= -0.3 and mpdd_delta >= -0.3 and best["mean_delta"] > 0.3 and positive >= 2 and complex_ok:
        return "STRUCTURE_NORMALIZATION_HAS_COMPETITIVE_VALUE", info
    if visa_delta >= 0.7 and btad_delta > -0.5 and mpdd_delta > -0.5 and best["mean_delta"] >= 0 and complex_ok:
        return "STRUCTURE_NORMALIZATION_HAS_LIMITED_BUT_USEFUL_VALUE", info
    return "STRUCTURE_NORMALIZATION_HAS_NO_COMPETITIVE_VALUE", info


def write_report(summary, raw_rows, visa_cat, corr_rows, outcome, info, runtime):
    lines = [
        "# Phase SNTS: Structure-Normalized Two-State Final Pilot",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "SAM2 was used only as a frozen structural reference frame. No region CLIP re-encoding, target statistics, target calibration, learned gate, or parameter update was used.",
        "",
        "## Part A: Dataset Structure",
        "",
        "| Dataset | Masks | Entropy | Overlap | Nesting | Fragmentation | CLIP Dispersion | Local Entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['dataset']} | {fnum(r,'number_retained_sam_masks_mean'):.3f} | {fnum(r,'mask_area_entropy_mean'):.4f} | "
            f"{fnum(r,'mask_overlap_ratio_mean'):.4f} | {fnum(r,'nested_mask_ratio_mean'):.4f} | {fnum(r,'foreground_fragmentation_proxy_mean'):.3f} | "
            f"{fnum(r,'clip_patch_feature_dispersion_mean'):.4f} | {fnum(r,'local_patch_feature_entropy_mean'):.4f} |"
        )
    lines.extend(["", "## Pilot Metrics", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO | ΔP-AP |", "|---|---|---|---:|---:|---:|---:|---:|"])
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw_rows}
    for source, target in PAIRS:
        base = by.get((source, target, E0), {})
        for v in VARIANTS:
            r = by.get((source, target, v), {})
            delta = fnum(r, "M_FR_P_AP") - fnum(base, "M_FR_P_AP") if v != E0 else 0.0
            lines.append(f"| {source} | {target} | {v} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} | {delta:.4f} |")
    lines.extend(["", "## VisA Focus Categories", "", "| Category | E0 P-AP | E1 Δ | E2 Δ | E3 Δ | Fragmentation | Nesting |", "|---|---:|---:|---:|---:|---:|---:|"])
    for r in visa_cat:
        if r["focus_category"]:
            lines.append(f"| {r['category']} | {fnum(r,'E0_P_AP'):.4f} | {fnum(r,f'{E1}_Delta_P_AP'):.4f} | {fnum(r,f'{E2}_Delta_P_AP'):.4f} | {fnum(r,f'{E3}_Delta_P_AP'):.4f} | {fnum(r,'foreground_fragmentation_proxy_mean'):.3f} | {fnum(r,'nested_mask_ratio_mean'):.4f} |")
    lines.extend(["", "## Exploratory Correlations", "", "| Metric | Factor | Spearman |", "|---|---|---:|"])
    for r in corr_rows:
        if r["metric"] in ("E0_P_AP", f"{info.get('best_variant', E1)}_Delta_P_AP"):
            lines.append(f"| {r['metric']} | {r['factor']} | {fnum(r,'spearman'):.4f} |")
    lines.extend(["", "## Decision Notes", ""])
    if info:
        lines.append(f"- Best variant: `{info.get('best_variant')}`.")
        lines.append(f"- MVTec->VisA ΔP-AP: `{info.get('visa_delta_pap', float('nan')):.4f}`; BTAD: `{info.get('btad_delta_pap', float('nan')):.4f}`; VisA->MPDD: `{info.get('mpdd_delta_pap', float('nan')):.4f}`.")
        lines.append(f"- Mean ΔP-AP: `{info.get('mean_delta_pap', float('nan')):.4f}`; positive transfers: `{info.get('positive_transfers')}/3`.")
    if outcome == "STRUCTURE_NORMALIZATION_HAS_NO_COMPETITIVE_VALUE":
        lines.append("- Stop VisA-directed structural modification and freeze the method for paper writing.")
    lines.append(f"- Runtime: `{runtime:.1f}s`.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    STRUCTURE_CACHE.mkdir(parents=True, exist_ok=True)
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"

    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    generator = load_sam_generator(device)
    all_structure_rows = []
    metas_by_dataset = {}
    for ds in DATASETS:
        metas = load_dataset_metas(ds, device, smoke=args.smoke)
        metas_by_dataset[ds] = metas
        all_structure_rows.extend(dataset_structure_stats(ds, metas, generator, clip_model, preprocess, device, smoke=args.smoke, force=args.force_structure))
        write_csv(OUT / "DATASET_STRUCTURE_STATS.csv", all_structure_rows)
    summary = structure_summary(all_structure_rows)
    write_csv(OUT / "DATASET_STRUCTURE_SUMMARY.csv", summary)
    supported, support_info = structure_hypothesis_supported(summary, all_structure_rows)

    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.part_a_only or not supported:
        outcome = "PART_A_ONLY_COMPLETE" if args.part_a_only else "STRUCTURAL_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, **support_info, "runtime_sec": time.time() - start})
        write_report(summary, [], [], [], outcome, support_info, time.time() - start)
        print(f"[SNTS] stop after Part A: {outcome}", flush=True)
        return

    raw_rows, cat_rows, diag_rows = [], [], []
    for source, target in (PAIRS[:1] if args.smoke else PAIRS):
        metas = load_metas_direct(source, target, device, smoke=args.smoke)
        maps = run_pair(source, target, metas)
        variants = (E0, E1, E2)
        rr, cc = evaluate_maps(source, target, maps, metas, variants)
        raw_rows.extend(rr)
        cat_rows.extend(cc)
        diag_rows.extend(diagnostic_rows(source, target, maps, metas, variants))
        write_csv(OUT / "RAW_METRICS.csv", raw_rows)
        write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", cat_rows)
        write_csv(OUT / "SNTS_DIAGNOSTICS.csv", diag_rows)

    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw_rows}
    e1_or_e2_positive_visa = False
    for v in (E1, E2):
        e0 = by.get(("Full MVTec", "VisA", E0), {})
        cur = by.get(("Full MVTec", "VisA", v), {})
        e1_or_e2_positive_visa = e1_or_e2_positive_visa or (fnum(cur, "M_FR_P_AP") - fnum(e0, "M_FR_P_AP") > 0)
    if e1_or_e2_positive_visa and not args.smoke:
        raw_rows, cat_rows, diag_rows = [], [], []
        for source, target in PAIRS:
            metas = load_metas_direct(source, target, device, smoke=False)
            maps = run_pair(source, target, metas)
            rr, cc = evaluate_maps(source, target, maps, metas, VARIANTS)
            raw_rows.extend(rr)
            cat_rows.extend(cc)
            diag_rows.extend(diagnostic_rows(source, target, maps, metas, VARIANTS))
            write_csv(OUT / "RAW_METRICS.csv", raw_rows)
            write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", cat_rows)
            write_csv(OUT / "SNTS_DIAGNOSTICS.csv", diag_rows)

    visa_cat = visa_category_analysis(cat_rows, all_structure_rows)
    corr_rows = correlation_analysis(visa_cat)
    outcome, info = decide(raw_rows, visa_cat)
    info.update(support_info)
    write_csv(OUT / "VISA_STRUCTURE_CORRELATION.csv", corr_rows)
    write_csv(OUT / "VISA_CATEGORY_STRUCTURE_GAIN.csv", visa_cat)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, **info, "runtime_sec": time.time() - start})
    write_report(summary, raw_rows, visa_cat, corr_rows, outcome, info, time.time() - start)
    print(f"[SNTS] complete: {outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--part-a-only", action="store_true")
    p.add_argument("--force-structure", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
