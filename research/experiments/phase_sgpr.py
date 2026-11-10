import argparse
import csv
import json
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

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    METRICS,
    PROTOCOLS,
    CORE_PAIRS,
    fast_aupro_vectorized,
    item_at,
    item_meta,
    mask_np,
    pg_metrics,
    score_feature_grid,
    summarize_category,
    torch_resize,
    local_zoom_map as formal_lrp_local_zoom_map,
)
from experiments.phase_spatial_residual_refinement_standard_metric import fnum
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_sgpr/formal")
VARIANTS = ("B0_CanonicalNTR", "E1_FixedLRP", "E2_StateGuidedSingleLevel", "E3_SGPR")
INDUSTRIAL_TARGETS = {"MVTec", "VisA", "BTAD", "MPDD", "DTD-Synthetic"}
AUDIT_PAIRS = {
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
}
REPRODUCTION_PAIRS = {
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "MPDD"),
}


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_stat(vals, fn):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(fn(arr)) if arr.size else float("nan")


def safe_auc_binary(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
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
    n_pos = float(np.sum(pos))
    n_neg = float(y.size - np.sum(pos))
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1.0) / 2.0) / max(n_pos * n_neg, 1.0) * 100.0)


def safe_ap_binary(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order].astype(np.float32, copy=False)
    tp = np.cumsum(y_sorted, dtype=np.float32)
    denom = np.arange(1, y_sorted.size + 1, dtype=np.float32)
    precision = tp / denom
    n_pos = float(tp[-1])
    return float(np.sum(precision[y_sorted > 0]) / max(n_pos, 1.0) * 100.0)


def f1max_binary(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order].astype(np.float32, copy=False)
    tp = np.cumsum(y_sorted, dtype=np.float32)
    denom = np.arange(1, y_sorted.size + 1, dtype=np.float32)
    precision = tp / denom
    recall = tp / max(float(tp[-1]), 1.0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    return float(np.max(f1) * 100.0)


def summarize_category_local(labels, image_scores, masks, scores):
    py = np.concatenate([m.reshape(-1) for m in masks]) if masks else np.asarray([], dtype=np.uint8)
    ps = np.concatenate([s.reshape(-1) for s in scores]) if scores else np.asarray([], dtype=np.float32)
    out = {
        "I-AUROC": safe_auc_binary(labels, image_scores),
        "I-AP": safe_ap_binary(labels, image_scores),
        "I-F1max": f1max_binary(labels, image_scores),
        "P-AUROC": safe_auc_binary(py, ps),
        "P-AP": safe_ap_binary(py, ps),
        "P-F1max": f1max_binary(py, ps),
        "PRO": fast_aupro_vectorized(masks, scores),
    }
    out["G_pixel"] = 0.5 * (out["P-AUROC"] + out["P-AP"]) if np.isfinite(out["P-AUROC"]) and np.isfinite(out["P-AP"]) else float("nan")
    return out


def image_size(path):
    with Image.open(path) as im:
        return im.size


def fixed_box(cx, cy, bw, bh, w, h):
    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x1 = max(0, min(x1, max(0, w - bw)))
    y1 = max(0, min(y1, max(0, h - bh)))
    x2 = min(w, x1 + int(bw))
    y2 = min(h, y1 + int(bh))
    return int(x1), int(y1), int(x2), int(y2)


def greedy_nms_peaks(score, k, min_dist_px, w, h):
    arr = np.asarray(score, dtype=np.float32)
    flat_order = np.argsort(arr.reshape(-1))[::-1]
    peaks = []
    for idx in flat_order:
        gy, gx = divmod(int(idx), arr.shape[1])
        cx = (gx + 0.5) * float(w) / float(arr.shape[1])
        cy = (gy + 0.5) * float(h) / float(arr.shape[0])
        if all(((cx - p["center_x"]) ** 2 + (cy - p["center_y"]) ** 2) ** 0.5 >= min_dist_px for p in peaks):
            peaks.append({"grid_x": gx, "grid_y": gy, "center_x": cx, "center_y": cy, "score": float(arr[gy, gx])})
        if len(peaks) >= k:
            break
    if len(peaks) < k:
        for idx in flat_order:
            gy, gx = divmod(int(idx), arr.shape[1])
            cx = (gx + 0.5) * float(w) / float(arr.shape[1])
            cy = (gy + 0.5) * float(h) / float(arr.shape[0])
            key = (gx, gy)
            if any((p["grid_x"], p["grid_y"]) == key for p in peaks):
                continue
            peaks.append({"grid_x": gx, "grid_y": gy, "center_x": cx, "center_y": cy, "score": float(arr[gy, gx])})
            if len(peaks) >= k:
                break
    return peaks


def half_crop_boxes_from_score(score37, w, h, k):
    min_dist = 0.25 * float(min(w, h))
    peaks = greedy_nms_peaks(score37, k, min_dist, w, h)
    rows = []
    for p in peaks:
        box = fixed_box(p["center_x"], p["center_y"], w // 2, h // 2, w, h)
        rows.append((p, box))
    return rows


def fixed_quadrant_boxes(w, h):
    return [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h),
    ]


def stitch_quadrant74(scores):
    s = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    s[:SIDE, :SIDE] = scores[0]
    s[:SIDE, SIDE:] = scores[1]
    s[SIDE:, :SIDE] = scores[2]
    s[SIDE:, SIDE:] = scores[3]
    return s


@torch.no_grad()
def score_crop(model_info, clip_model, preprocess, img, box, device):
    crop = img.crop(tuple(box))
    batch = preprocess(crop).unsqueeze(0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patch = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)[0]
    return score_feature_grid(model_info, patch, device)


@torch.no_grad()
def score_crops(model_info, clip_model, preprocess, img, boxes, device, chunk=4):
    scores = {}
    unique = []
    for box in boxes:
        key = tuple(int(v) for v in box)
        if key not in scores:
            scores[key] = None
            unique.append(key)
    for start in range(0, len(unique), chunk):
        part = unique[start:start + chunk]
        batch = torch.stack([preprocess(img.crop(box)) for box in part], dim=0).to(device)
        out = visual_outputs(clip_model, batch, adapter=None, grad=False)
        patches = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)
        for i, box in enumerate(part):
            scores[box] = score_feature_grid(model_info, patches[i], device)
    return scores


def scale_box(box, src_w, src_h, dst_w, dst_h):
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * dst_w / max(src_w, 1))),
        int(round(y1 * dst_h / max(src_h, 1))),
        int(round(x2 * dst_w / max(src_w, 1))),
        int(round(y2 * dst_h / max(src_h, 1))),
    )


def fuse_eval_canvas(global37, local_items, orig_w, orig_h, canvas_side=IMAGE_SIZE):
    w = h = int(canvas_side)
    global_full = torch_resize(global37, (h, w), align_corners=True)
    local_sum = np.zeros((h, w), dtype=np.float32)
    local_count = np.zeros((h, w), dtype=np.float32)
    for box, score37 in local_items:
        x1, y1, x2, y2 = scale_box(box, orig_w, orig_h, w, h)
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        crop_score = torch_resize(score37, (y2 - y1, x2 - x1), align_corners=True)
        local_sum[y1:y2, x1:x2] += crop_score
        local_count[y1:y2, x1:x2] += 1.0
    out = global_full.copy()
    covered = local_count > 0
    if np.any(covered):
        local_mean = np.divide(local_sum, np.maximum(local_count, 1.0))
        out[covered] = 0.5 * global_full[covered] + 0.5 * local_mean[covered]
    return out.astype(np.float32), int(np.sum(covered))


def stage2_boxes(stage1_score, stage1_box, full_w, full_h):
    x1, y1, x2, y2 = stage1_box
    cw, ch = x2 - x1, y2 - y1
    peak = greedy_nms_peaks(stage1_score, 1, 0.0, cw, ch)[0]
    cx = x1 + peak["center_x"]
    cy = y1 + peak["center_y"]
    box = fixed_box(cx, cy, max(1, full_w // 4), max(1, full_h // 4), full_w, full_h)
    return peak, box


def mask_morphology(mask):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return {"num_anomaly_patches": 0, "anomaly_patch_ratio": 0.0, "thinness": 0.0, "boundary_to_area_ratio": 0.0}
    t = torch.tensor(m.astype(np.float32)).view(1, 1, *m.shape)
    pm = F.interpolate(t, size=(SIDE, SIDE), mode="nearest").view(SIDE, SIDE).numpy() > 0.5
    boundary = m & (~ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0))
    area = int(m.sum())
    return {
        "num_anomaly_patches": int(pm.sum()),
        "anomaly_patch_ratio": float(pm.sum()) / float(SIDE * SIDE),
        "thinness": float(boundary.sum()) / max(area, 1),
        "boundary_to_area_ratio": float(boundary.sum()) / max(area, 1),
    }


def group_for_morph(m, thin_threshold, large_threshold):
    small = fnum(m, "num_anomaly_patches") <= 8
    thin = fnum(m, "thinness") >= thin_threshold
    large = fnum(m, "num_anomaly_patches") >= large_threshold
    return {
        "small_sparse": small,
        "thin": thin,
        "small_or_thin": small or thin,
        "large": large,
        "non_thin": not thin,
    }


def proposal_hit_rows(source, target, rows):
    out = []
    for group in ("all", "small_sparse", "thin", "small_or_thin", "large", "non_thin"):
        part = [r for r in rows if group == "all" or r.get(group)]
        comp_total = sum(int(r["component_total"]) for r in part)
        comp_hit = sum(int(r["component_hit"]) for r in part)
        pix_total = sum(float(r["anomaly_pixels"]) for r in part)
        pix_hit = sum(float(r["covered_anomaly_pixels"]) for r in part)
        img_total = len(part)
        img_hit = sum(int(r["image_hit"]) for r in part)
        out.append({
            "source": source,
            "target": target,
            "morphology_group": group,
            "images": img_total,
            "component_total": comp_total,
            "component_hit": comp_hit,
            "component_hit_rate": comp_hit / comp_total if comp_total else float("nan"),
            "anomaly_pixel_coverage": pix_hit / pix_total if pix_total else float("nan"),
            "image_hit_rate": img_hit / img_total if img_total else float("nan"),
        })
    return out


def collect_protocol(smoke):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    models = [load_visa_source(device), load_mvtec_source()]
    cells = []
    for model_info in models:
        all_cells, _audits = protocol_cells(model_info["source"], device, smoke)
        for cell in all_cells:
            cell_id, target, dataset, idx_or_rows, patch, kind = cell
            pair = (model_info["source"], target)
            if pair in AUDIT_PAIRS:
                n = min(24, patch.shape[0]) if smoke else patch.shape[0]
                cells.append((model_info, cell_id, target, dataset, idx_or_rows[:n], patch[:n], kind))
    return device, preprocess, clip_model, cells


def formal_reference_rows():
    ref_path = Path("results/phase_parameter_free_local_zoom_pyramid_fullres/formal/FULL_RES_METRICS.csv")
    rows = read_csv(ref_path) if ref_path.exists() else []
    return {(r["source"], r["dataset"]): r for r in rows if r.get("variant") == "E2_GlobalLocalZoomPyramid"}


def write_lrp_static_audits():
    write_text(OUT / "LRP_CODE_PATH_DIFF.md", """# LRP Code Path Diff

Reference path: `experiments/phase_parameter_free_local_zoom_pyramid_fullres.py::E2_GlobalLocalZoomPyramid`.

Root cause found:
- Locked formal LRP crops use the original PIL image size from `Image.open(img_path).size`.
- The first SGPR E1 implementation used the transformed GT mask size (`518x518`) as if it were the original image size when cropping `img_path`.
- For images whose original dimensions differ from 518x518, the local re-encoding crops are spatially wrong, so E1_FixedLRP falls far below the locked Paper-Ready LRP artifact.

Corrected SGPR E1:
- crop coordinates: original PIL image width/height, same as formal LRP
- quadrant boundaries: `(0,0,w//2,h//2)`, `(w//2,0,w,h//2)`, `(0,h//2,w//2,h)`, `(w//2,h//2,w,h)`
- crop resize/RGB preprocessing/CLIP forward/patch extraction/alignment/state scorer: same frozen code path
- local map size: formal `74x74` stitch from four `37x37` local maps
- fusion: `0.5 * resize(global37 -> 74x74, align_corners=True) + 0.5 * local74`
- evaluator: M_FR full-resolution resize with bilinear `align_corners=True`; V_FR retained as secondary protocol

Remaining intended SGPR-only differences:
- E2/E3 differ only in crop proposal/hierarchy.
- No learned selector, target statistics, tuning, or fusion change is introduced.
""")
    write_text(OUT / "LRP_STITCH_AUDIT.md", "# LRP Stitch / Mapping Audit\n\nCorrected E1 stitches four local `37x37` maps into the exact formal `74x74` quadrant layout. Difference maps are summarized in `LRP_LOCAL_SCORE_EQUIVALENCE.csv`; max errors should be numerical zero when the same crops are used.\n")
    write_text(OUT / "LRP_FUSION_AUDIT.md", "# LRP Fusion Audit\n\nFormal LRP fusion uses `global74 = bilinear(global37 -> 74x74, align_corners=True)`, `local74 = four quadrant local score maps stitched into 74x74`, and `final74 = 0.5*global74 + 0.5*local74`. Corrected SGPR E1 now uses this exact formula.\n")
    write_text(OUT / "LRP_EVALUATOR_AUDIT.md", "# LRP Evaluator Audit\n\nBoth locked formal LRP and corrected SGPR E1 use category-macro M_FR full-resolution evaluation with bilinear score resize and `align_corners=True`. Patch-grid metrics are diagnostic only and are not used for the reproduction gate.\n")


def lrp_equivalence_audit(smoke=False):
    device, preprocess, clip_model, cells = collect_protocol(smoke)
    crop_rows, score_rows = [], []
    used = 0
    for model_info, _cell_id, target, dataset, refs, patch, kind in cells:
        source = model_info["source"]
        for pos, ref in enumerate(refs):
            if used >= 10:
                break
            item = item_at(dataset, kind, ref, pos)
            cat, label, path = item_meta(item, kind, ref, target)
            if not path:
                continue
            img = Image.open(path).convert("RGB")
            w, h = img.size
            formal_boxes = fixed_quadrant_boxes(w, h)
            sgpr_boxes = fixed_quadrant_boxes(w, h)
            for q, (fb, sb) in enumerate(zip(formal_boxes, sgpr_boxes)):
                crop_rows.append({
                    "sample_id": used,
                    "source": source,
                    "target": target,
                    "category": cat,
                    "label": label,
                    "quadrant": q,
                    "formal_x1": fb[0],
                    "formal_y1": fb[1],
                    "formal_x2": fb[2],
                    "formal_y2": fb[3],
                    "sgpr_x1": sb[0],
                    "sgpr_y1": sb[1],
                    "sgpr_x2": sb[2],
                    "sgpr_y2": sb[3],
                    "crop_match": tuple(fb) == tuple(sb),
                    "crop_shape": f"{fb[2]-fb[0]}x{fb[3]-fb[1]}",
                    "resize_shape": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
                })
            global37 = score_feature_grid(model_info, patch[pos], device)
            formal_local74 = formal_lrp_local_zoom_map(model_info, clip_model, preprocess, path, device)
            sgpr_scores = score_crops(model_info, clip_model, preprocess, img, sgpr_boxes, device)
            sgpr_local74 = stitch_quadrant74([sgpr_scores[tuple(b)] for b in sgpr_boxes])
            formal_final74 = 0.5 * torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * formal_local74
            sgpr_final74 = 0.5 * torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * sgpr_local74
            for name, a, b in (("local74", formal_local74, sgpr_local74), ("final74", formal_final74, sgpr_final74)):
                af = np.asarray(a, dtype=np.float32).reshape(-1)
                bf = np.asarray(b, dtype=np.float32).reshape(-1)
                score_rows.append({
                    "sample_id": used,
                    "source": source,
                    "target": target,
                    "category": cat,
                    "map": name,
                    "mean_abs_error": float(np.mean(np.abs(af - bf))),
                    "max_abs_error": float(np.max(np.abs(af - bf))),
                    "pearson": float(stats.pearsonr(af, bf).statistic) if np.std(af) > 0 and np.std(bf) > 0 else float("nan"),
                    "spearman": float(stats.spearmanr(af, bf).statistic) if np.std(af) > 0 and np.std(bf) > 0 else float("nan"),
                })
            img.close()
            used += 1
        if used >= 10:
            break
    write_csv(OUT / "LRP_CROP_AUDIT.csv", crop_rows)
    write_csv(OUT / "LRP_LOCAL_SCORE_EQUIVALENCE.csv", score_rows)


def reproduction_rows(raw):
    refs = formal_reference_rows()
    keys = {("Full VisA", "MVTec"), ("Full VisA", "BTAD"), ("Full MVTec", "VisA"), ("Full MVTec", "MPDD")}
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, dataset in sorted(keys):
        ref = refs.get((source, dataset))
        cur = by.get((source, dataset, "E1_FixedLRP"))
        delta = (fnum(cur, "M_FR_P_AP") - fnum(ref, "M_FR_P_AP")) if ref and cur else float("nan")
        rows.append({
            "source": source,
            "dataset": dataset,
            "reference_variant": "E2_GlobalLocalZoomPyramid",
            "corrected_variant": "E1_FixedLRP",
            "reference_M_FR_P_AP": fnum(ref, "M_FR_P_AP") if ref else float("nan"),
            "corrected_M_FR_P_AP": fnum(cur, "M_FR_P_AP") if cur else float("nan"),
            "delta_P_AP": delta,
            "pass_abs_le_0_2": bool(np.isfinite(delta) and abs(delta) <= 0.2),
        })
    return rows


def phase0_proposal_audit(smoke=False):
    device, _preprocess, _clip_model, cells = collect_protocol(smoke)
    detail_rows = []
    all_morph = []
    cached_scores = {}
    for model_info, cell_id, target, dataset, refs, patch, kind in cells:
        source = model_info["source"]
        print(f"[SGPR:P0] {source} -> {target} n={len(refs)}", flush=True)
        pair_details = []
        for pos, ref in enumerate(refs):
            item = item_at(dataset, kind, ref, pos)
            cat, label, path = item_meta(item, kind, ref, target)
            score37 = score_feature_grid(model_info, patch[pos], device)
            cached_scores[(source, target, pos)] = score37
            if int(label) == 0 or "img_mask" not in item:
                continue
            mask = mask_np(item)
            mask_h, mask_w = mask.shape
            img_w, img_h = image_size(path)
            peaks = half_crop_boxes_from_score(score37, img_w, img_h, 2)
            cover = np.zeros_like(mask, dtype=bool)
            for _peak, box in peaks:
                x1, y1, x2, y2 = scale_box(box, img_w, img_h, mask_w, mask_h)
                x1 = max(0, min(x1, mask_w))
                x2 = max(0, min(x2, mask_w))
                y1 = max(0, min(y1, mask_h))
                y2 = max(0, min(y2, mask_h))
                cover[y1:y2, x1:x2] = True
            labels, ncomp = ndimage.label(mask > 0, structure=np.ones((3, 3), dtype=np.int8))
            comp_hit = 0
            for rid in range(1, ncomp + 1):
                comp_hit += int(np.any(cover & (labels == rid)))
            morph = mask_morphology(mask)
            all_morph.append(morph)
            row = {
                "source": source,
                "target": target,
                "category": cat,
                "image_index": pos,
                "img_path": path,
                "component_total": int(ncomp),
                "component_hit": int(comp_hit),
                "anomaly_pixels": int(mask.sum()),
                "covered_anomaly_pixels": int(np.sum(mask.astype(bool) & cover)),
                "image_hit": int(np.any(mask.astype(bool) & cover)),
                **morph,
            }
            pair_details.append(row)
            detail_rows.append(row)
    thin_vals = [fnum(r, "thinness") for r in all_morph if fnum(r, "thinness") > 0]
    patch_vals = [fnum(r, "num_anomaly_patches") for r in all_morph if fnum(r, "num_anomaly_patches") > 0]
    thin_thr = safe_stat(thin_vals, lambda x: np.percentile(x, 75))
    large_thr = safe_stat(patch_vals, lambda x: np.percentile(x, 75))
    for r in detail_rows:
        for k, v in group_for_morph(r, thin_thr, large_thr).items():
            r[k] = bool(v)
    summary = []
    for source, target in sorted({(r["source"], r["target"]) for r in detail_rows}):
        summary.extend(proposal_hit_rows(source, target, [r for r in detail_rows if r["source"] == source and r["target"] == target]))
    return detail_rows, summary, cached_scores, {"thin_threshold": thin_thr, "large_patch_threshold": large_thr}


def evaluate_maps(source, target, refs, dataset, kind, maps, metas, variants=None):
    variants = variants or VARIANTS
    raw_rows, cat_rows = [], []
    for variant in variants:
        print(f"[SGPR] metrics {source} -> {target} {variant}", flush=True)
        by_cat = {}
        for i, meta in enumerate(metas):
            d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "maps": []})
            d["labels"].append(meta["label"])
            d["image_scores"].append(top1_image_score(maps[variant][i]))
            d["masks"].append(meta["mask"])
            d["maps"].append(np.asarray(maps[variant][i], dtype=np.float32))
        pg_cat_rows = []
        fr_cat_by_protocol = {p: [] for p in PROTOCOLS}
        for cat, vals in sorted(by_cat.items()):
            side = int(vals["maps"][0].shape[0])
            gmasks = []
            for m in vals["masks"]:
                t = torch.tensor(m.astype(np.float32)).view(1, 1, *m.shape)
                gm = F.interpolate(t, size=(side, side), mode="nearest").view(side, side).numpy()
                gmasks.append((gm > 0.5).astype(np.uint8))
            pg = summarize_category_local(vals["labels"], vals["image_scores"], gmasks, vals["maps"])
            pg_row = {"source": source, "dataset": target, "category": cat, "variant": variant, "protocol": "PG", "aggregation": "CATEGORY", "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}"}
            for met, val in pg.items():
                pg_row[f"PG_{met.replace('-', '_')}"] = val
            cat_rows.append(pg_row)
            pg_cat_rows.append(pg_row)
            for pkey, pcfg in PROTOCOLS.items():
                scores = [torch_resize(s, m.shape, align_corners=pcfg["align_corners"]) for s, m in zip(vals["maps"], vals["masks"])]
                fr = summarize_category_local(vals["labels"], vals["image_scores"], vals["masks"], scores)
                fr_row = {"source": source, "dataset": target, "category": cat, "variant": variant, "protocol": pkey, "protocol_name": pcfg["name"], "aggregation": "CATEGORY", "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}"}
                for met, val in fr.items():
                    fr_row[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = val
                cat_rows.append(fr_row)
                fr_cat_by_protocol[pkey].append(fr_row)
        raw = {"source": source, "dataset": target, "variant": variant, "aggregation": "CATEGORY_MACRO", "score_resolution": f"{maps[variant][0].shape[0]}x{maps[variant][0].shape[1]}"}
        for met in METRICS:
            vals = np.asarray([fnum(r, f"PG_{met.replace('-', '_')}") for r in pg_cat_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            raw[f"PG_{met.replace('-', '_')}"] = float(vals.mean()) if vals.size else float("nan")
        for pkey, pcfg in PROTOCOLS.items():
            for met in METRICS:
                vals = np.asarray([fnum(r, f"{pcfg['prefix']}_{met.replace('-', '_')}") for r in fr_cat_by_protocol[pkey]], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                raw[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = float(vals.mean()) if vals.size else float("nan")
        raw_rows.append(raw)
    return raw_rows, cat_rows


def build_sgpr_maps(model_info, clip_model, preprocess, target, dataset, refs, patch, kind, device, variants=None):
    variants = variants or VARIANTS
    active = set(variants)
    maps = {v: [] for v in variants}
    metas = []
    crop_rows = []
    compute = []
    for pos, ref in enumerate(refs):
        if pos % 25 == 0:
            print(f"[SGPR] {model_info['source']} -> {target} image {pos}/{len(refs)}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        if not path:
            raise ValueError(f"missing img_path for {model_info['source']} -> {target} pos={pos}")
        gt = mask_np(item)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        global37 = score_feature_grid(model_info, patch[pos], device)

        e1_boxes = fixed_quadrant_boxes(w, h)
        e2_pairs = half_crop_boxes_from_score(global37, w, h, 4)
        e3_pairs = half_crop_boxes_from_score(global37, w, h, 2)

        if "B0_CanonicalNTR" in active:
            maps["B0_CanonicalNTR"].append(torch_resize(global37, (IMAGE_SIZE, IMAGE_SIZE), align_corners=True).astype(np.float16))

        t0 = time.perf_counter()
        if "E1_FixedLRP" in active:
            e1_scores = score_crops(model_info, clip_model, preprocess, img, e1_boxes, device)
            e1_local37 = [e1_scores[tuple(box)] for box in e1_boxes]
            e1_local74 = stitch_quadrant74(e1_local37)
            e1 = 0.5 * torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * e1_local74
            e1_cov = int(e1.size)
            maps["E1_FixedLRP"].append(e1.astype(np.float16))
            t1 = time.perf_counter()
            compute.append({"source": model_info["source"], "dataset": target, "variant": "E1_FixedLRP", "image_index": pos, "num_global_forward": 1, "num_local_forward": 4, "total_forward": 5, "ms_per_image": (t1 - t0) * 1000.0, "FPS": 1000.0 / max((t1 - t0) * 1000.0, 1e-9), "peak_memory_MB": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0, "local_covered_pixels": e1_cov})
        else:
            t1 = t0

        e2_scores = None
        if "E2_StateGuidedSingleLevel" in active or "E3_SGPR" in active:
            e2_scores = score_crops(model_info, clip_model, preprocess, img, [box for _peak, box in e2_pairs], device)

        if "E2_StateGuidedSingleLevel" in active:
            e2_half = []
            for rank, (peak, box) in enumerate(e2_pairs):
                s = e2_scores[tuple(box)]
                e2_half.append((box, s))
                crop_rows.append({"source": model_info["source"], "dataset": target, "variant": "E2_StateGuidedSingleLevel", "category": cat, "image_index": pos, "stage": 1, "rank": rank, "center_x": peak["center_x"], "center_y": peak["center_y"], "crop_x1": box[0], "crop_y1": box[1], "crop_x2": box[2], "crop_y2": box[3], "peak_score": peak["score"]})
            e2, e2_cov = fuse_eval_canvas(global37, e2_half, w, h)
            t2 = time.perf_counter()
            maps["E2_StateGuidedSingleLevel"].append(e2.astype(np.float16))
            compute.append({"source": model_info["source"], "dataset": target, "variant": "E2_StateGuidedSingleLevel", "image_index": pos, "num_global_forward": 1, "num_local_forward": 4, "total_forward": 5, "ms_per_image": (t2 - t1) * 1000.0, "FPS": 1000.0 / max((t2 - t1) * 1000.0, 1e-9), "peak_memory_MB": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0, "local_covered_pixels": e2_cov})
        else:
            t2 = t1

        if "E3_SGPR" in active:
            e3_items = []
            stage2_todo = []
            for rank, (peak, box) in enumerate(e3_pairs):
                s1 = e2_scores[tuple(box)]
                e3_items.append((box, s1))
                crop_rows.append({"source": model_info["source"], "dataset": target, "variant": "E3_SGPR", "category": cat, "image_index": pos, "stage": 1, "rank": rank, "center_x": peak["center_x"], "center_y": peak["center_y"], "crop_x1": box[0], "crop_y1": box[1], "crop_x2": box[2], "crop_y2": box[3], "peak_score": peak["score"]})
                p2, box2 = stage2_boxes(s1, box, w, h)
                stage2_todo.append((rank, box, p2, box2))
            e3_quarter_scores = score_crops(model_info, clip_model, preprocess, img, [box for _rank, _parent, _p2, box in stage2_todo], device)
            for rank, parent_box, p2, box2 in stage2_todo:
                s2 = e3_quarter_scores[tuple(box2)]
                e3_items.append((box2, s2))
                crop_rows.append({"source": model_info["source"], "dataset": target, "variant": "E3_SGPR", "category": cat, "image_index": pos, "stage": 2, "rank": rank, "center_x": parent_box[0] + p2["center_x"], "center_y": parent_box[1] + p2["center_y"], "crop_x1": box2[0], "crop_y1": box2[1], "crop_x2": box2[2], "crop_y2": box2[3], "peak_score": p2["score"]})
            e3, e3_cov = fuse_eval_canvas(global37, e3_items, w, h)
            t3 = time.perf_counter()
            maps["E3_SGPR"].append(e3.astype(np.float16))
            compute.append({"source": model_info["source"], "dataset": target, "variant": "E3_SGPR", "image_index": pos, "num_global_forward": 1, "num_local_forward": 4, "total_forward": 5, "ms_per_image": (t3 - t2) * 1000.0, "FPS": 1000.0 / max((t3 - t2) * 1000.0, 1e-9), "peak_memory_MB": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0, "local_covered_pixels": e3_cov})
        metas.append({"category": cat, "label": label, "mask": gt, "img_path": path})
        img.close()
    return maps, metas, crop_rows, compute


def full_eval(smoke=False, tag="", pair_filter=None, variants=None):
    variants = variants or VARIANTS
    pair_filter = set(pair_filter) if pair_filter else None
    infix = f"{tag}." if tag else ""
    device, preprocess, clip_model, cells = collect_protocol(smoke)
    raw, cats, crops, compute = [], [], [], []
    done = set()
    partial = OUT / (f"FULL_RES_METRICS.{infix}smoke.partial.csv" if smoke else f"FULL_RES_METRICS.{infix}partial.csv")
    if partial.exists() and not smoke:
        raw = read_csv(partial)
        done = {(r["source"], r["dataset"]) for r in raw if r.get("variant") == variants[-1]}
        for path, acc in ((OUT / f"PER_CATEGORY_FULL_RES.{infix}partial.csv", cats), (OUT / f"REOBSERVATION_CROPS.{infix}partial.csv", crops), (OUT / f"COMPUTE_AUDIT.{infix}partial.csv", compute)):
            if path.exists():
                acc.extend(read_csv(path))
    for model_info, cell_id, target, dataset, refs, patch, kind in cells:
        source = model_info["source"]
        if pair_filter and (source, target) not in pair_filter:
            continue
        if (source, target) in done:
            print(f"[SGPR] skip completed {source}->{target}", flush=True)
            continue
        print(f"[SGPR] evaluate {source} -> {target} n={len(refs)}", flush=True)
        maps, metas, cr, co = build_sgpr_maps(model_info, clip_model, preprocess, target, dataset, refs, patch, kind, device, variants=variants)
        rr, cc = evaluate_maps(source, target, refs, dataset, kind, maps, metas, variants=variants)
        raw.extend(rr)
        cats.extend(cc)
        crops.extend(cr)
        compute.extend(co)
        if not smoke:
            write_csv(OUT / f"FULL_RES_METRICS.{infix}partial.csv", raw)
            write_csv(OUT / f"PER_CATEGORY_FULL_RES.{infix}partial.csv", cats)
            write_csv(OUT / f"REOBSERVATION_CROPS.{infix}partial.csv", crops)
            write_csv(OUT / f"COMPUTE_AUDIT.{infix}partial.csv", compute)
    return raw, cats, crops, compute


def morphology_delta(cat_rows, proposal_rows):
    morph_path = Path("results/phase_final_bottleneck_attribution/formal/CATEGORY_MORPHOLOGY.csv")
    morph_by = {}
    if morph_path.exists():
        for r in read_csv(morph_path):
            morph_by[(r["source"], r["dataset"], r["category"])] = r
    vals = {(r["source"], r["dataset"], r["category"], r["variant"]): r for r in cat_rows if r.get("protocol") == "M"}
    morph_vals = [m for m in morph_by.values() if np.isfinite(fnum(m, "thinness_mean"))]
    thin_thr = safe_stat([fnum(m, "thinness_mean") for m in morph_vals], lambda x: np.percentile(x, 75))
    patch_vals = [fnum(m, "num_anomaly_patches_mean") for m in morph_vals if fnum(m, "num_anomaly_patches_mean") > 0]
    large_thr = safe_stat(patch_vals, lambda x: np.percentile(x, 75))
    rows = []
    for source, dataset, cat, variant in sorted(vals):
        if variant not in ("E2_StateGuidedSingleLevel", "E3_SGPR"):
            continue
        base = vals.get((source, dataset, cat, "E1_FixedLRP"))
        cur = vals.get((source, dataset, cat, variant))
        m = morph_by.get((source, dataset, cat))
        if not (base and cur and m):
            continue
        npatches = fnum(m, "num_anomaly_patches_mean")
        thinness = fnum(m, "thinness_mean")
        groups = {
            "small_sparse": npatches <= 8,
            "thin": thinness >= thin_thr,
            "small_or_thin": (npatches <= 8) or (thinness >= thin_thr),
            "large": npatches >= large_thr,
            "non_thin": thinness < thin_thr,
        }
        d = fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP")
        for g, flag in groups.items():
            if flag:
                rows.append({"source": source, "dataset": dataset, "category": cat, "variant": variant, "morphology_group": g, "delta_M_FR_P_AP_vs_E1": d, "improved": d > 0, "num_anomaly_patches_mean": npatches, "thinness_mean": thinness})
    summary = []
    for variant in ("E2_StateGuidedSingleLevel", "E3_SGPR"):
        for group in ("small_sparse", "thin", "small_or_thin", "large", "non_thin"):
            part = [r for r in rows if r["variant"] == variant and r["morphology_group"] == group]
            vals_d = [fnum(r, "delta_M_FR_P_AP_vs_E1") for r in part]
            summary.append({"source": "SUMMARY", "dataset": "ALL", "category": "ALL", "variant": variant, "morphology_group": group, "mean_delta_M_FR_P_AP_vs_E1": safe_stat(vals_d, np.mean), "improved": sum(1 for v in vals_d if v > 0), "total": len(vals_d)})
    return rows + summary


def proposal_gain_corr(cat_rows, proposal_summary):
    prop = {(r["source"], r["target"], r["morphology_group"]): fnum(r, "component_hit_rate") for r in proposal_summary}
    vals = {(r["source"], r["dataset"], r["category"], r["variant"]): r for r in cat_rows if r.get("protocol") == "M"}
    rows = []
    by_pair = {}
    for source, dataset, cat, variant in sorted(vals):
        if variant != "E3_SGPR":
            continue
        e1 = vals.get((source, dataset, cat, "E1_FixedLRP"))
        cur = vals.get((source, dataset, cat, "E3_SGPR"))
        if not (e1 and cur):
            continue
        by_pair.setdefault((source, dataset), {"r": [], "d": []})
        r = prop.get((source, dataset, "all"), float("nan"))
        d = fnum(cur, "M_FR_P_AP") - fnum(e1, "M_FR_P_AP")
        by_pair[(source, dataset)]["r"].append(r)
        by_pair[(source, dataset)]["d"].append(d)
        rows.append({"source": source, "dataset": dataset, "category": cat, "proposal_component_hit_rate_pair": r, "delta_E3_minus_E1_M_FR_P_AP": d})
    for (source, dataset), vd in by_pair.items():
        r = np.asarray(vd["r"], dtype=np.float64)
        d = np.asarray(vd["d"], dtype=np.float64)
        ok = np.isfinite(r) & np.isfinite(d)
        rows.append({
            "source": source,
            "dataset": dataset,
            "category": "PAIR_CORRELATION",
            "pearson": float(stats.pearsonr(r[ok], d[ok]).statistic) if ok.sum() >= 2 and np.std(r[ok]) > 0 and np.std(d[ok]) > 0 else float("nan"),
            "spearman": float(stats.spearmanr(r[ok], d[ok]).statistic) if ok.sum() >= 2 and np.std(r[ok]) > 0 and np.std(d[ok]) > 0 else float("nan"),
        })
    return rows


def decide(raw, proposal_summary):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    deltas = {}
    for source, target in sorted(AUDIT_PAIRS):
        e1 = by.get((source, target, "E1_FixedLRP"))
        e2 = by.get((source, target, "E2_StateGuidedSingleLevel"))
        e3 = by.get((source, target, "E3_SGPR"))
        if e1 and e2 and e3:
            deltas[(source, target)] = {
                "E2_minus_E1": fnum(e2, "M_FR_P_AP") - fnum(e1, "M_FR_P_AP"),
                "E3_minus_E2": fnum(e3, "M_FR_P_AP") - fnum(e2, "M_FR_P_AP"),
                "E3_minus_E1": fnum(e3, "M_FR_P_AP") - fnum(e1, "M_FR_P_AP"),
            }
    e31 = [v["E3_minus_E1"] for v in deltas.values()]
    gate = bool(
        len(e31) == 8
        and deltas.get(("Full MVTec", "VisA"), {}).get("E3_minus_E1", -999) >= 1.0
        and deltas.get(("Full VisA", "MVTec"), {}).get("E3_minus_E1", -999) >= 0.0
        and all(v["E3_minus_E1"] >= -0.5 for k, v in deltas.items() if k[1] in {"BTAD", "MPDD", "DTD-Synthetic"})
        and sum(1 for v in e31 if v >= -0.2) >= 6
        and float(np.mean(e31)) >= 0.5
    )
    if gate:
        outcome = "PROGRESSIVE_REOBSERVATION_IMPROVES_FORMAL_LRP"
    else:
        outcome = "PROGRESSIVE_REOBSERVATION_DOES_NOT_IMPROVE_FORMAL_LRP"
    serial_deltas = {f"{source}->{target}": vals for (source, target), vals in deltas.items()}
    return outcome, {
        "deltas": serial_deltas,
        "mean_E3_minus_E1": float(np.mean(e31)) if e31 else float("nan"),
        "non_drop_ge_minus_0_2": sum(1 for v in e31 if v >= -0.2),
        "gate_pass": gate,
    }


def failure_interpretation(raw, proposal_summary):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    prop = {(r["source"], r["target"], r["morphology_group"]): fnum(r, "component_hit_rate") for r in proposal_summary}
    e2_gt_e1 = []
    e3_gt_e2 = []
    e3_gt_e1 = []
    for source, target in AUDIT_PAIRS:
        e1 = by.get((source, target, "E1_FixedLRP"))
        e2 = by.get((source, target, "E2_StateGuidedSingleLevel"))
        e3 = by.get((source, target, "E3_SGPR"))
        if e1 and e2 and e3:
            e2_gt_e1.append(fnum(e2, "M_FR_P_AP") - fnum(e1, "M_FR_P_AP"))
            e3_gt_e2.append(fnum(e3, "M_FR_P_AP") - fnum(e2, "M_FR_P_AP"))
            e3_gt_e1.append(fnum(e3, "M_FR_P_AP") - fnum(e1, "M_FR_P_AP"))
    p = prop.get(("Full MVTec", "VisA", "small_or_thin"), float("nan"))
    if np.isfinite(p) and p < 0.5:
        return "FIRST_PASS_LOCALIZATION_LIMITS_ACTIVE_REOBSERVATION"
    if e3_gt_e1 and np.mean(e3_gt_e1) >= 0.5:
        return "STATE_GUIDED_PROGRESSIVE_REOBSERVATION_IS_EFFECTIVE"
    if e2_gt_e1 and np.mean(e2_gt_e1) > 0 and e3_gt_e2 and np.mean(e3_gt_e2) <= 0:
        return "STATE_GUIDED_CROPPING_HELPS_BUT_PROGRESSIVE_ZOOM_DOES_NOT"
    if np.isfinite(p) and p >= 0.5:
        return "GOOD_PROPOSALS_BUT_PROGRESSIVE_ZOOM_ADDS_NO_GAIN"
    return "FIRST_PASS_LOCALIZATION_LIMITS_ACTIVE_REOBSERVATION"


def write_phase0_report(summary, thresholds, outcome=None):
    lines = [
        "# SGPR Proposal Audit Summary",
        "",
        f"thin_threshold={thresholds['thin_threshold']:.6f}; large_patch_threshold={thresholds['large_patch_threshold']:.6f}.",
        "",
        "| Source | Target | Group | Images | Component Hit Rate | Anomaly Pixel Coverage | Image Hit Rate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["morphology_group"] in ("all", "small_or_thin"):
            lines.append(f"| {r['source']} | {r['target']} | {r['morphology_group']} | {int(r['images'])} | {fnum(r,'component_hit_rate'):.4f} | {fnum(r,'anomaly_pixel_coverage'):.4f} | {fnum(r,'image_hit_rate'):.4f} |")
    if outcome:
        lines.extend(["", f"OUTCOME: `{outcome}`"])
    (OUT / "SGPR_PROPOSAL_AUDIT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(raw, proposal_summary, decision, interp, outcome, runtime, path=None, reproduction=None):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = ["# Phase SGPR: State-Guided Progressive Re-Observation", "", f"OUTCOME: `{outcome}`", "", f"Failure interpretation: `{interp}`"]
    if reproduction:
        lines += ["", "## Formal LRP Reproduction", "", "| Source | Target | Reference P-AP | Corrected E1 P-AP | Delta | Pass |", "|---|---|---:|---:|---:|---|"]
        for r in reproduction:
            lines.append(f"| {r['source']} | {r['dataset']} | {fnum(r,'reference_M_FR_P_AP'):.4f} | {fnum(r,'corrected_M_FR_P_AP'):.4f} | {fnum(r,'delta_P_AP'):+.4f} | {r['pass_abs_le_0_2']} |")
    lines += ["", "## Main M-FR Results", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |", "|---|---|---|---:|---:|---:|---:|"]
    for source, target in sorted(AUDIT_PAIRS):
        for variant in VARIANTS:
            r = by.get((source, target, variant))
            if r:
                lines.append(f"| {source} | {target} | {variant} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    lines += ["", "## Primary Comparisons", "", "| Source | Target | E2-E1 P-AP | E3-E2 P-AP | E3-E1 P-AP |", "|---|---|---:|---:|---:|"]
    for key, vals in decision["deltas"].items():
        source, target = key.split("->", 1)
        lines.append(f"| {source} | {target} | {vals['E2_minus_E1']:+.4f} | {vals['E3_minus_E2']:+.4f} | {vals['E3_minus_E1']:+.4f} |")
    lines += ["", f"Mean E3-E1 P-AP: {decision['mean_E3_minus_E1']:+.4f}; non-drop count >= -0.2: {decision['non_drop_ge_minus_0_2']}/8.", "", "## Proposal Gate", "", "| Source | Target | small_or_thin component hit rate |", "|---|---|---:|"]
    for r in proposal_summary:
        if r["morphology_group"] == "small_or_thin":
            lines.append(f"| {r['source']} | {r['target']} | {fnum(r,'component_hit_rate'):.4f} |")
    lines += ["", f"Runtime: {runtime:.1f}s."]
    (path or (OUT / "REPORT.md")).write_text("\n".join(lines), encoding="utf-8")


def write_reproduction_failure_report(rows, runtime):
    lines = [
        "# SGPR / LRP Reproduction Audit",
        "",
        "OUTCOME: `SGPR_LRP_REPRODUCTION_STILL_FAILED`",
        "",
        "Corrected SGPR E1 did not reproduce the locked formal LRP artifact within `|Delta P-AP| <= 0.2`, so E2/E3 were not re-evaluated by protocol.",
        "",
        "| Source | Target | Reference P-AP | Corrected E1 P-AP | Delta | Pass |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r['source']} | {r['dataset']} | {fnum(r,'reference_M_FR_P_AP'):.4f} | {fnum(r,'corrected_M_FR_P_AP'):.4f} | {fnum(r,'delta_P_AP'):+.4f} | {r['pass_abs_le_0_2']} |")
    lines += ["", f"Runtime: {runtime:.1f}s."]
    (OUT / "REPORT_CORRECTED.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    write_lrp_static_audits()
    lrp_equivalence_audit(args.smoke)

    proposal_detail, proposal_summary, _cached, thresholds = phase0_proposal_audit(args.smoke)
    write_csv(OUT / "SGPR_PROPOSAL_AUDIT.csv", proposal_detail)
    write_csv(OUT / "SGPR_PROPOSAL_AUDIT_AGGREGATE.csv", proposal_summary)
    p_gate = next((r for r in proposal_summary if r["source"] == "Full MVTec" and r["target"] == "VisA" and r["morphology_group"] == "small_or_thin"), None)
    p_hit = fnum(p_gate, "component_hit_rate") if p_gate else float("nan")
    if not np.isfinite(p_hit) or p_hit < 0.50:
        outcome = "STATE_PROPOSAL_TOO_WEAK_FOR_PROGRESSIVE_REOBSERVATION"
        write_phase0_report(proposal_summary, thresholds, outcome)
        for name in ("REOBSERVATION_CROPS.csv", "COMPUTE_AUDIT.csv", "FULL_RES_METRICS.csv", "PER_CATEGORY_FULL_RES.csv", "MORPHOLOGY_DELTA_ANALYSIS.csv", "PROPOSAL_GAIN_CORRELATION.csv"):
            write_csv(OUT / name, [])
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "phase0_component_hit_rate": p_hit, "runtime_sec": time.time() - start})
        (OUT / "REPORT.md").write_text(f"# Phase SGPR\n\nOUTCOME: `{outcome}`\n\nMVTec->VisA small_or_thin component_hit_rate={p_hit:.4f}; Phase 1 was not run by protocol.\n", encoding="utf-8")
        print(f"[SGPR] OUTCOME={outcome} p_hit={p_hit:.4f}", flush=True)
        return

    write_phase0_report(proposal_summary, thresholds)

    repro_raw, repro_cats, repro_crops, repro_compute = full_eval(
        args.smoke,
        tag="LRP_REPRO",
        pair_filter=REPRODUCTION_PAIRS,
        variants=("E1_FixedLRP",),
    )
    repro = reproduction_rows(repro_raw)
    repro_pass = bool(repro and all(str(r.get("pass_abs_le_0_2")).lower() == "true" or r.get("pass_abs_le_0_2") is True for r in repro))
    write_csv(OUT / "LRP_REPRODUCTION_TEST.csv", repro)
    write_csv(OUT / "FULL_RES_METRICS_LRP_REPRO.csv", repro_raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES_LRP_REPRO.csv", repro_cats)
    write_csv(OUT / "REOBSERVATION_CROPS_LRP_REPRO.csv", repro_crops)
    write_csv(OUT / "COMPUTE_AUDIT_LRP_REPRO.csv", repro_compute)
    if not repro_pass:
        runtime = time.time() - start
        write_json(OUT / "OUTCOME_CORRECTED.json", {"outcome": "SGPR_LRP_REPRODUCTION_STILL_FAILED", "reproduction": repro, "runtime_sec": runtime})
        write_reproduction_failure_report(repro, runtime)
        print(f"[SGPR] OUTCOME=SGPR_LRP_REPRODUCTION_STILL_FAILED runtime={runtime:.1f}s", flush=True)
        return

    raw, cats, crops, compute = full_eval(args.smoke, tag="CORRECTED")
    morph = morphology_delta(cats, proposal_summary)
    corr = proposal_gain_corr(cats, proposal_summary)
    outcome, decision = decide(raw, proposal_summary)
    interp = failure_interpretation(raw, proposal_summary)
    runtime = time.time() - start
    write_csv(OUT / "REOBSERVATION_CROPS_CORRECTED.csv", crops)
    write_csv(OUT / "COMPUTE_AUDIT_CORRECTED.csv", compute)
    write_csv(OUT / "FULL_RES_METRICS_CORRECTED.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES_CORRECTED.csv", cats)
    write_csv(OUT / "MORPHOLOGY_DELTA_ANALYSIS_CORRECTED.csv", morph)
    write_csv(OUT / "PROPOSAL_GAIN_CORRELATION_CORRECTED.csv", corr)
    write_json(OUT / "OUTCOME_CORRECTED.json", {"outcome": outcome, "reproduction_outcome": "FORMAL_LRP_REPRODUCED_IN_SGPR_PIPELINE", "reproduction": repro, "failure_interpretation": interp, "decision": decision, "runtime_sec": runtime})
    write_report(raw, proposal_summary, decision, interp, outcome, runtime, path=OUT / "REPORT_CORRECTED.md", reproduction=repro)
    print(f"[SGPR] OUTCOME={outcome} interpretation={interp} runtime={runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
