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
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAM2_ROOT = ROOT / "_external" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from experiments.phase_mva_mrad_same_protocol_visa_factorization import load_metas_direct, pair_slug
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    CHECKPOINT,
    IMAGE_SIZE,
    load_components,
    load_mvtec_source,
    load_visa_source,
    score_feature_grid,
    summarize_category,
    torch_resize,
)
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from utils.transforms import get_transform

OUT = ROOT / "results" / "phase_isro_instance_structure_reobservation" / "formal"
REGION_CACHE = OUT / "REGION_CACHE"
MAP_CACHE = OUT / "MAP_CACHE"

E0 = "E0_Core_NTR_FixedLRP"
E1 = "E1_Instance_ReObservation"
E2 = "E2_Instance_Part_ReObservation"
VARIANTS = (E0, E1, E2)
PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "MPDD"),
)
FOCUS_CATEGORIES = {"macaroni2", "pcb1", "pcb2", "pcb3", "cashew", "fryum", "pipe_fryum"}

SAM2_CKPT = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_POINTS_PER_SIDE = 16
SAM_POINTS_PER_BATCH = 32
MAX_INSTANCE_REGIONS = 8
MAX_PART_REGIONS = 8
MAX_TOTAL_REGIONS = 16
MIN_REGION_AREA_RATIO = 0.002
MAX_REGION_AREA_RATIO = 0.85
INSTANCE_AREA_RATIO = 0.06
PART_MIN_AREA_RATIO = 0.005
PART_MAX_AREA_RATIO = 0.18
CROP_PADDING_RATIO = 0.10
COVERAGE_RECALL_GATE = 0.50
COVERAGE_COMPONENT_GATE = 0.40


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


def fnum(row, key):
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


def ensure_sam2_ready():
    missing = []
    if not SAM2_ROOT.exists():
        missing.append(str(SAM2_ROOT))
    if not SAM2_CKPT.exists() or SAM2_CKPT.stat().st_size < 1_000_000:
        missing.append(str(SAM2_CKPT))
    if missing:
        raise FileNotFoundError("Missing official SAM2 dependency: " + "; ".join(missing))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    return build_sam2, SAM2AutomaticMaskGenerator


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


def e0_maps(source, target, metas):
    path = ROOT / "results" / "phase_fova_field_of_view_amplification_audit" / "formal" / "MAP_CACHE" / f"{pair_slug(source, target)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing locked Core/LRP map cache: {path}")
    data = np.load(path, allow_pickle=True)
    q = np.asarray(data["q_lrp"], dtype=np.float32)
    return q[: len(metas)]


def region_cache_path(source, target, image_index):
    return REGION_CACHE / f"{pair_slug(source, target)}__{image_index:06d}.npz"


def load_regions(source, target, image_index):
    p = region_cache_path(source, target, image_index)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    return list(data["regions"])


def save_regions(source, target, image_index, regions):
    p = region_cache_path(source, target, image_index)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, regions=np.asarray(regions, dtype=object))


def bbox_xyxy_from_xywh(b):
    x, y, w, h = [float(v) for v in b]
    return [x, y, x + w, y + h]


def box_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def containment_ratio(inner, outer):
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = box_area([x1, y1, x2, y2])
    return inter / max(box_area(inner), 1e-12)


def filter_regions(anns, h, w):
    image_area = float(h * w)
    rows = []
    for ann in anns:
        mask = np.asarray(ann["segmentation"], dtype=bool)
        area = float(ann.get("area", mask.sum()))
        ar = area / max(image_area, 1.0)
        if ar < MIN_REGION_AREA_RATIO or ar > MAX_REGION_AREA_RATIO:
            continue
        box = bbox_xyxy_from_xywh(ann["bbox"])
        rows.append(
            {
                "segmentation": mask,
                "bbox": np.asarray(box, dtype=np.float32),
                "area": area,
                "area_ratio": ar,
                "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "stability_score": float(ann.get("stability_score", 0.0)),
                "region_type": "candidate",
            }
        )
    rows.sort(key=lambda r: (r["predicted_iou"] + r["stability_score"], r["area"]), reverse=True)
    large = []
    for r in rows:
        if r["area_ratio"] >= INSTANCE_AREA_RATIO:
            r = dict(r)
            r["region_type"] = "instance"
            large.append(r)
        if len(large) >= MAX_INSTANCE_REGIONS:
            break
    part = []
    large_boxes = [r["bbox"] for r in large]
    for r in rows:
        nested = any(containment_ratio(r["bbox"], b) >= 0.80 and r["area_ratio"] < (box_area(b) / max(image_area, 1.0)) * 0.75 for b in large_boxes)
        medium = PART_MIN_AREA_RATIO <= r["area_ratio"] <= PART_MAX_AREA_RATIO
        if medium or nested:
            if any(np.allclose(r["bbox"], x["bbox"]) for x in large):
                continue
            rr = dict(r)
            rr["region_type"] = "part"
            part.append(rr)
        if len(part) >= MAX_PART_REGIONS:
            break
    selected = (large + part)[:MAX_TOTAL_REGIONS]
    return selected


@torch.no_grad()
def generate_regions(generator, source, target, metas, smoke=False, force=False):
    rows = []
    n = min(24, len(metas)) if smoke else len(metas)
    for i, meta in enumerate(metas[:n]):
        if i % 25 == 0:
            print(f"[ISRO:SAM2] {source}->{target} {i}/{n}", flush=True)
        cached = None if force else load_regions(source, target, i)
        img = Image.open(meta["img_path"]).convert("RGB")
        w, h = img.size
        if cached is None:
            arr = np.asarray(img, dtype=np.uint8)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                anns = generator.generate(arr.copy())
            regions = filter_regions(anns, h, w)
            save_regions(source, target, i, regions)
        else:
            regions = cached
        rows.append(coverage_row(source, target, i, meta, regions, h, w))
    return rows


def coverage_row(source, target, idx, meta, regions, h, w):
    gt = np.asarray(meta["mask"], dtype=bool)
    union_all = np.zeros((h, w), dtype=bool)
    union_inst = np.zeros((h, w), dtype=bool)
    union_part = np.zeros((h, w), dtype=bool)
    areas = []
    nested = 0
    boxes = [np.asarray(r["bbox"], dtype=np.float32) for r in regions]
    for ri, r in enumerate(regions):
        m = np.asarray(r["segmentation"], dtype=bool)
        if m.shape != (h, w):
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
        union_all |= m
        if r["region_type"] == "instance":
            union_inst |= m
        elif r["region_type"] == "part":
            union_part |= m
        areas.append(float(m.mean()))
        nested += int(any(j != ri and containment_ratio(boxes[ri], boxes[j]) >= 0.80 for j in range(len(boxes))))
    if gt.shape != (h, w):
        gt = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
    labels, ncomp = ndimage.label(gt, structure=np.ones((3, 3), dtype=np.uint8))
    comp_hits = 0
    small_hits = 0
    small_total = 0
    for rid in range(1, ncomp + 1):
        comp = labels == rid
        hit = bool(np.any(union_all & comp))
        comp_hits += int(hit)
        if comp.sum() / max(gt.size, 1) <= 0.002 or boundary_ratio(comp) >= 0.45:
            small_total += 1
            small_hits += int(hit)
    anomaly_pixels = int(gt.sum())
    proposal_recall = float((union_all & gt).sum() / max(anomaly_pixels, 1)) if anomaly_pixels else float("nan")
    inst_recall = float((union_inst & gt).sum() / max(anomaly_pixels, 1)) if anomaly_pixels else float("nan")
    part_recall = float((union_part & gt).sum() / max(anomaly_pixels, 1)) if anomaly_pixels else float("nan")
    return {
        "source": source,
        "dataset": target,
        "image_index": idx,
        "category": meta["category"],
        "label": int(meta["label"]),
        "mask_count": int(len(regions)),
        "instance_mask_count": int(sum(r["region_type"] == "instance" for r in regions)),
        "part_mask_count": int(sum(r["region_type"] == "part" for r in regions)),
        "nested_mask_count": int(nested),
        "mask_area_mean": float(np.mean(areas)) if areas else 0.0,
        "mask_area_median": float(np.median(areas)) if areas else 0.0,
        "mask_area_p95": float(np.percentile(areas, 95)) if areas else 0.0,
        "overlap_ratio": float(sum(np.asarray(r["segmentation"], dtype=bool).sum() for r in regions) / max((union_all.sum() if union_all.any() else 1), 1)),
        "proposal_recall": proposal_recall,
        "instance_proposal_recall": inst_recall,
        "part_proposal_recall": part_recall,
        "component_recall": float(comp_hits / max(ncomp, 1)) if anomaly_pixels else float("nan"),
        "small_thin_component_recall": float(small_hits / max(small_total, 1)) if small_total else float("nan"),
        "anomaly_area_ratio": float(gt.mean()),
        "thinness": boundary_ratio(gt) if anomaly_pixels else 0.0,
    }


def boundary_ratio(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return float((mask & ~eroded).sum() / max(mask.sum(), 1))


def coverage_gate(rows):
    anom = [r for r in rows if int(r["label"]) == 1 and np.isfinite(fnum(r, "proposal_recall"))]
    if not anom:
        return True, {"mean_proposal_recall": float("nan"), "mean_component_recall": float("nan"), "coverage_gate_note": "no anomaly images in sampled rows; gate skipped"}
    mean_prop = float(np.mean([fnum(r, "proposal_recall") for r in anom]))
    mean_comp = float(np.mean([fnum(r, "component_recall") for r in anom]))
    return bool(mean_prop >= COVERAGE_RECALL_GATE and mean_comp >= COVERAGE_COMPONENT_GATE), {
        "mean_proposal_recall": mean_prop,
        "mean_component_recall": mean_comp,
    }


def padded_box(box, w, h):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    pad = CROP_PADDING_RATIO * max(bw, bh)
    return (
        int(max(0, math.floor(x1 - pad))),
        int(max(0, math.floor(y1 - pad))),
        int(min(w, math.ceil(x2 + pad))),
        int(min(h, math.ceil(y2 + pad))),
    )


@torch.no_grad()
def score_crop(model_info, clip_model, preprocess, crop, device):
    batch = preprocess(crop).unsqueeze(0).to(device)
    from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs

    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patch = normalized_patch_feature(out).float().cpu().numpy()[0].astype(np.float32)
    del batch, out
    return score_feature_grid(model_info, patch, device)


def grid_box(box, image_size, grid_shape):
    x1, y1, x2, y2 = box
    w, h = image_size
    gh, gw = grid_shape
    gx1 = int(np.floor(x1 / max(w, 1) * gw))
    gx2 = int(np.ceil(x2 / max(w, 1) * gw))
    gy1 = int(np.floor(y1 / max(h, 1) * gh))
    gy2 = int(np.ceil(y2 / max(h, 1) * gh))
    gx1, gy1 = max(0, gx1), max(0, gy1)
    gx2, gy2 = min(gw, max(gx1 + 1, gx2)), min(gh, max(gy1 + 1, gy2))
    return gx1, gy1, gx2, gy2


def map_cache_path(source, target):
    return MAP_CACHE / f"{pair_slug(source, target)}.npz"


def load_pair_maps(source, target):
    p = map_cache_path(source, target)
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return {k: np.asarray(d[k], dtype=np.float32) for k in d.files}
    return None


def save_pair_maps(source, target, maps):
    p = map_cache_path(source, target)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{k: np.asarray(v, dtype=np.float16) for k, v in maps.items()})


@torch.no_grad()
def build_isro_maps(source, target, metas, model_info, clip_model, preprocess, device, smoke=False, force=False):
    cached = None if force or smoke else load_pair_maps(source, target)
    if cached is not None and all(k in cached for k in VARIANTS):
        return cached
    n = min(24, len(metas)) if smoke else len(metas)
    q0 = e0_maps(source, target, metas)[:n]
    maps = {E0: [], E1: [], E2: []}
    for i, meta in enumerate(metas[:n]):
        if i % 10 == 0:
            print(f"[ISRO:region] {source}->{target} {i}/{n}", flush=True)
        img = Image.open(meta["img_path"]).convert("RGB")
        w, h = img.size
        regions = load_regions(source, target, i)
        if regions is None:
            raise RuntimeError(f"missing SAM2 regions for {source}->{target} image {i}")
        global_map = np.asarray(q0[i], dtype=np.float32)
        inst_sum = np.zeros_like(global_map, dtype=np.float32)
        inst_cnt = np.zeros_like(global_map, dtype=np.float32)
        all_sum = np.zeros_like(global_map, dtype=np.float32)
        all_cnt = np.zeros_like(global_map, dtype=np.float32)
        for r in regions:
            typ = str(r["region_type"])
            if typ not in ("instance", "part"):
                continue
            box = padded_box(r["bbox"], w, h)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop_score = score_crop(model_info, clip_model, preprocess, img.crop(box), device)
            gx1, gy1, gx2, gy2 = grid_box(box, (w, h), global_map.shape)
            resized = torch_resize(crop_score, (gy2 - gy1, gx2 - gx1), align_corners=True)
            all_sum[gy1:gy2, gx1:gx2] += resized
            all_cnt[gy1:gy2, gx1:gx2] += 1.0
            if typ == "instance":
                inst_sum[gy1:gy2, gx1:gx2] += resized
                inst_cnt[gy1:gy2, gx1:gx2] += 1.0
        inst_region = np.divide(inst_sum, inst_cnt, out=np.zeros_like(inst_sum), where=inst_cnt > 0)
        all_region = np.divide(all_sum, all_cnt, out=np.zeros_like(all_sum), where=all_cnt > 0)
        e1 = global_map.copy()
        m1 = inst_cnt > 0
        e1[m1] = 0.5 * global_map[m1] + 0.5 * inst_region[m1]
        e2 = global_map.copy()
        m2 = all_cnt > 0
        e2[m2] = 0.5 * global_map[m2] + 0.5 * all_region[m2]
        maps[E0].append(global_map)
        maps[E1].append(e1)
        maps[E2].append(e2)
    if not smoke:
        save_pair_maps(source, target, maps)
    return {k: np.asarray(v, dtype=np.float32) for k, v in maps.items()}


def evaluate_maps(source, target, maps, metas):
    raw_rows, cat_rows = [], []
    for variant in VARIANTS:
        print(f"[ISRO] metrics {source}->{target} {variant}", flush=True)
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
            row = {"source": source, "dataset": target, "category": cat, "variant": variant, "protocol": "M_FR", "aggregation": "CATEGORY", "score_resolution": f"{vals['scores'][0].shape[0]}x{vals['scores'][0].shape[1]}"}
            for k, v in met.items():
                row[f"M_FR_{k.replace('-', '_')}"] = v
            cat_rows.append(row)
            fr_cat.append(row)
        raw = {"source": source, "dataset": target, "variant": variant, "aggregation": "CATEGORY_MACRO", "score_resolution": f"{maps[variant][0].shape[0]}x{maps[variant][0].shape[1]}"}
        for met in ("I_AUROC", "I_AP", "I_F1max", "P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
            vals = np.asarray([fnum(r, f"M_FR_{met}") for r in fr_cat], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            raw[f"M_FR_{met}"] = float(vals.mean()) if vals.size else float("nan")
        raw_rows.append(raw)
    return raw_rows, cat_rows


def diagnostic_rows(source, target, maps, metas, coverage):
    cov_by_idx = {int(r["image_index"]): r for r in coverage if r["source"] == source and r["dataset"] == target}
    rows = []
    for i, meta in enumerate(metas[: len(maps[E0])]):
        mask = np.asarray(meta["mask"], dtype=bool)
        q0 = torch_resize(maps[E0][i], mask.shape, align_corners=True)
        for variant in (E1, E2):
            q = torch_resize(maps[variant][i], mask.shape, align_corners=True)
            delta = q - q0
            thr0 = np.percentile(q0.reshape(-1), 95)
            thr = np.percentile(q.reshape(-1), 95)
            pred0 = q0 >= thr0
            pred = q >= thr
            fn0 = mask & (~pred0)
            tp0 = mask & pred0
            fp0 = (~mask) & pred0
            rows.append(
                {
                    "source": source,
                    "dataset": target,
                    "category": meta["category"],
                    "image_index": i,
                    "variant": variant,
                    "mean_delta_anomaly": float(delta[mask].mean()) if mask.any() else float("nan"),
                    "mean_delta_normal": float(delta[~mask].mean()) if (~mask).any() else float("nan"),
                    "E0_FN_rescue_ratio": float((fn0 & pred).sum() / max(fn0.sum(), 1)),
                    "E0_TP_preserved_ratio": float((tp0 & pred).sum() / max(tp0.sum(), 1)),
                    "E0_FP_kept_ratio": float((fp0 & pred).sum() / max(fp0.sum(), 1)),
                    "proposal_recall": fnum(cov_by_idx.get(i, {}), "proposal_recall"),
                    "instance_proposal_recall": fnum(cov_by_idx.get(i, {}), "instance_proposal_recall"),
                    "part_proposal_recall": fnum(cov_by_idx.get(i, {}), "part_proposal_recall"),
                    "small": bool(mask.mean() <= 0.01),
                    "thin": bool(boundary_ratio(mask) >= 0.35) if mask.any() else False,
                }
            )
    return rows


def delta_summary(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in PAIRS:
        base = by.get((source, target, E0))
        if not base:
            continue
        for variant in (E1, E2):
            cur = by.get((source, target, variant))
            if not cur:
                continue
            rows.append(
                {
                    "source": source,
                    "dataset": target,
                    "variant": variant,
                    "Delta_M_FR_P_AP": fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"),
                    "Delta_M_FR_P_AUROC": fnum(cur, "M_FR_P_AUROC") - fnum(base, "M_FR_P_AUROC"),
                    "Delta_M_FR_P_F1max": fnum(cur, "M_FR_P_F1max") - fnum(base, "M_FR_P_F1max"),
                    "Delta_M_FR_PRO": fnum(cur, "M_FR_PRO") - fnum(base, "M_FR_PRO"),
                }
            )
    for variant in (E1, E2):
        vals = np.asarray([fnum(r, "Delta_M_FR_P_AP") for r in rows if r["variant"] == variant], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        rows.append({"source": "SUMMARY", "dataset": "pilot", "variant": variant, "mean_delta_pap": float(vals.mean()) if vals.size else float("nan"), "median_delta_pap": float(np.median(vals)) if vals.size else float("nan"), "improved": int(np.sum(vals > 0)), "total": int(vals.size)})
    return rows


def decide(raw, coverage):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    base = by.get(("Full MVTec", "VisA", E0))
    e2v = by.get(("Full MVTec", "VisA", E2))
    btad = by.get(("Full MVTec", "BTAD", E2))
    mpdd = by.get(("Full VisA", "MPDD", E2))
    if base is None or e2v is None:
        return "INSTANCE_STRUCTURE_REOBSERVATION_HAS_NO_COMPETITIVE_VALUE"
    visa_delta = fnum(e2v, "M_FR_P_AP") - fnum(base, "M_FR_P_AP")
    btad_delta = fnum(btad, "M_FR_P_AP") - fnum(by.get(("Full MVTec", "BTAD", E0), {}), "M_FR_P_AP") if btad else float("nan")
    mpdd_delta = fnum(mpdd, "M_FR_P_AP") - fnum(by.get(("Full VisA", "MPDD", E0), {}), "M_FR_P_AP") if mpdd else float("nan")
    vals = np.asarray([visa_delta, btad_delta, mpdd_delta], dtype=np.float64)
    mean_delta = float(np.nanmean(vals))
    focus = [r for r in coverage if r["source"] == "Full MVTec" and r["dataset"] == "VisA" and int(r["label"]) == 1]
    cov_ok = bool(focus and np.nanmean([fnum(r, "proposal_recall") for r in focus]) >= COVERAGE_RECALL_GATE)
    if visa_delta >= 2.0 and np.nanmin([btad_delta, mpdd_delta]) >= -0.5 and mean_delta > 0 and cov_ok:
        return "INSTANCE_STRUCTURE_ALIGNED_REOBSERVATION_HAS_STRONG_VALUE"
    if visa_delta >= 1.0 and np.nanmin([btad_delta, mpdd_delta]) >= -2.0 and cov_ok:
        return "STRUCTURE_ALIGNED_OBSERVATION_HAS_PARTIAL_VALUE"
    return "INSTANCE_STRUCTURE_REOBSERVATION_HAS_NO_COMPETITIVE_VALUE"


def write_report(raw, coverage, visa_cat, outcome, runtime):
    lines = [
        "# Phase ISRO: Instance / Structure-Aware Re-Observation Pilot",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "SAM2 and CLIP/Two-State were frozen. Region proposals were selected only from SAM2 mask quality/area, not from GT or anomaly scores.",
        "",
        "## Fixed Configuration",
        "",
        f"- SAM2: official `sam2.1_hiera_tiny`, config `{SAM2_CFG}`, checkpoint `{SAM2_CKPT.name}`.",
        f"- Automatic mask generation: points_per_side={SAM_POINTS_PER_SIDE}, points_per_batch={SAM_POINTS_PER_BATCH}.",
        f"- Region cap: instance={MAX_INSTANCE_REGIONS}, part={MAX_PART_REGIONS}, total={MAX_TOTAL_REGIONS}.",
        f"- Fusion: covered pixels use `(q_global + q_region)/2`; uncovered pixels keep `q_global`.",
        "",
        "## Pilot Metrics",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    cov = [r for r in coverage if r["source"] == "Full MVTec" and r["dataset"] == "VisA" and int(r["label"]) == 1]
    lines.extend(["", "## Proposal Coverage", ""])
    if cov:
        lines.append(f"- VisA anomaly mean proposal recall: `{np.nanmean([fnum(r,'proposal_recall') for r in cov]):.4f}`.")
        lines.append(f"- VisA anomaly mean component recall: `{np.nanmean([fnum(r,'component_recall') for r in cov]):.4f}`.")
        lines.append(f"- VisA small/thin component recall: `{np.nanmean([fnum(r,'small_thin_component_recall') for r in cov]):.4f}`.")
    lines.extend(["", "## VisA Focus Categories", "", "| Category | E0 P-AP | E1 Delta | E2 Delta | Proposal Recall | Thinness |", "|---|---:|---:|---:|---:|---:|"])
    for r in visa_cat:
        if r["category"] in FOCUS_CATEGORIES:
            lines.append(f"| {r['category']} | {fnum(r,'E0_P_AP'):.4f} | {fnum(r,'E1_delta_P_AP'):.4f} | {fnum(r,'E2_delta_P_AP'):.4f} | {fnum(r,'proposal_recall_mean'):.4f} | {fnum(r,'thinness_mean'):.4f} |")
    lines.append("")
    lines.append(f"Runtime: {runtime:.1f}s.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def visa_category_region_audit(raw_cat, coverage):
    by = {(r["category"], r["variant"]): r for r in raw_cat if r["source"] == "Full MVTec" and r["dataset"] == "VisA"}
    cats = sorted({r["category"] for r in raw_cat if r["source"] == "Full MVTec" and r["dataset"] == "VisA"})
    rows = []
    for cat in cats:
        b, e1, e2 = by.get((cat, E0), {}), by.get((cat, E1), {}), by.get((cat, E2), {})
        cov = [r for r in coverage if r["source"] == "Full MVTec" and r["dataset"] == "VisA" and r["category"] == cat and int(r["label"]) == 1]
        rows.append(
            {
                "source": "Full MVTec",
                "dataset": "VisA",
                "category": cat,
                "E0_P_AP": fnum(b, "M_FR_P_AP"),
                "E1_P_AP": fnum(e1, "M_FR_P_AP"),
                "E2_P_AP": fnum(e2, "M_FR_P_AP"),
                "E1_delta_P_AP": fnum(e1, "M_FR_P_AP") - fnum(b, "M_FR_P_AP"),
                "E2_delta_P_AP": fnum(e2, "M_FR_P_AP") - fnum(b, "M_FR_P_AP"),
                "proposal_recall_mean": float(np.nanmean([fnum(r, "proposal_recall") for r in cov])) if cov else float("nan"),
                "component_recall_mean": float(np.nanmean([fnum(r, "component_recall") for r in cov])) if cov else float("nan"),
                "thinness_mean": float(np.nanmean([fnum(r, "thinness") for r in cov])) if cov else float("nan"),
                "mask_count_mean": float(np.nanmean([fnum(r, "mask_count") for r in cov])) if cov else float("nan"),
            }
        )
    return rows


def correlation_rows(visa_cat):
    rows = []
    if not visa_cat:
        return rows
    metrics = ["E1_delta_P_AP", "E2_delta_P_AP"]
    factors = ["proposal_recall_mean", "component_recall_mean", "thinness_mean", "mask_count_mean"]
    for metric in metrics:
        y = np.asarray([fnum(r, metric) for r in visa_cat], dtype=np.float64)
        for factor in factors:
            x = np.asarray([fnum(r, factor) for r in visa_cat], dtype=np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
                pr = sp = float("nan")
            else:
                pr = float(pearsonr(x[ok], y[ok]).statistic)
                sp = float(spearmanr(x[ok], y[ok]).statistic)
            rows.append({"dataset": "VisA", "metric": metric, "factor": factor, "pearson": pr, "spearman": sp, "n": int(ok.sum())})
    return rows


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    REGION_CACHE.mkdir(parents=True, exist_ok=True)
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"

    generator = load_sam_generator(device)
    coverage_rows = []
    metas_by_pair = {}
    selected_pairs = PAIRS if not args.smoke else PAIRS[:1]
    for source, target in selected_pairs:
        metas = load_metas_direct(source, target, device, smoke=args.smoke)
        metas_by_pair[(source, target)] = metas
        coverage_rows.extend(generate_regions(generator, source, target, metas, smoke=args.smoke, force=args.force_regions))
        if args.coverage_only and (source, target) == ("Full MVTec", "VisA"):
            break
    write_csv(OUT / "REGION_COVERAGE.csv", coverage_rows)

    gate_rows = [r for r in coverage_rows if r["source"] == "Full MVTec" and r["dataset"] == "VisA"]
    gate_ok, gate_info = coverage_gate(gate_rows)
    if not gate_ok:
        outcome = "SAM2_REGIONS_DO_NOT_COVER_VISA_FAILURE_MODE"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, **gate_info, "runtime_sec": time.time() - start})
        write_report([], coverage_rows, [], outcome, time.time() - start)
        print(f"[ISRO] stop at coverage gate: {outcome}", flush=True)
        return
    if args.coverage_only:
        outcome = "COVERAGE_GATE_PASSED"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, **gate_info, "runtime_sec": time.time() - start})
        write_report([], coverage_rows, [], outcome, time.time() - start)
        print(f"[ISRO] coverage-only complete", flush=True)
        return

    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    raw_rows, cat_rows, diag_rows = [], [], []
    model_info_cache = {}
    for source, target in selected_pairs:
        model_info = model_info_cache.get(source)
        if model_info is None:
            model_info = load_mvtec_source() if source == "Full MVTec" else load_visa_source(device)
            model_info_cache[source] = model_info
        metas = metas_by_pair[(source, target)]
        maps = build_isro_maps(source, target, metas, model_info, clip_model, preprocess, device, smoke=args.smoke, force=args.force_maps)
        rr, cc = evaluate_maps(source, target, maps, metas)
        raw_rows.extend(rr)
        cat_rows.extend(cc)
        diag_rows.extend(diagnostic_rows(source, target, maps, metas, coverage_rows))
        write_csv(OUT / "RAW_METRICS.csv", raw_rows)
        write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", cat_rows)
        write_csv(OUT / "REGION_DIAGNOSTICS.csv", diag_rows)

    visa_cat = visa_category_region_audit(cat_rows, coverage_rows)
    corr = correlation_rows(visa_cat)
    deltas = delta_summary(raw_rows)
    outcome = decide(raw_rows, coverage_rows)
    write_csv(OUT / "VISA_CATEGORY_REGION_AUDIT.csv", visa_cat)
    write_csv(OUT / "REGION_GAIN_CORRELATION.csv", corr)
    write_csv(OUT / "DELTA_SUMMARY.csv", deltas)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, **gate_info, "runtime_sec": time.time() - start})
    write_report(raw_rows, coverage_rows, visa_cat, outcome, time.time() - start)
    print(f"[ISRO] complete: {outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--coverage-only", action="store_true")
    p.add_argument("--force-regions", action="store_true")
    p.add_argument("--force-maps", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
