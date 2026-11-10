import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    PROTOCOLS,
    METRICS,
    evaluate_maps,
    fast_aupro_vectorized,
    fnum,
    item_at,
    item_meta,
    mask_np,
    summarize_category,
    torch_resize,
)
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE
from experiments.standard_zsad_metrics import f1max, safe_ap, safe_auc


OUT = Path("results/phase_core_specialist_complementarity_oracle/formal")
MAP_CACHE = OUT / "MAP_CACHE"
E0_CACHE = ROOT / "results" / "phase_fova_field_of_view_amplification_audit" / "formal" / "MAP_CACHE"
VISUALAD_METRICS = ROOT / "results" / "phase_visualad_exact_protocol_audit" / "formal" / "RAW_METRICS.csv"
FBCLIP_METRICS = ROOT / "results" / "phase_fbclip_pretrained_headroom_audit" / "formal" / "RAW_METRICS.csv"

PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "MVTec"),
)
VAR_E0 = "E0_Core_NTR_FixedLRP"
VAR_VIS = "VisualAD_E1c_Full"
VAR_FB = "FBCLIP_Official_Pretrained"
SPECIALISTS = (VAR_VIS, VAR_FB)
ALL_VARIANTS = (VAR_E0, VAR_VIS, VAR_FB)


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


def load_metric_rows():
    rows = []
    for r in read_csv(ROOT / "results" / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "FULL_RES_METRICS.csv"):
        if r.get("variant") == "E2_GlobalLocalZoomPyramid" and (r.get("source"), r.get("dataset")) in set(PAIRS):
            rr = dict(r)
            rr["variant"] = VAR_E0
            rows.append(rr)
    if VISUALAD_METRICS.exists():
        for r in read_csv(VISUALAD_METRICS):
            if r.get("variant") == "E1c_FullVisualAD" and (r.get("source"), r.get("dataset")) in set(PAIRS):
                rr = dict(r)
                rr["variant"] = VAR_VIS
                rows.append(rr)
    if FBCLIP_METRICS.exists():
        for r in read_csv(FBCLIP_METRICS):
            if r.get("variant") == "E1_Official_FBCLIP_Pretrained" and (r.get("source"), r.get("dataset")) in set(PAIRS):
                rr = dict(r)
                rr["variant"] = VAR_FB
                rows.append(rr)
    return rows


def pair_slug(source, target):
    return f"{source.replace(' ', '_')}__{target}"


def specialist_cache_path(source, target, variant):
    return MAP_CACHE / f"{pair_slug(source, target)}__{variant}.npz"


def load_metas(source, target, device, smoke=False):
    cells, _audits = protocol_cells(source, device, smoke)
    for _cell_id, t, dataset, refs, _patch, kind in cells:
        if t != target:
            continue
        n = min(24, len(refs)) if smoke else len(refs)
        metas = []
        for pos, ref in enumerate(refs[:n]):
            item = item_at(dataset, kind, ref, pos)
            cat, label, path = item_meta(item, kind, ref, target)
            metas.append({"category": cat, "label": label, "img_path": path, "mask": mask_np(item)})
        return metas
    raise KeyError(f"missing protocol cell: {source}->{target}")


def load_e0_maps(source, target, metas):
    path = E0_CACHE / f"{pair_slug(source, target)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing E0 FOVA map cache: {path}")
    data = np.load(path, allow_pickle=True)
    q = np.asarray(data["q_lrp"], dtype=np.float32)
    if q.shape[0] >= len(metas):
        q = q[:len(metas)]
    if q.shape[0] != len(metas):
        raise RuntimeError(f"E0 map/meta length mismatch for {source}->{target}: {q.shape[0]} vs {len(metas)}")
    return q


def save_maps(path, maps):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.asarray(m, dtype=np.float16) for m in maps], axis=0)
    np.savez(path, maps=arr)


def load_specialist_maps(source, target, variant, device, smoke=False, batch_size=2):
    path = specialist_cache_path(source, target, variant)
    if path.exists() and not smoke:
        return np.asarray(np.load(path, allow_pickle=True)["maps"], dtype=np.float16)
    if variant == VAR_VIS:
        from experiments import phase_visualad_exact_protocol_audit as vad
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]
        official_root = str(vad.OFFICIAL_ROOT)
        if official_root in sys.path:
            sys.path.remove(official_root)
        sys.path.insert(0, official_root)
        official = vad.ensure_official_imports()
        model, layer_transforms, cross_attn, _checkpoint = vad.load_visualad_bundle(source, "E1c_FullVisualAD", official, device)
        preprocess = vad.official_transform(official)
        _cell_id, _t, dataset, refs, kind = vad.find_cell(source, target, device, smoke)
        n = min(8, len(refs)) if smoke else len(refs)
        maps = []
        pending = []
        for pos, ref in enumerate(refs[:n]):
            if pos % 50 == 0:
                print(f"[CSCA:VisualAD] {source} -> {target} image {pos}/{n}", flush=True)
            item = item_at(dataset, kind, ref, pos)
            meta_ref = ref if kind != "source_dataset" else None
            img_path = vad.resolve_image_path(item, dataset, meta_ref)
            pending.append(preprocess(Image.open(img_path).convert("RGB")))
            if len(pending) == batch_size or pos == n - 1:
                batch = torch.stack(pending, dim=0).to(device)
                final, _stats_rows = vad.infer_batch(model, layer_transforms, cross_attn, batch)
                for bi in range(final.shape[0]):
                    maps.append(gaussian_filter(final[bi], sigma=vad.SIGMA).astype(np.float32))
                pending = []
                del batch
        del model, layer_transforms, cross_attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not smoke:
            save_maps(path, maps)
        return np.asarray(maps, dtype=np.float16)
    if variant == VAR_FB:
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]
        from experiments import phase_fbclip_pretrained_headroom_audit as fb
        fb_root = str(fb.FB_ROOT)
        if fb_root in sys.path:
            sys.path.remove(fb_root)
        sys.path.insert(0, fb_root)
        model, prompt, fb_args, _info = fb.load_fb_model(source, device)
        if smoke:
            fb_args.test_batch_size = 1
        maps, metas = fb.build_maps_for_pair(source, target, model, prompt, fb_args, device, smoke)
        arr = np.asarray(maps[fb.VARIANT], dtype=np.float32)
        if not smoke:
            save_maps(path, arr)
        arr = arr.astype(np.float16)
        del model, prompt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return arr
    raise KeyError(variant)


def resize_to_mask(score, mask, align_corners=True):
    return torch_resize(score, mask.shape, align_corners=align_corners)


def per_image_rank(score):
    flat = np.asarray(score, dtype=np.float32).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks.reshape(score.shape)


def f1_threshold(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    if y.size == 0 or y.max() == y.min():
        return float(np.nanpercentile(s, 95))
    vals = np.unique(np.quantile(s, np.linspace(0, 1, 256)))
    best_t, best_f = float(vals[-1]), -1.0
    for t in vals:
        pred = s >= t
        tp = float(np.sum(pred & (y > 0)))
        fp = float(np.sum(pred & (y == 0)))
        fn = float(np.sum((~pred) & (y > 0)))
        f = 2 * tp / max(2 * tp + fp + fn, 1e-12)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def morphology(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"group": "normal", "small": False, "thin": False, "large": False, "thick": False}
    area = int(mask.sum())
    ratio = area / float(mask.size)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    comp_areas = [int((labels == i).sum()) for i in range(1, n + 1)]
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=np.uint8))
    boundary = mask & (~eroded)
    thinness = float(boundary.sum() / max(area, 1))
    small = ratio <= 0.01 or (np.median(comp_areas) if comp_areas else area) <= 0.005 * mask.size
    thin = thinness >= 0.35
    return {
        "group": "small_thin" if small or thin else "large_thick",
        "small": bool(small),
        "thin": bool(thin),
        "large": not bool(small),
        "thick": not bool(thin),
        "area_ratio": ratio,
        "thinness": thinness,
        "component_count": int(n),
    }


def flatten_pair_maps(metas, maps_by_variant):
    y_all, s_all = [], {v: [] for v in maps_by_variant}
    cats = []
    morphs = []
    for i, meta in enumerate(metas):
        mask = meta["mask"].astype(np.uint8)
        y_all.append(mask.reshape(-1))
        cats.append(meta["category"])
        morphs.append(morphology(mask))
        for v, maps in maps_by_variant.items():
            score = resize_to_mask(maps[i], mask, align_corners=True)
            s_all[v].append(score.reshape(-1))
    y = np.concatenate(y_all).astype(np.uint8)
    s = {v: np.concatenate(vals).astype(np.float32) for v, vals in s_all.items()}
    return y, s, cats, morphs


def basic_table(metrics):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in metrics}
    rows = []
    for source, target in PAIRS:
        row = {"source": source, "dataset": target, "transfer": f"{source}->{target}"}
        for variant in ALL_VARIANTS:
            r = by.get((source, target, variant))
            for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
                row[f"{variant}_M_FR_{met}"] = fnum(r, f"M_FR_{met}") if r else float("nan")
        for specialist in SPECIALISTS:
            row[f"{specialist}_delta_P_AP_vs_E0"] = row[f"{specialist}_M_FR_P_AP"] - row[f"{VAR_E0}_M_FR_P_AP"]
        rows.append(row)
    return rows


def evaluate_variant_set(source, target, maps_by_variant, metas):
    old = None
    try:
        import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
        old = lzp.VARIANTS
        lzp.VARIANTS = tuple(maps_by_variant)
        raw, cats = evaluate_maps(source, target, maps_by_variant, metas)
    finally:
        if old is not None:
            lzp.VARIANTS = old
    return raw, cats


def oracle_map(maps_by_variant, metas, variants, mode):
    out = []
    for i, meta in enumerate(metas):
        mask = meta["mask"].astype(bool)
        base = resize_to_mask(maps_by_variant[VAR_E0][i], mask, align_corners=True)
        resized = [resize_to_mask(maps_by_variant[v][i], mask, align_corners=True) for v in variants]
        q = base.copy()
        if mode == "pixel":
            if mask.any():
                q[mask] = np.maximum.reduce([base[mask], *[s[mask] for s in resized]])
        elif mode == "region":
            labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
            for rid in range(1, n + 1):
                region = labels == rid
                vals = [float(base[region].mean())] + [float(s[region].mean()) for s in resized]
                best = int(np.argmax(vals))
                if best > 0:
                    q[region] = resized[best - 1][region]
        else:
            raise ValueError(mode)
        out.append(q.astype(np.float32))
    return out


def fast_global_eval(source, target, variant, maps, metas):
    masks, scores, labels, image_scores = [], [], [], []
    for score, meta in zip(maps, metas):
        mask = np.asarray(meta["mask"], dtype=np.uint8)
        s = resize_to_mask(score, mask, align_corners=True)
        masks.append(mask)
        scores.append(s.astype(np.float32))
        labels.append(int(meta["label"]))
        image_scores.append(float(np.max(s)))
    py = np.concatenate([m.reshape(-1) for m in masks]).astype(np.uint8)
    ps = np.concatenate([s.reshape(-1) for s in scores]).astype(np.float32)
    p_auc = safe_auc(py, ps)
    p_ap = safe_ap(py, ps)
    p_f1 = f1max(py, ps)
    pro = fast_aupro_vectorized(masks, scores)
    row = {
        "source": source,
        "dataset": target,
        "variant": variant,
        "aggregation": "DATASET_GLOBAL_FAST",
        "score_resolution": f"{maps[0].shape[0]}x{maps[0].shape[1]}" if maps else "",
        "M_FR_I_AUROC": safe_auc(labels, image_scores),
        "M_FR_I_AP": safe_ap(labels, image_scores),
        "M_FR_I_F1max": f1max(labels, image_scores),
        "M_FR_P_AUROC": p_auc,
        "M_FR_P_AP": p_ap,
        "M_FR_P_F1max": p_f1,
        "M_FR_PRO": pro,
        "M_FR_G_pixel": 0.5 * (p_auc + p_ap) if np.isfinite(p_auc) and np.isfinite(p_ap) else float("nan"),
    }
    row.update({k.replace("M_FR", "PG"): v for k, v in row.items() if k.startswith("M_FR")})
    row.update({k.replace("M_FR", "V_FR"): v for k, v in row.items() if k.startswith("M_FR")})
    return [row], []


def sampled_pixel_indices(mask, rng):
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    pos = np.flatnonzero(flat > 0)
    neg = np.flatnonzero(flat == 0)
    if pos.size > 5000:
        pos = rng.choice(pos, size=5000, replace=False)
    max_neg = min(5000, max(pos.size * 5, 1000), neg.size)
    if neg.size > max_neg:
        neg = rng.choice(neg, size=max_neg, replace=False)
    return np.concatenate([pos, neg]) if pos.size else neg


def sampled_oracle_eval(source, target, variant, maps_by_variant, metas, variants, mode):
    rng = np.random.default_rng(0)
    ys, ss, image_scores, labels = [], [], [], []
    for i, meta in enumerate(metas):
        mask = np.asarray(meta["mask"], dtype=np.uint8)
        mb = mask.astype(bool)
        base = resize_to_mask(maps_by_variant[VAR_E0][i], mask, align_corners=True)
        resized = [resize_to_mask(maps_by_variant[v][i], mask, align_corners=True) for v in variants]
        q = base.copy()
        if mode == "pixel":
            if mb.any():
                q[mb] = np.maximum.reduce([base[mb], *[s[mb] for s in resized]])
        elif mode == "region":
            labels_cc, n = ndimage.label(mb, structure=np.ones((3, 3), dtype=np.uint8))
            for rid in range(1, n + 1):
                region = labels_cc == rid
                vals = [float(base[region].mean())] + [float(s[region].mean()) for s in resized]
                best = int(np.argmax(vals))
                if best > 0:
                    q[region] = resized[best - 1][region]
        idx = sampled_pixel_indices(mask, rng)
        if idx.size:
            ys.append(mask.reshape(-1)[idx])
            ss.append(q.reshape(-1)[idx].astype(np.float32))
        image_scores.append(float(np.max(q)))
        labels.append(int(meta["label"]))
    y = np.concatenate(ys).astype(np.uint8) if ys else np.asarray([], dtype=np.uint8)
    s = np.concatenate(ss).astype(np.float32) if ss else np.asarray([], dtype=np.float32)
    p_auc = safe_auc(y, s)
    p_ap = safe_ap(y, s)
    p_f1 = f1max(y, s)
    row = {
        "source": source,
        "dataset": target,
        "variant": variant,
        "aggregation": "SAMPLED_FULL_RES_POSTHOC",
        "sampled_pixels": int(y.size),
        "sampled_anomaly_pixels": int(y.sum()) if y.size else 0,
        "M_FR_I_AUROC": safe_auc(labels, image_scores),
        "M_FR_I_AP": safe_ap(labels, image_scores),
        "M_FR_I_F1max": f1max(labels, image_scores),
        "M_FR_P_AUROC": p_auc,
        "M_FR_P_AP": p_ap,
        "M_FR_P_F1max": p_f1,
        "M_FR_PRO": float("nan"),
        "M_FR_G_pixel": 0.5 * (p_auc + p_ap) if np.isfinite(p_auc) and np.isfinite(p_ap) else float("nan"),
    }
    row.update({k.replace("M_FR", "PG"): v for k, v in row.items() if k.startswith("M_FR")})
    row.update({k.replace("M_FR", "V_FR"): v for k, v in row.items() if k.startswith("M_FR")})
    return [row], []


def sampled_sim_eval(source, target, variant, maps_by_variant, metas, specialist):
    rng = np.random.default_rng(0)
    ys, ss, image_scores, labels = [], [], [], []
    for i, meta in enumerate(metas):
        mask = np.asarray(meta["mask"], dtype=np.uint8)
        q0 = resize_to_mask(maps_by_variant[VAR_E0][i], mask, align_corners=True)
        qs = resize_to_mask(maps_by_variant[specialist][i], mask, align_corners=True)
        lo, hi = np.percentile(q0, [50, 90])
        band = (q0 >= lo) & (q0 <= hi)
        residual = np.clip(qs - q0, -1.0, 1.0)
        q = q0.copy()
        q[band] = q[band] + 0.25 * residual[band]
        idx = sampled_pixel_indices(mask, rng)
        if idx.size:
            ys.append(mask.reshape(-1)[idx])
            ss.append(q.reshape(-1)[idx].astype(np.float32))
        image_scores.append(float(np.max(q)))
        labels.append(int(meta["label"]))
    y = np.concatenate(ys).astype(np.uint8) if ys else np.asarray([], dtype=np.uint8)
    s = np.concatenate(ss).astype(np.float32) if ss else np.asarray([], dtype=np.float32)
    p_auc = safe_auc(y, s)
    p_ap = safe_ap(y, s)
    p_f1 = f1max(y, s)
    row = {
        "source": source,
        "dataset": target,
        "variant": variant,
        "aggregation": "SAMPLED_FULL_RES_POSTHOC",
        "sampled_pixels": int(y.size),
        "sampled_anomaly_pixels": int(y.sum()) if y.size else 0,
        "M_FR_I_AUROC": safe_auc(labels, image_scores),
        "M_FR_I_AP": safe_ap(labels, image_scores),
        "M_FR_I_F1max": f1max(labels, image_scores),
        "M_FR_P_AUROC": p_auc,
        "M_FR_P_AP": p_ap,
        "M_FR_P_F1max": p_f1,
        "M_FR_PRO": float("nan"),
        "M_FR_G_pixel": 0.5 * (p_auc + p_ap) if np.isfinite(p_auc) and np.isfinite(p_ap) else float("nan"),
    }
    row.update({k.replace("M_FR", "PG"): v for k, v in row.items() if k.startswith("M_FR")})
    row.update({k.replace("M_FR", "V_FR"): v for k, v in row.items() if k.startswith("M_FR")})
    return [row], []


def complementarity_rows(source, target, metas, maps_by_variant):
    rows = []
    totals = {
        specialist: {
            "e0_fn_pixels": 0,
            "fn_rescue_pixels": 0,
            "e0_tp_pixels": 0,
            "tp_destruction_pixels": 0,
            "e0_tn_pixels": 0,
            "normal_fp_amplification_pixels": 0,
            "specialist_score_on_e0_fn": [],
            "e0_score_on_e0_fn": [],
            "e0_fn_regions": 0,
            "fn_rescue_regions": 0,
            "e0_tp_regions": 0,
            "tp_destruction_regions": 0,
        }
        for specialist in SPECIALISTS + ("VisualAD_AND_FBCLIP",)
    }
    cons_vis, cons_fb, cons_corr = [], [], []
    for i, meta in enumerate(metas):
        mask = meta["mask"].astype(bool)
        q0 = resize_to_mask(maps_by_variant[VAR_E0][i], mask, align_corners=True)
        qv = resize_to_mask(maps_by_variant[VAR_VIS][i], mask, align_corners=True)
        qf = resize_to_mask(maps_by_variant[VAR_FB][i], mask, align_corners=True)
        scores = {VAR_E0: q0, VAR_VIS: qv, VAR_FB: qf}
        pred = {v: scores[v] >= np.percentile(scores[v], 95) for v in scores}
        e0_tp = mask & pred[VAR_E0]
        e0_fn = mask & (~pred[VAR_E0])
        e0_tn = (~mask) & (~pred[VAR_E0])
        for specialist in SPECIALISTS:
            t = totals[specialist]
            t["e0_fn_pixels"] += int(e0_fn.sum())
            t["fn_rescue_pixels"] += int((e0_fn & pred[specialist]).sum())
            t["e0_tp_pixels"] += int(e0_tp.sum())
            t["tp_destruction_pixels"] += int((e0_tp & (~pred[specialist])).sum())
            t["e0_tn_pixels"] += int(e0_tn.sum())
            t["normal_fp_amplification_pixels"] += int((e0_tn & pred[specialist]).sum())
            if e0_fn.any():
                t["specialist_score_on_e0_fn"].append(float(scores[specialist][e0_fn].mean()))
                t["e0_score_on_e0_fn"].append(float(q0[e0_fn].mean()))
        both = totals["VisualAD_AND_FBCLIP"]
        both["e0_fn_pixels"] += int(e0_fn.sum())
        both["fn_rescue_pixels"] += int((e0_fn & pred[VAR_VIS] & pred[VAR_FB]).sum())
        labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        for specialist in SPECIALISTS + ("VisualAD_AND_FBCLIP",):
            ps = (pred[VAR_VIS] & pred[VAR_FB]) if specialist == "VisualAD_AND_FBCLIP" else pred[specialist]
            t = totals[specialist]
            for rid in range(1, n + 1):
                reg = labels == rid
                if np.any(reg & (~pred[VAR_E0])):
                    t["e0_fn_regions"] += 1
                    if np.any(reg & ps):
                        t["fn_rescue_regions"] += 1
                if np.any(reg & pred[VAR_E0]):
                    t["e0_tp_regions"] += 1
                    if not np.any(reg & ps):
                        t["tp_destruction_regions"] += 1
        local_fn = mask & (q0 < np.percentile(q0, 90))
        if local_fn.any():
            rv, rf = per_image_rank(qv), per_image_rank(qf)
            cons_vis.append(float(rv[local_fn].mean()))
            cons_fb.append(float(rf[local_fn].mean()))
            if local_fn.sum() > 2:
                cons_corr.append(float(np.corrcoef(qv[local_fn].reshape(-1), qf[local_fn].reshape(-1))[0, 1]))
    for specialist, t in totals.items():
        rows.append({
            "source": source,
            "dataset": target,
            "specialist": specialist,
            "scope": "all_pixels_per_image_top5pct_posthoc",
            "e0_fn_pixels": t["e0_fn_pixels"],
            "fn_rescue_pixels": t["fn_rescue_pixels"],
            "fn_rescue_ratio": float(t["fn_rescue_pixels"] / max(t["e0_fn_pixels"], 1)),
            "e0_tp_pixels": t["e0_tp_pixels"],
            "tp_destruction_pixels": t["tp_destruction_pixels"],
            "tp_destruction_ratio": float(t["tp_destruction_pixels"] / max(t["e0_tp_pixels"], 1)),
            "e0_tn_pixels": t["e0_tn_pixels"],
            "normal_fp_amplification_pixels": t["normal_fp_amplification_pixels"],
            "normal_fp_amplification_ratio": float(t["normal_fp_amplification_pixels"] / max(t["e0_tn_pixels"], 1)),
            "specialist_score_on_e0_fn_mean": float(np.mean(t["specialist_score_on_e0_fn"])) if t["specialist_score_on_e0_fn"] else float("nan"),
            "e0_score_on_e0_fn_mean": float(np.mean(t["e0_score_on_e0_fn"])) if t["e0_score_on_e0_fn"] else float("nan"),
        })
        rows.append({
            "source": source,
            "dataset": target,
            "specialist": specialist,
            "scope": "connected_anomaly_regions_per_image_top5pct_posthoc",
            "e0_fn_regions": t["e0_fn_regions"],
            "fn_rescue_regions": t["fn_rescue_regions"],
            "fn_rescue_region_ratio": float(t["fn_rescue_regions"] / max(t["e0_fn_regions"], 1)),
            "e0_tp_regions": t["e0_tp_regions"],
            "tp_destruction_regions": t["tp_destruction_regions"],
            "tp_destruction_region_ratio": float(t["tp_destruction_regions"] / max(t["e0_tp_regions"], 1)),
        })
    rows.append({
        "source": source,
        "dataset": target,
        "specialist": "VisualAD_FBCLIP_consensus",
        "scope": "e0_low_rank_anomaly_pixels",
        "visualad_rank_mean": float(np.nanmean(cons_vis)) if cons_vis else float("nan"),
        "fbclip_rank_mean": float(np.nanmean(cons_fb)) if cons_fb else float("nan"),
        "specialist_corr_mean": float(np.nanmean(cons_corr)) if cons_corr else float("nan"),
    })
    return rows


def morphology_rows(source, target, metas, maps_by_variant):
    rng = np.random.default_rng(0)
    rows = []
    for group_name, selector in (
        ("small", lambda m: m.get("small", False)),
        ("thin", lambda m: m.get("thin", False)),
        ("large", lambda m: m.get("large", False)),
        ("thick", lambda m: m.get("thick", False)),
    ):
        keep = [i for i, meta in enumerate(metas) if int(meta["label"]) == 1 and selector(morphology(meta["mask"]))]
        if not keep:
            continue
        samples_y = []
        samples_s = {v: [] for v in ALL_VARIANTS}
        for i in keep:
            mask = metas[i]["mask"].astype(np.uint8)
            flat_mask = mask.reshape(-1)
            pos = np.flatnonzero(flat_mask > 0)
            neg = np.flatnonzero(flat_mask == 0)
            max_neg = min(20000, max(pos.size * 10, 2000), neg.size)
            if neg.size > max_neg:
                neg = rng.choice(neg, size=max_neg, replace=False)
            idx = np.concatenate([pos, neg]) if pos.size else neg
            if idx.size == 0:
                continue
            samples_y.append(flat_mask[idx])
            for v in ALL_VARIANTS:
                samples_s[v].append(resize_to_mask(maps_by_variant[v][i], mask, align_corners=True).reshape(-1)[idx].astype(np.float32))
        if not samples_y:
            continue
        py = np.concatenate(samples_y).astype(np.uint8)
        row = {
            "source": source,
            "dataset": target,
            "morphology_group": group_name,
            "images": len(keep),
            "sampled_pixels": int(py.size),
            "anomaly_pixels": int(py.sum()),
            "diagnostic": "all anomaly pixels plus deterministic sampled normal pixels",
        }
        for v in ALL_VARIANTS:
            ps = np.concatenate(samples_s[v]).astype(np.float32)
            row[f"{v}_P_AP"] = safe_ap(py, ps)
            row[f"{v}_P_AUROC"] = safe_auc(py, ps)
            row[f"{v}_P_F1max"] = f1max(py, ps)
        for specialist in SPECIALISTS:
            row[f"{specialist}_delta_P_AP_vs_E0"] = row[f"{specialist}_P_AP"] - row[f"{VAR_E0}_P_AP"]
        rows.append(row)
    return rows


def simulation_maps(maps_by_variant, specialist):
    out = []
    for q0, qs in zip(maps_by_variant[VAR_E0], maps_by_variant[specialist]):
        q0 = np.asarray(q0, dtype=np.float32)
        qs = torch_resize(qs, q0.shape, align_corners=True) if qs.shape != q0.shape else np.asarray(qs, dtype=np.float32)
        lo, hi = np.percentile(q0, [50, 90])
        band = (q0 >= lo) & (q0 <= hi)
        residual = np.clip(qs - q0, -1.0, 1.0)
        q = q0.copy()
        q[band] = q[band] + 0.25 * residual[band]
        out.append(q.astype(np.float32))
    return out


def run(args):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    metric_rows = load_metric_rows()
    basic = basic_table(metric_rows)
    write_csv(OUT / "RAW_COMPLEMENTARITY.csv", basic)

    all_raw = []
    oracle_rows = []
    comp_rows = []
    morph_rows = []
    sim_rows = []
    cache_status = []

    for source, target in PAIRS:
        print(f"[CSCA] {source}->{target}", flush=True)
        metas = load_metas(source, target, device, args.smoke)
        e0 = load_e0_maps(source, target, metas)
        maps = {VAR_E0: e0}
        for variant in SPECIALISTS:
            before = specialist_cache_path(source, target, variant).exists()
            maps[variant] = load_specialist_maps(source, target, variant, device, args.smoke, args.batch_size)
            cache_status.append({"source": source, "dataset": target, "variant": variant, "cache_reused": before, "maps": int(maps[variant].shape[0])})
            if maps[variant].shape[0] != len(metas) and not args.smoke:
                raise RuntimeError(f"{variant} map/meta mismatch {source}->{target}: {maps[variant].shape[0]} vs {len(metas)}")
        if args.smoke:
            n_common = min(len(metas), *(maps[v].shape[0] for v in ALL_VARIANTS))
            metas = metas[:n_common]
            maps = {v: maps[v][:n_common] for v in maps}

        comp_rows.extend(complementarity_rows(source, target, metas, maps))
        morph_rows.extend(morphology_rows(source, target, metas, maps))

        oracle_defs = {
            "O1_E0_VisualAD_pixel": (VAR_VIS,),
            "O2_E0_FBCLIP_pixel": (VAR_FB,),
            "O3_E0_VisualAD_FBCLIP_pixel": (VAR_VIS, VAR_FB),
            "O1_E0_VisualAD_region": (VAR_VIS,),
            "O2_E0_FBCLIP_region": (VAR_FB,),
            "O3_E0_VisualAD_FBCLIP_region": (VAR_VIS, VAR_FB),
        }
        for name, variants in oracle_defs.items():
            mode = "pixel" if name.endswith("_pixel") else "region"
            raw, _cats = sampled_oracle_eval(source, target, name, maps, metas, variants, mode)
            all_raw.extend(raw)
            r = raw[0] if raw else {}
            e0_metric = next((rr for rr in metric_rows if rr.get("source") == source and rr.get("dataset") == target and rr.get("variant") == VAR_E0), None)
            oracle_rows.append({
                "source": source,
                "dataset": target,
                "oracle": name,
                "E0_M_FR_P_AP": fnum(e0_metric, "M_FR_P_AP"),
                "Oracle_M_FR_P_AP": fnum(r, "M_FR_P_AP"),
                "Delta_M_FR_P_AP": fnum(r, "M_FR_P_AP") - fnum(e0_metric, "M_FR_P_AP"),
                "Oracle_M_FR_P_AUROC": fnum(r, "M_FR_P_AUROC"),
                "Oracle_M_FR_P_F1max": fnum(r, "M_FR_P_F1max"),
                "Oracle_M_FR_PRO": fnum(r, "M_FR_PRO"),
            })
            del raw
            gc.collect()
        e0_metric = next((r for r in metric_rows if r.get("source") == source and r.get("dataset") == target and r.get("variant") == VAR_E0), None)
        for name in ("SIM_E0_plus_VisualAD_ambiguity", "SIM_E0_plus_FBCLIP_ambiguity"):
            raw, _cats = sampled_sim_eval(source, target, name, maps, metas, VAR_VIS if "VisualAD" in name else VAR_FB)
            all_raw.extend(raw)
            r = raw[0] if raw else {}
            sim_rows.append({
                "source": source,
                "dataset": target,
                "variant": name,
                "E0_M_FR_P_AP": fnum(e0_metric, "M_FR_P_AP"),
                "M_FR_P_AP": fnum(r, "M_FR_P_AP"),
                "Delta_M_FR_P_AP": fnum(r, "M_FR_P_AP") - fnum(e0_metric, "M_FR_P_AP"),
                "M_FR_P_AUROC": fnum(r, "M_FR_P_AUROC"),
                "M_FR_P_F1max": fnum(r, "M_FR_P_F1max"),
                "M_FR_PRO": fnum(r, "M_FR_PRO"),
            })
            del raw
            gc.collect()
        write_csv(OUT / "ORACLE_HEADROOM.partial.csv", oracle_rows)
        write_csv(OUT / "ERROR_COMPLEMENTARITY.partial.csv", comp_rows)
        write_csv(OUT / "MORPHOLOGY_COMPLEMENTARITY.partial.csv", morph_rows)
        write_csv(OUT / "AMBIGUITY_SIMULATION.partial.csv", sim_rows)
        del maps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    outcome, decision = decide(oracle_rows, comp_rows, basic)
    write_csv(OUT / "ORACLE_HEADROOM.csv", oracle_rows)
    write_csv(OUT / "ERROR_COMPLEMENTARITY.csv", comp_rows)
    write_csv(OUT / "MORPHOLOGY_COMPLEMENTARITY.csv", morph_rows)
    write_csv(OUT / "AMBIGUITY_SIMULATION.csv", sim_rows)
    write_csv(OUT / "ORACLE_EVAL_RAW_METRICS.csv", all_raw)
    write_csv(OUT / "MAP_CACHE_STATUS.csv", cache_status)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision": decision, "runtime_sec": time.time() - t0, "no_training": True, "no_target_tuning": True})
    write_report(outcome, decision, basic, oracle_rows, comp_rows, morph_rows, sim_rows, time.time() - t0)
    print(json.dumps({"outcome": outcome, **decision, "runtime_sec": time.time() - t0}, indent=2), flush=True)


def decide(oracle_rows, comp_rows, basic_rows):
    best_by_pair = {}
    for r in oracle_rows:
        if not r["oracle"].endswith("_region"):
            continue
        key = (r["source"], r["dataset"])
        val = fnum(r, "Delta_M_FR_P_AP")
        if np.isfinite(val):
            best_by_pair[key] = max(best_by_pair.get(key, -1e9), val)
    vals = np.asarray(list(best_by_pair.values()), dtype=np.float64)
    mean_oracle = float(np.nanmean(vals)) if vals.size else float("nan")
    positive = int(np.sum(vals > 0)) if vals.size else 0
    visa_only = False
    if vals.size:
        visa_gain = best_by_pair.get(("Full MVTec", "VisA"), float("nan"))
        others = [v for k, v in best_by_pair.items() if k != ("Full MVTec", "VisA")]
        visa_only = np.isfinite(visa_gain) and visa_gain >= 2.0 and all((not np.isfinite(x)) or x < 1.0 for x in others)
    fp_amp = np.nanmean([fnum(r, "normal_fp_amplification_ratio") for r in comp_rows if r.get("scope") == "all_pixels_per_image_top5pct_posthoc"])
    tp_dest = np.nanmean([fnum(r, "tp_destruction_ratio") for r in comp_rows if r.get("scope") == "all_pixels_per_image_top5pct_posthoc"])
    ambiguity_positive = False
    if mean_oracle >= 2.0 and positive >= 3:
        ambiguity_positive = True
    if mean_oracle < 0.5:
        outcome = "SPECIALIST_INTEGRATION_HAS_NO_USEFUL_HEADROOM"
    elif visa_only:
        outcome = "SPECIALIST_COMPLEMENTARITY_IS_VISA_SPECIFIC"
    elif np.isfinite(fp_amp) and fp_amp > 0.02 and fp_amp >= tp_dest:
        outcome = "SPECIALIST_SIGNAL_EXISTS_BUT_NORMAL_RANKING_IS_UNSAFE"
    elif np.isfinite(tp_dest) and tp_dest > 0.4:
        outcome = "SPECIALIST_SIGNAL_EXISTS_BUT_STRUCTURAL_EVIDENCE_IS_LOST"
    elif mean_oracle >= 2.0 and positive >= 3 and ambiguity_positive:
        outcome = "CORE_SPECIALIST_COMPLEMENTARITY_HAS_STRONG_HEADROOM"
    else:
        outcome = "SPECIALIST_INTEGRATION_HAS_NO_USEFUL_HEADROOM"
    return outcome, {
        "mean_best_region_oracle_delta_P_AP": mean_oracle,
        "positive_oracle_headroom_transfers": positive,
        "mean_fp_amplification_ratio": float(fp_amp) if np.isfinite(fp_amp) else float("nan"),
        "mean_tp_destruction_ratio": float(tp_dest) if np.isfinite(tp_dest) else float("nan"),
    }


def summarize_delta(rows, prefix):
    vals = [fnum(r, prefix) for r in rows]
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), 0
    return float(np.mean(vals)), int(np.sum(vals > 0))


def write_report(outcome, decision, basic, oracle, comp, morph, sim, runtime):
    lines = [
        "# Phase CSCA: Core-Specialist Complementarity and Oracle Headroom Audit",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "No training, no target tuning. E0 uses the locked `Canonical Two-State + NTR + Fixed LRP` maps from FOVA/LZP cache. VisualAD and FB-CLIP maps are reused from CSCA cache when present; otherwise official frozen/pretrained inference is rerun only to create maps.",
        "",
        "## Basic P-AP Comparison",
        "",
        "| Transfer | E0 P-AP | VisualAD P-AP | FB-CLIP P-AP | VisualAD Delta | FB Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in basic:
        lines.append(
            f"| {r['transfer']} | {fnum(r, VAR_E0 + '_M_FR_P_AP'):.4f} | "
            f"{fnum(r, VAR_VIS + '_M_FR_P_AP'):.4f} | {fnum(r, VAR_FB + '_M_FR_P_AP'):.4f} | "
            f"{fnum(r, VAR_VIS + '_delta_P_AP_vs_E0'):+.4f} | {fnum(r, VAR_FB + '_delta_P_AP_vs_E0'):+.4f} |"
        )
    lines += [
        "",
        "## Oracle Headroom",
        "",
        "| Transfer | Oracle | E0 P-AP | Oracle P-AP | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for r in oracle:
        if not r["oracle"].endswith("_region"):
            continue
        lines.append(f"| {r['source']}->{r['dataset']} | {r['oracle']} | {fnum(r,'E0_M_FR_P_AP'):.4f} | {fnum(r,'Oracle_M_FR_P_AP'):.4f} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} |")
    lines += [
        "",
        "## Error Complementarity",
        "",
        "| Transfer | Specialist | FN rescue | TP destruction | FP amplification |",
        "|---|---|---:|---:|---:|",
    ]
    for r in comp:
        if r.get("scope") != "all_pixels_per_image_top5pct_posthoc" or r.get("specialist") == "VisualAD_AND_FBCLIP":
            continue
        lines.append(f"| {r['source']}->{r['dataset']} | {r['specialist']} | {fnum(r,'fn_rescue_ratio'):.4f} | {fnum(r,'tp_destruction_ratio'):.4f} | {fnum(r,'normal_fp_amplification_ratio'):.4f} |")
    lines += [
        "",
        "## Ambiguity Simulation",
        "",
        "| Transfer | Variant | Delta P-AP |",
        "|---|---|---:|",
    ]
    for r in sim:
        lines.append(f"| {r['source']}->{r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} |")
    lines += [
        "",
        "## Decision Notes",
        "",
        f"- mean best region-oracle Delta P-AP: {decision['mean_best_region_oracle_delta_P_AP']:+.4f}",
        f"- positive oracle-headroom transfers: {decision['positive_oracle_headroom_transfers']}/4",
        f"- mean FP amplification ratio: {decision['mean_fp_amplification_ratio']:.4f}",
        f"- mean TP destruction ratio: {decision['mean_tp_destruction_ratio']:.4f}",
        "",
        "Q1-Q6 are answered from `ERROR_COMPLEMENTARITY.csv`, `MORPHOLOGY_COMPLEMENTARITY.csv`, `ORACLE_HEADROOM.csv`, and `AMBIGUITY_SIMULATION.csv`; oracle rows are explicitly post-hoc upper bounds and are not deployable methods.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
