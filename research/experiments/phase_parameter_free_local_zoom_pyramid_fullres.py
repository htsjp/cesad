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

from experiments.phase_final_bottleneck_attribution import OUT as BOTTLENECK_OUT
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells, sota_reference
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, fnum, logits_from_z
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np
from experiments.standard_zsad_metrics import f1max, resize_mask_flat, safe_ap, safe_auc
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_parameter_free_local_zoom_pyramid_fullres/formal")
CORE_PAIRS = {
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "CVC-ClinicDB"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "CVC-ClinicDB"),
    ("Full MVTec", "MPDD"),
}
VARIANTS = ("B0_HardNormal", "E1_LocalZoomOnly", "E2_GlobalLocalZoomPyramid", "E3_ReencodingResolutionControl")
PROTOCOLS = {
    "M": {"prefix": "M_FR", "align_corners": True, "name": "MoECLIP-compatible full-res"},
    "V": {"prefix": "V_FR", "align_corners": False, "name": "VisualAD-compatible full-res"},
}
METRICS = ("I-AUROC", "I-AP", "I-F1max", "P-AUROC", "P-AP", "P-F1max", "PRO", "G_pixel")


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


def torch_resize(score, size, align_corners=True):
    arr = np.asarray(score, dtype=np.float32)
    x = torch.tensor(arr, dtype=torch.float32).view(1, 1, arr.shape[0], arr.shape[1])
    y = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=align_corners)
    return y.view(*size).numpy().astype(np.float32)


def mask_np(item):
    m = item["img_mask"].detach().cpu().numpy()
    return (np.squeeze(m) > 0.5).astype(np.uint8)


def item_at(dataset, kind, ref, pos):
    return dataset[int(ref)] if kind == "source_dataset" else dataset[pos]


def item_meta(item, kind, ref, target):
    if kind == "source_dataset":
        return str(item.get("cls_name", target)), int(item.get("anomaly", 0)), str(item.get("img_path", ""))
    row = dict(ref)
    return str(row.get("category", item.get("cls_name", target))), int(row.get("label", item.get("anomaly", 0))), str(row.get("img_path", item.get("img_path", "")))


def summarize_category(labels, image_scores, masks, scores):
    py = np.concatenate([m.reshape(-1) for m in masks]) if masks else np.asarray([], dtype=np.uint8)
    ps = np.concatenate([s.reshape(-1) for s in scores]) if scores else np.asarray([], dtype=np.float32)
    out = {
        "I-AUROC": safe_auc(labels, image_scores),
        "I-AP": safe_ap(labels, image_scores),
        "I-F1max": f1max(labels, image_scores),
        "P-AUROC": safe_auc(py, ps),
        "P-AP": safe_ap(py, ps),
        "P-F1max": f1max(py, ps),
        "PRO": fast_aupro_vectorized(masks, scores),
    }
    out["G_pixel"] = 0.5 * (out["P-AUROC"] + out["P-AP"]) if np.isfinite(out["P-AUROC"]) and np.isfinite(out["P-AP"]) else float("nan")
    return out


def fast_aupro_vectorized(mask_list, score_list, fpr_limit=0.3, steps=200):
    masks = [np.asarray(m, dtype=bool) for m in mask_list]
    scores = [np.asarray(s, dtype=np.float32) for s in score_list]
    if not masks or not scores:
        return float("nan")
    all_min, all_max = [], []
    for s in scores:
        if s.size:
            all_min.append(float(np.min(s)))
            all_max.append(float(np.max(s)))
    if not all_min:
        return float("nan")
    mn, mx = min(all_min), max(all_max)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return float("nan")
    thresholds = np.linspace(mn, mx, int(steps), dtype=np.float32)

    neg_total = 0
    fp_counts = np.zeros_like(thresholds, dtype=np.float64)
    region_sum = np.zeros_like(thresholds, dtype=np.float64)
    region_count = 0
    structure = np.ones((3, 3), dtype=np.int8)
    for mask, score in zip(masks, scores):
        if mask.shape != score.shape:
            raise ValueError(f"mask/score shape mismatch: {mask.shape} vs {score.shape}")
        neg = np.sort(score[~mask].reshape(-1))
        if neg.size:
            neg_total += int(neg.size)
            fp_counts += neg.size - np.searchsorted(neg, thresholds, side="right")
        labels, n = ndimage.label(mask, structure=structure)
        for rid in range(1, n + 1):
            vals = np.sort(score[labels == rid].reshape(-1))
            if vals.size:
                region_sum += (vals.size - np.searchsorted(vals, thresholds, side="right")) / float(vals.size)
                region_count += 1
    if neg_total == 0 or region_count == 0:
        return float("nan")
    fprs = fp_counts / float(neg_total)
    pros = region_sum / float(region_count)
    keep = np.isfinite(fprs) & np.isfinite(pros) & (fprs <= fpr_limit)
    if keep.sum() < 2:
        return float("nan")
    f = fprs[keep]
    p = pros[keep]
    order = np.argsort(f)
    f = f[order]
    p = p[order]
    if f[-1] <= f[0]:
        return float("nan")
    return float(np.trapezoid(p, (f - f[0]) / (f[-1] - f[0])) * 100.0)


def macro(rows, prefix=None):
    out = {}
    for met in METRICS:
        key = f"{prefix}_{met.replace('-', '_')}" if prefix else met
        vals = np.asarray([fnum(r, key if prefix else met) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[key] = float(vals.mean()) if vals.size else float("nan")
    return out


def pg_metrics(labels, image_scores, masks, maps):
    side = int(maps[0].shape[0]) if maps else SIDE
    gmasks = []
    for m in masks:
        t = torch.tensor(m.astype(np.float32)).view(1, 1, *m.shape)
        gm = F.interpolate(t, size=(side, side), mode="nearest").view(side, side).numpy()
        gmasks.append((gm > 0.5).astype(np.uint8))
    return summarize_category(labels, image_scores, gmasks, maps)


def score_feature_grid(model_info, patch_np, device):
    m_t = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v_t = torch.tensor(model_info["v1"], dtype=torch.float32, device=device)
    if model_info["use_align"]:
        z = aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).unsqueeze(0)
        score = logits_from_z(z, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    else:
        z = norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
        zt = torch.tensor(z, dtype=torch.float32, device=device)
        score = contrast_logits(zt, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    return score


def quadrant_crops(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    boxes = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h),
    ]
    return [img.crop(b) for b in boxes]


@torch.no_grad()
def local_zoom_map(model_info, clip_model, preprocess, img_path, device):
    crops = quadrant_crops(img_path)
    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patches = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)
    qs = [score_feature_grid(model_info, patches[i], device) for i in range(4)]
    s = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    s[:SIDE, :SIDE] = qs[0]
    s[:SIDE, SIDE:] = qs[1]
    s[SIDE:, :SIDE] = qs[2]
    s[SIDE:, SIDE:] = qs[3]
    return s


def build_maps(model_info, clip_model, preprocess, patch, refs, dataset, kind, target, device):
    n = len(refs)
    maps = {v: [] for v in VARIANTS}
    metas = []
    scale_rows = []
    for pos, ref in enumerate(refs):
        if pos % 25 == 0:
            print(f"[LZP] {model_info['source']} -> {target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        if not path:
            raise ValueError(f"missing img_path for {model_info['source']} -> {target} pos={pos}")
        gt = mask_np(item)
        global37 = score_feature_grid(model_info, patch[pos], device)
        local74 = local_zoom_map(model_info, clip_model, preprocess, path, device)
        global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
        final74 = 0.5 * global74 + 0.5 * local74
        local_down37 = torch_resize(local74, (SIDE, SIDE), align_corners=True)
        e3 = 0.5 * global37 + 0.5 * local_down37
        maps["B0_HardNormal"].append(global37)
        maps["E1_LocalZoomOnly"].append(local74)
        maps["E2_GlobalLocalZoomPyramid"].append(final74)
        maps["E3_ReencodingResolutionControl"].append(e3)
        metas.append({"category": cat, "label": label, "mask": gt, "img_path": path})
        scale_rows.append(score_scale_row(model_info["source"], target, cat, pos, global37, local74, final74))
    return maps, metas, scale_rows


def score_scale_row(source, target, category, pos, global37, local74, final74):
    row = {"source": source, "dataset": target, "category": category, "image_index": pos}
    for name, score in (("global37", global37), ("local74", local74), ("final74", final74)):
        flat = np.asarray(score, dtype=np.float32).reshape(-1)
        row[f"{name}_mean"] = float(np.mean(flat))
        row[f"{name}_std"] = float(np.std(flat))
        row[f"{name}_P95"] = float(np.percentile(flat, 95))
        row[f"{name}_P99"] = float(np.percentile(flat, 99))
    return row


def evaluate_maps(source, target, maps, metas):
    raw_rows, cat_rows = [], []
    for variant in VARIANTS:
        print(f"[LZP] metrics {source} -> {target} {variant}", flush=True)
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
            print(f"[LZP] metrics {source} -> {target} {variant} category={cat}", flush=True)
            pg = pg_metrics(vals["labels"], vals["image_scores"], vals["masks"], vals["maps"])
            pg_row = {
                "source": source,
                "dataset": target,
                "category": cat,
                "variant": variant,
                "protocol": "PG",
                "aggregation": "CATEGORY",
                "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}",
            }
            for met, val in pg.items():
                pg_row[f"PG_{met.replace('-', '_')}"] = val
            cat_rows.append(pg_row)
            pg_cat_rows.append(pg_row)
            for pkey, pcfg in PROTOCOLS.items():
                scores = [torch_resize(s, m.shape, align_corners=pcfg["align_corners"]) for s, m in zip(vals["maps"], vals["masks"])]
                fr = summarize_category(vals["labels"], vals["image_scores"], vals["masks"], scores)
                fr_row = {
                    "source": source,
                    "dataset": target,
                    "category": cat,
                    "variant": variant,
                    "protocol": pkey,
                    "protocol_name": pcfg["name"],
                    "aggregation": "CATEGORY",
                    "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}",
                }
                for met, val in fr.items():
                    fr_row[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = val
                cat_rows.append(fr_row)
                fr_cat_by_protocol[pkey].append(fr_row)
        raw = {
            "source": source,
            "dataset": target,
            "variant": variant,
            "aggregation": "CATEGORY_MACRO",
            "score_resolution": f"{maps[variant][0].shape[0]}x{maps[variant][0].shape[1]}",
        }
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


def boundary_band(side):
    b = np.zeros((side, side), dtype=bool)
    mid = side // 2
    for k in (mid - 2, mid - 1, mid, mid + 1):
        if 0 <= k < side:
            b[k, :] = True
            b[:, k] = True
    return b


def stitch_boundary_diagnostics(source, target, maps, metas):
    rows = []
    band = boundary_band(SIDE * 2)
    for i, meta in enumerate(metas):
        s = np.asarray(maps["E1_LocalZoomOnly"][i], dtype=np.float32)
        gt = torch.tensor(meta["mask"].astype(np.float32)).view(1, 1, *meta["mask"].shape)
        gm = F.interpolate(gt, size=s.shape, mode="nearest").view(*s.shape).numpy() > 0.5
        flat = s.reshape(-1)
        k = max(1, int(np.ceil(flat.size * 0.01)))
        thr = np.partition(flat, flat.size - k)[flat.size - k]
        top = s >= thr
        normal = ~gm
        anomaly = gm
        disc_h = np.abs(s[SIDE - 1, :] - s[SIDE, :])
        disc_v = np.abs(s[:, SIDE - 1] - s[:, SIDE])
        rows.append({
            "source": source,
            "dataset": target,
            "category": meta["category"],
            "image_index": i,
            "boundary_score_discontinuity_mean": float(np.mean(np.concatenate([disc_h, disc_v]))),
            "boundary_score_discontinuity_std": float(np.std(np.concatenate([disc_h, disc_v]))),
            "top1_boundary_normal_fpr": float(np.sum(top & band & normal) / max(np.sum(band & normal), 1)),
            "top1_nonboundary_normal_fpr": float(np.sum(top & (~band) & normal) / max(np.sum((~band) & normal), 1)),
            "top1_boundary_anomaly_recall": float(np.sum(top & band & anomaly) / max(np.sum(band & anomaly), 1)),
            "top1_nonboundary_anomaly_recall": float(np.sum(top & (~band) & anomaly) / max(np.sum((~band) & anomaly), 1)),
        })
    return rows


def localization_diagnostics(source, target, maps, metas):
    rows = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for variant in VARIANTS:
        for i, meta in enumerate(metas):
            s = np.asarray(maps[variant][i], dtype=np.float32)
            gt = torch.tensor(meta["mask"].astype(np.float32)).view(1, 1, *meta["mask"].shape)
            gm = F.interpolate(gt, size=s.shape, mode="nearest").view(*s.shape).numpy() > 0.5
            dil = ndimage.binary_dilation(gm, structure=structure)
            ero = ndimage.binary_erosion(gm, structure=structure) if gm.any() else gm
            boundary = gm & (~ero)
            interior = ero
            near = dil & (~gm)
            far = ~dil
            row = {"source": source, "dataset": target, "category": meta["category"], "variant": variant, "image_index": i}
            for name, region in (("anomaly", gm), ("anomaly_interior", interior), ("anomaly_boundary", boundary), ("normal_near_boundary", near), ("far_normal", far)):
                vals = s[region]
                row[f"{name}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
                row[f"{name}_P90"] = float(np.percentile(vals, 90)) if vals.size else float("nan")
            rows.append(row)
    return rows


def evaluate_all(smoke=False, core_only=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    models = [load_visa_source(device), load_mvtec_source()]
    raw, cats, loc, stitch, scale = [], [], [], [], []
    suffix = ".smoke" if smoke else ""
    partial = OUT / f"RAW_METRICS{suffix}.partial.csv"
    done = set()
    if partial.exists() and not smoke:
        done = {(r["source"], r["dataset"]) for r in read_csv(partial) if r.get("variant") == "E3_ReencodingResolutionControl"}
        raw = read_csv(partial)
        for path, target_list in (
            (OUT / f"PER_CATEGORY_FULL_RES{suffix}.partial.csv", cats),
            (OUT / f"LOCALIZATION_DIAGNOSTICS{suffix}.partial.csv", loc),
            (OUT / f"STITCH_BOUNDARY_DIAGNOSTICS{suffix}.partial.csv", stitch),
            (OUT / f"SCORE_SCALE_DIAGNOSTICS{suffix}.partial.csv", scale),
        ):
            if path.exists():
                target_list.extend(read_csv(path))
    for model_info in models:
        cells, _audits = protocol_cells(model_info["source"], device, smoke)
        for cell in cells:
            _cell_id, target, dataset, idx_or_rows, patch, kind = cell
            pair = (model_info["source"], target)
            if core_only and pair not in CORE_PAIRS:
                continue
            if pair in done:
                print(f"[LZP] skip completed {pair}", flush=True)
                continue
            n = min(24, patch.shape[0]) if smoke else patch.shape[0]
            refs = idx_or_rows[:n]
            print(f"[LZP] evaluate {model_info['source']} -> {target} n={n}", flush=True)
            maps, metas, scale_rows = build_maps(model_info, clip_model, preprocess, patch[:n], refs, dataset, kind, target, device)
            rr, cc = evaluate_maps(model_info["source"], target, maps, metas)
            raw.extend(rr)
            cats.extend(cc)
            loc.extend(localization_diagnostics(model_info["source"], target, maps, metas))
            stitch.extend(stitch_boundary_diagnostics(model_info["source"], target, maps, metas))
            scale.extend(scale_rows)
            write_csv(OUT / f"RAW_METRICS{suffix}.partial.csv", raw)
            write_csv(OUT / f"PER_CATEGORY_FULL_RES{suffix}.partial.csv", cats)
            write_csv(OUT / f"LOCALIZATION_DIAGNOSTICS{suffix}.partial.csv", loc)
            write_csv(OUT / f"STITCH_BOUNDARY_DIAGNOSTICS{suffix}.partial.csv", stitch)
            write_csv(OUT / f"SCORE_SCALE_DIAGNOSTICS{suffix}.partial.csv", scale)
    return raw, cats, loc, stitch, scale


def resolution_control(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, dataset in sorted({(r["source"], r["dataset"]) for r in raw}):
        b0 = by.get((source, dataset, "B0_HardNormal"))
        e2 = by.get((source, dataset, "E2_GlobalLocalZoomPyramid"))
        e3 = by.get((source, dataset, "E3_ReencodingResolutionControl"))
        if not (b0 and e2 and e3):
            continue
        row = {"source": source, "dataset": dataset}
        for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
            row[f"E2_minus_B0_M_FR_{met}"] = fnum(e2, f"M_FR_{met}") - fnum(b0, f"M_FR_{met}")
            row[f"E3_minus_B0_M_FR_{met}"] = fnum(e3, f"M_FR_{met}") - fnum(b0, f"M_FR_{met}")
            row[f"E2_minus_E3_M_FR_{met}"] = fnum(e2, f"M_FR_{met}") - fnum(e3, f"M_FR_{met}")
        rows.append(row)
    return rows


def small_sparse_analysis(cat_rows):
    morph_path = BOTTLENECK_OUT / "CATEGORY_MORPHOLOGY.csv"
    if not morph_path.exists():
        return [{"status": "MORPHOLOGY_FILE_MISSING", "path": str(morph_path)}]
    morph = read_csv(morph_path)
    morph_by = {(r["source"], r["dataset"], r["category"]): r for r in morph}
    vals = {}
    for r in cat_rows:
        if r.get("protocol") != "M":
            continue
        key = (r["source"], r["dataset"], r["category"], r["variant"])
        vals[key] = r
    rows = []
    for source, dataset, cat, variant in sorted(vals):
        if variant != "E2_GlobalLocalZoomPyramid":
            continue
        b = vals.get((source, dataset, cat, "B0_HardNormal"))
        e = vals[(source, dataset, cat, variant)]
        m = morph_by.get((source, dataset, cat))
        if not (b and m):
            continue
        rows.append({
            "source": source,
            "dataset": dataset,
            "category": cat,
            "anomaly_patch_ratio_mean": fnum(m, "anomaly_patch_ratio_mean"),
            "boundary_to_area_ratio_mean": fnum(m, "boundary_to_area_ratio_mean"),
            "delta_E2_minus_B0_M_FR_P_AP": fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP"),
            "delta_E2_minus_B0_M_FR_G_pixel": fnum(e, "M_FR_G_pixel") - fnum(b, "M_FR_G_pixel"),
        })
    arr_patch = np.asarray([fnum(r, "anomaly_patch_ratio_mean") for r in rows], dtype=np.float64)
    arr_bound = np.asarray([fnum(r, "boundary_to_area_ratio_mean") for r in rows], dtype=np.float64)
    if rows and np.isfinite(arr_patch).sum() >= 4:
        q25, q75 = np.nanpercentile(arr_patch, [25, 75])
        b75 = np.nanpercentile(arr_bound, 75)
        small = [fnum(r, "delta_E2_minus_B0_M_FR_P_AP") for r in rows if fnum(r, "anomaly_patch_ratio_mean") <= q25 and np.isfinite(fnum(r, "delta_E2_minus_B0_M_FR_P_AP"))]
        large = [fnum(r, "delta_E2_minus_B0_M_FR_P_AP") for r in rows if fnum(r, "anomaly_patch_ratio_mean") >= q75 and np.isfinite(fnum(r, "delta_E2_minus_B0_M_FR_P_AP"))]
        high_boundary = [fnum(r, "delta_E2_minus_B0_M_FR_P_AP") for r in rows if fnum(r, "boundary_to_area_ratio_mean") >= b75 and np.isfinite(fnum(r, "delta_E2_minus_B0_M_FR_P_AP"))]
        rows.append({
            "source": "ALL",
            "dataset": "ALL",
            "category": "QUARTILE_SUMMARY",
            "small_patch_q25": q25,
            "large_patch_q75": q75,
            "boundary_q75": b75,
            "small_sparse_mean_delta_P_AP": float(np.nanmean(small)) if small else float("nan"),
            "large_mean_delta_P_AP": float(np.nanmean(large)) if large else float("nan"),
            "small_minus_large_delta_P_AP": (float(np.nanmean(small)) - float(np.nanmean(large))) if small and large else float("nan"),
            "high_boundary_mean_delta_P_AP": float(np.nanmean(high_boundary)) if high_boundary else float("nan"),
        })
    return rows


def sota_gap(raw):
    refs = sota_reference()
    rows = []
    for r in raw:
        refs_here = [x for x in refs if x.get("comparison_type") == "DIRECT_COMPARABLE" and x.get("training_source") == r["source"] and x.get("target") == r["dataset"]]
        if not refs_here:
            rows.append({"source": r["source"], "dataset": r["dataset"], "variant": r["variant"], "reference": "no_direct_reference_available", "comparison_type": "REFERENCE_ONLY"})
            continue
        for ref in refs_here:
            paper = ref.get("paper", "")
            protocol = "M_FR" if paper == "MoECLIP" else "V_FR"
            row = {"source": r["source"], "dataset": r["dataset"], "variant": r["variant"], "reference": f"{paper} {ref.get('venue','')} {ref.get('year','')}", "protocol_used": protocol, "comparison_type": "DIRECT_COMPARABLE"}
            for cur, refk in (("P_AUROC", "P_AUROC"), ("P_AP", "P_AP"), ("G_pixel", "G_pixel")):
                cv = fnum(r, f"{protocol}_{cur}")
                rv = fnum(ref, refk)
                row[f"current_{protocol}_{cur}"] = cv
                row[f"reference_{cur}"] = rv
                row[f"Delta_{cur}"] = cv - rv if np.isfinite(cv) and np.isfinite(rv) else "NA"
            rows.append(row)
    return rows


def decide(raw, small_rows, res_rows):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    core = sorted(CORE_PAIRS)
    deltas_ap, deltas_g = [], []
    nondec = 0
    worst = float("inf")
    for source, dataset in core:
        b = by.get((source, dataset, "B0_HardNormal"))
        e = by.get((source, dataset, "E2_GlobalLocalZoomPyramid"))
        if not (b and e):
            continue
        d = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
        g = fnum(e, "M_FR_G_pixel") - fnum(b, "M_FR_G_pixel")
        deltas_ap.append(d)
        deltas_g.append(g)
        nondec += int(d >= -0.3)
        worst = min(worst, d)
    primary_pair = by.get(("Full MVTec", "VisA", "E2_GlobalLocalZoomPyramid")), by.get(("Full MVTec", "VisA", "B0_HardNormal"))
    if primary_pair[0] and primary_pair[1]:
        visa_ap = fnum(primary_pair[0], "M_FR_P_AP") - fnum(primary_pair[1], "M_FR_P_AP")
        visa_auc = fnum(primary_pair[0], "M_FR_P_AUROC") - fnum(primary_pair[1], "M_FR_P_AUROC")
        visa_pro = fnum(primary_pair[0], "M_FR_PRO") - fnum(primary_pair[1], "M_FR_PRO")
    else:
        visa_ap = visa_auc = visa_pro = float("nan")
    summary = next((r for r in small_rows if r.get("category") == "QUARTILE_SUMMARY"), {})
    small_minus_large = fnum(summary, "small_minus_large_delta_P_AP")
    general_ok = bool(len(deltas_ap) == 8 and nondec >= 6 and np.mean(deltas_ap) >= 0.8 and worst > -1.0)
    primary_ok = bool(visa_ap >= 3.0 and visa_auc >= -0.5 and visa_pro >= -1.0)
    small_ok = bool(small_minus_large >= 1.0)
    if primary_ok and general_ok and small_ok:
        outcome = "LOCAL_ZOOM_PYRAMID_RESOLVES_SMALL_SPARSE_LOCALIZATION"
    elif visa_ap >= 1.0 and not general_ok:
        outcome = "LOCAL_ZOOM_IS_VISA_SPECIFIC"
    elif general_ok and not small_ok:
        outcome = "LOCAL_VIEW_HAS_GENERIC_SIGNAL_NOT_BOTTLENECK_SPECIFIC"
    elif not np.isfinite(visa_ap) or visa_ap < 1.0:
        outcome = "LOCAL_ZOOM_PYRAMID_NO_MEANINGFUL_SIGNAL"
    else:
        outcome = "LOCAL_ZOOM_PYRAMID_NO_MEANINGFUL_SIGNAL"
    return outcome, {
        "core_evaluations": len(deltas_ap),
        "core_non_decrease_ge_minus_minus_0_3": nondec,
        "core_mean_M_FR_P_AP_gain": float(np.mean(deltas_ap)) if deltas_ap else float("nan"),
        "core_worst_M_FR_P_AP_gain": worst,
        "core_mean_M_FR_G_gain": float(np.mean(deltas_g)) if deltas_g else float("nan"),
        "mvtec_source_to_visa_delta_M_FR_P_AP": visa_ap,
        "mvtec_source_to_visa_delta_M_FR_P_AUROC": visa_auc,
        "mvtec_source_to_visa_delta_M_FR_PRO": visa_pro,
        "small_minus_large_delta_P_AP": small_minus_large,
        "primary_gate": primary_ok,
        "generalization_gate": general_ok,
        "small_sparse_gate": small_ok,
    }


def write_report(raw, outcome, decision, res_rows, gaps, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Parameter-Free Local Zoom Pyramid -- Final Full-Resolution Performance Gate",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "No training, tuning, Median baseline, score normalization, target statistics, or target calibration was used. E1/E2/E3 only re-encode fixed 2x2 local crops through the frozen final HardNormal model.",
        "",
        "## Core M-FR Results",
        "",
        "| Source | Target | Variant | M_FR P-AUROC | M_FR P-AP | M_FR P-F1max | M_FR PRO | M_FR G |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for source, target in sorted(CORE_PAIRS):
        for variant in VARIANTS:
            r = by.get((source, target, variant))
            if r:
                lines.append(f"| {source} | {target} | {variant} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} | {fnum(r,'M_FR_G_pixel'):.4f} |")
    lines += [
        "",
        "## Gate Summary",
        "",
        f"- MVTec-source -> VisA E2-B0: P-AP {decision['mvtec_source_to_visa_delta_M_FR_P_AP']:+.4f}, P-AUROC {decision['mvtec_source_to_visa_delta_M_FR_P_AUROC']:+.4f}, PRO {decision['mvtec_source_to_visa_delta_M_FR_PRO']:+.4f}.",
        f"- 8-core non-decrease count (Delta P-AP >= -0.3): {decision['core_non_decrease_ge_minus_minus_0_3']}/{decision['core_evaluations']}; mean P-AP gain {decision['core_mean_M_FR_P_AP_gain']:+.4f}; worst {decision['core_worst_M_FR_P_AP_gain']:+.4f}.",
        f"- Mean M-FR G gain: {decision['core_mean_M_FR_G_gain']:+.4f}.",
        f"- Small-sparse minus large-category P-AP gain: {decision['small_minus_large_delta_P_AP']:+.4f}.",
        "",
        "## Resolution Causality",
        "",
        "| Source | Target | E2-E3 M_FR P-AP | E2-E3 M_FR G |",
        "|---|---|---:|---:|",
    ]
    for r in res_rows:
        if (r["source"], r["dataset"]) in CORE_PAIRS:
            lines.append(f"| {r['source']} | {r['dataset']} | {fnum(r,'E2_minus_E3_M_FR_P_AP'):+.4f} | {fnum(r,'E2_minus_E3_M_FR_G_pixel'):+.4f} |")
    lines += [
        "",
        "## SOTA Gap",
        "",
        "MoECLIP is compared only under M_FR rows; VisualAD only under V_FR rows. Other pairs without matching public direct protocol are marked reference-only in `SOTA_GAP_FULL_RES.csv`.",
        "",
        "| Source | Target | Variant | Reference | Protocol | Delta P-AUROC | Delta P-AP | Delta G |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for g in gaps:
        if g.get("comparison_type") == "DIRECT_COMPARABLE" and g.get("variant") == "E2_GlobalLocalZoomPyramid":
            lines.append(f"| {g['source']} | {g['dataset']} | {g['variant']} | {g['reference']} | {g['protocol_used']} | {fnum(g,'Delta_P_AUROC'):+.4f} | {fnum(g,'Delta_P_AP'):+.4f} | {fnum(g,'Delta_G_pixel'):+.4f} |")
    lines += [
        "",
        "## Notes",
        "",
        "The 37x37 `PG_*` columns are retained only as legacy diagnostics. `M_FR_*` is the primary MoECLIP-compatible full-resolution protocol using bilinear `align_corners=True`; `V_FR_*` uses VisualAD-compatible `align_corners=False`.",
        f"",
        f"Runtime: {runtime:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all-ready", action="store_true", help="also run non-core READY holdouts if compute allows")
    args = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    raw, cats, loc, stitch, scale = evaluate_all(smoke=args.smoke, core_only=not args.all_ready)
    res = resolution_control(raw)
    small = small_sparse_analysis(cats)
    gaps = sota_gap(raw)
    outcome, decision = decide(raw, small, res)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "FULL_RES_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "SMALL_SPARSE_ANALYSIS.csv", small)
    write_csv(OUT / "RESOLUTION_CAUSAL_CONTROL.csv", res)
    write_csv(OUT / "STITCH_BOUNDARY_DIAGNOSTICS.csv", stitch)
    write_csv(OUT / "SCORE_SCALE_DIAGNOSTICS.csv", scale)
    write_csv(OUT / "LOCALIZATION_DIAGNOSTICS.csv", loc)
    write_csv(OUT / "SOTA_GAP_FULL_RES.csv", gaps)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision": decision, "runtime_sec": runtime, "core_only": not args.all_ready, "variants": VARIANTS})
    write_report(raw, outcome, decision, res, gaps, runtime)
    print(f"[LZP] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
