import argparse
import csv
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

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np, torch_resize
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_sgpr import evaluate_maps, fixed_quadrant_boxes, stitch_quadrant74
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, logits_from_z
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_qinr_query_internal_normality_refinement/formal")
VARIANTS = (
    "E0_FixedLRP_Baseline",
    "E1_NormalityOnly",
    "E2_HighRiskLocalReranking",
    "E3_ConservativeIntersection",
)
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "DTD-Synthetic"),
)
FULL_PAIRS = (
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
)
NORMAL_RATIO = 0.40
RISK_RATIO = 0.30
PCA_RANK = 8
SEED = 0


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fnum(row, key):
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


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


def safe_auc_np(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
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


def rank01(values):
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    if x.size <= 1:
        ranks[:] = 0.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    return ranks.reshape(values.shape)


def load_runtime(device):
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    model = load_components(str(CHECKPOINT), device)[0]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return preprocess, model


def model_infos(device):
    return [load_visa_source(device), load_mvtec_source()]


def scorer_z_grid(model_info, patch_np, device):
    if model_info["use_align"]:
        return aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).detach()
    z = norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    return torch.tensor(z, dtype=torch.float32, device=device).reshape(SIDE, SIDE, -1)


def score_from_scorer_z(model_info, z_grid, device):
    m_t = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v_t = torch.tensor(model_info["v1"], dtype=torch.float32, device=device)
    flat = z_grid.reshape(-1, z_grid.shape[-1])
    if model_info["use_align"]:
        out = logits_from_z(z_grid.unsqueeze(0), m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(z_grid.shape[:2])
    else:
        out = contrast_logits(flat, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(z_grid.shape[:2])
    return out.astype(np.float32)


@torch.no_grad()
def crop_patches(clip_model, preprocess, img, boxes, device):
    batch = torch.stack([preprocess(img.crop(tuple(b))) for b in boxes], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    return normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)


def stitch_feature74(features37):
    top = torch.cat([features37[0], features37[1]], dim=1)
    bot = torch.cat([features37[2], features37[3]], dim=1)
    return torch.cat([top, bot], dim=0).contiguous()


@torch.no_grad()
def low_rank_nonconformity(z74, q74):
    flat_z = F.normalize(z74.reshape(-1, z74.shape[-1]).float(), dim=-1)
    q = torch.tensor(np.asarray(q74, dtype=np.float32).reshape(-1), device=flat_z.device)
    n = flat_z.shape[0]
    k_normal = max(PCA_RANK + 2, int(round(NORMAL_RATIO * n)))
    k_normal = min(k_normal, n)
    normal_idx = torch.topk(q, k=k_normal, largest=False).indices
    z_n = flat_z[normal_idx]
    mu = z_n.mean(dim=0, keepdim=True)
    xc_n = z_n - mu
    rank = min(PCA_RANK, max(1, int(xc_n.shape[0] - 1)), int(xc_n.shape[1]))
    gen = torch.Generator(device=flat_z.device)
    gen.manual_seed(SEED)
    omega = torch.randn(xc_n.shape[1], rank + 4, generator=gen, device=flat_z.device, dtype=flat_z.dtype)
    y = xc_n @ omega
    q_basis, _ = torch.linalg.qr(y, mode="reduced")
    b = q_basis.transpose(0, 1) @ xc_n
    _u, _s, vh = torch.linalg.svd(b, full_matrices=False)
    basis = vh[:rank].transpose(0, 1).contiguous()
    xc = flat_z - mu
    recon = (xc @ basis) @ basis.transpose(0, 1)
    e = torch.linalg.norm(xc - recon, dim=-1)
    e_np = e.detach().cpu().numpy().reshape(SIDE * 2, SIDE * 2).astype(np.float32)
    nmask = np.zeros(n, dtype=bool)
    nmask[normal_idx.detach().cpu().numpy()] = True
    return e_np, nmask.reshape(SIDE * 2, SIDE * 2)


def high_risk_rerank(q74, e74):
    q_flat = np.asarray(q74, dtype=np.float32).reshape(-1)
    e_flat = np.asarray(e74, dtype=np.float32).reshape(-1)
    n = q_flat.size
    k = max(1, int(round(RISK_RATIO * n)))
    q_order = np.argsort(q_flat, kind="mergesort")
    high = q_order[-k:]
    low = q_order[:-k]
    out = np.empty(n, dtype=np.float32)
    if low.size:
        out[low] = np.linspace(0.0, 1.0 - RISK_RATIO, low.size, endpoint=False, dtype=np.float32)
    e_order_high = high[np.argsort(e_flat[high], kind="mergesort")]
    out[e_order_high] = np.linspace(1.0 - RISK_RATIO, 1.0, high.size, dtype=np.float32)
    return out.reshape(q74.shape)


def conservative_intersection(q74, e74):
    q_flat = np.asarray(q74, dtype=np.float32).reshape(-1)
    e_flat = np.asarray(e74, dtype=np.float32).reshape(-1)
    n = q_flat.size
    k = max(1, int(round(RISK_RATIO * n)))
    q_top = np.zeros(n, dtype=bool)
    e_top = np.zeros(n, dtype=bool)
    q_top[np.argsort(q_flat, kind="mergesort")[-k:]] = True
    e_top[np.argsort(e_flat, kind="mergesort")[-k:]] = True
    inter = q_top & e_top
    base_order = np.argsort(q_flat, kind="mergesort")
    non_inter = base_order[~inter[base_order]]
    inter_order = base_order[inter[base_order]]
    out = np.empty(n, dtype=np.float32)
    if non_inter.size:
        out[non_inter] = np.linspace(0.0, 1.0 - inter.mean(), non_inter.size, endpoint=False, dtype=np.float32)
    if inter_order.size:
        # Within the conservative intersection, use non-conformity as secondary evidence.
        inter_by_e = inter_order[np.argsort(e_flat[inter_order], kind="mergesort")]
        out[inter_by_e] = np.linspace(1.0 - inter.mean(), 1.0, inter_by_e.size, dtype=np.float32)
    return out.reshape(q74.shape)


def resize_gt74(mask):
    t = torch.tensor(mask.astype(np.float32)).view(1, 1, *mask.shape)
    return (F.interpolate(t, size=(SIDE * 2, SIDE * 2), mode="nearest").view(SIDE * 2, SIDE * 2).numpy() > 0.5)


def morphology(mask):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return {"num_anomaly_patches": 0, "anomaly_patch_ratio": 0.0, "thinness": 0.0}
    t = torch.tensor(m.astype(np.float32)).view(1, 1, *m.shape)
    pm = F.interpolate(t, size=(SIDE, SIDE), mode="nearest").view(SIDE, SIDE).numpy() > 0.5
    boundary = m & (~ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0))
    return {
        "num_anomaly_patches": int(pm.sum()),
        "anomaly_patch_ratio": float(pm.mean()),
        "thinness": float(boundary.sum()) / max(int(m.sum()), 1),
    }


def build_maps(model_info, clip_model, preprocess, target, dataset, refs, patch, kind, device, smoke):
    maps = {v: [] for v in VARIANTS}
    metas, diag_rows, image_rows = [], [], []
    n = len(refs)
    for pos, ref in enumerate(refs):
        if pos % 25 == 0:
            print(f"[QINR] {model_info['source']} -> {target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        if not path:
            raise ValueError(f"missing img_path for {model_info['source']}->{target} pos={pos}")
        mask = mask_np(item)
        img = Image.open(path).convert("RGB")
        boxes = fixed_quadrant_boxes(*img.size)
        crop_patch = crop_patches(clip_model, preprocess, img, boxes, device)
        crop_z = [scorer_z_grid(model_info, crop_patch[i], device) for i in range(4)]
        crop_q = [score_from_scorer_z(model_info, crop_z[i], device) for i in range(4)]
        local_z74 = stitch_feature74(crop_z)
        local_q74 = stitch_quadrant74(crop_q)
        global37 = score_from_scorer_z(model_info, scorer_z_grid(model_info, patch[pos], device), device)
        global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
        q0 = (0.5 * global74 + 0.5 * local_q74).astype(np.float32)
        e_raw, normal_set = low_rank_nonconformity(local_z74, q0)
        e_rank = rank01(e_raw).astype(np.float32)
        q_rank = rank01(q0).astype(np.float32)
        e2 = high_risk_rerank(q0, e_raw).astype(np.float32)
        e3 = conservative_intersection(q0, e_raw).astype(np.float32)
        maps["E0_FixedLRP_Baseline"].append(q0)
        maps["E1_NormalityOnly"].append(e_rank)
        maps["E2_HighRiskLocalReranking"].append(e2)
        maps["E3_ConservativeIntersection"].append(e3)
        metas.append({"category": cat, "label": int(label), "mask": mask, "img_path": path})
        gt74 = resize_gt74(mask)
        morph = morphology(mask)
        image_rows.append({"source": model_info["source"], "dataset": target, "category": cat, "image_index": pos, "label": int(label), **morph})
        if gt74.size and (int(label) == 1 or pos % (20 if not smoke else 1) == 0):
            qf, ef, yf, nsf = q0.reshape(-1), e_raw.reshape(-1), gt74.reshape(-1), normal_set.reshape(-1)
            normal = ~yf
            anomaly = yf
            fn_thr = np.percentile(qf[anomaly], 20) if anomaly.any() else float("nan")
            fp_thr = np.percentile(qf[normal], 95) if normal.any() else float("nan")
            fn = anomaly & np.isfinite(fn_thr) & (qf <= fn_thr)
            fp = normal & np.isfinite(fp_thr) & (qf >= fp_thr)
            tn = normal & np.isfinite(fp_thr) & (qf < fp_thr)
            diag_rows.append({
                "source": model_info["source"],
                "dataset": target,
                "category": cat,
                "image_index": pos,
                "label": int(label),
                "normal_set_contamination": float(np.mean(yf[nsf])) if nsf.any() else float("nan"),
                "e_normal_mean": float(np.mean(ef[normal])) if normal.any() else float("nan"),
                "e_normal_median": float(np.median(ef[normal])) if normal.any() else float("nan"),
                "e_anomaly_mean": float(np.mean(ef[anomaly])) if anomaly.any() else float("nan"),
                "e_anomaly_median": float(np.median(ef[anomaly])) if anomaly.any() else float("nan"),
                "e_auc": safe_auc_np(yf, ef),
                "baseline_fn_e_mean": float(np.mean(ef[fn])) if fn.any() else float("nan"),
                "baseline_anomaly_correct_e_mean": float(np.mean(ef[anomaly & ~fn])) if (anomaly & ~fn).any() else float("nan"),
                "baseline_fp_e_mean": float(np.mean(ef[fp])) if fp.any() else float("nan"),
                "baseline_tn_e_mean": float(np.mean(ef[tn])) if tn.any() else float("nan"),
                "fn_patch_count": int(fn.sum()),
                "fp_patch_count": int(fp.sum()),
                **morph,
            })
        img.close()
        if smoke and pos + 1 >= min(n, 24):
            break
    return maps, metas, diag_rows, image_rows


def selected_eval(source, target, refs, dataset, kind, maps, metas):
    return evaluate_maps(source, target, refs, dataset, kind, maps, metas, variants=VARIANTS)


def delta_summary(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in PILOT_PAIRS:
        base = by.get((source, target, "E0_FixedLRP_Baseline"))
        for variant in VARIANTS[1:]:
            cur = by.get((source, target, variant))
            if not (base and cur):
                continue
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(cur, "M_FR_P_AUROC") - fnum(base, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"),
                "Delta_M_FR_P_F1max": fnum(cur, "M_FR_P_F1max") - fnum(base, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(cur, "M_FR_PRO") - fnum(base, "M_FR_PRO"),
            })
    for variant in VARIANTS[1:]:
        vals = [fnum(r, "Delta_M_FR_P_AP") for r in rows if r["variant"] == variant]
        rows.append({
            "source": "SUMMARY",
            "dataset": "PILOT",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(vals),
            "median_delta_M_FR_P_AP": finite_median(vals),
            "improved": int(sum(v > 0 for v in vals if np.isfinite(v))),
            "non_drop_ge_minus_0p2": int(sum(v >= -0.2 for v in vals if np.isfinite(v))),
        })
    return rows


def diagnostic_summary(diag_rows, image_rows):
    rows = []
    morph_by = {(r["source"], r["dataset"], r["category"], r["image_index"]): r for r in image_rows}
    for source, target in PILOT_PAIRS:
        pair = [r for r in diag_rows if r["source"] == source and r["dataset"] == target]
        vals = {
            "normal_set_contamination": [fnum(r, "normal_set_contamination") for r in pair],
            "e_auc": [fnum(r, "e_auc") for r in pair],
            "fn_minus_correct_e": [fnum(r, "baseline_fn_e_mean") - fnum(r, "baseline_anomaly_correct_e_mean") for r in pair],
            "fp_minus_tn_e": [fnum(r, "baseline_fp_e_mean") - fnum(r, "baseline_tn_e_mean") for r in pair],
        }
        rows.append({
            "source": source,
            "dataset": target,
            "group": "all",
            "image_count": len(pair),
            "normal_set_contamination_mean": finite_mean(vals["normal_set_contamination"]),
            "e_auc_mean": finite_mean(vals["e_auc"]),
            "fn_minus_correct_e_mean": finite_mean(vals["fn_minus_correct_e"]),
            "fp_minus_tn_e_mean": finite_mean(vals["fp_minus_tn_e"]),
        })
        pos_morph = [r for r in image_rows if r["source"] == source and r["dataset"] == target and int(r["label"]) == 1 and fnum(r, "num_anomaly_patches") > 0]
        thin_thr = np.percentile([fnum(r, "thinness") for r in pos_morph], 75) if pos_morph else float("nan")
        large_thr = np.percentile([fnum(r, "num_anomaly_patches") for r in pos_morph], 75) if pos_morph else float("nan")
        for group, pred in (
            ("small", lambda m: fnum(m, "num_anomaly_patches") <= 8 and fnum(m, "num_anomaly_patches") > 0),
            ("thin", lambda m: np.isfinite(thin_thr) and fnum(m, "thinness") >= thin_thr and fnum(m, "num_anomaly_patches") > 0),
            ("large", lambda m: np.isfinite(large_thr) and fnum(m, "num_anomaly_patches") >= large_thr),
            ("thick", lambda m: np.isfinite(thin_thr) and fnum(m, "thinness") < thin_thr and fnum(m, "num_anomaly_patches") > 0),
        ):
            sub = []
            for r in pair:
                m = morph_by.get((r["source"], r["dataset"], r["category"], r["image_index"]))
                if m and pred(m):
                    sub.append(r)
            rows.append({
                "source": source,
                "dataset": target,
                "group": group,
                "image_count": len(sub),
                "normal_set_contamination_mean": finite_mean([fnum(r, "normal_set_contamination") for r in sub]),
                "e_auc_mean": finite_mean([fnum(r, "e_auc") for r in sub]),
                "fn_minus_correct_e_mean": finite_mean([fnum(r, "baseline_fn_e_mean") - fnum(r, "baseline_anomaly_correct_e_mean") for r in sub]),
                "fp_minus_tn_e_mean": finite_mean([fnum(r, "baseline_fp_e_mean") - fnum(r, "baseline_tn_e_mean") for r in sub]),
            })
    return rows


def decide(summary):
    stats_out = {}
    best_variant, best_mean = None, -1e9
    for v in VARIANTS[1:]:
        row = next((r for r in summary if r["source"] == "SUMMARY" and r["variant"] == v), None)
        if row and fnum(row, "mean_delta_M_FR_P_AP") > best_mean:
            best_mean = fnum(row, "mean_delta_M_FR_P_AP")
            best_variant = v
    vals = [r for r in summary if r.get("variant") == best_variant and r.get("source") != "SUMMARY"]
    deltas = [fnum(r, "Delta_M_FR_P_AP") for r in vals]
    improved = sum(v > 0 for v in deltas if np.isfinite(v))
    focus = [fnum(r, "Delta_M_FR_P_AP") for r in vals if (r["source"], r["dataset"]) in {("Full MVTec", "VisA"), ("Full VisA", "MPDD")}]
    catastrophic = any(v < -2.0 for v in deltas if np.isfinite(v))
    stats_out.update({
        "best_variant": best_variant,
        "mean_delta_P_AP": best_mean,
        "median_delta_P_AP": finite_median(deltas),
        "improved": improved,
        "non_drop_ge_minus_0p2": sum(v >= -0.2 for v in deltas if np.isfinite(v)),
        "focus_best_delta_P_AP": max(focus) if focus else float("nan"),
        "catastrophic_drop": catastrophic,
    })
    if best_mean >= 1.0 and improved >= 3 and (focus and max(focus) > 0.5) and not catastrophic:
        return "QUERY_INTERNAL_NORMALITY_SIGNIFICANTLY_IMPROVES_TRANSFER", stats_out
    if best_mean > 0 and improved >= 2:
        return "QUERY_NORMALITY_HAS_WEAK_BUT_REAL_VALUE", stats_out
    # E1 can be diagnostic even if reranking fails.
    e1 = next((r for r in summary if r["source"] == "SUMMARY" and r["variant"] == "E1_NormalityOnly"), None)
    if e1 and fnum(e1, "mean_delta_M_FR_P_AP") > -0.5:
        return "QUERY_NORMALITY_IS_DIAGNOSTIC_ONLY", stats_out
    return "QUERY_INTERNAL_NORMALITY_HAS_NO_COMPETITIVE_VALUE", stats_out


def write_report(raw, summary, diag_summary_rows, outcome, stats_out, runtime):
    lines = [
        "# Phase QINR: Query-Internal Normality Refinement",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR + fixed 2x2 Local Re-encoding Pyramid. No model parameters are trained or modified. Query normality is rebuilt independently inside each test image and discarded immediately.",
        "",
        "## Pilot Metrics",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    lines += [
        "",
        "## Delta vs E0",
        "",
        "| Source | Target | Variant | Delta P-AP |",
        "|---|---|---|---:|",
    ]
    for r in summary:
        if r["source"] != "SUMMARY":
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} |")
    lines += [
        "",
        "## Summary",
        "",
        "| Variant | mean Delta P-AP | median | improved | non-drop >= -0.2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["source"] == "SUMMARY":
            lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {r.get('improved','')} | {r.get('non_drop_ge_minus_0p2','')} |")
    lines += [
        "",
        "## Query Normality Diagnostics",
        "",
        "| Source | Target | Group | contamination | e AUROC | FN e - correct anomaly e | FP e - TN e |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in diag_summary_rows:
        if r["group"] in ("all", "small", "thin"):
            lines.append(f"| {r['source']} | {r['dataset']} | {r['group']} | {fnum(r,'normal_set_contamination_mean'):.4f} | {fnum(r,'e_auc_mean'):.4f} | {fnum(r,'fn_minus_correct_e_mean'):+.4f} | {fnum(r,'fp_minus_tn_e_mean'):+.4f} |")
    lines += [
        "",
        f"Decision stats: `{json.dumps(stats_out, default=json_default)}`",
        "",
        f"Runtime: {runtime:.1f}s.",
    ]
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[QINR] device={device}", flush=True)
    preprocess, clip_model = load_runtime(device)
    infos = {m["source"]: m for m in model_infos(device)}
    pairs = FULL_PAIRS if args.full else PILOT_PAIRS
    raw_rows, cat_rows, diag_rows, image_rows = [], [], [], []
    for source, target in pairs:
        cells, _audits = protocol_cells(source, device, args.smoke)
        matches = [c for c in cells if c[1] == target]
        if not matches:
            raise RuntimeError(f"No protocol cell for {source}->{target}")
        _cell_id, _target, dataset, refs, patch, kind = matches[0]
        if args.smoke:
            refs = refs[: min(len(refs), 24)] if not isinstance(refs, np.ndarray) else refs[: min(len(refs), 24)]
            patch = patch[: min(len(patch), 24)]
        print(f"[QINR] evaluate {source}->{target} n={len(refs)}", flush=True)
        maps, metas, dr, ir = build_maps(infos[source], clip_model, preprocess, target, dataset, refs, patch, kind, device, args.smoke)
        rr, cc = selected_eval(source, target, refs, dataset, kind, maps, metas)
        raw_rows.extend(rr)
        cat_rows.extend(cc)
        diag_rows.extend(dr)
        image_rows.extend(ir)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw_rows)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", cat_rows)
        write_csv(OUT / "QINR_DIAGNOSTIC.partial.csv", diag_rows)
    summary = delta_summary(raw_rows)
    diag_summary_rows = diagnostic_summary(diag_rows, image_rows)
    outcome, stats_out = decide(summary)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw_rows)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cat_rows)
    write_csv(OUT / "QINR_DIAGNOSTIC.csv", diag_rows)
    write_csv(OUT / "QINR_DIAGNOSTIC_SUMMARY.csv", diag_summary_rows)
    write_csv(OUT / "IMAGE_MORPHOLOGY.csv", image_rows)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", summary)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats_out,
        "runtime_sec": runtime,
        "normal_ratio": NORMAL_RATIO,
        "risk_ratio": RISK_RATIO,
        "pca_rank": PCA_RANK,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_tuning": False,
        "train_new_module": False,
        "full_stage": bool(args.full),
    })
    write_report(raw_rows, summary, diag_summary_rows, outcome, stats_out, runtime)
    print(f"[QINR] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
