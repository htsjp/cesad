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
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import protocol_cells
from experiments.phase_fova_field_of_view_amplification_audit import MAP_CACHE as FOVA_MAP_CACHE, slug as pair_slug
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np, torch_resize
from experiments.phase_sgpr import AUDIT_PAIRS, evaluate_maps
from experiments.phase_spsd_state_preserving_structural_diffusion import (
    DINO_BACKBONE,
    K_FIXED,
    dino_transform,
    extract_dino_patch,
    load_dino,
    safe_auc,
)
from experiments.phase_spatial_residual_refinement_standard_metric import fnum
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE


OUT = Path("results/phase_gscr_graph_spectral_contrast_residual/formal")
VARIANTS = ("E0_FixedLRP", "E1_GSCR_lambda0p5", "E2_GSCR_lambda1p0")
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "DTD-Synthetic"),
)


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


def load_pair(source, target, device, smoke):
    cells, _audits = protocol_cells(source, device, smoke)
    for _cell_id, dst, dataset, refs, _patch, kind in cells:
        if dst == target:
            return dataset, refs, kind
    raise KeyError(f"missing target cell {source}->{target}")


def load_e0_maps(source, target, n):
    path = FOVA_MAP_CACHE / f"{pair_slug(source, target)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Fixed LRP/FOVA map cache for {source}->{target}: {path}")
    data = np.load(path)
    q = np.asarray(data["q_lrp"][:n], dtype=np.float32)
    if q.ndim != 3 or q.shape[1:] != (SIDE * 2, SIDE * 2):
        raise ValueError(f"unexpected q_lrp shape {q.shape} in {path}")
    return q


def morphology(mask):
    mask = np.asarray(mask, dtype=bool)
    pix = int(mask.sum())
    h, w = mask.shape
    if pix == 0:
        return {"num_anomaly_patches": 0, "thinness": 0.0, "label": 0}
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = mask & (~eroded)
    gm = torch.tensor(mask.astype(np.float32)).view(1, 1, h, w)
    patch = F.interpolate(gm, size=(SIDE, SIDE), mode="nearest").view(SIDE, SIDE).numpy() > 0.5
    return {"num_anomaly_patches": int(patch.sum()), "thinness": float(boundary.sum() / max(pix, 1)), "label": 1}


def assign_groups(rows):
    pos = [r for r in rows if int(r["label"]) == 1 and fnum(r, "num_anomaly_patches") > 0]
    patch_vals = [fnum(r, "num_anomaly_patches") for r in pos]
    thin_vals = [fnum(r, "thinness") for r in pos]
    small_thr = float(np.percentile(patch_vals, 25)) if patch_vals else float("nan")
    large_thr = float(np.percentile(patch_vals, 75)) if patch_vals else float("nan")
    thin_thr = float(np.percentile(thin_vals, 75)) if thin_vals else float("nan")
    for r in rows:
        p = fnum(r, "num_anomaly_patches")
        t = fnum(r, "thinness")
        r["group_small"] = bool(np.isfinite(small_thr) and p > 0 and p <= small_thr)
        r["group_large"] = bool(np.isfinite(large_thr) and p >= large_thr)
        r["group_thin"] = bool(np.isfinite(thin_thr) and p > 0 and t >= thin_thr)
        r["group_thick"] = bool(np.isfinite(thin_thr) and p > 0 and t < thin_thr)


def pair_meta(source, target, dataset, refs, kind):
    metas, rows = [], []
    for pos, ref in enumerate(refs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        mask = mask_np(item)
        morph = morphology(mask)
        row = {"source": source, "dataset": target, "category": cat, "image_index": pos, "label": int(label), "img_path": path, **morph}
        rows.append(row)
        metas.append({"category": cat, "label": int(label), "mask": mask, "img_path": path})
    assign_groups(rows)
    for meta, row in zip(metas, rows):
        meta.update({k: row[k] for k in ("group_small", "group_large", "group_thin", "group_thick", "num_anomaly_patches", "thinness")})
    return metas, rows


def topk_graph_lowpass(q74, feat, k=K_FIXED):
    n = feat.shape[0]
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"DINO patch count is not square: {n}")
    q = torch_resize(q74, (side, side), align_corners=True).reshape(-1).astype(np.float32)
    x = feat.astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    aff = x @ x.T
    np.fill_diagonal(aff, 1.0)
    kk = min(int(k), max(n, 1))
    top = np.argpartition(-aff, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(n)[:, None]
    w = np.maximum(aff[row, top], 0.0)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    q_low = np.sum(w * q[top], axis=1).astype(np.float32)
    residual = (q - q_low).astype(np.float32)
    r_pos = np.maximum(residual, 0.0)
    out = {
        "q_low74": torch_resize(q_low.reshape(side, side), (SIDE * 2, SIDE * 2), align_corners=True),
        "residual74": torch_resize(residual.reshape(side, side), (SIDE * 2, SIDE * 2), align_corners=True),
        "r_pos74": torch_resize(r_pos.reshape(side, side), (SIDE * 2, SIDE * 2), align_corners=True),
        "dino_grid": f"{side}x{side}",
        "mean_topk_affinity": float(np.mean(w)),
        "mean_residual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "mean_positive_residual": float(np.mean(r_pos)),
        "positive_residual_fraction": float(np.mean(r_pos > 0)),
    }
    return out


def residual_stats(source, target, meta, residual74, r_pos74):
    gt = torch.tensor(meta["mask"].astype(np.float32)).view(1, 1, *meta["mask"].shape)
    gm = F.interpolate(gt, size=residual74.shape, mode="nearest").view(*residual74.shape).numpy() > 0.5
    y = gm.reshape(-1).astype(np.uint8)
    r = residual74.reshape(-1).astype(np.float32)
    rp = r_pos74.reshape(-1).astype(np.float32)
    normal = y == 0
    anomaly = y == 1
    row = {
        "source": source,
        "dataset": target,
        "category": meta["category"],
        "image_index": meta.get("image_index", ""),
        "label": int(meta["label"]),
        "group_small": bool(meta.get("group_small", False)),
        "group_large": bool(meta.get("group_large", False)),
        "group_thin": bool(meta.get("group_thin", False)),
        "group_thick": bool(meta.get("group_thick", False)),
        "residual_auc_anomaly_vs_normal": safe_auc(y, r) if anomaly.any() and normal.any() else float("nan"),
        "rpos_auc_anomaly_vs_normal": safe_auc(y, rp) if anomaly.any() and normal.any() else float("nan"),
        "anomaly_residual_mean": float(np.mean(r[anomaly])) if anomaly.any() else float("nan"),
        "normal_residual_mean": float(np.mean(r[normal])) if normal.any() else float("nan"),
        "anomaly_rpos_mean": float(np.mean(rp[anomaly])) if anomaly.any() else float("nan"),
        "normal_rpos_mean": float(np.mean(rp[normal])) if normal.any() else float("nan"),
    }
    return row


def build_maps(source, target, e0_maps, metas, dino_model, tfm, device):
    maps = {v: [] for v in VARIANTS}
    audit, residual_rows = [], []
    for i, (q, meta) in enumerate(zip(e0_maps, metas)):
        if i % 50 == 0:
            print(f"[GSCR] {source}->{target} image {i}/{len(metas)}", flush=True)
        feat = extract_dino_patch(dino_model, tfm, meta["img_path"], device)
        spec = topk_graph_lowpass(q, feat, K_FIXED)
        rpos = spec["r_pos74"]
        e1 = q + 0.5 * rpos
        e2 = q + 1.0 * rpos
        maps["E0_FixedLRP"].append(q.astype(np.float32))
        maps["E1_GSCR_lambda0p5"].append(e1.astype(np.float32))
        maps["E2_GSCR_lambda1p0"].append(e2.astype(np.float32))
        audit.append({"source": source, "dataset": target, "image_index": i, "K": K_FIXED, **{k: v for k, v in spec.items() if not isinstance(v, np.ndarray)}})
        residual_rows.append(residual_stats(source, target, {**meta, "image_index": i}, spec["residual74"], spec["r_pos74"]))
    return maps, audit, residual_rows


def summarize_metrics(raw, pairs):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in VARIANTS[1:]:
        vals, improved, nondrop = [], 0, 0
        for source, target in pairs:
            b = by.get((source, target, VARIANTS[0]))
            e = by.get((source, target, variant))
            if not (b and e):
                continue
            d = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            vals.append(d)
            improved += int(np.isfinite(d) and d > 0)
            nondrop += int(np.isfinite(d) and d >= -0.2)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "E_minus_E0_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "E_minus_E0_M_FR_P_AP": d,
                "E_minus_E0_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "E_minus_E0_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
        rows.append({
            "source": "SUMMARY",
            "dataset": "PILOT",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(vals),
            "median_delta_M_FR_P_AP": finite_median(vals),
            "improved": improved,
            "non_drop_ge_minus_0_2": nondrop,
            "total": len(vals),
        })
    return rows


def category_morphology_diagnostics(cat_rows, meta_rows):
    meta_by_cat = {}
    for r in meta_rows:
        key = (r["source"], r["dataset"], r["category"])
        d = meta_by_cat.setdefault(key, {g: [] for g in ("small", "large", "thin", "thick")})
        for g in ("small", "large", "thin", "thick"):
            d[g].append(str(r.get(f"group_{g}", False)).lower() == "true")
    cat_group = {}
    for key, vals in meta_by_cat.items():
        cat_group[key] = {g: (np.mean(v) >= 0.5 if v else False) for g, v in vals.items()}
    by = {(r["source"], r["dataset"], r["category"], r["variant"], r.get("protocol")): r for r in cat_rows}
    rows = []
    for key, groups in cat_group.items():
        source, target, cat = key
        base = by.get((source, target, cat, VARIANTS[0], "M"))
        if not base:
            continue
        for variant in VARIANTS[1:]:
            e = by.get((source, target, cat, variant, "M"))
            if not e:
                continue
            row = {"source": source, "dataset": target, "category": cat, "variant": variant}
            row.update({f"group_{g}": bool(v) for g, v in groups.items()})
            row["Delta_M_FR_P_AP"] = fnum(e, "M_FR_P_AP") - fnum(base, "M_FR_P_AP")
            row["Delta_M_FR_P_F1max"] = fnum(e, "M_FR_P_F1max") - fnum(base, "M_FR_P_F1max")
            row["Delta_M_FR_PRO"] = fnum(e, "M_FR_PRO") - fnum(base, "M_FR_PRO")
            rows.append(row)
    for g in ("small", "large", "thin", "thick"):
        for variant in VARIANTS[1:]:
            vals = [fnum(r, "Delta_M_FR_P_AP") for r in rows if r["variant"] == variant and r.get(f"group_{g}") is True]
            rows.append({"source": "SUMMARY", "dataset": "PILOT", "category": "ALL", "variant": variant, "group": g, "mean_delta_M_FR_P_AP": finite_mean(vals), "median_delta_M_FR_P_AP": finite_median(vals), "improved": sum(1 for v in vals if np.isfinite(v) and v > 0), "total": sum(1 for v in vals if np.isfinite(v))})
    return rows


def residual_summary(residual_rows):
    rows = list(residual_rows)
    for source, target in sorted({(r["source"], r["dataset"]) for r in residual_rows}):
        part = [r for r in residual_rows if r["source"] == source and r["dataset"] == target and int(r["label"]) == 1]
        for group in ("all_anomaly", "small", "large", "thin", "thick"):
            if group == "all_anomaly":
                gp = part
            else:
                gp = [r for r in part if str(r.get(f"group_{group}", False)).lower() == "true"]
            rows.append({
                "source": source,
                "dataset": target,
                "category": "SUMMARY",
                "image_index": "",
                "label": 1,
                "group": group,
                "mean_residual_auc_anomaly_vs_normal": finite_mean([fnum(r, "residual_auc_anomaly_vs_normal") for r in gp]),
                "mean_rpos_auc_anomaly_vs_normal": finite_mean([fnum(r, "rpos_auc_anomaly_vs_normal") for r in gp]),
                "mean_anomaly_minus_normal_rpos": finite_mean([fnum(r, "anomaly_rpos_mean") - fnum(r, "normal_rpos_mean") for r in gp]),
                "images": len(gp),
            })
    return rows


def best_variant(summary):
    candidates = [r for r in summary if r.get("source") == "SUMMARY"]
    if not candidates:
        return None, float("nan"), 0, 0, 0
    best = max(candidates, key=lambda r: fnum(r, "mean_delta_M_FR_P_AP"))
    return best["variant"], fnum(best, "mean_delta_M_FR_P_AP"), int(float(best.get("improved", 0))), int(float(best.get("non_drop_ge_minus_0_2", 0))), int(float(best.get("total", 0)))


def decide(summary, morph_rows):
    best, mean_delta, improved, nondrop, total = best_variant(summary)
    small_rows = [r for r in morph_rows if r.get("source") == "SUMMARY" and r.get("variant") == best and r.get("group") == "small"]
    thin_rows = [r for r in morph_rows if r.get("source") == "SUMMARY" and r.get("variant") == best and r.get("group") == "thin"]
    large_rows = [r for r in morph_rows if r.get("source") == "SUMMARY" and r.get("variant") == best and r.get("group") == "large"]
    thick_rows = [r for r in morph_rows if r.get("source") == "SUMMARY" and r.get("variant") == best and r.get("group") == "thick"]
    small_gain = fnum(small_rows[0], "mean_delta_M_FR_P_AP") if small_rows else float("nan")
    thin_gain = fnum(thin_rows[0], "mean_delta_M_FR_P_AP") if thin_rows else float("nan")
    large_gain = fnum(large_rows[0], "mean_delta_M_FR_P_AP") if large_rows else float("nan")
    thick_gain = fnum(thick_rows[0], "mean_delta_M_FR_P_AP") if thick_rows else float("nan")
    fine_vals = [v for v in (small_gain, thin_gain) if np.isfinite(v)]
    coarse_vals = [v for v in (large_gain, thick_gain) if np.isfinite(v)]
    fine_gain = float(np.mean(fine_vals)) if fine_vals else float("nan")
    coarse_gain = float(np.mean(coarse_vals)) if coarse_vals else float("nan")
    catastrophic = nondrop < total - 1
    stats = {
        "best_variant": best,
        "mean_delta_P_AP": mean_delta,
        "improved": improved,
        "non_drop_ge_minus_0_2": nondrop,
        "total": total,
        "small_mean_delta_P_AP": small_gain,
        "thin_mean_delta_P_AP": thin_gain,
        "large_mean_delta_P_AP": large_gain,
        "thick_mean_delta_P_AP": thick_gain,
        "fine_minus_coarse_delta_P_AP": float(fine_gain - coarse_gain) if np.isfinite(fine_gain) and np.isfinite(coarse_gain) else float("nan"),
        "pilot_gate_pass": False,
    }
    gate = mean_delta >= 1.0 and improved >= 3 and np.isfinite(fine_gain) and fine_gain > 0 and np.isfinite(thin_gain) and thin_gain > 0 and not catastrophic
    stats["pilot_gate_pass"] = bool(gate)
    if gate and mean_delta >= 1.5:
        return "GRAPH_HIGH_PASS_SIGNIFICANTLY_IMPROVES_FINE_GRAINED_TRANSFER", stats
    if gate:
        return "GRAPH_HIGH_PASS_HAS_WEAK_BUT_REAL_VALUE", stats
    if np.isfinite(fine_gain) and fine_gain > 0 and improved >= 2:
        return "GRAPH_HIGH_PASS_IS_DIAGNOSTIC_ONLY", stats
    return "GRAPH_SPECTRAL_RESIDUAL_HAS_NO_COMPETITIVE_VALUE", stats


def write_report(raw, summary, morph_rows, residual_rows, outcome, stats, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Phase GSCR: Graph-Spectral Contrast Residual",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        f"NO-TRAIN / NO-TUNING. Baseline is Canonical Two-State + NTR + Fixed LRP. DINO graph is the frozen `{DINO_BACKBONE}` affinity from SPSD with K={K_FIXED}.",
        "",
        "GSCR tests graph high-pass positive residual only: `r_pos = ReLU(q - Ahat q)`, with fixed lambda 0.5 and 1.0.",
        "",
        "## Pilot Metrics",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for source, target in PILOT_PAIRS:
        for variant in VARIANTS:
            r = by.get((source, target, variant))
            if r:
                lines.append(f"| {source} | {target} | {variant} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    lines.extend(["", "## P-AP Delta", "", "| Source | Target | Variant | Delta P-AP | Delta PRO |", "|---|---|---|---:|---:|"])
    for r in summary:
        if r.get("source") == "SUMMARY":
            continue
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'E_minus_E0_M_FR_P_AP'):+.4f} | {fnum(r,'E_minus_E0_M_FR_PRO'):+.4f} |")
    lines.extend(["", "## Summary", "", "| Variant | mean Delta P-AP | median | improved | non-drop |", "|---|---:|---:|---:|---:|"])
    for r in summary:
        if r.get("source") == "SUMMARY":
            lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {r.get('improved')}/{r.get('total')} | {r.get('non_drop_ge_minus_0_2')}/{r.get('total')} |")
    lines.extend(["", "## Morphology Delta", "", "| Group | Variant | mean Delta P-AP | improved |", "|---|---|---:|---:|"])
    for r in morph_rows:
        if r.get("source") == "SUMMARY":
            lines.append(f"| {r['group']} | {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {r.get('improved')}/{r.get('total')} |")
    lines.extend(["", "## Residual Discrimination", "", "| Source | Target | Group | residual AUROC | r_pos AUROC | anomaly-normal r_pos |", "|---|---|---|---:|---:|---:|"])
    for r in residual_rows:
        if r.get("category") == "SUMMARY":
            lines.append(f"| {r['source']} | {r['dataset']} | {r['group']} | {fnum(r,'mean_residual_auc_anomaly_vs_normal'):.4f} | {fnum(r,'mean_rpos_auc_anomaly_vs_normal'):.4f} | {fnum(r,'mean_anomaly_minus_normal_rpos'):+.6f} |")
    lines.extend(["", f"Decision stats: `{json.dumps(stats, default=json_default)}`", "", f"Runtime: {runtime:.1f}s."])
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GSCR] device={device}", flush=True)
    pairs = list(PILOT_PAIRS)
    dino_model = load_dino(device)
    tfm = dino_transform()
    raw, cats, meta_rows_all, audit_rows, residual_rows = [], [], [], [], []
    for source, target in pairs:
        dataset, refs, kind = load_pair(source, target, device, args.smoke)
        n = min(24, len(refs)) if args.smoke else len(refs)
        refs_use = refs[:n]
        print(f"[GSCR] evaluate {source}->{target} n={n}", flush=True)
        metas, meta_rows = pair_meta(source, target, dataset, refs_use, kind)
        e0 = load_e0_maps(source, target, n)
        maps, audit, resid = build_maps(source, target, e0, metas, dino_model, tfm, device)
        rr, cc = evaluate_maps(source, target, refs_use, dataset, kind, maps, metas, variants=VARIANTS)
        raw.extend(rr)
        cats.extend(cc)
        meta_rows_all.extend(meta_rows)
        audit_rows.extend(audit)
        residual_rows.extend(resid)
        if not args.smoke:
            write_csv(OUT / "RAW_METRICS.partial.csv", raw)
            write_csv(OUT / "GRAPH_RESIDUAL_AUDIT.partial.csv", audit_rows)
            write_csv(OUT / "RESIDUAL_DISCRIMINATION.partial.csv", residual_rows)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = summarize_metrics(raw, pairs)
    morph = category_morphology_diagnostics(cats, meta_rows_all)
    residual_sum = residual_summary(residual_rows)
    outcome, stats = decide(summary, morph)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", summary)
    write_csv(OUT / "MORPHOLOGY_DIAGNOSTICS.csv", morph)
    write_csv(OUT / "MORPHOLOGY_GROUPS_USED.csv", meta_rows_all)
    write_csv(OUT / "GRAPH_RESIDUAL_AUDIT.csv", audit_rows)
    write_csv(OUT / "RESIDUAL_DISCRIMINATION.csv", residual_sum)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "no_train": True,
        "no_target_tuning": True,
        "baseline": "Canonical Two-State + NTR + Fixed LRP",
        "dino_backbone": DINO_BACKBONE,
        "K": K_FIXED,
        "variants": VARIANTS,
        "pilot_pairs": PILOT_PAIRS,
    })
    write_report(raw, summary, morph, residual_sum, outcome, stats, runtime)
    print(f"[GSCR] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
