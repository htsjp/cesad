import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_bottleneck_attribution import (
    OUT as BOTTLENECK_OUT,
    category_failure_table,
    correlation_analysis,
    decide,
    read_csv as read_csv_bottleneck,
)
from experiments.phase_final_cross_dataset_benchmark import (
    OUT as FINAL_OUT,
    VARIANTS,
    load_mvtec_source,
    load_visa_source,
    protocol_cells,
    score_cell,
    sota_reference,
)
from experiments.phase_spatial_residual_refinement_standard_metric import fnum
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE
from experiments.standard_zsad_metrics import PRO_FPR_LIMIT, PRO_STEPS, f1max, safe_ap, safe_auc


OUT = Path("results/phase_full_resolution_benchmark_revalidation/formal")
PROTOCOLS = {
    "M": {"name": "MoECLIP-compatible", "align_corners": True},
    "V": {"name": "VisualAD-compatible", "align_corners": False},
}
METRICS = ("P-AUROC", "P-AP", "P-F1max", "PRO", "AUPRO", "G_pixel")
CORE_FULL_RES_PAIRS = {
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "CVC-ClinicDB"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "CVC-ClinicDB"),
    ("Full MVTec", "MPDD"),
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def item_at(dataset, kind, ref, pos):
    return dataset[int(ref)] if kind == "source_dataset" else dataset[pos]


def item_meta(item, kind, ref, target):
    if kind == "source_dataset":
        return str(item.get("cls_name", target)), int(item.get("anomaly", 0))
    row = dict(ref)
    return str(row.get("category", item.get("cls_name", target))), int(row.get("label", item.get("anomaly", 0)))


def mask_np(item):
    m = item["img_mask"].detach().cpu().numpy()
    return (np.squeeze(m) > 0.5).astype(np.uint8)


def resize_score(score, shape, align_corners):
    x = torch.tensor(np.asarray(score, dtype=np.float32), dtype=torch.float32).view(1, 1, SIDE, SIDE)
    y = F.interpolate(x, size=tuple(shape), mode="bilinear", align_corners=align_corners)
    return y.view(*shape).numpy().astype(np.float32)


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
        "PRO": fast_aupro_sorted(masks, scores),
    }
    out["AUPRO"] = out["PRO"]
    out["G_pixel"] = 0.5 * (out["P-AUROC"] + out["P-AP"]) if np.isfinite(out["P-AUROC"]) and np.isfinite(out["P-AP"]) else float("nan")
    return out


def fast_aupro_sorted(mask_list, score_list, fpr_limit=PRO_FPR_LIMIT, steps=PRO_STEPS):
    masks = [np.asarray(m, dtype=bool) for m in mask_list]
    scores = [np.asarray(s, dtype=np.float32) for s in score_list]
    if not masks or not scores:
        return float("nan")
    all_scores = np.concatenate([s.reshape(-1) for s in scores])
    if all_scores.size == 0:
        return float("nan")
    mn, mx = float(np.min(all_scores)), float(np.max(all_scores))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return float("nan")
    thresholds = np.linspace(mn, mx, int(steps), dtype=np.float32)

    neg_chunks = []
    region_chunks = []
    region_areas = []
    structure = np.ones((3, 3), dtype=np.int8)
    for mask, score in zip(masks, scores):
        if mask.shape != score.shape:
            raise ValueError(f"mask/score shape mismatch: {mask.shape} vs {score.shape}")
        neg = np.sort(score[~mask].reshape(-1))
        if neg.size:
            neg_chunks.append(neg)
        labels, n = ndimage.label(mask, structure=structure)
        for rid in range(1, n + 1):
            vals = np.sort(score[labels == rid].reshape(-1))
            if vals.size:
                region_chunks.append(vals)
                region_areas.append(float(vals.size))
    neg_total = sum(x.size for x in neg_chunks)
    if neg_total == 0 or not region_chunks:
        return float("nan")

    fprs, pros = [], []
    for th in thresholds:
        fp = 0
        for neg in neg_chunks:
            fp += int(neg.size - np.searchsorted(neg, th, side="right"))
        fprs.append(float(fp) / float(max(neg_total, 1)))
        overlaps = []
        for vals, area in zip(region_chunks, region_areas):
            overlaps.append((vals.size - np.searchsorted(vals, th, side="right")) / area)
        pros.append(float(np.mean(overlaps)) if overlaps else float("nan"))
    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
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


def macro_from_category_rows(rows):
    out = {}
    for met in ("I-AUROC", "I-AP", "I-F1max", *METRICS):
        vals = np.asarray([fnum(r, met) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[met] = float(vals.mean()) if vals.size else float("nan")
    return out


def prefix_metrics(row, prefix, src):
    for met in ("I-AUROC", "I-AP", "I-F1max", *METRICS):
        row[f"{prefix}_{met.replace('-', '_')}"] = src.get(met, float("nan"))


def evaluate_cell(model, cell, device, smoke):
    cell_id, target, dataset, idx_or_rows, patch, kind = cell
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    refs = idx_or_rows[:n]
    print(f"[FRReval] {model['source']} -> {target} n={n}", flush=True)
    image_scores, maps = score_cell(model, patch, device, smoke)
    meta = []
    for pos, ref in enumerate(refs):
        item = item_at(dataset, kind, ref, pos)
        cat, label = item_meta(item, kind, ref, target)
        gt = mask_np(item) if "img_mask" in item else None
        meta.append((cat, label, gt))

    raw_rows, cat_rows = [], []
    for variant in VARIANTS:
        protocol_macro = {}
        for pkey, pcfg in PROTOCOLS.items():
            by_cat = {}
            for pos, (cat, label, gt) in enumerate(meta):
                d = by_cat.setdefault(cat, {"labels": [], "image_scores": [], "masks": [], "scores": []})
                d["labels"].append(label)
                d["image_scores"].append(float(image_scores[variant][pos]))
                if gt is not None:
                    d["masks"].append(gt)
                    d["scores"].append(resize_score(maps[variant][pos], gt.shape, pcfg["align_corners"]))
            per = []
            for cat, vals in sorted(by_cat.items()):
                metrics = summarize_category(vals["labels"], vals["image_scores"], vals["masks"], vals["scores"])
                row = {
                    "source": model["source"],
                    "dataset": target,
                    "category": cat,
                    "variant": variant,
                    "protocol": pkey,
                    "protocol_name": pcfg["name"],
                    "aggregation": "CATEGORY",
                    "mask_images": len(vals["masks"]),
                    **metrics,
                }
                cat_rows.append(row)
                per.append(row)
            protocol_macro[pkey] = macro_from_category_rows(per)
        raw = {
            "source": model["source"],
            "dataset": target,
            "variant": variant,
            "cell": cell_id,
            "FR_aggregation": "CATEGORY_MACRO",
            "score_source_resolution": f"{SIDE}x{SIDE}",
            "gt_resolution": "transformed_image_mask_shape",
        }
        for pkey in PROTOCOLS:
            prefix_metrics(raw, f"{pkey}_CATEGORY_MACRO_FR", protocol_macro[pkey])
        raw_rows.append(raw)
    return raw_rows, cat_rows


def run_full_res(smoke, core_only=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_visa_source(device), load_mvtec_source()]
    partial_raw = OUT / "FULL_RES_RAW_METRICS.partial.csv"
    partial_cat = OUT / "FULL_RES_CATEGORY_METRICS.partial.csv"
    raw = read_csv(partial_raw) if partial_raw.exists() and not smoke else []
    cats = read_csv(partial_cat) if partial_cat.exists() and not smoke else []
    done = {(r["source"], r["dataset"]) for r in raw if r.get("variant") == "M2_Ours_HardNormal_Median" and r.get("FR_status", "OK") == "OK"}
    for model in models:
        cells, _audits = protocol_cells(model["source"], device, smoke)
        for cell in cells:
            target = cell[1]
            key = (model["source"], target)
            if core_only and key not in CORE_FULL_RES_PAIRS:
                continue
            if key in done:
                print(f"[FRReval] skip completed {model['source']} -> {target}", flush=True)
                continue
            rr, cc = evaluate_cell(model, cell, device, smoke)
            for r in rr:
                r["FR_status"] = "OK"
            for r in cc:
                r["FR_status"] = "OK"
            raw.extend(rr)
            cats.extend(cc)
            write_csv(OUT / "FULL_RES_RAW_METRICS.partial.csv", raw)
            write_csv(OUT / "FULL_RES_CATEGORY_METRICS.partial.csv", cats)
    return raw, cats


def add_missing_full_rows(full_rows, core_only=True):
    if not core_only:
        return full_rows
    patch_rows = read_csv(FINAL_OUT / "RAW_METRICS.csv")
    present = {(r["source"], r["dataset"], r["variant"]) for r in full_rows}
    rows = list(full_rows)
    for p in patch_rows:
        key = (p["source"], p["dataset"], p["variant"])
        if key in present or (p["source"], p["dataset"]) in CORE_FULL_RES_PAIRS:
            continue
        row = {
            "source": p["source"],
            "dataset": p["dataset"],
            "variant": p["variant"],
            "cell": p.get("cell", ""),
            "FR_aggregation": "CATEGORY_MACRO",
            "FR_status": "NOT_RUN_FULL_RES_COMPUTE_LIMIT",
            "score_source_resolution": f"{SIDE}x{SIDE}",
            "gt_resolution": "transformed_image_mask_shape",
        }
        for pkey in PROTOCOLS:
            for met in ("I_AUROC", "I_AP", "I_F1max", "P_AUROC", "P_AP", "P_F1max", "PRO", "AUPRO", "G_pixel"):
                row[f"{pkey}_CATEGORY_MACRO_FR_{met}"] = "NA"
        rows.append(row)
    return rows


def patch_grid_vs_full(full_rows):
    pg = read_csv(FINAL_OUT / "RAW_METRICS.csv")
    by_pg = {(r["source"], r["dataset"], r["variant"]): r for r in pg}
    rows = []
    for r in full_rows:
        base = by_pg.get((r["source"], r["dataset"], r["variant"]), {})
        row = {"source": r["source"], "dataset": r["dataset"], "variant": r["variant"]}
        for met in ("P-AUROC", "P-AP", "P-F1max", "PRO", "G_pixel"):
            pg_key = f"PG_{met.replace('-', '_')}"
            row[pg_key] = base.get(met, "NA")
            for pkey in PROTOCOLS:
                fr_key = f"{pkey}_CATEGORY_MACRO_FR_{met.replace('-', '_')}"
                row[fr_key] = r.get(fr_key, "NA")
                row[f"Delta_{fr_key}_minus_{pg_key}"] = fnum(r, fr_key) - fnum(base, met)
        rows.append(row)
    return rows


def protocol_sensitivity(full_rows):
    rows = []
    diffs_by_metric = {m: [] for m in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel")}
    for r in full_rows:
        row = {"source": r["source"], "dataset": r["dataset"], "variant": r["variant"]}
        for met in diffs_by_metric:
            d = fnum(r, f"M_CATEGORY_MACRO_FR_{met}") - fnum(r, f"V_CATEGORY_MACRO_FR_{met}")
            row[f"Delta_M_minus_V_{met}"] = d
            if np.isfinite(d):
                diffs_by_metric[met].append(d)
        rows.append(row)
    summary = []
    for met, vals in diffs_by_metric.items():
        a = np.asarray(vals, dtype=np.float64)
        summary.append({
            "source": "ALL",
            "dataset": "ALL",
            "variant": "ALL",
            "metric": met,
            "mean_delta": float(a.mean()) if a.size else float("nan"),
            "median_delta": float(np.median(a)) if a.size else float("nan"),
            "mean_abs_delta": float(np.mean(np.abs(a))) if a.size else float("nan"),
            "max_abs_delta": float(np.max(np.abs(a))) if a.size else float("nan"),
        })
    return rows + summary


def delta_summary(full_rows, comparison):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in full_rows}
    pairs = sorted({(r["source"], r["dataset"]) for r in full_rows})
    if comparison == "Median_vs_HardNormal":
        a_name, b_name = "M2_Ours_HardNormal_Median", "M1_HardNormal"
    else:
        a_name, b_name = "M1_HardNormal", "M0_StateAligned"
    rows = []
    for source, dataset in pairs:
        a, b = by[(source, dataset, a_name)], by[(source, dataset, b_name)]
        row = {"source": source, "dataset": dataset, "comparison": comparison}
        for pkey in PROTOCOLS:
            for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
                row[f"{pkey}_Delta_FR_{met}"] = fnum(a, f"{pkey}_CATEGORY_MACRO_FR_{met}") - fnum(b, f"{pkey}_CATEGORY_MACRO_FR_{met}")
        rows.append(row)
    for pkey in PROTOCOLS:
        for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
            vals = np.asarray([fnum(r, f"{pkey}_Delta_FR_{met}") for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            rows.append({
                "source": "ALL",
                "dataset": "ALL",
                "comparison": comparison,
                "protocol": pkey,
                "metric": met,
                "targets": int(vals.size),
                "improved": int(np.sum(vals > 0)),
                "mean_delta": float(vals.mean()) if vals.size else float("nan"),
                "median_delta": float(np.median(vals)) if vals.size else float("nan"),
                "worst_delta": float(vals.min()) if vals.size else float("nan"),
            })
    return rows


def full_res_correlation(cat_rows):
    m2m = [r for r in cat_rows if r["variant"] == "M2_Ours_HardNormal_Median" and r["protocol"] == "M"]
    metric_by = {(r["source"], r["dataset"], r["category"]): {
        "P-AP": fnum(r, "P-AP"),
        "P-F1max": fnum(r, "P-F1max"),
        "PRO": fnum(r, "PRO"),
    } for r in m2m}
    category_morph = read_csv_bottleneck(BOTTLENECK_OUT / "CATEGORY_MORPHOLOGY.csv")
    joined = []
    for r in category_morph:
        m = metric_by.get((r["source"], r["dataset"], r["category"]))
        if not m:
            continue
        row = dict(r)
        row.update(m)
        joined.append(row)
    factors = [
        "anomaly_area_ratio_mean",
        "anomaly_patch_ratio_mean",
        "num_anomaly_patches_mean",
        "patch_mean_component_area_mean",
        "boundary_to_area_ratio_mean",
        "thinness_mean",
        "normal_P99_mean",
        "top1_recall_mean",
    ]
    rows = []
    for metric in ("P-AP", "P-F1max", "PRO"):
        for factor in factors:
            x = np.asarray([fnum(r, factor) for r in joined], dtype=np.float64)
            y = np.asarray([fnum(r, metric) for r in joined], dtype=np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3 or np.std(x[ok]) <= 0 or np.std(y[ok]) <= 0:
                pear, spear = float("nan"), float("nan")
            else:
                pear = float(stats.pearsonr(x[ok], y[ok]).statistic)
                spear = float(stats.spearmanr(x[ok], y[ok]).statistic)
            rows.append({"metric": f"CATEGORY_MACRO_FR_{metric}", "factor": factor, "protocol": "M", "n_categories": int(ok.sum()), "pearson": pear, "spearman": spear})
    return rows, joined


def reciprocal_and_failure(joined):
    recip = []
    focus = {"MVTec", "VisA", "BTAD", "MPDD", "CVC-ClinicDB"}
    for source in sorted({r["source"] for r in joined}):
        for dataset in sorted({r["dataset"] for r in joined if r["source"] == source and r["dataset"] in focus}):
            part = [r for r in joined if r["source"] == source and r["dataset"] == dataset]
            row = {"source": source, "dataset": dataset, "n_categories": len(part)}
            for f in ("P-AP", "P-F1max", "PRO", "anomaly_area_ratio_mean", "anomaly_patch_ratio_mean", "num_anomaly_patches_mean", "boundary_to_area_ratio_mean", "thinness_mean", "normal_P99_mean", "top1_recall_mean"):
                vals = np.asarray([fnum(r, f) for r in part], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                row[f"{f}_mean"] = float(vals.mean()) if vals.size else float("nan")
            recip.append(row)
    failure = category_failure_table(joined)
    return recip, failure


def sota_gap_full(full_rows):
    refs = sota_reference()
    rows = []
    for r in full_rows:
        if r["variant"] != "M2_Ours_HardNormal_Median":
            continue
        for ref in refs:
            if ref.get("comparison_type") != "DIRECT_COMPARABLE" or ref.get("training_source") != r["source"] or ref.get("target") != r["dataset"]:
                continue
            if ref["paper"] == "MoECLIP":
                prefix, comp = "M_CATEGORY_MACRO_FR", "DIRECT_COMPARABLE"
            elif ref["paper"] == "VisualAD":
                prefix, comp = "V_CATEGORY_MACRO_FR", "DIRECT_COMPARABLE"
            else:
                prefix, comp = "M_CATEGORY_MACRO_FR", "REFERENCE_ONLY"
            row = {"source": r["source"], "dataset": r["dataset"], "variant": r["variant"], "reference": f"{ref['paper']} {ref['venue']} {ref['year']}", "comparison_type": comp, "evaluation_protocol": prefix}
            for met, refk in (("P_AUROC", "P_AUROC"), ("P_AP", "P_AP"), ("P_F1max", "P_F1max"), ("PRO", "PRO"), ("G_pixel", "G_pixel")):
                row[f"CURRENT_FR_{met}"] = r.get(f"{prefix}_{met}", "NA")
                row[f"REFERENCE_{met}"] = ref.get(refk, "NA")
                rv = fnum(ref, refk)
                row[f"Delta_FR_{met}"] = fnum(r, f"{prefix}_{met}") - rv if np.isfinite(rv) else "NA"
            rows.append(row)
    return rows


def decide_full(corr_rows, failure_rows):
    def sp(metric, factor):
        vals = [fnum(r, "spearman") for r in corr_rows if r["metric"] == metric and r["factor"] == factor]
        return vals[0] if vals else float("nan")
    size_signal = max(abs(sp("CATEGORY_MACRO_FR_P-AP", "anomaly_area_ratio_mean")), abs(sp("CATEGORY_MACRO_FR_P-AP", "anomaly_patch_ratio_mean")), abs(sp("CATEGORY_MACRO_FR_P-AP", "num_anomaly_patches_mean")))
    worst = [r for r in failure_rows if r["group"] == "worst"]
    best = [r for r in failure_rows if r["group"] == "best"]
    worst_size = np.nanmean([fnum(r, "anomaly_patch_ratio") for r in worst]) if worst else float("nan")
    best_size = np.nanmean([fnum(r, "anomaly_patch_ratio") for r in best]) if best else float("nan")
    if size_signal >= 0.4 and np.isfinite(worst_size) and np.isfinite(best_size) and worst_size < best_size:
        return "FULL_RES_SMALL_SPARSE_BOTTLENECK_RECONFIRMED"
    if size_signal < 0.3:
        return "PATCH_GRID_BOTTLENECK_DOES_NOT_TRANSFER_TO_OFFICIAL_METRIC"
    visa_worst = [r for r in worst if r["source"] == "Full MVTec" and r["dataset"] == "VisA"]
    mvtec_worst = [r for r in worst if r["source"] == "Full VisA" and r["dataset"] == "MVTec"]
    if visa_worst and mvtec_worst:
        visa_size = np.nanmean([fnum(r, "anomaly_patch_ratio") for r in visa_worst])
        mvtec_size = np.nanmean([fnum(r, "anomaly_patch_ratio") for r in mvtec_worst])
        if visa_size < mvtec_size * 0.5:
            return "VISA_SMALL_SPARSE_LOCALIZATION_IS_REMAINING_PRIMARY_BOTTLENECK"
    return "PATCH_GRID_BOTTLENECK_DOES_NOT_TRANSFER_TO_OFFICIAL_METRIC"


def write_report(outcome, full_rows, sensitivity, ablation, corr, recip, gaps, runtime):
    lines = ["# Full-Resolution Final Benchmark and Bottleneck Revalidation", "", f"OUTCOME: `{outcome}`", "", "No training, tuning, or method change was performed. Metrics are explicitly named with `PG_*`, `M_FR_*`, and `V_FR_*` conventions.", "", "## M2 Full-Resolution Main Rows", "", "| Source | Dataset | M_FR_P-AUROC | M_FR_P-AP | V_FR_P-AP | M_FR_PRO |", "|---|---|---:|---:|---:|---:|"]
    for r in full_rows:
        if r["variant"] == "M2_Ours_HardNormal_Median":
            lines.append(f"| {r['source']} | {r['dataset']} | {fnum(r,'M_CATEGORY_MACRO_FR_P_AUROC'):.4f} | {fnum(r,'M_CATEGORY_MACRO_FR_P_AP'):.4f} | {fnum(r,'V_CATEGORY_MACRO_FR_P_AP'):.4f} | {fnum(r,'M_CATEGORY_MACRO_FR_PRO'):.4f} |")
    lines.extend(["", "## Protocol Sensitivity", "", "| Metric | Mean Abs | Max Abs |", "|---|---:|---:|"])
    for r in sensitivity:
        if r.get("source") == "ALL":
            lines.append(f"| {r['metric']} | {fnum(r,'mean_abs_delta'):.4f} | {fnum(r,'max_abs_delta'):.4f} |")
    lines.extend(["", "## Full-Res Ablation Summary", "", "| Comparison | Protocol | Metric | Improved/Targets | Mean | Median | Worst |", "|---|---|---|---:|---:|---:|---:|"])
    for r in ablation:
        if r.get("source") == "ALL":
            lines.append(f"| {r['comparison']} | {r['protocol']} | {r['metric']} | {r['improved']}/{r['targets']} | {fnum(r,'mean_delta'):.4f} | {fnum(r,'median_delta'):.4f} | {fnum(r,'worst_delta'):.4f} |")
    lines.extend(["", "## Bottleneck Correlations", "", "| Metric | Factor | Spearman |", "|---|---|---:|"])
    for r in corr:
        if r["metric"] == "CATEGORY_MACRO_FR_P-AP":
            lines.append(f"| {r['metric']} | {r['factor']} | {fnum(r,'spearman'):.4f} |")
    lines.extend(["", "## SOTA Gap Full-Res", "", "| Source | Dataset | Reference | Protocol | Delta P-AUROC | Delta P-AP | Delta G |", "|---|---|---|---|---:|---:|---:|"])
    for r in gaps:
        lines.append(f"| {r['source']} | {r['dataset']} | {r['reference']} | {r['evaluation_protocol']} | {fnum(r,'Delta_FR_P_AUROC'):.4f} | {fnum(r,'Delta_FR_P_AP'):.4f} | {fnum(r,'Delta_FR_G_pixel'):.4f} |")
    lines.append(f"\nRuntime: {runtime:.1f}s.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    full_rows, cat_rows = run_full_res(args.smoke, core_only=not args.all_targets)
    full_rows = add_missing_full_rows(full_rows, core_only=not args.all_targets)
    pg_full = patch_grid_vs_full(full_rows)
    sensitivity = protocol_sensitivity(full_rows)
    ablation = delta_summary(full_rows, "HardNormal_vs_StateAligned") + delta_summary(full_rows, "Median_vs_HardNormal")
    general = [r for r in ablation if r.get("source") == "ALL"]
    corr, joined = full_res_correlation(cat_rows)
    recip, failure = reciprocal_and_failure(joined)
    gaps = sota_gap_full(full_rows)
    outcome = decide_full(corr, failure)
    write_csv(OUT / "FULL_RES_RAW_METRICS.csv", full_rows)
    write_csv(OUT / "PATCH_GRID_VS_FULL_RES.csv", pg_full)
    write_csv(OUT / "INTERPOLATION_PROTOCOL_SENSITIVITY.csv", sensitivity)
    write_csv(OUT / "FULL_RES_ABLATION_SUMMARY.csv", ablation)
    write_csv(OUT / "FULL_RES_GENERALIZATION_SUMMARY.csv", general)
    write_csv(OUT / "FULL_RES_CATEGORY_METRICS.csv", cat_rows)
    write_csv(OUT / "FULL_RES_CORRELATION_ANALYSIS.csv", corr)
    write_csv(OUT / "FULL_RES_RECIPROCAL_PAIR_ANALYSIS.csv", recip)
    write_csv(OUT / "CATEGORY_FAILURE_TABLE.csv", failure)
    write_csv(OUT / "SOTA_GAP_FULL_RES.csv", gaps)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "runtime_sec": time.time() - start, "no_train": True, "no_tuning": True, "metric_naming": "PG_* and M/V CATEGORY_MACRO_FR_*", "protocols": PROTOCOLS})
    write_report(outcome, full_rows, sensitivity, general, corr, recip, gaps, time.time() - start)
    print(f"[FRReval] outcome={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all-targets", action="store_true", help="Run full-res exact metrics for every READY target; default keeps full-res exact to core/SOTA pairs and marks the rest.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
