import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import (
    OUT as FINAL_OUT,
    INDUSTRIAL,
    MEDICAL,
    load_mvtec_source,
    load_visa_source,
    median3x3,
    protocol_cells,
    score_cell,
)
from experiments.phase_spatial_residual_refinement_standard_metric import fnum
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE
from experiments.standard_zsad_metrics import resize_mask_flat


OUT = Path("results/phase_final_bottleneck_attribution/formal")
TOP_FRACS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
PREC_FRACS = (0.001, 0.005, 0.01, 0.02, 0.05)


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


def mask_pixel(item):
    if "img_mask" not in item:
        return None
    m = item["img_mask"].detach().cpu().numpy()
    return (np.squeeze(m) > 0.5).astype(np.uint8)


def patch_mask(item):
    return resize_mask_flat(item, SIDE).reshape(SIDE, SIDE).astype(np.uint8)


def comp_stats(mask):
    labels, n = ndimage.label(mask > 0, structure=np.ones((3, 3), dtype=np.int8))
    if n == 0:
        return 0, np.asarray([], dtype=np.float64)
    return n, np.bincount(labels.reshape(-1), minlength=n + 1)[1:].astype(np.float64)


def boundary_mask(mask):
    m = mask > 0
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_and(m, ~er)


def interior_mask(mask):
    m = mask > 0
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    return ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)


def image_morphology(source, target, category, image_index, item):
    mp = mask_pixel(item)
    mg = patch_mask(item)
    pix_area = int(mp.sum()) if mp is not None else 0
    pix_total = int(mp.size) if mp is not None else 0
    ncomp, areas = comp_stats(mp) if mp is not None else (0, np.asarray([]))
    b = boundary_mask(mp) if mp is not None else np.zeros((1, 1), dtype=bool)
    patch_comp, patch_areas = comp_stats(mg)
    an_patches = int(mg.sum())
    row = {
        "source": source,
        "dataset": target,
        "category": category,
        "image_index": image_index,
        "anomaly_area_ratio": pix_area / max(pix_total, 1),
        "num_connected_components": ncomp,
        "mean_component_area": float(areas.mean()) if areas.size else 0.0,
        "median_component_area": float(np.median(areas)) if areas.size else 0.0,
        "min_component_area": float(areas.min()) if areas.size else 0.0,
        "max_component_area": float(areas.max()) if areas.size else 0.0,
        "boundary_pixel_ratio": float(b.sum()) / max(pix_total, 1),
        "boundary_to_area_ratio": float(b.sum()) / max(pix_area, 1),
        "thinness": float(b.sum()) / max(pix_area, 1),
        "num_anomaly_patches": an_patches,
        "anomaly_patch_ratio": an_patches / float(SIDE * SIDE),
        "num_components_patch_grid": patch_comp,
        "patch_mean_component_area": float(patch_areas.mean()) if patch_areas.size else 0.0,
        "small_le_1_patch": an_patches <= 1,
        "small_le_2_patch": an_patches <= 2,
        "small_le_4_patch": an_patches <= 4,
        "small_le_8_patch": an_patches <= 8,
    }
    return row


def aggregate(rows, keys, fields, prefix=""):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups.setdefault(key, []).append(r)
    out = []
    for key, vals in sorted(groups.items()):
        row = {k: v for k, v in zip(keys, key)}
        row["n_images"] = len(vals)
        for f in fields:
            arr = np.asarray([fnum(v, f) for v in vals], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            row[prefix + f + "_mean"] = float(arr.mean()) if arr.size else float("nan")
            row[prefix + f + "_median"] = float(np.median(arr)) if arr.size else float("nan")
            row[prefix + f + "_p90"] = float(np.percentile(arr, 90)) if arr.size else float("nan")
        for k in ("small_le_1_patch", "small_le_2_patch", "small_le_4_patch", "small_le_8_patch"):
            if k in vals[0]:
                row[k + "_frac"] = float(np.mean([bool(v[k]) for v in vals]))
        out.append(row)
    return out


def top_mask(score, frac):
    flat = score.reshape(-1)
    k = max(1, int(math.ceil(frac * flat.size)))
    idx = np.argpartition(flat, flat.size - k)[flat.size - k :]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(score.shape)


def ranking_rows(source, target, category, image_index, mask, score):
    rows = []
    gt = mask > 0
    for frac in TOP_FRACS:
        pred = top_mask(score, frac)
        hit = np.logical_and(pred, gt).sum()
        rows.append({
            "source": source,
            "dataset": target,
            "category": category,
            "image_index": image_index,
            "top_fraction": frac,
            "recall_at_top": float(hit) / max(int(gt.sum()), 1),
            "precision_at_top": float(hit) / max(int(pred.sum()), 1),
            "top_normal_patches": int(np.logical_and(pred, ~gt).sum()),
            "top_anomaly_patches": int(hit),
        })
    return rows


def normal_tail_row(source, target, category, image_index, mask, score):
    gt = mask > 0
    normal = score[~gt]
    anomaly = score[gt]
    row = {"source": source, "dataset": target, "category": category, "image_index": image_index}
    for name, arr in (("normal", normal), ("anomaly", anomaly)):
        if arr.size:
            row[f"{name}_P50"] = float(np.percentile(arr, 50))
            row[f"{name}_P90"] = float(np.percentile(arr, 90))
            row[f"{name}_P95"] = float(np.percentile(arr, 95))
            row[f"{name}_P99"] = float(np.percentile(arr, 99))
            row[f"{name}_max"] = float(arr.max())
        else:
            for p in ("P50", "P90", "P95", "P99", "max"):
                row[f"{name}_{p}"] = float("nan")
    row["margin_anomaly_P50_minus_normal_P95"] = row["anomaly_P50"] - row["normal_P95"]
    row["margin_anomaly_P90_minus_normal_P99"] = row["anomaly_P90"] - row["normal_P99"]
    return row


def ring_masks(mask):
    gt = mask > 0
    boundary = boundary_mask(gt)
    interior = interior_mask(gt)
    dil = ndimage.binary_dilation(gt, structure=np.ones((3, 3), dtype=bool), iterations=1)
    near = np.logical_and(dil, ~gt)
    far = np.logical_and(~dil, ~gt)
    return interior, boundary, near, far


def region_mean(score, region):
    vals = score[region]
    return float(vals.mean()) if vals.size else float("nan")


def median_mechanism_rows(source, target, category, image_index, mask, s_base, s_med):
    rows = []
    gt = mask > 0
    boundary = boundary_mask(gt)
    interior = interior_mask(gt)
    for name, score in (("HardNormal", s_base), ("HardNormal_Median", s_med)):
        pred = top_mask(score, 0.01)
        labels, n = ndimage.label(pred, structure=np.ones((3, 3), dtype=np.int8))
        isolated_fp = 0
        component_sizes = []
        for cid in range(1, n + 1):
            comp = labels == cid
            if not np.any(np.logical_and(comp, gt)):
                isolated_fp += 1
                component_sizes.append(int(comp.sum()))
        rows.append({
            "source": source,
            "dataset": target,
            "category": category,
            "image_index": image_index,
            "variant": name,
            "isolated_fp_component_count_top1": isolated_fp,
            "isolated_fp_component_mean_size": float(np.mean(component_sizes)) if component_sizes else 0.0,
            "top1_normal_patches": int(np.logical_and(pred, ~gt).sum()),
            "anomaly_top1_recall": float(np.logical_and(pred, gt).sum()) / max(int(gt.sum()), 1),
            "boundary_anomaly_recall_top1": float(np.logical_and(pred, boundary).sum()) / max(int(boundary.sum()), 1),
            "interior_anomaly_recall_top1": float(np.logical_and(pred, interior).sum()) / max(int(interior.sum()), 1),
            "interior_score_mean": region_mean(score, interior),
            "boundary_score_mean": region_mean(score, boundary),
            "near_normal_score_mean": region_mean(score, ring_masks(gt)[2]),
            "far_normal_score_mean": region_mean(score, ring_masks(gt)[3]),
        })
    return rows


def corr_pair(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) <= 0 or np.std(y[ok]) <= 0:
        return float("nan"), float("nan")
    return float(stats.pearsonr(x[ok], y[ok]).statistic), float(stats.spearmanr(x[ok], y[ok]).statistic)


def correlation_analysis(category_rows, percat_rows):
    metric_by = {(r["source"], r["dataset"], r["category"]): r for r in percat_rows if r.get("variant") == "M2_Ours_HardNormal_Median"}
    joined = []
    for r in category_rows:
        m = metric_by.get((r["source"], r["dataset"], r["category"]))
        if not m:
            continue
        row = dict(r)
        for met in ("P-AP", "P-F1max", "PRO"):
            row[met] = fnum(m, met)
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
            pear, spear = corr_pair([r.get(factor, float("nan")) for r in joined], [r.get(metric, float("nan")) for r in joined])
            rows.append({"metric": metric, "factor": factor, "n_categories": len(joined), "pearson": pear, "spearman": spear})
    return rows, joined


def build_reciprocal(category_joined):
    focus = {"MVTec", "VisA", "BTAD", "MPDD", "CVC-ClinicDB"}
    rows = []
    for source in sorted({r["source"] for r in category_joined}):
        for dataset in sorted({r["dataset"] for r in category_joined if r["source"] == source and r["dataset"] in focus}):
            part = [r for r in category_joined if r["source"] == source and r["dataset"] == dataset]
            if not part:
                continue
            row = {"source": source, "dataset": dataset, "n_categories": len(part)}
            for f in ("P-AP", "anomaly_area_ratio_mean", "anomaly_patch_ratio_mean", "num_anomaly_patches_mean", "boundary_to_area_ratio_mean", "normal_P99_mean", "top1_recall_mean"):
                arr = np.asarray([fnum(r, f) for r in part], dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                row[f + "_mean"] = float(arr.mean()) if arr.size else float("nan")
            rows.append(row)
    return rows


def category_failure_table(category_joined):
    rows = []
    for source, dataset in (("Full VisA", "MVTec"), ("Full MVTec", "VisA")):
        part = [r for r in category_joined if r["source"] == source and r["dataset"] == dataset and np.isfinite(fnum(r, "P-AP"))]
        part = sorted(part, key=lambda r: fnum(r, "P-AP"))
        for rank, r in enumerate(part[:5], start=1):
            rows.append(failure_row(r, "worst", rank))
        for rank, r in enumerate(reversed(part[-5:]), start=1):
            rows.append(failure_row(r, "best", rank))
    return rows


def failure_row(r, group, rank):
    return {
        "source": r["source"],
        "dataset": r["dataset"],
        "category": r["category"],
        "group": group,
        "rank": rank,
        "P-AP": r["P-AP"],
        "P-F1max": r["P-F1max"],
        "PRO": r["PRO"],
        "anomaly_patch_ratio": r["anomaly_patch_ratio_mean"],
        "median_component_size": r["median_component_area_mean"],
        "boundary_ratio": r["boundary_to_area_ratio_mean"],
        "top1_recall": r["top1_recall_mean"],
        "normal_P99": r["normal_P99_mean"],
    }


def decide(corr_rows, failure_rows):
    def sp(metric, factor):
        vals = [fnum(r, "spearman") for r in corr_rows if r["metric"] == metric and r["factor"] == factor]
        return vals[0] if vals else float("nan")
    ap_size = max(abs(sp("P-AP", "anomaly_area_ratio_mean")), abs(sp("P-AP", "num_anomaly_patches_mean")))
    ap_tail = max(abs(sp("P-AP", "normal_P99_mean")), abs(sp("P-AP", "top1_recall_mean")))
    pro_boundary = max(abs(sp("PRO", "boundary_to_area_ratio_mean")), abs(sp("PRO", "thinness_mean")))
    ap_boundary = max(abs(sp("P-AP", "boundary_to_area_ratio_mean")), abs(sp("P-AP", "thinness_mean")))
    worst = [r for r in failure_rows if r["group"] == "worst"]
    best = [r for r in failure_rows if r["group"] == "best"]
    worst_size = np.nanmean([fnum(r, "anomaly_patch_ratio") for r in worst]) if worst else float("nan")
    best_size = np.nanmean([fnum(r, "anomaly_patch_ratio") for r in best]) if best else float("nan")
    worst_sparse = np.isfinite(worst_size) and np.isfinite(best_size) and worst_size < best_size
    if ap_size >= 0.4 and worst_sparse:
        return "SMALL_SPARSE_ANOMALY_LOCALIZATION_IS_PRIMARY_BOTTLENECK"
    if ap_tail >= 0.4:
        return "NORMAL_TAIL_RANKING_REMAINS_PRIMARY_BOTTLENECK"
    if pro_boundary >= 0.4 and ap_boundary < 0.4:
        return "BOUNDARY_COVERAGE_EXPLAINS_PRO_NOT_MAIN_AP_GAP"
    return "NO_SINGLE_STRUCTURAL_BOTTLENECK_IDENTIFIED"


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    raw_metrics = read_csv(FINAL_OUT / "RAW_METRICS.csv")
    percat_metrics = read_csv(FINAL_OUT / "PER_CATEGORY_STANDARD_METRICS.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_visa_source(device), load_mvtec_source()]
    morphology, ranking, normal_tail, median_rows = [], [], [], []
    for model in models:
        cells, _audits = protocol_cells(model["source"], device, args.smoke)
        for cell_id, target, dataset, idx_or_rows, patch, kind in cells:
            n = min(24, patch.shape[0]) if args.smoke else patch.shape[0]
            refs = idx_or_rows[:n]
            print(f"[Bottleneck] {model['source']} -> {target} n={n}", flush=True)
            _image, maps = score_cell(model, patch, device, args.smoke)
            s1_maps = maps["M1_HardNormal"].astype(np.float32)
            s2_maps = maps["M2_Ours_HardNormal_Median"].astype(np.float32)
            for pos, ref in enumerate(refs):
                item = item_at(dataset, kind, ref, pos)
                cat, label = item_meta(item, kind, ref, target)
                if label != 1 or "img_mask" not in item:
                    continue
                mg = patch_mask(item)
                morphology.append(image_morphology(model["source"], target, cat, pos, item))
                ranking.extend(ranking_rows(model["source"], target, cat, pos, mg, s2_maps[pos]))
                normal_tail.append(normal_tail_row(model["source"], target, cat, pos, mg, s2_maps[pos]))
                median_rows.extend(median_mechanism_rows(model["source"], target, cat, pos, mg, s1_maps[pos], s2_maps[pos]))
    category_morph = aggregate(
        morphology,
        ["source", "dataset", "category"],
        ["anomaly_area_ratio", "num_anomaly_patches", "anomaly_patch_ratio", "median_component_area", "patch_mean_component_area", "boundary_to_area_ratio", "thinness"],
    )
    top1 = [r for r in ranking if abs(fnum(r, "top_fraction") - 0.01) < 1e-9]
    cat_rank = aggregate(top1, ["source", "dataset", "category"], ["recall_at_top", "precision_at_top", "top_normal_patches"], prefix="top1_")
    for r in cat_rank:
        r["top1_recall_mean"] = r.get("top1_recall_at_top_mean", float("nan"))
        r["top1_precision_mean"] = r.get("top1_precision_at_top_mean", float("nan"))
    cat_tail = aggregate(normal_tail, ["source", "dataset", "category"], ["normal_P95", "normal_P99", "anomaly_P50", "anomaly_P90", "margin_anomaly_P50_minus_normal_P95", "margin_anomaly_P90_minus_normal_P99"])
    cat_by = {(r["source"], r["dataset"], r["category"]): dict(r) for r in category_morph}
    for extra in cat_rank + cat_tail:
        key = (extra["source"], extra["dataset"], extra["category"])
        cat_by.setdefault(key, {"source": key[0], "dataset": key[1], "category": key[2]}).update(extra)
    category_rows = list(cat_by.values())
    corr_rows, category_joined = correlation_analysis(category_rows, percat_metrics)
    reciprocal = build_reciprocal(category_joined)
    failure = category_failure_table(category_joined)
    outcome = decide(corr_rows, failure)
    write_csv(OUT / "DATASET_MORPHOLOGY.csv", morphology)
    write_csv(OUT / "CATEGORY_MORPHOLOGY.csv", category_rows)
    write_csv(OUT / "RANKING_DIAGNOSTICS.csv", ranking)
    write_csv(OUT / "NORMAL_TAIL_DIAGNOSTICS.csv", normal_tail)
    write_csv(OUT / "MEDIAN_MECHANISM.csv", median_rows)
    write_csv(OUT / "CORRELATION_ANALYSIS.csv", corr_rows)
    write_csv(OUT / "RECIPROCAL_PAIR_ANALYSIS.csv", reciprocal)
    write_csv(OUT / "CATEGORY_FAILURE_TABLE.csv", failure)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "runtime_sec": time.time() - start, "device": device, "no_train": True, "no_tuning": True})
    write_report(outcome, raw_metrics, corr_rows, reciprocal, failure, time.time() - start)
    print(f"[Bottleneck] outcome={outcome}", flush=True)


def write_report(outcome, raw_metrics, corr_rows, reciprocal, failure, runtime):
    lines = ["# Final ZSAD Bottleneck Attribution Audit", "", f"OUTCOME: `{outcome}`", "", "This is a no-train/no-tuning diagnostic over the frozen final method: Canonical Two-State + HardNormal + fixed 3x3 Median.", "", "## Key Correlations", "", "| Metric | Factor | Pearson | Spearman |", "|---|---|---:|---:|"]
    focus = {"anomaly_area_ratio_mean", "anomaly_patch_ratio_mean", "num_anomaly_patches_mean", "normal_P99_mean", "top1_recall_mean", "boundary_to_area_ratio_mean", "thinness_mean"}
    for r in corr_rows:
        if r["metric"] in {"P-AP", "PRO"} and r["factor"] in focus:
            lines.append(f"| {r['metric']} | {r['factor']} | {fnum(r,'pearson'):.4f} | {fnum(r,'spearman'):.4f} |")
    lines.extend(["", "## Reciprocal Pairs", "", "| Source | Dataset | P-AP | Patch Ratio | Top1 Recall | Normal P99 | Boundary/Area |", "|---|---|---:|---:|---:|---:|---:|"])
    for r in reciprocal:
        if (r["source"], r["dataset"]) in {("Full VisA", "MVTec"), ("Full MVTec", "VisA")} or r["dataset"] in {"BTAD", "MPDD", "CVC-ClinicDB"}:
            lines.append(f"| {r['source']} | {r['dataset']} | {fnum(r,'P-AP_mean'):.4f} | {fnum(r,'anomaly_patch_ratio_mean_mean'):.4f} | {fnum(r,'top1_recall_mean_mean'):.4f} | {fnum(r,'normal_P99_mean_mean'):.4f} | {fnum(r,'boundary_to_area_ratio_mean_mean'):.4f} |")
    lines.extend(["", "## MVTec / VisA Category Extremes", "", "| Source | Dataset | Group | Category | P-AP | Patch Ratio | Top1 Recall | Normal P99 |", "|---|---|---|---|---:|---:|---:|---:|"])
    for r in failure:
        lines.append(f"| {r['source']} | {r['dataset']} | {r['group']} | {r['category']} | {fnum(r,'P-AP'):.4f} | {fnum(r,'anomaly_patch_ratio'):.4f} | {fnum(r,'top1_recall'):.4f} | {fnum(r,'normal_P99'):.4f} |")
    lines.extend(["", "## Interpretation", ""])
    if outcome == "SMALL_SPARSE_ANOMALY_LOCALIZATION_IS_PRIMARY_BOTTLENECK":
        lines.append("Category-level P-AP tracks anomaly size / patch occupancy, and the reciprocal MVTec/VisA failures concentrate in sparse categories. This supports treating small dense localization as the primary remaining bottleneck.")
    elif outcome == "NORMAL_TAIL_RANKING_REMAINS_PRIMARY_BOTTLENECK":
        lines.append("The strongest P-AP relationship is with top-rank recall or normal-tail contamination, so the remaining gap is better explained by score ranking errors than by morphology alone.")
    elif outcome == "BOUNDARY_COVERAGE_EXPLAINS_PRO_NOT_MAIN_AP_GAP":
        lines.append("Boundary/thinness mainly explains PRO variation, not the main P-AP gap. Optimizing only boundary coverage would be a distraction for the current AP deficit.")
    else:
        lines.append("No single structural factor passed the attribution gate cleanly. The benchmark is better treated as a mixed-error regime for writing rather than a green light for another method branch.")
    lines.append(f"\nRuntime: {runtime:.1f}s.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
