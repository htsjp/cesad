from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import auc, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xxiii_b_dual_carrier_local import IMAGE_SIZE, SIGMA, apply_cdf
from experiments.phase_xxviii_b2_standardized_pro import MetaDataset, InsPLADDataset, DATA_ROOT, anomaly_get_transform
from utils.transforms import get_transform as mrad_get_transform


OUT = ROOT / "results" / "phase_xxix_pro_sensitivity"
METRIC_CACHE = Path(r"E:\WorkSpace\phase_xxix_metric_cache")
XXIV = ROOT / "results" / "phase_xxiv_pixel_calibration" / "final_target"
XXVI = ROOT / "results" / "phase_xxvi_cross_model" / "anomalyclip"
XXVIII_CACHE = Path(r"E:\WorkSpace\phase_xxviii_eval_cache")
MAX_FPR = 0.3


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mrad_dataset(name: str, path: Path):
    transform, target_transform = mrad_get_transform(type("Args", (), {"image_size": IMAGE_SIZE})())
    return MetaDataset(path, transform, target_transform, name, mode="test")


def load_masks_from_dataset(dataset) -> tuple[np.ndarray, list[dict]]:
    masks, manifest = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        masks.append((item["img_mask"].detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8))
        manifest.append({"pos": i, "category": item["cls_name"], "anomaly": int(item["anomaly"]), "img_path": item["img_path"]})
    return np.asarray(masks, dtype=np.uint8), manifest


def regions_by_image(masks: np.ndarray):
    for i, m in enumerate(masks):
        for region in measure.regionprops(measure.label(m)):
            yield i, region.coords[:, 0], region.coords[:, 1], region.area


def thresholds_for(evaluator: str, maps: np.ndarray, count: int | None = None) -> np.ndarray:
    mn, mx = float(np.min(maps)), float(np.max(maps))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.asarray([], dtype=np.float32)
    if evaluator == "E0_REFERENCE":
        c = 200 if count is None else count
        return np.arange(mn, mx, (mx - mn) / c, dtype=np.float32)
    if evaluator == "E1_REFERENCE_FAITHFUL_DENSE_THRESHOLD":
        c = 1000 if count is None else count
        return np.linspace(mn, mx, c, endpoint=False, dtype=np.float32)
    if evaluator == "E2_REFERENCE_FAITHFUL_SCORE_THRESHOLD":
        flat = np.asarray(maps, dtype=np.float32).reshape(-1)
        if flat.size > 2_000_000:
            qs = np.linspace(0, 1, 5000 if count is None else count, endpoint=False)
            return np.unique(np.quantile(flat, qs).astype(np.float32))
        vals = np.unique(flat)
        if vals.size > 5000:
            qs = np.linspace(0, 1, 5000 if count is None else count, endpoint=False)
            return np.unique(np.quantile(vals, qs).astype(np.float32))
        return vals.astype(np.float32)
    raise KeyError(evaluator)


def pro_with_thresholds(masks: np.ndarray, maps: np.ndarray, thresholds: np.ndarray, exact_reference: bool = True) -> tuple[float | None, dict, list[dict]]:
    if thresholds.size == 0:
        return None, {"defined": False, "reason": "empty_thresholds", "curve_points": 0, "points_below_cap": 0}, []
    masks = np.asarray(masks, dtype=np.uint8)
    maps = np.asarray(maps, dtype=np.float32)
    regions = []
    for i, rr, cc, area in regions_by_image(masks):
        vals = np.sort(maps[i, rr, cc].astype(np.float32))
        regions.append((vals, float(area)))
    bg_vals = np.sort(maps[masks == 0].astype(np.float32))
    denom = int(bg_vals.size)
    if not regions or denom <= 0:
        return None, {"defined": False, "reason": "no_regions_or_background", "curve_points": 0, "points_below_cap": 0}, []
    pros, fprs = [], []
    curve = []
    for th in thresholds:
        bg_pos = bg_vals.size - np.searchsorted(bg_vals, th, side="right")
        overlaps = [
            float((vals.size - np.searchsorted(vals, th, side="right")) / area)
            for vals, area in regions
        ]
        fpr = float(bg_pos / denom)
        pro = float(np.mean(overlaps))
        if np.isfinite(fpr) and np.isfinite(pro):
            fprs.append(fpr); pros.append(pro); curve.append({"fpr": fpr, "pro": pro})
    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    idx = fprs < MAX_FPR
    if int(idx.sum()) < 2:
        return None, {"defined": False, "reason": "insufficient_points_below_cap", "curve_points": int(len(fprs)), "points_below_cap": int(idx.sum())}, curve
    x = fprs[idx]
    y = pros[idx]
    try:
        xnorm = (x - x.min()) / (x.max() - x.min())
        value = float(auc(xnorm, y))
    except Exception as exc:
        return None, {"defined": False, "reason": f"{type(exc).__name__}:{exc}", "curve_points": int(len(fprs)), "points_below_cap": int(idx.sum())}, curve
    return value * 100.0, {"defined": True, "reason": "", "curve_points": int(len(fprs)), "points_below_cap": int(idx.sum())}, curve


def evaluate_pair(cell: dict, masks: np.ndarray, maps0: np.ndarray, maps1: np.ndarray, evaluators: list[str]) -> tuple[list[dict], list[dict]]:
    rows, curves = [], []
    for ev in evaluators:
        th0 = thresholds_for(ev, maps0)
        th1 = thresholds_for(ev, maps1)
        pro0, meta0, curve0 = pro_with_thresholds(masks, maps0, th0)
        pro1, meta1, curve1 = pro_with_thresholds(masks, maps1, th1)
        delta = None if pro0 is None or pro1 is None else pro1 - pro0
        rows.append({**cell, "evaluator": ev, "T0_PRO": pro0 if pro0 is not None else "", "T1_PRO": pro1 if pro1 is not None else "", "DeltaPRO": delta if delta is not None else "", "T0_defined": meta0["defined"], "T1_defined": meta1["defined"], "T0_curve_points": meta0["curve_points"], "T1_curve_points": meta1["curve_points"], "T0_points_below_cap": meta0["points_below_cap"], "T1_points_below_cap": meta1["points_below_cap"], "T0_reason": meta0["reason"], "T1_reason": meta1["reason"]})
    return rows, curves


def core_cells() -> list[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    specs = [
        ("MRAD", "MVTec", "VisA", "D1", XXIV / "cache" / "D1_target_raw.npy", XXIV / "D1_PIXEL_CDF.npz", ROOT / "data" / "VISA", "visa"),
        ("MRAD", "VisA", "MVTec", "D2", XXIV / "cache" / "D2_target_raw.npy", XXIV / "D2_PIXEL_CDF.npz", ROOT / "data" / "mvtec", "mvtec"),
        ("AnomalyCLIP", "MVTec", "VisA", "D1", XXVI / "cache" / "D1_target_raw.npy", XXVI / "D1_ANOMALYCLIP_PIXEL_CDF.npz", ROOT / "data" / "VISA", "visa"),
        ("AnomalyCLIP", "VisA", "MVTec", "D2", XXVI / "cache" / "D2_target_raw.npy", XXVI / "D2_ANOMALYCLIP_PIXEL_CDF.npz", ROOT / "data" / "mvtec", "mvtec"),
    ]
    for model, source, target, direction, raw_path, cdf_path, data_path, dataset_name in specs:
        cache_dir = METRIC_CACHE / "core" / model / source / target
        cache_dir.mkdir(parents=True, exist_ok=True)
        masks_path, m0_path, m1_path, manifest_path = cache_dir / "gt_masks.npy", cache_dir / "T0_final_maps.npy", cache_dir / "T1_final_maps.npy", cache_dir / "manifest.json"
        if not (masks_path.exists() and m0_path.exists() and m1_path.exists()):
            ds = load_mrad_dataset(dataset_name, data_path)
            masks, manifest = load_masks_from_dataset(ds)
            raw = np.load(raw_path, mmap_mode="r")
            cdf = np.load(cdf_path)["cdf"].astype(np.float32)
            maps0 = np.lib.format.open_memmap(m0_path, mode="w+", dtype=np.float16, shape=raw.shape)
            maps1 = np.lib.format.open_memmap(m1_path, mode="w+", dtype=np.float16, shape=raw.shape)
            for i in range(raw.shape[0]):
                r = np.asarray(raw[i], dtype=np.float32)
                maps0[i] = gaussian_filter(r, sigma=SIGMA).astype(np.float16)
                maps1[i] = gaussian_filter(apply_cdf(r, cdf), sigma=SIGMA).astype(np.float16)
                if i % 100 == 0:
                    maps0.flush(); maps1.flush()
            maps0.flush(); maps1.flush()
            np.save(masks_path, masks)
            write_json(manifest_path, manifest)
        cell = {"model": model, "source": source, "target": target, "system": f"{model}_{source}_to_{target}", "primary_group": "core"}
        out.append((cell, np.load(masks_path, mmap_mode="r"), np.load(m0_path, mmap_mode="r"), np.load(m1_path, mmap_mode="r")))
    return out


def mpdd_cells() -> list[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for model in ["MRAD", "AnomalyCLIP"]:
        for source in ["MVTec", "VisA"]:
            root = XXVIII_CACHE / "MPDD" / model / source
            cell = {"model": model, "source": source, "target": "MPDD", "system": ("M" if model == "MRAD" else "A") + f"-{source}", "primary_group": "MPDD"}
            out.append((cell, np.load(root / "gt_masks.npy", mmap_mode="r"), np.load(root / "T0_OFFICIAL" / "final_maps.npy", mmap_mode="r"), np.load(root / "T1_SOURCE_NORMAL_PERCENTILE" / "final_maps.npy", mmap_mode="r")))
    return out


def insplad_vari_grip() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    root = XXVIII_CACHE / "InsPLAD_AD" / "MRAD" / "VisA"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    idx = [m["pos"] for m in manifest if m["category"] == "vari-grip"]
    return (
        {"model": "MRAD", "source": "VisA", "target": "InsPLAD_AD", "system": "M-VisA", "category": "vari-grip", "primary_group": "InsPLAD_diagnostic"},
        np.asarray(np.load(root / "gt_masks.npy", mmap_mode="r")[idx]),
        np.asarray(np.load(root / "T0_OFFICIAL" / "final_maps.npy", mmap_mode="r")[idx], dtype=np.float32),
        np.asarray(np.load(root / "T1_SOURCE_NORMAL_PERCENTILE" / "final_maps.npy", mmap_mode="r")[idx], dtype=np.float32),
    )


def label_cell(rows: list[dict]) -> dict:
    defined = [r for r in rows if r["DeltaPRO"] != ""]
    vals = [float(r["DeltaPRO"]) for r in defined]
    if len(vals) < 2:
        label = "UNRESOLVED"
    elif any(v < 0 for v in vals):
        label = "EVALUATOR_SENSITIVE"
    elif all(v > 0.50 for v in vals):
        label = "ROBUST_POSITIVE"
    elif all(v > 0 for v in vals):
        label = "SIGN_STABLE_POSITIVE"
    else:
        label = "EVALUATOR_SENSITIVE"
    return {
        "model": rows[0]["model"], "source": rows[0]["source"], "target": rows[0]["target"], "system": rows[0]["system"], "primary_group": rows[0]["primary_group"],
        "sign_label": label, "defined_evaluators": len(vals),
        "min_DeltaPRO": min(vals) if vals else "", "median_DeltaPRO": float(np.median(vals)) if vals else "", "max_DeltaPRO": max(vals) if vals else "", "range_DeltaPRO": (max(vals) - min(vals)) if vals else "",
        "strong_magnitude_stability": bool(vals and min(vals) >= 1.0 and (max(vals) - min(vals)) <= 2.0),
        "moderate_magnitude_support": bool(vals and min(vals) >= 0.5),
    }


def threshold_convergence(cell, masks, maps0, maps1) -> list[dict]:
    rows = []
    for count in [100, 250, 500, 1000, 2500, 5000]:
        for method, maps in [("T0", maps0), ("T1", maps1)]:
            th = thresholds_for("E1_REFERENCE_FAITHFUL_DENSE_THRESHOLD", maps, count=count)
            pro, meta, _ = pro_with_thresholds(masks, maps, th)
            rows.append({**cell, "threshold_count": count, "method": method, "PRO": pro if pro is not None else "", **meta})
        if rows[-1]["PRO"] != "" and rows[-2]["PRO"] != "":
            rows.append({**cell, "threshold_count": count, "method": "Delta", "PRO": float(rows[-1]["PRO"]) - float(rows[-2]["PRO"]), "defined": True})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    METRIC_CACHE.mkdir(parents=True, exist_ok=True)
    impl_rows = [
        {"id": "R0/E0", "implementation": "utils.metrics.cal_pro_score", "threshold_generation": "np.arange(min,max,(max-min)/200)", "max_fpr": MAX_FPR, "duplicate_fpr_handling": "none", "boundary_interpolation": "none", "normalization": "normalize retained fprs to [0,1]", "degenerate_behavior": "exception/NaN possible"},
        {"id": "R2/B2", "implementation": "phase_xxviii_b2_standardized_pro", "threshold_generation": "reference-like plus duplicate max and boundary interpolation", "max_fpr": MAX_FPR, "duplicate_fpr_handling": "max PRO per FPR", "boundary_interpolation": "linear at cap", "normalization": "divide by max_fpr", "degenerate_behavior": "undefined"},
        {"id": "E1", "implementation": "reference-faithful dense thresholds", "threshold_generation": "1000 linspace thresholds", "max_fpr": MAX_FPR},
        {"id": "E2", "implementation": "reference-faithful score thresholds", "threshold_generation": "unique scores or 5000 quantile thresholds", "max_fpr": MAX_FPR},
    ]
    write_csv(OUT / "PRO_IMPLEMENTATION_COMPARISON.csv", impl_rows)
    (OUT / "PRO_IMPLEMENTATION_FORENSICS.md").write_text(
        "# PRO Implementation Forensics\n\n"
        "R0/E0 uses the project reference `cal_pro_score`: 200 equally spaced score thresholds, FPR over background pixels, mean overlap over connected GT regions, retain `fpr < 0.3`, normalize retained FPR values to [0,1], then sklearn AUC. It has no duplicate-FPR consolidation or boundary interpolation and can fail on degenerate curves.\n\n"
        "R2/B2 changed multiple semantics at once: duplicate-FPR consolidation, explicit origin, boundary interpolation, and max-FPR normalization. XXIX-A therefore treats B2 as forensic context, not a replacement.\n\n"
        "E1/E2 preserve R0 curve semantics and vary only threshold sampling density/policy for DeltaPRO sensitivity.\n",
        encoding="utf-8",
    )
    write_json(OUT / "EVALUATOR_PROTOCOL.json", {"phase": "XXIX-A", "evaluators": impl_rows, "no_dataset_expansion": True})

    evaluators = ["E0_REFERENCE", "E1_REFERENCE_FAITHFUL_DENSE_THRESHOLD", "E2_REFERENCE_FAITHFUL_SCORE_THRESHOLD"]
    all_rows, labels, mag = [], [], []
    for cell, masks, maps0, maps1 in core_cells() + mpdd_cells():
        rows, _ = evaluate_pair(cell, masks, maps0, maps1, evaluators)
        all_rows.extend(rows)
        lab = label_cell(rows); labels.append(lab); mag.append(lab)
    write_csv(OUT / "RAW_PRO_RESULTS.csv", all_rows)
    delta_rows = [{k: v for k, v in r.items() if k not in ["T0_reason", "T1_reason"]} for r in all_rows]
    write_csv(OUT / "DELTA_PRO_RESULTS.csv", delta_rows)
    write_csv(OUT / "SIGN_ROBUSTNESS.csv", labels)
    write_csv(OUT / "MAGNITUDE_ROBUSTNESS.csv", mag)
    mpdd_labels = [r for r in labels if r["primary_group"] == "MPDD"]
    write_csv(OUT / "MPDD_ROBUSTNESS.csv", mpdd_labels)

    # Difference decomposition: compare B2 parity rows already computed.
    b2_parity = ROOT / "results" / "phase_xxviii_multidataset" / "batch_b2_standardized_pro" / "PRO_REFERENCE_PARITY.csv"
    decomp = []
    if b2_parity.exists():
        with b2_parity.open("r", newline="", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        diffs = [float(r["abs_diff_pp"]) for r in rows if r.get("abs_diff_pp") not in ("", None)]
        decomp.append({"factor": "B2_ALL_SEMANTIC_CHANGES", "mean_abs_PRO_diff_pp": float(np.mean(diffs)) if diffs else "", "max_abs_PRO_diff_pp": float(np.max(diffs)) if diffs else "", "note": "B2 differs from R0 in duplicate-FPR handling, origin, boundary interpolation, and normalization."})
    write_csv(OUT / "DIFFERENCE_DECOMPOSITION.csv", decomp)

    # Threshold convergence on representative core cell and MPDD cell.
    conv = []
    reps = [core_cells()[0], mpdd_cells()[0]]
    for cell, masks, maps0, maps1 in reps:
        conv.extend(threshold_convergence(cell, masks, maps0, maps1))
    write_csv(OUT / "THRESHOLD_CONVERGENCE.csv", conv)

    # Curve comparison: deterministic first category for MPDD MRAD-MVTec.
    curve_rows = []
    cell, masks, maps0, maps1 = mpdd_cells()[0]
    manifest = json.loads((XXVIII_CACHE / "MPDD" / "MRAD" / "MVTec" / "manifest.json").read_text(encoding="utf-8"))
    first_cat = sorted({m["category"] for m in manifest})[0]
    idx = [m["pos"] for m in manifest if m["category"] == first_cat]
    for ev in evaluators:
        for method, maps in [("T0", np.asarray(maps0[idx], dtype=np.float32)), ("T1", np.asarray(maps1[idx], dtype=np.float32))]:
            pro, meta, curve = pro_with_thresholds(np.asarray(masks[idx]), maps, thresholds_for(ev, maps))
            for p in curve[:1000]:
                curve_rows.append({**cell, "category_selection": "first_alphabetical", "category": first_cat, "evaluator": ev, "method": method, **p, "PRO": pro if pro is not None else "", **meta})
    write_csv(OUT / "CURVE_COMPARISON.csv", curve_rows)

    # Absolute vs delta stability.
    stab = []
    for key in {(r["model"], r["source"], r["target"]) for r in all_rows}:
        rs = [r for r in all_rows if (r["model"], r["source"], r["target"]) == key and r["DeltaPRO"] != ""]
        if len(rs) >= 2:
            base = next((r for r in rs if r["evaluator"] == "E0_REFERENCE"), rs[0])
            for r in rs:
                stab.append({"model": key[0], "source": key[1], "target": key[2], "evaluator": r["evaluator"], "absolute_T0_diff_from_E0": float(r["T0_PRO"]) - float(base["T0_PRO"]), "absolute_T1_diff_from_E0": float(r["T1_PRO"]) - float(base["T1_PRO"]), "DeltaPRO_diff_from_E0": float(r["DeltaPRO"]) - float(base["DeltaPRO"])})
    write_csv(OUT / "ABSOLUTE_VS_DELTA_STABILITY.csv", stab)

    # InsPLAD vari-grip diagnostic only.
    diag_cell, dmasks, dm0, dm1 = insplad_vari_grip()
    diag_rows, _ = evaluate_pair(diag_cell, dmasks, dm0, dm1, evaluators)
    write_csv(OUT / "INSPLAD_VARI_GRIP_DIAGNOSTIC.csv", diag_rows)

    core_labels = [r for r in labels if r["primary_group"] == "core"]
    core_neg = any(float(r["min_DeltaPRO"]) < 0 for r in core_labels if r["min_DeltaPRO"] != "")
    robust_count = sum(r["sign_label"] == "ROBUST_POSITIVE" for r in core_labels)
    sign_stable_count = sum(r["sign_label"] in ("ROBUST_POSITIVE", "SIGN_STABLE_POSITIVE") for r in core_labels)
    mpdd_neg = any(float(r["min_DeltaPRO"]) < 0 for r in mpdd_labels if r["min_DeltaPRO"] != "")
    if robust_count >= 3 and sign_stable_count == 4 and not core_neg and not mpdd_neg:
        outcome = "XXIX_A_PRO_GAIN_ROBUST"
    elif sign_stable_count == 4 and not core_neg:
        outcome = "XXIX_A_PRO_GAIN_SIGN_STABLE_BUT_MAGNITUDE_SENSITIVE"
    elif core_neg:
        outcome = "XXIX_A_PRO_GAIN_EVALUATOR_SPECIFIC"
    else:
        outcome = "XXIX_A_REFERENCE_PRO_IMPLEMENTATION_UNSTABLE"
    write_json(OUT / "OUTCOME.json", {"phase": "XXIX-A", "outcome": outcome, "core_labels": core_labels, "mpdd_labels": mpdd_labels})
    report = ["# Phase XXIX-A PRO Metric Sensitivity", "", f"Outcome: `{outcome}`", "", "## Core Sign Robustness", "", "| Cell | Label | min dPRO | median dPRO | max dPRO | range |", "|---|---|---:|---:|---:|---:|"]
    for r in core_labels:
        report.append(f"| {r['system']} | {r['sign_label']} | {r['min_DeltaPRO']} | {r['median_DeltaPRO']} | {r['max_DeltaPRO']} | {r['range_DeltaPRO']} |")
    report.extend(["", "## MPDD", "", "| Cell | Label | min dPRO | median dPRO | max dPRO |", "|---|---|---:|---:|---:|"])
    for r in mpdd_labels:
        report.append(f"| {r['system']} | {r['sign_label']} | {r['min_DeltaPRO']} | {r['median_DeltaPRO']} | {r['max_DeltaPRO']} |")
    report.append("")
    report.append("STOP: no new dataset, training, calibration change, or evaluator replacement was run.")
    (OUT / "XXIX_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXIX-A PRO Metric Sensitivity\n\n")
        fp.write(f"- outcome: `{outcome}`\n")
        for r in core_labels:
            fp.write(f"- core {r['system']}: {r['sign_label']}, min dPRO {r['min_DeltaPRO']}, median dPRO {r['median_DeltaPRO']}, max dPRO {r['max_DeltaPRO']}.\n")
        for r in mpdd_labels:
            fp.write(f"- MPDD {r['system']}: {r['sign_label']}, min dPRO {r['min_DeltaPRO']}, median dPRO {r['median_DeltaPRO']}, max dPRO {r['max_DeltaPRO']}.\n")
    print(outcome)


if __name__ == "__main__":
    main()
