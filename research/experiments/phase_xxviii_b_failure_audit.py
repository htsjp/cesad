from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from skimage import measure
from sklearn.metrics import auc


ROOT = Path(__file__).resolve().parents[1]
PHASE_A = ROOT / "results" / "phase_xxviii_multidataset" / "batch_a"
AUDIT_A = ROOT / "results" / "phase_xxviii_multidataset" / "audit"
OUT = ROOT / "results" / "phase_xxviii_multidataset" / "batch_b_failure_audit"


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


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def safe_pro_score(masks: np.ndarray, amaps: np.ndarray, max_step: int = 200, expect_fpr: float = 0.3) -> tuple[float, dict]:
    masks = np.asarray(masks)
    amaps = np.asarray(amaps, dtype=np.float32)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if amaps.ndim == 4:
        amaps = amaps.squeeze(1)
    min_th, max_th = float(np.min(amaps)), float(np.max(amaps))
    detail = {"min_th": min_th, "max_th": max_th, "candidate_points": 0, "fpr_points_below_0_3": 0, "fallback_reason": ""}
    if not np.isfinite(min_th) or not np.isfinite(max_th) or max_th <= min_th:
        detail["fallback_reason"] = "constant_or_invalid_prediction"
        return 0.0, detail
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    delta = (max_th - min_th) / max_step
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        denom = inverse_masks.sum()
        if denom <= 0 or not pro:
            continue
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        pros.append(float(np.mean(pro)))
        fprs.append(float(fp_pixels / denom))
    pros = np.asarray(pros, dtype=np.float32)
    fprs = np.asarray(fprs, dtype=np.float32)
    idxes = np.isfinite(fprs) & np.isfinite(pros) & (fprs < expect_fpr)
    detail["candidate_points"] = int(len(fprs))
    detail["fpr_points_below_0_3"] = int(idxes.sum())
    if int(idxes.sum()) < 2:
        detail["fallback_reason"] = "insufficient_fpr_points"
        return 0.0, detail
    ff = fprs[idxes]
    pp = pros[idxes]
    if float(ff.max()) <= float(ff.min()):
        detail["fallback_reason"] = "zero_fpr_span"
        return 0.0, detail
    ff = (ff - ff.min()) / (ff.max() - ff.min())
    return float(auc(ff, pp)), detail


def original_pro_score(masks: np.ndarray, amaps: np.ndarray, max_step: int = 200, expect_fpr: float = 0.3) -> float:
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
    pros, fprs = np.array(pros), np.array(fprs)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    return float(auc(fprs, pros[idxes]))


def synthetic_cases() -> list[dict]:
    rng = np.random.default_rng(0)
    masks = np.zeros((4, 32, 32), dtype=np.uint8)
    masks[:, 8:16, 8:16] = 1
    masks_multi = masks.copy()
    masks_multi[:, 22:27, 22:27] = 1
    cases = {
        "perfect_prediction": masks.astype(np.float32),
        "perfect_inverse_prediction": 1.0 - masks.astype(np.float32),
        "constant_prediction": np.zeros_like(masks, dtype=np.float32) + 0.5,
        "random_prediction": rng.random(masks.shape, dtype=np.float32),
        "single_region_anomaly": masks.astype(np.float32) + 0.05 * rng.random(masks.shape, dtype=np.float32),
        "multiple_anomaly_regions": masks_multi.astype(np.float32) + 0.05 * rng.random(masks.shape, dtype=np.float32),
        "very_sparse_predicted_positives": (rng.random(masks.shape) > 0.998).astype(np.float32),
    }
    rows = []
    for name, amap in cases.items():
        safe, detail = safe_pro_score(masks_multi if name == "multiple_anomaly_regions" else masks, amap)
        try:
            orig = original_pro_score(masks_multi if name == "multiple_anomaly_regions" else masks, amap)
            orig_status = "defined"
            diff = abs(orig - safe) * 100.0
        except Exception as exc:
            orig = float("nan")
            orig_status = f"undefined:{type(exc).__name__}:{exc}"
            diff = float("nan")
        rows.append({
            "case": name,
            "original_status": orig_status,
            "original_PRO": orig * 100.0 if np.isfinite(orig) else "",
            "fallback_PRO": safe * 100.0,
            "abs_diff_pp_when_defined": diff if np.isfinite(diff) else "",
            **detail,
            "finite": math.isfinite(safe),
            "valid_range": 0.0 <= safe <= 1.0,
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_metrics = read_csv(PHASE_A / "RAW_METRICS.csv")
    cell_outcomes = read_csv(PHASE_A / "CELL_OUTCOMES.csv")
    support = read_csv(PHASE_A / "CALIBRATION_SUPPORT_SHIFT.csv")
    percat = read_csv(PHASE_A / "PER_CATEGORY.csv")
    artifact = json.loads((AUDIT_A / "FROZEN_ARTIFACT_AUDIT.json").read_text(encoding="utf-8"))

    insplad_metrics = [r for r in raw_metrics if r["target"] == "InsPLAD_AD"]
    insplad_cells = [r for r in cell_outcomes if r["target"] == "InsPLAD_AD"]
    failing = next(r for r in cell_outcomes if r["system"] == "M-VisA" and r["target"] == "InsPLAD_AD")
    failing_cat = [r for r in percat if r["system"] == "M-VisA" and r["target"] == "InsPLAD_AD"]

    integrity = {
        "phase": "XXVIII-B",
        "artifact_audit_source": str(AUDIT_A / "FROZEN_ARTIFACT_AUDIT.json"),
        "artifacts_ok": bool(artifact.get("artifacts_ok")),
        "insplad_cell_jsons_present": {
            name: (PHASE_A / "cell_results" / name).exists()
            for name in [
                "M-MVTec_InsPLAD_AD.json",
                "M-VisA_InsPLAD_AD.json",
                "A-MVTec_InsPLAD_AD.json",
                "A-VisA_InsPLAD_AD.json",
            ]
        },
        "target_image_count": 225,
        "gt_mask_count": 170,
        "t0_t1_pairing": "All XXVIII-A cells stored max_abs_image_score_diff=0.0 and image_parity_pass=True.",
        "artifacts": artifact.get("artifacts", []),
    }
    integrity["pass"] = bool(integrity["artifacts_ok"] and all(integrity["insplad_cell_jsons_present"].values()))
    write_json(OUT / "ARTIFACT_INTEGRITY.json", integrity)

    fallback_audit = {
        "phase": "XXVIII-B",
        "original_failure": {
            "exception": "ValueError: At least 2 points are needed to compute area under curve, but x.shape = 1",
            "cell": "M-VisA -> InsPLAD_AD",
            "method": "T1_SOURCE_NORMAL_PERCENTILE",
            "category": "vari-grip",
            "observed_stored_PRO": 0.0,
            "reason": "Original PRO implementation produced an insufficient FPR curve after fpr < 0.3 filtering; fallback returned 0.0.",
            "fallback_invoked_for": ["M-VisA -> InsPLAD_AD / T1_SOURCE_NORMAL_PERCENTILE / vari-grip"],
            "fallback_not_invoked_for": "All completed aggregate cells except this per-category edge case completed without raising an exception during XXVIII-A.",
        },
        "validation_limitation": "XXVIII-A intentionally did not save full maps to conserve D: disk, so original-vs-fallback parity cannot be recomputed for every historical XXVIII cell from stored artifacts alone without rerunning all cells.",
    }
    write_json(OUT / "PRO_FALLBACK_AUDIT.json", fallback_audit)

    sanity = synthetic_cases()
    write_csv(OUT / "PRO_SYNTHETIC_SANITY.csv", sanity)
    defined = [r for r in sanity if r["original_status"] == "defined" and r["abs_diff_pp_when_defined"] != ""]
    mean_diff = float(np.mean([float(r["abs_diff_pp_when_defined"]) for r in defined])) if defined else float("nan")
    max_diff = float(np.max([float(r["abs_diff_pp_when_defined"]) for r in defined])) if defined else float("nan")
    parity_rows = [{
        "scope": "synthetic_defined_cases",
        "defined_cases": len(defined),
        "mean_abs_diff_pp": mean_diff,
        "max_abs_diff_pp": max_diff,
        "threshold_mean_pp": 0.02,
        "threshold_max_pp": 0.05,
        "passes_threshold": bool(np.isfinite(mean_diff) and np.isfinite(max_diff) and mean_diff <= 0.02 and max_diff <= 0.05),
    }, {
        "scope": "all_xxviii_cells_where_original_works",
        "defined_cases": "",
        "mean_abs_diff_pp": "",
        "max_abs_diff_pp": "",
        "threshold_mean_pp": 0.02,
        "threshold_max_pp": 0.05,
        "passes_threshold": False,
        "reason": "Not validated from stored XXVIII-A artifacts because full maps were not retained; protocol requires this parity before interpreting fallback-dependent negative transfer.",
    }]
    write_csv(OUT / "PRO_FALLBACK_PARITY.csv", parity_rows)

    repro_rows = []
    for r in insplad_metrics:
        repro_rows.append({**r, "source": "XXVIII-A stored result", "recomputed_in_xxviii_b": False, "reason": "PRO fallback parity did not pass; protocol stops before recomputing maps."})
    write_csv(OUT / "INSPLAD_REPRODUCTION.csv", repro_rows)

    decomposition = [{
        "system": "M-VisA",
        "target": "InsPLAD_AD",
        "dP-AUROC": f(failing, "dP-AUROC"),
        "dPRO": f(failing, "dPRO"),
        "G_pixel": f(failing, "G_pixel"),
        "failure_type": "almost_entirely_PRO",
        "note": "P-AUROC increased slightly, while PRO dropped by -16.6591 pp."
    }]
    write_csv(OUT / "FAILURE_DECOMPOSITION.csv", decomposition)

    write_csv(OUT / "MONOTONICITY.csv", [{
        "system": "M-VisA",
        "target": "InsPLAD_AD",
        "status": "not_computed",
        "reason": "Protocol stops at PRO fallback validation failure before recomputing maps.",
    }])
    write_csv(OUT / "CALIBRATION_SUPPORT_COMPARISON.csv", [r for r in support if r["target"] == "InsPLAD_AD"])

    cross_target = [
        {
            "model": "MRAD",
            "source": "VisA",
            "target": "MVTec",
            "baseline_P-AUROC": 92.8762,
            "calibrated_P-AUROC": 92.8752,
            "baseline_PRO": 86.8162,
            "calibrated_PRO": 91.6972,
            "dP-AUROC": -0.0010,
            "dPRO": 4.8810,
            "G_pixel": 2.4400,
            "source": "XXIV-T frozen historical",
        },
        {
            "model": "MRAD",
            "source": "VisA",
            "target": "InsPLAD_AD",
            "baseline_P-AUROC": 76.1774827952148,
            "calibrated_P-AUROC": 76.44908438613565,
            "baseline_PRO": 52.21233707318314,
            "calibrated_PRO": 35.553258073227035,
            "dP-AUROC": f(failing, "dP-AUROC"),
            "dPRO": f(failing, "dPRO"),
            "G_pixel": f(failing, "G_pixel"),
            "source": "XXVIII-A stored result; fallback-dependent",
        },
    ]
    write_csv(OUT / "CROSS_TARGET_REFERENCE.csv", cross_target)

    avisa = next(r for r in cell_outcomes if r["system"] == "A-VisA" and r["target"] == "InsPLAD_AD")
    cross_model = [{
        "target": "InsPLAD_AD",
        "source_dataset_concept": "VisA",
        "system": "M-VisA",
        "model": "MRAD",
        "dP-AUROC": f(failing, "dP-AUROC"),
        "dPRO": f(failing, "dPRO"),
        "G_pixel": f(failing, "G_pixel"),
        "positive": failing["positive"],
    }, {
        "target": "InsPLAD_AD",
        "source_dataset_concept": "VisA",
        "system": "A-VisA",
        "model": "AnomalyCLIP",
        "dP-AUROC": f(avisa, "dP-AUROC"),
        "dPRO": f(avisa, "dPRO"),
        "G_pixel": f(avisa, "G_pixel"),
        "positive": avisa["positive"],
    }]
    write_csv(OUT / "CROSS_MODEL_REFERENCE.csv", cross_model)

    neg_total = sum(abs(min(0.0, f(r, "G_pixel"))) for r in failing_cat)
    rows = []
    for r in failing_cat:
        g = f(r, "G_pixel")
        share = abs(min(0.0, g)) / neg_total if neg_total > 0 else 0.0
        rows.append({**r, "negative_contribution_share": share, "CATEGORY_CONCENTRATED_NEGATIVE_TRANSFER_gt50": share > 0.5, "CATEGORY_CONCENTRATED_NEGATIVE_TRANSFER_gt75": share > 0.75})
    write_csv(OUT / "PER_CATEGORY_FAILURE.csv", rows)

    write_csv(OUT / "MAP_DISTRIBUTIONS.csv", [{
        "system": "M-VisA",
        "target": "InsPLAD_AD",
        "status": "not_computed",
        "reason": "Stopped at fallback validation failure; full maps were not retained in XXVIII-A.",
    }])
    write_csv(OUT / "MAP_NUMERICAL_AUDIT.csv", [{
        "system": "M-VisA",
        "target": "InsPLAD_AD",
        "status": "not_computed",
        "reason": "Stopped at fallback validation failure; full maps were not retained in XXVIII-A.",
    }])

    outcome = "XXVIII_B_PRO_FALLBACK_NOT_VALIDATED"
    report = [
        "# Phase XXVIII-B Failure Adjudication",
        "",
        f"Outcome: `{outcome}`",
        "",
        "XXVIII-B stopped at the required PRO fallback validation gate. No training, target calibration, alternate normalization, new dataset, or performance rescue was run.",
        "",
        "## Key Finding",
        "",
        "The critical XXVIII-A failing cell is `M-VisA -> InsPLAD_AD`. Its P-AUROC increased by `+0.2716 pp`, but PRO dropped by `-16.6591 pp`, so the failure is almost entirely PRO-driven.",
        "",
        "The decisive negative component is category-level: `vari-grip` has `T1_PRO = 0.0` after the local fallback, with `G_pixel = -35.7964 pp`. That category contributes essentially all aggregate negative degradation.",
        "",
        "## Fallback Gate",
        "",
        "The original PRO failure was `ValueError: At least 2 points are needed to compute area under curve, but x.shape = 1`. Because XXVIII-A did not retain full maps, the required original-vs-fallback parity test for every XXVIII cell where original PRO works cannot be completed from stored artifacts alone. Synthetic cases were finite and sane, but that is not enough for the mandatory parity gate.",
        "",
        "Therefore the fallback is not sufficiently validated for interpreting the fallback-dependent `vari-grip` negative PRO as a real scientific negative transfer.",
        "",
        "## Stopped Before",
        "",
        "Per protocol, monotonicity, full map numerical audit, and full recomputation are not used to assign a real negative-transfer outcome because the fallback validation gate did not pass.",
    ]
    (OUT / "XXVIII_B_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {
        "phase": "XXVIII-B",
        "outcome": outcome,
        "reason": "Mandatory original-vs-fallback PRO parity for every XXVIII cell where original works cannot be validated from stored artifacts; failing result depends on fallback T1 PRO=0.0 for vari-grip.",
        "no_training": True,
        "no_target_calibration": True,
        "no_additional_dataset": True,
    })
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXVIII-B Failure Adjudication\n\n")
        fp.write(f"- outcome: `{outcome}`\n")
        fp.write("- Critical cell: `M-VisA -> InsPLAD_AD`, dP `+0.2716 pp`, dPRO `-16.6591 pp`, G `-8.1937 pp`.\n")
        fp.write("- Fallback-dependent category: `vari-grip`, T1 PRO `0.0`, G `-35.7964 pp`.\n")
        fp.write("- No training, target calibration, alternate method, or additional dataset was run.\n")
    print(outcome)


if __name__ == "__main__":
    main()
