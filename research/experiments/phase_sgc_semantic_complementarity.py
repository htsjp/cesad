from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "phase_sgc_semantic_complementarity"
DATA_ROOT = Path(r"E:\WorkSpace\dataset")

MEDICAL_TARGETS = ["CVC-ClinicDB", "CVC-ColonDB", "Kvasir"]
NORMAL_PROMPTS = ["a normal object", "an intact object"]
ANOMALY_PROMPTS = ["an anomalous object", "a damaged object"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def num(v: Any) -> float:
    try:
        if pd.isna(v):
            return float("nan")
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def fmt(v: Any) -> str:
    x = num(v)
    return "NA" if not math.isfinite(x) else f"{x:.4f}"


def mask_path_for(root: Path, row: dict[str, Any]) -> Path | None:
    mask = str(row.get("mask_path") or "")
    if mask:
        p = root / mask
        if p.exists():
            return p
    img_rel = Path(str(row.get("img_path") or ""))
    if int(row.get("anomaly", 0)) and "test" in img_rel.parts:
        parts = list(img_rel.parts)
        parts[parts.index("test")] = "ground_truth"
        parts[-1] = f"{Path(parts[-1]).stem}_mask.png"
        p = root.joinpath(*parts)
        if p.exists():
            return p
    return None


def anomaly_pixel_ratio(dataset: str) -> dict[str, Any]:
    root = DATA_ROOT / dataset
    meta_path = root / "meta.json"
    out = {
        "dataset": dataset,
        "mask_images": 0,
        "total_pixels": 0,
        "anomaly_pixels": 0,
        "anomaly_pixel_ratio": float("nan"),
        "status": "NOT_AVAILABLE",
    }
    if not meta_path.exists():
        return out
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for _cat, items in meta.get("test", {}).items():
        for item in items:
            if not int(item.get("anomaly", 0)):
                continue
            mp = mask_path_for(root, item)
            if mp is None:
                continue
            arr = np.asarray(Image.open(mp).convert("L")) > 0
            out["mask_images"] += 1
            out["total_pixels"] += int(arr.size)
            out["anomaly_pixels"] += int(arr.sum())
    if out["total_pixels"] > 0:
        out["anomaly_pixel_ratio"] = float(out["anomaly_pixels"] / out["total_pixels"])
        out["status"] = "OK"
    return out


def build_medical_error_audit() -> list[dict[str, Any]]:
    metrics_path = RESULTS / "paper_assets" / "medical_visa_source_comparison" / "MEDICAL_VISA_SOURCE_FULL_METRICS.csv"
    metrics = read_csv(metrics_path)
    ratios = {d: anomaly_pixel_ratio(d) for d in MEDICAL_TARGETS}
    rows: list[dict[str, Any]] = []
    for target in MEDICAL_TARGETS:
        sub = metrics[metrics["target"].eq(target)] if not metrics.empty else pd.DataFrame()
        ours = sub[sub["method"].eq("Ours")].iloc[0].to_dict() if not sub[sub["method"].eq("Ours")].empty else {}
        fb = sub[sub["method"].eq("FB-CLIP")].iloc[0].to_dict() if not sub[sub["method"].eq("FB-CLIP")].empty else {}
        va = sub[sub["method"].eq("VisualAD")].iloc[0].to_dict() if not sub[sub["method"].eq("VisualAD")].empty else {}
        ratio = ratios[target]
        rows.append(
            {
                "source": "Full VisA",
                "dataset": target,
                "ces_variant": ours.get("variant", "NOT_AVAILABLE"),
                "ces_P_AUROC": num(ours.get("P_AUROC")),
                "ces_P_AP": num(ours.get("P_AP")),
                "ces_P_F1MAX": num(ours.get("P_F1MAX")),
                "ces_AUPRO": num(ours.get("AUPRO")),
                "fbclip_P_AP": num(fb.get("P_AP")),
                "visualad_P_AP": num(va.get("P_AP")),
                "fbclip_minus_ces_P_AP": num(fb.get("P_AP")) - num(ours.get("P_AP")),
                "visualad_minus_ces_P_AP": num(va.get("P_AP")) - num(ours.get("P_AP")),
                "anomaly_pixel_ratio": ratio["anomaly_pixel_ratio"],
                "mask_images": ratio["mask_images"],
                "interpretation_scope": "metric_level_only_no_raw_map_pairing",
                "raw_map_alignment_status": "PROTOCOL_BLOCKED_RAW_MAPS_NOT_SAVED",
                "source_file": str(metrics_path.relative_to(ROOT)) if metrics_path.exists() else "NOT_AVAILABLE",
            }
        )
    return rows


def build_paired_complementarity() -> list[dict[str, Any]]:
    rows = []
    for target in MEDICAL_TARGETS:
        for method in ["FB-CLIP", "VisualAD", "AnomalyCLIP"]:
            if method == "AnomalyCLIP":
                status = "NOT_AVAILABLE_PROMPT_TENSOR_LOAD_ERROR_NO_ALIGNED_RAW_MAP"
            else:
                status = "PROTOCOL_BLOCKED_RAW_MAPS_NOT_PERSISTED"
            rows.append(
                {
                    "source": "Full VisA",
                    "dataset": target,
                    "comparison": f"CESAD_vs_{method}",
                    "paired_image_count": 0,
                    "ces_detected_anomaly_pixels": "NOT_COMPUTED",
                    "specialist_detected_ces_missed_pixels": "NOT_COMPUTED",
                    "ces_unique_correct_pixels": "NOT_COMPUTED",
                    "status": status,
                    "note": "Existing artifacts contain full-resolution metrics, not per-image raw maps with stable sample IDs. SGC forbids inferring pixel complementarity from tables.",
                }
            )
    return rows


def build_source_category_diagnostics() -> list[dict[str, Any]]:
    geom_path = RESULTS / "phase_text_state_complementarity_audit" / "formal" / "TEXT_STATE_GEOMETRY.csv"
    geom = read_csv(geom_path)
    rows: list[dict[str, Any]] = []
    for source in ["Full VisA", "Full MVTec"]:
        g = geom[geom["source"].eq(source)].iloc[0].to_dict() if not geom.empty and not geom[geom["source"].eq(source)].empty else {}
        rows.append(
            {
                "source": source,
                "diagnostic": "state_text_geometry",
                "cosine_d_s_d_t": num(g.get("cosine_d_s_d_t")),
                "angle_deg": num(g.get("angle_deg")),
                "d_t_perp_norm_over_d_t_norm": num(g.get("d_t_perp_norm_over_d_t_norm")),
                "status": "AVAILABLE_FROM_EXISTING_TEXT_STATE_AUDIT" if g else "NOT_AVAILABLE",
                "source_file": str(geom_path.relative_to(ROOT)) if geom_path.exists() else "NOT_AVAILABLE",
            }
        )
        rows.append(
            {
                "source": source,
                "diagnostic": "source_category_patch_ap_and_hard_normal",
                "status": "NOT_RUN_IN_SGC",
                "note": "Requires source patch labels and category-level scoring pass. Existing SGC run does not infer category bias without raw source-category diagnostics.",
            }
        )
    rows.append(
        {
            "source": "Full MVTec_vs_Full VisA",
            "diagnostic": "source_transfer_bias",
            "status": "ASSOCIATION_ONLY",
            "note": "Existing source comparisons are confounded by different fixed preprocessing: VisA-source uses PC1 removal/frozen rank-4 adapter; MVTec-source does not. Do not interpret performance gap causally.",
        }
    )
    return rows


def copy_semantic_metrics() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_path = RESULTS / "phase_text_state_complementarity_audit" / "formal" / "RAW_METRICS.csv"
    mean_path = RESULTS / "phase_text_state_complementarity_audit" / "formal" / "MEAN_DELTA.csv"
    raw = read_csv(raw_path)
    mean = read_csv(mean_path)
    raw_rows = []
    for _, r in raw.iterrows():
        row = r.to_dict()
        row["sgc_scope"] = "existing_industrial_full_evaluation"
        row["medical_semantic_status"] = "NOT_RUN_LARGE_CLIP_LRP_COMPUTE_REQUIRED"
        raw_rows.append(row)
    comp_rows = []
    for _, r in mean.iterrows():
        row = r.to_dict()
        row["scope"] = "industrial_8_transfer_existing_text_state_audit"
        row["medical_semantic_status"] = "PROTOCOL_BLOCKED_PENDING_EXPLICIT_FULL_RUN"
        comp_rows.append(row)
    for target in MEDICAL_TARGETS:
        comp_rows.append(
            {
                "variant": "Medical_FrozenSemanticProbe",
                "dataset": target,
                "mean_delta_P_AP_vs_E0": "NOT_RUN",
                "scope": "medical",
                "medical_semantic_status": "NOT_RUN: full medical semantic LRP requires per-image CLIP global/local inference; no persisted semantic maps exist.",
            }
        )
    return raw_rows, comp_rows


def write_reports(
    medical_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    semantic_comp: list[dict[str, Any]],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = [
        "# SGC Preflight Audit",
        "",
        "This phase is read/consolidation-only unless explicitly rerun. It does not modify CES/NTR/LRP checkpoints.",
        "",
        "## Available",
        "- Medical full-resolution metric rows for `Full VisA -> CVC-ClinicDB / CVC-ColonDB / Kvasir`: Ours, VisualAD, FB-CLIP; AnomalyCLIP P-AP-only legacy row.",
        "- Existing frozen text-state complementarity audit on industrial 8 transfers.",
        "- Text/visual alignment path: frozen CLIP text direction is lifted from 768 to 1024 patch state space with `clip_model.visual.proj` in `phase_text_state_complementarity_audit.py`.",
        "",
        "## Blocked / Not Available",
        "- Medical paired pixel complementarity is blocked because aligned per-image raw maps were not persisted for CESAD/FB-CLIP/VisualAD in the current artifacts.",
        "- AnomalyCLIP cannot be used for paired pixel analysis: current wrapper failed at prompt tensor dimensionality, and no aligned raw maps are available.",
        "- Medical frozen semantic probe was not run in this consolidation because it requires full global+local CLIP inference for three medical targets; SGC protocol requires a preflight estimate before large compute.",
        "",
    ]
    (OUT / "PREFLIGHT_AUDIT.md").write_text("\n".join(preflight), encoding="utf-8")

    err_lines = [
        "# Medical Error Report",
        "",
        "Only metric-level evidence is available. Raw map paired analysis is marked blocked rather than inferred.",
        "",
        "| Dataset | CES P-AP | FB-CLIP P-AP | VisualAD P-AP | FB-CES | VisualAD-CES | anomaly pixel ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in medical_rows:
        err_lines.append(
            f"| {r['dataset']} | {fmt(r['ces_P_AP'])} | {fmt(r['fbclip_P_AP'])} | {fmt(r['visualad_P_AP'])} | "
            f"{fmt(r['fbclip_minus_ces_P_AP'])} | {fmt(r['visualad_minus_ces_P_AP'])} | {fmt(r['anomaly_pixel_ratio'])} |"
        )
    err_lines += [
        "",
        "Metric-level observation: FB-CLIP is higher than CESAD-Full on all three medical stress targets. This does not by itself identify whether the gain comes from boundary coverage, foreground coverage, or score calibration, because raw aligned maps are unavailable.",
    ]
    (OUT / "MEDICAL_ERROR_REPORT.md").write_text("\n".join(err_lines), encoding="utf-8")

    bias_lines = [
        "# Source Transfer Bias Report",
        "",
        "Source-bias evidence is association-only in this consolidation.",
        "",
        "- Full VisA and Full MVTec source models use different fixed preprocessing paths; VisA-source includes PC1 removal and a frozen rank-4 adapter, while MVTec-source does not.",
        "- Therefore performance differences between sources cannot be interpreted as causal evidence that one source dataset is intrinsically better.",
        "- Source-category patch AP / hard-normal concentration was not recomputed in this SGC consolidation.",
    ]
    (OUT / "SOURCE_TRANSFER_BIAS_REPORT.md").write_text("\n".join(bias_lines), encoding="utf-8")

    final_lines = [
        "# SGC Final Report",
        "",
        "OUTCOME: `PROTOCOL_OR_RESOURCE_BLOCKED`",
        "",
        "## Answers",
        "",
        "Q1/Q2: Current metric-level evidence shows CESAD-Full underperforms FB-CLIP on ClinicDB, ColonDB, and Kvasir, but the stored artifacts do not contain aligned raw maps. Therefore the audit cannot reliably decide whether the main error is false-negative anomaly coverage, boundary localization, or background false positives.",
        "",
        "Q3: FB-CLIP has higher P-AP on all three medical stress targets, but independent pixel evidence in CES-failure regions is not computable without raw maps.",
        "",
        "Q4/Q5: Existing frozen text-state audit on industrial transfers shows text-only is much worse and equal fusion does not improve transfer on average. Medical frozen semantic probe was not run because it would require a new full CLIP global/local inference pass; no medical semantic maps exist.",
        "",
        "Q6: Industrial 8-transfer evidence does not support frozen text complementarity: mean P-AP delta for equal fusion is negative.",
        "",
        "Q7: Source bias remains association-only due to source-specific preprocessing confounds.",
        "",
        "## Recommendation",
        "",
        "D. Existing evidence is insufficient to justify adding a frozen semantic prior to CESAD. If this route is revisited, first persist aligned CESAD/FB-CLIP/VisualAD medical raw maps and run a small semantic-probe smoke test before full evaluation.",
    ]
    (OUT / "SGC_FINAL_REPORT.md").write_text("\n".join(final_lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    medical_rows = build_medical_error_audit()
    paired_rows = build_paired_complementarity()
    source_rows = build_source_category_diagnostics()
    semantic_raw, semantic_comp = copy_semantic_metrics()

    write_csv(OUT / "MEDICAL_ERROR_AUDIT.csv", medical_rows)
    write_csv(OUT / "PAIRED_COMPLEMENTARITY.csv", paired_rows)
    write_csv(OUT / "SOURCE_CATEGORY_DIAGNOSTICS.csv", source_rows)
    write_json(
        OUT / "SEMANTIC_PROBE_CONFIG.json",
        {
            "normal_prompts": NORMAL_PROMPTS,
            "anomaly_prompts": ANOMALY_PROMPTS,
            "text_visual_alignment": "Existing phase_text_state_complementarity_audit lifts CLIP 768-d text direction to 1024-d patch state space with frozen visual.proj.",
            "medical_probe_status": "NOT_RUN_LARGE_CLIP_LRP_COMPUTE_REQUIRED",
            "target_tuning": False,
        },
    )
    write_csv(OUT / "SEMANTIC_PROBE_RAW_METRICS.csv", semantic_raw)
    write_csv(OUT / "SEMANTIC_PROBE_COMPARISON.csv", semantic_comp)
    write_reports(medical_rows, paired_rows, source_rows, semantic_comp)
    write_json(
        OUT / "OUTCOME.json",
        {
            "outcome": "PROTOCOL_OR_RESOURCE_BLOCKED",
            "reason": "Medical raw maps and full medical frozen-semantic maps are not available in persisted artifacts; industrial text audit shows no stable benefit.",
            "output_dir": str(OUT),
        },
    )
    print(OUT)


if __name__ == "__main__":
    main()
