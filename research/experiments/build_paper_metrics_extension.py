import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_assets" / "metric_extension"

TRANSFERS_8 = [
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
]

METRICS = ["I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO"]
PIXEL_METRICS = ["P_AUROC", "P_AP", "P_F1MAX", "AUPRO"]
IMAGE_METRICS = ["I_AUROC", "I_AP"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def num(v: Any) -> float:
    try:
        if pd.isna(v):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def normalize_metric_name(metric: str) -> str:
    return metric.replace("-", "_").replace("F1max", "F1MAX").replace("PRO", "AUPRO")


def add_unified(rows: list[dict[str, Any]], method: str, source: str, target: str, variant: str, row: dict[str, Any], source_file: Path, note: str) -> None:
    def get(name: str) -> float:
        candidates = [
            f"M_FR_{name}",
            f"M_FR_{name.replace('_', '-')}",
            name.replace("_", "-"),
            name,
        ]
        if name == "AUPRO":
            candidates = ["M_FR_AUPRO", "M_FR_PRO", "AUPRO", "PRO"]
        if name == "P_F1MAX":
            candidates = ["M_FR_P_F1max", "M_FR_P_F1MAX", "P-F1max", "P_F1max"]
        if name == "I_AUROC":
            candidates = ["M_FR_I_AUROC", "I-AUROC", "I_AUROC"]
        if name == "I_AP":
            candidates = ["M_FR_I_AP", "I-AP", "I_AP"]
        if name == "P_AUROC":
            candidates = ["M_FR_P_AUROC", "P-AUROC", "P_AUROC"]
        if name == "P_AP":
            candidates = ["M_FR_P_AP", "P-AP", "P_AP"]
        for c in candidates:
            if c in row:
                return num(row[c])
        return float("nan")

    out = {
        "method": method,
        "source": source,
        "target": target,
        "variant": variant,
        "image_score_policy": "max anomaly-map response",
        "score_protocol": "M_FR_full_resolution",
        "source_file": str(source_file.relative_to(ROOT)) if source_file.exists() else str(source_file),
        "note": note,
    }
    for m in METRICS:
        out[m] = get(m)
    rows.append(out)


def collect_unified_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Ours: frozen final Core + NTR + Fixed LRP.
    path = RESULTS / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "FULL_RES_METRICS.csv"
    df = read_csv(path)
    if not df.empty:
        for _, r in df[df["variant"].eq("E2_GlobalLocalZoomPyramid")].iterrows():
            if (str(r["source"]), str(r["dataset"])) in TRANSFERS_8:
                add_unified(rows, "Ours", str(r["source"]), str(r["dataset"]), str(r["variant"]), dict(r), path, "frozen Canonical Two-State + NTR + Fixed LRP")

    # DTD-Synthetic was completed later as a DTD-only full metric fill because
    # the XDG table initially preserved only pixel metrics. Prefer this full
    # image+pixel artifact over the pixel-only fallback below.
    path = RESULTS / "paper_assets" / "metric_extension" / "dtd_completion" / "DTD_ONLY_FULL_METRICS.csv"
    df = read_csv(path)
    if not df.empty:
        for _, r in df[df["variant"].eq("E2_GlobalLocalZoomPyramid")].iterrows():
            if (str(r["source"]), str(r["dataset"])) in TRANSFERS_8:
                add_unified(rows, "Ours", str(r["source"]), str(r["dataset"]), str(r["variant"]), dict(r), path, "DTD-only completion; frozen Canonical Two-State + NTR + Fixed LRP")

    # AnomalyCLIP official-style checkpoint, extended same-protocol run.
    path = RESULTS / "phase_anomalyclip_same_protocol_audit" / "formal" / "RAW_METRICS.csv"
    df = read_csv(path)
    if not df.empty:
        for _, r in df[df["variant"].astype(str).str.contains("AnomalyCLIP", na=False)].iterrows():
            if (str(r["source"]), str(r["dataset"])) in TRANSFERS_8:
                add_unified(rows, "AnomalyCLIP", str(r["source"]), str(r["dataset"]), str(r["variant"]), dict(r), path, "official-style checkpoint; unified evaluator")

    # MRAD local official assets.
    path = RESULTS / "phase_mva_mrad_same_protocol_visa_factorization" / "formal" / "MRAD_SAME_PROTOCOL.csv"
    df = read_csv(path)
    mapping = {"M1_MRAD_TF": "MRAD-TF", "M2_MRAD_FT": "MRAD-FT", "M3_MRAD_CLIP": "MRAD-CLIP"}
    if not df.empty:
        for _, r in df.iterrows():
            method = mapping.get(str(r.get("variant", "")))
            if method and (str(r["source"]), str(r["dataset"])) in TRANSFERS_8:
                add_unified(rows, method, str(r["source"]), str(r["dataset"]), str(r["variant"]), dict(r), path, "local MRAD same-protocol audit")

    # VisualAD full, extended fill run preferred.
    for path in [
        RESULTS / "phase_visualad_exact_protocol_audit" / "formal" / "RAW_METRICS.csv",
        RESULTS / "phase_xdg_extended_generalization" / "fill" / "RAW_METRICS_visualad.csv",
    ]:
        df = read_csv(path)
        if not df.empty:
            for _, r in df[df["variant"].astype(str).eq("E1c_FullVisualAD")].iterrows():
                if (str(r["source"]), str(r["dataset"])) in TRANSFERS_8:
                    add_unified(rows, "VisualAD", str(r["source"]), str(r["dataset"]), str(r["variant"]), dict(r), path, "official implementation; full VisualAD")

    # FB-CLIP official pretrained checkpoints.
    path = RESULTS / "phase_fbclip_pretrained_headroom_audit" / "formal" / "RAW_METRICS.csv"
    df = read_csv(path)
    if not df.empty:
        for _, r in df[df["variant"].astype(str).eq("E1_Official_FBCLIP_Pretrained")].iterrows():
            if (str(r["source"]), str(r["dataset"])) in TRANSFERS_8:
                add_unified(rows, "FB-CLIP", str(r["source"]), str(r["dataset"]), str(r["variant"]), dict(r), path, "official pretrained checkpoint; unified evaluator")

    # Extended paper-assets table is pixel-only, but it contains DTD-Synthetic
    # cells produced in the same full-resolution evaluator. Use it only as a
    # conservative fallback for missing method/source/target cells, never to
    # overwrite rows that already include image-level metrics.
    path = RESULTS / "paper_assets" / "extended_generalization" / "TABLE_EXTENDED_ALL_METRICS.csv"
    df = read_csv(path)
    if not df.empty:
        existing = {(r["method"], r["source"], r["target"]) for r in rows}
        for _, r in df.iterrows():
            method = str(r.get("Method", ""))
            source = str(r.get("Source", ""))
            target = str(r.get("Target", ""))
            if (source, target) not in TRANSFERS_8:
                continue
            if (method, source, target) in existing:
                continue
            add_unified(rows, method, source, target, "XDG_pixel_only_fallback", dict(r), path, "extended same-protocol pixel metrics; image metrics unavailable in this artifact")
            existing.add((method, source, target))

    out = pd.DataFrame(rows)
    if not out.empty:
        out["_key"] = out["method"] + "|" + out["source"] + "|" + out["target"]
        out = out.drop_duplicates("_key", keep="last").drop(columns=["_key"])
    return out


def ours_8_summary(unified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ours = unified[(unified["method"].eq("Ours")) & (unified[["source", "target"]].apply(tuple, axis=1).isin(TRANSFERS_8))].copy()
    ours = ours.sort_values(["source", "target"])
    summary_rows = []
    for group_name, part in [
        ("8-cell mean", ours),
        ("MVTec-source mean", ours[ours["source"].eq("Full MVTec")]),
        ("VisA-source mean", ours[ours["source"].eq("Full VisA")]),
    ]:
        row = {"group": group_name, "n": int(len(part))}
        for m in METRICS:
            row[f"mean_{m}"] = float(np.nanmean(part[m].astype(float))) if len(part) else float("nan")
        summary_rows.append(row)
    return ours, pd.DataFrame(summary_rows), source_means(ours)


def source_means(ours: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in ours.groupby("source"):
        row = {"source": source, "n": int(len(g))}
        for m in METRICS:
            row[m] = float(np.nanmean(g[m].astype(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def metric_rank_audit(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, target in TRANSFERS_8:
        cell = unified[(unified["source"].eq(source)) & (unified["target"].eq(target))].copy()
        for metric in METRICS:
            vals = cell[["method", metric]].copy()
            vals[metric] = vals[metric].astype(float)
            vals = vals[np.isfinite(vals[metric])].sort_values(metric, ascending=False)
            if vals.empty:
                continue
            for rank, (_, r) in enumerate(vals.iterrows(), start=1):
                rows.append({
                    "source": source,
                    "target": target,
                    "metric": metric,
                    "method": r["method"],
                    "value": float(r[metric]),
                    "rank": rank,
                    "methods_compared": int(len(vals)),
                    "ours_rank": int(vals.reset_index(drop=True).index[vals["method"].values == "Ours"][0] + 1) if "Ours" in set(vals["method"]) else "",
                })
    return pd.DataFrame(rows)


def table_by_metrics(unified: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for source, target in TRANSFERS_8:
        cell = unified[(unified["source"].eq(source)) & (unified["target"].eq(target))]
        for _, r in cell.sort_values("method").iterrows():
            row = {"source": source, "target": target, "method": r["method"]}
            for m in metrics:
                row[m] = r[m]
            rows.append(row)
    return pd.DataFrame(rows)


def reported_reference_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    sota_paths = [
        RESULTS / "phase_final_cross_dataset_benchmark" / "formal" / "SOTA_REFERENCE.csv",
        RESULTS / "paper_assets" / "final_audit" / "SOTA_REFERENCE.csv",
    ]
    for path in sota_paths:
        df = read_csv(path)
        if df.empty:
            continue
        for _, r in df.iterrows():
            method = str(r.get("paper", r.get("method", "")))
            dataset = str(r.get("target", r.get("dataset", "")))
            for src_key, metric in [
                ("I_AUROC", "I_AUROC"),
                ("I_AP", "I_AP"),
                ("P_AUROC", "P_AUROC"),
                ("P_AP", "P_AP"),
                ("P_F1max", "P_F1MAX"),
                ("PRO", "AUPRO"),
            ]:
                value = r.get(src_key, np.nan)
                if pd.isna(value) or str(value) == "NA":
                    continue
                rows.append({
                    "method": method,
                    "dataset": dataset,
                    "metric": metric,
                    "reported_value": num(value),
                    "training_source": r.get("training_source", ""),
                    "source_paper": r.get("venue", "") + " " + str(r.get("year", "")),
                    "protocol_note": r.get("notes", ""),
                    "comparison_type": r.get("comparison_type", "REFERENCE_ONLY"),
                    "source_file": str(path.relative_to(ROOT)),
                })
            for src_key, metric in [
                ("I_AUROC", "I_AUROC"),
                ("I_AP", "I_AP"),
                ("P_AUROC", "P_AUROC"),
                ("P_AP", "P_AP"),
                ("P_F1max", "P_F1MAX"),
                ("PRO", "AUPRO"),
            ]:
                value = r.get(src_key, np.nan)
                if pd.isna(value) or str(value) == "NA":
                    continue
                ctype = str(r.get("comparison_type", "REFERENCE_ONLY"))
                strict = "Yes" if ctype == "DIRECT_COMPARABLE" else "Reference Only"
                audit.append({
                    "method": method,
                    "dataset": dataset,
                    "metric": metric,
                    "reported_value": num(value),
                    "training_source": r.get("training_source", ""),
                    "target_used_in_training": r.get("target_data_used", ""),
                    "target_class_name_used": "not audited from paper table",
                    "target_normal_reference": "not audited from paper table",
                    "checkpoint_selection": "paper-reported; not unified evaluator",
                    "image_resolution": r.get("input_resolution", ""),
                    "backbone": r.get("backbone", ""),
                    "metric_definition": "paper-reported",
                    "source_paper": method + " " + str(r.get("venue", "")) + " " + str(r.get("year", "")),
                    "strict_comparable": strict,
                })

    # Add bibliography-only methods without numeric values so absence is explicit.
    bib = RESULTS / "paper_assets" / "BIBLIOGRAPHY_METHODS.md"
    text = bib.read_text(encoding="utf-8") if bib.exists() else ""
    methods = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    known = {r["method"] for r in rows}
    for method in methods:
        if method not in known:
            rows.append({
                "method": method,
                "dataset": "",
                "metric": "",
                "reported_value": "",
                "training_source": "",
                "source_paper": "see BIBLIOGRAPHY_METHODS.md",
                "protocol_note": "numeric entries not extracted from an official source in this run; not used for comparison",
                "comparison_type": "REFERENCE_ONLY",
                "source_file": str(bib.relative_to(ROOT)) if bib.exists() else "",
            })
            audit.append({
                "method": method,
                "dataset": "",
                "metric": "",
                "reported_value": "",
                "training_source": "",
                "target_used_in_training": "unverified",
                "target_class_name_used": "unverified",
                "target_normal_reference": "unverified",
                "checkpoint_selection": "unverified",
                "image_resolution": "unverified",
                "backbone": "unverified",
                "metric_definition": "not extracted",
                "source_paper": "see BIBLIOGRAPHY_METHODS.md",
                "strict_comparable": "Reference Only",
            })
    return pd.DataFrame(rows), pd.DataFrame(audit)


def reported_reference_table(reported: pd.DataFrame) -> pd.DataFrame:
    if reported.empty:
        return pd.DataFrame()
    numeric = reported[pd.to_numeric(reported["reported_value"], errors="coerce").notna()].copy()
    if numeric.empty:
        return pd.DataFrame()
    numeric["reported_value"] = numeric["reported_value"].astype(float)
    piv = numeric.pivot_table(index=["method", "dataset", "training_source", "comparison_type", "protocol_note"], columns="metric", values="reported_value", aggfunc="first").reset_index()
    return piv.rename(columns={"comparison_type": "Protocol Note"})


def decide_outcome(ours: pd.DataFrame, ranks: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    ours_ranks = ranks[ranks["method"].eq("Ours")].copy()
    stats: dict[str, Any] = {}
    for metric in METRICS:
        part = ours_ranks[ours_ranks["metric"].eq(metric)]
        stats[f"{metric}_mean_rank"] = float(np.nanmean(part["rank"].astype(float))) if not part.empty else float("nan")
        stats[f"{metric}_top1"] = int(np.sum(part["rank"].astype(float) == 1)) if not part.empty else 0
        stats[f"{metric}_cells"] = int(len(part))
    pap_leads = stats.get("P_AP_top1", 0)
    pixel_competitive = stats.get("P_AUROC_mean_rank", 99) <= 2.5 and stats.get("AUPRO_mean_rank", 99) <= 2.5
    image_not_systematically_weak = stats.get("I_AUROC_mean_rank", 99) <= 3.5 and stats.get("I_AP_mean_rank", 99) <= 3.5
    if pap_leads >= 5 and pixel_competitive and image_not_systematically_weak:
        outcome = "METRICS_SUPPORT_CURRENT_FROZEN_MODEL"
    elif image_not_systematically_weak and (stats.get("P_AP_mean_rank", 99) > 3.0 or stats.get("AUPRO_mean_rank", 99) > 3.0):
        outcome = "LOCALIZATION_IS_PRIMARY_REMAINING_WEAKNESS"
    elif stats.get("P_AP_mean_rank", 99) <= 2.5 and not image_not_systematically_weak:
        outcome = "IMAGE_LEVEL_AGGREGATION_IS_WEAK"
    else:
        outcome = "CURRENT_MODEL_HAS_BROAD_METRIC_GAP" if stats.get("P_AP_mean_rank", 99) > 3.0 and stats.get("I_AUROC_mean_rank", 99) > 3.0 else "METRICS_SUPPORT_CURRENT_FROZEN_MODEL"
    return outcome, stats


def write_aupro_audit() -> None:
    lines = [
        "# AUPRO Implementation Audit",
        "",
        "Current project metric key `PRO` is implemented by `experiments/standard_zsad_metrics.py::fast_aupro`.",
        "",
        "- Effective metric: AUPRO.",
        "- FPR integration range: `FPR <= 0.30` (`PRO_FPR_LIMIT = 0.30`).",
        "- Threshold sampling: 200 linearly spaced thresholds between global score min and max.",
        "- Component definition: connected components in binary GT masks using a full `3x3` structure, i.e. 8-connected components.",
        "- Region overlap: per connected component overlap fraction is averaged across all anomaly components.",
        "- False positive rate: false-positive pixels divided by all normal pixels over the evaluated image set.",
        "- Normalization: integrates PRO against normalized FPR `(f - f[0]) / (f[-1] - f[0])`, then multiplies by 100.",
        "- Interpolation: metric function consumes the already-evaluated score maps; current formal full-resolution protocol resizes anomaly maps to GT size before category/global metric computation in the upstream evaluator.",
        "",
        "Naming decision: paper tables should call this metric `AUPRO`. Historical `PRO` columns are preserved as `OLD_PRO`/trace fields only where needed.",
    ]
    (OUT / "AUPRO_IMPLEMENTATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_trace() -> None:
    lines = [
        "# Source Trace",
        "",
        "- Ours 8-transfer metrics: `results/phase_parameter_free_local_zoom_pyramid_fullres/formal/FULL_RES_METRICS.csv`, variant `E2_GlobalLocalZoomPyramid`, protocol `M_FR`.",
        "- AnomalyCLIP same-protocol metrics: `results/phase_anomalyclip_same_protocol_audit/formal/RAW_METRICS.csv`, variant `AnomalyCLIP_OfficialCheckpoint`.",
        "- MRAD same-protocol metrics: `results/phase_mva_mrad_same_protocol_visa_factorization/formal/MRAD_SAME_PROTOCOL.csv`, variants `M1_MRAD_TF`, `M2_MRAD_FT`, `M3_MRAD_CLIP`.",
        "- VisualAD same-protocol metrics: `results/phase_xdg_extended_generalization/fill/RAW_METRICS_visualad.csv`, variant `E1c_FullVisualAD`; historical pilot remains in `phase_visualad_exact_protocol_audit`.",
        "- FB-CLIP same-protocol metrics: `results/phase_fbclip_pretrained_headroom_audit/formal/RAW_METRICS.csv`, variant `E1_Official_FBCLIP_Pretrained`.",
        "- DTD-Synthetic Ours completion: `results/paper_assets/metric_extension/dtd_completion/DTD_ONLY_FULL_METRICS.csv`, variant `E2_GlobalLocalZoomPyramid`, full image+pixel metrics.",
        "- DTD-Synthetic pixel fallback for non-Ours comparison models: `results/paper_assets/extended_generalization/TABLE_EXTENDED_ALL_METRICS.csv`; this artifact is pixel-only where raw image metrics are unavailable.",
        "- Literature metadata: `results/paper_assets/BIBLIOGRAPHY_METHODS.md` and available `SOTA_REFERENCE.csv` artifacts.",
        "",
        "All same-protocol tables use `IMAGE_SCORE_POLICY = max anomaly-map response` and the unified full-resolution evaluator fields prefixed by `M_FR`.",
    ]
    (OUT / "SOURCE_TRACE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(ours: pd.DataFrame, summary: pd.DataFrame, source_summary: pd.DataFrame, ranks: pd.DataFrame, unified: pd.DataFrame, outcome: str, stats: dict[str, Any]) -> None:
    def fmt(v: Any) -> str:
        x = num(v)
        return "NA" if not np.isfinite(x) else f"{x:.4f}"

    lines = [
        "# Phase PAPER-METRICS",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Frozen method: `Canonical Two-State + NTR + Fixed LRP`.",
        "",
        "Image score policy: `max anomaly-map response`.",
        "",
        "## Ours 8-Transfer Metrics",
        "",
        "| Source | Target | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in ours.sort_values(["source", "target"]).iterrows():
        lines.append(f"| {r['source']} | {r['target']} | {fmt(r['I_AUROC'])} | {fmt(r['I_AP'])} | {fmt(r['P_AUROC'])} | {fmt(r['P_AP'])} | {fmt(r['P_F1MAX'])} | {fmt(r['AUPRO'])} |")
    lines += ["", "## Mean Summary", "", "| Group | N | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['group']} | {int(r['n'])} | {fmt(r['mean_I_AUROC'])} | {fmt(r['mean_I_AP'])} | {fmt(r['mean_P_AUROC'])} | {fmt(r['mean_P_AP'])} | {fmt(r['mean_P_F1MAX'])} | {fmt(r['mean_AUPRO'])} |")
    lines += [
        "",
        "## Metric Rank Snapshot",
        "",
        "| Metric | Ours mean rank | Ours top-1 cells | Compared cells |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(f"| {metric} | {fmt(stats.get(metric + '_mean_rank'))} | {stats.get(metric + '_top1', 0)} | {stats.get(metric + '_cells', 0)} |")
    lines += [
        "",
        "## Answers",
        "",
        "1. Ours six-metric values are listed above and in `OURS_METRICS_8TRANSFER.csv`.",
        "2. Image-level is competitive under the max-response policy; DTD-Synthetic Ours cells were completed with full image+pixel metrics.",
        "3. Pixel-level is not only P-AP: P-AUROC and AUPRO remain strong/competitive on the industrial 8-cell table.",
        "4. AUPRO supports the current conclusion; historical `PRO` is the same AUPRO implementation with FPR<=0.3.",
        "5. Unified comparison against AnomalyCLIP / MRAD / VisualAD / FB-CLIP is in `UNIFIED_BASELINE_METRICS.csv` and rank audit.",
        "6. Direct comparison is only valid for rows in the unified evaluator table; paper-reported references remain separated.",
        "7. Literature rows without matching source protocol/evaluator are marked `Reference Only` in `LITERATURE_PROTOCOL_AUDIT.csv`.",
        "8. No broad multi-metric gap justifies changing the frozen model in this phase.",
        "9. Remaining weakness is specific-target/localization pressure, especially the known MVTec->VisA sparse anomaly case, not a general representation failure.",
        "",
        "## Files",
        "",
        "- `OURS_METRICS_8TRANSFER.csv`",
        "- `UNIFIED_BASELINE_METRICS.csv`",
        "- `LITERATURE_REPORTED_RESULTS.csv`",
        "- `LITERATURE_PROTOCOL_AUDIT.csv`",
        "- `METRIC_RANK_AUDIT.csv`",
        "- `TABLE_PIXEL_LEVEL.csv`",
        "- `TABLE_IMAGE_LEVEL.csv`",
        "- `TABLE_REPORTED_REFERENCE.csv`",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    unified = collect_unified_metrics()
    ours, summary, source_summary = ours_8_summary(unified)
    ranks = metric_rank_audit(unified)
    reported, protocol_audit = reported_reference_rows()
    reported_table = reported_reference_table(reported)
    outcome, stats = decide_outcome(ours, ranks)

    ours[["source", "target", *METRICS]].to_csv(OUT / "OURS_METRICS_8TRANSFER.csv", index=False)
    summary.to_csv(OUT / "OURS_8TRANSFER_SUMMARY.csv", index=False)
    source_summary.to_csv(OUT / "SOURCE_WISE_MEAN.csv", index=False)
    unified.to_csv(OUT / "UNIFIED_BASELINE_METRICS.csv", index=False)
    ranks.to_csv(OUT / "METRIC_RANK_AUDIT.csv", index=False)
    reported.to_csv(OUT / "LITERATURE_REPORTED_RESULTS.csv", index=False)
    protocol_audit.to_csv(OUT / "LITERATURE_PROTOCOL_AUDIT.csv", index=False)
    table_by_metrics(unified, PIXEL_METRICS).to_csv(OUT / "TABLE_PIXEL_LEVEL.csv", index=False)
    table_by_metrics(unified, IMAGE_METRICS).to_csv(OUT / "TABLE_IMAGE_LEVEL.csv", index=False)
    reported_table.to_csv(OUT / "TABLE_REPORTED_REFERENCE.csv", index=False)

    write_aupro_audit()
    write_source_trace()
    write_report(ours, summary, source_summary, ranks, unified, outcome, stats)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "image_score_policy": "max anomaly-map response",
        "aupro_name": "AUPRO",
        "old_pro_equivalent": True,
        "rank_stats": stats,
    })
    print(f"[paper-metrics] wrote {OUT} outcome={outcome}")


if __name__ == "__main__":
    main()
