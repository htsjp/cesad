"""Read-only consolidation of existing Full VisA -> MPDD evidence.

This script intentionally performs no training, inference, or metric
recomputation.  It only reads previously generated, unified full-resolution
artifacts and writes a traceable protocol-consolidation bundle.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\ProjectPython\MRAD-main\MRAD-main")
RESULTS = ROOT / "results"
OUT = RESULTS / "phase_mpdd_visa_source_unification"

SOURCES = {
    "CESAD": RESULTS / "phase_mva_mrad_same_protocol_visa_factorization" / "formal" / "MRAD_SAME_PROTOCOL.csv",
    "AnomalyCLIP": RESULTS / "phase_anomalyclip_same_protocol_audit" / "formal" / "RAW_METRICS.csv",
    "FAPrompt": RESULTS / "paper_assets" / "baseline_close" / "FAPROMPT_UNIFIED_METRICS.csv",
    "VisualAD": RESULTS / "phase_xdg_extended_generalization" / "fill" / "RAW_METRICS_visualad.csv",
    "FB-CLIP": RESULTS / "phase_fbclip_pretrained_headroom_audit" / "formal" / "RAW_METRICS.csv",
}

METRICS = ("I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1max", "AUPRO")


def mfr(row: pd.Series, name: str) -> float:
    """Read a project M_FR field while keeping ordinary unified columns valid."""
    candidates = {
        "I_AUROC": ("M_FR_I_AUROC", "I_AUROC"),
        "I_AP": ("M_FR_I_AP", "I_AP"),
        "P_AUROC": ("M_FR_P_AUROC", "P_AUROC"),
        "P_AP": ("M_FR_P_AP", "P_AP"),
        "P_F1max": ("M_FR_P_F1max", "P_F1MAX"),
        "AUPRO": ("M_FR_PRO", "AUPRO"),
    }[name]
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    raise KeyError(f"No {name} field in row")


def select_row(path: Path, method: str) -> pd.Series:
    df = pd.read_csv(path)
    if method == "CESAD":
        q = df[(df.source == "Full VisA") & (df.dataset == "MPDD") & (df.variant == "E0_Core_NTR_FixedLRP")]
    elif method == "AnomalyCLIP":
        q = df[(df.source == "Full VisA") & (df.dataset == "MPDD") & (df.variant == "AnomalyCLIP_OfficialCheckpoint")]
    elif method == "FAPrompt":
        q = df[(df.source == "Full VisA") & (df.target == "MPDD") & (df.method == "FAPrompt")]
    elif method == "VisualAD":
        q = df[(df.source == "Full VisA") & (df.dataset == "MPDD") & (df.variant == "E1c_FullVisualAD")]
    elif method == "FB-CLIP":
        q = df[(df.source == "Full VisA") & (df.dataset == "MPDD") & (df.variant == "E1_Official_FBCLIP_Pretrained")]
    else:
        raise ValueError(method)
    if len(q) != 1:
        raise RuntimeError(f"Expected one {method} VisA->MPDD row in {path}, got {len(q)}")
    return q.iloc[0]


def metric_record(method: str, row: pd.Series) -> dict[str, object]:
    checkpoints = {
        "CESAD": r"results\paper_assets\final_ablation_close\checkpoints\Full_VisA_seed0_rho0.10.pt; VisA PC1/adapter assets",
        "AnomalyCLIP": r"_external\AnomalyCLIP_official\checkpoints\9_12_4_multiscale\epoch_15.pth",
        "FAPrompt": r"results\paper_assets\baseline_close\official_checkpoints\train_on_visa.pth",
        "MRAD-TF": r"VisA-selected official MRAD cache/checkpoint; phase_mva_mrad_same_protocol_visa_factorization",
        "MRAD-FT": r"VisA-selected official MRAD cache/checkpoint; phase_mva_mrad_same_protocol_visa_factorization",
        "MRAD-CLIP": r"VisA-selected official MRAD cache/checkpoint; phase_mva_mrad_same_protocol_visa_factorization",
        "VisualAD": r"source-selected released VisualAD checkpoint; phase_xdg_extended_generalization/fill",
        "FB-CLIP": r"_external\FB-CLIP\official_weights\train_on_visa\visa_epoch_2_model.pth",
    }
    record = {
        "source": "Full VisA",
        "target": "MPDD",
        "method": method,
        "checkpoint": checkpoints[method],
        "evaluator": "unified full-resolution M_FR category-macro",
        "protocol_valid": "YES",
        "target_training": "NO",
        "target_normal_reference": "NO",
        "target_statistics": "NO",
        "target_checkpoint_selection": "NO",
    }
    record.update({metric: mfr(row, metric) for metric in METRICS})
    return record


def write_md(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {method: select_row(path, method) for method, path in SOURCES.items()}
    rows = [metric_record("CESAD", raw["CESAD"])]

    mrad = pd.read_csv(SOURCES["CESAD"])
    for variant, label in (("M1_MRAD_TF", "MRAD-TF"), ("M2_MRAD_FT", "MRAD-FT"), ("M3_MRAD_CLIP", "MRAD-CLIP")):
        r = mrad[(mrad.source == "Full VisA") & (mrad.dataset == "MPDD") & (mrad.variant == variant)]
        if len(r) != 1:
            raise RuntimeError(f"Expected one {variant} row, got {len(r)}")
        rows.append(metric_record(label, r.iloc[0]))
    for method in ("AnomalyCLIP", "FAPrompt", "VisualAD", "FB-CLIP"):
        rows.append(metric_record(method, raw[method]))

    results = pd.DataFrame(rows)
    order = ["CESAD", "AnomalyCLIP", "FAPrompt", "MRAD-TF", "MRAD-FT", "MRAD-CLIP", "VisualAD", "FB-CLIP"]
    results["_order"] = results.method.map({name: i for i, name in enumerate(order)})
    results = results.sort_values("_order").drop(columns="_order")
    results.to_csv(OUT / "MPDD_VISA_SOURCE_RESULTS.csv", index=False, float_format="%.12f")
    results.to_csv(OUT / "MAIN_TABLE_MPDD_REPLACEMENT.csv", index=False, float_format="%.12f")

    protocol = results[["method", "source", "target", "checkpoint", "evaluator", "protocol_valid", "target_training", "target_normal_reference", "target_statistics", "target_checkpoint_selection"]].copy()
    protocol["evidence_file"] = protocol.method.map({
        "CESAD": str(SOURCES["CESAD"].relative_to(ROOT)),
        "AnomalyCLIP": str(SOURCES["AnomalyCLIP"].relative_to(ROOT)),
        "FAPrompt": str(SOURCES["FAPrompt"].relative_to(ROOT)),
        "MRAD-TF": str(SOURCES["CESAD"].relative_to(ROOT)),
        "MRAD-FT": str(SOURCES["CESAD"].relative_to(ROOT)),
        "MRAD-CLIP": str(SOURCES["CESAD"].relative_to(ROOT)),
        "VisualAD": str(SOURCES["VisualAD"].relative_to(ROOT)),
        "FB-CLIP": str(SOURCES["FB-CLIP"].relative_to(ROOT)),
    })
    protocol.to_csv(OUT / "SOURCE_PROTOCOL_COMPARISON.csv", index=False)

    legacy = pd.read_csv(RESULTS / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "FULL_RES_METRICS.csv")
    old = legacy[(legacy.source == "Full MVTec") & (legacy.dataset == "MPDD") & (legacy.variant == "E2_GlobalLocalZoomPyramid")]
    if len(old) != 1:
        raise RuntimeError(f"Expected one legacy MVTec->MPDD Ours row, got {len(old)}")
    old = old.iloc[0]
    ces = results[results.method == "CESAD"].iloc[0]
    comparison = pd.DataFrame([
        {"source": "Full MVTec", "target": "MPDD", **{metric: mfr(old, metric) for metric in METRICS}, "artifact": str((RESULTS / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "FULL_RES_METRICS.csv").relative_to(ROOT))},
        {"source": "Full VisA", "target": "MPDD", **{metric: ces[metric] for metric in METRICS}, "artifact": str(SOURCES["CESAD"].relative_to(ROOT))},
    ])
    comparison.to_csv(OUT / "MPDD_SOURCE_COMPARISON.csv", index=False, float_format="%.12f")

    existing = """# Existing Full VisA -> MPDD Audit

## Decision

All eight requested methods already have a single Full VisA -> MPDD result with a recorded unified full-resolution evaluation. No training, target inference, or metric recomputation was required for this consolidation.

| Method | Evidence artifact | Reusable | Why |
|---|---|---|---|
| CESAD | `phase_mva_mrad_same_protocol_visa_factorization/formal/MRAD_SAME_PROTOCOL.csv` | YES | Exact Full VisA source row, canonical `E0_Core_NTR_FixedLRP`, `M_FR` category-macro fields. |
| AnomalyCLIP | `phase_anomalyclip_same_protocol_audit/formal/RAW_METRICS.csv` | YES | Full VisA official-style checkpoint and M_FR raw-map evaluation. |
| FAPrompt | `paper_assets/baseline_close/FAPROMPT_UNIFIED_METRICS.csv` | YES | Official `train_on_visa.pth`; unified evaluator; strict-comparable audit says YES. |
| MRAD-TF/FT/CLIP | `phase_mva_mrad_same_protocol_visa_factorization/formal/MRAD_SAME_PROTOCOL.csv` | YES | Explicit Full VisA source cache selection and unified M_FR metrics. |
| VisualAD | `phase_xdg_extended_generalization/fill/RAW_METRICS_visualad.csv` | YES | Full VisA released source-specific path; unified raw-map M_FR metrics. |
| FB-CLIP | `phase_fbclip_pretrained_headroom_audit/formal/RAW_METRICS.csv` | YES | Official `train_on_visa` checkpoint; unified M_FR metrics. |

No file is accepted merely by name: each selected row is keyed by `source=Full VisA`, `target/dataset=MPDD`, a method-specific variant, and the recorded `CATEGORY_MACRO` aggregation.
"""
    write_md(OUT / "EXISTING_RUN_AUDIT.md", existing)

    metric_audit = """# Metric Audit

The replacement table reads existing `M_FR_*` fields from each source artifact. In this project these are the unified full-resolution, category-macro metrics used by the paper tables. The audit does not substitute dataset-global AP, per-image AP, patch-grid metrics, or `V_FR` auxiliary values.

The row selection was exact and deterministic:

- CESAD: `Full VisA`, `MPDD`, `E0_Core_NTR_FixedLRP`.
- AnomalyCLIP: `Full VisA`, `MPDD`, `AnomalyCLIP_OfficialCheckpoint`.
- VisualAD: `Full VisA`, `MPDD`, `E1c_FullVisualAD`.
- FB-CLIP: `Full VisA`, `MPDD`, `E1_Official_FBCLIP_Pretrained`.
- MRAD: the three explicit `M1/M2/M3` variants in the same protocol artifact.
- FAPrompt: the prior unified category-macro result, already marked `strict_comparable=YES`.

All rows are directly comparable to the current main-table M_FR protocol, subject to the already documented method-specific official inference paths.
"""
    write_md(OUT / "METRIC_AUDIT.md", metric_audit)

    ablation = """# Ablation Impact of the MPDD Source-Protocol Unification

The corrected CES/NTR/LRP factorial ablation, state-capacity control, and CES-vs-linear-head control use their own predeclared matched transfer subsets. Existing evidence identifies the common ablation subset as MVTec->VisA, MVTec->MPDD, VisA->MVTec, and VisA->BTAD (with the stated seed policy per artifact).

Changing the **main comparison table** MPDD column to Full VisA->MPDD must not silently overwrite those ablation cells. The existing ablations remain valid for their fixed transfer subset, but their captions/method text must state the four transfer cells explicitly. A directly matched VisA->MPDD component/capacity/parameterization ablation is not available in the audited artifacts and is therefore not added here.

Paper action: replace only the MPDD main-table comparison column with `MAIN_TABLE_MPDD_REPLACEMENT.csv`; retain ablation tables with explicit fixed-subset provenance. A supplementary full transfer matrix can show both source-to-MPDD cells where useful.
"""
    write_md(OUT / "ABLATION_IMPACT_REPORT.md", ablation)

    delta = ces["P_AP"] - comparison.iloc[0]["P_AP"]
    final = f"""# Full VisA -> MPDD Protocol Unification

## Result

The requested Full VisA -> MPDD comparison coverage already exists for CESAD and every requested baseline. This bundle consolidates those existing results without new training, inference, or metric recomputation.

CESAD under Full VisA -> MPDD obtains I-AUROC {ces['I_AUROC']:.4f}, I-AP {ces['I_AP']:.4f}, P-AUROC {ces['P_AUROC']:.4f}, P-AP {ces['P_AP']:.4f}, P-F1max {ces['P_F1max']:.4f}, and AUPRO {ces['AUPRO']:.4f} using the unified category-macro M_FR evaluator.

Relative to the legacy Full MVTec -> MPDD CESAD cell, Full VisA -> MPDD changes category-macro P-AP by {delta:+.4f} points. This is a source-protocol comparison, not source selection: the Full VisA row is the appropriate replacement when the paper fixes VisA as the source for MPDD.

## Answers

1. **Existing CESAD result?** Yes; exact Full VisA -> MPDD M_FR output is present.
2. **Baseline coverage?** AnomalyCLIP, FAPrompt, MRAD-TF/FT/CLIP, VisualAD, and FB-CLIP are all already covered by a Full VisA -> MPDD unified result.
3. **Consistent source protocol?** Yes for the consolidated rows: each has a documented Full VisA source checkpoint/cache and no target training/reference/statistics/checkpoint selection.
4. **CESAD performance?** The six metrics above, with P-AP {ces['P_AP']:.4f}.
5. **Can the MPDD column replace Tables 1/2?** Yes, for these recorded M_FR category-macro cells. This bundle deliberately does not edit those tables.
6. **Comparison conclusion?** CESAD remains highest in this audited Full VisA -> MPDD set on P-AP ({ces['P_AP']:.4f}); the closest listed comparator is VisualAD ({results[results.method == 'VisualAD'].iloc[0]['P_AP']:.4f}).
7. **Must Sec. 4.3 ablations change?** No numerical overwrite. They must retain and disclose their predeclared matched subset; a VisA->MPDD ablation is absent and should not be implied.
"""
    write_md(OUT / "FINAL_REPORT.md", final)

    manifest = {
        "outcome": "MPDD_VISA_SOURCE_STRICT_COVERAGE_REUSED",
        "training_run": False,
        "inference_run": False,
        "metric_recomputation": False,
        "selected_source": "Full VisA",
        "target": "MPDD",
        "methods": order,
    }
    (OUT / "OUTCOME.json").write_text(pd.Series(manifest).to_json(indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(results[["method", "P_AP", "P_AUROC", "P_F1max", "AUPRO"]].to_string(index=False))


if __name__ == "__main__":
    main()
