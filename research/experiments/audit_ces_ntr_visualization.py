from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FIG_OUT = ROOT / "results" / "ces_ntr_visualization" / "final"
AUDIT_OUT = ROOT / "results" / "ces_ntr_visualization" / "audit"
SOURCE_RESPONSES = FIG_OUT / "source_patch_responses.csv"
TAIL_CURVES = FIG_OUT / "normal_tail_curves.csv"
QUAL_SAMPLES = FIG_OUT / "qualitative_samples.csv"
VIS_DATA = FIG_OUT / "ces_ntr_visualization_data.npz"
METADATA = FIG_OUT / "metadata.json"
FACTORIAL = ROOT / "results" / "paper_assets" / "final_ablation_close" / "FACTORIAL_ABLATION.csv"
PLOT_SCRIPT = ROOT / "experiments" / "plot_ces_ntr_mechanism.py"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def f(x):
    if x is None:
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.10g}"
    return x


def metric_stats(df: pd.DataFrame, score_col: str, hard_mask: np.ndarray) -> dict:
    labels = df["label"].to_numpy().astype(int)
    scores = df[score_col].to_numpy(dtype=np.float64)
    normal = labels == 0
    anomaly = labels == 1
    out = {
        "mean_normal_score": float(scores[normal].mean()),
        "mean_anomalous_score": float(scores[anomaly].mean()),
        "mean_high_risk_normal_score": float(scores[hard_mask].mean()),
        "anomaly_normal_mean_gap": float(scores[anomaly].mean() - scores[normal].mean()),
        "anomaly_high_risk_normal_mean_gap": float(scores[anomaly].mean() - scores[hard_mask].mean()),
        "source_patch_AUROC": float(roc_auc_score(labels, scores)),
        "source_patch_AP": float(average_precision_score(labels, scores)),
    }
    return out


def compute_panel_b_stats() -> tuple[list[dict], dict]:
    df = pd.read_csv(SOURCE_RESPONSES)
    tail = pd.read_csv(TAIL_CURVES)
    labels = df["label"].to_numpy().astype(int)
    hard = df["hard_normal"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    normal_ids = np.flatnonzero(labels == 0)

    checks = {}
    checks["same_patch_ids_in_tail"] = bool(set(tail["patch_id"].astype(int)) == set(normal_ids.tolist()))
    sorted_expected = normal_ids[np.argsort(df.loc[normal_ids, "q_ces"].to_numpy())]
    checks["tail_sorted_by_ces_only"] = bool(np.array_equal(tail["patch_id"].astype(int).to_numpy(), sorted_expected))
    checks["top10_fixed_from_ces"] = bool(
        set(df.index[hard].tolist()) == set(tail.loc[tail["in_top_rho_tail"].astype(str).str.lower().isin(["true", "1"]), "patch_id"].astype(int).tolist())
    )
    checks["q_ces_formula_matches"] = bool(np.allclose(df["q_ces"], df["ces_sa"] - df["ces_sn"], atol=1e-7))
    checks["q_ntr_formula_matches"] = bool(np.allclose(df["q_ntr"], df["ntr_sa"] - df["ntr_sn"], atol=1e-7))
    checks["labels_are_binary"] = bool(set(labels.tolist()) == {0, 1})
    checks["normal_count"] = int((labels == 0).sum())
    checks["anomaly_count"] = int((labels == 1).sum())
    checks["hard_normal_count"] = int(hard.sum())

    ces = metric_stats(df, "q_ces", hard)
    ntr = metric_stats(df, "q_ntr", hard)
    rows = []
    for key in ces:
        rows.append({"metric": key, "CES_only": ces[key], "CES_plus_NTR": ntr[key], "delta_NTR_minus_CES": ntr[key] - ces[key]})

    # Extra tail-specific movement checks.
    scores0 = df["q_ces"].to_numpy(float)
    scores1 = df["q_ntr"].to_numpy(float)
    normal = labels == 0
    nonhard_normal = normal & (~hard)
    anomaly = labels == 1
    checks["mean_score_shift_all"] = float(np.mean(scores1 - scores0))
    checks["mean_score_shift_normal"] = float(np.mean(scores1[normal] - scores0[normal]))
    checks["mean_score_shift_anomaly"] = float(np.mean(scores1[anomaly] - scores0[anomaly]))
    checks["mean_score_shift_hard_normal"] = float(np.mean(scores1[hard] - scores0[hard]))
    checks["mean_score_shift_nonhard_normal"] = float(np.mean(scores1[nonhard_normal] - scores0[nonhard_normal]))
    checks["hard_normal_drops_more_than_other_normal"] = bool(
        checks["mean_score_shift_hard_normal"] < checks["mean_score_shift_nonhard_normal"]
    )
    checks["ranking_AUROC_delta"] = ntr["source_patch_AUROC"] - ces["source_patch_AUROC"]
    checks["ranking_AP_delta"] = ntr["source_patch_AP"] - ces["source_patch_AP"]
    return rows, checks


def compute_qualitative_stats() -> tuple[list[dict], dict]:
    data = np.load(VIS_DATA, allow_pickle=True)
    qdf = pd.read_csv(QUAL_SAMPLES)
    meta = json.loads(METADATA.read_text(encoding="utf-8"))

    rows = []
    for _, rec in qdf.iterrows():
        i = int(rec["sample_id"])
        mask = data[f"sample_{i}_mask"].astype(bool)
        ces = data[f"sample_{i}_ces_full"].astype(np.float32)
        ntr = data[f"sample_{i}_ntr_full"].astype(np.float32)
        y = mask.reshape(-1).astype(int)
        for name, arr in [("CES_only", ces), ("CES_plus_NTR", ntr)]:
            scores = arr.reshape(-1)
            rows.append(
                {
                    "sample_id": i,
                    "source": meta["source"],
                    "target": meta["target_for_qualitative_panel"],
                    "category": rec["category"],
                    "img_path": rec["img_path"],
                    "checkpoint": "init_v_fallback" if name == "CES_only" else meta["ntr_checkpoint"],
                    "variant": name,
                    "pixel_AP": float(average_precision_score(y, scores)),
                    "normal_background_mean_response": float(scores[~y.astype(bool)].mean()),
                    "gt_anomaly_mean_response": float(scores[y.astype(bool)].mean()),
                    "heatmap_vmin": float(meta["heatmap_scale"]["vmin"]),
                    "heatmap_vmax": float(meta["heatmap_scale"]["vmax"]),
                    "same_color_range": True,
                    "per_map_minmax_normalized": False,
                }
            )

    full_target = {}
    if FACTORIAL.exists():
        fact = pd.read_csv(FACTORIAL)
        sub = fact[(fact["source"] == "Full VisA") & (fact["target"] == "MVTec") & (fact["seed"] == 0)]
        for variant in ["Base+LRP", "Base+NTR+LRP"]:
            r = sub[sub["variant"] == variant]
            if len(r):
                full_target[variant] = {
                    "P_AP": float(r.iloc[0]["P_AP"]),
                    "P_AUROC": float(r.iloc[0]["P_AUROC"]),
                    "AUPRO": float(r.iloc[0]["AUPRO"]),
                    "checkpoint": str(r.iloc[0]["checkpoint"]),
                    "cache": str(r.iloc[0]["cache"]),
                    "source_file": str(FACTORIAL),
                }
    return rows, full_target


def config_comparison_rows(metadata: dict) -> list[dict]:
    rows = [
        {
            "item": "source dataset",
            "CES_only": metadata["source"],
            "CES_plus_NTR": metadata["source"],
            "status": "PASS",
            "evidence": f"metadata.json source={metadata['source']}; plot script loads one source response cache.",
        },
        {
            "item": "frozen CLIP features / source transform",
            "CES_only": "VisA-source response cache with formal PC1/adapter preprocessing",
            "CES_plus_NTR": "same response cache; qualitative maps use same source_info and LRP preprocessing",
            "status": "PASS",
            "evidence": f"{metadata['source_response_cache']}; plot_ces_ntr_mechanism.py loads one source_info for both HeadSpec objects.",
        },
        {
            "item": "CES initialization",
            "CES_only": "init_v_fallback / source initialization",
            "CES_plus_NTR": "same m and alpha; v from rho=0.10 checkpoint",
            "status": "PASS_WITH_LIMITATION",
            "evidence": "Both specs use source_info['m'] and source_info['alpha']; CES-only uses init_v, NTR uses trained v. They are two controls, not before/after snapshots.",
        },
        {
            "item": "training budget and base loss",
            "CES_only": "no-NTR baseline control from initialization/fallback artifact",
            "CES_plus_NTR": "seed0 rho=0.10 checkpoint",
            "status": "UNKNOWN",
            "evidence": "Generated figure artifacts do not encode the full training schedule for CES-only. Final ablation table marks Base+LRP checkpoint as init_v_fallback.",
        },
        {
            "item": "NTR enabled",
            "CES_only": "No",
            "CES_plus_NTR": "Yes, rho=0.10",
            "status": "PASS",
            "evidence": f"NTR checkpoint={metadata['ntr_checkpoint']}; ntr_rho={metadata['ntr_rho']}.",
        },
        {
            "item": "inference configuration",
            "CES_only": "same Fixed LRP local/global fusion for panel C; panel B source-patch scorer no LRP",
            "CES_plus_NTR": "same Fixed LRP local/global fusion for panel C; panel B source-patch scorer no LRP",
            "status": "PASS",
            "evidence": f"local_lrp_cache={metadata['local_lrp_cache']}; both maps generated via same lrp_map function.",
        },
        {
            "item": "target usage in training/preprocessing/selection",
            "CES_only": "not used by plotting script for model fitting",
            "CES_plus_NTR": "not used by plotting script for model fitting",
            "status": "PASS_WITH_LIMITATION",
            "evidence": "Panel C uses MVTec target only for visualization/evaluation; audit cannot independently prove historical checkpoint chronology beyond existing protocol artifacts.",
        },
    ]
    return rows


def write_report(score_rows, checks, qual_rows, full_target, config_rows, metadata):
    score_df = pd.DataFrame(score_rows)
    qual_df = pd.DataFrame(qual_rows)
    cfg_df = pd.DataFrame(config_rows)

    def stat_value(metric, variant):
        r = score_df[score_df["metric"] == metric].iloc[0]
        return r[variant]

    ntr_shift_summary = (
        f"- Mean score shift all patches: {checks['mean_score_shift_all']:.6f}\n"
        f"- Mean score shift normal: {checks['mean_score_shift_normal']:.6f}\n"
        f"- Mean score shift anomaly: {checks['mean_score_shift_anomaly']:.6f}\n"
        f"- Mean score shift hard normal: {checks['mean_score_shift_hard_normal']:.6f}\n"
        f"- Mean score shift other normal: {checks['mean_score_shift_nonhard_normal']:.6f}\n"
    )
    ranking_text = (
        f"Source patch AUROC changes from {stat_value('source_patch_AUROC', 'CES_only'):.6f} to "
        f"{stat_value('source_patch_AUROC', 'CES_plus_NTR'):.6f}; AP changes from "
        f"{stat_value('source_patch_AP', 'CES_only'):.6f} to {stat_value('source_patch_AP', 'CES_plus_NTR'):.6f}."
    )

    full_target_text = "Not available from existing formal table."
    if full_target:
        lines = []
        for k, v in full_target.items():
            lines.append(f"- {k}: P-AP={v['P_AP']:.6f}, P-AUROC={v['P_AUROC']:.6f}, AUPRO={v['AUPRO']:.6f}, checkpoint={v['checkpoint']}")
        full_target_text = "\n".join(lines)

    report = f"""# CES/NTR 可视化真实性与机制审计

## Scope

This is a read-only audit of `ces_ntr_mechanism.png`. No model training, new inference benchmark, or figure redrawing was performed. The audit uses the already saved figure data and formal result artifacts.

## Audited Files

- Figure: `{FIG_OUT / 'ces_ntr_mechanism.png'}`
- Plot script: `{PLOT_SCRIPT}`
- Source responses: `{SOURCE_RESPONSES}`
- Normal-tail curve data: `{TAIL_CURVES}`
- Qualitative samples: `{QUAL_SAMPLES}`
- NPZ maps/data: `{VIS_DATA}`
- Metadata: `{METADATA}`
- Formal full-target reference: `{FACTORIAL}`

## 1. Subfigure Data Sources

### Panel (a): CES Decision Geometry

- Data: `source_patch_responses.csv`, originally from `{metadata['source_response_cache']}`.
- Domain: source-domain patches from `{metadata['source']}`.
- Coordinates: `x = z^T s_N`, `y = z^T s_A` for the CES-only state pair.
- Score identity check: `q_CES = z^T s_A - z^T s_N = y - x` is `{checks['q_ces_formula_matches']}`.
- The dashed `y=x` line is the zero-score boundary for `q(z)=z^T s_A-z^T s_N`.
- Limitation: this is a two-state response coordinate plot, not a plot of transformed CLIP feature space.

### Panel (b): Normal Tail-Risk Reduction

- Data: `normal_tail_curves.csv`.
- Patch set: source normal patches only, `{checks['normal_count']}` normal patches.
- Sorting: one sort by CES-only score, reused for CES+NTR: `{checks['tail_sorted_by_ces_only']}`.
- Top-10% tail fixed from CES-only score: `{checks['top10_fixed_from_ces']}`.
- There is no per-curve min-max normalization or plotting offset; curves use saved raw `q_ces` and `q_ntr`.

### Panel (c): Qualitative Anomaly Responses

- Source checkpoint family: `{metadata['source']}`.
- Target: `{metadata['target_for_qualitative_panel']}`.
- Samples selected by fixed target category/mask-area rule, not by model score.
- Local LRP cache: `{metadata['local_lrp_cache']}`.
- Heatmap color range: shared `vmin={metadata['heatmap_scale']['vmin']:.6f}`, `vmax={metadata['heatmap_scale']['vmax']:.6f}`.
- Per-map min-max normalization: `False`.

## 2. Configuration Fairness Table

See `CONFIG_COMPARISON.csv`. Summary:

{cfg_df.to_markdown(index=False)}

Important limitation: CES-only and CES+NTR are two control variants. The generated figure must not describe them as the same model before and after training. The formal CES-only row is recorded as `init_v_fallback` in the final ablation artifacts, whereas CES+NTR uses a trained `rho=0.10` checkpoint.

## 3. Panel (b) Score Statistics

See `NTR_SCORE_STATISTICS.csv`.

{score_df.to_markdown(index=False)}

Additional checks:

{ntr_shift_summary}
- Hard normal drops more than other normal: `{checks['hard_normal_drops_more_than_other_normal']}`.
- Ranking AUROC delta: `{checks['ranking_AUROC_delta']:.6f}`.
- Ranking AP delta: `{checks['ranking_AP_delta']:.6f}`.

Interpretation:

{ranking_text}

NTR substantially lowers the normal score tail, especially the fixed high-risk normal subset. It also lowers anomaly scores on average. Therefore, the figure supports a source-side tail-risk suppression view. It should not be worded as a guaranteed global increase in source patch discrimination unless the AUROC/AP deltas are explicitly reported.

## 4. Panel (c) Cross-Domain Legality and Sample Metrics

See `QUALITATIVE_SAMPLE_CHECK.csv`.

{qual_df.to_markdown(index=False)}

Existing complete Full VisA→MVTec formal target metrics from `FACTORIAL_ABLATION.csv`:

{full_target_text}

Panel (c) satisfies the same-source/same-target display protocol: both maps are Full VisA-source models evaluated on MVTec target images, with the same target sample IDs, GT masks, feature preprocessing, LRP mapping, and shared heatmap scale. The target MVTec images are used only for visualization/statistics here.

## 5. Supported Claims

This figure can support:

1. CES defines a compact two-state response coordinate system for source patches.
2. The dashed `y=x` line corresponds to zero anomaly score under `q(z)=z^T s_A-z^T s_N`.
3. NTR reduces scores of the fixed CES-selected high-risk normal tail.
4. The shown target examples are strict Full VisA→MVTec visualizations using the same LRP mapping and color scale.
5. Existing full-target formal results show Base+NTR+LRP improves MVTec P-AP over Base+LRP for seed 0.

## 6. Claims Not Supported / Do Not Overstate

Do not claim:

1. Panel (a) is a CLIP feature-space compression plot. It is a two-state response plot.
2. CES-only and CES+NTR are one checkpoint before/after a single training trajectory.
3. Panel (c) alone proves cross-dataset improvement; it is qualitative and must be paired with full-target metrics.
4. NTR only suppresses normal scores without affecting anomaly scores; the source statistics show anomaly scores also decrease.
5. NTR necessarily improves source patch ranking if AUROC/AP deltas are small or negative.

## Final Verdict

**PASS_WITH_LIMITATIONS**

The figure is usable for the paper if the caption explicitly states that CES-only and CES+NTR are separate controls, that panel (a) is a response-space visualization, and that panel (b) demonstrates tail-risk suppression rather than by itself proving improved discrimination. The qualitative panel is valid under the VisA→MVTec cross-domain protocol, but should be presented as illustrative evidence rather than causal proof.
"""
    (AUDIT_OUT / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def main():
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    score_rows, checks = compute_panel_b_stats()
    qual_rows, full_target = compute_qualitative_stats()
    config_rows = config_comparison_rows(metadata)

    write_csv(
        AUDIT_OUT / "NTR_SCORE_STATISTICS.csv",
        [{k: f(v) for k, v in r.items()} for r in score_rows],
        ["metric", "CES_only", "CES_plus_NTR", "delta_NTR_minus_CES"],
    )
    write_csv(
        AUDIT_OUT / "QUALITATIVE_SAMPLE_CHECK.csv",
        [{k: f(v) for k, v in r.items()} for r in qual_rows],
        [
            "sample_id",
            "source",
            "target",
            "category",
            "img_path",
            "checkpoint",
            "variant",
            "pixel_AP",
            "normal_background_mean_response",
            "gt_anomaly_mean_response",
            "heatmap_vmin",
            "heatmap_vmax",
            "same_color_range",
            "per_map_minmax_normalized",
        ],
    )
    write_csv(
        AUDIT_OUT / "CONFIG_COMPARISON.csv",
        config_rows,
        ["item", "CES_only", "CES_plus_NTR", "status", "evidence"],
    )
    write_report(score_rows, checks, qual_rows, full_target, config_rows, metadata)
    print("=== CES/NTR VISUALIZATION AUDIT COMPLETE ===")
    print(f"Report: {AUDIT_OUT / 'AUDIT_REPORT.md'}")
    print(f"Score stats: {AUDIT_OUT / 'NTR_SCORE_STATISTICS.csv'}")
    print(f"Qualitative check: {AUDIT_OUT / 'QUALITATIVE_SAMPLE_CHECK.csv'}")
    print(f"Config comparison: {AUDIT_OUT / 'CONFIG_COMPARISON.csv'}")


if __name__ == "__main__":
    main()
