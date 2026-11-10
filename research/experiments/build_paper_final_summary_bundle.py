from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_assets" / "metric_extension"
ABL = ROOT / "results" / "paper_assets" / "ablation_6metrics"
BUNDLE = ROOT / "results" / "paper_assets" / "PAPER_RESULTS_COMBINED_SUMMARY.md"


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def f(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "NA"


def main():
    ours = read_csv(OUT / "OURS_METRICS_8TRANSFER.csv")
    big_common = read_csv(OUT / "BIG_MODEL_COMMON6_MEAN_SUMMARY.csv")
    big_mean = read_csv(OUT / "BIG_MODEL_MEAN_SUMMARY.csv")
    ablation = read_csv(ABL / "TABLE_ABLATION_6METRICS.csv")
    delta = read_csv(ABL / "ABLATION_METRIC_DELTA.csv")

    lines = [
        "# Paper Results Combined Summary",
        "",
        "Frozen final method: `Canonical Two-State + NTR + Fixed LRP`.",
        "",
        "All values come from existing cached/evaluated artifacts. No model training, hyperparameter tuning, or target calibration was performed in these paper-table consolidation steps.",
        "",
        "## 1. Final Ours 8-Transfer Metrics",
        "",
        "| Source | Target | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in ours.sort_values(["source", "target"]).iterrows():
        lines.append(
            f"| {r['source']} | {r['target']} | {f(r['I_AUROC'])} | {f(r['I_AP'])} | {f(r['P_AUROC'])} | {f(r['P_AP'])} | {f(r['P_F1MAX'])} | {f(r['AUPRO'])} |"
        )

    lines += [
        "",
        "DTD-Synthetic is now complete for Ours at both image and pixel level.",
        "",
        "## 2. Same-Protocol Model Comparison",
        "",
        "Full 8-cell means use every available same-protocol cell for each method. MRAD variants have 6 cells because DTD-Synthetic was not available for local MRAD same-protocol assets.",
        "",
        "| Method | I-AUROC mean/n | I-AP mean/n | P-AUROC mean/n | P-AP mean/n | P-F1max mean/n | AUPRO mean/n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in big_mean.iterrows():
        vals = []
        for metric in ["I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO"]:
            vals.append(f"{f(r[f'{metric}_mean'])}/{int(r[f'{metric}_n'])}")
        lines.append(f"| {r['method']} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "Common 6-cell means exclude DTD-Synthetic so every listed method uses the same transfer cells.",
        "",
        "| Method | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in big_common.iterrows():
        lines.append(
            f"| {r['method']} | {f(r['I_AUROC'])} | {f(r['I_AP'])} | {f(r['P_AUROC'])} | {f(r['P_AP'])} | {f(r['P_F1MAX'])} | {f(r['AUPRO'])} |"
        )

    lines += [
        "",
        "## 3. Core 6-Metric Ablation",
        "",
        "Matched multiseed subset: MVTec→MPDD, MVTec→VisA, VisA→BTAD, VisA→MVTec with seeds 0/1/2 for every variant.",
        "",
        "| Variant | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO | ΔP-AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in ablation.iterrows():
        vals = []
        for metric in ["I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO"]:
            vals.append(f"{f(r[f'{metric}_mean'])} ± {f(r[f'{metric}_mean_of_seed_std'])}")
        lines.append(f"| {r['display_variant']} | " + " | ".join(vals) + f" | {float(r['Delta_P_AP_vs_StateAligned']):+.2f} |")

    lines += ["", "## 4. Module Deltas", "", "| Step | Metric | Delta | Relative |", "|---|---|---:|---:|"]
    for _, r in delta.iterrows():
        step = f"{r['variant_from']} → {r['variant_to']}"
        lines.append(f"| {step} | {r['metric']} | {float(r['delta']):+.2f} | {float(r['relative_delta_percent']):+.2f}% |")

    lines += [
        "",
        "## 5. Takeaways",
        "",
        "- The final frozen method is strongest on pixel-level ranking/localization metrics across same-protocol comparisons, especially P-AP, P-F1max, and AUPRO.",
        "- On the common 6-cell model comparison, Ours has the highest mean I-AUROC, I-AP, P-AP, P-F1max, and AUPRO; MRAD-CLIP is slightly higher on mean P-AUROC.",
        "- NTR is the largest ablation contributor by P-AP: +7.68 over StateAligned.",
        "- Fixed LRP adds another +4.11 P-AP over NTR and improves P-AUROC, P-F1max, and AUPRO, while I-AUROC changes by -0.21 and I-AP by +0.37.",
        "- The paper tables should call historical `PRO` as `AUPRO` because the implementation is AUPRO@FPR<=0.3.",
        "",
        "## 6. Primary Files",
        "",
        "- `results/paper_assets/metric_extension/BIG_MODEL_COMPARISON.md`",
        "- `results/paper_assets/metric_extension/BIG_MODEL_COMPARISON.csv`",
        "- `results/paper_assets/metric_extension/OURS_METRICS_8TRANSFER.csv`",
        "- `results/paper_assets/ablation_6metrics/REPORT.md`",
        "- `results/paper_assets/ablation_6metrics/TABLE_ABLATION_6METRICS.tex`",
        "- `results/paper_assets/ablation_6metrics/figures/ablation_core_metrics.pdf`",
    ]
    BUNDLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[combined-summary] wrote {BUNDLE}")


if __name__ == "__main__":
    main()
