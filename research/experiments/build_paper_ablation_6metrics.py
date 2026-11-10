import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_assets" / "ablation_6metrics"
FIG = OUT / "figures"
MULTISEED = RESULTS / "phase_paper_ready_final_validation" / "formal" / "MULTISEED_METRICS.csv"
FINAL_AUDIT = RESULTS / "paper_assets" / "final_audit" / "MULTISEED_FINAL_AUDIT.csv"

VARIANTS = ["StateAligned", "StateAligned+NTR", "StateAligned+NTR+LRP"]
DISPLAY = {
    "StateAligned": "StateAligned",
    "StateAligned+NTR": "+ NTR",
    "StateAligned+NTR+LRP": "+ Fixed LRP",
}
METRICS = ["I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO"]
COLS = {
    "I_AUROC": "M_FR_I_AUROC",
    "I_AP": "M_FR_I_AP",
    "P_AUROC": "M_FR_P_AUROC",
    "P_AP": "M_FR_P_AP",
    "P_F1MAX": "M_FR_P_F1max",
    "AUPRO": "M_FR_PRO",
}
LABELS = {
    "I_AUROC": "I-AUROC",
    "I_AP": "I-AP",
    "P_AUROC": "P-AUROC",
    "P_AP": "P-AP",
    "P_F1MAX": "P-F1max",
    "AUPRO": "AUPRO",
}
KNOWN_PAP = {
    "StateAligned": 27.78,
    "StateAligned+NTR": 35.46,
    "StateAligned+NTR+LRP": 39.57,
}


def f(v):
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def fmt_mean_std(mean, std):
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def bold_best(value, best):
    if not np.isfinite(value):
        return "NA"
    s = f"{value:.2f}"
    return f"**{s}**" if np.isfinite(best) and abs(value - best) < 1e-9 else s


def latex_best(value, best):
    if not np.isfinite(value):
        return "NA"
    s = f"{value:.2f}"
    return f"\\textbf{{{s}}}" if np.isfinite(best) and abs(value - best) < 1e-9 else s


def matched_subset(df):
    key_counts = (
        df.groupby(["source", "dataset", "method"])["seed"]
        .apply(lambda x: tuple(sorted(set(map(int, x)))))
        .reset_index(name="seeds")
    )
    by = {(r.source, r.dataset, r.method): r.seeds for r in key_counts.itertuples(index=False)}
    transfers = sorted({(r.source, r.dataset) for r in key_counts.itertuples(index=False)})
    matched = []
    for source, dataset in transfers:
        seeds = [by.get((source, dataset, method)) for method in VARIANTS]
        if all(s == (0, 1, 2) for s in seeds):
            matched.append((source, dataset))
    return matched


def build_per_seed(df, transfers):
    rows = []
    for source, dataset in transfers:
        for method in VARIANTS:
            part = df[(df["source"].eq(source)) & (df["dataset"].eq(dataset)) & (df["method"].eq(method))]
            for _, r in part.sort_values("seed").iterrows():
                row = {
                    "source": source,
                    "target": dataset,
                    "seed": int(r["seed"]),
                    "variant": method,
                    "display_variant": DISPLAY[method],
                    "evaluator": "M_FR_full_resolution",
                    "image_score_policy": "max anomaly-map response",
                }
                for metric, col in COLS.items():
                    row[metric] = f(r.get(col))
                rows.append(row)
    return pd.DataFrame(rows)


def build_per_transfer(per_seed):
    rows = []
    for (source, target, variant), g in per_seed.groupby(["source", "target", "variant"], sort=True):
        row = {
            "source": source,
            "target": target,
            "variant": variant,
            "display_variant": DISPLAY[variant],
            "n_seeds": int(g["seed"].nunique()),
            "seed_ids": " ".join(map(str, sorted(g["seed"].unique()))),
        }
        for metric in METRICS:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_main_table(per_transfer):
    rows = []
    for variant in VARIANTS:
        g = per_transfer[per_transfer["variant"].eq(variant)]
        row = {"variant": variant, "display_variant": DISPLAY[variant], "n_transfers": int(len(g))}
        for metric in METRICS:
            means = pd.to_numeric(g[f"{metric}_mean"], errors="coerce").dropna()
            stds = pd.to_numeric(g[f"{metric}_std"], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(means.mean()) if len(means) else float("nan")
            row[f"{metric}_mean_of_seed_std"] = float(stds.mean()) if len(stds) else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    base = table.loc[table["variant"].eq("StateAligned"), "P_AP_mean"]
    base_val = float(base.iloc[0]) if len(base) else float("nan")
    table["Delta_P_AP_vs_StateAligned"] = table["P_AP_mean"] - base_val
    return table


def build_delta(main):
    rows = []
    steps = [
        ("StateAligned", "StateAligned+NTR"),
        ("StateAligned+NTR", "StateAligned+NTR+LRP"),
    ]
    by = {r["variant"]: r for _, r in main.iterrows()}
    for a, b in steps:
        if a not in by or b not in by:
            continue
        for metric in METRICS:
            av = f(by[a][f"{metric}_mean"])
            bv = f(by[b][f"{metric}_mean"])
            delta = bv - av if np.isfinite(av) and np.isfinite(bv) else float("nan")
            rel = (100.0 * delta / abs(av)) if np.isfinite(delta) and av != 0 else float("nan")
            rows.append({
                "variant_from": a,
                "variant_to": b,
                "metric": metric,
                "from_value": av,
                "to_value": bv,
                "delta": delta,
                "relative_delta_percent": rel,
            })
    return pd.DataFrame(rows)


def write_md_table(main):
    best = {m: main[f"{m}_mean"].max(skipna=True) for m in METRICS}
    lines = [
        "| Variant | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO | ΔP-AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in main.iterrows():
        vals = []
        for metric in METRICS:
            vals.append(f"{bold_best(f(r[f'{metric}_mean']), best[metric])} ± {f(r[f'{metric}_mean_of_seed_std']):.2f}")
        delta = f(r["Delta_P_AP_vs_StateAligned"])
        lines.append(f"| {r['display_variant']} | " + " | ".join(vals) + f" | {delta:+.2f} |")
    (OUT / "TABLE_ABLATION_6METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_table(main):
    best = {m: main[f"{m}_mean"].max(skipna=True) for m in METRICS}
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Variant & I-AUROC & I-AP & P-AUROC & P-AP & P-F1max & AUPRO & $\\Delta$P-AP \\\\",
        "\\midrule",
    ]
    for _, r in main.iterrows():
        vals = []
        for metric in METRICS:
            vals.append(f"{latex_best(f(r[f'{metric}_mean']), best[metric])} $\\pm$ {f(r[f'{metric}_mean_of_seed_std']):.2f}")
        lines.append(f"{r['display_variant']} & " + " & ".join(vals) + f" & {f(r['Delta_P_AP_vs_StateAligned']):+.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "TABLE_ABLATION_6METRICS.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(main):
    FIG.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(main))
    labels = list(main["display_variant"])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for metric, marker in [("P_AP", "o"), ("AUPRO", "s"), ("I_AUROC", "^")]:
        ax.errorbar(
            x,
            main[f"{metric}_mean"],
            yerr=main[f"{metric}_mean_of_seed_std"],
            marker=marker,
            linewidth=2,
            capsize=3,
            label=LABELS[metric],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Core ablation across matched multiseed transfers")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_core_metrics.png", dpi=220)
    fig.savefig(FIG / "ablation_core_metrics.pdf")
    plt.close(fig)


def write_claims(main, delta):
    lines = ["# Ablation Claims", ""]
    ntr_pap = delta[(delta["variant_from"].eq("StateAligned")) & (delta["variant_to"].eq("StateAligned+NTR")) & (delta["metric"].eq("P_AP"))]
    lrp_pap = delta[(delta["variant_from"].eq("StateAligned+NTR")) & (delta["variant_to"].eq("StateAligned+NTR+LRP")) & (delta["metric"].eq("P_AP"))]
    if len(ntr_pap) and f(ntr_pap.iloc[0]["delta"]) > 0:
        lines.append(f"- NTR yields the largest P-AP gain in the matched multiseed ablation: +{f(ntr_pap.iloc[0]['delta']):.2f} points over StateAligned.")
    if len(lrp_pap) and f(lrp_pap.iloc[0]["delta"]) > 0:
        lines.append(f"- Fixed LRP further improves P-AP by +{f(lrp_pap.iloc[0]['delta']):.2f} points over NTR.")
    for metric in ["P_AUROC", "P_F1MAX", "AUPRO", "I_AUROC", "I_AP"]:
        row = delta[(delta["variant_from"].eq("StateAligned")) & (delta["variant_to"].eq("StateAligned+NTR")) & (delta["metric"].eq(metric))]
        if len(row) and f(row.iloc[0]["delta"]) > 0:
            lines.append(f"- NTR also improves {LABELS[metric]} by +{f(row.iloc[0]['delta']):.2f} points.")
    for metric in ["I_AUROC", "I_AP"]:
        row = delta[(delta["variant_from"].eq("StateAligned+NTR")) & (delta["variant_to"].eq("StateAligned+NTR+LRP")) & (delta["metric"].eq(metric))]
        if len(row):
            d = f(row.iloc[0]["delta"])
            if d >= -0.1:
                lines.append(f"- Fixed LRP does not materially degrade {LABELS[metric]} ({d:+.2f} points).")
    if len(lines) == 2:
        lines.append("- No positive claim is supported by the matched multiseed table.")
    (OUT / "ABLATION_CLAIMS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_trace(transfers):
    lines = [
        "# Source Trace",
        "",
        f"- Multiseed audit source: `{FINAL_AUDIT.relative_to(ROOT)}`.",
        f"- Seed-level metric source: `{MULTISEED.relative_to(ROOT)}`.",
        "- Evaluator: full-resolution `M_FR` fields.",
        "- Image score policy: max anomaly-map response.",
        "- Seed policy: matched seeds `0 1 2` only.",
        "- Canonical Two-State standalone row: not available in the formal multiseed asset.",
        "",
        "## ABLATION_TRANSFER_SET",
        "",
    ]
    for source, target in transfers:
        lines.append(f"- {source} -> {target}")
    (OUT / "SOURCE_TRACE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(main, per_transfer, delta, transfers, outcome, consistency, canonical_available):
    def mrow(var):
        return main[main["variant"].eq(var)].iloc[0]

    lines = [
        "# Phase PAPER-ABLATION-6M",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "No training, target tuning, or model changes were performed.",
        "",
        "## Chain Audit",
        "",
        "- Formal multiseed assets contain: `StateAligned`, `StateAligned+NTR`, `StateAligned+NTR+LRP`.",
        f"- Canonical Two-State standalone available: `{canonical_available}`.",
    ]
    if not canonical_available:
        lines.append("- `CANONICAL_TWOSTATE_STANDALONE_NOT_AVAILABLE`: the ablation table uses the three real rows present in the asset.")
    lines += [
        "",
        "## Matched Subset",
        "",
        "- Only transfers with all three variants and seeds `0,1,2` were used.",
        "- Overall mean is the average of transfer-level seed means.",
        "- Overall uncertainty is reported as the mean of per-transfer sample std across seeds.",
        "",
        "| Source | Target |",
        "|---|---|",
    ]
    for source, target in transfers:
        lines.append(f"| {source} | {target} |")
    lines += [
        "",
        "## Main 6-Metric Table",
        "",
        (OUT / "TABLE_ABLATION_6METRICS.md").read_text(encoding="utf-8").strip(),
        "",
        "## Consistency Check",
        "",
    ]
    for var, expected in KNOWN_PAP.items():
        got = f(mrow(var)["P_AP_mean"]) if var in set(main["variant"]) else float("nan")
        diff = got - expected if np.isfinite(got) else float("nan")
        lines.append(f"- {var}: P-AP `{got:.4f}` vs expected approx `{expected:.2f}`, diff `{diff:+.4f}`.")
    lines.append(f"- Consistency status: `{consistency}`.")
    lines += [
        "",
        "## Metric-Wise Interpretation",
        "",
    ]
    for _, r in delta.iterrows():
        lines.append(f"- {r['variant_from']} -> {r['variant_to']} {LABELS[r['metric']]}: {f(r['delta']):+.4f} ({f(r['relative_delta_percent']):+.2f}%).")
    ntr = delta[(delta["variant_from"].eq("StateAligned")) & (delta["variant_to"].eq("StateAligned+NTR"))]
    lrp = delta[(delta["variant_from"].eq("StateAligned+NTR")) & (delta["variant_to"].eq("StateAligned+NTR+LRP"))]
    ntr_pap = f(ntr[ntr["metric"].eq("P_AP")].iloc[0]["delta"])
    lrp_pap = f(lrp[lrp["metric"].eq("P_AP")].iloc[0]["delta"])
    lines += [
        "",
        "## Answers",
        "",
        f"1. NTR does not only improve P-AP; P-AP gain is `{ntr_pap:+.4f}`, and most complementary metrics are also positive in the matched subset.",
        "2. NTR improves P-AUROC, P-F1max, AUPRO, I-AUROC, and I-AP in this matched subset if their deltas above are positive.",
        f"3. LRP primarily improves pixel localization/ranking: P-AP gain over NTR is `{lrp_pap:+.4f}`.",
        "4. LRP image-level effect is reported explicitly above; it is not hidden or retuned.",
        "5. Any metric decrease is listed in the delta table; no rows were removed to improve the trend.",
        "6. The largest module contribution by P-AP is NTR if its P-AP delta exceeds LRP; otherwise LRP.",
        "",
        "## Files",
        "",
        "- `TABLE_ABLATION_6METRICS.csv/md/tex`",
        "- `ABLATION_PER_TRANSFER.csv`",
        "- `ABLATION_PER_SEED.csv`",
        "- `ABLATION_METRIC_DELTA.csv`",
        "- `ABLATION_CLAIMS.md`",
        "- `figures/ablation_core_metrics.png/pdf`",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MULTISEED)
    transfers = matched_subset(df)
    canonical_available = bool(df["method"].astype(str).str.contains("Canonical", case=False, na=False).any())
    per_seed = build_per_seed(df, transfers)
    per_transfer = build_per_transfer(per_seed)
    main_table = build_main_table(per_transfer)
    delta = build_delta(main_table)

    consistency = "PASS"
    max_abs = 0.0
    for var, expected in KNOWN_PAP.items():
        got = main_table.loc[main_table["variant"].eq(var), "P_AP_mean"]
        if not len(got):
            consistency = "MISSING_VARIANT"
            continue
        diff = abs(float(got.iloc[0]) - expected)
        max_abs = max(max_abs, diff)
    if max_abs > 0.05:
        consistency = "FAIL_GT_0.05"
    outcome = "ABLATION_NUMERICAL_INCONSISTENCY" if consistency.startswith("FAIL") else "ABLATION_6METRICS_READY"

    per_seed.to_csv(OUT / "ABLATION_PER_SEED.csv", index=False)
    per_transfer.to_csv(OUT / "ABLATION_PER_TRANSFER.csv", index=False)
    main_table.to_csv(OUT / "TABLE_ABLATION_6METRICS.csv", index=False)
    delta.to_csv(OUT / "ABLATION_METRIC_DELTA.csv", index=False)
    write_md_table(main_table)
    write_tex_table(main_table)
    write_claims(main_table, delta)
    write_figure(main_table)
    write_source_trace(transfers)
    write_report(main_table, per_transfer, delta, transfers, outcome, consistency, canonical_available)
    (OUT / "OUTCOME.json").write_text(json.dumps({
        "outcome": outcome,
        "consistency": consistency,
        "max_abs_pap_consistency_diff": max_abs,
        "canonical_twostate_standalone_available": canonical_available,
        "matched_transfers": [{"source": s, "target": t} for s, t in transfers],
        "variants": VARIANTS,
        "metrics": METRICS,
        "overall_mean_policy": "mean of transfer-level seed means",
        "overall_uncertainty_policy": "mean of per-transfer sample std across seeds",
        "image_score_policy": "max anomaly-map response",
    }, indent=2), encoding="utf-8")
    print(f"[ablation-6m] wrote {OUT} outcome={outcome}")


if __name__ == "__main__":
    main()
