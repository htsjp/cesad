import csv
import math
from pathlib import Path


OUT = Path("results/phase_xa_memory_distillation")


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def fmt(x):
    return "" if not math.isfinite(x) else f"{x:.4f}"


def load_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_id(row):
    variant = row["variant"]
    seed = int(row["seed"])
    return f"{variant}" if variant != "XA6_RandomMemoryTeacher" else f"{variant}_seed{seed}"


def classify(rows, deltas, random_mean):
    xa2 = next((r for r in rows if r["variant"] == "XA2_CLIPPreservedAdapter"), None)
    xa1 = next((r for r in rows if r["variant"] == "XA1_HardLabelAdapter"), None)
    xa4 = next((r for r in rows if r["variant"] == "XA4_DualTeacherFullMemory"), None)
    if not xa2 or not xa4:
        return "B5_AMBIGUOUS", "XA2 or XA4 is missing."

    xa4_delta = deltas[row_id(xa4)]
    improved_groups = []
    if xa4_delta["G_image"] >= 0.15:
        improved_groups.append("image")
    if xa4_delta["G_pixel"] >= 0.15:
        improved_groups.append("pixel")

    specificity_image = xa4_delta["G_image"] - random_mean["G_image"]
    specificity_pixel = xa4_delta["G_pixel"] - random_mean["G_pixel"]
    specificity_balanced = xa4_delta["G_balanced"] - random_mean["G_balanced"]
    xa4_gt_xa2 = xa4_delta["G_balanced"] > 0.0
    if improved_groups:
        ok_specific = (
            ("image" in improved_groups and specificity_image >= 0.10)
            or ("pixel" in improved_groups and specificity_pixel >= 0.10)
        )
        if ok_specific and xa4_gt_xa2:
            return "B1_REAL_MEMORY_POSITIVE", "XA4 improves over XA2 and separates from XA6 on the improved metric group."

    if xa4_gt_xa2 and abs(specificity_balanced) < 0.10:
        return "B2_REAL_AND_RANDOM_BOTH_POSITIVE", "XA4 improves, but random memory is essentially tied."
    if xa4_gt_xa2 and random_mean["G_balanced"] >= xa4_delta["G_balanced"]:
        return "B2_REAL_AND_RANDOM_BOTH_POSITIVE", "Random memory matches or exceeds XA4."

    random_best_metric_gain = max(random_mean["G_image"], random_mean["G_pixel"])
    xa4_best_metric_gain = max(xa4_delta["G_image"], xa4_delta["G_pixel"])
    if xa4_best_metric_gain < 0.10 and random_best_metric_gain < 0.10:
        return "B3_NO_TASK_GAIN", "XA4 and XA6 both fail to improve over XA2 by >=0.10 pp on primary metric groups."

    if xa1:
        xa1_delta = deltas[row_id(xa1)]
        if xa1_delta["G_balanced"] > xa4_delta["G_balanced"] + 0.10:
            return "B4_HARDLABEL_OR_PRESERVE_BEST", "XA1 clearly exceeds XA4."
    if xa4_delta["G_balanced"] < -0.10:
        return "B4_HARDLABEL_OR_PRESERVE_BEST", "XA2 exceeds XA4."
    return "B5_AMBIGUOUS", "Metric differences are mixed or too small."


def main():
    readout = OUT / "source_checkpoint_readout_D1_fold0.csv"
    per_cat = OUT / "per_category_source_readout_D1_fold0.csv"
    rows = load_rows(readout)
    if not rows:
        raise SystemExit(f"Missing readout rows: {readout}")

    xa2 = next((r for r in rows if r["variant"] == "XA2_CLIPPreservedAdapter"), None)
    if not xa2:
        raise SystemExit("Missing XA2 baseline row")

    metrics = ["I-AUROC", "I-AP", "P-AUROC", "PRO"]
    base = {m: f(xa2[m]) for m in metrics}
    deltas = {}
    for row in rows:
        rid = row_id(row)
        d = {m: f(row[m]) - base[m] for m in metrics}
        d["G_image"] = 0.5 * d["I-AUROC"] + 0.5 * d["I-AP"]
        d["G_pixel"] = 0.5 * d["P-AUROC"] + 0.5 * d["PRO"]
        d["G_balanced"] = 0.5 * d["G_image"] + 0.5 * d["G_pixel"]
        deltas[rid] = d

    random_ids = [row_id(r) for r in rows if r["variant"] == "XA6_RandomMemoryTeacher" and int(r["seed"]) in (0, 1, 2)]
    random_mean = {}
    for key in ["G_image", "G_pixel", "G_balanced"]:
        vals = [deltas[rid][key] for rid in random_ids if math.isfinite(deltas[rid][key])]
        random_mean[key] = sum(vals) / len(vals) if vals else float("nan")

    xa4 = next((r for r in rows if r["variant"] == "XA4_DualTeacherFullMemory"), None)
    xa4_delta = deltas[row_id(xa4)] if xa4 else {}
    specificity = {
        "G_image": xa4_delta.get("G_image", float("nan")) - random_mean.get("G_image", float("nan")),
        "G_pixel": xa4_delta.get("G_pixel", float("nan")) - random_mean.get("G_pixel", float("nan")),
        "G_balanced": xa4_delta.get("G_balanced", float("nan")) - random_mean.get("G_balanced", float("nan")),
    }
    label, reason = classify(rows, deltas, random_mean)

    cat_rows = load_rows(per_cat)
    cat_summary = []
    if cat_rows:
        by_key = {(r["variant"], r["seed"], r["category"]): r for r in cat_rows}
        cats = sorted({r["category"] for r in cat_rows})
        for metric in ["I-AUROC", "P-AUROC", "PRO"]:
            wins_xa2 = ties_xa2 = losses_xa2 = 0
            wins_xa6 = ties_xa6 = losses_xa6 = 0
            deltas_xa2 = []
            deltas_xa6 = []
            for cat in cats:
                r4 = by_key.get(("XA4_DualTeacherFullMemory", "0", cat))
                r2 = by_key.get(("XA2_CLIPPreservedAdapter", "0", cat))
                r6s = [by_key.get(("XA6_RandomMemoryTeacher", str(seed), cat)) for seed in (0, 1, 2)]
                r6s = [r for r in r6s if r]
                if not r4 or not r2:
                    continue
                d2 = f(r4[metric]) - f(r2[metric])
                deltas_xa2.append(d2)
                if abs(d2) < 0.05:
                    ties_xa2 += 1
                elif d2 > 0:
                    wins_xa2 += 1
                else:
                    losses_xa2 += 1
                if r6s:
                    mean6 = sum(f(r[metric]) for r in r6s) / len(r6s)
                    d6 = f(r4[metric]) - mean6
                    deltas_xa6.append(d6)
                    if abs(d6) < 0.05:
                        ties_xa6 += 1
                    elif d6 > 0:
                        wins_xa6 += 1
                    else:
                        losses_xa6 += 1
            cat_summary.append({
                "metric": metric,
                "xa4_vs_xa2_wtl": f"{wins_xa2}/{ties_xa2}/{losses_xa2}",
                "xa4_vs_xa6_wtl": f"{wins_xa6}/{ties_xa6}/{losses_xa6}",
                "worst_delta_xa2": min(deltas_xa2) if deltas_xa2 else float("nan"),
                "median_delta_xa2": sorted(deltas_xa2)[len(deltas_xa2)//2] if deltas_xa2 else float("nan"),
            })

    lines = [
        "# Readout Batch B: D1 Fold0 Source Validation",
        "",
        f"Outcome: **{label}**",
        "",
        "## Completed",
        "",
        "Evaluated existing D1 fold0 checkpoints only: XA0, XA1, XA2, XA4, XA5, and XA6 seeds0-2.",
        "",
        "## Not Run",
        "",
        "No training. No target evaluation. XA6 seeds3-4, XA7 and XA8 were intentionally skipped.",
        "",
        "## Core Metrics",
        "",
        "| Variant | Seed | I-AUROC | I-AP | P-AUROC | PRO | G_image | G_pixel | G_balanced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rid = row_id(row)
        d = deltas[rid]
        lines.append(
            f"| {row['variant']} | {row['seed']} | {fmt(f(row['I-AUROC']))} | {fmt(f(row['I-AP']))} | "
            f"{fmt(f(row['P-AUROC']))} | {fmt(f(row['PRO']))} | {fmt(d['G_image'])} | {fmt(d['G_pixel'])} | {fmt(d['G_balanced'])} |"
        )
    lines.extend([
        "",
        "## Causal Comparison",
        "",
        f"- RealSpecificity G_image: `{fmt(specificity['G_image'])}` pp",
        f"- RealSpecificity G_pixel: `{fmt(specificity['G_pixel'])}` pp",
        f"- RealSpecificity G_balanced: `{fmt(specificity['G_balanced'])}` pp",
        f"- Classification reason: {reason}",
        "",
        "## Per-Category Summary",
        "",
        "| Metric | XA4 vs XA2 win/tie/loss | XA4 vs XA6 win/tie/loss | Worst XA4-XA2 | Median XA4-XA2 |",
        "|---|---:|---:|---:|---:|",
    ])
    for s in cat_summary:
        lines.append(f"| {s['metric']} | {s['xa4_vs_xa2_wtl']} | {s['xa4_vs_xa6_wtl']} | {fmt(s['worst_delta_xa2'])} | {fmt(s['median_delta_xa2'])} |")
    lines.extend([
        "",
        "## Mechanism Interpretation",
        "",
        "This Batch B decision uses held-out source validation metrics as primary evidence. Feature cosine diagnostics are not used for the decision.",
        "",
        "## Next Minimal Batch",
        "",
        "Stop here for user review. If approved and Batch B is not a clear stop, evaluate D1 fold1 existing checkpoints only: XA0, XA1, XA2, XA4, XA5, XA6 seeds0-2.",
    ])
    (OUT / "READOUT_BATCH_B_D1_FOLD0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "READOUT_BATCH_B_D1_FOLD0.md")


if __name__ == "__main__":
    main()
