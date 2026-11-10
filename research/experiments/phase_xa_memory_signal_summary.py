import csv
import json
import math
from pathlib import Path


OUT = Path("results/phase_xa_memory_distillation")
METRICS = ["I-AUROC", "I-AP", "P-AUROC", "PRO"]
MAIN_ROWS = [
    "XA0_OfficialMRADCLIP",
    "XA1_HardLabelAdapter",
    "XA2_CLIPPreservedAdapter",
    "XA4_DualTeacherFullMemory",
    "XA5_DualTeacherCompactMemory",
    "XA6mean",
]
TIE_PP = 0.05


def load_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value, digits=4):
    value = float(value)
    return "N/A" if not math.isfinite(value) else f"{value:.{digits}f}"


def mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return sum(values) / len(values) if values else float("nan")


def std(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if len(values) < 2:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def row_label(row):
    if row["variant"] == "XA6_RandomMemoryTeacher":
        return f"XA6 seed{row['seed']}"
    return row["variant"]


def readout_rows(direction, fold):
    return load_csv(OUT / f"source_checkpoint_readout_{direction}_fold{fold}.csv")


def summarized_rows(direction, fold):
    rows = readout_rows(direction, fold)
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    out = {}
    for variant in MAIN_ROWS:
        if variant == "XA6mean":
            seeds = by_variant.get("XA6_RandomMemoryTeacher", [])
            if seeds:
                out[variant] = {m: mean(f(row[m]) for row in seeds) for m in METRICS}
            continue
        vals = by_variant.get(variant, [])
        if vals:
            out[variant] = {m: f(vals[0][m]) for m in METRICS}
    return out


def deltas(direction, fold):
    rows = readout_rows(direction, fold)
    xa2 = next((row for row in rows if row["variant"] == "XA2_CLIPPreservedAdapter"), None)
    if not xa2:
        return {}
    base = {m: f(xa2[m]) for m in METRICS}
    out = {}
    for row in rows:
        d = {m: f(row[m]) - base[m] for m in METRICS}
        d["G_image"] = 0.5 * d["I-AUROC"] + 0.5 * d["I-AP"]
        d["G_pixel"] = 0.5 * d["P-AUROC"] + 0.5 * d["PRO"]
        d["G_balanced"] = 0.5 * d["G_image"] + 0.5 * d["G_pixel"]
        out[row_label(row)] = d
    xa6_rows = [row for row in rows if row["variant"] == "XA6_RandomMemoryTeacher"]
    if xa6_rows:
        out["XA6mean"] = {
            key: mean(out[row_label(row)][key] for row in xa6_rows)
            for key in ["I-AUROC", "I-AP", "P-AUROC", "PRO", "G_image", "G_pixel", "G_balanced"]
        }
    return out


def xa6_stats(direction, fold):
    rows = [row for row in readout_rows(direction, fold) if row["variant"] == "XA6_RandomMemoryTeacher"]
    if not rows:
        return {}
    ds = deltas(direction, fold)
    out = {}
    for key in ["I-AUROC", "I-AP", "P-AUROC", "PRO", "G_image", "G_pixel", "G_balanced"]:
        values = []
        for row in rows:
            values.append(ds[row_label(row)][key] if key.startswith("G_") else f(row[key]))
        out[key] = {"mean": mean(values), "std": std(values), "min": min(values), "max": max(values), "n": len(values)}
    return out


def compare_full_compact(direction, fold):
    rows = summarized_rows(direction, fold)
    xa4 = rows.get("XA4_DualTeacherFullMemory")
    xa5 = rows.get("XA5_DualTeacherCompactMemory")
    if not xa4 or not xa5:
        return None
    return {m: xa4[m] - xa5[m] for m in METRICS}


def per_category_wtl(direction, fold):
    rows = load_csv(OUT / f"per_category_source_readout_{direction}_fold{fold}.csv")
    if not rows:
        return []
    by_key = {(row["variant"], row["seed"], row["category"]): row for row in rows}
    categories = sorted({row["category"] for row in rows})
    out = []
    for metric in ["I-AUROC", "P-AUROC", "PRO"]:
        for comparison in ["XA4-XA2", "XA4-XA6mean"]:
            deltas_pp = []
            for cat in categories:
                xa4 = by_key.get(("XA4_DualTeacherFullMemory", "0", cat))
                if not xa4:
                    continue
                if comparison == "XA4-XA2":
                    other = by_key.get(("XA2_CLIPPreservedAdapter", "0", cat))
                    if not other:
                        continue
                    delta = f(xa4[metric]) - f(other[metric])
                else:
                    random_rows = [
                        by_key.get(("XA6_RandomMemoryTeacher", str(seed), cat))
                        for seed in range(5)
                    ]
                    random_rows = [row for row in random_rows if row]
                    if not random_rows:
                        continue
                    delta = f(xa4[metric]) - mean(f(row[metric]) for row in random_rows)
                if math.isfinite(delta):
                    deltas_pp.append(delta)
            wins = sum(1 for x in deltas_pp if x > TIE_PP)
            ties = sum(1 for x in deltas_pp if abs(x) <= TIE_PP)
            losses = sum(1 for x in deltas_pp if x < -TIE_PP)
            sorted_delta = sorted(deltas_pp)
            out.append({
                "direction": direction,
                "fold": fold,
                "metric": metric,
                "comparison": comparison,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "median": sorted_delta[len(sorted_delta) // 2] if sorted_delta else float("nan"),
                "worst": min(sorted_delta) if sorted_delta else float("nan"),
                "best": max(sorted_delta) if sorted_delta else float("nan"),
                "n_categories": len(deltas_pp),
            })
    return out


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics_table(title, direction, folds):
    lines = [f"## {title}", "", "| Fold | Row | I-AUROC | I-AP | P-AUROC | PRO |", "|---:|---|---:|---:|---:|---:|"]
    for fold in folds:
        rows = summarized_rows(direction, fold)
        for label in MAIN_ROWS:
            vals = rows.get(label)
            if vals:
                lines.append(
                    f"| {fold} | {label} | {fmt(vals['I-AUROC'], 6)} | {fmt(vals['I-AP'], 6)} | "
                    f"{fmt(vals['P-AUROC'], 6)} | {fmt(vals['PRO'], 6)} |"
                )
            elif direction == "D1" and fold == 2 and label in {"XA0_OfficialMRADCLIP", "XA5_DualTeacherCompactMemory"}:
                lines.append(f"| {fold} | {label} | N/A | N/A | N/A | N/A |")
    lines.append("")
    return lines


def delta_table(title, direction, folds):
    lines = [
        f"## {title}",
        "",
        "| Fold | Row | dI-AUROC | dI-AP | dP-AUROC | dPRO | G_image | G_pixel | G_balanced |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in folds:
        ds = deltas(direction, fold)
        for label in ["XA1_HardLabelAdapter", "XA4_DualTeacherFullMemory", "XA5_DualTeacherCompactMemory", "XA6mean"]:
            if label not in ds:
                continue
            d = ds[label]
            lines.append(
                f"| {fold} | {label} | {fmt(d['I-AUROC'])} | {fmt(d['I-AP'])} | {fmt(d['P-AUROC'])} | "
                f"{fmt(d['PRO'])} | {fmt(d['G_image'])} | {fmt(d['G_pixel'])} | {fmt(d['G_balanced'])} |"
            )
    lines.append("")
    return lines


def xa6_table(direction, folds):
    lines = [
        f"## {direction} XA6 Random-Seed Variability",
        "",
        "| Fold | Metric | Mean | Std | Min | Max | N |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for fold in folds:
        stats = xa6_stats(direction, fold)
        for key in ["I-AUROC", "I-AP", "P-AUROC", "PRO", "G_image", "G_pixel", "G_balanced"]:
            s = stats.get(key)
            if not s:
                continue
            lines.append(f"| {fold} | {key} | {fmt(s['mean'])} | {fmt(s['std'])} | {fmt(s['min'])} | {fmt(s['max'])} | {s['n']} |")
    lines.append("")
    return lines


def per_category_table(title, rows):
    lines = [
        f"## {title}",
        "",
        f"Tie threshold: abs(delta) <= {TIE_PP:.2f} pp.",
        "",
        "| Direction | Fold | Metric | Comparison | W/T/L | Median | Worst | Best | Categories |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['direction']} | {row['fold']} | {row['metric']} | {row['comparison']} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {fmt(row['median'])} | "
            f"{fmt(row['worst'])} | {fmt(row['best'])} | {row['n_categories']} |"
        )
    lines.append("")
    return lines


def compact_table():
    lines = [
        "## XA4 Full Teacher vs XA5 Compact Teacher",
        "",
        "| Direction | Fold | dI-AUROC | dI-AP | dP-AUROC | dPRO | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for direction, folds in [("D1", [0, 1, 2]), ("D2", [0])]:
        for fold in folds:
            diff = compare_full_compact(direction, fold)
            if diff is None:
                status = "N/A by tie-break protocol" if direction == "D1" and fold == 2 else "N/A"
                lines.append(f"| {direction} | {fold} | N/A | N/A | N/A | N/A | {status} |")
                continue
            max_abs = max(abs(x) for x in diff.values())
            status = "identical at readout precision" if max_abs <= 1e-9 else "different"
            lines.append(
                f"| {direction} | {fold} | {fmt(diff['I-AUROC'], 6)} | {fmt(diff['I-AP'], 6)} | "
                f"{fmt(diff['P-AUROC'], 6)} | {fmt(diff['PRO'], 6)} | {status} |"
            )
    lines.append("")
    return lines


def main():
    per_cat_rows = []
    for fold in [0, 1, 2]:
        per_cat_rows.extend(per_category_wtl("D1", fold))
    per_cat_rows.extend(per_category_wtl("D2", 0))
    write_csv(OUT / "memory_signal_per_category_wtl.csv", per_cat_rows)

    payload = {
        "d1_fold2_missing_rows": ["XA0_OfficialMRADCLIP", "XA5_DualTeacherCompactMemory"],
        "d2_fold0_outcome": json.loads((OUT / "D2_FOLD0_EXISTING_OUTCOME.json").read_text(encoding="utf-8")),
        "d1_fold2_outcome": json.loads((OUT / "D1_FOLD2_TIEBREAK_OUTCOME.json").read_text(encoding="utf-8")),
    }
    (OUT / "MEMORY_SIGNAL_SUMMARY_OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase X-A Memory Signal Summary",
        "",
        "Scope: D1 fold0/fold1/fold2 source readout and existing D2 fold0 source readout.",
        "",
        "Main conclusion: D1 is fold-sensitive but the fold2 tie-break restores positive real-memory specificity; D2 fold0 gives a strong cross-direction positive readout. No additional X-A training is recommended from these results.",
        "",
        "Important caveat: D1 fold2 was a minimal tie-break run, so XA0 and XA5 were not evaluated under the source-readout protocol for that fold.",
        "",
    ]
    lines.extend(metrics_table("D1 Full Raw Four-Metric Readout", "D1", [0, 1, 2]))
    lines.extend(delta_table("D1 Deltas vs XA2", "D1", [0, 1, 2]))
    lines.extend(xa6_table("D1", [0, 1, 2]))
    lines.extend(metrics_table("D2 Fold0 Raw Four-Metric Readout", "D2", [0]))
    lines.extend(delta_table("D2 Fold0 Deltas vs XA2", "D2", [0]))
    lines.extend(xa6_table("D2", [0]))
    lines.extend(compact_table())
    lines.extend(per_category_table("Per-Category Win/Tie/Loss", per_cat_rows))
    lines.extend([
        "## Interpretation",
        "",
        "- D1 fold1 is not a severe mechanism failure: XA4 still beats XA2 on all four raw metrics, but its real-vs-random specificity turns slightly negative because XA6 also improves strongly.",
        "- D1 fold2 is the tie-break: XA4 again beats XA2 and also beats XA6 mean on image, pixel, and balanced groups.",
        "- D2 fold0 is the strongest evidence: XA4 beats both XA2 and XA1 on image/balanced groups, and XA6 seed variability is much smaller than the XA4-vs-XA6mean specificity.",
        "- XA4 and XA5 are identical at readout precision wherever both were evaluated, supporting the compact-teacher conclusion.",
        "",
    ])
    report = OUT / "PHASE_XA_MEMORY_SIGNAL_SUMMARY.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
