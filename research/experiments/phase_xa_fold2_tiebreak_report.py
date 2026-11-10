import csv
import json
import math
from pathlib import Path


OUT = Path("results/phase_xa_memory_distillation")
METRICS = ["I-AUROC", "I-AP", "P-AUROC", "PRO"]


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
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return sum(values) / len(values) if values else float("nan")


def row_name(row):
    if row["variant"] == "XA6_RandomMemoryTeacher":
        return f"XA6_seed{row['seed']}"
    return row["variant"]


def fold_summary(fold):
    rows = load_csv(OUT / f"source_checkpoint_readout_D1_fold{fold}.csv")
    if not rows:
        return None
    wanted = []
    for row in rows:
        if row["variant"] in {
            "XA1_HardLabelAdapter",
            "XA2_CLIPPreservedAdapter",
            "XA4_DualTeacherFullMemory",
            "XA6_RandomMemoryTeacher",
        }:
            wanted.append(row)
    xa2 = next((r for r in wanted if r["variant"] == "XA2_CLIPPreservedAdapter"), None)
    xa4 = next((r for r in wanted if r["variant"] == "XA4_DualTeacherFullMemory"), None)
    xa6 = [r for r in wanted if r["variant"] == "XA6_RandomMemoryTeacher"]
    if not xa2 or not xa4 or not xa6:
        return None

    base = {m: f(xa2[m]) for m in METRICS}
    deltas = {}
    for row in wanted:
        d = {m: f(row[m]) - base[m] for m in METRICS}
        d["G_image"] = 0.5 * d["I-AUROC"] + 0.5 * d["I-AP"]
        d["G_pixel"] = 0.5 * d["P-AUROC"] + 0.5 * d["PRO"]
        d["G_balanced"] = 0.5 * d["G_image"] + 0.5 * d["G_pixel"]
        deltas[row_name(row)] = d

    random = {}
    for key in ["I-AUROC", "I-AP", "P-AUROC", "PRO", "G_image", "G_pixel", "G_balanced"]:
        random[key] = mean([deltas[row_name(r)][key] for r in xa6])

    xa4_delta = deltas["XA4_DualTeacherFullMemory"]
    specificity = {key: xa4_delta[key] - random[key] for key in random}
    return {
        "fold": fold,
        "rows": wanted,
        "deltas": deltas,
        "random": random,
        "specificity": specificity,
    }


def classify(summaries):
    valid = [s for s in summaries if s]
    fold2 = next((s for s in valid if s["fold"] == 2), None)
    if not fold2:
        return "FOLD2_TIEBREAK_INCOMPLETE", "Fold2 readout is missing or incomplete.", "Run no further analysis until fold2 exists."

    image_gain = fold2["deltas"]["XA4_DualTeacherFullMemory"]["G_image"]
    pixel_gain = fold2["deltas"]["XA4_DualTeacherFullMemory"]["G_pixel"]
    balanced_gain = fold2["deltas"]["XA4_DualTeacherFullMemory"]["G_balanced"]
    image_spec = fold2["specificity"]["G_image"]
    balanced_spec = fold2["specificity"]["G_balanced"]

    all_image_specs = [s["specificity"]["G_image"] for s in valid]
    mean_image_spec = mean(all_image_specs)
    positive_image_spec = sum(1 for x in all_image_specs if x > 0.10)

    if image_gain > 0.15 and image_spec > 0.10 and mean_image_spec > 0.10:
        return (
            "D1_TIEBREAK_MEMORY_SIGNAL_POSITIVE",
            "Fold2 restores positive image-side real-memory specificity.",
            "Stop D1. Next useful action is existing D2 fold0 readout only, not more D1 training.",
        )
    if balanced_gain > 0 and balanced_spec <= 0 and pixel_gain > 0:
        return (
            "D1_TIEBREAK_GENERIC_REGULARIZATION",
            "XA4 still improves over XA2, but random memory matches or exceeds real memory.",
            "Stop new X-A training. Existing D2 fold0 readout is optional for cross-direction confirmation.",
        )
    if image_gain <= 0.15 and balanced_gain <= 0.10:
        return (
            "D1_TIEBREAK_NO_REPRODUCIBLE_MEMORY_GAIN",
            "Fold2 does not show a practically meaningful XA4 gain over XA2.",
            "Stop. No more X-A training is justified from D1.",
        )
    if positive_image_spec >= 2 and mean_image_spec > 0.10:
        return (
            "D1_TIEBREAK_IMAGE_SIGNAL_PARTIAL",
            "Image specificity is positive in at least two folds but remains fold-sensitive.",
            "Stop D1. Prefer existing D2 fold0 readout before any new training.",
        )
    return (
        "D1_TIEBREAK_UNRESOLVED",
        "Fold2 remains mixed across gain and specificity tests.",
        "Stop here for review; do not broaden experiments automatically.",
    )


def main():
    summaries = [fold_summary(fold) for fold in (0, 1, 2)]
    label, reason, next_action = classify(summaries)
    valid = [s for s in summaries if s]

    lines = [
        "# Phase X-A D1 Fold2 Tie-Break",
        "",
        f"Outcome: **{label}**",
        "",
        f"Reason: {reason}",
        "",
        f"Next action: {next_action}",
        "",
        "## Fold Metrics",
        "",
        "| Fold | Row | I-AUROC | I-AP | P-AUROC | PRO | G_image | G_pixel | G_balanced |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in valid:
        for row in summary["rows"]:
            name = row_name(row)
            d = summary["deltas"][name]
            lines.append(
                f"| {summary['fold']} | {name} | {fmt(f(row['I-AUROC']), 6)} | {fmt(f(row['I-AP']), 6)} | "
                f"{fmt(f(row['P-AUROC']), 6)} | {fmt(f(row['PRO']), 6)} | {fmt(d['G_image'])} | "
                f"{fmt(d['G_pixel'])} | {fmt(d['G_balanced'])} |"
            )
    lines.extend([
        "",
        "## Real Memory Specificity",
        "",
        "| Fold | G_image | G_pixel | G_balanced |",
        "|---:|---:|---:|---:|",
    ])
    for summary in valid:
        spec = summary["specificity"]
        lines.append(f"| {summary['fold']} | {fmt(spec['G_image'])} | {fmt(spec['G_pixel'])} | {fmt(spec['G_balanced'])} |")

    payload = {
        "outcome": label,
        "reason": reason,
        "next_action": next_action,
        "folds": [
            {
                "fold": s["fold"],
                "xa4_g_image": s["deltas"]["XA4_DualTeacherFullMemory"]["G_image"],
                "xa4_g_pixel": s["deltas"]["XA4_DualTeacherFullMemory"]["G_pixel"],
                "xa4_g_balanced": s["deltas"]["XA4_DualTeacherFullMemory"]["G_balanced"],
                "specificity_image": s["specificity"]["G_image"],
                "specificity_pixel": s["specificity"]["G_pixel"],
                "specificity_balanced": s["specificity"]["G_balanced"],
            }
            for s in valid
        ],
    }
    (OUT / "READOUT_D1_FOLD2_TIEBREAK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "D1_FOLD2_TIEBREAK_OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"outcome": label, "report": str(OUT / "READOUT_D1_FOLD2_TIEBREAK.md")}, indent=2))


if __name__ == "__main__":
    main()
