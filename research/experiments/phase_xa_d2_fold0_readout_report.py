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
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def rid(row):
    if row["variant"] == "XA6_RandomMemoryTeacher":
        return f"XA6_seed{row['seed']}"
    return row["variant"]


def mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return sum(values) / len(values) if values else float("nan")


def main():
    rows = load_csv(OUT / "source_checkpoint_readout_D2_fold0.csv")
    if not rows:
        raise SystemExit("Missing D2 fold0 source checkpoint readout")
    xa2 = next((r for r in rows if r["variant"] == "XA2_CLIPPreservedAdapter"), None)
    xa4 = next((r for r in rows if r["variant"] == "XA4_DualTeacherFullMemory"), None)
    xa1 = next((r for r in rows if r["variant"] == "XA1_HardLabelAdapter"), None)
    xa6 = [r for r in rows if r["variant"] == "XA6_RandomMemoryTeacher" and int(r["seed"]) in (0, 1, 2)]
    if not xa2 or not xa4 or len(xa6) < 3:
        raise SystemExit("D2 fold0 readout is incomplete")

    base = {m: f(xa2[m]) for m in METRICS}
    deltas = {}
    for row in rows:
        d = {m: f(row[m]) - base[m] for m in METRICS}
        d["G_image"] = 0.5 * d["I-AUROC"] + 0.5 * d["I-AP"]
        d["G_pixel"] = 0.5 * d["P-AUROC"] + 0.5 * d["PRO"]
        d["G_balanced"] = 0.5 * d["G_image"] + 0.5 * d["G_pixel"]
        deltas[rid(row)] = d

    random = {}
    for key in ["I-AUROC", "I-AP", "P-AUROC", "PRO", "G_image", "G_pixel", "G_balanced"]:
        random[key] = mean([deltas[rid(row)][key] for row in xa6])
    xa4_delta = deltas["XA4_DualTeacherFullMemory"]
    specificity = {key: xa4_delta[key] - random[key] for key in random}

    if xa4_delta["G_image"] > 0.15 and specificity["G_image"] > 0.10 and xa4_delta["G_balanced"] > 0:
        outcome = "D2_EXISTING_READOUT_MEMORY_SIGNAL_POSITIVE"
        next_action = "Stop. Cross-direction existing-checkpoint evidence is positive; do not start new X-A training automatically."
    elif xa4_delta["G_balanced"] > 0 and specificity["G_balanced"] <= 0:
        outcome = "D2_EXISTING_READOUT_GENERIC_REGULARIZATION"
        next_action = "Stop. Random memory matches/exceeds real memory on D2 fold0."
    elif xa4_delta["G_balanced"] <= 0:
        outcome = "D2_EXISTING_READOUT_NO_GAIN"
        next_action = "Stop. D2 fold0 does not support more X-A training."
    else:
        outcome = "D2_EXISTING_READOUT_MIXED"
        next_action = "Stop for review; do not broaden the grid automatically."

    lines = [
        "# Phase X-A Existing D2 Fold0 Source Readout",
        "",
        f"Outcome: **{outcome}**",
        "",
        f"Next action: {next_action}",
        "",
        "## Raw Metrics",
        "",
        "| Row | I-AUROC | I-AP | P-AUROC | PRO | G_image | G_pixel | G_balanced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        d = deltas[rid(row)]
        lines.append(
            f"| {rid(row)} | {fmt(f(row['I-AUROC']), 6)} | {fmt(f(row['I-AP']), 6)} | "
            f"{fmt(f(row['P-AUROC']), 6)} | {fmt(f(row['PRO']), 6)} | {fmt(d['G_image'])} | "
            f"{fmt(d['G_pixel'])} | {fmt(d['G_balanced'])} |"
        )
    lines.extend([
        "",
        "## Real Memory Specificity",
        "",
        f"- G_image: `{fmt(specificity['G_image'])}` pp",
        f"- G_pixel: `{fmt(specificity['G_pixel'])}` pp",
        f"- G_balanced: `{fmt(specificity['G_balanced'])}` pp",
        "",
        "## Hard-Label / Preservation Check",
        "",
    ])
    if xa1:
        xa1_delta = deltas["XA1_HardLabelAdapter"]
        lines.extend([
            f"- XA4-XA1 G_image: `{fmt(xa4_delta['G_image'] - xa1_delta['G_image'])}` pp",
            f"- XA4-XA1 G_pixel: `{fmt(xa4_delta['G_pixel'] - xa1_delta['G_pixel'])}` pp",
            f"- XA4-XA1 G_balanced: `{fmt(xa4_delta['G_balanced'] - xa1_delta['G_balanced'])}` pp",
        ])
    payload = {
        "outcome": outcome,
        "next_action": next_action,
        "xa4_g_image": xa4_delta["G_image"],
        "xa4_g_pixel": xa4_delta["G_pixel"],
        "xa4_g_balanced": xa4_delta["G_balanced"],
        "specificity_image": specificity["G_image"],
        "specificity_pixel": specificity["G_pixel"],
        "specificity_balanced": specificity["G_balanced"],
    }
    (OUT / "READOUT_D2_FOLD0_EXISTING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "D2_FOLD0_EXISTING_OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "READOUT_D2_FOLD0_EXISTING.md")}, indent=2))


if __name__ == "__main__":
    main()
