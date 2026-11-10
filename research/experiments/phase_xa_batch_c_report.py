import csv
import json
import math
from pathlib import Path

import numpy as np


OUT = Path("results/phase_xa_memory_distillation")
PRIMARY = ["I-AUROC", "I-AP", "P-AUROC", "PRO"]
MAIN_VARIANTS = [
    "XA0_OfficialMRADCLIP",
    "XA1_HardLabelAdapter",
    "XA2_CLIPPreservedAdapter",
    "XA4_DualTeacherFullMemory",
    "XA5_DualTeacherCompactMemory",
    "XA6_RandomMemoryTeacher",
]


def load_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def fmt(x, nd=4):
    return "nan" if not math.isfinite(float(x)) else f"{float(x):.{nd}f}"


def rid(row):
    if row["variant"] == "XA6_RandomMemoryTeacher":
        return f"XA6_seed{row['seed']}"
    return row["variant"]


def rows_for_fold(fold):
    rows = load_csv(OUT / f"source_checkpoint_readout_D1_fold{fold}.csv")
    wanted = []
    for row in rows:
        if row["variant"] in MAIN_VARIANTS:
            if row["variant"] == "XA6_RandomMemoryTeacher" and int(row["seed"]) not in (0, 1, 2):
                continue
            wanted.append(row)
    return wanted


def deltas(rows):
    xa2 = next(r for r in rows if r["variant"] == "XA2_CLIPPreservedAdapter")
    base = {m: f(xa2[m]) for m in PRIMARY}
    out = {}
    for row in rows:
        d = {m: f(row[m]) - base[m] for m in PRIMARY}
        d["G_image"] = 0.5 * d["I-AUROC"] + 0.5 * d["I-AP"]
        d["G_pixel"] = 0.5 * d["P-AUROC"] + 0.5 * d["PRO"]
        d["G_balanced"] = 0.5 * d["G_image"] + 0.5 * d["G_pixel"]
        out[rid(row)] = d
    return out


def random_stats(rows, delta):
    ids = [rid(r) for r in rows if r["variant"] == "XA6_RandomMemoryTeacher" and int(r["seed"]) in (0, 1, 2)]
    stats = {}
    for key in PRIMARY + ["G_image", "G_pixel", "G_balanced"]:
        vals = []
        for row in rows:
            if rid(row) not in ids:
                continue
            vals.append(delta[rid(row)][key] if key.startswith("G_") else f(row[key]))
        vals = np.asarray(vals, dtype=float)
        stats[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
    return stats


def specificity(rows, delta):
    xa4 = delta["XA4_DualTeacherFullMemory"]
    rand = random_stats(rows, delta)
    out = {}
    for key in PRIMARY + ["G_image", "G_pixel", "G_balanced"]:
        real = xa4[key] if key.startswith("G_") else f(next(r for r in rows if r["variant"] == "XA4_DualTeacherFullMemory")[key])
        out[key] = real - rand[key]["mean"]
    return out


def mean_row_for_variant(rows, delta, variant):
    if variant == "XA6mean":
        rand = random_stats(rows, delta)
        return {m: rand[m]["mean"] for m in PRIMARY + ["G_image", "G_pixel", "G_balanced"]}
    row = next(r for r in rows if r["variant"] == variant)
    d = delta[rid(row)]
    return {m: f(row[m]) for m in PRIMARY} | {k: d[k] for k in ["G_image", "G_pixel", "G_balanced"]}


def pearson(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    def ranks(x):
        order = np.argsort(x)
        out = np.empty_like(order, dtype=float)
        out[order] = np.arange(len(x), dtype=float)
        return out
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return pearson(ranks(a[mask]), ranks(b[mask]))


def score_map_by_variant(fold):
    rows = load_csv(OUT / f"source_checkpoint_scores_D1_fold{fold}.csv")
    out = {}
    for row in rows:
        key = rid(row)
        out.setdefault(key, []).append((int(row["idx"]), f(row["score"])))
    return {k: np.asarray([v for _, v in sorted(vals)], dtype=float) for k, vals in out.items()}


def load_map64(fold, variant, seed=0):
    path = OUT / f"source_checkpoint_map64_D1_fold{fold}_{variant}_seed{seed}.npz"
    if not path.exists():
        return None
    return np.load(path)["maps"].astype(np.float32)


def correlation_diagnostics():
    rows = []
    fold = 1
    scores = score_map_by_variant(fold)
    if "XA6_seed0" in scores and "XA6_seed1" in scores and "XA6_seed2" in scores:
        scores["XA6mean"] = np.mean([scores[f"XA6_seed{i}"] for i in (0, 1, 2)], axis=0)
    pairs = [("XA4_DualTeacherFullMemory", "XA2_CLIPPreservedAdapter"), ("XA4_DualTeacherFullMemory", "XA6mean")]
    maps = {
        "XA4_DualTeacherFullMemory": load_map64(fold, "XA4_DualTeacherFullMemory", 0),
        "XA2_CLIPPreservedAdapter": load_map64(fold, "XA2_CLIPPreservedAdapter", 0),
    }
    m6 = [load_map64(fold, "XA6_RandomMemoryTeacher", i) for i in (0, 1, 2)]
    if all(x is not None for x in m6):
        maps["XA6mean"] = np.mean(m6, axis=0)
    for a, b in pairs:
        row = {"direction": "D1", "fold": 1, "pair": f"{a}_vs_{b}"}
        if a in scores and b in scores:
            row["image_score_pearson"] = pearson(scores[a], scores[b])
            row["image_score_spearman"] = spearman(scores[a], scores[b])
        else:
            row["image_score_pearson"] = "NA"
            row["image_score_spearman"] = "NA"
        if a in maps and b in maps:
            row["pixel_map_pearson"] = pearson(maps[a], maps[b])
            row["pixel_map_spearman"] = spearman(maps[a], maps[b])
            row["pixel_map_resolution"] = "64x64"
        else:
            row["pixel_map_pearson"] = "NA"
            row["pixel_map_spearman"] = "NA"
            row["pixel_map_resolution"] = "not_available"
        rows.append(row)
    write_csv(OUT / "output_correlation_diagnostics.csv", rows)
    return rows


def per_category_analysis():
    all_rows = []
    summaries = []
    for fold in (0, 1):
        rows = load_csv(OUT / f"per_category_source_readout_D1_fold{fold}.csv")
        all_rows.extend(rows)
        cats = sorted({r["category"] for r in rows})
        lookup = {(r["variant"], r["seed"], r["category"]): r for r in rows}
        for metric in ["I-AUROC", "P-AUROC", "PRO"]:
            d_xa2, d_xa6 = [], []
            for cat in cats:
                r4 = lookup.get(("XA4_DualTeacherFullMemory", "0", cat))
                r2 = lookup.get(("XA2_CLIPPreservedAdapter", "0", cat))
                r6s = [lookup.get(("XA6_RandomMemoryTeacher", str(i), cat)) for i in (0, 1, 2)]
                r6s = [r for r in r6s if r]
                if not r4 or not r2:
                    continue
                d_xa2.append((cat, f(r4[metric]) - f(r2[metric])))
                if r6s:
                    d_xa6.append((cat, f(r4[metric]) - float(np.nanmean([f(r[metric]) for r in r6s]))))
            vals = [v for _, v in d_xa2 if math.isfinite(v)]
            vals6 = [v for _, v in d_xa6 if math.isfinite(v)]
            if vals:
                pos = [max(0.0, v) for v in vals]
                concentration = max(pos) / sum(pos) if sum(pos) > 0 else 0.0
                summaries.append({
                    "direction": "D1",
                    "fold": fold,
                    "metric": metric,
                    "xa4_vs_xa2_wins": sum(v > 0.05 for v in vals),
                    "xa4_vs_xa2_ties": sum(abs(v) <= 0.05 for v in vals),
                    "xa4_vs_xa2_losses": sum(v < -0.05 for v in vals),
                    "xa4_vs_xa2_median_delta": float(np.nanmedian(vals)),
                    "xa4_vs_xa2_worst_delta": float(np.nanmin(vals)),
                    "xa4_vs_xa2_best_delta": float(np.nanmax(vals)),
                    "xa4_vs_xa6_wins": sum(v > 0.05 for v in vals6),
                    "xa4_vs_xa6_ties": sum(abs(v) <= 0.05 for v in vals6),
                    "xa4_vs_xa6_losses": sum(v < -0.05 for v in vals6),
                    "gain_concentration": "CATEGORY_CONCENTRATED_GAIN" if concentration > 0.70 else "CATEGORY_DISTRIBUTED_GAIN",
                })
    write_csv(OUT / "per_category_source_readout.csv", all_rows)
    return summaries


def main():
    fold_rows = {0: rows_for_fold(0), 1: rows_for_fold(1)}
    if not fold_rows[0] or not fold_rows[1]:
        raise SystemExit("Missing fold0/fold1 readout artifacts")
    fold_delta = {fold: deltas(rows) for fold, rows in fold_rows.items()}
    fold_rand = {fold: random_stats(fold_rows[fold], fold_delta[fold]) for fold in (0, 1)}
    fold_spec = {fold: specificity(fold_rows[fold], fold_delta[fold]) for fold in (0, 1)}

    # Verify Batch B values from artifacts.
    b_verify = {
        "G_image_XA4": fold_delta[0]["XA4_DualTeacherFullMemory"]["G_image"],
        "G_pixel_XA4": fold_delta[0]["XA4_DualTeacherFullMemory"]["G_pixel"],
        "G_balanced_XA4": fold_delta[0]["XA4_DualTeacherFullMemory"]["G_balanced"],
        "specificity_image": fold_spec[0]["G_image"],
        "specificity_pixel": fold_spec[0]["G_pixel"],
        "specificity_balanced": fold_spec[0]["G_balanced"],
        "units": "percentage_points",
    }

    spec_rows = []
    for fold in (0, 1):
        for key in PRIMARY + ["G_image", "G_pixel", "G_balanced"]:
            spec_rows.append({
                "direction": "D1",
                "fold": fold,
                "metric": key,
                "xa4_minus_xa6mean": fold_spec[fold][key],
            })
    write_csv(OUT / "memory_specificity.csv", spec_rows)

    pooled_rows = []
    for variant in ["XA1_HardLabelAdapter", "XA2_CLIPPreservedAdapter", "XA4_DualTeacherFullMemory", "XA5_DualTeacherCompactMemory", "XA6mean"]:
        values_by_fold = []
        for fold in (0, 1):
            values_by_fold.append(mean_row_for_variant(fold_rows[fold], fold_delta[fold], variant))
        row = {"direction": "D1", "variant": variant, "folds": "0,1"}
        for key in PRIMARY + ["G_image", "G_pixel", "G_balanced"]:
            vals = [v[key] for v in values_by_fold]
            row[f"mean_{key}"] = float(np.nanmean(vals))
            row[f"worst_{key}"] = float(np.nanmin(vals))
        pooled_rows.append(row)

    corr_rows = correlation_diagnostics()
    cat_summary = per_category_analysis()

    image_signal = "IMAGE_MEMORY_SIGNAL_NOT_REPLICATED"
    image_fold_ok = [
        fold_delta[fold]["XA4_DualTeacherFullMemory"]["G_image"] > 0.15
        and fold_spec[fold]["G_image"] > 0.10
        for fold in (0, 1)
    ]
    if all(image_fold_ok):
        image_signal = "IMAGE_MEMORY_SIGNAL_STRONG"
    elif all(fold_delta[fold]["XA4_DualTeacherFullMemory"]["G_image"] > 0 for fold in (0, 1)) and np.mean([fold_spec[fold]["G_image"] for fold in (0, 1)]) > 0.10:
        image_signal = "IMAGE_MEMORY_SIGNAL_PARTIAL"

    pixel_fold = []
    for fold in (0, 1):
        xa4_pix = fold_delta[fold]["XA4_DualTeacherFullMemory"]["G_pixel"]
        xa6_pix = fold_rand[fold]["G_pixel"]["mean"]
        if xa4_pix > 0 and xa6_pix > 0 and xa6_pix >= xa4_pix:
            pixel_fold.append("generic")
        elif xa4_pix > xa6_pix:
            pixel_fold.append("specific")
        elif xa4_pix <= 0 and xa6_pix <= 0:
            pixel_fold.append("none")
        else:
            pixel_fold.append("mixed")
    if all(x == "specific" for x in pixel_fold):
        pixel_signal = "PIXEL_MEMORY_SPECIFIC"
    elif all(x in ("generic", "mixed") for x in pixel_fold):
        pixel_signal = "PIXEL_GENERIC_REGULARIZATION"
    elif all(x == "none" for x in pixel_fold):
        pixel_signal = "PIXEL_NO_GAIN"
    else:
        pixel_signal = "PIXEL_INCONSISTENT"

    compact_equiv = True
    compact_diffs = []
    for fold in (0, 1):
        xa4 = next(r for r in fold_rows[fold] if r["variant"] == "XA4_DualTeacherFullMemory")
        xa5 = next(r for r in fold_rows[fold] if r["variant"] == "XA5_DualTeacherCompactMemory")
        d4 = fold_delta[fold]["XA4_DualTeacherFullMemory"]
        d5 = fold_delta[fold]["XA5_DualTeacherCompactMemory"]
        for key in PRIMARY:
            diff = f(xa4[key]) - f(xa5[key])
            compact_diffs.append({"fold": fold, "metric": key, "xa4_minus_xa5": diff})
            compact_equiv = compact_equiv and abs(diff) <= 0.05
        for key in ["G_image", "G_pixel", "G_balanced"]:
            diff = d4[key] - d5[key]
            compact_diffs.append({"fold": fold, "metric": key, "xa4_minus_xa5": diff})
            compact_equiv = compact_equiv and abs(diff) <= 0.05

    mean_spec_image = float(np.mean([fold_spec[f]["G_image"] for f in (0, 1)]))
    mean_spec_pixel = float(np.mean([fold_spec[f]["G_pixel"] for f in (0, 1)]))
    mean_spec_bal = float(np.mean([fold_spec[f]["G_balanced"] for f in (0, 1)]))
    xa4_mean_g_image = float(np.mean([fold_delta[f]["XA4_DualTeacherFullMemory"]["G_image"] for f in (0, 1)]))
    xa4_mean_g_pixel = float(np.mean([fold_delta[f]["XA4_DualTeacherFullMemory"]["G_pixel"] for f in (0, 1)]))
    xa4_beats_xa1 = all(
        f(next(r for r in fold_rows[fid] if r["variant"] == "XA4_DualTeacherFullMemory")["I-AUROC"])
        >= f(next(r for r in fold_rows[fid] if r["variant"] == "XA1_HardLabelAdapter")["I-AUROC"])
        for fid in (0, 1)
    )
    major_drops = []
    for key in PRIMARY:
        mean_delta = float(np.mean([fold_delta[f]["XA4_DualTeacherFullMemory"][key] for f in (0, 1)]))
        if mean_delta < -0.20:
            major_drops.append((key, mean_delta))

    if image_signal == "IMAGE_MEMORY_SIGNAL_STRONG" and xa4_mean_g_image >= 0.15 and mean_spec_image >= 0.10 and xa4_beats_xa1 and not major_drops:
        outcome = "C1_REAL_MEMORY_REPLICATED_IMAGE_ONLY"
    elif (
        all(fold_delta[f]["XA4_DualTeacherFullMemory"]["G_balanced"] > 0 for f in (0, 1))
        and all(fold_rand[f]["G_balanced"]["mean"] > 0 for f in (0, 1))
        and mean_spec_bal < 0.10
    ):
        outcome = "C2_GENERIC_REGULARIZATION_REPLICATED"
    elif image_signal == "IMAGE_MEMORY_SIGNAL_NOT_REPLICATED":
        outcome = "C3_FOLD_DEPENDENT_MEMORY_SIGNAL"
    elif any(fold_delta[f]["XA4_DualTeacherFullMemory"]["G_balanced"] <= 0 for f in (0, 1)):
        outcome = "C4_NO_REPRODUCIBLE_GAIN"
    else:
        outcome = "C3_FOLD_DEPENDENT_MEMORY_SIGNAL"

    next_batch = "existing D2 fold0 checkpoint readout only" if outcome.startswith("C1") else "NO_NEW_TRAINING_JUSTIFIED"
    if outcome == "C3_FOLD_DEPENDENT_MEMORY_SIGNAL":
        next_batch = "D1 fold2 tie-break only: XA1, XA2, XA4, XA6 seed0, XA6 seed1"

    combined = []
    for fold in (0, 1):
        combined.extend(fold_rows[fold])
    write_csv(OUT / "source_checkpoint_readout.csv", combined)

    lines = [
        "# Readout Batch C: D1 Fold1 and Two-Fold Confirmation",
        "",
        f"Outcome: **{outcome}**",
        "",
        "## Batch B Verification",
        "",
        f"- Fold0 XA4 G_image: `{fmt(b_verify['G_image_XA4'])}` pp",
        f"- Fold0 XA4 G_pixel: `{fmt(b_verify['G_pixel_XA4'])}` pp",
        f"- Fold0 XA4 G_balanced: `{fmt(b_verify['G_balanced_XA4'])}` pp",
        f"- Fold0 specificity image: `{fmt(b_verify['specificity_image'])}` pp",
        f"- Fold0 specificity pixel: `{fmt(b_verify['specificity_pixel'])}` pp",
        f"- Fold0 specificity balanced: `{fmt(b_verify['specificity_balanced'])}` pp",
        "- Units verified as percentage points.",
        "",
        "## Fold1 Raw Metrics",
        "",
        "| Variant | Seed | I-AUROC | I-AP | P-AUROC | PRO | G_image | G_pixel | G_balanced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_rows[1]:
        d = fold_delta[1][rid(row)]
        lines.append(
            f"| {row['variant']} | {row['seed']} | {fmt(f(row['I-AUROC']),6)} | {fmt(f(row['I-AP']),6)} | {fmt(f(row['P-AUROC']),6)} | {fmt(f(row['PRO']),6)} | {fmt(d['G_image'])} | {fmt(d['G_pixel'])} | {fmt(d['G_balanced'])} |"
        )
    lines.extend([
        "",
        "## Fold1 Random Control",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ])
    for key in PRIMARY + ["G_image", "G_pixel", "G_balanced"]:
        s = fold_rand[1][key]
        lines.append(f"| {key} | {fmt(s['mean'])} | {fmt(s['std'])} | {fmt(s['min'])} | {fmt(s['max'])} |")
    lines.extend([
        "",
        "## Fold1 Real Memory Specificity",
        "",
        "| Metric | XA4 - XA6mean |",
        "|---|---:|",
    ])
    for key in PRIMARY + ["G_image", "G_pixel", "G_balanced"]:
        lines.append(f"| {key} | {fmt(fold_spec[1][key])} |")
    lines.extend([
        "",
        "## Replication Test",
        "",
        f"- XA4 > XA2 on image: `{'REPLICATED' if fold_delta[1]['XA4_DualTeacherFullMemory']['G_image'] > 0 else 'NOT_REPLICATED'}`",
        f"- XA4 > XA2 on pixel: `{'REPLICATED' if fold_delta[1]['XA4_DualTeacherFullMemory']['G_pixel'] > 0 else 'NOT_REPLICATED'}`",
        f"- RealMemory > RandomMemory on image: `{'REPLICATED' if fold_spec[1]['G_image'] > 0 else 'NOT_REPLICATED'}`",
        f"- Random may exceed real on pixel: `{'REPLICATED' if fold_spec[1]['G_pixel'] <= 0 else 'NOT_REPLICATED'}`",
        f"- Balanced specificity positive: `{'REPLICATED' if fold_spec[1]['G_balanced'] > 0 else 'NOT_REPLICATED'}`",
        "",
        "## Two-Fold Pooled Analysis",
        "",
        "| Variant | Mean I-AUROC | Mean I-AP | Mean P-AUROC | Mean PRO | Mean G_image | Mean G_pixel | Mean G_balanced | Worst G_balanced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in pooled_rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['mean_I-AUROC'])} | {fmt(row['mean_I-AP'])} | {fmt(row['mean_P-AUROC'])} | {fmt(row['mean_PRO'])} | {fmt(row['mean_G_image'])} | {fmt(row['mean_G_pixel'])} | {fmt(row['mean_G_balanced'])} | {fmt(row['worst_G_balanced'])} |"
        )
    lines.extend([
        "",
        "## Image-Side Causal Test",
        "",
        f"- Classification: `{image_signal}`",
        f"- MeanRealSpecificity_image: `{fmt(mean_spec_image)}` pp",
        "",
        "## Pixel-Side Causal Test",
        "",
        f"- Classification: `{pixel_signal}`",
        f"- MeanRealSpecificity_pixel: `{fmt(mean_spec_pixel)}` pp",
        "",
        "## Full vs Compact Teacher",
        "",
        f"- Status: `{'FULL_COMPACT_TEACHER_TASK_EQUIVALENT_ON_D1' if compact_equiv else 'FULL_COMPACT_TEACHER_DIFFERS_ON_D1'}`",
        "",
        "## Hard-Label / Preservation Comparison",
        "",
        "| Fold | Metric | XA4-XA1 | XA4-XA2 |",
        "|---:|---|---:|---:|",
    ])
    for fold in (0, 1):
        r1 = next(r for r in fold_rows[fold] if r["variant"] == "XA1_HardLabelAdapter")
        r2 = next(r for r in fold_rows[fold] if r["variant"] == "XA2_CLIPPreservedAdapter")
        r4 = next(r for r in fold_rows[fold] if r["variant"] == "XA4_DualTeacherFullMemory")
        for key in PRIMARY:
            lines.append(f"| {fold} | {key} | {fmt(f(r4[key])-f(r1[key]))} | {fmt(f(r4[key])-f(r2[key]))} |")
    lines.extend([
        "",
        "## Per-Category Causal Analysis",
        "",
        "| Fold | Metric | XA4-XA2 W/T/L | XA4-XA6 W/T/L | Median XA4-XA2 | Worst XA4-XA2 | Best XA4-XA2 | Gain Spread |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for s in cat_summary:
        lines.append(
            f"| {s['fold']} | {s['metric']} | {s['xa4_vs_xa2_wins']}/{s['xa4_vs_xa2_ties']}/{s['xa4_vs_xa2_losses']} | "
            f"{s['xa4_vs_xa6_wins']}/{s['xa4_vs_xa6_ties']}/{s['xa4_vs_xa6_losses']} | {fmt(s['xa4_vs_xa2_median_delta'])} | {fmt(s['xa4_vs_xa2_worst_delta'])} | {fmt(s['xa4_vs_xa2_best_delta'])} | {s['gain_concentration']} |"
        )
    lines.extend([
        "",
        "## Output Correlation Diagnostic",
        "",
        "| Pair | Image Pearson | Image Spearman | Pixel Pearson | Pixel Spearman | Pixel Resolution |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for r in corr_rows:
        lines.append(f"| {r['pair']} | {fmt(f(r['image_score_pearson']))} | {fmt(f(r['image_score_spearman']))} | {fmt(f(r['pixel_map_pearson']))} | {fmt(f(r['pixel_map_spearman']))} | {r['pixel_map_resolution']} |")
    lines.extend([
        "",
        "## Next Minimal Batch",
        "",
        next_batch,
    ])
    (OUT / "READOUT_BATCH_C_D1_FOLD1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "READOUT_BATCH_C_D1_TWO_FOLDS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "BATCH_C_OUTCOME.json").write_text(json.dumps({
        "outcome": outcome,
        "image_signal": image_signal,
        "pixel_signal": pixel_signal,
        "mean_specificity_image": mean_spec_image,
        "mean_specificity_pixel": mean_spec_pixel,
        "mean_specificity_balanced": mean_spec_bal,
        "next_minimal_batch": next_batch,
        "batch_b_verification": b_verify,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "READOUT_BATCH_C_D1_TWO_FOLDS.md")}, indent=2))


if __name__ == "__main__":
    main()
