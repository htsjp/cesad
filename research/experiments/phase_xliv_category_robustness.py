import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv


IN = Path("results/phase_xliv_state_alignment_confirmation")
OUT = Path("results/phase_xliv_category_robustness")
FOLDS = [("D1", 0), ("D1", 1), ("D2", 0), ("D2", 1)]
TIE = 0.05


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def ifold(v):
    return int(float(v))


def gp(row):
    return 0.5 * (fnum(row["P-AUROC"]) + fnum(row["P-AP"]))


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def rankdata(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            idx = inv == i
            ranks[idx] = ranks[idx].mean()
    return ranks


def gini(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.sum(x) <= 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    return float((2.0 * np.sum((np.arange(1, n + 1) * x)) / (n * np.sum(x))) - (n + 1.0) / n)


def wl(delta):
    return "tie" if abs(delta) <= TIE else ("win" if delta > 0 else "loss")


def load_per_category():
    rows = read_csv(IN / "PER_CATEGORY.csv")
    by = {}
    for r in rows:
        direction = r["direction"]
        fold = ifold(r["fold"])
        category = r["category"]
        variant = r["variant"]
        # C1/A0 duplicates are intentional in XLIV-A; keep first exact key only.
        key = (direction, fold, category, variant)
        if key not in by:
            rr = dict(r)
            rr["fold"] = fold
            rr["G_pixel"] = gp(rr)
            by[key] = rr
    return by


def primary_gain_rows(by):
    rows = []
    for direction, fold in FOLDS:
        cats = sorted({k[2] for k in by if k[0] == direction and k[1] == fold})
        for cat in cats:
            c0 = by.get((direction, fold, cat, "C0_SINGLE_STATE"))
            c1 = by.get((direction, fold, cat, "C1_STATE_ALIGNED"))
            c2 = by.get((direction, fold, cat, "C2_INSTANCE_MEMORY"))
            if not (c0 and c1 and c2):
                continue
            rows.append({
                "direction": direction,
                "fold": fold,
                "category": cat,
                "C0_P-AUROC": fnum(c0["P-AUROC"]),
                "C1_P-AUROC": fnum(c1["P-AUROC"]),
                "delta_P-AUROC": fnum(c1["P-AUROC"]) - fnum(c0["P-AUROC"]),
                "C0_P-AP": fnum(c0["P-AP"]),
                "C1_P-AP": fnum(c1["P-AP"]),
                "delta_P-AP": fnum(c1["P-AP"]) - fnum(c0["P-AP"]),
                "C0_G_pixel": gp(c0),
                "C1_G_pixel": gp(c1),
                "C2_G_pixel": gp(c2),
                "delta_G": gp(c1) - gp(c0),
                "C1_minus_C2_delta_G": gp(c1) - gp(c2),
            })
    return rows


def fold_distribution(rows):
    out = []
    for direction, fold in FOLDS:
        vals = np.asarray([r["delta_G"] for r in rows if r["direction"] == direction and r["fold"] == fold], dtype=np.float64)
        out.append({
            "direction": direction,
            "fold": fold,
            "category_count": int(vals.size),
            "mean_delta_G": float(np.mean(vals)),
            "median_delta_G": float(np.median(vals)),
            "q25_delta_G": float(np.quantile(vals, 0.25)),
            "q75_delta_G": float(np.quantile(vals, 0.75)),
            "min_delta_G": float(np.min(vals)),
            "max_delta_G": float(np.max(vals)),
            "positive_category_ratio": float(np.mean(vals > 0)),
            "gt_1pp_ratio": float(np.mean(vals > 1.0)),
            "gt_3pp_ratio": float(np.mean(vals > 3.0)),
            "negative_ratio": float(np.mean(vals < 0)),
        })
    return out


def equal_category_mean(rows):
    raw_gain = {(r["direction"], ifold(r["fold"])): fnum(r["delta_G_pixel_pp"]) for r in read_csv(IN / "STATE_ALIGNMENT_GAIN.csv")}
    out = []
    for direction, fold in FOLDS:
        vals = np.asarray([r["delta_G"] for r in rows if r["direction"] == direction and r["fold"] == fold], dtype=np.float64)
        out.append({
            "direction": direction,
            "fold": fold,
            "equal_category_mean_delta_G": float(np.mean(vals)),
            "median_category_delta_G": float(np.median(vals)),
            "original_aggregate_delta_G": raw_gain[(direction, fold)],
            "equal_minus_original_delta_G": float(np.mean(vals) - raw_gain[(direction, fold)]),
        })
    return out


def leave_one_out(rows):
    out = []
    for direction, fold in FOLDS:
        fold_rows = [r for r in rows if r["direction"] == direction and r["fold"] == fold]
        for removed in sorted(r["category"] for r in fold_rows):
            vals = np.asarray([r["delta_G"] for r in fold_rows if r["category"] != removed], dtype=np.float64)
            out.append({
                "direction": direction,
                "fold": fold,
                "removed_category": removed,
                "remaining_category_count": int(vals.size),
                "remaining_mean_gain": float(np.mean(vals)),
                "remaining_median_gain": float(np.median(vals)),
                "remaining_positive_ratio": float(np.mean(vals > 0)),
            })
    return out


def concentration(rows):
    out = []
    for direction, fold in FOLDS:
        vals = [(r["category"], max(r["delta_G"], 0.0)) for r in rows if r["direction"] == direction and r["fold"] == fold]
        pos = np.asarray([v for _c, v in vals if v > 0], dtype=np.float64)
        total = float(np.sum(pos))
        shares = sorted([v / total for _c, v in vals if v > 0], reverse=True) if total > 0 else []
        out.append({
            "direction": direction,
            "fold": fold,
            "positive_gain_sum": total,
            "positive_category_count": int(pos.size),
            "HHI": float(np.sum(np.asarray(shares) ** 2)) if shares else float("nan"),
            "Gini": gini(pos),
            "top1_share": float(shares[0]) if shares else float("nan"),
            "top2_share": float(sum(shares[:2])) if shares else float("nan"),
        })
    return out


def baseline_correlation(rows):
    out = []
    for direction, fold in FOLDS:
        vals = [r for r in rows if r["direction"] == direction and r["fold"] == fold]
        rho_g = spearman([r["C0_G_pixel"] for r in vals], [r["delta_G"] for r in vals])
        rho_ap = spearman([r["C0_P-AP"] for r in vals], [r["delta_P-AP"] for r in vals])
        out.append({
            "direction": direction,
            "fold": fold,
            "spearman_C0_G_vs_delta_G": rho_g,
            "spearman_C0_PAP_vs_delta_PAP": rho_ap,
            "baseline_floor_dependent": bool(np.isfinite(rho_g) and rho_g <= -0.70),
        })
    for direction in ["D1", "D2"]:
        vals = [r for r in rows if r["direction"] == direction]
        rho_g = spearman([r["C0_G_pixel"] for r in vals], [r["delta_G"] for r in vals])
        rho_ap = spearman([r["C0_P-AP"] for r in vals], [r["delta_P-AP"] for r in vals])
        out.append({
            "direction": direction,
            "fold": "all",
            "spearman_C0_G_vs_delta_G": rho_g,
            "spearman_C0_PAP_vs_delta_PAP": rho_ap,
            "baseline_floor_dependent": bool(np.isfinite(rho_g) and rho_g <= -0.70),
        })
    return out


def instance_memory_rows(rows):
    out = []
    for r in rows:
        d = r["C1_minus_C2_delta_G"]
        out.append({
            "direction": r["direction"],
            "fold": r["fold"],
            "category": r["category"],
            "C1_G_pixel": r["C1_G_pixel"],
            "C2_G_pixel": r["C2_G_pixel"],
            "C1_minus_C2_delta_G": d,
            "C1_vs_C2": wl(d),
        })
    for direction, fold in FOLDS:
        vals = [r for r in out if r["direction"] == direction and r["fold"] == fold]
        ratio = np.mean([r["C1_vs_C2"] in ("win", "tie") for r in vals])
        out.append({
            "direction": direction,
            "fold": fold,
            "category": "__SUMMARY__",
            "C1_G_pixel": float("nan"),
            "C2_G_pixel": float("nan"),
            "C1_minus_C2_delta_G": float("nan"),
            "C1_vs_C2": f"C1_ge_C2_ratio={ratio:.4f}",
        })
    return out


def anchor_specificity_rows(by):
    out = []
    for direction, fold in FOLDS:
        cats = sorted({k[2] for k in by if k[0] == direction and k[1] == fold})
        fold_rows = []
        for cat in cats:
            a0 = by.get((direction, fold, cat, "A0_LEARNED"))
            a2 = by.get((direction, fold, cat, "A2_SHUFFLED"))
            a3 = by.get((direction, fold, cat, "A3_RANDOM_ANGLE"))
            if not (a0 and a2 and a3):
                continue
            row = {
                "direction": direction,
                "fold": fold,
                "category": cat,
                "learned_G": gp(a0),
                "shuffled_G": gp(a2),
                "random_G": gp(a3),
                "learned_minus_shuffled": gp(a0) - gp(a2),
                "learned_minus_random": gp(a0) - gp(a3),
            }
            out.append(row)
            fold_rows.append(row)
        if fold_rows:
            out.append({
                "direction": direction,
                "fold": fold,
                "category": "__SUMMARY__",
                "learned_G": float("nan"),
                "shuffled_G": float("nan"),
                "random_G": float("nan"),
                "learned_minus_shuffled": float(np.median([r["learned_minus_shuffled"] for r in fold_rows])),
                "learned_minus_random": float(np.median([r["learned_minus_random"] for r in fold_rows])),
                "positive_ratio_learned_minus_shuffled": float(np.mean([r["learned_minus_shuffled"] > 0 for r in fold_rows])),
                "positive_ratio_learned_minus_random": float(np.mean([r["learned_minus_random"] > 0 for r in fold_rows])),
            })
    return out


def chewinggum_sensitivity(rows, eq_rows, loco_rows):
    d2f0 = [r for r in rows if r["direction"] == "D2" and r["fold"] == 0]
    remain = [r["delta_G"] for r in d2f0 if r["category"] != "chewinggum"]
    original = next(r for r in eq_rows if r["direction"] == "D2" and r["fold"] == 0)
    removed = next(r for r in loco_rows if r["direction"] == "D2" and r["fold"] == 0 and r["removed_category"] == "chewinggum")
    ok = removed["remaining_mean_gain"] > 2.0 and removed["remaining_positive_ratio"] >= 0.60
    return {
        "original_equal_category_mean": original["equal_category_mean_delta_G"],
        "original_median_category_gain": original["median_category_delta_G"],
        "original_aggregate_gain": original["original_aggregate_delta_G"],
        "remove_chewinggum_equal_category_mean": removed["remaining_mean_gain"],
        "remove_chewinggum_median_category_gain": removed["remaining_median_gain"],
        "remove_chewinggum_positive_category_ratio": removed["remaining_positive_ratio"],
        "CHEWINGGUM_NOT_REQUIRED_FOR_SIGNAL": ok,
    }


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    by = load_per_category()
    gains = primary_gain_rows(by)
    dist = fold_distribution(gains)
    eq = equal_category_mean(gains)
    loco = leave_one_out(gains)
    chew = chewinggum_sensitivity(gains, eq, loco)
    conc = concentration(gains)
    corr = baseline_correlation(gains)
    im = instance_memory_rows(gains)
    anchor = anchor_specificity_rows(by)

    dists_by = {(r["direction"], r["fold"]): r for r in dist}
    eq_by = {(r["direction"], r["fold"]): r for r in eq}
    med_positive = all(dists_by[(d, f)]["median_delta_G"] > 0 for d, f in FOLDS)
    pos_ratio = all(dists_by[(d, f)]["positive_category_ratio"] >= 0.60 for d, f in FOLDS)
    eq_count = sum(eq_by[(d, f)]["equal_category_mean_delta_G"] >= 2.0 for d, f in FOLDS)
    d1_floor = any(r["direction"] == "D1" and r["baseline_floor_dependent"] for r in corr)
    d2_floor = any(r["direction"] == "D2" and r["baseline_floor_dependent"] for r in corr)
    robust = med_positive and pos_ratio and chew["CHEWINGGUM_NOT_REQUIRED_FOR_SIGNAL"] and eq_count >= 3 and not (d1_floor and d2_floor)
    if robust:
        outcome = "XLIV_B_CATEGORY_GENERALIZATION_ROBUST"
    elif not chew["CHEWINGGUM_NOT_REQUIRED_FOR_SIGNAL"] or any(dists_by[(d, f)]["median_delta_G"] <= 0 for d, f in FOLDS):
        outcome = "XLIV_B_CATEGORY_GENERALIZATION_WEAK"
    else:
        outcome = "XLIV_B_FINE_GRAINED_ANOMALY_DEPENDENCE"

    write_csv(OUT / "PER_CATEGORY_GAIN.csv", gains)
    write_csv(OUT / "FOLD_GAIN_DISTRIBUTION.csv", dist)
    write_csv(OUT / "EQUAL_CATEGORY_MEAN.csv", eq)
    write_csv(OUT / "LEAVE_ONE_CATEGORY_OUT.csv", loco)
    write_json(OUT / "CHEWINGGUM_SENSITIVITY.json", chew)
    write_csv(OUT / "GAIN_CONCENTRATION.csv", conc)
    write_csv(OUT / "BASELINE_DIFFICULTY_CORRELATION.csv", corr)
    write_csv(OUT / "INSTANCE_MEMORY_PER_CATEGORY.csv", im)
    write_csv(OUT / "ANCHOR_SPECIFICITY_PER_CATEGORY.csv", anchor)

    report = ["# Phase XLIV-B Category Generalization Robustness Audit", "", f"Outcome: **{outcome}**", "", "## Fold Gain Distribution", ""]
    for r in dist:
        report.append(f"- {r['direction']} fold{r['fold']}: mean {r['mean_delta_G']:+.4f}, median {r['median_delta_G']:+.4f}, positive ratio {r['positive_category_ratio']:.2f}, >3pp ratio {r['gt_3pp_ratio']:.2f}.")
    report.extend(["", "## Equal-Category Mean", ""])
    for r in eq:
        report.append(f"- {r['direction']} fold{r['fold']}: equal mean {r['equal_category_mean_delta_G']:+.4f}, original aggregate {r['original_aggregate_delta_G']:+.4f}.")
    report.extend(["", "## Chewinggum Sensitivity", ""])
    report.append(f"- D2 fold0 without chewinggum: mean {chew['remove_chewinggum_equal_category_mean']:+.4f}, median {chew['remove_chewinggum_median_category_gain']:+.4f}, positive ratio {chew['remove_chewinggum_positive_category_ratio']:.2f}.")
    report.append(f"- CHEWINGGUM_NOT_REQUIRED_FOR_SIGNAL: `{chew['CHEWINGGUM_NOT_REQUIRED_FOR_SIGNAL']}`.")
    report.extend(["", "## Concentration", ""])
    for r in conc:
        report.append(f"- {r['direction']} fold{r['fold']}: HHI {r['HHI']:.4f}, Gini {r['Gini']:.4f}, Top1 {r['top1_share']:.4f}, Top2 {r['top2_share']:.4f}.")
    report.extend(["", "## Baseline Difficulty", ""])
    for r in corr:
        report.append(f"- {r['direction']} fold{r['fold']}: rho(C0G,DeltaG) {r['spearman_C0_G_vs_delta_G']:+.4f}, floor dependent `{r['baseline_floor_dependent']}`.")
    report.extend(["", "## Gate", ""])
    report.append(f"- Median positive in 4/4 folds: `{med_positive}`.")
    report.append(f"- Positive category ratio >=60% in 4/4 folds: `{pos_ratio}`.")
    report.append(f"- Equal-category mean >= +2pp in {eq_count}/4 folds.")
    report.append(f"- Baseline-floor dependent in both directions: `{d1_floor and d2_floor}`.")
    report.append("Stop: XLIV-B was statistics-only; no training or inference was run.")
    report.append(f"Runtime: {time.time() - start:.1f}s.")
    (OUT / "XLIV_B_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {
        "phase": "XLIV-B",
        "outcome": outcome,
        "median_positive_all_folds": med_positive,
        "positive_ratio_gate": pos_ratio,
        "equal_category_mean_ge_2pp_folds": eq_count,
        "chewinggum_not_required": chew["CHEWINGGUM_NOT_REQUIRED_FOR_SIGNAL"],
        "baseline_floor_dependent_both_directions": d1_floor and d2_floor,
        "runtime_sec": time.time() - start,
    })
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLIV-B Category Generalization Robustness Audit\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- Fold gain distribution: {json.dumps(dist, default=json_default)}.\n")
        fp.write(f"- Equal-category mean: {json.dumps(eq, default=json_default)}.\n")
        fp.write(f"- Chewinggum sensitivity: {json.dumps(chew, default=json_default)}.\n")
        fp.write(f"- Concentration: {json.dumps(conc, default=json_default)}.\n")
        fp.write(f"- Baseline difficulty correlation: {json.dumps(corr, default=json_default)}.\n")
    print(json.dumps(json.loads((OUT / "OUTCOME.json").read_text(encoding="utf-8")), indent=2), flush=True)


if __name__ == "__main__":
    main()
