import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xvii_prompt_memory_complementarity_audit import (
    Calibrator,
    metric_pair,
    metric_rows_scores,
    read_csv,
    safe_corr,
    safe_auc,
)
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xvii_prompt_memory_complementarity/calibration_reconciliation")
AUDIT = Path("results/phase_xvii_prompt_memory_complementarity/audit")
STATIC = Path("results/phase_xvii_prompt_memory_complementarity/static_confirmation")
ALPHA = 0.7
CAL_MAP = {"C0_RAW": "C0_raw", "C1_ZSCORE": "C1_zscore", "C2_RANK": "C2_rank"}


def load_d2_rows():
    rows = [r for r in read_csv(AUDIT / "cached_component_scores.csv") if r.get("direction") == "D2" and int(r.get("fold", -1)) == 0]
    rows.extend(read_csv(STATIC / "D2_COMPONENT_SCORES.csv"))
    if not rows:
        raise RuntimeError("missing D2 cached component scores")
    return rows


def by(rows, fold, split):
    return [r for r in rows if r["direction"] == "D2" and int(r["fold"]) == fold and r["split"] == split]


def fused_scores(rows, method):
    if method == "C0_RAW":
        return [ALPHA * float(r["score_memory_raw"]) + (1.0 - ALPHA) * float(r["score_prompt_raw"]) for r in rows]
    return [ALPHA * float(r["memory_cal"]) + (1.0 - ALPHA) * float(r["prompt_cal"]) for r in rows]


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize_values(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "iqr": percentile(arr, 75) - percentile(arr, 25),
        "p01": percentile(arr, 1),
        "p05": percentile(arr, 5),
        "p25": percentile(arr, 25),
        "p50": percentile(arr, 50),
        "p75": percentile(arr, 75),
        "p95": percentile(arr, 95),
        "p99": percentile(arr, 99),
    }


def effect_size(normal, anomaly):
    n = np.asarray(normal, dtype=np.float64)
    a = np.asarray(anomaly, dtype=np.float64)
    pooled = np.sqrt(0.5 * (n.var() + a.var()))
    return float((a.mean() - n.mean()) / (pooled + 1e-12))


def component_rows(train_raw, held_raw, method):
    if method == "C0_RAW":
        train = [dict(r, memory_cal=float(r["score_memory_raw"]), prompt_cal=float(r["score_prompt_raw"])) for r in train_raw]
        held = [dict(r, memory_cal=float(r["score_memory_raw"]), prompt_cal=float(r["score_prompt_raw"])) for r in held_raw]
    else:
        cal = Calibrator(train_raw, CAL_MAP[method])
        train = cal.apply(train_raw)
        held = cal.apply(held_raw)
    return train, held


def compute_tables(rows):
    metrics, deltas, dist, contrib = [], [], [], []
    parity = []
    by_method_fold = {}
    for fold in [0, 1, 2]:
        train_raw, held_raw = by(rows, fold, "train"), by(rows, fold, "heldout")
        if not train_raw or not held_raw:
            raise RuntimeError(f"missing fold {fold} train/heldout rows")
        official = metric_pair(held_raw, "score_official_final")
        c0_raw_scores = fused_scores(held_raw, "C0_RAW")
        official_scores = [float(r["score_official_final"]) for r in held_raw]
        max_abs = float(np.max(np.abs(np.asarray(c0_raw_scores) - np.asarray(official_scores))))
        c0_metric = metric_rows_scores(held_raw, c0_raw_scores)
        parity.append({
            "fold": fold,
            "max_abs_score_error": max_abs,
            "official_I-AUROC": official["I-AUROC"],
            "c0_I-AUROC": c0_metric["I-AUROC"],
            "official_I-AP": official["I-AP"],
            "c0_I-AP": c0_metric["I-AP"],
            "pass": max_abs <= 1e-6 and abs(float(official["I-AUROC"]) - float(c0_metric["I-AUROC"])) <= 1e-9 and abs(float(official["I-AP"]) - float(c0_metric["I-AP"])) <= 1e-9,
        })
        for method in ["C0_RAW", "C1_ZSCORE", "C2_RANK"]:
            train, held = component_rows(train_raw, held_raw, method)
            scores = fused_scores(held, method)
            m = metric_rows_scores(held, scores)
            d_i = float(m["I-AUROC"]) - float(c0_metric["I-AUROC"])
            d_ap = float(m["I-AP"]) - float(c0_metric["I-AP"])
            g = 0.5 * d_i + 0.5 * d_ap
            row = {
                "direction": "D2",
                "fold": fold,
                "calibration": method,
                "alpha": ALPHA,
                "I-AUROC": m["I-AUROC"],
                "I-AP": m["I-AP"],
                "C0_I-AUROC": c0_metric["I-AUROC"],
                "C0_I-AP": c0_metric["I-AP"],
                "dI-AUROC": d_i,
                "dI-AP": d_ap,
                "G_image": g,
            }
            metrics.append(row)
            deltas.append({k: row[k] for k in ["direction", "fold", "calibration", "dI-AUROC", "dI-AP", "G_image"]})
            by_method_fold[(method, fold)] = row
            for split_name, split_rows in [("source_train", train), ("heldout", held)]:
                mem = [float(r["memory_cal"]) for r in split_rows]
                prm = [float(r["prompt_cal"]) for r in split_rows]
                for branch, vals in [("memory", mem), ("prompt", prm)]:
                    d = summarize_values(vals)
                    d.update({"direction": "D2", "fold": fold, "calibration": method, "split": split_name, "branch": branch})
                    d["pearson_memory_prompt"] = safe_corr(mem, prm, "pearson")
                    d["spearman_memory_prompt"] = safe_corr(mem, prm, "spearman")
                    dist.append(d)
                labels = np.asarray([int(r["label"]) for r in split_rows], dtype=np.int64)
                for branch, vals in [("memory", mem), ("prompt", prm), ("fused", fused_scores(split_rows, method))]:
                    vals_arr = np.asarray(vals, dtype=np.float64)
                    normal = vals_arr[labels == 0]
                    anomaly = vals_arr[labels == 1]
                    if normal.size and anomaly.size:
                        dist.append({
                            "direction": "D2",
                            "fold": fold,
                            "calibration": method,
                            "split": split_name,
                            "branch": f"{branch}_normal_vs_anomaly",
                            "normal_mean": float(normal.mean()),
                            "normal_std": float(normal.std()),
                            "normal_median": float(np.median(normal)),
                            "anomaly_mean": float(anomaly.mean()),
                            "anomaly_std": float(anomaly.std()),
                            "anomaly_median": float(np.median(anomaly)),
                            "effect_size": effect_size(normal, anomaly),
                            "ranking_AUROC": safe_auc(labels, vals_arr),
                        })
            for split_name, split_rows in [("source_train", train), ("heldout", held)]:
                mem_comp = np.asarray([abs(ALPHA * float(r["memory_cal"])) for r in split_rows], dtype=np.float64)
                prm_comp = np.asarray([abs((1.0 - ALPHA) * float(r["prompt_cal"])) for r in split_rows], dtype=np.float64)
                raw_mem = np.asarray([float(r["score_memory_raw"]) for r in split_rows], dtype=np.float64)
                raw_prm = np.asarray([float(r["score_prompt_raw"]) for r in split_rows], dtype=np.float64)
                contrib.append({
                    "direction": "D2",
                    "fold": fold,
                    "calibration": method,
                    "split": split_name,
                    "E_abs_memory_contribution": float(mem_comp.mean()),
                    "E_abs_prompt_contribution": float(prm_comp.mean()),
                    "R_memory_over_prompt": float(mem_comp.mean() / (prm_comp.mean() + 1e-12)),
                    "raw_memory_std": float(raw_mem.std()),
                    "raw_prompt_std": float(raw_prm.std()),
                    "raw_variance_ratio_memory_over_prompt": float((raw_mem.var() + 1e-12) / (raw_prm.var() + 1e-12)),
                    "memory_rank_saturation_fraction": float(np.mean((np.asarray([float(r["memory_cal"]) for r in split_rows]) <= 0.01) | (np.asarray([float(r["memory_cal"]) for r in split_rows]) >= 0.99))) if method == "C2_RANK" else "",
                    "prompt_rank_saturation_fraction": float(np.mean((np.asarray([float(r["prompt_cal"]) for r in split_rows]) <= 0.01) | (np.asarray([float(r["prompt_cal"]) for r in split_rows]) >= 0.99))) if method == "C2_RANK" else "",
                })
    return metrics, deltas, dist, contrib, parity, by_method_fold


def summarize_and_decide(metrics):
    summaries = []
    for method in ["C0_RAW", "C1_ZSCORE", "C2_RANK"]:
        rows = [r for r in metrics if r["calibration"] == method]
        gains = np.asarray([float(r["G_image"]) for r in rows], dtype=np.float64)
        di = np.asarray([float(r["dI-AUROC"]) for r in rows], dtype=np.float64)
        dap = np.asarray([float(r["dI-AP"]) for r in rows], dtype=np.float64)
        positive = gains[gains > 0]
        concentration = float(positive.max() / positive.sum()) if positive.size else 1.0
        summaries.append({
            "calibration": method,
            "mean_G_image": float(gains.mean()),
            "median_G_image": float(np.median(gains)),
            "worst_G_image": float(gains.min()),
            "best_G_image": float(gains.max()),
            "positive_folds": int((gains > 0).sum()),
            "mean_dI-AUROC": float(di.mean()),
            "mean_dI-AP": float(dap.mean()),
            "fold_concentration": concentration,
        })
    complexity = {"C0_RAW": 0, "C1_ZSCORE": 1, "C2_RANK": 2}
    ordered = sorted(summaries, key=lambda r: (float(r["mean_G_image"]), float(r["median_G_image"]), float(r["worst_G_image"]), -complexity[r["calibration"]]), reverse=True)
    best = ordered[0]
    def passes(r):
        mean_gain = float(r["mean_G_image"])
        mean_di = float(r["mean_dI-AUROC"])
        mean_dap = float(r["mean_dI-AP"])
        metric_ok = (mean_di >= 0.50 and mean_dap > -0.10) or (mean_dap >= 0.50 and mean_di > -0.10)
        return (
            r["calibration"] != "C0_RAW"
            and mean_gain >= 0.50
            and int(r["positive_folds"]) >= 2
            and float(r["worst_G_image"]) >= -0.20
            and metric_ok
            and (float(r["fold_concentration"]) <= 0.80 or int(r["positive_folds"]) == 3)
        )
    passers = [r for r in summaries if passes(r)]
    if passers:
        selected = sorted(passers, key=lambda r: (float(r["mean_G_image"]), float(r["median_G_image"]), float(r["worst_G_image"]), -complexity[r["calibration"]]), reverse=True)[0]
        outcome = "PASS_TO_D2_TARGET_CALIBRATION_EVAL"
    else:
        selected = best
        if best["calibration"] != "C0_RAW" and float(best["fold_concentration"]) > 0.80 and int(best["positive_folds"]) >= 2:
            outcome = "XVII_S2_GAIN_FOLD_CONCENTRATED"
        elif best["calibration"] != "C0_RAW" and int(best["positive_folds"]) >= 2:
            outcome = "XVII_S2_CALIBRATION_NOT_REPRODUCIBLE"
        else:
            outcome = "XVII_S2_FOLD_SPECIFIC_CALIBRATION"
    strong = (
        outcome == "PASS_TO_D2_TARGET_CALIBRATION_EVAL"
        and float(selected["mean_G_image"]) >= 1.0
        and int(selected["positive_folds"]) == 3
        and float(selected["worst_G_image"]) >= -0.05
    )
    return summaries, selected, outcome, strong


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_d2_rows()
    metrics, deltas, dist, contrib, parity, _ = compute_tables(rows)
    if not all(r["pass"] for r in parity):
        outcome = "XVII_S2_IMPLEMENTATION_FAILURE"
        summaries, selected, strong = [], {}, False
    else:
        summaries, selected, outcome, strong = summarize_and_decide(metrics)
    write_csv(OUT / "FIXED_CALIBRATION_METRICS.csv", metrics)
    write_csv(OUT / "FIXED_CALIBRATION_DELTAS.csv", deltas)
    write_csv(OUT / "CROSS_FOLD_SUMMARY.csv", summaries)
    write_csv(OUT / "SCORE_DISTRIBUTIONS.csv", dist)
    write_csv(OUT / "EFFECTIVE_BRANCH_CONTRIBUTIONS.csv", contrib)
    protocol = {
        "phase": "XVII-S2",
        "target_accessed_before_protocol": False,
        "final_alpha": ALPHA if outcome == "PASS_TO_D2_TARGET_CALIBRATION_EVAL" else None,
        "final_calibration": selected.get("calibration") if outcome == "PASS_TO_D2_TARGET_CALIBRATION_EVAL" else None,
        "future_target_scoring_algorithm": [
            "Estimate calibration statistics using ALL VisA source data only.",
            "Compute official raw CLS-memory anomaly score for each unseen MVTec image.",
            "Compute official raw prompt top1% anomaly score for each unseen MVTec image.",
            "Apply the frozen VisA-source calibration independently to memory and prompt scores.",
            "Fuse with score = 0.7 * calibrated_memory + 0.3 * calibrated_prompt.",
            "Evaluate MVTec target once with no target-dependent calibration or normalization.",
        ],
    }
    (OUT / "FINAL_CALIBRATION_PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    lines = [
        "# Phase XVII-S2 Calibration Reconciliation",
        "",
        f"Outcome: **{outcome}**",
        "",
        "No target access, no router, no model training, and alpha fixed at 0.7.",
        "",
        "## C0 Parity",
        "",
        "| Fold | Max Abs Score Error | Pass |",
        "|---:|---:|---|",
    ]
    for r in parity:
        lines.append(f"| {r['fold']} | {float(r['max_abs_score_error']):.3e} | {r['pass']} |")
    lines.extend(["", "## Cross-Fold Summary", "", "| Calibration | Mean G | Median G | Worst G | Positive Folds | Mean dI | Mean dAP | Concentration |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in summaries:
        lines.append(f"| {r['calibration']} | {float(r['mean_G_image']):.4f} | {float(r['median_G_image']):.4f} | {float(r['worst_G_image']):.4f} | {r['positive_folds']} | {float(r['mean_dI-AUROC']):.4f} | {float(r['mean_dI-AP']):.4f} | {float(r['fold_concentration']):.4f} |")
    lines.extend(["", "## Selected", "", f"- selected calibration: `{selected.get('calibration', '')}`", f"- secondary tag: `{'SOURCE_CALIBRATION_STRONG' if strong else 'none'}`", "", "Mandatory stop: target evaluation was not started."])
    (OUT / "CALIBRATION_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "phase": "XVII-S2",
        "outcome": outcome,
        "selected_calibration": selected,
        "secondary_tag": "SOURCE_CALIBRATION_STRONG" if strong else None,
        "c0_parity": parity,
        "target_accessed": False,
        "trained": False,
        "router_run": False,
        "alpha": ALPHA,
        "next": "STOP before target calibration evaluation.",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write("\n\n## Phase XVII-S2\n")
        f.write("- Command: python experiments/phase_xvii_s2_calibration_reconciliation.py\n")
        f.write("- Scope: D2 source cached scores only; alpha fixed 0.7; no target; no training; no router.\n")
        f.write(f"- C0 parity max errors: {[r['max_abs_score_error'] for r in parity]}.\n")
        f.write(f"- Selected calibration: {selected.get('calibration', '')}; outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "CALIBRATION_DIAGNOSTICS.md")}, indent=2))


if __name__ == "__main__":
    main()
