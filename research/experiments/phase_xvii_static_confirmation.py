import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, to_device_cache, write_csv
from experiments.phase_xvii_prompt_memory_complementarity_audit import (
    ALPHAS,
    CALIBRATIONS,
    Calibrator,
    g_image,
    metric_pair,
    metric_rows_scores,
    read_csv,
    score_one,
)
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xvii_prompt_memory_complementarity/static_confirmation")
AUDIT = Path("results/phase_xvii_prompt_memory_complementarity/audit")


def d2_rows_from_audit():
    return [r for r in read_csv(AUDIT / "cached_component_scores.csv") if r["direction"] == "D2"]


def completed_keys(rows):
    return {(int(r["fold"]), r["split"], int(r["idx"])) for r in rows}


def cache_d2_fold_scores():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = d2_rows_from_audit()
    local_cache = OUT / "D2_COMPONENT_SCORES.csv"
    extra = read_csv(local_cache)
    rows.extend(extra)
    done = completed_keys(rows)
    device = "cuda:0"
    direction, dataset_name, data_path, checkpoint = "D2", *direction_cfg("D2")
    model = prompt_learner = image_proj = patch_proj = prompt_proj = None
    dataset = None
    for fold in [1, 2]:
        if dataset is None:
            model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
            dataset = load_dataset(dataset_name, data_path)
        rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
        train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
        val_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
        expected = {(fold, "train", idx) for idx in train_indices} | {(fold, "heldout", idx) for idx in val_indices}
        if expected.issubset(done):
            continue
        image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None)
        image_cache, patch_cache = to_device_cache(image_cache, device), to_device_cache(patch_cache, device)
        for split, indices in [("train", train_indices), ("heldout", val_indices)]:
            for idx in indices:
                key = (fold, split, idx)
                if key in done:
                    continue
                item = dataset[idx]
                scores = score_one(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, dataset_name, item, device)
                row = {
                    "direction": direction,
                    "dataset": dataset_name,
                    "fold": fold,
                    "split": split,
                    "idx": idx,
                    "category": item["cls_name"],
                    "label": int(item["anomaly"]),
                    **scores,
                }
                row["abs_disagreement"] = abs(row["score_memory_raw"] - row["score_prompt_raw"])
                row["signed_disagreement"] = row["score_memory_raw"] - row["score_prompt_raw"]
                rows.append(row)
                done.add(key)
            write_csv(local_cache, [r for r in rows if int(r["fold"]) in {1, 2}])
    return rows


def by(rows, fold, split):
    return [r for r in rows if r["direction"] == "D2" and int(r["fold"]) == fold and r["split"] == split]


def score_alpha(rows, alpha):
    return [alpha * float(r["memory_cal"]) + (1.0 - alpha) * float(r["prompt_cal"]) for r in rows]


def compute_static(rows):
    fold_metrics, curves, selections = [], [], []
    split_rows = []
    for fold in [0, 1, 2]:
        train_raw, held_raw = by(rows, fold, "train"), by(rows, fold, "heldout")
        if not train_raw or not held_raw:
            raise RuntimeError(f"missing D2 fold{fold} rows")
        official = metric_pair(held_raw, "score_official_final")
        best = None
        best_payload = None
        held_curve = []
        for cal in CALIBRATIONS:
            calibrator = Calibrator(train_raw, cal)
            train = calibrator.apply(train_raw)
            held = calibrator.apply(held_raw)
            for alpha in ALPHAS:
                train_scores = score_alpha(train, alpha)
                held_scores = score_alpha(held, alpha)
                train_metric = metric_rows_scores(train, train_scores)
                held_metric = metric_rows_scores(held, held_scores)
                train_selection_metric = 0.5 * float(train_metric["I-AUROC"]) + 0.5 * float(train_metric["I-AP"])
                row = {
                    "direction": "D2",
                    "fold": fold,
                    "calibration": cal,
                    "alpha": alpha,
                    "train_selection_metric": train_selection_metric,
                    "train_I-AUROC": train_metric["I-AUROC"],
                    "train_I-AP": train_metric["I-AP"],
                    "heldout_I-AUROC": held_metric["I-AUROC"],
                    "heldout_I-AP": held_metric["I-AP"],
                    "official_I-AUROC": official["I-AUROC"],
                    "official_I-AP": official["I-AP"],
                    "delta_I-AUROC": float(held_metric["I-AUROC"]) - float(official["I-AUROC"]),
                    "delta_I-AP": float(held_metric["I-AP"]) - float(official["I-AP"]),
                    "G_image_delta": g_image(held_metric, official),
                }
                curves.append(row)
                held_curve.append(row)
                if best is None or train_selection_metric > best:
                    best = train_selection_metric
                    best_payload = row
        selected = dict(best_payload)
        selections.append({
            "direction": "D2",
            "fold": fold,
            "selected_calibration": selected["calibration"],
            "selected_alpha": selected["alpha"],
            "source_train_selection_metric": selected["train_selection_metric"],
            "train_I-AUROC": selected["train_I-AUROC"],
            "train_I-AP": selected["train_I-AP"],
        })
        fold_metrics.append({
            "direction": "D2",
            "fold": fold,
            "selected_calibration": selected["calibration"],
            "selected_alpha": selected["alpha"],
            "source_train_selection_metric": selected["train_selection_metric"],
            "heldout_I-AUROC": selected["heldout_I-AUROC"],
            "heldout_I-AP": selected["heldout_I-AP"],
            "official_I-AUROC": selected["official_I-AUROC"],
            "official_I-AP": selected["official_I-AP"],
            "delta_I-AUROC": selected["delta_I-AUROC"],
            "delta_I-AP": selected["delta_I-AP"],
            "G_image_delta": selected["G_image_delta"],
        })
        rotation = deterministic_source_folds("visa")["rotations"][fold]
        split_rows.append({
            "direction": "D2",
            "fold": fold,
            "train_categories": ";".join(rotation["train_categories"]),
            "heldout_categories": ";".join(rotation["val_categories"]),
        })
    return fold_metrics, curves, selections, split_rows


def decide(fold_metrics):
    gains = np.asarray([float(r["G_image_delta"]) for r in fold_metrics], dtype=np.float64)
    alphas = np.asarray([float(r["selected_alpha"]) for r in fold_metrics], dtype=np.float64)
    alpha_range = float(alphas.max() - alphas.min())
    if alpha_range <= 0.20 + 1e-9:
        alpha_status = "ALPHA_STABLE"
    elif alpha_range <= 0.30 + 1e-9:
        alpha_status = "ALPHA_MODERATELY_STABLE"
    else:
        alpha_status = "ALPHA_UNSTABLE"
    stability = {
        "selected_alphas": [float(x) for x in alphas],
        "mean_alpha": float(alphas.mean()),
        "median_alpha": float(np.median(alphas)),
        "range": alpha_range,
        "std": float(alphas.std(ddof=1)) if alphas.size > 1 else 0.0,
        "alpha_status": alpha_status,
        "final_static_alpha": float(np.round(np.median(alphas), 1)),
    }
    positive = int((gains > 0).sum())
    mean_gain = float(gains.mean())
    no_bad = bool(gains.min() >= -0.20)
    gate = positive >= 2 and mean_gain >= 0.30 and no_bad and alpha_status in {"ALPHA_STABLE", "ALPHA_MODERATELY_STABLE"}
    strong = positive == 3 and mean_gain >= 0.50 and alpha_range <= 0.20 + 1e-9
    if gate:
        outcome = "PASS_TO_STATIC_TARGET_EVAL"
    elif positive <= 1:
        outcome = "D2_STATIC_FUSION_NOT_REPRODUCIBLE"
    elif alpha_status == "ALPHA_UNSTABLE":
        outcome = "D2_STATIC_ALPHA_UNSTABLE"
    else:
        outcome = "D2_STATIC_GAIN_FOLD_SPECIFIC"
    return outcome, stability, {
        "positive_folds": positive,
        "mean_G_image_delta": mean_gain,
        "min_G_image_delta": float(gains.min()),
        "max_G_image_delta": float(gains.max()),
        "no_fold_below_minus_0p20": no_bad,
        "strong_source_support": bool(strong),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = cache_d2_fold_scores()
    fold_metrics, curves, selections, split_rows = compute_static(rows)
    outcome, stability, summary = decide(fold_metrics)
    write_csv(OUT / "D2_STATIC_FOLD_METRICS.csv", fold_metrics)
    write_csv(OUT / "D2_STATIC_ALPHA_CURVES.csv", curves)
    write_csv(OUT / "D2_STATIC_ALPHA_SELECTION.csv", selections)
    write_csv(OUT / "D2_SOURCE_SPLITS.csv", split_rows)
    (OUT / "D2_STATIC_ALPHA_STABILITY.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    payload = {
        "phase": "XVII-S",
        "outcome": outcome,
        "summary": summary,
        "alpha_stability": stability,
        "target_accessed": False,
        "trained": False,
        "router_run": False,
        "next": "STOP before target evaluation.",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Phase XVII-S D2 Static Prompt-Memory Fusion Confirmation",
        "",
        f"Outcome: **{outcome}**",
        "",
        "No target access, no router, and no model training were performed.",
        "",
        "## Fold Metrics",
        "",
        "| Fold | Cal | Alpha | Official I-AUROC | Official I-AP | Static I-AUROC | Static I-AP | dI | dAP | G_image |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in fold_metrics:
        lines.append(
            f"| {r['fold']} | {r['selected_calibration']} | {float(r['selected_alpha']):.1f} | "
            f"{float(r['official_I-AUROC']):.4f} | {float(r['official_I-AP']):.4f} | "
            f"{float(r['heldout_I-AUROC']):.4f} | {float(r['heldout_I-AP']):.4f} | "
            f"{float(r['delta_I-AUROC']):.4f} | {float(r['delta_I-AP']):.4f} | {float(r['G_image_delta']):.4f} |"
        )
    lines.extend([
        "",
        "## Alpha Stability",
        "",
        f"- selected alphas: `{stability['selected_alphas']}`",
        f"- mean/median/range/std: `{stability['mean_alpha']:.4f}` / `{stability['median_alpha']:.4f}` / `{stability['range']:.4f}` / `{stability['std']:.4f}`",
        f"- status: `{stability['alpha_status']}`",
        f"- FINAL_STATIC_ALPHA: `{stability['final_static_alpha']}`" if outcome == "PASS_TO_STATIC_TARGET_EVAL" else "- FINAL_STATIC_ALPHA: `not frozen because continuation gate failed`",
        "",
        "Mandatory stop: target evaluation was not started.",
    ])
    (OUT / "D2_STATIC_SOURCE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write("\n\n## Phase XVII-S\n")
        f.write("- Command: python experiments/phase_xvii_static_confirmation.py\n")
        f.write("- Scope: D2 VisA fold1/fold2 static confirmation; fold0 reused; no target; no training; no router.\n")
        f.write(f"- Selected alphas: {stability['selected_alphas']}; alpha_status: {stability['alpha_status']}.\n")
        f.write(f"- Mean G_image delta: {summary['mean_G_image_delta']}; positive folds: {summary['positive_folds']}/3.\n")
        f.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "D2_STATIC_SOURCE_SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()
