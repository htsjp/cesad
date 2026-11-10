import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_clip_radio_additive_complementarity_audit import build_full_source_scores
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE


OUT = Path("results/phase_source_trained_backbone_reliability_selector_audit/formal")
SEED = 0


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


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def gp(row):
    return 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"]))


def entropy_score(scores):
    x = np.asarray(scores, dtype=np.float64)
    x = x - np.max(x)
    p = np.exp(x)
    p = p / np.maximum(p.sum(), 1e-12)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(len(p)))


def image_features(sc, sr):
    diff = np.abs(sc - sr)
    corr = float(pearsonr(sc, sr).statistic) if len(sc) > 2 and np.std(sc) > 1e-8 and np.std(sr) > 1e-8 else 0.0
    vals = []
    for s in (sc, sr):
        vals.extend([
            float(np.mean(s)),
            float(np.std(s)),
            float(np.percentile(s, 90)),
            float(np.percentile(s, 95)),
            float(np.max(s)),
            entropy_score(s),
        ])
    vals.extend([float(np.mean(diff)), float(np.percentile(diff, 95)), corr])
    return np.asarray(vals, dtype=np.float32)


def per_image_table(pool):
    rows = []
    n = len(pool["S_C"]) // (SIDE * SIDE)
    sc_all = pool["S_C"].reshape(n, SIDE * SIDE)
    sr_all = pool["S_R"].reshape(n, SIDE * SIDE)
    y_all = pool["y"].reshape(n, SIDE * SIDE)
    cat_all = pool["category"].reshape(n, SIDE * SIDE)
    pos_all = pool["position"].reshape(n, SIDE * SIDE)
    for i in range(n):
        pos = int(pos_all[i, 0])
        sc = sc_all[i].astype(np.float32)
        sr = sr_all[i].astype(np.float32)
        y = y_all[i].astype(np.int64)
        cat = str(cat_all[i, 0])
        fc = image_features(sc, sr)
        c_row = {"P-AUROC": safe_auc(y, sc), "P-AP": safe_ap(y, sc)}
        r_row = {"P-AUROC": safe_auc(y, sr), "P-AP": safe_ap(y, sr)}
        c_g = gp(c_row) if np.isfinite(c_row["P-AUROC"]) and np.isfinite(c_row["P-AP"]) else float("nan")
        r_g = gp(r_row) if np.isfinite(r_row["P-AUROC"]) and np.isfinite(r_row["P-AP"]) else float("nan")
        if np.isfinite(c_g) and np.isfinite(r_g):
            label = int(r_g > c_g)
        else:
            label = -1
        rows.append({
            "position": int(pos),
            "category": cat,
            "has_pixel_label": bool(np.unique(y).size == 2),
            "clip_G": c_g,
            "radio_G": r_g,
            "selector_label": label,
            "features": fc,
        })
    return rows


def metric_for_positions(maps, pos_to_score):
    ys, ss = [], []
    for pos, score in pos_to_score.items():
        ys.append(maps[pos]["y"])
        ss.append(np.asarray(score, dtype=np.float32))
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    row = {"P-AUROC": safe_auc(y, s), "P-AP": safe_ap(y, s)}
    row["G_pixel"] = gp(row)
    return row


def score_maps_by_position(pool):
    out = {}
    n = len(pool["S_C"]) // (SIDE * SIDE)
    sc_all = pool["S_C"].reshape(n, SIDE * SIDE)
    sr_all = pool["S_R"].reshape(n, SIDE * SIDE)
    y_all = pool["y"].reshape(n, SIDE * SIDE)
    cat_all = pool["category"].reshape(n, SIDE * SIDE)
    pos_all = pool["position"].reshape(n, SIDE * SIDE)
    for i in range(n):
        pos = int(pos_all[i, 0])
        out[int(pos)] = {
            "category": str(cat_all[i, 0]),
            "y": y_all[i].astype(np.int64),
            "clip": sc_all[i].astype(np.float32),
            "radio": sr_all[i].astype(np.float32),
        }
    return out


def train_selector(image_rows, train_cats):
    train = [r for r in image_rows if r["category"] in train_cats and int(r["selector_label"]) >= 0]
    if not train:
        return None, 0, 0
    x = np.stack([r["features"] for r in train]).astype(np.float32)
    y = np.asarray([r["selector_label"] for r in train], dtype=np.int64)
    if np.unique(y).size < 2:
        return None, int(y[0]) if y.size else 0, len(train)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=SEED, class_weight="balanced", max_iter=1000),
    )
    clf.fit(x, y)
    return clf, None, len(train)


def evaluate_loo(pool, image_rows):
    maps = score_maps_by_position(pool)
    cats = sorted(set(r["category"] for r in image_rows))
    metric_rows, decision_rows, selector_rows = [], [], []
    for held_cat in cats:
        train_cats = [c for c in cats if c != held_cat]
        clf, constant, train_n = train_selector(image_rows, train_cats)
        train_positions = [r["position"] for r in image_rows if r["category"] in train_cats]
        held_positions = [r["position"] for r in image_rows if r["category"] == held_cat]
        b0_train = metric_for_positions(maps, {p: maps[p]["clip"] for p in train_positions})
        b1_train = metric_for_positions(maps, {p: maps[p]["radio"] for p in train_positions})
        global_choice = "clip" if b0_train["G_pixel"] >= b1_train["G_pixel"] else "radio"

        variant_scores = {
            "B0_CLIP_HardNormal": {},
            "B1_RADIO_StateAligned": {},
            "B2_SourceGlobalFixedChoice": {},
            "E1_LearnedSelector": {},
            "Oracle_PerImageGT": {},
        }
        y_true_sel, y_pred_sel = [], []
        for row in [r for r in image_rows if r["category"] == held_cat]:
            p = row["position"]
            clip = maps[p]["clip"]
            radio = maps[p]["radio"]
            variant_scores["B0_CLIP_HardNormal"][p] = clip
            variant_scores["B1_RADIO_StateAligned"][p] = radio
            variant_scores["B2_SourceGlobalFixedChoice"][p] = maps[p][global_choice]
            if clf is None:
                pred = int(constant)
            else:
                pred = int(clf.predict(row["features"].reshape(1, -1))[0])
            choice = "radio" if pred == 1 else "clip"
            variant_scores["E1_LearnedSelector"][p] = maps[p][choice]
            if int(row["selector_label"]) >= 0:
                y_true_sel.append(int(row["selector_label"]))
                y_pred_sel.append(pred)
            if int(row["selector_label"]) >= 0:
                oracle_choice = "radio" if int(row["selector_label"]) == 1 else "clip"
            else:
                oracle_choice = "clip" if float(np.mean(clip)) <= float(np.mean(radio)) else "radio"
            variant_scores["Oracle_PerImageGT"][p] = maps[p][oracle_choice]
            selector_rows.append({
                "heldout_category": held_cat,
                "position": p,
                "true_best": row["selector_label"],
                "pred_best": pred,
                "chosen": choice,
                "clip_G": row["clip_G"],
                "radio_G": row["radio_G"],
            })

        local_metrics = {}
        for variant, pos_scores in variant_scores.items():
            m = metric_for_positions(maps, pos_scores)
            rec = {
                "heldout_category": held_cat,
                "variant": variant,
                **m,
                "train_selector_images": train_n,
                "global_choice": global_choice,
                "selector_accuracy": float(accuracy_score(y_true_sel, y_pred_sel) * 100.0) if y_true_sel else float("nan"),
                "selector_eval_images": int(len(y_true_sel)),
            }
            metric_rows.append(rec)
            local_metrics[variant] = rec
        e1 = local_metrics["E1_LearnedSelector"]["G_pixel"]
        b0 = local_metrics["B0_CLIP_HardNormal"]["G_pixel"]
        b1 = local_metrics["B1_RADIO_StateAligned"]["G_pixel"]
        b2 = local_metrics["B2_SourceGlobalFixedChoice"]["G_pixel"]
        oracle = local_metrics["Oracle_PerImageGT"]["G_pixel"]
        decision_rows.append({
            "heldout_category": held_cat,
            "E1_minus_B0_G": e1 - b0,
            "E1_minus_max_B0_B1_G": e1 - max(b0, b1),
            "E1_minus_B2_G": e1 - b2,
            "Oracle_minus_E1_G": oracle - e1,
            "selector_accuracy": local_metrics["E1_LearnedSelector"]["selector_accuracy"],
            "global_choice": global_choice,
        })
        print(f"[ReliabilitySelector] LOO {held_cat}: E1-B0={e1-b0:+.4f} E1-max={e1-max(b0,b1):+.4f} acc={local_metrics['E1_LearnedSelector']['selector_accuracy']:.2f}", flush=True)
    return metric_rows, decision_rows, selector_rows


def summarize(decision_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in decision_rows], dtype=np.float32)
    e1_max = np.asarray([r["E1_minus_max_B0_B1_G"] for r in decision_rows], dtype=np.float32)
    e1_b2 = np.asarray([r["E1_minus_B2_G"] for r in decision_rows], dtype=np.float32)
    acc = np.asarray([r["selector_accuracy"] for r in decision_rows if np.isfinite(r["selector_accuracy"])], dtype=np.float32)
    stats = {
        "E1_improved_over_CLIP_categories": int(np.sum(e1_b0 > 0)),
        "mean_E1_minus_CLIP_G": float(np.mean(e1_b0)),
        "median_E1_minus_CLIP_G": float(np.median(e1_b0)),
        "mean_E1_minus_max_CLIP_RADIO_G": float(np.mean(e1_max)),
        "mean_E1_minus_global_fixed_G": float(np.mean(e1_b2)),
        "mean_selector_accuracy": float(np.mean(acc)) if acc.size else float("nan"),
        "median_selector_accuracy": float(np.median(acc)) if acc.size else float("nan"),
    }
    if (
        stats["E1_improved_over_CLIP_categories"] >= 8
        and stats["mean_E1_minus_CLIP_G"] >= 1.0
        and stats["mean_E1_minus_max_CLIP_RADIO_G"] >= 0.5
        and stats["mean_E1_minus_global_fixed_G"] > 0
        and stats["mean_selector_accuracy"] >= 65.0
    ):
        outcome = "SOURCE_ONLY_BACKBONE_SELECTOR_HAS_SIGNAL"
    else:
        outcome = "SOURCE_ONLY_BACKBONE_SELECTOR_NO_SIGNAL"
    return outcome, stats


def write_report(outcome, stats, decision_rows, runtime):
    lines = [
        "# Source-Trained Backbone Reliability Selector Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TARGET / NO-NEW-TRAINING. Full VisA source only. CLIP Symmetric HardNormal and RADIO StateAligned are frozen. The selector uses only per-image score-map statistics: six CLIP stats, six RADIO stats, and three disagreement/correlation stats.",
        "",
        "## Gate Summary",
        f"- E1 improved over CLIP in {stats['E1_improved_over_CLIP_categories']}/12 categories.",
        f"- mean E1-CLIP G: {stats['mean_E1_minus_CLIP_G']:+.4f}",
        f"- median E1-CLIP G: {stats['median_E1_minus_CLIP_G']:+.4f}",
        f"- mean E1-max(CLIP,RADIO) G: {stats['mean_E1_minus_max_CLIP_RADIO_G']:+.4f}",
        f"- mean E1-source-global-fixed G: {stats['mean_E1_minus_global_fixed_G']:+.4f}",
        f"- mean selector accuracy: {stats['mean_selector_accuracy']:.4f}%",
        "",
        "| Held-out | E1-CLIP G | E1-max G | E1-B2 G | Oracle-E1 G | Selector Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in decision_rows:
        lines.append(
            f"| {r['heldout_category']} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_max_B0_B1_G']:+.4f} | {r['E1_minus_B2_G']:+.4f} | {r['Oracle_minus_E1_G']:+.4f} | {r['selector_accuracy']:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "- The selector must beat CLIP, the stronger single backbone per category, and a source-global fixed choice. Passing only the first condition is not enough.",
        "- Per-image oracle is reported only as an upper bound from source GT and is not a deployable rule.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    pool = build_full_source_scores(device, smoke=args.smoke)
    image_rows = per_image_table(pool)
    metric_rows, decision_rows, selector_rows = evaluate_loo(pool, image_rows)
    outcome, stats = summarize(decision_rows)
    runtime = time.time() - start
    image_out = [{k: v for k, v in r.items() if k != "features"} for r in image_rows]
    write_csv(OUT / "IMAGE_FEATURE_LABELS.csv", image_out)
    write_csv(OUT / "LOO_METRICS.csv", metric_rows)
    write_csv(OUT / "SELECTOR_DECISIONS.csv", selector_rows)
    write_csv(OUT / "SELECTOR_GATE.csv", decision_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "no_target": True,
        "no_new_training": True,
        "features": [
            "clip_mean", "clip_std", "clip_p90", "clip_p95", "clip_max", "clip_spatial_entropy",
            "radio_mean", "radio_std", "radio_p90", "radio_p95", "radio_max", "radio_spatial_entropy",
            "mean_abs_disagreement", "p95_abs_disagreement", "pearson_correlation",
        ],
    })
    write_report(outcome, stats, decision_rows, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
