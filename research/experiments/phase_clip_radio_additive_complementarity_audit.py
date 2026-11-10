import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cross_foundation_stable_hardness_audit import ensure_radio_source_scores_37, initialize_clip_context
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np


OUT = Path("results/phase_clip_radio_additive_complementarity_audit/formal")
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


def g_pixel(row):
    return 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"]))


def zscore_from_train(train_scores, scores):
    mu = float(np.mean(train_scores))
    sd = float(np.std(train_scores))
    sd = sd if sd > 1e-8 else 1.0
    return (scores - mu) / sd, mu, sd


def build_full_source_scores(device, smoke=False):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / "cache" / f"full_source_clip_radio_scores_{suffix}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}

    q_pc1, m_c, alpha, v_hard, align_model = initialize_clip_context(device)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    radio37 = ensure_radio_source_scores_37(device, smoke=False)
    use_positions = np.arange(len(indices), dtype=np.int64)
    if smoke:
        cats_seen = defaultdict(int)
        keep = []
        for pos, idx in enumerate(indices):
            cat = str(dataset.data_all[int(idx)]["cls_name"])
            if cats_seen[cat] < 8:
                keep.append(pos)
                cats_seen[cat] += 1
        use_positions = np.asarray(keep, dtype=np.int64)

    s_c_parts, s_r_parts, y_parts, cat_parts, pos_parts, pid_parts = [], [], [], [], [], []
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    m_t = torch.tensor(m_c, dtype=torch.float32, device=device)
    v_t = torch.tensor(v_hard, dtype=torch.float32, device=device)
    for n, pos in enumerate(use_positions):
        if n % 100 == 0:
            print(f"[AdditiveAudit] CLIP/RADIO source score {n}/{len(use_positions)}", flush=True)
        idx = int(indices[int(pos)])
        cat = str(dataset.data_all[idx]["cls_name"])
        y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
        z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
        z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
        with torch.no_grad():
            z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device))
            sc = contrast_logits(z, m_t, alpha, v_t).detach().cpu().numpy().astype(np.float32)
        s_c_parts.append(sc)
        s_r_parts.append(np.asarray(radio37[int(pos)], dtype=np.float32))
        y_parts.append(y.astype(np.int64))
        cat_parts.extend([cat] * (SIDE * SIDE))
        pos_parts.extend([int(pos)] * (SIDE * SIDE))
        pid_parts.extend(list(range(SIDE * SIDE)))

    payload = {
        "S_C": np.concatenate(s_c_parts).astype(np.float32),
        "S_R": np.concatenate(s_r_parts).astype(np.float32),
        "y": np.concatenate(y_parts).astype(np.int64),
        "category": np.asarray(cat_parts, dtype=object),
        "position": np.asarray(pos_parts, dtype=np.int32),
        "patch_id": np.asarray(pid_parts, dtype=np.int16),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    return payload


def score_alignment(pool):
    rows = []
    for name, mask in [
        ("all", np.ones_like(pool["y"], dtype=bool)),
        ("normal", pool["y"] == 0),
        ("anomaly", pool["y"] == 1),
    ]:
        sc = pool["S_C"][mask]
        sr = pool["S_R"][mask]
        rows.append({
            "scope": name,
            "count": int(sc.size),
            "pearson": float(pearsonr(sc, sr).statistic) if sc.size > 2 else float("nan"),
            "spearman": float(spearmanr(sc, sr).statistic) if sc.size > 2 else float("nan"),
        })
    return rows


def conditional_radio_auc(pool, bins=20):
    sc = pool["S_C"]
    sr = pool["S_R"]
    y = pool["y"]
    order = np.argsort(sc)
    chunks = np.array_split(order, bins)
    rows = []
    for i, ids in enumerate(chunks):
        rows.append({
            "bin": i,
            "count": int(ids.size),
            "S_C_min": float(np.min(sc[ids])),
            "S_C_max": float(np.max(sc[ids])),
            "S_R_conditional_P-AUROC": safe_auc(y[ids], sr[ids]),
            "S_R_conditional_P-AP": safe_ap(y[ids], sr[ids]),
        })
    vals = [r["S_R_conditional_P-AUROC"] for r in rows if np.isfinite(r["S_R_conditional_P-AUROC"])]
    rows.append({
        "bin": "mean",
        "count": int(sum(r["count"] for r in rows if isinstance(r["bin"], int))),
        "S_C_min": float("nan"),
        "S_C_max": float("nan"),
        "S_R_conditional_P-AUROC": float(np.mean(vals)) if vals else float("nan"),
        "S_R_conditional_P-AP": float("nan"),
    })
    return rows


def loo_metrics(pool):
    cats = sorted(set(pool["category"].tolist()))
    rng = np.random.default_rng(SEED)
    random_r = pool["S_R"].copy()
    rng.shuffle(random_r)
    rows = []
    rand_rows = []
    for cat in cats:
        train = pool["category"] != cat
        held = pool["category"] == cat
        zc, mu_c, sd_c = zscore_from_train(pool["S_C"][train], pool["S_C"])
        zr, mu_r, sd_r = zscore_from_train(pool["S_R"][train], pool["S_R"])
        zrr, mu_rr, sd_rr = zscore_from_train(random_r[train], random_r)
        s_add = 0.5 * zc + 0.5 * zr
        s_rand = 0.5 * zc + 0.5 * zrr
        variants = {
            "P0_CLIP_HardNormal": pool["S_C"],
            "P1_RADIO_StateAligned": pool["S_R"],
            "P2_Additive": s_add,
            "P3_RandomAdditive": s_rand,
        }
        local = {}
        for variant, score in variants.items():
            row = {
                "heldout_category": cat,
                "variant": variant,
                "P-AUROC": safe_auc(pool["y"][held], score[held]),
                "P-AP": safe_ap(pool["y"][held], score[held]),
                "mu_C_train": mu_c,
                "std_C_train": sd_c,
                "mu_R_train": mu_r,
                "std_R_train": sd_r,
                "mu_R_random_train": mu_rr,
                "std_R_random_train": sd_rr,
                "patches": int(np.sum(held)),
                "positive_patches": int(np.sum(pool["y"][held] == 1)),
            }
            row["G_pixel"] = g_pixel(row)
            rows.append(row)
            local[variant] = row
        p0 = local["P0_CLIP_HardNormal"]["G_pixel"]
        p1 = local["P1_RADIO_StateAligned"]["G_pixel"]
        p2 = local["P2_Additive"]["G_pixel"]
        pr = local["P3_RandomAdditive"]["G_pixel"]
        rand_rows.append({
            "heldout_category": cat,
            "P2_minus_P0_G": p2 - p0,
            "P2_minus_max_P0_P1_G": p2 - max(p0, p1),
            "P2_minus_Srand_G": p2 - pr,
            "P2_improves_P0": bool(p2 > p0),
            "P2_improves_max": bool(p2 > max(p0, p1)),
        })
        print(f"[AdditiveAudit] LOO {cat}: P2-P0={p2-p0:+.4f} P2-max={p2-max(p0,p1):+.4f}", flush=True)
    return rows, rand_rows


def error_complementarity(pool, loo_rows):
    rows = []
    by_cat = defaultdict(dict)
    for r in loo_rows:
        by_cat[r["heldout_category"]][r["variant"]] = r
    for cat in sorted(by_cat):
        held = pool["category"] == cat
        y = pool["y"][held]
        sc = pool["S_C"][held]
        sr = pool["S_R"][held]
        normal = y == 0
        anomaly = y == 1
        if not np.any(normal) or not np.any(anomaly):
            continue
        c_fp_thr = np.percentile(sc[normal], 95)
        c_fn_thr = np.percentile(sc[anomaly], 20)
        r_fp_thr = np.percentile(sr[normal], 95)
        r_fn_thr = np.percentile(sr[anomaly], 20)
        clip_fp = normal & (sc > c_fp_thr)
        clip_fn = anomaly & (sc < c_fn_thr)
        radio_fp = normal & (sr > r_fp_thr)
        radio_fn = anomaly & (sr < r_fn_thr)
        rows.append({
            "heldout_category": cat,
            "clip_fp_count": int(np.sum(clip_fp)),
            "clip_fn_count": int(np.sum(clip_fn)),
            "radio_fp_count": int(np.sum(radio_fp)),
            "radio_fn_count": int(np.sum(radio_fn)),
            "radio_repairs_clip_fp_ratio": float(np.mean(sr[clip_fp] <= r_fp_thr)) if np.any(clip_fp) else float("nan"),
            "radio_repairs_clip_fn_ratio": float(np.mean(sr[clip_fn] >= r_fn_thr)) if np.any(clip_fn) else float("nan"),
            "clip_repairs_radio_fp_ratio": float(np.mean(sc[radio_fp] <= c_fp_thr)) if np.any(radio_fp) else float("nan"),
            "clip_repairs_radio_fn_ratio": float(np.mean(sc[radio_fn] >= c_fn_thr)) if np.any(radio_fn) else float("nan"),
            "RADIO_score_on_CLIP_FP_mean": float(np.mean(sr[clip_fp])) if np.any(clip_fp) else float("nan"),
            "RADIO_score_on_CLIP_FN_mean": float(np.mean(sr[clip_fn])) if np.any(clip_fn) else float("nan"),
            "CLIP_score_on_RADIO_FP_mean": float(np.mean(sc[radio_fp])) if np.any(radio_fp) else float("nan"),
            "CLIP_score_on_RADIO_FN_mean": float(np.mean(sc[radio_fn])) if np.any(radio_fn) else float("nan"),
        })
    return rows


def decide(delta_rows):
    p2_p0 = np.asarray([r["P2_minus_P0_G"] for r in delta_rows], dtype=np.float32)
    p2_max = np.asarray([r["P2_minus_max_P0_P1_G"] for r in delta_rows], dtype=np.float32)
    p2_rand = np.asarray([r["P2_minus_Srand_G"] for r in delta_rows], dtype=np.float32)
    improved = int(np.sum(p2_p0 > 0))
    stats = {
        "improved_categories": improved,
        "mean_P2_minus_P0_G": float(np.mean(p2_p0)),
        "median_P2_minus_P0_G": float(np.median(p2_p0)),
        "mean_P2_minus_Srand_G": float(np.mean(p2_rand)),
        "mean_P2_minus_max_P0_P1_G": float(np.mean(p2_max)),
    }
    if improved >= 8 and stats["mean_P2_minus_P0_G"] >= 0.7 and stats["median_P2_minus_P0_G"] > 0 and stats["mean_P2_minus_Srand_G"] >= 0.5 and stats["mean_P2_minus_max_P0_P1_G"] > 0:
        return "FOUNDATION_SCORES_HAVE_ACTIONABLE_COMPLEMENTARITY", stats
    if stats["mean_P2_minus_P0_G"] > 0 and stats["mean_P2_minus_max_P0_P1_G"] <= 0:
        return "BACKBONES_COMPLEMENTARY_ONLY_AT_DATASET_LEVEL", stats
    if abs(stats["mean_P2_minus_P0_G"]) < 0.25:
        return "FOUNDATION_SCORE_COMPLEMENTARITY_NO_SIGNAL", stats
    return "FOUNDATION_SCORE_COMPLEMENTARITY_NO_SIGNAL", stats


def write_report(outcome, stats, align_rows, cond_rows, delta_rows, runtime):
    lines = [
        "# CLIP-RADIO Additive Complementarity Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TARGET / NO-NEW-TRAINING. Full VisA source only. CLIP Symmetric HardNormal and RADIO StateAligned checkpoints are frozen; RADIO scores are bilinearly resized from 32x32 to 37x37 without GT.",
        "",
        "## Score Alignment",
    ]
    for r in align_rows:
        lines.append(f"- {r['scope']}: Pearson {r['pearson']:.6f}, Spearman {r['spearman']:.6f}, count {r['count']}.")
    mean_cond = next((r for r in cond_rows if r["bin"] == "mean"), None)
    if mean_cond:
        lines.append(f"- Conditional S_R AUROC within S_C bins: {mean_cond['S_R_conditional_P-AUROC']:.4f}.")
    lines.extend([
        "",
        "## LOO Gate",
        f"- P2 improved over P0 in {stats['improved_categories']}/12 categories.",
        f"- mean G(P2-P0): {stats['mean_P2_minus_P0_G']:+.4f}",
        f"- median G(P2-P0): {stats['median_P2_minus_P0_G']:+.4f}",
        f"- mean G(P2-S_rand): {stats['mean_P2_minus_Srand_G']:+.4f}",
        f"- mean G(P2-max(P0,P1)): {stats['mean_P2_minus_max_P0_P1_G']:+.4f}",
        "",
        "## Per-Category Delta",
        "| Category | P2-P0 G | P2-max(P0,P1) G | P2-Srand G |",
        "|---|---:|---:|---:|",
    ])
    for r in delta_rows:
        lines.append(f"| {r['heldout_category']} | {r['P2_minus_P0_G']:+.4f} | {r['P2_minus_max_P0_P1_G']:+.4f} | {r['P2_minus_Srand_G']:+.4f} |")
    lines.extend([
        "",
        "## Interpretation",
        "- Fixed 1:1 additive fusion is useful only if it beats CLIP alone, beats a shuffled RADIO control, and still exceeds the stronger single foundation score in LOO.",
        "- If it improves over CLIP but not over max(P0,P1), the evidence is treated as dataset-level complementarity rather than actionable patch-level fusion value.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AdditiveAudit] device={device}", flush=True)
    pool = build_full_source_scores(device, smoke=args.smoke)
    align_rows = score_alignment(pool)
    cond_rows = conditional_radio_auc(pool)
    loo_rows, delta_rows = loo_metrics(pool)
    err_rows = error_complementarity(pool, loo_rows)
    outcome, stats = decide(delta_rows)
    runtime = time.time() - start
    write_csv(OUT / "SCORE_ALIGNMENT.csv", align_rows)
    write_csv(OUT / "CONDITIONAL_COMPLEMENTARITY.csv", cond_rows)
    write_csv(OUT / "LOO_METRICS.csv", loo_rows)
    write_csv(OUT / "RANDOM_CONTROL.csv", delta_rows)
    write_csv(OUT / "ERROR_COMPLEMENTARITY.csv", err_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "no_target": True,
        "no_new_training": True,
        "clip_model": "formal CLIP Symmetric HardNormal",
        "radio_model": "formal RADIO StateAligned source-side scores only",
        "fusion": "0.5 source-standardized CLIP + 0.5 source-standardized RADIO",
    })
    write_report(outcome, stats, align_rows, cond_rows, delta_rows, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
