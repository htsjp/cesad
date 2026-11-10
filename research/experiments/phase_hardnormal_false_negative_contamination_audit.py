import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC
from experiments.phase_patch_occupancy_supervision_audit import current_binary_label, load_context, true_patch_occupancy
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import balanced_loss
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np


OUT = Path("results/phase_hardnormal_false_negative_contamination_audit/formal")
EPS = 1e-12


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


def transform_score(pos, patch, q_pc1, align_model, m_c, alpha, v_hn, device):
    z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
    z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
    with torch.no_grad():
        z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device))
        m = torch.tensor(m_c, dtype=torch.float32, device=device)
        v = torch.tensor(v_hn, dtype=torch.float32, device=device)
        score = contrast_logits(z, m, alpha, F.normalize(v, dim=-1)).detach().cpu().numpy().astype(np.float32)
    return z.detach().cpu().numpy().astype(np.float32), score


def percentile_stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "P50": float("nan"), "P90": float("nan"), "P95": float("nan")}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "P50": float(np.percentile(x, 50)),
        "P90": float(np.percentile(x, 90)),
        "P95": float(np.percentile(x, 95)),
    }


def build_source_patch_table(dataset, indices, patch, q_pc1, align_model, m_c, alpha, v_hn, device, smoke):
    rows = []
    feats = []
    max_images = 80 if smoke else len(indices)
    for pos, idx in enumerate(indices[:max_images]):
        item = dataset[int(idx)]
        cat = str(item["cls_name"])
        if int(item["anomaly"]) == 1:
            q, _method = true_patch_occupancy(item)
        else:
            q = np.zeros((SIDE * SIDE,), dtype=np.float32)
        y = (q > 0.5).astype(np.float32)
        y_current = current_binary_label(item).astype(np.float32)
        z, score = transform_score(pos, patch, q_pc1, align_model, m_c, alpha, v_hn, device)
        if not np.array_equal(y, y_current):
            # The audit definition is authoritative; keep a mismatch counter in rows.
            mismatch = int(np.sum(y != y_current))
        else:
            mismatch = 0
        feats.append(z)
        for p in range(SIDE * SIDE):
            rows.append({
                "image_pos": int(pos),
                "index": int(idx),
                "category": cat,
                "patch": int(p),
                "q": float(q[p]),
                "y_bin": int(y[p]),
                "score": float(score[p]),
                "binary_mismatch_with_current_patch_labels": mismatch,
                "feature_row": len(feats) - 1,
            })
    return rows, feats


def contamination_tables(rows):
    q = np.asarray([r["q"] for r in rows], dtype=np.float32)
    y = np.asarray([r["y_bin"] for r in rows], dtype=np.int64)
    score = np.asarray([r["score"] for r in rows], dtype=np.float32)
    cat = np.asarray([r["category"] for r in rows], dtype=object)
    neg = y == 0
    pure = neg & (q == 0)
    mixed_fn = neg & (q > 0)
    pos = y == 1
    k = max(1, int(np.ceil(float(np.sum(neg)) * TOP_NEG_FRAC)))
    neg_ids = np.flatnonzero(neg)
    hard_ids = neg_ids[np.argpartition(score[neg_ids], -k)[-k:]]
    hard = np.zeros_like(neg, dtype=bool)
    hard[hard_ids] = True
    h_mixed = hard & mixed_fn
    pool_cont = float(np.sum(mixed_fn) / max(1, np.sum(neg)))
    hard_cont = float(np.sum(h_mixed) / max(1, np.sum(hard)))
    enrich = float(hard_cont / max(EPS, pool_cont))
    pool_rows = [{
        "scope": "__all__",
        "negative_pool_count": int(np.sum(neg)),
        "PureNormal_count": int(np.sum(pure)),
        "MixedFalseNegative_count": int(np.sum(mixed_fn)),
        "contamination_pool": pool_cont,
        "mixed_q_mean": float(q[mixed_fn].mean()) if np.any(mixed_fn) else 0.0,
        "mixed_q_median": float(np.percentile(q[mixed_fn], 50)) if np.any(mixed_fn) else 0.0,
        "mixed_q_P95": float(np.percentile(q[mixed_fn], 95)) if np.any(mixed_fn) else 0.0,
    }]
    hard_rows = [{
        "scope": "__all__",
        "hard_count": int(np.sum(hard)),
        "hard_pure_count": int(np.sum(hard & pure)),
        "hard_mixed_false_negative_count": int(np.sum(h_mixed)),
        "contamination_hard": hard_cont,
        "contamination_pool": pool_cont,
        "enrichment_ratio": enrich,
        "top_fraction": TOP_NEG_FRAC,
    }]
    score_rows = []
    groups = {
        "PureNormal_q_eq_0": pure,
        "MixedFalseNegative_0_lt_q_le_0p5": mixed_fn,
        "BinaryPositive_q_gt_0p5": pos,
    }
    for name, mask in groups.items():
        stats = percentile_stats(score[mask])
        score_rows.append({"group": name, **stats})
    pure_median = np.percentile(score[pure], 50) if np.any(pure) else float("nan")
    pure_p90 = np.percentile(score[pure], 90) if np.any(pure) else float("nan")
    score_rows.append({
        "group": "MixedFalseNegative_probability_above_pure_thresholds",
        "P_S_mixed_gt_median_S_pure": float(np.mean(score[mixed_fn] > pure_median)) if np.any(mixed_fn) else float("nan"),
        "P_S_mixed_gt_P90_S_pure": float(np.mean(score[mixed_fn] > pure_p90)) if np.any(mixed_fn) else float("nan"),
    })
    cat_rows = []
    for c in sorted(set(cat.tolist())):
        cm = cat == c
        cneg = cm & neg
        cpure = cm & pure
        cmix = cm & mixed_fn
        chard = cm & hard
        chmix = cm & hard & mixed_fn
        cp = float(np.sum(cmix) / max(1, np.sum(cneg)))
        ch = float(np.sum(chmix) / max(1, np.sum(chard)))
        cat_rows.append({
            "category": c,
            "negative_pool_count": int(np.sum(cneg)),
            "PureNormal_count": int(np.sum(cpure)),
            "MixedFalseNegative_count": int(np.sum(cmix)),
            "hard_count": int(np.sum(chard)),
            "hard_mixed_false_negative_count": int(np.sum(chmix)),
            "contamination_pool": cp,
            "contamination_hard": ch,
            "enrichment_ratio": float(ch / max(EPS, cp)),
        })
    masks = {"pure": pure, "mixed": mixed_fn, "positive": pos, "hard": hard, "hard_pure": hard & pure, "hard_mixed": h_mixed}
    arrays = {"q": q, "y": y, "score": score, "cat": cat}
    return pool_rows, hard_rows, score_rows, cat_rows, masks, arrays


def grad_vector(z_np, y_np, m_c, alpha, v_hn, device, loss_type):
    if z_np.shape[0] == 0:
        return np.zeros_like(v_hn, dtype=np.float64), float("nan")
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    v = torch.nn.Parameter(torch.tensor(v_hn, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    logits = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
    if loss_type == "hard_normal":
        loss = F.softplus(logits).mean()
    elif loss_type == "occupancy":
        loss, _bce, _dice = balanced_loss(logits, y)
    else:
        raise ValueError(loss_type)
    loss.backward()
    g = v.grad.detach().cpu().numpy().astype(np.float64)
    return g, float(loss.detach().cpu())


def gradient_conflict(rows, feats, masks, arrays, m_c, alpha, v_hn, device):
    feature_rows = np.asarray([r["feature_row"] for r in rows], dtype=np.int64)
    patch_ids = np.asarray([r["patch"] for r in rows], dtype=np.int64)
    feat_stack = np.stack(feats, axis=0)
    def gather(mask, cap=50000):
        ids = np.flatnonzero(mask)
        if ids.size > cap:
            rng = np.random.default_rng(0)
            ids = rng.choice(ids, size=cap, replace=False)
        return feat_stack[feature_rows[ids], patch_ids[ids]], arrays["q"][ids]
    z_pure, q_pure = gather(masks["hard_pure"])
    z_mixed, q_mixed = gather(masks["hard_mixed"])
    g_pure, loss_pure = grad_vector(z_pure, np.zeros_like(q_pure), m_c, alpha, v_hn, device, "hard_normal")
    g_mixed, loss_mixed = grad_vector(z_mixed, np.zeros_like(q_mixed), m_c, alpha, v_hn, device, "hard_normal")
    g_occ, loss_occ = grad_vector(z_mixed, q_mixed, m_c, alpha, v_hn, device, "occupancy")
    def cosine(a, b):
        return float(a @ b / max(EPS, np.linalg.norm(a) * np.linalg.norm(b)))
    rows_out = [{
        "scope": "__all__",
        "hard_pure_count_used": int(z_pure.shape[0]),
        "hard_mixed_count_used": int(z_mixed.shape[0]),
        "cos_g_pure_g_mixed": cosine(g_pure, g_mixed),
        "cos_g_mixed_g_occ": cosine(g_mixed, g_occ),
        "norm_g_pure": float(np.linalg.norm(g_pure)),
        "norm_g_mixed": float(np.linalg.norm(g_mixed)),
        "norm_g_occ": float(np.linalg.norm(g_occ)),
        "loss_hard_pure": loss_pure,
        "loss_hard_mixed": loss_mixed,
        "loss_occ_mixed": loss_occ,
        "direct_conflict_cos_mixed_occ_lt_0": bool(cosine(g_mixed, g_occ) < 0),
    }]
    return rows_out


def decide(hard_rows):
    row = hard_rows[0]
    hc = float(row["contamination_hard"])
    enrich = float(row["enrichment_ratio"])
    if hc >= 0.10 and enrich >= 1.5:
        outcome = "BINARY_HARDNORMAL_FALSE_NEGATIVE_CONTAMINATION_CONFIRMED"
    elif hc >= 0.10:
        outcome = "MIXED_PATCH_CONTAMINATION_PRESENT_NOT_HARD_SPECIFIC"
    elif hc < 0.05:
        outcome = "HARDNORMAL_CONTAMINATION_NEGLIGIBLE"
    else:
        outcome = "HARDNORMAL_CONTAMINATION_INCONCLUSIVE"
    return outcome


def write_report(pool_rows, hard_rows, score_rows, grad_rows, cat_rows, outcome, runtime):
    p = pool_rows[0]
    h = hard_rows[0]
    g = grad_rows[0]
    score = {r["group"]: r for r in score_rows}
    lines = [
        "# HardNormal False-Negative Contamination Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN audit on Full VisA source with the current formal Symmetric HardNormal scorer. Binary target is defined as y=1(q>0.5), where q is true patch occupancy.",
        "",
        "## Negative Pool",
        "",
        f"- N_bin patches: {p['negative_pool_count']}.",
        f"- PureNormal: {p['PureNormal_count']}.",
        f"- MixedFalseNegative: {p['MixedFalseNegative_count']}.",
        f"- contamination_pool: {float(p['contamination_pool']) * 100:.4f}%.",
        f"- mixed q mean/median/P95: {float(p['mixed_q_mean']):.6f} / {float(p['mixed_q_median']):.6f} / {float(p['mixed_q_P95']):.6f}.",
        "",
        "## Hard Selection",
        "",
        f"- H_bin top10% patches: {h['hard_count']}.",
        f"- hard mixed false negatives: {h['hard_mixed_false_negative_count']}.",
        f"- contamination_hard: {float(h['contamination_hard']) * 100:.4f}%.",
        f"- enrichment ratio: {float(h['enrichment_ratio']):.4f}.",
        "",
        "## Score Distribution",
        "",
        "| Group | Count | Mean | Std | P50 | P90 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("PureNormal_q_eq_0", "MixedFalseNegative_0_lt_q_le_0p5", "BinaryPositive_q_gt_0p5"):
        r = score[key]
        lines.append(f"| {key} | {r['count']} | {float(r['mean']):.6f} | {float(r['std']):.6f} | {float(r['P50']):.6f} | {float(r['P90']):.6f} | {float(r['P95']):.6f} |")
    prob = score["MixedFalseNegative_probability_above_pure_thresholds"]
    lines.extend([
        "",
        f"P(S_mixed > median(S_pure)): {float(prob['P_S_mixed_gt_median_S_pure']):.6f}.",
        f"P(S_mixed > P90(S_pure)): {float(prob['P_S_mixed_gt_P90_S_pure']):.6f}.",
        "",
        "## Gradient Conflict",
        "",
        f"- cos(g_pure,g_mixed): {float(g['cos_g_pure_g_mixed']):.6f}.",
        f"- cos(g_mixed,g_occ): {float(g['cos_g_mixed_g_occ']):.6f}.",
        f"- Direct conflict cos(g_mixed,g_occ)<0: {g['direct_conflict_cos_mixed_occ_lt_0']}.",
        "",
        "## Per-Category",
        "",
        f"Per-category contamination is saved in `CATEGORY_CONTAMINATION.csv`; {sum(float(r['contamination_hard']) >= 0.10 for r in cat_rows)}/{len(cat_rows)} categories have hard contamination >=10%.",
        "",
        "STOP: no model training or result modification was performed.",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FNContamination] device={device}", flush=True)
    q_pc1, align_model, m_c, alpha, v_hn = load_context(device)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    rows, feats = build_source_patch_table(dataset, indices, patch, q_pc1, align_model, m_c, alpha, v_hn, device, args.smoke)
    pool_rows, hard_rows, score_rows, cat_rows, masks, arrays = contamination_tables(rows)
    grad_rows = gradient_conflict(rows, feats, masks, arrays, m_c, alpha, v_hn, device)
    outcome = decide(hard_rows)
    runtime = time.time() - start
    write_csv(OUT / "POOL_CONTAMINATION.csv", pool_rows)
    write_csv(OUT / "HARD_SELECTION_CONTAMINATION.csv", hard_rows)
    write_csv(OUT / "SCORE_DISTRIBUTION.csv", score_rows)
    write_csv(OUT / "GRADIENT_CONFLICT.csv", grad_rows)
    write_csv(OUT / "CATEGORY_CONTAMINATION.csv", cat_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "runtime_sec": runtime,
        "device": device,
        "no_training": True,
        "optimizer_step": False,
        "source": "Full VisA",
        "smoke": bool(args.smoke),
        "top_negative_fraction": TOP_NEG_FRAC,
        "hard_contamination": hard_rows[0]["contamination_hard"],
        "pool_contamination": pool_rows[0]["contamination_pool"],
        "enrichment_ratio": hard_rows[0]["enrichment_ratio"],
        "cos_g_mixed_g_occ": grad_rows[0]["cos_g_mixed_g_occ"],
        "direct_gradient_conflict": grad_rows[0]["direct_conflict_cos_mixed_occ_lt_0"],
    })
    write_report(pool_rows, hard_rows, score_rows, grad_rows, cat_rows, outcome, runtime)
    print(f"[FNContamination] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
