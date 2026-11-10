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

from experiments.phase_cross_foundation_stable_hardness_audit import build_clip_radio_pool
from experiments.phase_hard_boundary_spherical_sharpness_audit import cos_np, gradient, logits_np, top_hard_indices
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xliii_adapter_head_decomposition import balanced_loss


OUT = Path("results/phase_hardnormal_meta_transfer_audit/formal")
SEED = 0
LR = 0.001


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


def unit_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def subset_ids(pool, category):
    return np.flatnonzero(pool["category"] == category)


def base_hard_ids(pool, subset):
    scores = logits_np(pool, pool["v_hard"], subset)
    neg = np.flatnonzero((pool["y"] < 0.5) & subset)
    return top_hard_indices(scores, neg)


def loss_torch(pool, v, ids_task=None, hard_ids=None, objective="full"):
    m = torch.tensor(np.asarray(pool["m_c"], dtype=np.float32), dtype=torch.float32)
    alpha = float(pool["alpha"])
    losses = []
    if objective in ("full", "task") and ids_task is not None and len(ids_task):
        z = torch.tensor(np.asarray(pool["x"][ids_task], dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(np.asarray(pool["y"][ids_task], dtype=np.float32), dtype=torch.float32)
        logits = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
        task, _bce, _dice = balanced_loss(logits, y)
        losses.append(task)
    if objective in ("full", "hard") and hard_ids is not None and len(hard_ids):
        z_h = torch.tensor(np.asarray(pool["x"][hard_ids], dtype=np.float32), dtype=torch.float32)
        logits_h = contrast_logits(z_h, m, alpha, F.normalize(v, dim=-1))
        losses.append(F.softplus(logits_h).mean())
    if not losses:
        return torch.zeros((), dtype=torch.float32)
    return sum(losses)


def grad_np_from_loss(loss, v):
    g = torch.autograd.grad(loss, v, create_graph=False, retain_graph=False, allow_unused=False)[0]
    return g.detach().cpu().numpy().astype(np.float32)


def category_gradients(pool):
    cats = sorted(set(pool["category"].tolist()))
    hard_grads = {}
    full_grads = {}
    rows = []
    for cat in cats:
        subset = pool["category"] == cat
        ids = np.flatnonzero(subset)
        hard = base_hard_ids(pool, subset)
        g_hard = gradient(pool, pool["v_hard"], ids[:0], hard)[0]
        g_full = gradient(pool, pool["v_hard"], ids, hard)[0]
        hard_grads[cat] = g_hard
        full_grads[cat] = g_full
        rows.append({
            "record": "category_gradient_norm",
            "category_i": cat,
            "category_j": cat,
            "hard_norm": float(np.linalg.norm(g_hard)),
            "full_norm": float(np.linalg.norm(g_full)),
        })
        print(f"[MetaTransfer] category gradients {cat}", flush=True)
    for kind, grads in [("hard", hard_grads), ("full", full_grads)]:
        vals = []
        for ci in cats:
            for cj in cats:
                c = cos_np(grads[ci], grads[cj])
                if ci < cj:
                    vals.append(c)
                rows.append({"record": f"{kind}_pairwise_cosine", "category_i": ci, "category_j": cj, "cosine": c})
        rows.append({
            "record": f"{kind}_pairwise_summary",
            "category_i": "__offdiag__",
            "category_j": "__offdiag__",
            "cosine_mean": float(np.mean(vals)),
            "cosine_std": float(np.std(vals)),
            "cosine_min": float(np.min(vals)),
            "cosine_max": float(np.max(vals)),
        })
    return rows, hard_grads, full_grads


def meta_gradient(pool, train_subset, val_subset, objective="hard"):
    v = torch.tensor(unit_np(pool["v_hard"]), dtype=torch.float32, requires_grad=True)
    train_ids = np.flatnonzero(train_subset)
    val_ids = np.flatnonzero(val_subset)
    train_hard = base_hard_ids(pool, train_subset)
    val_hard = base_hard_ids(pool, val_subset)
    l_tr = loss_torch(pool, v, train_ids, train_hard, objective=objective)
    g_tr = torch.autograd.grad(l_tr, v, create_graph=True, retain_graph=True)[0]
    v_prime = v - LR * g_tr
    l_val = loss_torch(pool, v_prime, val_ids, val_hard, objective=objective)
    l_meta = l_tr + l_val
    g_meta = torch.autograd.grad(l_meta, v, create_graph=False, retain_graph=False)[0]
    return g_meta.detach().cpu().numpy().astype(np.float32), float(l_tr.detach().cpu()), float(l_val.detach().cpu()), len(train_hard), len(val_hard)


def consensus_gradient(pool, train_subset, val_subset):
    train_hard = base_hard_ids(pool, train_subset)
    val_hard = base_hard_ids(pool, val_subset)
    g_tr = gradient(pool, pool["v_hard"], np.asarray([], dtype=np.int64), train_hard)[0]
    g_val = gradient(pool, pool["v_hard"], np.asarray([], dtype=np.int64), val_hard)[0]
    return g_tr + unit_np(g_val), g_tr, g_val


def random_meta_subset(pool, val_count, seed):
    rng = np.random.default_rng(seed)
    all_ids = np.arange(pool["y"].shape[0])
    val = rng.choice(all_ids, size=val_count, replace=False)
    val_mask = np.zeros(pool["y"].shape, dtype=bool)
    val_mask[val] = True
    return ~val_mask, val_mask


def loo_meta(pool):
    cats = sorted(set(pool["category"].tolist()))
    rows = []
    control = []
    meta_geoms = []
    previous_available = False
    for i, cat in enumerate(cats):
        train = pool["category"] != cat
        val = pool["category"] == cat
        train_ids = np.flatnonzero(train)
        hold_ids = np.flatnonzero(val)
        train_hard = base_hard_ids(pool, train)
        hold_hard = base_hard_ids(pool, val)
        g_b0_train = gradient(pool, pool["v_hard"], train_ids, train_hard)[0]
        g_holdout = gradient(pool, pool["v_hard"], hold_ids, hold_hard)[0]
        g_meta_hard, ltr_h, lval_h, ntr_h, nval_h = meta_gradient(pool, train, val, objective="hard")
        g_meta_full, ltr_f, lval_f, ntr_f, nval_f = meta_gradient(pool, train, val, objective="full")
        g_cons, _gtr, _gval = consensus_gradient(pool, train, val)
        rtrain, rval = random_meta_subset(pool, int(np.sum(val)), SEED + i)
        g_meta_random, _ltr_r, _lval_r, _ntr_r, _nval_r = meta_gradient(pool, rtrain, rval, objective="hard")
        a0 = cos_np(g_b0_train, g_holdout)
        a1 = cos_np(g_meta_hard, g_holdout)
        a2 = cos_np(g_meta_full, g_holdout)
        ar = cos_np(g_meta_random, g_holdout)
        rows.append({
            "heldout_category": cat,
            "A0_cos_B0_train_holdout": a0,
            "A1_cos_meta_hard_holdout": a1,
            "A2_cos_meta_full_holdout": a2,
            "Ar_cos_meta_random_holdout": ar,
            "A1_minus_A0": a1 - a0,
            "A1_minus_A2": a1 - a2,
            "A1_minus_Ar": a1 - ar,
            "train_hard_count": int(ntr_h),
            "val_hard_count": int(nval_h),
            "L_hard_train": ltr_h,
            "L_hard_val_virtual": lval_h,
            "L_full_train": ltr_f,
            "L_full_val_virtual": lval_f,
        })
        meta_geoms.append({
            "heldout_category": cat,
            "cos_meta_hard_B0_train": cos_np(g_meta_hard, g_b0_train),
            "cos_meta_hard_consensus": cos_np(g_meta_hard, g_cons),
            "cos_meta_hard_meta_full": cos_np(g_meta_hard, g_meta_full),
            "cos_meta_hard_meta_random": cos_np(g_meta_hard, g_meta_random),
            "norm_B0_train": float(np.linalg.norm(g_b0_train)),
            "norm_meta_hard": float(np.linalg.norm(g_meta_hard)),
            "norm_meta_full": float(np.linalg.norm(g_meta_full)),
            "norm_consensus": float(np.linalg.norm(g_cons)),
            "norm_meta_random": float(np.linalg.norm(g_meta_random)),
        })
        control.append({
            "heldout_category": cat,
            "previous_category_method_comparable": previous_available,
            "note": "No strict previous final-gradient artifact was available for direct comparison; not fabricated.",
        })
        print(f"[MetaTransfer] LOO {cat}: A1-A0={a1-a0:+.4f} A1-A2={a1-a2:+.4f} A1-Ar={a1-ar:+.4f}", flush=True)
    return rows, meta_geoms, control


def decide(cat_rows, meta_geoms, loo_rows):
    hard_summary = next(r for r in cat_rows if r["record"] == "hard_pairwise_summary")
    hard_mean = float(hard_summary["cosine_mean"])
    cos_meta_b0 = np.asarray([r["cos_meta_hard_B0_train"] for r in meta_geoms], dtype=np.float32)
    cos_meta_cons = np.asarray([r["cos_meta_hard_consensus"] for r in meta_geoms], dtype=np.float32)
    a10 = np.asarray([r["A1_minus_A0"] for r in loo_rows], dtype=np.float32)
    a12 = np.asarray([r["A1_minus_A2"] for r in loo_rows], dtype=np.float32)
    a1r = np.asarray([r["A1_minus_Ar"] for r in loo_rows], dtype=np.float32)
    stats = {
        "hard_pairwise_cosine_mean": hard_mean,
        "hard_pairwise_cosine_std": float(hard_summary["cosine_std"]),
        "mean_cos_meta_hard_B0_train": float(np.mean(cos_meta_b0)),
        "mean_cos_meta_hard_consensus": float(np.mean(cos_meta_cons)),
        "improved_categories_A1_gt_A0": int(np.sum(a10 > 0)),
        "mean_A1_minus_A0": float(np.mean(a10)),
        "median_A1_minus_A0": float(np.median(a10)),
        "mean_A1_minus_A2": float(np.mean(a12)),
        "mean_A1_minus_Ar": float(np.mean(a1r)),
    }
    if hard_mean > 0.95:
        collapse_risk = True
    else:
        collapse_risk = False
    if stats["mean_cos_meta_hard_consensus"] >= 0.98:
        return "META_TRANSFER_COLLAPSES_TO_GRADIENT_ALIGNMENT", stats
    if stats["mean_cos_meta_hard_B0_train"] >= 0.99:
        return "META_TRANSFER_REDUNDANT", stats
    if stats["mean_A1_minus_A2"] < 0 and (stats["improved_categories_A1_gt_A0"] >= 8 or stats["mean_A1_minus_A0"] >= 0.02):
        return "GENERIC_META_DG_SIGNAL_NOT_HARD_SPECIFIC", stats
    if (
        hard_mean <= 0.95
        and stats["mean_cos_meta_hard_B0_train"] <= 0.99
        and stats["mean_cos_meta_hard_consensus"] <= 0.98
        and stats["improved_categories_A1_gt_A0"] >= 8
        and stats["mean_A1_minus_A0"] >= 0.02
        and stats["mean_A1_minus_Ar"] >= 0.015
        and stats["mean_A1_minus_A2"] >= 0
    ):
        return "HARDNORMAL_META_TRANSFER_HAS_SIGNAL", stats
    if collapse_risk:
        stats["collapse_risk"] = True
    return "HARDNORMAL_META_TRANSFER_NO_SIGNAL", stats


def write_report(outcome, stats, loo_rows, runtime):
    lines = [
        "# HardNormal Meta-Transfer Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN / NO-TARGET. The formal Symmetric HardNormal checkpoint is frozen. All meta steps are differentiable virtual updates with alpha=0.001; no optimizer.step is called.",
        "",
        "## Gate Summary",
        f"- hard pairwise gradient cosine mean/std: {stats['hard_pairwise_cosine_mean']:.6f} / {stats['hard_pairwise_cosine_std']:.6f}",
        f"- mean cos(g_meta_hard, g_B0_train): {stats['mean_cos_meta_hard_B0_train']:.6f}",
        f"- mean cos(g_meta_hard, g_consensus): {stats['mean_cos_meta_hard_consensus']:.6f}",
        f"- A1>A0 categories: {stats['improved_categories_A1_gt_A0']}/12",
        f"- mean(A1-A0): {stats['mean_A1_minus_A0']:+.6f}",
        f"- median(A1-A0): {stats['median_A1_minus_A0']:+.6f}",
        f"- mean(A1-Ar): {stats['mean_A1_minus_Ar']:+.6f}",
        f"- mean(A1-A2): {stats['mean_A1_minus_A2']:+.6f}",
        "",
        "| Held-out | A1-A0 | A1-A2 | A1-Ar |",
        "|---|---:|---:|---:|",
    ]
    for r in loo_rows:
        lines.append(f"| {r['heldout_category']} | {r['A1_minus_A0']:+.6f} | {r['A1_minus_A2']:+.6f} | {r['A1_minus_Ar']:+.6f} |")
    lines.extend([
        "",
        "## Historical Control",
        "- A strict previous category-method final-gradient artifact was not available, so that comparison is reported as unavailable rather than reconstructed loosely.",
        "",
        "## Interpretation",
        "- Passing requires hard category gradients to be non-collapsed, the meta-gradient to differ from B0 and simple gradient consensus, and the hard-specific meta-gradient to align better with held-out categories than random and full-objective meta controls.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MetaTransfer] device={device}", flush=True)
    pool = build_clip_radio_pool(device, smoke=args.smoke)
    cat_rows, _hard_grads, _full_grads = category_gradients(pool)
    loo_rows, meta_geoms, controls = loo_meta(pool)
    outcome, stats = decide(cat_rows, meta_geoms, loo_rows)
    runtime = time.time() - start
    write_csv(OUT / "CATEGORY_GRADIENT_GEOMETRY.csv", cat_rows)
    write_csv(OUT / "META_GRADIENT_GEOMETRY.csv", meta_geoms)
    write_csv(OUT / "LOO_META_TRANSFER.csv", loo_rows)
    write_csv(OUT / "CONTROL_COMPARISON.csv", controls)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "no_train": True,
        "no_target": True,
        "optimizer_step": False,
        "virtual_alpha": LR,
        "second_order_gradient": True,
        "parameterization": "formal HardNormal normalized contrast/scorer direction",
    })
    write_report(outcome, stats, loo_rows, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
