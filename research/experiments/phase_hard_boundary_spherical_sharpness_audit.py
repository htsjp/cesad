import argparse
import json
import math
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
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xliii_adapter_head_decomposition import balanced_loss


OUT = Path("results/phase_hard_boundary_spherical_sharpness_audit/formal")
SEED = 0
TOP_FRAC = 0.10
ALPHA_DEG = 5.0


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


def cos_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def top_hard_indices(scores, neg_ids):
    scores = np.asarray(scores, dtype=np.float32)
    neg_ids = np.asarray(neg_ids, dtype=np.int64)
    k = max(1, int(math.ceil(len(neg_ids) * TOP_FRAC)))
    local = np.argpartition(scores[neg_ids], len(neg_ids) - k)[len(neg_ids) - k :]
    return neg_ids[local]


def tensor_context(pool, subset=None):
    if subset is None:
        subset = np.ones(pool["y"].shape, dtype=bool)
    ids = np.flatnonzero(subset)
    z = torch.tensor(np.asarray(pool["x"][ids], dtype=np.float32), dtype=torch.float32)
    y = torch.tensor(np.asarray(pool["y"][ids], dtype=np.float32), dtype=torch.float32)
    m = torch.tensor(np.asarray(pool["m_c"], dtype=np.float32), dtype=torch.float32)
    alpha = float(pool["alpha"])
    return ids, z, y, m, alpha


def logits_np(pool, v, subset=None):
    ids, z, _y, m, alpha = tensor_context(pool, subset)
    with torch.no_grad():
        vt = torch.tensor(unit_np(v), dtype=torch.float32)
        out = contrast_logits(z, m, alpha, vt).detach().cpu().numpy().astype(np.float32)
    full = np.zeros(pool["y"].shape, dtype=np.float32)
    full[ids] = out
    return full


def loss_value(pool, v, ids, kind="normal"):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return float("nan")
    z = torch.tensor(np.asarray(pool["x"][ids], dtype=np.float32), dtype=torch.float32)
    y = torch.tensor(np.asarray(pool["y"][ids], dtype=np.float32), dtype=torch.float32)
    m = torch.tensor(np.asarray(pool["m_c"], dtype=np.float32), dtype=torch.float32)
    vt = torch.tensor(unit_np(v), dtype=torch.float32)
    logits = contrast_logits(z, m, float(pool["alpha"]), vt)
    if kind == "normal":
        return float(F.softplus(logits).mean().detach().cpu())
    loss, _bce, _dice = balanced_loss(logits, y)
    return float(loss.detach().cpu())


def gradient(pool, v_init, ids_task, hard_ids=None):
    v = torch.tensor(unit_np(v_init), dtype=torch.float32, requires_grad=True)
    m = torch.tensor(np.asarray(pool["m_c"], dtype=np.float32), dtype=torch.float32)
    alpha = float(pool["alpha"])
    z_task = torch.tensor(np.asarray(pool["x"][ids_task], dtype=np.float32), dtype=torch.float32)
    y_task = torch.tensor(np.asarray(pool["y"][ids_task], dtype=np.float32), dtype=torch.float32)
    logits = contrast_logits(z_task, m, alpha, F.normalize(v, dim=-1))
    loss, _bce, _dice = balanced_loss(logits, y_task)
    if hard_ids is not None and len(hard_ids):
        z_h = torch.tensor(np.asarray(pool["x"][hard_ids], dtype=np.float32), dtype=torch.float32)
        logits_h = contrast_logits(z_h, m, alpha, F.normalize(v, dim=-1))
        loss = loss + F.softplus(logits_h).mean()
    loss.backward()
    return v.grad.detach().cpu().numpy().astype(np.float32), float(loss.detach().cpu())


def hard_loss_gradient(pool, v_init, hard_ids):
    v = torch.tensor(unit_np(v_init), dtype=torch.float32, requires_grad=True)
    m = torch.tensor(np.asarray(pool["m_c"], dtype=np.float32), dtype=torch.float32)
    z = torch.tensor(np.asarray(pool["x"][hard_ids], dtype=np.float32), dtype=torch.float32)
    logits = contrast_logits(z, m, float(pool["alpha"]), F.normalize(v, dim=-1))
    loss = F.softplus(logits).mean()
    loss.backward()
    return v.grad.detach().cpu().numpy().astype(np.float32), float(loss.detach().cpu())


def tangent_project(g, d_hat):
    d = unit_np(d_hat)
    return np.asarray(g, dtype=np.float32) - float(np.dot(g, d)) * d


def angular_perturb(d_hat, tangent, deg=ALPHA_DEG):
    theta = math.radians(deg)
    u = unit_np(tangent)
    return unit_np(math.cos(theta) * unit_np(d_hat) + math.sin(theta) * u)


def random_tangent(d_hat, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=d_hat.shape).astype(np.float32)
    return unit_np(tangent_project(v, d_hat))


def sharpness_for_subset(pool, subset, seed=SEED, source="hard"):
    d_hat = unit_np(pool["v_hard"])
    base_scores = logits_np(pool, d_hat, subset)
    neg_ids = np.flatnonzero((pool["y"] < 0.5) & subset)
    hard_ids = top_hard_indices(base_scores, neg_ids)
    all_ids = np.flatnonzero(subset)
    if source == "hard":
        g_seed, hard_base = hard_loss_gradient(pool, d_hat, hard_ids)
    elif source == "full":
        g_seed, _loss = gradient(pool, d_hat, all_ids, hard_ids)
        hard_base = loss_value(pool, d_hat, hard_ids, "normal")
    else:
        raise ValueError(source)
    g_t = tangent_project(g_seed, d_hat)
    d_adv = angular_perturb(d_hat, g_t)
    d_rand = angular_perturb(d_hat, random_tangent(d_hat, seed))
    hard_adv = loss_value(pool, d_adv, hard_ids, "normal")
    hard_rand = loss_value(pool, d_rand, hard_ids, "normal")
    adv_scores = logits_np(pool, d_adv, subset)
    rand_scores = logits_np(pool, d_rand, subset)
    hard_adv_ids = top_hard_indices(adv_scores, neg_ids)
    hard_rand_ids = top_hard_indices(rand_scores, neg_ids)
    g_b0, loss_b0 = gradient(pool, d_hat, all_ids, hard_ids)
    g_adv, loss_adv = gradient(pool, d_adv, all_ids, hard_adv_ids)
    g_rand, loss_rand = gradient(pool, d_rand, all_ids, hard_rand_ids)
    return {
        "d_hat": d_hat,
        "hard_ids": hard_ids,
        "d_adv": d_adv,
        "d_rand": d_rand,
        "g_seed": g_seed,
        "g_tangent": g_t,
        "g_B0": g_b0,
        "g_adv": g_adv,
        "g_rand": g_rand,
        "loss_B0": loss_b0,
        "loss_adv": loss_adv,
        "loss_rand": loss_rand,
        "L_hard_base": hard_base,
        "L_hard_adv": hard_adv,
        "L_hard_rand": hard_rand,
        "sharpness_gap": hard_adv - hard_base,
        "random_gap": hard_rand - hard_base,
        "tangent_ratio": float(np.linalg.norm(g_t) / max(np.linalg.norm(g_seed), 1e-12)),
    }


def global_rows(res_hard, res_full):
    ratio = res_hard["sharpness_gap"] / max(abs(res_hard["random_gap"]), 1e-12)
    stats = [{
        "scope": "all",
        "alpha_deg": ALPHA_DEG,
        "source": "hard",
        "grad_norm": float(np.linalg.norm(res_hard["g_seed"])),
        "tangent_norm": float(np.linalg.norm(res_hard["g_tangent"])),
        "tangent_ratio": res_hard["tangent_ratio"],
        "L_hard_base": res_hard["L_hard_base"],
        "L_hard_adv": res_hard["L_hard_adv"],
        "L_hard_rand": res_hard["L_hard_rand"],
        "sharpness_gap": res_hard["sharpness_gap"],
        "random_gap": res_hard["random_gap"],
        "sharpness_gap_over_random_gap_abs": ratio,
        "hard_set_size": int(len(res_hard["hard_ids"])),
    }]
    geom = [{
        "scope": "all",
        "cos_g_B0_g_adv": cos_np(res_hard["g_B0"], res_hard["g_adv"]),
        "cos_g_B0_g_rand": cos_np(res_hard["g_B0"], res_hard["g_rand"]),
        "cos_g_adv_g_rand": cos_np(res_hard["g_adv"], res_hard["g_rand"]),
        "norm_g_B0": float(np.linalg.norm(res_hard["g_B0"])),
        "norm_g_adv": float(np.linalg.norm(res_hard["g_adv"])),
        "norm_g_rand": float(np.linalg.norm(res_hard["g_rand"])),
        "norm_ratio_adv_B0": float(np.linalg.norm(res_hard["g_adv"]) / max(np.linalg.norm(res_hard["g_B0"]), 1e-12)),
        "norm_ratio_rand_B0": float(np.linalg.norm(res_hard["g_rand"]) / max(np.linalg.norm(res_hard["g_B0"]), 1e-12)),
        "cos_hard_adv_full_adv_gradient": cos_np(res_hard["g_adv"], res_full["g_adv"]),
    }]
    return stats, geom


def loo_rows(pool):
    cats = sorted(set(pool["category"].tolist()))
    rows = []
    for i, cat in enumerate(cats):
        train = pool["category"] != cat
        hold = pool["category"] == cat
        res_hard = sharpness_for_subset(pool, train, seed=SEED + i, source="hard")
        res_full = sharpness_for_subset(pool, train, seed=SEED + i, source="full")
        hold_scores = logits_np(pool, pool["v_hard"], hold)
        hold_neg = np.flatnonzero((pool["y"] < 0.5) & hold)
        hold_hard = top_hard_indices(hold_scores, hold_neg)
        hold_ids = np.flatnonzero(hold)
        g_hold, _loss = gradient(pool, pool["v_hard"], hold_ids, hold_hard)
        a0 = cos_np(res_hard["g_B0"], g_hold)
        a1 = cos_np(res_hard["g_adv"], g_hold)
        a2 = cos_np(res_hard["g_rand"], g_hold)
        afull = cos_np(res_full["g_adv"], g_hold)
        rows.append({
            "heldout_category": cat,
            "A0_cos_B0_train_holdout": a0,
            "A1_cos_hard_adv_train_holdout": a1,
            "A2_cos_random_adv_train_holdout": a2,
            "Afull_cos_full_adv_train_holdout": afull,
            "A1_minus_A0": a1 - a0,
            "A1_minus_A2": a1 - a2,
            "A1_minus_Afull": a1 - afull,
            "train_tangent_ratio": res_hard["tangent_ratio"],
            "train_sharpness_gap": res_hard["sharpness_gap"],
            "train_random_gap": res_hard["random_gap"],
            "train_hard_set_size": int(len(res_hard["hard_ids"])),
        })
        print(f"[SphericalSharpness] LOO {cat}: A1-A0={a1-a0:+.4f} A1-A2={a1-a2:+.4f} A1-Afull={a1-afull:+.4f}", flush=True)
    return rows


def decide(sharp_rows, geom_rows, loo):
    s = sharp_rows[0]
    g = geom_rows[0]
    a10 = np.asarray([r["A1_minus_A0"] for r in loo], dtype=np.float32)
    a12 = np.asarray([r["A1_minus_A2"] for r in loo], dtype=np.float32)
    a1f = np.asarray([r["A1_minus_Afull"] for r in loo], dtype=np.float32)
    stats = {
        "tangent_ratio": float(s["tangent_ratio"]),
        "sharpness_gap": float(s["sharpness_gap"]),
        "random_gap": float(s["random_gap"]),
        "sharpness_ratio_abs": float(s["sharpness_gap_over_random_gap_abs"]),
        "cos_g_B0_g_adv": float(g["cos_g_B0_g_adv"]),
        "improved_categories_A1_gt_A0": int(np.sum(a10 > 0)),
        "mean_A1_minus_A0": float(np.mean(a10)),
        "median_A1_minus_A0": float(np.median(a10)),
        "mean_A1_minus_A2": float(np.mean(a12)),
        "mean_A1_minus_Afull": float(np.mean(a1f)),
    }
    if stats["tangent_ratio"] < 0.2:
        return "HARD_BOUNDARY_SHARPNESS_NO_TRANSFER_SIGNAL", stats
    if stats["sharpness_gap"] <= stats["random_gap"] * 1.5:
        return "ANGULAR_SHARPNESS_NOT_SPECIFIC", stats
    if stats["cos_g_B0_g_adv"] >= 0.995:
        return "HARD_BOUNDARY_SHARPNESS_COLLAPSES_TO_RESCALING", stats
    if stats["mean_A1_minus_Afull"] < 0.01:
        return "GENERIC_SHARPNESS_NOT_HARD_SPECIFIC", stats
    if (
        stats["cos_g_B0_g_adv"] <= 0.99
        and stats["improved_categories_A1_gt_A0"] >= 8
        and stats["mean_A1_minus_A0"] >= 0.02
        and stats["mean_A1_minus_A2"] >= 0.015
        and stats["mean_A1_minus_Afull"] >= 0.01
    ):
        return "HARD_BOUNDARY_ANGULAR_SHARPNESS_HAS_TRANSFER_SIGNAL", stats
    return "HARD_BOUNDARY_SHARPNESS_NO_TRANSFER_SIGNAL", stats


def write_report(outcome, stats, sharp_rows, geom_rows, loo, runtime):
    lines = [
        "# Hard-Boundary Spherical Sharpness Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN / NO-TARGET. The formal Symmetric HardNormal contrast is frozen. All perturbations are virtual 5-degree tangent-space rotations of the state contrast direction; no optimizer step is performed.",
        "",
        "## Global Sharpness",
        f"- tangent_ratio: {stats['tangent_ratio']:.6f}",
        f"- sharpness_gap: {stats['sharpness_gap']:.6f}",
        f"- random_gap: {stats['random_gap']:.6f}",
        f"- sharpness/random(abs): {stats['sharpness_ratio_abs']:.6f}",
        f"- cos(g_B0,g_adv): {stats['cos_g_B0_g_adv']:.6f}",
        "",
        "## LOO Transfer",
        f"- A1>A0 categories: {stats['improved_categories_A1_gt_A0']}/12",
        f"- mean(A1-A0): {stats['mean_A1_minus_A0']:+.6f}",
        f"- median(A1-A0): {stats['median_A1_minus_A0']:+.6f}",
        f"- mean(A1-A2): {stats['mean_A1_minus_A2']:+.6f}",
        f"- mean hard-specific advantage over generic-full: {stats['mean_A1_minus_Afull']:+.6f}",
        "",
        "| Held-out | A1-A0 | A1-A2 | A1-Afull | tangent_ratio | sharp_gap | rand_gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in loo:
        lines.append(f"| {r['heldout_category']} | {r['A1_minus_A0']:+.6f} | {r['A1_minus_A2']:+.6f} | {r['A1_minus_Afull']:+.6f} | {r['train_tangent_ratio']:.6f} | {r['train_sharpness_gap']:+.6f} | {r['train_random_gap']:+.6f} |")
    lines.extend([
        "",
        "## Interpretation",
        "- A useful signal requires nontrivial tangent gradient, hard-specific sharpness greater than random, non-collinear adversarial gradient, and better held-out category alignment than both B0 and random tangent controls.",
        "- If generic full-objective perturbation matches or beats hard-specific perturbation, the effect is not attributable to HardNormal boundary sharpness.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SphericalSharpness] device={device}", flush=True)
    pool = build_clip_radio_pool(device, smoke=args.smoke)
    all_subset = np.ones(pool["y"].shape, dtype=bool)
    res_hard = sharpness_for_subset(pool, all_subset, seed=SEED, source="hard")
    res_full = sharpness_for_subset(pool, all_subset, seed=SEED, source="full")
    sharp_rows, geom_rows = global_rows(res_hard, res_full)
    loo = loo_rows(pool)
    outcome, stats = decide(sharp_rows, geom_rows, loo)
    runtime = time.time() - start
    write_csv(OUT / "SHARPNESS_STATS.csv", sharp_rows)
    write_csv(OUT / "GRADIENT_GEOMETRY.csv", geom_rows)
    write_csv(OUT / "LOO_GRADIENT_TRANSFER.csv", loo)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "no_train": True,
        "no_target": True,
        "optimizer_step": False,
        "alpha_deg": ALPHA_DEG,
        "top_hard_fraction": TOP_FRAC,
        "source_pool": "formal HardNormal capped source training pool",
    })
    write_report(outcome, stats, sharp_rows, geom_rows, loo, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
