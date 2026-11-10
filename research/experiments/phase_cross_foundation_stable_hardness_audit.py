import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_frozen_spatial_backbone_compatibility import (
    OUT as RADIO_OUT,
    PatchStateModel,
    RadioFeatureExtractor,
    raw_source_rows,
)
from experiments.phase_hard_normal_contrast import OUT as CLIP_HN_OUT
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_state_pair_decomposition import decompose
from experiments.phase_statealigned_pc1_debias import OUT as PC1_DEBIAS_OUT, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np


OUT = Path("results/phase_cross_foundation_stable_hardness_audit/formal")
TOP_FRAC = 0.10


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


def load_payload(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def cos(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def top_mask(scores, frac=TOP_FRAC):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(math.ceil(scores.size * frac)))
    ids = np.argpartition(scores, scores.size - k)[scores.size - k :]
    mask = np.zeros(scores.shape, dtype=bool)
    mask[ids] = True
    return mask


def initialize_clip_context(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    m_c, _d_c = decompose(states_c)
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    alpha = float(np.linalg.norm(d_t))
    v_hard = np.load(CLIP_HN_OUT / "contrast_vectors.npz")["E1"].astype(np.float32)
    align_model = make_model(states_c, corrupt_payload, device)
    return q_pc1, m_c.astype(np.float32), alpha, unit_np(v_hard), align_model


@torch.no_grad()
def ensure_radio_source_scores_37(device, smoke=False):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / "cache" / f"radio_statealigned_source_scores37_{suffix}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    rows = raw_source_rows()
    if smoke:
        rows = rows[:120]
    model = PatchStateModel(1024).to(device).eval()
    ckpt = RADIO_OUT / "checkpoints" / "E1_RADIO_STATEALIGNED_formal.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    extractor = RadioFeatureExtractor(device, batch_size=2)
    scores = np.zeros((len(rows), SIDE * SIDE), dtype=np.float16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), extractor.batch_size):
        if start % 50 == 0:
            print(f"[StableHardness] RADIO source score {start}/{len(rows)}", flush=True)
        batch = rows[start : start + extractor.batch_size]
        feats = extractor.extract_paths([r["image_path"] for r in batch])
        for off, patch in enumerate(feats):
            z = torch.tensor(patch, dtype=torch.float32, device=device)
            logits32 = model.logits(z).float().view(1, 1, 32, 32)
            score37 = F.interpolate(logits32, size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape(-1)
            scores[start + off] = score37.detach().cpu().numpy().astype(np.float16)
    np.save(cache, scores)
    return np.load(cache, mmap_mode="r")


def source_slots(dataset, indices, smoke=False):
    cats = sorted({str(dataset.data_all[int(i)]["cls_name"]) for i in indices})
    if smoke:
        cats = cats[:4]
    return {cat: [p for p, idx in enumerate(indices) if str(dataset.data_all[int(idx)]["cls_name"]) == cat] for cat in cats}


def build_clip_radio_pool(device, smoke=False):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / "cache" / f"clip_radio_negative_pool_{suffix}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}

    q_pc1, m_c, alpha, v_hard, align_model = initialize_clip_context(device)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    radio37 = ensure_radio_source_scores_37(device, smoke=False)
    slots = source_slots(dataset, indices, smoke)
    rng = np.random.default_rng(SEED)
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    x_parts, y_parts, cat_parts, pos_parts, pid_parts, origin_parts, sr_parts = [], [], [], [], [], [], []

    m_t = torch.tensor(m_c, dtype=torch.float32, device=device)
    v_t = torch.tensor(v_hard, dtype=torch.float32, device=device)
    for cat, positions in slots.items():
        counts = defaultdict(int)
        for pos in positions:
            y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
            image_is_anomaly = bool(y.max() > 0)
            z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
            z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
            with torch.no_grad():
                z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
            pools = []
            if image_is_anomaly:
                pools.append(("anomaly_mask", np.flatnonzero(y == 1), 1))
                pools.append(("anomaly_mask_out", np.flatnonzero(y == 0), 0))
            else:
                pools.append(("normal_image", np.flatnonzero(y == 0), 0))
            for origin, ids, label in pools:
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - counts[(cat, origin)]
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                x_parts.append(z[ids].astype(np.float32))
                y_parts.append(np.full(ids.shape, label, dtype=np.float32))
                cat_parts.extend([cat] * len(ids))
                pos_parts.extend([pos] * len(ids))
                pid_parts.extend(ids.tolist())
                origin_parts.extend([origin] * len(ids))
                sr_parts.append(np.asarray(radio37[pos, ids], dtype=np.float32))
                counts[(cat, origin)] += int(len(ids))
    x = norm_np(np.concatenate(x_parts, axis=0).astype(np.float32))
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    with torch.no_grad():
        s_c = contrast_logits(
            torch.tensor(x, dtype=torch.float32, device=device),
            m_t,
            alpha,
            v_t,
        ).detach().cpu().numpy().astype(np.float32)
    payload = {
        "x": x.astype(np.float16),
        "y": y.astype(np.float32),
        "category": np.asarray(cat_parts, dtype=object),
        "position": np.asarray(pos_parts, dtype=np.int32),
        "patch_id": np.asarray(pid_parts, dtype=np.int32),
        "origin": np.asarray(origin_parts, dtype=object),
        "S_C": s_c.astype(np.float32),
        "S_R": np.concatenate(sr_parts, axis=0).astype(np.float32),
        "m_c": m_c.astype(np.float32),
        "alpha": np.asarray(alpha, dtype=np.float32),
        "v_hard": v_hard.astype(np.float32),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    return payload


def hard_sets(pool, subset=None):
    is_neg = pool["y"] < 0.5
    if subset is not None:
        is_neg &= subset
    neg_ids = np.flatnonzero(is_neg)
    hc_local = top_mask(pool["S_C"][neg_ids])
    hr_local = top_mask(pool["S_R"][neg_ids])
    hc = neg_ids[hc_local]
    hr = neg_ids[hr_local]
    stable = np.intersect1d(hc, hr, assume_unique=False)
    clip_only = np.setdiff1d(hc, hr, assume_unique=False)
    radio_only = np.setdiff1d(hr, hc, assume_unique=False)
    return neg_ids, hc, hr, stable, clip_only, radio_only


def correlation_rows(pool):
    neg = pool["y"] < 0.5
    sc = pool["S_C"][neg]
    sr = pool["S_R"][neg]
    return [
        {"scope": "all_negative_pool", "metric": "pearson_S_C_S_R", "value": float(pearsonr(sc, sr).statistic)},
        {"scope": "all_negative_pool", "metric": "spearman_S_C_S_R", "value": float(spearmanr(sc, sr).statistic)},
    ]


def overlap_rows(pool):
    rows = []
    cats = ["__all__"] + sorted(set(pool["category"].tolist()))
    for cat in cats:
        subset = None if cat == "__all__" else (pool["category"] == cat)
        neg, hc, hr, stable, clip_only, radio_only = hard_sets(pool, subset)
        union = np.union1d(hc, hr)
        rows.append({
            "category": cat,
            "normal_pool": int(len(neg)),
            "H_C": int(len(hc)),
            "H_R": int(len(hr)),
            "H_stable": int(len(stable)),
            "H_clip_only": int(len(clip_only)),
            "H_radio_only": int(len(radio_only)),
            "H_stable_pool_ratio": float(len(stable) / max(len(neg), 1)),
            "Jaccard_HC_HR": float(len(stable) / max(len(union), 1)),
            "stable_over_HC": float(len(stable) / max(len(hc), 1)),
        })
    return rows


def grad_for_indices(pool, ids, m_c, alpha, v_hard, loss_kind="normal"):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.zeros_like(v_hard, dtype=np.float32)
    v = torch.tensor(v_hard, dtype=torch.float32, requires_grad=True)
    z = torch.tensor(np.asarray(pool["x"][ids], dtype=np.float32), dtype=torch.float32)
    m = torch.tensor(m_c, dtype=torch.float32)
    logits = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
    if loss_kind == "normal":
        loss = F.softplus(logits).mean()
    elif loss_kind == "task":
        y = torch.tensor(pool["y"][ids], dtype=torch.float32)
        loss, _bce, _dice = balanced_loss(logits, y)
    else:
        raise ValueError(loss_kind)
    loss.backward()
    return v.grad.detach().cpu().numpy().astype(np.float32)


def b0_gradient(pool, subset, m_c, alpha, v_hard):
    all_ids = np.flatnonzero(subset)
    _neg, hc, _hr, _stable, _clip_only, _radio_only = hard_sets(pool, subset)
    g_task = grad_for_indices(pool, all_ids, m_c, alpha, v_hard, "task")
    g_h = grad_for_indices(pool, hc, m_c, alpha, v_hard, "normal")
    return g_task + g_h


def gradient_geometry(pool):
    m_c = pool["m_c"].astype(np.float32)
    alpha = float(pool["alpha"])
    v_hard = pool["v_hard"].astype(np.float32)
    all_subset = np.ones(pool["y"].shape, dtype=bool)
    _neg, hc, _hr, stable, clip_only, radio_only = hard_sets(pool)
    rng = np.random.default_rng(SEED)
    random_h = rng.choice(hc, size=len(stable), replace=False) if len(stable) and len(hc) >= len(stable) else np.asarray([], dtype=np.int64)
    g_h = grad_for_indices(pool, hc, m_c, alpha, v_hard, "normal")
    g_stable = grad_for_indices(pool, stable, m_c, alpha, v_hard, "normal")
    g_clip = grad_for_indices(pool, clip_only, m_c, alpha, v_hard, "normal")
    g_radio = grad_for_indices(pool, radio_only, m_c, alpha, v_hard, "normal")
    g_random = grad_for_indices(pool, random_h, m_c, alpha, v_hard, "normal")
    g_b0 = b0_gradient(pool, all_subset, m_c, alpha, v_hard)
    g_new = g_b0 + g_stable
    g_random_new = g_b0 + g_random
    rows = []
    for name, g in [("g_H", g_h), ("g_stable", g_stable), ("g_clip_only", g_clip), ("g_radio_only", g_radio), ("g_random", g_random), ("g_B0", g_b0)]:
        rows.append({"scope": "all", "gradient": name, "norm": float(np.linalg.norm(g)), "cos_gH": cos(g_h, g)})
    rows.extend([
        {"scope": "all", "gradient": "g_stable", "metric": "cos_g_H_g_stable", "value": cos(g_h, g_stable), "norm": float(np.linalg.norm(g_stable))},
        {"scope": "all", "gradient": "g_clip_only", "metric": "cos_g_H_g_clip_only", "value": cos(g_h, g_clip), "norm": float(np.linalg.norm(g_clip))},
        {"scope": "all", "gradient": "g_radio_only", "metric": "cos_g_H_g_radio_only", "value": cos(g_h, g_radio), "norm": float(np.linalg.norm(g_radio))},
        {"scope": "all", "gradient": "g_new", "metric": "cos_g_new_g_B0", "value": cos(g_new, g_b0), "norm": float(np.linalg.norm(g_new))},
        {"scope": "all", "gradient": "g_random_new", "metric": "cos_g_random_new_g_B0", "value": cos(g_random_new, g_b0), "norm": float(np.linalg.norm(g_random_new))},
    ])
    return rows, {"g_b0": g_b0, "g_stable": g_stable, "g_random": g_random, "g_clip_only": g_clip}


def loo_transfer(pool):
    m_c = pool["m_c"].astype(np.float32)
    alpha = float(pool["alpha"])
    v_hard = pool["v_hard"].astype(np.float32)
    cats = sorted(set(pool["category"].tolist()))
    rows = []
    for cat in cats:
        train = pool["category"] != cat
        hold = pool["category"] == cat
        _neg, _hc, _hr, stable, clip_only, _radio_only = hard_sets(pool, train)
        rng = np.random.default_rng(SEED)
        hc_train = hard_sets(pool, train)[1]
        random_h = rng.choice(hc_train, size=len(stable), replace=False) if len(stable) and len(hc_train) >= len(stable) else np.asarray([], dtype=np.int64)
        g_b0_train = b0_gradient(pool, train, m_c, alpha, v_hard)
        g_stable_train = grad_for_indices(pool, stable, m_c, alpha, v_hard, "normal")
        g_random_train = grad_for_indices(pool, random_h, m_c, alpha, v_hard, "normal")
        g_clip_train = grad_for_indices(pool, clip_only, m_c, alpha, v_hard, "normal")
        g_holdout = b0_gradient(pool, hold, m_c, alpha, v_hard)
        a0 = cos(g_b0_train, g_holdout)
        a1 = cos(g_b0_train + g_stable_train, g_holdout)
        a2 = cos(g_b0_train + g_random_train, g_holdout)
        a_clip = cos(g_b0_train + g_clip_train, g_holdout)
        rows.append({
            "heldout_category": cat,
            "A0_cos_B0_train_holdout": a0,
            "A1_cos_stable_new_holdout": a1,
            "A2_cos_random_new_holdout": a2,
            "Aclip_cos_cliponly_new_holdout": a_clip,
            "A1_minus_A0": a1 - a0,
            "A1_minus_A2": a1 - a2,
            "A1_minus_Aclip": a1 - a_clip,
            "train_stable_count": int(len(stable)),
            "train_clip_only_count": int(len(clip_only)),
        })
        print(f"[StableHardness] LOO {cat}: A1-A0={a1-a0:+.4f} A1-A2={a1-a2:+.4f}", flush=True)
    return rows


def category_hardness(pool):
    rows = []
    for cat in sorted(set(pool["category"].tolist())):
        subset = pool["category"] == cat
        neg, hc, hr, stable, clip_only, radio_only = hard_sets(pool, subset)
        rows.append({
            "category": cat,
            "normal_pool": int(len(neg)),
            "S_C_mean": float(np.mean(pool["S_C"][neg])),
            "S_R_mean": float(np.mean(pool["S_R"][neg])),
            "S_C_S_R_pearson": float(pearsonr(pool["S_C"][neg], pool["S_R"][neg]).statistic) if len(neg) > 2 else float("nan"),
            "S_C_S_R_spearman": float(spearmanr(pool["S_C"][neg], pool["S_R"][neg]).statistic) if len(neg) > 2 else float("nan"),
            "H_C": int(len(hc)),
            "H_R": int(len(hr)),
            "H_stable": int(len(stable)),
            "H_clip_only": int(len(clip_only)),
            "H_radio_only": int(len(radio_only)),
            "stable_over_HC": float(len(stable) / max(len(hc), 1)),
        })
    return rows


def decide(overlap, grad_rows, loo_rows):
    all_overlap = next(r for r in overlap if r["category"] == "__all__")
    stable_ratio = all_overlap["stable_over_HC"] * 100.0
    cos_new = next(r for r in grad_rows if r.get("metric") == "cos_g_new_g_B0")["value"]
    mean_a10 = float(np.mean([r["A1_minus_A0"] for r in loo_rows]))
    mean_a12 = float(np.mean([r["A1_minus_A2"] for r in loo_rows]))
    mean_a1clip = float(np.mean([r["A1_minus_Aclip"] for r in loo_rows]))
    improved = int(np.sum([r["A1_minus_A0"] > 0 for r in loo_rows]))
    if not (10.0 <= stable_ratio <= 80.0):
        return "CROSS_FOUNDATION_STABLE_HARDNESS_WEAK", locals()
    if cos_new >= 0.995:
        return "STABLE_HARDNESS_COLLAPSES_TO_HARDNORMAL_REWEIGHTING", locals()
    if mean_a12 < 0.015:
        return "CROSS_FOUNDATION_STABILITY_NOT_SPECIFIC", locals()
    if mean_a1clip < 0.015:
        return "FOUNDATION_CONSENSUS_NOT_MORE_TRANSFERABLE", locals()
    if improved >= 8 and mean_a10 >= 0.02 and mean_a12 >= 0.015 and mean_a1clip >= 0.015 and cos_new <= 0.99:
        return "CROSS_FOUNDATION_STABLE_HARDNESS_HAS_TRANSFER_SIGNAL", locals()
    return "CROSS_FOUNDATION_STABLE_HARDNESS_WEAK", locals()


def write_report(outcome, stats, corr, overlap, grad, loo, runtime):
    all_overlap = next(r for r in overlap if r["category"] == "__all__")
    lines = [
        "# Cross-Foundation Stable Hardness Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN / NO-TARGET. Full VisA source only. CLIP HardNormal is the formal contrast checkpoint; RADIO StateAligned is used only as a source-side verifier. No optimizer step, target inference, fusion, or distillation was run.",
        "",
        "## Score Alignment",
    ]
    for r in corr:
        lines.append(f"- {r['metric']}: {r['value']:.6f}")
    lines.extend([
        "",
        "## Hard Set",
        f"- H_C: {all_overlap['H_C']}, H_R: {all_overlap['H_R']}, H_stable: {all_overlap['H_stable']}",
        f"- Jaccard(H_C,H_R): {all_overlap['Jaccard_HC_HR']:.6f}",
        f"- |H_stable| / |H_C|: {all_overlap['stable_over_HC'] * 100.0:.2f}%",
        "",
        "## Gradient Gate",
    ])
    for key in ["cos_new", "mean_a10", "mean_a12", "mean_a1clip", "improved"]:
        if key in stats:
            lines.append(f"- {key}: {stats[key]}")
    lines.extend([
        "",
        "## Interpretation",
        "- The audit checks whether cross-foundation stable hard normals form a nontrivial subset and whether adding their CLIP normal-loss gradient changes the formal HardNormal gradient toward held-out source categories.",
        "- If the additive gradient is almost collinear with B0, the stable set is treated as reweighting rather than new transferable evidence.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[StableHardness] device={device}", flush=True)
    pool = build_clip_radio_pool(device, smoke=args.smoke)
    corr = correlation_rows(pool)
    overlap = overlap_rows(pool)
    grad_rows, _grads = gradient_geometry(pool)
    loo = loo_transfer(pool)
    cat_rows = category_hardness(pool)
    outcome, stats = decide(overlap, grad_rows, loo)
    runtime = time.time() - start
    write_csv(OUT / "HARD_SET_OVERLAP.csv", overlap)
    write_csv(OUT / "GRADIENT_GEOMETRY.csv", corr + grad_rows)
    write_csv(OUT / "LOO_GRADIENT_TRANSFER.csv", loo)
    write_csv(OUT / "CATEGORY_HARDNESS.csv", cat_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "strict_no_train": True,
        "target_data_used": False,
        "radio_usage": "source_side_verifier_only",
        "gradient_parameters": "formal CLIP HardNormal global contrast vector",
        "normal_pool": "current capped category/origin-balanced HardNormal source training pool",
    })
    write_report(outcome, stats, corr, overlap, grad_rows, loo, runtime)
    print(json.dumps({"outcome": outcome, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
