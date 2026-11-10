import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cmab_compact_multi_anomaly_basis import load_mvtec_groups, load_visa_groups
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC
from experiments.phase_snir_source_nuisance_invariant_state_regularization import (
    contrast_direction_np,
    evaluate_pairs,
    fnum,
    finite_mean,
    finite_median,
    source_state_pair,
)
from experiments.phase_ssec_state_space_experience_consolidation import two_state_logits
from experiments.phase_trsl_tail_ranked_state_learning import hardnormal_loss
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, LR
from utils.transforms import get_transform


OUT = Path("results/phase_normal_envelope_distillation/formal")
VARIANTS = ("E0_Baseline", "E1_TrainingOnlyNormalChildren", "E2_NormalEnvelopeDistillation")
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
)
K_CHILD = 4
EPOCHS = 10
MARGIN = 0.05
LAMBDA_ENV = 0.1
LAMBDA_DISTILL = 0.1
TAIL_IMBALANCE_GATE = 1.5
TAIL_SPREAD_GATE = 0.05


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_source_groups(source, model_info, device, smoke):
    if source == "Full VisA":
        return load_visa_groups(model_info, device, smoke)
    return load_mvtec_groups(smoke)


def stack_pool(groups):
    xs, ys, cats = [], [], []
    for cat in sorted(groups):
        g = groups[cat]
        x = np.asarray(g["x"], dtype=np.float32)
        y = np.asarray(g["y"], dtype=np.float32)
        xs.append(x)
        ys.append(y)
        cats.extend([cat] * len(y))
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32), np.asarray(cats, dtype=object)


def category_balanced_batch(groups, assignments_by_cat, rng, total_batch):
    names = sorted(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, ks, cats = [], [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(np.asarray(data["y"]) == cls)
            if ids.size == 0:
                continue
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(np.asarray(data["x"], dtype=np.float32)[choice])
            ys.append(np.asarray(data["y"], dtype=np.float32)[choice])
            ks.append(assignments_by_cat[cat][choice])
            cats.extend([cat] * len(choice))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    k = np.concatenate(ks, axis=0).astype(np.int64)
    cat = np.asarray(cats, dtype=object)
    order = rng.permutation(len(y))
    return x[order], y[order], k[order], cat[order]


def kmeans_normal_modes(source, groups, init_states, device, smoke):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / "cache" / f"{source.replace(' ', '_')}_normal_kmeans_{suffix}.npz"
    x, y, cat = stack_pool(groups)
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        centers = data["centers"].astype(np.float32)
        assign_all = data["assign_all"].astype(np.int64)
    else:
        normal = x[y < 0.5]
        km = KMeans(n_clusters=K_CHILD, random_state=SEED, n_init=10, max_iter=300)
        km.fit(normal)
        centers = km.cluster_centers_.astype(np.float32)
        centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        assign_all = km.predict(x).astype(np.int64)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, centers=centers, assign_all=assign_all)
    assignments_by_cat = {}
    start = 0
    for c in sorted(groups):
        n = len(groups[c]["y"])
        assignments_by_cat[c] = assign_all[start:start + n]
        start += n
    s_n, s_a = init_states
    zt = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        q = two_state_logits(
            zt,
            torch.tensor(s_n, dtype=torch.float32, device=device),
            torch.tensor(s_a, dtype=torch.float32, device=device),
        ).detach().cpu().numpy().astype(np.float32)
    normal_ids = np.flatnonzero(y < 0.5)
    tail_k = max(1, int(math.ceil(len(normal_ids) * TOP_NEG_FRAC)))
    tail_ids = normal_ids[np.argsort(q[normal_ids])[-tail_k:]]
    tail_mask = np.zeros(len(y), dtype=bool)
    tail_mask[tail_ids] = True
    rows = []
    tail_fracs = []
    for k in range(K_CHILD):
        ids = normal_ids[assign_all[normal_ids] == k]
        scores = q[ids]
        if ids.size:
            tail_frac = float(tail_mask[ids].mean())
            high_risk = float(np.mean(scores >= np.percentile(q[normal_ids], 95)))
        else:
            tail_frac = float("nan")
            high_risk = float("nan")
        tail_fracs.append(tail_frac)
        rows.append({
            "source": source,
            "cluster": k,
            "patch_count": int(ids.size),
            "patch_fraction": float(ids.size / max(len(normal_ids), 1)),
            "baseline_score_mean": float(np.mean(scores)) if ids.size else float("nan"),
            "baseline_score_P95": float(np.percentile(scores, 95)) if ids.size else float("nan"),
            "ntr_tail_fraction": tail_frac,
            "source_high_risk_rate": high_risk,
        })
    cos = centers @ centers.T
    dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(cos, -1.0, 1.0)))
    pairs = []
    for i in range(K_CHILD):
        for j in range(i + 1, K_CHILD):
            pairs.append({
                "source": source,
                "cluster_i": i,
                "cluster_j": j,
                "centroid_cosine": float(cos[i, j]),
                "centroid_distance": float(dist[i, j]),
            })
    finite_tail = np.asarray([v for v in tail_fracs if np.isfinite(v)], dtype=np.float64)
    summary = {
        "source": source,
        "K": K_CHILD,
        "normal_patch_count": int(len(normal_ids)),
        "tail_fraction_min": float(finite_tail.min()) if finite_tail.size else float("nan"),
        "tail_fraction_max": float(finite_tail.max()) if finite_tail.size else float("nan"),
        "tail_fraction_spread": float(finite_tail.max() - finite_tail.min()) if finite_tail.size else float("nan"),
        "tail_imbalance_max_over_min": float(finite_tail.max() / max(float(finite_tail.min()), 1e-12)) if finite_tail.size else float("nan"),
        "centroid_pairwise_cosine_mean": finite_mean([p["centroid_cosine"] for p in pairs]),
        "centroid_pairwise_distance_mean": finite_mean([p["centroid_distance"] for p in pairs]),
    }
    return centers, assignments_by_cat, rows, pairs, summary


def has_hardness_heterogeneity(summary_rows):
    for row in summary_rows:
        if fnum(row, "tail_imbalance_max_over_min") >= TAIL_IMBALANCE_GATE and fnum(row, "tail_fraction_spread") >= TAIL_SPREAD_GATE:
            return True
    return False


def ned_losses(z, y, child_assign, s_n, s_a, children, variant):
    logits = two_state_logits(z, s_n, s_a)
    state_loss, bce, dice = balanced_loss(logits, y)
    ntr, hard_count = hardnormal_loss(logits, y)
    z_norm = F.normalize(z, dim=-1)
    s_a_norm = F.normalize(s_a, dim=-1)
    children_norm = F.normalize(children, dim=-1)
    child_logits = z_norm @ children_norm.T - (z_norm @ s_a_norm).unsqueeze(1)
    normal_ids = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    child_loss = torch.zeros((), dtype=z.dtype, device=z.device)
    if normal_ids.numel() > 0:
        ck = child_assign[normal_ids].long().clamp(0, children.shape[0] - 1)
        child_margin = child_logits[normal_ids, ck]
        child_loss = F.softplus(-child_margin).mean()
    envelope = torch.zeros((), dtype=z.dtype, device=z.device)
    distill = torch.zeros((), dtype=z.dtype, device=z.device)
    if variant == "E2_NormalEnvelopeDistillation":
        pos = torch.nonzero(y > 0.5, as_tuple=False).flatten()
        if pos.numel() > 0:
            sim_a = z_norm[pos] @ s_a_norm
            sim_child_max = (z_norm[pos] @ children_norm.T).max(dim=1).values
            envelope = F.softplus(sim_child_max - sim_a + MARGIN).mean()
        if normal_ids.numel() > 0:
            ck = child_assign[normal_ids].long().clamp(0, children.shape[0] - 1)
            m_child = child_logits[normal_ids, ck]
            m_global = -logits[normal_ids]
            distill = F.softplus(m_child - m_global).mean()
    loss = state_loss + ntr + child_loss + LAMBDA_ENV * envelope + LAMBDA_DISTILL * distill
    return loss, state_loss.detach(), ntr.detach(), child_loss.detach(), envelope.detach(), distill.detach(), bce.detach(), dice.detach(), hard_count


def train_variant(source, groups, assignments, init_states, centers, variant, device, smoke):
    if variant == "E0_Baseline":
        return init_states[0].astype(np.float32), init_states[1].astype(np.float32), centers.astype(np.float32), [], True
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "checkpoints" / f"{source.replace(' ', '_')}_{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["s_n"].astype(np.float32), payload["s_a"].astype(np.float32), payload["children"].astype(np.float32), payload["curves"], True
    print(f"[NED] train {source} {variant}", flush=True)
    s_n0, s_a0 = init_states
    s_n = torch.nn.Parameter(torch.tensor(s_n0, dtype=torch.float32, device=device))
    s_a = torch.nn.Parameter(torch.tensor(s_a0, dtype=torch.float32, device=device))
    children = torch.nn.Parameter(torch.tensor(centers, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([s_n, s_a, children], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, int(math.ceil(total / BATCH_SIZE)))
    if smoke:
        steps = min(steps, 2)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _ in range(steps):
            xb, yb, kb, _cb = category_balanced_batch(groups, assignments, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            k = torch.tensor(kb, dtype=torch.long, device=device)
            loss, state, ntr, child_loss, env, dist, bce, dice, hard_count = ned_losses(z, y, k, s_n, s_a, children, variant)
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(state.cpu()), float(ntr.cpu()), float(child_loss.cpu()), float(env.cpu()), float(dist.cpu()), float(bce.cpu()), float(dice.cpu()), hard_count))
        arr = np.asarray(vals, dtype=np.float64)
        row = {
            "source": source,
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "state_loss": float(arr[:, 1].mean()),
            "ntr_loss": float(arr[:, 2].mean()),
            "child_loss": float(arr[:, 3].mean()),
            "envelope_loss": float(arr[:, 4].mean()),
            "distillation_loss": float(arr[:, 5].mean()),
            "bce": float(arr[:, 6].mean()),
            "dice": float(arr[:, 7].mean()),
            "hard_patch_count": float(arr[:, 8].mean()),
        }
        curves.append(row)
        print(f"[NED] {source} {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} env={row['envelope_loss']:.4f} dist={row['distillation_loss']:.4f}", flush=True)
    payload = {
        "s_n": s_n.detach().cpu().numpy().astype(np.float32),
        "s_a": s_a.detach().cpu().numpy().astype(np.float32),
        "children": children.detach().cpu().numpy().astype(np.float32),
        "curves": curves,
    }
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt)
    return payload["s_n"], payload["s_a"], payload["children"], curves, False


def child_diagnostics(source, groups, assignments, states, children, device):
    x, y, _cat = stack_pool(groups)
    z = torch.tensor(x, dtype=torch.float32, device=device)
    sn, sa = states
    with torch.no_grad():
        q = two_state_logits(z, torch.tensor(sn, dtype=torch.float32, device=device), torch.tensor(sa, dtype=torch.float32, device=device)).detach().cpu().numpy()
    assign = np.concatenate([assignments[c] for c in sorted(groups)], axis=0)
    normal_ids = np.flatnonzero(y < 0.5)
    tail_k = max(1, int(math.ceil(len(normal_ids) * TOP_NEG_FRAC)))
    tail = normal_ids[np.argsort(q[normal_ids])[-tail_k:]]
    tail_mask = np.zeros(len(y), dtype=bool)
    tail_mask[tail] = True
    c = children / np.maximum(np.linalg.norm(children, axis=1, keepdims=True), 1e-12)
    snu = sn / max(float(np.linalg.norm(sn)), 1e-12)
    rows = []
    for k in range(K_CHILD):
        ids = normal_ids[assign[normal_ids] == k]
        rows.append({
            "source": source,
            "cluster": k,
            "usage_fraction": float(ids.size / max(len(normal_ids), 1)),
            "tail_fraction": float(tail_mask[ids].mean()) if ids.size else float("nan"),
            "child_to_global_normal_cosine": float(np.dot(c[k], snu)),
        })
    pair_rows = []
    sim = c @ c.T
    for i in range(K_CHILD):
        for j in range(i + 1, K_CHILD):
            pair_rows.append({"source": source, "child_i": i, "child_j": j, "child_pairwise_cosine": float(sim[i, j])})
    return rows, pair_rows


def state_geometry(states_by_source_variant):
    rows = []
    for source, variants in states_by_source_variant.items():
        d0 = contrast_direction_np(*variants["E0_Baseline"])
        for variant, states in variants.items():
            d = contrast_direction_np(*states)
            cos = float(np.dot(d0, d))
            rows.append({
                "source": source,
                "variant": variant,
                "cos_d_E0": cos,
                "state_direction_angle": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
                "sN_norm": float(np.linalg.norm(states[0])),
                "sA_norm": float(np.linalg.norm(states[1])),
                "inference_state_count": 2,
            })
    return rows


def delta_summary(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in VARIANTS[1:]:
        vals = []
        for source, target in PILOT_PAIRS:
            b = by.get((source, target, "E0_Baseline"))
            e = by.get((source, target, variant))
            if not b or not e:
                continue
            dap = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            vals.append(dap)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": dap,
                "Delta_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
        rows.append({
            "source": "SUMMARY",
            "dataset": "PILOT",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(vals),
            "median_delta_M_FR_P_AP": finite_median(vals),
            "improved": int(sum(v > 0 for v in vals if np.isfinite(v))),
            "non_drop_ge_minus_0_2": int(sum(v >= -0.2 for v in vals if np.isfinite(v))),
            "total": int(sum(np.isfinite(v) for v in vals)),
        })
    return rows


def decide(delta):
    e1s = next((r for r in delta if r.get("source") == "SUMMARY" and r.get("variant") == "E1_TrainingOnlyNormalChildren"), None)
    e2s = next((r for r in delta if r.get("source") == "SUMMARY" and r.get("variant") == "E2_NormalEnvelopeDistillation"), None)
    if not e2s:
        return "NORMAL_ENVELOPE_DISTILLATION_HAS_NO_COMPETITIVE_VALUE", {}
    e2 = [r for r in delta if r.get("variant") == "E2_NormalEnvelopeDistillation" and r.get("source") != "SUMMARY"]
    mean_delta = fnum(e2s, "mean_delta_M_FR_P_AP")
    improved = int(fnum(e2s, "improved", 0))
    non_drop = int(fnum(e2s, "non_drop_ge_minus_0_2", 0))
    focus = max([fnum(r, "Delta_M_FR_P_AP") for r in e2 if (r["source"], r["dataset"]) in {("Full MVTec", "VisA"), ("Full VisA", "MPDD")}], default=float("nan"))
    catastrophic = any(fnum(r, "Delta_M_FR_P_AP") < -5.0 for r in e2)
    e2_minus_e1 = mean_delta - (fnum(e1s, "mean_delta_M_FR_P_AP") if e1s else float("nan"))
    stats = {
        "E2_mean_delta_P_AP": mean_delta,
        "E2_improved": improved,
        "E2_non_drop_ge_minus_0_2": non_drop,
        "focus_best_delta_P_AP": focus,
        "catastrophic_drop": catastrophic,
        "E2_minus_E1_mean_delta_P_AP": e2_minus_e1,
        "inference_state_count": 2,
    }
    if mean_delta >= 1.0 and improved >= 2 and focus >= 1.0 and not catastrophic:
        return "NORMAL_ENVELOPE_DISTILLATION_SIGNIFICANTLY_IMPROVES_TRANSFER", stats
    if e1s and fnum(e1s, "mean_delta_M_FR_P_AP") > 0 and e2_minus_e1 <= 0.2:
        return "NORMAL_CHILDREN_HELP_BUT_DISTILLATION_HAS_NO_ADDED_VALUE", stats
    if mean_delta > 0 or (e1s and fnum(e1s, "mean_delta_M_FR_P_AP") > 0):
        return "NORMAL_STRUCTURAL_MODES_EXIST_BUT_DO_NOT_IMPROVE_TRANSFER", stats
    return "NORMAL_ENVELOPE_DISTILLATION_HAS_NO_COMPETITIVE_VALUE", stats


def write_report(outcome, gate_rows, gate_pair_rows, raw, delta, geometry, child_rows, train_rows, stats, runtime):
    lines = [
        "# Phase NED: Normal Envelope Distillation",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR + fixed 2x2 Local Re-encoding Pyramid. Normal child states exist only during source training; inference rows confirm the final scorer remains 1 normal state + 1 anomaly state.",
        "",
        f"K-means normal modes: `{K_CHILD}`. margin: `{MARGIN}`. lambda_env: `{LAMBDA_ENV}`. lambda_distill: `{LAMBDA_DISTILL}`.",
        "",
        "## Gate 0 Normal Mode Hardness",
        "",
        "| Source | tail imbalance | tail spread | mean centroid cosine | mean centroid distance |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in gate_rows:
        lines.append(f"| {r['source']} | {fnum(r,'tail_imbalance_max_over_min'):.4f} | {fnum(r,'tail_fraction_spread'):.4f} | {fnum(r,'centroid_pairwise_cosine_mean'):.4f} | {fnum(r,'centroid_pairwise_distance_mean'):.4f} |")
    lines += ["", "### Per-Cluster Tail Fractions", "", "| Source | cluster | patch fraction | score mean | score P95 | tail fraction | high-risk rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in gate_pair_rows:
        lines.append(f"| {r['source']} | {int(fnum(r,'cluster'))} | {fnum(r,'patch_fraction'):.4f} | {fnum(r,'baseline_score_mean'):.4f} | {fnum(r,'baseline_score_P95'):.4f} | {fnum(r,'ntr_tail_fraction'):.4f} | {fnum(r,'source_high_risk_rate'):.4f} |")
    if raw:
        lines += ["", "## Pilot Metrics", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |", "|---|---|---|---:|---:|---:|---:|"]
        for r in raw:
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
        lines += ["", "## Delta vs E0", "", "| Source | Target | Variant | Delta P-AP | Delta PRO |", "|---|---|---|---:|---:|"]
        for r in delta:
            if r.get("source") != "SUMMARY":
                lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} | {fnum(r,'Delta_M_FR_PRO'):+.4f} |")
        lines += ["", "## Summary", "", "| Variant | mean Delta P-AP | median | improved | non-drop |", "|---|---:|---:|---:|---:|"]
        for r in delta:
            if r.get("source") == "SUMMARY":
                lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {int(fnum(r,'improved',0))}/{int(fnum(r,'total',0))} | {int(fnum(r,'non_drop_ge_minus_0_2',0))}/{int(fnum(r,'total',0))} |")
    if geometry:
        lines += ["", "## State Geometry", "", "| Source | Variant | cos(d,E0) | angle | inference states |", "|---|---|---:|---:|---:|"]
        for r in geometry:
            lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'cos_d_E0'):.4f} | {fnum(r,'state_direction_angle'):.3f} | {int(fnum(r,'inference_state_count'))} |")
    if child_rows:
        lines += ["", "## Child Diagnostics", "", "| Source | Variant | cluster | usage | tail fraction | child-global cosine |", "|---|---|---:|---:|---:|---:|"]
        for r in child_rows:
            lines.append(f"| {r['source']} | {r['variant']} | {int(fnum(r,'cluster'))} | {fnum(r,'usage_fraction'):.4f} | {fnum(r,'tail_fraction'):.4f} | {fnum(r,'child_to_global_normal_cosine'):.4f} |")
    last = {}
    for r in train_rows:
        last[(r["source"], r["variant"])] = r
    if last:
        lines += ["", "## Training Endpoints", "", "| Source | Variant | source loss | NTR | envelope | distill |", "|---|---|---:|---:|---:|---:|"]
        for (_source, _variant), r in sorted(last.items()):
            lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'loss'):.4f} | {fnum(r,'ntr_loss'):.4f} | {fnum(r,'envelope_loss'):.4f} | {fnum(r,'distillation_loss'):.4f} |")
    lines += ["", f"Decision stats: `{json.dumps(stats, default=json_default)}`", "", f"Runtime: {runtime:.1f}s."]
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NED] device={device}", flush=True)
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    model_infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    groups_by_source, assignments_by_source, centers_by_source = {}, {}, {}
    gate_cluster_rows, gate_pair_rows, gate_summary_rows = [], [], []
    states_by_source_variant = {}
    for source, info in model_infos.items():
        print(f"[NED] load source groups {source}", flush=True)
        groups, _audit = load_source_groups(source, info, device, args.smoke)
        groups_by_source[source] = groups
        init = source_state_pair(info)
        centers, assignments, rows, pairs, summary = kmeans_normal_modes(source, groups, init, device, args.smoke)
        centers_by_source[source] = centers
        assignments_by_source[source] = assignments
        gate_cluster_rows.extend(rows)
        gate_pair_rows.extend(pairs)
        gate_summary_rows.append(summary)
        states_by_source_variant[source] = {"E0_Baseline": init}
    write_csv(OUT / "NORMAL_MODE_HARDNESS_AUDIT.csv", gate_cluster_rows)
    write_csv(OUT / "NORMAL_CLUSTER_GEOMETRY.csv", gate_pair_rows)
    write_csv(OUT / "NORMAL_MODE_HARDNESS_SUMMARY.csv", gate_summary_rows)
    if not has_hardness_heterogeneity(gate_summary_rows):
        outcome = "NORMAL_STRUCTURAL_MODES_DO_NOT_EXPLAIN_HARD_NORMALS"
        geometry = state_geometry(states_by_source_variant)
        runtime = time.time() - start
        stats = {"gate0_hardness_heterogeneity": False}
        write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": stats, "runtime_sec": runtime})
        write_report(outcome, gate_summary_rows, gate_cluster_rows, [], [], geometry, [], [], stats, runtime)
        print(f"[NED] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)
        return
    train_rows = []
    child_rows, child_pair_rows = [], []
    for source in model_infos:
        init = states_by_source_variant[source]["E0_Baseline"]
        for variant in VARIANTS[1:]:
            s_n, s_a, children, curves, reused = train_variant(source, groups_by_source[source], assignments_by_source[source], init, centers_by_source[source], variant, device, args.smoke)
            states_by_source_variant[source][variant] = (s_n, s_a)
            for r in curves:
                r["checkpoint_reused"] = bool(reused)
                train_rows.append(r)
            cr, cp = child_diagnostics(source, groups_by_source[source], assignments_by_source[source], (s_n, s_a), children, device)
            for r in cr:
                r["variant"] = variant
            for r in cp:
                r["variant"] = variant
            child_rows.extend(cr)
            child_pair_rows.extend(cp)
    raw, cats = evaluate_pairs(model_infos, states_by_source_variant, clip_model, preprocess, PILOT_PAIRS, device, args.smoke, VARIANTS)
    delta = delta_summary(raw)
    geometry = state_geometry(states_by_source_variant)
    outcome, stats = decide(delta)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", delta)
    write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
    write_csv(OUT / "CHILD_STATE_DIAGNOSTICS.csv", child_rows)
    write_csv(OUT / "CHILD_PAIRWISE_COSINE.csv", child_pair_rows)
    write_csv(OUT / "SOURCE_TRAINING_DIAGNOSTICS.csv", train_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "K_child": K_CHILD,
        "margin": MARGIN,
        "lambda_env": LAMBDA_ENV,
        "lambda_distill": LAMBDA_DISTILL,
        "inference_state_count": 2,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_tuning": False,
        "fixed_lrp": True,
    })
    write_report(outcome, gate_summary_rows, gate_cluster_rows, raw, delta, geometry, child_rows, train_rows, stats, runtime)
    print(f"[NED] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
