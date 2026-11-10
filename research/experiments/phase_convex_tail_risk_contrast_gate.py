import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_frozen_semantic_anchor_fusion import load_b0_rows
from experiments.phase_hard_normal_contrast import collect_source_patch_pool
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import SEED
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np


OUT = Path("results/phase_convex_tail_risk_contrast_gate/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
ALPHA_CVAR = 0.90
MAX_ITER = 500
TARGETS = ("MVTec", "BTAD", "CVC-ClinicDB")
EPS = 1e-12
OBJECTIVE_VERSION = "weighted_bce_convex_eta0"


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
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def cos_np(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))


def corr_np(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_payloads(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = torch.load(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", map_location=device, weights_only=False)
    corrupt_payload = torch.load(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", map_location=device, weights_only=False)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    contrast = np.load(HN_OUT / "contrast_vectors.npz")
    v_hn = unit_np(contrast["E1"].astype(np.float32))
    alpha = float(np.linalg.norm(d_t))
    states_hn = hybrid_states(m_c, alpha * v_hn).astype(np.float32)
    states_hn_norm = norm_np(states_hn)
    w_b0 = (states_hn_norm[1] - states_hn_norm[0]).astype(np.float32)
    return q_pc1, corrupt_payload, states_c, states_hn, w_b0


def prepare_source(q_pc1, states_c, corrupt_payload, device, smoke):
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, smoke)
    groups, pool_rows = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    xs, ys = [], []
    for data in groups.values():
        xs.append(data["x"])
        ys.append(data["y"])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    return x, y, pool_rows


def source_loss_terms(logits, y, eta=None):
    y = y.float()
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    weights = torch.where(y > 0.5, 0.5 * (pos + neg) / pos, 0.5 * (pos + neg) / neg)
    base = F.binary_cross_entropy_with_logits(logits, y, weight=weights)
    with torch.no_grad():
        p = torch.sigmoid(logits)
        dice_det = 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)
        bce_det = base.detach()
    neg_loss = F.softplus(logits[y < 0.5])
    if neg_loss.numel() == 0:
        cvar = torch.zeros((), dtype=logits.dtype, device=logits.device)
    elif eta is None:
        k = max(1, int(np.ceil(float(neg_loss.numel()) * (1.0 - ALPHA_CVAR))))
        cvar = torch.topk(neg_loss, k=k, largest=True).values.mean()
    else:
        cvar = eta + F.relu(neg_loss - eta).mean() / max(1.0 - ALPHA_CVAR, EPS)
    return base, cvar, bce_det, dice_det


def objective_values(x, y, w_np, eta_value=None):
    with torch.no_grad():
        z = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        w = torch.tensor(w_np, dtype=torch.float32)
        logits = z @ w
        base, cvar, bce, dice = source_loss_terms(logits, yt, None if eta_value is None else torch.tensor(float(eta_value)))
        score = logits.cpu().numpy().astype(np.float32)
    p_auc = safe_auc(y, score)
    p_ap = safe_ap(y, score)
    return {
        "P-AUROC": p_auc,
        "P-AP": p_ap,
        "G_pixel": 0.5 * (p_auc + p_ap),
        "L_base": float(base.cpu()),
        "L_base_bce": float(bce.cpu()),
        "L_base_dice": float(dice.cpu()),
        "normal_CVaR": float(cvar.cpu()),
        "L_total_CTRC": float((base + cvar).cpu()),
        "w_norm": float(np.linalg.norm(w_np)),
    }


def algebraic_equivalence(x, states_hn, w_b0):
    states = norm_np(states_hn)
    q_state = x @ states[1] - x @ states[0]
    q_direct = x @ w_b0
    err = np.abs(q_state - q_direct)
    return [{
        "temperature_tau": 1.0,
        "state_normalization": "normalize(sA), normalize(sN)",
        "feature_normalization": "aligned z is already L2 normalized by AdapterToState.adapt",
        "sample_count": int(x.shape[0]),
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "passed": bool(float(err.max()) <= 1e-5),
    }]


def init_vector(name, w_b0, dim):
    if name == "I0_zero":
        return np.zeros((dim,), dtype=np.float32)
    if name == "I1_w_B0":
        return np.asarray(w_b0, dtype=np.float32).copy()
    seed = 0 if name == "I2_seed0_random_small" else 1
    rng = np.random.default_rng(seed)
    return (0.01 * rng.standard_normal(dim)).astype(np.float32)


def initial_eta(x, y, w_init):
    return 0.0


def optimal_cvar_eta(x, y, w_np):
    logits = x @ np.asarray(w_np, dtype=np.float32)
    losses = np.logaddexp(logits[y < 0.5], 0.0).astype(np.float32)
    return float(np.quantile(losses, ALPHA_CVAR)) if losses.size else 0.0


def optimize_direct(variant, objective, init_name, w_init, eta_init, x, y, smoke, device):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{init_name}_{suffix}_{OBJECTIVE_VERSION}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload
    print(f"[CTRC] optimize {variant} {init_name}", flush=True)
    z = torch.tensor(x, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    w = nn.Parameter(torch.tensor(w_init, dtype=torch.float32, device=device))
    params = [w]
    eta = None
    if objective == "ctrc":
        eta = nn.Parameter(torch.tensor(float(eta_init), dtype=torch.float32, device=device))
        params.append(eta)
    opt = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=20 if smoke else MAX_ITER,
        tolerance_change=1e-8,
        tolerance_grad=1e-9,
        line_search_fn="strong_wolfe",
    )
    history = []

    def closure():
        opt.zero_grad()
        logits = z @ w
        base, cvar, bce, dice = source_loss_terms(logits, yt, eta if objective == "ctrc" else None)
        loss = base if objective == "base" else base + cvar
        loss.backward()
        history.append({
            "variant": variant,
            "objective": objective,
            "init": init_name,
            "closure": len(history) + 1,
            "L_base": float(base.detach().cpu()),
            "L_base_bce": float(bce.detach().cpu()),
            "L_base_dice": float(dice.detach().cpu()),
            "normal_CVaR": float(cvar.detach().cpu()),
            "L_total": float(loss.detach().cpu()),
            "eta": float(eta.detach().cpu()) if eta is not None else float("nan"),
            "w_norm": float(w.detach().norm().cpu()),
        })
        return loss

    opt.step(closure)
    with torch.no_grad():
        w_np = w.detach().cpu().numpy().astype(np.float32)
        logits = (z @ w).detach().cpu().numpy().astype(np.float32)
    payload = {
        "variant": variant,
        "objective": objective,
        "init": init_name,
        "w": w_np,
        "eta": float(eta.detach().cpu()) if eta is not None else float("nan"),
        "history": history,
        "final_logits": logits,
    }
    torch.save(payload, ckpt)
    return payload


class DirectContrastModel(nn.Module):
    def __init__(self, w, align_payload):
        super().__init__()
        states_dummy = np.stack([-unit_np(w), unit_np(w)]).astype(np.float32)
        from experiments.phase_xliii_adapter_head_decomposition import AdapterToState

        base = AdapterToState(1024, states_dummy)
        state = base.state_dict()
        src = align_payload["model"]
        state["a.weight"] = src["a.weight"].detach().clone()
        state["b.weight"] = src["b.weight"].detach().clone()
        state["states"] = torch.tensor(states_dummy, dtype=torch.float32)
        base.load_state_dict(state, strict=True)
        self.base = base
        self.register_buffer("w", torch.tensor(np.asarray(w, dtype=np.float32)))

    def logits(self, x):
        z = self.base.adapt(x)
        return z @ self.w.to(z.device), z


def convergence_rows(payloads):
    rows = []
    for variant, plist in payloads.items():
        for p in plist:
            vals = objective_values_from_payload(p)
            rows.append({
                "variant": variant,
                "objective": p["objective"],
                "init": p["init"],
                "final_objective": vals["selected_objective"],
                "L_base": vals["L_base"],
                "normal_CVaR": vals["normal_CVaR"],
                "w_norm": float(np.linalg.norm(p["w"])),
                "eta": p["eta"],
                "closures": len(p["history"]),
            })
        for i, a in enumerate(plist):
            for b in plist[i + 1 :]:
                rows.append({
                    "variant": variant,
                    "objective": "pairwise",
                    "init": f"{a['init']}|{b['init']}",
                    "final_objective": float("nan"),
                    "L_base": float("nan"),
                    "normal_CVaR": float("nan"),
                    "w_norm": cos_np(a["w"], b["w"]),
                    "eta": corr_np(a["final_logits"], b["final_logits"]),
                    "closures": 0,
                })
    return rows


def objective_values_from_payload(payload):
    last = payload["history"][-1]
    selected = last["L_base"] if payload["objective"] == "base" else last["L_total"]
    return {**last, "selected_objective": selected}


def source_objective_rows(x, y, w_b0, payload_e1, payload_e2):
    rows = []
    for name, w, eta in (
        ("B0_EffectiveTwoState_w", w_b0, optimal_cvar_eta(x, y, w_b0)),
        ("E1_DirectContrastBaseOnly", payload_e1["w"], float("nan")),
        ("E2_ConvexTailRiskContrast", payload_e2["w"], payload_e2["eta"]),
    ):
        vals_base = objective_values(x, y, w, None)
        vals_ctrc = objective_values(x, y, w, optimal_cvar_eta(x, y, w) if not np.isfinite(eta) else eta)
        rows.append({
            "variant": name,
            "source_P-AUROC": vals_base["P-AUROC"],
            "source_P-AP": vals_base["P-AP"],
            "source_G": vals_base["G_pixel"],
            "L_base": vals_base["L_base"],
            "L_base_bce": vals_base["L_base_bce"],
            "L_base_dice": vals_base["L_base_dice"],
            "normal_CVaR": vals_ctrc["normal_CVaR"],
            "L_total_CTRC": vals_ctrc["L_total_CTRC"],
            "w_norm": vals_base["w_norm"],
            "eta": float(eta) if np.isfinite(eta) else float("nan"),
        })
    b0_obj = rows[0]["L_total_CTRC"]
    e2_obj = rows[2]["L_total_CTRC"]
    rows.append({
        "variant": "optimization_gap_B0_w_minus_E2",
        "source_P-AUROC": float("nan"),
        "source_P-AP": float("nan"),
        "source_G": float("nan"),
        "L_base": float("nan"),
        "L_base_bce": float("nan"),
        "L_base_dice": float("nan"),
        "normal_CVaR": float("nan"),
        "L_total_CTRC": b0_obj - e2_obj,
        "w_norm": float("nan"),
        "eta": float("nan"),
    })
    return rows


def load_b0_target_rows():
    raw, percat = load_b0_rows()
    for row in raw:
        row["variant"] = "B0_CurrentTwoStateHardNormal"
    for row in percat:
        row["variant"] = "B0_CurrentTwoStateHardNormal"
    return raw, percat


def summarize(raw, source_rows):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in TARGETS:
        b0 = by[(ds, "B0_CurrentTwoStateHardNormal")]
        e1 = by[(ds, "E1_DirectContrastBaseOnly")]
        e2 = by[(ds, "E2_ConvexTailRiskContrast")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_E1_G": float(e2["G_pixel"]) - float(e1["G_pixel"]),
            "E2_minus_B0_P-AUROC": float(e2["P-AUROC"]) - float(b0["P-AUROC"]),
            "E2_minus_B0_P-AP": float(e2["P-AP"]) - float(b0["P-AP"]),
        })
    src = {r["variant"]: r for r in source_rows}
    rows.append({
        "dataset": "source_objective",
        "B0_G": src["B0_EffectiveTwoState_w"]["source_G"],
        "E1_G": src["E1_DirectContrastBaseOnly"]["source_G"],
        "E2_G": src["E2_ConvexTailRiskContrast"]["source_G"],
        "E2_minus_B0_G": src["E2_ConvexTailRiskContrast"]["source_G"] - src["B0_EffectiveTwoState_w"]["source_G"],
        "E1_minus_B0_G": src["E1_DirectContrastBaseOnly"]["source_G"] - src["B0_EffectiveTwoState_w"]["source_G"],
        "E2_minus_E1_G": src["E2_ConvexTailRiskContrast"]["source_G"] - src["E1_DirectContrastBaseOnly"]["source_G"],
        "E2_minus_B0_P-AUROC": src["E2_ConvexTailRiskContrast"]["source_P-AUROC"] - src["B0_EffectiveTwoState_w"]["source_P-AUROC"],
        "E2_minus_B0_P-AP": src["E2_ConvexTailRiskContrast"]["source_P-AP"] - src["B0_EffectiveTwoState_w"]["source_P-AP"],
    })
    return rows


def contrast_geometry(w_b0, w_e1, w_e2):
    tau = 1.0
    return [{
        "metric": "norm_w_B0",
        "value": float(np.linalg.norm(w_b0)),
        "note": "effective two-state contrast tau*(normalize(sA)-normalize(sN)); tau=1 in current scorer",
    }, {
        "metric": "norm_w_E1",
        "value": float(np.linalg.norm(w_e1)),
        "note": "",
    }, {
        "metric": "norm_w_E2",
        "value": float(np.linalg.norm(w_e2)),
        "note": "inside_two_state_representable_contrast_ball" if float(np.linalg.norm(w_e2)) <= 2 * tau else "DIRECT_CONTRAST_EXCEEDS_TWO_STATE_NORM_BALL",
    }, {
        "metric": "cos_w_B0_w_E1",
        "value": cos_np(w_b0, w_e1),
        "note": "",
    }, {
        "metric": "cos_w_B0_w_E2",
        "value": cos_np(w_b0, w_e2),
        "note": "",
    }, {
        "metric": "cos_w_E1_w_E2",
        "value": cos_np(w_e1, w_e2),
        "note": "",
    }]


def decide(eq_rows, conv_rows, summary_rows, source_rows):
    if not bool(eq_rows[0]["passed"]):
        return "STATE_SCORER_NOT_EXACT_SINGLE_CONTRAST", {"max_abs_error": eq_rows[0]["max_abs_error"]}
    by = {r["dataset"]: r for r in summary_rows if r["dataset"] in TARGETS}
    e2_b0 = np.asarray([by[ds]["E2_minus_B0_G"] for ds in TARGETS], dtype=np.float32)
    e2_e1 = np.asarray([by[ds]["E2_minus_E1_G"] for ds in TARGETS], dtype=np.float32)
    src = {r["variant"]: r for r in source_rows}
    b0_obj = float(src["B0_EffectiveTwoState_w"]["L_total_CTRC"])
    e2_obj = float(src["E2_ConvexTailRiskContrast"]["L_total_CTRC"])
    rel_gap = (b0_obj - e2_obj) / max(abs(b0_obj), EPS)
    stats = {
        "mean_E2_minus_B0_G": float(e2_b0.mean()),
        "targets_E2_improved_over_B0": int(np.sum(e2_b0 > 0)),
        "MVTec_or_ClinicDB_best_E2_minus_B0_G": float(max(by["MVTec"]["E2_minus_B0_G"], by["CVC-ClinicDB"]["E2_minus_B0_G"])),
        "mean_E2_minus_E1_G": float(e2_e1.mean()),
        "relative_optimization_gap_B0_w_minus_E2": float(rel_gap),
        "optimization_gap_B0_w_minus_E2": float(b0_obj - e2_obj),
    }
    pair_rows = [r for r in conv_rows if r["objective"] == "pairwise" and r["variant"] == "E2_ConvexTailRiskContrast"]
    bad_corr = [r for r in pair_rows if not np.isnan(float(r["eta"])) and float(r["eta"]) < 0.999]
    final_objs = {}
    for r in conv_rows:
        if r["objective"] != "pairwise" and r["variant"] == "E2_ConvexTailRiskContrast":
            final_objs.setdefault(r["variant"], []).append(float(r["final_objective"]))
    bad_obj = any((max(v) - min(v)) / max(abs(min(v)), EPS) > 1e-4 for v in final_objs.values() if len(v) > 1)
    if bad_corr or bad_obj:
        stats.update({"bad_pairwise_logit_corr_count": len(bad_corr), "objective_spread_check_failed": bool(bad_obj)})
        return "DETERMINISTIC_CONVEX_SOLVER_NOT_CONVERGED", stats
    target_close = abs(float(e2_b0.mean())) < 0.5
    source_close = abs(rel_gap) < 0.01
    if int(np.sum(e2_b0 > 0)) >= 2 and float(e2_b0.mean()) >= 1.0 and stats["MVTec_or_ClinicDB_best_E2_minus_B0_G"] >= 1.5 and rel_gap >= 0.01:
        outcome = "CONVEX_TAIL_RISK_CONTRAST_IMPROVES_TRANSFER"
    elif rel_gap >= 0.05 and float(e2_b0.mean()) < -0.5:
        outcome = "NONCONVEX_STATE_IMPLICIT_BIAS_AIDS_TRANSFER"
    elif source_close and target_close:
        outcome = "TWO_STATE_PARAMETERIZATION_REDUNDANT"
    elif float(e2_e1.mean()) > 1.0 and float(e2_b0.mean()) >= -0.5:
        outcome = "NORMAL_CVAR_IS_ESSENTIAL_IN_DIRECT_CONTRAST"
    else:
        outcome = "CONVEX_TAIL_RISK_CONTRAST_NO_TRANSFER_GAIN"
    return outcome, stats


def write_report(outcome, stats, eq_rows, summary_rows, source_rows, geom_rows, runtime):
    lines = [
        "# Convex Tail-Risk Contrast Gate",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Stage 0 uses the exact current scorer algebra: `logit_A-logit_N = z dot (normalize(sA)-normalize(sN))`; current scorer has tau=1.",
        "",
        f"Algebraic max abs error: {eq_rows[0]['max_abs_error']:.8g}; mean abs error: {eq_rows[0]['mean_abs_error']:.8g}.",
        "",
        "| Dataset | B0 G | E1 BaseOnly G | E2 CTRC G | E2-B0 | E2-E1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        if r["dataset"] == "source_objective":
            continue
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E2_minus_E1_G']:+.4f} |")
    src = {r["variant"]: r for r in source_rows}
    gap = src["optimization_gap_B0_w_minus_E2"]["L_total_CTRC"]
    lines.extend([
        "",
        "## Source Objective",
        f"- B0 effective w total CTRC objective: {src['B0_EffectiveTwoState_w']['L_total_CTRC']:.6f}",
        f"- E2 final total CTRC objective: {src['E2_ConvexTailRiskContrast']['L_total_CTRC']:.6f}",
        f"- optimization gap B0_w - E2: {gap:+.6f}",
        "",
        "## Gate",
        f"- mean E2-B0 target G: {stats.get('mean_E2_minus_B0_G', float('nan')):+.4f}",
        f"- targets improved: {stats.get('targets_E2_improved_over_B0', 0)}/3",
        f"- relative optimization gap: {stats.get('relative_optimization_gap_B0_w_minus_E2', float('nan')):+.4%}",
        f"- norm(w_E2): {[r for r in geom_rows if r['metric']=='norm_w_E2'][0]['value']:.6f}",
        f"- capacity note: {[r for r in geom_rows if r['metric']=='norm_w_E2'][0]['note']}",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CTRC] device={device}", flush=True)
    q_pc1, corrupt_payload, states_c, states_hn, w_b0 = load_payloads(device)
    x, y, pool_rows = prepare_source(q_pc1, states_c, corrupt_payload, device, args.smoke)
    eq_rows = algebraic_equivalence(x, states_hn, w_b0)
    write_csv(OUT / "ALGEBRAIC_EQUIVALENCE.csv", eq_rows)
    if not bool(eq_rows[0]["passed"]):
        write_json(OUT / "OUTCOME.json", {"outcome": "STATE_SCORER_NOT_EXACT_SINGLE_CONTRAST", "algebraic_equivalence": eq_rows[0]})
        print("[CTRC] algebraic equivalence failed", flush=True)
        return

    init_names = ["I0_zero", "I1_w_B0", "I2_seed0_random_small", "I3_seed1_random_small"]
    payloads = {"E1_DirectContrastBaseOnly": [], "E2_ConvexTailRiskContrast": []}
    for init in init_names:
        w_init = init_vector(init, w_b0, x.shape[1])
        eta_init = initial_eta(x, y, w_init)
        payloads["E1_DirectContrastBaseOnly"].append(optimize_direct("E1_DirectContrastBaseOnly", "base", init, w_init, eta_init, x, y, args.smoke, device))
        payloads["E2_ConvexTailRiskContrast"].append(optimize_direct("E2_ConvexTailRiskContrast", "ctrc", init, w_init, eta_init, x, y, args.smoke, device))
    e1 = payloads["E1_DirectContrastBaseOnly"][0]
    e2 = payloads["E2_ConvexTailRiskContrast"][0]

    models = {
        "E1_DirectContrastBaseOnly": DirectContrastModel(e1["w"], corrupt_payload).to(device).eval(),
        "E2_ConvexTailRiskContrast": DirectContrastModel(e2["w"], corrupt_payload).to(device).eval(),
    }
    raw_new, per_new = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    raw_b0, per_b0 = load_b0_target_rows()
    if args.smoke:
        raw_b0 = raw_b0[:3]
        per_b0 = per_b0[:3]
    raw = raw_b0 + raw_new
    percat = per_b0 + per_new
    conv_rows = convergence_rows(payloads)
    source_rows = source_objective_rows(x, y, w_b0, e1, e2)
    summary_rows = summarize(raw, source_rows)
    geom_rows = contrast_geometry(w_b0, e1["w"], e2["w"])
    outcome, stats = decide(eq_rows, conv_rows, summary_rows, source_rows)
    runtime = time.time() - start

    write_csv(OUT / "MULTISTART_CONVERGENCE.csv", conv_rows)
    hist_rows = []
    for plist in payloads.values():
        for p in plist:
            hist_rows.extend(p["history"])
    write_csv(OUT / "OPTIMIZATION_TRACE.csv", hist_rows)
    write_csv(OUT / "SOURCE_OBJECTIVE.csv", source_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "AVERAGING_COMPARISON.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom_rows)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_rows)
    np.savez_compressed(OUT / "direct_contrasts.npz", w_B0=w_b0.astype(np.float32), w_E1=e1["w"].astype(np.float32), w_E2=e2["w"].astype(np.float32))
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "representation_frozen": True,
        "alignment_retrained": False,
        "optimizer": "torch.optim.LBFGS full fixed source pool closure",
        "max_iter": 20 if args.smoke else MAX_ITER,
        "alpha_cvar": ALPHA_CVAR,
        "multi_start_initializations": init_names,
        "base_loss_note": "L_base uses the same source labels and class-balancing weights as the formal StateAligned BCE term, but excludes Dice to preserve the requested direct convex BCE-only objective.",
    })
    write_report(outcome, stats, eq_rows, summary_rows, source_rows, geom_rows, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
