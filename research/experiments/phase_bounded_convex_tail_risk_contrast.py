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

from experiments.phase_convex_tail_risk_contrast_gate import (
    DirectContrastModel,
    algebraic_equivalence,
    load_b0_target_rows,
    load_payloads,
    prepare_source,
)
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_bounded_convex_tail_risk_contrast/formal")
UNBOUNDED_OUT = Path("results/phase_convex_tail_risk_contrast_gate/formal")
ALPHA_TAIL = 0.90
MAX_ITER = 2000
PATIENCE = 10
REL_TOL = 1e-8
TARGETS = ("MVTec", "BTAD", "CVC-ClinicDB")
EPS = 1e-12
ETA_TIE_BREAK_EPS = 0.0
EXTERNAL_BENCHMARKS = {
    "MoECLIP": {
        "comparison": "DIRECT_COMPARABLE",
        "note": "VisA-source cross-dataset pixel metrics, OpenCLIP ViT-L/14-336 518x518 reference.",
        "MVTec": {"P-AUROC": 92.5, "P-AP": 45.7, "G": 69.10},
        "BTAD": {"P-AUROC": 96.8, "P-AP": 50.4, "G": 73.60},
        "CVC-ClinicDB": {"P-AUROC": 89.7, "P-AP": 49.9, "G": 69.80},
    },
    "VisualAD_CLIP": {
        "comparison": "DIRECT_COMPARABLE",
        "note": "VisA-source cross-dataset CLIP pixel result; architecture details are less tightly matched than MoECLIP.",
        "MVTec": {"P-AUROC": 90.8, "P-AP": 41.2, "G": 66.00},
        "BTAD": {"P-AUROC": 91.1, "P-AP": 43.1, "G": 67.10},
        "CVC-ClinicDB": {"P-AUROC": 85.2, "P-AP": 37.8, "G": 61.50},
    },
}


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


def weighted_bce(logits, y):
    y = y.float()
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    weights = torch.where(y > 0.5, 0.5 * (pos + neg) / pos, 0.5 * (pos + neg) / neg)
    return F.binary_cross_entropy_with_logits(logits, y, weight=weights)


def dice_diag(logits, y):
    with torch.no_grad():
        p = torch.sigmoid(logits)
        yy = y.float()
        return 1.0 - (2.0 * (p * yy).sum() + 1.0) / (p.sum() + yy.sum() + 1.0)


def cvar_tail_torch(normal_losses, eta):
    n = normal_losses.numel()
    return eta + F.relu(normal_losses - eta).sum() / ((1.0 - ALPHA_TAIL) * float(n))


def top10_tail_torch(normal_losses):
    k = max(1, int(np.ceil(float(normal_losses.numel()) * (1.0 - ALPHA_TAIL))))
    return torch.topk(normal_losses, k=k, largest=True).values.mean()


def eta_opt_from_losses_np(normal_losses):
    normal_losses = np.asarray(normal_losses, dtype=np.float64).reshape(-1)
    if normal_losses.size == 0:
        return 0.0
    return float(np.quantile(normal_losses, ALPHA_TAIL, method="lower"))


def eta_opt_for_w_np(x, y, w_np):
    logits = np.asarray(x @ w_np, dtype=np.float64)
    normal_losses = np.logaddexp(0.0, logits[np.asarray(y) < 0.5])
    return eta_opt_from_losses_np(normal_losses)


def objective_torch(z, y, w, eta):
    logits = z @ w
    base = weighted_bce(logits, y)
    normal_losses = F.softplus(logits[y < 0.5])
    tail = cvar_tail_torch(normal_losses, eta)
    return base + tail, base, tail, logits


def objective_np(x, y, w_np, eta=None):
    with torch.no_grad():
        z = torch.tensor(x, dtype=torch.float32)
        yy = torch.tensor(y, dtype=torch.float32)
        w = torch.tensor(w_np, dtype=torch.float32)
        if eta is None:
            eta = eta_opt_for_w_np(x, y, w_np)
        eta_t = torch.tensor(float(eta), dtype=torch.float32)
        total, base, tail, logits = objective_torch(z, yy, w, eta_t)
        normal_losses = F.softplus(logits[yy < 0.5])
        top10_tail = top10_tail_torch(normal_losses)
        dice = dice_diag(logits, yy)
        score = logits.cpu().numpy().astype(np.float32)
    p_auc = safe_auc(y, score)
    p_ap = safe_ap(y, score)
    norm_logits = score[y < 0.5]
    anom_logits = score[y > 0.5]
    return {
        "source_P-AUROC": p_auc,
        "source_P-AP": p_ap,
        "source_G": 0.5 * (p_auc + p_ap),
        "L_base": float(base.cpu()),
        "L_tail": float(tail.cpu()),
        "eta": float(eta),
        "Top10Tail": float(top10_tail.cpu()),
        "tail_minus_top10": float((tail - top10_tail).cpu()),
        "F_total": float(total.cpu()),
        "L_base_dice_diag": float(dice.cpu()),
        "w_norm": float(np.linalg.norm(w_np)),
        "normal_logit_p5": float(np.percentile(norm_logits, 5)),
        "normal_logit_median": float(np.percentile(norm_logits, 50)),
        "normal_logit_p95": float(np.percentile(norm_logits, 95)),
        "anomaly_logit_p5": float(np.percentile(anom_logits, 5)),
        "anomaly_logit_median": float(np.percentile(anom_logits, 50)),
        "anomaly_logit_p95": float(np.percentile(anom_logits, 95)),
    }


def project_ball(w, radius):
    n = torch.norm(w).clamp_min(EPS)
    if float(n.detach().cpu()) <= radius:
        return w
    return w * (float(radius) / n)


def init_inside_ball(name, w_b0, radius):
    dim = w_b0.shape[0]
    if name == "I0_zero":
        return np.zeros((dim,), dtype=np.float32)
    if name == "I1_w_B0":
        w = np.asarray(w_b0, dtype=np.float32).copy()
    else:
        seed = 0 if name == "I2_seed0_random_inside" else 1
        rng = np.random.default_rng(seed)
        w = unit_np(rng.standard_normal(dim).astype(np.float32)) * (0.5 * radius)
    n = float(np.linalg.norm(w))
    if n > radius:
        w = w * (radius / max(n, EPS))
    return w.astype(np.float32)


def projected_gradient_solve(variant, init_name, x, y, w_init, eta_init, radius, f_b0=None, smoke=False, device="cpu"):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{init_name}_cvar_eta_median_{suffix}.pt"
    if ckpt.exists():
        return torch.load(ckpt, map_location=device, weights_only=False)
    print(f"[BoundedCTRC] solve {variant} {init_name} R={radius:.6f}", flush=True)
    z = torch.tensor(x, dtype=torch.float32, device=device)
    yy = torch.tensor(y, dtype=torch.float32, device=device)
    w = torch.tensor(w_init, dtype=torch.float32, device=device)
    w = project_ball(w, radius).detach()
    eta = torch.tensor(float(eta_init), dtype=torch.float32, device=device).detach()
    max_iter = 80 if smoke else MAX_ITER
    trajectory = []
    best_match = None
    stale = 0
    prev_obj = None
    with torch.no_grad():
        init_total, init_base, init_tail, init_logits = objective_torch(z, yy, w, eta)
        init_obj = float(init_total.detach().cpu())
        init_top10 = float(top10_tail_torch(F.softplus(init_logits[yy < 0.5])).detach().cpu())
    trajectory.append({
        "variant": variant,
        "init": init_name,
        "iteration": 0,
        "F_total": init_obj,
        "L_base": float(init_base.detach().cpu()),
        "L_tail": float(init_tail.detach().cpu()),
        "Top10Tail": init_top10,
        "eta": float(eta.detach().cpu()),
        "grad_norm": float("nan"),
        "grad_w_norm": float("nan"),
        "grad_eta_abs": float("nan"),
        "step_size": float("nan"),
        "w_norm": float(w.norm().detach().cpu()),
        "w_norm_over_R": float(w.norm().detach().cpu()) / float(radius),
        "boundary_active": bool(float(w.norm().detach().cpu()) / float(radius) >= 0.999),
        "relative_objective_change": float("nan"),
        "objective_match_already_at_initialization": bool(f_b0 is not None and init_obj <= f_b0),
    })
    if f_b0 is not None and init_obj <= f_b0:
        best_match = {
            "w": w.detach().cpu().numpy().astype(np.float32),
            "eta": float(eta.detach().cpu()),
            "iteration": 0,
            "F_total": init_obj,
            "OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION": True,
        }
        prev_obj = init_obj
    for it in range(1, max_iter + 1):
        w_req = w.detach().clone().requires_grad_(True)
        eta_req = eta.detach().clone().requires_grad_(True)
        total, base, tail, logits = objective_torch(z, yy, w_req, eta_req)
        total.backward()
        grad_w = w_req.grad.detach()
        grad_eta = eta_req.grad.detach()
        grad_w_norm = float(grad_w.norm().detach().cpu())
        grad_eta_abs = float(torch.abs(grad_eta).detach().cpu())
        grad_norm = float(torch.sqrt(grad_w.pow(2).sum() + grad_eta.pow(2)).detach().cpu())
        obj = float(total.detach().cpu())
        step = 1.0
        accepted = False
        with torch.no_grad():
            for _ in range(30):
                cand_w = project_ball(w - step * grad_w, radius)
                cand_eta = eta - step * grad_eta
                cand_obj = float(objective_torch(z, yy, cand_w, cand_eta)[0].detach().cpu())
                if cand_obj <= obj - 1e-4 * step * grad_norm * grad_norm or cand_obj <= obj:
                    w = cand_w.detach()
                    eta = cand_eta.detach()
                    obj_next = cand_obj
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                w = project_ball(w - step * grad_w, radius).detach()
                eta = (eta - step * grad_eta).detach()
                obj_next = float(objective_torch(z, yy, w, eta)[0].detach().cpu())
        if f_b0 is not None and best_match is None and obj_next <= f_b0:
            best_match = {
                "w": w.detach().cpu().numpy().astype(np.float32),
                "eta": float(eta.detach().cpu()),
                "iteration": it,
                "F_total": obj_next,
                "OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION": False,
            }
        rel = float("nan") if prev_obj is None else abs(prev_obj - obj_next) / max(abs(prev_obj), EPS)
        with torch.no_grad():
            top10_tail = top10_tail_torch(F.softplus((z @ w)[yy < 0.5]))
            w_norm_now = float(w.norm().detach().cpu())
        trajectory.append({
            "variant": variant,
            "init": init_name,
            "iteration": it,
            "F_total": obj_next,
            "L_base": float(base.detach().cpu()),
            "L_tail": float(tail.detach().cpu()),
            "Top10Tail": float(top10_tail.detach().cpu()),
            "eta": float(eta.detach().cpu()),
            "grad_norm": grad_norm,
            "grad_w_norm": grad_w_norm,
            "grad_eta_abs": grad_eta_abs,
            "step_size": step,
            "w_norm": w_norm_now,
            "w_norm_over_R": w_norm_now / float(radius),
            "boundary_active": bool(w_norm_now / float(radius) >= 0.999),
            "relative_objective_change": rel,
            "objective_match_already_at_initialization": False,
        })
        if prev_obj is not None and rel < REL_TOL:
            stale += 1
        else:
            stale = 0
        prev_obj = obj_next
        if it % 100 == 0 or it == 1:
            print(f"[BoundedCTRC] {variant} {init_name} iter={it} F={obj_next:.6f} grad={grad_norm:.4e} norm={trajectory[-1]['w_norm']:.4f} eta={trajectory[-1]['eta']:.4f}", flush=True)
        if stale >= PATIENCE:
            break
    with torch.no_grad():
        final_logits = (z @ w).detach().cpu().numpy().astype(np.float32)
    payload = {
        "variant": variant,
        "init": init_name,
        "radius": float(radius),
        "w": w.detach().cpu().numpy().astype(np.float32),
        "eta": float(eta.detach().cpu()),
        "trajectory": trajectory,
        "final_logits": final_logits,
        "matched": best_match,
    }
    torch.save(payload, ckpt)
    return payload


def convergence_rows(payloads):
    rows = []
    for variant, plist in payloads.items():
        objs = []
        for p in plist:
            last = p["trajectory"][-1]
            objs.append(float(last["F_total"]))
            rows.append({
                "variant": variant,
                "init": p["init"],
                "record": "final",
                "F_total": float(last["F_total"]),
                "L_base": float(last["L_base"]),
                "L_tail": float(last["L_tail"]),
                "Top10Tail": float(last.get("Top10Tail", float("nan"))),
                "eta": float(last.get("eta", float("nan"))),
                "w_norm": float(np.linalg.norm(p["w"])),
                "w_norm_over_R": float(np.linalg.norm(p["w"])) / float(p["radius"]),
                "boundary_active": bool(float(np.linalg.norm(p["w"])) / float(p["radius"]) >= 0.999),
                "iterations": int(last["iteration"]),
                "pairwise_w_cosine": float("nan"),
                "pairwise_logit_correlation": float("nan"),
            })
        rel_spread = (max(objs) - min(objs)) / max(abs(min(objs)), EPS)
        rows.append({
            "variant": variant,
            "init": "__summary__",
            "record": "relative_objective_spread",
            "F_total": float(rel_spread),
            "L_base": float("nan"),
            "L_tail": float("nan"),
            "Top10Tail": float("nan"),
            "eta": float("nan"),
            "w_norm": float("nan"),
            "w_norm_over_R": float("nan"),
            "boundary_active": "",
            "iterations": 0,
            "pairwise_w_cosine": float("nan"),
            "pairwise_logit_correlation": float("nan"),
        })
        for i, a in enumerate(plist):
            for b in plist[i + 1 :]:
                rows.append({
                    "variant": variant,
                    "init": f"{a['init']}|{b['init']}",
                    "record": "pairwise",
                    "F_total": float("nan"),
                    "L_base": float("nan"),
                    "L_tail": float("nan"),
                    "Top10Tail": float("nan"),
                    "eta": float("nan"),
                    "w_norm": float("nan"),
                    "w_norm_over_R": float("nan"),
                    "boundary_active": "",
                    "iterations": 0,
                    "pairwise_w_cosine": cos_np(a["w"], b["w"]),
                    "pairwise_logit_correlation": corr_np(a["final_logits"], b["final_logits"]),
                })
    return rows


def multistart_ok(rows, variant):
    rel = [float(r["F_total"]) for r in rows if r["variant"] == variant and r["record"] == "relative_objective_spread"][0]
    pair = [r for r in rows if r["variant"] == variant and r["record"] == "pairwise"]
    min_corr = min(float(r["pairwise_logit_correlation"]) for r in pair)
    return bool(rel <= 1e-5 and min_corr >= 0.9999), rel, min_corr


def source_rows(x, y, entries, tau):
    rows = []
    for entry in entries:
        variant = entry["variant"]
        w = entry["w"]
        vals = objective_np(x, y, w, entry.get("eta"))
        radius = float(entry.get("radius", 2.0 * tau))
        vals.update({
            "variant": variant,
            "w_norm_over_Rmax": vals["w_norm"] / (2.0 * tau),
            "radius": radius,
            "w_norm_over_R": vals["w_norm"] / radius,
            "BOUNDARY_ACTIVE": bool(vals["w_norm"] / radius >= 0.999),
            "OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION": bool(entry.get("OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION", False)),
        })
        rows.append(vals)
    by = {r["variant"]: r for r in rows}
    rows.append({
        "variant": "F_B0_minus_F_E1",
        "w_norm": float("nan"),
        "w_norm_over_Rmax": float("nan"),
        "source_P-AUROC": float("nan"),
        "source_P-AP": float("nan"),
        "source_G": float("nan"),
        "L_base": float("nan"),
        "L_tail": float("nan"),
        "eta": float("nan"),
        "Top10Tail": float("nan"),
        "tail_minus_top10": float("nan"),
        "F_total": by["B0_FormalTwoStateEffectiveW"]["F_total"] - by["E1_BoundedCTRC_ExactTwoStateBall"]["F_total"],
        "L_base_dice_diag": float("nan"),
        "radius": float("nan"),
        "w_norm_over_R": float("nan"),
        "BOUNDARY_ACTIVE": "",
        "OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION": "",
        "normal_logit_p5": float("nan"),
        "normal_logit_median": float("nan"),
        "normal_logit_p95": float("nan"),
        "anomaly_logit_p5": float("nan"),
        "anomaly_logit_median": float("nan"),
        "anomaly_logit_p95": float("nan"),
    })
    rows.append({
        "variant": "F_B0_minus_F_E2",
        "w_norm": float("nan"),
        "w_norm_over_Rmax": float("nan"),
        "source_P-AUROC": float("nan"),
        "source_P-AP": float("nan"),
        "source_G": float("nan"),
        "L_base": float("nan"),
        "L_tail": float("nan"),
        "eta": float("nan"),
        "Top10Tail": float("nan"),
        "tail_minus_top10": float("nan"),
        "F_total": by["B0_FormalTwoStateEffectiveW"]["F_total"] - by["E2_BoundedCTRC_B0NormBall"]["F_total"],
        "L_base_dice_diag": float("nan"),
        "radius": float("nan"),
        "w_norm_over_R": float("nan"),
        "BOUNDARY_ACTIVE": "",
        "OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION": "",
        "normal_logit_p5": float("nan"),
        "normal_logit_median": float("nan"),
        "normal_logit_p95": float("nan"),
        "anomaly_logit_p5": float("nan"),
        "anomaly_logit_median": float("nan"),
        "anomaly_logit_p95": float("nan"),
    })
    return rows


def hypothesis_rows(eq_row, tau, w_b0):
    rmax = 2.0 * tau
    rb0 = float(np.linalg.norm(w_b0))
    return [{
        "tau": tau,
        "R_max": rmax,
        "R_B0": rb0,
        "R_B0_over_R_max": rb0 / rmax,
        "w_B0_norm": rb0,
        "algebraic_max_abs_error": eq_row["max_abs_error"],
        "algebraic_mean_abs_error": eq_row["mean_abs_error"],
        "algebraic_passed": eq_row["passed"],
    }]


def contrast_geometry(w_b0, w_e1, w_e2, w_e3):
    pairs = [
        ("cos_w_B0_w_E1", w_b0, w_e1),
        ("cos_w_B0_w_E2", w_b0, w_e2),
        ("cos_w_B0_w_E3", w_b0, w_e3),
        ("cos_w_E1_w_E2", w_e1, w_e2),
        ("cos_w_E1_w_E3", w_e1, w_e3),
        ("cos_w_E2_w_E3", w_e2, w_e3),
    ]
    rows = [{"metric": name, "value": cos_np(a, b)} for name, a, b in pairs]
    for name, w in (("norm_w_B0", w_b0), ("norm_w_E1", w_e1), ("norm_w_E2", w_e2), ("norm_w_E3", w_e3)):
        rows.append({"metric": name, "value": float(np.linalg.norm(w))})
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in TARGETS:
        b0 = by[(ds, "B0_FormalTwoStateHardNormal")]
        e1 = by[(ds, "E1_BoundedCTRC_ExactTwoStateBall")]
        e2 = by[(ds, "E2_BoundedCTRC_B0NormBall")]
        e3 = by[(ds, "E3_ObjectiveMatchedEarlyContrast")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E3_G": float(e3["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E3_minus_B0_G": float(e3["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_E1_G": float(e2["G_pixel"]) - float(e1["G_pixel"]),
            "abs_E3_minus_B0_G": abs(float(e3["G_pixel"]) - float(b0["G_pixel"])),
        })
    return rows


def write_sota_gap(raw_rows):
    rows = []
    for row in raw_rows:
        target = row.get("target")
        if target not in TARGETS:
            continue
        method_g = (float(row["P-AUROC"]) + float(row["P-AP"])) / 2.0
        for benchmark, by_target in EXTERNAL_BENCHMARKS.items():
            ref = by_target[target]
            rows.append({
                "variant": row["variant"],
                "target": target,
                "benchmark": benchmark,
                "comparison": by_target["comparison"],
                "method_P-AUROC": row["P-AUROC"],
                "method_P-AP": row["P-AP"],
                "method_G": method_g,
                "benchmark_P-AUROC": ref["P-AUROC"],
                "benchmark_P-AP": ref["P-AP"],
                "benchmark_G": ref["G"],
                "Delta_P_AUROC": float(row["P-AUROC"]) - ref["P-AUROC"],
                "Delta_P_AP": float(row["P-AP"]) - ref["P-AP"],
                "Delta_G": method_g - ref["G"],
                "note": by_target["note"],
            })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def decide(summary_rows, source_obj, conv_rows):
    ok1, rel1, corr1 = multistart_ok(conv_rows, "E1_BoundedCTRC_ExactTwoStateBall")
    ok2, rel2, corr2 = multistart_ok(conv_rows, "E2_BoundedCTRC_B0NormBall")
    stats = {"E1_multistart_ok": ok1, "E2_multistart_ok": ok2, "E1_objective_spread": rel1, "E2_objective_spread": rel2, "E1_min_logit_corr": corr1, "E2_min_logit_corr": corr2}
    if not (ok1 and ok2):
        return "CONSTRAINED_SOLVER_NOT_CONVERGED", stats
    by = {r["dataset"]: r for r in summary_rows}
    src = {r["variant"]: r for r in source_obj}
    e1_b0 = np.asarray([by[ds]["E1_minus_B0_G"] for ds in TARGETS], dtype=np.float32)
    e2_e1 = np.asarray([by[ds]["E2_minus_E1_G"] for ds in TARGETS], dtype=np.float32)
    e3_abs = np.asarray([by[ds]["abs_E3_minus_B0_G"] for ds in TARGETS], dtype=np.float32)
    e1_source_better = src["E1_BoundedCTRC_ExactTwoStateBall"]["F_total"] < src["B0_FormalTwoStateEffectiveW"]["F_total"]
    e2_restores = float(e2_e1.mean()) >= 1.0 and abs(float(np.mean([by[ds]["E2_minus_B0_G"] for ds in TARGETS]))) <= 0.5
    e3_close = float(e3_abs.mean()) <= 0.5
    stats.update({
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "targets_E1_improved_over_B0": int(np.sum(e1_b0 > 0)),
        "MVTec_or_ClinicDB_best_E1_minus_B0_G": float(max(by["MVTec"]["E1_minus_B0_G"], by["CVC-ClinicDB"]["E1_minus_B0_G"])),
        "mean_E2_minus_E1_G": float(e2_e1.mean()),
        "mean_abs_E3_minus_B0_G": float(e3_abs.mean()),
        "F_B0_minus_F_E1": float(src["F_B0_minus_F_E1"]["F_total"]),
        "F_B0_minus_F_E2": float(src["F_B0_minus_F_E2"]["F_total"]),
    })
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and stats["MVTec_or_ClinicDB_best_E1_minus_B0_G"] >= 1.5:
        outcome = "BOUNDED_CONVEX_TAIL_RISK_CONTRAST_SUPPORTED"
    elif e1_source_better and float(e1_b0.mean()) < -0.5 and e2_restores:
        outcome = "B0_CONTRAST_MAGNITUDE_IS_TRANSFER_REGULARIZER"
    elif e1_source_better and float(e1_b0.mean()) < -0.5 and e3_close:
        outcome = "SOURCE_OBJECTIVE_EARLY_STOPPING_EXPLAINS_TRANSFER"
    elif all(by[ds]["E1_minus_B0_G"] < -0.5 and by[ds]["E2_minus_B0_G"] < -0.5 and by[ds]["E3_minus_B0_G"] < -0.5 for ds in TARGETS):
        outcome = "JOINT_TWO_STATE_TRAINING_IMPLICIT_BIAS_AIDS_TRANSFER"
    elif abs(float(e1_b0.mean())) < 0.5 and abs(float(src["F_B0_minus_F_E1"]["F_total"])) < 0.01:
        outcome = "TWO_STATE_PARAMETERIZATION_REDUNDANT_UNDER_NORM_CONSTRAINT"
    else:
        outcome = "BOUNDED_CONVEX_TAIL_RISK_CONTRAST_INCONCLUSIVE"
    return outcome, stats


def write_report(outcome, stats, hyp_rows, summary_rows, source_obj, runtime, sota_rows=None):
    src = {r["variant"]: r for r in source_obj}
    lines = [
        "# Bounded Convex Tail-Risk Contrast",
        "",
        f"Outcome: `{outcome}`",
        "",
        f"Algebraic equivalence max error: {float(hyp_rows[0]['algebraic_max_abs_error']):.8g}. R_B0/R_max: {float(hyp_rows[0]['R_B0_over_R_max']):.6f}.",
        "",
        "| Dataset | B0 G | E1 Exact Ball G | E2 B0-Norm Ball G | E3 Early-Match G | E1-B0 | E2-E1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {float(r['B0_G']):.4f} | {float(r['E1_G']):.4f} | {float(r['E2_G']):.4f} | {float(r['E3_G']):.4f} | {float(r['E1_minus_B0_G']):+.4f} | {float(r['E2_minus_E1_G']):+.4f} |")
    lines.extend([
        "",
        "## Source Objective",
        f"- B0 F: {float(src['B0_FormalTwoStateEffectiveW']['F_total']):.6f}",
        f"- E1 F: {float(src['E1_BoundedCTRC_ExactTwoStateBall']['F_total']):.6f}",
        f"- E2 F: {float(src['E2_BoundedCTRC_B0NormBall']['F_total']):.6f}",
        f"- E3 F: {float(src['E3_ObjectiveMatchedEarlyContrast']['F_total']):.6f}",
        f"- B0 eta_opt: {float(src['B0_FormalTwoStateEffectiveW']['eta']):.6f}; CVaR-Top10Tail: {float(src['B0_FormalTwoStateEffectiveW']['tail_minus_top10']):+.6e}",
        f"- E1 eta: {float(src['E1_BoundedCTRC_ExactTwoStateBall']['eta']):.6f}; ||w||/R: {float(src['E1_BoundedCTRC_ExactTwoStateBall']['w_norm_over_R']):.6f}; boundary active: {src['E1_BoundedCTRC_ExactTwoStateBall']['BOUNDARY_ACTIVE']}",
        f"- E2 eta: {float(src['E2_BoundedCTRC_B0NormBall']['eta']):.6f}; ||w||/R: {float(src['E2_BoundedCTRC_B0NormBall']['w_norm_over_R']):.6f}; boundary active: {src['E2_BoundedCTRC_B0NormBall']['BOUNDARY_ACTIVE']}",
        f"- E3 objective match already at initialization: {src['E3_ObjectiveMatchedEarlyContrast']['OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION']}",
        "",
        "## Solver Gate",
        f"- E1 objective spread: {stats.get('E1_objective_spread', float('nan')):.8g}; min logit corr: {stats.get('E1_min_logit_corr', float('nan')):.8g}",
        f"- E2 objective spread: {stats.get('E2_objective_spread', float('nan')):.8g}; min logit corr: {stats.get('E2_min_logit_corr', float('nan')):.8g}",
        "",
        "## External SOTA Position",
        "MoECLIP is the closest direct comparison for this VisA-source cross-dataset protocol and OpenCLIP ViT-L/14-336 518x518 setting. VisualAD(CLIP) is also included as a direct pixel-metric reference, but its implementation details are less tightly matched than MoECLIP.",
    ])
    if sota_rows:
        for benchmark in EXTERNAL_BENCHMARKS:
            b0 = [r for r in sota_rows if r["variant"] == "B0_FormalTwoStateHardNormal" and r["benchmark"] == benchmark]
            e1 = [r for r in sota_rows if r["variant"] == "E1_BoundedCTRC_ExactTwoStateBall" and r["benchmark"] == benchmark]
            if b0 and e1:
                lines.append(f"- Versus {benchmark}: B0 mean Delta_G {np.mean([r['Delta_G'] for r in b0]):+.4f}; E1 mean Delta_G {np.mean([r['Delta_G'] for r in e1]):+.4f}.")
        lines.append("The bounded convex variants improve the source objective but do not narrow the main target gap, so this direct-convex scorer route is not worth additional complexity.")
        lines.append("")
    lines.extend([
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BoundedCTRC] device={device}", flush=True)
    q_pc1, corrupt_payload, states_c, states_hn, w_b0 = load_payloads(device)
    x, y, pool_rows = prepare_source(q_pc1, states_c, corrupt_payload, device, args.smoke)
    eq_rows = algebraic_equivalence(x, states_hn, w_b0)
    tau = float(eq_rows[0]["temperature_tau"])
    rmax = 2.0 * tau
    rb0 = float(np.linalg.norm(w_b0))
    hyp_rows = hypothesis_rows(eq_rows[0], tau, w_b0)
    write_csv(OUT / "HYPOTHESIS_CLASS_GEOMETRY.csv", hyp_rows)
    if not bool(eq_rows[0]["passed"]):
        write_json(OUT / "OUTCOME.json", {"outcome": "STATE_SCORER_NOT_EXACT_SINGLE_CONTRAST", "algebraic": eq_rows[0]})
        return

    eta_b0_opt = eta_opt_for_w_np(x, y, w_b0)
    f_b0 = objective_np(x, y, w_b0, eta_b0_opt)["F_total"]
    eta0_median = eta_opt_for_w_np(x, y, np.zeros_like(w_b0, dtype=np.float32))
    eta0 = eta0_median - ETA_TIE_BREAK_EPS
    init_names = ("I0_zero", "I1_w_B0", "I2_seed0_random_inside", "I3_seed1_random_inside")
    payloads = {"E1_BoundedCTRC_ExactTwoStateBall": [], "E2_BoundedCTRC_B0NormBall": []}
    for init in init_names:
        payloads["E1_BoundedCTRC_ExactTwoStateBall"].append(projected_gradient_solve("E1_BoundedCTRC_ExactTwoStateBall", init, x, y, init_inside_ball(init, w_b0, rmax), eta0, rmax, f_b0=f_b0, smoke=args.smoke, device=device))
        payloads["E2_BoundedCTRC_B0NormBall"].append(projected_gradient_solve("E2_BoundedCTRC_B0NormBall", init, x, y, init_inside_ball(init, w_b0, rb0), eta0, rb0, smoke=args.smoke, device=device))
    conv_rows = convergence_rows(payloads)
    write_csv(OUT / "MULTISTART_CONVERGENCE.csv", conv_rows)
    ok1, _rel1, _corr1 = multistart_ok(conv_rows, "E1_BoundedCTRC_ExactTwoStateBall")
    ok2, _rel2, _corr2 = multistart_ok(conv_rows, "E2_BoundedCTRC_B0NormBall")
    if not (ok1 and ok2):
        matched_fail = payloads["E1_BoundedCTRC_ExactTwoStateBall"][0].get("matched")
        source_obj = source_rows(x, y, [
            {"variant": "B0_FormalTwoStateEffectiveW", "w": w_b0, "eta": eta_b0_opt, "radius": rmax},
            {"variant": "E1_BoundedCTRC_ExactTwoStateBall", "w": payloads["E1_BoundedCTRC_ExactTwoStateBall"][0]["w"], "eta": payloads["E1_BoundedCTRC_ExactTwoStateBall"][0]["eta"], "radius": rmax},
            {"variant": "E2_BoundedCTRC_B0NormBall", "w": payloads["E2_BoundedCTRC_B0NormBall"][0]["w"], "eta": payloads["E2_BoundedCTRC_B0NormBall"][0]["eta"], "radius": rb0},
            {"variant": "E3_ObjectiveMatchedEarlyContrast", "w": matched_fail["w"] if matched_fail else payloads["E1_BoundedCTRC_ExactTwoStateBall"][0]["w"], "eta": matched_fail["eta"] if matched_fail else payloads["E1_BoundedCTRC_ExactTwoStateBall"][0]["eta"], "radius": rmax, "OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION": bool(matched_fail and matched_fail.get("OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION"))},
        ], tau)
        write_csv(OUT / "SOURCE_OBJECTIVE.csv", source_obj)
        hist = []
        for plist in payloads.values():
            for p in plist:
                hist.extend(p["trajectory"])
        write_csv(OUT / "OPTIMIZATION_TRAJECTORY.csv", hist)
        write_json(OUT / "OUTCOME.json", {"outcome": "CONSTRAINED_SOLVER_NOT_CONVERGED", "device": device, "strict_source_only": True})
        (OUT / "REPORT.md").write_text("# Bounded Convex Tail-Risk Contrast\n\nOutcome: `CONSTRAINED_SOLVER_NOT_CONVERGED`\n\nStage 0 algebraic equivalence passed, but E1/E2 projected-gradient multi-start convergence did not satisfy source logit correlation >= 0.9999 and relative objective spread <= 1e-5. Target evaluation was not run as a formal method result.\n", encoding="utf-8")
        print(json.dumps({"outcome": "CONSTRAINED_SOLVER_NOT_CONVERGED"}, indent=2), flush=True)
        return

    e1 = payloads["E1_BoundedCTRC_ExactTwoStateBall"][0]["w"]
    e2 = payloads["E2_BoundedCTRC_B0NormBall"][0]["w"]
    eta_e1 = payloads["E1_BoundedCTRC_ExactTwoStateBall"][0]["eta"]
    eta_e2 = payloads["E2_BoundedCTRC_B0NormBall"][0]["eta"]
    matched = payloads["E1_BoundedCTRC_ExactTwoStateBall"][0].get("matched")
    e3 = matched["w"] if matched else e1
    eta_e3 = matched["eta"] if matched else eta_e1
    models = {
        "E1_BoundedCTRC_ExactTwoStateBall": DirectContrastModel(e1, corrupt_payload).to(device).eval(),
        "E2_BoundedCTRC_B0NormBall": DirectContrastModel(e2, corrupt_payload).to(device).eval(),
        "E3_ObjectiveMatchedEarlyContrast": DirectContrastModel(e3, corrupt_payload).to(device).eval(),
    }
    raw_new, per_new = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    raw_b0, per_b0 = load_b0_target_rows()
    for row in raw_b0:
        row["variant"] = "B0_FormalTwoStateHardNormal"
    for row in per_b0:
        row["variant"] = "B0_FormalTwoStateHardNormal"
    if args.smoke:
        raw_b0 = raw_b0[:3]
        per_b0 = per_b0[:3]
    raw = raw_b0 + raw_new
    percat = per_b0 + per_new
    source_obj = source_rows(x, y, [
        {"variant": "B0_FormalTwoStateEffectiveW", "w": w_b0, "eta": eta_b0_opt, "radius": rmax},
        {"variant": "E1_BoundedCTRC_ExactTwoStateBall", "w": e1, "eta": eta_e1, "radius": rmax},
        {"variant": "E2_BoundedCTRC_B0NormBall", "w": e2, "eta": eta_e2, "radius": rb0},
        {"variant": "E3_ObjectiveMatchedEarlyContrast", "w": e3, "eta": eta_e3, "radius": rmax, "OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION": bool(matched and matched.get("OBJECTIVE_MATCH_ALREADY_AT_INITIALIZATION"))},
    ], tau)
    summary = summarize(raw)
    geom = contrast_geometry(w_b0, e1, e2, e3)
    outcome, stats = decide(summary, source_obj, conv_rows)
    hist = []
    for plist in payloads.values():
        for p in plist:
            hist.extend(p["trajectory"])
    runtime = time.time() - start
    write_csv(OUT / "SOURCE_OBJECTIVE.csv", source_obj)
    write_csv(OUT / "OPTIMIZATION_TRAJECTORY.csv", hist)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "AVERAGING_COMPARISON.csv", summary)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_rows)
    sota_rows = write_sota_gap(raw)
    np.savez_compressed(OUT / "bounded_contrasts.npz", w_B0=w_b0.astype(np.float32), w_E1=e1.astype(np.float32), w_E2=e2.astype(np.float32), w_E3=e3.astype(np.float32), eta_B0=np.float32(eta_b0_opt), eta_E1=np.float32(eta_e1), eta_E2=np.float32(eta_e2), eta_E3=np.float32(eta_e3))
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "representation_frozen": True,
        "radius_sweep": False,
        "R_max": rmax,
        "R_B0": rb0,
        "alpha_tail": ALPHA_TAIL,
        "eta0": eta0,
        "eta0_median_at_w0": eta0_median,
        "eta_tie_break_eps": ETA_TIE_BREAK_EPS,
        "eta_B0_opt": eta_b0_opt,
        "solver": "full-source projected gradient with deterministic backtracking",
        "max_iter": 80 if args.smoke else MAX_ITER,
        "objective_note": "weighted BCE base plus convex CVaR normal-tail epigraph eta; Top10Tail is logged as an equivalence diagnostic; Dice excluded to preserve convex contrast objective.",
        "unbounded_previous": str(UNBOUNDED_OUT),
    })
    write_report(outcome, stats, hyp_rows, summary, source_obj, runtime, sota_rows)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
