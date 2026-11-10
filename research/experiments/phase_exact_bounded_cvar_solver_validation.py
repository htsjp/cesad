import argparse
import csv
import json
import math
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_bounded_convex_tail_risk_contrast import cos_np, corr_np, safe_ap, safe_auc, weighted_bce
from experiments.phase_convex_tail_risk_contrast_gate import algebraic_equivalence, load_payloads, prepare_source
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_xlii_patch_specialized_adaptation import SEED


OUT = Path("results/phase_exact_bounded_cvar_solver_validation/formal")
ALPHA = 0.90
MAX_ITER = 5000
EPS = 1e-12
INIT_NAMES = ("I0_zero", "I1_w_B0", "I2_seed0_random_inside", "I3_seed1_random_inside")


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


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = sorted({k for row in rows for k in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_np(w, radius):
    n = float(np.linalg.norm(w))
    if n <= radius:
        return w.astype(np.float32)
    return (w * (radius / max(n, EPS))).astype(np.float32)


def init_inside_ball(name, w_b0, radius):
    dim = int(w_b0.shape[0])
    if name == "I0_zero":
        return np.zeros(dim, dtype=np.float32)
    if name == "I1_w_B0":
        return project_np(np.asarray(w_b0, dtype=np.float32).copy(), radius)
    seed = 0 if name == "I2_seed0_random_inside" else 1
    rng = np.random.default_rng(seed)
    direction = unit_np(rng.standard_normal(dim).astype(np.float32))
    radial = float(radius) * 0.5
    return (radial * direction).astype(np.float32)


def canonical_cvar_weights_np(losses, alpha=ALPHA):
    losses = np.asarray(losses, dtype=np.float64).reshape(-1)
    n = int(losses.size)
    if n == 0:
        raise ValueError("Cannot compute CVaR weights for an empty normal-loss set.")
    m = (1.0 - float(alpha)) * float(n)
    cap = 1.0 / m
    order = np.argsort(-losses, kind="mergesort")
    q = np.zeros(n, dtype=np.float64)
    remaining = 1.0
    start = 0
    while start < n and remaining > 1e-15:
        value = losses[order[start]]
        end = start + 1
        while end < n and losses[order[end]] == value:
            end += 1
        idx = order[start:end]
        group_cap = cap * float(idx.size)
        if remaining >= group_cap - 1e-14:
            q[idx] = cap
            remaining -= group_cap
        else:
            q[idx] = remaining / float(idx.size)
            remaining = 0.0
        start = end
    q[np.abs(q) < 1e-18] = 0.0
    return q, {"n_normal": n, "m_tail_mass": m, "capacity": cap, "sum_q": float(q.sum()), "min_q": float(q.min()), "max_q": float(q.max())}


def objective_and_grad(z, y, w_np):
    w = torch.tensor(w_np, dtype=torch.float32, device=z.device, requires_grad=True)
    logits = z @ w
    base = weighted_bce(logits, y)
    normal_losses = F.softplus(logits[y < 0.5])
    q_np, _meta = canonical_cvar_weights_np(normal_losses.detach().cpu().numpy())
    q = torch.tensor(q_np, dtype=torch.float32, device=z.device)
    tail = (q * normal_losses).sum()
    total = base + tail
    total.backward()
    grad = w.grad.detach().cpu().numpy().astype(np.float32)
    return {
        "F": float(total.detach().cpu()),
        "L_base": float(base.detach().cpu()),
        "L_tail": float(tail.detach().cpu()),
        "grad": grad,
        "grad_norm": float(np.linalg.norm(grad)),
        "logits": logits.detach().cpu().numpy().astype(np.float32),
        "q": q_np,
        "normal_losses": normal_losses.detach().cpu().numpy().astype(np.float64),
    }


def objective_only_np(x, y, w_np):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.tensor(x, dtype=torch.float32, device=device)
    yy = torch.tensor(y, dtype=torch.float32, device=device)
    with torch.no_grad():
        w = torch.tensor(w_np, dtype=torch.float32, device=device)
        logits = z @ w
        base = weighted_bce(logits, yy)
        normal_losses = F.softplus(logits[yy < 0.5]).detach().cpu().numpy().astype(np.float64)
        q, meta = canonical_cvar_weights_np(normal_losses)
        tail = float(np.dot(q, normal_losses))
        score = logits.detach().cpu().numpy().astype(np.float32)
    p_auc = safe_auc(y, score)
    p_ap = safe_ap(y, score)
    k = max(1, int(math.ceil(float(normal_losses.size) * (1.0 - ALPHA))))
    top_tail = float(np.sort(normal_losses)[-k:].mean())
    return {
        "F": float(base.detach().cpu()) + tail,
        "L_base": float(base.detach().cpu()),
        "L_tail": tail,
        "Top10Tail": top_tail,
        "tail_minus_top10": tail - top_tail,
        "source_P-AUROC": p_auc,
        "source_P-AP": p_ap,
        "source_G": 0.5 * (p_auc + p_ap),
        "w_norm": float(np.linalg.norm(w_np)),
        **meta,
    }


def solve_projected(variant, init_name, x, y, w_init, radius, smoke, device):
    seed_all(SEED if init_name in ("I0_zero", "I1_w_B0") else (0 if "seed0" in init_name else 1))
    max_iter = 80 if smoke else MAX_ITER
    z = torch.tensor(x, dtype=torch.float32, device=device)
    yy = torch.tensor(y, dtype=torch.float32, device=device)
    w = project_np(w_init.astype(np.float32), radius)
    first = objective_and_grad(z, yy, w)
    step0 = float(radius) / (first["grad_norm"] + EPS)
    step_sum = 0.0
    w_avg_acc = np.zeros_like(w, dtype=np.float64)
    best = {"w": w.copy(), "F": first["F"], "iteration": 0, "kind": "best"}
    last = {"w": w.copy(), "F": first["F"], "iteration": 0, "kind": "last"}
    trajectory = []
    prev_f = None
    stable = 0
    for it in range(max_iter + 1):
        info = first if it == 0 else objective_and_grad(z, yy, w)
        fval = info["F"]
        grad = info["grad"]
        gnorm = info["grad_norm"]
        if fval < best["F"]:
            best = {"w": w.copy(), "F": fval, "iteration": it, "kind": "best"}
        rel = float("nan") if prev_f is None else abs(prev_f - fval) / max(abs(prev_f), EPS)
        if prev_f is not None and rel < 1e-8:
            stable += 1
        else:
            stable = 0
        if it == 0 or it % 25 == 0 or it == max_iter:
            trajectory.append({
                "variant": variant,
                "init": init_name,
                "iteration": it,
                "F": fval,
                "L_base": info["L_base"],
                "L_tail": info["L_tail"],
                "grad_norm": gnorm,
                "step": float("nan") if it == max_iter else step0 / math.sqrt(it + 1.0),
                "w_norm": float(np.linalg.norm(w)),
                "w_norm_over_R": float(np.linalg.norm(w)) / float(radius),
                "boundary_active": bool(float(np.linalg.norm(w)) / float(radius) >= 0.999),
                "relative_objective_change": rel,
            })
        last = {"w": w.copy(), "F": fval, "iteration": it, "kind": "last"}
        if stable >= 10:
            break
        step = step0 / math.sqrt(it + 1.0)
        step_sum += step
        w_avg_acc += float(step) * w.astype(np.float64)
        prev_f = fval
        w = project_np(w - step * grad, radius)
    if step_sum > 0.0:
        w_avg = (w_avg_acc / step_sum).astype(np.float32)
    else:
        w_avg = last["w"].copy()
    avg_vals = objective_only_np(x, y, w_avg)
    candidates = [
        {**last, "kind": "last"},
        {**best, "kind": "best"},
        {"w": w_avg, "F": avg_vals["F"], "iteration": -1, "kind": "averaged"},
    ]
    chosen = min(candidates, key=lambda row: row["F"])
    final_vals = objective_only_np(x, y, chosen["w"])
    return {
        "variant": variant,
        "init": init_name,
        "radius": float(radius),
        "step0": step0,
        "iterations_run": int(last["iteration"]),
        "trajectory": trajectory,
        "candidate_rows": [
            {
                "variant": variant,
                "init": init_name,
                "candidate": cand["kind"],
                "iteration": cand["iteration"],
                **objective_only_np(x, y, cand["w"]),
            }
            for cand in candidates
        ],
        "chosen_kind": chosen["kind"],
        "chosen_iteration": int(chosen["iteration"]),
        "w": chosen["w"].astype(np.float32),
        "source": final_vals,
    }


def kkt_residual(x, y, w_np, radius, device):
    z = torch.tensor(x, dtype=torch.float32, device=device)
    yy = torch.tensor(y, dtype=torch.float32, device=device)
    info = objective_and_grad(z, yy, w_np)
    proj = project_np(w_np - info["grad"], radius)
    res = float(np.linalg.norm(w_np - proj) / max(1.0, float(np.linalg.norm(w_np))))
    return res, info


def cvar_dual_checks(losses):
    q1, meta1 = canonical_cvar_weights_np(losses)
    q2, meta2 = canonical_cvar_weights_np(losses)
    positive = q1 > 1e-12
    cap = meta1["capacity"]
    fractional = positive & (q1 < cap - 1e-10)
    zero = q1 <= 1e-12
    min_pos_loss = float(np.min(losses[positive])) if np.any(positive) else float("nan")
    max_zero_loss = float(np.max(losses[zero])) if np.any(zero) else float("-inf")
    return {
        **meta1,
        "sum_q_abs_error": abs(meta1["sum_q"] - 1.0),
        "bounds_ok": bool(meta1["min_q"] >= -1e-10 and meta1["max_q"] <= cap + 1e-10),
        "positive_losses_dominate_zero": bool(min_pos_loss + 1e-10 >= max_zero_loss),
        "fractional_tie_count": int(fractional.sum()),
        "bitwise_recompute_equal": bool(np.array_equal(q1, q2)),
        "meta_recompute_equal": bool(meta1 == meta2),
    }


def pairwise_logit_stats(x, solutions):
    rows = []
    for (a_name, a), (b_name, b) in combinations(solutions.items(), 2):
        la = x @ a["w"]
        lb = x @ b["w"]
        dw = a["w"] - b["w"]
        rows.append({
            "pair": f"{a_name}__{b_name}",
            "source_logit_pearson": corr_np(la, lb),
            "w_cosine": cos_np(a["w"], b["w"]),
            "delta_w_norm": float(np.linalg.norm(dw)),
            "Z_delta_over_delta_w": float(np.linalg.norm(x @ dw) / max(float(np.linalg.norm(dw)), EPS)),
        })
    return rows


def convergence_summary(variant, rows, pair_rows, kkt_rows):
    vals = [float(r["F"]) for r in rows if r["variant"] == variant]
    rel_spread = (max(vals) - min(vals)) / max(abs(float(np.mean(vals))), EPS)
    pairs = [r for r in pair_rows if r["variant"] == variant]
    kkts = [r for r in kkt_rows if r["variant"] == variant]
    min_corr = min(float(r["source_logit_pearson"]) for r in pairs)
    max_kkt = max(float(r["projected_residual"]) for r in kkts)
    ok = bool(rel_spread <= 1e-4 and min_corr >= 0.9999 and max_kkt <= 1e-4)
    nonunique = bool(rel_spread <= 1e-4 and max_kkt <= 1e-4 and min_corr < 0.9999)
    return {
        "variant": variant,
        "relative_objective_spread": rel_spread,
        "min_source_logit_pearson": min_corr,
        "max_kkt_projected_residual": max_kkt,
        "multistart_converged": ok,
        "prediction_null_nonuniqueness_candidate": nonunique,
    }


def source_objective_rows(x, y, w_b0, entries, tau):
    rows = []
    for variant, w in entries:
        vals = objective_only_np(x, y, w)
        radius = 2.0 * tau if variant.startswith("E1") else (float(np.linalg.norm(w_b0)) if variant.startswith("E2") else float("nan"))
        rows.append({
            "variant": variant,
            **vals,
            "radius": radius,
            "w_norm_over_R": vals["w_norm"] / radius if radius == radius and radius > 0 else float("nan"),
            "boundary_active": bool(vals["w_norm"] / radius >= 0.999) if radius == radius and radius > 0 else False,
        })
    return rows


def write_report(outcome, summary_rows, source_rows, geom_rows, runtime):
    lines = [
        "# Exact Empirical CVaR Bounded Contrast - Solver Validation",
        "",
        f"Outcome: `{outcome}`",
        "",
        "This run removed the eta optimization variable and used canonical empirical CVaR dual/top-tail weights. No target evaluation was run.",
        "",
        "## Multi-Start",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['variant']}: rel spread {float(row['relative_objective_spread']):.6g}, "
            f"min logit Pearson {float(row['min_source_logit_pearson']):.8g}, "
            f"max KKT residual {float(row['max_kkt_projected_residual']):.6g}, "
            f"converged={row['multistart_converged']}"
        )
    lines += ["", "## Source Objectives"]
    for row in source_rows:
        lines.append(
            f"- {row['variant']}: F={float(row['F']):.6f}, base={float(row['L_base']):.6f}, "
            f"tail={float(row['L_tail']):.6f}, ||w||={float(row['w_norm']):.6f}, "
            f"boundary={row['boundary_active']}"
        )
    lines += ["", "## Geometry"]
    for row in geom_rows:
        lines.append(f"- {row['metric']}: {row['value']}")
    lines += ["", f"Runtime seconds: {runtime:.2f}", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    seed_all(SEED)
    q_pc1, corrupt_payload, states_c, states_hn, w_b0 = load_payloads(device)
    x, y, _pool_rows = prepare_source(q_pc1, states_c, corrupt_payload, device, args.smoke)
    eq_rows = algebraic_equivalence(x, states_hn, w_b0)
    tau = float(eq_rows[0]["temperature_tau"])
    rmax = 2.0 * tau
    rb0 = float(np.linalg.norm(w_b0))

    zero_losses = np.logaddexp(np.zeros(int((y < 0.5).sum()), dtype=np.float64), 0.0)
    q_zero, qmeta = canonical_cvar_weights_np(zero_losses)
    zero_err = float(np.max(np.abs(q_zero - (1.0 / float(q_zero.size)))))
    zero_rows = [{
        "n_normal": int(q_zero.size),
        "expected_uniform_q": 1.0 / float(q_zero.size),
        "max_abs_error": zero_err,
        "sum_q_abs_error": abs(float(q_zero.sum()) - 1.0),
        "bounds_ok": bool(qmeta["min_q"] >= -1e-10 and qmeta["max_q"] <= qmeta["capacity"] + 1e-10),
        "passed": bool(zero_err <= 1e-10 and abs(float(q_zero.sum()) - 1.0) <= 1e-10),
    }]
    write_csv(OUT / "ZERO_INIT_TIE_CHECK.csv", zero_rows)
    if not zero_rows[0]["passed"]:
        outcome = "CANONICAL_CVAR_TIE_IMPLEMENTATION_ERROR"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "strict_source_only": True, "target_evaluation_run": False})
        write_report(outcome, [], [], [], time.time() - start)
        return

    all_trajectory = []
    multistart_rows = []
    kkt_rows = []
    dual_rows = []
    candidate_rows = []
    chosen = {}
    for variant, radius in (("E1_ExactBall_R2tau", rmax), ("E2_B0NormBall", rb0)):
        chosen[variant] = {}
        for init_name in INIT_NAMES:
            w_init = init_inside_ball(init_name, w_b0, radius)
            payload = solve_projected(variant, init_name, x, y, w_init, radius, args.smoke, device)
            all_trajectory.extend(payload["trajectory"])
            candidate_rows.extend(payload["candidate_rows"])
            w_final = payload["w"]
            kres, info = kkt_residual(x, y, w_final, radius, device)
            dcheck = cvar_dual_checks(info["normal_losses"])
            source_vals = objective_only_np(x, y, w_final)
            row = {
                "variant": variant,
                "init": init_name,
                "chosen_kind": payload["chosen_kind"],
                "chosen_iteration": payload["chosen_iteration"],
                "radius": float(radius),
                **source_vals,
            }
            multistart_rows.append(row)
            kkt_rows.append({"variant": variant, "init": init_name, "projected_residual": kres, "grad_norm": info["grad_norm"], "radius": float(radius), "w_norm": float(np.linalg.norm(w_final))})
            dual_rows.append({"variant": variant, "init": init_name, **dcheck, "L_tail": source_vals["L_tail"], "Top10Tail": source_vals["Top10Tail"], "tail_minus_top10": source_vals["tail_minus_top10"]})
            chosen[variant][init_name] = {"w": w_final, **source_vals}

    pair_rows = []
    for variant in chosen:
        for row in pairwise_logit_stats(x, chosen[variant]):
            row["variant"] = variant
            pair_rows.append(row)
    summary_rows = [convergence_summary(variant, multistart_rows, pair_rows, kkt_rows) for variant in chosen]

    e1_best_name = min(chosen["E1_ExactBall_R2tau"], key=lambda n: chosen["E1_ExactBall_R2tau"][n]["F"])
    e2_best_name = min(chosen["E2_B0NormBall"], key=lambda n: chosen["E2_B0NormBall"][n]["F"])
    w_e1 = chosen["E1_ExactBall_R2tau"][e1_best_name]["w"]
    w_e2 = chosen["E2_B0NormBall"][e2_best_name]["w"]
    source_rows = source_objective_rows(x, y, w_b0, [
        ("B0_FormalEffectiveContrast", w_b0),
        ("E1_ExactBall_R2tau", w_e1),
        ("E2_B0NormBall", w_e2),
    ], tau)
    geom_rows = [
        {"metric": "R_max_2tau", "value": rmax},
        {"metric": "R_B0", "value": rb0},
        {"metric": "R_B0_over_R_max", "value": rb0 / rmax},
        {"metric": "E1_final_norm", "value": float(np.linalg.norm(w_e1))},
        {"metric": "E1_final_norm_over_R", "value": float(np.linalg.norm(w_e1)) / rmax},
        {"metric": "E1_boundary_active", "value": bool(float(np.linalg.norm(w_e1)) / rmax >= 0.999)},
        {"metric": "E2_final_norm", "value": float(np.linalg.norm(w_e2))},
        {"metric": "E2_final_norm_over_R", "value": float(np.linalg.norm(w_e2)) / rb0},
        {"metric": "E2_boundary_active", "value": bool(float(np.linalg.norm(w_e2)) / rb0 >= 0.999)},
        {"metric": "cos_w_B0_w_E1", "value": cos_np(w_b0, w_e1)},
        {"metric": "cos_w_B0_w_E2", "value": cos_np(w_b0, w_e2)},
        {"metric": "cos_w_E1_w_E2", "value": cos_np(w_e1, w_e2)},
        {"metric": "E1_best_init", "value": e1_best_name},
        {"metric": "E2_best_init", "value": e2_best_name},
    ]

    write_csv(OUT / "MULTISTART_CONVERGENCE.csv", multistart_rows + [{**r, "record_type": "summary"} for r in summary_rows])
    write_csv(OUT / "OPTIMIZATION_TRAJECTORY.csv", all_trajectory)
    write_csv(OUT / "KKT_RESIDUAL.csv", kkt_rows)
    write_csv(OUT / "CVAR_DUAL_CHECK.csv", dual_rows)
    write_csv(OUT / "SOURCE_OBJECTIVE.csv", source_rows + candidate_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom_rows)

    all_converged = all(bool(r["multistart_converged"]) for r in summary_rows)
    any_nonunique = any(bool(r["prediction_null_nonuniqueness_candidate"]) for r in summary_rows)
    if all_converged:
        outcome = "EXACT_BOUNDED_CVAR_SOLVER_VALIDATED"
    elif any_nonunique:
        outcome = "BOUNDED_CVAR_HAS_PREDICTION_NULL_NONUNIQUENESS"
    else:
        outcome = "EXACT_BOUNDED_CVAR_SOLVER_NOT_CONVERGED"
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "strict_source_only": True,
        "target_evaluation_run": False,
        "scientific_objective_changed": False,
        "eta_variable_used": False,
        "alpha": ALPHA,
        "max_iter": 80 if args.smoke else MAX_ITER,
        "device": device,
        "summary": summary_rows,
        "zero_init_tie_check": zero_rows[0],
        "runtime_sec": time.time() - start,
    })
    write_report(outcome, summary_rows, source_rows, geom_rows, time.time() - start)
    print(f"[ExactCVaR] outcome={outcome}")
    print(f"[ExactCVaR] wrote {OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
