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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_joint_contrast_parameterization_causal_decomposition import (
    JointDirect,
    JointTwoState,
    JointWeightNorm,
    build_batch_plan,
    collect_raw_source_patch_pool,
    materialize_batch,
    sigmoid_inverse,
)
from experiments.phase_prediction_null_state_gauge_causal_test import (
    HISTORICAL_B0,
    TOP_NEG_FRAC,
    load_b0_raw_metrics,
    reproduction_rows,
)
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_spherical_state_trajectory_averaging import (
    balanced_batch,
    load_payloads as load_canonical_payloads,
    prepare_groups,
    train_b0_trajectory,
)
from experiments.phase_state_pair_decomposition import hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, RANK, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_canonical_joint_contrast_causal_decomposition/formal")
TARGETS = ("MVTec", "BTAD", "CVC-ClinicDB")
EPOCHS = 10
EPS = 1e-12
VIRTUAL_EPS = 1e-4


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


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hardnormal_loss(logits, y):
    task_loss, bce, dice = balanced_loss(logits, y)
    neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
    selected = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
    hard = F.softplus(logits[selected]).mean()
    return task_loss + hard, task_loss, bce, dice, hard, selected


def model_from_kind(kind, align_init, init_states, w0, rmax):
    if kind == "two_state":
        return JointTwoState(align_init, init_states)
    if kind in ("direct", "direct_replay"):
        return JointDirect(align_init, w0)
    if kind == "weight_norm":
        return JointWeightNorm(align_init, w0, rmax)
    raise ValueError(kind)


def effective_w_np(model):
    with torch.no_grad():
        if hasattr(model, "effective_w"):
            return model.effective_w().detach().cpu().numpy().astype(np.float32)
        states = F.normalize(model.states, dim=-1)
        return (states[1] - states[0]).detach().cpu().numpy().astype(np.float32)


def function_equivalence(models, batch, device):
    xb, yb, _origin, _cats = batch
    x = torch.tensor(xb, dtype=torch.float32, device=device)
    y = torch.tensor(yb, dtype=torch.float32, device=device)
    cache = {}
    for name, model in models.items():
        model = model.to(device).eval()
        with torch.no_grad():
            logits, z = model.logits(x)
            prob = torch.sigmoid(logits)
        loss, task, _bce, _dice, hard, selected = hardnormal_loss(logits, y)
        cache[name] = {
            "logits": logits.detach().cpu().numpy(),
            "prob": prob.detach().cpu().numpy(),
            "loss": float(loss.detach().cpu()),
            "task": float(task.detach().cpu()),
            "hard": float(hard.detach().cpu()),
            "selected": selected.detach().cpu().numpy(),
        }
    ref = cache["B0_JointTwoStateDiagnostic"]
    rows = []
    for name, vals in cache.items():
        rows.append({
            "variant": name,
            "max_abs_logit_diff_vs_B0": float(np.max(np.abs(vals["logits"] - ref["logits"]))),
            "max_abs_probability_diff_vs_B0": float(np.max(np.abs(vals["prob"] - ref["prob"]))),
            "loss": vals["loss"],
            "statealigned_loss": vals["task"],
            "hardnormal_loss": vals["hard"],
            "loss_diff_vs_B0": vals["loss"] - ref["loss"],
            "hard_set_overlap_vs_B0": float(len(np.intersect1d(vals["selected"], ref["selected"])) / max(1, len(ref["selected"]))),
        })
    ok = all(r["max_abs_logit_diff_vs_B0"] <= 1e-6 and abs(r["loss_diff_vs_B0"]) <= 1e-7 and r["hard_set_overlap_vs_B0"] == 1.0 for r in rows)
    return rows, ok


def diagnostic_step_delta(model, batch, device):
    model = model.to(device)
    xb, yb, _origin, _cats = batch
    x = torch.tensor(xb, dtype=torch.float32, device=device)
    y = torch.tensor(yb, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    w_before = effective_w_np(model)
    logits, _z = model.logits(x)
    loss, _task, _bce, _dice, _hard, _sel = hardnormal_loss(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if isinstance(model, JointDirect):
        with torch.no_grad():
            n = model.w.norm().clamp_min(EPS)
            if float(n.cpu()) > 2.0:
                model.w.mul_(2.0 / n)
    w_after = effective_w_np(model)
    return w_after - w_before


def first_step_geometry(models, batch, device):
    deltas = {}
    rows = []
    for name, model in models.items():
        seed_all(SEED)
        delta = diagnostic_step_delta(model, batch, device)
        deltas[name] = delta
        rows.append({"variant": name, "delta_w_norm": float(np.linalg.norm(delta))})
    for a, b in combinations(models.keys(), 2):
        rows.append({"variant": f"{a}__{b}", "cos_delta_w": cos_np(deltas[a], deltas[b]), "delta_w_norm_ratio": float(np.linalg.norm(deltas[a]) / max(np.linalg.norm(deltas[b]), EPS))})
    return rows


def source_metrics(groups, model, device):
    y_all, s_all = [], []
    losses = []
    model.eval()
    with torch.no_grad():
        for data in groups.values():
            x = torch.tensor(data["x"], dtype=torch.float32, device=device)
            y = torch.tensor(data["y"], dtype=torch.float32, device=device)
            logits, _z = model.logits(x)
            loss, _task, _bce, _dice, hard, _sel = hardnormal_loss(logits, y)
            losses.append((float(loss.cpu()), float(hard.cpu())))
            y_all.append(data["y"])
            s_all.append(logits.detach().cpu().numpy())
    y = np.concatenate(y_all)
    s = np.concatenate(s_all)
    auc = safe_auc(y, s)
    ap = safe_ap(y, s)
    return {"source_P-AUROC": auc, "source_P-AP": ap, "source_G": 0.5 * (auc + ap), "source_total_loss": float(np.mean([v[0] for v in losses])), "source_hardnormal_loss": float(np.mean([v[1] for v in losses])), "contrast_norm": float(np.linalg.norm(effective_w_np(model)))}


def train_joint_variant(variant, kind, groups, plan, align_init, init_states, w0, rmax, b0_norm_schedule, b0_probe_cache, probe_spec, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["model_obj"].to(device).eval(), payload, True
    print(f"[CanonicalJoint] train {variant}", flush=True)
    seed_all(SEED)
    model = model_from_kind(kind, align_init, init_states, w0, rmax).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    traj, align_rows, grad_rows = [], [], []
    step_idx = 0
    prev_w = effective_w_np(model)
    for spec in plan:
        step_idx += 1
        xb, yb, origin, cats = materialize_batch(groups, spec)
        x = torch.tensor(xb, dtype=torch.float32, device=device)
        y = torch.tensor(yb, dtype=torch.float32, device=device)
        logits, _z = model.logits(x)
        loss, task, bce, dice, hard, selected = hardnormal_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if kind == "direct":
                n = model.w.norm().clamp_min(EPS)
                if float(n.cpu()) > rmax:
                    model.w.mul_(rmax / n)
            elif kind == "direct_replay":
                target = float(b0_norm_schedule[step_idx - 1])
                model.w.mul_(target / model.w.norm().clamp_min(EPS))
        if spec["step"] == 1 or step_idx == len(plan):
            w = effective_w_np(model)
            traj.append({
                "variant": variant,
                "epoch": spec["epoch"],
                "step": spec["step"],
                "global_step": step_idx,
                "loss": float(loss.detach().cpu()),
                "statealigned_loss": float(task.detach().cpu()),
                "hardnormal_loss": float(hard.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "dice": float(dice.detach().cpu()),
                "contrast_norm": float(np.linalg.norm(w)),
                "cos_w_prev": cos_np(w, prev_w),
                "selected_hard_count": int(selected.numel()),
                "selected_normal_image_ratio": float(np.mean(origin[selected.detach().cpu().numpy()] == "normal_image")) if selected.numel() else float("nan"),
            })
            prev_w = w.copy()
            ar, gr = alignment_and_gradient_diag(variant, model, groups, probe_spec, b0_probe_cache, device)
            ar.update({"epoch": spec["epoch"], "global_step": step_idx})
            gr.update({"epoch": spec["epoch"], "global_step": step_idx})
            align_rows.append(ar)
            grad_rows.append(gr)
        if step_idx % 20 == 0 or step_idx == 1:
            print(f"[CanonicalJoint] {variant} step={step_idx}/{len(plan)} loss={float(loss.detach().cpu()):.4f}", flush=True)
    payload = {"model_obj": model.eval().cpu(), "trajectory": traj, "alignment_rows": align_rows, "gradient_rows": grad_rows}
    torch.save(payload, ckpt)
    return model.to(device).eval(), payload, False


def capture_canonical_b0_norm_schedule(groups, m0, r0, init_v, total_steps, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"canonical_b0_step_norm_schedule_{suffix}.pt"
    if ckpt.exists():
        return torch.load(ckpt, map_location=device, weights_only=False)["norm_schedule"]
    seed_all(SEED)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m0, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    schedule = []
    for _step in range(total_steps):
        xb, yb = balanced_batch(groups, rng, BATCH_SIZE)
        z = torch.tensor(xb, dtype=torch.float32, device=device)
        y = torch.tensor(yb, dtype=torch.float32, device=device)
        v = F.normalize(model, dim=-1)
        s_a = F.normalize(m + r0 * v, dim=-1)
        s_n = F.normalize(m - r0 * v, dim=-1)
        logits = z @ s_a - z @ s_n
        loss, _task, _bce, _dice, _hard, _sel = hardnormal_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v_after = F.normalize(model, dim=-1)
            w_after = F.normalize(m + r0 * v_after, dim=-1) - F.normalize(m - r0 * v_after, dim=-1)
            schedule.append(float(w_after.norm().detach().cpu()))
    torch.save({"norm_schedule": schedule}, ckpt)
    return schedule


def alignment_and_gradient_diag(variant, model, groups, probe_spec, b0_probe_cache, device):
    xb, yb, _origin, _cats = materialize_batch(groups, probe_spec)
    x = torch.tensor(xb, dtype=torch.float32, device=device)
    y = torch.tensor(yb, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, z = model.logits(x)
        w = model.effective_w().detach()
        margin_pos = float(logits[y > 0.5].mean().detach().cpu()) if torch.any(y > 0.5) else float("nan")
        margin_neg = float(logits[y < 0.5].mean().detach().cpu()) if torch.any(y < 0.5) else float("nan")
    b0_z = b0_probe_cache["z"].to(device)
    b0_w = b0_probe_cache["w"].to(device)
    align_dist = 0.0
    for name, p in model.named_parameters():
        if name.startswith(("a.", "b.")):
            refp = b0_probe_cache["params"][name].to(device)
            align_dist += float(torch.sum((p.detach() - refp) ** 2).cpu())
    align_row = {
        "variant": variant,
        "cos_w_variant_w_B0_initial": float(F.cosine_similarity(w, b0_w, dim=0).detach().cpu()),
        "probe_feature_cos_mean_vs_B0_initial": float(F.cosine_similarity(z, b0_z, dim=-1).mean().detach().cpu()),
        "alignment_parameter_l2_distance_vs_init": math.sqrt(align_dist),
        "source_margin_anomaly_mean": margin_pos,
        "source_margin_normal_mean": margin_neg,
    }

    logits, _z = model.logits(x)
    loss, _task, _bce, _dice, _hard, _sel = hardnormal_loss(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    a_grad = model.a.weight.grad.detach().clone() if model.a.weight.grad is not None else torch.zeros_like(model.a.weight)
    b_grad = model.b.weight.grad.detach().clone() if model.b.weight.grad is not None else torch.zeros_like(model.b.weight)
    scorer_grads = []
    for name, p in model.named_parameters():
        if not name.startswith(("a.", "b.")) and p.grad is not None:
            scorer_grads.append(p.grad.detach().flatten())
    g_scorer_norm = float(torch.cat(scorer_grads).norm().detach().cpu()) if scorer_grads else 0.0
    with torch.no_grad():
        z_before = model.adapt(x)
        q_before = z_before @ model.effective_w()
        model.a.weight.sub_(VIRTUAL_EPS * a_grad)
        model.b.weight.sub_(VIRTUAL_EPS * b_grad)
        z_after = model.adapt(x)
        q_align_after = z_after @ model.effective_w()
        model.a.weight.add_(VIRTUAL_EPS * a_grad)
        model.b.weight.add_(VIRTUAL_EPS * b_grad)
        dq_align = (q_align_after - q_before).detach().cpu().numpy()
    if hasattr(model, "head") and model.head.grad is not None:
        with torch.no_grad():
            old = model.head.detach().clone()
            model.head.sub_(VIRTUAL_EPS * model.head.grad)
            q_s = model.logits(x)[0]
            model.head.copy_(old)
            dq_scorer = (q_s - q_before).detach().cpu().numpy()
    elif hasattr(model, "w") and model.w.grad is not None:
        with torch.no_grad():
            old = model.w.detach().clone()
            model.w.sub_(VIRTUAL_EPS * model.w.grad)
            q_s = model.logits(x)[0]
            model.w.copy_(old)
            dq_scorer = (q_s - q_before).detach().cpu().numpy()
    else:
        dq_scorer = np.zeros_like(dq_align)
    grad_row = {
        "variant": variant,
        "alignment_grad_norm": float(torch.sqrt(torch.sum(a_grad ** 2) + torch.sum(b_grad ** 2)).detach().cpu()),
        "scorer_grad_norm": g_scorer_norm,
        "corr_delta_q_alignment_scorer": corr_np(dq_align, dq_scorer),
        "norm_delta_q_alignment_over_scorer": float(np.linalg.norm(dq_align) / max(np.linalg.norm(dq_scorer), EPS)),
    }
    model.zero_grad(set_to_none=True)
    return align_row, grad_row


def evaluate_joint_models(models, q_pc1, device, smoke):
    return evaluate_pc1_removed(models, q_pc1, device, smoke)


def decide(raw, source_rows):
    by = {(r["dataset"], r["variant"]): float(r["G_pixel"]) for r in raw if r["dataset"] in TARGETS}
    b0 = np.asarray([by[(ds, "B0_Canonical")] for ds in TARGETS], dtype=np.float32)
    e1 = np.asarray([by[(ds, "E1_JointBoundedSingleContrast")] for ds in TARGETS], dtype=np.float32)
    e2 = np.asarray([by[(ds, "E2_NormTrajectoryMatchedSingleContrast")] for ds in TARGETS], dtype=np.float32)
    e3 = np.asarray([by[(ds, "E3_WeightNormalizedSingleContrast")] for ds in TARGETS], dtype=np.float32)
    if float(np.mean(np.abs(e1 - b0))) <= 0.5 and int(np.sum(np.abs(e1 - b0) <= 0.5)) >= 2:
        outcome = "JOINT_FEATURE_BOUNDARY_COADAPTATION_EXPLAINS_TRANSFER"
    elif float(np.mean(b0 - e1)) >= 1.0 and float(np.mean(np.abs(e2 - b0))) <= 0.5 and float(np.mean(e2 - e1)) >= 1.0:
        outcome = "CONTRAST_NORM_TRAJECTORY_EXPLAINS_TWO_STATE_TRANSFER"
    elif float(np.mean(b0 - e1)) >= 1.0 and float(np.mean(np.abs(e3 - b0))) <= 0.5:
        outcome = "ANGULAR_NORMALIZED_OPTIMIZATION_EXPLAINS_TRANSFER"
    elif all(float(np.mean(b0 - arr)) >= 1.0 for arr in (e1, e2, e3)):
        outcome = "CANONICAL_TWO_STATE_PARAMETERIZATION_HAS_TRANSFER_IMPLICIT_BIAS"
    else:
        outcome = "CANONICAL_JOINT_CONTRAST_CAUSALITY_INCONCLUSIVE"
    src_g = [float(r["source_G"]) for r in source_rows]
    extra = "SOURCE_EQUIVALENT_PARAMETERIZATIONS_DIVERGE_IN_TRANSFER" if (max(src_g) - min(src_g) <= 1.0 and max(abs(float(np.mean(arr - b0))) for arr in (e1, e2, e3)) >= 1.0) else ""
    return outcome, {
        "mean_abs_E1_minus_B0_G": float(np.mean(np.abs(e1 - b0))),
        "mean_E2_minus_E1_G": float(np.mean(e2 - e1)),
        "mean_abs_E2_minus_B0_G": float(np.mean(np.abs(e2 - b0))),
        "mean_abs_E3_minus_B0_G": float(np.mean(np.abs(e3 - b0))),
        "extra_flag": extra,
    }


def write_report(outcome, stats, repro, raw, source_rows, runtime):
    lines = [
        "# Canonical Joint Contrast Causal Decomposition",
        "",
        f"Outcome: `{outcome}`",
    ]
    if stats.get("extra_flag"):
        lines.append(f"Extra flag: `{stats['extra_flag']}`")
    lines += ["", "## B0 Reproduction"]
    for r in repro:
        lines.append(f"- {r['dataset']}: {float(r['rerun_G']):.4f} vs {float(r['historical_G']):.4f}, delta {float(r['delta_G']):+.4f}, passed={r['passed']}")
    lines += ["", "## Target G"]
    for r in raw:
        if r["dataset"] in TARGETS:
            lines.append(f"- {r['variant']} on {r['dataset']}: G {float(r['G_pixel']):.4f}")
    lines += ["", "## Source"]
    for r in source_rows:
        lines.append(f"- {r['variant']}: source G {float(r['source_G']):.4f}, loss {float(r['source_total_loss']):.4f}, hard {float(r['source_hardnormal_loss']):.4f}, ||w|| {float(r['contrast_norm']):.4f}")
    lines += ["", "## Decision Stats"]
    for k, v in stats.items():
        lines.append(f"- {k}: {v}")
    lines += ["", f"Runtime seconds: {runtime:.1f}", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[CanonicalJoint] device={device}", flush=True)
    seed_all(SEED)
    q_pc1, corrupt_payload, states_c, m0, r0 = load_canonical_payloads(device)
    init_v = unit_np(np.load(Path("results/phase_hard_normal_contrast/formal/contrast_vectors.npz"))["B0"].astype(np.float32))
    canonical_groups, _pool_rows = prepare_groups(q_pc1, states_c, corrupt_payload, device, args.smoke)
    epoch_raw, epoch_v, trajectory_b0, _reused = train_b0_trajectory(canonical_groups, m0, r0, init_v, device, args.smoke)
    b0_model = make_model(hybrid_states(m0, r0 * epoch_v[10]), corrupt_payload, device)
    b0_raw, b0_per = evaluate_pc1_removed({"B0_Canonical": b0_model}, q_pc1, device, args.smoke)
    repro = reproduction_rows(b0_raw) if not args.smoke else [{"dataset": r["dataset"], "historical_G": HISTORICAL_B0.get(r["dataset"], float("nan")), "rerun_G": r["G_pixel"], "delta_G": float("nan"), "passed": True} for r in b0_raw]
    write_csv(OUT / "B0_REPRODUCTION.csv", repro)
    if not all(bool(r["passed"]) for r in repro):
        outcome = "B0_REPRODUCTION_FAILED"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "strict_source_only": True, "target_tuning": False})
        write_report(outcome, {}, repro, b0_raw, [], time.time() - start)
        return

    align_init = make_model(states_c, corrupt_payload, device)
    init_states = hybrid_states(m0, r0 * init_v).astype(np.float32)
    w0 = (np.asarray(init_states[1]) / np.linalg.norm(init_states[1]) - np.asarray(init_states[0]) / np.linalg.norm(init_states[0])).astype(np.float32)
    rmax = 2.0
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    raw_groups, _raw_audit = collect_raw_source_patch_pool(slots, labels_by_pos, patch, q_pc1)
    plan = build_batch_plan(raw_groups, args.smoke)
    first_batch = materialize_batch(raw_groups, plan[0])
    init_models = {
        "B0_JointTwoStateDiagnostic": model_from_kind("two_state", align_init, init_states, w0, rmax),
        "E1_JointBoundedSingleContrast": model_from_kind("direct", align_init, init_states, w0, rmax),
        "E2_NormTrajectoryMatchedSingleContrast": model_from_kind("direct_replay", align_init, init_states, w0, rmax),
        "E3_WeightNormalizedSingleContrast": model_from_kind("weight_norm", align_init, init_states, w0, rmax),
    }
    func_rows, func_ok = function_equivalence(init_models, first_batch, device)
    write_csv(OUT / "FUNCTION_EQUIVALENCE.csv", func_rows)
    if not func_ok:
        outcome = "FUNCTION_MATCHED_INITIALIZATION_FAILED"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "strict_source_only": True, "target_tuning": False})
        write_report(outcome, {}, repro, b0_raw, [], time.time() - start)
        return

    first_models = {
        name: model_from_kind(("two_state" if "B0" in name else "direct" if "E1" in name else "direct_replay" if "E2" in name else "weight_norm"), align_init, init_states, w0, rmax)
        for name in init_models
    }
    first_rows = first_step_geometry(first_models, first_batch, device)
    write_csv(OUT / "FIRST_STEP_GEOMETRY.csv", first_rows)

    b0_norm_schedule = capture_canonical_b0_norm_schedule(canonical_groups, m0, r0, init_v, len(plan), device, args.smoke)

    probe_spec = plan[0]
    xb_probe, _yb, _o, _c = materialize_batch(raw_groups, probe_spec)
    with torch.no_grad():
        x_probe = torch.tensor(xb_probe, dtype=torch.float32, device=device)
        z_probe = align_init.adapt(x_probe).detach().cpu()
        w_probe = torch.tensor(w0, dtype=torch.float32).detach().cpu()
        params_probe = {name: p.detach().cpu().clone() for name, p in align_init.named_parameters()}
    b0_probe_cache = {"z": z_probe, "w": w_probe, "params": params_probe}

    variants = [
        ("E1_JointBoundedSingleContrast", "direct"),
        ("E2_NormTrajectoryMatchedSingleContrast", "direct_replay"),
        ("E3_WeightNormalizedSingleContrast", "weight_norm"),
    ]
    models, payloads = {}, {}
    for variant, kind in variants:
        model, payload, _reused = train_joint_variant(variant, kind, raw_groups, plan, align_init, init_states, w0, rmax, b0_norm_schedule, b0_probe_cache, probe_spec, device, args.smoke)
        models[variant] = model
        payloads[variant] = payload
    raw_more, per_more = evaluate_joint_models(models, q_pc1, device, args.smoke)
    raw = b0_raw + raw_more
    per = b0_per + per_more
    source_rows = [{"variant": "B0_Canonical", **source_metrics(raw_groups, b0_model, device)}]
    for variant, _kind in variants:
        source_rows.append({"variant": variant, **source_metrics(raw_groups, models[variant], device)})
    traj = []
    for r in trajectory_b0:
        row = dict(r)
        row["variant"] = "B0_Canonical"
        traj.append(row)
    align_rows, grad_rows = [], []
    for payload in payloads.values():
        traj.extend(payload["trajectory"])
        align_rows.extend(payload["alignment_rows"])
        grad_rows.extend(payload["gradient_rows"])
    outcome, stats = decide(raw, source_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", per)
    write_csv(OUT / "CONTRAST_TRAJECTORY.csv", traj)
    write_csv(OUT / "ALIGNMENT_TRAJECTORY.csv", align_rows)
    write_csv(OUT / "GRADIENT_COADAPTATION.csv", grad_rows)
    write_csv(OUT / "SOURCE_COMPARISON.csv", source_rows)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "stats": stats, "strict_source_only": True, "target_tuning": False, "target_statistics": False, "seed": SEED, "lr": LR, "epochs": EPOCHS, "top_neg_frac": TOP_NEG_FRAC, "runtime_sec": time.time() - start})
    write_report(outcome, stats, repro, raw, source_rows, time.time() - start)
    print(f"[CanonicalJoint] OUTCOME={outcome}")
    print(f"[CanonicalJoint] wrote {OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
