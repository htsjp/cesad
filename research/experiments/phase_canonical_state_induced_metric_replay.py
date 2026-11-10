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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_joint_contrast_parameterization_causal_decomposition import (
    JointDirect,
    build_batch_plan,
    collect_raw_source_patch_pool,
    materialize_batch,
)
from experiments.phase_prediction_null_state_gauge_causal_test import HISTORICAL_B0, TOP_NEG_FRAC, reproduction_rows
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
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
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_canonical_state_induced_metric_replay/formal")
TARGETS = ("MVTec", "BTAD", "CVC-ClinicDB")
EPOCHS = 10
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


def state_metric(g, a, b, r_a, r_n):
    return ((g - a * torch.dot(a, g)) / max(r_a * r_a, EPS)) + ((g - b * torch.dot(b, g)) / max(r_n * r_n, EPS))


def random_state_pair_like(a, b, seed):
    rng = np.random.default_rng(seed)
    dim = a.shape[0]
    cos_ab = float(np.dot(a, b))
    ar = unit_np(rng.standard_normal(dim).astype(np.float32))
    v = rng.standard_normal(dim).astype(np.float32)
    v = unit_np(v - float(v @ ar) * ar)
    br = cos_ab * ar + math.sqrt(max(0.0, 1.0 - cos_ab * cos_ab)) * v
    return ar.astype(np.float32), unit_np(br.astype(np.float32))


def canonical_metric_schedule(groups, m0, r0, init_v, total_steps, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"canonical_metric_schedule_{suffix}.pt"
    if ckpt.exists():
        return torch.load(ckpt, map_location=device, weights_only=False)
    seed_all(SEED)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m0, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    metrics, rows = [], []
    first_delta = None
    for step_idx in range(1, total_steps + 1):
        xb, yb = balanced_batch(groups, rng, BATCH_SIZE)
        z = torch.tensor(xb, dtype=torch.float32, device=device)
        y = torch.tensor(yb, dtype=torch.float32, device=device)
        with torch.no_grad():
            v_before = F.normalize(model, dim=-1)
            s_a = m + r0 * v_before
            s_n = m - r0 * v_before
            a = F.normalize(s_a, dim=-1)
            b = F.normalize(s_n, dim=-1)
            w_before = a - b
            metrics.append({
                "a": a.detach().cpu().numpy().astype(np.float32),
                "b": b.detach().cpu().numpy().astype(np.float32),
                "rA": float(s_a.norm().detach().cpu()),
                "rN": float(s_n.norm().detach().cpu()),
                "w": w_before.detach().cpu().numpy().astype(np.float32),
            })
        logits = contrast_logits(z, m, r0, F.normalize(model, dim=-1))
        loss, task, _bce, _dice, hard, _sel = hardnormal_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v_after = F.normalize(model, dim=-1)
            w_after = F.normalize(m + r0 * v_after, dim=-1) - F.normalize(m - r0 * v_after, dim=-1)
            delta = (w_after - w_before).detach().cpu().numpy().astype(np.float32)
            if step_idx == 1:
                first_delta = delta
            rows.append({
                "variant": "B0_CanonicalMetric",
                "global_step": step_idx,
                "epoch": int(math.ceil(step_idx / max(1, total_steps / EPOCHS))),
                "loss": float(loss.detach().cpu()),
                "statealigned_loss": float(task.detach().cpu()),
                "hardnormal_loss": float(hard.detach().cpu()),
                "w_norm": float(w_before.norm().detach().cpu()),
                "rA": metrics[-1]["rA"],
                "rN": metrics[-1]["rN"],
                "cos_a_b": float(torch.dot(a, b).detach().cpu()),
            })
    payload = {"metrics": metrics, "trajectory": rows, "first_delta": first_delta}
    torch.save(payload, ckpt)
    return payload


def metric_for_variant(metrics, step_idx, mode):
    idx = step_idx
    if mode == "canonical":
        return metrics[idx]
    if mode == "shuffled":
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(metrics))
        return metrics[int(perm[idx])]
    if mode == "random":
        base = metrics[idx]
        a, b = random_state_pair_like(base["a"], base["b"], 100000 + idx)
        return {"a": a, "b": b, "rA": base["rA"], "rN": base["rN"], "w": base["w"]}
    raise ValueError(mode)


def first_step_delta(model, first_batch, metric, mode, device):
    model = model.to(device)
    xb, yb, _origin, _cats = first_batch
    x = torch.tensor(xb, dtype=torch.float32, device=device)
    y = torch.tensor(yb, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    w_before = model.effective_w().detach().cpu().numpy().astype(np.float32)
    logits, _z = model.logits(x)
    loss, _task, _bce, _dice, _hard, _sel = hardnormal_loss(logits, y)
    opt.zero_grad()
    loss.backward()
    if mode != "ordinary":
        g = model.w.grad.detach()
        a = torch.tensor(metric["a"], dtype=torch.float32, device=device)
        b = torch.tensor(metric["b"], dtype=torch.float32, device=device)
        model.w.grad.copy_(state_metric(g, a, b, metric["rA"], metric["rN"]))
    opt.step()
    with torch.no_grad():
        n = model.w.norm().clamp_min(EPS)
        if float(n.detach().cpu()) > 2.0:
            model.w.mul_(2.0 / n)
    w_after = model.effective_w().detach().cpu().numpy().astype(np.float32)
    return w_after - w_before


def train_variant(variant, mode, groups, plan, align_init, w0, metric_payload, b0_probe_cache, probe_spec, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["model_obj"].to(device).eval(), payload
    print(f"[MetricReplay] train {variant}", flush=True)
    seed_all(SEED)
    model = JointDirect(align_init, w0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    rows, align_rows = [], []
    metrics = metric_payload["metrics"]
    for step_idx, spec in enumerate(plan):
        xb, yb, origin, _cats = materialize_batch(groups, spec)
        x = torch.tensor(xb, dtype=torch.float32, device=device)
        y = torch.tensor(yb, dtype=torch.float32, device=device)
        logits, _z = model.logits(x)
        loss, task, bce, dice, hard, selected = hardnormal_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        if mode != "ordinary":
            met = metric_for_variant(metrics, step_idx, mode)
            g = model.w.grad.detach()
            a = torch.tensor(met["a"], dtype=torch.float32, device=device)
            b = torch.tensor(met["b"], dtype=torch.float32, device=device)
            model.w.grad.copy_(state_metric(g, a, b, met["rA"], met["rN"]))
        opt.step()
        with torch.no_grad():
            n = model.w.norm().clamp_min(EPS)
            if float(n.detach().cpu()) > 2.0:
                model.w.mul_(2.0 / n)
        if spec["step"] == 1 or step_idx == len(plan) - 1:
            w = model.effective_w().detach().cpu().numpy().astype(np.float32)
            rows.append({
                "variant": variant,
                "epoch": spec["epoch"],
                "step": spec["step"],
                "global_step": step_idx + 1,
                "loss": float(loss.detach().cpu()),
                "statealigned_loss": float(task.detach().cpu()),
                "hardnormal_loss": float(hard.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "dice": float(dice.detach().cpu()),
                "contrast_norm": float(np.linalg.norm(w)),
                "cos_w_B0_step": cos_np(w, metrics[min(step_idx, len(metrics) - 1)]["w"]),
                "selected_hard_count": int(selected.numel()),
                "selected_normal_image_ratio": float(np.mean(origin[selected.detach().cpu().numpy()] == "normal_image")) if selected.numel() else float("nan"),
            })
            align_rows.append(alignment_diag(variant, model, groups, probe_spec, b0_probe_cache, device, spec["epoch"], step_idx + 1))
        if step_idx % 20 == 0 or step_idx == len(plan) - 1:
            print(f"[MetricReplay] {variant} step={step_idx+1}/{len(plan)} loss={float(loss.detach().cpu()):.4f}", flush=True)
    payload = {"model_obj": model.eval().cpu(), "trajectory": rows, "alignment_rows": align_rows}
    torch.save(payload, ckpt)
    return model.to(device).eval(), payload


def alignment_diag(variant, model, groups, probe_spec, b0_probe_cache, device, epoch, step):
    xb, yb, _origin, _cats = materialize_batch(groups, probe_spec)
    x = torch.tensor(xb, dtype=torch.float32, device=device)
    with torch.no_grad():
        _logits, z = model.logits(x)
        w = model.effective_w().detach()
        b0_z = b0_probe_cache["z"].to(device)
        b0_w = b0_probe_cache["w"].to(device)
    dist = 0.0
    for name, p in model.named_parameters():
        if name.startswith(("a.", "b.")):
            ref = b0_probe_cache["params"][name].to(device)
            dist += float(torch.sum((p.detach() - ref) ** 2).cpu())
    return {
        "variant": variant,
        "epoch": epoch,
        "global_step": step,
        "cos_w_B0_initial": float(F.cosine_similarity(w, b0_w, dim=0).detach().cpu()),
        "probe_feature_cos_mean_vs_B0_initial": float(F.cosine_similarity(z, b0_z, dim=-1).mean().detach().cpu()),
        "alignment_parameter_l2_distance_vs_init": math.sqrt(dist),
    }


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
    return {"source_P-AUROC": auc, "source_P-AP": ap, "source_G": 0.5 * (auc + ap), "source_total_loss": float(np.mean([v[0] for v in losses])), "source_hardnormal_loss": float(np.mean([v[1] for v in losses])), "contrast_norm": float(np.linalg.norm(model.effective_w().detach().cpu().numpy()))}


def decide(raw, first_rows):
    by = {(r["dataset"], r["variant"]): float(r["G_pixel"]) for r in raw if r["dataset"] in TARGETS}
    b0 = np.asarray([by[(ds, "B0_Canonical")] for ds in TARGETS], dtype=np.float32)
    e1 = np.asarray([by[(ds, "E1_OrdinarySingleContrast")] for ds in TARGETS], dtype=np.float32)
    e2 = np.asarray([by[(ds, "E2_CanonicalMetricReplay")] for ds in TARGETS], dtype=np.float32)
    e3 = np.asarray([by[(ds, "E3_TimeShuffledMetric")] for ds in TARGETS], dtype=np.float32)
    e4 = np.asarray([by[(ds, "E4_RandomStateMetric")] for ds in TARGETS], dtype=np.float32)
    e2_vs_e1 = float(np.mean(e2 - e1))
    recovers = int(np.sum(np.abs(e2 - b0) < np.abs(e1 - b0)))
    if e2_vs_e1 >= 1.0 and recovers >= 2 and float(np.mean(np.abs(e2 - b0))) <= 1.0 and float(np.mean(e2 - e3)) >= 0.5 and float(np.mean(e2 - e4)) >= 0.5:
        outcome = "CANONICAL_STATE_INDUCED_METRIC_EXPLAINS_TRANSFER_BIAS"
        if float(np.mean(np.abs(e2 - b0))) <= 0.5:
            outcome += "__TWO_STATE_IMPLICIT_BIAS_IS_METRIC_COMPRESSIBLE"
    elif float(np.mean(e2 - e3)) < 0.5:
        outcome = "ANISOTROPIC_PRECONDITIONING_NOT_CANONICAL_TRAJECTORY_SPECIFIC"
    elif float(np.mean(e2 - e4)) < 0.5:
        outcome = "GENERIC_ANISOTROPIC_METRIC_SIGNAL_NOT_STATE_SPECIFIC"
    else:
        outcome = "LOCAL_STATE_METRIC_INSUFFICIENT_FOR_GLOBAL_TRANSFER_BIAS"
    return outcome, {
        "mean_E2_minus_E1_G": e2_vs_e1,
        "targets_E2_closer_to_B0_than_E1": recovers,
        "mean_abs_E2_minus_B0_G": float(np.mean(np.abs(e2 - b0))),
        "mean_E2_minus_E3_G": float(np.mean(e2 - e3)),
        "mean_E2_minus_E4_G": float(np.mean(e2 - e4)),
    }


def write_report(outcome, stats, first_rows, raw, source_rows, runtime):
    lines = [
        "# Canonical State-Induced Metric Replay",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## First Step",
    ]
    for r in first_rows:
        if "cos_delta_w" in r:
            lines.append(f"- {r['variant']}: cos {float(r['cos_delta_w']):.6f}, norm ratio {float(r['delta_w_norm_ratio']):.6f}")
    lines += ["", "## Target G"]
    for r in raw:
        if r["dataset"] in TARGETS:
            lines.append(f"- {r['variant']} on {r['dataset']}: G {float(r['G_pixel']):.4f}")
    lines += ["", "## Source"]
    for r in source_rows:
        lines.append(f"- {r['variant']}: source G {float(r['source_G']):.4f}, loss {float(r['source_total_loss']):.4f}, hard {float(r['source_hardnormal_loss']):.4f}")
    lines += ["", "## Decision Stats"]
    for k, v in stats.items():
        lines.append(f"- {k}: {v}")
    lines += ["", f"Runtime seconds: {runtime:.1f}", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def not_run_rows(stage, reason):
    return [{"stage": stage, "status": "not_run", "reason": reason}]


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[MetricReplay] device={device}", flush=True)
    seed_all(SEED)
    q_pc1, corrupt_payload, states_c, m0, r0 = load_canonical_payloads(device)
    init_v = unit_np(np.load(Path("results/phase_hard_normal_contrast/formal/contrast_vectors.npz"))["B0"].astype(np.float32))
    canonical_groups, _pool_rows = prepare_groups(q_pc1, states_c, corrupt_payload, device, args.smoke)
    epoch_raw, epoch_v, _trajectory_b0, _reused = train_b0_trajectory(canonical_groups, m0, r0, init_v, device, args.smoke)
    b0_model = make_model(hybrid_states(m0, r0 * epoch_v[10]), corrupt_payload, device)
    b0_raw, b0_per = evaluate_pc1_removed({"B0_Canonical": b0_model}, q_pc1, device, args.smoke)
    repro = reproduction_rows(b0_raw) if not args.smoke else [{"dataset": r["dataset"], "historical_G": HISTORICAL_B0.get(r["dataset"], float("nan")), "rerun_G": r["G_pixel"], "delta_G": float("nan"), "passed": True} for r in b0_raw]
    write_csv(OUT / "B0_REPRODUCTION.csv", repro)
    if not all(bool(r["passed"]) for r in repro):
        outcome = "B0_REPRODUCTION_FAILED"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "strict_source_only": True})
        write_report(outcome, {}, [], b0_raw, [], time.time() - start)
        return

    align_init = make_model(states_c, corrupt_payload, device)
    init_states = hybrid_states(m0, r0 * init_v).astype(np.float32)
    s_norm = init_states / np.linalg.norm(init_states, axis=1, keepdims=True)
    w0 = (s_norm[1] - s_norm[0]).astype(np.float32)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    raw_groups, _raw_audit = collect_raw_source_patch_pool(slots, labels_by_pos, patch, q_pc1)
    plan = build_batch_plan(raw_groups, args.smoke)
    metric_payload = canonical_metric_schedule(canonical_groups, m0, r0, init_v, len(plan), device, args.smoke)

    first_batch = materialize_batch(raw_groups, plan[0])
    first_models = {
        "E1_OrdinarySingleContrast": JointDirect(align_init, w0),
        "E2_CanonicalMetricReplay": JointDirect(align_init, w0),
    }
    first_deltas = {"B0_CanonicalActual": metric_payload["first_delta"]}
    first_deltas["E1_OrdinarySingleContrast"] = first_step_delta(first_models["E1_OrdinarySingleContrast"], first_batch, metric_payload["metrics"][0], "ordinary", device)
    first_deltas["E2_CanonicalMetricReplay"] = first_step_delta(first_models["E2_CanonicalMetricReplay"], first_batch, metric_payload["metrics"][0], "canonical", device)
    first_rows = [{"variant": k, "delta_w_norm": float(np.linalg.norm(v))} for k, v in first_deltas.items()]
    for a, b in combinations(first_deltas.keys(), 2):
        first_rows.append({"variant": f"{a}__{b}", "cos_delta_w": cos_np(first_deltas[a], first_deltas[b]), "delta_w_norm_ratio": float(np.linalg.norm(first_deltas[a]) / max(np.linalg.norm(first_deltas[b]), EPS))})
    write_csv(OUT / "FIRST_STEP_MATCHING.csv", first_rows)
    cos_e1 = [r for r in first_rows if r["variant"] == "B0_CanonicalActual__E1_OrdinarySingleContrast"][0]["cos_delta_w"]
    cos_e2 = [r for r in first_rows if r["variant"] == "B0_CanonicalActual__E2_CanonicalMetricReplay"][0]["cos_delta_w"]
    if not (float(cos_e2) >= 0.9 and float(cos_e2) > float(cos_e1)):
        outcome = "METRIC_FORMULA_DOES_NOT_CAPTURE_B0_UPDATE"
        reason = "First-step metric replay did not reach the required cos(B0,E2) >= 0.9 gate; target variants were not run."
        write_csv(OUT / "METRIC_TRAJECTORY.csv", metric_payload["trajectory"])
        write_csv(OUT / "RAW_METRICS.csv", b0_raw)
        write_csv(OUT / "PER_CATEGORY.csv", b0_per)
        write_csv(OUT / "CONTRAST_TRAJECTORY.csv", metric_payload["trajectory"])
        write_csv(OUT / "ALIGNMENT_TRAJECTORY.csv", not_run_rows("ALIGNMENT_TRAJECTORY", reason))
        write_csv(OUT / "SOURCE_COMPARISON.csv", not_run_rows("SOURCE_COMPARISON", reason))
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "strict_source_only": True, "target_evaluation_run": False, "first_step_cos_B0_E1": cos_e1, "first_step_cos_B0_E2": cos_e2})
        write_report(outcome, {"first_step_cos_B0_E1": cos_e1, "first_step_cos_B0_E2": cos_e2}, first_rows, b0_raw, [], time.time() - start)
        print(f"[MetricReplay] OUTCOME={outcome}")
        return

    xb_probe, _yb, _o, _c = materialize_batch(raw_groups, plan[0])
    with torch.no_grad():
        x_probe = torch.tensor(xb_probe, dtype=torch.float32, device=device)
        b0_probe = {"z": align_init.adapt(x_probe).detach().cpu(), "w": torch.tensor(w0).detach().cpu(), "params": {n: p.detach().cpu().clone() for n, p in align_init.named_parameters()}}
    configs = [
        ("E1_OrdinarySingleContrast", "ordinary"),
        ("E2_CanonicalMetricReplay", "canonical"),
        ("E3_TimeShuffledMetric", "shuffled"),
        ("E4_RandomStateMetric", "random"),
    ]
    models, payloads = {}, {}
    for variant, mode in configs:
        model, payload = train_variant(variant, mode, raw_groups, plan, align_init, w0, metric_payload, b0_probe, plan[0], device, args.smoke)
        models[variant] = model
        payloads[variant] = payload
    raw_more, per_more = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    raw = b0_raw + raw_more
    per = b0_per + per_more
    source_rows = [{"variant": "B0_Canonical", "source_P-AUROC": float("nan"), "source_P-AP": float("nan"), "source_G": float("nan"), "source_total_loss": float("nan"), "source_hardnormal_loss": float("nan"), "contrast_norm": float(np.linalg.norm(epoch_v[10]))}]
    for variant, _mode in configs:
        source_rows.append({"variant": variant, **source_metrics(raw_groups, models[variant], device)})
    contrast_rows = metric_payload["trajectory"].copy()
    align_rows = []
    for payload in payloads.values():
        contrast_rows.extend(payload["trajectory"])
        align_rows.extend(payload["alignment_rows"])
    outcome, stats = decide(raw, first_rows)
    write_csv(OUT / "METRIC_TRAJECTORY.csv", metric_payload["trajectory"])
    write_csv(OUT / "CONTRAST_TRAJECTORY.csv", contrast_rows)
    write_csv(OUT / "ALIGNMENT_TRAJECTORY.csv", align_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", per)
    write_csv(OUT / "SOURCE_COMPARISON.csv", source_rows)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "stats": stats, "strict_source_only": True, "target_tuning": False, "target_statistics": False, "seed": SEED, "lr": LR, "epochs": EPOCHS, "top_neg_frac": TOP_NEG_FRAC, "runtime_sec": time.time() - start})
    write_report(outcome, stats, first_rows, raw, source_rows, time.time() - start)
    print(f"[MetricReplay] OUTCOME={outcome}")
    print(f"[MetricReplay] wrote {OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
