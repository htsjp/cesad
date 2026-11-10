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

from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_spherical_state_trajectory_averaging import (
    HISTORICAL_B0,
    TOP_NEG_FRAC,
    balanced_batch,
    load_payloads,
    prepare_groups,
    train_b0_trajectory,
)
from experiments.phase_state_pair_decomposition import hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_prediction_null_state_gauge_causal_test/formal")
EPOCHS = 10
TARGETS = ("MVTec", "BTAD", "CVC-ClinicDB")
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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_states_t(states):
    return F.normalize(states, dim=-1)


def effective_w_t(states):
    s = norm_states_t(states)
    return s[1] - s[0]


def logits_from_states(z, states):
    s = norm_states_t(states)
    l_n = z @ s[0]
    l_a = z @ s[1]
    return l_n, l_a, l_a - l_n


def hard_loss_parts(logits, y):
    task_loss, bce, dice = balanced_loss(logits, y)
    neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
    selected = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
    hard = F.softplus(logits[selected]).mean()
    return task_loss, bce, dice, hard, selected


def source_metrics(groups, states_np, device):
    y_all, s_all = [], []
    losses = []
    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        for data in groups.values():
            z = torch.tensor(data["x"], dtype=torch.float32, device=device)
            y = torch.tensor(data["y"], dtype=torch.float32, device=device)
            _ln, _la, logits = logits_from_states(z, states)
            task, _bce, _dice, hard, _sel = hard_loss_parts(logits, y)
            losses.append((float((task + hard).cpu()), float(hard.cpu())))
            y_all.append(data["y"])
            s_all.append(logits.detach().cpu().numpy().astype(np.float32))
    y_np = np.concatenate(y_all)
    s_np = np.concatenate(s_all)
    auc = safe_auc(y_np, s_np)
    ap = safe_ap(y_np, s_np)
    return {
        "source_P-AUROC": auc,
        "source_P-AP": ap,
        "source_G": 0.5 * (auc + ap),
        "source_total_loss": float(np.mean([v[0] for v in losses])),
        "source_hardnormal_loss": float(np.mean([v[1] for v in losses])),
    }


def decompose_normalized(states_np):
    s = np.asarray(states_np, dtype=np.float32)
    r_n = float(np.linalg.norm(s[0]))
    r_a = float(np.linalg.norm(s[1]))
    b = s[0] / max(r_n, EPS)
    a = s[1] / max(r_a, EPS)
    c = (a + b) / 2.0
    d = (a - b) / 2.0
    return r_a, r_n, a.astype(np.float32), b.astype(np.float32), c.astype(np.float32), d.astype(np.float32)


def rotate_gauge(states_np, seed):
    r_a, r_n, _a, _b, c, d = decompose_normalized(states_np)
    d_hat = unit_np(d)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d.shape[0]).astype(np.float32)
    v = v - float(v @ d_hat) * d_hat
    u = unit_np(v)
    c_new = float(np.linalg.norm(c)) * u
    a_new = c_new + d
    b_new = c_new - d
    out = np.stack([r_n * b_new, r_a * a_new]).astype(np.float32)
    return out


def gauge_checks(name, states_np, base_states):
    r_a, r_n, a, b, c, d = decompose_normalized(states_np)
    _r_a0, _r_n0, a0, b0, _c0, _d0 = decompose_normalized(base_states)
    return [
        {"variant": name, "metric": "raw_sA_norm", "value": r_a},
        {"variant": name, "metric": "raw_sN_norm", "value": r_n},
        {"variant": name, "metric": "abs_c_dot_d", "value": abs(float(c @ d))},
        {"variant": name, "metric": "abs_norm_identity_error", "value": abs(float(c @ c + d @ d) - 1.0)},
        {"variant": name, "metric": "norm_a_prime", "value": float(np.linalg.norm(a))},
        {"variant": name, "metric": "norm_b_prime", "value": float(np.linalg.norm(b))},
        {"variant": name, "metric": "max_abs_delta_a_minus_b", "value": float(np.max(np.abs((a - b) - (a0 - b0))))},
    ]


def preservation_gate(states_by_variant, batch, groups, device):
    xb, yb = batch
    z = torch.tensor(xb, dtype=torch.float32, device=device)
    y = torch.tensor(yb, dtype=torch.float32, device=device)
    rows = []
    cache = {}
    for name, states_np in states_by_variant.items():
        states = torch.tensor(states_np, dtype=torch.float32, device=device)
        l_n, l_a, logits = logits_from_states(z, states)
        task, _bce, _dice, hard, selected = hard_loss_parts(logits, y)
        total = task + hard
        cache[name] = {
            "lN": l_n.detach().cpu().numpy(),
            "lA": l_a.detach().cpu().numpy(),
            "logits": logits.detach().cpu().numpy(),
            "prob": torch.sigmoid(logits).detach().cpu().numpy(),
            "task": float(task.detach().cpu()),
            "hard": float(hard.detach().cpu()),
            "total": float(total.detach().cpu()),
            "selected": selected.detach().cpu().numpy(),
        }
    b0 = cache["B0_CanonicalGauge"]
    x_sample, _y_sample = sample_global_source(groups, 10000)
    z_global = torch.tensor(x_sample, dtype=torch.float32, device=device)
    global_logits = {}
    for name, states_np in states_by_variant.items():
        states = torch.tensor(states_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            global_logits[name] = logits_from_states(z_global, states)[2].detach().cpu().numpy()
    for name, vals in cache.items():
        rows.append({
            "variant": name,
            "max_abs_normal_logit_diff_vs_B0": float(np.max(np.abs(vals["lN"] - b0["lN"]))),
            "max_abs_anomaly_logit_diff_vs_B0": float(np.max(np.abs(vals["lA"] - b0["lA"]))),
            "max_abs_delta_logit_diff_vs_B0": float(np.max(np.abs(vals["logits"] - b0["logits"]))),
            "max_abs_probability_diff_vs_B0": float(np.max(np.abs(vals["prob"] - b0["prob"]))),
            "statealigned_loss": vals["task"],
            "hardnormal_loss": vals["hard"],
            "total_loss": vals["total"],
            "loss_diff_vs_B0": vals["total"] - b0["total"],
            "hard_set_overlap_vs_B0": float(len(np.intersect1d(vals["selected"], b0["selected"])) / max(1, len(b0["selected"]))),
            "global_10000_max_abs_logit_diff_vs_B0": float(np.max(np.abs(global_logits[name] - global_logits["B0_CanonicalGauge"]))),
        })
    passed = all(
        r["max_abs_delta_logit_diff_vs_B0"] <= 1e-6
        and abs(r["loss_diff_vs_B0"]) <= 1e-7
        and r["hard_set_overlap_vs_B0"] == 1.0
        and r["global_10000_max_abs_logit_diff_vs_B0"] <= 1e-6
        for r in rows
    )
    return rows, passed


def sample_global_source(groups, n):
    xs, ys = [], []
    for data in groups.values():
        xs.append(data["x"])
        ys.append(data["y"])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    take = min(int(n), len(y))
    rng = np.random.default_rng(909)
    ids = rng.choice(np.arange(len(y)), size=take, replace=False)
    return x[ids], y[ids]


def first_step_delta(states_np, batch, device):
    xb, yb = batch
    seed_all(SEED)
    params = torch.nn.Parameter(torch.tensor(states_np, dtype=torch.float32, device=device))
    z = torch.tensor(xb, dtype=torch.float32, device=device)
    y = torch.tensor(yb, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([params], lr=LR, weight_decay=0.0)
    w_before = effective_w_t(params).detach().cpu().numpy().astype(np.float32)
    logits = logits_from_states(z, params)[2]
    task, _bce, _dice, hard, _sel = hard_loss_parts(logits, y)
    loss = task + hard
    opt.zero_grad()
    loss.backward()
    grad = params.grad.detach().cpu().numpy().astype(np.float32)
    opt.step()
    w_after = effective_w_t(params).detach().cpu().numpy().astype(np.float32)
    return w_after - w_before, grad


def effective_w_grad(batch, states_np, device):
    xb, yb = batch
    z = torch.tensor(xb, dtype=torch.float32, device=device)
    y = torch.tensor(yb, dtype=torch.float32, device=device)
    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    w0 = effective_w_t(states).detach()
    w = w0.clone().requires_grad_(True)
    logits = z @ w
    task, _bce, _dice, hard, _sel = hard_loss_parts(logits, y)
    loss = task + hard
    loss.backward()
    return w.grad.detach().cpu().numpy().astype(np.float32)


def induced_metric_rows(states_by_variant, batch, device):
    g_w = effective_w_grad(batch, states_by_variant["B0_CanonicalGauge"], device)
    rows = []
    mg = {}
    for name, states_np in states_by_variant.items():
        r_a, r_n, a, b, _c, _d = decompose_normalized(states_np)
        eye_g = g_w
        mg_np = (eye_g - a * float(a @ eye_g)) / max(r_a * r_a, EPS)
        mg_np += (eye_g - b * float(b @ eye_g)) / max(r_n * r_n, EPS)
        mg[name] = mg_np.astype(np.float32)
        rows.append({
            "variant": name,
            "cos_Mg_gw": cos_np(mg_np, g_w),
            "norm_Mg_over_gw": float(np.linalg.norm(mg_np) / max(np.linalg.norm(g_w), EPS)),
        })
    for a_name, b_name in combinations(states_by_variant, 2):
        rows.append({"variant": f"{a_name}__{b_name}", "cos_Mg_pair": cos_np(mg[a_name], mg[b_name]), "norm_Mg_ratio": float(np.linalg.norm(mg[a_name]) / max(np.linalg.norm(mg[b_name]), EPS))})
    return rows


def first_step_rows(states_by_variant, batch, device):
    deltas = {}
    rows = []
    for name, states_np in states_by_variant.items():
        delta, grad = first_step_delta(states_np, batch, device)
        deltas[name] = delta
        rows.append({"variant": name, "delta_w_norm": float(np.linalg.norm(delta)), "state_grad_norm": float(np.linalg.norm(grad))})
    for a_name, b_name in combinations(states_by_variant, 2):
        rows.append({
            "variant": f"{a_name}__{b_name}",
            "cos_delta_w": cos_np(deltas[a_name], deltas[b_name]),
            "delta_w_norm_ratio": float(np.linalg.norm(deltas[a_name]) / max(np.linalg.norm(deltas[b_name]), EPS)),
        })
    inactive = all(
        row.get("cos_delta_w", 0.0) >= 0.9999 and abs(row.get("delta_w_norm_ratio", 1.0) - 1.0) <= 1e-3
        for row in rows
        if "cos_delta_w" in row
    )
    return rows, inactive


def train_gauge_variant(variant, states_init, groups, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        return torch.load(ckpt, map_location=device, weights_only=False)
    print(f"[StateGauge] train {variant}", flush=True)
    seed_all(SEED)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    states = torch.nn.Parameter(torch.tensor(states_init, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([states], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    trajectory = []
    b0_init_norm = norm_states_t(torch.tensor(states_init, dtype=torch.float32, device=device)).detach().cpu().numpy()
    d0 = unit_np(b0_init_norm[1] - b0_init_norm[0])
    c0 = ((b0_init_norm[1] + b0_init_norm[0]) / 2.0).astype(np.float32)
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _step in range(steps_per_epoch):
            xb, yb = balanced_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            logits = logits_from_states(z, states)[2]
            task, bce, dice, hard, _sel = hard_loss_parts(logits, y)
            loss = task + hard
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task.detach().cpu()), float(hard.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        with torch.no_grad():
            ns = norm_states_t(states).detach().cpu().numpy().astype(np.float32)
            d = unit_np(ns[1] - ns[0])
            c = ((ns[1] + ns[0]) / 2.0).astype(np.float32)
            w = ns[1] - ns[0]
        trajectory.append({
            "variant": variant,
            "epoch": epoch,
            "source_total_loss": float(arr[:, 0].mean()),
            "source_statealigned_loss": float(arr[:, 1].mean()),
            "source_hardnormal_loss": float(arr[:, 2].mean()),
            "source_bce": float(arr[:, 3].mean()),
            "source_dice": float(arr[:, 4].mean()),
            "d_norm": float(np.linalg.norm((ns[1] - ns[0]) / 2.0)),
            "c_norm": float(np.linalg.norm(c)),
            "cos_d_t_d_init": cos_np(d, d0),
            "cos_c_t_c_init": cos_np(c, c0),
            "effective_contrast_norm": float(np.linalg.norm(w)),
        })
        print(f"[StateGauge] {variant} epoch {epoch}/{EPOCHS} loss={trajectory[-1]['source_total_loss']:.4f}", flush=True)
    final_states = states.detach().cpu().numpy().astype(np.float32)
    payload = {"variant": variant, "states": final_states, "trajectory": trajectory}
    torch.save(payload, ckpt)
    return payload


def load_b0_raw_metrics():
    path = Path("results/phase_spherical_state_trajectory_averaging/formal/RAW_METRICS.csv")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["variant"] != "B0_Epoch10":
                continue
            out = dict(row)
            out["variant"] = "B0_Canonical"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            rows.append(out)
    return rows


def reproduction_rows(raw):
    rows = []
    by = {r["dataset"]: r for r in raw}
    for ds, hist in HISTORICAL_B0.items():
        got = float(by[ds]["G_pixel"])
        rows.append({"dataset": ds, "historical_G": hist, "rerun_G": got, "delta_G": got - hist, "passed": bool(abs(got - hist) <= 0.05)})
    return rows


def decide(raw_rows, source_rows, first_rows, b0_states_np, final_states):
    by = {(r["dataset"], r["variant"]): float(r["G_pixel"]) for r in raw_rows}
    b0_mean = np.mean([by[(ds, "B0_Canonical")] for ds in TARGETS])
    diffs = []
    any_2g = False
    for var in ("E1_GaugeSeed101", "E2_GaugeSeed202"):
        vals = [by[(ds, var)] - by[(ds, "B0_Canonical")] for ds in TARGETS]
        diffs.extend(vals)
        any_2g = any_2g or any(abs(v) >= 2.0 for v in vals)
    mean_abs = float(np.mean(np.abs(diffs)))
    max_source_g = max(float(r["source_G"]) for r in source_rows)
    min_source_g = min(float(r["source_G"]) for r in source_rows)
    max_loss = max(float(r["source_total_loss"]) for r in source_rows)
    min_loss = min(float(r["source_total_loss"]) for r in source_rows)
    source_near = (max_source_g - min_source_g) <= 1.0 and ((max_loss - min_loss) / max(abs(min_loss), EPS)) <= 0.05
    pair_rows = [r for r in first_rows if "cos_delta_w" in r and "B0_CanonicalGauge__" in r["variant"]]
    first_diff = any(float(r["cos_delta_w"]) < 0.98 for r in pair_rows)
    final_cos = []
    b0_d = unit_np((norm_states_t(torch.tensor(b0_states_np)).numpy()[1] - norm_states_t(torch.tensor(b0_states_np)).numpy()[0]).astype(np.float32))
    for states in final_states.values():
        ns = norm_states_t(torch.tensor(states)).numpy()
        final_cos.append(cos_np(unit_np(ns[1] - ns[0]), b0_d))
    if first_diff and (mean_abs >= 1.0 or any_2g) and source_near:
        return "PREDICTION_NULL_GAUGE_CAUSALLY_CONTROLS_TRANSFER"
    if first_diff and mean_abs < 0.5:
        return "GAUGE_CHANGES_OPTIMIZATION_NOT_TRANSFER"
    if (max_source_g - min_source_g) > 1.0:
        return "GAUGE_CONTROLS_GENERAL_OPTIMIZATION_BASIN"
    if min(final_cos) >= 0.995 and mean_abs < 0.5:
        return "GAUGE_INITIALIZATION_IS_FORGOTTEN"
    return "GAUGE_CHANGES_OPTIMIZATION_NOT_TRANSFER" if first_diff else "PREDICTION_NULL_GAUGE_OPTIMIZATION_INACTIVE"


def write_report(outcome, repro, first_rows, source_rows, raw_rows, runtime):
    lines = [
        "# Prediction-Null State Gauge - Function-Preserving Causal Test",
        "",
        f"Outcome: `{outcome}`",
        "",
        "B0 reproduction used the canonical `phase_spherical_state_trajectory_averaging` path. Gauge variants were initialized to preserve normalized state contrast exactly, then trained as raw two-state gauge branches on the same frozen aligned source representation.",
        "",
        "## B0 Reproduction",
    ]
    for r in repro:
        lines.append(f"- {r['dataset']}: rerun G {float(r['rerun_G']):.4f}, historical {float(r['historical_G']):.4f}, delta {float(r['delta_G']):+.4f}, passed={r['passed']}")
    lines.append("")
    lines.append("## First Step")
    for r in first_rows:
        if "cos_delta_w" in r:
            lines.append(f"- {r['variant']}: cos delta-w {float(r['cos_delta_w']):.6f}, norm ratio {float(r['delta_w_norm_ratio']):.6f}")
    lines.append("")
    lines.append("## Target G")
    for r in raw_rows:
        if r["dataset"] in TARGETS:
            lines.append(f"- {r['variant']} on {r['dataset']}: G {float(r['G_pixel']):.4f}")
    lines.append("")
    lines.append("## Source")
    for r in source_rows:
        lines.append(f"- {r['variant']}: source G {float(r['source_G']):.4f}, loss {float(r['source_total_loss']):.4f}, hard {float(r['source_hardnormal_loss']):.4f}")
    lines.append("")
    lines.append(f"Runtime seconds: {runtime:.1f}")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[StateGauge] device={device}", flush=True)
    seed_all(SEED)
    q_pc1, corrupt_payload, states_c, m0, r0 = load_payloads(device)
    init_v = unit_np(np.load(Path("results/phase_hard_normal_contrast/formal/contrast_vectors.npz"))["B0"].astype(np.float32))
    groups, _pool_rows = prepare_groups(q_pc1, states_c, corrupt_payload, device, args.smoke)
    epoch_raw, epoch_v, trajectory_b0, _reused = train_b0_trajectory(groups, m0, r0, init_v, device, args.smoke)
    b0_states_init = hybrid_states(m0, r0 * init_v).astype(np.float32)
    b0_model = make_model(hybrid_states(m0, r0 * epoch_v[10]), corrupt_payload, device)
    b0_eval, b0_percat = evaluate_pc1_removed({"B0_Canonical": b0_model}, q_pc1, device, args.smoke)
    repro = reproduction_rows(b0_eval) if not args.smoke else [{"dataset": r["dataset"], "historical_G": HISTORICAL_B0.get(r["dataset"], float("nan")), "rerun_G": r["G_pixel"], "delta_G": float("nan"), "passed": True} for r in b0_eval]
    write_csv(OUT / "B0_REPRODUCTION.csv", repro)
    if not all(bool(r["passed"]) for r in repro):
        outcome = "GAUGE_TEST_B0_REPRODUCTION_FAILED"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "strict_source_only": True, "target_tuning": False})
        write_report(outcome, repro, [], [], b0_eval, time.time() - start)
        return

    states_by_variant = {
        "B0_CanonicalGauge": b0_states_init,
        "E1_GaugeSeed101": rotate_gauge(b0_states_init, 101),
        "E2_GaugeSeed202": rotate_gauge(b0_states_init, 202),
    }
    gauge_rows = []
    for name, states_np in states_by_variant.items():
        gauge_rows.extend(gauge_checks(name, states_np, b0_states_init))
    first_batch_np = balanced_batch(groups, np.random.default_rng(SEED), BATCH_SIZE)[:2]
    func_rows, func_ok = preservation_gate(states_by_variant, first_batch_np, groups, device)
    write_csv(OUT / "FUNCTION_PRESERVATION.csv", gauge_rows + func_rows)
    if not func_ok:
        outcome = "GAUGE_TRANSFORMATION_NOT_FUNCTION_PRESERVING"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "strict_source_only": True, "target_tuning": False})
        write_report(outcome, repro, [], [], b0_eval, time.time() - start)
        return

    first_rows, inactive = first_step_rows(states_by_variant, first_batch_np, device)
    metric_rows = induced_metric_rows(states_by_variant, first_batch_np, device)
    write_csv(OUT / "FIRST_STEP_GEOMETRY.csv", first_rows)
    write_csv(OUT / "INDUCED_METRIC.csv", metric_rows)
    if inactive:
        outcome = "PREDICTION_NULL_GAUGE_OPTIMIZATION_INACTIVE"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "strict_source_only": True, "target_tuning": False})
        write_report(outcome, repro, first_rows, [], b0_eval, time.time() - start)
        return

    train_variants = {
        "E1_GaugeSeed101": states_by_variant["E1_GaugeSeed101"],
        "E2_GaugeSeed202": states_by_variant["E2_GaugeSeed202"],
        "E3_ScaleGauge_x2": (2.0 * states_by_variant["B0_CanonicalGauge"]).astype(np.float32),
    }
    payloads = {name: train_gauge_variant(name, init, groups, device, args.smoke) for name, init in train_variants.items()}
    final_models = {name: make_model(payload["states"], corrupt_payload, device) for name, payload in payloads.items()}
    raw_more, per_more = evaluate_pc1_removed(final_models, q_pc1, device, args.smoke)
    raw = b0_eval + raw_more
    percat = b0_percat + per_more
    source_rows = [{
        "variant": "B0_Canonical",
        **source_metrics(groups, hybrid_states(m0, r0 * epoch_v[10]).astype(np.float32), device),
    }]
    for name, payload in payloads.items():
        source_rows.append({"variant": name, **source_metrics(groups, payload["states"], device)})
    traj = []
    for r in trajectory_b0:
        row = dict(r)
        row["variant"] = "B0_Canonical"
        traj.append(row)
    for payload in payloads.values():
        traj.extend(payload["trajectory"])
    write_csv(OUT / "TRAINING_TRAJECTORY.csv", traj)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "SOURCE_COMPARISON.csv", source_rows)
    outcome = decide(raw, source_rows, first_rows, b0_states_init, {k: v["states"] for k, v in payloads.items() if k != "E3_ScaleGauge_x2"})
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "strict_source_only": True,
        "target_tuning": False,
        "target_statistics": False,
        "seed": SEED,
        "lr": LR,
        "epochs": EPOCHS,
        "top_neg_frac": TOP_NEG_FRAC,
        "note": "B0 reproduction directly used canonical trajectory training; E1/E2/E3 are raw-state gauge causal branches.",
        "runtime_sec": time.time() - start,
    })
    write_report(outcome, repro, first_rows, source_rows, raw, time.time() - start)
    print(f"[StateGauge] OUTCOME={outcome}")
    print(f"[StateGauge] wrote {OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
