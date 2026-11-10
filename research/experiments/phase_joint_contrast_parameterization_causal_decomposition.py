import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.external_sota_gate import apply_gate
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC, collect_source_patch_pool
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, RANK, SEED, WEIGHT_DECAY, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np


OUT = Path("results/phase_joint_contrast_parameterization_causal_decomposition/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
HARD_OUT = Path("results/phase_hard_normal_contrast/formal")
TARGETS = ("MVTec", "BTAD", "CVC-ClinicDB")
EPS = 1e-12
EPOCHS = 10
HISTORICAL_B0_G = {"MVTec": 61.4250, "BTAD": 72.5939, "CVC-ClinicDB": 65.0998}


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


def sigmoid_inverse(x):
    x = float(np.clip(x, 1e-6, 1.0 - 1e-6))
    return float(math.log(x / (1.0 - x)))


def load_setup(device):
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    m_c, _d_c = decompose(states_c)
    _m_t, d_t = decompose(states_t)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    init_states = hybrid_states(m_c, alpha * init_v).astype(np.float32)
    align_init = make_model(states_c, corrupt_payload, device).eval()
    dataset_name, dataset, indices, cls, patch, y_img, cat_img = load_full_visa_source_lazy(False)
    return q_pc1, corrupt_payload, init_states, alpha, init_v, align_init, dataset_name, dataset, indices, cls, patch, y_img, cat_img


def collect_raw_source_patch_pool(slots, labels_by_pos, patch, q_pc1):
    rng = np.random.default_rng(SEED)
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    by_key = {}
    audit = []
    for cat, positions in slots.items():
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
            image_is_anomaly = bool(y.max() > 0)
            z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
            z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
            pools = []
            if image_is_anomaly:
                pools.append(("anomaly_mask_out", np.flatnonzero(y == 0)))
                pools.append(("anomaly_mask", np.flatnonzero(y == 1)))
            else:
                pools.append(("normal_image", np.flatnonzero(y == 0)))
            for origin_name, ids in pools:
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(part.shape[0] for part in parts[origin_name])
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin_name].append(z0[ids])
        xs, ys, origins = [], [], []
        for origin_name, cls_id in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if not parts[origin_name]:
                continue
            arr = np.concatenate(parts[origin_name], axis=0).astype(np.float32)
            xs.append(arr)
            ys.extend([cls_id] * arr.shape[0])
            origins.extend([origin_name] * arr.shape[0])
        if not xs or len(set(ys)) < 2:
            raise RuntimeError(f"{cat}: insufficient source patch samples.")
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.asarray(ys, dtype=np.float32)
        origin = np.asarray(origins, dtype=object)
        by_key[cat] = {"x": x, "y": y, "origin": origin}
        audit.append({"category": cat, "samples": int(x.shape[0]), "positive_samples": int(y.sum()), "negative_samples": int((y == 0).sum())})
    return by_key, audit


def build_batch_plan(groups, smoke):
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 2)
    rng = np.random.default_rng(SEED)
    names = list(groups)
    per_cat = max(2, BATCH_SIZE // max(1, len(names)))
    per_cls = max(1, per_cat // 2)
    plan = []
    for epoch in range(1, EPOCHS + 1):
        for step in range(steps_per_epoch):
            entries = []
            for cat in names:
                data = groups[cat]
                for cls_id in (0, 1):
                    ids = np.flatnonzero(data["y"] == cls_id)
                    choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
                    entries.extend((cat, int(i)) for i in choice)
            order = rng.permutation(len(entries))
            plan.append({"epoch": epoch, "step": step + 1, "entries": [entries[int(i)] for i in order]})
    return plan


def materialize_batch(groups, spec):
    xs, ys, origins, cats = [], [], [], []
    for cat, idx in spec["entries"]:
        data = groups[cat]
        xs.append(data["x"][idx])
        ys.append(data["y"][idx])
        origins.append(data["origin"][idx])
        cats.append(cat)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), np.asarray(origins, dtype=object), np.asarray(cats, dtype=object)


class JointBase(nn.Module):
    def __init__(self, align_init):
        super().__init__()
        self.a = nn.Linear(1024, RANK, bias=False)
        self.b = nn.Linear(RANK, 1024, bias=False)
        self.a.load_state_dict(align_init.a.state_dict())
        self.b.load_state_dict(align_init.b.state_dict())

    def adapt(self, x):
        return F.normalize(x + self.b(self.a(x)), dim=-1)


class JointTwoState(JointBase):
    def __init__(self, align_init, states):
        super().__init__(align_init)
        self.head = nn.Parameter(torch.tensor(states, dtype=torch.float32))

    def effective_w(self):
        h = F.normalize(self.head, dim=-1)
        return h[1] - h[0]

    def logits(self, x):
        z = self.adapt(x)
        return z @ self.effective_w(), z


class JointDirect(JointBase):
    def __init__(self, align_init, w0):
        super().__init__(align_init)
        self.w = nn.Parameter(torch.tensor(w0, dtype=torch.float32))

    def logits(self, x):
        z = self.adapt(x)
        return z @ self.w, z

    def effective_w(self):
        return self.w


class JointWeightNorm(JointBase):
    def __init__(self, align_init, w0, rmax):
        super().__init__(align_init)
        n = float(np.linalg.norm(w0))
        self.v = nn.Parameter(torch.tensor(unit_np(w0), dtype=torch.float32))
        self.rho = nn.Parameter(torch.tensor(sigmoid_inverse(n / rmax), dtype=torch.float32))
        self.rmax = float(rmax)

    def effective_w(self):
        return (self.rmax * torch.sigmoid(self.rho)) * F.normalize(self.v, dim=0)

    def logits(self, x):
        z = self.adapt(x)
        return z @ self.effective_w(), z


def hardnormal_loss(logits, y, rng, mode="hard"):
    task_loss, bce, dice = balanced_loss(logits, y)
    neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
    if mode == "none" or len(neg_idx) == 0:
        return task_loss, bce, dice, torch.zeros((), dtype=logits.dtype, device=logits.device), neg_idx[:0]
    scores = logits.detach()[neg_idx]
    if mode == "hard":
        selected_local = torch.topk(scores, k=k, largest=True).indices
    else:
        selected_local = torch.tensor(rng.permutation(len(neg_idx))[:k], dtype=torch.long, device=logits.device)
    selected = neg_idx[selected_local]
    extra = F.softplus(logits[selected]).mean()
    return task_loss + extra, bce, dice, extra, selected


def source_metrics(groups, model, device):
    y_all, s_all = [], []
    losses = []
    rng = np.random.default_rng(SEED)
    model.eval()
    with torch.no_grad():
        for cat in groups:
            x = torch.tensor(groups[cat]["x"], dtype=torch.float32, device=device)
            y = torch.tensor(groups[cat]["y"], dtype=torch.float32, device=device)
            logits, _ = model.logits(x)
            loss, _bce, _dice, extra, _sel = hardnormal_loss(logits, y, rng, "hard")
            losses.append((float(loss.cpu()), float(extra.cpu())))
            y_all.append(groups[cat]["y"])
            s_all.append(logits.detach().cpu().numpy())
    y = np.concatenate(y_all)
    s = np.concatenate(s_all)
    from sklearn.metrics import average_precision_score, roc_auc_score
    auc = float(roc_auc_score(y, s) * 100.0)
    ap = float(average_precision_score(y, s) * 100.0)
    return {"source_P-AUROC": auc, "source_P-AP": ap, "source_G": 0.5 * (auc + ap), "source_final_loss": float(np.mean([v[0] for v in losses])), "source_hardnormal_loss": float(np.mean([v[1] for v in losses]))}


def state_geometry(model):
    with torch.no_grad():
        h = F.normalize(model.head, dim=-1)
        a = h[1].detach().cpu().numpy().astype(np.float32)
        b = h[0].detach().cpu().numpy().astype(np.float32)
        raw_a = model.head[1].detach().cpu().numpy().astype(np.float32)
        raw_b = model.head[0].detach().cpu().numpy().astype(np.float32)
    c = 0.5 * (a + b)
    d = 0.5 * (a - b)
    return raw_a, raw_b, a, b, c, d, a - b


def preconditioner_diag(model, groups, batch_spec, device):
    if not isinstance(model, JointTwoState):
        return {}
    xb, yb, _origin, _cats = materialize_batch(groups, batch_spec)
    x = torch.tensor(xb, dtype=torch.float32, device=device)
    y = torch.tensor(yb, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = model.adapt(x).detach()
        w_eff = model.effective_w().detach()
    w_var = w_eff.clone().detach().requires_grad_(True)
    logits = z @ w_var
    rng = np.random.default_rng(SEED)
    loss, _bce, _dice, _extra, _sel = hardnormal_loss(logits, y, rng, "hard")
    loss.backward()
    gw = w_var.grad.detach()
    raw_a, raw_b, a_np, b_np, _c, d_np, _w_np = state_geometry(model)
    a = torch.tensor(a_np, dtype=torch.float32, device=device)
    b = torch.tensor(b_np, dtype=torch.float32, device=device)
    sa_norm = max(float(np.linalg.norm(raw_a)), EPS)
    sn_norm = max(float(np.linalg.norm(raw_b)), EPS)
    gpre = ((gw - a * torch.dot(a, gw)) / (sa_norm * sa_norm)) + ((gw - b * torch.dot(b, gw)) / (sn_norm * sn_norm))
    dh = F.normalize(torch.tensor(d_np, dtype=torch.float32, device=device), dim=0)
    gt = gw - dh * torch.dot(dh, gw)
    def coss(u, v):
        return float(F.cosine_similarity(u.flatten(), v.flatten(), dim=0).detach().cpu())
    vals = np.linalg.eigvalsh(((np.eye(1024) - np.outer(a_np, a_np)) / (sa_norm * sa_norm)) + ((np.eye(1024) - np.outer(b_np, b_np)) / (sn_norm * sn_norm)))
    vals = vals[vals > 1e-8]
    return {
        "cos_g_pre_g_w": coss(gpre, gw),
        "norm_g_pre_over_g_w": float(gpre.norm().detach().cpu() / gw.norm().clamp_min(EPS).detach().cpu()),
        "effective_condition_number": float(vals.max() / max(vals.min(), EPS)) if vals.size else float("nan"),
        "cos_g_pre_g_tangent": coss(gpre, gt),
    }


def train_variant(variant, kind, groups, plan, align_init, init_states, w0, rmax, norm_schedule, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_epoch10_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model = payload["model_obj"].to(device).eval()
        return model, payload, True
    print(f"[JointContrast] train {variant}", flush=True)
    if kind == "two_state":
        model = JointTwoState(align_init, init_states).to(device)
    elif kind in ("direct", "direct_replay"):
        model = JointDirect(align_init, w0).to(device)
    elif kind == "weight_norm":
        model = JointWeightNorm(align_init, w0, rmax).to(device)
    else:
        raise ValueError(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    contrast_rows, gauge_rows, pre_rows, curves = [], [], [], []
    step_idx = 0
    probe_batch = plan[0]
    for spec in plan:
        step_idx += 1
        xb, yb, origin, cats = materialize_batch(groups, spec)
        x = torch.tensor(xb, dtype=torch.float32, device=device)
        y = torch.tensor(yb, dtype=torch.float32, device=device)
        logits, _z = model.logits(x)
        loss, bce, dice, extra, selected = hardnormal_loss(logits, y, rng, "hard")
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if kind == "direct":
                n = model.w.norm().clamp_min(EPS)
                if float(n.cpu()) > rmax:
                    model.w.mul_(rmax / n)
            elif kind == "direct_replay":
                target_norm = float(norm_schedule[step_idx - 1])
                model.w.mul_(target_norm / model.w.norm().clamp_min(EPS))
        with torch.no_grad():
            w = model.effective_w().detach().cpu().numpy().astype(np.float32)
        contrast_rows.append({
            "variant": variant,
            "epoch": spec["epoch"],
            "step": spec["step"],
            "global_step": step_idx,
            "loss": float(loss.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "dice": float(dice.detach().cpu()),
            "hardnormal_loss": float(extra.detach().cpu()),
            "contrast_norm": float(np.linalg.norm(w)),
            "boundary_active": bool(np.linalg.norm(w) / rmax >= 0.999),
            "selected_hard_count": int(selected.numel()),
            "selected_normal_image_ratio": float(np.mean(origin[selected.detach().cpu().numpy()] == "normal_image")) if selected.numel() else float("nan"),
        })
        if kind == "two_state":
            raw_a, raw_b, a, b, c, d, w_np = state_geometry(model)
            prev = gauge_rows[-1] if gauge_rows else None
            gauge_rows.append({
                "variant": variant,
                "epoch": spec["epoch"],
                "step": spec["step"],
                "global_step": step_idx,
                "raw_sA_norm": float(np.linalg.norm(raw_a)),
                "raw_sN_norm": float(np.linalg.norm(raw_b)),
                "c_dot_d": float(c @ d),
                "c_norm": float(np.linalg.norm(c)),
                "d_norm": float(np.linalg.norm(d)),
                "c_norm_sq_plus_d_norm_sq": float(np.linalg.norm(c) ** 2 + np.linalg.norm(d) ** 2),
                "cos_c_prev": float(prev["c_prev_cos_ref"] @ c / max(np.linalg.norm(prev["c_prev_cos_ref"]) * np.linalg.norm(c), EPS)) if prev else float("nan"),
                "cos_d_prev": float(prev["d_prev_cos_ref"] @ d / max(np.linalg.norm(prev["d_prev_cos_ref"]) * np.linalg.norm(d), EPS)) if prev else float("nan"),
                "c_prev_cos_ref": c,
                "d_prev_cos_ref": d,
            })
        if spec["step"] == 1:
            vals = [r for r in contrast_rows if r["epoch"] == spec["epoch"]]
            curves.append({"variant": variant, "epoch": spec["epoch"], "loss": float(np.mean([r["loss"] for r in vals])), "hardnormal_loss": float(np.mean([r["hardnormal_loss"] for r in vals]))})
            if kind == "two_state":
                pr = preconditioner_diag(model, groups, probe_batch, device)
                pr.update({"variant": variant, "epoch": spec["epoch"], "global_step": step_idx})
                pre_rows.append(pr)
        if step_idx % 20 == 0 or step_idx == 1:
            print(f"[JointContrast] {variant} step={step_idx}/{len(plan)} loss={float(loss.detach().cpu()):.4f} R={float(np.linalg.norm(w)):.4f}", flush=True)
    # Drop array-only helper fields before writing CSV.
    for row in gauge_rows:
        row.pop("c_prev_cos_ref", None)
        row.pop("d_prev_cos_ref", None)
    payload = {
        "model_obj": model.eval().cpu(),
        "contrast_rows": contrast_rows,
        "gauge_rows": gauge_rows,
        "preconditioner_rows": pre_rows,
        "curves": curves,
        "variant": variant,
        "kind": kind,
    }
    torch.save(payload, ckpt)
    return model.to(device).eval(), payload, False


def adapter_output_cosine(model, b0_model, groups, device):
    xs = []
    for cat in list(groups)[:4]:
        xs.append(groups[cat]["x"][:128])
    x = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32, device=device)
    with torch.no_grad():
        z = model.adapt(x)
        z0 = b0_model.adapt(x)
        cos = F.cosine_similarity(z, z0, dim=-1).mean()
    dist = 0.0
    for name in ("a", "b"):
        p = torch.cat([v.detach().flatten().cpu() for v in getattr(model, name).parameters()])
        q = torch.cat([v.detach().flatten().cpu() for v in getattr(b0_model, name).parameters()])
        dist += float(torch.sum((p - q) ** 2))
    return float(math.sqrt(dist)), float(cos.cpu())


def reproduction_stats(summary):
    rows = []
    for r in summary:
        hist = HISTORICAL_B0_G[r["dataset"]]
        rows.append({
            "dataset": r["dataset"],
            "historical_B0_G": hist,
            "rerun_B0_G": r["B0_G"],
            "rerun_minus_historical_G": r["B0_G"] - hist,
            "abs_delta_G": abs(r["B0_G"] - hist),
        })
    return rows


def decide(summary, pre_rows):
    repro = reproduction_stats(summary)
    if max(r["abs_delta_G"] for r in repro) > 1.0:
        return "B0_REPRODUCTION_MISMATCH_PROTOCOL_NOT_VALIDATED"
    by = {r["dataset"]: r for r in summary}
    diffs = {v: np.asarray([by[ds][f"{v}_minus_B0_G"] for ds in TARGETS], dtype=np.float32) for v in ("E1", "E2", "E3")}
    mean_abs_e1 = float(np.mean(np.abs(diffs["E1"])))
    if mean_abs_e1 <= 0.5:
        return "JOINT_FEATURE_SCORER_COTRAINING_EXPLAINS_TRANSFER"
    if float(diffs["E2"].mean()) >= -0.5 and float((diffs["E2"] - diffs["E1"]).mean()) >= 1.0:
        return "TWO_STATE_CONTRAST_NORM_TRAJECTORY_EXPLAINS_TRANSFER"
    if float(np.mean(np.abs(diffs["E3"]))) <= 0.5:
        return "NORMALIZED_CONTRAST_PARAMETERIZATION_EXPLAINS_TRANSFER"
    all_weak = all(float((-diffs[v]).mean()) >= 1.0 for v in ("E1", "E2", "E3"))
    pre_ok = bool(pre_rows) and float(np.mean([r["cos_g_pre_g_w"] for r in pre_rows])) < 0.98 and float(np.mean([r["cos_g_pre_g_tangent"] for r in pre_rows])) < 0.98
    if all_weak and pre_ok:
        return "TWO_STATE_PREDICTION_NULL_GAUGE_HAS_TRANSFER_BIAS"
    return "JOINT_CONTRAST_PARAMETERIZATION_CAUSALITY_INCONCLUSIVE"


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in TARGETS:
        b0 = by[(ds, "B0_JointTwoState")]
        e1 = by[(ds, "E1_JointBoundedDirect")]
        e2 = by[(ds, "E2_JointDirectB0NormReplay")]
        e3 = by[(ds, "E3_JointWeightNormalized")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E3_G": float(e3["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E3_minus_B0_G": float(e3["G_pixel"]) - float(b0["G_pixel"]),
        })
    return rows


def contrast_geometry_rows(final_ws):
    names = list(final_ws)
    rows = []
    for name, w in final_ws.items():
        rows.append({"metric": f"norm_{name}", "value": float(np.linalg.norm(w))})
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rows.append({"metric": f"cos_{a}_{b}", "value": float(np.dot(unit_np(final_ws[a]), unit_np(final_ws[b])))})
    return rows


def write_report(summary, repro_rows, outcome, runtime, reused):
    lines = [
        "# Joint Contrast Parameterization Causal Decomposition",
        "",
        f"Outcome: `{outcome}`",
        "",
        "| Dataset | B0 Joint Two-State | E1 Direct | E2 Norm Replay | E3 WeightNorm | E1-B0 | E2-B0 | E3-B0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E3_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E3_minus_B0_G']:+.4f} |")
    lines.extend(["", "## B0 Reproduction Check", "", "| Dataset | Historical B0 G | Rerun B0 G | Delta |", "|---|---:|---:|---:|"])
    for r in repro_rows:
        lines.append(f"| {r['dataset']} | {r['historical_B0_G']:.4f} | {r['rerun_B0_G']:.4f} | {r['rerun_minus_historical_G']:+.4f} |")
    lines.extend(["", "If B0 differs from the historical formal baseline by more than 1G on any target, the causal decision tree is treated as protocol-not-validated rather than a positive mechanism claim.", "", f"Runtime: {runtime:.1f}s. Reused checkpoints: {reused}."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[JointContrast] device={device}", flush=True)
    q_pc1, corrupt_payload, init_states, alpha, init_v, align_init, _dataset_name, _dataset, indices, _cls, patch, _y_img, _cat_img = load_setup(device)
    labels_by_pos = source_label_maps(False)[0]
    slots = source_slots_by_category(_dataset, indices, args.smoke)
    groups, pool_audit = collect_raw_source_patch_pool(slots, labels_by_pos, patch, q_pc1)
    plan = build_batch_plan(groups, args.smoke)
    w0 = init_v.astype(np.float32) * float(alpha)
    rmax = 2.0
    b0_model, b0_payload, reused_b0 = train_variant("B0_JointTwoState", "two_state", groups, plan, align_init, init_states, w0, rmax, None, device, args.smoke)
    norm_schedule = [r["contrast_norm"] for r in b0_payload["contrast_rows"]]
    e1_model, e1_payload, reused_e1 = train_variant("E1_JointBoundedDirect", "direct", groups, plan, align_init, init_states, w0, rmax, norm_schedule, device, args.smoke)
    e2_model, e2_payload, reused_e2 = train_variant("E2_JointDirectB0NormReplay", "direct_replay", groups, plan, align_init, init_states, w0, rmax, norm_schedule, device, args.smoke)
    e3_model, e3_payload, reused_e3 = train_variant("E3_JointWeightNormalized", "weight_norm", groups, plan, align_init, init_states, w0, rmax, norm_schedule, device, args.smoke)
    models = {
        "B0_JointTwoState": b0_model,
        "E1_JointBoundedDirect": e1_model,
        "E2_JointDirectB0NormReplay": e2_model,
        "E3_JointWeightNormalized": e3_model,
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    summary = summarize(raw)
    repro_rows = reproduction_stats(summary)
    final_ws = {k: m.effective_w().detach().cpu().numpy().astype(np.float32) for k, m in models.items()}
    source_rows = []
    for name, model in models.items():
        row = {"variant": name, **source_metrics(groups, model, device), "final_contrast_norm": float(np.linalg.norm(final_ws[name])), "cos_with_B0_final": float(np.dot(unit_np(final_ws[name]), unit_np(final_ws["B0_JointTwoState"])))}
        source_rows.append(row)
    align_rows = []
    for name, model in models.items():
        dist, cos = adapter_output_cosine(model, b0_model, groups, device)
        align_rows.append({"variant": name, "adapter_param_l2_distance_to_B0": dist, "probe_output_feature_cosine_to_B0": cos})
    pre_rows = b0_payload["preconditioner_rows"]
    outcome = decide(summary, pre_rows)
    contrast_rows = b0_payload["contrast_rows"] + e1_payload["contrast_rows"] + e2_payload["contrast_rows"] + e3_payload["contrast_rows"]
    gauge_rows = b0_payload["gauge_rows"]
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "CONTRAST_TRAJECTORY.csv", contrast_rows)
    write_csv(OUT / "GAUGE_TRAJECTORY.csv", gauge_rows)
    write_csv(OUT / "PRECONDITIONER_DIAGNOSTICS.csv", pre_rows)
    write_csv(OUT / "JOINT_TRAIN_GEOMETRY.csv", source_rows)
    write_csv(OUT / "ALIGNMENT_DIAGNOSTICS.csv", align_rows)
    write_csv(OUT / "B0_REPRODUCTION.csv", repro_rows)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_audit)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", contrast_geometry_rows(final_ws))
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "runtime_sec": runtime, "strict_source_only": True, "target_training": False, "target_statistics": False, "seed": SEED, "lr": LR, "epochs": EPOCHS, "top_neg_frac": TOP_NEG_FRAC, "b0_reproduction": repro_rows, "reused": {"B0": reused_b0, "E1": reused_e1, "E2": reused_e2, "E3": reused_e3}})
    write_report(summary, repro_rows, outcome, runtime, {"B0": reused_b0, "E1": reused_e1, "E2": reused_e2, "E3": reused_e3})
    try:
        apply_gate(OUT, "B0_JointTwoState", "B0_JointTwoState")
    except Exception as exc:
        print(f"[JointContrast] SOTA gate warning: {exc}", flush=True)
    print(json.dumps({"outcome": outcome, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
