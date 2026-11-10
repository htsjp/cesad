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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_hard_normal_contrast import collect_source_patch_pool
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np, patch_labels


OUT = Path("results/phase_spherical_state_trajectory_averaging/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPOCHS = 10
AVG_EPOCHS = (6, 7, 8, 9, 10)
TOP_NEG_FRAC = 0.10
TARGETS = ("MVTec", "BTAD", "CVC-ClinicDB")
HISTORICAL_B0 = {
    "MVTec": 61.42496625199354,
    "BTAD": 72.59389890285473,
    "CVC-ClinicDB": 65.09979163145577,
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


def safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a, b):
    if len(a) < 2:
        return float("nan")
    return float(spearmanr(a, b, nan_policy="omit").correlation)


def cos_np(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def angle_deg(a, b):
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos_np(a, b))))))


def load_payloads(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = torch.load(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", map_location=device, weights_only=False)
    corrupt_payload = torch.load(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", map_location=device, weights_only=False)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    return q_pc1, corrupt_payload, states_c, m_c.astype(np.float32), alpha


def prepare_groups(q_pc1, states_c, corrupt_payload, device, smoke):
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, smoke)
    return collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)


def balanced_batch(groups, rng, total_batch):
    names = list(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys = [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order]


def compute_source_metrics(run, epoch, groups, raw_v, m0, r0, final_training_row=None):
    xs, ys = [], []
    for data in groups.values():
        xs.append(data["x"])
        ys.append(data["y"])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    v = unit_np(raw_v)
    with torch.no_grad():
        logits = contrast_logits(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(m0, dtype=torch.float32),
            r0,
            torch.tensor(v, dtype=torch.float32),
        )
        yt = torch.tensor(y, dtype=torch.float32)
        task_loss, bce, dice = balanced_loss(logits, yt)
        neg_idx = torch.nonzero(yt < 0.5, as_tuple=False).flatten()
        k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
        hard_idx = neg_idx[torch.topk(logits[neg_idx], k=k, largest=True).indices]
        hard_loss = F.softplus(logits[hard_idx]).mean()
        score = logits.cpu().numpy().astype(np.float32)
    p_auc = safe_auc(y, score)
    p_ap = safe_ap(y, score)
    row = {
        "run": run,
        "epoch": epoch,
        "split": "source",
        "dataset": "Full VisA source sample pool",
        "P-AUROC": p_auc,
        "P-AP": p_ap,
        "G_pixel": 0.5 * (p_auc + p_ap),
        "source_eval_statealigned_loss": float(task_loss.cpu()),
        "source_eval_hardnormal_loss": float(hard_loss.cpu()),
        "source_eval_bce": float(bce.cpu()),
        "source_eval_dice": float(dice.cpu()),
    }
    if final_training_row:
        row.update({
            "source_total_loss": final_training_row["source_total_loss"],
            "source_statealigned_loss": final_training_row["source_statealigned_loss"],
            "source_hardnormal_loss": final_training_row["source_hardnormal_loss"],
            "train_total_loss": final_training_row["source_total_loss"],
            "train_statealigned_loss": final_training_row["source_statealigned_loss"],
            "train_hardnormal_loss": final_training_row["source_hardnormal_loss"],
        })
    return row


def train_b0_trajectory(groups, m0, r0, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"B0_trajectory_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["epoch_raw"], payload["epoch_v"], payload["trajectory"], True
    print("[TrajectoryAvg] train B0 trajectory", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m0, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    trajectory = []
    epoch_raw = {}
    epoch_v = {}
    prev_v = unit_np(init_v)
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _step in range(steps_per_epoch):
            xb, yb = balanced_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, r0, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            hard_idx = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
            hard_loss = F.softplus(logits[hard_idx]).mean()
            loss = task_loss + hard_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        with torch.no_grad():
            raw = model.detach().cpu().numpy().astype(np.float32)
            v_np = unit_np(raw)
        row = {
            "run": "B0_Epoch10",
            "epoch": epoch,
            "source_total_loss": float(arr[:, 0].mean()),
            "source_statealigned_loss": float(arr[:, 1].mean()),
            "source_hardnormal_loss": float(arr[:, 2].mean()),
            "source_bce": float(arr[:, 3].mean()),
            "source_dice": float(arr[:, 4].mean()),
            "cos_d_epoch_d_init": cos_np(v_np, init_v),
            "angle_d_epoch_prev_deg": angle_deg(v_np, prev_v),
        }
        trajectory.append(row)
        epoch_raw[epoch] = raw
        epoch_v[epoch] = v_np
        prev_v = v_np
        torch.save({"raw_v": raw, "v": v_np, "epoch": epoch, "trajectory_row": row}, OUT / f"B0_epoch{epoch:02d}_{suffix}.pt")
        print(f"[TrajectoryAvg] epoch {epoch}/{EPOCHS} loss={row['source_total_loss']:.4f} hard={row['source_hardnormal_loss']:.4f}", flush=True)
    torch.save({"epoch_raw": epoch_raw, "epoch_v": epoch_v, "trajectory": trajectory}, ckpt)
    return epoch_raw, epoch_v, trajectory, False


def score_model_on_patch(model, raw_patch, q_pc1, device):
    raw_state = norm_np(remove_pc1(np.asarray(raw_patch, dtype=np.float32), q_pc1).reshape(-1, raw_patch.shape[-1]))
    x = torch.tensor(raw_state, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, _ = model.logits(x)
    return logits.detach().cpu().numpy().astype(np.float32).reshape(SIDE, SIDE)


def target_item(dataset, ref, kind, pos):
    return dataset[int(ref)] if kind == "source_dataset" else dataset[pos]


def evaluate_logit_ensemble(models, q_pc1, device, smoke):
    raw, percat = [], []
    variant = "E3_LogitEnsembleDiagnostic"
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
        image_scores = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            parts = [score_model_on_patch(model, patch[i], q_pc1, device) for model in models]
            avg = np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)
            maps[i] = avg.astype(np.float16)
            image_scores[i] = top1_image_score(avg)
        metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image_scores, maps)
        metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
        raw.append(metric)
        for cat, vals in per.items():
            row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
            row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
            percat.append(row)
        print(f"[TrajectoryAvg] eval {variant} on {cell_id}", flush=True)
    return raw, percat


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in TARGETS:
        b0 = by[(ds, "B0_Epoch10")]
        e1 = by[(ds, "E1_FullParameterSWA")]
        e2 = by[(ds, "E2_SphericalStateAverage")]
        e3 = by[(ds, "E3_LogitEnsembleDiagnostic")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E3_G": float(e3["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_E1_G": float(e2["G_pixel"]) - float(e1["G_pixel"]),
            "E3_minus_E2_G": float(e3["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E3_minus_B0_G": float(e3["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_B0_P-AUROC": float(e2["P-AUROC"]) - float(b0["P-AUROC"]),
            "E2_minus_B0_P-AP": float(e2["P-AP"]) - float(b0["P-AP"]),
            "B0_minus_historical_G": float(b0["G_pixel"]) - HISTORICAL_B0[ds],
        })
    return rows


def state_trajectory_rows(epoch_v, d_bar):
    rows = []
    prev = None
    path_len = 0.0
    for epoch in range(1, EPOCHS + 1):
        v = epoch_v[epoch]
        angle = float("nan") if prev is None else angle_deg(v, prev)
        if prev is not None:
            path_len += angle
        rows.append({
            "epoch": epoch,
            "cos_d_epoch_d10": cos_np(v, epoch_v[10]),
            "cos_d_epoch_d_bar": cos_np(v, d_bar),
            "angle_d_t_d_t_minus_1_deg": angle,
            "cumulative_spherical_path_length_deg": path_len,
        })
        prev = v
    return rows


def state_cosine_rows(epoch_v, d_bar):
    rows = []
    late_vals = []
    for i in range(1, EPOCHS + 1):
        for j in range(1, EPOCHS + 1):
            c = cos_np(epoch_v[i], epoch_v[j])
            rows.append({"epoch_i": i, "epoch_j": j, "cosine": c})
            if i < j and i in AVG_EPOCHS and j in AVG_EPOCHS:
                late_vals.append(c)
    arr = np.asarray(late_vals, dtype=np.float32)
    rows.extend([
        {"epoch_i": "Epoch6-10", "epoch_j": "mean_pairwise_cosine", "cosine": float(arr.mean())},
        {"epoch_i": "Epoch6-10", "epoch_j": "min_pairwise_cosine", "cosine": float(arr.min())},
        {"epoch_i": "Epoch6-10", "epoch_j": "max_pairwise_cosine", "cosine": float(arr.max())},
        {"epoch_i": "d_bar", "epoch_j": "d10", "cosine": cos_np(d_bar, epoch_v[10])},
    ])
    return rows


def source_target_correlation(per_epoch):
    epoch_to_source = {}
    epoch_to_target = {}
    for row in per_epoch:
        if row["split"] == "source":
            epoch_to_source[int(row["epoch"])] = row
        elif row["split"] == "target":
            epoch_to_target.setdefault(int(row["epoch"]), []).append(float(row["G_pixel"]))
    epochs = sorted(set(epoch_to_source) & set(epoch_to_target))
    target_mean = [float(np.mean(epoch_to_target[e])) for e in epochs]
    rows = []
    for key in ("G_pixel", "source_total_loss", "source_hardnormal_loss"):
        vals = [float(epoch_to_source[e].get(key, float("nan"))) for e in epochs]
        rows.append({"source_quantity": key, "target_quantity": "epoch_three_target_mean_G", "pearson": safe_pearson(vals, target_mean), "spearman": safe_spearman(vals, target_mean)})
    return rows


def decide(summary_rows, cosine_rows):
    e2_b0 = np.asarray([r["E2_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e2_e1 = np.asarray([r["E2_minus_E1_G"] for r in summary_rows], dtype=np.float32)
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e3_b0 = np.asarray([r["E3_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e3_e2 = np.asarray([r["E3_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    by = {r["dataset"]: r for r in summary_rows}
    c = {(r["epoch_i"], r["epoch_j"]): float(r["cosine"]) for r in cosine_rows}
    late_mean = c[("Epoch6-10", "mean_pairwise_cosine")]
    success = (
        int(np.sum(e2_b0 > 0)) >= 2
        and float(e2_b0.mean()) >= 1.0
        and max(float(by["MVTec"]["E2_minus_B0_G"]), float(by["CVC-ClinicDB"]["E2_minus_B0_G"])) >= 1.5
        and float(by["BTAD"]["E2_minus_B0_G"]) >= -1.0
        and late_mean < 0.995
    )
    if success and float(e2_e1.mean()) >= 0.3:
        outcome = "SPHERICAL_STATE_TRAJECTORY_AVERAGING_SUPPORTED"
    elif int(np.sum(e1_b0 > 0)) >= 2 and int(np.sum(e2_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e2_b0.mean()) >= 1.0 and abs(float(e2_e1.mean())) < 0.3:
        outcome = "GENERIC_SWA_IMPROVES_TRANSFER_NOT_STATE_SPECIFIC"
    elif float(e3_b0.mean()) >= 1.0 and float(e2_b0.mean()) < 1.0 and float(e3_e2.mean()) >= 1.0:
        outcome = "TRAJECTORY_ENSEMBLE_HAS_SIGNAL_NOT_COMPRESSIBLE"
    elif late_mean >= 0.995:
        outcome = "LATE_STATE_TRAJECTORY_ALREADY_CONVERGED"
    elif abs(float(e2_b0.mean())) < 0.5:
        outcome = "STATE_TRAJECTORY_AVERAGING_NO_SIGNAL"
    elif float(e2_b0.mean()) <= -0.5:
        outcome = "FINAL_STATE_DIRECTION_IS_PREFERABLE_TO_TRAJECTORY_MEAN"
    else:
        outcome = "STATE_TRAJECTORY_AVERAGING_NO_SIGNAL"
    return outcome, {
        "mean_E2_minus_B0_G": float(e2_b0.mean()),
        "mean_E2_minus_E1_G": float(e2_e1.mean()),
        "mean_E3_minus_E2_G": float(e3_e2.mean()),
        "targets_E2_improved_over_B0": int(np.sum(e2_b0 > 0)),
        "MVTec_E2_minus_B0_G": float(by["MVTec"]["E2_minus_B0_G"]),
        "BTAD_E2_minus_B0_G": float(by["BTAD"]["E2_minus_B0_G"]),
        "ClinicDB_E2_minus_B0_G": float(by["CVC-ClinicDB"]["E2_minus_B0_G"]),
        "epoch6_10_mean_pairwise_cosine": late_mean,
        "cos_d_bar_d10": c[("d_bar", "d10")],
        "mean_B0_epoch10_minus_historical_G": float(np.mean([r["B0_minus_historical_G"] for r in summary_rows])),
    }


def write_report(outcome, stats, summary_rows, runtime):
    lines = [
        "# Spherical State Trajectory Averaging",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: reran Full VisA StateAligned + Symmetric HardNormal with the formal original initialization, saved Epoch1-10 trainable contrast checkpoints, and evaluated fixed Epoch6-10 averaging controls without target selection.",
        "",
        "| Dataset | B0 Epoch10 | E1 Full SWA | E2 Spherical Avg | E3 Logit Ens | E2-B0 | E2-E1 | E3-E2 | B0-hist |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E3_G']:.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E2_minus_E1_G']:+.4f} | {r['E3_minus_E2_G']:+.4f} | {r['B0_minus_historical_G']:+.4f} |")
    lines.extend([
        "",
        "## Gate",
        f"- targets E2 improved over B0: {stats['targets_E2_improved_over_B0']}/3",
        f"- mean E2-B0 G: {stats['mean_E2_minus_B0_G']:+.4f}",
        f"- mean E2-E1 G: {stats['mean_E2_minus_E1_G']:+.4f}",
        f"- Epoch6-10 mean pairwise cosine: {stats['epoch6_10_mean_pairwise_cosine']:.6f}",
        f"- cos(d_bar,d10): {stats['cos_d_bar_d10']:.6f}",
        f"- mean B0 Epoch10 minus historical G: {stats['mean_B0_epoch10_minus_historical_G']:+.4f}",
        "",
        "Note: in this formal contrast pipeline the only trainable parameter is the state contrast vector; therefore E1 is arithmetic averaging of that full trainable parameter, while E2 averages normalized directions on the sphere.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TrajectoryAvg] device={device}", flush=True)
    q_pc1, corrupt_payload, states_c, m0, r0 = load_payloads(device)
    init_v = unit_np(np.load(HN_OUT / "contrast_vectors.npz")["B0"].astype(np.float32))
    groups, pool_rows = prepare_groups(q_pc1, states_c, corrupt_payload, device, args.smoke)
    epoch_raw, epoch_v, trajectory_train, reused = train_b0_trajectory(groups, m0, r0, init_v, device, args.smoke)

    v_swa = unit_np(np.mean([epoch_raw[e] for e in AVG_EPOCHS], axis=0).astype(np.float32))
    d_bar = unit_np(np.mean([epoch_v[e] for e in AVG_EPOCHS], axis=0).astype(np.float32))
    final_dirs = {
        "B0_Epoch10": epoch_v[10],
        "E1_FullParameterSWA": v_swa,
        "E2_SphericalStateAverage": d_bar,
    }
    models = {name: make_model(hybrid_states(m0, r0 * v), corrupt_payload, device) for name, v in final_dirs.items()}
    raw_main, percat_main = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    ensemble_models = [make_model(hybrid_states(m0, r0 * epoch_v[e]), corrupt_payload, device) for e in AVG_EPOCHS]
    raw_ens, percat_ens = evaluate_logit_ensemble(ensemble_models, q_pc1, device, args.smoke)
    raw = raw_main + raw_ens
    percat = percat_main + percat_ens

    epoch_models = {f"Epoch{e:02d}": make_model(hybrid_states(m0, r0 * epoch_v[e]), corrupt_payload, device) for e in range(1, EPOCHS + 1)}
    epoch_raw, _epoch_percat = evaluate_pc1_removed(epoch_models, q_pc1, device, args.smoke)
    per_epoch = []
    train_by_epoch = {int(r["epoch"]): r for r in trajectory_train}
    for e in range(1, EPOCHS + 1):
        per_epoch.append(compute_source_metrics("EpochTrajectory", e, groups, epoch_v[e], m0, r0, train_by_epoch[e]))
    for row in epoch_raw:
        e = int(str(row["variant"]).replace("Epoch", ""))
        out = dict(row)
        out["run"] = "EpochTrajectory"
        out["epoch"] = e
        out["split"] = "target"
        per_epoch.append(out)

    summary_rows = summarize(raw)
    traj_rows = state_trajectory_rows(epoch_v, d_bar)
    cosine_rows = state_cosine_rows(epoch_v, d_bar)
    corr_rows = source_target_correlation(per_epoch)
    outcome, stats = decide(summary_rows, cosine_rows)
    runtime = time.time() - start

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "PER_EPOCH_METRICS.csv", per_epoch)
    write_csv(OUT / "STATE_TRAJECTORY.csv", trajectory_train + traj_rows)
    write_csv(OUT / "STATE_COSINE_MATRIX.csv", cosine_rows)
    write_csv(OUT / "AVERAGING_COMPARISON.csv", summary_rows)
    write_csv(OUT / "SOURCE_TARGET_CORRELATION.csv", corr_rows)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_rows)
    np.savez_compressed(OUT / "contrast_vectors.npz", **{k: v.astype(np.float32) for k, v in final_dirs.items()}, **{f"Epoch{e:02d}": epoch_v[e].astype(np.float32) for e in range(1, EPOCHS + 1)})
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "cross_seed_averaging": False,
        "epoch_window": list(AVG_EPOCHS),
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "trajectory_checkpoint_reused": bool(reused),
    })
    write_report(outcome, stats, summary_rows, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
