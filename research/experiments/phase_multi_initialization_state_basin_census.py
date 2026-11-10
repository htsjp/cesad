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

from experiments.phase_frozen_semantic_anchor_fusion import load_b0_rows
from experiments.phase_hard_normal_contrast import balanced_category_batch, collect_source_patch_pool
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_multi_initialization_state_basin_census/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPOCHS = 10
TOP_NEG_FRAC = 0.10
TARGETS = ("MVTec", "BTAD", "CVC-ClinicDB")


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


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


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


def random_unit(seed, dim):
    rng = np.random.default_rng(seed)
    return unit_np(rng.standard_normal(dim).astype(np.float32))


def train_run(run_id, init_v, groups, m0, r0, b0_final_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{run_id}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["trajectory"], True

    print(f"[BasinCensus] train {run_id}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m0, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    trajectory = []
    prev_v = unit_np(init_v)

    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _step in range(steps_per_epoch):
            xb, yb, _origin, _cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, r0, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            selected_idx = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
            hard_loss = F.softplus(logits[selected_idx]).mean()
            loss = task_loss + hard_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        with torch.no_grad():
            epoch_v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
        row = {
            "run": run_id,
            "epoch": epoch,
            "source_total_loss": float(arr[:, 0].mean()),
            "source_statealigned_loss": float(arr[:, 1].mean()),
            "source_hardnormal_loss": float(arr[:, 2].mean()),
            "source_bce": float(arr[:, 3].mean()),
            "source_dice": float(arr[:, 4].mean()),
            "cos_d_epoch_d_init": cos_np(epoch_v, init_v),
            "cos_d_epoch_d_B0_final": cos_np(epoch_v, b0_final_v),
            "angle_d_epoch_prev_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, cos_np(epoch_v, prev_v)))))),
        }
        trajectory.append(row)
        prev_v = epoch_v
        print(f"[BasinCensus] {run_id} epoch {epoch}/{EPOCHS} loss={row['source_total_loss']:.4f} hard={row['source_hardnormal_loss']:.4f} cosB0={row['cos_d_epoch_d_B0_final']:.4f}", flush=True)

    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "init_v": np.asarray(init_v, dtype=np.float32), "trajectory": trajectory}, ckpt)
    return v, trajectory, False


def source_metrics(run_id, groups, v, m0, r0, trajectory):
    xs, ys = [], []
    for data in groups.values():
        xs.append(data["x"])
        ys.append(data["y"])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
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
    last = trajectory[-1] if trajectory else {}
    return {
        "run": run_id,
        "split": "source",
        "dataset": "Full VisA source sample pool",
        "P-AUROC": p_auc,
        "P-AP": p_ap,
        "G_pixel": 0.5 * (p_auc + p_ap),
        "final_training_loss": float(last.get("source_total_loss", float(task_loss + hard_loss))),
        "final_statealigned_loss": float(last.get("source_statealigned_loss", float(task_loss))),
        "final_hardnormal_loss": float(last.get("source_hardnormal_loss", float(hard_loss))),
        "source_eval_statealigned_loss": float(task_loss.cpu()),
        "source_eval_hardnormal_loss": float(hard_loss.cpu()),
        "source_eval_bce": float(bce.cpu()),
        "source_eval_dice": float(dice.cpu()),
    }


def r0_trajectory():
    path = HN_OUT / "E1_HardNormalContrast_TRAINING_CURVES.csv"
    rows = []
    if not path.exists():
        return rows
    for row in read_csv_rows(path):
        rows.append({
            "run": "R0_FormalOriginalInit",
            "epoch": int(float(row["epoch"])),
            "source_total_loss": float(row["loss"]),
            "source_statealigned_loss": float(row["task_loss"]),
            "source_hardnormal_loss": float(row["extra_normal_loss"]),
            "source_bce": float(row["bce"]),
            "source_dice": float(row["dice"]),
            "cos_d_epoch_d_init": float("nan"),
            "cos_d_epoch_d_B0_final": float("nan"),
            "angle_d_epoch_prev_deg": float("nan"),
        })
    return rows


def b0_target_rows():
    raw, percat = load_b0_rows()
    for row in raw:
        row["run"] = "R0_FormalOriginalInit"
        row["split"] = "target"
        row["variant"] = "R0_FormalOriginalInit"
    for row in percat:
        row["run"] = "R0_FormalOriginalInit"
        row["variant"] = "R0_FormalOriginalInit"
    return raw, percat


def state_cosine_matrix(final_dirs):
    names = list(final_dirs)
    rows = []
    vals = []
    for a in names:
        for b in names:
            c = cos_np(final_dirs[a], final_dirs[b])
            rows.append({"run_i": a, "run_j": b, "cosine": c})
            if a < b:
                vals.append(c)
    arr = np.asarray(vals, dtype=np.float32)
    rows.append({"run_i": "__summary__", "run_j": "off_diagonal_mean", "cosine": float(arr.mean())})
    rows.append({"run_i": "__summary__", "run_j": "off_diagonal_std", "cosine": float(arr.std())})
    rows.append({"run_i": "__summary__", "run_j": "off_diagonal_min", "cosine": float(arr.min())})
    rows.append({"run_i": "__summary__", "run_j": "off_diagonal_max", "cosine": float(arr.max())})
    return rows


def target_variance(run_metrics):
    rows = []
    by_run = {}
    for r in run_metrics:
        if r["split"] == "target":
            by_run.setdefault(r["run"], {})[r["dataset"]] = float(r["G_pixel"])
    for ds in TARGETS:
        vals = np.asarray([by_run[r][ds] for r in by_run if ds in by_run[r]], dtype=np.float32)
        rows.append({"scope": ds, "mean_G": float(vals.mean()), "std_G": float(vals.std()), "min_G": float(vals.min()), "max_G": float(vals.max()), "range_G": float(vals.max() - vals.min())})
    mean_vals = np.asarray([np.mean([by_run[r][ds] for ds in TARGETS]) for r in by_run], dtype=np.float32)
    rows.append({"scope": "three_target_mean", "mean_G": float(mean_vals.mean()), "std_G": float(mean_vals.std()), "min_G": float(mean_vals.min()), "max_G": float(mean_vals.max()), "range_G": float(mean_vals.max() - mean_vals.min())})
    b0 = np.mean([by_run["R0_FormalOriginalInit"][ds] for ds in TARGETS])
    random_runs = [r for r in by_run if r.startswith("R") and r != "R0_FormalOriginalInit"]
    rows.append({
        "scope": "random_runs_over_B0_counts",
        "mean_G": float(b0),
        "std_G": float("nan"),
        "min_G": int(sum(np.mean([by_run[r][ds] for ds in TARGETS]) > b0 for r in random_runs)),
        "max_G": int(sum(sum(by_run[r][ds] > by_run["R0_FormalOriginalInit"][ds] for ds in TARGETS) >= 2 for r in random_runs)),
        "range_G": float("nan"),
    })
    return rows


def source_target_correlation(run_metrics):
    by_run = {}
    for r in run_metrics:
        by_run.setdefault(r["run"], {})[(r["split"], r["dataset"])] = r
    rows = []
    runs = sorted(by_run)
    source_g = []
    source_loss = []
    source_hard = []
    target_mean = []
    for run in runs:
        src = by_run[run][("source", "Full VisA source sample pool")]
        tg = [float(by_run[run][("target", ds)]["G_pixel"]) for ds in TARGETS]
        source_g.append(float(src["G_pixel"]))
        source_loss.append(float(src["final_training_loss"]))
        source_hard.append(float(src["final_hardnormal_loss"]))
        target_mean.append(float(np.mean(tg)))
    for name, vals in (("source_G", source_g), ("final_training_loss", source_loss), ("final_hardnormal_loss", source_hard)):
        rows.append({"source_quantity": name, "target_quantity": "three_target_mean_G", "pearson": safe_pearson(vals, target_mean), "spearman": safe_spearman(vals, target_mean)})
    return rows


def decide(run_metrics, variance_rows):
    by_run = {}
    for r in run_metrics:
        by_run.setdefault(r["run"], {})[(r["split"], r["dataset"])] = r
    random_runs = sorted([r for r in by_run if r.startswith("R") and r != "R0_FormalOriginalInit"])
    random_source_g = np.asarray([float(by_run[r][("source", "Full VisA source sample pool")]["G_pixel"]) for r in random_runs], dtype=np.float32)
    b0_target_mean = float(np.mean([float(by_run["R0_FormalOriginalInit"][("target", ds)]["G_pixel"]) for ds in TARGETS]))
    random_target_mean = np.asarray([np.mean([float(by_run[r][("target", ds)]["G_pixel"]) for ds in TARGETS]) for r in random_runs], dtype=np.float32)
    tv = {r["scope"]: r for r in variance_rows}
    source_range = float(random_source_g.max() - random_source_g.min()) if random_source_g.size else float("nan")
    target_std = float(tv["three_target_mean"]["std_G"])
    target_range = float(tv["three_target_mean"]["range_G"])
    random_over_b0 = int(np.sum(random_target_mean > b0_target_mean))
    random_all_weak = int(np.sum(random_target_mean > b0_target_mean)) == 0
    if source_range <= 1.0 and (target_std >= 2.0 or target_range >= 5.0) and random_over_b0 >= 2:
        outcome = "SOURCE_EQUIVALENT_STATE_BASINS_HAVE_TRANSFER_VARIANCE"
    elif (target_std < 1.0 and target_range < 3.0):
        outcome = "STATE_DIRECTION_VARIANCE_TARGET_INVARIANT"
    elif source_range > 1.0:
        outcome = "INITIALIZATION_AFFECTS_SOURCE_OPTIMIZATION_NOT_ONLY_TRANSFER"
    elif random_all_weak:
        outcome = "FORMAL_INITIALIZATION_IS_SPECIAL"
    else:
        outcome = "STATE_BASIN_VARIANCE_WEAK"
    return outcome, {
        "random_source_G_range": source_range,
        "three_target_mean_G_std_all_runs": target_std,
        "three_target_mean_G_range_all_runs": target_range,
        "B0_three_target_mean_G": b0_target_mean,
        "random_runs_mean_target_over_B0_count": random_over_b0,
        "random_runs_2of3_targets_over_B0_count": int(sum(sum(float(by_run[r][("target", ds)]["G_pixel"]) > float(by_run["R0_FormalOriginalInit"][("target", ds)]["G_pixel"]) for ds in TARGETS) >= 2 for r in random_runs)),
    }


def write_report(outcome, stats, run_metrics, variance_rows, corr_rows, runtime):
    by_run = {}
    for r in run_metrics:
        by_run.setdefault(r["run"], {})[(r["split"], r["dataset"])] = r
    lines = [
        "# Multi-Initialization State Basin Census",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source, StateAligned + Symmetric HardNormal, frozen CLIP, lr=0.001, epoch=10, top10% hard normals. R0 reuses the formal original initialization checkpoint; R1-R8 use random unit contrast initializations with seeds 1-8.",
        "",
        "| Run | Source G | Final Loss | Hard Loss | MVTec G | BTAD G | ClinicDB G | Mean Target G |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in sorted(by_run):
        src = by_run[run][("source", "Full VisA source sample pool")]
        tg = [float(by_run[run][("target", ds)]["G_pixel"]) for ds in TARGETS]
        lines.append(f"| {run} | {float(src['G_pixel']):.4f} | {float(src['final_training_loss']):.4f} | {float(src['final_hardnormal_loss']):.4f} | {tg[0]:.4f} | {tg[1]:.4f} | {tg[2]:.4f} | {float(np.mean(tg)):.4f} |")
    lines.extend([
        "",
        "## Census",
        f"- random source G range: {stats['random_source_G_range']:.4f}",
        f"- three-target mean G std: {stats['three_target_mean_G_std_all_runs']:.4f}",
        f"- three-target mean G range: {stats['three_target_mean_G_range_all_runs']:.4f}",
        f"- random runs with mean target G > B0: {stats['random_runs_mean_target_over_B0_count']}/8",
        f"- random runs with at least 2/3 targets > B0: {stats['random_runs_2of3_targets_over_B0_count']}/8",
        "",
        "## Source-Target Correlation",
    ])
    for row in corr_rows:
        lines.append(f"- {row['source_quantity']} vs target mean G: Pearson {float(row['pearson']):+.4f}, Spearman {float(row['spearman']):+.4f}")
    lines.extend(["", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BasinCensus] device={device}", flush=True)
    q_pc1, corrupt_payload, states_c, m0, r0 = load_payloads(device)
    hn_vectors = np.load(HN_OUT / "contrast_vectors.npz")
    b0_init_v = unit_np(hn_vectors["B0"].astype(np.float32))
    b0_final_v = unit_np(hn_vectors["E1"].astype(np.float32))
    groups, pool_rows = prepare_groups(q_pc1, states_c, corrupt_payload, device, args.smoke)

    final_dirs = {"R0_FormalOriginalInit": b0_final_v}
    init_dirs = {"R0_FormalOriginalInit": b0_init_v}
    trajectories = r0_trajectory()
    reused = {"R0_FormalOriginalInit": True}
    for seed in range(1, 9):
        run_id = f"R{seed}_RandomSeed{seed}"
        init_v = random_unit(seed, b0_init_v.shape[0])
        init_dirs[run_id] = init_v
        final_v, rows, was_reused = train_run(run_id, init_v, groups, m0, r0, b0_final_v, device, args.smoke)
        final_dirs[run_id] = final_v
        trajectories.extend(rows)
        reused[run_id] = bool(was_reused)

    models = {run_id: make_model(hybrid_states(m0, r0 * v), corrupt_payload, device) for run_id, v in final_dirs.items() if run_id != "R0_FormalOriginalInit"}
    raw_new, per_new = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    raw_b0, per_b0 = b0_target_rows()
    if args.smoke:
        raw_b0 = raw_b0[:3]
        per_b0 = per_b0[:3]
    target_rows = []
    for row in raw_b0 + raw_new:
        run_id = row.get("run") or row.get("variant")
        out = dict(row)
        out["run"] = run_id
        out["variant"] = run_id
        out["split"] = "target"
        target_rows.append(out)
    percat = per_b0 + per_new

    run_metrics = []
    for run_id, v in final_dirs.items():
        traj = [r for r in trajectories if r["run"] == run_id]
        run_metrics.append(source_metrics(run_id, groups, v, m0, r0, traj))
    run_metrics.extend(target_rows)

    geom_rows = []
    for run_id in sorted(final_dirs):
        geom_rows.append({
            "run": run_id,
            "cos_final_to_B0_final": cos_np(final_dirs[run_id], b0_final_v),
            "cos_init_to_B0_init": cos_np(init_dirs[run_id], b0_init_v),
            "cos_init_to_B0_final": cos_np(init_dirs[run_id], b0_final_v),
            "semantic_retention_cos_final_init": cos_np(final_dirs[run_id], init_dirs[run_id]),
        })
    cosine_rows = state_cosine_matrix(final_dirs)
    variance_rows = target_variance(run_metrics)
    corr_rows = source_target_correlation(run_metrics)
    outcome, stats = decide(run_metrics, variance_rows)
    runtime = time.time() - start

    write_csv(OUT / "RUN_METRICS.csv", run_metrics)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "TARGET_VARIANCE.csv", variance_rows)
    write_csv(OUT / "STATE_COSINE_MATRIX.csv", cosine_rows)
    write_csv(OUT / "STATE_GEOMETRY.csv", geom_rows)
    write_csv(OUT / "SOURCE_TARGET_CORRELATION.csv", corr_rows)
    write_csv(OUT / "STATE_TRAJECTORY.csv", trajectories)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_rows)
    np.savez_compressed(OUT / "contrast_vectors.npz", **{k: v.astype(np.float32) for k, v in final_dirs.items()})
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "new_loss": False,
        "text_used": False,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "random_initialization_seeds": list(range(1, 9)),
        "top_negative_fraction": TOP_NEG_FRAC,
        "state_midpoint_source": "formal contrast pipeline m0",
        "state_magnitude": r0,
        "checkpoints_reused": reused,
    })
    write_report(outcome, stats, run_metrics, variance_rows, corr_rows, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
