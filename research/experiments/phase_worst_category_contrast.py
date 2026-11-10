import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_category_dropout_contrast import collect_group_samples, source_validation
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import random_slots_like, source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR

OUT = Path("results/phase_worst_category_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPOCHS = 10


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


def class_balanced_group_batch(data, rng, n):
    xs, ys = [], []
    per_cls = max(1, n // 2)
    for cls in (0, 1):
        ids = np.flatnonzero(data["y"] == cls)
        choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
        xs.append(data["x"][choice])
        ys.append(data["y"][choice])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order]


def train_worst_variant(variant, objective, slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["audit_rows"], True

    print(f"[WorstCategory] prepare samples for {variant}", flush=True)
    groups, sample_rows = collect_group_samples(slots, labels_by_pos, patch, align_model, q_pc1, device)
    names = list(groups)
    total_samples = sum(len(groups[n]["y"]) for n in names)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    per_group_batch = max(64, BATCH_SIZE // max(1, len(names)))

    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    max_counts = defaultdict(int)
    curves = []
    audit_rows = []

    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            v = F.normalize(model, dim=-1)
            losses = []
            loss_rows = []
            for group in names:
                xb, yb = class_balanced_group_batch(groups[group], rng, per_group_batch)
                z = torch.tensor(xb, dtype=torch.float32, device=device)
                y = torch.tensor(yb, dtype=torch.float32, device=device)
                logits = contrast_logits(z, m, alpha, v)
                loss_c, bce_c, dice_c = balanced_loss(logits, y)
                losses.append(loss_c)
                loss_rows.append((group, float(loss_c.detach().cpu()), float(bce_c.detach().cpu()), float(dice_c.detach().cpu())))
            loss_tensor = torch.stack(losses)
            worst_idx = int(torch.argmax(loss_tensor.detach()).cpu().item())
            worst_group = names[worst_idx]
            max_counts[worst_group] += 1
            if objective == "mean":
                loss = loss_tensor.mean()
            elif objective == "max":
                loss = loss_tensor.max()
            else:
                raise ValueError(objective)
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(loss_tensor.detach().mean().cpu()), float(loss_tensor.detach().max().cpu())))
            for group, lc, bce, dice in loss_rows:
                audit_rows.append({
                    "variant": variant,
                    "record_type": "step_loss",
                    "epoch": epoch,
                    "step": step + 1,
                    "group": group,
                    "loss_c": lc,
                    "bce_c": bce,
                    "dice_c": dice,
                    "is_max_loss_group": int(group == worst_group),
                })
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "objective": objective,
            "loss": float(arr[:, 0].mean()),
            "mean_group_loss": float(arr[:, 1].mean()),
            "max_group_loss": float(arr[:, 2].mean()),
        }
        curves.append(row)
        print(f"[WorstCategory] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} max={row['max_group_loss']:.4f}", flush=True)

    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    for row in sample_rows:
        group = row["group"]
        out = dict(row)
        out.update({
            "variant": variant,
            "record_type": "group_summary",
            "max_loss_count": int(max_counts[group]),
            "total_optimizer_steps": int(EPOCHS * steps_per_epoch),
            "objective": objective,
        })
        audit_rows.append(out)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "audit_rows": audit_rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, audit_rows, False


def source_diagnostics(variant, true_groups, v, m_c, alpha):
    rows = source_validation(variant, true_groups, v, m_c, alpha)
    vals = np.asarray([r["G_pixel"] for r in rows], dtype=np.float32)
    rows.append({
        "variant": variant,
        "source_group": "__summary__",
        "P-AUROC": float("nan"),
        "P-AP": float("nan"),
        "G_pixel": float(vals.mean()),
        "samples": int(sum(int(r["samples"]) for r in rows)),
        "positive_samples": int(sum(int(r["positive_samples"]) for r in rows)),
        "mean_source_category_G": float(vals.mean()),
        "worst_source_category_G": float(vals.min()),
        "source_category_G_std": float(vals.std()),
    })
    for r in rows:
        r.setdefault("mean_source_category_G", "")
        r.setdefault("worst_source_category_G", "")
        r.setdefault("source_category_G_std", "")
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_GroupMeanGlobalContrast")]
        e1 = by[(ds, "E1_TrueWorstCategory")]
        e2 = by[(ds, "E2_RandomWorstGroup")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
            "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def decide(summary_rows, source_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    stable = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0
    if stable and float(e1_e2.mean()) >= 0.5:
        outcome = "WORST_CATEGORY_CONTRAST_SUPPORTED"
    elif stable and abs(float(e1_e2.mean())) < 0.5:
        outcome = "GENERIC_WORST_GROUP_REGULARIZATION"
    else:
        summaries = {r["variant"]: r for r in source_rows if r["source_group"] == "__summary__"}
        improved_source_worst = (
            "B0_GroupMeanGlobalContrast" in summaries
            and "E1_TrueWorstCategory" in summaries
            and float(summaries["E1_TrueWorstCategory"]["worst_source_category_G"]) > float(summaries["B0_GroupMeanGlobalContrast"]["worst_source_category_G"]) + 0.5
        )
        outcome = "SOURCE_ROBUSTNESS_NOT_TRANSFERABILITY" if improved_source_worst else "WORST_CATEGORY_CONTRAST_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_report(summary_rows, outcome, stats, geom, runtime):
    lines = [
        "# Worst-Category Contrast Learning Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Strict Full VisA source-only. B0/E1/E2 are retrained with one final global contrast; target inference has no memory, router, prompt, target statistics, or target adaptation.",
        "",
        "| Dataset | B0 GroupMean G | E1 WorstCategory G | E2 RandomWorst G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        f"Contrast cosines: B0/E1={geom[0]['cos_v_B0_v_E1']:.6f}, B0/E2={geom[0]['cos_v_B0_v_E2']:.6f}, E1/E2={geom[0]['cos_v_E1_v_E2']:.6f}.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[WorstCategory] device={device}", flush=True)
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    true_slots = source_slots_by_category(dataset, indices, args.smoke)
    random_slots = random_slots_like(true_slots, dataset, indices, SEED)

    v_b0, audit_b0, reused_b0 = train_worst_variant("B0_GroupMeanGlobalContrast", "mean", true_slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, args.smoke)
    v_e1, audit_e1, reused_e1 = train_worst_variant("E1_TrueWorstCategory", "max", true_slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, args.smoke)
    v_e2, audit_e2, reused_e2 = train_worst_variant("E2_RandomWorstGroup", "max", random_slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, args.smoke)

    models = {
        "B0_GroupMeanGlobalContrast": make_model(hybrid_states(m_c, alpha * v_b0), corrupt_payload, device),
        "E1_TrueWorstCategory": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_RandomWorstGroup": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    summary_rows = summarize(raw)
    true_groups, _sample_rows = collect_group_samples(true_slots, labels_by_pos, patch, align_model, q_pc1, device)
    source_rows = []
    for variant, v in (("B0_GroupMeanGlobalContrast", v_b0), ("E1_TrueWorstCategory", v_e1), ("E2_RandomWorstGroup", v_e2)):
        source_rows.extend(source_diagnostics(variant, true_groups, v, m_c, alpha))
    geom = contrast_geometry(v_b0, v_e1, v_e2)
    outcome, stats = decide(summary_rows, source_rows)
    runtime = time.time() - start

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "SOURCE_DIAGNOSTICS.csv", source_rows)
    write_csv(OUT / "MAX_LOSS_AUDIT.csv", audit_b0 + audit_e1 + audit_e2)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    np.savez_compressed(OUT / "contrast_vectors.npz", B0=v_b0.astype(np.float32), E1=v_e1.astype(np.float32), E2=v_e2.astype(np.float32), init=init_v.astype(np.float32))
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "alpha": alpha,
        "checkpoints_reused": {"B0": bool(reused_b0), "E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, stats, geom, runtime)
    print(f"[WorstCategory] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
