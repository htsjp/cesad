import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_category_dropout_contrast import (
    collect_group_samples,
    load_b0_rows,
    sample_batch,
    source_validation,
)
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import random_slots_like, source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR

OUT = Path("results/phase_source_partition_contrast_invariance/formal")
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


def split_partition(names, rng):
    shuffled = list(names)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


def batch_induced_contrast(z, y, m, alpha, v):
    logits = contrast_logits(z, m, alpha, v)
    pos_mask = y > 0.5
    neg_mask = ~pos_mask
    if int(pos_mask.sum().item()) == 0 or int(neg_mask.sum().item()) == 0:
        return v
    pos_w = torch.sigmoid(-logits[pos_mask]).detach() + 1e-3
    neg_w = torch.sigmoid(logits[neg_mask]).detach() + 1e-3
    pos_mu = (z[pos_mask] * pos_w[:, None]).sum(dim=0) / pos_w.sum().clamp_min(1e-6)
    neg_mu = (z[neg_mask] * neg_w[:, None]).sum(dim=0) / neg_w.sum().clamp_min(1e-6)
    return F.normalize(pos_mu - neg_mu, dim=-1)


def train_partition_variant(variant, slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["audit_rows"], payload["source_validation_rows"], True
    print(f"[PartitionInvariant] prepare samples for {variant}", flush=True)
    groups, sample_rows = collect_group_samples(slots, labels_by_pos, patch, align_model, q_pc1, device)
    names = list(groups)
    total_samples = sum(len(groups[n]["y"]) for n in names)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)

    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng_np = np.random.default_rng(SEED)
    rng_py = random.Random(SEED)
    curves = []
    inv_cosines = []
    partition_rows = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            p1, p2 = split_partition(names, rng_py)
            x1, y1 = sample_batch(groups, p1, rng_np, max(2, BATCH_SIZE // 2))
            x2, y2 = sample_batch(groups, p2, rng_np, max(2, BATCH_SIZE // 2))
            z1 = torch.tensor(x1, dtype=torch.float32, device=device)
            yy1 = torch.tensor(y1, dtype=torch.float32, device=device)
            z2 = torch.tensor(x2, dtype=torch.float32, device=device)
            yy2 = torch.tensor(y2, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits1 = contrast_logits(z1, m, alpha, v)
            logits2 = contrast_logits(z2, m, alpha, v)
            loss1, _bce1, _dice1 = balanced_loss(logits1, yy1)
            loss2, _bce2, _dice2 = balanced_loss(logits2, yy2)
            v1 = batch_induced_contrast(z1, yy1, m, alpha, v)
            v2 = batch_induced_contrast(z2, yy2, m, alpha, v)
            inv_cos = torch.sum(v1 * v2)
            inv_loss = 1.0 - inv_cos
            task_loss = 0.5 * (loss1 + loss2)
            loss = task_loss + inv_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(inv_loss.detach().cpu()), float(inv_cos.detach().cpu())))
            inv_cosines.append(float(inv_cos.detach().cpu()))
            partition_rows.append({
                "variant": variant,
                "record_type": "train_partition",
                "epoch": epoch,
                "step": step + 1,
                "P1_groups": "|".join(p1),
                "P2_groups": "|".join(p2),
                "cos_v1_v2": float(inv_cos.detach().cpu()),
                "L_inv": float(inv_loss.detach().cpu()),
            })
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "L_inv": float(arr[:, 2].mean()),
            "cos_v1_v2_mean": float(arr[:, 3].mean()),
            "cos_v1_v2_std": float(arr[:, 3].std()),
        }
        curves.append(row)
        print(f"[PartitionInvariant] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} inv_cos={row['cos_v1_v2_mean']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    source_rows = source_validation(variant, groups, v, m_c, alpha)
    stability_rows = leave_one_group_stability(variant, groups, v)
    inv = np.asarray(inv_cosines, dtype=np.float32)
    audit_rows = sample_rows + partition_rows + stability_rows + [{
        "variant": variant,
        "record_type": "summary",
        "epoch": "",
        "step": "",
        "P1_groups": "",
        "P2_groups": "",
        "cos_v1_v2": float(inv.mean()),
        "L_inv": float(1.0 - inv.mean()),
        "cos_v1_v2_std": float(inv.std()),
        "total_optimizer_steps": int(EPOCHS * steps_per_epoch),
    }]
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "audit_rows": audit_rows, "source_validation_rows": source_rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, audit_rows, source_rows, False


def leave_one_group_stability(variant, groups, v):
    rows = []
    names = list(groups)
    all_x = []
    all_y = []
    for name in names:
        all_x.append(groups[name]["x"])
        all_y.append(groups[name]["y"])
    for held in names:
        keep_x = [groups[n]["x"] for n in names if n != held]
        keep_y = [groups[n]["y"] for n in names if n != held]
        x = np.concatenate(keep_x, axis=0)
        y = np.concatenate(keep_y, axis=0)
        pos = x[y > 0.5]
        neg = x[y <= 0.5]
        d = unit_np(pos.mean(axis=0) - neg.mean(axis=0))
        rows.append({
            "variant": variant,
            "record_type": "leave_one_group_stability",
            "held_group": held,
            "cos_leave_one_contrast_to_final": float(d @ unit_np(v)),
        })
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_GlobalContrast")]
        e1 = by[(ds, "E1_TruePartitionInvariance")]
        e2 = by[(ds, "E2_MatchedRandomPartitionInvariance")]
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


def contrast_geometry(v_b0, v_e1, v_e2):
    return [{
        "cos_v_B0_v_E1": float(unit_np(v_b0) @ unit_np(v_e1)),
        "cos_v_B0_v_E2": float(unit_np(v_b0) @ unit_np(v_e2)),
        "cos_v_E1_v_E2": float(unit_np(v_e1) @ unit_np(v_e2)),
        "norm_v_B0": float(np.linalg.norm(v_b0)),
        "norm_v_E1": float(np.linalg.norm(v_e1)),
        "norm_v_E2": float(np.linalg.norm(v_e2)),
    }]


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    stable = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0
    if stable and float(e1_e2.mean()) >= 0.5:
        outcome = "SOURCE_PARTITION_INVARIANCE_SUPPORTED"
    elif stable and abs(float(e1_e2.mean())) < 0.5:
        outcome = "PARTITION_INVARIANCE_REGULARIZATION_NOT_SEMANTIC"
    else:
        outcome = "SOURCE_PARTITION_INVARIANCE_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_report(summary_rows, outcome, stats, geom, runtime):
    lines = [
        "# Source-Partition Contrast Invariance Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Strict Full VisA source-only. E1/E2 train one final global contrast; partition consistency is training-only and no target statistics, memory, routing, prompt, or DINO are used.",
        "",
        "| Dataset | B0 G | E1 True Partition G | E2 Random Partition G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        f"Contrast cosines: B0/E1={geom[0]['cos_v_B0_v_E1']:.6f}, B0/E2={geom[0]['cos_v_B0_v_E2']:.6f}.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PartitionInvariant] device={device}", flush=True)
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
    cat_slots = source_slots_by_category(dataset, indices, args.smoke)
    random_slots = random_slots_like(cat_slots, dataset, indices, SEED)

    v_e1, audit_e1, source_e1, reused_e1 = train_partition_variant("E1_TruePartitionInvariance", cat_slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, args.smoke)
    v_e2, audit_e2, source_e2, reused_e2 = train_partition_variant("E2_MatchedRandomPartitionInvariance", random_slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, args.smoke)
    models = {
        "E1_TruePartitionInvariance": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_MatchedRandomPartitionInvariance": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    geom = contrast_geometry(init_v, v_e1, v_e2)
    outcome, stats = decide(summary_rows)
    runtime = time.time() - start

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "DROPOUT_AUDIT.csv", audit_e1 + audit_e2)
    write_csv(OUT / "SOURCE_PER_CATEGORY_VALIDATION.csv", source_e1 + source_e2)
    np.savez_compressed(OUT / "contrast_vectors.npz", B0=init_v.astype(np.float32), E1=v_e1.astype(np.float32), E2=v_e2.astype(np.float32))
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
        "lambda_inv": 1.0,
        "seed": SEED,
        "alpha": alpha,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, stats, geom, runtime)
    print(f"[PartitionInvariant] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
