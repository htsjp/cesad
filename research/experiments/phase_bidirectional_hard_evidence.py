import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cross_layer_persistent_hard_normal import MOECLIP, VISUALAD
from experiments.phase_hard_normal_contrast import (
    TOP_NEG_FRAC,
    balanced_category_batch,
    collect_source_patch_pool,
    final_negative_distribution,
)
from experiments.phase_hard_normal_rejection_state import load_b0_rows
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xliii_adapter_head_decomposition import SEED
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_bidirectional_hard_evidence/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
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


def load_setup(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    vecs = np.load(HN_OUT / "contrast_vectors.npz")
    b0_v = unit_np(np.asarray(vecs["E1"], dtype=np.float32))
    align_model = make_model(states_c, corrupt_payload, device)
    return q_pc1, corrupt_payload, align_model, m_c, alpha, init_v, b0_v


def train_variant(variant, pos_mode, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["rows"], payload["curves"], True
    print(f"[BiHardEvidence] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    rows, curves = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin, cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            pos_idx = torch.nonzero(y > 0.5, as_tuple=False).flatten()
            k_neg = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            hard_neg = neg_idx[torch.topk(logits.detach()[neg_idx], k=k_neg, largest=True).indices]

            # Approximate per-image bottom-10% within the sampled positive pool by category-balanced batch groups.
            pos_scores = logits.detach()[pos_idx]
            k_pos = max(1, int(np.ceil(float(len(pos_idx)) * TOP_NEG_FRAC)))
            if pos_mode == "hard":
                chosen_pos = pos_idx[torch.topk(pos_scores, k=k_pos, largest=False).indices]
            elif pos_mode == "random":
                chosen_pos = pos_idx[torch.tensor(rng.permutation(len(pos_idx))[:k_pos], dtype=torch.long, device=device)]
            else:
                raise ValueError(pos_mode)

            hard_neg_loss = F.softplus(logits[hard_neg]).mean()
            pos_loss = F.softplus(-logits[chosen_pos]).mean()
            loss = task_loss + hard_neg_loss + pos_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            hard_np = logits.detach()[hard_neg].cpu().numpy()
            pos_np = logits.detach()[chosen_pos].cpu().numpy()
            all_pos_np = logits.detach()[pos_idx].cpu().numpy()
            neg_origin = origin[hard_neg.detach().cpu().numpy()]
            pos_cats = cats[chosen_pos.detach().cpu().numpy()]
            rows.append({
                "variant": variant,
                "epoch": epoch,
                "step": step + 1,
                "positive_mode": pos_mode,
                "hard_negative_count": int(hard_neg.numel()),
                "selected_positive_count": int(chosen_pos.numel()),
                "hard_negative_score_mean": float(hard_np.mean()),
                "hard_negative_score_std": float(hard_np.std()),
                "selected_positive_score_mean": float(pos_np.mean()),
                "selected_positive_score_std": float(pos_np.std()),
                "all_positive_score_mean": float(all_pos_np.mean()),
                "positive_score_gap_all_minus_selected": float(all_pos_np.mean() - pos_np.mean()),
                "hard_positive_fraction_of_positive_pool": float(chosen_pos.numel() / max(int(pos_idx.numel()), 1)),
                "normal_image_ratio": float(np.mean(neg_origin == "normal_image")),
                "anomaly_mask_out_ratio": float(np.mean(neg_origin == "anomaly_mask_out")),
                "selected_positive_categories": ";".join(sorted(set(map(str, pos_cats.tolist())))),
                "loss": float(loss.detach().cpu()),
                "task_loss": float(task_loss.detach().cpu()),
                "hard_negative_loss": float(hard_neg_loss.detach().cpu()),
                "hard_positive_loss": float(pos_loss.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "dice": float(dice.detach().cpu()),
            })
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_neg_loss.detach().cpu()), float(pos_loss.detach().cpu()), float(pos_np.mean()), float(hard_np.mean()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "hard_negative_loss": float(arr[:, 2].mean()),
            "hard_positive_loss": float(arr[:, 3].mean()),
            "selected_positive_score_mean": float(arr[:, 4].mean()),
            "hard_negative_score_mean": float(arr[:, 5].mean()),
            "bce": float(arr[:, 6].mean()),
            "dice": float(arr[:, 7].mean()),
        }
        curves.append(row)
        print(f"[BiHardEvidence] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} hn={row['hard_negative_loss']:.4f} hp={row['hard_positive_loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "rows": rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, rows, curves, False


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_BidirectionalHardEvidence")]
        e2 = by[(ds, "E2_RandomPositiveControl")]
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


def decide(summary_rows, train_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and (e1_b0[0] >= 1.5 or e1_b0[2] >= 1.5) and e1_b0[1] >= -1.0:
        outcome = "BIDIRECTIONAL_HARD_EVIDENCE_SUPPORTED"
    elif abs(float(e1_b0.mean())) < 1.0 and float(e1_e2.mean()) >= 0.5:
        outcome = "HARD_POSITIVE_SELECTION_HAS_SIGNAL_NO_ADDITIONAL_GAIN"
    else:
        e1 = [r for r in train_rows if r.get("variant") == "E1_BidirectionalHardEvidence" and "epoch" in r and "selected_positive_score_mean" in r]
        source_improved = False
        if e1:
            early = np.asarray([r["selected_positive_score_mean"] for r in e1 if int(r["epoch"]) <= 2], dtype=np.float32)
            late = np.asarray([r["selected_positive_score_mean"] for r in e1 if int(r["epoch"]) >= EPOCHS - 1], dtype=np.float32)
            source_improved = bool(early.size and late.size and float(late.mean()) > float(early.mean()) + 0.02)
        outcome = "SOURCE_HARD_POSITIVE_NOT_TRANSFERABLE" if source_improved else "BIDIRECTIONAL_HARD_EVIDENCE_NO_SIGNAL"
    return outcome, {"mean_E1_minus_B0_G": float(e1_b0.mean()), "mean_E1_minus_E2_G": float(e1_e2.mean()), "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0))}


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_BidirectionalHardEvidence"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_BidirectionalHardEvidence",
                "current_P-AUROC": float(row["P-AUROC"]),
                "current_P-AP": float(row["P-AP"]),
                "current_G": float(row["G_pixel"]),
                "reference": ref_name,
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": float(row["P-AUROC"]) - ref["P-AUROC"],
                "Delta_P_AP": float(row["P-AP"]) - ref["P-AP"],
                "Delta_G": float(row["G_pixel"]) - ref["G_pixel"],
                "comparison_status": "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS",
            })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(summary_rows, outcome, stats, geom, sota_rows, runtime):
    lines = [
        "# Bidirectional Hard-Evidence Contrast Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. E1 keeps HardNormal top10% hard negatives and adds bottom10% hard-positive anomaly loss; E2 uses matched random positive patches. Target inference is the same single contrast scorer.",
        "",
        "| Dataset | B0 HardNormal G | E1 Bidirectional G | E2 RandomPositive G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        f"Final contrast cosines: B0/E1={geom[0]['cos_v_B0_v_E1']:.6f}, B0/E2={geom[0]['cos_v_B0_v_E2']:.6f}, E1/E2={geom[0]['cos_v_E1_v_E2']:.6f}.",
        "",
        "## External SOTA Position",
        "",
        "MoECLIP is the primary external gate; VisualAD(CLIP) is retained as a secondary reference. Comparisons are limited to VisA-source pixel metrics with implementation/preprocessing caveats.",
        "",
        "| Reference | Dataset | Current G | Reference G | Delta G | Status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for r in sota_rows:
        lines.append(f"| {r['reference']} | {r['dataset']} | {float(r['current_G']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend(["", "Stop-loss: if bidirectional hard evidence does not beat HardNormal and random-positive control, do not sweep top ratios, loss weights, ranking losses, adapters, Memory, or target tuning.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BiHardEvidence] device={device}", flush=True)
    q_pc1, corrupt_payload, align_model, m_c, alpha, init_v, b0_v = load_setup(device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    v_e1, rows_e1, _curves_e1, reused_e1 = train_variant("E1_BidirectionalHardEvidence", "hard", groups, m_c, alpha, init_v, device, args.smoke)
    v_e2, rows_e2, _curves_e2, reused_e2 = train_variant("E2_RandomPositiveControl", "random", groups, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_BidirectionalHardEvidence": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_RandomPositiveControl": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw_new, percat_new = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw_new
    percat = b0_percat + percat_new
    summary_rows = summarize(raw)
    train_rows = pool_audit + rows_e1 + rows_e2
    for variant, v in (("B0_HardNormal", b0_v), ("E1_BidirectionalHardEvidence", v_e1), ("E2_RandomPositiveControl", v_e2)):
        train_rows.extend(final_negative_distribution(variant, groups, v, m_c, alpha))
    geom = contrast_geometry(b0_v, v_e1, v_e2)
    outcome, decision_stats = decide(summary_rows, train_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "HARD_EVIDENCE_STATS.csv", train_rows)
    np.savez_compressed(OUT / "contrast_vectors.npz", B0=b0_v.astype(np.float32), E1=v_e1.astype(np.float32), E2=v_e2.astype(np.float32), init=init_v.astype(np.float32))
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "hard_positive_fraction": TOP_NEG_FRAC,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, decision_stats, geom, sota_rows, runtime)
    print(f"[BiHardEvidence] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
