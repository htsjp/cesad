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


OUT = Path("results/phase_error_correction_experience_memory/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
EPOCHS = 10
MEMORY_K = 64
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


def load_payload(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def load_hardnormal_setup(device):
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
    vecs = np.load(HN_OUT / "contrast_vectors.npz")
    b0_v = unit_np(np.asarray(vecs["E1"], dtype=np.float32))
    return q_pc1, corrupt_payload, align_model, m_c, alpha, init_v, b0_v


def normalized_grad(loss, param):
    grad = torch.autograd.grad(loss, param, retain_graph=True, create_graph=False, allow_unused=False)[0]
    return F.normalize(grad.detach(), dim=0)


def memory_consensus(memory, device):
    if not memory:
        return None
    mat = torch.stack(memory, dim=0).to(device)
    return F.normalize(mat.mean(dim=0), dim=0)


def pairwise_cosine_stats(memory):
    if len(memory) < 2:
        return {"pairwise_cosine_mean": float("nan"), "pairwise_cosine_std": float("nan"), "experience_diversity": float("nan")}
    mat = F.normalize(torch.stack(memory, dim=0).detach().cpu(), dim=-1).numpy()
    sim = mat @ mat.T
    tri = sim[np.triu_indices_from(sim, k=1)]
    return {
        "pairwise_cosine_mean": float(tri.mean()),
        "pairwise_cosine_std": float(tri.std()),
        "experience_diversity": float(1.0 - tri.mean()),
    }


def train_experience_variant(variant, experience_mode, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["memory_rows"], payload["curves"], True
    print(f"[ExperienceMemory] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    memory = []
    memory_rows = []
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin, _cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            neg_scores = logits.detach()[neg_idx]
            hard_local = torch.topk(neg_scores, k=k, largest=True).indices
            random_local = torch.tensor(rng.permutation(len(neg_idx))[:k], dtype=torch.long, device=device)
            hard_idx = neg_idx[hard_local]
            random_idx = neg_idx[random_local]
            hard_loss = F.softplus(logits[hard_idx]).mean()
            current_loss = task_loss + hard_loss
            exp_idx = hard_idx if experience_mode == "hard" else random_idx
            exp_loss = F.softplus(logits[exp_idx]).mean()
            exp_vec = normalized_grad(exp_loss, model)
            hard_vec_for_log = exp_vec if experience_mode == "hard" else normalized_grad(hard_loss, model)
            mem_vec_before = memory_consensus(memory, device)
            opt.zero_grad()
            current_loss.backward()
            grad = model.grad.detach().clone()
            grad_norm = float(torch.linalg.norm(grad).detach().cpu())
            mem_cos = float("nan")
            if mem_vec_before is not None:
                mem_cos = float(F.cosine_similarity(F.normalize(grad, dim=0), mem_vec_before, dim=0).detach().cpu())
                model.grad = grad + torch.linalg.norm(grad).clamp_min(EPS) * mem_vec_before
            opt.step()
            memory.append(exp_vec.detach().cpu())
            if len(memory) > MEMORY_K:
                memory.pop(0)
            sel_origin = origin[hard_idx.detach().cpu().numpy()]
            exp_origin = origin[exp_idx.detach().cpu().numpy()]
            mem_stats = pairwise_cosine_stats(memory)
            memory_rows.append({
                "variant": variant,
                "epoch": epoch,
                "step": step + 1,
                "experience_mode": experience_mode,
                "memory_size": int(len(memory)),
                "gradient_norm": grad_norm,
                "current_memory_cosine": mem_cos,
                "experience_hard_cosine": float(F.cosine_similarity(exp_vec, hard_vec_for_log, dim=0).detach().cpu()),
                "selected_hard_count": int(hard_idx.numel()),
                "experience_count": int(exp_idx.numel()),
                "hard_score_mean": float(logits.detach()[hard_idx].cpu().numpy().mean()),
                "random_score_mean": float(logits.detach()[random_idx].cpu().numpy().mean()),
                "hard_normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                "hard_anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
                "experience_normal_image_ratio": float(np.mean(exp_origin == "normal_image")),
                "experience_anomaly_mask_out_ratio": float(np.mean(exp_origin == "anomaly_mask_out")),
                **mem_stats,
            })
            vals.append((float(current_loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu()), grad_norm, mem_cos))
        arr = np.asarray(vals, dtype=np.float32)
        finite_cos = arr[:, 6][np.isfinite(arr[:, 6])]
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "hard_loss": float(arr[:, 2].mean()),
            "bce": float(arr[:, 3].mean()),
            "dice": float(arr[:, 4].mean()),
            "gradient_norm": float(arr[:, 5].mean()),
            "current_memory_cosine": float(finite_cos.mean()) if finite_cos.size else float("nan"),
            "memory_size": int(len(memory)),
        }
        curves.append(row)
        print(f"[ExperienceMemory] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} mem_cos={row['current_memory_cosine']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "memory_rows": memory_rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, memory_rows, curves, False


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_HardNormalExperienceMemory")]
        e2 = by[(ds, "E2_RandomNormalExperienceMemory")]
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


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and (e1_b0[0] >= 1.0 or e1_b0[2] >= 1.0):
        outcome = "ERROR_CORRECTION_EXPERIENCE_MEMORY_SUPPORTED"
    elif float(e1_e2.mean()) > 0.5 and not (int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0):
        outcome = "HISTORICAL_HARD_EXPERIENCE_HAS_STRUCTURE_NO_GAIN"
    else:
        outcome = "ERROR_CORRECTION_MEMORY_NO_SIGNAL"
    return outcome, {"mean_E1_minus_B0_G": float(e1_b0.mean()), "mean_E1_minus_E2_G": float(e1_e2.mean()), "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0))}


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_HardNormalExperienceMemory"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_HardNormalExperienceMemory",
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
        "# Error-Correction Experience Memory Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. E1/E2 train only the shared state-contrast direction. FIFO experience memory stores normalized source-training correction gradients only; inference is identical to HardNormal and uses no memory.",
        "",
        "| Dataset | B0 HardNormal G | E1 HardExperience G | E2 RandomExperience G | E1-B0 | E1-E2 |",
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
        "MoECLIP is the primary external gate; VisualAD(CLIP) is retained as a secondary reference. These are directly comparable only at the VisA-source pixel metric level, with implementation and preprocessing caveats.",
        "",
        "| Reference | Dataset | Current G | Reference G | Delta G | Status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for r in sota_rows:
        lines.append(f"| {r['reference']} | {r['dataset']} | {float(r['current_G']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend(["", "Stop-loss: if historical hard-error replay does not beat HardNormal and random-experience replay, do not add retrieval, dynamic K, category slots, or gradient-weight sweeps.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ExperienceMemory] device={device}", flush=True)
    q_pc1, corrupt_payload, align_model, m_c, alpha, init_v, b0_v = load_hardnormal_setup(device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    v_e1, rows_e1, curves_e1, reused_e1 = train_experience_variant("E1_HardNormalExperienceMemory", "hard", groups, m_c, alpha, init_v, device, args.smoke)
    v_e2, rows_e2, curves_e2, reused_e2 = train_experience_variant("E2_RandomNormalExperienceMemory", "random", groups, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_HardNormalExperienceMemory": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_RandomNormalExperienceMemory": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw_new, percat_new = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw_new
    percat = b0_percat + percat_new
    summary_rows = summarize(raw)
    geom = contrast_geometry(b0_v, v_e1, v_e2)
    stats_rows = pool_audit + rows_e1 + rows_e2
    for variant, v in (("B0_HardNormal", b0_v), ("E1_HardNormalExperienceMemory", v_e1), ("E2_RandomNormalExperienceMemory", v_e2)):
        stats_rows.extend(final_negative_distribution(variant, groups, v, m_c, alpha))
    outcome, decision_stats = decide(summary_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "EXPERIENCE_MEMORY_STATS.csv", stats_rows)
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
        "inference_memory": False,
        "memory_contents": "normalized source-training contrast correction gradients only",
        "memory_k": MEMORY_K,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
        "implementation_note": "Gradient replay is applied to the shared contrast parameterization used by this codebase; this is equivalent to a state-pair contrast correction direction under the fixed midpoint/alpha reconstruction.",
    })
    write_report(summary_rows, outcome, decision_stats, geom, sota_rows, runtime)
    print(f"[ExperienceMemory] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
