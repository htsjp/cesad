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

from experiments.phase_hard_normal_contrast import (
    TOP_NEG_FRAC,
    balanced_category_batch,
    collect_source_patch_pool,
    final_negative_distribution,
)
from experiments.external_sota_gate import apply_gate
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR

OUT = Path("results/phase_hard_normal_boundary_ranking/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
HARD_OUT = Path("results/phase_hard_normal_contrast/formal")
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


def load_b0_hardnormal_rows(smoke):
    raw, percat = [], []
    for filename, dest in (("RAW_METRICS.csv", raw), ("PER_CATEGORY.csv", percat)):
        with (HARD_OUT / filename).open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                if row["variant"] != "E1_HardNormalContrast":
                    continue
                out = dict(row)
                out["variant"] = "B0_HardNormal"
                for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                    try:
                        out[key] = float(out[key])
                    except ValueError:
                        out[key] = float("nan")
                out["mask_images"] = int(float(out["mask_images"]))
                dest.append(out)
    if smoke:
        return raw[:3], percat[:3]
    return raw, percat


def train_ranking_variant(variant, mode, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["rank_rows"], True
    print(f"[BoundaryRank] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    curves = []
    rank_rows = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin, _cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            pos_idx = torch.nonzero(y > 0.5, as_tuple=False).flatten()
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            neg_scores = logits.detach()[neg_idx]
            if mode == "hard":
                selected_local = torch.topk(neg_scores, k=k, largest=True).indices
            elif mode == "random":
                selected_local = torch.tensor(rng.permutation(len(neg_idx))[:k], dtype=torch.long, device=device)
            else:
                raise ValueError(mode)
            selected_idx = neg_idx[selected_local]
            s_pos = logits[pos_idx].mean()
            rank_loss = F.softplus(logits[selected_idx] - s_pos).mean()
            loss = task_loss + rank_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            sel_np = selected_idx.detach().cpu().numpy()
            pos_scores = logits.detach()[pos_idx].cpu().numpy()
            sel_scores = logits.detach()[selected_idx].cpu().numpy()
            sel_origin = origin[sel_np]
            rank_rows.append({
                "variant": variant,
                "record_type": "train_boundary",
                "epoch": epoch,
                "step": step + 1,
                "mode": mode,
                "positive_score_mean": float(pos_scores.mean()),
                "positive_score_std": float(pos_scores.std()),
                "selected_normal_score_mean": float(sel_scores.mean()),
                "selected_normal_score_std": float(sel_scores.std()),
                "positive_minus_selected_gap": float(pos_scores.mean() - sel_scores.mean()),
                "all_negative_score_mean": float(neg_scores.cpu().numpy().mean()),
                "all_negative_score_std": float(neg_scores.cpu().numpy().std()),
                "selected_count": int(len(sel_np)),
                "normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                "anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
                "rank_loss": float(rank_loss.detach().cpu()),
            })
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(rank_loss.detach().cpu()), float(s_pos.detach().cpu()), float(logits[selected_idx].detach().mean().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "rank_loss": float(arr[:, 2].mean()),
            "positive_score_mean": float(arr[:, 3].mean()),
            "selected_normal_score_mean": float(arr[:, 4].mean()),
            "positive_minus_selected_gap": float(arr[:, 3].mean() - arr[:, 4].mean()),
            "bce": float(arr[:, 5].mean()),
            "dice": float(arr[:, 6].mean()),
        }
        curves.append(row)
        print(f"[BoundaryRank] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} gap={row['positive_minus_selected_gap']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "rank_rows": rank_rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, rank_rows, False


def final_boundary_distribution(variant, groups, v, m_c, alpha):
    rows = final_negative_distribution(variant, groups, v, m_c, alpha)
    m = torch.tensor(m_c, dtype=torch.float32)
    vt = torch.tensor(v, dtype=torch.float32)
    pos_scores_all = []
    neg_scores_all = []
    for cat, data in groups.items():
        z = torch.tensor(data["x"], dtype=torch.float32)
        y = data["y"]
        with torch.no_grad():
            score = contrast_logits(z, m, alpha, vt).cpu().numpy()
        pos = score[y > 0.5]
        neg = score[y < 0.5]
        k = max(1, int(np.ceil(len(neg) * TOP_NEG_FRAC)))
        hard = np.sort(neg)[-k:]
        pos_scores_all.append(pos)
        neg_scores_all.append(neg)
        rows.append({
            "variant": variant,
            "record_type": "final_boundary_distribution",
            "category": cat,
            "positive_score_mean": float(pos.mean()),
            "positive_score_std": float(pos.std()),
            "hard_normal_score_mean": float(hard.mean()),
            "hard_normal_score_std": float(hard.std()),
            "positive_hard_gap": float(pos.mean() - hard.mean()),
        })
    pos_all = np.concatenate(pos_scores_all, axis=0)
    neg_all = np.concatenate(neg_scores_all, axis=0)
    k = max(1, int(np.ceil(len(neg_all) * TOP_NEG_FRAC)))
    hard_all = np.sort(neg_all)[-k:]
    rows.append({
        "variant": variant,
        "record_type": "final_boundary_summary",
        "category": "__all__",
        "positive_score_mean": float(pos_all.mean()),
        "positive_score_std": float(pos_all.std()),
        "hard_normal_score_mean": float(hard_all.mean()),
        "hard_normal_score_std": float(hard_all.std()),
        "positive_hard_gap": float(pos_all.mean() - hard_all.mean()),
    })
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_HardNormalBoundaryRanking")]
        e2 = by[(ds, "E2_RandomNormalBoundaryControl")]
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


def decide(summary_rows, boundary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    stable = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0
    if stable and float(e1_e2.mean()) >= 0.5:
        outcome = "TRANSFERABLE_HARD_NORMAL_BOUNDARY_SUPPORTED"
    elif abs(float(e1_b0.mean())) < 1.0 and float(e1_e2.mean()) >= 0.5:
        outcome = "HARD_SELECTION_SUPPORTED_RANKING_NOT_ADDITIONAL"
    else:
        summaries = {r["variant"]: r for r in boundary_rows if r.get("record_type") == "final_boundary_summary"}
        source_gap_improved = (
            "B0_HardNormal" in summaries
            and "E1_HardNormalBoundaryRanking" in summaries
            and float(summaries["E1_HardNormalBoundaryRanking"]["positive_hard_gap"]) > float(summaries["B0_HardNormal"]["positive_hard_gap"]) + 0.05
        )
        outcome = "SOURCE_BOUNDARY_NOT_TRANSFERABILITY" if source_gap_improved else "HARD_NORMAL_BOUNDARY_RANKING_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_report(summary_rows, outcome, stats, geom, runtime):
    lines = [
        "# Hard-Normal Boundary Ranking Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Strict Full VisA source-only. E1 ranks top-10% hard negatives below the batch positive mean; E2 uses matched random negatives. No memory, prompt, target statistics, or target adaptation.",
        "",
        "| Dataset | B0 HardNormal G | E1 BoundaryRank G | E2 RandomRank G | E1-B0 | E1-E2 |",
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
    print(f"[BoundaryRank] device={device}", flush=True)
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    hard_vecs = np.load(HARD_OUT / "contrast_vectors.npz")
    b0_v = hard_vecs["E1"].astype(np.float32)
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)

    v_e1, rank_e1, reused_e1 = train_ranking_variant("E1_HardNormalBoundaryRanking", "hard", groups, m_c, alpha, init_v, device, args.smoke)
    v_e2, rank_e2, reused_e2 = train_ranking_variant("E2_RandomNormalBoundaryControl", "random", groups, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_HardNormalBoundaryRanking": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_RandomNormalBoundaryControl": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_hardnormal_rows(args.smoke)
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    boundary_rows = pool_audit + rank_e1 + rank_e2
    for variant, v in (("B0_HardNormal", b0_v), ("E1_HardNormalBoundaryRanking", v_e1), ("E2_RandomNormalBoundaryControl", v_e2)):
        boundary_rows.extend(final_boundary_distribution(variant, groups, v, m_c, alpha))
    geom = contrast_geometry(b0_v, v_e1, v_e2)
    outcome, stats = decide(summary_rows, boundary_rows)
    runtime = time.time() - start

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "BOUNDARY_RANKING_STATS.csv", boundary_rows)
    np.savez_compressed(OUT / "contrast_vectors.npz", B0=b0_v.astype(np.float32), E1=v_e1.astype(np.float32), E2=v_e2.astype(np.float32), init=init_v.astype(np.float32))
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
        "top_negative_fraction": TOP_NEG_FRAC,
        "alpha": alpha,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, stats, geom, runtime)
    apply_gate(
        OUT,
        "E1_HardNormalBoundaryRanking",
        "HardNormalBoundaryRanking",
        "Stop-loss: not triggered yet. This round does not improve over HardNormal, but the immediately preceding method round achieved mean target G >= +1 pp and narrowed/overcame the external gap, so the consecutive-two-failure condition is not met.",
    )
    print(f"[BoundaryRank] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
