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


OUT = Path("results/phase_historical_hard_normal_memory/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
EPOCHS = 10
MEMORY_K = 2048
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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


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
    align_model = make_model(states_c, corrupt_payload, device)
    vecs = np.load(HN_OUT / "contrast_vectors.npz")
    b0_v = unit_np(np.asarray(vecs["E1"], dtype=np.float32))
    return q_pc1, corrupt_payload, align_model, m_c, alpha, init_v, b0_v


class FeatureMemory:
    def __init__(self, cap, dim, device):
        self.cap = int(cap)
        self.dim = int(dim)
        self.device = device
        self.x = torch.empty((0, dim), dtype=torch.float32, device=device)
        self.origins = []
        self.insert_steps = []

    def __len__(self):
        return int(self.x.shape[0])

    def enqueue(self, feats, origins, global_step):
        if feats.numel() == 0:
            return
        feats = feats.detach().to(self.device, dtype=torch.float32)
        self.x = torch.cat([self.x, feats], dim=0)
        self.origins.extend([str(o) for o in origins])
        self.insert_steps.extend([int(global_step)] * int(feats.shape[0]))
        if len(self) > self.cap:
            excess = len(self) - self.cap
            self.x = self.x[excess:]
            self.origins = self.origins[excess:]
            self.insert_steps = self.insert_steps[excess:]

    def top_replay(self, m, alpha, v, h):
        if len(self) == 0 or h <= 0:
            return None, None, None
        scores = contrast_logits(self.x, m, alpha, v)
        k = min(int(h), len(self))
        vals, ids = torch.topk(scores.detach(), k=k, largest=True)
        return self.x[ids], vals, ids.detach().cpu().numpy()

    def origin_ratio(self, ids):
        if ids is None or len(ids) == 0:
            return float("nan"), float("nan")
        vals = np.asarray([self.origins[int(i)] for i in ids], dtype=object)
        return float(np.mean(vals == "normal_image")), float(np.mean(vals == "anomaly_mask_out"))

    def age_stats(self, ids, global_step):
        if ids is None or len(ids) == 0:
            return float("nan"), float("nan"), float("nan")
        ages = np.asarray([int(global_step) - self.insert_steps[int(i)] for i in ids], dtype=np.float32)
        return float(ages.mean()), float(np.percentile(ages, 50)), float(np.percentile(ages, 95))


def centroid_cosine(a, b):
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        return float("nan")
    ca = F.normalize(a.mean(dim=0), dim=0)
    cb = F.normalize(b.mean(dim=0), dim=0)
    return float(F.cosine_similarity(ca, cb, dim=0).detach().cpu())


def train_memory_variant(variant, enqueue_mode, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["memory_rows"], payload["curves"], True
    print(f"[HistoricalHNMemory] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    memory = FeatureMemory(MEMORY_K, len(init_v), device)
    memory_rows = []
    curves = []
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            global_step += 1
            xb, yb, origin, _cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            h = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            neg_scores = logits.detach()[neg_idx]
            hard_local = torch.topk(neg_scores, k=h, largest=True).indices
            random_local = torch.tensor(rng.permutation(len(neg_idx))[:h], dtype=torch.long, device=device)
            hard_idx = neg_idx[hard_local]
            random_idx = neg_idx[random_local]
            hard_loss = F.softplus(logits[hard_idx]).mean()
            replay_x, replay_scores, replay_ids = memory.top_replay(m, alpha, v, h)
            if replay_x is None:
                history_loss = torch.zeros((), dtype=torch.float32, device=device)
                replay_mean = float("nan")
                replay_p95 = float("nan")
                persistent_hard_ratio = float("nan")
            else:
                replay_logits = contrast_logits(replay_x, m, alpha, v)
                history_loss = F.softplus(replay_logits).mean()
                replay_np = replay_scores.detach().cpu().numpy()
                replay_mean = float(replay_np.mean())
                replay_p95 = float(np.percentile(replay_np, 95))
                threshold = float(neg_scores.topk(k=h, largest=True).values.min().detach().cpu())
                persistent_hard_ratio = float(np.mean(replay_np >= threshold))
            loss = task_loss + hard_loss + history_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            enqueue_idx = hard_idx if enqueue_mode == "hard" else random_idx
            enqueue_origins = origin[enqueue_idx.detach().cpu().numpy()]
            memory.enqueue(z[enqueue_idx], enqueue_origins, global_step)
            replay_normal_ratio, replay_maskout_ratio = memory.origin_ratio(replay_ids)
            age_mean, age_p50, age_p95 = memory.age_stats(replay_ids, global_step)
            current_replay_cos = centroid_cosine(z[hard_idx].detach(), replay_x.detach() if replay_x is not None else None)
            sel_origin = origin[hard_idx.detach().cpu().numpy()]
            enq_origin = origin[enqueue_idx.detach().cpu().numpy()]
            memory_rows.append({
                "variant": variant,
                "epoch": epoch,
                "step": step + 1,
                "global_step": global_step,
                "enqueue_mode": enqueue_mode,
                "memory_size": len(memory),
                "current_hard_count": int(hard_idx.numel()),
                "history_replay_count": int(0 if replay_x is None else replay_x.shape[0]),
                "current_hard_score_mean": float(logits.detach()[hard_idx].cpu().numpy().mean()),
                "random_negative_score_mean": float(logits.detach()[random_idx].cpu().numpy().mean()),
                "history_replay_score_mean": replay_mean,
                "history_replay_score_p95": replay_p95,
                "persistent_hard_ratio": persistent_hard_ratio,
                "current_history_centroid_cosine": current_replay_cos,
                "history_age_mean_steps": age_mean,
                "history_age_p50_steps": age_p50,
                "history_age_p95_steps": age_p95,
                "current_hard_normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                "current_hard_anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
                "enqueue_normal_image_ratio": float(np.mean(enq_origin == "normal_image")),
                "enqueue_anomaly_mask_out_ratio": float(np.mean(enq_origin == "anomaly_mask_out")),
                "replay_normal_image_ratio": replay_normal_ratio,
                "replay_anomaly_mask_out_ratio": replay_maskout_ratio,
                "task_loss": float(task_loss.detach().cpu()),
                "current_hard_loss": float(hard_loss.detach().cpu()),
                "history_hard_loss": float(history_loss.detach().cpu()),
            })
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(history_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu()), replay_mean, persistent_hard_ratio))
        arr = np.asarray(vals, dtype=np.float32)
        replay = arr[:, 6][np.isfinite(arr[:, 6])]
        persist = arr[:, 7][np.isfinite(arr[:, 7])]
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "current_hard_loss": float(arr[:, 2].mean()),
            "history_hard_loss": float(arr[:, 3].mean()),
            "bce": float(arr[:, 4].mean()),
            "dice": float(arr[:, 5].mean()),
            "history_replay_score_mean": float(replay.mean()) if replay.size else float("nan"),
            "persistent_hard_ratio": float(persist.mean()) if persist.size else float("nan"),
            "memory_size": len(memory),
        }
        curves.append(row)
        print(f"[HistoricalHNMemory] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} hist={row['history_hard_loss']:.4f} persist={row['persistent_hard_ratio']:.4f}", flush=True)
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
        e1 = by[(ds, "E1_HistoricalHardNormalMemory")]
        e2 = by[(ds, "E2_RandomHistoricalMemory")]
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
        outcome = "HISTORICAL_HARD_NORMAL_MEMORY_SUPPORTED"
    elif float(e1_e2.mean()) > 0.5 and not (int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0):
        outcome = "HISTORICAL_ERRORS_HAVE_STRUCTURE_NO_GAIN"
    else:
        outcome = "HISTORICAL_HARD_NORMAL_MEMORY_NO_SIGNAL"
    return outcome, {"mean_E1_minus_B0_G": float(e1_b0.mean()), "mean_E1_minus_E2_G": float(e1_e2.mean()), "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0))}


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_HistoricalHardNormalMemory"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_HistoricalHardNormalMemory",
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
        "# Historical Hard-Normal Experience Memory",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. Memory exists only during source training and stores detached normal patch features; target inference is identical to HardNormal with one shared contrast.",
        "",
        "| Dataset | B0 HardNormal G | E1 HistoricalHardMemory G | E2 RandomHistoricalMemory G | E1-B0 | E1-E2 |",
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
        "MoECLIP is the primary external gate; VisualAD(CLIP) is retained as a secondary reference. These comparisons are limited to VisA-source pixel metrics with implementation/preprocessing caveats.",
        "",
        "| Reference | Dataset | Current G | Reference G | Delta G | Status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for r in sota_rows:
        lines.append(f"| {r['reference']} | {r['dataset']} | {float(r['current_G']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend(["", "Stop-loss: if historical patch replay does not beat HardNormal and random historical replay, do not sweep K/top ratio or add inference memory, category memory, learned attention, or target tuning.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HistoricalHNMemory] device={device}", flush=True)
    q_pc1, corrupt_payload, align_model, m_c, alpha, init_v, b0_v = load_setup(device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    v_e1, rows_e1, _curves_e1, reused_e1 = train_memory_variant("E1_HistoricalHardNormalMemory", "hard", groups, m_c, alpha, init_v, device, args.smoke)
    v_e2, rows_e2, _curves_e2, reused_e2 = train_memory_variant("E2_RandomHistoricalMemory", "random", groups, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_HistoricalHardNormalMemory": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_RandomHistoricalMemory": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
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
    memory_rows = pool_audit + rows_e1 + rows_e2
    for variant, v in (("B0_HardNormal", b0_v), ("E1_HistoricalHardNormalMemory", v_e1), ("E2_RandomHistoricalMemory", v_e2)):
        memory_rows.extend(final_negative_distribution(variant, groups, v, m_c, alpha))
    outcome, decision_stats = decide(summary_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "HISTORICAL_MEMORY_STATS.csv", memory_rows)
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
        "memory_phase": "source_training_only",
        "memory_contents": "detached projected normal patch features and normal labels; diagnostic origin/age are not used by training",
        "memory_k": MEMORY_K,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, decision_stats, geom, sota_rows, runtime)
    print(f"[HistoricalHNMemory] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
