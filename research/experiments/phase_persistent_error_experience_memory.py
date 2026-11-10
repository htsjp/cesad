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
from experiments.phase_historical_hard_normal_memory import load_setup
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_state_pair_decomposition import hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xliii_adapter_head_decomposition import SEED
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_persistent_error_experience_memory/formal")
HIST_OUT = Path("results/phase_historical_hard_normal_memory/formal")
EPOCHS = 10
MEMORY_K = 2048


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


def coerce_row(row, variant):
    out = dict(row)
    out["variant"] = variant
    for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
        try:
            out[key] = float(out[key])
        except Exception:
            out[key] = float("nan")
    if "mask_images" in out:
        out["mask_images"] = int(float(out["mask_images"]))
    return out


def load_b1_rows(smoke):
    raw, percat = [], []
    for row in read_csv(HIST_OUT / "RAW_METRICS.csv"):
        if row.get("variant") == "E1_HistoricalHardNormalMemory":
            raw.append(coerce_row(row, "B1_FIFOHistoricalHardMemory"))
    for row in read_csv(HIST_OUT / "PER_CATEGORY.csv"):
        if row.get("variant") == "E1_HistoricalHardNormalMemory":
            percat.append(coerce_row(row, "B1_FIFOHistoricalHardMemory"))
    if smoke:
        return raw[:3], percat[:3]
    return raw, percat


class PersistentMemory:
    def __init__(self, cap, dim, device):
        self.cap = int(cap)
        self.device = device
        self.x = torch.empty((0, dim), dtype=torch.float32, device=device)
        self.counts = torch.empty((0,), dtype=torch.long, device=device)
        self.birth = torch.empty((0,), dtype=torch.long, device=device)
        self.origins = []

    def __len__(self):
        return int(self.x.shape[0])

    def enqueue(self, feats, origins, step):
        if feats.numel() == 0:
            return
        feats = feats.detach().to(self.device, dtype=torch.float32)
        n = int(feats.shape[0])
        self.x = torch.cat([self.x, feats], dim=0)
        self.counts = torch.cat([self.counts, torch.zeros((n,), dtype=torch.long, device=self.device)], dim=0)
        self.birth = torch.cat([self.birth, torch.full((n,), int(step), dtype=torch.long, device=self.device)], dim=0)
        self.origins.extend([str(o) for o in origins])

    def update_persistent_counts(self, scores):
        if len(self) == 0:
            return
        k = max(1, int(np.ceil(0.20 * len(self))))
        ids = torch.topk(scores.detach(), k=k, largest=True).indices
        self.counts[ids] += 1

    def prune(self):
        if len(self) <= self.cap:
            return
        keep_n = self.cap
        # Keep highest hardness_count; for ties keep younger entries.
        score = self.counts.to(torch.float64) * 1e9 + self.birth.to(torch.float64)
        keep = torch.topk(score, k=keep_n, largest=True).indices
        keep_sorted = torch.sort(keep).values
        self.x = self.x[keep_sorted]
        self.counts = self.counts[keep_sorted]
        self.birth = self.birth[keep_sorted]
        keep_np = keep_sorted.detach().cpu().numpy()
        self.origins = [self.origins[int(i)] for i in keep_np]

    def replay(self, scores, h, mode):
        if len(self) == 0 or h <= 0:
            return None, None, None
        k = min(int(h), len(self))
        if mode == "persistent":
            priority = self.counts.to(torch.float64) * 1e9 + scores.detach().to(torch.float64)
            ids = torch.topk(priority, k=k, largest=True).indices
        elif mode == "transient":
            priority = -self.counts.to(torch.float64) * 1e9 + scores.detach().to(torch.float64)
            ids = torch.topk(priority, k=k, largest=True).indices
        else:
            raise ValueError(mode)
        return self.x[ids], scores.detach()[ids], ids.detach().cpu().numpy()

    def origin_ratio(self, ids):
        if ids is None or len(ids) == 0:
            return float("nan"), float("nan")
        vals = np.asarray([self.origins[int(i)] for i in ids], dtype=object)
        return float(np.mean(vals == "normal_image")), float(np.mean(vals == "anomaly_mask_out"))

    def age_stats(self, ids, step):
        if ids is None or len(ids) == 0:
            return float("nan"), float("nan"), float("nan")
        ages = np.asarray([int(step) - int(self.birth[int(i)].detach().cpu()) for i in ids], dtype=np.float32)
        return float(ages.mean()), float(np.percentile(ages, 50)), float(np.percentile(ages, 95))

    def count_stats(self, ids=None):
        if len(self) == 0:
            return {k: float("nan") for k in ("count_mean", "count_p50", "count_p90", "count_p95", "persistent_ratio")}
        arr = self.counts.detach().cpu().numpy().astype(np.float32) if ids is None else self.counts[torch.tensor(ids, dtype=torch.long, device=self.device)].detach().cpu().numpy().astype(np.float32)
        return {
            "count_mean": float(arr.mean()),
            "count_p50": float(np.percentile(arr, 50)),
            "count_p90": float(np.percentile(arr, 90)),
            "count_p95": float(np.percentile(arr, 95)),
            "persistent_ratio": float(np.mean(arr >= 2)),
        }


def centroid_cosine(a, b):
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        return float("nan")
    ca = F.normalize(a.mean(dim=0), dim=0)
    cb = F.normalize(b.mean(dim=0), dim=0)
    return float(F.cosine_similarity(ca, cb, dim=0).detach().cpu())


def train_variant(variant, replay_mode, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["memory_rows"], payload["curves"], True
    print(f"[PersistentErrorMemory] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    memory = PersistentMemory(MEMORY_K, len(init_v), device)
    rows, curves = [], []
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
            hard_idx = neg_idx[hard_local]
            hard_loss = F.softplus(logits[hard_idx]).mean()
            hard_origin = origin[hard_idx.detach().cpu().numpy()]

            memory.enqueue(z[hard_idx], hard_origin, global_step)
            mem_scores_for_update = contrast_logits(memory.x, m, alpha, v).detach()
            memory.update_persistent_counts(mem_scores_for_update)
            memory.prune()
            mem_scores = contrast_logits(memory.x, m, alpha, v)
            replay_x, replay_scores, replay_ids = memory.replay(mem_scores, h, replay_mode)
            replay_logits = contrast_logits(replay_x, m, alpha, v)
            history_loss = F.softplus(replay_logits).mean()
            loss = task_loss + hard_loss + history_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            replay_np = replay_scores.detach().cpu().numpy()
            threshold = float(neg_scores.topk(k=h, largest=True).values.min().detach().cpu())
            persistent_hard_ratio = float(np.mean(replay_np >= threshold))
            replay_norm_ratio, replay_maskout_ratio = memory.origin_ratio(replay_ids)
            age_mean, age_p50, age_p95 = memory.age_stats(replay_ids, global_step)
            cstats = memory.count_stats()
            rstats = memory.count_stats(replay_ids)
            rows.append({
                "variant": variant,
                "epoch": epoch,
                "step": step + 1,
                "global_step": global_step,
                "replay_mode": replay_mode,
                "memory_size": len(memory),
                "current_hard_count": int(hard_idx.numel()),
                "history_replay_count": int(replay_x.shape[0]),
                "current_hard_score_mean": float(logits.detach()[hard_idx].cpu().numpy().mean()),
                "history_replay_score_mean": float(replay_np.mean()),
                "history_replay_score_p95": float(np.percentile(replay_np, 95)),
                "persistent_hard_ratio": persistent_hard_ratio,
                "current_history_centroid_cosine": centroid_cosine(z[hard_idx].detach(), replay_x.detach()),
                "history_age_mean_steps": age_mean,
                "history_age_p50_steps": age_p50,
                "history_age_p95_steps": age_p95,
                "current_hard_normal_image_ratio": float(np.mean(hard_origin == "normal_image")),
                "current_hard_anomaly_mask_out_ratio": float(np.mean(hard_origin == "anomaly_mask_out")),
                "replay_normal_image_ratio": replay_norm_ratio,
                "replay_anomaly_mask_out_ratio": replay_maskout_ratio,
                "memory_count_mean": cstats["count_mean"],
                "memory_count_p50": cstats["count_p50"],
                "memory_count_p90": cstats["count_p90"],
                "memory_count_p95": cstats["count_p95"],
                "memory_persistent_ratio": cstats["persistent_ratio"],
                "replay_count_mean": rstats["count_mean"],
                "replay_count_p50": rstats["count_p50"],
                "replay_count_p90": rstats["count_p90"],
                "replay_count_p95": rstats["count_p95"],
                "replay_persistent_ratio": rstats["persistent_ratio"],
                "task_loss": float(task_loss.detach().cpu()),
                "current_hard_loss": float(hard_loss.detach().cpu()),
                "persistent_history_loss": float(history_loss.detach().cpu()),
            })
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(history_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu()), persistent_hard_ratio, cstats["persistent_ratio"]))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "current_hard_loss": float(arr[:, 2].mean()),
            "persistent_history_loss": float(arr[:, 3].mean()),
            "bce": float(arr[:, 4].mean()),
            "dice": float(arr[:, 5].mean()),
            "replay_persistent_hard_ratio": float(arr[:, 6].mean()),
            "memory_persistent_ratio": float(arr[:, 7].mean()),
            "memory_size": len(memory),
        }
        curves.append(row)
        print(f"[PersistentErrorMemory] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} hist={row['persistent_history_loss']:.4f} replay_persist={row['replay_persistent_hard_ratio']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "memory_rows": rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, rows, curves, False


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        b1 = by[(ds, "B1_FIFOHistoricalHardMemory")]
        e1 = by[(ds, "E1_PersistentErrorMemory")]
        e2 = by[(ds, "E2_TransientHardControl")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "B1_G": float(b1["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_B1_G": float(e1["G_pixel"]) - float(b1["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
            "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_b1 = np.asarray([r["E1_minus_B1_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    ok = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_b1.mean()) >= 0.5 and float(e1_e2.mean()) >= 0.5 and (e1_b0[0] >= 1.0 or e1_b0[2] >= 1.0)
    outcome = "PERSISTENT_ERROR_MEMORY_SUPPORTED" if ok else "PERSISTENT_ERROR_MEMORY_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_B1_G": float(e1_b1.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_PersistentErrorMemory"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_PersistentErrorMemory",
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
        "# Persistent Error Experience Memory",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. Memory is source-training-only and stores detached normal patch features with hardness counts and age. Target inference remains one shared HardNormal-style contrast with no memory.",
        "",
        "| Dataset | B0 HardNormal G | B1 FIFOHist G | E1 Persistent G | E2 Transient G | E1-B0 | E1-B1 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['B1_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_B1_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-B1 G: {stats['mean_E1_minus_B1_G']:+.4f} pp.",
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
    lines.extend(["", "Stop-loss: if persistent hard-error memory fails, close this memory route; do not tune K, top ratios, retrieval, attention, category memory, or target memory.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PersistentErrorMemory] device={device}", flush=True)
    q_pc1, corrupt_payload, align_model, m_c, alpha, init_v, b0_v = load_setup(device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    v_e1, rows_e1, _curves_e1, reused_e1 = train_variant("E1_PersistentErrorMemory", "persistent", groups, m_c, alpha, init_v, device, args.smoke)
    v_e2, rows_e2, _curves_e2, reused_e2 = train_variant("E2_TransientHardControl", "transient", groups, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_PersistentErrorMemory": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_TransientHardControl": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw_new, percat_new = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    b1_raw, b1_percat = load_b1_rows(args.smoke)
    raw = b0_raw + b1_raw + raw_new
    percat = b0_percat + b1_percat + percat_new
    summary_rows = summarize(raw)
    geom = contrast_geometry(b0_v, v_e1, v_e2)
    memory_rows = pool_audit + rows_e1 + rows_e2
    for variant, v in (("B0_HardNormal", b0_v), ("E1_PersistentErrorMemory", v_e1), ("E2_TransientHardControl", v_e2)):
        memory_rows.extend(final_negative_distribution(variant, groups, v, m_c, alpha))
    outcome, decision_stats = decide(summary_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "PERSISTENT_MEMORY_STATS.csv", memory_rows)
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
        "memory_contents": "detached projected normal patch features, normal label, hardness_count, age; no category/image/defect identity used",
        "memory_k": MEMORY_K,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, decision_stats, geom, sota_rows, runtime)
    print(f"[PersistentErrorMemory] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
