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
from experiments.phase_hard_guided_selective_patch_adapter import (
    Rank4PatchAdapter,
    hardnormal_logits,
    source_diagnostics,
)
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC, balanced_category_batch, collect_source_patch_pool
from experiments.phase_hard_normal_rejection_state import load_b0_rows, load_hardnormal_components
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xliii_adapter_head_decomposition import SEED
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, WEIGHT_DECAY, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np


OUT = Path("results/phase_anomaly_preserved_hard_adapter/formal")
B1_OUT = Path("results/phase_hard_guided_selective_patch_adapter/formal")
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
    for row in read_csv(B1_OUT / "RAW_METRICS.csv"):
        if row.get("variant") == "E1_HardGuidedSelectiveAdapter":
            raw.append(coerce_row(row, "B1_HardGuidedAdapter"))
    for row in read_csv(B1_OUT / "PER_CATEGORY.csv"):
        if row.get("variant") == "E1_HardGuidedSelectiveAdapter":
            percat.append(coerce_row(row, "B1_HardGuidedAdapter"))
    if smoke:
        return raw[:3], percat[:3]
    return raw, percat


def train_variant(variant, selection_mode, groups, m_c, alpha, v_hn, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model = Rank4PatchAdapter(int(payload["dim"])).to(device)
        model.load_state_dict(payload["adapter"], strict=True)
        return model.eval(), payload["train_rows"], payload["curves"], True
    print(f"[AnomPreservedAdapter] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    dim = next(iter(groups.values()))["x"].shape[1]
    adapter = Rank4PatchAdapter(dim).to(device)
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    v = torch.tensor(v_hn, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    rows, curves = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin, _cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            with torch.no_grad():
                base_logits = hardnormal_logits(z, m, alpha, v)
                neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
                pos_idx = torch.nonzero(y > 0.5, as_tuple=False).flatten()
                k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
                if selection_mode == "hard":
                    selected_local = torch.topk(base_logits[neg_idx], k=k, largest=True).indices
                elif selection_mode == "random":
                    selected_local = torch.tensor(rng.permutation(len(neg_idx))[:k], dtype=torch.long, device=device)
                else:
                    raise ValueError(selection_mode)
                selected_idx = neg_idx[selected_local]
                mask = torch.zeros_like(y, dtype=torch.float32)
                mask[pos_idx] = 1.0
                mask[selected_idx] = 1.0
            z_adapt = adapter(z)
            z_mixed = F.normalize(z + mask.unsqueeze(-1) * (z_adapt - z), dim=-1)
            logits = hardnormal_logits(z_mixed, m, alpha, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            hard_loss = F.softplus(logits[selected_idx]).mean()
            keep_loss = (1.0 - (z_adapt[pos_idx] * z[pos_idx]).sum(dim=-1)).mean()
            loss = task_loss + hard_loss + keep_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                residual = torch.linalg.norm(z_adapt - z, dim=-1).detach().cpu().numpy()
                cos_za = (z * z_adapt).sum(dim=-1).detach().cpu().numpy()
                base_pos = base_logits[pos_idx].detach().cpu().numpy()
                after_pos = logits[pos_idx].detach().cpu().numpy()
                base_sel = base_logits[selected_idx].detach().cpu().numpy()
                after_sel = logits[selected_idx].detach().cpu().numpy()
            sel_np = selected_idx.detach().cpu().numpy()
            pos_np = pos_idx.detach().cpu().numpy()
            easy_np = np.setdiff1d(np.arange(len(yb)), np.concatenate([sel_np, pos_np]), assume_unique=False)
            sel_origin = origin[sel_np]
            rows.append({
                "variant": variant,
                "epoch": epoch,
                "step": step + 1,
                "selection_mode": selection_mode,
                "selected_count": int(selected_idx.numel()),
                "hard_score_before_mean": float(base_sel.mean()),
                "hard_score_after_mean": float(after_sel.mean()),
                "hard_score_delta_mean": float(after_sel.mean() - base_sel.mean()),
                "anomaly_score_before_mean": float(base_pos.mean()),
                "anomaly_score_after_mean": float(after_pos.mean()),
                "anomaly_score_delta_mean": float(after_pos.mean() - base_pos.mean()),
                "keep_loss": float(keep_loss.detach().cpu()),
                "residual_norm_hard_mean": float(residual[sel_np].mean()) if sel_np.size else float("nan"),
                "residual_norm_anomaly_mean": float(residual[pos_np].mean()) if pos_np.size else float("nan"),
                "residual_norm_easy_mean": float(residual[easy_np].mean()) if easy_np.size else float("nan"),
                "cos_z_Az_hard_mean": float(cos_za[sel_np].mean()) if sel_np.size else float("nan"),
                "cos_z_Az_anomaly_mean": float(cos_za[pos_np].mean()) if pos_np.size else float("nan"),
                "cos_z_Az_easy_mean": float(cos_za[easy_np].mean()) if easy_np.size else float("nan"),
                "normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                "anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
                "loss": float(loss.detach().cpu()),
                "task_loss": float(task_loss.detach().cpu()),
                "hard_loss": float(hard_loss.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "dice": float(dice.detach().cpu()),
            })
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(keep_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "hard_loss": float(arr[:, 2].mean()),
            "keep_loss": float(arr[:, 3].mean()),
            "bce": float(arr[:, 4].mean()),
            "dice": float(arr[:, 5].mean()),
        }
        curves.append(row)
        print(f"[AnomPreservedAdapter] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} hard={row['hard_loss']:.4f} keep={row['keep_loss']:.4f}", flush=True)
    torch.save({"adapter": adapter.eval().state_dict(), "dim": dim, "train_rows": rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return adapter.eval(), rows, curves, False


@torch.no_grad()
def score_adapter_target(patch, base_align, adapter, q_pc1, m_c, alpha, v_hn, device, smoke, batch_images=8):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    v = torch.tensor(v_hn, dtype=torch.float32, device=device)
    base_align.eval()
    adapter.eval()
    for start in range(0, n, batch_images):
        end = min(start + batch_images, n)
        arr = remove_pc1(np.asarray(patch[start:end], dtype=np.float32), q_pc1)
        b = arr.shape[0]
        flat = norm_np(arr.reshape(-1, arr.shape[-1]))
        x = torch.tensor(flat, dtype=torch.float32, device=device)
        z = base_align.adapt(x)
        za = adapter(z)
        score = torch.sigmoid(hardnormal_logits(za, m, alpha, v)).detach().cpu().numpy().reshape(b, SIDE, SIDE).astype(np.float32)
        maps[start:end] = score.astype(np.float16)
        image[start:end] = np.asarray([top1_image_score(s) for s in score], dtype=np.float32)
    return image, maps


def evaluate_models(models, q_pc1, align_model, m_c, alpha, v_hn, device, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        if cell_id not in ("MVTec", "BTAD", "CVC-ClinicDB"):
            continue
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, adapter in models.items():
            print(f"[AnomPreservedAdapter] eval {variant} on {cell_id}", flush=True)
            image, maps = score_adapter_target(patch, align_model, adapter, q_pc1, m_c, alpha, v_hn, device, smoke)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        b1 = by[(ds, "B1_HardGuidedAdapter")]
        e1 = by[(ds, "E1_AnomalyPreservedHardAdapter")]
        e2 = by[(ds, "E2_RandomGuidedAnomalyPreserved")]
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
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_b1.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and (e1_b0[0] >= 1.5 or e1_b0[2] >= 1.5) and e1_b0[1] >= -1.0:
        outcome = "ANOMALY_PRESERVED_HARD_ADAPTER_SUPPORTED"
    elif float(e1_b1.mean()) >= 1.0 and float(e1_b0.mean()) < 1.0:
        outcome = "ANOMALY_PRESERVATION_FIXES_ADAPTER_BUT_NO_METHOD_GAIN"
    elif abs(float(e1_e2.mean())) < 0.5:
        outcome = "GENERIC_ADAPTER_PRESERVATION_NOT_HARD_SPECIFIC"
    else:
        outcome = "ANOMALY_PRESERVED_ADAPTER_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_B1_G": float(e1_b1.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_AnomalyPreservedHardAdapter"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_AnomalyPreservedHardAdapter",
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
        "# Anomaly-Preserved Hard-Guided Adapter",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. CLIP, existing StateAligned alignment, and HardNormal scorer are frozen. E1/E2 train one rank4 residual adapter with anomaly identity preservation; target inference applies the adapter uniformly to every patch.",
        "",
        "| Dataset | B0 HardNormal G | B1 HardGuided G | E1 PreservedHard G | E2 PreservedRandom G | E1-B0 | E1-B1 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['B1_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_B1_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-B1 G: {stats['mean_E1_minus_B1_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        f"Frozen contrast cosine proxy: B0/E1={geom[0]['cos_v_B0_v_E1']:.6f}, B0/E2={geom[0]['cos_v_B0_v_E2']:.6f}, E1/E2={geom[0]['cos_v_E1_v_E2']:.6f}.",
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
    lines.extend(["", "Stop-loss: if anomaly preservation does not fix adapter transfer, do not sweep preservation weight/rank or add dual branches, routers, Memory, or target tuning.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AnomPreservedAdapter] device={device}", flush=True)
    q_pc1, _payload, align_model, _states_hn, m_c, alpha, v_hn = load_hardnormal_components(device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    e1, rows_e1, _curves_e1, reused_e1 = train_variant("E1_AnomalyPreservedHardAdapter", "hard", groups, m_c, alpha, v_hn, device, args.smoke)
    e2, rows_e2, _curves_e2, reused_e2 = train_variant("E2_RandomGuidedAnomalyPreserved", "random", groups, m_c, alpha, v_hn, device, args.smoke)
    raw_new, percat_new = evaluate_models({"E1_AnomalyPreservedHardAdapter": e1, "E2_RandomGuidedAnomalyPreserved": e2}, q_pc1, align_model, m_c, alpha, v_hn, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    b1_raw, b1_percat = load_b1_rows(args.smoke)
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + b1_raw + raw_new
    percat = b0_percat + b1_percat + percat_new
    summary_rows = summarize(raw)
    outcome, decision_stats = decide(summary_rows)
    diag_rows = pool_audit + rows_e1 + rows_e2 + source_diagnostics(groups, {"E1_AnomalyPreservedHardAdapter": e1, "E2_RandomGuidedAnomalyPreserved": e2}, m_c, alpha, v_hn, device)
    geom = contrast_geometry(v_hn, v_hn, v_hn)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "ADAPTER_DIAGNOSTICS.csv", diag_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "clip_frozen": True,
        "hardnormal_scorer_frozen": True,
        "adapter_rank": 4,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "preservation_loss": "mean_pos_1_minus_cosine_Az_z",
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, decision_stats, geom, sota_rows, runtime)
    print(f"[AnomPreservedAdapter] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
