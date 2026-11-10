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
from experiments.phase_hard_normal_rejection_state import load_b0_rows, load_hardnormal_components
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xliii_adapter_head_decomposition import SEED
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np


OUT = Path("results/phase_one_sided_hard_normal_state_update/formal")
EPOCHS = 10
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


def state_logits(z, s_n, s_a):
    sn = F.normalize(s_n, dim=-1)
    sa = F.normalize(s_a, dim=-1)
    return z @ sa - z @ sn


def one_sided_hard_logits(z, s_n, s_a, side):
    if side == "normal":
        return z @ F.normalize(s_a.detach(), dim=-1) - z @ F.normalize(s_n, dim=-1)
    if side == "anomaly":
        return z @ F.normalize(s_a, dim=-1) - z @ F.normalize(s_n.detach(), dim=-1)
    raise ValueError(side)


def train_variant(variant, hard_side, groups, init_states, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["states"].astype(np.float32), payload["rows"], payload["curves"], True
    print(f"[OneSidedHN] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    s_n = torch.nn.Parameter(torch.tensor(init_states[0], dtype=torch.float32, device=device))
    s_a = torch.nn.Parameter(torch.tensor(init_states[1], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([s_n, s_a], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    rows, curves = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin, _cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            logits = state_logits(z, s_n, s_a)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            pos_idx = torch.nonzero(y > 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            hard_idx = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
            hard_logits = one_sided_hard_logits(z[hard_idx], s_n, s_a, hard_side)
            hard_loss = F.softplus(hard_logits).mean()
            loss = task_loss + hard_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                post_logits = state_logits(z, s_n, s_a)
                hard_before = logits.detach()[hard_idx].cpu().numpy()
                hard_after = post_logits.detach()[hard_idx].cpu().numpy()
                pos_before = logits.detach()[pos_idx].cpu().numpy()
                pos_after = post_logits.detach()[pos_idx].cpu().numpy()
            sel_origin = origin[hard_idx.detach().cpu().numpy()]
            rows.append({
                "variant": variant,
                "epoch": epoch,
                "step": step + 1,
                "hard_side": hard_side,
                "selected_count": int(hard_idx.numel()),
                "hard_normal_score_before_mean": float(hard_before.mean()),
                "hard_normal_score_after_mean": float(hard_after.mean()),
                "hard_normal_score_delta_mean": float(hard_after.mean() - hard_before.mean()),
                "anomaly_positive_score_before_mean": float(pos_before.mean()),
                "anomaly_positive_score_after_mean": float(pos_after.mean()),
                "anomaly_positive_score_delta_mean": float(pos_after.mean() - pos_before.mean()),
                "normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                "anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
                "loss": float(loss.detach().cpu()),
                "task_loss": float(task_loss.detach().cpu()),
                "hard_loss": float(hard_loss.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "dice": float(dice.detach().cpu()),
            })
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu()), float(hard_before.mean()), float(pos_before.mean())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "hard_loss": float(arr[:, 2].mean()),
            "bce": float(arr[:, 3].mean()),
            "dice": float(arr[:, 4].mean()),
            "hard_normal_score_mean": float(arr[:, 5].mean()),
            "anomaly_positive_score_mean": float(arr[:, 6].mean()),
        }
        curves.append(row)
        print(f"[OneSidedHN] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} hard={row['hard_loss']:.4f}", flush=True)
    with torch.no_grad():
        states = torch.stack([F.normalize(s_n, dim=-1), F.normalize(s_a, dim=-1)], dim=0).detach().cpu().numpy().astype(np.float32)
    torch.save({"states": states, "rows": rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return states, rows, curves, False


@torch.no_grad()
def score_target(patch, align_model, q_pc1, states, device, smoke, batch_images=8):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    s = torch.tensor(norm_np(states), dtype=torch.float32, device=device)
    align_model.eval()
    for start in range(0, n, batch_images):
        end = min(start + batch_images, n)
        arr = remove_pc1(np.asarray(patch[start:end], dtype=np.float32), q_pc1)
        b = arr.shape[0]
        flat = norm_np(arr.reshape(-1, arr.shape[-1]))
        x = torch.tensor(flat, dtype=torch.float32, device=device)
        z = align_model.adapt(x)
        logits = z @ s[1] - z @ s[0]
        score = torch.sigmoid(logits).detach().cpu().numpy().reshape(b, SIDE, SIDE).astype(np.float32)
        maps[start:end] = score.astype(np.float16)
        image[start:end] = np.asarray([top1_image_score(m) for m in score], dtype=np.float32)
    return image, maps


def evaluate(states_by_variant, align_model, q_pc1, device, smoke):
    raw, percat, fp_rows = [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        if cell_id not in ("MVTec", "BTAD", "CVC-ClinicDB"):
            continue
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, states in states_by_variant.items():
            print(f"[OneSidedHN] eval {variant} on {cell_id}", flush=True)
            image, maps = score_target(patch, align_model, q_pc1, states, device, smoke)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
            normal_vals = []
            for pos in range(n):
                item = dataset[int(idx_use[pos])] if kind == "source_dataset" else dataset[pos]
                mask = item["img_mask"].float()
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.ndim == 3:
                    mask = mask.unsqueeze(0)
                small = F.interpolate(mask, size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape(-1).numpy() > 0.5
                normal_vals.append(np.asarray(maps[pos], dtype=np.float32).reshape(-1)[~small])
            arr = np.concatenate(normal_vals).astype(np.float32)
            fp_rows.append({"dataset": cell_id, "variant": variant, "normal_pixel_count": int(arr.size), "normal_score_mean": float(arr.mean()), "normal_score_p95": float(np.percentile(arr, 95))})
    return raw, percat, fp_rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_NormalSideHardUpdate")]
        e2 = by[(ds, "E2_AnomalySideHardUpdate")]
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


def state_geometry(b0_states, e1_states, e2_states):
    rows = []
    for name, states in (("E1_NormalSideHardUpdate", e1_states), ("E2_AnomalySideHardUpdate", e2_states)):
        rows.append({
            "variant": name,
            "cos_sN_B0_variant": float(unit_np(b0_states[0]) @ unit_np(states[0])),
            "cos_sA_B0_variant": float(unit_np(b0_states[1]) @ unit_np(states[1])),
            "contrast_cos_B0_variant": float(unit_np(b0_states[1] - b0_states[0]) @ unit_np(states[1] - states[0])),
            "sN_norm": float(np.linalg.norm(states[0])),
            "sA_norm": float(np.linalg.norm(states[1])),
        })
    rows.append({
        "variant": "E1_vs_E2",
        "cos_sN_B0_variant": float(unit_np(e1_states[0]) @ unit_np(e2_states[0])),
        "cos_sA_B0_variant": float(unit_np(e1_states[1]) @ unit_np(e2_states[1])),
        "contrast_cos_B0_variant": float(unit_np(e1_states[1] - e1_states[0]) @ unit_np(e2_states[1] - e2_states[0])),
        "sN_norm": "",
        "sA_norm": "",
    })
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and (e1_b0[0] >= 1.5 or e1_b0[2] >= 1.5) and e1_b0[1] >= -1.0:
        outcome = "ONE_SIDED_HARD_NORMAL_SUPPORTED"
    elif float(e1_e2.mean()) >= 0.5 and float(e1_b0.mean()) < 1.0:
        outcome = "NORMAL_SIDE_ASYMMETRY_CONFIRMED_NO_METHOD_GAIN"
    elif abs(float(e1_b0.mean())) < 0.5 and abs(float(e1_e2.mean())) < 0.5:
        outcome = "ONE_SIDED_UPDATE_NO_SIGNAL"
    else:
        outcome = "ONE_SIDED_UPDATE_NO_SIGNAL"
    return outcome, {"mean_E1_minus_B0_G": float(e1_b0.mean()), "mean_E1_minus_E2_G": float(e1_e2.mean()), "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0))}


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_NormalSideHardUpdate"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_NormalSideHardUpdate",
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


def write_report(summary_rows, outcome, stats, sota_rows, runtime):
    lines = [
        "# One-Sided Hard-Normal State Update",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. E1/E2 train explicit normal/anomaly state parameters with the existing aligned source features; L_task updates both states, while the extra hard-normal loss updates only the normal side or anomaly side respectively. Target inference uses the same frozen alignment and two-state scorer.",
        "",
        "| Dataset | B0 HardNormal G | E1 NormalSide G | E2 AnomalySide G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
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
    lines.extend(["", "Stop-loss: if one-sided hard-normal updates do not beat symmetric HardNormal and anomaly-side control, do not sweep loss weights, hard ratios, adapters, Memory, or target tuning.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OneSidedHN] device={device}", flush=True)
    q_pc1, _payload, align_model, states_hn, m_c, alpha, v_hn = load_hardnormal_components(device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    e1_states, rows_e1, _curves_e1, reused_e1 = train_variant("E1_NormalSideHardUpdate", "normal", groups, states_hn, device, args.smoke)
    e2_states, rows_e2, _curves_e2, reused_e2 = train_variant("E2_AnomalySideHardUpdate", "anomaly", groups, states_hn, device, args.smoke)
    raw_new, percat_new, fp_rows = evaluate({"E1_NormalSideHardUpdate": e1_states, "E2_AnomalySideHardUpdate": e2_states}, align_model, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw_new
    percat = b0_percat + percat_new
    summary_rows = summarize(raw)
    geom_rows = state_geometry(states_hn, e1_states, e2_states)
    train_rows = pool_audit + rows_e1 + rows_e2
    for variant, states in (("B0_HardNormal", states_hn), ("E1_NormalSideHardUpdate", e1_states), ("E2_AnomalySideHardUpdate", e2_states)):
        train_rows.extend(final_negative_distribution(variant, groups, unit_np(states[1] - states[0]), m_c, alpha))
    outcome, decision_stats = decide(summary_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "STATE_GEOMETRY.csv", geom_rows)
    write_csv(OUT / "ONE_SIDED_TRAINING_STATS.csv", train_rows)
    write_csv(OUT / "TARGET_FALSE_POSITIVE_STATS.csv", fp_rows)
    np.savez_compressed(OUT / "states.npz", B0=states_hn.astype(np.float32), E1=e1_states.astype(np.float32), E2=e2_states.astype(np.float32), v_hn=np.asarray(v_hn, dtype=np.float32))
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
        "implementation_note": "Existing codebase HardNormal is contrast-vector based; this audit expands it to explicit sN/sA states initialized from the validated HardNormal state reconstruction so the extra hard-normal loss can be one-sided.",
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, decision_stats, sota_rows, runtime)
    print(f"[OneSidedHN] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
