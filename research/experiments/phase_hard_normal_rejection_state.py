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
    CACHE_VERSION,
    PC1_DEBIAS_OUT,
    TOP_NEG_FRAC,
    collect_source_patch_pool,
    final_negative_distribution,
    json_default,
    load_payload,
)
from experiments.phase_liv_memory_support_modes import eval_maps
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_spatial_supervision_transfer_audit import target_cells
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np


OUT = Path("results/phase_hard_normal_rejection_state/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
EPOCHS = 10


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


def load_b0_rows():
    raw, percat = [], []
    for row in read_csv(HN_OUT / "RAW_METRICS.csv"):
        if row.get("variant") == "E1_HardNormalContrast":
            raw.append(coerce_row(row, "B0_HardNormal"))
    for row in read_csv(HN_OUT / "PER_CATEGORY.csv"):
        if row.get("variant") == "E1_HardNormalContrast":
            percat.append(coerce_row(row, "B0_HardNormal"))
    return raw, percat


def balanced_batch(groups, rng, total_batch):
    names = list(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, origins, cats = [], [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            origins.extend(data["origin"][choice].tolist())
            cats.extend([cat] * len(choice))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], np.asarray(origins, dtype=object)[order], np.asarray(cats, dtype=object)[order]


def load_hardnormal_components(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    vecs = np.load(HN_OUT / "contrast_vectors.npz")
    v_hn = unit_np(np.asarray(vecs["E1"], dtype=np.float32))
    states_hn = hybrid_states(m_c, alpha * v_hn)
    align_model = make_model(states_hn, corrupt_payload, device).eval()
    return q_pc1, corrupt_payload, align_model, states_hn, m_c, alpha, v_hn


def three_logits(z, states2, reject_state):
    states = F.normalize(torch.cat([states2, reject_state.unsqueeze(0)], dim=0), dim=-1)
    return z @ states.T


def train_reject_state(variant, mode, groups, states2_np, m_c, alpha, v_hn, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}_{CACHE_VERSION}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["reject_state"].astype(np.float32), payload["train_rows"], payload["curves"], True
    print(f"[RejectState] train {variant}", flush=True)
    rng = np.random.default_rng(SEED)
    states2 = torch.tensor(norm_np(states2_np), dtype=torch.float32, device=device)
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    v = torch.tensor(v_hn, dtype=torch.float32, device=device)
    init = states2[0].detach().clone()
    reject = torch.nn.Parameter(init)
    opt = torch.optim.AdamW([reject], lr=LR, weight_decay=0.0)
    total_samples = sum(len(groups[k]["y"]) for k in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    train_rows = []
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin, _cats = balanced_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            with torch.no_grad():
                base_margin = contrast_logits(z, m, alpha, v)
                neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
                k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
                if mode == "hard":
                    selected_local = torch.topk(base_margin[neg_idx], k=k, largest=True).indices
                elif mode == "random":
                    selected_local = torch.tensor(rng.permutation(len(neg_idx))[:k], dtype=torch.long, device=device)
                else:
                    raise ValueError(mode)
                selected_idx = neg_idx[selected_local]
            logits3 = three_logits(z, states2, reject)
            targets = torch.full((selected_idx.numel(),), 2, dtype=torch.long, device=device)
            reject_loss = F.cross_entropy(logits3[selected_idx], targets)
            opt.zero_grad()
            reject_loss.backward()
            opt.step()
            sel_np = selected_idx.detach().cpu().numpy()
            sel_origin = origin[sel_np]
            with torch.no_grad():
                pa = torch.softmax(logits3[selected_idx], dim=-1)[:, 1].detach().cpu().numpy()
            train_rows.append({
                "variant": variant,
                "record_type": "train_selection",
                "epoch": epoch,
                "step": step + 1,
                "mode": mode,
                "selected_count": int(selected_idx.numel()),
                "reject_loss": float(reject_loss.detach().cpu()),
                "base_selected_margin_mean": float(base_margin[selected_idx].detach().cpu().numpy().mean()),
                "selected_P_A_mean_before_update": float(np.mean(pa)),
                "normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                "anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
            })
            vals.append(float(reject_loss.detach().cpu()))
        row = {"variant": variant, "epoch": epoch, "reject_loss": float(np.mean(vals))}
        curves.append(row)
        print(f"[RejectState] {variant} epoch {epoch}/{EPOCHS} loss={row['reject_loss']:.4f}", flush=True)
    with torch.no_grad():
        s = F.normalize(reject, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"reject_state": s, "train_rows": train_rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return s, train_rows, curves, False


@torch.no_grad()
def score_reject_batch(patch, align_model, q_pc1, states2_np, reject_np, device, smoke=False, batch_images=8):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    states2 = torch.tensor(norm_np(states2_np), dtype=torch.float32, device=device)
    reject = torch.tensor(unit_np(reject_np), dtype=torch.float32, device=device)
    align_model.eval()
    for start in range(0, n, batch_images):
        arr = remove_pc1(np.asarray(patch[start : start + batch_images], dtype=np.float32), q_pc1)
        b = arr.shape[0]
        flat = norm_np(arr.reshape(-1, arr.shape[-1]))
        x = torch.tensor(flat, dtype=torch.float32, device=device)
        z = align_model.adapt(x)
        logits3 = three_logits(z, states2, reject)
        s = torch.softmax(logits3, dim=-1)[:, 1].detach().cpu().numpy().reshape(b, SIDE, SIDE).astype(np.float32)
        maps[start : start + b] = s.astype(np.float16)
        image[start : start + b] = np.asarray([top1_image_score(m) for m in s], dtype=np.float32)
    return image, maps


def evaluate_models(models, q_pc1, device, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        if cell_id not in ("MVTec", "BTAD", "CVC-ClinicDB"):
            continue
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, cfg in models.items():
            print(f"[RejectState] eval {variant} on {cell_id}", flush=True)
            image, maps = score_reject_batch(patch, cfg["align"], q_pc1, cfg["states2"], cfg["reject"], device, smoke=smoke)
            metric, per = eval_maps(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat


def patch_mask(item):
    mask = item["img_mask"].float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    small = F.interpolate(mask, size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape(-1)
    return (small.numpy() > 0.5)


def false_positive_rows(raw_maps_by_variant, smoke):
    rows = []
    for cell_id, target, dataset, idx_or_rows, _patch, kind in target_cells(smoke):
        if cell_id not in raw_maps_by_variant:
            continue
        n = next(iter(raw_maps_by_variant[cell_id].values())).shape[0]
        for variant, maps in raw_maps_by_variant[cell_id].items():
            vals = []
            for pos in range(n):
                item = dataset[int(idx_or_rows[pos])] if kind == "source_dataset" else dataset[pos]
                normal = ~patch_mask(item)
                vals.append(maps[pos].reshape(-1)[normal])
            arr = np.concatenate(vals).astype(np.float32)
            rows.append({
                "dataset": cell_id,
                "variant": variant,
                "normal_pixel_count": int(arr.size),
                "normal_score_mean": float(arr.mean()),
                "normal_score_p95": float(np.percentile(arr, 95)),
                "normal_score_p99": float(np.percentile(arr, 99)),
            })
    return rows


def evaluate_with_fp(models, q_pc1, device, smoke):
    raw, percat, fp = [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        if cell_id not in ("MVTec", "BTAD", "CVC-ClinicDB"):
            continue
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        cell_maps = {}
        for variant, cfg in models.items():
            print(f"[RejectState] eval {variant} on {cell_id}", flush=True)
            image, maps = score_reject_batch(patch, cfg["align"], q_pc1, cfg["states2"], cfg["reject"], device, smoke=smoke)
            cell_maps[variant] = maps.astype(np.float32)
            metric, per = eval_maps(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
        for variant, maps in cell_maps.items():
            vals = []
            for pos in range(n):
                item = dataset[int(idx_use[pos])] if kind == "source_dataset" else dataset[pos]
                normal = ~patch_mask(item)
                vals.append(maps[pos].reshape(-1)[normal])
            arr = np.concatenate(vals).astype(np.float32)
            fp.append({
                "dataset": cell_id,
                "variant": variant,
                "normal_pixel_count": int(arr.size),
                "normal_score_mean": float(arr.mean()),
                "normal_score_p95": float(np.percentile(arr, 95)),
                "normal_score_p99": float(np.percentile(arr, 99)),
            })
    return raw, percat, fp


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_HardNormalRejectionState")]
        e2 = by[(ds, "E2_RandomNormalRejectionControl")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def state_geometry(states2, reject_h, reject_r):
    s_n = unit_np(states2[0])
    s_a = unit_np(states2[1])
    s_h = unit_np(reject_h)
    s_r = unit_np(reject_r)
    d = unit_np(s_a - s_n)
    return [
        {"variant": "E1_HardNormalRejectionState", "cos_reject_sN": float(s_h @ s_n), "cos_reject_sA": float(s_h @ s_a), "cos_sA_minus_sN_reject_minus_sN": float(d @ unit_np(s_h - s_n)), "reject_state_norm": float(np.linalg.norm(reject_h))},
        {"variant": "E2_RandomNormalRejectionControl", "cos_reject_sN": float(s_r @ s_n), "cos_reject_sA": float(s_r @ s_a), "cos_sA_minus_sN_reject_minus_sN": float(d @ unit_np(s_r - s_n)), "reject_state_norm": float(np.linalg.norm(reject_r))},
    ]


def decide(summary_rows, fp_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    supported = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and (e1_b0[0] >= 1.0 or e1_b0[2] >= 1.0) and e1_b0[1] >= -1.0
    if supported:
        outcome = "HARD_NORMAL_REJECTION_STATE_SUPPORTED"
    else:
        by = {(r["dataset"], r["variant"]): r for r in fp_rows}
        drops = []
        for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
            # B0 false-positive rows are not directly produced by this script; compare E1 vs E2 as a conservative calibration signal.
            if (ds, "E1_HardNormalRejectionState") in by and (ds, "E2_RandomNormalRejectionControl") in by:
                drops.append(float(by[(ds, "E1_HardNormalRejectionState")]["normal_score_p95"]) < float(by[(ds, "E2_RandomNormalRejectionControl")]["normal_score_p95"]))
        outcome = "REJECTION_STATE_CALIBRATION_NO_TRANSFER_GAIN" if any(drops) else "HARD_NORMAL_REJECTION_STATE_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_HardNormalRejectionState"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            p_auroc = float(row["P-AUROC"])
            p_ap = float(row["P-AP"])
            g = float(row["G_pixel"])
            rows.append({
                "dataset": ds,
                "current_method": "E1_HardNormalRejectionState",
                "current_P-AUROC": p_auroc,
                "current_P-AP": p_ap,
                "current_G": g,
                "reference": ref_name,
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": p_auroc - ref["P-AUROC"],
                "Delta_P_AP": p_ap - ref["P-AP"],
                "Delta_G": g - ref["G_pixel"],
                "comparison_status": "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS",
            })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(summary_rows, outcome, stats, sota_rows, runtime):
    lines = [
        "# Hard-Normal Rejection State Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. The validated Final-Layer HardNormal normal/anomaly states and rank4 alignment are frozen. E1/E2 train only one third rejection state from hard or matched random source negatives.",
        "",
        "| Dataset | B0 HardNormal G | E1 Rejection G | E2 Random Rejection G | E1-B0 | E1-E2 |",
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
        "MoECLIP is the primary direct reference requested here; VisualAD(CLIP) is retained as a secondary direct reference. Comparability is limited to VisA-source pixel P-AUROC/P-AP/G with implementation caveats.",
        "",
        "| Reference | Dataset | Current G | Reference G | Delta G | Status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for r in sota_rows:
        lines.append(f"| {r['reference']} | {r['dataset']} | {float(r['current_G']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend([
        "",
        "Stop-loss: if a third rejection state does not beat frozen HardNormal and the random rejection control, do not add more states or learned fusion.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RejectState] device={device}", flush=True)
    q_pc1, _payload, align_model, states_hn, m_c, alpha, v_hn = load_hardnormal_components(device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    s_h, rows_h, curves_h, reused_h = train_reject_state("E1_HardNormalRejectionState", "hard", groups, states_hn, m_c, alpha, v_hn, device, args.smoke)
    s_r, rows_r, curves_r, reused_r = train_reject_state("E2_RandomNormalRejectionControl", "random", groups, states_hn, m_c, alpha, v_hn, device, args.smoke)
    models = {
        "E1_HardNormalRejectionState": {"align": align_model, "states2": states_hn, "reject": s_h},
        "E2_RandomNormalRejectionControl": {"align": align_model, "states2": states_hn, "reject": s_r},
    }
    raw, percat, fp_rows = evaluate_with_fp(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    geom = state_geometry(states_hn, s_h, s_r)
    outcome, stats = decide(summary_rows, fp_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "STATE_GEOMETRY.csv", geom)
    write_csv(OUT / "REJECTION_STATE_TRAINING.csv", pool_audit + rows_h + rows_r)
    write_csv(OUT / "TARGET_FALSE_POSITIVE_STATS.csv", fp_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "hardnormal_retrained": False,
        "trainable_parameters": "single rejection state only",
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "checkpoints_reused": {"E1": bool(reused_h), "E2": bool(reused_r)},
    })
    write_report(summary_rows, outcome, stats, sota_rows, runtime)
    print(f"[RejectState] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
