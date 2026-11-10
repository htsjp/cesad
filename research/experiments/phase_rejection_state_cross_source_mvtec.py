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

from experiments.phase_hard_normal_cross_source_mvtec import (
    OUT as HN_OUT,
    TOP_NEG_FRAC,
    balanced_category_batch,
    collect_source_pool,
    init_midpoint_contrast,
    json_default,
    load_full_mvtec_source,
    source_slots_by_category,
    target_cells,
)
from experiments.phase_liv_memory_support_modes import eval_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np, patch_labels


OUT = Path("results/phase_rejection_state_cross_source_mvtec/formal")
EPOCHS = 10
MOECLIP_MVTEC_SOURCE_VISA = {"VisA": {"P-AUROC": 95.6, "P-AP": 26.1, "G_pixel": 60.85}}


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


def load_b0_hardnormal_rows(smoke):
    raw, percat = [], []
    for row in read_csv(HN_OUT / "RAW_METRICS.csv"):
        if row.get("variant") == "E1_HardNormal":
            raw.append(coerce_row(row, "B0_HardNormal"))
    for row in read_csv(HN_OUT / "PER_CATEGORY.csv"):
        if row.get("variant") == "E1_HardNormal":
            percat.append(coerce_row(row, "B0_HardNormal"))
    if smoke:
        raw = raw[:3]
        percat = percat[:6]
    return raw, percat


def load_hardnormal_vector_midpoint_alpha():
    path = HN_OUT / "contrast_vectors.npz"
    if not path.exists():
        raise RuntimeError(f"Missing MVTec-source HardNormal vectors: {path}")
    z = np.load(path)
    return (
        unit_np(np.asarray(z["E1"], dtype=np.float32)),
        np.asarray(z["midpoint"], dtype=np.float32),
        float(np.asarray(z["alpha"]).item()),
    )


def two_states(midpoint, alpha, v):
    s_n = midpoint - alpha * v
    s_a = midpoint + alpha * v
    return norm_np(np.stack([s_n, s_a]).astype(np.float32))


def three_logits(z, states2, reject_state):
    states = F.normalize(torch.cat([states2, reject_state.unsqueeze(0)], dim=0), dim=-1)
    return z @ states.T


def train_reject_state(variant, mode, groups, states2_np, midpoint, alpha, v_hn, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["reject_state"].astype(np.float32), payload["train_rows"], payload["curves"], True
    print(f"[RejectCrossSource] train {variant}", flush=True)
    rng = np.random.default_rng(SEED)
    states2 = torch.tensor(norm_np(states2_np), dtype=torch.float32, device=device)
    m = torch.tensor(midpoint, dtype=torch.float32, device=device)
    v = torch.tensor(unit_np(v_hn), dtype=torch.float32, device=device)
    reject = torch.nn.Parameter(states2[0].detach().clone())
    opt = torch.optim.AdamW([reject], lr=LR, weight_decay=0.0)
    total_samples = sum(len(g["y"]) for g in groups.values())
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    train_rows, curves = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin = balanced_category_batch(groups, rng, BATCH_SIZE)
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
            loss = F.cross_entropy(logits3[selected_idx], targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sel_np = selected_idx.detach().cpu().numpy()
            sel_origin = origin[sel_np]
            train_rows.append({
                "variant": variant,
                "record_type": "train_selection",
                "epoch": epoch,
                "step": step + 1,
                "mode": mode,
                "selected_count": int(selected_idx.numel()),
                "reject_loss": float(loss.detach().cpu()),
                "base_selected_margin_mean": float(base_margin[selected_idx].detach().cpu().numpy().mean()),
                "normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                "anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
            })
            vals.append(float(loss.detach().cpu()))
        row = {"variant": variant, "epoch": epoch, "reject_loss": float(np.mean(vals))}
        curves.append(row)
        print(f"[RejectCrossSource] {variant} epoch {epoch}/{EPOCHS} loss={row['reject_loss']:.4f}", flush=True)
    with torch.no_grad():
        s = F.normalize(reject, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"reject_state": s, "train_rows": train_rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return s, train_rows, curves, False


@torch.no_grad()
def score_reject_targets(states2_np, rejects, device, smoke):
    raw, percat, fp_rows = [], [], []
    states2 = torch.tensor(norm_np(states2_np), dtype=torch.float32, device=device)
    reject_t = {k: torch.tensor(unit_np(v), dtype=torch.float32, device=device) for k, v in rejects.items()}
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, rej in reject_t.items():
            print(f"[RejectCrossSource] eval {variant} on {target}", flush=True)
            maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
            image = np.zeros((n,), dtype=np.float32)
            for i in range(n):
                z = norm_np(np.asarray(patch[i], dtype=np.float32).reshape(SIDE * SIDE, 1024))
                zt = torch.tensor(z, dtype=torch.float32, device=device)
                logits = three_logits(zt, states2, rej)
                s = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
                maps[i] = s.astype(np.float16)
                image[i] = top1_image_score(s)
            metric, per = eval_maps(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": target, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": target, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
            fp_rows.extend(false_positive_stats(target, dataset, idx_use, kind, variant, maps))
    return raw, percat, fp_rows


def patch_mask(item):
    mask = item["img_mask"].float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    small = F.interpolate(mask, size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape(-1)
    return small.numpy() > 0.5


def false_positive_stats(target, dataset, idx_use, kind, variant, maps):
    vals = []
    for pos in range(maps.shape[0]):
        item = dataset[int(idx_use[pos])] if kind == "source_dataset" else dataset[pos]
        normal = ~patch_mask(item)
        vals.append(maps[pos].reshape(-1)[normal])
    arr = np.concatenate(vals).astype(np.float32)
    return [{
        "dataset": target,
        "variant": variant,
        "normal_pixel_count": int(arr.size),
        "normal_score_mean": float(arr.mean()),
        "normal_score_p95": float(np.percentile(arr, 95)),
        "normal_score_p99": float(np.percentile(arr, 99)),
    }]


def state_geometry(states2_np, s_h, s_r):
    s_n = unit_np(states2_np[0])
    s_a = unit_np(states2_np[1])
    d = unit_np(s_a - s_n)
    h = unit_np(s_h)
    r = unit_np(s_r)
    return [
        {"variant": "E1_HardNormalRejectionState", "cos_reject_sN": float(h @ s_n), "cos_reject_sA": float(h @ s_a), "cos_sA_minus_sN_reject_minus_sN": float(d @ unit_np(h - s_n)), "reject_state_norm": float(np.linalg.norm(s_h))},
        {"variant": "E2_RandomNormalRejectionState", "cos_reject_sN": float(r @ s_n), "cos_reject_sA": float(r @ s_a), "cos_sA_minus_sN_reject_minus_sN": float(d @ unit_np(r - s_n)), "reject_state_norm": float(np.linalg.norm(s_r))},
    ]


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("VisA", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_HardNormalRejectionState")]
        e2 = by[(ds, "E2_RandomNormalRejectionState")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e2_b0 = np.asarray([r["E2_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5:
        outcome = "HARD_REJECTION_CROSS_SOURCE_SUPPORTED"
    elif float(e1_b0.mean()) >= 1.0 and float(e2_b0.mean()) >= 1.0 and abs(float(e1_e2.mean())) < 0.5 and int(np.sum(e1_b0 > 0)) >= 2 and int(np.sum(e2_b0 > 0)) >= 2:
        outcome = "GENERIC_NORMAL_REJECTION_CROSS_SOURCE_SUPPORTED"
    elif int(np.sum(e1_b0 > 1.0)) == 1:
        outcome = "REJECTION_STATE_DOMAIN_DEPENDENT"
    else:
        outcome = "REJECTION_STATE_CROSS_SOURCE_NOT_REPLICATED"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E2_minus_B0_G": float(e2_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
        "E2_targets_improved_over_B0": int(np.sum(e2_b0 > 0)),
    }


def write_sota_gap(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds, ref in MOECLIP_MVTEC_SOURCE_VISA.items():
        row = by[(ds, "E1_HardNormalRejectionState")]
        rows.append({
            "dataset": ds,
            "current_method": "E1_HardNormalRejectionState_MVTecSource",
            "current_P-AUROC": float(row["P-AUROC"]),
            "current_P-AP": float(row["P-AP"]),
            "current_G": float(row["G_pixel"]),
            "reference": "MoECLIP MVTec-source -> VisA pixel",
            "reference_P-AUROC": ref["P-AUROC"],
            "reference_P-AP": ref["P-AP"],
            "reference_G": ref["G_pixel"],
            "Delta_P_AUROC": float(row["P-AUROC"]) - ref["P-AUROC"],
            "Delta_P_AP": float(row["P-AP"]) - ref["P-AP"],
            "Delta_G": float(row["G_pixel"]) - ref["G_pixel"],
            "comparison_status": "DIRECT_COMPARABLE_WITH_CAUTION_MVTEC_SOURCE_TO_VISA_PIXEL_METRICS",
        })
    for ds in ("BTAD", "CVC-ClinicDB"):
        row = by[(ds, "E1_HardNormalRejectionState")]
        rows.append({
            "dataset": ds,
            "current_method": "E1_HardNormalRejectionState_MVTecSource",
            "current_P-AUROC": float(row["P-AUROC"]),
            "current_P-AP": float(row["P-AP"]),
            "current_G": float(row["G_pixel"]),
            "reference": "No directly specified MVTec-source external reference in protocol",
            "reference_P-AUROC": "",
            "reference_P-AP": "",
            "reference_G": "",
            "Delta_P_AUROC": "",
            "Delta_P_AP": "",
            "Delta_G": "",
            "comparison_status": "REFERENCE_ONLY_NO_DIRECT_EXTERNAL_VALUE_PROVIDED",
        })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(summary_rows, outcome, stats, sota_rows, runtime):
    lines = [
        "# Rejection-State Cross-Source Replication",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full MVTec AD source-only. B0 reuses the completed MVTec-source HardNormal. E1/E2 add one third rejection state and use three-state softmax at inference; target data are evaluation-only.",
        "",
        "| Target | B0 HardNormal G | E1 Hard Rejection G | E2 Random Rejection G | E1-B0 | E2-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E2-B0 G: {stats['mean_E2_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
        "## External SOTA Position",
        "",
        "The protocol provides a directly comparable MoECLIP value for MVTec-source -> VisA. BTAD and CVC-ClinicDB are reported without a specified direct external reference here, so they are internal cross-source evidence only.",
        "",
        "| Dataset | Current G | Reference G | Delta G | Status |",
        "|---|---:|---:|---:|---|",
    ])
    for r in sota_rows:
        if r["reference_G"] == "":
            lines.append(f"| {r['dataset']} | {float(r['current_G']):.4f} |  |  | {r['comparison_status']} |")
        else:
            lines.append(f"| {r['dataset']} | {float(r['current_G']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend([
        "",
        "Stop-loss: if hard rejection does not exceed both B0 and random rejection, do not add more rejection states or learned fusion.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RejectCrossSource] device={device}", flush=True)
    _dataset_name, dataset, indices, patch = load_full_mvtec_source(args.smoke)
    groups, pool_audit = collect_source_pool(dataset, indices, patch, source_slots_by_category(dataset, indices))
    v_hn, midpoint, alpha = load_hardnormal_vector_midpoint_alpha()
    states2 = two_states(midpoint, alpha, v_hn)
    s_h, rows_h, _curves_h, reused_h = train_reject_state("E1_HardNormalRejectionState", "hard", groups, states2, midpoint, alpha, v_hn, device, args.smoke)
    s_r, rows_r, _curves_r, reused_r = train_reject_state("E2_RandomNormalRejectionState", "random", groups, states2, midpoint, alpha, v_hn, device, args.smoke)
    raw, percat, fp_rows = score_reject_targets(states2, {"E1_HardNormalRejectionState": s_h, "E2_RandomNormalRejectionState": s_r}, device, args.smoke)
    b0_raw, b0_percat = load_b0_hardnormal_rows(args.smoke)
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    geom_rows = state_geometry(states2, s_h, s_r)
    outcome, stats = decide(summary_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "STATE_GEOMETRY.csv", geom_rows)
    write_csv(OUT / "REJECTION_STATE_TRAINING.csv", pool_audit + rows_h + rows_r)
    write_csv(OUT / "TARGET_FALSE_POSITIVE_STATS.csv", fp_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "source_dataset": "mvtec",
        "targets": ["VisA", "BTAD", "CVC-ClinicDB"],
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
    print(f"[RejectCrossSource] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
