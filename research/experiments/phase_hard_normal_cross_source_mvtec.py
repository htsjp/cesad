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

from experiments.phase_defect_query_transfer_pilot import light_fold
from experiments.phase_liv_memory_support_modes import ArrayConcat, external_cell, npz_array_mmap
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, LR, SIDE, norm_np, patch_labels

OUT = Path("results/phase_hard_normal_cross_source_mvtec/formal")
EPOCHS = 10
TOP_NEG_FRAC = 0.10


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


def load_full_mvtec_source(smoke):
    dataset_name, dataset, train_indices, held_indices = light_fold("D1", 0)
    cache = FEATURE_CACHE / "D1_fold0_clip.npz"
    patch = ArrayConcat([npz_array_mmap(cache, "train_patch"), npz_array_mmap(cache, "held_patch")])
    indices = list(train_indices) + list(held_indices)
    if smoke:
        by_cat = {}
        for pos, idx in enumerate(indices):
            item = dataset.data_all[int(idx)]
            cat = str(item["cls_name"])
            bucket = by_cat.setdefault(cat, {0: [], 1: []})
            lab = int(item["anomaly"])
            if len(bucket[lab]) < 4:
                bucket[lab].append((pos, idx))
        chosen = []
        for cat in sorted(by_cat)[:6]:
            chosen.extend(by_cat[cat][0])
            chosen.extend(by_cat[cat][1])
        patch = np.stack([np.asarray(patch[p], dtype=np.float32) for p, _idx in chosen], axis=0)
        indices = [idx for _p, idx in chosen]
    return dataset_name, dataset, indices, patch


def source_slots_by_category(dataset, indices):
    cats = sorted({str(dataset.data_all[int(idx)]["cls_name"]) for idx in indices})
    return {cat: [p for p, idx in enumerate(indices) if str(dataset.data_all[int(idx)]["cls_name"]) == cat] for cat in cats}


def collect_source_pool(dataset, indices, patch, slots):
    rng = np.random.default_rng(SEED)
    groups = {}
    audit = []
    for cat, positions in slots.items():
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            idx = int(indices[pos])
            item = dataset[idx]
            y = patch_labels(item).astype(np.int64)
            image_is_anomaly = int(item["anomaly"]) == 1
            z = norm_np(np.asarray(patch[pos], dtype=np.float32).reshape(SIDE * SIDE, 1024))
            pools = []
            if image_is_anomaly:
                pools.append(("anomaly_mask_out", np.flatnonzero(y == 0)))
                pools.append(("anomaly_mask", np.flatnonzero(y == 1)))
            else:
                pools.append(("normal_image", np.flatnonzero(y == 0)))
            for origin, ids in pools:
                if ids.size == 0:
                    continue
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(part.shape[0] for part in parts[origin])
                if room <= 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin].append(z[ids])
        xs, ys, origins = [], [], []
        for origin, cls in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if not parts[origin]:
                continue
            arr = np.concatenate(parts[origin], axis=0).astype(np.float32)
            xs.append(arr)
            ys.extend([cls] * arr.shape[0])
            origins.extend([origin] * arr.shape[0])
        if not xs or len(set(ys)) < 2:
            continue
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.asarray(ys, dtype=np.float32)
        origin = np.asarray(origins, dtype=object)
        groups[cat] = {"x": x, "y": y, "origin": origin}
        audit.append({
            "variant": "source_pool",
            "record_type": "pool",
            "category": cat,
            "samples": int(x.shape[0]),
            "positive_samples": int(y.sum()),
            "negative_samples": int((y == 0).sum()),
            "normal_image_negatives": int(np.sum(origin[y == 0] == "normal_image")),
            "anomaly_mask_out_negatives": int(np.sum(origin[y == 0] == "anomaly_mask_out")),
        })
    if len(groups) < 2:
        raise RuntimeError("Insufficient MVTec source groups.")
    return groups, audit


def init_midpoint_contrast(groups):
    pos, neg = [], []
    for data in groups.values():
        pos.append(data["x"][data["y"] > 0.5])
        neg.append(data["x"][data["y"] < 0.5])
    pos_mu = unit_np(np.concatenate(pos, axis=0).mean(axis=0))
    neg_mu = unit_np(np.concatenate(neg, axis=0).mean(axis=0))
    midpoint = 0.5 * (pos_mu + neg_mu)
    contrast = 0.5 * (pos_mu - neg_mu)
    alpha = float(np.linalg.norm(contrast))
    return midpoint.astype(np.float32), unit_np(contrast), alpha


def balanced_category_batch(groups, rng, total_batch):
    names = list(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, origins = [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            origins.extend(data["origin"][choice].tolist())
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], np.asarray(origins, dtype=object)[order]


def train_variant(variant, mode, groups, midpoint, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["stats"], True
    print(f"[CrossSourceHardNormal] train {variant}", flush=True)
    total_samples = sum(len(g["y"]) for g in groups.values())
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(midpoint, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    curves, stats = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            neg_scores = logits.detach()[neg_idx]
            if mode == "hard":
                selected_local = torch.topk(neg_scores, k=k, largest=True).indices
            elif mode == "random":
                selected_local = torch.tensor(rng.permutation(len(neg_idx))[:k], dtype=torch.long, device=device)
            else:
                selected_local = torch.empty((0,), dtype=torch.long, device=device)
            if mode == "none":
                extra = torch.zeros((), dtype=torch.float32, device=device)
                selected_idx = neg_idx[:0]
            else:
                selected_idx = neg_idx[selected_local]
                extra = F.softplus(logits[selected_idx]).mean()
            loss = task_loss + extra
            opt.zero_grad()
            loss.backward()
            opt.step()
            if selected_idx.numel():
                sel_np = selected_idx.detach().cpu().numpy()
                sel_score = logits.detach()[selected_idx].cpu().numpy()
                sel_origin = origin[sel_np]
                stats.append({
                    "variant": variant,
                    "record_type": "train_selection",
                    "epoch": epoch,
                    "step": step + 1,
                    "mode": mode,
                    "selected_count": int(len(sel_np)),
                    "selected_score_mean": float(sel_score.mean()),
                    "selected_score_std": float(sel_score.std()),
                    "normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                    "anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
                    "all_negative_score_mean": float(neg_scores.cpu().numpy().mean()),
                    "all_negative_score_std": float(neg_scores.cpu().numpy().std()),
                })
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(extra.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {"variant": variant, "epoch": epoch, "loss": float(arr[:, 0].mean()), "task_loss": float(arr[:, 1].mean()), "extra_normal_loss": float(arr[:, 2].mean()), "bce": float(arr[:, 3].mean()), "dice": float(arr[:, 4].mean())}
        curves.append(row)
        print(f"[CrossSourceHardNormal] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} extra={row['extra_normal_loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "stats": stats, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, stats, False


def target_cells(smoke):
    dataset_name, dataset, train_indices, held_indices = light_fold("D2", 0)
    cache = FEATURE_CACHE / "D2_fold0_clip.npz"
    patch = ArrayConcat([npz_array_mmap(cache, "train_patch"), npz_array_mmap(cache, "held_patch")])
    indices = list(train_indices) + list(held_indices)
    if smoke:
        chosen = []
        for pos, idx in enumerate(indices):
            if int(dataset.data_all[int(idx)]["anomaly"]) == 1:
                chosen.append((pos, idx))
            if len(chosen) >= 24:
                break
        patch = np.stack([np.asarray(patch[p], dtype=np.float32) for p, _idx in chosen], axis=0)
        indices = [idx for _p, idx in chosen]
    cells = [("VisA", "VisA", dataset, indices, patch, "source_dataset")]
    for name in ("BTAD", "CVC-ClinicDB"):
        cell_id, target, ds, rows, p, kind = external_cell(name, smoke)
        cells.append((target, target, ds, rows, p, kind))
    return cells


@torch.no_grad()
def score_targets(vectors, midpoint, alpha, device, smoke):
    raw, percat = [], []
    m = torch.tensor(midpoint, dtype=torch.float32, device=device)
    vec_t = {k: torch.tensor(unit_np(v), dtype=torch.float32, device=device) for k, v in vectors.items()}
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, v in vec_t.items():
            print(f"[CrossSourceHardNormal] eval {variant} on {target}", flush=True)
            maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
            image = np.zeros((n,), dtype=np.float32)
            for i in range(n):
                z = norm_np(np.asarray(patch[i], dtype=np.float32).reshape(SIDE * SIDE, 1024))
                zt = torch.tensor(z, dtype=torch.float32, device=device)
                s = contrast_logits(zt, m, alpha, v).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
                maps[i] = s.astype(np.float16)
                image[i] = top1_image_score(s)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": target, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": target, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat


def final_negative_stats(variant, groups, v, midpoint, alpha):
    rows = []
    m = torch.tensor(midpoint, dtype=torch.float32)
    vt = torch.tensor(unit_np(v), dtype=torch.float32)
    all_scores, all_origins = [], []
    for cat, data in groups.items():
        z = torch.tensor(data["x"], dtype=torch.float32)
        with torch.no_grad():
            scores = contrast_logits(z, m, alpha, vt).cpu().numpy()
        neg = data["y"] < 0.5
        ns = scores[neg]
        origins = data["origin"][neg]
        all_scores.append(ns)
        all_origins.extend(origins.tolist())
        rows.append({"variant": variant, "record_type": "final_negative_distribution", "category": cat, "negative_count": int(ns.size), "score_mean": float(ns.mean()), "score_std": float(ns.std()), "score_p90": float(np.percentile(ns, 90)), "score_p95": float(np.percentile(ns, 95)), "normal_image_ratio": float(np.mean(origins == "normal_image")), "anomaly_mask_out_ratio": float(np.mean(origins == "anomaly_mask_out"))})
    s = np.concatenate(all_scores, axis=0)
    origins = np.asarray(all_origins, dtype=object)
    rows.append({"variant": variant, "record_type": "final_negative_distribution_summary", "category": "__all__", "negative_count": int(s.size), "score_mean": float(s.mean()), "score_std": float(s.std()), "score_p90": float(np.percentile(s, 90)), "score_p95": float(np.percentile(s, 95)), "normal_image_ratio": float(np.mean(origins == "normal_image")), "anomaly_mask_out_ratio": float(np.mean(origins == "anomaly_mask_out"))})
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("VisA", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_GlobalContrast")]
        e1 = by[(ds, "E1_HardNormal")]
        e2 = by[(ds, "E2_RandomNormal")]
        rows.append({"dataset": ds, "B0_G": float(b0["G_pixel"]), "E1_G": float(e1["G_pixel"]), "E2_G": float(e2["G_pixel"]), "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]), "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]), "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]), "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]), "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]), "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"])})
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5:
        out = "HARD_NORMAL_CROSS_SOURCE_REPLICATED"
    elif int(np.sum(e1_b0 > 1.0)) == 1:
        out = "HARD_NORMAL_SOURCE_OR_DOMAIN_DEPENDENT"
    else:
        out = "HARD_NORMAL_CROSS_SOURCE_NOT_REPLICATED"
    return out, {"mean_E1_minus_B0_G": float(e1_b0.mean()), "mean_E1_minus_E2_G": float(e1_e2.mean()), "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0))}


def write_sota_gap(raw):
    refs = {
        "MVTec": {"P-AUROC": 90.8, "P-AP": 41.2, "G_pixel": 66.0},
        "BTAD": {"P-AUROC": 91.1, "P-AP": 43.1, "G_pixel": 67.1},
        "CVC-ClinicDB": {"P-AUROC": 85.2, "P-AP": 37.8, "G_pixel": 61.5},
    }
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds, ref in refs.items():
        row = by.get((ds, "E1_HardNormal"))
        if row is None:
            rows.append({"dataset": ds, "current_method": "E1_HardNormal_MVTecSource", "current_P-AUROC": "", "current_P-AP": "", "current_G": "", "reference": "CVPR 2026 VisualAD(CLIP) pixel, VisA-source cross-dataset", "reference_P-AUROC": ref["P-AUROC"], "reference_P-AP": ref["P-AP"], "reference_G": ref["G_pixel"], "Delta_P_AUROC": "", "Delta_P_AP": "", "Delta_G": "", "comparison_status": "reference_only_target_not_evaluated_or_source_mismatch"})
            continue
        rows.append({"dataset": ds, "current_method": "E1_HardNormal_MVTecSource", "current_P-AUROC": float(row["P-AUROC"]), "current_P-AP": float(row["P-AP"]), "current_G": float(row["G_pixel"]), "reference": "CVPR 2026 VisualAD(CLIP) pixel, VisA-source cross-dataset", "reference_P-AUROC": ref["P-AUROC"], "reference_P-AP": ref["P-AP"], "reference_G": ref["G_pixel"], "Delta_P_AUROC": float(row["P-AUROC"]) - ref["P-AUROC"], "Delta_P_AP": float(row["P-AP"]) - ref["P-AP"], "Delta_G": float(row["G_pixel"]) - ref["G_pixel"], "comparison_status": "reference_only_source_mismatch_MVTec_source_not_directly_comparable"})
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(summary_rows, outcome, stats, geom, sota_rows, runtime):
    lines = [
        "# Hard-Normal Cross-Source Replication",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Source is Full MVTec AD. Targets are VisA, BTAD, and CVC-ClinicDB. No target training, target statistics, target memory, prompt, router, or adaptation are used.",
        "",
        "| Target | B0 Global G | E1 HardNormal G | E2 RandomNormal G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend(["", f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.", f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.", f"Contrast cosines: B0/E1={geom[0]['cos_v_B0_v_E1']:.6f}, B0/E2={geom[0]['cos_v_B0_v_E2']:.6f}, E1/E2={geom[0]['cos_v_E1_v_E2']:.6f}.", "", "## External SOTA Position", "", "VisualAD(CLIP) values provided for the mandatory gate are VisA-source cross-dataset results. This experiment uses MVTec as source, so BTAD/CVC deltas are reference-only and must not be claimed as direct leaderboard comparisons; MVTec target is not evaluated in this cross-source design.", "", "| Dataset | Current G | VisualAD G | Delta G | Status |", "|---|---:|---:|---:|---|"])
    for r in sota_rows:
        cg = r["current_G"]
        dg = r["Delta_G"]
        lines.append(f"| {r['dataset']} | {float(cg):.4f} | {float(r['reference_G']):.4f} | {float(dg):+.4f} | {r['comparison_status']} |" if cg != "" else f"| {r['dataset']} |  | {float(r['reference_G']):.4f} |  | {r['comparison_status']} |")
    lines.extend(["", "Stop-loss: evaluated as replication, not a new complexity increase. If this fails to replicate across sources, future HardNormal variants should be treated as source/domain dependent until another source confirms it.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CrossSourceHardNormal] device={device}", flush=True)
    dataset_name, dataset, indices, patch = load_full_mvtec_source(args.smoke)
    slots = source_slots_by_category(dataset, indices)
    groups, pool_audit = collect_source_pool(dataset, indices, patch, slots)
    midpoint, init_v, alpha = init_midpoint_contrast(groups)
    v_b0, stats_b0, reused_b0 = train_variant("B0_GlobalContrast", "none", groups, midpoint, alpha, init_v, device, args.smoke)
    v_e1, stats_e1, reused_e1 = train_variant("E1_HardNormal", "hard", groups, midpoint, alpha, init_v, device, args.smoke)
    v_e2, stats_e2, reused_e2 = train_variant("E2_RandomNormal", "random", groups, midpoint, alpha, init_v, device, args.smoke)
    raw, percat = score_targets({"B0_GlobalContrast": v_b0, "E1_HardNormal": v_e1, "E2_RandomNormal": v_e2}, midpoint, alpha, device, args.smoke)
    summary_rows = summarize(raw)
    hard_stats = pool_audit + stats_b0 + stats_e1 + stats_e2
    for variant, v in (("B0_GlobalContrast", v_b0), ("E1_HardNormal", v_e1), ("E2_RandomNormal", v_e2)):
        hard_stats.extend(final_negative_stats(variant, groups, v, midpoint, alpha))
    geom = contrast_geometry(v_b0, v_e1, v_e2)
    outcome, stats = decide(summary_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "HARD_NEGATIVE_STATS.csv", hard_stats)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    np.savez_compressed(OUT / "contrast_vectors.npz", B0=v_b0.astype(np.float32), E1=v_e1.astype(np.float32), E2=v_e2.astype(np.float32), midpoint=midpoint.astype(np.float32), alpha=np.asarray(alpha, dtype=np.float32), init=init_v.astype(np.float32))
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": stats, "runtime_sec": runtime, "device": device, "source_dataset": dataset_name, "targets": ["VisA", "BTAD", "CVC-ClinicDB"], "strict_source_only": True, "target_training": False, "target_statistics": False, "target_memory": False, "epochs": EPOCHS, "lr": LR, "seed": SEED, "top_negative_fraction": TOP_NEG_FRAC, "alpha": alpha, "checkpoints_reused": {"B0": bool(reused_b0), "E1": bool(reused_e1), "E2": bool(reused_e2)}})
    write_report(summary_rows, outcome, stats, geom, sota_rows, runtime)
    print(f"[CrossSourceHardNormal] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
