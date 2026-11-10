import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_defect_query_transfer_pilot import (
    CHECKPOINT,
    EPOCHS,
    LR,
    SEED,
    SIDE,
    dice_loss,
    make_mvec_target,
    make_source_records,
    project_patch_image,
    resize_binary_mask,
    safe_ap,
    safe_auc,
)
from experiments.phase_liv_memory_support_modes import eval_maps, external_cell
from experiments.phase_type_factorized_consensus_query import load_patch_proj
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score


OUT = Path("results/phase_spatial_supervision_transfer_audit")
VARIANTS = ["B0_ImageBinary", "E1_TrueMask", "E2_CorruptedMask"]


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


class SharedAnomalyQuery(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        torch.manual_seed(SEED)
        self.q = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self):
        return self.q / self.q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def union_source_records(smoke):
    dataset, indices, patch, typed_records, _source_names = make_source_records(smoke)
    by_pos = {}
    order = []
    for rec in typed_records:
        pos = int(rec["pos"])
        if pos not in by_pos:
            by_pos[pos] = {
                "pos": pos,
                "idx": int(rec["idx"]),
                "mask": np.zeros((SIDE * SIDE,), dtype=np.float32),
                "normal": True,
            }
            order.append(pos)
        if not bool(rec["normal"]):
            by_pos[pos]["mask"] = np.maximum(by_pos[pos]["mask"], np.asarray(rec["mask"], dtype=np.float32))
            by_pos[pos]["normal"] = False
    records = [by_pos[p] for p in order]
    return dataset, indices, patch, records


def projected_source_cache(out, records, patch, patch_proj, device):
    path = out / "source_projected_images.float16.npy"
    shape = (len(records), SIDE * SIDE, 768)
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        if arr.shape == shape:
            return arr
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    with torch.no_grad():
        for i, rec in enumerate(records):
            if i % 100 == 0:
                print(f"[SpatialAudit] project source {i}/{len(records)}", flush=True)
            z = project_patch_image(patch_proj, patch[rec["pos"]], device).detach().cpu().numpy().astype(np.float16)
            arr[i] = z
    del arr
    return np.load(path, mmap_mode="r")


def centroid(mask):
    yy, xx = np.nonzero(mask.reshape(SIDE, SIDE) > 0.5)
    if yy.size == 0:
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array([float(yy.mean()), float(xx.mean())], dtype=np.float32)


def corrupt_masks(records):
    rng = random.Random(SEED)
    masks = []
    audit = []
    for i, rec in enumerate(records):
        m = np.asarray(rec["mask"], dtype=np.float32).reshape(SIDE, SIDE)
        if bool(rec["normal"]) or float(m.sum()) <= 0:
            masks.append(np.zeros((SIDE * SIDE,), dtype=np.float32))
            continue
        best = None
        candidates = []
        for _ in range(128):
            dy = rng.randrange(SIDE)
            dx = rng.randrange(SIDE)
            if dy == 0 and dx == 0:
                continue
            c = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
            inter = float(np.logical_and(m > 0.5, c > 0.5).sum())
            union = float(np.logical_or(m > 0.5, c > 0.5).sum())
            iou = inter / max(union, 1.0)
            candidates.append((iou, dy, dx, c))
            if iou < 0.1:
                best = (iou, dy, dx, c)
                break
        if best is None:
            best = min(candidates, key=lambda x: x[0])
        iou, dy, dx, c = best
        masks.append(c.reshape(-1).astype(np.float32))
        ct, cc = centroid(m), centroid(c)
        audit.append({
            "record_id": i,
            "image_idx": int(rec["idx"]),
            "image_pos": int(rec["pos"]),
            "true_area": float(m.sum()),
            "corrupted_area": float(c.sum()),
            "area_ratio_difference": float(abs(c.sum() - m.sum()) / max(m.size, 1)),
            "iou": float(iou),
            "centroid_distance": float(np.linalg.norm(ct - cc)) if np.isfinite(ct).all() and np.isfinite(cc).all() else float("nan"),
            "shift_dy": int(dy),
            "shift_dx": int(dx),
        })
    return masks, audit


def train_query(z_cache, records, labels, variant, device):
    random.seed(SEED)
    torch.manual_seed(SEED)
    qmod = SharedAnomalyQuery().to(device)
    opt = torch.optim.AdamW(qmod.parameters(), lr=LR)
    rng = random.Random(SEED)
    curves = []
    for epoch in range(EPOCHS):
        order = list(range(len(records)))
        rng.shuffle(order)
        total = 0.0
        pos_density = 0.0
        for rid in order:
            z = torch.tensor(np.asarray(z_cache[rid], dtype=np.float32), device=device)
            logits = z @ qmod()
            y = torch.tensor(labels[rid], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            pos_density += float(y.mean().detach().cpu())
        row = {
            "variant": variant,
            "epoch": epoch + 1,
            "loss": total / max(1, len(order)),
            "mean_positive_patch_fraction": pos_density / max(1, len(order)),
        }
        curves.append(row)
        print(f"[SpatialAudit] {variant} epoch {epoch+1}/{EPOCHS} loss={row['loss']:.4f}", flush=True)
    with torch.no_grad():
        q = qmod().detach().cpu().numpy().astype(np.float32)
    return q, curves


def make_labels(records, corrupted_masks):
    labels = {}
    labels["B0_ImageBinary"] = [
        np.full((SIDE * SIDE,), 0.0 if bool(r["normal"]) else 1.0, dtype=np.float32)
        for r in records
    ]
    labels["E1_TrueMask"] = [np.asarray(r["mask"], dtype=np.float32) for r in records]
    labels["E2_CorruptedMask"] = [
        np.zeros((SIDE * SIDE,), dtype=np.float32) if bool(r["normal"]) else np.asarray(corrupted_masks[i], dtype=np.float32)
        for i, r in enumerate(records)
    ]
    return labels


def target_cells(smoke):
    target_dataset, target_indices, target_patch, _vocab = make_mvec_target(smoke)
    cells = [("MVTec", "MVTec", target_dataset, target_indices, target_patch, "source_dataset")]
    cells.append(("BTAD", *external_cell("BTAD", smoke)[1:]))
    cells.append(("CVC-ClinicDB", *external_cell("CVC-ClinicDB", smoke)[1:]))
    return cells


def score_query_on_target(q, patch, patch_proj, device, smoke):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    tq = torch.tensor(q, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(n):
            z = project_patch_image(patch_proj, patch[i], device)
            s = (z @ tq).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            maps[i] = s.astype(np.float16)
            image[i] = top1_image_score(s)
    return image, maps


def evaluate_all(queries, patch_proj, device, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, q in queries.items():
            print(f"[SpatialAudit] eval {variant} on {cell_id}", flush=True)
            image, maps = score_query_on_target(q, patch, patch_proj, device, smoke)
            if kind == "source_dataset":
                metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            else:
                metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat


def fast_patch_grid_eval(cell_id, target, dataset, idx_or_rows, kind, image_scores, maps):
    by_cat = {}
    all_iy, all_is, all_py, all_ps = [], [], [], []
    for pos, ref in enumerate(idx_or_rows):
        item = dataset[int(ref)] if kind == "source_dataset" else dataset[pos]
        if kind == "source_dataset":
            cat = str(item["cls_name"])
            label = int(item["anomaly"])
            mask_available = True
        else:
            row = dict(ref)
            cat = str(row.get("category", item.get("cls_name", target)))
            label = int(row.get("label", item.get("anomaly", 0)))
            mask_available = bool(row.get("mask_available", True))
        d = by_cat.setdefault(cat, {"iy": [], "is": [], "py": [], "ps": [], "mask_images": 0})
        score = float(image_scores[pos])
        d["iy"].append(label)
        d["is"].append(score)
        all_iy.append(label)
        all_is.append(score)
        if mask_available:
            mask = resize_binary_mask(item).astype(np.uint8)
            s = np.asarray(maps[pos], dtype=np.float32).reshape(-1)
            d["py"].append(mask)
            d["ps"].append(s)
            d["mask_images"] += 1
            all_py.append(mask)
            all_ps.append(s)
    per = {}
    for cat, vals in sorted(by_cat.items()):
        py = np.concatenate(vals["py"]) if vals["py"] else np.asarray([], dtype=np.uint8)
        ps = np.concatenate(vals["ps"]) if vals["ps"] else np.asarray([], dtype=np.float32)
        per[cat] = {
            "I-AUROC": safe_auc(vals["iy"], vals["is"]),
            "I-AP": safe_ap(vals["iy"], vals["is"]),
            "P-AUROC": safe_auc(py, ps),
            "P-AP": safe_ap(py, ps),
            "PRO": float("nan"),
            "mask_images": vals["mask_images"],
        }
    py_all = np.concatenate(all_py) if all_py else np.asarray([], dtype=np.uint8)
    ps_all = np.concatenate(all_ps) if all_ps else np.asarray([], dtype=np.float32)
    metric = {
        "cell": cell_id,
        "target": target,
        "I-AUROC": safe_auc(all_iy, all_is),
        "I-AP": safe_ap(all_iy, all_is),
        "P-AUROC": safe_auc(py_all, ps_all),
        "P-AP": safe_ap(py_all, ps_all),
        "PRO": float("nan"),
        "mask_images": len(all_py),
        "pixel_eval_resolution": f"{SIDE}x{SIDE}",
    }
    return metric, per


def query_geometry(queries):
    rows = []
    for name, q in queries.items():
        rows.append({"variant": name, "query_norm": float(np.linalg.norm(q))})
    pairs = [("B0_ImageBinary", "E1_TrueMask"), ("B0_ImageBinary", "E2_CorruptedMask"), ("E1_TrueMask", "E2_CorruptedMask")]
    for a, b in pairs:
        qa = queries[a] / max(float(np.linalg.norm(queries[a])), 1e-12)
        qb = queries[b] / max(float(np.linalg.norm(queries[b])), 1e-12)
        rows.append({
            "pair": f"{a}_vs_{b}",
            "cosine": float(np.dot(qa, qb)),
            "l2_distance": float(np.linalg.norm(qa - qb)),
        })
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in sorted({r["dataset"] for r in raw}):
        b0 = by[(ds, "B0_ImageBinary")]
        e1 = by[(ds, "E1_TrueMask")]
        e2 = by[(ds, "E2_CorruptedMask")]
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
    e1_b0 = [r["E1_minus_B0_G"] for r in summary_rows]
    e1_e2 = [r["E1_minus_E2_G"] for r in summary_rows]
    mean_b0 = float(np.mean(e1_b0))
    mean_e2 = float(np.mean(e1_e2))
    stable_b0 = sum(v > 0 for v in e1_b0) >= 2 and mean_b0 >= 0.5
    if stable_b0 and mean_e2 >= 0.5:
        outcome = "TRANSFERABLE_SPATIAL_CORRESPONDENCE_SUPPORTED"
    elif stable_b0 and mean_e2 < 0.5:
        outcome = "DENSE_SUPERVISION_GAIN_NOT_SPATIAL"
    elif mean_e2 >= 0.5 and not stable_b0:
        outcome = "SPATIAL_SIGNAL_PRESENT_BUT_CAUSALITY_INCOMPLETE"
    else:
        outcome = "PIXEL_LOCALIZATION_SUPERVISION_NO_SIGNAL"
    return outcome, mean_b0, mean_e2


def corruption_summary(audit):
    vals = np.asarray([r["iou"] for r in audit], dtype=np.float32)
    area = np.asarray([r["area_ratio_difference"] for r in audit], dtype=np.float32)
    return [{
        "num_anomaly_images": len(audit),
        "mean_iou": float(np.mean(vals)) if vals.size else float("nan"),
        "median_iou": float(np.median(vals)) if vals.size else float("nan"),
        "max_iou": float(np.max(vals)) if vals.size else float("nan"),
        "mean_area_ratio_error": float(np.mean(area)) if area.size else float("nan"),
        "fraction_iou_lt_0_1": float(np.mean(vals < 0.1)) if vals.size else float("nan"),
    }]


def write_report(out, outcome, raw, summary_rows, geom, corrupt_summary_rows, source_stats, mean_b0, mean_e2):
    lines = [
        "# Spatial Supervision Transfer Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## 1. Experiment Question",
        "",
        "This audit asks whether transferable source anomaly knowledge comes from correct pixel-level anomaly-location correspondence, rather than image-level anomaly labels, dense patch supervision, or optimization side effects.",
        "",
        "## 2. Link To Type-Factorized Results",
        "",
        "The previous type-factorized consensus experiment found no stable gain from true defect-type partitioning, and source type queries collapsed to an almost shared direction. This moves the causal question from defect identity to spatial supervision content.",
        "",
        "## 3. Zero-Shot Protocol",
        "",
        "Source is Full VisA only. Target MVTec, BTAD, and CVC-ClinicDB are used only for final evaluation. No target normal reference, target statistics, target categories for training, text, Memory, DINO, prompts, or StateAligned retraining are used.",
        "",
        "## 4. Variants",
        "",
        "- B0_ImageBinary: one shared query; normal images use all-zero patch labels, anomaly images broadcast the image anomaly label to all patches. Loss: BCEWithLogits(s_i, y_img) + Dice(s_i, y_img) over all patches.",
        "- E1_TrueMask: one shared query; anomaly images use the true VisA binary union mask; normal images are all-zero.",
        "- E2_CorruptedMask: one shared query; anomaly images use a circularly shifted version of the true mask; normal images are all-zero.",
        "",
        "## 5. Corrupted Mask Rule",
        "",
        "Each anomaly mask is circularly shifted with seed0. The first sampled shift with IoU < 0.1 is used; if none is found in 128 attempts, the lowest-IoU shift is used. Area and shape are preserved exactly by construction.",
        "",
        "## 6. Source Corruption Audit",
        "",
        f"- Source records: {source_stats['records']}; anomaly images: {source_stats['anomaly_images']}; normal images: {source_stats['normal_images']}.",
    ]
    for r in corrupt_summary_rows:
        lines.append(f"- Mean IoU={float(r['mean_iou']):.4f}, median IoU={float(r['median_iou']):.4f}, max IoU={float(r['max_iou']):.4f}, mean area error={float(r['mean_area_ratio_error']):.6f}.")
    lines.extend(["", "## 7. Overall Target Results", "", "| Dataset | Variant | P-AUROC | P-AP | G_pixel | PRO |", "|---|---|---:|---:|---:|---:|"])
    for r in raw:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} | {float(r.get('PRO', float('nan'))):.4f} |")
    lines.extend(["", "## 8. Per-Category Results", "", "Saved in `PER_CATEGORY.csv` for all target/category/variant cells.", "", "## 9. Query Geometry", ""])
    for r in geom:
        if "variant" in r:
            lines.append(f"- {r['variant']}: norm={float(r['query_norm']):.4f}.")
        else:
            lines.append(f"- {r['pair']}: cosine={float(r['cosine']):.4f}, L2={float(r['l2_distance']):.4f}.")
    lines.extend(["", "## 10. Pre-Registered Gate", "", "| Dataset | E1-B0 G | E1-E2 G |", "|---|---:|---:|"])
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"- Mean E1-B0 G: {mean_b0:+.4f} pp.",
        f"- Mean E1-E2 G: {mean_e2:+.4f} pp.",
        "",
        "## 11. Supported Conclusion",
        "",
        "The only supported conclusion is the one encoded by the pre-registered outcome above.",
        "",
        "## 12. Unsupported Claims",
        "",
        "This experiment cannot support claims about defect type semantics, open-vocabulary recognition, Memory, DINO, prompt learning, target adaptation, or target-normal reference.",
        "",
        "## 13. Continue?",
        "",
        "STOP. No additional variants or rescue experiments were started.",
    ])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    out = OUT / ("smoke" if args.smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[SpatialAudit] start smoke={args.smoke} device={device} checkpoint={CHECKPOINT}", flush=True)
    patch_proj = load_patch_proj(device)
    _dataset, _indices, source_patch, records = union_source_records(args.smoke)
    z_cache = projected_source_cache(out, records, source_patch, patch_proj, device)
    corrupted_masks, audit = corrupt_masks(records)
    labels = make_labels(records, corrupted_masks)
    source_stats = {
        "records": len(records),
        "normal_images": int(sum(1 for r in records if bool(r["normal"]))),
        "anomaly_images": int(sum(1 for r in records if not bool(r["normal"]))),
        "b0_loss": "BCEWithLogits(s_i, y_img) + Dice(s_i, y_img), y_img broadcast to every patch; no GT mask passed to B0 loss.",
    }
    queries = {}
    curves = []
    for variant in VARIANTS:
        q, cur = train_query(z_cache, records, labels[variant], variant, device)
        queries[variant] = q
        curves.extend(cur)
    raw, percat = evaluate_all(queries, patch_proj, device, args.smoke)
    geom = query_geometry(queries)
    summary_rows = summarize(raw)
    outcome, mean_b0, mean_e2 = decide(summary_rows)
    corrupt_summary_rows = corruption_summary(audit)
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "QUERY_GEOMETRY.csv", geom)
    write_csv(out / "MASK_CORRUPTION_AUDIT.csv", audit)
    write_csv(out / "TRUE_VS_CORRUPTED_MASK.csv", corrupt_summary_rows)
    write_csv(out / "BINARY_VS_PIXEL_SUPERVISION.csv", summary_rows)
    write_csv(out / "TRAIN_CURVE.csv", curves)
    write_json(out / "OUTCOME.json", {
        "phase": "spatial_supervision_transfer_audit",
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "protocol": "strict_source_only_cross_dataset_zsad",
        "variants": VARIANTS,
        "outcome": outcome,
        "mean_gain_E1_vs_B0": mean_b0,
        "mean_gain_E1_vs_E2": mean_e2,
        "notes": "B0 uses only image-level labels broadcast to patches; E2 preserves mask area/shape via circular shift.",
        "smoke": args.smoke,
        "runtime_sec": time.time() - start,
        "checkpoint": str(CHECKPOINT),
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "strict_source_only": True,
        "target_information_used_for_training": False,
        "text_used": False,
        "memory_used": False,
        "dino_used": False,
        "statealigned_retrained": False,
        "source_stats": source_stats,
    })
    write_report(out, outcome, raw, summary_rows, geom, corrupt_summary_rows, source_stats, mean_b0, mean_e2)
    print(f"[SpatialAudit] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
