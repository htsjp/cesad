import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_defect_query_transfer_pilot import CHECKPOINT, SEED, SIDE
from experiments.phase_spatial_supervision_transfer_audit import evaluate_all, projected_source_cache, union_source_records
from experiments.phase_type_factorized_consensus_query import load_patch_proj
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_source_mask_geometry_confound")
SPATIAL_OUT = Path("results/phase_spatial_supervision_transfer_audit/formal")


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


def norm_vec(x):
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    return (x / max(n, 1e-12)).astype(np.float32), n


def cls_name(dataset, rec):
    return str(dataset.data_all[int(rec["idx"])]["cls_name"])


def normal_spatial_maps(z_cache, records, dataset):
    sums = defaultdict(lambda: np.zeros((SIDE * SIDE, 768), dtype=np.float64))
    counts = defaultdict(int)
    for i, rec in enumerate(records):
        if not bool(rec["normal"]):
            continue
        c = cls_name(dataset, rec)
        sums[c] += np.asarray(z_cache[i], dtype=np.float32)
        counts[c] += 1
    return {c: sums[c] / max(counts[c], 1) for c in sums}, counts


def anomaly_mask_records(records, dataset):
    out = []
    for i, rec in enumerate(records):
        if bool(rec["normal"]):
            continue
        m = np.asarray(rec["mask"], dtype=np.float32).reshape(SIDE, SIDE)
        if m.sum() <= 0 or m.sum() >= m.size:
            continue
        out.append({
            "record_id": i,
            "image_idx": int(rec["idx"]),
            "cls_name": cls_name(dataset, rec),
            "mask": (m > 0.5),
            "area": int((m > 0.5).sum()),
        })
    return out


def mask_iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def relocate_masks(mask_rows):
    rng = random.Random(SEED)
    relocated = []
    audits = []
    for r in mask_rows:
        m = r["mask"]
        best = None
        candidates = []
        for _ in range(128):
            dy = rng.randrange(SIDE)
            dx = rng.randrange(SIDE)
            if dx == 0 and dy == 0:
                continue
            c = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
            iou = mask_iou(m, c)
            candidates.append((iou, dy, dx, c))
            if iou < 0.1:
                best = (iou, dy, dx, c)
                break
        if best is None:
            best = min(candidates, key=lambda x: x[0])
        iou, dy, dx, c = best
        rr = dict(r)
        rr["mask"] = c
        relocated.append(rr)
        audits.append({
            "variant": "E1_RelocatedMaskSpatial",
            "record_id": r["record_id"],
            "image_idx": r["image_idx"],
            "cls_name": r["cls_name"],
            "true_area": r["area"],
            "control_area": int(c.sum()),
            "area_error": int(c.sum()) - int(r["area"]),
            "iou": float(iou),
            "shift_dy": int(dy),
            "shift_dx": int(dx),
        })
    return relocated, audits


def category_derangement(categories):
    rng = random.Random(SEED)
    cats = sorted(categories)
    shuffled = cats[:]
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and any(a == b for a, b in zip(cats, shuffled)):
        shuffled = shuffled[1:] + shuffled[:1]
    return dict(zip(cats, shuffled))


def cross_category_masks(mask_rows):
    rng = random.Random(SEED)
    by_cls = defaultdict(list)
    for r in mask_rows:
        by_cls[r["cls_name"]].append(r)
    cmap = category_derangement(by_cls.keys())
    out = []
    audits = []
    for c in sorted(by_cls):
        donors = by_cls[cmap[c]][:]
        rng.shuffle(donors)
        targets = by_cls[c]
        for k, r in enumerate(targets):
            donor = donors[k % len(donors)]
            rr = dict(r)
            rr["mask"] = donor["mask"]
            rr["donor_cls_name"] = donor["cls_name"]
            rr["donor_record_id"] = donor["record_id"]
            out.append(rr)
            audits.append({
                "variant": "E2_CrossCategoryMaskSpatial",
                "record_id": r["record_id"],
                "image_idx": r["image_idx"],
                "cls_name": r["cls_name"],
                "donor_cls_name": donor["cls_name"],
                "donor_record_id": donor["record_id"],
                "true_area": r["area"],
                "control_area": donor["area"],
                "area_error": int(donor["area"]) - int(r["area"]),
                "iou": mask_iou(r["mask"], donor["mask"]),
                "category_derangement": f"{c}->{cmap[c]}",
            })
    return out, audits, cmap


def direction_from_masks(normal_maps, mask_rows, variant):
    sum_in = np.zeros((768,), dtype=np.float64)
    sum_out = np.zeros((768,), dtype=np.float64)
    count_in = 0
    count_out = 0
    used = 0
    for r in mask_rows:
        fmap = normal_maps.get(r["cls_name"])
        if fmap is None:
            continue
        m = np.asarray(r["mask"], dtype=bool).reshape(-1)
        if not m.any() or not (~m).any():
            continue
        sum_in += fmap[m].sum(axis=0, dtype=np.float64)
        sum_out += fmap[~m].sum(axis=0, dtype=np.float64)
        count_in += int(m.sum())
        count_out += int((~m).sum())
        used += 1
    mu_in = sum_in / max(count_in, 1)
    mu_out = sum_out / max(count_out, 1)
    q, dn = norm_vec(mu_in - mu_out)
    rows = [
        {"variant": variant, "stat": "mask_records_used", "value": used},
        {"variant": variant, "stat": "mask_in_patch_count", "value": count_in},
        {"variant": variant, "stat": "mask_out_patch_count", "value": count_out},
        {"variant": variant, "stat": "raw_difference_norm", "value": dn},
        {"variant": variant, "stat": "mu_in_norm", "value": float(np.linalg.norm(mu_in))},
        {"variant": variant, "stat": "mu_out_norm", "value": float(np.linalg.norm(mu_out))},
    ]
    return q, rows


def geometry(queries):
    rows = []
    for name, q in queries.items():
        rows.append({"variant": name, "query_norm": float(np.linalg.norm(q))})
    pairs = [
        ("B0_TrueMaskSpatial", "E1_RelocatedMaskSpatial"),
        ("B0_TrueMaskSpatial", "E2_CrossCategoryMaskSpatial"),
        ("E1_RelocatedMaskSpatial", "E2_CrossCategoryMaskSpatial"),
    ]
    for a, b in pairs:
        qa, _ = norm_vec(queries[a])
        qb, _ = norm_vec(queries[b])
        rows.append({"pair": f"{a}_vs_{b}", "cosine": float(np.dot(qa, qb)), "l2_distance": float(np.linalg.norm(qa - qb))})
    return rows


def audit_summary(audits):
    rows = list(audits)
    for variant in sorted({r["variant"] for r in audits}):
        vals = [r for r in audits if r["variant"] == variant]
        iou = np.asarray([float(r["iou"]) for r in vals], dtype=np.float64)
        err = np.asarray([float(r["area_error"]) for r in vals], dtype=np.float64)
        rows.append({
            "variant": variant,
            "stat": "summary",
            "count": len(vals),
            "mean_iou": float(np.mean(iou)) if iou.size else float("nan"),
            "median_iou": float(np.median(iou)) if iou.size else float("nan"),
            "max_iou": float(np.max(iou)) if iou.size else float("nan"),
            "mean_abs_area_error": float(np.mean(np.abs(err))) if err.size else float("nan"),
            "median_abs_area_error": float(np.median(np.abs(err))) if err.size else float("nan"),
        })
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in sorted({r["dataset"] for r in raw}):
        b0 = by[(ds, "B0_TrueMaskSpatial")]
        e1 = by[(ds, "E1_RelocatedMaskSpatial")]
        e2 = by[(ds, "E2_CrossCategoryMaskSpatial")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "B0_minus_E1_G": float(b0["G_pixel"]) - float(e1["G_pixel"]),
            "B0_minus_E2_G": float(b0["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
        })
    return rows


def decide(summary_rows, geom_rows):
    b0_e1 = float(np.mean([r["B0_minus_E1_G"] for r in summary_rows]))
    b0_e2 = float(np.mean([r["B0_minus_E2_G"] for r in summary_rows]))
    e1_e2 = float(np.mean([r["E1_minus_E2_G"] for r in summary_rows]))
    cos01 = next(float(r["cosine"]) for r in geom_rows if r.get("pair") == "B0_TrueMaskSpatial_vs_E1_RelocatedMaskSpatial")
    cos02 = next(float(r["cosine"]) for r in geom_rows if r.get("pair") == "B0_TrueMaskSpatial_vs_E2_CrossCategoryMaskSpatial")
    all_close = max(abs(b0_e1), abs(b0_e2), abs(e1_e2)) < 1.0
    high_sim = min(cos01, cos02) >= 0.90
    if b0_e1 >= 1.0 and b0_e2 >= 1.0:
        outcome = "SOURCE_DEFECT_MASK_GEOMETRY_SIGNAL"
    elif all_close and high_sim:
        outcome = "GENERIC_FOREGROUND_DIRECTION_SUFFICIENT"
    elif abs(b0_e1) < 1.0 and b0_e2 >= 1.0:
        outcome = "CATEGORY_FOREGROUND_STRUCTURE_DOMINANT"
    else:
        outcome = "SPATIAL_GEOMETRY_CONFOUND_INCONCLUSIVE"
    return outcome, {
        "mean_B0_minus_E1_G": b0_e1,
        "mean_B0_minus_E2_G": b0_e2,
        "mean_E1_minus_E2_G": e1_e2,
        "cos_B0_E1": cos01,
        "cos_B0_E2": cos02,
    }


def write_report(out, outcome, gate, raw, summary_rows, geom_rows, source_rows, audit_rows):
    lines = [
        "# Source Mask Geometry Confound Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## Protocol",
        "",
        "No training. All directions are constructed only from Full VisA normal image feature maps plus source mask coordinates. No anomaly visual feature, target statistics, Memory, prompt, adapter, or DINO is used.",
        "",
        "## Target Results",
        "",
        "| Dataset | Variant | P-AUROC | P-AP | G |",
        "|---|---|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} |")
    lines.extend(["", "## Deltas", "", "| Dataset | B0-E1 G | B0-E2 G | E1-E2 G |", "|---|---:|---:|---:|"])
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_minus_E1_G']:+.4f} | {r['B0_minus_E2_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend(["", "## Geometry", ""])
    for r in geom_rows:
        if "variant" in r:
            lines.append(f"- {r['variant']}: norm={float(r['query_norm']):.4f}.")
        else:
            lines.append(f"- {r['pair']}: cosine={float(r['cosine']):.4f}, L2={float(r['l2_distance']):.4f}.")
    summary_audits = [r for r in audit_rows if r.get("stat") == "summary"]
    lines.extend(["", "## Mask Geometry Audit", ""])
    for r in summary_audits:
        lines.append(f"- {r['variant']}: count={int(r['count'])}, mean IoU={float(r['mean_iou']):.4f}, median IoU={float(r['median_iou']):.4f}, mean abs area error={float(r['mean_abs_area_error']):.4f}.")
    lines.extend(["", "## Source Statistics", "", "| Variant | Statistic | Value |", "|---|---|---:|"])
    for r in source_rows:
        lines.append(f"| {r['variant']} | {r['stat']} | {float(r['value']):.6f} |")
    lines.extend([
        "",
        "## Gate",
        "",
        json.dumps(gate, indent=2, default=json_default),
        "",
        "Per-category target results are saved in `PER_CATEGORY.csv`; per-mask relocation/cross-category audit is saved in `MASK_GEOMETRY_AUDIT.csv`.",
        "",
        "STOP. No center prior, PCA, Memory, learned query, or new variant was started.",
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
    print(f"[MaskGeometry] start smoke={args.smoke} device={device}", flush=True)
    patch_proj = load_patch_proj(device)
    dataset, _indices, source_patch, records = union_source_records(args.smoke)
    cache_out = SPATIAL_OUT if not args.smoke else out
    z_cache = projected_source_cache(cache_out, records, source_patch, patch_proj, device)
    normal_maps, normal_counts = normal_spatial_maps(z_cache, records, dataset)
    masks = anomaly_mask_records(records, dataset)
    relocated, rel_audit = relocate_masks(masks)
    crossed, cross_audit, category_map = cross_category_masks(masks)
    q0, s0 = direction_from_masks(normal_maps, masks, "B0_TrueMaskSpatial")
    q1, s1 = direction_from_masks(normal_maps, relocated, "E1_RelocatedMaskSpatial")
    q2, s2 = direction_from_masks(normal_maps, crossed, "E2_CrossCategoryMaskSpatial")
    queries = {"B0_TrueMaskSpatial": q0, "E1_RelocatedMaskSpatial": q1, "E2_CrossCategoryMaskSpatial": q2}
    for name, q in queries.items():
        np.save(out / f"{name}.npy", q)
    raw, percat = evaluate_all(queries, patch_proj, device, args.smoke)
    geom_rows = geometry(queries)
    summary_rows = summarize(raw)
    outcome, gate = decide(summary_rows, geom_rows)
    source_rows = s0 + s1 + s2
    audit_rows = audit_summary(rel_audit + cross_audit)
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "DIRECTION_GEOMETRY.csv", geom_rows)
    write_csv(out / "MASK_GEOMETRY_AUDIT.csv", audit_rows)
    write_json(out / "OUTCOME.json", {
        "phase": "source_mask_geometry_confound_audit",
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "protocol": "strict_source_only_cross_dataset_zsad",
        "variants": list(queries.keys()),
        "outcome": outcome,
        "gate": gate,
        "smoke": args.smoke,
        "runtime_sec": time.time() - start,
        "checkpoint": str(CHECKPOINT),
        "seed": SEED,
        "normal_categories": len(normal_counts),
        "anomaly_masks": len(masks),
        "cross_category_derangement": category_map,
        "strict_source_only": True,
        "training_used": False,
        "anomaly_visual_feature_used_for_direction": False,
        "target_information_used_for_training": False,
        "memory_used": False,
        "prompt_used": False,
        "adapter_used": False,
        "dino_used": False,
    })
    write_report(out, outcome, gate, raw, summary_rows, geom_rows, source_rows, audit_rows)
    print(f"[MaskGeometry] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
