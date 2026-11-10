import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_defect_query_transfer_pilot import CHECKPOINT, SEED, SIDE
from experiments.phase_empirical_defect_occupancy_prior import direction_from_weights, normal_global_spatial_mean, occupancy_map
from experiments.phase_normal_spatial_low_rank_mode import evaluate_all_multi, pc1_direction
from experiments.phase_spatial_supervision_transfer_audit import projected_source_cache, union_source_records
from experiments.phase_type_factorized_consensus_query import load_patch_proj
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_pc1_anomaly_orientation")
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


def orient(v1, d):
    d, dn = norm_vec(d)
    c = float(np.dot(v1, d))
    sign = 1.0 if c >= 0 else -1.0
    return (sign * v1).astype(np.float32), sign, c, dn


def binary_direction(z_cache, records):
    sum_a = np.zeros((768,), dtype=np.float64)
    sum_n = np.zeros((768,), dtype=np.float64)
    count_a = 0
    count_n = 0
    img_a = 0
    img_n = 0
    for i, rec in enumerate(records):
        z = np.asarray(z_cache[i], dtype=np.float32)
        if bool(rec["normal"]):
            sum_n += z.sum(axis=0, dtype=np.float64)
            count_n += int(z.shape[0])
            img_n += 1
        else:
            sum_a += z.sum(axis=0, dtype=np.float64)
            count_a += int(z.shape[0])
            img_a += 1
    d = sum_a / max(count_a, 1) - sum_n / max(count_n, 1)
    return d, {
        "anomaly_images": img_a,
        "normal_images": img_n,
        "anomaly_patch_count": count_a,
        "normal_patch_count": count_n,
    }


def center_border_direction(fmap):
    yy, xx = np.divmod(np.arange(SIDE * SIDE), SIDE)
    lo = int(np.floor(SIDE * 0.25))
    hi = int(np.ceil(SIDE * 0.75))
    center = (yy >= lo) & (yy < hi) & (xx >= lo) & (xx < hi)
    border = ~center
    d = np.asarray(fmap[center], dtype=np.float64).mean(axis=0) - np.asarray(fmap[border], dtype=np.float64).mean(axis=0)
    return d, {
        "center_lo_index": lo,
        "center_hi_exclusive": hi,
        "center_patch_count": int(center.sum()),
        "border_patch_count": int(border.sum()),
    }


def build_queries(z_cache, records):
    fmap, normal_count = normal_global_spatial_mean(z_cache, records)
    occ, mask_count = occupancy_map(records)
    q_occ, _occ_rows = direction_from_weights(fmap, occ, "B0_OccupancyDirection")
    v1, spectrum = pc1_direction(fmap, q_occ)
    # pc1_direction already flips v1 to q_occ; keep explicit mask diagnostic.
    q_mask, sign_mask, cos_mask, norm_mask = orient(v1, q_occ)
    d_binary, binary_counts = binary_direction(z_cache, records)
    q_binary, sign_binary, cos_binary, norm_binary = orient(v1, d_binary)
    d_normal, normal_counts = center_border_direction(fmap)
    q_normal, sign_normal, cos_normal, norm_normal = orient(v1, d_normal)
    queries = {
        "B0_MaskOrientedPC1": q_mask,
        "E1_BinaryOrientedPC1": q_binary,
        "E2_NormalOnlyOrientedPC1": q_normal,
    }
    diag = [
        {"variant": "B0_MaskOrientedPC1", "stat": "orientation_sign", "value": sign_mask},
        {"variant": "B0_MaskOrientedPC1", "stat": "cos_v1_d_mask_occ", "value": cos_mask},
        {"variant": "B0_MaskOrientedPC1", "stat": "d_mask_occ_norm", "value": norm_mask},
        {"variant": "E1_BinaryOrientedPC1", "stat": "orientation_sign", "value": sign_binary},
        {"variant": "E1_BinaryOrientedPC1", "stat": "cos_v1_d_binary", "value": cos_binary},
        {"variant": "E1_BinaryOrientedPC1", "stat": "d_binary_norm", "value": norm_binary},
        {"variant": "E2_NormalOnlyOrientedPC1", "stat": "orientation_sign", "value": sign_normal},
        {"variant": "E2_NormalOnlyOrientedPC1", "stat": "cos_v1_d_normal", "value": cos_normal},
        {"variant": "E2_NormalOnlyOrientedPC1", "stat": "d_normal_norm", "value": norm_normal},
        {"variant": "source", "stat": "normal_images_used_for_pc1", "value": normal_count},
        {"variant": "source", "stat": "anomaly_masks_used_for_mask_orientation", "value": mask_count},
        {"variant": "E2_NormalOnlyOrientedPC1", "stat": "center_patch_count", "value": normal_counts["center_patch_count"]},
        {"variant": "E2_NormalOnlyOrientedPC1", "stat": "border_patch_count", "value": normal_counts["border_patch_count"]},
        {"variant": "E2_NormalOnlyOrientedPC1", "stat": "center_lo_index", "value": normal_counts["center_lo_index"]},
        {"variant": "E2_NormalOnlyOrientedPC1", "stat": "center_hi_exclusive", "value": normal_counts["center_hi_exclusive"]},
        {"variant": "E1_BinaryOrientedPC1", "stat": "binary_anomaly_images", "value": binary_counts["anomaly_images"]},
        {"variant": "E1_BinaryOrientedPC1", "stat": "binary_normal_images", "value": binary_counts["normal_images"]},
        {"variant": "E2_NormalSpatialPC1", "stat": "PC1_explained_variance_ratio", "value": spectrum[0]["explained_variance_ratio"]},
    ]
    signs = {k: float(np.dot(v1, q)) for k, q in queries.items()}
    diag.extend([
        {"variant": "orientation", "stat": "B0_E1_final_sign_consistent", "value": float(signs["B0_MaskOrientedPC1"] * signs["E1_BinaryOrientedPC1"] > 0)},
        {"variant": "orientation", "stat": "B0_E2_final_sign_consistent", "value": float(signs["B0_MaskOrientedPC1"] * signs["E2_NormalOnlyOrientedPC1"] > 0)},
        {"variant": "orientation", "stat": "E1_E2_final_sign_consistent", "value": float(signs["E1_BinaryOrientedPC1"] * signs["E2_NormalOnlyOrientedPC1"] > 0)},
    ])
    return queries, diag, spectrum


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in sorted({r["dataset"] for r in raw}):
        b0 = by[(ds, "B0_MaskOrientedPC1")]
        e1 = by[(ds, "E1_BinaryOrientedPC1")]
        e2 = by[(ds, "E2_NormalOnlyOrientedPC1")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
        })
    return rows


def decide(summary_rows):
    e1_b0 = float(np.mean([r["E1_minus_B0_G"] for r in summary_rows]))
    e2_b0 = float(np.mean([r["E2_minus_B0_G"] for r in summary_rows]))
    if abs(e2_b0) <= 1.0:
        outcome = "ANOMALY_SUPERVISION_NOT_REQUIRED_FOR_ORIENTATION"
    elif abs(e1_b0) <= 1.0 and e2_b0 < -1.0:
        outcome = "BINARY_LABEL_SUFFICIENT_FOR_PC1_ORIENTATION"
    elif abs(e1_b0) > 1.0 and e2_b0 < -1.0:
        outcome = "MASK_SUPERVISION_REQUIRED_ONLY_FOR_ORIENTATION"
    else:
        outcome = "PC1_ORIENTATION_INCONCLUSIVE"
    return outcome, {"mean_E1_minus_B0_G": e1_b0, "mean_E2_minus_B0_G": e2_b0}


def write_report(out, outcome, gate, raw, summary_rows, diag):
    lines = [
        "# PC1 Anomaly Orientation Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## Protocol",
        "",
        "No training. All variants use the same normal-only spatial PC1 axis and differ only in the source-only rule used to choose its sign.",
        "",
        "## Target Results",
        "",
        "| Dataset | Variant | P-AUROC | P-AP | G |",
        "|---|---|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} |")
    lines.extend(["", "## Deltas", "", "| Dataset | E1-B0 G | E2-B0 G |", "|---|---:|---:|"])
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['E1_minus_B0_G']:+.4f} | {r['E2_minus_B0_G']:+.4f} |")
    lines.extend(["", "## Source Diagnostics", "", "| Variant | Statistic | Value |", "|---|---|---:|"])
    for r in diag:
        lines.append(f"| {r['variant']} | {r['stat']} | {float(r['value']):.6f} |")
    lines.extend([
        "",
        "## Gate",
        "",
        json.dumps(gate, indent=2, default=json_default),
        "",
        "Per-category results are saved in `PER_CATEGORY.csv`.",
        "",
        "STOP. No extra variant or Memory was started.",
    ])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def run(args):
    start = time.time()
    out = OUT / ("smoke" if args.smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[PC1Orientation] start smoke={args.smoke} device={device}", flush=True)
    patch_proj = load_patch_proj(device)
    _dataset, _indices, source_patch, records = union_source_records(args.smoke)
    cache_out = SPATIAL_OUT if not args.smoke else out
    z_cache = projected_source_cache(cache_out, records, source_patch, patch_proj, device)
    queries, diag, spectrum = build_queries(z_cache, records)
    for name, q in queries.items():
        np.save(out / f"{name}.npy", q)
    raw, percat = evaluate_all_multi(queries, patch_proj, device, args.smoke)
    summary_rows = summarize(raw)
    outcome, gate = decide(summary_rows)
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "SOURCE_DIAGNOSTICS.csv", diag + summary_rows)
    write_csv(out / "SINGULAR_SPECTRUM.csv", spectrum)
    write_json(out / "OUTCOME.json", {
        "phase": "pc1_anomaly_orientation_audit",
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
        "training_used": False,
        "target_information_used_for_training": False,
        "memory_used": False,
        "prompt_used": False,
        "adapter_used": False,
        "dino_used": False,
    })
    write_report(out, outcome, gate, raw, summary_rows, diag)
    print(f"[PC1Orientation] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
