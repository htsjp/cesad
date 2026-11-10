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


OUT = Path("results/phase_anomaly_content_vs_spatial_bias")
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


def class_name(dataset, rec):
    return str(dataset.data_all[int(rec["idx"])]["cls_name"])


def normal_spatial_means(z_cache, records, dataset):
    sums = defaultdict(lambda: np.zeros((SIDE * SIDE, 768), dtype=np.float64))
    counts = defaultdict(int)
    global_sum = np.zeros((SIDE * SIDE, 768), dtype=np.float64)
    global_count = 0
    for i, rec in enumerate(records):
        if not bool(rec["normal"]):
            continue
        z = np.asarray(z_cache[i], dtype=np.float32)
        cls = class_name(dataset, rec)
        sums[cls] += z
        counts[cls] += 1
        global_sum += z
        global_count += 1
    means = {cls: sums[cls] / max(counts[cls], 1) for cls in sums}
    global_mean = global_sum / max(global_count, 1)
    return means, counts, global_mean, global_count


def build_directions(z_cache, records, dataset):
    normal_maps, normal_counts, global_normal_map, global_normal_count = normal_spatial_means(z_cache, records, dataset)
    sum_in = np.zeros((768,), dtype=np.float64)
    sum_out = np.zeros((768,), dtype=np.float64)
    sum_sp_in = np.zeros((768,), dtype=np.float64)
    sum_sp_out = np.zeros((768,), dtype=np.float64)
    sum_content_in = np.zeros((768,), dtype=np.float64)
    sum_content_normal = np.zeros((768,), dtype=np.float64)
    count_in = count_out = 0
    count_sp_in = count_sp_out = 0
    anomaly_images = 0
    fallback_categories = set()
    category_rows = defaultdict(lambda: {"anomaly_images": 0, "mask_area": 0, "context_area": 0})
    for i, rec in enumerate(records):
        if bool(rec["normal"]):
            continue
        z = np.asarray(z_cache[i], dtype=np.float32)
        m = np.asarray(rec["mask"], dtype=bool)
        if not m.any() or not (~m).any():
            continue
        cls = class_name(dataset, rec)
        fmap = normal_maps.get(cls, global_normal_map)
        if cls not in normal_maps:
            fallback_categories.add(cls)
        anomaly_images += 1
        category_rows[cls]["anomaly_images"] += 1
        category_rows[cls]["mask_area"] += int(m.sum())
        category_rows[cls]["context_area"] += int((~m).sum())

        sum_in += z[m].sum(axis=0, dtype=np.float64)
        sum_out += z[~m].sum(axis=0, dtype=np.float64)
        count_in += int(m.sum())
        count_out += int((~m).sum())

        sum_sp_in += fmap[m].sum(axis=0, dtype=np.float64)
        sum_sp_out += fmap[~m].sum(axis=0, dtype=np.float64)
        count_sp_in += int(m.sum())
        count_sp_out += int((~m).sum())

        sum_content_in += z[m].sum(axis=0, dtype=np.float64)
        sum_content_normal += fmap[m].sum(axis=0, dtype=np.float64)

    mu_in = sum_in / max(count_in, 1)
    mu_out = sum_out / max(count_out, 1)
    mu_pos_in = sum_sp_in / max(count_sp_in, 1)
    mu_pos_out = sum_sp_out / max(count_sp_out, 1)
    mu_content_in = sum_content_in / max(count_in, 1)
    mu_normal_matched = sum_content_normal / max(count_in, 1)

    q_local, n_local = norm_vec(mu_in - mu_out)
    q_spatial, n_spatial = norm_vec(mu_pos_in - mu_pos_out)
    q_content, n_content = norm_vec(mu_content_in - mu_normal_matched)
    queries = {
        "B0_LocalContrast": q_local,
        "E1_SpatialForegroundControl": q_spatial,
        "E2_ContentResidual": q_content,
    }
    source_rows = [
        {"stat": "anomaly_images_used", "value": anomaly_images},
        {"stat": "normal_images_used_for_spatial_means", "value": global_normal_count},
        {"stat": "normal_categories_with_spatial_mean", "value": len(normal_maps)},
        {"stat": "fallback_category_count", "value": len(fallback_categories)},
        {"stat": "mask_in_patch_count", "value": count_in},
        {"stat": "mask_out_patch_count", "value": count_out},
        {"stat": "norm_local_mu_in_minus_mu_out", "value": n_local},
        {"stat": "norm_spatial_mu_pos_in_minus_pos_out", "value": n_spatial},
        {"stat": "norm_content_mu_in_minus_normal_matched", "value": n_content},
        {"stat": "mu_in_norm", "value": float(np.linalg.norm(mu_in))},
        {"stat": "mu_out_norm", "value": float(np.linalg.norm(mu_out))},
        {"stat": "mu_pos_in_norm", "value": float(np.linalg.norm(mu_pos_in))},
        {"stat": "mu_pos_out_norm", "value": float(np.linalg.norm(mu_pos_out))},
        {"stat": "mu_normal_matched_norm", "value": float(np.linalg.norm(mu_normal_matched))},
    ]
    for cls, count in sorted(normal_counts.items()):
        row = category_rows[cls]
        source_rows.append({
            "stat": "category_summary",
            "category": cls,
            "normal_images": count,
            "anomaly_images": row["anomaly_images"],
            "mask_area": row["mask_area"],
            "context_area": row["context_area"],
        })
    return queries, source_rows


def direction_geometry(queries):
    rows = []
    for name, q in queries.items():
        rows.append({"variant": name, "query_norm": float(np.linalg.norm(q))})
    pairs = [
        ("B0_LocalContrast", "E1_SpatialForegroundControl"),
        ("B0_LocalContrast", "E2_ContentResidual"),
        ("E1_SpatialForegroundControl", "E2_ContentResidual"),
    ]
    for a, b in pairs:
        qa, _ = norm_vec(queries[a])
        qb, _ = norm_vec(queries[b])
        rows.append({"pair": f"{a}_vs_{b}", "cosine": float(np.dot(qa, qb)), "l2_distance": float(np.linalg.norm(qa - qb))})
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in sorted({r["dataset"] for r in raw}):
        b0 = by[(ds, "B0_LocalContrast")]
        e1 = by[(ds, "E1_SpatialForegroundControl")]
        e2 = by[(ds, "E2_ContentResidual")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_E1_G": float(e2["G_pixel"]) - float(e1["G_pixel"]),
        })
    return rows


def decide(summary_rows, geom):
    e1_gap = float(np.mean([r["E1_minus_B0_G"] for r in summary_rows]))
    e2_gap = float(np.mean([r["E2_minus_B0_G"] for r in summary_rows]))
    b0_e1_cos = next(float(r["cosine"]) for r in geom if r.get("pair") == "B0_LocalContrast_vs_E1_SpatialForegroundControl")
    e2_close = abs(e2_gap) <= 1.0
    e1_close = abs(e1_gap) <= 1.0
    e1_low = e1_gap <= -3.0
    spatial_aligned = b0_e1_cos >= 0.90
    if e2_close and e1_low:
        outcome = "ANOMALY_CONTENT_DOMINANT"
    elif e1_close or spatial_aligned:
        outcome = "SPATIAL_FOREGROUND_CONFOUND"
    elif e1_gap > -5.0 and e2_gap > -5.0:
        outcome = "MIXED_CONTENT_STRUCTURE_SIGNAL"
    else:
        outcome = "CONTENT_DECOMPOSITION_INCONCLUSIVE"
    return outcome, {
        "mean_E1_minus_B0_G": e1_gap,
        "mean_E2_minus_B0_G": e2_gap,
        "cos_q_local_q_spatial": b0_e1_cos,
        "content_close_threshold_pp": 1.0,
        "spatial_low_threshold_pp": -3.0,
        "spatial_alignment_threshold": 0.90,
    }


def write_report(out, outcome, gate, raw, geom, source_rows, summary_rows):
    lines = [
        "# Anomaly Content vs Spatial Bias Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## Question",
        "",
        "Does the strong local contrast direction come from true anomaly visual content, or from anomaly-mask spatial/foreground bias?",
        "",
        "## Protocol",
        "",
        "Full VisA source only; MVTec, BTAD, and CVC-ClinicDB are final evaluation only. No training, Memory, prompt, adapter, DINO, target statistics, or target normal reference.",
        "",
        "## Variants",
        "",
        "- B0_LocalContrast: normalize(mu_in_anomaly - mu_out_anomaly).",
        "- E1_SpatialForegroundControl: normal-image category spatial mean maps sampled at anomaly-mask coordinates.",
        "- E2_ContentResidual: anomaly mask-in feature minus category normal spatial mean at the same mask coordinates.",
        "",
        "## Target Results",
        "",
        "| Dataset | Variant | P-AUROC | P-AP | G |",
        "|---|---|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} |")
    lines.extend(["", "## Deltas", "", "| Dataset | E1-B0 G | E2-B0 G | E2-E1 G |", "|---|---:|---:|---:|"])
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['E1_minus_B0_G']:+.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E2_minus_E1_G']:+.4f} |")
    lines.extend(["", "## Geometry", ""])
    for r in geom:
        if "variant" in r:
            lines.append(f"- {r['variant']}: norm={float(r['query_norm']):.4f}.")
        else:
            lines.append(f"- {r['pair']}: cosine={float(r['cosine']):.4f}, L2={float(r['l2_distance']):.4f}.")
    lines.extend(["", "## Source Statistics", "", "| Statistic | Value |", "|---|---:|"])
    for r in source_rows:
        if r.get("stat") != "category_summary":
            lines.append(f"| {r['stat']} | {float(r['value']):.6f} |")
    lines.extend([
        "",
        "## Gate",
        "",
        f"- mean E1-B0 G = {gate['mean_E1_minus_B0_G']:+.4f} pp.",
        f"- mean E2-B0 G = {gate['mean_E2_minus_B0_G']:+.4f} pp.",
        f"- cos(q_local, q_spatial) = {gate['cos_q_local_q_spatial']:.4f}.",
        "",
        "Per-category target results are saved in `PER_CATEGORY.csv`; per-source-category normal/anomaly counts are saved in `SOURCE_STATISTICS.csv`.",
        "",
        "STOP. No Memory, covariance, PCA, or new method was started.",
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
    print(f"[ContentSpatial] start smoke={args.smoke} device={device}", flush=True)
    patch_proj = load_patch_proj(device)
    dataset, _indices, source_patch, records = union_source_records(args.smoke)
    cache_out = SPATIAL_OUT if not args.smoke else out
    z_cache = projected_source_cache(cache_out, records, source_patch, patch_proj, device)
    queries, source_rows = build_directions(z_cache, records, dataset)
    for name, q in queries.items():
        np.save(out / f"{name}.npy", q)
    raw, percat = evaluate_all(queries, patch_proj, device, args.smoke)
    geom = direction_geometry(queries)
    summary_rows = summarize(raw)
    outcome, gate = decide(summary_rows, geom)
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "DIRECTION_GEOMETRY.csv", geom)
    write_csv(out / "SOURCE_STATISTICS.csv", source_rows + summary_rows)
    write_json(out / "OUTCOME.json", {
        "phase": "anomaly_content_vs_spatial_bias_audit",
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "protocol": "strict_source_only_cross_dataset_zsad",
        "variants": ["B0_LocalContrast", "E1_SpatialForegroundControl", "E2_ContentResidual"],
        "outcome": outcome,
        "gate": gate,
        "smoke": args.smoke,
        "runtime_sec": time.time() - start,
        "checkpoint": str(CHECKPOINT),
        "seed": SEED,
        "strict_source_only": True,
        "target_information_used_for_training": False,
        "training_used": False,
        "memory_used": False,
        "prompt_used": False,
        "adapter_used": False,
        "dino_used": False,
    })
    write_report(out, outcome, gate, raw, geom, source_rows, summary_rows)
    print(f"[ContentSpatial] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
