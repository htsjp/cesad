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

from experiments.phase_defect_query_transfer_pilot import CHECKPOINT, SEED
from experiments.phase_spatial_supervision_transfer_audit import (
    evaluate_all,
    projected_source_cache,
    union_source_records,
)
from experiments.phase_type_factorized_consensus_query import load_patch_proj
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_mean_difference_signal_decomposition")
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


def compute_means(z_cache, records):
    sum_in = np.zeros((768,), dtype=np.float64)
    sum_out = np.zeros((768,), dtype=np.float64)
    sum_normal = np.zeros((768,), dtype=np.float64)
    count_in = 0
    count_out = 0
    count_normal = 0
    anomaly_images = 0
    normal_images = 0
    for i, rec in enumerate(records):
        z = np.asarray(z_cache[i], dtype=np.float32)
        if bool(rec["normal"]):
            sum_normal += z.sum(axis=0, dtype=np.float64)
            count_normal += int(z.shape[0])
            normal_images += 1
            continue
        m = np.asarray(rec["mask"], dtype=bool)
        anomaly_images += 1
        if m.any():
            sum_in += z[m].sum(axis=0, dtype=np.float64)
            count_in += int(m.sum())
        out = ~m
        if out.any():
            sum_out += z[out].sum(axis=0, dtype=np.float64)
            count_out += int(out.sum())
    mu_in = sum_in / max(count_in, 1)
    mu_out = sum_out / max(count_out, 1)
    mu_normal = sum_normal / max(count_normal, 1)
    mu_mix = (sum_out + sum_normal) / max(count_out + count_normal, 1)
    stats = {
        "anomaly_images": anomaly_images,
        "normal_images": normal_images,
        "mask_in_patch_count": count_in,
        "mask_out_patch_count": count_out,
        "normal_image_patch_count": count_normal,
        "mu_in_norm": float(np.linalg.norm(mu_in)),
        "mu_out_norm": float(np.linalg.norm(mu_out)),
        "mu_normal_norm": float(np.linalg.norm(mu_normal)),
        "mu_N_mix_norm": float(np.linalg.norm(mu_mix)),
    }
    return mu_in, mu_out, mu_normal, mu_mix, stats


def build_queries(mu_in, mu_out, mu_normal, mu_mix):
    q_full, n_full = norm_vec(mu_in - mu_mix)
    q_local, n_local = norm_vec(mu_in - mu_out)
    q_context, n_context = norm_vec(mu_out - mu_normal)
    q_in_normal, n_in_normal = norm_vec(mu_in - mu_normal)
    queries = {
        "B0_FullMeanDifference": q_full,
        "E1_LocalContrast": q_local,
        "E2_ImageContext": q_context,
    }
    dist_rows = [
        {"stat": "norm_mu_in_minus_mu_N_mix", "value": n_full},
        {"stat": "norm_mu_in_minus_mu_out", "value": n_local},
        {"stat": "norm_mu_out_minus_mu_normal", "value": n_context},
        {"stat": "norm_mu_in_minus_mu_normal", "value": n_in_normal},
    ]
    return queries, dist_rows


def query_geometry(queries):
    rows = []
    for name, q in queries.items():
        rows.append({"variant": name, "query_norm": float(np.linalg.norm(q))})
    pairs = [
        ("B0_FullMeanDifference", "E1_LocalContrast"),
        ("B0_FullMeanDifference", "E2_ImageContext"),
        ("E1_LocalContrast", "E2_ImageContext"),
    ]
    for a, b in pairs:
        qa, _ = norm_vec(queries[a])
        qb, _ = norm_vec(queries[b])
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
        b0 = by[(ds, "B0_FullMeanDifference")]
        e1 = by[(ds, "E1_LocalContrast")]
        e2 = by[(ds, "E2_ImageContext")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
        })
    return rows


def decide(summary_rows):
    e1_gap = float(np.mean([r["E1_minus_B0_G"] for r in summary_rows]))
    e2_gap = float(np.mean([r["E2_minus_B0_G"] for r in summary_rows]))
    e1_close = abs(e1_gap) <= 1.0
    e2_close = abs(e2_gap) <= 1.0
    e1_weak = e1_gap <= -5.0
    e2_weak = e2_gap <= -5.0
    if e1_close and e2_weak:
        outcome = "LOCAL_ANOMALY_CONTRAST_DOMINANT"
    elif e2_close and e1_weak:
        outcome = "ANOMALY_IMAGE_CONTEXT_DOMINANT"
    elif e1_weak and e2_weak:
        outcome = "COMPOSITE_FIRST_ORDER_SIGNAL"
    elif e1_close and e2_close:
        outcome = "FIRST_ORDER_SIGNAL_HIGHLY_REDUNDANT"
    elif abs(e1_gap) < abs(e2_gap):
        outcome = "LOCAL_ANOMALY_CONTRAST_DOMINANT"
    elif abs(e2_gap) < abs(e1_gap):
        outcome = "ANOMALY_IMAGE_CONTEXT_DOMINANT"
    else:
        outcome = "FIRST_ORDER_SIGNAL_HIGHLY_REDUNDANT"
    return outcome, {
        "mean_E1_minus_B0_G": e1_gap,
        "mean_E2_minus_B0_G": e2_gap,
        "close_threshold_pp": 1.0,
        "weak_threshold_pp": -5.0,
    }


def write_report(out, outcome, gate, raw, geom, source_rows, summary_rows):
    lines = [
        "# Mean-Difference Signal Decomposition Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## Question",
        "",
        "This source-only statistics audit decomposes the strong true-mask MeanDifference direction into local anomaly contrast and anomaly-image context components.",
        "",
        "## Protocol",
        "",
        "Full VisA source only. MVTec, BTAD, and CVC-ClinicDB are used only for final evaluation. Frozen feature extractor/projection; no target statistics, target normal reference, Memory, prompt, adapter, DINO, PCA, covariance, or training.",
        "",
        "## Variants",
        "",
        "- B0_FullMeanDifference: normalize(mu_in - mu_N_mix), identical to the previous true-mask MeanDifference statistic.",
        "- E1_LocalContrast: normalize(mu_in - mu_out).",
        "- E2_ImageContext: normalize(mu_out - mu_normal).",
        "",
        "## Target Results",
        "",
        "| Dataset | Variant | P-AUROC | P-AP | G |",
        "|---|---|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} |")
    lines.extend(["", "## Decomposition Deltas", "", "| Dataset | E1-B0 G | E2-B0 G | E1-E2 G |", "|---|---:|---:|---:|"])
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['E1_minus_B0_G']:+.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        "## Geometry",
        "",
    ])
    for r in geom:
        if "variant" in r:
            lines.append(f"- {r['variant']}: norm={float(r['query_norm']):.4f}.")
        else:
            lines.append(f"- {r['pair']}: cosine={float(r['cosine']):.4f}, L2={float(r['l2_distance']):.4f}.")
    lines.extend([
        "",
        "## Source Statistics",
        "",
        "| Statistic | Value |",
        "|---|---:|",
    ])
    for r in source_rows:
        lines.append(f"| {r['stat']} | {float(r['value']):.6f} |")
    lines.extend([
        "",
        "## Gate",
        "",
        f"- Close-to-B0 threshold: abs(mean delta) <= {gate['close_threshold_pp']} pp.",
        f"- Weak threshold: mean delta <= {gate['weak_threshold_pp']} pp.",
        f"- mean E1-B0 G = {gate['mean_E1_minus_B0_G']:+.4f} pp.",
        f"- mean E2-B0 G = {gate['mean_E2_minus_B0_G']:+.4f} pp.",
        "",
        "Per-category results are saved in `PER_CATEGORY.csv`.",
        "",
        "STOP. No additional variant, PCA, covariance, LDA, Memory, or new method was started.",
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
    print(f"[MeanDiffDecomp] start smoke={args.smoke} device={device}", flush=True)
    patch_proj = load_patch_proj(device)
    _dataset, _indices, source_patch, records = union_source_records(args.smoke)
    cache_out = SPATIAL_OUT if not args.smoke else out
    z_cache = projected_source_cache(cache_out, records, source_patch, patch_proj, device)
    mu_in, mu_out, mu_normal, mu_mix, stats = compute_means(z_cache, records)
    queries, dist_rows = build_queries(mu_in, mu_out, mu_normal, mu_mix)
    for name, q in queries.items():
        np.save(out / f"{name}.npy", q)
    raw, percat = evaluate_all(queries, patch_proj, device, args.smoke)
    geom = query_geometry(queries)
    source_rows = [{"stat": k, "value": v} for k, v in stats.items()] + dist_rows
    summary_rows = summarize(raw)
    outcome, gate = decide(summary_rows)
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "QUERY_GEOMETRY.csv", geom)
    write_csv(out / "SOURCE_STATISTICS.csv", source_rows + summary_rows)
    write_json(out / "OUTCOME.json", {
        "phase": "mean_difference_signal_decomposition_audit",
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "protocol": "strict_source_only_cross_dataset_zsad",
        "variants": ["B0_FullMeanDifference", "E1_LocalContrast", "E2_ImageContext"],
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
    print(f"[MeanDiffDecomp] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
