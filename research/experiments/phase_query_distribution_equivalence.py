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

from experiments.phase_defect_query_transfer_pilot import CHECKPOINT, EPOCHS, LR, SEED, SIDE
from experiments.phase_spatial_supervision_transfer_audit import (
    corrupt_masks,
    evaluate_all,
    make_labels,
    projected_source_cache,
    train_query,
    union_source_records,
)
from experiments.phase_type_factorized_consensus_query import load_patch_proj
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_query_distribution_equivalence")
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


def mean_difference_query(z_cache, labels, records, tag):
    sum_a = np.zeros((768,), dtype=np.float64)
    sum_n = np.zeros((768,), dtype=np.float64)
    count_a = 0
    count_n = 0
    pos_fracs = []
    for i, rec in enumerate(records):
        z = np.asarray(z_cache[i], dtype=np.float32)
        y = np.asarray(labels[i], dtype=bool)
        if y.any():
            sum_a += z[y].sum(axis=0, dtype=np.float64)
            count_a += int(y.sum())
        neg = ~y
        if neg.any():
            sum_n += z[neg].sum(axis=0, dtype=np.float64)
            count_n += int(neg.sum())
        pos_fracs.append(float(y.mean()))
    mu_a = sum_a / max(count_a, 1)
    mu_n = sum_n / max(count_n, 1)
    delta = mu_a - mu_n
    q, delta_norm = norm_vec(delta)
    rows = [
        {"variant": tag, "stat": "anomaly_patch_count", "value": count_a},
        {"variant": tag, "stat": "normal_patch_count", "value": count_n},
        {"variant": tag, "stat": "mean_positive_patch_fraction", "value": float(np.mean(pos_fracs))},
        {"variant": tag, "stat": "mu_A_minus_mu_N_norm", "value": delta_norm},
        {"variant": tag, "stat": "mu_A_norm", "value": float(np.linalg.norm(mu_a))},
        {"variant": tag, "stat": "mu_N_norm", "value": float(np.linalg.norm(mu_n))},
    ]
    return q, rows


def query_geometry(queries, source_stats):
    rows = []
    for name, q in queries.items():
        rows.append({"variant": name, "query_norm": float(np.linalg.norm(q))})
    pairs = [
        ("A0_LearnedQuery", "E1_MeanDifference"),
        ("A0_LearnedQuery", "E2_CorruptedMeanDifference"),
        ("E1_MeanDifference", "E2_CorruptedMeanDifference"),
    ]
    for a, b in pairs:
        qa, _ = norm_vec(queries[a])
        qb, _ = norm_vec(queries[b])
        rows.append({
            "pair": f"{a}_vs_{b}",
            "cosine": float(np.dot(qa, qb)),
            "l2_distance": float(np.linalg.norm(qa - qb)),
        })
    rows.extend({
        "variant": r["variant"],
        "source_stat": r["stat"],
        "value": r["value"],
    } for r in source_stats if r["stat"] == "mu_A_minus_mu_N_norm")
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in sorted({r["dataset"] for r in raw}):
        a0 = by[(ds, "A0_LearnedQuery")]
        e1 = by[(ds, "E1_MeanDifference")]
        e2 = by[(ds, "E2_CorruptedMeanDifference")]
        rows.append({
            "dataset": ds,
            "A0_G": float(a0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_A0_G": float(e1["G_pixel"]) - float(a0["G_pixel"]),
            "E2_minus_A0_G": float(e2["G_pixel"]) - float(a0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_A0_P-AUROC": float(e1["P-AUROC"]) - float(a0["P-AUROC"]),
            "E1_minus_A0_P-AP": float(e1["P-AP"]) - float(a0["P-AP"]),
            "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def decide(geom, summary_rows):
    cos_ld = next(float(r["cosine"]) for r in geom if r.get("pair") == "A0_LearnedQuery_vs_E1_MeanDifference")
    mean_e1_a0 = float(np.mean([r["E1_minus_A0_G"] for r in summary_rows]))
    gap = -mean_e1_a0
    if cos_ld >= 0.90 and gap <= 1.0:
        outcome = "FIRST_ORDER_DISTRIBUTION_EQUIVALENCE_SUPPORTED"
    elif cos_ld >= 0.90 and gap > 1.0:
        outcome = "GEOMETRIC_EQUIVALENCE_NOT_FUNCTIONAL_EQUIVALENCE"
    elif cos_ld < 0.90 and gap <= 1.0:
        outcome = "FUNCTIONAL_EQUIVALENCE_WITH_DIFFERENT_GEOMETRY"
    else:
        outcome = "FIRST_ORDER_DISTRIBUTION_EQUIVALENCE_REJECTED"
    return outcome, cos_ld, mean_e1_a0


def write_report(out, outcome, raw, percat, geom, source_stats, summary_rows, cos_ld, mean_e1_a0):
    cos_lc = next(float(r["cosine"]) for r in geom if r.get("pair") == "A0_LearnedQuery_vs_E2_CorruptedMeanDifference")
    cos_dc = next(float(r["cosine"]) for r in geom if r.get("pair") == "E1_MeanDifference_vs_E2_CorruptedMeanDifference")
    lines = [
        "# Query-Distribution Equivalence Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## Question",
        "",
        "Does the transferable behavior of the Shared Anomaly Query reduce to the first-order source abnormal-minus-normal patch distribution direction?",
        "",
        "## Protocol",
        "",
        "Full VisA source only. Targets are MVTec, BTAD, and CVC-ClinicDB. CLIP backbone and patch projection are frozen. No text, prompt, Memory, DINO, adapter, defect type, target training, target statistics, target normal reference, or target tuning is used.",
        "",
        "## Variants",
        "",
        "- A0_LearnedQuery: same fixed Shared Query protocol as previous TrueMask E1, regenerated with seed0 because the prior run did not persist the query vector.",
        "- E1_MeanDifference: no training; q_delta = normalize(mean(source anomaly-mask patches) - mean(source normal/background patches)).",
        "- E2_CorruptedMeanDifference: no training; same statistic computed with the previous seed0 corrupted masks.",
        "",
        "## Target Results",
        "",
        "| Dataset | Variant | P-AUROC | P-AP | G |",
        "|---|---|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} |")
    lines.extend(["", "## Geometry", ""])
    for r in geom:
        if "variant" in r and "query_norm" in r:
            lines.append(f"- {r['variant']}: norm={float(r['query_norm']):.4f}.")
        elif "pair" in r:
            lines.append(f"- {r['pair']}: cosine={float(r['cosine']):.4f}, L2={float(r['l2_distance']):.4f}.")
    lines.extend([
        "",
        "## Source Statistics",
        "",
        "| Variant | Statistic | Value |",
        "|---|---|---:|",
    ])
    for r in source_stats:
        lines.append(f"| {r['variant']} | {r['stat']} | {float(r['value']):.6f} |")
    lines.extend([
        "",
        "## Gate",
        "",
        f"- cos(q_learned, q_delta) = {cos_ld:.4f}.",
        f"- mean G(E1 - A0) = {mean_e1_a0:+.4f} pp.",
        f"- cos(q_learned, q_delta_corrupt) = {cos_lc:.4f}.",
        f"- cos(q_delta, q_delta_corrupt) = {cos_dc:.4f}.",
        "",
        "## E1 vs E2",
        "",
        "| Dataset | E1-A0 G | E2-A0 G | E1-E2 G |",
        "|---|---:|---:|---:|",
    ])
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['E1_minus_A0_G']:+.4f} | {r['E2_minus_A0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        "Per-category results are saved in `PER_CATEGORY.csv`.",
        "",
        "STOP. No covariance, PCA, LDA, Memory, or new method was started.",
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
    print(f"[QueryDistribution] start smoke={args.smoke} device={device}", flush=True)
    patch_proj = load_patch_proj(device)
    _dataset, _indices, source_patch, records = union_source_records(args.smoke)
    cache_out = SPATIAL_OUT if not args.smoke else out
    z_cache = projected_source_cache(cache_out, records, source_patch, patch_proj, device)
    corrupted_masks, _audit = corrupt_masks(records)
    labels = make_labels(records, corrupted_masks)

    q_learned, curve = train_query(z_cache, records, labels["E1_TrueMask"], "A0_LearnedQuery", device)
    q_delta, stats_delta = mean_difference_query(z_cache, labels["E1_TrueMask"], records, "E1_MeanDifference")
    q_corrupt, stats_corrupt = mean_difference_query(z_cache, labels["E2_CorruptedMask"], records, "E2_CorruptedMeanDifference")
    queries = {
        "A0_LearnedQuery": q_learned,
        "E1_MeanDifference": q_delta,
        "E2_CorruptedMeanDifference": q_corrupt,
    }
    np.save(out / "q_learned.npy", q_learned)
    np.save(out / "q_delta.npy", q_delta)
    np.save(out / "q_delta_corrupt.npy", q_corrupt)

    raw, percat = evaluate_all(queries, patch_proj, device, args.smoke)
    source_stats = stats_delta + stats_corrupt
    geom = query_geometry(queries, source_stats)
    summary_rows = summarize(raw)
    outcome, cos_ld, mean_e1_a0 = decide(geom, summary_rows)

    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "QUERY_GEOMETRY.csv", geom)
    write_csv(out / "SOURCE_STATISTICS.csv", source_stats + summary_rows)
    write_csv(out / "TRAIN_CURVE_A0.csv", curve)
    write_json(out / "OUTCOME.json", {
        "phase": "query_distribution_equivalence_audit",
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "protocol": "strict_source_only_cross_dataset_zsad",
        "variants": ["A0_LearnedQuery", "E1_MeanDifference", "E2_CorruptedMeanDifference"],
        "outcome": outcome,
        "cos_q_learned_q_delta": cos_ld,
        "mean_gain_E1_vs_A0_G": mean_e1_a0,
        "smoke": args.smoke,
        "runtime_sec": time.time() - start,
        "checkpoint": str(CHECKPOINT),
        "epochs_A0": EPOCHS,
        "lr_A0": LR,
        "seed": SEED,
        "strict_source_only": True,
        "target_information_used_for_training": False,
        "text_used": False,
        "memory_used": False,
        "dino_used": False,
        "adapter_used": False,
        "notes": "A0 query regenerated with the previous fixed TrueMask protocol because the preceding phase did not persist q_A. Pixel metrics use the same 37x37 query patch-grid evaluator.",
    })
    write_report(out, outcome, raw, percat, geom, source_stats, summary_rows, cos_ld, mean_e1_a0)
    print(f"[QueryDistribution] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
