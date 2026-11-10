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

from experiments.phase_defect_query_transfer_pilot import CHECKPOINT, SEED
from experiments.phase_spatial_supervision_transfer_audit import evaluate_all, projected_source_cache, union_source_records
from experiments.phase_type_factorized_consensus_query import load_patch_proj
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_local_anomaly_residual_consistency")
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


def anomaly_image_means(z_cache, records, dataset):
    rows = []
    sum_in = np.zeros((768,), dtype=np.float64)
    sum_out = np.zeros((768,), dtype=np.float64)
    count_in = 0
    count_out = 0
    for rid, rec in enumerate(records):
        if bool(rec["normal"]):
            continue
        z = np.asarray(z_cache[rid], dtype=np.float32)
        m = np.asarray(rec["mask"], dtype=bool)
        if not m.any() or not (~m).any():
            continue
        mu_in = z[m].mean(axis=0, dtype=np.float64)
        mu_out = z[~m].mean(axis=0, dtype=np.float64)
        d, dn = norm_vec(mu_in - mu_out)
        cls_name = str(dataset.data_all[int(rec["idx"])]["cls_name"])
        rows.append({
            "record_id": rid,
            "image_idx": int(rec["idx"]),
            "cls_name": cls_name,
            "mu_in": mu_in,
            "mu_out": mu_out,
            "d_pair": d,
            "residual_norm": dn,
            "mask_area": int(m.sum()),
            "context_area": int((~m).sum()),
        })
        sum_in += z[m].sum(axis=0, dtype=np.float64)
        sum_out += z[~m].sum(axis=0, dtype=np.float64)
        count_in += int(m.sum())
        count_out += int((~m).sum())
    mu_in_global = sum_in / max(count_in, 1)
    mu_out_global = sum_out / max(count_out, 1)
    q_global, n_global = norm_vec(mu_in_global - mu_out_global)
    return rows, q_global, n_global, count_in, count_out


def derangement(rows):
    rng = random.Random(SEED)
    by_cls = defaultdict(list)
    for i, r in enumerate(rows):
        by_cls[r["cls_name"]].append(i)
    mapping = {}
    leftovers = []
    for cls, ids in sorted(by_cls.items()):
        ids = list(ids)
        if len(ids) == 1:
            leftovers.extend(ids)
            continue
        shuffled = ids[:]
        rng.shuffle(shuffled)
        if any(a == b for a, b in zip(ids, shuffled)):
            shuffled = shuffled[1:] + shuffled[:1]
        for a, b in zip(ids, shuffled):
            mapping[a] = b
    all_ids = list(range(len(rows)))
    for a in leftovers:
        candidates = [x for x in all_ids if x != a]
        mapping[a] = rng.choice(candidates)
    return mapping


def build_queries(rows, q_global):
    pair_stack = np.stack([r["d_pair"] for r in rows], axis=0).astype(np.float32)
    q_pair, pair_strength = norm_vec(pair_stack.mean(axis=0))
    mapping = derangement(rows)
    mismatch = []
    map_rows = []
    for i, r in enumerate(rows):
        j = mapping[i]
        d, dn = norm_vec(r["mu_in"] - rows[j]["mu_out"])
        mismatch.append(d)
        map_rows.append({
            "record_id": r["record_id"],
            "image_idx": r["image_idx"],
            "cls_name": r["cls_name"],
            "matched_context_record_id": rows[j]["record_id"],
            "matched_context_image_idx": rows[j]["image_idx"],
            "matched_context_cls_name": rows[j]["cls_name"],
            "same_category": r["cls_name"] == rows[j]["cls_name"],
            "mismatched_residual_norm": dn,
        })
    mismatch_stack = np.stack(mismatch, axis=0).astype(np.float32)
    q_mismatch, mismatch_strength = norm_vec(mismatch_stack.mean(axis=0))
    return {
        "B0_GlobalLocalContrast": q_global,
        "E1_PairedResidualConsensus": q_pair,
        "E2_MismatchedContextConsensus": q_mismatch,
    }, pair_stack, mismatch_stack, map_rows, pair_strength, mismatch_strength


def pairwise_summary(stack, q, variant, category=None):
    sims = stack @ stack.T
    tri = sims[np.triu_indices_from(sims, k=1)]
    align = stack @ q
    prefix = {"variant": variant}
    if category is not None:
        prefix["category"] = category
    return {
        **prefix,
        "num_residuals": int(stack.shape[0]),
        "pairwise_cosine_mean": float(np.mean(tri)) if tri.size else float("nan"),
        "pairwise_cosine_std": float(np.std(tri)) if tri.size else float("nan"),
        "cos_to_consensus_mean": float(np.mean(align)),
        "cos_to_consensus_std": float(np.std(align)),
        "ConsensusStrength": float(np.linalg.norm(stack.mean(axis=0))),
    }


def residual_geometry(queries, rows, pair_stack, mismatch_stack, pair_strength, mismatch_strength):
    out = []
    for name, q in queries.items():
        out.append({"variant": name, "query_norm": float(np.linalg.norm(q))})
    pairs = [
        ("B0_GlobalLocalContrast", "E1_PairedResidualConsensus"),
        ("B0_GlobalLocalContrast", "E2_MismatchedContextConsensus"),
        ("E1_PairedResidualConsensus", "E2_MismatchedContextConsensus"),
    ]
    for a, b in pairs:
        qa, _ = norm_vec(queries[a])
        qb, _ = norm_vec(queries[b])
        out.append({"pair": f"{a}_vs_{b}", "cosine": float(np.dot(qa, qb)), "l2_distance": float(np.linalg.norm(qa - qb))})
    out.append(pairwise_summary(pair_stack, queries["E1_PairedResidualConsensus"], "E1_PairedResidualConsensus"))
    out[-1]["ConsensusStrength"] = pair_strength
    out.append(pairwise_summary(mismatch_stack, queries["E2_MismatchedContextConsensus"], "E2_MismatchedContextConsensus"))
    out[-1]["ConsensusStrength"] = mismatch_strength
    by_cls = defaultdict(list)
    for i, r in enumerate(rows):
        by_cls[r["cls_name"]].append(i)
    for cls, ids in sorted(by_cls.items()):
        if len(ids) < 2:
            continue
        ids = np.asarray(ids, dtype=np.int64)
        out.append(pairwise_summary(pair_stack[ids], queries["E1_PairedResidualConsensus"], "E1_PairedResidualConsensus", cls))
        out.append(pairwise_summary(mismatch_stack[ids], queries["E2_MismatchedContextConsensus"], "E2_MismatchedContextConsensus", cls))
    return out


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in sorted({r["dataset"] for r in raw}):
        b0 = by[(ds, "B0_GlobalLocalContrast")]
        e1 = by[(ds, "E1_PairedResidualConsensus")]
        e2 = by[(ds, "E2_MismatchedContextConsensus")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
        })
    return rows


def decide(summary_rows):
    e1_b0 = float(np.mean([r["E1_minus_B0_G"] for r in summary_rows]))
    e1_e2 = float(np.mean([r["E1_minus_E2_G"] for r in summary_rows]))
    if e1_b0 < -1.0:
        outcome = "INSTANCE_RESIDUAL_CONSENSUS_NOT_SUFFICIENT"
    elif abs(e1_b0) <= 1.0 and e1_e2 >= 0.5:
        outcome = "PAIRED_LOCAL_RESIDUAL_SUPPORTED"
    elif abs(e1_b0) <= 1.0 and e1_e2 < 0.5:
        outcome = "MARGINAL_LOCAL_CONTRAST_SUFFICIENT"
    else:
        outcome = "INSTANCE_RESIDUAL_CONSENSUS_NOT_SUFFICIENT"
    return outcome, {"mean_E1_minus_B0_G": e1_b0, "mean_E1_minus_E2_G": e1_e2}


def write_report(out, outcome, gate, raw, geom, source_rows, summary_rows):
    lines = [
        "# Local Anomaly Residual Consistency Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## Question",
        "",
        "Does transferable local contrast arise from consistent same-image anomaly-minus-context residual directions?",
        "",
        "## Protocol",
        "",
        "Full VisA source only; MVTec, BTAD, and CVC-ClinicDB are final evaluation only. No training, Memory, prompt, adapter, DINO, target statistics, or target normal reference.",
        "",
        "## Target Results",
        "",
        "| Dataset | Variant | P-AUROC | P-AP | G |",
        "|---|---|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} |")
    lines.extend(["", "## Deltas", "", "| Dataset | E1-B0 G | E1-E2 G |", "|---|---:|---:|"])
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend(["", "## Geometry", ""])
    for r in geom:
        if "pair" in r:
            lines.append(f"- {r['pair']}: cosine={float(r['cosine']):.4f}, L2={float(r['l2_distance']):.4f}.")
        elif "pairwise_cosine_mean" in r and "category" not in r:
            lines.append(f"- {r['variant']}: pairwise mean={float(r['pairwise_cosine_mean']):.4f}, pairwise std={float(r['pairwise_cosine_std']):.4f}, cos-to-consensus mean={float(r['cos_to_consensus_mean']):.4f}, ConsensusStrength={float(r['ConsensusStrength']):.4f}.")
        elif "query_norm" in r:
            lines.append(f"- {r['variant']}: norm={float(r['query_norm']):.4f}.")
    lines.extend(["", "## Source Statistics", "", "| Statistic | Value |", "|---|---:|"])
    for r in source_rows:
        lines.append(f"| {r['stat']} | {float(r['value']):.6f} |")
    lines.extend([
        "",
        "## Gate",
        "",
        f"- mean E1-B0 G = {gate['mean_E1_minus_B0_G']:+.4f} pp.",
        f"- mean E1-E2 G = {gate['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
        "Per-category target results are saved in `PER_CATEGORY.csv`; per-object source geometry is saved in `RESIDUAL_GEOMETRY.csv`.",
        "",
        "STOP. No extra variant was added.",
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
    print(f"[LocalResidual] start smoke={args.smoke} device={device}", flush=True)
    patch_proj = load_patch_proj(device)
    dataset, _indices, source_patch, records = union_source_records(args.smoke)
    cache_out = SPATIAL_OUT if not args.smoke else out
    z_cache = projected_source_cache(cache_out, records, source_patch, patch_proj, device)
    rows, q_global, n_global, count_in, count_out = anomaly_image_means(z_cache, records, dataset)
    queries, pair_stack, mismatch_stack, mismatch_map, pair_strength, mismatch_strength = build_queries(rows, q_global)
    for name, q in queries.items():
        np.save(out / f"{name}.npy", q)
    raw, percat = evaluate_all(queries, patch_proj, device, args.smoke)
    geom = residual_geometry(queries, rows, pair_stack, mismatch_stack, pair_strength, mismatch_strength)
    summary_rows = summarize(raw)
    outcome, gate = decide(summary_rows)
    source_rows = [
        {"stat": "anomaly_images_used", "value": len(rows)},
        {"stat": "mask_in_patch_count", "value": count_in},
        {"stat": "mask_out_patch_count", "value": count_out},
        {"stat": "norm_global_mu_in_minus_mu_out", "value": n_global},
        {"stat": "pair_consensus_strength", "value": pair_strength},
        {"stat": "mismatch_consensus_strength", "value": mismatch_strength},
        {"stat": "same_category_mismatch_fraction", "value": float(np.mean([bool(r["same_category"]) for r in mismatch_map])) if mismatch_map else float("nan")},
    ]
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "RESIDUAL_GEOMETRY.csv", geom)
    write_csv(out / "SOURCE_STATISTICS.csv", source_rows + summary_rows + mismatch_map)
    write_json(out / "OUTCOME.json", {
        "phase": "local_anomaly_residual_consistency_audit",
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "protocol": "strict_source_only_cross_dataset_zsad",
        "variants": ["B0_GlobalLocalContrast", "E1_PairedResidualConsensus", "E2_MismatchedContextConsensus"],
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
    print(f"[LocalResidual] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
