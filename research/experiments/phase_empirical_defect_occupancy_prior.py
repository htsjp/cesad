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
from experiments.phase_spatial_supervision_transfer_audit import evaluate_all, projected_source_cache, union_source_records
from experiments.phase_type_factorized_consensus_query import load_patch_proj
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_empirical_defect_occupancy_prior")
SPATIAL_OUT = Path("results/phase_spatial_supervision_transfer_audit/formal")
MASK_GEOM_OUT = Path("results/phase_source_mask_geometry_confound/formal")


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


def normal_global_spatial_mean(z_cache, records):
    total = np.zeros((SIDE * SIDE, 768), dtype=np.float64)
    count = 0
    for i, rec in enumerate(records):
        if bool(rec["normal"]):
            total += np.asarray(z_cache[i], dtype=np.float32)
            count += 1
    return total / max(count, 1), count


def occupancy_map(records):
    total = np.zeros((SIDE * SIDE,), dtype=np.float64)
    count = 0
    for rec in records:
        if bool(rec["normal"]):
            continue
        m = np.asarray(rec["mask"], dtype=np.float64).reshape(-1)
        if m.sum() <= 0:
            continue
        total += m
        count += 1
    return total / max(count, 1), count


def direction_from_weights(fmap, w, variant):
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    comp = 1.0 - w
    mu_in = (w[:, None] * fmap).sum(axis=0) / max(float(w.sum()), 1e-12)
    mu_out = (comp[:, None] * fmap).sum(axis=0) / max(float(comp.sum()), 1e-12)
    q, dn = norm_vec(mu_in - mu_out)
    rows = [
        {"variant": variant, "stat": "raw_difference_norm", "value": dn},
        {"variant": variant, "stat": "weight_sum", "value": float(w.sum())},
        {"variant": variant, "stat": "complement_weight_sum", "value": float(comp.sum())},
        {"variant": variant, "stat": "mu_weighted_norm", "value": float(np.linalg.norm(mu_in))},
        {"variant": variant, "stat": "mu_complement_norm", "value": float(np.linalg.norm(mu_out))},
    ]
    return q, rows


def true_mask_spatial_direction(z_cache, records):
    fmap, _normal_count = normal_global_spatial_mean(z_cache, records)
    sum_in = np.zeros((768,), dtype=np.float64)
    sum_out = np.zeros((768,), dtype=np.float64)
    count_in = 0
    count_out = 0
    used = 0
    for rec in records:
        if bool(rec["normal"]):
            continue
        m = np.asarray(rec["mask"], dtype=bool).reshape(-1)
        if not m.any() or not (~m).any():
            continue
        sum_in += fmap[m].sum(axis=0, dtype=np.float64)
        sum_out += fmap[~m].sum(axis=0, dtype=np.float64)
        count_in += int(m.sum())
        count_out += int((~m).sum())
        used += 1
    q, dn = norm_vec(sum_in / max(count_in, 1) - sum_out / max(count_out, 1))
    rows = [
        {"variant": "B0_TrueMaskSpatial", "stat": "raw_difference_norm", "value": dn},
        {"variant": "B0_TrueMaskSpatial", "stat": "mask_records_used", "value": used},
        {"variant": "B0_TrueMaskSpatial", "stat": "mask_in_patch_count", "value": count_in},
        {"variant": "B0_TrueMaskSpatial", "stat": "mask_out_patch_count", "value": count_out},
    ]
    return q, rows


def occupancy_stats(p, p_shuffle):
    yy, xx = np.divmod(np.arange(SIDE * SIDE), SIDE)
    def one(name, arr):
        arr = np.asarray(arr, dtype=np.float64)
        s = float(arr.sum())
        return [
            {"map": name, "stat": "min", "value": float(arr.min())},
            {"map": name, "stat": "max", "value": float(arr.max())},
            {"map": name, "stat": "mean", "value": float(arr.mean())},
            {"map": name, "stat": "std", "value": float(arr.std())},
            {"map": name, "stat": "effective_occupied_area_sum", "value": s},
            {"map": name, "stat": "effective_cells_participation_ratio", "value": float((s * s) / max(float((arr * arr).sum()), 1e-12))},
            {"map": name, "stat": "center_of_mass_y", "value": float((arr * yy).sum() / max(s, 1e-12))},
            {"map": name, "stat": "center_of_mass_x", "value": float((arr * xx).sum() / max(s, 1e-12))},
        ]
    return one("empirical_occupancy", p) + one("shuffled_occupancy", p_shuffle)


def geometry(queries):
    rows = []
    for name, q in queries.items():
        rows.append({"variant": name, "query_norm": float(np.linalg.norm(q))})
    pairs = [
        ("B0_TrueMaskSpatial", "E1_EmpiricalOccupancySpatial"),
        ("B0_TrueMaskSpatial", "E2_ShuffledOccupancySpatial"),
        ("E1_EmpiricalOccupancySpatial", "E2_ShuffledOccupancySpatial"),
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
        b0 = by[(ds, "B0_TrueMaskSpatial")]
        e1 = by[(ds, "E1_EmpiricalOccupancySpatial")]
        e2 = by[(ds, "E2_ShuffledOccupancySpatial")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "B0_minus_E1_G": float(b0["G_pixel"]) - float(e1["G_pixel"]),
        })
    return rows


def decide(summary_rows, geom_rows):
    e1_b0 = float(np.mean([r["E1_minus_B0_G"] for r in summary_rows]))
    e1_e2 = float(np.mean([r["E1_minus_E2_G"] for r in summary_rows]))
    b0_e1 = -e1_b0
    vals = []
    for r in summary_rows:
        vals.extend([r["B0_G"], r["E1_G"], r["E2_G"]])
    max_mean_spread = max(
        abs(float(np.mean([r["B0_G"] for r in summary_rows])) - float(np.mean([r["E1_G"] for r in summary_rows]))),
        abs(float(np.mean([r["B0_G"] for r in summary_rows])) - float(np.mean([r["E2_G"] for r in summary_rows]))),
        abs(float(np.mean([r["E1_G"] for r in summary_rows])) - float(np.mean([r["E2_G"] for r in summary_rows]))),
    )
    cos01 = next(float(r["cosine"]) for r in geom_rows if r.get("pair") == "B0_TrueMaskSpatial_vs_E1_EmpiricalOccupancySpatial")
    cos12 = next(float(r["cosine"]) for r in geom_rows if r.get("pair") == "E1_EmpiricalOccupancySpatial_vs_E2_ShuffledOccupancySpatial")
    if abs(e1_b0) <= 1.0 and e1_e2 >= 3.0:
        outcome = "AGGREGATE_DEFECT_OCCUPANCY_PRIOR_SUFFICIENT"
    elif b0_e1 >= 3.0:
        outcome = "INSTANCE_MASK_GEOMETRY_REQUIRED"
    elif abs(e1_e2) <= 1.0:
        outcome = "OCCUPANCY_POSITION_NOT_CAUSAL"
    elif max_mean_spread < 1.0 and min(cos01, cos12) >= 0.90:
        outcome = "OCCUPANCY_POSITION_NOT_CAUSAL"
    else:
        outcome = "OCCUPANCY_PRIOR_INCONCLUSIVE"
    return outcome, {
        "mean_E1_minus_B0_G": e1_b0,
        "mean_E1_minus_E2_G": e1_e2,
        "mean_B0_minus_E1_G": b0_e1,
        "max_mean_variant_spread_G": max_mean_spread,
        "cos_B0_E1": cos01,
        "cos_E1_E2": cos12,
    }


def write_report(out, outcome, gate, raw, summary_rows, geom_rows, occ_rows):
    lines = [
        "# Empirical Defect Occupancy Prior Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## Protocol",
        "",
        "No training. B0 uses true source mask coordinates on global normal spatial features; E1 replaces individual masks with the aggregate source occupancy map; E2 spatially permutes the occupancy map values with seed0.",
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
    for r in geom_rows:
        if "variant" in r:
            lines.append(f"- {r['variant']}: norm={float(r['query_norm']):.4f}.")
        else:
            lines.append(f"- {r['pair']}: cosine={float(r['cosine']):.4f}, L2={float(r['l2_distance']):.4f}.")
    lines.extend(["", "## Occupancy Statistics", "", "| Map/Variant | Statistic | Value |", "|---|---|---:|"])
    for r in occ_rows:
        label = r.get("map") or r.get("variant")
        lines.append(f"| {label} | {r['stat']} | {float(r['value']):.6f} |")
    lines.extend([
        "",
        "## Gate",
        "",
        json.dumps(gate, indent=2, default=json_default),
        "",
        "Per-category target results are saved in `PER_CATEGORY.csv`.",
        "",
        "STOP. No center prior, Gaussian prior, Memory, PCA, or new variant was started.",
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
    print(f"[OccupancyPrior] start smoke={args.smoke} device={device}", flush=True)
    patch_proj = load_patch_proj(device)
    _dataset, _indices, source_patch, records = union_source_records(args.smoke)
    cache_out = SPATIAL_OUT if not args.smoke else out
    z_cache = projected_source_cache(cache_out, records, source_patch, patch_proj, device)
    fmap, normal_count = normal_global_spatial_mean(z_cache, records)
    p, mask_count = occupancy_map(records)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(SIDE * SIDE)
    p_shuffle = p[perm]
    q0, s0 = true_mask_spatial_direction(z_cache, records)
    q1, s1 = direction_from_weights(fmap, p, "E1_EmpiricalOccupancySpatial")
    q2, s2 = direction_from_weights(fmap, p_shuffle, "E2_ShuffledOccupancySpatial")
    queries = {"B0_TrueMaskSpatial": q0, "E1_EmpiricalOccupancySpatial": q1, "E2_ShuffledOccupancySpatial": q2}
    for name, q in queries.items():
        np.save(out / f"{name}.npy", q)
    np.save(out / "empirical_occupancy_map.npy", p.reshape(SIDE, SIDE).astype(np.float32))
    np.save(out / "shuffled_occupancy_map.npy", p_shuffle.reshape(SIDE, SIDE).astype(np.float32))
    raw, percat = evaluate_all(queries, patch_proj, device, args.smoke)
    geom_rows = geometry(queries)
    summary_rows = summarize(raw)
    occ_rows = occupancy_stats(p, p_shuffle) + s0 + s1 + s2 + [
        {"variant": "source", "stat": "normal_images_used", "value": normal_count},
        {"variant": "source", "stat": "anomaly_masks_used", "value": mask_count},
    ]
    outcome, gate = decide(summary_rows, geom_rows)
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "DIRECTION_GEOMETRY.csv", geom_rows)
    write_csv(out / "OCCUPANCY_STATISTICS.csv", occ_rows + summary_rows)
    write_json(out / "OUTCOME.json", {
        "phase": "empirical_defect_occupancy_prior_audit",
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
        "normal_images_used": normal_count,
        "anomaly_masks_used": mask_count,
        "strict_source_only": True,
        "training_used": False,
        "target_information_used_for_training": False,
        "memory_used": False,
        "prompt_used": False,
        "adapter_used": False,
        "dino_used": False,
    })
    write_report(out, outcome, gate, raw, summary_rows, geom_rows, occ_rows)
    print(f"[OccupancyPrior] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
