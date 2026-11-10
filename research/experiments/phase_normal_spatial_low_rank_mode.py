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

from experiments.phase_defect_query_transfer_pilot import CHECKPOINT, SEED, SIDE, resize_binary_mask, safe_ap, safe_auc
from experiments.phase_empirical_defect_occupancy_prior import direction_from_weights, normal_global_spatial_mean, occupancy_map
from experiments.phase_spatial_supervision_transfer_audit import (
    fast_patch_grid_eval,
    projected_source_cache,
    score_query_on_target,
    target_cells,
    union_source_records,
)
from experiments.phase_type_factorized_consensus_query import load_patch_proj
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score


OUT = Path("results/phase_normal_spatial_low_rank_mode")
SPATIAL_OUT = Path("results/phase_spatial_supervision_transfer_audit/formal")
RANDOM_SEEDS = list(range(20))


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


def pc1_direction(fmap, q_occ):
    x = np.asarray(fmap, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    v1, _ = norm_vec(vt[0])
    if float(np.dot(v1, q_occ)) < 0:
        v1 = -v1
    total = float(np.sum(s * s))
    spectrum = []
    for i, sv in enumerate(s):
        spectrum.append({
            "rank": i + 1,
            "singular_value": float(sv),
            "explained_variance_ratio": float((sv * sv) / max(total, 1e-12)),
            "cumulative_explained_variance_ratio": float(np.sum(s[: i + 1] * s[: i + 1]) / max(total, 1e-12)),
        })
    return v1.astype(np.float32), spectrum


def build_directions(fmap, occ):
    q_occ, occ_rows = direction_from_weights(fmap, occ, "B0_OccupancyDirection")
    queries = {"B0_OccupancyDirection": q_occ}
    random_rows = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        p = occ[rng.permutation(SIDE * SIDE)]
        q, rows = direction_from_weights(fmap, p, f"E1_RandomWeighting_seed{seed}")
        queries[f"E1_RandomWeighting_seed{seed}"] = q
        for r in rows:
            r["seed"] = seed
            random_rows.append(r)
    q_pc1, spectrum = pc1_direction(fmap, q_occ)
    queries["E2_NormalSpatialPC1"] = q_pc1
    return queries, occ_rows + random_rows, spectrum


def batched_top1(scores):
    # scores: B x P x V
    k = max(1, int(scores.shape[1] * 0.01))
    part = np.partition(scores, scores.shape[1] - k, axis=1)[:, -k:, :]
    return part.mean(axis=1)


def score_queries_on_target(queries, patch, patch_proj, device, smoke, batch_size=16):
    names = list(queries.keys())
    q = torch.tensor(np.stack([queries[n] for n in names], axis=0), dtype=torch.float32, device=device)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = {name: np.zeros((n, SIDE, SIDE), dtype=np.float16) for name in names}
    image = {name: np.zeros((n,), dtype=np.float32) for name in names}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            arr = np.asarray(patch[start:end], dtype=np.float32)
            b = arr.shape[0]
            z = torch.tensor(arr.reshape(b * SIDE * SIDE, arr.shape[-1]), device=device)
            z = patch_proj(z, 0).float()
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scores = (z @ q.T).detach().cpu().numpy().astype(np.float32).reshape(b, SIDE * SIDE, len(names))
            top = batched_top1(scores)
            score_maps = scores.reshape(b, SIDE, SIDE, len(names))
            for j, name in enumerate(names):
                maps[name][start:end] = score_maps[:, :, :, j].astype(np.float16)
                image[name][start:end] = top[:, j].astype(np.float32)
    return names, image, maps


def evaluate_all_multi(queries, patch_proj, device, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        print(f"[LowRankMode] score all directions on {cell_id}", flush=True)
        names, images, maps = score_queries_on_target(queries, patch, patch_proj, device, smoke)
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        eval_cache = prepare_eval_cache(dataset, idx_use, kind, target)
        for name in names:
            metric, per = eval_cached(cell_id, target, eval_cache, images[name], maps[name])
            metric.update({"dataset": cell_id, "variant": name, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": name, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat


def prepare_eval_cache(dataset, idx_or_rows, kind, target):
    cached = []
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
        mask = resize_binary_mask(item).astype(np.uint8) if mask_available else np.zeros((SIDE * SIDE,), dtype=np.uint8)
        cached.append({"category": cat, "label": label, "mask_available": mask_available, "mask": mask})
    return cached


def eval_cached(cell_id, target, cached, image_scores, maps):
    by_cat = defaultdict(lambda: {"iy": [], "is": [], "py": [], "ps": [], "mask_images": 0})
    all_iy, all_is, all_py, all_ps = [], [], [], []
    for pos, item in enumerate(cached):
        cat = item["category"]
        label = item["label"]
        score = float(image_scores[pos])
        d = by_cat[cat]
        d["iy"].append(label)
        d["is"].append(score)
        all_iy.append(label)
        all_is.append(score)
        if item["mask_available"]:
            m = item["mask"]
            s = np.asarray(maps[pos], dtype=np.float32).reshape(-1)
            d["py"].append(m)
            d["ps"].append(s)
            d["mask_images"] += 1
            all_py.append(m)
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
    mean = {
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
    return mean, per


def direction_geometry(queries, spectrum):
    q_occ = queries["B0_OccupancyDirection"]
    q_pc1 = queries["E2_NormalSpatialPC1"]
    rows = []
    for name, q in queries.items():
        base = {"variant": name, "query_norm": float(np.linalg.norm(q))}
        if name.startswith("E1_RandomWeighting_seed"):
            seed = int(name.rsplit("seed", 1)[1])
            base["seed"] = seed
            base["cos_to_q_occ"] = float(np.dot(q, q_occ))
            base["cos_to_q_pc1"] = float(np.dot(q, q_pc1))
        rows.append(base)
    rows.append({"pair": "B0_OccupancyDirection_vs_E2_NormalSpatialPC1", "cosine": float(np.dot(q_occ, q_pc1)), "l2_distance": float(np.linalg.norm(q_occ - q_pc1))})
    rand_cos_occ = [r["cos_to_q_occ"] for r in rows if "cos_to_q_occ" in r]
    rand_cos_pc1 = [r["cos_to_q_pc1"] for r in rows if "cos_to_q_pc1" in r]
    rows.append({
        "variant": "E1_RandomWeighting_summary",
        "random_count": len(rand_cos_occ),
        "cos_to_q_occ_mean": float(np.mean(rand_cos_occ)),
        "cos_to_q_occ_std": float(np.std(rand_cos_occ)),
        "cos_to_q_occ_min": float(np.min(rand_cos_occ)),
        "cos_to_q_occ_max": float(np.max(rand_cos_occ)),
        "cos_to_q_pc1_mean": float(np.mean(rand_cos_pc1)),
        "cos_to_q_pc1_std": float(np.std(rand_cos_pc1)),
        "cos_to_q_pc1_min": float(np.min(rand_cos_pc1)),
        "cos_to_q_pc1_max": float(np.max(rand_cos_pc1)),
        "fraction_cos_pc1_ge_0_9": float(np.mean(np.asarray(rand_cos_pc1) >= 0.9)),
    })
    rows.append({
        "variant": "E2_NormalSpatialPC1",
        "PC1_explained_variance_ratio": float(spectrum[0]["explained_variance_ratio"]),
    })
    return rows


def random_summary(raw):
    rows = []
    random_rows = [r for r in raw if r["variant"].startswith("E1_RandomWeighting_seed")]
    by_dataset = defaultdict(list)
    for r in random_rows:
        by_dataset[r["dataset"]].append(r)
    for ds, vals in sorted(by_dataset.items()):
        g = np.asarray([float(r["G_pixel"]) for r in vals], dtype=np.float64)
        pa = np.asarray([float(r["P-AUROC"]) for r in vals], dtype=np.float64)
        pap = np.asarray([float(r["P-AP"]) for r in vals], dtype=np.float64)
        rows.append({
            "dataset": ds,
            "variant": "E1_RandomWeighting_summary",
            "seeds": len(vals),
            "G_mean": float(np.mean(g)),
            "G_std": float(np.std(g)),
            "G_min": float(np.min(g)),
            "G_max": float(np.max(g)),
            "P-AUROC_mean": float(np.mean(pa)),
            "P-AP_mean": float(np.mean(pap)),
        })
    all_g = np.asarray([float(r["G_pixel"]) for r in random_rows], dtype=np.float64)
    rows.append({
        "dataset": "ALL_TARGET_ROWS",
        "variant": "E1_RandomWeighting_summary",
        "seeds": len(random_rows),
        "G_mean": float(np.mean(all_g)),
        "G_std": float(np.std(all_g)),
        "G_min": float(np.min(all_g)),
        "G_max": float(np.max(all_g)),
    })
    return random_rows + rows


def decide(raw, geom_rows):
    nonrand = [r for r in raw if r["variant"] in {"B0_OccupancyDirection", "E2_NormalSpatialPC1"}]
    by = {(r["dataset"], r["variant"]): r for r in nonrand}
    datasets = sorted({r["dataset"] for r in raw})
    b0_mean = float(np.mean([float(by[(d, "B0_OccupancyDirection")]["G_pixel"]) for d in datasets]))
    pc1_mean = float(np.mean([float(by[(d, "E2_NormalSpatialPC1")]["G_pixel"]) for d in datasets]))
    rand_rows = [r for r in raw if r["variant"].startswith("E1_RandomWeighting_seed")]
    rand_by_ds = defaultdict(list)
    for r in rand_rows:
        rand_by_ds[r["dataset"]].append(float(r["G_pixel"]))
    rand_mean_by_ds = {d: float(np.mean(v)) for d, v in rand_by_ds.items()}
    rand_mean = float(np.mean([rand_mean_by_ds[d] for d in datasets]))
    cos_occ_pc1 = next(float(r["cosine"]) for r in geom_rows if r.get("pair") == "B0_OccupancyDirection_vs_E2_NormalSpatialPC1")
    rand_summary_row = next(r for r in geom_rows if r.get("variant") == "E1_RandomWeighting_summary")
    majority_rand_pc1 = float(rand_summary_row["fraction_cos_pc1_ge_0_9"]) > 0.5
    pc1_close = abs(pc1_mean - b0_mean) <= 1.0
    rand_close = abs(rand_mean - b0_mean) <= 1.0
    if cos_occ_pc1 >= 0.9 and pc1_close and rand_close and majority_rand_pc1:
        outcome = "NORMAL_SPATIAL_LOW_RANK_MODE_SUFFICIENT"
    elif cos_occ_pc1 >= 0.9 and pc1_close and (not rand_close or not majority_rand_pc1):
        outcome = "LOW_RANK_MODE_PRESENT_BUT_WEIGHTING_SIGNIFICANT"
    else:
        outcome = "NORMAL_SPATIAL_LOW_RANK_EXPLANATION_REJECTED"
    return outcome, {
        "B0_mean_G": b0_mean,
        "PC1_mean_G": pc1_mean,
        "RandomPermutation_mean_G": rand_mean,
        "PC1_minus_B0_mean_G": pc1_mean - b0_mean,
        "Random_minus_B0_mean_G": rand_mean - b0_mean,
        "cos_q_occ_q_pc1": cos_occ_pc1,
        "random_fraction_cos_pc1_ge_0_9": float(rand_summary_row["fraction_cos_pc1_ge_0_9"]),
    }


def write_report(out, outcome, gate, raw, rand_summary_rows, geom_rows, spectrum):
    lines = [
        "# Normal Spatial Low-Rank Mode Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## Protocol",
        "",
        "No training. Full VisA source normal spatial feature map and empirical occupancy map only; target datasets are used only for final evaluation. No target statistics, Memory, prompt, adapter, or DINO.",
        "",
        "## Fixed Directions",
        "",
        "| Dataset | Variant | P-AUROC | P-AP | G |",
        "|---|---|---:|---:|---:|",
    ]
    for r in raw:
        if r["variant"].startswith("E1_RandomWeighting_seed"):
            continue
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} |")
    lines.extend(["", "## Random Null Summary", "", "| Dataset | Mean G | Std | Min | Max |", "|---|---:|---:|---:|---:|"])
    for r in rand_summary_rows:
        if r.get("variant") == "E1_RandomWeighting_summary":
            lines.append(f"| {r['dataset']} | {float(r['G_mean']):.4f} | {float(r['G_std']):.4f} | {float(r['G_min']):.4f} | {float(r['G_max']):.4f} |")
    lines.extend(["", "## Geometry", ""])
    for r in geom_rows:
        if "pair" in r:
            lines.append(f"- {r['pair']}: cosine={float(r['cosine']):.4f}, L2={float(r['l2_distance']):.4f}.")
        elif r.get("variant") == "E1_RandomWeighting_summary":
            lines.append(f"- Random to PC1: mean={float(r['cos_to_q_pc1_mean']):.4f}, std={float(r['cos_to_q_pc1_std']):.4f}, fraction>=0.9={float(r['fraction_cos_pc1_ge_0_9']):.4f}.")
        elif r.get("variant") == "E2_NormalSpatialPC1" and "PC1_explained_variance_ratio" in r:
            lines.append(f"- PC1 explained variance ratio={float(r['PC1_explained_variance_ratio']):.6f}.")
    lines.extend([
        "",
        "## Gate",
        "",
        json.dumps(gate, indent=2, default=json_default),
        "",
        "Top singular values are saved in `SINGULAR_SPECTRUM.csv`; every random seed result is saved in `RANDOM_NULL_METRICS.csv`.",
        "",
        "STOP. No method or Memory was designed.",
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
    print(f"[LowRankMode] start smoke={args.smoke} device={device}", flush=True)
    patch_proj = load_patch_proj(device)
    _dataset, _indices, source_patch, records = union_source_records(args.smoke)
    cache_out = SPATIAL_OUT if not args.smoke else out
    z_cache = projected_source_cache(cache_out, records, source_patch, patch_proj, device)
    fmap, normal_count = normal_global_spatial_mean(z_cache, records)
    occ, mask_count = occupancy_map(records)
    queries, source_rows, spectrum = build_directions(fmap, occ)
    for name, q in queries.items():
        np.save(out / f"{name}.npy", q)
    raw, percat = evaluate_all_multi(queries, patch_proj, device, args.smoke)
    geom_rows = direction_geometry(queries, spectrum)
    random_rows = random_summary(raw)
    outcome, gate = decide(raw, geom_rows)
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "RANDOM_NULL_METRICS.csv", random_rows)
    write_csv(out / "DIRECTION_GEOMETRY.csv", geom_rows + source_rows)
    write_csv(out / "SINGULAR_SPECTRUM.csv", spectrum)
    write_json(out / "OUTCOME.json", {
        "phase": "normal_spatial_low_rank_mode_audit",
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "protocol": "strict_source_only_cross_dataset_zsad",
        "variants": ["B0_OccupancyDirection", "E1_RandomWeighting_seed0_to_19", "E2_NormalSpatialPC1"],
        "outcome": outcome,
        "gate": gate,
        "smoke": args.smoke,
        "runtime_sec": time.time() - start,
        "checkpoint": str(CHECKPOINT),
        "seed": SEED,
        "random_seeds": RANDOM_SEEDS,
        "normal_images_used": normal_count,
        "anomaly_masks_used": mask_count,
        "training_used": False,
        "target_information_used_for_training": False,
        "memory_used": False,
        "prompt_used": False,
        "adapter_used": False,
        "dino_used": False,
    })
    write_report(out, outcome, gate, raw, random_rows, geom_rows, spectrum)
    print(f"[LowRankMode] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
