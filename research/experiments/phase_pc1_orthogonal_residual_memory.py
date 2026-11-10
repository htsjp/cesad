import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, fps_indices, norm_np

OUT = Path("results/phase_pc1_orthogonal_residual_memory/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
SUBSPACE_OUT = Path("results/phase_pc1_orthogonal_residual_subspace/formal")
K = 64


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


def collect_residuals(dataset, indices, patch, labels_by_pos, q_pc1, source_positions, variant):
    residuals = []
    bg_sum = np.zeros((1024,), dtype=np.float64)
    bg_count = 0
    zero = np.zeros((SIDE * SIDE,), dtype=np.float32)
    pos_count = 0
    img_count = 0
    for pos in source_positions:
        z = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1).reshape(SIDE * SIDE, 1024)
        y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=bool)
        bg = z[~y]
        if bg.size:
            mu_out = bg.mean(axis=0, dtype=np.float64).astype(np.float32)
            bg_sum += bg.sum(axis=0, dtype=np.float64)
            bg_count += int(bg.shape[0])
        else:
            mu_out = z.mean(axis=0, dtype=np.float64).astype(np.float32)
        if y.any():
            residuals.append((z[y] - mu_out[None, :]).astype(np.float32))
            pos_count += int(y.sum())
            img_count += 1
        elif int(dataset.data_all[int(indices[int(pos)])]["anomaly"]) == 0:
            bg_sum += z.sum(axis=0, dtype=np.float64)
            bg_count += int(z.shape[0])
    if not residuals:
        raise RuntimeError(f"{variant}: no residuals collected.")
    r = np.concatenate(residuals, axis=0).astype(np.float32)
    r_norm = norm_np(r)
    mu_bg = (bg_sum / max(bg_count, 1)).astype(np.float32)
    print(f"[ResidualMemory] {variant} residuals {r_norm.shape}", flush=True)
    return r_norm, mu_bg, {
        "variant": variant,
        "residual_count": int(r_norm.shape[0]),
        "positive_patch_count": int(pos_count),
        "source_anomaly_images_with_positive_mask": int(img_count),
        "background_patch_count": int(bg_count),
        "mu_background_norm": float(np.linalg.norm(mu_bg)),
    }


def build_memory(residuals, variant, k=K):
    idx = fps_indices(residuals, min(k, residuals.shape[0]))
    mem = norm_np(residuals[idx]).astype(np.float32)
    sim = residuals @ mem.T
    best = sim.max(axis=1)
    pair = mem @ mem.T
    off = pair[~np.eye(pair.shape[0], dtype=bool)]
    rows = [
        {"variant": variant, "stat": "memory_size", "value": int(mem.shape[0])},
        {"variant": variant, "stat": "coverage_mean_max_cosine", "value": float(best.mean())},
        {"variant": variant, "stat": "coverage_std_max_cosine", "value": float(best.std())},
        {"variant": variant, "stat": "coverage_min_max_cosine", "value": float(best.min())},
        {"variant": variant, "stat": "coverage_p05_max_cosine", "value": float(np.percentile(best, 5))},
        {"variant": variant, "stat": "pairwise_cosine_mean", "value": float(off.mean())},
        {"variant": variant, "stat": "pairwise_cosine_std", "value": float(off.std())},
        {"variant": variant, "stat": "pairwise_cosine_min", "value": float(off.min())},
        {"variant": variant, "stat": "pairwise_cosine_max", "value": float(off.max())},
    ]
    return mem, rows


def cross_similarity_rows(mem_true, mem_corrupt):
    sims = mem_true @ mem_corrupt.T
    return [
        {"variant": "true_vs_corrupt", "stat": "cross_similarity_mean", "value": float(sims.mean())},
        {"variant": "true_vs_corrupt", "stat": "cross_similarity_std", "value": float(sims.std())},
        {"variant": "true_vs_corrupt", "stat": "true_to_corrupt_mean_max", "value": float(sims.max(axis=1).mean())},
        {"variant": "true_vs_corrupt", "stat": "corrupt_to_true_mean_max", "value": float(sims.max(axis=0).mean())},
        {"variant": "true_vs_corrupt", "stat": "cross_similarity_max", "value": float(sims.max())},
        {"variant": "true_vs_corrupt", "stat": "cross_similarity_min", "value": float(sims.min())},
    ]


def flatten_base_stats(row):
    variant = row["variant"]
    return [{"variant": variant, "stat": k, "value": v} for k, v in row.items() if k != "variant"]


def score_memory_on_target(memory, mu_bg, q_pc1, patch, smoke, batch_images=16):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    mem = np.asarray(memory, dtype=np.float32).T
    mu = np.asarray(mu_bg, dtype=np.float32)
    for start in range(0, n, batch_images):
        end = min(n, start + batch_images)
        z = remove_pc1(np.asarray(patch[start:end], dtype=np.float32), q_pc1).reshape(end - start, SIDE * SIDE, 1024)
        x = norm_np(z - mu[None, None, :])
        scores = (x @ mem).max(axis=-1).astype(np.float32)
        for bi in range(scores.shape[0]):
            m = scores[bi].reshape(SIDE, SIDE)
            maps[start + bi] = m.astype(np.float16)
            image[start + bi] = top1_image_score(m)
    return image, maps


def evaluate_memories(variants, q_pc1, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for name, payload in variants.items():
            print(f"[ResidualMemory] eval {name} on {cell_id}", flush=True)
            image, maps = score_memory_on_target(payload["memory"], payload["mu_bg"], q_pc1, patch, smoke)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": cell_id, "variant": name, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": name, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat


def load_reuse_rows():
    raw, percat = [], []
    wanted = {
        "B0_StateAligned": "B0_StateAligned-PC1Removed",
        "E2_TrueMaskRank4ResidualSubspace": "B1_TrueRank4",
    }
    with (SUBSPACE_OUT / "RAW_METRICS.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] not in wanted:
                continue
            out = dict(row)
            out["variant"] = wanted[row["variant"]]
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            raw.append(out)
    with (SUBSPACE_OUT / "PER_CATEGORY.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] not in wanted:
                continue
            out = dict(row)
            out["variant"] = wanted[row["variant"]]
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            percat.append(out)
    return raw, percat


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_StateAligned-PC1Removed")]
        b1 = by[(ds, "B1_TrueRank4")]
        e1 = by[(ds, "E1_TrueResidualMemory")]
        e2 = by[(ds, "E2_CorruptedResidualMemory")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "B1_G": float(b1["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_B1_G": float(e1["G_pixel"]) - float(b1["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
            "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_b1 = np.asarray([r["E1_minus_B1_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    mean_b0 = float(e1_b0.mean())
    mean_b1 = float(e1_b1.mean())
    mean_e2 = float(e1_e2.mean())
    if mean_e2 >= 3.0 and mean_b0 >= -3.0:
        outcome = "DISCRETE_RESIDUAL_MEMORY_SUFFICIENT"
    elif mean_e2 >= 3.0 and mean_b1 >= 3.0 and mean_b0 < -3.0:
        outcome = "MEMORY_CAPTURES_ADDITIONAL_GENUINE_STRUCTURE"
    elif mean_e2 >= 3.0 and mean_b1 < 3.0:
        outcome = "STATIC_MEMORY_NOT_BETTER_THAN_SUBSPACE"
    elif abs(mean_e2) < 1.0:
        outcome = "RESIDUAL_MEMORY_NOT_GENUINE"
    else:
        outcome = "RESIDUAL_MEMORY_AUDIT_INCONCLUSIVE"
    return outcome, {"mean_E1_minus_B0_G": mean_b0, "mean_E1_minus_B1_G": mean_b1, "mean_E1_minus_E2_G": mean_e2}


def write_report(summary_rows, memory_stats, outcome, decision_stats, runtime_sec):
    stats = {(r["variant"], r["stat"]): r["value"] for r in memory_stats}
    lines = [
        "# PC1-Orthogonal Residual Memory Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source only. All source and target patch features are projected orthogonal to the fixed source-normal PC1. Residual memory uses FPS with fixed K=64.",
        "",
        "| Dataset | B0 StateAligned G | B1 Rank4 G | E1 True Memory G | E2 Corrupt Memory G | E1-B0 | E1-B1 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['B1_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_B1_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {decision_stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-B1 G: {decision_stats['mean_E1_minus_B1_G']:+.4f} pp.",
        f"Mean E1-E2 G: {decision_stats['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
        f"True memory coverage mean max cosine: {stats[('E1_TrueResidualMemory', 'coverage_mean_max_cosine')]:.6f}",
        f"Corrupt memory coverage mean max cosine: {stats[('E2_CorruptedResidualMemory', 'coverage_mean_max_cosine')]:.6f}",
        f"True/corrupt memory cross mean max: {stats[('true_vs_corrupt', 'true_to_corrupt_mean_max')]:.6f}",
        "",
        f"Runtime: {runtime_sec:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    _dataset_name, dataset, indices, _cls, patch, _y, _cats = load_full_visa_source_lazy(False)
    true_by_pos, corrupt_by_pos, audit = source_label_maps(args.smoke)
    source_positions = sorted(true_by_pos) if args.smoke else list(range(len(indices)))

    r_true, mu_true, stats_true = collect_residuals(dataset, indices, patch, true_by_pos, q_pc1, source_positions, "E1_TrueResidualMemory")
    r_corrupt, mu_corrupt, stats_corrupt = collect_residuals(dataset, indices, patch, corrupt_by_pos, q_pc1, source_positions, "E2_CorruptedResidualMemory")
    mem_true, mem_stats_true = build_memory(r_true, "E1_TrueResidualMemory")
    mem_corrupt, mem_stats_corrupt = build_memory(r_corrupt, "E2_CorruptedResidualMemory")
    memory_stats = flatten_base_stats(stats_true) + flatten_base_stats(stats_corrupt) + mem_stats_true + mem_stats_corrupt + cross_similarity_rows(mem_true, mem_corrupt)

    raw, percat = evaluate_memories(
        {
            "E1_TrueResidualMemory": {"memory": mem_true, "mu_bg": mu_true},
            "E2_CorruptedResidualMemory": {"memory": mem_corrupt, "mu_bg": mu_corrupt},
        },
        q_pc1,
        args.smoke,
    )
    base_raw, base_percat = load_reuse_rows()
    if args.smoke:
        base_raw = base_raw[:6]
        base_percat = base_percat[:6]
    raw = base_raw + raw
    percat = base_percat + percat
    summary_rows = summarize(raw)
    outcome, decision_stats = decide(summary_rows)

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "MEMORY_STATS.csv", memory_stats)
    write_csv(OUT / "MASK_CORRUPTION_AUDIT.csv", audit)
    np.save(OUT / "E1_true_residual_memory_k64.npy", mem_true.astype(np.float32))
    np.save(OUT / "E2_corrupt_residual_memory_k64.npy", mem_corrupt.astype(np.float32))
    runtime_sec = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime_sec,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "memory_used": "source_residual_memory_only",
        "K": K,
        "pc1_source": str(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy"),
    })
    write_report(summary_rows, memory_stats, outcome, decision_stats, runtime_sec)
    print(f"[ResidualMemory] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
