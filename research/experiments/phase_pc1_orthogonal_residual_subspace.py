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
from experiments.phase_xxxviii_visual_state_representation import SIDE

OUT = Path("results/phase_pc1_orthogonal_residual_subspace/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
SUFF_OUT = Path("results/phase_pc1_orthogonal_anomaly_sufficiency/formal")


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


def l2norm(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def build_residual_subspace(dataset, indices, patch, labels_by_pos, q_pc1, source_positions, variant):
    residuals = []
    bg_sum = np.zeros((1024,), dtype=np.float64)
    bg_count = 0
    pos_count = 0
    img_count = 0
    zero = np.zeros((SIDE * SIDE,), dtype=np.float32)
    for pos in source_positions:
        item = dataset.data_all[int(indices[int(pos)])]
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
        elif int(item["anomaly"]) == 0:
            bg_sum += z.sum(axis=0, dtype=np.float64)
            bg_count += int(z.shape[0])
    if not residuals:
        raise RuntimeError(f"{variant}: no residuals collected.")
    r = np.concatenate(residuals, axis=0).astype(np.float32)
    print(f"[ResidualSubspace] {variant} residual matrix {r.shape}", flush=True)
    _u, s, vt = np.linalg.svd(r.astype(np.float64), full_matrices=False)
    basis = vt[:4].astype(np.float32)
    basis = l2norm(basis)
    total = float(np.sum(s * s))
    spectrum = []
    for i, sv in enumerate(s[:32]):
        spectrum.append(
            {
                "variant": variant,
                "rank": i + 1,
                "singular_value": float(sv),
                "explained_variance_ratio": float((sv * sv) / max(total, 1e-12)),
                "cumulative_explained_variance_ratio": float(np.sum(s[: i + 1] * s[: i + 1]) / max(total, 1e-12)),
            }
        )
    mu_bg = (bg_sum / max(bg_count, 1)).astype(np.float32)
    stats = {
        "variant": variant,
        "residual_count": int(r.shape[0]),
        "source_anomaly_images_with_positive_mask": int(img_count),
        "background_patch_count": int(bg_count),
        "top4_explained_variance": float(np.sum(s[:4] * s[:4]) / max(total, 1e-12)),
        "top1_explained_variance": float((s[0] * s[0]) / max(total, 1e-12)),
        "residual_frobenius_norm": float(np.sqrt(total)),
        "mu_background_norm": float(np.linalg.norm(mu_bg)),
        "positive_patch_count": int(pos_count),
    }
    return basis, mu_bg, spectrum, stats


def principal_angles_rows(basis_true, basis_corrupt):
    m = basis_true @ basis_corrupt.T
    _u, s, _vt = np.linalg.svd(m, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.degrees(np.arccos(s))
    rows = []
    for i, (sv, angle) in enumerate(zip(s, angles)):
        rows.append({"metric": "principal_angle", "index": i + 1, "cosine": float(sv), "angle_degrees": float(angle)})
    rows.append({"metric": "mean_principal_angle", "index": 0, "cosine": float(np.mean(s)), "angle_degrees": float(np.mean(angles))})
    rows.append({"metric": "max_principal_angle", "index": 0, "cosine": float(np.min(s)), "angle_degrees": float(np.max(angles))})
    return rows


def score_subspace_on_target(basis, mu_bg, q_pc1, patch, smoke, batch_images=16):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    basis_t = np.asarray(basis, dtype=np.float32).T
    mu = np.asarray(mu_bg, dtype=np.float32)
    for start in range(0, n, batch_images):
        end = min(n, start + batch_images)
        z = remove_pc1(np.asarray(patch[start:end], dtype=np.float32), q_pc1).reshape(end - start, SIDE * SIDE, 1024)
        x = z - mu[None, None, :]
        proj = x @ basis_t
        scores = np.linalg.norm(proj, axis=-1).astype(np.float32)
        for bi in range(scores.shape[0]):
            m = scores[bi].reshape(SIDE, SIDE)
            maps[start + bi] = m.astype(np.float16)
            image[start + bi] = top1_image_score(m)
    return image, maps


def evaluate_subspaces(variants, q_pc1, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for name, payload in variants.items():
            print(f"[ResidualSubspace] eval {name} on {cell_id}", flush=True)
            image, maps = score_subspace_on_target(payload["basis"], payload["mu_bg"], q_pc1, patch, smoke)
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
        "B0_StateAligned-PC1Removed": "B0_StateAligned",
        "E1_TrueMaskMeanDifference-PC1Removed": "E1_Rank1MeanDiff",
    }
    with (SUFF_OUT / "RAW_METRICS.csv").open("r", encoding="utf-8", newline="") as fp:
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
    with (SUFF_OUT / "PER_CATEGORY.csv").open("r", encoding="utf-8", newline="") as fp:
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
        b0 = by[(ds, "B0_StateAligned")]
        e1 = by[(ds, "E1_Rank1MeanDiff")]
        e2 = by[(ds, "E2_TrueMaskRank4ResidualSubspace")]
        e3 = by[(ds, "E3_CorruptedMaskRank4ResidualSubspace")]
        rows.append(
            {
                "dataset": ds,
                "B0_G": float(b0["G_pixel"]),
                "E1_G": float(e1["G_pixel"]),
                "E2_G": float(e2["G_pixel"]),
                "E3_G": float(e3["G_pixel"]),
                "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
                "E2_minus_E1_G": float(e2["G_pixel"]) - float(e1["G_pixel"]),
                "E2_minus_E3_G": float(e2["G_pixel"]) - float(e3["G_pixel"]),
                "E2_minus_B0_P-AUROC": float(e2["P-AUROC"]) - float(b0["P-AUROC"]),
                "E2_minus_B0_P-AP": float(e2["P-AP"]) - float(b0["P-AP"]),
                "E2_minus_E3_P-AUROC": float(e2["P-AUROC"]) - float(e3["P-AUROC"]),
                "E2_minus_E3_P-AP": float(e2["P-AP"]) - float(e3["P-AP"]),
            }
        )
    return rows


def decide(summary_rows):
    e2_b0 = np.asarray([r["E2_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e2_e3 = np.asarray([r["E2_minus_E3_G"] for r in summary_rows], dtype=np.float32)
    mean_e2_b0 = float(np.mean(e2_b0))
    mean_e2_e3 = float(np.mean(e2_e3))
    if mean_e2_e3 >= 3.0 and mean_e2_b0 >= -3.0:
        outcome = "LOW_RANK_ANOMALY_GEOMETRY_SUFFICIENT"
    elif mean_e2_e3 >= 3.0 and mean_e2_b0 < -3.0:
        outcome = "LOW_RANK_GEOMETRY_INSUFFICIENT_FOR_STATEALIGNED"
    elif abs(mean_e2_e3) < 1.0:
        outcome = "RESIDUAL_SUBSPACE_NOT_GENUINE"
    else:
        outcome = "RESIDUAL_SUBSPACE_AUDIT_INCONCLUSIVE"
    return outcome, {
        "mean_E2_minus_B0_G": mean_e2_b0,
        "mean_E2_minus_E1_G": float(np.mean([r["E2_minus_E1_G"] for r in summary_rows])),
        "mean_E2_minus_E3_G": mean_e2_e3,
    }


def write_report(summary_rows, subspace_stats, angle_rows, outcome, stats, runtime_sec):
    lines = [
        "# PC1-Orthogonal Residual Subspace Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source only. All source and target patch features are projected orthogonal to the fixed source-normal PC1 before residual construction and scoring.",
        "",
        "| Dataset | B0 StateAligned G | E1 Rank1 G | E2 True Rank4 G | E3 Corrupt Rank4 G | E2-B0 | E2-E1 | E2-E3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E3_G']:.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E2_minus_E1_G']:+.4f} | {r['E2_minus_E3_G']:+.4f} |"
        )
    stats_by_variant = {r["variant"]: r for r in subspace_stats}
    lines.extend(
        [
            "",
            f"Mean E2-B0 G: {stats['mean_E2_minus_B0_G']:+.4f} pp.",
            f"Mean E2-E1 G: {stats['mean_E2_minus_E1_G']:+.4f} pp.",
            f"Mean E2-E3 G: {stats['mean_E2_minus_E3_G']:+.4f} pp.",
            "",
            f"E2 top4 explained variance: {stats_by_variant['E2_TrueMaskRank4ResidualSubspace']['top4_explained_variance']:.6f}",
            f"E3 top4 explained variance: {stats_by_variant['E3_CorruptedMaskRank4ResidualSubspace']['top4_explained_variance']:.6f}",
            f"Mean principal angle true/corrupt: {[r for r in angle_rows if r['metric']=='mean_principal_angle'][0]['angle_degrees']:.4f} degrees",
            "",
            f"Runtime: {runtime_sec:.1f}s.",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    _dataset_name, dataset, indices, _cls, patch, _y, _cats = load_full_visa_source_lazy(False)
    true_by_pos, corrupt_by_pos, audit = source_label_maps(args.smoke)
    source_positions = sorted(true_by_pos) if args.smoke else list(range(len(indices)))

    b_true, mu_true, spec_true, stats_true = build_residual_subspace(
        dataset, indices, patch, true_by_pos, q_pc1, source_positions, "E2_TrueMaskRank4ResidualSubspace"
    )
    b_corrupt, mu_corrupt, spec_corrupt, stats_corrupt = build_residual_subspace(
        dataset, indices, patch, corrupt_by_pos, q_pc1, source_positions, "E3_CorruptedMaskRank4ResidualSubspace"
    )
    angle_rows = principal_angles_rows(b_true, b_corrupt)
    raw, percat = evaluate_subspaces(
        {
            "E2_TrueMaskRank4ResidualSubspace": {"basis": b_true, "mu_bg": mu_true},
            "E3_CorruptedMaskRank4ResidualSubspace": {"basis": b_corrupt, "mu_bg": mu_corrupt},
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
    subspace_stats = [stats_true, stats_corrupt]
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "RESIDUAL_SPECTRUM.csv", spec_true + spec_corrupt)
    write_csv(OUT / "SUBSPACE_STATS.csv", subspace_stats)
    write_csv(OUT / "PRINCIPAL_ANGLES.csv", angle_rows)
    write_csv(OUT / "MASK_CORRUPTION_AUDIT.csv", audit)
    np.save(OUT / "E2_true_rank4_basis.npy", b_true.astype(np.float32))
    np.save(OUT / "E3_corrupt_rank4_basis.npy", b_corrupt.astype(np.float32))
    runtime_sec = time.time() - start
    write_json(
        OUT / "OUTCOME.json",
        {
            "outcome": outcome,
            "decision_stats": decision_stats,
            "runtime_sec": runtime_sec,
            "strict_source_only": True,
            "target_training": False,
            "target_statistics": False,
            "memory_used": False,
            "rank_fixed": 4,
            "pc1_source": str(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy"),
        },
    )
    write_report(summary_rows, subspace_stats, angle_rows, outcome, decision_stats, runtime_sec)
    print(f"[ResidualSubspace] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
