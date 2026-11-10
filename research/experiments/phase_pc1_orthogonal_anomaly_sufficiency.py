import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_spatial_supervision_transfer_audit import (
    corrupt_masks,
    fast_patch_grid_eval,
    target_cells,
    union_source_records,
)
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE

OUT = Path("results/phase_pc1_orthogonal_anomaly_sufficiency/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")


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


def source_label_maps(smoke):
    _dataset_u, _indices_u, _patch_u, records = union_source_records(smoke)
    corrupt, audit = corrupt_masks(records)
    true_by_pos = {}
    corrupt_by_pos = {}
    for i, rec in enumerate(records):
        pos = int(rec["pos"])
        if bool(rec["normal"]):
            z = np.zeros((SIDE * SIDE,), dtype=np.float32)
            true_by_pos[pos] = z
            corrupt_by_pos[pos] = z
        else:
            true_by_pos[pos] = np.asarray(rec["mask"], dtype=np.float32)
            corrupt_by_pos[pos] = np.asarray(corrupt[i], dtype=np.float32)
    return true_by_pos, corrupt_by_pos, audit


def mean_difference_pc1_removed(patch, labels_by_pos, q_pc1, source_positions, tag):
    sum_a = np.zeros((1024,), dtype=np.float64)
    sum_n = np.zeros((1024,), dtype=np.float64)
    count_a = 0
    count_n = 0
    pos_fracs = []
    zero = np.zeros((SIDE * SIDE,), dtype=np.float32)
    for pos in source_positions:
        labels = np.asarray(labels_by_pos.get(int(pos), zero), dtype=bool)
        z = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1).reshape(SIDE * SIDE, 1024)
        if labels.any():
            sum_a += z[labels].sum(axis=0, dtype=np.float64)
            count_a += int(labels.sum())
        neg = ~labels
        if neg.any():
            sum_n += z[neg].sum(axis=0, dtype=np.float64)
            count_n += int(neg.sum())
        pos_fracs.append(float(labels.mean()))
    mu_a = sum_a / max(count_a, 1)
    mu_n = sum_n / max(count_n, 1)
    delta = mu_a - mu_n
    q, dnorm = norm_vec(delta)
    rows = [
        {"variant": tag, "stat": "anomaly_patch_count", "value": count_a},
        {"variant": tag, "stat": "normal_patch_count", "value": count_n},
        {"variant": tag, "stat": "mean_positive_patch_fraction", "value": float(np.mean(pos_fracs))},
        {"variant": tag, "stat": "raw_mu_A_minus_mu_N_norm", "value": dnorm},
        {"variant": tag, "stat": "mu_A_norm", "value": float(np.linalg.norm(mu_a))},
        {"variant": tag, "stat": "mu_N_norm", "value": float(np.linalg.norm(mu_n))},
    ]
    return q, rows


@torch.no_grad()
def score_queries_pc1_removed(queries, patch, q_pc1, smoke, batch_images=16):
    names = list(queries)
    q = np.stack([queries[n] for n in names], axis=0).astype(np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = {name: np.zeros((n, SIDE, SIDE), dtype=np.float16) for name in names}
    image = {name: np.zeros((n,), dtype=np.float32) for name in names}
    for start in range(0, n, batch_images):
        end = min(n, start + batch_images)
        z = remove_pc1(np.asarray(patch[start:end], dtype=np.float32), q_pc1).reshape(end - start, SIDE * SIDE, 1024)
        z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
        scores = (z @ q.T).astype(np.float32)
        k = max(1, int(SIDE * SIDE * 0.01))
        top = np.partition(scores, scores.shape[1] - k, axis=1)[:, -k:, :].mean(axis=1)
        smaps = scores.reshape(end - start, SIDE, SIDE, len(names))
        for j, name in enumerate(names):
            maps[name][start:end] = smaps[:, :, :, j].astype(np.float16)
            image[name][start:end] = top[:, j].astype(np.float32)
    return image, maps


def evaluate_queries(queries, q_pc1, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        print(f"[PC1Orthogonal] score mean-difference queries on {cell_id}", flush=True)
        image, maps = score_queries_pc1_removed(queries, patch, q_pc1, smoke)
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for name in queries:
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image[name], maps[name])
            metric.update({"dataset": cell_id, "variant": name, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": name, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat


def load_b0_reuse():
    raw, percat = [], []
    with (PC1_DEBIAS_OUT / "RAW_METRICS.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] != "E1_PC1Removed-TrueMask":
                continue
            out = dict(row)
            out["variant"] = "B0_StateAligned-PC1Removed"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            raw.append(out)
    with (PC1_DEBIAS_OUT / "PER_CATEGORY.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] != "E1_PC1Removed-TrueMask":
                continue
            out = dict(row)
            out["variant"] = "B0_StateAligned-PC1Removed"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            percat.append(out)
    return raw, percat


def geometry_rows(queries, source_rows):
    qt = queries["E1_TrueMaskMeanDifference-PC1Removed"]
    qc = queries["E2_CorruptedMaskMeanDifference-PC1Removed"]
    rows = [
        {"metric": "cos_q_true_q_corrupt", "value": float(np.dot(qt, qc) / max(float(np.linalg.norm(qt) * np.linalg.norm(qc)), 1e-12))},
        {"metric": "q_true_norm", "value": float(np.linalg.norm(qt))},
        {"metric": "q_corrupt_norm", "value": float(np.linalg.norm(qc))},
    ]
    for r in source_rows:
        rows.append({"variant": r["variant"], "metric": r["stat"], "value": r["value"]})
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_StateAligned-PC1Removed")]
        e1 = by[(ds, "E1_TrueMaskMeanDifference-PC1Removed")]
        e2 = by[(ds, "E2_CorruptedMaskMeanDifference-PC1Removed")]
        rows.append(
            {
                "dataset": ds,
                "B0_G": float(b0["G_pixel"]),
                "E1_G": float(e1["G_pixel"]),
                "E2_G": float(e2["G_pixel"]),
                "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
                "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
                "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
                "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
                "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
                "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
            }
        )
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    mean_e1_b0 = float(np.mean(e1_b0))
    mean_e1_e2 = float(np.mean(e1_e2))
    if mean_e1_e2 >= 3.0 and mean_e1_b0 >= -3.0:
        outcome = "FIRST_ORDER_GENUINE_ANOMALY_SIGNAL_SUFFICIENT"
    elif mean_e1_e2 >= 3.0 and mean_e1_b0 < -3.0:
        outcome = "GENUINE_ANOMALY_SIGNAL_REQUIRES_HIGHER_COMPLEXITY"
    elif abs(mean_e1_e2) < 1.0:
        outcome = "FIRST_ORDER_SIGNAL_STILL_CONFOUNDED"
    else:
        outcome = "PC1_ORTHOGONAL_SUFFICIENCY_INCONCLUSIVE"
    return outcome, {"mean_E1_minus_B0_G": mean_e1_b0, "mean_E1_minus_E2_G": mean_e1_e2}


def write_report(summary_rows, geom, outcome, stats, runtime_sec):
    lines = [
        "# PC1-Orthogonal Anomaly Sufficiency Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source only. The fixed 1024-d source-normal PC1 from the prior StateAligned PC1 debias audit is removed from all source and target patch features.",
        "",
        "| Dataset | B0 StateAligned-PC1Removed G | E1 TrueMask MeanDiff G | E2 CorruptedMask MeanDiff G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    gmap = {(r.get("variant", ""), r["metric"]): r["value"] for r in geom}
    lines.extend(
        [
            "",
            f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
            f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
            "",
            f"cos(q_true, q_corrupt): {float([r for r in geom if r['metric']=='cos_q_true_q_corrupt'][0]['value']):.6f}",
            f"raw true difference norm: {float(gmap[('E1_TrueMaskMeanDifference-PC1Removed', 'raw_mu_A_minus_mu_N_norm')]):.6f}",
            f"raw corrupted difference norm: {float(gmap[('E2_CorruptedMaskMeanDifference-PC1Removed', 'raw_mu_A_minus_mu_N_norm')]):.6f}",
            "",
            f"Runtime: {runtime_sec:.1f}s.",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("[PC1Orthogonal] load source and fixed PC1", flush=True)
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    _dataset_name, _dataset, indices, _cls, patch, _y, _cats = load_full_visa_source_lazy(False)
    true_by_pos, corrupt_by_pos, audit = source_label_maps(args.smoke)
    source_positions = sorted(true_by_pos) if args.smoke else list(range(len(indices)))

    q_true, rows_true = mean_difference_pc1_removed(patch, true_by_pos, q_pc1, source_positions, "E1_TrueMaskMeanDifference-PC1Removed")
    q_corrupt, rows_corrupt = mean_difference_pc1_removed(patch, corrupt_by_pos, q_pc1, source_positions, "E2_CorruptedMaskMeanDifference-PC1Removed")
    queries = {
        "E1_TrueMaskMeanDifference-PC1Removed": q_true,
        "E2_CorruptedMaskMeanDifference-PC1Removed": q_corrupt,
    }
    raw, percat = evaluate_queries(queries, q_pc1, args.smoke)
    b0_raw, b0_percat = load_b0_reuse()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    geom = geometry_rows(queries, rows_true + rows_corrupt)
    outcome, stats = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "QUERY_GEOMETRY.csv", geom)
    write_csv(OUT / "MASK_CORRUPTION_AUDIT.csv", audit)
    np.save(OUT / "q_true_pc1_removed.npy", q_true.astype(np.float32))
    np.save(OUT / "q_corrupt_pc1_removed.npy", q_corrupt.astype(np.float32))
    runtime_sec = time.time() - start
    write_json(
        OUT / "OUTCOME.json",
        {
            "outcome": outcome,
            "decision_stats": stats,
            "runtime_sec": runtime_sec,
            "strict_source_only": True,
            "target_training": False,
            "target_statistics": False,
            "memory_used": False,
            "pc1_source": str(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy"),
        },
    )
    write_report(summary_rows, geom, outcome, stats, runtime_sec)
    print(f"[PC1Orthogonal] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
