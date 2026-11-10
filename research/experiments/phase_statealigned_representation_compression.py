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

from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np

OUT = Path("results/phase_statealigned_representation_compression/formal")
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


def load_pc1removed_adapter(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    states = np.asarray(payload["patch_states"], dtype=np.float32)
    model = AdapterToState(1024, states).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return model, states


@torch.no_grad()
def aligned_features(model, arr, q_pc1, device):
    z = remove_pc1(np.asarray(arr, dtype=np.float32), q_pc1).reshape(-1, 1024)
    z = norm_np(z)
    x = torch.tensor(z, dtype=torch.float32, device=device)
    out = model.adapt(x).detach().cpu().numpy().astype(np.float32)
    return out


def norm_vec(x):
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    return (x / max(n, 1e-12)).astype(np.float32), n


def aligned_mean_diff(model, patch, labels_by_pos, q_pc1, source_positions, device, tag):
    sum_a = np.zeros((1024,), dtype=np.float64)
    sum_n = np.zeros((1024,), dtype=np.float64)
    count_a = 0
    count_n = 0
    pos_fracs = []
    sep_vals = []
    zero = np.zeros((SIDE * SIDE,), dtype=np.float32)
    for npos, pos in enumerate(source_positions):
        if npos % 500 == 0:
            print(f"[RepCompression] source aligned mean {tag} {npos}/{len(source_positions)}", flush=True)
        y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=bool)
        z = aligned_features(model, patch[int(pos)], q_pc1, device)
        if y.any():
            za = z[y]
            zn = z[~y]
            sum_a += za.sum(axis=0, dtype=np.float64)
            count_a += int(za.shape[0])
            if zn.size:
                sep_vals.append(float(za.mean(axis=0) @ zn.mean(axis=0)))
        else:
            zn = z
        if zn.size:
            sum_n += zn.sum(axis=0, dtype=np.float64)
            count_n += int(zn.shape[0])
        pos_fracs.append(float(y.mean()))
    mu_a = sum_a / max(count_a, 1)
    mu_n = sum_n / max(count_n, 1)
    delta = mu_a - mu_n
    q, dnorm = norm_vec(delta)
    rows = [
        {"variant": tag, "stat": "anomaly_patch_count", "value": count_a},
        {"variant": tag, "stat": "normal_patch_count", "value": count_n},
        {"variant": tag, "stat": "mean_positive_patch_fraction", "value": float(np.mean(pos_fracs))},
        {"variant": tag, "stat": "aligned_mu_A_minus_mu_N_norm", "value": dnorm},
        {"variant": tag, "stat": "aligned_mu_A_norm", "value": float(np.linalg.norm(mu_a))},
        {"variant": tag, "stat": "aligned_mu_N_norm", "value": float(np.linalg.norm(mu_n))},
        {"variant": tag, "stat": "aligned_global_mu_A_mu_N_cosine", "value": float(np.dot(mu_a, mu_n) / max(float(np.linalg.norm(mu_a) * np.linalg.norm(mu_n)), 1e-12))},
        {"variant": tag, "stat": "aligned_per_image_anomaly_background_cosine_mean", "value": float(np.mean(sep_vals)) if sep_vals else float("nan")},
        {"variant": tag, "stat": "aligned_per_image_anomaly_background_cosine_std", "value": float(np.std(sep_vals)) if sep_vals else float("nan")},
    ]
    return q, rows


@torch.no_grad()
def score_aligned_queries(models, queries, q_pc1, patch, device, smoke, batch_images=8):
    names = list(queries)
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = {name: np.zeros((n, SIDE, SIDE), dtype=np.float16) for name in names}
    image = {name: np.zeros((n,), dtype=np.float32) for name in names}
    for start in range(0, n, batch_images):
        end = min(n, start + batch_images)
        arr = np.asarray(patch[start:end], dtype=np.float32)
        for name in names:
            model = models[name]
            q = np.asarray(queries[name], dtype=np.float32)
            z = aligned_features(model, arr, q_pc1, device).reshape(end - start, SIDE * SIDE, 1024)
            scores = (z @ q).astype(np.float32)
            for bi in range(scores.shape[0]):
                m = scores[bi].reshape(SIDE, SIDE)
                maps[name][start + bi] = m.astype(np.float16)
                image[name][start + bi] = top1_image_score(m)
    return image, maps


def evaluate_aligned(models, queries, q_pc1, device, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        print(f"[RepCompression] eval aligned mean-diff on {cell_id}", flush=True)
        image, maps = score_aligned_queries(models, queries, q_pc1, patch, device, smoke)
        for name in queries:
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image[name], maps[name])
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
        "B0_StateAligned-PC1Removed": "B0_FullStateAligned",
        "E1_TrueMaskMeanDifference-PC1Removed": "B1_PreAlignmentMeanDiff",
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
        b0 = by[(ds, "B0_FullStateAligned")]
        b1 = by[(ds, "B1_PreAlignmentMeanDiff")]
        e1 = by[(ds, "E1_TrueAlignedMeanDiff")]
        e2 = by[(ds, "E2_CorruptAlignedMeanDiff")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "B1_G": float(b1["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B1_G": float(e1["G_pixel"]) - float(b1["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B1_P-AUROC": float(e1["P-AUROC"]) - float(b1["P-AUROC"]),
            "E1_minus_B1_P-AP": float(e1["P-AP"]) - float(b1["P-AP"]),
            "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def decide(summary_rows):
    e1_b1 = np.asarray([r["E1_minus_B1_G"] for r in summary_rows], dtype=np.float32)
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    mean_b1 = float(e1_b1.mean())
    mean_b0 = float(e1_b0.mean())
    mean_e2 = float(e1_e2.mean())
    if mean_b1 >= 3.0 and mean_b0 >= -3.0 and mean_e2 >= 3.0:
        outcome = "LEARNED_ALIGNMENT_COMPRESSES_TRANSFERABLE_ANOMALY_KNOWLEDGE"
    elif mean_b1 >= 3.0 and mean_e2 >= 3.0 and mean_b0 < -3.0:
        outcome = "ALIGNMENT_HELPFUL_BUT_STATE_SCORER_STILL_REQUIRED"
    elif abs(mean_b1) < 1.0:
        outcome = "ALIGNMENT_NOT_MAIN_SOURCE_OF_STATEALIGNED_GAIN"
    else:
        outcome = "STATEALIGNED_REPRESENTATION_MECHANISM_INCONCLUSIVE"
    return outcome, {"mean_E1_minus_B1_G": mean_b1, "mean_E1_minus_B0_G": mean_b0, "mean_E1_minus_E2_G": mean_e2}


def write_report(summary_rows, geom_rows, outcome, stats, runtime_sec):
    g = {(r["variant"], r["stat"]): r["value"] for r in geom_rows}
    lines = [
        "# StateAligned Representation Compression Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source only under the PC1Removed setup. E1/E2 use only aligned patch representations and a source true-mask MeanDiff scorer; final StateAligned states/scorer are not used.",
        "",
        "| Dataset | B0 Full StateAligned G | B1 PreAlign MeanDiff G | E1 TrueAligned MeanDiff G | E2 CorruptAligned MeanDiff G | E1-B1 | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['B1_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B1_G']:+.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B1 G: {stats['mean_E1_minus_B1_G']:+.4f} pp.",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
        f"E1 aligned MeanDiff norm: {g[('E1_TrueAlignedMeanDiff', 'aligned_mu_A_minus_mu_N_norm')]:.6f}",
        f"E2 aligned MeanDiff norm: {g[('E2_CorruptAlignedMeanDiff', 'aligned_mu_A_minus_mu_N_norm')]:.6f}",
        f"E1 aligned anomaly/background cosine: {g[('E1_TrueAlignedMeanDiff', 'aligned_global_mu_A_mu_N_cosine')]:.6f}",
        f"E2 aligned anomaly/background cosine: {g[('E2_CorruptAlignedMeanDiff', 'aligned_global_mu_A_mu_N_cosine')]:.6f}",
        "",
        f"Runtime: {runtime_sec:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RepCompression] device={device}", flush=True)
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    true_model, _true_states = load_pc1removed_adapter(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_model, _corrupt_states = load_pc1removed_adapter(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    _dataset_name, _dataset, indices, _cls, patch, _y, _cats = load_full_visa_source_lazy(False)
    true_by_pos, _corrupt_by_pos, _audit = source_label_maps(args.smoke)
    source_positions = sorted(true_by_pos) if args.smoke else list(range(len(indices)))

    q_true, rows_true = aligned_mean_diff(true_model, patch, true_by_pos, q_pc1, source_positions, device, "E1_TrueAlignedMeanDiff")
    q_corrupt, rows_corrupt = aligned_mean_diff(corrupt_model, patch, true_by_pos, q_pc1, source_positions, device, "E2_CorruptAlignedMeanDiff")
    models = {"E1_TrueAlignedMeanDiff": true_model, "E2_CorruptAlignedMeanDiff": corrupt_model}
    queries = {"E1_TrueAlignedMeanDiff": q_true, "E2_CorruptAlignedMeanDiff": q_corrupt}
    raw, percat = evaluate_aligned(models, queries, q_pc1, device, args.smoke)
    base_raw, base_percat = load_reuse_rows()
    if args.smoke:
        base_raw = base_raw[:6]
        base_percat = base_percat[:6]
    raw = base_raw + raw
    percat = base_percat + percat
    summary_rows = summarize(raw)
    geom_rows = rows_true + rows_corrupt
    outcome, decision_stats = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "ALIGNED_GEOMETRY.csv", geom_rows)
    np.save(OUT / "q_true_aligned_mean_diff.npy", q_true.astype(np.float32))
    np.save(OUT / "q_corrupt_aligned_mean_diff.npy", q_corrupt.astype(np.float32))
    runtime_sec = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime_sec,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "memory_used": False,
        "final_state_scorer_used_for_E1_E2": False,
        "pc1_source": str(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy"),
        "true_adapter_checkpoint": str(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt"),
        "corrupt_adapter_checkpoint": str(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt"),
    })
    write_report(summary_rows, geom_rows, outcome, decision_stats, runtime_sec)
    print(f"[RepCompression] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
