import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cross_layer_persistent_hard_normal import MOECLIP, VISUALAD
from experiments.phase_hard_normal_rejection_state import load_b0_rows, load_hardnormal_components
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xliii_adapter_head_decomposition import SEED
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np


OUT = Path("results/phase_local_residual_hard_normal_scoring/formal")
EPS = 1e-8


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def random_neighbor_indices(seed=0):
    rng = np.random.default_rng(seed)
    idx = np.zeros((SIDE * SIDE, 8), dtype=np.int64)
    valid = np.zeros((SIDE * SIDE, 8), dtype=np.float32)
    for y in range(SIDE):
        for x in range(SIDE):
            p = y * SIDE + x
            spatial = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < SIDE and 0 <= xx < SIDE:
                        spatial.append(yy * SIDE + xx)
            pool = np.setdiff1d(np.arange(SIDE * SIDE), np.asarray([p], dtype=np.int64), assume_unique=False)
            chosen = rng.choice(pool, size=len(spatial), replace=False)
            idx[p, : len(spatial)] = chosen
            if len(spatial) < 8:
                idx[p, len(spatial) :] = chosen[0]
            valid[p, : len(spatial)] = 1.0
    return idx, valid


def local_neighbor_mean(z):
    # z: B,N,D normalized aligned features.
    b, n, d = z.shape
    fmap = z.reshape(b, SIDE, SIDE, d).permute(0, 3, 1, 2)
    summed = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1, count_include_pad=False) * local_counts(z.device).view(1, 1, SIDE, SIDE)
    counts = local_counts(z.device).view(1, 1, SIDE, SIDE) - 1.0
    neigh = (summed - fmap) / counts.clamp_min(1.0)
    return neigh.permute(0, 2, 3, 1).reshape(b, n, d)


_COUNTS = None


def local_counts(device):
    global _COUNTS
    if _COUNTS is None or _COUNTS.device != device:
        ones = torch.ones((1, 1, SIDE, SIDE), dtype=torch.float32, device=device)
        _COUNTS = F.avg_pool2d(ones, kernel_size=3, stride=1, padding=1, count_include_pad=False) * 9.0
    return _COUNTS.squeeze(0).squeeze(0)


def random_context_mean(z, rand_idx, rand_valid, device):
    idx = torch.tensor(rand_idx, dtype=torch.long, device=device)
    valid = torch.tensor(rand_valid, dtype=torch.float32, device=device)
    gathered = z[:, idx.reshape(-1), :].reshape(z.shape[0], SIDE * SIDE, 8, z.shape[-1])
    return (gathered * valid.view(1, SIDE * SIDE, 8, 1)).sum(dim=2) / valid.sum(dim=1).view(1, SIDE * SIDE, 1).clamp_min(1.0)


@torch.no_grad()
def aligned_features(raw_patch, align_model, q_pc1, device):
    arr = remove_pc1(np.asarray(raw_patch, dtype=np.float32), q_pc1)
    flat = norm_np(arr.reshape(-1, arr.shape[-1]))
    x = torch.tensor(flat, dtype=torch.float32, device=device)
    z = align_model.adapt(x)
    return z.reshape(raw_patch.shape[0], SIDE * SIDE, raw_patch.shape[-1])


@torch.no_grad()
def score_components(raw_patch, align_model, q_pc1, states_hn, v_hn, rand_idx, rand_valid, device, batch_images=8):
    n = raw_patch.shape[0]
    base = np.zeros((n, SIDE, SIDE), dtype=np.float32)
    local = np.zeros((n, SIDE, SIDE), dtype=np.float32)
    rand = np.zeros((n, SIDE, SIDE), dtype=np.float32)
    s = torch.tensor(norm_np(states_hn), dtype=torch.float32, device=device)
    v = torch.tensor(v_hn / max(float(np.linalg.norm(v_hn)), EPS), dtype=torch.float32, device=device)
    for start in range(0, n, batch_images):
        patch = np.asarray(raw_patch[start : start + batch_images], dtype=np.float32)
        z = aligned_features(patch, align_model, q_pc1, device)
        logits = z @ s.T
        bscore = torch.softmax(logits, dim=-1)[:, :, 1]
        n_local = local_neighbor_mean(z)
        r_local = F.normalize(z - n_local, dim=-1)
        n_rand = random_context_mean(z, rand_idx, rand_valid, device)
        r_rand = F.normalize(z - n_rand, dim=-1)
        b = z.shape[0]
        base[start : start + b] = bscore.detach().cpu().numpy().reshape(b, SIDE, SIDE)
        local[start : start + b] = (r_local @ v).detach().cpu().numpy().reshape(b, SIDE, SIDE)
        rand[start : start + b] = (r_rand @ v).detach().cpu().numpy().reshape(b, SIDE, SIDE)
    return base, local, rand


def source_calibration(smoke, align_model, q_pc1, states_hn, v_hn, rand_idx, rand_valid, device):
    suffix = "smoke" if smoke else "formal"
    path = OUT / f"SOURCE_CALIBRATION_{suffix}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print("[LocalResidual] source-only calibration", flush=True)
    _name, _dataset, _indices, _cls, patch, _y, _cat = load_full_visa_source_lazy(False)
    n = min(96, patch.shape[0]) if smoke else patch.shape[0]
    sums = {"base": 0.0, "local": 0.0, "random": 0.0}
    sums2 = {"base": 0.0, "local": 0.0, "random": 0.0}
    count = 0
    start_time = time.time()
    for start in range(0, n, 16):
        arr = np.asarray(patch[start : min(start + 16, n)], dtype=np.float32)
        b, l, r = score_components(arr, align_model, q_pc1, states_hn, v_hn, rand_idx, rand_valid, device, batch_images=8)
        for name, vals in (("base", b), ("local", l), ("random", r)):
            x = vals.reshape(-1).astype(np.float64)
            sums[name] += float(x.sum())
            sums2[name] += float((x * x).sum())
        count += b.size
        if (start // 16 + 1) % 25 == 0:
            print(f"[LocalResidual] calibration images {min(start+16,n)}/{n}", flush=True)
    rows = []
    payload = {"target_information_used": False, "source_images": int(n), "patch_count": int(count), "seconds": time.time() - start_time, "rows": rows}
    for name in ("base", "local", "random"):
        mean = sums[name] / max(count, 1)
        var = max(sums2[name] / max(count, 1) - mean * mean, 0.0)
        rows.append({"score": name, "mean": float(mean), "std": float(np.sqrt(var) + EPS)})
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "SOURCE_CALIBRATION.csv", rows)
    return payload


def cal_params(calibration):
    return {r["score"]: (float(r["mean"]), float(r["std"])) for r in calibration["rows"]}


def zscore(x, params, name):
    mu, sd = params[name]
    return (x - mu) / max(sd, EPS)


def patch_mask(item):
    mask = item["img_mask"].float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    small = F.interpolate(mask, size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape(-1)
    return small.numpy() > 0.5


def eval_target_scores(smoke, align_model, q_pc1, states_hn, v_hn, rand_idx, rand_valid, calibration, device):
    params = cal_params(calibration)
    raw, percat, corr_rows, dist_rows = [], [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        if cell_id not in ("MVTec", "BTAD", "CVC-ClinicDB"):
            continue
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        print(f"[LocalResidual] score {cell_id} n={n}", flush=True)
        base = np.zeros((n, SIDE, SIDE), dtype=np.float32)
        local = np.zeros((n, SIDE, SIDE), dtype=np.float32)
        rand = np.zeros((n, SIDE, SIDE), dtype=np.float32)
        chunk = 16
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            arr = np.stack([np.asarray(patch[i], dtype=np.float32) for i in range(start, end)], axis=0)
            b, l, r = score_components(arr, align_model, q_pc1, states_hn, v_hn, rand_idx, rand_valid, device)
            base[start:end] = b
            local[start:end] = l
            rand[start:end] = r
            if (start // chunk + 1) % 25 == 0:
                print(f"[LocalResidual] {cell_id} scored {end}/{n}", flush=True)
        maps = {
            "E1_LocalResidualScorer": 0.5 * (zscore(base, params, "base") + zscore(local, params, "local")),
            "E2_RandomContextControl": 0.5 * (zscore(base, params, "base") + zscore(rand, params, "random")),
        }
        idx_use = idx_or_rows[:n]
        for variant, m in maps.items():
            image = np.asarray([top1_image_score(x) for x in m], dtype=np.float32)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, m.astype(np.float32))
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
        for name, res in (("local", local), ("random", rand)):
            flat_b = base.reshape(-1)
            flat_r = res.reshape(-1)
            if flat_b.size > 200000:
                ids = np.linspace(0, flat_b.size - 1, 200000).astype(np.int64)
                fb, fr = flat_b[ids], flat_r[ids]
            else:
                fb, fr = flat_b, flat_r
            corr_rows.append({"dataset": cell_id, "residual": name, "pearson_base_residual": float(pearsonr(fb, fr).statistic), "spearman_base_residual": float(spearmanr(fb, fr).statistic), "residual_norm_mean_proxy_abs_score": float(np.mean(np.abs(flat_r)))})
        for name, res in (("local", local), ("random", rand)):
            pos_vals, neg_vals = [], []
            for pos in range(n):
                item = dataset[int(idx_use[pos])] if kind == "source_dataset" else dataset[pos]
                gt = patch_mask(item)
                vals = res[pos].reshape(-1)
                if gt.any():
                    pos_vals.append(vals[gt])
                neg_vals.append(vals[~gt])
            p = np.concatenate(pos_vals).astype(np.float32) if pos_vals else np.asarray([], dtype=np.float32)
            q = np.concatenate(neg_vals).astype(np.float32)
            dist_rows.append({
                "dataset": cell_id,
                "residual": name,
                "anomaly_pixel_count": int(p.size),
                "normal_pixel_count": int(q.size),
                "anomaly_residual_mean": float(p.mean()) if p.size else float("nan"),
                "normal_residual_mean": float(q.mean()),
                "anomaly_residual_p95": float(np.percentile(p, 95)) if p.size else float("nan"),
                "normal_residual_p95": float(np.percentile(q, 95)),
                "mean_gap": float(p.mean() - q.mean()) if p.size else float("nan"),
            })
    return raw, percat, corr_rows, dist_rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_LocalResidualScorer")]
        e2 = by[(ds, "E2_RandomContextControl")]
        rows.append({
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
        })
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and (e1_b0[0] >= 1.5 or e1_b0[2] >= 1.5) and e1_b0[1] >= -1.0:
        outcome = "LOCAL_RESIDUAL_SCORING_SUPPORTED"
    elif e1_b0[1] > 0 and e1_b0[0] <= 0 and e1_b0[2] <= 0:
        outcome = "LOCAL_RESIDUAL_DOMAIN_DEPENDENT"
    else:
        outcome = "LOCAL_RESIDUAL_SCORING_NO_SIGNAL"
    return outcome, {"mean_E1_minus_B0_G": float(e1_b0.mean()), "mean_E1_minus_E2_G": float(e1_e2.mean()), "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0))}


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_LocalResidualScorer"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            p_auroc = float(row["P-AUROC"])
            p_ap = float(row["P-AP"])
            g = float(row["G_pixel"])
            rows.append({
                "dataset": ds,
                "current_method": "E1_LocalResidualScorer",
                "current_P-AUROC": p_auroc,
                "current_P-AP": p_ap,
                "current_G": g,
                "reference": ref_name,
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": p_auroc - ref["P-AUROC"],
                "Delta_P_AP": p_ap - ref["P-AP"],
                "Delta_G": g - ref["G_pixel"],
                "comparison_status": "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS",
            })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(summary_rows, outcome, stats, sota_rows, runtime):
    lines = [
        "# Local Residual Hard-Normal Scoring Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. Final-Layer HardNormal is reused without retraining. E1 adds fixed 3x3 local residual scoring; E2 replaces the neighborhood with same-image random context. Source-only z-score calibration is used for base/residual score fusion.",
        "",
        "| Dataset | B0 HardNormal G | E1 LocalResidual G | E2 RandomContext G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
        "## External SOTA Position",
        "",
        "MoECLIP is the primary direct reference; VisualAD(CLIP) is retained as a secondary direct reference. Comparability is limited to VisA-source pixel P-AUROC/P-AP/G with implementation caveats.",
        "",
        "| Reference | Dataset | Current G | Reference G | Delta G | Status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for r in sota_rows:
        lines.append(f"| {r['reference']} | {r['dataset']} | {float(r['current_G']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend(["", "Stop-loss: if local residual does not beat both HardNormal and random context, do not sweep windows or add learned fusion.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LocalResidual] device={device}", flush=True)
    q_pc1, _payload, align_model, states_hn, _m_c, _alpha, v_hn = load_hardnormal_components(device)
    rand_idx, rand_valid = random_neighbor_indices(SEED)
    calibration = source_calibration(args.smoke, align_model, q_pc1, states_hn, v_hn, rand_idx, rand_valid, device)
    raw_new, percat_new, corr_rows, dist_rows = eval_target_scores(args.smoke, align_model, q_pc1, states_hn, v_hn, rand_idx, rand_valid, calibration, device)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw_new
    percat = b0_percat + percat_new
    summary_rows = summarize(raw)
    outcome, stats = decide(summary_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "SCORE_CORRELATION.csv", corr_rows)
    write_csv(OUT / "RESIDUAL_SCORE_DISTRIBUTIONS.csv", dist_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "hardnormal_retrained": False,
        "window": "3x3_exclude_self",
        "fusion": "equal_weight_source_zscore_base_and_residual",
    })
    write_report(summary_rows, outcome, stats, sota_rows, runtime)
    print(f"[LocalResidual] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
