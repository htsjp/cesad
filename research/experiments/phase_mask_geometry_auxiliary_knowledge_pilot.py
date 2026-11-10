import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cross_foundation_stable_hardness_audit import build_clip_radio_pool
from experiments.phase_patch_occupancy_supervision_audit import mask_tensor, true_patch_occupancy
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE


OUT = Path("results/phase_mask_geometry_auxiliary_knowledge_pilot/formal")
SEED = 0
EPOCHS = 10
BATCH_SIZE = 4096
HIDDEN = 32
EPS = 1e-6


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


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def metric_row(cat, variant, y, score):
    auc = safe_auc(y, score)
    ap = safe_ap(y, score)
    return {
        "heldout_category": cat,
        "variant": variant,
        "P-AUROC": auc,
        "P-AP": ap,
        "G_pixel": 0.5 * (auc + ap) if np.isfinite(auc) and np.isfinite(ap) else float("nan"),
        "patches": int(len(y)),
        "positives": int(np.sum(np.asarray(y) > 0.5)),
    }


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def zscore(train, values):
    mu = float(np.mean(train))
    sd = float(np.std(train))
    if sd < 1e-8:
        sd = 1.0
    return (values - mu) / sd, mu, sd


def signed_distance_phi(item):
    m = mask_tensor(item).numpy().astype(bool)
    h, w = m.shape
    if not np.any(m):
        return -np.ones((SIDE * SIDE,), dtype=np.float32), "all_normal"
    inside = distance_transform_edt(m).astype(np.float32)
    outside = distance_transform_edt(~m).astype(np.float32)
    max_inside = float(inside[m].max()) if np.any(m) else 1.0
    patch_radius = max(float(h) / SIDE, float(w) / SIDE)
    phi = np.zeros((h, w), dtype=np.float32)
    phi[m] = inside[m] / max(max_inside, EPS)
    phi[~m] = -np.minimum(outside[~m] / max(patch_radius, EPS), 1.0)
    ys = np.clip(((np.arange(SIDE) + 0.5) * h / SIDE).astype(np.int64), 0, h - 1)
    xs = np.clip(((np.arange(SIDE) + 0.5) * w / SIDE).astype(np.int64), 0, w - 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return phi[yy, xx].reshape(-1).astype(np.float32), "center_sample_signed_distance"


def load_phi_q_by_position(positions, smoke=False):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / "cache" / f"visa_phi_q_by_position_v2_{suffix}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = set(int(k) for k in z["positions"].tolist())
        need = set(int(p) for p in positions)
        if need.issubset(have):
            return {int(k): {"phi": z[f"phi_{k}"], "q": z[f"q_{k}"]} for k in z["positions"].tolist() if int(k) in need}
    _name, dataset, indices, _cls, _patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    use_positions = sorted(set(int(p) for p in positions if int(p) < len(indices)))
    payload = {"positions": []}
    for n, pos in enumerate(use_positions):
        if n % 200 == 0:
            print(f"[MaskGeometry] signed distance {n}/{len(use_positions)}", flush=True)
        item = dataset[int(indices[pos])]
        phi, _method = signed_distance_phi(item)
        q, _q_method = true_patch_occupancy(item)
        payload["positions"].append(pos)
        payload[f"phi_{pos}"] = phi.astype(np.float32)
        payload[f"q_{pos}"] = q.astype(np.float32)
    payload["positions"] = np.asarray(payload["positions"], dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    return {int(k): {"phi": payload[f"phi_{int(k)}"], "q": payload[f"q_{int(k)}"]} for k in payload["positions"].tolist()}


def build_data(smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = build_clip_radio_pool(device, smoke=smoke)
    phi_q = load_phi_q_by_position(pool["position"], smoke=smoke)
    keep, phi, q = [], [], []
    for i, (pos, pid) in enumerate(zip(pool["position"], pool["patch_id"])):
        pos = int(pos)
        if pos not in phi_q:
            continue
        keep.append(i)
        phi.append(phi_q[pos]["phi"][int(pid)])
        q.append(phi_q[pos]["q"][int(pid)])
    ids = np.asarray(keep, dtype=np.int64)
    data = {
        "x": np.asarray(pool["x"][ids], dtype=np.float32),
        "y": np.asarray(pool["y"][ids], dtype=np.float32),
        "S_H": np.asarray(pool["S_C"][ids], dtype=np.float32),
        "category": np.asarray(pool["category"][ids], dtype=object),
        "position": np.asarray(pool["position"][ids], dtype=np.int64),
        "patch_id": np.asarray(pool["patch_id"][ids], dtype=np.int64),
        "phi": np.asarray(phi, dtype=np.float32),
        "q": np.asarray(q, dtype=np.float32),
    }
    return data


class AuxHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(variant, cat, x, target, mode, init_state, device, smoke):
    ckpt = OUT / "checkpoints" / f"{variant}_{cat}_lr{LR:g}.pt"
    if ckpt.exists() and not smoke:
        head = AuxHead(x.shape[1]).to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        head.load_state_dict(payload["state_dict"])
        return head, payload.get("curves", []), True
    head = AuxHead(x.shape[1]).to(device)
    head.load_state_dict(init_state)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    epochs = 2 if smoke else EPOCHS
    curves = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(x.shape[0])
        losses = []
        for start in range(0, x.shape[0], BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            xb = torch.tensor(x[ids], dtype=torch.float32, device=device)
            yb = torch.tensor(target[ids], dtype=torch.float32, device=device)
            pred = head(xb)
            if mode == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, yb)
            else:
                loss = F.smooth_l1_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"variant": variant, "heldout_category": cat, "epoch": epoch, "loss": float(np.mean(losses))}
        curves.append(row)
        print(f"[MaskGeometry] {variant}/{cat} epoch {epoch}/{epochs} loss={row['loss']:.6f}", flush=True)
    if not smoke:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "curves": curves}, ckpt)
    return head, curves, False


@torch.no_grad()
def predict(head, x, device):
    out = []
    head.eval()
    for start in range(0, x.shape[0], BATCH_SIZE):
        xb = torch.tensor(x[start : start + BATCH_SIZE], dtype=torch.float32, device=device)
        out.append(head(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def linear_r2_binary_occ(y_pred, q_pred, geo_pred):
    x = np.stack([y_pred, q_pred, np.ones_like(y_pred)], axis=1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(x, geo_pred.astype(np.float64), rcond=1e-8)
    pred = x @ coef
    ss_res = float(np.sum((geo_pred - pred) ** 2))
    ss_tot = float(np.sum((geo_pred - np.mean(geo_pred)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def conditional_auc(y, base, aux, bins=20):
    rows = []
    order = np.argsort(base)
    for b, ids in enumerate(np.array_split(order, bins)):
        rows.append({
            "bin": b,
            "count": int(ids.size),
            "S_H_min": float(np.min(base[ids])),
            "S_H_max": float(np.max(base[ids])),
            "aux_conditional_P-AUROC": safe_auc(y[ids], aux[ids]),
        })
    vals = [r["aux_conditional_P-AUROC"] for r in rows if np.isfinite(r["aux_conditional_P-AUROC"])]
    rows.append({"bin": "mean", "count": int(len(y)), "S_H_min": float("nan"), "S_H_max": float("nan"), "aux_conditional_P-AUROC": float(np.mean(vals)) if vals else float("nan")})
    return rows


def error_rows(cat, y, base, geo):
    normal = y < 0.5
    anomaly = y > 0.5
    if not np.any(normal) or not np.any(anomaly):
        return []
    fp_thr = np.percentile(base[normal], 95)
    fn_thr = np.percentile(base[anomaly], 20)
    fp = normal & (base >= fp_thr)
    fn = anomaly & (base <= fn_thr)
    correct_n = normal & (base < fp_thr)
    correct_a = anomaly & (base > fn_thr)
    return [{
        "heldout_category": cat,
        "FP_count": int(np.sum(fp)),
        "FN_count": int(np.sum(fn)),
        "geo_FP_mean": float(np.mean(geo[fp])) if np.any(fp) else float("nan"),
        "geo_correct_normal_mean": float(np.mean(geo[correct_n])) if np.any(correct_n) else float("nan"),
        "geo_FN_mean": float(np.mean(geo[fn])) if np.any(fn) else float("nan"),
        "geo_correct_anomaly_mean": float(np.mean(geo[correct_a])) if np.any(correct_a) else float("nan"),
    }]


def run_loo(data, device, smoke):
    cats = sorted(set(data["category"].tolist()))
    if smoke:
        cats = cats[:4]
    torch.manual_seed(SEED)
    init = AuxHead(data["x"].shape[1])
    init_state = {k: v.detach().clone() for k, v in init.state_dict().items()}
    raw_rows, diag_rows, cond_rows, err_rows, curve_rows = [], [], [], [], []
    for fold, cat in enumerate(cats):
        print(f"[MaskGeometry] LOO {fold + 1}/{len(cats)} heldout={cat}", flush=True)
        train = data["category"] != cat
        test = data["category"] == cat
        xtr = data["x"][train]
        heads = {}
        targets = {
            "E1_Geometry": (data["phi"][train], "smoothl1"),
            "E2_Binary": (data["y"][train], "bce"),
            "E3_Occupancy": (data["q"][train], "bce"),
        }
        for variant, (target, mode) in targets.items():
            head, curves, reused = train_head(variant, cat, xtr, target, mode, init_state, device, smoke)
            heads[variant] = (head, reused)
            curve_rows.extend(curves)
        train_base_z, base_mu, base_sd = zscore(data["S_H"][train], data["S_H"][test])
        raw_rows.append(metric_row(cat, "B0_HardNormal", data["y"][test], data["S_H"][test]))
        preds_train, preds_test = {}, {}
        aux_scores_train, aux_scores_test = {}, {}
        for variant, (head, reused) in heads.items():
            pt = predict(head, data["x"][train], device)
            pe = predict(head, data["x"][test], device)
            preds_train[variant] = pt
            preds_test[variant] = pe
            if variant == "E1_Geometry":
                st = (pt + 1.0) / 2.0
                se = (pe + 1.0) / 2.0
            else:
                st = 1.0 / (1.0 + np.exp(-pt))
                se = 1.0 / (1.0 + np.exp(-pe))
            aux_scores_train[variant] = st.astype(np.float32)
            aux_scores_test[variant] = se.astype(np.float32)
            aux_z, aux_mu, aux_sd = zscore(st, se)
            fused = 0.5 * train_base_z + 0.5 * aux_z
            raw_rows.append(metric_row(cat, variant, data["y"][test], fused))
            if variant == "E1_Geometry":
                cond_rows.extend({"heldout_category": cat, **r} for r in conditional_auc(data["y"][test], data["S_H"][test], se))
                err_rows.extend(error_rows(cat, data["y"][test], data["S_H"][test], se))
        y_pred = aux_scores_test["E2_Binary"]
        q_pred = aux_scores_test["E3_Occupancy"]
        geo_pred = preds_test["E1_Geometry"]
        normal = data["y"][test] < 0.5
        anomaly = data["y"][test] > 0.5
        diag_rows.append({
            "heldout_category": cat,
            "corr_pred_phi_binary_score": corr(geo_pred, y_pred),
            "R2_binary_occupancy_to_geometry": linear_r2_binary_occ(y_pred, q_pred, geo_pred),
            "corr_pred_phi_true_phi_all": corr(geo_pred, data["phi"][test]),
            "corr_pred_phi_true_phi_normal": corr(geo_pred[normal], data["phi"][test][normal]) if np.any(normal) else float("nan"),
            "corr_pred_phi_true_phi_anomaly": corr(geo_pred[anomaly], data["phi"][test][anomaly]) if np.any(anomaly) else float("nan"),
            "base_z_mu_train": base_mu,
            "base_z_sd_train": base_sd,
        })
    return raw_rows, diag_rows, cond_rows, err_rows, curve_rows


def summarize(raw_rows):
    variants = ["B0_HardNormal", "E1_Geometry", "E2_Binary", "E3_Occupancy"]
    summary = []
    for variant in variants:
        rows = [r for r in raw_rows if r["variant"] == variant]
        summary.append({
            "variant": variant,
            "heldout_categories": len(rows),
            "mean_P-AUROC": float(np.mean([r["P-AUROC"] for r in rows])),
            "mean_P-AP": float(np.mean([r["P-AP"] for r in rows])),
            "mean_G": float(np.mean([r["G_pixel"] for r in rows])),
            "median_G": float(np.median([r["G_pixel"] for r in rows])),
        })
    by = {(r["heldout_category"], r["variant"]): r for r in raw_rows}
    cats = sorted({r["heldout_category"] for r in raw_rows})
    deltas = []
    for cat in cats:
        b0 = by[(cat, "B0_HardNormal")]["G_pixel"]
        e1 = by[(cat, "E1_Geometry")]["G_pixel"]
        e2 = by[(cat, "E2_Binary")]["G_pixel"]
        e3 = by[(cat, "E3_Occupancy")]["G_pixel"]
        deltas.append({"heldout_category": cat, "E1_minus_B0_G": e1 - b0, "E1_minus_E2_G": e1 - e2, "E1_minus_E3_G": e1 - e3})
    stats = {
        "categories": len(cats),
        "E1_improved_over_B0": int(sum(d["E1_minus_B0_G"] > 0 for d in deltas)),
        "mean_E1_minus_B0_G": float(np.mean([d["E1_minus_B0_G"] for d in deltas])),
        "mean_E1_minus_E2_G": float(np.mean([d["E1_minus_E2_G"] for d in deltas])),
        "mean_E1_minus_E3_G": float(np.mean([d["E1_minus_E3_G"] for d in deltas])),
        "median_E1_minus_B0_G": float(np.median([d["E1_minus_B0_G"] for d in deltas])),
    }
    return summary, deltas, stats


def decide(stats, diag_rows):
    r2 = float(np.mean([r["R2_binary_occupancy_to_geometry"] for r in diag_rows]))
    corr_anom = [r["corr_pred_phi_true_phi_anomaly"] for r in diag_rows if np.isfinite(r["corr_pred_phi_true_phi_anomaly"])]
    corr_norm = [r["corr_pred_phi_true_phi_normal"] for r in diag_rows if np.isfinite(r["corr_pred_phi_true_phi_normal"])]
    max_internal = max(float(np.nanmean(corr_anom)) if corr_anom else float("nan"), float(np.nanmean(corr_norm)) if corr_norm else float("nan"))
    if (
        stats["E1_improved_over_B0"] >= 8
        and stats["mean_E1_minus_B0_G"] >= 0.7
        and stats["mean_E1_minus_E2_G"] >= 0.4
        and stats["mean_E1_minus_E3_G"] >= 0.3
        and stats["median_E1_minus_B0_G"] > 0
        and r2 < 0.95
        and max_internal >= 0.3
    ):
        outcome = "MASK_GEOMETRY_HAS_COMPLEMENTARY_TRANSFER_SIGNAL"
    elif abs(stats["mean_E1_minus_E3_G"]) < 0.3:
        outcome = "GEOMETRY_COLLAPSES_TO_OCCUPANCY_INFORMATION"
    elif abs(stats["mean_E1_minus_E2_G"]) < 0.4:
        outcome = "GENERIC_AUXILIARY_SEGMENTATION_HEAD_SIGNAL"
    elif stats["mean_E1_minus_B0_G"] > 0 and r2 >= 0.95:
        outcome = "GEOMETRY_HEAD_COLLAPSES_TO_ANOMALY_CLASSIFIER"
    else:
        outcome = "MASK_GEOMETRY_NO_TRANSFER_SIGNAL"
    return outcome, {"mean_R2_binary_occupancy_to_geometry": r2, "mean_internal_distance_corr_best": max_internal}


def write_report(outcome, summary_rows, stats, extra, runtime):
    lines = [
        "# Mask Geometry Auxiliary Knowledge Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "SOURCE-ONLY / NO-TARGET. CLIP Symmetric HardNormal is frozen. Aux heads are trained with 12-category VisA LOO and fused with HardNormal by fixed source-only z-score 1:1 averaging.",
        "",
        "| Variant | Mean P-AUROC | Mean P-AP | Mean G | Median G |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['variant']} | {r['mean_P-AUROC']:.4f} | {r['mean_P-AP']:.4f} | {r['mean_G']:.4f} | {r['median_G']:.4f} |")
    lines.extend([
        "",
        "## Gate",
        f"- E1 improved over B0: {stats['E1_improved_over_B0']}/{stats['categories']}",
        f"- mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f}",
        f"- mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f}",
        f"- mean E1-E3 G: {stats['mean_E1_minus_E3_G']:+.4f}",
        f"- median E1-B0 G: {stats['median_E1_minus_B0_G']:+.4f}",
        f"- R2(binary+occupancy -> geometry): {extra['mean_R2_binary_occupancy_to_geometry']:.6f}",
        f"- best same-label distance correlation: {extra['mean_internal_distance_corr_best']:.6f}",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MaskGeometry] device={device}", flush=True)
    data = build_data(smoke=args.smoke)
    raw_rows, diag_rows, cond_rows, err_rows, curve_rows = run_loo(data, device, args.smoke)
    summary_rows, delta_rows, stats = summarize(raw_rows)
    outcome, extra = decide(stats, diag_rows)
    runtime = time.time() - start
    write_csv(OUT / "LOO_METRICS.csv", raw_rows)
    write_csv(OUT / "SUMMARY_METRICS.csv", summary_rows)
    write_csv(OUT / "DELTA_METRICS.csv", delta_rows)
    write_csv(OUT / "NONDEGENERACY.csv", diag_rows)
    write_csv(OUT / "CONDITIONAL_COMPLEMENTARITY.csv", cond_rows)
    write_csv(OUT / "ERROR_ANALYSIS.csv", err_rows)
    write_csv(OUT / "TRAINING_CURVES.csv", curve_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "nondegeneracy": extra,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_used": False,
        "frozen_clip_hardnormal": True,
        "epochs": 2 if args.smoke else EPOCHS,
        "lr": LR,
        "head": "Linear(D,32)-GELU-Linear(32,1)",
        "signed_distance": "inside positive normalized per image; outside truncated at one CLIP patch width; sampled at 37x37 patch centers",
    })
    write_report(outcome, summary_rows, stats, extra, runtime)
    print(json.dumps({"outcome": outcome, **stats, **extra, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
