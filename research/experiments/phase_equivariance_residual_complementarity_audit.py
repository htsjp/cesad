import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import load_model
from experiments.phase_patch_occupancy_supervision_audit import load_context
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np
from models.mlp import average_neighbor


OUT = Path("results/phase_equivariance_residual_complementarity_audit/formal")
EPS = 1e-12
PATCH_GRID = SIDE * SIDE


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
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s) * 100.0)


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s) * 100.0)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "P50": float("nan"), "P90": float("nan"), "P95": float("nan")}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "P50": float(np.percentile(x, 50)),
        "P90": float(np.percentile(x, 90)),
        "P95": float(np.percentile(x, 95)),
    }


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if a.size < 2 or a.std() < EPS or b.std() < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_patch_features(patch_t, q_pc1, align_model, m_c, alpha, v_hn, device):
    patch = average_neighbor(patch_t).detach().float().cpu().numpy()[0].astype(np.float32)
    z0 = remove_pc1(patch, q_pc1)
    z0 = norm_np(z0.reshape(PATCH_GRID, 1024))
    with torch.no_grad():
        z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device))
        m = torch.tensor(m_c, dtype=torch.float32, device=device)
        v = torch.tensor(v_hn, dtype=torch.float32, device=device)
        score = contrast_logits(z, m, alpha, F.normalize(v, dim=-1)).detach().cpu().numpy().astype(np.float32)
    return score.reshape(SIDE, SIDE)


@torch.no_grad()
def forward_scores(model, image, q_pc1, align_model, m_c, alpha, v_hn, device):
    out0 = model.encode_image(image, [6, 12, 18, 24], DPAM_layer=24)
    s0 = score_patch_features(out0[1][3], q_pc1, align_model, m_c, alpha, v_hn, device)
    image_flip = torch.flip(image, dims=[-1])
    out1 = model.encode_image(image_flip, [6, 12, 18, 24], DPAM_layer=24)
    sf = score_patch_features(out1[1][3], q_pc1, align_model, m_c, alpha, v_hn, device)
    s1 = np.fliplr(sf).astype(np.float32)
    return s0.astype(np.float32), s1


def collect_source_orbit(smoke, batch_limit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"dpam_layer": 24, "n_ctx": 12, "depth": 9, "t_n_ctx": 4}
    model = load_model(cfg, str(device))
    q_pc1, align_model, m_c, alpha, v_hn = load_context(device)
    _name, dataset, indices, _cls, _patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    positions = list(range(len(indices)))
    if smoke:
        positions = positions[:80]
    if batch_limit is not None:
        positions = positions[: int(batch_limit)]
    rows = []
    for n, pos in enumerate(positions, 1):
        if n == 1 or n % 100 == 0:
            print(f"[EquivAudit] image {n}/{len(positions)}", flush=True)
        item = dataset[int(indices[int(pos)])]
        image = item["img"].unsqueeze(0).to(device)
        s0, s1 = forward_scores(model, image, q_pc1, align_model, m_c, alpha, v_hn, device)
        y = np.asarray(labels_by_pos.get(int(pos), np.zeros((PATCH_GRID,), dtype=np.float32)), dtype=np.uint8).reshape(SIDE, SIDE)
        stable = 0.5 * (s0 + s1)
        residual = np.abs(s0 - s1)
        rows.append({
            "category": str(item["cls_name"]),
            "position": int(pos),
            "index": int(indices[int(pos)]),
            "y": y.reshape(-1),
            "S0": s0.reshape(-1),
            "S1": s1.reshape(-1),
            "StableMean": stable.reshape(-1),
            "R": residual.reshape(-1),
        })
    data = {
        "category": np.concatenate([np.full(PATCH_GRID, r["category"], dtype=object) for r in rows]),
        "position": np.concatenate([np.full(PATCH_GRID, r["position"], dtype=np.int32) for r in rows]),
        "y": np.concatenate([r["y"] for r in rows]).astype(np.uint8),
        "S0": np.concatenate([r["S0"] for r in rows]).astype(np.float32),
        "S1": np.concatenate([r["S1"] for r in rows]).astype(np.float32),
        "StableMean": np.concatenate([r["StableMean"] for r in rows]).astype(np.float32),
        "R": np.concatenate([r["R"] for r in rows]).astype(np.float32),
    }
    data["image_records"] = rows
    return data, str(device)


def residual_stats(data):
    rows = []
    for name, mask in (("normal", data["y"] == 0), ("anomaly", data["y"] > 0), ("all", np.ones_like(data["y"], dtype=bool))):
        rows.append({"group": name, **stats(data["R"][mask])})
    rows.append({"group": "R_metric", "P-AUROC": safe_auc(data["y"], data["R"]), "P-AP": safe_ap(data["y"], data["R"]), "G": 0.5 * (safe_auc(data["y"], data["R"]) + safe_ap(data["y"], data["R"]))})
    rows.append({"group": "StableMean_metric", "P-AUROC": safe_auc(data["y"], data["StableMean"]), "P-AP": safe_ap(data["y"], data["StableMean"]), "G": 0.5 * (safe_auc(data["y"], data["StableMean"]) + safe_ap(data["y"], data["StableMean"]))})
    rows.append({"group": "S0_metric", "P-AUROC": safe_auc(data["y"], data["S0"]), "P-AP": safe_ap(data["y"], data["S0"]), "G": 0.5 * (safe_auc(data["y"], data["S0"]) + safe_ap(data["y"], data["S0"]))})
    rows.append({"group": "corr_S0_R", "value": corr(data["S0"], data["R"])})
    rows.append({"group": "corr_StableMean_R", "value": corr(data["StableMean"], data["R"])})
    return rows


def conditional_table(data):
    order = np.argsort(data["S0"], kind="mergesort")
    bins = np.array_split(order, 20)
    rows = []
    aucs = []
    for i, ids in enumerate(bins):
        y = data["y"][ids]
        r = data["R"][ids]
        auc = safe_auc(y, r)
        ap = safe_ap(y, r)
        if np.isfinite(auc):
            aucs.append(auc)
        rows.append({
            "bin": i,
            "count": int(ids.size),
            "S0_min": float(data["S0"][ids].min()),
            "S0_max": float(data["S0"][ids].max()),
            "anomaly_ratio": float(y.mean()),
            "R_anomaly_minus_normal": float(r[y > 0].mean() - r[y == 0].mean()) if np.unique(y).size == 2 else float("nan"),
            "R_AUROC": auc,
            "R_AP": ap,
        })
    arr = np.asarray(aucs, dtype=np.float64)
    rows.insert(0, {"bin": "__summary__", "valid_bins": int(arr.size), "R_AUROC": float(arr.mean()) if arr.size else float("nan"), "R_AUROC_std": float(arr.std()) if arr.size else float("nan")})
    return rows


def standardize(train_x, test_x):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)
    return (train_x - mu) / sd, (test_x - mu) / sd


def fit_probe(train_x, train_y, test_x):
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=0)
    clf.fit(train_x, train_y)
    return clf.predict_proba(test_x)[:, 1]


def loo_probe(data):
    rng = np.random.default_rng(0)
    r_random = data["R"].copy()
    rng.shuffle(r_random)
    cats = sorted(set(data["category"].tolist()))
    rows = []
    for cat in cats:
        test = data["category"] == cat
        train = ~test
        y_train = data["y"][train]
        y_test = data["y"][test]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        variants = {
            "P0_S0": (data["S0"][train].reshape(-1, 1), data["S0"][test].reshape(-1, 1)),
            "P1_S0_plus_R": (np.stack([data["S0"][train], data["R"][train]], axis=1), np.stack([data["S0"][test], data["R"][test]], axis=1)),
            "P2_StableMean": (data["StableMean"][train].reshape(-1, 1), data["StableMean"][test].reshape(-1, 1)),
            "P3_S0_plus_Rrandom": (np.stack([data["S0"][train], r_random[train]], axis=1), np.stack([data["S0"][test], r_random[test]], axis=1)),
        }
        metrics = {}
        for name, (tr, te) in variants.items():
            tr, te = standardize(tr.astype(np.float64), te.astype(np.float64))
            pred = fit_probe(tr, y_train, te)
            pa = safe_auc(y_test, pred)
            pap = safe_ap(y_test, pred)
            metrics[name] = {"P-AUROC": pa, "P-AP": pap, "G": 0.5 * (pa + pap)}
            rows.append({"heldout_category": cat, "variant": name, **metrics[name]})
        for name, a, b in (("DELTA_P1_minus_P0", "P1_S0_plus_R", "P0_S0"), ("DELTA_P1_minus_P2", "P1_S0_plus_R", "P2_StableMean"), ("DELTA_P1_minus_P3", "P1_S0_plus_R", "P3_S0_plus_Rrandom")):
            rows.append({
                "heldout_category": cat,
                "variant": name,
                "P-AUROC": metrics[a]["P-AUROC"] - metrics[b]["P-AUROC"],
                "P-AP": metrics[a]["P-AP"] - metrics[b]["P-AP"],
                "G": metrics[a]["G"] - metrics[b]["G"],
            })
    summary = {"heldout_category": "__summary__", "variant": "summary"}
    for dname in ("DELTA_P1_minus_P0", "DELTA_P1_minus_P2", "DELTA_P1_minus_P3"):
        vals = np.asarray([r["G"] for r in rows if r["variant"] == dname], dtype=np.float64)
        summary[f"{dname}_categories"] = int(vals.size)
        summary[f"{dname}_improved"] = int(np.sum(vals > 0))
        summary[f"{dname}_mean_G"] = float(vals.mean()) if vals.size else float("nan")
        summary[f"{dname}_median_G"] = float(np.median(vals)) if vals.size else float("nan")
    return [summary] + rows


def error_analysis(data):
    y, s, r = data["y"], data["S0"], data["R"]
    normal_s = s[y == 0]
    anomaly_s = s[y > 0]
    n95 = np.percentile(normal_s, 95)
    a20 = np.percentile(anomaly_s, 20)
    masks = {
        "FP_H_normal_S0_gt_normal_P95": (y == 0) & (s > n95),
        "correct_normal": (y == 0) & (s <= n95),
        "FN_H_anomaly_S0_lt_anomaly_P20": (y > 0) & (s < a20),
        "correct_anomaly": (y > 0) & (s >= a20),
    }
    rows = []
    for name, mask in masks.items():
        rows.append({"group": name, **stats(r[mask])})
    rows.append({"group": "delta_FP_minus_correct_normal", "mean": float(r[masks["FP_H_normal_S0_gt_normal_P95"]].mean() - r[masks["correct_normal"]].mean())})
    rows.append({"group": "delta_FN_minus_correct_anomaly", "mean": float(r[masks["FN_H_anomaly_S0_lt_anomaly_P20"]].mean() - r[masks["correct_anomaly"]].mean())})
    return rows


def boundary_control(data):
    groups = defaultdict(list)
    for rec in data["image_records"]:
        mask = rec["y"].reshape(SIDE, SIDE).astype(bool)
        r = rec["R"].reshape(SIDE, SIDE)
        if mask.any():
            eroded = binary_erosion(mask, iterations=1, border_value=0)
            dilated = binary_dilation(mask, iterations=3)
            boundary = mask & ~eroded
            interior = mask & eroded
            near_normal = dilated & ~mask
            far_normal = ~dilated
        else:
            boundary = np.zeros_like(mask)
            interior = np.zeros_like(mask)
            near_normal = np.zeros_like(mask)
            far_normal = ~mask
        for name, m in (("anomaly_interior", interior), ("anomaly_boundary", boundary), ("normal_near_boundary", near_normal), ("far_normal", far_normal)):
            if m.any():
                groups[name].extend(r[m].tolist())
    return [{"region": name, **stats(vals)} for name, vals in groups.items()]


def decide(residual_rows, loo_rows, boundary_rows):
    all_r = next(r for r in residual_rows if r.get("group") == "all")
    corr_s0 = next(r for r in residual_rows if r.get("group") == "corr_S0_R")
    summary = next(r for r in loo_rows if r.get("heldout_category") == "__summary__")
    mean_r = float(all_r["mean"])
    p95_r = float(all_r["P95"])
    nonzero = not (mean_r < 1e-6 and p95_r < 1e-5)
    abs_corr = abs(float(corr_s0["value"]))
    cats = max(1, int(summary.get("DELTA_P1_minus_P0_categories", 0)))
    improve_frac = float(summary.get("DELTA_P1_minus_P0_improved", 0)) / cats
    d10 = float(summary.get("DELTA_P1_minus_P0_mean_G", float("nan")))
    d12 = float(summary.get("DELTA_P1_minus_P2_mean_G", float("nan")))
    d13 = float(summary.get("DELTA_P1_minus_P3_mean_G", float("nan")))
    boundary_artifact = False
    b = {r["region"]: r for r in boundary_rows}
    if "anomaly_boundary" in b and "anomaly_interior" in b:
        boundary_artifact = float(b["anomaly_boundary"]["mean"]) > float(b["anomaly_interior"]["mean"]) * 1.5 and d10 < 0.5
    if nonzero and abs_corr <= 0.8 and improve_frac >= 2.0 / 3.0 and d10 >= 0.5 and d13 >= 0.3 and d12 >= 0.3:
        outcome = "EQUIVARIANCE_RESIDUAL_HAS_COMPLEMENTARY_INFORMATION"
    elif d12 < 0.3 and d10 >= 0.3:
        outcome = "SIMPLE_TTA_SIGNAL_ONLY"
    elif boundary_artifact:
        outcome = "EQUIVARIANCE_RESIDUAL_BOUNDARY_ARTIFACT"
    elif abs(d10) < 0.3:
        outcome = "EQUIVARIANCE_RESIDUAL_REDUNDANT"
    else:
        outcome = "EQUIVARIANCE_RESIDUAL_NO_SIGNAL"
    return outcome, {
        "R_mean": mean_r,
        "R_P95": p95_r,
        "residual_nonzero": nonzero,
        "corr_S0_R": float(corr_s0["value"]),
        "P1_improve_fraction": improve_frac,
        "mean_delta_G_P1_minus_P0": d10,
        "mean_delta_G_P1_minus_P2": d12,
        "mean_delta_G_P1_minus_P3": d13,
        "boundary_artifact_flag": boundary_artifact,
    }


def write_report(outcome, stats_dict, residual_rows, cond_rows, loo_rows, err_rows, boundary_rows, runtime, smoke):
    def row(group):
        return next((r for r in residual_rows if r.get("group") == group), {})
    r_metric = row("R_metric")
    s0_metric = row("S0_metric")
    cond = next((r for r in cond_rows if r.get("bin") == "__summary__"), {})
    summary = next((r for r in loo_rows if r.get("heldout_category") == "__summary__"), {})
    lines = [
        "# Transformation-Orbit Equivariance Residual Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN / NO-TARGET. Horizontal flip is applied to Full VisA source images, then inverse-flipped back to compare frozen HardNormal score maps.",
        "",
        "## Residual",
        "",
        f"- mean(R): {stats_dict['R_mean']:.8f}; P95(R): {stats_dict['R_P95']:.8f}.",
        f"- AUROC(R): {float(r_metric.get('P-AUROC', float('nan'))):.4f}; AP(R): {float(r_metric.get('P-AP', float('nan'))):.4f}.",
        f"- AUROC(S0): {float(s0_metric.get('P-AUROC', float('nan'))):.4f}; AP(S0): {float(s0_metric.get('P-AP', float('nan'))):.4f}.",
        f"- corr(S0,R): {stats_dict['corr_S0_R']:.6f}.",
        "",
        "## Conditional Complementarity",
        "",
        f"- mean R AUROC within S0 bins: {float(cond.get('R_AUROC', float('nan'))):.4f}.",
        f"- std R AUROC within S0 bins: {float(cond.get('R_AUROC_std', float('nan'))):.4f}.",
        "",
        "## LOO Probe",
        "",
        f"- P1>S0 categories: {int(summary.get('DELTA_P1_minus_P0_improved', 0))}/{int(summary.get('DELTA_P1_minus_P0_categories', 0))}.",
        f"- mean Delta G(P1-P0): {stats_dict['mean_delta_G_P1_minus_P0']:+.4f}.",
        f"- mean Delta G(P1-P2): {stats_dict['mean_delta_G_P1_minus_P2']:+.4f}.",
        f"- mean Delta G(P1-P3): {stats_dict['mean_delta_G_P1_minus_P3']:+.4f}.",
        "",
        f"Boundary artifact flag: {stats_dict['boundary_artifact_flag']}.",
        f"Runtime: {runtime:.1f}s. Smoke: {smoke}.",
        "",
        "STOP: no target experiment and no training were started.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data, device = collect_source_orbit(args.smoke, args.limit)
    residual_rows = residual_stats(data)
    cond_rows = conditional_table(data)
    loo_rows = loo_probe(data)
    err_rows = error_analysis(data)
    boundary_rows = boundary_control(data)
    outcome, stats_dict = decide(residual_rows, loo_rows, boundary_rows)
    runtime = time.time() - start
    write_csv(OUT / "RESIDUAL_STATS.csv", residual_rows)
    write_csv(OUT / "CONDITIONAL_COMPLEMENTARITY.csv", cond_rows)
    write_csv(OUT / "LOO_PROBE.csv", loo_rows)
    write_csv(OUT / "ERROR_ANALYSIS.csv", err_rows)
    write_csv(OUT / "BOUNDARY_CONTROL.csv", boundary_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats_dict,
        "runtime_sec": runtime,
        "device": device,
        "no_training": True,
        "target_data_used": False,
        "source": "Full VisA",
        "transform": "horizontal_flip",
        "smoke": bool(args.smoke),
        "limit": args.limit,
    })
    write_report(outcome, stats_dict, residual_rows, cond_rows, loo_rows, err_rows, boundary_rows, runtime, bool(args.smoke))
    print(f"[EquivAudit] OUTCOME={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
