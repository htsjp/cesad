import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_patch_occupancy_supervision_audit import load_context
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np


OUT = Path("results/phase_state_pair_information_completion_audit/formal")
PATCH_GRID = SIDE * SIDE
EPS = 1e-12


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


def metric_row(name, y, s):
    pa = safe_auc(y, s)
    pap = safe_ap(y, s)
    return {"scope": "source", "score": name, "P-AUROC": pa, "P-AP": pap, "G": 0.5 * (pa + pap)}


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "P5": float("nan"), "P50": float("nan"), "P95": float("nan"), "cv": float("nan")}
    mean = float(x.mean())
    return {
        "count": int(x.size),
        "mean": mean,
        "std": float(x.std()),
        "P5": float(np.percentile(x, 5)),
        "P50": float(np.percentile(x, 50)),
        "P95": float(np.percentile(x, 95)),
        "cv": float(x.std() / max(abs(mean), EPS)),
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


def standardize(train_x, test_x):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)
    return (train_x - mu) / sd, (test_x - mu) / sd


def fit_probe(train_x, train_y, test_x):
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=0)
    clf.fit(train_x, train_y)
    return clf.predict_proba(test_x)[:, 1]


def current_states(m_c, alpha, v_hn):
    m = torch.tensor(m_c, dtype=torch.float32)
    v = F.normalize(torch.tensor(v_hn, dtype=torch.float32), dim=-1)
    s_a = F.normalize(m + alpha * v, dim=-1)
    s_n = F.normalize(m - alpha * v, dim=-1)
    return s_n.numpy().astype(np.float32), s_a.numpy().astype(np.float32)


def orth_basis(s_n, s_a):
    mat = np.stack([s_n, s_a], axis=1).astype(np.float32)
    q, _r = np.linalg.qr(mat)
    return q[:, :2].astype(np.float32)


def collect_state_pair(smoke, limit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_pc1, align_model, m_c, alpha, v_hn = load_context(device)
    s_n, s_a = current_states(m_c, alpha, v_hn)
    u = orth_basis(s_n, s_a)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    positions = list(range(len(indices)))
    if smoke:
        positions = positions[:96]
    if limit is not None:
        positions = positions[: int(limit)]

    arrays = {k: [] for k in ["y", "Delta", "Common", "Residual", "P_current", "P_reconstructed"]}
    cat_ids = []
    pos_ids = []
    cats = []
    cat_to_id = {}
    s_n_t = torch.tensor(s_n, dtype=torch.float32, device=device)
    s_a_t = torch.tensor(s_a, dtype=torch.float32, device=device)
    u_t = torch.tensor(u, dtype=torch.float32, device=device)
    m_t = torch.tensor(m_c, dtype=torch.float32, device=device)
    v_t = F.normalize(torch.tensor(v_hn, dtype=torch.float32, device=device), dim=-1)
    for n, pos in enumerate(positions, 1):
        if n == 1 or n % 100 == 0:
            print(f"[StatePairInfo] image {n}/{len(positions)}", flush=True)
        item = dataset[int(indices[int(pos)])]
        cat = str(item["cls_name"])
        if cat not in cat_to_id:
            cat_to_id[cat] = len(cats)
            cats.append(cat)
        y = np.asarray(labels_by_pos.get(int(pos), np.zeros((PATCH_GRID,), dtype=np.float32)), dtype=np.uint8).reshape(-1)
        x0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
        x0 = norm_np(x0.reshape(PATCH_GRID, 1024))
        with torch.no_grad():
            z = align_model.adapt(torch.tensor(x0, dtype=torch.float32, device=device))
            l_n = z @ s_n_t
            l_a = z @ s_a_t
            delta = l_a - l_n
            common = l_a + l_n
            proj = z @ u_t
            residual = torch.sqrt(torch.clamp(1.0 - proj.pow(2).sum(dim=-1), min=0.0))
            delta_current = contrast_logits(z, m_t, alpha, v_t)
            p_current = torch.sigmoid(delta_current)
            p_recon = torch.sigmoid(delta)
        arrays["y"].append(y)
        arrays["Delta"].append(delta.detach().cpu().numpy().astype(np.float32))
        arrays["Common"].append(common.detach().cpu().numpy().astype(np.float32))
        arrays["Residual"].append(residual.detach().cpu().numpy().astype(np.float32))
        arrays["P_current"].append(p_current.detach().cpu().numpy().astype(np.float32))
        arrays["P_reconstructed"].append(p_recon.detach().cpu().numpy().astype(np.float32))
        cat_ids.append(np.full((PATCH_GRID,), cat_to_id[cat], dtype=np.int16))
        pos_ids.append(np.full((PATCH_GRID,), int(pos), dtype=np.int32))

    data = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    data["category_id"] = np.concatenate(cat_ids, axis=0)
    data["position"] = np.concatenate(pos_ids, axis=0)
    data["categories"] = np.asarray(cats, dtype=object)
    data["max_probability_reconstruction_error"] = np.asarray([np.max(np.abs(data["P_current"] - data["P_reconstructed"]))], dtype=np.float32)
    data["state_norms"] = np.asarray([np.linalg.norm(s_n), np.linalg.norm(s_a), float(s_n @ s_a)], dtype=np.float32)
    return data, str(device)


def state_pair_stats(data):
    rows = []
    y = data["y"]
    for score in ["Common", "Residual"]:
        for group, mask in [("normal", y == 0), ("anomaly", y > 0), ("all", np.ones_like(y, dtype=bool))]:
            rows.append({"score": score, "group": group, **stats(data[score][mask])})
        rows.append(metric_row(score, y, data[score]))
    rows.append(metric_row("Delta", y, data["Delta"]))
    rows.append({"scope": "source", "score": "corr_Delta_Common", "G": corr(data["Delta"], data["Common"])})
    rows.append({"scope": "source", "score": "corr_Delta_Residual", "G": corr(data["Delta"], data["Residual"])})
    rows.append({"scope": "source", "score": "corr_Common_Residual", "G": corr(data["Common"], data["Residual"])})
    rows.append({"scope": "implementation", "score": "max_probability_reconstruction_error", "G": float(data["max_probability_reconstruction_error"][0])})
    rows.append({"scope": "implementation", "score": "state_sN_norm", "G": float(data["state_norms"][0])})
    rows.append({"scope": "implementation", "score": "state_sA_norm", "G": float(data["state_norms"][1])})
    rows.append({"scope": "implementation", "score": "cos_sN_sA", "G": float(data["state_norms"][2])})
    return rows


def conditional_information(data):
    order = np.argsort(data["Delta"], kind="mergesort")
    bins = np.array_split(order, 20)
    rows = []
    auc_common = []
    auc_res = []
    for i, ids in enumerate(bins):
        y = data["y"][ids]
        row = {
            "bin": i,
            "count": int(ids.size),
            "Delta_min": float(data["Delta"][ids].min()),
            "Delta_max": float(data["Delta"][ids].max()),
            "anomaly_ratio": float(y.mean()),
        }
        for score, bucket in [("Common", auc_common), ("Residual", auc_res)]:
            auc = safe_auc(y, data[score][ids])
            ap = safe_ap(y, data[score][ids])
            if np.isfinite(auc):
                bucket.append(auc)
            row[f"{score}_AUROC"] = auc
            row[f"{score}_AP"] = ap
            row[f"{score}_anomaly_minus_normal"] = float(data[score][ids][y > 0].mean() - data[score][ids][y == 0].mean()) if np.unique(y).size == 2 else float("nan")
        rows.append(row)
    ac = np.asarray(auc_common, dtype=np.float64)
    ar = np.asarray(auc_res, dtype=np.float64)
    rows.insert(0, {
        "bin": "__summary__",
        "valid_common_bins": int(ac.size),
        "Common_AUROC": float(ac.mean()) if ac.size else float("nan"),
        "Common_AUROC_std": float(ac.std()) if ac.size else float("nan"),
        "valid_residual_bins": int(ar.size),
        "Residual_AUROC": float(ar.mean()) if ar.size else float("nan"),
        "Residual_AUROC_std": float(ar.std()) if ar.size else float("nan"),
    })
    return rows


def loo_probe(data):
    rng = np.random.default_rng(0)
    common_rand = data["Common"].copy()
    residual_rand = data["Residual"].copy()
    rng.shuffle(common_rand)
    rng.shuffle(residual_rand)
    rows = []
    cats = data["categories"].tolist()
    for cid, cat in enumerate(cats):
        test = data["category_id"] == cid
        train = ~test
        y_train = data["y"][train]
        y_test = data["y"][test]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        variants = {
            "P0_Delta": ([data["Delta"][train]], [data["Delta"][test]]),
            "P1_Delta_Common": ([data["Delta"][train], data["Common"][train]], [data["Delta"][test], data["Common"][test]]),
            "P2_Delta_Residual": ([data["Delta"][train], data["Residual"][train]], [data["Delta"][test], data["Residual"][test]]),
            "P3_Delta_Common_Residual": ([data["Delta"][train], data["Common"][train], data["Residual"][train]], [data["Delta"][test], data["Common"][test], data["Residual"][test]]),
            "Prand_Delta_CommonResidualRandom": ([data["Delta"][train], common_rand[train], residual_rand[train]], [data["Delta"][test], common_rand[test], residual_rand[test]]),
        }
        metrics = {}
        for name, (tr_cols, te_cols) in variants.items():
            tr = np.stack(tr_cols, axis=1).astype(np.float64)
            te = np.stack(te_cols, axis=1).astype(np.float64)
            tr, te = standardize(tr, te)
            pred = fit_probe(tr, y_train, te)
            pa = safe_auc(y_test, pred)
            pap = safe_ap(y_test, pred)
            metrics[name] = {"P-AUROC": pa, "P-AP": pap, "G": 0.5 * (pa + pap)}
            rows.append({"heldout_category": cat, "variant": name, **metrics[name]})
        for name, a, b in [
            ("DELTA_P1_minus_P0", "P1_Delta_Common", "P0_Delta"),
            ("DELTA_P2_minus_P0", "P2_Delta_Residual", "P0_Delta"),
            ("DELTA_P3_minus_P0", "P3_Delta_Common_Residual", "P0_Delta"),
            ("DELTA_P3_minus_Prand", "P3_Delta_Common_Residual", "Prand_Delta_CommonResidualRandom"),
        ]:
            rows.append({
                "heldout_category": cat,
                "variant": name,
                "P-AUROC": metrics[a]["P-AUROC"] - metrics[b]["P-AUROC"],
                "P-AP": metrics[a]["P-AP"] - metrics[b]["P-AP"],
                "G": metrics[a]["G"] - metrics[b]["G"],
            })
    summary = {"heldout_category": "__summary__", "variant": "summary"}
    for name in ["DELTA_P1_minus_P0", "DELTA_P2_minus_P0", "DELTA_P3_minus_P0", "DELTA_P3_minus_Prand"]:
        vals = np.asarray([r["G"] for r in rows if r["variant"] == name], dtype=np.float64)
        summary[f"{name}_categories"] = int(vals.size)
        summary[f"{name}_improved"] = int(np.sum(vals > 0))
        summary[f"{name}_mean_G"] = float(vals.mean()) if vals.size else float("nan")
        summary[f"{name}_median_G"] = float(np.median(vals)) if vals.size else float("nan")
    return [summary] + rows


def error_complementarity(data):
    y = data["y"]
    delta = data["Delta"]
    common = data["Common"]
    residual = data["Residual"]
    n95 = np.percentile(delta[y == 0], 95)
    a20 = np.percentile(delta[y > 0], 20)
    masks = {
        "FP_H_normal_Delta_gt_normal_P95": (y == 0) & (delta > n95),
        "correct_normal": (y == 0) & (delta <= n95),
        "FN_H_anomaly_Delta_lt_anomaly_P20": (y > 0) & (delta < a20),
        "correct_anomaly": (y > 0) & (delta >= a20),
    }
    rows = []
    common_stats = {}
    residual_stats = {}
    for name, mask in masks.items():
        cs = stats(common[mask])
        rs = stats(residual[mask])
        common_stats[name] = cs
        residual_stats[name] = rs
        rows.append({"group": name, "signal": "Common", **cs})
        rows.append({"group": name, "signal": "Residual", **rs})
    rows.append({"group": "delta_FP_minus_correct_normal", "signal": "Common", "mean": common_stats["FP_H_normal_Delta_gt_normal_P95"]["mean"] - common_stats["correct_normal"]["mean"]})
    rows.append({"group": "delta_FN_minus_correct_anomaly", "signal": "Common", "mean": common_stats["FN_H_anomaly_Delta_lt_anomaly_P20"]["mean"] - common_stats["correct_anomaly"]["mean"]})
    rows.append({"group": "delta_FP_minus_correct_normal", "signal": "Residual", "mean": residual_stats["FP_H_normal_Delta_gt_normal_P95"]["mean"] - residual_stats["correct_normal"]["mean"]})
    rows.append({"group": "delta_FN_minus_correct_anomaly", "signal": "Residual", "mean": residual_stats["FN_H_anomaly_Delta_lt_anomaly_P20"]["mean"] - residual_stats["correct_anomaly"]["mean"]})
    return rows


def decide(stats_rows, cond_rows, loo_rows, err_rows):
    recon = next(r for r in stats_rows if r["score"] == "max_probability_reconstruction_error")["G"]
    common_all = next(r for r in stats_rows if r["score"] == "Common" and r.get("group") == "all")
    residual_all = next(r for r in stats_rows if r["score"] == "Residual" and r.get("group") == "all")
    cond = next(r for r in cond_rows if r["bin"] == "__summary__")
    loo = next(r for r in loo_rows if r["heldout_category"] == "__summary__")
    p3_cats = max(float(loo["DELTA_P3_minus_P0_categories"]), 1.0)
    p3_improved = float(loo["DELTA_P3_minus_P0_improved"])
    p3_mean = float(loo["DELTA_P3_minus_P0_mean_G"])
    p3_prand = float(loo["DELTA_P3_minus_Prand_mean_G"])
    p1_mean = float(loo["DELTA_P1_minus_P0_mean_G"])
    p2_mean = float(loo["DELTA_P2_minus_P0_mean_G"])
    common_cond = float(cond["Common_AUROC"])
    residual_cond = float(cond["Residual_AUROC"])
    stable_cond = max(abs(common_cond - 50.0), abs(residual_cond - 50.0)) >= 3.0
    fp_common = next(r for r in err_rows if r["group"] == "delta_FP_minus_correct_normal" and r["signal"] == "Common")["mean"]
    fn_common = next(r for r in err_rows if r["group"] == "delta_FN_minus_correct_anomaly" and r["signal"] == "Common")["mean"]
    midpoint_not_error = abs(float(fp_common)) < 0.01 and abs(float(fn_common)) < 0.01
    if float(recon) >= 1e-6:
        outcome = "STATE_PAIR_PROBABILITY_RECONSTRUCTION_FAILURE"
    elif abs(float(common_all["mean"])) < 1e-8 and float(common_all["std"]) < 1e-8:
        outcome = "STATE_PAIR_DISCARDED_INFORMATION_REDUNDANT"
    elif (
        p3_improved >= 8
        and p3_mean >= 0.5
        and p3_prand >= 0.3
        and max(p1_mean, p2_mean) >= 0.3
        and stable_cond
    ):
        outcome = "STATE_PAIR_DISCARDED_INFORMATION_COMPLEMENTARY"
    elif p1_mean >= 0.3 and midpoint_not_error:
        outcome = "STATE_MIDPOINT_SIGNAL_NOT_COMPLEMENTARY"
    elif abs(p3_mean) < 0.3:
        outcome = "STATE_PAIR_DISCARDED_INFORMATION_REDUNDANT"
    elif p2_mean < 0.3 and p1_mean < 0.5:
        outcome = "STATE_PAIR_INFORMATION_ALREADY_EXHAUSTED"
    else:
        outcome = "STATE_PAIR_INFORMATION_WEAK"
    return outcome, {
        "max_probability_reconstruction_error": float(recon),
        "Common_cv": float(common_all["cv"]),
        "Residual_cv": float(residual_all["cv"]),
        "Common_conditional_AUROC_mean": common_cond,
        "Residual_conditional_AUROC_mean": residual_cond,
        "P3_improve_fraction": float(p3_improved / p3_cats),
        "mean_delta_G_P1_minus_P0": p1_mean,
        "mean_delta_G_P2_minus_P0": p2_mean,
        "mean_delta_G_P3_minus_P0": p3_mean,
        "mean_delta_G_P3_minus_Prand": p3_prand,
        "midpoint_class_signal_not_error_complement_flag": bool(midpoint_not_error),
        "residual_collapsed": bool(float(residual_all["cv"]) < 0.01),
    }


def write_report(outcome, decision, stats_rows, cond_rows, loo_rows, err_rows, runtime, smoke):
    delta = next(r for r in stats_rows if r.get("score") == "Delta" and r.get("scope") == "source")
    common = next(r for r in stats_rows if r.get("score") == "Common" and r.get("scope") == "source")
    residual = next(r for r in stats_rows if r.get("score") == "Residual" and r.get("scope") == "source")
    corr_dc = next(r for r in stats_rows if r.get("score") == "corr_Delta_Common")["G"]
    corr_dr = next(r for r in stats_rows if r.get("score") == "corr_Delta_Residual")["G"]
    corr_cr = next(r for r in stats_rows if r.get("score") == "corr_Common_Residual")["G"]
    loo = next(r for r in loo_rows if r["heldout_category"] == "__summary__")
    lines = [
        "# State-Pair Information Completion Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN / NO-TARGET. Full VisA source only; current formal Symmetric HardNormal state pair, alignment and scorer are frozen.",
        "",
        "## Implementation",
        "",
        f"- Max anomaly probability reconstruction error from sigmoid(lA-lN): {decision['max_probability_reconstruction_error']:.8g}.",
        "- Delta is the current HardNormal decision coordinate: lA-lN.",
        "- Common is discarded by binary softmax: lA+lN.",
        "- Residual is the normalized patch feature mass outside orth([sN,sA]).",
        "",
        "## Source Signal",
        "",
        f"- Delta source AUROC/AP/G: {float(delta['P-AUROC']):.4f} / {float(delta['P-AP']):.4f} / {float(delta['G']):.4f}.",
        f"- Common source AUROC/AP/G: {float(common['P-AUROC']):.4f} / {float(common['P-AP']):.4f} / {float(common['G']):.4f}.",
        f"- Residual source AUROC/AP/G: {float(residual['P-AUROC']):.4f} / {float(residual['P-AP']):.4f} / {float(residual['G']):.4f}.",
        f"- corr(Delta,Common): {float(corr_dc):.4f}; corr(Delta,Residual): {float(corr_dr):.4f}; corr(Common,Residual): {float(corr_cr):.4f}.",
        f"- Residual CV: {decision['Residual_cv']:.6f}; collapsed={decision['residual_collapsed']}.",
        "",
        "## Conditional And LOO",
        "",
        f"- Conditional Common AUROC mean: {decision['Common_conditional_AUROC_mean']:.4f}.",
        f"- Conditional Residual AUROC mean: {decision['Residual_conditional_AUROC_mean']:.4f}.",
        f"- P3=[Delta,Common,Residual] improves P0=[Delta] in {int(loo['DELTA_P3_minus_P0_improved'])}/{int(loo['DELTA_P3_minus_P0_categories'])} categories.",
        f"- mean Delta G(P1-P0): {decision['mean_delta_G_P1_minus_P0']:+.4f}.",
        f"- mean Delta G(P2-P0): {decision['mean_delta_G_P2_minus_P0']:+.4f}.",
        f"- mean Delta G(P3-P0): {decision['mean_delta_G_P3_minus_P0']:+.4f}.",
        f"- mean Delta G(P3-Prand): {decision['mean_delta_G_P3_minus_Prand']:+.4f}.",
        "",
        "STOP: no target experiment, trainable MLP, fusion weight sweep, or retraining was started.",
        "",
        f"Runtime: {runtime:.1f}s. Smoke: {smoke}.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ("state_pair_scalars_smoke.npz" if args.smoke else "state_pair_scalars_formal.npz")
    if cache.exists() and not args.no_cache:
        z = np.load(cache, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        data["categories"] = data["categories"].astype(object)
        device = "cache"
    else:
        data, device = collect_state_pair(args.smoke, args.limit)
        np.savez_compressed(cache, **data)

    stats_rows = state_pair_stats(data)
    cond_rows = conditional_information(data)
    loo_rows = loo_probe(data)
    err_rows = error_complementarity(data)
    outcome, decision = decide(stats_rows, cond_rows, loo_rows, err_rows)
    runtime = time.time() - start

    write_csv(OUT / "STATE_PAIR_STATS.csv", stats_rows)
    write_csv(OUT / "CONDITIONAL_INFORMATION.csv", cond_rows)
    write_csv(OUT / "LOO_PROBE.csv", loo_rows)
    write_csv(OUT / "ERROR_COMPLEMENTARITY.csv", err_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision,
        "runtime_sec": runtime,
        "device": device,
        "no_training": True,
        "target_data_used": False,
        "source": "Full VisA",
        "checkpoint": "current formal Symmetric HardNormal",
        "smoke": args.smoke,
        "limit": args.limit,
    })
    write_report(outcome, decision, stats_rows, cond_rows, loo_rows, err_rows, runtime, args.smoke)
    print(f"[StatePairInfo] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
