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
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np


OUT = Path("results/phase_radial_angular_complementarity_audit/formal")
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


def metric_row(name, y, s):
    pa = safe_auc(y, s)
    pap = safe_ap(y, s)
    return {"scope": "source", "score": name, "P-AUROC": pa, "P-AP": pap, "G": 0.5 * (pa + pap)}


def stats(values):
    x = np.asarray(values, dtype=np.float64)
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
        "cv": float(x.std() / max(EPS, abs(mean))),
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


def implementation_audit():
    return [
        {"item": "pre_normalized_feature_h", "value": "h_i = x_i + B(Ax_i), where x_i is PC1-removed and L2-normalized patch feature before rank4 alignment output normalization"},
        {"item": "normalized_feature_u", "value": "u_i = normalize(h_i)"},
        {"item": "rho", "value": "rho_i = ||h_i||_2"},
        {"item": "current_adapter_return", "value": "AdapterToState.adapt returns F.normalize(x + self.b(self.a(x)), dim=-1)"},
        {"item": "current_scorer", "value": "contrast_logits normalizes states and computes cosine anomaly-normal contrast from u_i"},
        {"item": "rho_used_by_current_scorer", "value": False},
    ]


def extract_scalars(dataset, indices, patch, q_pc1, align_model, m_c, alpha, v_hn, device, smoke):
    slots = source_slots_by_category(dataset, indices, smoke)
    allowed_positions = sorted({p for positions in slots.values() for p in positions})
    if smoke:
        allowed_positions = allowed_positions[:96]
    rng = np.random.default_rng(0)
    random_v = rng.normal(size=(1024,)).astype(np.float32)
    random_v /= max(float(np.linalg.norm(random_v)), EPS)
    rows = []
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    v = torch.tensor(v_hn, dtype=torch.float32, device=device)
    rv = torch.tensor(random_v, dtype=torch.float32, device=device)
    for pos in allowed_positions:
        item = dataset[int(indices[int(pos)])]
        cat = str(item["cls_name"])
        y = np.asarray(labels_by_pos.get(int(pos), np.zeros((PATCH_GRID,), dtype=np.float32)), dtype=np.float32)
        x0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
        x0 = norm_np(x0.reshape(PATCH_GRID, 1024))
        with torch.no_grad():
            x = torch.tensor(x0, dtype=torch.float32, device=device)
            h = x + align_model.b(align_model.a(x))
            rho = torch.linalg.norm(h, dim=-1)
            u = F.normalize(h, dim=-1)
            score = contrast_logits(u, m, alpha, F.normalize(v, dim=-1))
            rnd = h @ rv
        rows.append({
            "cat": np.full((PATCH_GRID,), cat, dtype=object),
            "y": y.astype(np.uint8),
            "score": score.detach().cpu().numpy().astype(np.float32),
            "rho": rho.detach().cpu().numpy().astype(np.float32),
            "rand": rnd.detach().cpu().numpy().astype(np.float32),
        })
    cat = np.concatenate([r["cat"] for r in rows], axis=0)
    y = np.concatenate([r["y"] for r in rows], axis=0).astype(np.uint8)
    score = np.concatenate([r["score"] for r in rows], axis=0).astype(np.float32)
    rho = np.concatenate([r["rho"] for r in rows], axis=0).astype(np.float32)
    rand = np.concatenate([r["rand"] for r in rows], axis=0).astype(np.float32)
    log_rho = np.log(np.maximum(rho, EPS)).astype(np.float32)
    sign = 1.0 if float(log_rho[y > 0].mean() - log_rho[y == 0].mean()) >= 0 else -1.0
    return {"category": cat, "y": y, "S_H": score, "rho": rho, "logrho": log_rho, "R_signed": (sign * log_rho).astype(np.float32), "R_random": rand}


def radial_stats_table(data):
    rows = []
    for label_name, mask in (("normal", data["y"] == 0), ("anomaly", data["y"] > 0), ("all", np.ones_like(data["y"], dtype=bool))):
        for name in ("rho", "logrho"):
            rows.append({"group": label_name, "variable": name, **stats(data[name][mask])})
    rows.extend([
        metric_row("HardNormal_S_H", data["y"], data["S_H"]),
        metric_row("Radial_R_signed", data["y"], data["R_signed"]),
        {"scope": "source", "score": "corr_S_H_R_signed", "P-AUROC": float("nan"), "P-AP": float("nan"), "G": corr(data["S_H"], data["R_signed"])},
    ])
    return rows


def conditional_table(data):
    order = np.argsort(data["S_H"], kind="mergesort")
    bins = np.array_split(order, 20)
    rows = []
    aucs = []
    for i, ids in enumerate(bins):
        y = data["y"][ids]
        r = data["R_signed"][ids]
        if np.unique(y).size < 2:
            auc = float("nan")
            ap = float("nan")
            diff = float("nan")
        else:
            auc = safe_auc(y, r)
            ap = safe_ap(y, r)
            diff = float(r[y > 0].mean() - r[y == 0].mean())
            aucs.append(auc)
        rows.append({
            "bin": i,
            "count": int(ids.size),
            "S_H_min": float(data["S_H"][ids].min()),
            "S_H_max": float(data["S_H"][ids].max()),
            "anomaly_ratio": float(y.mean()),
            "mean_R_signed_anomaly_minus_normal": diff,
            "radial_AUROC": auc,
            "radial_AP": ap,
        })
    arr = np.asarray([x for x in aucs if np.isfinite(x)], dtype=np.float64)
    rows.insert(0, {
        "bin": "__summary__",
        "count": int(arr.size),
        "S_H_min": float("nan"),
        "S_H_max": float("nan"),
        "anomaly_ratio": float("nan"),
        "mean_R_signed_anomaly_minus_normal": float("nan"),
        "radial_AUROC": float(arr.mean()) if arr.size else float("nan"),
        "radial_AP": float(arr.std()) if arr.size else float("nan"),
        "radial_AUROC_std": float(arr.std()) if arr.size else float("nan"),
    })
    return rows


def standardize(train_x, test_x):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    return (train_x - mu) / sd, (test_x - mu) / sd


def fit_probe(train_x, train_y, test_x):
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=0)
    clf.fit(train_x, train_y)
    return clf.predict_proba(test_x)[:, 1]


def loo_probe(data):
    cats = sorted(set(data["category"].tolist()))
    rows = []
    for cat in cats:
        test = data["category"] == cat
        train = ~test
        y_train = data["y"][train]
        y_test = data["y"][test]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        x0_train = data["S_H"][train].reshape(-1, 1)
        x0_test = data["S_H"][test].reshape(-1, 1)
        x1_train = np.stack([data["S_H"][train], data["logrho"][train]], axis=1)
        x1_test = np.stack([data["S_H"][test], data["logrho"][test]], axis=1)
        x2_train = np.stack([data["S_H"][train], data["R_random"][train]], axis=1)
        x2_test = np.stack([data["S_H"][test], data["R_random"][test]], axis=1)
        variants = {
            "P0_SH": (x0_train, x0_test),
            "P1_SH_plus_logrho": (x1_train, x1_test),
            "P2_SH_plus_random_scalar": (x2_train, x2_test),
        }
        metrics = {}
        for name, (tr, te) in variants.items():
            tr, te = standardize(tr.astype(np.float64), te.astype(np.float64))
            pred = fit_probe(tr, y_train, te)
            pa = safe_auc(y_test, pred)
            pap = safe_ap(y_test, pred)
            metrics[name] = {"P-AUROC": pa, "P-AP": pap, "G": 0.5 * (pa + pap)}
            rows.append({"heldout_category": cat, "variant": name, **metrics[name]})
        rows.append({
            "heldout_category": cat,
            "variant": "DELTA_P1_minus_P0",
            "P-AUROC": metrics["P1_SH_plus_logrho"]["P-AUROC"] - metrics["P0_SH"]["P-AUROC"],
            "P-AP": metrics["P1_SH_plus_logrho"]["P-AP"] - metrics["P0_SH"]["P-AP"],
            "G": metrics["P1_SH_plus_logrho"]["G"] - metrics["P0_SH"]["G"],
        })
        rows.append({
            "heldout_category": cat,
            "variant": "DELTA_P1_minus_P2",
            "P-AUROC": metrics["P1_SH_plus_logrho"]["P-AUROC"] - metrics["P2_SH_plus_random_scalar"]["P-AUROC"],
            "P-AP": metrics["P1_SH_plus_logrho"]["P-AP"] - metrics["P2_SH_plus_random_scalar"]["P-AP"],
            "G": metrics["P1_SH_plus_logrho"]["G"] - metrics["P2_SH_plus_random_scalar"]["G"],
        })
    d10 = np.asarray([r["G"] for r in rows if r["variant"] == "DELTA_P1_minus_P0"], dtype=np.float64)
    d12 = np.asarray([r["G"] for r in rows if r["variant"] == "DELTA_P1_minus_P2"], dtype=np.float64)
    summary = {
        "heldout_category": "__summary__",
        "variant": "summary",
        "categories": int(d10.size),
        "P1_improved_over_P0_categories": int(np.sum(d10 > 0)),
        "P1_improved_fraction": float(np.mean(d10 > 0)) if d10.size else float("nan"),
        "mean_delta_G_P1_minus_P0": float(d10.mean()) if d10.size else float("nan"),
        "median_delta_G_P1_minus_P0": float(np.median(d10)) if d10.size else float("nan"),
        "mean_delta_G_P1_minus_P2": float(d12.mean()) if d12.size else float("nan"),
        "median_delta_G_P1_minus_P2": float(np.median(d12)) if d12.size else float("nan"),
    }
    return [summary] + rows


def error_complementarity(data):
    y = data["y"]
    s = data["S_H"]
    r = data["R_signed"]
    normal_s = s[y == 0]
    anomaly_s = s[y > 0]
    n95 = np.percentile(normal_s, 95)
    a20 = np.percentile(anomaly_s, 20)
    fp = (y == 0) & (s >= n95)
    correct_n = (y == 0) & (s < n95)
    fn = (y > 0) & (s <= a20)
    correct_a = (y > 0) & (s > a20)
    groups = {
        "FP_H_normal_above_normal_P95": fp,
        "correct_normal": correct_n,
        "FN_H_anomaly_below_anomaly_P20": fn,
        "correct_anomaly": correct_a,
    }
    rows = []
    for name, mask in groups.items():
        rows.append({"group": name, **stats(r[mask])})
    rows.append({"group": "delta_FP_minus_correct_normal", "mean": float(r[fp].mean() - r[correct_n].mean()), "count": int(fp.sum())})
    rows.append({"group": "delta_FN_minus_correct_anomaly", "mean": float(r[fn].mean() - r[correct_a].mean()), "count": int(fn.sum())})
    return rows


def decide(radial_rows, loo_rows):
    all_rho = next(r for r in radial_rows if r.get("group") == "all" and r.get("variable") == "rho")
    cv = float(all_rho["cv"])
    if cv < 0.01:
        return "RADIAL_SIGNAL_COLLAPSED_BY_NORMALIZATION", {"rho_cv": cv}
    corr_row = next(r for r in radial_rows if r.get("score") == "corr_S_H_R_signed")
    c = float(corr_row["G"])
    summary = next(r for r in loo_rows if r.get("heldout_category") == "__summary__")
    improved = float(summary["P1_improved_fraction"])
    d10 = float(summary["mean_delta_G_P1_minus_P0"])
    d12 = float(summary["mean_delta_G_P1_minus_P2"])
    if cv >= 0.01 and abs(c) <= 0.8 and improved >= (2.0 / 3.0) and d10 >= 0.5 and d12 >= 0.3:
        outcome = "RADIAL_EVIDENCE_HAS_COMPLEMENTARY_INFORMATION"
    elif d10 < 0.5 and abs(c) > 0.8:
        outcome = "RADIAL_SIGNAL_REDUNDANT_WITH_HARDNORMAL"
    elif abs(d12) < 0.3:
        outcome = "RADIAL_SIGNAL_NOT_SPECIFIC"
    else:
        outcome = "RADIAL_EVIDENCE_NO_SIGNAL"
    return outcome, {
        "rho_cv": cv,
        "corr_S_H_R_signed": c,
        "P1_improved_fraction": improved,
        "mean_delta_G_P1_minus_P0": d10,
        "mean_delta_G_P1_minus_P2": d12,
    }


def write_report(outcome, stats_dict, radial_rows, conditional_rows, loo_rows, err_rows, runtime, smoke):
    def find_metric(score):
        return next((r for r in radial_rows if r.get("score") == score), {})
    hard = find_metric("HardNormal_S_H")
    radial = find_metric("Radial_R_signed")
    loo = next((r for r in loo_rows if r.get("heldout_category") == "__summary__"), {})
    cond = next((r for r in conditional_rows if r.get("bin") == "__summary__"), {})
    fp_delta = next((r for r in err_rows if r.get("group") == "delta_FP_minus_correct_normal"), {})
    fn_delta = next((r for r in err_rows if r.get("group") == "delta_FN_minus_correct_anomaly"), {})
    lines = [
        "# Radial-Angular Complementarity Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN / NO-TARGET audit on Full VisA source with the current formal HardNormal checkpoint/pipeline.",
        "",
        "## Implementation Audit",
        "",
        "- The pre-normalized feature is `h_i = x_i + B(Ax_i)`.",
        "- The current scorer uses `u_i = normalize(h_i)` and cosine contrast only; `rho_i = ||h_i||` is not explicitly used.",
        "",
        "## Radial Signal",
        "",
        f"- rho CV: {stats_dict.get('rho_cv', float('nan')):.6f}.",
        f"- HardNormal source G: {float(hard.get('G', float('nan'))):.4f}; radial-only source G: {float(radial.get('G', float('nan'))):.4f}.",
        f"- corr(S_H, R_signed): {stats_dict.get('corr_S_H_R_signed', float('nan')):.6f}.",
        "",
        "## Conditional Complementarity",
        "",
        f"- mean within-bin radial AUROC over S_H quantile bins: {float(cond.get('radial_AUROC', float('nan'))):.4f}.",
        f"- std within-bin radial AUROC: {float(cond.get('radial_AUROC_std', float('nan'))):.4f}.",
        "",
        "## LOO Probe",
        "",
        f"- categories improved P1 over P0: {int(loo.get('P1_improved_over_P0_categories', 0))}/{int(loo.get('categories', 0))}.",
        f"- mean Delta G(P1-P0): {float(loo.get('mean_delta_G_P1_minus_P0', float('nan'))):+.4f}.",
        f"- mean Delta G(P1-P2): {float(loo.get('mean_delta_G_P1_minus_P2', float('nan'))):+.4f}.",
        "",
        "## Error Complementarity",
        "",
        f"- R_signed(FP_H) - R_signed(correct normal): {float(fp_delta.get('mean', float('nan'))):+.6f}.",
        f"- R_signed(FN_H) - R_signed(correct anomaly): {float(fn_delta.get('mean', float('nan'))):+.6f}.",
        "",
        f"Runtime: {runtime:.1f}s. Smoke: {smoke}.",
        "",
        "STOP: no training was started.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RadialAudit] device={device}", flush=True)
    write_csv(OUT / "IMPLEMENTATION_AUDIT.csv", implementation_audit())
    q_pc1, align_model, m_c, alpha, v_hn = load_context(device)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    data = extract_scalars(dataset, indices, patch, q_pc1, align_model, m_c, alpha, v_hn, device, args.smoke)
    radial_rows = radial_stats_table(data)
    write_csv(OUT / "RADIAL_STATS.csv", radial_rows)
    all_rho = next(r for r in radial_rows if r.get("group") == "all" and r.get("variable") == "rho")
    if float(all_rho["cv"]) < 0.01:
        outcome, stats_dict = decide(radial_rows, [{"heldout_category": "__summary__", "P1_improved_fraction": float("nan"), "mean_delta_G_P1_minus_P0": float("nan"), "mean_delta_G_P1_minus_P2": float("nan")}])
        conditional_rows = []
        loo_rows = []
        err_rows = []
    else:
        conditional_rows = conditional_table(data)
        loo_rows = loo_probe(data)
        err_rows = error_complementarity(data)
        write_csv(OUT / "CONDITIONAL_COMPLEMENTARITY.csv", conditional_rows)
        write_csv(OUT / "LOO_PROBE.csv", loo_rows)
        write_csv(OUT / "ERROR_COMPLEMENTARITY.csv", err_rows)
        outcome, stats_dict = decide(radial_rows, loo_rows)
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats_dict,
        "runtime_sec": runtime,
        "device": str(device),
        "no_training": True,
        "target_data_used": False,
        "source": "Full VisA",
        "smoke": bool(args.smoke),
    })
    write_report(outcome, stats_dict, radial_rows, conditional_rows, loo_rows, err_rows, runtime, bool(args.smoke))
    print(f"[RadialAudit] OUTCOME={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
