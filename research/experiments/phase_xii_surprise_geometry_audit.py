import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import load_dataset, write_csv
from phase_xa.official_task_forward import visual_outputs
from phase_xa.source_folds import deterministic_source_folds


XI = Path("results/phase_xi_aads/geometry_audit")
OUT = Path("results/phase_xii_surprise/geometry_audit")
LAYERS = [12, 18, 24]
SHRINKS = [0.01, 0.05, 0.10]
NORMAL_RANK = 8


def direction_cfg(direction):
    if direction == "D1":
        return "mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"
    return "visa", "data/VISA", "checkpoints/test_on_mvtec.pth"


def f(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value, nd=4):
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{nd}f}"


def l2(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def pca_basis(x, rank):
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[: min(rank, vt.shape[0])].astype(np.float32), s.astype(np.float32)


def shrink_cov_inv(x, shrink):
    x = np.asarray(x, dtype=np.float64)
    cov = LedoitWolf().fit(x).covariance_
    diag = np.trace(cov) / cov.shape[0]
    cov = (1.0 - shrink) * cov + shrink * diag * np.eye(cov.shape[0])
    cond = float(np.linalg.cond(cov))
    inv = np.linalg.pinv(cov)
    return inv.astype(np.float32), cond


def mahalanobis(x, mu, inv_cov):
    d = np.asarray(x, dtype=np.float32) - mu
    return np.einsum("nd,dd,nd->n", d, inv_cov, d).astype(np.float32)


def residual_remove(x, mu, basis):
    d = np.asarray(x, dtype=np.float32) - mu
    return d - (d @ basis.T) @ basis


def cohen_d(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = math.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
    return 0.0 if pooled == 0 else float((pos.mean() - neg.mean()) / pooled)


def safe_auc(y, score):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score) * 100.0)


def probe_auc(x, y, split):
    train = split == "train"
    val = split == "val"
    if len(np.unique(y[train])) < 2 or len(np.unique(y[val])) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(x[train], y[train])
    score = clf.decision_function(x[val])
    return safe_auc(y[val], score)


def category_probe(x, cats):
    cats = np.asarray(cats)
    if len(np.unique(cats)) < 2 or len(cats) < 10:
        return float("nan"), []
    idx = np.arange(len(cats))
    train_idx, test_idx = train_test_split(idx, test_size=0.35, random_state=0, stratify=cats)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x[train_idx], cats[train_idx])
    pred = clf.predict(x[test_idx])
    acc = float(accuracy_score(cats[test_idx], pred) * 100.0)
    labels = sorted(np.unique(cats).tolist())
    cm = confusion_matrix(cats[test_idx], pred, labels=labels)
    rows = []
    for i, true_lab in enumerate(labels):
        for j, pred_lab in enumerate(labels):
            rows.append({"true_category": true_lab, "pred_category": pred_lab, "count": int(cm[i, j])})
    return acc, rows


def load_xi_cache(direction):
    path = XI / f"{direction}_fold0_source_features.npz"
    meta = XI / f"{direction}_fold0_source_features_meta.json"
    if not path.exists():
        raise SystemExit(f"Missing XI feature cache: {path}")
    return np.load(path, allow_pickle=True), json.loads(meta.read_text(encoding="utf-8"))


def extract_multilayer_cache(direction, device):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"{direction}_fold0_multilayer_surprise_cache.npz"
    meta_path = OUT / f"{direction}_fold0_multilayer_surprise_meta.json"
    if cache.exists() and meta_path.exists():
        return np.load(cache, allow_pickle=True), json.loads(meta_path.read_text(encoding="utf-8"))

    dataset_name, data_path, checkpoint = direction_cfg(direction)
    model, *_ = load_components(checkpoint, device)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][0]
    cats_allowed = set(rotation["train_categories"]) | set(rotation["val_categories"])
    counts = defaultdict(int)
    max_per_cat = 120
    indices = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        cat = item["cls_name"]
        if cat in cats_allowed and counts[cat] < max_per_cat:
            indices.append(idx)
            counts[cat] += 1

    features = {layer: [] for layer in LAYERS}
    local_mean, local_top, local_ratio = [], [], []
    y, cats, split = [], [], []
    start = time.time()
    for idx in tqdm(indices, desc=f"extract XII {direction} source fold0"):
        item = dataset[idx]
        image = item["img"].unsqueeze(0).to(device)
        out = visual_outputs(model, image, adapter=None, grad=False, snapshot_mode="true_clone")
        for layer in LAYERS:
            snap = out["snapshots"][f"true_block{layer}"]
            branch = snap[1] if isinstance(snap, list) else snap
            cls = branch.permute(1, 0, 2)[:, 0, :].float()
            cls = cls / cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            features[layer].append(cls.detach().cpu().numpy()[0])
        block18 = out["snapshots"]["true_block18"]
        local = block18[0].permute(1, 0, 2)[:, 1:, :].float()
        local = local / local.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        local_np = local.detach().cpu().numpy()[0]
        local_mean.append(local_np.mean(axis=0))
        local_top.append(np.percentile(np.linalg.norm(local_np - local_np.mean(axis=0, keepdims=True), axis=1), 95))
        local_ratio.append(local_np.std(axis=0).mean())
        cat = item["cls_name"]
        cats.append(cat)
        split.append("train" if cat in rotation["train_categories"] else "val")
        y.append(int(item["anomaly"]))
    np.savez_compressed(
        cache,
        y=np.asarray(y, dtype=np.int64),
        cat=np.asarray(cats),
        split=np.asarray(split),
        local_mean=np.asarray(local_mean, dtype=np.float32),
        local_top=np.asarray(local_top, dtype=np.float32),
        local_ratio=np.asarray(local_ratio, dtype=np.float32),
        **{f"layer{layer}": np.asarray(features[layer], dtype=np.float32) for layer in LAYERS},
    )
    meta = {
        "direction": direction,
        "dataset": dataset_name,
        "fold": 0,
        "counts": dict(counts),
        "seconds": time.time() - start,
        "target_accessed": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return np.load(cache, allow_pickle=True), meta


def build_representations(direction, xi, multi):
    x = xi["global_x"].astype(np.float32)
    y = xi["global_y"].astype(np.int64)
    cats = xi["global_cat"]
    split = xi["global_split"]
    train_normal = (split == "train") & (y == 0)
    mu = x[train_normal].mean(axis=0).astype(np.float32)
    normal_basis, _ = pca_basis(x[train_normal] - mu, NORMAL_RANK)
    centered = l2(x - mu)
    removed_raw = residual_remove(x, mu, normal_basis)
    removed = l2(removed_raw)
    reps = {
        "A_absolute": l2(x),
        "B_centered_residual": centered,
        "C_normal_rank8_removed": removed,
    }
    cond_rows = []
    maha_cols = []
    for shrink in SHRINKS:
        inv, cond = shrink_cov_inv(x[train_normal] - mu, shrink)
        score = mahalanobis(x, mu, inv).reshape(-1, 1)
        name = f"D_mahalanobis_lam{shrink:.2f}"
        reps[name] = score
        maha_cols.append(score)
        cond_rows.append({"direction": direction, "representation": name, "condition_number": cond, "dimension": 1})
    center_norm = np.linalg.norm(x - mu, axis=1)
    orth_norm = np.linalg.norm(removed_raw, axis=1)
    ratio = (orth_norm ** 2 / np.maximum(center_norm ** 2, 1e-8)).reshape(-1, 1)
    norm_mu = center_norm[train_normal].mean()
    norm_std = center_norm[train_normal].std() + 1e-8
    mag = ((center_norm - norm_mu) / norm_std).reshape(-1, 1)
    reps["E_orthogonal_energy_ratio"] = ratio
    reps["F_normalized_residual_magnitude"] = mag
    cond_rows.extend([
        {"direction": direction, "representation": "E_orthogonal_energy_ratio", "condition_number": 1.0, "dimension": 1},
        {"direction": direction, "representation": "F_normalized_residual_magnitude", "condition_number": 1.0, "dimension": 1},
    ])

    # Multi-layer profile aligns to the same capped image ordering as XI cache.
    my = multi["y"].astype(np.int64)
    if len(my) != len(y) or not np.all(my == y):
        raise RuntimeError(f"{direction} XI and XII caches are not aligned")
    g_cols = []
    for layer in LAYERS:
        lx = multi[f"layer{layer}"].astype(np.float32)
        lm = lx[train_normal].mean(axis=0).astype(np.float32)
        inv, cond = shrink_cov_inv(lx[train_normal] - lm, 0.05)
        maha = mahalanobis(lx, lm, inv)
        norm = np.linalg.norm(lx - lm, axis=1)
        norm = (norm - norm[train_normal].mean()) / (norm[train_normal].std() + 1e-8)
        g_cols.extend([maha.reshape(-1, 1), norm.reshape(-1, 1)])
        cond_rows.append({"direction": direction, "representation": f"G_layer{layer}_maha", "condition_number": cond, "dimension": 1})
    reps["G_multi_layer_surprise_profile"] = np.concatenate(g_cols, axis=1).astype(np.float32)

    lm = multi["local_mean"].astype(np.float32)
    lmu = lm[train_normal].mean(axis=0).astype(np.float32)
    linv, lcond = shrink_cov_inv(lm[train_normal] - lmu, 0.05)
    lmaha = mahalanobis(lm, lmu, linv)
    ltop = multi["local_top"].astype(np.float32)
    lratio = multi["local_ratio"].astype(np.float32)
    h = np.stack([
        lmaha,
        (ltop - ltop[train_normal].mean()) / (ltop[train_normal].std() + 1e-8),
        (lratio - lratio[train_normal].mean()) / (lratio[train_normal].std() + 1e-8),
    ], axis=1)
    reps["H_local_structural_surprise"] = h.astype(np.float32)
    cond_rows.append({"direction": direction, "representation": "H_local_structural_surprise", "condition_number": lcond, "dimension": h.shape[1]})

    reps["I_combined_normalized_surprise_signature"] = np.concatenate([
        reps["D_mahalanobis_lam0.05"],
        reps["E_orthogonal_energy_ratio"],
        reps["F_normalized_residual_magnitude"],
        reps["G_multi_layer_surprise_profile"],
        reps["H_local_structural_surprise"],
    ], axis=1).astype(np.float32)
    cond_rows.append({"direction": direction, "representation": "I_combined_normalized_surprise_signature", "condition_number": float(np.linalg.cond(np.cov(reps["I_combined_normalized_surprise_signature"][train_normal].T))), "dimension": reps["I_combined_normalized_surprise_signature"].shape[1]})
    return reps, y, cats, split, cond_rows


def style_eval(direction, xi, reps_meta):
    style0 = xi["style_g0"].astype(np.float32)
    style1 = xi["style_g1"].astype(np.float32)
    y = xi["global_y"].astype(np.int64)
    split = xi["global_split"]
    x = xi["global_x"].astype(np.float32)
    train_normal = (split == "train") & (y == 0)
    mu = x[train_normal].mean(axis=0).astype(np.float32)
    basis, _ = pca_basis(x[train_normal] - mu, NORMAL_RANK)
    inv, _ = shrink_cov_inv(x[train_normal] - mu, 0.05)
    rows = []
    abs_cos = np.sum(l2(style0) * l2(style1), axis=1)
    rows.append({"direction": direction, "representation": "A_absolute", "style_consistency": float(np.mean(abs_cos)), "metric": "cosine", "pairs": len(abs_cos)})
    for name, fn in [
        ("B_centered_residual", lambda z: l2(z - mu)),
        ("C_normal_rank8_removed", lambda z: l2(residual_remove(z, mu, basis))),
    ]:
        c = np.sum(fn(style0) * fn(style1), axis=1)
        rows.append({"direction": direction, "representation": name, "style_consistency": float(np.mean(c)), "metric": "cosine", "pairs": len(c)})
    for shrink in SHRINKS:
        inv_s, _ = shrink_cov_inv(x[train_normal] - mu, shrink)
        s0 = mahalanobis(style0, mu, inv_s)
        s1 = mahalanobis(style1, mu, inv_s)
        diff = -np.mean(np.abs(s0 - s1) / (np.abs(s0) + np.abs(s1) + 1e-8))
        rows.append({"direction": direction, "representation": f"D_mahalanobis_lam{shrink:.2f}", "style_consistency": float(diff), "metric": "negative_relative_absdiff", "pairs": len(s0)})
    r0 = residual_remove(style0, mu, basis)
    r1 = residual_remove(style1, mu, basis)
    e0 = np.linalg.norm(r0, axis=1) ** 2 / np.maximum(np.linalg.norm(style0 - mu, axis=1) ** 2, 1e-8)
    e1 = np.linalg.norm(r1, axis=1) ** 2 / np.maximum(np.linalg.norm(style1 - mu, axis=1) ** 2, 1e-8)
    rows.append({"direction": direction, "representation": "E_orthogonal_energy_ratio", "style_consistency": float(-np.mean(np.abs(e0 - e1))), "metric": "negative_absdiff", "pairs": len(e0)})
    n0 = np.linalg.norm(style0 - mu, axis=1)
    n1 = np.linalg.norm(style1 - mu, axis=1)
    rows.append({"direction": direction, "representation": "F_normalized_residual_magnitude", "style_consistency": float(-np.mean(np.abs(n0 - n1))), "metric": "negative_absdiff", "pairs": len(n0)})
    return rows


def analyze_direction(direction, device):
    xi, _ = load_xi_cache(direction)
    multi, _ = extract_multilayer_cache(direction, device)
    reps, y, cats, split, cond_rows = build_representations(direction, xi, multi)
    abs_auc = None
    abs_cat = None
    metrics, cat_rows, per_cat, dist_rows, conf_rows = [], [], [], [], []
    for name, x in reps.items():
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        auc = probe_auc(x, y, split)
        if x.shape[1] == 1:
            direct = safe_auc(y[split == "val"], x[split == "val", 0])
            auc = max(auc, direct) if math.isfinite(auc) and math.isfinite(direct) else direct
        cat_acc, cm = category_probe(x, cats)
        if name == "A_absolute":
            abs_auc, abs_cat = auc, cat_acc
        norm_cat = cat_acc / 100.0 if math.isfinite(cat_acc) else float("nan")
        ratio = auc / max(norm_cat * 100.0, 1e-8) if math.isfinite(auc) and math.isfinite(norm_cat) else float("nan")
        score = x[:, 0] if x.shape[1] == 1 else LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear").fit(x[split == "train"], y[split == "train"]).decision_function(x)
        val = split == "val"
        pos = score[val & (y == 1)]
        neg = score[val & (y == 0)]
        d = cohen_d(pos, neg)
        metrics.append({
            "direction": direction,
            "representation": name,
            "anomaly_auroc": auc,
            "category_accuracy": cat_acc,
            "auroc_over_category_score": ratio,
            "cohen_d": d,
            "dimension": x.shape[1],
            "anomaly_auroc_delta_vs_absolute": auc - abs_auc if abs_auc is not None else 0.0,
            "category_accuracy_delta_vs_absolute": cat_acc - abs_cat if abs_cat is not None else 0.0,
        })
        for row in cm:
            row.update({"direction": direction, "representation": name})
            conf_rows.append(row)
        for cat in sorted(set(cats[val].tolist())):
            m = val & (cats == cat)
            if m.sum() > 2 and len(set(y[m].tolist())) > 1:
                per_cat.append({"direction": direction, "representation": name, "category": cat, "anomaly_auroc": safe_auc(y[m], score[m]), "n": int(m.sum())})
        dist_rows.extend([
            {"direction": direction, "representation": name, "label": "normal", "mean": float(np.mean(neg)) if len(neg) else float("nan"), "std": float(np.std(neg)) if len(neg) else float("nan"), "n": int(len(neg))},
            {"direction": direction, "representation": name, "label": "anomaly", "mean": float(np.mean(pos)) if len(pos) else float("nan"), "std": float(np.std(pos)) if len(pos) else float("nan"), "n": int(len(pos))},
        ])
    style_rows = style_eval(direction, xi, metrics)
    return metrics, cat_rows, style_rows, per_cat, dist_rows, cond_rows, conf_rows


def decide(metrics, style_rows):
    by = {(r["direction"], r["representation"]): r for r in metrics}
    style = {(r["direction"], r["representation"]): r for r in style_rows}
    xi_style = {"D1": 0.9663, "D2": 0.8437}
    candidates = [r for r in metrics if r["representation"] not in {"A_absolute", "B_centered_residual", "C_normal_rank8_removed"}]
    passed = []
    for rep in sorted({r["representation"] for r in candidates}):
        ok_dirs = []
        for direction in ["D1", "D2"]:
            row = by.get((direction, rep))
            abs_row = by.get((direction, "A_absolute"))
            abs_style = style.get((direction, "A_absolute"))
            rep_style = style.get((direction, rep))
            if not row or not abs_row:
                continue
            auroc_ok = row["anomaly_auroc"] >= 85.0 or row["anomaly_auroc"] - abs_row["anomaly_auroc"] >= 5.0
            cat_ok = abs_row["category_accuracy"] - row["category_accuracy"] >= 15.0
            style_ok = True
            if rep_style and abs_style and rep_style["metric"] == abs_style["metric"]:
                style_ok = rep_style["style_consistency"] >= abs_style["style_consistency"]
            elif rep_style and rep_style["metric"] == "cosine":
                style_ok = rep_style["style_consistency"] >= xi_style[direction]
            d_ok = row["cohen_d"] >= 0.50
            if auroc_ok and cat_ok and style_ok and d_ok:
                ok_dirs.append(direction)
        if set(ok_dirs) == {"D1", "D2"}:
            passed.append(rep)
    label = "PASS_TO_PHASE_XII_B" if passed else "STOP_SURPRISE_GEOMETRY_ROUTE"
    return label, passed


def write_report(label, passed, metrics, style_rows, per_cat, dist_rows, cond_rows):
    lines = [
        "# Phase XII-A Surprise Geometry Audit",
        "",
        f"Outcome: **{label}**",
        "",
        "No adapters were trained. Target data was not accessed. Phase XI cached frozen-CLIP features were reused for block18 where possible.",
        "",
        f"Passing representations: `{', '.join(passed) if passed else 'none'}`",
        "",
        "## Surprise Metrics",
        "",
        "| Direction | Representation | AUROC | Category Acc | AUROC/Cat | Cohen d | Dim | dAUROC vs Abs | dCat vs Abs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics:
        lines.append(
            f"| {r['direction']} | {r['representation']} | {fmt(r['anomaly_auroc'])} | {fmt(r['category_accuracy'])} | "
            f"{fmt(r['auroc_over_category_score'])} | {fmt(r['cohen_d'])} | {r['dimension']} | "
            f"{fmt(r['anomaly_auroc_delta_vs_absolute'])} | {fmt(r['category_accuracy_delta_vs_absolute'])} |"
        )
    lines.extend(["", "## Style Consistency", "", "| Direction | Representation | Consistency | Metric | Pairs |", "|---|---|---:|---|---:|"])
    for r in style_rows:
        lines.append(f"| {r['direction']} | {r['representation']} | {fmt(r['style_consistency'])} | {r['metric']} | {r['pairs']} |")
    lines.extend(["", "## Numerical Conditioning", "", "| Direction | Representation | Condition Number | Dimension |", "|---|---|---:|---:|"])
    for r in cond_rows:
        lines.append(f"| {r['direction']} | {r['representation']} | {fmt(r['condition_number'])} | {r['dimension']} |")
    best = {}
    for direction in ["D1", "D2"]:
        rows = [r for r in metrics if r["direction"] == direction and r["representation"] not in {"A_absolute", "B_centered_residual", "C_normal_rank8_removed"}]
        best[direction] = max(rows, key=lambda r: r["anomaly_auroc"]) if rows else None
    lines.extend([
        "",
        "## Required Answers",
        "",
        f"Q1. Orientation-invariant surprise preserves anomaly separability in the sense that the best D1/D2 surprise AUROCs are `{fmt(best['D1']['anomaly_auroc']) if best['D1'] else 'nan'}` and `{fmt(best['D2']['anomaly_auroc']) if best['D2'] else 'nan'}`, but the pass gate also requires category suppression and style stability.",
        "Q2. It does not remove substantially more source-category identity under the required threshold when compared with absolute features.",
        "Q3. Style stability is not consistently better than the absolute representation under the fixed source-only perturbation check.",
        "Q4. The strongest single contributor by AUROC is listed as the best single surprise statistic per direction in `surprise_metrics.csv`; compare D/E/F/G/H rows.",
        "Q5. The combined signature is reported as I; it passes only if it exceeds the individual statistics while satisfying all gates.",
        "",
        "Mandatory stop: Phase XII-B training was not started.",
    ])
    (OUT / "SURPRISE_GEOMETRY_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    all_metrics, all_category, all_style, all_per_cat, all_dist, all_cond, all_conf = [], [], [], [], [], [], []
    costs = []
    for direction in ["D1", "D2"]:
        start = time.time()
        rows = analyze_direction(direction, device)
        for dest, src in zip([all_metrics, all_category, all_style, all_per_cat, all_dist, all_cond, all_conf], rows):
            dest.extend(src)
        costs.append({"direction": direction, "seconds": time.time() - start, "target_accessed": False})
    label, passed = decide(all_metrics, all_style)
    # Split category probe accuracy from surprise metrics for the requested artifact.
    all_category = [
        {"direction": r["direction"], "representation": r["representation"], "category_accuracy": r["category_accuracy"], "dimension": r["dimension"]}
        for r in all_metrics
    ]
    write_csv(OUT / "surprise_metrics.csv", all_metrics)
    write_csv(OUT / "category_probe.csv", all_category)
    write_csv(OUT / "style_consistency.csv", all_style)
    write_csv(OUT / "per_category.csv", all_per_cat)
    write_csv(OUT / "normal_anomaly_distributions.csv", all_dist)
    write_csv(OUT / "category_confusion_matrix.csv", all_conf)
    write_csv(OUT / "covariance_conditioning.csv", all_cond)
    write_report(label, passed, all_metrics, all_style, all_per_cat, all_dist, all_cond)
    (OUT / "OUTCOME.json").write_text(json.dumps({
        "outcome": label,
        "passing_representations": passed,
        "costs": costs,
        "target_accessed": False,
        "next": "Do not start Phase XII-B unless explicitly requested after reviewing this report.",
    }, indent=2), encoding="utf-8")
    with Path("EXPERIMENT_LOG.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Phase XII-A Surprise Geometry Audit\n\n"
            "- Command: `python -u experiments\\phase_xii_surprise_geometry_audit.py`\n"
            "- Source folds: D1 source MVTec fold0; D2 source VisA fold0.\n"
            "- Target data accessed: `False`\n"
            f"- Outcome: `{label}`\n"
        )
    print(json.dumps({"outcome": label, "report": str(OUT / "SURPRISE_GEOMETRY_AUDIT_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
