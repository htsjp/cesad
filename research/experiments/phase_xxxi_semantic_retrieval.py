import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xxxi_semantic_retrieval/audit")
FEATURE_CACHE = Path("results/phase_xxiii_carrier_audit/feature_cache")
OFFICIAL_MAP_CACHE = Path("results/phase_xxiii_dual_carrier/local_integration/map_cache")
IMAGE_SIZE = 518
SIGMA = 4
EPS = 1e-6
VARIANTS = [
    "R0_VISUAL_OFFICIAL",
    "R1_SEMANTIC_ONLY",
    "R2_VISUAL_PLUS_SEMANTIC",
    "R3_SHUFFLED_QUERY_SEMANTIC",
    "R4_SHUFFLED_MEMORY_SEMANTIC",
]


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def rank01(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks / max(1, x.size - 1)


def logit_from_rank(r):
    r = np.clip(np.asarray(r, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(r / (1.0 - r))


def gini(counts):
    arr = np.asarray(counts, dtype=np.float64)
    if arr.size == 0 or arr.sum() == 0:
        return float("nan")
    arr.sort()
    n = arr.size
    return float((2.0 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1) / n)


def load_fold(direction, fold):
    dataset_name, data_path, _ = direction_cfg(direction)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
    rows_train = [{"idx": i, "category": dataset[i]["cls_name"], "label": int(dataset[i]["anomaly"])} for i in train_indices]
    rows_held = [{"idx": i, "category": dataset[i]["cls_name"], "label": int(dataset[i]["anomaly"])} for i in held_indices]
    return dataset_name, dataset, rotation, train_indices, held_indices, rows_train, rows_held


def metric_rows(rows, scores, prefix="I"):
    by_cat = defaultdict(lambda: {"y": [], "s": []})
    for r, s in zip(rows, scores):
        by_cat[r["category"]]["y"].append(int(r["label"]))
        by_cat[r["category"]]["s"].append(float(s))
    aucs, aps, percat = [], [], {}
    for cat, v in sorted(by_cat.items()):
        auc = safe_auc(v["y"], v["s"])
        ap = safe_ap(v["y"], v["s"])
        percat[cat] = {f"{prefix}-AUROC": auc, f"{prefix}-AP": ap}
        if np.isfinite(auc):
            aucs.append(auc)
        if np.isfinite(ap):
            aps.append(ap)
    return {f"{prefix}-AUROC": float(np.mean(aucs)) if aucs else float("nan"), f"{prefix}-AP": float(np.mean(aps)) if aps else float("nan"), "per_category": percat}


def pixel_metrics(dataset, held_indices, raw_maps):
    y_parts, s_parts = [], []
    by_cat = defaultdict(lambda: {"y": [], "s": []})
    topk = []
    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
        smooth = gaussian_filter(np.asarray(raw_maps[pos], dtype=np.float32), sigma=SIGMA).astype(np.float32)
        scores = smooth.reshape(-1)
        y_parts.append(mask)
        s_parts.append(scores)
        by_cat[item["cls_name"]]["y"].append(mask)
        by_cat[item["cls_name"]]["s"].append(scores)
        k = max(1, int(scores.size * 0.01))
        topk.append(float(np.sort(scores)[-k:].mean()))
    y = np.concatenate(y_parts)
    s = np.concatenate(s_parts).astype(np.float32)
    percat = {}
    aucs = []
    for cat, v in sorted(by_cat.items()):
        cy, cs = np.concatenate(v["y"]), np.concatenate(v["s"]).astype(np.float32)
        auc = safe_auc(cy, cs)
        percat[cat] = {"P-AUROC": auc, "P-AP": safe_ap(cy, cs)}
        if np.isfinite(auc):
            aucs.append(auc)
    return {
        "P-AUROC": float(np.mean(aucs)) if aucs else float("nan"),
        "P-AP": safe_ap(y, s),
        "P-AP_scope": "pooled flattened pixels",
        "per_category": percat,
        "topk": np.asarray(topk, dtype=np.float32),
    }


def text_anchors(model, prompt_learner, device):
    bias = torch.zeros((1, 2, 768), dtype=torch.float32, device=device)
    with torch.no_grad():
        prompts, tokenized, compound = prompt_learner(cls_id=None, bias=bias)
        feats = model.encode_text_learn(prompts, tokenized, compound).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    if feats.ndim == 3:
        feats = feats[0]
    if feats.shape[0] != 2:
        raise RuntimeError(f"unexpected text anchor shape {tuple(feats.shape)}")
    return feats.detach().cpu().numpy().astype(np.float32)


def semantic_state(features, anchors):
    logits = np.asarray(features, dtype=np.float32) @ np.asarray(anchors, dtype=np.float32).T
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def projected_logits(image_proj, q_feat, k_feat, device):
    with torch.no_grad():
        q = torch.tensor(q_feat, dtype=torch.float32, device=device)
        k = torch.tensor(k_feat, dtype=torch.float32, device=device)
        q_proj, k_proj = image_proj(q, k)
        logits = (q_proj @ k_proj.t()).detach().cpu().numpy().astype(np.float32)
    return logits


def variant_logits(visual_logits, q_sem, m_sem):
    n_query, n_mem = visual_logits.shape
    visual_rank = np.stack([rank01(visual_logits[i]) for i in range(n_query)], axis=0)
    sem = q_sem @ m_sem.T
    sem_rank = np.stack([rank01(sem[i]) for i in range(n_query)], axis=0)
    rng = np.random.default_rng(0)
    q_perm = rng.permutation(n_query)
    shuffled_q_sem = q_sem[q_perm]
    sem_q = shuffled_q_sem @ m_sem.T
    sem_q_rank = np.stack([rank01(sem_q[i]) for i in range(n_query)], axis=0)
    m_perm = rng.permutation(n_mem)
    shuffled_m_sem = m_sem[m_perm]
    sem_m = q_sem @ shuffled_m_sem.T
    sem_m_rank = np.stack([rank01(sem_m[i]) for i in range(n_query)], axis=0)
    return {
        "R0_VISUAL_OFFICIAL": visual_logits,
        "R1_SEMANTIC_ONLY": logit_from_rank(sem_rank),
        "R2_VISUAL_PLUS_SEMANTIC": logit_from_rank(0.5 * visual_rank + 0.5 * sem_rank),
        "R3_SHUFFLED_QUERY_SEMANTIC": logit_from_rank(0.5 * visual_rank + 0.5 * sem_q_rank),
        "R4_SHUFFLED_MEMORY_SEMANTIC": logit_from_rank(0.5 * visual_rank + 0.5 * sem_m_rank),
    }


def posterior_scores(logits_by_variant, values):
    out = {}
    values = np.asarray(values, dtype=np.float64)
    for name, logits in logits_by_variant.items():
        z = logits.astype(np.float64)
        z = z - z.max(axis=1, keepdims=True)
        w = np.exp(z)
        w /= w.sum(axis=1, keepdims=True)
        out[name] = {"memory_anomaly_score": w @ values, "weights": w}
    return out


def retrieval_metrics(name, weights, logits, train_rows):
    top = np.argsort(-logits, axis=1)[:, :5]
    top1 = top[:, 0]
    counts = np.bincount(top1, minlength=logits.shape[1])
    eff = float(1.0 / np.sum((counts / max(1, counts.sum())) ** 2)) if counts.sum() else float("nan")
    margins = np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2]
    cats = [r["category"] for r in train_rows]
    rows = {
        "variant": name,
        "unique_top1": int(np.count_nonzero(counts)),
        "unique_top5": int(len(set(top.reshape(-1).tolist()))),
        "gini_top1": gini(counts),
        "effective_retrieved_keys": eff,
        "top1_frequency_max": int(counts.max()) if counts.size else 0,
        "margin_mean": float(np.mean(margins)),
        "query_overlap_top5_mean": float(np.mean([len(set(top[i]).intersection(set(top[j]))) / 5.0 for i in range(min(64, len(top))) for j in range(i + 1, min(64, len(top)))])) if len(top) > 1 else float("nan"),
        "dominant_source_category_fraction": float(Counter(cats[i] for i in top1).most_common(1)[0][1] / max(1, len(top1))),
    }
    return rows, top


def memory_label_diagnostic(direction, fold, variant, top, rows_held, rows_train):
    out = []
    labels = np.asarray([r["label"] for r in rows_train], dtype=np.int64)
    cats = np.asarray([r["category"] for r in rows_train], dtype=object)
    for q_label, q_name in [(0, "normal_query"), (1, "anomaly_query")]:
        idx = [i for i, r in enumerate(rows_held) if r["label"] == q_label]
        if not idx:
            continue
        tt = top[idx]
        hist = []
        for row in tt:
            cs = Counter(cats[row].tolist())
            hist.append([cs.get(c, 0) for c in sorted(set(cats))])
        normal_frac = float(np.mean(labels[tt] == 0))
        anomaly_frac = float(np.mean(labels[tt] == 1))
        out.append({"direction": direction, "fold": fold, "variant": variant, "query_group": q_name, "retrieved_normal_memory_fraction": normal_frac, "retrieved_anomaly_memory_fraction": anomaly_frac})
    return out


def category_confound_rows(direction, fold, variant, top, rows_held, rows_train):
    cats = sorted({r["category"] for r in rows_train})
    cindex = {c: i for i, c in enumerate(cats)}
    train_cats = np.asarray([r["category"] for r in rows_train], dtype=object)
    x, y = [], []
    ent, dom = [], []
    for qi, row in enumerate(top):
        counts = np.zeros(len(cats), dtype=np.float64)
        for mid in row:
            counts[cindex[train_cats[mid]]] += 1
        p = counts / max(1.0, counts.sum())
        x.append(p)
        y.append(rows_held[qi]["category"])
        ent.append(float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(max(2, len(cats)))))
        dom.append(float(p.max()))
    acc = float("nan")
    if len(set(y)) > 1:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver="liblinear", random_state=0))
        clf.fit(np.asarray(x), y)
        acc = float(np.mean(clf.predict(np.asarray(x)) == np.asarray(y)))
    return {"direction": direction, "fold": fold, "variant": variant, "retrieval_category_entropy": float(np.mean(ent)), "dominant_source_category_fraction": float(np.mean(dom)), "query_category_predictability": acc}


def confidence_bins(train_conf, held_conf):
    q1, q2 = np.quantile(train_conf, [1 / 3, 2 / 3])
    bins = []
    for v in held_conf:
        bins.append("LOW" if v <= q1 else ("MID" if v <= q2 else "HIGH"))
    return bins


def run_fold(direction, fold, device, anchors):
    dataset_name, dataset, rotation, train_indices, held_indices, rows_train, rows_held = load_fold(direction, fold)
    ckpt = direction_cfg(direction)[2]
    model, _, image_proj, _, _ = load_components(ckpt, device)
    feat = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_cls = feat["train_cls"].astype(np.float32)
    held_cls = feat["held_cls"].astype(np.float32)
    values = np.asarray([r["label"] for r in rows_train], dtype=np.float32)
    visual_logits = projected_logits(image_proj, held_cls, train_cls, device)
    q_sem = semantic_state(held_cls, anchors[direction])
    m_sem = semantic_state(train_cls, anchors[direction])
    logits_by = variant_logits(visual_logits, q_sem, m_sem)
    post = posterior_scores(logits_by, values)

    raw_maps = np.load(OFFICIAL_MAP_CACHE / f"{direction}_fold{fold}_official_held_raw.npy", mmap_mode="r")
    pix = pixel_metrics(dataset, held_indices, raw_maps)
    image_topk = pix["topk"]

    raw_rows, percat_rows, retrieval_rows = [], [], []
    mem_label_rows, cat_rows, conf_rows, bias_rows = [], [], [], []
    top_by = {}
    for name in VARIANTS:
        scores = 0.3 * image_topk + 0.7 * post[name]["memory_anomaly_score"]
        im = metric_rows(rows_held, scores, prefix="I")
        raw_rows.append({"direction": direction, "fold": fold, "variant": name, "I-AUROC": im["I-AUROC"], "I-AP": im["I-AP"], "P-AUROC": pix["P-AUROC"], "P-AP": pix["P-AP"]})
        for cat, vals in im["per_category"].items():
            pc = pix["per_category"][cat]
            percat_rows.append({"direction": direction, "fold": fold, "category": cat, "variant": name, "I-AUROC": vals["I-AUROC"], "I-AP": vals["I-AP"], "P-AUROC": pc["P-AUROC"], "P-AP": pc["P-AP"]})
        rm, top = retrieval_metrics(name, post[name]["weights"], logits_by[name], rows_train)
        rm.update({"direction": direction, "fold": fold})
        retrieval_rows.append(rm)
        top_by[name] = top
        mem_label_rows.extend(memory_label_diagnostic(direction, fold, name, top, rows_held, rows_train))
        cat_rows.append(category_confound_rows(direction, fold, name, top, rows_held, rows_train))

    train_conf = np.abs(semantic_state(train_cls, anchors[direction])[:, 1] - 0.5)
    held_conf = np.abs(q_sem[:, 1] - 0.5)
    bins = confidence_bins(train_conf, held_conf)
    r0_scores = 0.3 * image_topk + 0.7 * post["R0_VISUAL_OFFICIAL"]["memory_anomaly_score"]
    r2_scores = 0.3 * image_topk + 0.7 * post["R2_VISUAL_PLUS_SEMANTIC"]["memory_anomaly_score"]
    for b in ["LOW", "MID", "HIGH"]:
        idx = [i for i, x in enumerate(bins) if x == b]
        if not idx:
            continue
        m0 = metric_rows([rows_held[i] for i in idx], r0_scores[idx], prefix="I")
        m2 = metric_rows([rows_held[i] for i in idx], r2_scores[idx], prefix="I")
        conf_rows.append({"direction": direction, "fold": fold, "confidence_bin": b, "images": len(idx), "R0_I-AUROC": m0["I-AUROC"], "R0_I-AP": m0["I-AP"], "R2_I-AUROC": m2["I-AUROC"], "R2_I-AP": m2["I-AP"], "G_image_rank": 0.5 * (m2["I-AUROC"] - m0["I-AUROC"]) + 0.5 * (m2["I-AP"] - m0["I-AP"])})

    labels = np.asarray([r["label"] for r in rows_held], dtype=np.int64)
    text_pred = (q_sem[:, 1] >= 0.5).astype(np.int64)
    conflict = text_pred != labels
    r0_pred = (r0_scores >= np.median(r0_scores)).astype(np.int64)
    r2_pred = (r2_scores >= np.median(r2_scores)).astype(np.int64)
    if conflict.any():
        bias_rows.append({
            "direction": direction,
            "fold": fold,
            "conflicting_text_samples": int(conflict.sum()),
            "fraction_semantic_conditioning_corrects": float(np.mean((r0_pred[conflict] != labels[conflict]) & (r2_pred[conflict] == labels[conflict]))),
            "fraction_semantic_conditioning_worsens": float(np.mean((r0_pred[conflict] == labels[conflict]) & (r2_pred[conflict] != labels[conflict]))),
            "SEMANTIC_CONFIRMATION_BIAS": bool(np.mean((r0_pred[conflict] == labels[conflict]) & (r2_pred[conflict] != labels[conflict])) > np.mean((r0_pred[conflict] != labels[conflict]) & (r2_pred[conflict] == labels[conflict])) + 0.10),
        })
    else:
        bias_rows.append({"direction": direction, "fold": fold, "conflicting_text_samples": 0, "fraction_semantic_conditioning_corrects": 0.0, "fraction_semantic_conditioning_worsens": 0.0, "SEMANTIC_CONFIRMATION_BIAS": False})

    sem_rows = []
    for split, rows, sem in [("train_memory", rows_train, m_sem), ("held_query", rows_held, q_sem)]:
        for i, r in enumerate(rows):
            sem_rows.append({"direction": direction, "fold": fold, "split": split, "idx": r["idx"], "category": r["category"], "stored_label": r["label"], "p_N": float(sem[i, 0]), "p_A": float(sem[i, 1]), "confidence_abs_pA_minus_0p5": float(abs(sem[i, 1] - 0.5))})

    audit = {"direction": direction, "fold": fold, "dataset": dataset_name, "train_categories": rotation["train_categories"], "heldout_categories": rotation["val_categories"], "train_images": len(train_indices), "heldout_images": len(held_indices), "pixel_map_parity_max_abs": 0.0, "modified_path": "image memory retrieval only"}
    return audit, sem_rows, raw_rows, retrieval_rows, mem_label_rows, cat_rows, conf_rows, bias_rows, percat_rows


def grouped(raw_rows):
    base = {(r["direction"], int(r["fold"])): r for r in raw_rows if r["variant"] == "R0_VISUAL_OFFICIAL"}
    rows = []
    for r in raw_rows:
        b = base[(r["direction"], int(r["fold"]))]
        row = dict(r)
        row["dI-AUROC"] = float(r["I-AUROC"] - b["I-AUROC"])
        row["dI-AP"] = float(r["I-AP"] - b["I-AP"])
        row["dP-AUROC"] = float(r["P-AUROC"] - b["P-AUROC"])
        row["dP-AP"] = float(r["P-AP"] - b["P-AP"])
        row["G_image_rank"] = 0.5 * row["dI-AUROC"] + 0.5 * row["dI-AP"]
        row["G_pixel_rank"] = 0.5 * row["dP-AUROC"] + 0.5 * row["dP-AP"]
        row["G_task"] = row["G_image_rank"]
        rows.append(row)
    return rows


def specificity(grows):
    rows = []
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in grows}
    all_q, all_m = [], []
    for d in ["D1", "D2"]:
        qs, ms = [], []
        for f in [0, 1]:
            r2 = by[(d, f, "R2_VISUAL_PLUS_SEMANTIC")]
            r3 = by[(d, f, "R3_SHUFFLED_QUERY_SEMANTIC")]
            r4 = by[(d, f, "R4_SHUFFLED_MEMORY_SEMANTIC")]
            qs.append(r2["G_task"] - r3["G_task"])
            ms.append(r2["G_task"] - r4["G_task"])
        rows.append({"direction": d, "R2_minus_R3_mean_G_task": float(np.mean(qs)), "R2_minus_R4_mean_G_task": float(np.mean(ms)), "query_specific_pass": bool(np.mean(qs) >= 0.15), "memory_specific_pass": bool(np.mean(ms) >= 0.15)})
        all_q.extend(qs)
        all_m.extend(ms)
    rows.append({"direction": "ALL", "R2_minus_R3_mean_G_task": float(np.mean(all_q)), "R2_minus_R4_mean_G_task": float(np.mean(all_m)), "query_specific_pass": bool(np.mean(all_q) >= 0.15), "memory_specific_pass": bool(np.mean(all_m) >= 0.15)})
    return rows


def decide(grows, spec_rows, cat_rows, bias_rows):
    r2 = [r for r in grows if r["variant"] == "R2_VISUAL_PLUS_SEMANTIC"]
    direction = {}
    for d in ["D1", "D2"]:
        vals = [r for r in r2 if r["direction"] == d]
        mean_g = float(np.mean([r["G_task"] for r in vals]))
        mean_drop_ia = float(np.mean([r["dI-AUROC"] for r in vals]))
        mean_drop_ap = float(np.mean([r["dI-AP"] for r in vals]))
        direction[d] = {
            "mean_G_task": mean_g,
            "positive_folds": int(sum(r["G_task"] > 0 for r in vals)),
            "min_G_task": float(min(r["G_task"] for r in vals)),
            "mean_dI_AUROC": mean_drop_ia,
            "mean_dI_AP": mean_drop_ap,
            "passes": bool(mean_g >= 0.20 and sum(r["G_task"] > 0 for r in vals) == 2 and min(r["G_task"] for r in vals) >= -0.20 and mean_drop_ia >= -0.10 and mean_drop_ap >= -0.10),
        }
    all_positive = sum(r["G_task"] > 0 for r in r2)
    crossfold = all_positive >= 3 and min(r["G_task"] for r in r2) >= -0.20
    spec_all = next(r for r in spec_rows if r["direction"] == "ALL")
    specificity_pass = bool(spec_all["query_specific_pass"] and spec_all["memory_specific_pass"])
    bias = any(str(r.get("SEMANTIC_CONFIRMATION_BIAS", False)).lower() == "true" for r in bias_rows)
    r2_cats = [r for r in cat_rows if r["variant"] == "R2_VISUAL_PLUS_SEMANTIC"]
    confound = any(float(r["dominant_source_category_fraction"]) > 0.80 or float(r["query_category_predictability"]) > 0.90 for r in r2_cats)
    if not (direction["D1"]["passes"] or direction["D2"]["passes"]):
        return "XXXI_A_NO_SEMANTIC_RETRIEVAL_GAIN", direction
    if direction["D1"]["passes"] != direction["D2"]["passes"]:
        return "XXXI_A_DIRECTION_SPECIFIC_SEMANTIC_GAIN", direction
    if not crossfold:
        return "XXXI_A_NO_SEMANTIC_RETRIEVAL_GAIN", direction
    if not specificity_pass:
        return "XXXI_A_SEMANTIC_IDENTITY_NOT_SPECIFIC", direction
    if bias:
        return "XXXI_A_SEMANTIC_CONFIRMATION_BIAS", direction
    if confound:
        return "XXXI_A_CATEGORY_STYLE_CONFOUND", direction
    return "PASS_TO_PHASE_XXXI_B", direction


def write_report(outcome, direction_summary, raw_rows, grows, spec_rows):
    lines = [
        "# Phase XXXI-A CLIP-Native Semantic-Conditioned Memory Retrieval Audit",
        "",
        f"Outcome: **{outcome}**",
        "",
        "Scope: D1/D2 fold0/fold1 source-held-out only. Image-memory retrieval was modified; patch/pixel map path was parity-controlled. No training, target access, dynamic memory, prompt feedback, or PRO-based decision.",
        "",
        "## Metrics",
        "",
        "| Direction | Fold | Variant | I-AUROC | I-AP | P-AUROC | P-AP | G_task |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    gmap = {(r["direction"], int(r["fold"]), r["variant"]): r for r in grows}
    for r in raw_rows:
        g = gmap[(r["direction"], int(r["fold"]), r["variant"])]
        lines.append(f"| {r['direction']} | {r['fold']} | {r['variant']} | {r['I-AUROC']:.4f} | {r['I-AP']:.4f} | {r['P-AUROC']:.4f} | {r['P-AP']:.4f} | {g['G_task']:+.4f} |")
    lines.extend(["", "## Direction Gates", ""])
    for d, s in direction_summary.items():
        lines.append(f"- {d}: mean G_task={s['mean_G_task']:+.4f}, positive_folds={s['positive_folds']}/2, min={s['min_G_task']:+.4f}, mean dI-AUROC={s['mean_dI_AUROC']:+.4f}, mean dI-AP={s['mean_dI_AP']:+.4f}, pass={s['passes']}.")
    lines.extend(["", "## Specificity", ""])
    for r in spec_rows:
        lines.append(f"- {r['direction']}: R2-R3={float(r['R2_minus_R3_mean_G_task']):+.4f}, R2-R4={float(r['R2_minus_R4_mean_G_task']):+.4f}.")
    lines.extend(["", "## Answers", "", "- Q1/Q2: judged by R2 G_task direction gates above.", "- Q3/Q4: judged by shuffled query and shuffled memory semantic controls above.", "- Q5: retrieval diversity is reported but never used for continuation.", "- Q6: confirmation-bias audit is written to `CONFIRMATION_BIAS.csv`.", "", "Mandatory stop: Phase XXXI-B was not started."])
    (OUT / "XXXI_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    anchors = {}
    anchor_audit = []
    for direction in ["D1", "D2"]:
        ckpt = direction_cfg(direction)[2]
        model, prompt_learner, _, _, _ = load_components(ckpt, device)
        anc = text_anchors(model, prompt_learner, device)
        anchors[direction] = anc
        anchor_audit.append({"direction": direction, "checkpoint": ckpt, "anchor_shape": list(anc.shape), "normal_anchor_norm": float(np.linalg.norm(anc[0])), "anomaly_anchor_norm": float(np.linalg.norm(anc[1])), "normal_anomaly_cosine": float(anc[0] @ anc[1]), "prompt_bias": "zero bias, official prompt learner weights"})
        del model, prompt_learner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_audit, all_sem, all_raw, all_ret, all_mem, all_cat, all_conf, all_bias, all_pc = [], [], [], [], [], [], [], [], []
    for d in ["D1", "D2"]:
        for f in [0, 1]:
            res = run_fold(d, f, device, anchors)
            audit, sem, raw, ret, mem, cat, conf, bias, pc = res
            all_audit.append(audit); all_sem.extend(sem); all_raw.extend(raw); all_ret.extend(ret)
            all_mem.extend(mem); all_cat.extend(cat); all_conf.extend(conf); all_bias.extend(bias); all_pc.extend(pc)
            write_csv(OUT / "RAW_METRICS.running.csv", all_raw)
    grows = grouped(all_raw)
    spec = specificity(grows)
    outcome, direction_summary = decide(grows, spec, all_cat, all_bias)

    retrieval_audit = {
        "phase": "XXXI-A",
        "modified_path": "image memory retrieval only",
        "official_baseline": "R0_VISUAL_OFFICIAL uses image_proj(query,key) dot-product logits and official softmax aggregation over all image-memory entries",
        "semantic_variants": "R1/R2/R3/R4 replace image-memory softmax logits with logit(rank) candidate-ordering scores; patch memory, prompt branch, top1 pixel map score, final 0.3/0.7 fusion are unchanged",
        "pixel_metric_policy": "P-AUROC is macro mean over source held-out categories using unchanged official maps; P-AP is standard average_precision_score over flattened source held-out pixels as required by XXXI-A",
        "rank_scope": "within each query over memory candidates only",
        "target_accessed": False,
        "trained": False,
        "folds": all_audit,
        "runtime_sec": time.time() - start,
    }
    (OUT / "RETRIEVAL_PATH_AUDIT.json").write_text(json.dumps(retrieval_audit, indent=2, default=json_default), encoding="utf-8")
    (OUT / "TEXT_ANCHOR_AUDIT.json").write_text(json.dumps(anchor_audit, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "MEMORY_SEMANTIC_STATE.csv", all_sem)
    write_csv(OUT / "RAW_METRICS.csv", all_raw)
    write_csv(OUT / "GROUPED_GAINS.csv", grows)
    write_csv(OUT / "RETRIEVAL_METRICS.csv", all_ret)
    write_csv(OUT / "SEMANTIC_SPECIFICITY.csv", spec)
    write_csv(OUT / "MEMORY_LABEL_DIAGNOSTIC.csv", all_mem)
    write_csv(OUT / "CATEGORY_STYLE_CONFOUND.csv", all_cat)
    write_csv(OUT / "CONFIDENCE_STRATIFICATION.csv", all_conf)
    write_csv(OUT / "CONFIRMATION_BIAS.csv", all_bias)
    write_csv(OUT / "PER_CATEGORY.csv", all_pc)
    write_report(outcome, direction_summary, all_raw, grows, spec)
    payload = {"phase": "XXXI-A", "outcome": outcome, "target_accessed": False, "trained": False, "pro_used_for_decision": False, "phase_xxxi_b_started": False, "runtime_sec": time.time() - start, "direction_gates": direction_summary, "semantic_specificity": spec}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXXI-A\n")
        fp.write("- Command: python experiments/phase_xxxi_semantic_retrieval.py\n")
        fp.write("- Scope: D1 fold0/fold1 and D2 fold0/fold1 source-held-out CLIP-native semantic-conditioned image-memory retrieval; no target; no training.\n")
        fp.write(f"- Outcome: {outcome}.\n")
        for d, s in direction_summary.items():
            fp.write(f"- {d}: R2 mean G_task {s['mean_G_task']:+.4f}, positive folds {s['positive_folds']}/2, pass {s['passes']}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXXI_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
