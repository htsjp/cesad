import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    item_at,
    item_meta,
    mask_np,
    score_feature_grid,
    torch_resize,
)
from experiments.phase_sgpr import (
    AUDIT_PAIRS,
    IMAGE_SIZE,
    OUT as SGPR_OUT,
    SIDE,
    build_sgpr_maps,
    evaluate_maps,
    fixed_box,
    greedy_nms_peaks,
    read_csv,
    safe_ap_binary,
    safe_auc_binary,
    summarize_category_local,
    write_json,
)
from experiments.phase_spatial_residual_refinement_standard_metric import fnum
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import patch_labels
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from utils.transforms import get_transform


OUT = Path("results/phase_prgr/formal")
FEATURE_NAMES = [
    "top1_score",
    "top2_score",
    "top1_minus_top2",
    "top1_minus_median",
    "top1_minus_p90",
    "map_mean",
    "map_std",
    "map_entropy",
    "top5pct_mass_ratio",
    "top2_peak_distance_normalized",
    "proposal1_mean_score",
    "proposal2_mean_score",
    "proposal_mean_minus_global_mean",
]
VARIANTS = ("B0_CanonicalNTR", "E1_FixedLRP", "E3_SGPR", "E4_PRGR")


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_runtime(device):
    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    return preprocess, clip_model


def half_boxes(score37, w, h, k=2):
    peaks = greedy_nms_peaks(score37, k, 0.25 * float(min(w, h)), w, h)
    out = []
    for p in peaks:
        out.append((p, fixed_box(p["center_x"], p["center_y"], w // 2, h // 2, w, h)))
    return out


def grid_box_mask(box, w, h, side=SIDE):
    x1, y1, x2, y2 = box
    xs = (np.arange(side, dtype=np.float32) + 0.5) * float(w) / float(side)
    ys = (np.arange(side, dtype=np.float32) + 0.5) * float(h) / float(side)
    xx, yy = np.meshgrid(xs, ys)
    return (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)


def map_entropy(score):
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    s = s - np.max(s)
    p = np.exp(s)
    p = p / max(float(p.sum()), 1e-12)
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    return ent / np.log(max(len(p), 2))


def reliability_features(score37, w, h):
    s = np.asarray(score37, dtype=np.float32).reshape(SIDE, SIDE)
    flat = s.reshape(-1)
    order = np.argsort(flat)[::-1]
    top1 = float(flat[order[0]])
    top2 = float(flat[order[1]]) if flat.size > 1 else top1
    median = float(np.median(flat))
    p90 = float(np.percentile(flat, 90))
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    topk = max(1, int(np.ceil(0.05 * flat.size)))
    pos = flat - float(np.min(flat))
    mass = float(np.sum(pos[order[:topk]]) / max(np.sum(pos), 1e-12))
    pairs = half_boxes(s, w, h, 2)
    if len(pairs) >= 2:
        p1, b1 = pairs[0]
        p2, b2 = pairs[1]
        dist = ((p1["center_x"] - p2["center_x"]) ** 2 + (p1["center_y"] - p2["center_y"]) ** 2) ** 0.5
        dist_norm = float(dist / max((w * w + h * h) ** 0.5, 1e-12))
    else:
        b1 = b2 = (0, 0, w, h)
        dist_norm = 0.0
    means = []
    for b in (b1, b2):
        m = grid_box_mask(b, w, h)
        means.append(float(np.mean(s[m])) if np.any(m) else mean)
    vals = {
        "top1_score": top1,
        "top2_score": top2,
        "top1_minus_top2": top1 - top2,
        "top1_minus_median": top1 - median,
        "top1_minus_p90": top1 - p90,
        "map_mean": mean,
        "map_std": std,
        "map_entropy": map_entropy(s),
        "top5pct_mass_ratio": mass,
        "top2_peak_distance_normalized": dist_norm,
        "proposal1_mean_score": means[0],
        "proposal2_mean_score": means[1],
        "proposal_mean_minus_global_mean": float(np.mean(means) - mean),
    }
    return vals


def proposal_label_from_mask(score37, mask, w, h):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return 0
    labels, ncomp = ndimage.label(m, structure=np.ones((3, 3), dtype=np.int8))
    if ncomp == 0:
        return 0
    mh, mw = m.shape
    cover = np.zeros_like(m, dtype=bool)
    for _peak, box in half_boxes(score37, w, h, 2):
        x1, y1, x2, y2 = box
        x1 = int(round(x1 * mw / max(w, 1)))
        x2 = int(round(x2 * mw / max(w, 1)))
        y1 = int(round(y1 * mh / max(h, 1)))
        y2 = int(round(y2 * mh / max(h, 1)))
        x1, x2 = max(0, min(x1, mw)), max(0, min(x2, mw))
        y1, y2 = max(0, min(y1, mh)), max(0, min(y2, mh))
        cover[y1:y2, x1:x2] = True
    for rid in range(1, ncomp + 1):
        if np.any(cover & (labels == rid)):
            return 1
    return 0


def source_dataset_info(source, smoke):
    if source == "Full VisA":
        _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(smoke)
        kind = "source_dataset"
    elif source == "Full MVTec":
        _name, dataset, indices, patch = load_full_mvtec_source(smoke)
        kind = "source_dataset"
    else:
        raise ValueError(source)
    return dataset, indices, patch, kind


def source_mask(item):
    if int(item.get("anomaly", 0)) == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    try:
        y = patch_labels(item).reshape(SIDE, SIDE).astype(np.uint8)
        return torch_resize(y.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), align_corners=True) > 0.5
    except Exception:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)


def source_path(item):
    for key in ("img_path", "image_path", "path"):
        if key in item and item[key]:
            return str(item[key])
    raise ValueError("source item has no image path")


def collect_source_selector_rows(model_info, smoke):
    dataset, indices, patch, _kind = source_dataset_info(model_info["source"], smoke)
    n = min(96, len(indices)) if smoke else len(indices)
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for pos in range(n):
        if pos % 250 == 0:
            print(f"[PRGR] source selector {model_info['source']} {pos}/{n}", flush=True)
        idx = int(indices[pos])
        item = dataset[idx]
        path = source_path(item)
        with Image.open(path) as im:
            w, h = im.size
        score37 = score_feature_grid(model_info, patch[pos], device)
        mask = source_mask(item)
        label = proposal_label_from_mask(score37, mask, w, h) if int(item.get("anomaly", 0)) else 0
        feats = reliability_features(score37, w, h)
        rows.append({
            "source": model_info["source"],
            "image_index": pos,
            "category": str(item.get("cls_name", "")),
            "image_label": int(item.get("anomaly", 0)),
            "proposal_label": int(label),
            **feats,
        })
    return rows


def fit_selector(source, rows):
    y = np.asarray([int(r["proposal_label"]) for r in rows], dtype=np.int64)
    x = np.asarray([[fnum(r, k) for k in FEATURE_NAMES] for r in rows], dtype=np.float64)
    Cs = [0.01, 0.1, 1.0, 10.0]
    cv_rows = []
    best_c, best_score = Cs[0], -np.inf
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"{source} selector has only one class")
    splits = min(5, int(np.bincount(y).min()))
    splits = max(2, splits)
    for c in Cs:
        fold_scores = []
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
        for fold, (tr, va) in enumerate(skf.split(x, y)):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="liblinear", class_weight="balanced", random_state=0))
            clf.fit(x[tr], y[tr])
            p = clf.predict_proba(x[va])[:, 1]
            auc = roc_auc_score(y[va], p) if len(np.unique(y[va])) > 1 else np.nan
            ap = average_precision_score(y[va], p) if len(np.unique(y[va])) > 1 else np.nan
            fold_scores.append(auc)
            cv_rows.append({"source": source, "C": c, "fold": fold, "AUROC": float(auc), "AP": float(ap)})
        mean_auc = float(np.nanmean(fold_scores))
        cv_rows.append({"source": source, "C": c, "fold": "MEAN", "AUROC": mean_auc, "AP": ""})
        if mean_auc > best_score:
            best_score, best_c = mean_auc, c
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, solver="liblinear", class_weight="balanced", random_state=0))
    clf.fit(x, y)
    lr = clf.named_steps["logisticregression"]
    scaler = clf.named_steps["standardscaler"]
    coef_rows = []
    for name, coef, mu, sigma in zip(FEATURE_NAMES, lr.coef_[0], scaler.mean_, scaler.scale_):
        coef_rows.append({"source": source, "selected_C": best_c, "feature": name, "coef_standardized": float(coef), "scaler_mean": float(mu), "scaler_scale": float(sigma)})
    coef_rows.append({"source": source, "selected_C": best_c, "feature": "intercept", "coef_standardized": float(lr.intercept_[0]), "scaler_mean": "", "scaler_scale": ""})
    return clf, cv_rows, coef_rows


def per_image_patch_ap(mask, score):
    y = np.asarray(mask, dtype=np.uint8).reshape(-1)
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return safe_ap_binary(y, s)


def build_prgr_pair(model_info, clf, clip_model, preprocess, target, dataset, refs, patch, kind, device, smoke):
    maps, metas, crops, compute = build_sgpr_maps(
        model_info,
        clip_model,
        preprocess,
        target,
        dataset,
        refs,
        patch,
        kind,
        device,
        variants=("B0_CanonicalNTR", "E1_FixedLRP", "E3_SGPR"),
    )
    maps["E4_PRGR"] = []
    maps["ORACLE"] = []
    routing_rows = []
    audit_rows = []
    n = len(metas)
    for pos in range(n):
        meta = metas[pos]
        path = meta["img_path"]
        with Image.open(path) as im:
            w, h = im.size
        global37 = torch_resize(np.asarray(maps["B0_CanonicalNTR"][pos], dtype=np.float32), (SIDE, SIDE), align_corners=True)
        feats = reliability_features(global37, w, h)
        x = np.asarray([[feats[k] for k in FEATURE_NAMES]], dtype=np.float64)
        prob = float(clf.predict_proba(x)[0, 1])
        select_sgpr = prob >= 0.5
        final = maps["E3_SGPR"][pos] if select_sgpr else maps["E1_FixedLRP"][pos]
        maps["E4_PRGR"].append(np.asarray(final, dtype=np.float16))
        gt_hit = proposal_label_from_mask(global37, meta["mask"], w, h) if int(meta["label"]) else 0
        oracle = maps["E3_SGPR"][pos] if gt_hit else maps["E1_FixedLRP"][pos]
        maps["ORACLE"].append(np.asarray(oracle, dtype=np.float16))
        e1_ap = per_image_patch_ap(meta["mask"], maps["E1_FixedLRP"][pos])
        e3_ap = per_image_patch_ap(meta["mask"], maps["E3_SGPR"][pos])
        row = {
            "source": model_info["source"],
            "dataset": target,
            "image_index": pos,
            "category": meta["category"],
            "image_label": int(meta["label"]),
            "selector_probability": prob,
            "selected_policy": "E3_SGPR" if select_sgpr else "E1_FixedLRP",
            "actual_proposal_hit": int(gt_hit),
            "E3_minus_E1_image_P_AP": e3_ap - e1_ap if np.isfinite(e1_ap) and np.isfinite(e3_ap) else float("nan"),
            **feats,
        }
        routing_rows.append(row)
        audit_rows.append(row.copy())
    return maps, metas, crops, compute, routing_rows, audit_rows


def evaluate_oracle(source, target, maps, metas):
    return evaluate_variable_policy(source, target, maps, metas, "ORACLE")


def evaluate_variable_policy(source, target, maps, metas, variant):
    raw_rows, cat_rows = [], []
    by_cat = {}
    for i, meta in enumerate(metas):
        d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "maps": []})
        d["labels"].append(meta["label"])
        d["image_scores"].append(top1_image_score(maps[variant][i]))
        d["masks"].append(meta["mask"])
        d["maps"].append(np.asarray(maps[variant][i], dtype=np.float32))
    fr_rows = []
    for cat, vals in sorted(by_cat.items()):
        scores = [torch_resize(s, m.shape, align_corners=True) for s, m in zip(vals["maps"], vals["masks"])]
        fr = summarize_category_local(vals["labels"], vals["image_scores"], vals["masks"], scores)
        row = {"source": source, "dataset": target, "category": cat, "variant": variant, "protocol": "M", "aggregation": "CATEGORY"}
        for met, val in fr.items():
            row[f"M_FR_{met.replace('-', '_')}"] = val
        cat_rows.append(row)
        fr_rows.append(row)
    raw = {"source": source, "dataset": target, "variant": variant, "aggregation": "CATEGORY_MACRO"}
    for met in ("P-AUROC", "P-AP", "P-F1max", "PRO", "G_pixel", "I-AUROC", "I-AP", "I-F1max"):
        key = f"M_FR_{met.replace('-', '_')}"
        vals = np.asarray([fnum(r, key) for r in fr_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        raw[key] = float(vals.mean()) if vals.size else float("nan")
    raw_rows.append(raw)
    return raw_rows, cat_rows


def target_audit_rows(rows):
    out = []
    for source, dataset in sorted({(r["source"], r["dataset"]) for r in rows}):
        part = [r for r in rows if r["source"] == source and r["dataset"] == dataset]
        y = np.asarray([int(r["actual_proposal_hit"]) for r in part], dtype=np.uint8)
        p = np.asarray([fnum(r, "selector_probability") for r in part], dtype=np.float32)
        selected = [r for r in part if r["selected_policy"] == "E3_SGPR"]
        unselected = [r for r in part if r["selected_policy"] != "E3_SGPR"]
        out.append({
            "source": source,
            "dataset": dataset,
            "selector_AUROC_actual_hit": safe_auc_binary(y, p),
            "routing_fraction_SGPR": len(selected) / max(len(part), 1),
            "actual_hit_rate_routed_to_SGPR": float(np.mean([int(r["actual_proposal_hit"]) for r in selected])) if selected else float("nan"),
            "actual_hit_rate_routed_to_LRP": float(np.mean([int(r["actual_proposal_hit"]) for r in unselected])) if unselected else float("nan"),
            "mean_E3_minus_E1_image_P_AP_selected": float(np.nanmean([fnum(r, "E3_minus_E1_image_P_AP") for r in selected])) if selected else float("nan"),
            "mean_E3_minus_E1_image_P_AP_unselected": float(np.nanmean([fnum(r, "E3_minus_E1_image_P_AP") for r in unselected])) if unselected else float("nan"),
        })
    return out


def decide(raw, oracle_raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    deltas = {}
    vals = []
    e3_vals = []
    for source, dataset in sorted(AUDIT_PAIRS):
        e1 = by.get((source, dataset, "E1_FixedLRP"))
        e3 = by.get((source, dataset, "E3_SGPR"))
        e4 = by.get((source, dataset, "E4_PRGR"))
        if not (e1 and e3 and e4):
            continue
        d = fnum(e4, "M_FR_P_AP") - fnum(e1, "M_FR_P_AP")
        de3 = fnum(e4, "M_FR_P_AP") - fnum(e3, "M_FR_P_AP")
        vals.append(d)
        e3_vals.append(de3)
        deltas[f"{source}->{dataset}"] = {"E4_minus_E1_P_AP": d, "E4_minus_E3_P_AP": de3}
    gate = (
        len(vals) == 8
        and deltas.get("Full MVTec->VisA", {}).get("E4_minus_E1_P_AP", -999) >= 1.0
        and deltas.get("Full VisA->MVTec", {}).get("E4_minus_E1_P_AP", -999) >= 0.0
        and all(deltas.get(k, {}).get("E4_minus_E1_P_AP", -999) >= -0.3 for k in ("Full VisA->BTAD", "Full MVTec->BTAD"))
        and all(deltas.get(k, {}).get("E4_minus_E1_P_AP", -999) >= 0.0 for k in ("Full VisA->MPDD", "Full MVTec->MPDD"))
        and all(deltas.get(k, {}).get("E4_minus_E1_P_AP", -999) >= -0.3 for k in ("Full VisA->DTD-Synthetic", "Full MVTec->DTD-Synthetic"))
        and sum(v >= -0.2 for v in vals) >= 7
        and float(np.mean(vals)) >= 0.8
        and float(np.mean(e3_vals)) > 0
    )
    oracle_by = {(r["source"], r["dataset"], r["variant"]): r for r in oracle_raw}
    oracle_deltas = []
    for source, dataset in sorted(AUDIT_PAIRS):
        e1 = by.get((source, dataset, "E1_FixedLRP"))
        orow = oracle_by.get((source, dataset, "ORACLE"))
        if e1 and orow:
            oracle_deltas.append(fnum(orow, "M_FR_P_AP") - fnum(e1, "M_FR_P_AP"))
    oracle_strong = bool(oracle_deltas and np.mean(oracle_deltas) >= 0.8)
    if gate:
        outcome = "PROPOSAL_RELIABILITY_GATING_IMPROVES_REOBSERVATION"
    elif oracle_strong:
        outcome = "PROPOSAL_ROUTING_HAS_HEADROOM_BUT_RELIABILITY_DOES_NOT_TRANSFER"
    else:
        outcome = "PROPOSAL_RELIABILITY_ROUTING_HAS_NO_VALUE"
    return outcome, {
        "deltas": deltas,
        "mean_E4_minus_E1_P_AP": float(np.mean(vals)) if vals else float("nan"),
        "mean_E4_minus_E3_P_AP": float(np.mean(e3_vals)) if e3_vals else float("nan"),
        "non_drop_ge_minus_0_2": int(sum(v >= -0.2 for v in vals)),
        "gate_pass": gate,
        "oracle_mean_minus_E1_P_AP": float(np.mean(oracle_deltas)) if oracle_deltas else float("nan"),
    }


def write_report(raw, oracle_raw, audit, decision, outcome, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    oby = {(r["source"], r["dataset"], r["variant"]): r for r in oracle_raw}
    lines = [
        "# Phase PRGR: Proposal-Reliability-Gated Re-Observation",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "No CLIP/state/NTR parameters were modified. Source-only logistic predictors route each target image between locked Fixed LRP and SGPR.",
        "",
        "## Main M-FR P-AP",
        "",
        "| Source | Target | E1 Fixed LRP | E3 SGPR | E4 PRGR | Oracle | E4-E1 | E4-E3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, dataset in sorted(AUDIT_PAIRS):
        e1 = by.get((source, dataset, "E1_FixedLRP"))
        e3 = by.get((source, dataset, "E3_SGPR"))
        e4 = by.get((source, dataset, "E4_PRGR"))
        orow = oby.get((source, dataset, "ORACLE"))
        if e1 and e3 and e4:
            lines.append(f"| {source} | {dataset} | {fnum(e1,'M_FR_P_AP'):.4f} | {fnum(e3,'M_FR_P_AP'):.4f} | {fnum(e4,'M_FR_P_AP'):.4f} | {fnum(orow,'M_FR_P_AP'):.4f} | {fnum(e4,'M_FR_P_AP')-fnum(e1,'M_FR_P_AP'):+.4f} | {fnum(e4,'M_FR_P_AP')-fnum(e3,'M_FR_P_AP'):+.4f} |")
    lines += [
        "",
        f"Mean E4-E1 P-AP: {decision['mean_E4_minus_E1_P_AP']:+.4f}; E4-E3 mean: {decision['mean_E4_minus_E3_P_AP']:+.4f}; non-drop count >= -0.2: {decision['non_drop_ge_minus_0_2']}/8.",
        f"Oracle mean E1 gap: {decision['oracle_mean_minus_E1_P_AP']:+.4f}.",
        "",
        "## Selector Target Audit",
        "",
        "| Source | Target | Selector AUROC(hit) | SGPR Fraction | Hit Rate SGPR | Hit Rate LRP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in audit:
        lines.append(f"| {r['source']} | {r['dataset']} | {fnum(r,'selector_AUROC_actual_hit'):.4f} | {fnum(r,'routing_fraction_SGPR'):.4f} | {fnum(r,'actual_hit_rate_routed_to_SGPR'):.4f} | {fnum(r,'actual_hit_rate_routed_to_LRP'):.4f} |")
    lines += ["", f"Runtime: {runtime:.1f}s."]
    write_text(OUT / "REPORT.md", "\n".join(lines))


def run(smoke=False):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, clip_model = load_runtime(device)
    models = [load_visa_source(device), load_mvtec_source()]

    selector_rows, cv_rows, coef_rows, selectors = [], [], [], {}
    for model_info in models:
        rows = collect_source_selector_rows(model_info, smoke)
        selector_rows.extend(rows)
        clf, cv, coef = fit_selector(model_info["source"], rows)
        selectors[model_info["source"]] = clf
        cv_rows.extend(cv)
        coef_rows.extend(coef)
    write_csv(OUT / "SOURCE_SELECTOR_TRAINING_ROWS.csv", selector_rows)
    write_csv(OUT / "SOURCE_SELECTOR_CV.csv", cv_rows)
    write_csv(OUT / "SELECTOR_COEFFICIENTS.csv", coef_rows)

    raw, cats, oracle_raw, oracle_cats, routing, target_rows = [], [], [], [], [], []
    for model_info in models:
        cells, _audits = protocol_cells(model_info["source"], device, smoke)
        for cell_id, target, dataset, idx_or_rows, patch, kind in cells:
            if (model_info["source"], target) not in AUDIT_PAIRS:
                continue
            n = min(24, patch.shape[0]) if smoke else patch.shape[0]
            print(f"[PRGR] evaluate {model_info['source']} -> {target} n={n}", flush=True)
            maps, metas, cr, co, rr, ar = build_prgr_pair(
                model_info,
                selectors[model_info["source"]],
                clip_model,
                preprocess,
                target,
                dataset,
                idx_or_rows[:n],
                patch[:n],
                kind,
                device,
                smoke,
            )
            rr_raw, cc = evaluate_maps(
                model_info["source"],
                target,
                idx_or_rows[:n],
                dataset,
                kind,
                maps,
                metas,
                variants=("B0_CanonicalNTR", "E1_FixedLRP", "E3_SGPR"),
            )
            e4_raw, e4_cat = evaluate_variable_policy(model_info["source"], target, maps, metas, "E4_PRGR")
            orr, occ = evaluate_oracle(model_info["source"], target, maps, metas)
            rr_raw.extend(e4_raw)
            cc.extend(e4_cat)
            raw.extend(rr_raw)
            cats.extend(cc)
            oracle_raw.extend(orr)
            oracle_cats.extend(occ)
            routing.extend(rr)
            target_rows.extend(ar)
            if not smoke:
                write_csv(OUT / "FULL_RES_METRICS.partial.csv", raw)
                write_csv(OUT / "ROUTING_DECISIONS.partial.csv", routing)
                write_csv(OUT / "ORACLE_ROUTING_METRICS.partial.csv", oracle_raw)
    audit = target_audit_rows(target_rows)
    outcome, decision = decide(raw, oracle_raw)
    runtime = time.time() - start
    write_csv(OUT / "ROUTING_DECISIONS.csv", routing)
    write_csv(OUT / "FULL_RES_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "ORACLE_ROUTING_METRICS.csv", oracle_raw)
    write_csv(OUT / "ORACLE_PER_CATEGORY_FULL_RES.csv", oracle_cats)
    write_csv(OUT / "SELECTOR_TARGET_AUDIT.csv", audit)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision": decision, "runtime_sec": runtime, "strict_source_only": True, "target_statistics": False, "target_tuning": False})
    write_report(raw, oracle_raw, audit, decision, outcome, runtime)
    print(f"[PRGR] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args.smoke)


if __name__ == "__main__":
    main()
