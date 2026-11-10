import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np, score_feature_grid, torch_resize
from experiments.phase_sgpr import AUDIT_PAIRS, evaluate_maps, fixed_quadrant_boxes, score_crops, stitch_quadrant74
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE
from utils.transforms import get_transform


OUT = Path("results/phase_fova_field_of_view_amplification_audit/formal")
MAP_CACHE = OUT / "MAP_CACHE"
VARIANTS = (
    "E0_FixedLRP",
    "E1_FixedLRP_plus_0p5_ReLU_FOVA",
    "E2_FixedLRP_plus_1p0_ReLU_FOVA",
)


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fnum(row, key):
    try:
        v = float(row.get(key, float("nan")))
    except Exception:
        return float("nan")
    return v


def finite_mean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    n = len(sorted_s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    pos = y.astype(bool)
    n_pos = float(np.sum(pos))
    n_neg = float(y.size - np.sum(pos))
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1.0) / 2.0) / max(n_pos * n_neg, 1.0) * 100.0)


def sample_flat(values, rng, cap):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= cap:
        return values
    idx = rng.choice(values.size, size=cap, replace=False)
    return values[idx]


def load_runtime(device):
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    model = load_components(str(CHECKPOINT), device)[0]
    model.eval()
    return preprocess, model


def source_models(device):
    return [load_visa_source(device), load_mvtec_source()]


def slug(source, target):
    return source.replace(" ", "_").replace("/", "_") + "__" + target.replace("/", "_")


def morphology(mask):
    mask = np.asarray(mask, dtype=bool)
    pix = int(mask.sum())
    h, w = mask.shape
    if pix == 0:
        return {
            "anomaly_area_ratio": 0.0,
            "num_connected_components": 0,
            "mean_component_area": 0.0,
            "boundary_to_area_ratio": 0.0,
            "thinness": 0.0,
            "num_anomaly_patches": 0,
            "anomaly_patch_ratio": 0.0,
        }
    labels, ncomp = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    areas = [int(np.sum(labels == rid)) for rid in range(1, ncomp + 1)]
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = mask & ~eroded
    gm = torch.tensor(mask.astype(np.float32)).view(1, 1, h, w)
    patch = F.interpolate(gm, size=(SIDE, SIDE), mode="nearest").view(SIDE, SIDE).numpy() > 0.5
    return {
        "anomaly_area_ratio": float(pix / max(h * w, 1)),
        "num_connected_components": int(ncomp),
        "mean_component_area": float(np.mean(areas)) if areas else 0.0,
        "boundary_to_area_ratio": float(boundary.sum() / max(pix, 1)),
        "thinness": float(boundary.sum() / max(pix, 1)),
        "num_anomaly_patches": int(patch.sum()),
        "anomaly_patch_ratio": float(patch.mean()),
    }


def assign_groups(meta_rows):
    pos = [r for r in meta_rows if int(r["label"]) == 1 and fnum(r, "num_anomaly_patches") > 0]
    patch_vals = [fnum(r, "num_anomaly_patches") for r in pos]
    thin_vals = [fnum(r, "thinness") for r in pos]
    comp_vals = [fnum(r, "num_connected_components") for r in pos]
    small_thr = float(np.percentile(patch_vals, 25)) if patch_vals else float("nan")
    large_thr = float(np.percentile(patch_vals, 75)) if patch_vals else float("nan")
    thin_thr = float(np.percentile(thin_vals, 75)) if thin_vals else float("nan")
    sparse_thr = float(np.percentile(comp_vals, 75)) if comp_vals else float("nan")
    for r in meta_rows:
        patches = fnum(r, "num_anomaly_patches")
        thin = fnum(r, "thinness")
        comps = fnum(r, "num_connected_components")
        r["group_small"] = bool(np.isfinite(small_thr) and patches > 0 and patches <= small_thr)
        r["group_large"] = bool(np.isfinite(large_thr) and patches >= large_thr)
        r["group_thin"] = bool(np.isfinite(thin_thr) and thin >= thin_thr and patches > 0)
        r["group_thick"] = bool(np.isfinite(thin_thr) and thin < thin_thr and patches > 0)
        r["group_sparse"] = bool(np.isfinite(sparse_thr) and comps >= sparse_thr and patches > 0)
        r["group_non_sparse"] = bool(np.isfinite(sparse_thr) and comps < sparse_thr and patches > 0)
    return {"small_patch_p25": small_thr, "large_patch_p75": large_thr, "thinness_p75": thin_thr, "components_p75": sparse_thr}


def build_fova_maps(model_info, clip_model, preprocess, target, dataset, refs, patch, kind, device, smoke):
    maps = {v: [] for v in VARIANTS}
    global_maps, local_maps, amp_maps, lrp_maps = [], [], [], []
    metas, meta_rows, image_rows = [], [], []
    n = len(refs)
    for pos, ref in enumerate(refs):
        if pos % 50 == 0:
            print(f"[FOVA] {model_info['source']} -> {target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        if not path:
            raise ValueError(f"missing img_path for {model_info['source']} -> {target} pos={pos}")
        mask = mask_np(item)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        global37 = score_feature_grid(model_info, patch[pos], device)
        global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
        boxes = fixed_quadrant_boxes(w, h)
        crop_scores = score_crops(model_info, clip_model, preprocess, img, boxes, device)
        local74 = stitch_quadrant74([crop_scores[tuple(b)] for b in boxes])
        lrp74 = 0.5 * global74 + 0.5 * local74
        amp74 = local74 - global74
        relu_amp = np.maximum(amp74, 0.0)
        maps["E0_FixedLRP"].append(lrp74.astype(np.float32))
        maps["E1_FixedLRP_plus_0p5_ReLU_FOVA"].append((lrp74 + 0.5 * relu_amp).astype(np.float32))
        maps["E2_FixedLRP_plus_1p0_ReLU_FOVA"].append((lrp74 + relu_amp).astype(np.float32))
        global_maps.append(global74.astype(np.float16))
        local_maps.append(local74.astype(np.float16))
        amp_maps.append(amp74.astype(np.float16))
        lrp_maps.append(lrp74.astype(np.float16))
        metas.append({"category": cat, "label": int(label), "mask": mask, "img_path": path})
        morph = morphology(mask)
        row = {"source": model_info["source"], "dataset": target, "category": cat, "image_index": pos, "label": int(label), "img_path": path, **morph}
        meta_rows.append(row)
        image_rows.append({
            "source": model_info["source"],
            "dataset": target,
            "category": cat,
            "image_index": pos,
            "label": int(label),
            "q_g_mean": float(np.mean(global74)),
            "q_l_mean": float(np.mean(local74)),
            "q_lrp_mean": float(np.mean(lrp74)),
            "amplification_mean": float(np.mean(amp74)),
            "amplification_median": float(np.median(amp74)),
            "positive_amplification_mean": float(np.mean(relu_amp)),
            "positive_amplification_p95": float(np.percentile(relu_amp, 95)),
        })
        img.close()
    thresholds = assign_groups(meta_rows)
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MAP_CACHE / f"{slug(model_info['source'], target)}.npz",
        q_g=np.stack(global_maps),
        q_l=np.stack(local_maps),
        q_lrp=np.stack(lrp_maps),
        a=np.stack(amp_maps),
        variant_e1=np.stack([np.asarray(x, dtype=np.float16) for x in maps["E1_FixedLRP_plus_0p5_ReLU_FOVA"]]),
        variant_e2=np.stack([np.asarray(x, dtype=np.float16) for x in maps["E2_FixedLRP_plus_1p0_ReLU_FOVA"]]),
    )
    return maps, global_maps, local_maps, amp_maps, metas, meta_rows, image_rows, thresholds


def amplification_diagnostics(source, target, qg_maps, amp_maps, metas, meta_rows):
    seed = sum(ord(c) for c in f"{source}->{target}")
    rng = np.random.default_rng(seed)
    y_all, a_all = [], []
    anom_samples, norm_samples = [], []
    qg_anom, qg_norm = [], []
    rows = []
    groups = ("all", "small", "large", "thin", "thick", "sparse", "non_sparse")
    group_samples = {g: {"y": [], "a": [], "anom": [], "norm": []} for g in groups}

    for qg74, amp74, meta, morph in zip(qg_maps, amp_maps, metas, meta_rows):
        mask = np.asarray(meta["mask"], dtype=bool)
        qg = torch_resize(qg74, mask.shape, align_corners=True)
        amp = torch_resize(amp74, mask.shape, align_corners=True)
        an = np.empty(0, dtype=np.float32)
        no = np.empty(0, dtype=np.float32)
        if np.any(mask):
            an = sample_flat(amp[mask], rng, 2048)
            if an.size:
                anom_samples.append(an)
                qg_anom.append(sample_flat(qg[mask], rng, 2048))
                group_samples["all"]["anom"].append(an)
        if np.any(~mask):
            no = sample_flat(amp[~mask], rng, 2048)
            if no.size:
                norm_samples.append(no)
                qg_norm.append(sample_flat(qg[~mask], rng, 2048))
                group_samples["all"]["norm"].append(no)
        parts_y = []
        parts_s = []
        if an.size:
            parts_y.append(np.ones_like(an, dtype=np.uint8))
            parts_s.append(an)
        if no.size:
            parts_y.append(np.zeros_like(no, dtype=np.uint8))
            parts_s.append(no)
        yy = np.concatenate(parts_y) if parts_y else np.empty(0, dtype=np.uint8)
        ss = np.concatenate(parts_s) if parts_s else np.empty(0, dtype=np.float32)
        if yy.size:
            y_all.append(yy)
            a_all.append(ss)
        for g in ("small", "large", "thin", "thick", "sparse", "non_sparse"):
            if bool(morph.get(f"group_{g}", False)):
                if an.size:
                    group_samples[g]["anom"].append(an)
                if no.size:
                    group_samples[g]["norm"].append(no)

    norm_all = np.concatenate(norm_samples) if norm_samples else np.empty(0, dtype=np.float32)
    anom_all = np.concatenate(anom_samples) if anom_samples else np.empty(0, dtype=np.float32)
    qgn = np.concatenate(qg_norm) if qg_norm else np.empty(0, dtype=np.float32)
    qga = np.concatenate(qg_anom) if qg_anom else np.empty(0, dtype=np.float32)
    if qgn.size and qga.size:
        n95 = float(np.percentile(qgn, 95))
        a20 = float(np.percentile(qga, 20))
    else:
        n95, a20 = float("nan"), float("nan")

    fp_vals, tn_vals, fn_vals, ca_vals = [], [], [], []
    for qg74, amp74, meta in zip(qg_maps, amp_maps, metas):
        mask = np.asarray(meta["mask"], dtype=bool)
        qg = torch_resize(qg74, mask.shape, align_corners=True)
        amp = torch_resize(amp74, mask.shape, align_corners=True)
        if np.isfinite(n95) and np.any(~mask):
            fp_vals.append(sample_flat(amp[(~mask) & (qg >= n95)], rng, 2048))
            tn_vals.append(sample_flat(amp[(~mask) & (qg < n95)], rng, 2048))
        if np.isfinite(a20) and np.any(mask):
            fn_vals.append(sample_flat(amp[mask & (qg <= a20)], rng, 2048))
            ca_vals.append(sample_flat(amp[mask & (qg > a20)], rng, 2048))

    def add_bucket(name, vals):
        arr = np.concatenate([v for v in vals if v.size]) if vals else np.empty(0, dtype=np.float32)
        rows.append({
            "source": source,
            "dataset": target,
            "group": "all",
            "bucket": name,
            "count_sampled": int(arr.size),
            "amplification_mean": float(np.mean(arr)) if arr.size else float("nan"),
            "amplification_median": float(np.median(arr)) if arr.size else float("nan"),
            "amplification_p90": float(np.percentile(arr, 90)) if arr.size else float("nan"),
        })
        return arr

    fp = add_bucket("global_false_positive_normal", fp_vals)
    tn = add_bucket("global_true_negative_normal", tn_vals)
    fn = add_bucket("global_false_negative_anomaly", fn_vals)
    ca = add_bucket("global_correct_anomaly", ca_vals)

    y = np.concatenate(y_all) if y_all else np.empty(0, dtype=np.uint8)
    s = np.concatenate(a_all) if a_all else np.empty(0, dtype=np.float32)
    rows.append({
        "source": source,
        "dataset": target,
        "group": "all",
        "bucket": "summary",
        "normal_amp_mean": float(np.mean(norm_all)) if norm_all.size else float("nan"),
        "normal_amp_median": float(np.median(norm_all)) if norm_all.size else float("nan"),
        "anomaly_amp_mean": float(np.mean(anom_all)) if anom_all.size else float("nan"),
        "anomaly_amp_median": float(np.median(anom_all)) if anom_all.size else float("nan"),
        "amp_anomaly_vs_normal_AUROC": safe_auc(y, s),
        "FN_rescue_mean": float(np.mean(fn) - np.mean(ca)) if fn.size and ca.size else float("nan"),
        "FN_rescue_median": float(np.median(fn) - np.median(ca)) if fn.size and ca.size else float("nan"),
        "FP_minus_TN_mean": float(np.mean(fp) - np.mean(tn)) if fp.size and tn.size else float("nan"),
        "FP_minus_TN_median": float(np.median(fp) - np.median(tn)) if fp.size and tn.size else float("nan"),
    })
    for g in groups[1:]:
        an = np.concatenate(group_samples[g]["anom"]) if group_samples[g]["anom"] else np.empty(0, dtype=np.float32)
        no = np.concatenate(group_samples[g]["norm"]) if group_samples[g]["norm"] else np.empty(0, dtype=np.float32)
        if not (an.size or no.size):
            continue
        yy = np.concatenate([np.ones(an.size, dtype=np.uint8), np.zeros(no.size, dtype=np.uint8)])
        ss = np.concatenate([an, no])
        rows.append({
            "source": source,
            "dataset": target,
            "group": g,
            "bucket": "morphology_summary",
            "normal_amp_mean": float(np.mean(no)) if no.size else float("nan"),
            "normal_amp_median": float(np.median(no)) if no.size else float("nan"),
            "anomaly_amp_mean": float(np.mean(an)) if an.size else float("nan"),
            "anomaly_amp_median": float(np.median(an)) if an.size else float("nan"),
            "amp_anomaly_vs_normal_AUROC": safe_auc(yy, ss) if an.size and no.size else float("nan"),
        })
    return rows


def summarize_metrics(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in VARIANTS[1:]:
        vals = []
        improved = 0
        nondrop = 0
        for source, target in sorted(AUDIT_PAIRS):
            b = by.get((source, target, VARIANTS[0]))
            e = by.get((source, target, variant))
            if not (b and e):
                continue
            d = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            if not np.isfinite(d):
                continue
            vals.append(d)
            improved += int(d > 0)
            nondrop += int(d >= -0.2)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "E_minus_E0_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "E_minus_E0_M_FR_P_AP": d,
                "E_minus_E0_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "E_minus_E0_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
        rows.append({
            "source": "SUMMARY",
            "dataset": "ALL",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(vals),
            "median_delta_M_FR_P_AP": finite_median(vals),
            "improved": improved,
            "non_drop_ge_minus_0_2": nondrop,
            "total": len(vals),
        })
    return rows


def decide(summary, amp_rows):
    summaries = [r for r in summary if r["source"] == "SUMMARY"]
    best = max(summaries, key=lambda r: fnum(r, "mean_delta_M_FR_P_AP")) if summaries else {}
    consistent_diag = 0
    for r in amp_rows:
        if r.get("bucket") == "summary":
            auc = fnum(r, "amp_anomaly_vs_normal_AUROC")
            fn = fnum(r, "FN_rescue_mean")
            if np.isfinite(auc) and auc >= 55.0 and np.isfinite(fn) and fn > 0:
                consistent_diag += 1
    if fnum(best, "mean_delta_M_FR_P_AP") >= 0.5 and int(best.get("improved", 0)) >= 5 and int(best.get("non_drop_ge_minus_0_2", 0)) >= 6:
        return "FOV_AMPLIFICATION_PROVIDES_TRANSFERABLE_ANOMALY_SIGNAL"
    if consistent_diag >= 4:
        return "FOV_AMPLIFICATION_IS_DIAGNOSTIC_BUT_NOT_USEFUL_FOR_FUSION"
    return "FOV_AMPLIFICATION_HAS_NO_STABLE_TRANSFER_SIGNAL"


def write_report(raw, summary, amp_rows, outcome, thresholds, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Phase FOVA: Field-of-View Amplification Audit",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline E0 is locked Canonical Two-State + NTR + Fixed LRP. No model training, target statistics, threshold tuning, learned fusion, router, adapter, or prompt module was used.",
        "",
        "## Fusion Summary",
        "",
        "| Variant | mean dP-AP | median dP-AP | improved | non-drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["source"] == "SUMMARY":
            lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {r['improved']}/{r['total']} | {r['non_drop_ge_minus_0_2']}/{r['total']} |")
    lines += ["", "## Per-Transfer P-AP", "", "| Source | Target | E0 | E1 +0.5ReLU(a) | E2 +1.0ReLU(a) | E1-E0 | E2-E0 |", "|---|---|---:|---:|---:|---:|---:|"]
    for source, target in sorted(AUDIT_PAIRS):
        e0 = by.get((source, target, VARIANTS[0]))
        e1 = by.get((source, target, VARIANTS[1]))
        e2 = by.get((source, target, VARIANTS[2]))
        if not (e0 and e1 and e2):
            continue
        lines.append(f"| {source} | {target} | {fnum(e0,'M_FR_P_AP'):.4f} | {fnum(e1,'M_FR_P_AP'):.4f} | {fnum(e2,'M_FR_P_AP'):.4f} | {fnum(e1,'M_FR_P_AP')-fnum(e0,'M_FR_P_AP'):+.4f} | {fnum(e2,'M_FR_P_AP')-fnum(e0,'M_FR_P_AP'):+.4f} |")
    lines += ["", "## Amplification Diagnostic", "", "| Source | Target | AUC(a) | anomaly mean | normal mean | FN rescue mean | FP-TN mean |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in amp_rows:
        if r.get("bucket") == "summary":
            lines.append(f"| {r['source']} | {r['dataset']} | {fnum(r,'amp_anomaly_vs_normal_AUROC'):.4f} | {fnum(r,'anomaly_amp_mean'):.4f} | {fnum(r,'normal_amp_mean'):.4f} | {fnum(r,'FN_rescue_mean'):+.4f} | {fnum(r,'FP_minus_TN_mean'):+.4f} |")
    lines += ["", "## Morphology", "", "| Source | Target | Group | AUC(a) | anomaly mean | normal mean |", "|---|---|---|---:|---:|---:|"]
    for r in amp_rows:
        if r.get("bucket") == "morphology_summary":
            lines.append(f"| {r['source']} | {r['dataset']} | {r['group']} | {fnum(r,'amp_anomaly_vs_normal_AUROC'):.4f} | {fnum(r,'anomaly_amp_mean'):.4f} | {fnum(r,'normal_amp_mean'):.4f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- `q_g` is the global observation map resized to 74x74.",
        "- `q_l` is the fixed quadrant local re-encoding map stitched at 74x74.",
        "- `a = q_l - q_g`; E1/E2 add only positive amplification through fixed ReLU gates.",
        "- GT is used only for post-hoc diagnostics and full-resolution metric computation.",
        f"- Runtime: {runtime:.1f}s.",
    ]
    write_text(OUT / "REPORT.md", "\n".join(lines))


def run(smoke=False):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, clip_model = load_runtime(device)
    raw, cats, amp_rows, meta_all, image_rows = [], [], [], [], []
    threshold_rows = []
    for model_info in source_models(device):
        cells, _ = protocol_cells(model_info["source"], device, smoke)
        for _cell_id, target, dataset, refs, patch, kind in cells:
            if (model_info["source"], target) not in AUDIT_PAIRS:
                continue
            n = min(24, patch.shape[0]) if smoke else patch.shape[0]
            refs_use = refs[:n]
            patch_use = patch[:n]
            print(f"[FOVA] evaluate {model_info['source']} -> {target} n={n}", flush=True)
            maps, qg, ql, amp, metas, meta_rows, img_rows, thr = build_fova_maps(model_info, clip_model, preprocess, target, dataset, refs_use, patch_use, kind, device, smoke)
            rr, cc = evaluate_maps(model_info["source"], target, refs_use, dataset, kind, maps, metas, variants=VARIANTS)
            raw.extend(rr)
            cats.extend(cc)
            amp_rows.extend(amplification_diagnostics(model_info["source"], target, qg, amp, metas, meta_rows))
            meta_all.extend(meta_rows)
            image_rows.extend(img_rows)
            threshold_rows.append({"source": model_info["source"], "dataset": target, **thr})
            if not smoke:
                write_csv(OUT / "RAW_METRICS.partial.csv", raw)
                write_csv(OUT / "AMPLIFICATION_AUDIT.partial.csv", amp_rows)
    summary = summarize_metrics(raw)
    outcome = decide(summary, amp_rows)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "FUSION_DELTA_SUMMARY.csv", summary)
    write_csv(OUT / "AMPLIFICATION_AUDIT.csv", amp_rows)
    write_csv(OUT / "IMAGE_AMPLIFICATION_SUMMARY.csv", image_rows)
    write_csv(OUT / "MORPHOLOGY_GROUPS.csv", meta_all)
    write_csv(OUT / "MORPHOLOGY_THRESHOLDS.csv", threshold_rows)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "runtime_sec": runtime, "variants": VARIANTS, "no_train": True, "no_target_tuning": True})
    write_report(raw, summary, amp_rows, outcome, threshold_rows, runtime)
    print(f"[FOVA] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args.smoke)


if __name__ == "__main__":
    main()
