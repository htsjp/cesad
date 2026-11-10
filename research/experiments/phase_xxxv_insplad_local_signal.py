import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, label as cc_label
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxiv_insplad_cross_asset import (
    CACHE,
    DATA_ROOT,
    OUT as XXXIV_OUT,
    PATCHES,
    PREVIOUS_VERIFIED_BUDGET,
    SIDE,
    TOPK,
    asset_records,
    build_memory,
    collect_dataset,
    image_score_from_patch_scores,
    mask_for,
    safe_ap,
    safe_auc,
    topk_distances,
    upsample_patch_scores,
)


OUT = Path("results/phase_xxxv_insplad_local_signal")
EPS = 1e-12
PAIR_LIMIT = 1_000_000


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def cohen_d(normal, anomaly):
    a = np.asarray(normal, dtype=np.float64)
    b = np.asarray(anomaly, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = math.sqrt(((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / max(1, a.size + b.size - 2))
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else float("nan")


def stats(prefix, arr):
    x = np.asarray(arr, dtype=np.float64)
    if x.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ["mean", "median", "std", "q10", "q25", "q75", "q90"]}
    qs = np.quantile(x, [0.10, 0.25, 0.75, 0.90])
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_q10": float(qs[0]),
        f"{prefix}_q25": float(qs[1]),
        f"{prefix}_q75": float(qs[2]),
        f"{prefix}_q90": float(qs[3]),
    }


def method_name(carrier, mode):
    return {
        ("clip", "fps"): "M0_CLIP_FPS",
        ("clip", "random"): "M1_CLIP_RANDOM",
        ("dino", "fps"): "M2_DINO_FPS",
        ("dino", "random"): "M3_DINO_RANDOM",
    }[(carrier, mode)]


def load_feature_index(assets):
    feature_paths, metas = {}, {}
    for carrier in ["clip", "dino"]:
        for asset in assets:
            meta_path = CACHE / f"features_{carrier}_{asset['name']}.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            patch_path = Path(meta.get("patch_npy", CACHE / f"features_{carrier}_{asset['name']}.patch.npy"))
            feature_paths[(carrier, asset["name"])] = patch_path
            metas[(carrier, asset["name"])] = meta
    return feature_paths, metas


def border_distance_shape(shape):
    h, w = shape
    yy, xx = np.indices((h, w))
    d = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]).astype(np.float32)
    return np.clip(d / max(1.0, min(h, w) / 2.0), 0.0, 1.0)


def patch_labels_from_mask(mask, mode):
    h, w = mask.shape
    cell_h = h / SIDE
    cell_w = w / SIDE
    labels = np.zeros((SIDE, SIDE), dtype=np.uint8)
    for py in range(SIDE):
        y0 = int(round(py * cell_h))
        y1 = int(round((py + 1) * cell_h))
        for px in range(SIDE):
            x0 = int(round(px * cell_w))
            x1 = int(round((px + 1) * cell_w))
            crop = mask[y0:y1, x0:x1]
            frac = float(crop.mean()) if crop.size else 0.0
            labels[py, px] = int(frac > 0.0) if mode == "any" else int(frac >= 0.50)
    return labels.reshape(-1)


def pair_accuracy(normal_scores, anomaly_scores, rng):
    n = np.asarray(normal_scores, dtype=np.float32)
    a = np.asarray(anomaly_scores, dtype=np.float32)
    if n.size == 0 or a.size == 0:
        return float("nan"), 0
    total = n.size * a.size
    sample = min(PAIR_LIMIT, total)
    ai = rng.integers(0, a.size, size=sample)
    ni = rng.integers(0, n.size, size=sample)
    return float(np.mean(a[ai] > n[ni])), int(sample)


def evaluate_method(target, carrier, mode, bank, entries, feature_paths, metas, device, full_forensics=False):
    patch_arr = np.load(feature_paths[(carrier, target)], mmap_mode="r")
    meta = metas[(carrier, target)]
    rows = [r for r in meta["rows"] if r["split"] == "test"]
    positions = [i for i, r in enumerate(meta["rows"]) if r["split"] == "test"]
    method = method_name(carrier, mode)

    y_img, s_img = [], []
    y_pix_parts, s_pix_parts = [], []
    image_parts = []
    per_image_rows = []
    patch_rows = []
    border_rows = []
    boundary_rows = []
    hotspot_rows = []
    coverage_rows = []
    image_forensics = []
    normal_top_scores, anomaly_top_scores = [], []
    dist_top1_normal, dist_top1_anom, dist_top5_normal, dist_top5_anom, disp_normal, disp_anom = [], [], [], [], [], []

    for pos, r in zip(positions, rows):
        patch = patch_arr[pos].astype(np.float32)
        dist, idx = topk_distances(patch, bank, TOPK, device)
        patch_score = dist.mean(axis=1).reshape(SIDE, SIDE)
        top1 = dist[:, 0].reshape(SIDE, SIDE)
        disp = dist.std(axis=1).reshape(SIDE, SIDE)
        img_score = image_score_from_patch_scores(patch_score)
        y_img.append(int(r["label"]))
        s_img.append(img_score)

        if int(r["label"]) == 1:
            with Image.open(r["mask_path"]) as m:
                mask = (np.asarray(m.convert("L")) > 0).astype(np.uint8)
            full = upsample_patch_scores(patch_score, mask.shape)
            full_top1 = upsample_patch_scores(top1, mask.shape)
            full_disp = upsample_patch_scores(disp, mask.shape)
            if full_forensics and mask.max() > 0 and mask.min() == 0:
                pa = safe_auc(mask.reshape(-1), full.reshape(-1))
                pap = safe_ap(mask.reshape(-1), full.reshape(-1))
                prev = float(mask.mean() * 100.0)
                per_image_rows.append({"target_asset": target, "method": method, "image": r["path"], "P-AUROC": pa, "P-AP": pap, "pixel_prevalence": prev, "P-AP_lift": pap - prev})
            if full_forensics:
                for grid_mode in ["any", "majority"]:
                    py = patch_labels_from_mask(mask, grid_mode)
                    ps = patch_score.reshape(-1)
                    patch_rows.append({"target_asset": target, "method": method, "grid_definition": f"G1_ANY_OVERLAP" if grid_mode == "any" else "G2_MAJORITY", "Patch_AUROC": safe_auc(py, ps), "Patch_AP": safe_ap(py, ps), "positive_patches": int(py.sum()), "total_patches": int(py.size)})

            anomaly_vals = full[mask.astype(bool)]
            normal_vals = full[~mask.astype(bool)]
            if full_forensics:
                dist_top1_anom.extend(full_top1[mask.astype(bool)].ravel().tolist())
                dist_top1_normal.extend(full_top1[~mask.astype(bool)].ravel().tolist())
                dist_top5_anom.extend(full[mask.astype(bool)].ravel().tolist())
                dist_top5_normal.extend(full[~mask.astype(bool)].ravel().tolist())
                disp_anom.extend(full_disp[mask.astype(bool)].ravel().tolist())
                disp_normal.extend(full_disp[~mask.astype(bool)].ravel().tolist())

                eroded = binary_erosion(mask.astype(bool), iterations=3)
                boundary = mask.astype(bool) & ~eroded
                boundary_rows.append({"target_asset": target, "method": method, "image": r["path"], "interior_mean_score": float(np.mean(full[eroded])) if eroded.any() else float("nan"), "boundary_band_mean_score": float(np.mean(full[boundary])) if boundary.any() else float("nan"), "boundary_pixels": int(boundary.sum()), "interior_pixels": int(eroded.sum())})
                flat = full.reshape(-1)
                mflat = mask.reshape(-1).astype(bool)
                cov = {"target_asset": target, "method": method, "image": r["path"]}
                for pct in [1, 5, 10]:
                    k = max(1, int(math.ceil(flat.size * pct / 100.0)))
                    ids = np.argpartition(flat, -k)[-k:]
                    cov[f"Precision@Top{pct}%"] = float(np.mean(mflat[ids]))
                coverage_rows.append(cov)
        else:
            with Image.open(r["path"]) as im:
                shape = (im.size[1], im.size[0])
            mask = np.zeros(shape, dtype=np.uint8)
            full = upsample_patch_scores(patch_score, shape)
            full_top1 = upsample_patch_scores(top1, shape)
            full_disp = upsample_patch_scores(disp, shape)
            if full_forensics:
                d_border = border_distance_shape(shape)
                edge = 1.0 - d_border
                rho = float(spearmanr(full.reshape(-1), edge.reshape(-1)).statistic)
                thresh = np.quantile(full, 0.95)
                top = full >= thresh
                border_band = d_border <= 0.10
                border_rows.append({"target_asset": target, "method": method, "image": r["path"], "spearman_score_vs_1minus_border_distance": rho, "top5pct_in_outer10pct_border_band": float(np.mean(border_band[top])) if top.any() else float("nan"), "BORDER_CONFOUND_IMAGE": bool(rho >= 0.40 and (np.mean(border_band[top]) if top.any() else 0.0) > 0.50)})
                q99 = np.quantile(full, 0.99)
                labels, nlab = cc_label(full >= q99)
                max_area = 0
                for lab in range(1, nlab + 1):
                    max_area = max(max_area, int((labels == lab).sum()))
                hotspot_rows.append({"target_asset": target, "method": method, "image": r["path"], "top1pct_pixel_mean": image_score_from_patch_scores(full), "q99_score": float(q99), "max_connected_hotspot_area_q99": max_area})

        y_pix_parts.append(mask.reshape(-1).astype(np.uint8))
        s_pix_parts.append(full.reshape(-1).astype(np.float32))
        image_parts.append({"label": int(r["label"]), "path": r["path"], "y": mask.reshape(-1).astype(np.uint8), "s": full.reshape(-1).astype(np.float32)})
        flat = full.reshape(-1)
        if full_forensics:
            image_forensics.append({"target_asset": target, "method": method, "image": r["path"], "label": int(r["label"]), "frozen_I_score": img_score, "max_pixel_score": float(np.max(flat)), "q99_pixel_score": float(np.quantile(flat, 0.99)), "top1pct_pixel_mean": float(np.mean(np.sort(flat)[-max(1, int(math.ceil(flat.size * 0.01))):]))})
            if int(r["label"]) == 0:
                normal_top_scores.append(image_forensics[-1]["top1pct_pixel_mean"])
            else:
                anomaly_top_scores.append(image_forensics[-1]["top1pct_pixel_mean"])

    y_img = np.asarray(y_img, dtype=np.uint8)
    s_img = np.asarray(s_img, dtype=np.float32)
    y_pix = np.concatenate(y_pix_parts)
    s_pix = np.concatenate(s_pix_parts).astype(np.float32)
    metric = {
        "target_asset": target,
        "method": method,
        "carrier": carrier.upper(),
        "memory_mode": mode,
        "I-AUROC": safe_auc(y_img, s_img),
        "I-AP": safe_ap(y_img, s_img),
        "P-AUROC": safe_auc(y_pix, s_pix),
        "P-AP": safe_ap(y_pix, s_pix),
        "pixel_prevalence": float(y_pix.mean() * 100.0),
        "P-AP_lift": safe_ap(y_pix, s_pix) - float(y_pix.mean() * 100.0),
        "AP_RATIO": safe_ap(y_pix, s_pix) / max(EPS, float(y_pix.mean() * 100.0)),
    }

    conc_rows = []
    anom_parts = [p for p in image_parts if p["label"] == 1]
    base_pa, base_pap = metric["P-AUROC"], metric["P-AP"]
    if full_forensics:
        for part in anom_parts:
            conc_rows.append({"target_asset": target, "method": method, "removed_image": part["path"], "loo_P-AUROC": "", "loo_P-AP": "", "delta_P-AUROC": "", "delta_P-AP": "", "relative_abs_delta_P-AP": "", "status": "NOT_AVAILABLE_DUE_TO_COMPUTE_POLICY_FULL_RES_LOO_SORT"})

    normal_pix = s_pix[y_pix == 0]
    anomaly_pix = s_pix[y_pix == 1]
    rng = np.random.default_rng(0)
    pair_acc, pair_n = pair_accuracy(normal_pix, anomaly_pix, rng) if full_forensics else (float("nan"), 0)
    dist_row = {"target_asset": target, "method": method}
    if full_forensics:
        dist_row.update(stats("normal_score", normal_pix))
        dist_row.update(stats("anomaly_score", anomaly_pix))
        dist_row.update({"cohen_d": cohen_d(normal_pix, anomaly_pix), "ks_statistic": float(ks_2samp(normal_pix, anomaly_pix).statistic), "wasserstein_distance": float(wasserstein_distance(normal_pix, anomaly_pix)), "median_gap_anomaly_minus_normal": float(np.median(anomaly_pix) - np.median(normal_pix))})
    else:
        dist_row["status"] = "PRIMARY_ONLY_NOT_COMPUTED_FOR_SECONDARY"
    pair_row = {"target_asset": target, "method": method, "PairAccuracy": pair_acc, "sampled_pairs": pair_n, "normal_pixels": int(normal_pix.size), "anomaly_pixels": int(anomaly_pix.size)}
    local_to_image = {"target_asset": target, "method": method, "LOCAL_TO_IMAGE_AUROC_top1pct": safe_auc(y_img, np.asarray([r["top1pct_pixel_mean"] for r in image_forensics], dtype=np.float32))} if full_forensics else {"target_asset": target, "method": method, "status": "PRIMARY_ONLY_NOT_COMPUTED_FOR_SECONDARY"}
    retrieval_row = {"target_asset": target, "method": method, "top1_distance_AUROC": safe_auc(np.r_[np.zeros(len(dist_top1_normal)), np.ones(len(dist_top1_anom))], np.r_[dist_top1_normal, dist_top1_anom]), "top1_distance_AP": safe_ap(np.r_[np.zeros(len(dist_top1_normal)), np.ones(len(dist_top1_anom))], np.r_[dist_top1_normal, dist_top1_anom]), "top1_cohen_d": cohen_d(dist_top1_normal, dist_top1_anom), "top1_ks": float(ks_2samp(dist_top1_normal, dist_top1_anom).statistic), "top5_mean_distance_AUROC": safe_auc(np.r_[np.zeros(len(dist_top5_normal)), np.ones(len(dist_top5_anom))], np.r_[dist_top5_normal, dist_top5_anom]), "top5_mean_distance_AP": safe_ap(np.r_[np.zeros(len(dist_top5_normal)), np.ones(len(dist_top5_anom))], np.r_[dist_top5_normal, dist_top5_anom]), "top5_dispersion_AUROC": safe_auc(np.r_[np.zeros(len(disp_normal)), np.ones(len(disp_anom))], np.r_[disp_normal, disp_anom])} if full_forensics else {"target_asset": target, "method": method, "status": "PRIMARY_ONLY_NOT_COMPUTED_FOR_SECONDARY"}
    return {
        "metric": metric,
        "per_image": per_image_rows,
        "concentration": conc_rows,
        "distribution": dist_row,
        "pair": pair_row,
        "patch": patch_rows,
        "border": border_rows,
        "boundary": boundary_rows,
        "hotspot": hotspot_rows,
        "coverage": coverage_rows,
        "image_forensics": image_forensics,
        "local_to_image": local_to_image,
        "retrieval_distance": retrieval_row,
    }


def summarize_per_image(rows):
    out = []
    by = defaultdict(list)
    for r in rows:
        by[(r["target_asset"], r["method"])].append(r)
    for (asset, method), vals in sorted(by.items()):
        au = np.asarray([r["P-AUROC"] for r in vals], dtype=np.float64)
        ap_lift = np.asarray([r["P-AP_lift"] for r in vals], dtype=np.float64)
        out.append({"target_asset": asset, "method": method, "per_image_P-AUROC_median": float(np.median(au)), "per_image_P-AUROC_q25": float(np.quantile(au, 0.25)), "per_image_P-AUROC_q75": float(np.quantile(au, 0.75)), "per_image_P-AUROC_worst": float(np.min(au)), "per_image_P-AUROC_best": float(np.max(au)), "fraction_P-AUROC_gt70": float(np.mean(au > 70.0)), "per_image_AP_lift_median": float(np.median(ap_lift)), "fraction_AP_lift_gt0": float(np.mean(ap_lift > 0.0))})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    assets, count_rows, _, _ = collect_dataset()
    asset_names = [a["name"] for a in assets]
    feature_paths, metas = load_feature_index(assets)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    raw_xxxiv = read_csv(XXXIV_OUT / "RAW_METRICS.csv")
    method_map = {("CLIP", "fps"): "M0_CLIP_FPS", ("CLIP", "random"): "M1_CLIP_RANDOM", ("DINO", "fps"): "M2_DINO_FPS", ("DINO", "random"): "M3_DINO_RANDOM"}
    all_outputs = []
    memory_comp_rows = []
    for target in asset_names:
        sources = [s for s in asset_names if s != target]
        eligible = sum(len([r for r in metas[("clip", s)]["rows"] if r["split"] == "train" and int(r["label"]) == 0]) * PATCHES for s in sources)
        budget = min(PREVIOUS_VERIFIED_BUDGET, eligible)
        print(f"[XXXV-A] target {target}, budget {budget}", flush=True)
        for carrier, mode in [("clip", "fps"), ("clip", "random"), ("dino", "fps"), ("dino", "random")]:
            print(f"[XXXV-A] {target} {method_name(carrier, mode)}", flush=True)
            bank, entries, comp = build_memory(sources, carrier, feature_paths, metas, budget, mode, seed=0)
            memory_comp_rows.extend([{"target_asset": target, **r} for r in comp])
            all_outputs.append(evaluate_method(target, carrier, mode, bank, entries, feature_paths, metas, device, full_forensics=(carrier == "clip" and mode == "fps")))

    pixel_metrics = [o["metric"] for o in all_outputs]
    parity_rows = []
    max_pauroc_delta = 0.0
    max_primary_delta = 0.0
    for r in pixel_metrics:
        old = next(x for x in raw_xxxiv if x["target_asset"] == r["target_asset"] and x["carrier"] == r["carrier"] and x["memory_mode"] == r["memory_mode"])
        row = {
            "target_asset": r["target_asset"],
            "method": r["method"],
            "delta_I-AUROC": r["I-AUROC"] - float(old["I-AUROC"]),
            "delta_I-AP": r["I-AP"] - float(old["I-AP"]),
            "delta_P-AUROC": r["P-AUROC"] - float(old["P-AUROC"]),
            "delta_P-AP": r["P-AP"] - float(old["P-AP"]),
            "new_P-AUROC": r["P-AUROC"],
            "old_P-AUROC": float(old["P-AUROC"]),
        }
        max_pauroc_delta = max(max_pauroc_delta, abs(row["delta_P-AUROC"]))
        max_primary_delta = max(max_primary_delta, abs(row["delta_I-AUROC"]), abs(row["delta_I-AP"]), abs(row["delta_P-AUROC"]))
        parity_rows.append(row)
    clip_fps_mean = float(np.mean([r["P-AUROC"] for r in pixel_metrics if r["method"] == "M0_CLIP_FPS"]))
    parity = {"passed": bool(abs(clip_fps_mean - 84.02794718277613) <= 0.02 and max_primary_delta <= 0.02), "clip_fps_mean_P-AUROC": clip_fps_mean, "frozen_clip_fps_mean_P-AUROC": 84.02794718277613, "max_primary_metric_delta": max_primary_delta, "rows": parity_rows}
    (OUT / "PARITY_AUDIT.json").write_text(json.dumps(parity, indent=2, default=json_default), encoding="utf-8")

    per_image = sum([o["per_image"] for o in all_outputs], [])
    concentration = sum([o["concentration"] for o in all_outputs if o["metric"]["method"] == "M0_CLIP_FPS"], [])
    distributions = [o["distribution"] for o in all_outputs]
    pairs = [o["pair"] for o in all_outputs]
    patch_rows = sum([o["patch"] for o in all_outputs], [])
    border_rows = sum([o["border"] for o in all_outputs if o["metric"]["method"] == "M0_CLIP_FPS"], [])
    boundary_rows = sum([o["boundary"] for o in all_outputs if o["metric"]["method"] == "M0_CLIP_FPS"], [])
    hotspot_rows = sum([o["hotspot"] for o in all_outputs if o["metric"]["method"] == "M0_CLIP_FPS"], [])
    coverage_rows = sum([o["coverage"] for o in all_outputs if o["metric"]["method"] == "M0_CLIP_FPS"], [])
    image_forensics = sum([o["image_forensics"] for o in all_outputs if o["metric"]["method"] == "M0_CLIP_FPS"], [])
    local_to_image = [o["local_to_image"] for o in all_outputs if o["metric"]["method"] == "M0_CLIP_FPS"]
    retrieval_distance = [o["retrieval_distance"] for o in all_outputs if o["metric"]["method"] == "M0_CLIP_FPS"]

    prevalence_rows = [{"target_asset": r["target_asset"], "method": r["method"], "pixel_prevalence": r["pixel_prevalence"], "P-AP": r["P-AP"], "P-AP_lift": r["P-AP_lift"], "AP_RATIO": r["AP_RATIO"]} for r in pixel_metrics]
    per_asset_signal = []
    for r in [x for x in pixel_metrics if x["method"] == "M0_CLIP_FPS"]:
        count = next(c for c in count_rows if c["asset"] == r["target_asset"])
        conc = [c for c in concentration if c["target_asset"] == r["target_asset"]]
        numeric_dpau = [abs(c["delta_P-AUROC"]) for c in conc if isinstance(c["delta_P-AUROC"], (int, float, np.floating))]
        numeric_dpap = [c["relative_abs_delta_P-AP"] for c in conc if isinstance(c["relative_abs_delta_P-AP"], (int, float, np.floating))]
        max_dpau = max(numeric_dpau) if numeric_dpau else float("nan")
        max_dpap_rel = max(numeric_dpap) if numeric_dpap else float("nan")
        per_asset_signal.append({"target_asset": r["target_asset"], "normal_count": count["test_normal"], "anomaly_count": count["test_anomaly"], "pixel_prevalence": r["pixel_prevalence"], "P-AUROC": r["P-AUROC"], "P-AP": r["P-AP"], "P-AP_lift": r["P-AP_lift"], "AP_RATIO": r["AP_RATIO"], "asset_local_signal": bool(r["P-AUROC"] >= 75.0 and r["P-AP_lift"] > 2.0), "max_leave_one_image_abs_delta_P-AUROC": max_dpau, "max_leave_one_image_relative_abs_delta_P-AP": max_dpap_rel, "IMAGE_CONCENTRATED": bool((np.isfinite(max_dpau) and max_dpau > 2.0) or (np.isfinite(max_dpap_rel) and max_dpap_rel > 0.20)), "image_concentration_status": "NOT_AVAILABLE_DUE_TO_COMPUTE_POLICY_FULL_RES_LOO_SORT" if not numeric_dpau else "AVAILABLE"})

    fps_vs_random = []
    for asset in asset_names:
        fps = next(r for r in pixel_metrics if r["target_asset"] == asset and r["method"] == "M0_CLIP_FPS")
        rnd = next(r for r in pixel_metrics if r["target_asset"] == asset and r["method"] == "M1_CLIP_RANDOM")
        fps_vs_random.append({"target_asset": asset, "dP-AUROC": fps["P-AUROC"] - rnd["P-AUROC"], "dP-AP": fps["P-AP"] - rnd["P-AP"], "G_local": 0.5 * ((fps["P-AUROC"] - rnd["P-AUROC"]) + (fps["P-AP"] - rnd["P-AP"])), "FPS_better": 0.5 * ((fps["P-AUROC"] - rnd["P-AUROC"]) + (fps["P-AP"] - rnd["P-AP"])) > 0})

    border_summary = []
    for asset in asset_names:
        vals = [r for r in border_rows if r["target_asset"] == asset]
        if vals:
            border_summary.append({"target_asset": asset, "method": "M0_CLIP_FPS", "normal_image_mean_spearman": float(np.nanmean([r["spearman_score_vs_1minus_border_distance"] for r in vals])), "top5pct_outer10pct_border_band_mean": float(np.nanmean([r["top5pct_in_outer10pct_border_band"] for r in vals])), "BORDER_CONFOUND": bool(np.nanmean([r["spearman_score_vs_1minus_border_distance"] for r in vals]) >= 0.40 and np.nanmean([r["top5pct_in_outer10pct_border_band"] for r in vals]) > 0.50)})

    hotspot_summary = []
    for asset in asset_names:
        vals = [r for r in hotspot_rows if r["target_asset"] == asset]
        if vals:
            hotspot_summary.append({"target_asset": asset, "method": "M0_CLIP_FPS", "top1pct_pixel_mean_median": float(np.median([r["top1pct_pixel_mean"] for r in vals])), "top1pct_pixel_mean_q90": float(np.quantile([r["top1pct_pixel_mean"] for r in vals], 0.90)), "q99_score_median": float(np.median([r["q99_score"] for r in vals])), "q99_score_q90": float(np.quantile([r["q99_score"] for r in vals], 0.90)), "max_connected_hotspot_area_median": float(np.median([r["max_connected_hotspot_area_q99"] for r in vals])), "max_connected_hotspot_area_q90": float(np.quantile([r["max_connected_hotspot_area_q99"] for r in vals], 0.90))})

    image_corr_rows = []
    for asset in asset_names:
        vals = [r for r in image_forensics if r["target_asset"] == asset]
        for score_name in ["max_pixel_score", "q99_pixel_score", "top1pct_pixel_mean"]:
            image_corr_rows.append({"target_asset": asset, "method": "M0_CLIP_FPS", "score": score_name, "spearman_vs_frozen_I_score": float(spearmanr([r["frozen_I_score"] for r in vals], [r[score_name] for r in vals]).statistic)})

    write_csv(OUT / "PIXEL_METRICS.csv", pixel_metrics)
    write_csv(OUT / "PIXEL_PREVALENCE.csv", prevalence_rows)
    write_csv(OUT / "PER_IMAGE_PIXEL_METRICS.csv", per_image)
    write_csv(OUT / "PER_ASSET_LOCAL_SIGNAL.csv", per_asset_signal)
    write_csv(OUT / "SCORE_DISTRIBUTIONS.csv", distributions)
    write_csv(OUT / "PAIR_RANKING.csv", pairs)
    write_csv(OUT / "FPS_VS_RANDOM_PIXEL.csv", fps_vs_random)
    write_csv(OUT / "PATCH_GRID_ROBUSTNESS.csv", patch_rows)
    write_csv(OUT / "BORDER_CONFOUND.csv", border_summary)
    write_csv(OUT / "MASK_BOUNDARY_DIAGNOSTIC.csv", boundary_rows)
    write_csv(OUT / "NORMAL_FALSE_HOTSPOT.csv", hotspot_summary)
    write_csv(OUT / "DEFECT_TOPK_COVERAGE.csv", coverage_rows)
    write_csv(OUT / "IMAGE_LEVEL_FORENSICS.csv", image_corr_rows + image_forensics)
    write_csv(OUT / "LOCAL_TO_IMAGE.csv", local_to_image)
    write_csv(OUT / "RETRIEVAL_DISTANCE_PIXEL.csv", retrieval_distance)

    clip_fps = [r for r in pixel_metrics if r["method"] == "M0_CLIP_FPS"]
    mean_pauroc = float(np.mean([r["P-AUROC"] for r in clip_fps]))
    mean_ap_lift = float(np.mean([r["P-AP_lift"] for r in clip_fps]))
    asset_pauroc_pass = int(sum(r["P-AUROC"] >= 75.0 for r in clip_fps))
    asset_ap_pass = int(sum(r["P-AP_lift"] > 2.0 for r in clip_fps))
    pos_lifts = [max(0.0, r["P-AP_lift"]) for r in clip_fps]
    max_asset_share = max(pos_lifts) / max(EPS, sum(pos_lifts))
    local_supported = mean_pauroc >= 80.0 and asset_pauroc_pass >= 2 and mean_ap_lift >= 5.0 and asset_ap_pass >= 2 and max_asset_share <= 0.60
    auroc_only = mean_pauroc >= 80.0 and asset_pauroc_pass >= 2 and not (mean_ap_lift >= 5.0 and asset_ap_pass >= 2)
    asset_specific = sum(r["asset_local_signal"] for r in per_asset_signal) == 1
    mean_g_local = float(np.mean([r["G_local"] for r in fps_vs_random]))
    fps_win_frac = float(np.mean([r["FPS_better"] for r in fps_vs_random]))
    memory_content = mean_g_local >= 0.20 and fps_win_frac >= 2 / 3
    image_concentrated = any(r["IMAGE_CONCENTRATED"] for r in per_asset_signal)
    if not parity["passed"]:
        outcome = "XXXV_A_IMPLEMENTATION_FAILURE"
    elif local_supported and not image_concentrated and memory_content:
        outcome = "XXXV_A_LOCAL_MEMORY_SIGNAL_SUPPORTED"
    elif local_supported:
        outcome = "XXXV_A_LOCAL_SUPPORT_SIGNAL_IDENTITY_WEAK"
    elif asset_specific:
        outcome = "XXXV_A_LOCAL_SIGNAL_ASSET_SPECIFIC"
    elif auroc_only:
        outcome = "XXXV_A_AUROC_ONLY_IMBALANCE_SIGNAL"
    else:
        outcome = "XXXV_A_NO_MEANINGFUL_LOCAL_SIGNAL"

    report = ["# Phase XXXV-A InsPLAD Local Anomaly Signal Forensics", "", f"Outcome: **{outcome}**", "", "No backbone forward was run; XXXIV patch feature cache was reused.", "", "## CLIP FPS Local Signal", "", "| Asset | P-AUROC | P-AP | Pixel prev | AP lift | AP ratio | Image concentrated |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in per_asset_signal:
        report.append(f"| {r['target_asset']} | {r['P-AUROC']:.4f} | {r['P-AP']:.4f} | {r['pixel_prevalence']:.4f} | {r['P-AP_lift']:+.4f} | {r['AP_RATIO']:.4f} | {r['IMAGE_CONCENTRATED']} |")
    report.extend(["", "## Gates", "", f"- Parity passed: {parity['passed']} (CLIP FPS mean P-AUROC {clip_fps_mean:.4f}).", f"- mean P-AUROC={mean_pauroc:.4f}; assets P-AUROC>=75: {asset_pauroc_pass}/3.", f"- mean P-AP lift={mean_ap_lift:+.4f}; assets AP lift>2: {asset_ap_pass}/3; max positive AP-lift share={max_asset_share:.4f}.", f"- FPS vs Random mean G_local={mean_g_local:+.4f}; FPS win fraction={fps_win_frac:.4f}.", f"- Runtime: {time.time() - start:.1f}s.", "", "Stop: Phase XXXV-A ended here."])
    (OUT / "XXXV_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    payload = {"phase": "XXXV-A", "outcome": outcome, "parity_passed": parity["passed"], "mean_clip_fps_P-AUROC": mean_pauroc, "mean_clip_fps_P-AP_lift": mean_ap_lift, "asset_pauroc_pass_count": asset_pauroc_pass, "asset_ap_lift_pass_count": asset_ap_pass, "max_asset_ap_lift_share": max_asset_share, "mean_G_local_FPS_minus_RANDOM": mean_g_local, "fps_win_fraction": fps_win_frac, "image_concentrated": image_concentrated, "runtime_sec": time.time() - start}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXXV-A\n")
        fp.write("- Command: python experiments/phase_xxxv_insplad_local_signal.py\n")
        fp.write(f"- XXXIV parity passed: {parity['passed']}; CLIP FPS mean P-AUROC {mean_pauroc:.4f}.\n")
        for r in per_asset_signal:
            fp.write(f"- {r['target_asset']}: P-AUROC {r['P-AUROC']:.4f}; P-AP {r['P-AP']:.4f}; prevalence {r['pixel_prevalence']:.4f}; AP lift {r['P-AP_lift']:+.4f}; image concentrated {r['IMAGE_CONCENTRATED']}.\n")
        fp.write(f"- FPS vs Random mean G_local {mean_g_local:+.4f}; FPS win fraction {fps_win_frac:.4f}.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXXV_A_REPORT.md"), "runtime_sec": time.time() - start}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
