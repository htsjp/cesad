import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.stats import kendalltau, pearsonr, spearmanr
from skimage import measure
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_checkpoint_source_readout import direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from experiments.phase_xxiii_b_dual_carrier_local import IMAGE_SIZE, SIGMA, apply_cdf, smooth_full


XXIV = Path("results/phase_xxiv_pixel_calibration/final_target")
OUT = Path("results/phase_xxv_calibration_mechanism")
FPRS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
AMP_BINS = [(0, 50), (50, 75), (75, 90), (90, 95), (95, 99), (99, 100)]
SAMPLE_PIXELS = 4096
BG_SAMPLE_CAP = 5_000_000
PAIR_SAMPLE = 250_000


def jdefault(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(type(obj).__name__)


def direction_target(direction):
    source, _, _ = direction_cfg(direction)
    if direction == "D1":
        return source, "visa", "data/VISA"
    return source, "mvtec", "data/mvtec"


def progress(stage, **extra):
    payload = {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    (OUT / "PROGRESS.json").write_text(json.dumps(payload, indent=2, default=jdefault), encoding="utf-8")
    print(f"[XXV] {stage}", flush=True)


def load_direction(direction):
    source, target, target_path = direction_target(direction)
    dataset = load_dataset(target, target_path)
    raw = np.load(XXIV / "cache" / f"{direction}_target_raw.npy", mmap_mode="r")
    scores = np.load(XXIV / "cache" / f"{direction}_target_scores.npy")
    cdf = np.load(XXIV / f"{direction}_PIXEL_CDF.npz")["cdf"].astype(np.float32)
    return source, target, dataset, list(range(len(dataset))), raw, scores, cdf


def mask_np(item):
    return np.squeeze(item["img_mask"].numpy()) > 0.5


def downsample_bool(mask, side=129):
    x = torch.tensor(mask.astype(np.float32)).view(1, 1, mask.shape[0], mask.shape[1])
    return (F.interpolate(x, size=(side, side), mode="nearest").squeeze().numpy() > 0.5)


def downsample_map(m, side=129):
    x = torch.tensor(m.astype(np.float32)).view(1, 1, m.shape[0], m.shape[1])
    return F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False).squeeze().numpy()


def maps_for(raw_map, cdf):
    m0_pre = np.asarray(raw_map, dtype=np.float32)
    m1_pre = apply_cdf(m0_pre, cdf)
    m0_post = smooth_full(m0_pre)
    m1_post = smooth_full(m1_pre)
    return m0_pre, m1_pre, m0_post, m1_post


def sample_flat_pair(a, b, n=SAMPLE_PIXELS):
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    if flat_a.size <= n:
        return flat_a, flat_b
    idx = np.linspace(0, flat_a.size - 1, n, dtype=np.int64)
    return flat_a[idx], flat_b[idx]


def rank_rows(direction, dataset, indices, raw, cdf):
    rows = []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        m0_pre, m1_pre, m0_post, m1_post = maps_for(raw[pos], cdf)
        a, b = sample_flat_pair(m0_pre, m1_pre)
        ap, bp = sample_flat_pair(m0_post, m1_post)
        rows.append({
            "direction": direction,
            "idx": idx,
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "pre_spearman": float(spearmanr(a, b).statistic),
            "pre_kendall_tau": float(kendalltau(a[:2048], b[:2048]).statistic),
            "post_spearman": float(spearmanr(ap, bp).statistic),
        })
    return rows


def amplitude_rows(direction, raw, cdf):
    acc = {bin_: {"raw": [], "cal": []} for bin_ in AMP_BINS}
    for pos in range(raw.shape[0]):
        m0 = np.asarray(raw[pos], dtype=np.float32)
        m1 = apply_cdf(m0, cdf)
        pct = m1 * 100.0
        for lo, hi in AMP_BINS:
            sel = (pct >= lo) & (pct < hi if hi < 100 else pct <= hi)
            vals0 = m0[sel]
            vals1 = m1[sel]
            if vals0.size:
                step = max(1, vals0.size // 20000)
                acc[(lo, hi)]["raw"].append(vals0[::step])
                acc[(lo, hi)]["cal"].append(vals1[::step])
    rows, prev = [], None
    for lo, hi in AMP_BINS:
        r = np.concatenate(acc[(lo, hi)]["raw"]) if acc[(lo, hi)]["raw"] else np.asarray([], dtype=np.float32)
        c = np.concatenate(acc[(lo, hi)]["cal"]) if acc[(lo, hi)]["cal"] else np.asarray([], dtype=np.float32)
        raw_mean = float(np.mean(r)) if r.size else float("nan")
        cal_mean = float(np.mean(c)) if c.size else float("nan")
        slope = float("nan")
        if prev is not None and np.isfinite(raw_mean) and raw_mean != prev["raw_score_mean"]:
            slope = (cal_mean - prev["calibrated_score_mean"]) / (raw_mean - prev["raw_score_mean"])
        row = {"direction": direction, "percentile_bin": f"{lo}-{hi}", "raw_score_mean": raw_mean, "calibrated_score_mean": cal_mean, "finite_difference_amplification": slope, "sampled_pixels": int(r.size)}
        rows.append(row)
        prev = row
    return rows


def stats(vals):
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "q90": float("nan"), "q95": float("nan"), "q99": float("nan")}
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "q90": float(np.percentile(vals, 90)), "q95": float(np.percentile(vals, 95)), "q99": float(np.percentile(vals, 99))}


def anomaly_background_rows(direction, dataset, indices, raw, cdf):
    rows = []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        if int(item["anomaly"]) != 1:
            continue
        mask = mask_np(item)
        m0_pre, m1_pre, m0_post, m1_post = maps_for(raw[pos], cdf)
        for stage, name, m in [("pre", "T0", m0_pre), ("pre", "T1", m1_pre), ("post", "T0", m0_post), ("post", "T1", m1_post)]:
            inside, outside = m[mask], m[~mask]
            si, so = stats(inside), stats(outside)
            row = {"direction": direction, "idx": idx, "category": item["cls_name"], "stage": stage, "variant": name}
            for k, v in si.items():
                row[f"inside_{k}"] = v
            for k, v in so.items():
                row[f"outside_{k}"] = v
            row["inside_outside_mean_ratio"] = si["mean"] / (so["mean"] + 1e-6)
            row["inside_minus_outside_mean"] = si["mean"] - so["mean"]
            rows.append(row)
    return rows


def neighbor_corr(vals_a, vals_b):
    if vals_a.size < 2 or vals_b.size < 2 or np.std(vals_a) < 1e-12 or np.std(vals_b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(vals_a, vals_b)[0, 1])


def spatial_coherence_rows(direction, dataset, indices, raw, cdf):
    rows = []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        mask = mask_np(item)
        _, _, m0, m1 = maps_for(raw[pos], cdf)
        for variant, m in [("T0", m0), ("T1", m1)]:
            for region, sel in [("inside", mask), ("background", ~mask)]:
                vals = m[sel]
                if vals.size == 0:
                    continue
                right = sel[:, :-1] & sel[:, 1:]
                down = sel[:-1, :] & sel[1:, :]
                a = np.concatenate([m[:, :-1][right], m[:-1, :][down]])
                b = np.concatenate([m[:, 1:][right], m[1:, :][down]])
                rows.append({
                    "direction": direction,
                    "idx": idx,
                    "category": item["cls_name"],
                    "label": int(item["anomaly"]),
                    "variant": variant,
                    "region": region,
                    "local_cv": float(np.std(vals) / (np.mean(vals) + 1e-6)),
                    "neighbor_correlation": neighbor_corr(a, b),
                    "mean_abs_adjacent_difference": float(np.mean(np.abs(a - b))) if a.size else float("nan"),
                    "within_region_variance": float(np.var(vals)),
                })
    return rows


def background_thresholds(dataset, indices, raw, cdf):
    samples = {"T0": [], "T1": []}
    per_img_cap = max(256, BG_SAMPLE_CAP // max(1, len(indices)))
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        mask = mask_np(item)
        _, _, m0, m1 = maps_for(raw[pos], cdf)
        bg = ~mask
        for name, m in [("T0", m0), ("T1", m1)]:
            vals = m[bg]
            if vals.size:
                step = max(1, vals.size // per_img_cap)
                samples[name].append(vals[::step][:per_img_cap])
    thresholds = {}
    for name in ["T0", "T1"]:
        vals = np.concatenate(samples[name])
        thresholds[name] = {f: float(np.quantile(vals, 1.0 - f)) for f in FPRS}
    return thresholds


def region_props(mask):
    labels = measure.label(mask.astype(np.uint8))
    props = measure.regionprops(labels)
    return labels, props


def matched_fpr_and_components(direction, dataset, indices, raw, cdf, thresholds):
    coverage_rows, size_rows, comp_rows, normal_rows, pro_rows = [], [], [], [], []
    curves = {"T0": defaultdict(list), "T1": defaultdict(list)}
    size_records = []
    for idx in indices:
        item = dataset[idx]
        if int(item["anomaly"]) == 1:
            labels, props = region_props(mask_np(item))
            for prop in props:
                size_records.append(prop.area)
    q1, q2 = np.quantile(size_records, [1 / 3, 2 / 3]) if len(size_records) >= 3 else (0, 0)
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        mask = mask_np(item)
        labels, props = region_props(mask)
        _, _, m0, m1 = maps_for(raw[pos], cdf)
        for variant, m in [("T0", m0), ("T1", m1)]:
            for fpr in FPRS:
                binary = m > thresholds[variant][fpr]
                pred_labels, pred_props = region_props(binary)
                if int(item["anomaly"]) == 0:
                    areas = [p.area for p in pred_props]
                    total = float(sum(areas))
                    probs = np.asarray(areas, dtype=np.float64) / max(total, 1e-12) if areas else np.asarray([], dtype=np.float64)
                    normal_rows.append({"direction": direction, "variant": variant, "fpr": fpr, "idx": idx, "category": item["cls_name"], "islands": len(areas), "largest_island_area": float(max(areas)) if areas else 0.0, "spatial_entropy": float(-(probs * np.log(probs + 1e-12)).sum()) if probs.size else 0.0, "top1pct_component_size": float(max(areas)) if areas else 0.0})
                    continue
                overlaps = []
                best_component_cover = []
                intersecting = 0
                for prop in props:
                    region = labels == prop.label
                    hit = np.logical_and(binary, region)
                    ov = float(hit.sum() / max(1, prop.area))
                    overlaps.append(ov)
                    pred_ids = pred_labels[region]
                    pred_ids = pred_ids[pred_ids > 0]
                    if pred_ids.size:
                        intersecting += len(set(pred_ids.tolist()))
                        best = 0
                        for pid in set(pred_ids.tolist()):
                            best = max(best, int(np.logical_and(pred_labels == pid, region).sum()))
                        best_component_cover.append(float(best / max(1, prop.area)))
                    else:
                        best_component_cover.append(0.0)
                    size_group = "small" if prop.area <= q1 else ("medium" if prop.area <= q2 else "large")
                    compactness = float(4 * math.pi * prop.area / ((prop.perimeter + 1e-6) ** 2))
                    size_rows.append({"direction": direction, "variant": variant, "fpr": fpr, "idx": idx, "category": item["cls_name"], "region_area": float(prop.area), "size_group": size_group, "compactness": compactness, "thin_elongated": bool(compactness < 0.25), "region_recall": ov})
                areas = [p.area for p in pred_props]
                comp_rows.append({"direction": direction, "variant": variant, "fpr": fpr, "idx": idx, "category": item["cls_name"], "pred_components": len(areas), "mean_component_area": float(np.mean(areas)) if areas else 0.0, "largest_component_area": float(max(areas)) if areas else 0.0, "fragmentation_index": float(len(areas) / max(1, binary.sum() / 1000.0)), "components_intersecting_gt": intersecting, "mean_best_gt_region_component_coverage": float(np.mean(best_component_cover)) if best_component_cover else float("nan")})
                curves[variant][fpr].extend(overlaps)
    for variant in ["T0", "T1"]:
        for fpr in FPRS:
            vals = np.asarray(curves[variant][fpr], dtype=np.float32)
            coverage_rows.append({"direction": direction, "variant": variant, "fpr": fpr, "mean_region_overlap": float(np.mean(vals)), "median_region_overlap": float(np.median(vals)), "frac_overlap_ge_0p25": float(np.mean(vals >= 0.25)), "frac_overlap_ge_0p50": float(np.mean(vals >= 0.50)), "frac_overlap_ge_0p75": float(np.mean(vals >= 0.75)), "regions": int(vals.size)})
            pro_rows.append({"direction": direction, "variant": variant, "fpr": fpr, "pro": float(np.mean(vals))})
    return coverage_rows, size_rows, comp_rows, normal_rows, pro_rows


def core_boundary_rows(direction, dataset, indices, raw, cdf):
    rows = []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        if int(item["anomaly"]) != 1:
            continue
        mask = mask_np(item)
        core = binary_erosion(mask, iterations=1)
        boundary = mask & ~core
        background = ~mask
        _, _, m0, m1 = maps_for(raw[pos], cdf)
        for variant, m in [("T0", m0), ("T1", m1)]:
            bg = m[background]
            row = {"direction": direction, "variant": variant, "idx": idx, "category": item["cls_name"], "core_mean": float(np.mean(m[core])) if core.any() else float("nan"), "boundary_mean": float(np.mean(m[boundary])) if boundary.any() else float("nan"), "background_mean": float(np.mean(bg))}
            row["core_background_separation"] = row["core_mean"] - row["background_mean"] if np.isfinite(row["core_mean"]) else float("nan")
            row["boundary_background_separation"] = row["boundary_mean"] - row["background_mean"] if np.isfinite(row["boundary_mean"]) else float("nan")
            rows.append(row)
    return rows


def propagation_rows(direction, dataset, indices, raw, cdf):
    bins = [(0, 0), (1, 2), (3, 4), (5, 8), (9, 9999)]
    rows = []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        mask = mask_np(item)
        m0_pre, m1_pre, m0_post, m1_post = maps_for(raw[pos], cdf)
        for variant, pre, post in [("T0", m0_pre, m0_post), ("T1", m1_pre, m1_post)]:
            seeds = pre >= np.percentile(pre, 99)
            if not seeds.any():
                continue
            dist = distance_transform_edt(~seeds)
            for lo, hi in bins:
                sel = (dist >= lo) & (dist <= hi)
                for region, rsel in [("inside", mask), ("background", ~mask)]:
                    vals = post[sel & rsel]
                    rows.append({"direction": direction, "variant": variant, "idx": idx, "category": item["cls_name"], "label": int(item["anomaly"]), "distance_bin": f"{lo}-{hi if hi < 9999 else 'gt8'}", "region": region, "post_score_mean": float(np.mean(vals)) if vals.size else float("nan"), "pixels": int(vals.size)})
    return rows


def pixel_ranking_rows(direction, dataset, indices, raw, cdf):
    rng = np.random.default_rng(0)
    pos0 = []; neg0 = []; pos1 = []; neg1 = []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        if int(item["anomaly"]) != 1:
            continue
        mask = mask_np(item)
        _, _, m0, m1 = maps_for(raw[pos], cdf)
        p0, n0 = m0[mask], m0[~mask]
        p1, n1 = m1[mask], m1[~mask]
        for src, dst in [(p0, pos0), (n0, neg0), (p1, pos1), (n1, neg1)]:
            step = max(1, src.size // 2000)
            dst.append(src[::step][:2000])
    p0 = np.concatenate(pos0); n0 = np.concatenate(neg0); p1 = np.concatenate(pos1); n1 = np.concatenate(neg1)
    k = min(PAIR_SAMPLE, p0.size, n0.size, p1.size, n1.size)
    pi = rng.integers(0, min(p0.size, p1.size), size=k)
    ni = rng.integers(0, min(n0.size, n1.size), size=k)
    ord0 = p0[pi] > n0[ni]
    ord1 = p1[pi] > n1[ni]
    return [{"direction": direction, "sampled_pairs": int(k), "T0_pixel_AUROC_sample": float(roc_auc_score(np.r_[np.ones(p0.size), np.zeros(n0.size)], np.r_[p0, n0]) * 100.0), "T1_pixel_AUROC_sample": float(roc_auc_score(np.r_[np.ones(p1.size), np.zeros(n1.size)], np.r_[p1, n1]) * 100.0), "pairwise_order_change_fraction": float(np.mean(ord0 != ord1)), "pairwise_order_agreement": float(np.mean(ord0 == ord1))}]


def summarize_rank(rows):
    out = []
    for d in sorted({r["direction"] for r in rows}):
        vals = [float(r["pre_spearman"]) for r in rows if r["direction"] == d]
        post = [float(r["post_spearman"]) for r in rows if r["direction"] == d]
        out.append({"direction": d, "pre_mean": float(np.mean(vals)), "pre_median": float(np.median(vals)), "pre_q05": float(np.percentile(vals, 5)), "pre_q95": float(np.percentile(vals, 95)), "post_mean": float(np.mean(post)), "post_median": float(np.median(post)), "post_q05": float(np.percentile(post, 5)), "post_q95": float(np.percentile(post, 95))})
    return out


def per_category_mechanism(direction, percat_xxiv, rank_rows_, coverage, comps):
    rows = []
    pc = [r for r in percat_xxiv if r["direction"] == direction]
    for r in pc:
        cat = r["category"]
        ranks = [x for x in rank_rows_ if x["direction"] == direction and x["category"] == cat]
        cov0 = [float(x["mean_best_gt_region_component_coverage"]) for x in comps if x["direction"] == direction and x["variant"] == "T0" and x["category"] == cat and abs(float(x["fpr"]) - 0.05) < 1e-9]
        cov1 = [float(x["mean_best_gt_region_component_coverage"]) for x in comps if x["direction"] == direction and x["variant"] == "T1" and x["category"] == cat and abs(float(x["fpr"]) - 0.05) < 1e-9]
        c0 = [float(x["fragmentation_index"]) for x in comps if x["direction"] == direction and x["variant"] == "T0" and x["category"] == cat and abs(float(x["fpr"]) - 0.05) < 1e-9]
        c1 = [float(x["fragmentation_index"]) for x in comps if x["direction"] == direction and x["variant"] == "T1" and x["category"] == cat and abs(float(x["fpr"]) - 0.05) < 1e-9]
        rows.append({"direction": direction, "category": cat, "dPRO": r["dPRO"], "dP-AUROC": r["dP-AUROC"], "pre_smooth_rank_correlation": float(np.mean([x["pre_spearman"] for x in ranks])) if ranks else float("nan"), "post_smooth_rank_correlation": float(np.mean([x["post_spearman"] for x in ranks])) if ranks else float("nan"), "region_coverage_change_fpr5": float(np.mean(cov1) - np.mean(cov0)) if cov0 and cov1 else float("nan"), "fragmentation_change_fpr5": float(np.mean(c1) - np.mean(c0)) if c0 and c1 else float("nan")})
    base_rows = list(rows)
    dpro = np.asarray([float(r["dPRO"]) for r in base_rows], dtype=np.float64)
    for key in ["region_coverage_change_fpr5", "fragmentation_change_fpr5"]:
        vals = np.asarray([float(r[key]) for r in base_rows], dtype=np.float64)
        ok = np.isfinite(dpro) & np.isfinite(vals)
        pear = float(pearsonr(vals[ok], dpro[ok]).statistic) if ok.sum() >= 2 else float("nan")
        spear = float(spearmanr(vals[ok], dpro[ok]).statistic) if ok.sum() >= 2 else float("nan")
        rows.append({"direction": direction, "category": f"CORRELATION_{key}_vs_dPRO", "dPRO": float("nan"), "dP-AUROC": float("nan"), "pre_smooth_rank_correlation": float("nan"), "post_smooth_rank_correlation": float("nan"), "region_coverage_change_fpr5": float("nan"), "fragmentation_change_fpr5": float("nan"), "pearson": pear, "spearman": spear})
    return rows


def frozen_parity():
    outcome = json.loads((XXIV / "OUTCOME.json").read_text(encoding="utf-8"))
    metrics = list(csv.DictReader((XXIV / "TARGET_RAW_METRICS.csv").open(encoding="utf-8")))
    expected = {
        ("D1", "T0_OFFICIAL_RAW"): (95.8084, 87.9651),
        ("D1", "T1_SOURCE_NORMAL_RANK"): (95.8173, 90.4225),
        ("D2", "T0_OFFICIAL_RAW"): (92.8762, 86.8162),
        ("D2", "T1_SOURCE_NORMAL_RANK"): (92.8752, 91.6972),
    }
    rows = []
    ok = True
    for r in metrics:
        key = (r["direction"], r["variant"])
        p, pro = float(r["P-AUROC"]), float(r["PRO"])
        ep, epro = expected[key]
        row = {"direction": key[0], "variant": key[1], "P-AUROC": p, "PRO": pro, "expected_P_approx": ep, "expected_PRO_approx": epro, "abs_P_diff_from_approx": abs(p - ep), "abs_PRO_diff_from_approx": abs(pro - epro)}
        rows.append(row)
        ok = ok and abs(p - ep) < 0.02 and abs(pro - epro) < 0.02
    return {"xxiv_outcome": outcome["outcome"], "metric_rows": rows, "pass": ok}


def decide(parity, rank_summary, target_deltas, coverage, comps, coherence, pro_rows):
    if not parity["pass"]:
        return "XXV_IMPLEMENTATION_FAILURE", "XXV_MECHANISM_UNRESOLVED"
    pre_ok = all(r["pre_q05"] >= 0.999 for r in rank_summary)
    pauroc_ok = all(abs(float(d["dP-AUROC"])) < 0.05 for d in target_deltas)
    coverage_ok = []
    for d in ["D1", "D2"]:
        cov0 = np.mean([float(r["mean_region_overlap"]) for r in coverage if r["direction"] == d and r["variant"] == "T0"])
        cov1 = np.mean([float(r["mean_region_overlap"]) for r in coverage if r["direction"] == d and r["variant"] == "T1"])
        coverage_ok.append(cov1 > cov0)
    frag_or_coh = []
    for d in ["D1", "D2"]:
        f0 = np.nanmean([float(r["fragmentation_index"]) for r in comps if r["direction"] == d and r["variant"] == "T0"])
        f1 = np.nanmean([float(r["fragmentation_index"]) for r in comps if r["direction"] == d and r["variant"] == "T1"])
        c0 = np.nanmean([float(r["neighbor_correlation"]) for r in coherence if r["direction"] == d and r["variant"] == "T0" and r["region"] == "inside"])
        c1 = np.nanmean([float(r["neighbor_correlation"]) for r in coherence if r["direction"] == d and r["variant"] == "T1" and r["region"] == "inside"])
        frag_or_coh.append((f1 < f0) or (c1 > c0))
    pro_ok = []
    for d in ["D1", "D2"]:
        low = [float(r["pro"]) for r in pro_rows if r["direction"] == d and r["variant"] == "T1" and float(r["fpr"]) <= 0.05]
        low0 = [float(r["pro"]) for r in pro_rows if r["direction"] == d and r["variant"] == "T0" and float(r["fpr"]) <= 0.05]
        pro_ok.append(np.mean(low) > np.mean(low0))
    primary = "XXV_CALIBRATION_SPATIAL_AGGREGATION_CONFIRMED" if (pre_ok and pauroc_ok and all(coverage_ok) and all(frag_or_coh) and all(pro_ok)) else "XXV_MECHANISM_UNRESOLVED"
    anomaly_cov = np.mean([1 if c else 0 for c in coverage_ok])
    normal_rows = list(csv.DictReader((OUT / "NORMAL_FP_STRUCTURE.csv").open(encoding="utf-8"))) if (OUT / "NORMAL_FP_STRUCTURE.csv").exists() else []
    normal_supp = []
    for d in ["D1", "D2"]:
        n0 = np.mean([float(r["islands"]) for r in normal_rows if r["direction"] == d and r["variant"] == "T0"])
        n1 = np.mean([float(r["islands"]) for r in normal_rows if r["direction"] == d and r["variant"] == "T1"])
        normal_supp.append(n1 < n0)
    if all(coverage_ok) and all(normal_supp):
        secondary = "XXV_DUAL_SPATIAL_MECHANISM"
    elif all(normal_supp):
        secondary = "XXV_NORMAL_FP_SUPPRESSION_MECHANISM"
    elif all(coverage_ok):
        secondary = "XXV_ANOMALY_REGION_COHERENCE_MECHANISM"
    else:
        secondary = "XXV_MECHANISM_UNRESOLVED"
    return primary, secondary


def write_report(primary, secondary, parity, rank_summary, target_deltas, coverage, ranking, coherence, normal_fp):
    lines = ["# Phase XXV Calibration-Spatial Aggregation Mechanism Audit", "", f"Outcome: **{primary}**", f"Secondary mechanism: **{secondary}**", "", "No method, calibrator, smoothing, threshold, or target candidate was changed.", "", "## Frozen Metric Parity"]
    for r in parity["metric_rows"]:
        lines.append(f"- {r['direction']} {r['variant']}: P {r['P-AUROC']:.4f}, PRO {r['PRO']:.4f}.")
    lines.extend(["", "## Rank Preservation"])
    for r in rank_summary:
        lines.append(f"- {r['direction']}: pre Spearman mean {r['pre_mean']:.6f}, q05 {r['pre_q05']:.6f}; post mean {r['post_mean']:.6f}.")
    lines.extend(["", "## Answers"])
    lines.append("Q1. P-AUROC is almost invariant because pre-Gaussian rank mapping is monotonic and post-Gaussian positive/negative pixel ranking changes only weakly.")
    for r in ranking:
        lines.append(f"- {r['direction']} pairwise order-change estimate: {r['pairwise_order_change_fraction']:.6f}.")
    lines.append("Q2. The large scalar PRO gain is not explained by higher coverage at the fixed matched-FPR points audited here; the audit therefore leaves the exact PRO mechanism unresolved.")
    lines.append("Q3. Yes, the source-normal CDF changes amplitude geometry before smoothing; see `AMPLITUDE_RESPONSE.csv`.")
    lines.append("Q4. No. At the fixed matched global FPR points, mean region overlap does not improve; it is slightly lower for T1 in both directions.")
    for d in ["D1", "D2"]:
        c0 = np.mean([float(r["mean_region_overlap"]) for r in coverage if r["direction"] == d and r["variant"] == "T0"])
        c1 = np.mean([float(r["mean_region_overlap"]) for r in coverage if r["direction"] == d and r["variant"] == "T1"])
        lines.append(f"- {d} mean matched-FPR overlap: T0 {c0:.4f}, T1 {c1:.4f}, delta {c1-c0:+.4f}.")
    lines.append("Q5. Neither clean anomaly-region coherence nor normal-FP suppression consistently explains the PRO gain under these diagnostics.")
    for d in ["D1", "D2"]:
        f0 = np.nanmean([float(r["islands"]) for r in normal_fp if r["direction"] == d and r["variant"] == "T0"])
        f1 = np.nanmean([float(r["islands"]) for r in normal_fp if r["direction"] == d and r["variant"] == "T1"])
        lines.append(f"- {d} normal-image island count mean: T0 {f0:.4f}, T1 {f1:.4f}, delta {f1-f0:+.4f}.")
    lines.append("Q6. Rank preservation and amplitude remapping are shared across both directions, but the matched-FPR spatial explanation is not supported in either direction.")
    lines.append("")
    lines.append("STOP: no new detector or target rescue was run.")
    (OUT / "XXV_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_log(primary, secondary):
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXV Calibration-Spatial Aggregation Mechanism Audit\n\n")
        fp.write(f"- outcome: `{primary}`\n")
        fp.write(f"- secondary_mechanism: `{secondary}`\n")
        fp.write("- scope: frozen XXIV-T T0/T1 target-map mechanism audit only; no training, no new candidate, no tuning.\n")
        fp.write(f"- output: `{OUT.as_posix()}`\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    parity = frozen_parity()
    (OUT / "FROZEN_METRIC_PARITY.json").write_text(json.dumps(parity, indent=2, default=jdefault), encoding="utf-8")
    target_deltas = list(csv.DictReader((XXIV / "TARGET_DELTAS.csv").open(encoding="utf-8")))
    percat_xxiv = list(csv.DictReader((XXIV / "PER_CATEGORY.csv").open(encoding="utf-8")))
    all_rank, all_amp, all_ab, all_coh, all_cov, all_size, all_comp, all_core, all_prop, all_norm, all_pro, all_ranking, all_percat = [], [], [], [], [], [], [], [], [], [], [], [], []
    for direction in ["D1", "D2"]:
        ckpt = OUT / f"direction_result_{direction}.json"
        if ckpt.exists():
            saved = json.loads(ckpt.read_text(encoding="utf-8"))
            all_rank.extend(saved["rank"]); all_amp.extend(saved["amp"]); all_ab.extend(saved["ab"]); all_coh.extend(saved["coh"])
            all_cov.extend(saved["cov"]); all_size.extend(saved["size"]); all_comp.extend(saved["comp"]); all_core.extend(saved["core"])
            all_prop.extend(saved["prop"]); all_norm.extend(saved["norm"]); all_pro.extend(saved["pro"]); all_ranking.extend(saved["ranking"]); all_percat.extend(saved["percat"])
            all_percat = [r for r in all_percat if r.get("direction") != direction]
            all_percat.extend(per_category_mechanism(direction, percat_xxiv, saved["rank"], saved["cov"], saved["comp"]))
            continue
        progress(f"loading {direction}")
        _, _, dataset, indices, raw, _, cdf = load_direction(direction)
        progress(f"rank/amplitude {direction}")
        rr = rank_rows(direction, dataset, indices, raw, cdf); all_rank.extend(rr)
        all_amp.extend(amplitude_rows(direction, raw, cdf))
        all_ab.extend(anomaly_background_rows(direction, dataset, indices, raw, cdf))
        progress(f"spatial/coherence {direction}")
        all_coh.extend(spatial_coherence_rows(direction, dataset, indices, raw, cdf))
        thresholds = background_thresholds(dataset, indices, raw, cdf)
        cov, size, comp, norm, pro = matched_fpr_and_components(direction, dataset, indices, raw, cdf, thresholds)
        all_cov.extend(cov); all_size.extend(size); all_comp.extend(comp); all_norm.extend(norm); all_pro.extend(pro)
        all_core.extend(core_boundary_rows(direction, dataset, indices, raw, cdf))
        all_prop.extend(propagation_rows(direction, dataset, indices, raw, cdf))
        all_ranking.extend(pixel_ranking_rows(direction, dataset, indices, raw, cdf))
        pcmech = per_category_mechanism(direction, percat_xxiv, rr, cov, comp)
        all_percat.extend(pcmech)
        ckpt.write_text(json.dumps({"rank": rr, "amp": all_amp[-len(amplitude_rows(direction, raw[:0], cdf)):] if False else [r for r in all_amp if r["direction"] == direction], "ab": [r for r in all_ab if r["direction"] == direction], "coh": [r for r in all_coh if r["direction"] == direction], "cov": cov, "size": size, "comp": comp, "core": [r for r in all_core if r["direction"] == direction], "prop": [r for r in all_prop if r["direction"] == direction], "norm": norm, "pro": pro, "ranking": [r for r in all_ranking if r["direction"] == direction], "percat": pcmech}, default=jdefault), encoding="utf-8")
        write_csv(OUT / "RANK_PRESERVATION.running.csv", all_rank)
    rank_summary = summarize_rank(all_rank)
    write_csv(OUT / "RANK_PRESERVATION.csv", all_rank + rank_summary)
    write_csv(OUT / "AMPLITUDE_RESPONSE.csv", all_amp)
    write_csv(OUT / "ANOMALY_BACKGROUND_AMPLITUDE.csv", all_ab)
    write_csv(OUT / "SPATIAL_COHERENCE.csv", all_coh)
    write_csv(OUT / "MATCHED_FPR_REGION_COVERAGE.csv", all_cov)
    write_csv(OUT / "REGION_SIZE_ANALYSIS.csv", all_size)
    write_csv(OUT / "CONNECTED_COMPONENTS.csv", all_comp)
    write_csv(OUT / "CORE_BOUNDARY_ANALYSIS.csv", all_core)
    write_csv(OUT / "SPATIAL_PROPAGATION.csv", all_prop)
    write_csv(OUT / "NORMAL_FP_STRUCTURE.csv", all_norm)
    write_csv(OUT / "PER_CATEGORY_MECHANISM.csv", all_percat)
    write_csv(OUT / "PRO_FPR_CURVES.csv", all_pro)
    write_csv(OUT / "PIXEL_RANKING_INVARIANCE.csv", all_ranking)
    primary, secondary = decide(parity, rank_summary, target_deltas, all_cov, all_comp, all_coh, all_pro)
    write_report(primary, secondary, parity, rank_summary, target_deltas, all_cov, all_ranking, all_coh, all_norm)
    payload = {"phase": "XXV", "outcome": primary, "secondary_mechanism": secondary, "trained": False, "new_target_candidate": False, "runtime_sec": time.time() - start, "next": "STOP"}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=jdefault), encoding="utf-8")
    append_log(primary, secondary)


if __name__ == "__main__":
    main()
