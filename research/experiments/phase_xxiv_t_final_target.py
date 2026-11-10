import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import direction_cfg
from experiments.phase_xa_smoke import load_dataset, to_device_cache, write_csv
from experiments.phase_xxiii_b_dual_carrier_local import (
    IMAGE_SIZE,
    SIGMA,
    apply_cdf,
    eval_maps,
    fit_cdf,
    official_raw_map_score,
    smooth_full,
)
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xxiv_pixel_calibration/final_target")
CACHE = OUT / "cache"
EPS = 1e-6
SOURCE_MEAN = {"D1": 1.2737561481943942, "D2": 3.1655830688424658}


def jdefault(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(type(obj).__name__)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not-a-git-repo"


def direction_spec(direction):
    source_name, source_path, checkpoint = direction_cfg(direction)
    if direction == "D1":
        return {"direction": "D1", "source": source_name, "source_path": source_path, "target": "visa", "target_path": "data/VISA", "checkpoint": checkpoint}
    return {"direction": "D2", "source": source_name, "source_path": source_path, "target": "mvtec", "target_path": "data/mvtec", "checkpoint": checkpoint}


def normal_indices(dataset):
    return [i for i in range(len(dataset)) if int(dataset[i]["anomaly"]) == 0]


def all_indices(dataset):
    return list(range(len(dataset)))


def write_progress(direction, stage, **extra):
    payload = {"direction": direction, "stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    (OUT / f"progress_{direction}.json").write_text(json.dumps(payload, indent=2, default=jdefault), encoding="utf-8")
    print(f"[XXIV-T] {direction}: {stage}", flush=True)


def build_raw_cache(direction, split_name, eval_dataset_name, eval_dataset, eval_indices, memory_dataset_name, memory_dataset, memory_indices, checkpoint, device):
    base = CACHE / f"{direction}_{split_name}"
    raw_path = base.with_name(base.name + "_raw.npy")
    score_path = base.with_name(base.name + "_scores.npy")
    if raw_path.exists() and score_path.exists():
        return raw_path, score_path
    write_progress(direction, f"building raw cache {split_name}", images=len(eval_indices), eval_dataset=eval_dataset_name)
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
    image_cache, patch_cache = build_memory_from_indices(model, memory_dataset, memory_indices, memory_dataset_name, device, adapter=None)
    image_cache, patch_cache = to_device_cache(image_cache, device), to_device_cache(patch_cache, device)
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(len(eval_indices), IMAGE_SIZE, IMAGE_SIZE))
    scores = np.zeros((len(eval_indices),), dtype=np.float32)
    for pos, idx in enumerate(eval_indices):
        m, s = official_raw_map_score(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, eval_dataset[idx], device)
        raw[pos] = m.astype(np.float16)
        scores[pos] = s
        if pos % 32 == 0:
            raw.flush()
            write_progress(direction, f"building raw cache {split_name}", done=pos + 1, total=len(eval_indices))
    raw.flush()
    np.save(score_path, scores)
    del model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_path, score_path


def finite_hash(arr):
    h = hashlib.sha256()
    h.update(np.asarray([arr.size], dtype=np.int64).tobytes())
    if arr.size:
        step = max(1, arr.size // 1000000)
        h.update(np.asarray(arr[::step], dtype=np.float32).tobytes())
    return h.hexdigest()


def calibrator_metadata(direction, spec, source_dataset, source_indices, cdf, checkpoint):
    q = np.percentile(cdf, [1, 5, 25, 50, 75, 95, 99]).astype(float)
    manifest = [{"idx": int(i), "category": source_dataset[i]["cls_name"]} for i in source_indices]
    mh = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "direction": direction,
        "algorithm": "P1_NORMAL_RANK",
        "source_dataset": spec["source"],
        "source_normal_image_count": len(source_indices),
        "calibration_pixels_retained": int(cdf.size),
        "pixel_sampling_policy": "none; all source normal pixels retained, matching XXIV-A P1/B1",
        "empirical_cdf_implementation": "numpy.searchsorted(sorted_ref, values, side='right') / len(sorted_ref)",
        "interpolation": "none; right-sided empirical step CDF",
        "tie_handling": "side='right'",
        "out_of_range_handling": "below source min maps to 0; above/equal source max maps to 1 via searchsorted behavior",
        "clipping_behavior": "no explicit clipping beyond empirical CDF support",
        "integration_point": "official MRAD raw pixel anomaly map immediately before final Gaussian smoothing",
        "gaussian_order": "rank transform then Gaussian smoothing",
        "gaussian_sigma": SIGMA,
        "source_manifest_hash": mh,
        "code_commit_hash": git_hash(),
        "official_checkpoint": checkpoint,
        "official_checkpoint_sha256": sha256_file(ROOT / checkpoint) if (ROOT / checkpoint).exists() else "missing",
        "cdf_sha256": finite_hash(cdf),
        "score_min": float(cdf[0]),
        "score_max": float(cdf[-1]),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "q50": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
    }


def build_calibrator(direction, device):
    spec = direction_spec(direction)
    cdf_path = OUT / f"{direction}_PIXEL_CDF.npz"
    meta_path = OUT / f"{direction}_CALIBRATOR_METADATA.json"
    if cdf_path.exists() and meta_path.exists():
        cdf = np.load(cdf_path)["cdf"].astype(np.float32)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return spec, cdf, meta
    source_dataset = load_dataset(spec["source"], spec["source_path"])
    src_normals = normal_indices(source_dataset)
    mem_indices = all_indices(source_dataset)
    raw_path, _ = build_raw_cache(direction, "source_normal", spec["source"], source_dataset, src_normals, spec["source"], source_dataset, mem_indices, spec["checkpoint"], device)
    raw = np.load(raw_path)
    write_progress(direction, "fitting final source-normal CDF", normal_images=len(src_normals))
    cdf = fit_cdf(raw.reshape(-1).astype(np.float32))
    np.savez_compressed(cdf_path, cdf=cdf)
    meta = calibrator_metadata(direction, spec, source_dataset, src_normals, cdf, spec["checkpoint"])
    meta_path.write_text(json.dumps(meta, indent=2, default=jdefault), encoding="utf-8")
    return spec, cdf, meta


def freeze_hashes():
    files = [
        OUT / "D1_PIXEL_CDF.npz",
        OUT / "D2_PIXEL_CDF.npz",
        OUT / "D1_CALIBRATOR_METADATA.json",
        OUT / "D2_CALIBRATOR_METADATA.json",
        OUT / "FINAL_PIXEL_TARGET_PROTOCOL.json",
    ]
    lines = []
    for p in files:
        if p.exists():
            lines.append(f"{sha256_file(p)}  {p.name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smooth_raw_maps(raw):
    out = np.empty((raw.shape[0], IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i in range(raw.shape[0]):
        out[i] = smooth_full(np.asarray(raw[i], dtype=np.float32))
    return out


def rank_maps(raw, cdf):
    out = np.empty((raw.shape[0], IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i in range(raw.shape[0]):
        out[i] = smooth_full(apply_cdf(np.asarray(raw[i], dtype=np.float32), cdf))
    return out


def evaluate_direction(direction, spec, cdf, meta, device):
    target_dataset = load_dataset(spec["target"], spec["target_path"])
    source_dataset = load_dataset(spec["source"], spec["source_path"])
    target_indices = all_indices(target_dataset)
    source_indices = all_indices(source_dataset)
    raw_path, score_path = build_raw_cache(direction, "target", spec["target"], target_dataset, target_indices, spec["source"], source_dataset, source_indices, spec["checkpoint"], device)
    raw = np.load(raw_path)
    scores = np.load(score_path)
    write_progress(direction, "evaluating T0/T1 target maps", target_images=len(target_indices))
    maps0 = smooth_raw_maps(raw)
    maps1 = rank_maps(raw, cdf)
    m0 = eval_maps(target_dataset, target_indices, scores, maps0)
    m1 = eval_maps(target_dataset, target_indices, scores, maps1)
    rows = []
    for variant, metrics in [("T0_OFFICIAL_RAW", m0), ("T1_SOURCE_NORMAL_RANK", m1)]:
        rows.append({"direction": direction, "source": spec["source"], "target": spec["target"], "variant": variant, **{k: metrics[k] for k in ["I-AUROC", "I-AP", "P-AUROC", "PRO"]}})
    delta = {
        "direction": direction,
        "source": spec["source"],
        "target": spec["target"],
        "dI-AUROC": m1["I-AUROC"] - m0["I-AUROC"],
        "dI-AP": m1["I-AP"] - m0["I-AP"],
        "dP-AUROC": m1["P-AUROC"] - m0["P-AUROC"],
        "dPRO": m1["PRO"] - m0["PRO"],
    }
    delta["G_pixel"] = 0.5 * delta["dP-AUROC"] + 0.5 * delta["dPRO"]
    delta["target_positive"] = bool(((delta["dP-AUROC"] >= 0.30) or (delta["dPRO"] >= 0.40)) and delta["dP-AUROC"] >= -0.20 and delta["dPRO"] >= -0.20 and delta["G_pixel"] > 0)
    delta["strong_pixel_calibration_gain"] = bool(delta["dP-AUROC"] >= 0.30 and delta["dPRO"] >= 0.40 and delta["G_pixel"] >= 0.50)
    image_parity = {
        "direction": direction,
        "max_abs_image_score_diff": 0.0,
        "I-AUROC_T0": m0["I-AUROC"],
        "I-AUROC_T1": m1["I-AUROC"],
        "I-AP_T0": m0["I-AP"],
        "I-AP_T1": m1["I-AP"],
        "max_abs_I_AUROC_diff": abs(m1["I-AUROC"] - m0["I-AUROC"]),
        "max_abs_I_AP_diff": abs(m1["I-AP"] - m0["I-AP"]),
        "pass": abs(m1["I-AUROC"] - m0["I-AUROC"]) <= 1e-9 and abs(m1["I-AP"] - m0["I-AP"]) <= 1e-9,
    }
    percat = per_category_rows(direction, target_dataset, target_indices, scores, maps0, maps1, m0["per_class"], m1["per_class"])
    return {
        "metrics": rows,
        "delta": delta,
        "image_parity": image_parity,
        "per_category": percat,
        "analysis": posthoc_analysis(direction, target_dataset, target_indices, raw, maps0, maps1, cdf, meta, scores),
    }


def per_category_rows(direction, dataset, indices, scores, maps0, maps1, pc0, pc1):
    rows = []
    for cat in sorted(pc0):
        p0 = float(pc0[cat].get("P-AUROC", pc0[cat].get("p-auroc", float("nan"))))
        pro0 = float(pc0[cat].get("PRO", pc0[cat].get("pro", float("nan"))))
        p1 = float(pc1[cat].get("P-AUROC", pc1[cat].get("p-auroc", float("nan"))))
        pro1 = float(pc1[cat].get("PRO", pc1[cat].get("pro", float("nan"))))
        if np.isfinite(p0) and p0 <= 1.5: p0 *= 100.0
        if np.isfinite(pro0) and pro0 <= 1.5: pro0 *= 100.0
        if np.isfinite(p1) and p1 <= 1.5: p1 *= 100.0
        if np.isfinite(pro1) and pro1 <= 1.5: pro1 *= 100.0
        g = 0.5 * (p1 - p0) + 0.5 * (pro1 - pro0)
        rows.append({"direction": direction, "category": cat, "T0_P-AUROC": p0, "T0_PRO": pro0, "T1_P-AUROC": p1, "T1_PRO": pro1, "dP-AUROC": p1 - p0, "dPRO": pro1 - pro0, "G_pixel": g, "WTL": "T" if abs(g) <= 0.05 else ("W" if g > 0 else "L")})
    pos = [max(0.0, r["G_pixel"]) for r in rows]
    concentrated = bool(pos and max(pos) / max(1e-12, sum(pos)) > 0.5)
    for r in rows:
        r["target_gain_concentration"] = "CATEGORY_CONCENTRATED_TARGET_GAIN" if concentrated else "CATEGORY_DISTRIBUTED_TARGET_GAIN"
    return rows


def sample_auc(normal_vals, anomaly_vals):
    n = normal_vals[:: max(1, normal_vals.size // 200000)] if normal_vals.size else normal_vals
    a = anomaly_vals[:: max(1, anomaly_vals.size // 200000)] if anomaly_vals.size else anomaly_vals
    y = np.concatenate([np.zeros(n.size), np.ones(a.size)])
    s = np.concatenate([n, a])
    return float(roc_auc_score(y, s) * 100.0) if len(np.unique(y)) == 2 else float("nan")


def posthoc_analysis(direction, dataset, indices, raw, maps0, maps1, cdf, meta, scores):
    diag_rows, dist_rows, fp_rows, region_rows, support_rows, coverage_rows = [], [], [], [], [], []
    normal0, anomaly0, normal1, anomaly1 = [], [], [], []
    below_min = above_max = below_q01 = above_q99 = total_pix = 0
    pct_normal, pct_anomaly = [], []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        r = np.asarray(raw[pos], dtype=np.float32)
        rank_pre = apply_cdf(r, cdf)
        raw_post, rank_post = maps0[pos], maps1[pos]
        flat_r = r.reshape(-1)
        flat_rank = rank_pre.reshape(-1)
        denom_pre = max(EPS, float(np.linalg.norm(flat_r) * np.linalg.norm(flat_rank)))
        denom_post = max(EPS, float(np.linalg.norm(raw_post.reshape(-1)) * np.linalg.norm(rank_post.reshape(-1))))
        if pos < 64:
            diag_rows.append({
                "direction": direction,
                "idx": idx,
                "category": item["cls_name"],
                "label": int(item["anomaly"]),
                "presmooth_spearman_raw_vs_rank": float(spearmanr(flat_r, flat_rank).statistic),
                "postsmooth_spearman_official_vs_rank": float(spearmanr(raw_post.reshape(-1), rank_post.reshape(-1)).statistic),
                "presmooth_map_cosine": float(np.dot(flat_r, flat_rank) / denom_pre),
                "postsmooth_map_cosine": float(np.dot(raw_post.reshape(-1), rank_post.reshape(-1)) / denom_post),
            })
        total_pix += flat_r.size
        below_min += int(np.sum(flat_r < meta["score_min"]))
        above_max += int(np.sum(flat_r > meta["score_max"]))
        below_q01 += int(np.sum(flat_r < meta["q01"]))
        above_q99 += int(np.sum(flat_r > meta["q99"]))
        label = int(item["anomaly"])
        if label == 0:
            for name, m in [("T0_OFFICIAL_RAW", raw_post), ("T1_SOURCE_NORMAL_RANK", rank_post)]:
                flat = m.reshape(-1)
                fp_rows.append({"direction": direction, "variant": name, "idx": idx, "category": item["cls_name"], "mean": float(np.mean(flat)), "q95": float(np.percentile(flat, 95)), "q99": float(np.percentile(flat, 99)), "top1_mean": float(np.mean(np.sort(flat)[-max(1, int(flat.size * 0.01)):])), "max": float(np.max(flat))})
            normal0.append(raw_post.reshape(-1)); normal1.append(rank_post.reshape(-1)); pct_normal.append(rank_pre.reshape(-1))
        else:
            mask = np.squeeze(item["img_mask"].numpy()) > 0.5
            if mask.any():
                anomaly0.append(raw_post[mask]); anomaly1.append(rank_post[mask]); pct_anomaly.append(rank_pre[mask])
                for name, m in [("T0_OFFICIAL_RAW", raw_post), ("T1_SOURCE_NORMAL_RANK", rank_post)]:
                    er = binary_erosion(mask, iterations=2)
                    dil = binary_dilation(mask, iterations=2)
                    boundary = np.logical_and(dil, ~er)
                    outside = ~dil
                    ins = m[mask]
                    out = m[outside]
                    region_rows.append({"direction": direction, "variant": name, "idx": idx, "category": item["cls_name"], "inside_mean": float(np.mean(ins)), "boundary_mean": float(np.mean(m[boundary])) if boundary.any() else float("nan"), "outside_mean": float(np.mean(out)) if out.size else float("nan"), "inside_outside_ratio": float(np.mean(ins) / (np.mean(out) + EPS)) if out.size else float("nan")})
    support_rows.append({"direction": direction, "below_source_min_fraction": below_min / total_pix, "above_source_max_fraction": above_max / total_pix, "below_source_q01_fraction": below_q01 / total_pix, "above_source_q99_fraction": above_q99 / total_pix, "target_pixels": total_pix})
    for name, nlist, alist in [("T0_OFFICIAL_RAW", normal0, anomaly0), ("T1_SOURCE_NORMAL_RANK", normal1, anomaly1)]:
        n = np.concatenate(nlist) if nlist else np.asarray([], dtype=np.float32)
        a = np.concatenate(alist) if alist else np.asarray([], dtype=np.float32)
        pooled = math.sqrt(0.5 * (float(np.var(n)) + float(np.var(a)))) if n.size and a.size else float("nan")
        dist_rows.append({"direction": direction, "variant": name, "normal_mean": float(np.mean(n)), "normal_std": float(np.std(n)), "normal_median": float(np.median(n)), "normal_q95": float(np.percentile(n, 95)), "normal_q99": float(np.percentile(n, 99)), "anomaly_mean": float(np.mean(a)), "anomaly_std": float(np.std(a)), "anomaly_median": float(np.median(a)), "anomaly_q95": float(np.percentile(a, 95)), "anomaly_q99": float(np.percentile(a, 99)), "cohen_d": float((np.mean(a) - np.mean(n)) / (pooled + EPS)), "ks_distance": float(ks_2samp(n[::max(1, n.size // 200000)], a[::max(1, a.size // 200000)]).statistic), "pixel_AUROC": sample_auc(n, a)})
    for label, vals in [("target_normal_percentiles", pct_normal), ("target_anomaly_percentiles", pct_anomaly)]:
        v = np.concatenate(vals) if vals else np.asarray([], dtype=np.float32)
        coverage_rows.append({"direction": direction, "group": label, "mean": float(np.mean(v)) if v.size else float("nan"), "std": float(np.std(v)) if v.size else float("nan"), "q01": float(np.percentile(v, 1)) if v.size else float("nan"), "q05": float(np.percentile(v, 5)) if v.size else float("nan"), "q50": float(np.percentile(v, 50)) if v.size else float("nan"), "q95": float(np.percentile(v, 95)) if v.size else float("nan"), "q99": float(np.percentile(v, 99)) if v.size else float("nan")})
    return {"diagnostics": diag_rows, "distributions": dist_rows, "normal_fp": fp_rows, "regions": region_rows, "support": support_rows, "coverage": coverage_rows, "maps0": maps0, "maps1": maps1, "scores": scores}


def anomaly_size_rows(direction, dataset, indices, scores, maps0, maps1):
    ratios = []
    for idx in indices:
        if int(dataset[idx]["anomaly"]) == 1:
            ratios.append(float((np.squeeze(dataset[idx]["img_mask"].numpy()) > 0.5).mean()))
    if len(ratios) < 3:
        return []
    q1, q2 = np.quantile(ratios, [1 / 3, 2 / 3])
    normal_pos = [p for p, idx in enumerate(indices) if int(dataset[idx]["anomaly"]) == 0]
    buckets = {"small": [], "medium": [], "large": []}
    for pos, idx in enumerate(indices):
        if int(dataset[idx]["anomaly"]) != 1:
            continue
        r = float((np.squeeze(dataset[idx]["img_mask"].numpy()) > 0.5).mean())
        buckets["small" if r <= q1 else ("medium" if r <= q2 else "large")].append(pos)
    rows = []
    for name, poss in buckets.items():
        use = normal_pos + poss
        sub = [indices[p] for p in use]
        m0 = eval_maps(dataset, sub, scores[use], maps0[use])
        m1 = eval_maps(dataset, sub, scores[use], maps1[use])
        rows.append({"direction": direction, "size_group": name, "T0_P-AUROC": m0["P-AUROC"], "T0_PRO": m0["PRO"], "T1_P-AUROC": m1["P-AUROC"], "T1_PRO": m1["PRO"], "dP-AUROC": m1["P-AUROC"] - m0["P-AUROC"], "dPRO": m1["PRO"] - m0["PRO"], "G_pixel": 0.5 * (m1["P-AUROC"] - m0["P-AUROC"]) + 0.5 * (m1["PRO"] - m0["PRO"])})
    return rows


def decide(deltas, parity):
    if not all(p["pass"] and p["max_abs_image_score_diff"] <= 1e-7 for p in parity):
        return "XXIV_T_IMPLEMENTATION_FAILURE"
    if any((d["dP-AUROC"] < -0.20 or d["dPRO"] < -0.20) and (d["dP-AUROC"] > 0 or d["dPRO"] > 0) for d in deltas):
        return "XXIV_T_TARGET_METRIC_TRADEOFF"
    positives = [d for d in deltas if d["target_positive"]]
    if len(positives) == 2 and all(d["G_pixel"] >= 0.50 for d in deltas):
        return "XXIV_T_STRONG_DUAL_DIRECTION_TARGET_GAIN"
    if len(positives) == 2:
        return "XXIV_T_DUAL_DIRECTION_TARGET_GAIN"
    if len(positives) == 1:
        return "XXIV_T_DIRECTION_SPECIFIC_TARGET_GAIN"
    if all(d["G_pixel"] > 0 for d in deltas):
        return "XXIV_T_SMALL_POSITIVE_GAIN"
    return "XXIV_T_SOURCE_GAIN_NOT_TRANSFERABLE"


def write_report(outcome, metrics, deltas, parity, retention):
    lines = ["# Phase XXIV-T Final Target Evaluation", "", f"Outcome: **{outcome}**", "", "Frozen method: `P1_NORMAL_RANK`; no training, no target statistics, no DINO, no extra target candidates.", "", "## Target Metrics", "", "| Direction | Variant | I-AUROC | I-AP | P-AUROC | PRO |", "|---|---|---:|---:|---:|---:|"]
    for r in metrics:
        lines.append(f"| {r['direction']} | {r['variant']} | {r['I-AUROC']:.4f} | {r['I-AP']:.4f} | {r['P-AUROC']:.4f} | {r['PRO']:.4f} |")
    lines.extend(["", "## Deltas", "", "| Direction | dP-AUROC | dPRO | G_pixel | Target positive | Strong | Retention |", "|---|---:|---:|---:|---|---|---:|"])
    for d in deltas:
        ret = next(r for r in retention if r["direction"] == d["direction"])
        lines.append(f"| {d['direction']} | {d['dP-AUROC']:+.4f} | {d['dPRO']:+.4f} | {d['G_pixel']:+.4f} | {d['target_positive']} | {d['strong_pixel_calibration_gain']} | {ret['retention']:+.4f} |")
    lines.extend(["", "## Image Parity"])
    for p in parity:
        lines.append(f"- {p['direction']}: pass `{p['pass']}`, max image-score diff `{p['max_abs_image_score_diff']}`.")
    lines.append("")
    lines.append("STOP: no additional target candidate was evaluated.")
    (OUT / "XXIV_T_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_log(outcome, metas, metrics, deltas, parity):
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXIV-T Final Source-Normal Pixel Rank Calibration Target Evaluation\n\n")
        fp.write("- final_source_calibration_algorithm: `P1_NORMAL_RANK`\n")
        fp.write(f"- target_access_timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
        fp.write("- exact_command: `python experiments\\\\phase_xxiv_t_final_target.py`\n")
        for m in metas:
            fp.write(f"- {m['direction']} source_normal_images: `{m['source_normal_image_count']}`, calibration_pixels: `{m['calibration_pixels_retained']}`, cdf_sha256: `{m['cdf_sha256']}`\n")
        for r in metrics:
            fp.write(f"- {r['direction']} {r['variant']}: I `{r['I-AUROC']:.4f}`, AP `{r['I-AP']:.4f}`, P `{r['P-AUROC']:.4f}`, PRO `{r['PRO']:.4f}`\n")
        for d in deltas:
            fp.write(f"- {d['direction']} delta: dP `{d['dP-AUROC']:+.4f}`, dPRO `{d['dPRO']:+.4f}`, G `{d['G_pixel']:+.4f}`\n")
        for p in parity:
            fp.write(f"- {p['direction']} image_parity: `{p['pass']}`\n")
        fp.write(f"- final_outcome: `{outcome}`\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    start = time.time()
    calibrators = {}
    metas = []
    for direction in ["D1", "D2"]:
        spec, cdf, meta = build_calibrator(direction, device)
        calibrators[direction] = (spec, cdf, meta)
        metas.append(meta)
    protocol = {
        "phase": "XXIV-T",
        "frozen_before_target": True,
        "algorithm": "P1_NORMAL_RANK",
        "directions": {d: {"source": calibrators[d][0]["source"], "target": calibrators[d][0]["target"], "cdf_file": f"{d}_PIXEL_CDF.npz", "metadata_file": f"{d}_CALIBRATOR_METADATA.json"} for d in ["D1", "D2"]},
        "integration_point": "official MRAD raw pixel anomaly map immediately before final Gaussian smoothing",
        "operation_order": "source-normal empirical CDF rank transform then Gaussian sigma=4",
        "target_methods": ["T0_OFFICIAL_RAW", "T1_SOURCE_NORMAL_RANK"],
        "forbidden_methods_not_evaluated": ["zscore", "minmax", "smooth_then_rank", "no_smoothing", "DINO", "target_CDF", "category_specific_calibration"],
    }
    (OUT / "FINAL_PIXEL_TARGET_PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    freeze_hashes()
    all_metrics, all_deltas, all_parity, all_percat, all_diag, all_dist, all_fp, all_region, all_support, all_coverage, all_size, retention = [], [], [], [], [], [], [], [], [], [], [], []
    for direction in ["D1", "D2"]:
        spec, cdf, meta = calibrators[direction]
        res = evaluate_direction(direction, spec, cdf, meta, device)
        all_metrics.extend(res["metrics"]); all_deltas.append(res["delta"]); all_parity.append(res["image_parity"]); all_percat.extend(res["per_category"])
        a = res["analysis"]
        all_diag.extend(a["diagnostics"]); all_dist.extend(a["distributions"]); all_fp.extend(a["normal_fp"]); all_region.extend(a["regions"]); all_support.extend(a["support"]); all_coverage.extend(a["coverage"])
        target_dataset = load_dataset(spec["target"], spec["target_path"])
        idx = all_indices(target_dataset)
        all_size.extend(anomaly_size_rows(direction, target_dataset, idx, a["scores"], a["maps0"], a["maps1"]))
        retention.append({"direction": direction, "source_mean_G_pixel": SOURCE_MEAN[direction], "target_G_pixel": res["delta"]["G_pixel"], "retention": res["delta"]["G_pixel"] / SOURCE_MEAN[direction]})
        write_csv(OUT / "TARGET_DELTAS.running.csv", all_deltas)
    outcome = decide(all_deltas, all_parity)
    write_csv(OUT / "TARGET_RAW_METRICS.csv", all_metrics)
    write_csv(OUT / "TARGET_DELTAS.csv", all_deltas)
    (OUT / "IMAGE_PARITY.json").write_text(json.dumps(all_parity, indent=2, default=jdefault), encoding="utf-8")
    write_csv(OUT / "PRE_SMOOTH_RANK_PARITY.csv", all_diag)
    parity_summary = []
    for direction in ["D1", "D2"]:
        vals = [float(r["presmooth_spearman_raw_vs_rank"]) for r in all_diag if r["direction"] == direction and np.isfinite(float(r["presmooth_spearman_raw_vs_rank"]))]
        parity_summary.append({
            "direction": direction,
            "samples": len(vals),
            "mean_presmooth_spearman_raw_vs_rank": float(np.mean(vals)) if vals else float("nan"),
            "min_presmooth_spearman_raw_vs_rank": float(np.min(vals)) if vals else float("nan"),
            "approximately_monotonic": bool(vals and np.min(vals) >= 0.999),
            "detail_csv": "PRE_SMOOTH_RANK_PARITY.csv",
        })
    (OUT / "PRE_SMOOTH_RANK_PARITY.json").write_text(json.dumps(parity_summary, indent=2, default=jdefault), encoding="utf-8")
    write_csv(OUT / "MAP_TRANSFORMATION_DIAGNOSTICS.csv", all_diag)
    write_csv(OUT / "PIXEL_DISTRIBUTIONS.csv", all_dist)
    write_csv(OUT / "NORMAL_FP_ANALYSIS.csv", all_fp)
    write_csv(OUT / "ANOMALOUS_REGION_ANALYSIS.csv", all_region)
    write_csv(OUT / "ANOMALY_SIZE.csv", all_size)
    write_csv(OUT / "PER_CATEGORY.csv", all_percat)
    write_csv(OUT / "SOURCE_TARGET_RETENTION.csv", retention)
    write_csv(OUT / "CALIBRATION_SUPPORT_MISMATCH.csv", all_support)
    write_csv(OUT / "TARGET_CALIBRATION_COVERAGE.csv", all_coverage)
    write_report(outcome, all_metrics, all_deltas, all_parity, retention)
    payload = {"phase": "XXIV-T", "outcome": outcome, "target_accessed": True, "trained": False, "dino_used": False, "runtime_sec": time.time() - start, "deltas": all_deltas, "image_parity": all_parity, "next": "STOP"}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=jdefault), encoding="utf-8")
    append_log(outcome, metas, all_metrics, all_deltas, all_parity)


if __name__ == "__main__":
    main()
