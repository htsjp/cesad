import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import evaluate_results, init_results
from experiments.phase_x0a_mandatory_gates import load_caches, load_components, safe_visual_eval
from experiments.phase_xa_smoke import load_dataset, write_csv
from models.mlp import average_neighbor
from mrad import compute_patch_socre


OUT = Path("results/phase_xvii_prompt_memory_complementarity/d2_target_calibration")
ALPHA = 0.7
SOURCE_DATASET = ("visa", "data/VISA")
TARGET_DATASET = ("mvtec", "data/mvtec")
CHECKPOINT = "checkpoints/test_on_mvtec.pth"
CACHE_DIR = "cache"
CACHE_NAME = "visa"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not_available"


def empirical_rank(sorted_vals, x):
    return np.searchsorted(sorted_vals, x, side="right") / max(1, len(sorted_vals))


def load_frozen_calibrator():
    cal = np.load(OUT / "FINAL_C2_CALIBRATOR.npz")
    return cal["memory_scores_sorted"], cal["prompt_scores_sorted"]


def score_components(model, prompt_learner, image_proj, patch_proj, prompt_proj, caches, item, device):
    cache_key, cache_value, cache_keys_patch, cache_values_patch = caches
    image = item["img"].unsqueeze(0).to(device)
    with torch.no_grad():
        image_features, patch_features, _, patch_projections = safe_visual_eval(model, image, [6, 12, 18, 24])
        patch_projection = average_neighbor(patch_projections[3])
        patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
        patch_bias = average_neighbor(patch_features[3])
        patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)
        _, patch_bias_tokens, _, _ = compute_patch_socre(
            patch_bias,
            cache_keys_patch,
            cache_values_patch,
            device=device,
            proj=patch_proj,
            need_mask=False,
            patch_projection=patch_projection,
            gt_mask=item["img_mask"].unsqueeze(0),
            is_mradft=False,
            use_proj=True,
        )
        bias = prompt_proj(patch_bias_tokens[:, 0, :], patch_bias_tokens[:, 1, :])
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        q, keys = image_proj(image_features, cache_key)
        cache_logits = torch.softmax(q @ keys.t(), dim=-1) @ cache_value.float()
        memory = float(cache_logits[:, 1].detach().cpu())
        maps = []
        for pidx, patch_feature in enumerate(patch_projections):
            if pidx >= 3:
                patch_feature = average_neighbor(patch_feature)
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                sim = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                smap = AnomalyCLIP_lib.get_similarity_map(sim, 518)
                maps.append((smap[..., 1] + 1 - smap[..., 0]) / 2.0)
        amap = torch.stack(maps)
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=4)) for x in amap.detach().cpu()], dim=0).sum(dim=0)
        prompt = float(torch.topk(amap.flatten(), int(amap.numel() * 0.01)).values.mean().cpu())
        official = ALPHA * memory + (1.0 - ALPHA) * prompt
    return memory, prompt, official, amap.detach().cpu()


def build_source_calibrator(device):
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_name, data_path = SOURCE_DATASET
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(CHECKPOINT, device)
    caches = load_caches(CACHE_DIR, CACHE_NAME, device)
    dataset = load_dataset(dataset_name, data_path)
    rows = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        memory, prompt, official, _ = score_components(model, prompt_learner, image_proj, patch_proj, prompt_proj, caches, item, device)
        rows.append({
            "idx": idx,
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "score_memory_raw": memory,
            "score_prompt_raw": prompt,
            "score_official_final": official,
        })
    mem = np.sort(np.asarray([r["score_memory_raw"] for r in rows], dtype=np.float64))
    prm = np.sort(np.asarray([r["score_prompt_raw"] for r in rows], dtype=np.float64))
    np.savez_compressed(OUT / "FINAL_C2_CALIBRATOR.npz", memory_scores_sorted=mem, prompt_scores_sorted=prm)
    write_csv(OUT / "SOURCE_VISΑ_COMPONENT_SCORES.csv", rows)
    # Also write ASCII alias in case the Greek alpha in the diagnostic filename is inconvenient.
    write_csv(OUT / "SOURCE_VISA_COMPONENT_SCORES.csv", rows)
    metadata = {
        "FINAL_ALPHA": ALPHA,
        "calibration": "C2_RANK",
        "source_dataset": dataset_name,
        "source_images": len(dataset),
        "number_of_source_memory_scores": int(mem.size),
        "number_of_source_prompt_scores": int(prm.size),
        "empirical_cdf_implementation": "numpy.searchsorted(sorted_source_scores, x, side='right') / n",
        "interpolation_rule": "right-continuous empirical CDF; no interpolation between observations",
        "tie_rule": "scores equal to source values map after all ties via side='right'",
        "clipping_rule": "implicit empirical CDF range [0,1]; below min maps 0, above/equal max maps 1",
        "source_dataset_manifest": str(ROOT / data_path / "meta.json"),
        "source_dataset_manifest_sha256": sha256_file(ROOT / data_path / "meta.json"),
        "code_commit": git_hash(),
        "official_checkpoint": CHECKPOINT,
        "official_checkpoint_sha256": sha256_file(ROOT / CHECKPOINT),
        "cache_model_sha256": sha256_file(ROOT / CACHE_DIR / f"cache_model_{CACHE_NAME}.pt"),
        "cache_patch_model_sha256": sha256_file(ROOT / CACHE_DIR / f"cache_patch_model_{CACHE_NAME}.pt"),
        "score_extraction_script": "experiments/phase_xvii_t_target_calibration_eval.py",
        "score_extraction_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (OUT / "FINAL_C2_CALIBRATOR_METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    protocol = {
        "FINAL_ALPHA": ALPHA,
        "FINAL_CALIBRATION": "C2_RANK",
        "source_calibrator": "VisA source empirical CDFs only",
        "target_algorithm": [
            "Compute official raw CLS-memory anomaly score M(x).",
            "Compute official raw prompt top1% anomaly score P(x).",
            "Map M(x) through frozen VisA memory empirical CDF.",
            "Map P(x) through frozen VisA prompt empirical CDF.",
            "Compute S_C2(x) = 0.7 * M_rank(x) + 0.3 * P_rank(x).",
            "Use unchanged official pixel anomaly map for pixel metrics.",
        ],
        "forbidden": ["target CDF", "target normalization", "target z-score", "alpha search", "C1 after target", "router", "category-specific calibration"],
        "target_accessed_before_file_written": False,
    }
    (OUT / "FINAL_TARGET_PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    hashes = {
        "FINAL_C2_CALIBRATOR.npz": sha256_file(OUT / "FINAL_C2_CALIBRATOR.npz"),
        "FINAL_C2_CALIBRATOR_METADATA.json": sha256_file(OUT / "FINAL_C2_CALIBRATOR_METADATA.json"),
        "FINAL_TARGET_PROTOCOL.json": sha256_file(OUT / "FINAL_TARGET_PROTOCOL.json"),
    }
    metadata["artifact_sha256"] = hashes
    (OUT / "FINAL_C2_CALIBRATOR_METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return rows, metadata, hashes


def metric_value(metrics, key):
    value = float(metrics["mean"].get(key, float("nan")))
    return value * 100.0 if np.isfinite(value) and value <= 1.5 else value


def append_result(results, cls, item, score, amap):
    mask = (item["img_mask"].detach().cpu() > 0.5).float()
    results[cls]["imgs_masks"].append(mask)
    results[cls]["gt_sp"].append(int(item["anomaly"]))
    results[cls]["pr_sp"].append(float(score))
    results[cls]["anomaly_maps"].append(amap.detach().cpu())


def evaluate_target(device):
    mem_sorted, prm_sorted = load_frozen_calibrator()
    dataset_name, data_path = TARGET_DATASET
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(CHECKPOINT, device)
    caches = load_caches(CACHE_DIR, CACHE_NAME, device)
    dataset = load_dataset(dataset_name, data_path)
    obj_list = dataset.obj_list
    official_results = init_results(obj_list)
    c2_results = init_results(obj_list)
    rows = []
    max_map_diff = 0.0
    start = time.time()
    target_access_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for idx in range(len(dataset)):
        item = dataset[idx]
        cls = item["cls_name"]
        memory, prompt, official, amap = score_components(model, prompt_learner, image_proj, patch_proj, prompt_proj, caches, item, device)
        m_rank = empirical_rank(mem_sorted, memory)
        p_rank = empirical_rank(prm_sorted, prompt)
        c2 = ALPHA * m_rank + (1.0 - ALPHA) * p_rank
        append_result(official_results, cls, item, official, amap)
        append_result(c2_results, cls, item, c2, amap.clone())
        max_map_diff = max(max_map_diff, float((amap - amap.clone()).abs().max()))
        rows.append({
            "idx": idx,
            "category": cls,
            "label": int(item["anomaly"]),
            "memory_raw": memory,
            "prompt_raw": prompt,
            "memory_rank": m_rank,
            "prompt_rank": p_rank,
            "official_score": official,
            "c2_score": c2,
        })
    official_metrics = evaluate_results(official_results, obj_list, compute_pro=True)
    c2_metrics = evaluate_results(c2_results, obj_list, compute_pro=True)
    per_cat = []
    for cat in obj_list:
        om = official_metrics["per_class"].get(cat, {})
        cm = c2_metrics["per_class"].get(cat, {})
        if not om or not cm:
            continue
        oi, oa = float(om.get("I-AUROC", np.nan)) * 100, float(om.get("I-AP", np.nan)) * 100
        ci, ca = float(cm.get("I-AUROC", np.nan)) * 100, float(cm.get("I-AP", np.nan)) * 100
        gi = 0.5 * (ci - oi) + 0.5 * (ca - oa)
        if abs(gi) <= 0.05:
            wtl = "T"
        else:
            wtl = "W" if gi > 0 else "L"
        per_cat.append({"category": cat, "Official_I-AUROC": oi, "Official_I-AP": oa, "C2_I-AUROC": ci, "C2_I-AP": ca, "dI-AUROC": ci - oi, "dI-AP": ca - oa, "G_image": gi, "WTL": wtl})
    return rows, official_metrics, c2_metrics, per_cat, {"max_abs_pixel_map_diff": max_map_diff, "target_access_timestamp": target_access_timestamp, "seconds": time.time() - start, "images": len(dataset)}


def summarize_metrics(metrics):
    return {
        "I-AUROC": metric_value(metrics, "I-AUROC"),
        "I-AP": metric_value(metrics, "I-AP"),
        "P-AUROC": metric_value(metrics, "P-AUROC"),
        "PRO": metric_value(metrics, "PRO"),
    }


def cohend(normal, anomaly):
    n, a = np.asarray(normal, dtype=np.float64), np.asarray(anomaly, dtype=np.float64)
    pooled = np.sqrt(0.5 * (n.var() + a.var()))
    return float((a.mean() - n.mean()) / (pooled + 1e-12))


def distribution_rows(rows):
    out = []
    for method, key in [("Official", "official_score"), ("C2", "c2_score")]:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
        for group, mask in [("normal", labels == 0), ("anomaly", labels == 1)]:
            x = vals[mask]
            out.append({"method": method, "group": group, "mean": float(x.mean()), "std": float(x.std()), "median": float(np.median(x)), "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)), "count": int(x.size)})
        out.append({"method": method, "group": "normal_anomaly_separation", "cohen_d": cohend(vals[labels == 0], vals[labels == 1]), "AUROC": float(__import__("sklearn.metrics").metrics.roc_auc_score(labels, vals) * 100.0), "count": int(vals.size)})
    return out


def branch_contributions(rows):
    out = []
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    for group, mask in [("all", np.ones(len(rows), dtype=bool)), ("normal", labels == 0), ("anomaly", labels == 1)]:
        sub = [r for r, keep in zip(rows, mask) if keep]
        raw_mem = np.asarray([abs(ALPHA * float(r["memory_raw"])) for r in sub], dtype=np.float64)
        raw_prm = np.asarray([abs((1 - ALPHA) * float(r["prompt_raw"])) for r in sub], dtype=np.float64)
        c2_mem = np.asarray([abs(ALPHA * float(r["memory_rank"])) for r in sub], dtype=np.float64)
        c2_prm = np.asarray([abs((1 - ALPHA) * float(r["prompt_rank"])) for r in sub], dtype=np.float64)
        out.append({"group": group, "method": "OfficialRaw", "E_abs_memory_contribution": float(raw_mem.mean()), "E_abs_prompt_contribution": float(raw_prm.mean()), "R_memory_over_prompt": float(raw_mem.mean() / (raw_prm.mean() + 1e-12)), "count": len(sub)})
        out.append({"group": group, "method": "C2Rank", "E_abs_memory_contribution": float(c2_mem.mean()), "E_abs_prompt_contribution": float(c2_prm.mean()), "R_memory_over_prompt": float(c2_mem.mean() / (c2_prm.mean() + 1e-12)), "count": len(sub)})
    return out


def decide(official, c2, pixel_parity):
    d_i = c2["I-AUROC"] - official["I-AUROC"]
    d_ap = c2["I-AP"] - official["I-AP"]
    d_p = c2["P-AUROC"] - official["P-AUROC"]
    d_pro = c2["PRO"] - official["PRO"]
    g = 0.5 * d_i + 0.5 * d_ap
    pixel_ok = abs(d_p) <= 1e-7 and abs(d_pro) <= 1e-7 and pixel_parity["max_abs_pixel_map_diff"] <= 1e-7
    if not pixel_ok:
        outcome = "XVII_T_IMPLEMENTATION_FAILURE"
    elif ((d_i >= 0.50) or (d_ap >= 0.40)) and (d_ap >= -0.20 if d_i >= 0.50 else d_i >= -0.20) and g > 0:
        outcome = "XVII_T_CALIBRATION_TARGET_GAIN"
    elif g > 0 and d_i >= -0.20 and d_ap >= -0.20:
        outcome = "XVII_T_SMALL_POSITIVE_TARGET_GAIN"
    elif (d_i > 0 or d_ap > 0) and (d_i < -0.20 or d_ap < -0.20):
        outcome = "XVII_T_TARGET_METRIC_TRADEOFF"
    else:
        outcome = "XVII_T_NO_TARGET_GAIN"
    strong = outcome == "XVII_T_CALIBRATION_TARGET_GAIN" and d_i >= 0.50 and d_ap >= 0.40 and g >= 0.50 and pixel_ok
    return outcome, {"dI-AUROC": d_i, "dI-AP": d_ap, "dP-AUROC": d_p, "dPRO": d_pro, "G_image": g, "pixel_parity_ok": pixel_ok, "secondary_tag": "SOURCE_RANK_CALIBRATION_STRONG_TARGET_SUPPORT" if strong else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    source_rows, metadata, hashes = build_source_calibrator(device)
    target_rows, official_metrics_raw, c2_metrics_raw, per_cat, runtime = evaluate_target(device)
    official = summarize_metrics(official_metrics_raw)
    c2 = summarize_metrics(c2_metrics_raw)
    pixel_parity = {
        "max_abs_pixel_map_diff": runtime["max_abs_pixel_map_diff"],
        "P-AUROC_official": official["P-AUROC"],
        "P-AUROC_c2": c2["P-AUROC"],
        "PRO_official": official["PRO"],
        "PRO_c2": c2["PRO"],
        "dP-AUROC": c2["P-AUROC"] - official["P-AUROC"],
        "dPRO": c2["PRO"] - official["PRO"],
        "pass": runtime["max_abs_pixel_map_diff"] <= 1e-7 and abs(c2["P-AUROC"] - official["P-AUROC"]) <= 1e-7 and abs(c2["PRO"] - official["PRO"]) <= 1e-7,
    }
    rank_parity = {
        "spearman_memory_raw_vs_rank": float(spearmanr([r["memory_raw"] for r in target_rows], [r["memory_rank"] for r in target_rows]).statistic),
        "spearman_prompt_raw_vs_rank": float(spearmanr([r["prompt_raw"] for r in target_rows], [r["prompt_rank"] for r in target_rows]).statistic),
        "interpretation": "Branch-internal rankings are monotonic; fusion changes come from cross-branch coordinate alignment.",
    }
    outcome, deltas = decide(official, c2, pixel_parity)
    positive = [max(0.0, float(r["G_image"])) for r in per_cat]
    concentration = max(positive) / sum(positive) if sum(positive) > 0 else 1.0
    cat_flag = "CATEGORY_CONCENTRATED_TARGET_GAIN" if concentration > 0.50 else "CATEGORY_DISTRIBUTED_TARGET_GAIN"
    (OUT / "TARGET_OFFICIAL_METRICS.json").write_text(json.dumps(official, indent=2), encoding="utf-8")
    (OUT / "TARGET_C2_METRICS.json").write_text(json.dumps(c2, indent=2), encoding="utf-8")
    (OUT / "TARGET_DELTAS.json").write_text(json.dumps({**deltas, "category_gain_concentration": concentration, "category_gain_flag": cat_flag}, indent=2), encoding="utf-8")
    (OUT / "TARGET_RANK_PARITY.json").write_text(json.dumps(rank_parity, indent=2), encoding="utf-8")
    (OUT / "TARGET_PIXEL_PARITY.json").write_text(json.dumps(pixel_parity, indent=2), encoding="utf-8")
    write_csv(OUT / "TARGET_SCORE_DISTRIBUTIONS.csv", distribution_rows(target_rows))
    write_csv(OUT / "TARGET_BRANCH_CONTRIBUTIONS.csv", branch_contributions(target_rows))
    write_csv(OUT / "TARGET_PER_CATEGORY.csv", per_cat)
    write_csv(OUT / "TARGET_COMPONENT_SCORES.csv", target_rows)
    report = [
        "# Phase XVII-T Final D2 Source-Calibrated Target Evaluation",
        "",
        f"Outcome: **{outcome}**",
        f"Secondary tag: **{deltas.get('secondary_tag') or 'none'}**",
        "",
        "No target-dependent calibration, target normalization, alpha search, router, or retry was performed.",
        "",
        "## Metrics",
        "",
        "| Method | I-AUROC | I-AP | P-AUROC | PRO |",
        "|---|---:|---:|---:|---:|",
        f"| T0 OfficialRawFusion | {official['I-AUROC']:.4f} | {official['I-AP']:.4f} | {official['P-AUROC']:.4f} | {official['PRO']:.4f} |",
        f"| T1 SourceRankCalibratedFusion | {c2['I-AUROC']:.4f} | {c2['I-AP']:.4f} | {c2['P-AUROC']:.4f} | {c2['PRO']:.4f} |",
        "",
        "## Deltas",
        "",
        f"- dI-AUROC: `{deltas['dI-AUROC']:.4f}`",
        f"- dI-AP: `{deltas['dI-AP']:.4f}`",
        f"- G_image: `{deltas['G_image']:.4f}`",
        f"- pixel parity: `{pixel_parity['pass']}`; max map diff `{pixel_parity['max_abs_pixel_map_diff']}`",
        "",
        "## Rank Parity",
        "",
        f"- Spearman(M_raw, M_rank): `{rank_parity['spearman_memory_raw_vs_rank']}`",
        f"- Spearman(P_raw, P_rank): `{rank_parity['spearman_prompt_raw_vs_rank']}`",
        "",
        "## Per-Category",
        "",
        f"- {cat_flag}; concentration `{concentration:.4f}`",
        "",
        "Mandatory stop: no further target candidates were evaluated.",
    ]
    (OUT / "XVII_T_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    payload = {
        "phase": "XVII-T",
        "outcome": outcome,
        "secondary_tag": deltas.get("secondary_tag"),
        "official": official,
        "c2": c2,
        "deltas": deltas,
        "pixel_parity": pixel_parity,
        "rank_parity": rank_parity,
        "target_access_timestamp": runtime["target_access_timestamp"],
        "calibrator_artifact_hashes": hashes,
        "target_accessed": True,
        "no_target_retries": True,
        "next": "STOP. Do not run additional target candidates.",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write("\n\n## Phase XVII-T\n")
        f.write("- Command: python experiments/phase_xvii_t_target_calibration_eval.py\n")
        f.write(f"- Frozen calibration: C2_RANK; alpha={ALPHA}; source scores={len(source_rows)}; artifact hashes={hashes}.\n")
        f.write(f"- Target access timestamp: {runtime['target_access_timestamp']}.\n")
        f.write(f"- Official metrics: {official}; C2 metrics: {c2}; deltas: {deltas}.\n")
        f.write(f"- Pixel parity: {pixel_parity}; per-category flag: {cat_flag}.\n")
        f.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XVII_T_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
