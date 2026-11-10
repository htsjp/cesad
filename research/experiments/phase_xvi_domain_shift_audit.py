import csv
import io
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from experiments.mrad_audit_utils import evaluate_results, init_results
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import (
    all_indices_for_categories,
    direction_cfg,
    load_adapter,
    metric_value,
    selected_rows,
)
from experiments.phase_xa_smoke import load_dataset, to_device_cache, write_csv
from models.mlp import average_neighbor
from mrad import compute_patch_socre
from phase_xa.adapted_cache_builder import build_adapted_task_cache
from phase_xa.official_task_forward import image_memory_scores, normalized_image_feature, visual_outputs
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xvi_domain_generalization/audit")
XA_OUT = Path("results/phase_xa_memory_distillation")
VARIANTS = ["XA0_OfficialMRADCLIP", "XA2_CLIPPreservedAdapter", "XA4_DualTeacherFullMemory"]
CORRUPTIONS = [
    "clean",
    "A1_brightness_m20",
    "A2_brightness_p20",
    "A3_contrast_m20",
    "A4_contrast_p20",
    "A5_gamma_0p8",
    "A6_gamma_1p2",
    "A7_saturation_0p7",
    "A8_saturation_1p3",
    "A9_warm",
    "A10_cool",
    "A11_blur_sigma1",
    "A12_noise_mild",
    "A13_jpeg_q60",
]
FEATURE_SAMPLE_LIMIT = 64
EPS = 1e-8


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(device):
    mean = torch.tensor(OPENAI_DATASET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(OPENAI_DATASET_STD, device=device).view(3, 1, 1)
    return mean, std


def to_pixel(img, device):
    mean, std = mean_std(device)
    return (img.to(device) * std + mean).clamp(0.0, 1.0)


def to_normalized(px, device):
    mean, std = mean_std(device)
    return (px.clamp(0.0, 1.0) - mean) / std


def jpeg_tensor(px, quality=60):
    arr = (px.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    arr2 = np.asarray(out).astype(np.float32) / 255.0
    return torch.from_numpy(arr2).permute(2, 0, 1).to(px.device)


def corrupt_image(img, corruption, device, idx=0):
    if corruption == "clean":
        return img.to(device)
    px = to_pixel(img, device)
    if corruption == "A1_brightness_m20":
        px = px * 0.8
    elif corruption == "A2_brightness_p20":
        px = px * 1.2
    elif corruption == "A3_contrast_m20":
        m = px.mean(dim=(1, 2), keepdim=True)
        px = (px - m) * 0.8 + m
    elif corruption == "A4_contrast_p20":
        m = px.mean(dim=(1, 2), keepdim=True)
        px = (px - m) * 1.2 + m
    elif corruption == "A5_gamma_0p8":
        px = px.clamp_min(0).pow(0.8)
    elif corruption == "A6_gamma_1p2":
        px = px.clamp_min(0).pow(1.2)
    elif corruption == "A7_saturation_0p7":
        gray = (0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]).unsqueeze(0)
        px = gray + 0.7 * (px - gray)
    elif corruption == "A8_saturation_1p3":
        gray = (0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]).unsqueeze(0)
        px = gray + 1.3 * (px - gray)
    elif corruption == "A9_warm":
        px = px * torch.tensor([1.08, 1.00, 0.92], device=device).view(3, 1, 1)
    elif corruption == "A10_cool":
        px = px * torch.tensor([0.92, 1.00, 1.08], device=device).view(3, 1, 1)
    elif corruption == "A11_blur_sigma1":
        px = TVF.gaussian_blur(px, kernel_size=[7, 7], sigma=[1.0, 1.0])
    elif corruption == "A12_noise_mild":
        gen = torch.Generator(device=device)
        gen.manual_seed(20260813 + int(idx))
        px = px + torch.randn(px.shape, generator=gen, device=device) * 0.03
    elif corruption == "A13_jpeg_q60":
        px = jpeg_tensor(px, quality=60)
    else:
        raise ValueError(corruption)
    return to_normalized(px, device)


def append_result(results, cls_name, item, image_score, anomaly_map):
    mask = (item["img_mask"].detach().cpu() > 0.5).float()
    results[cls_name]["imgs_masks"].append(mask)
    results[cls_name]["gt_sp"].append(int(item["anomaly"]))
    results[cls_name]["pr_sp"].append(float(image_score))
    results[cls_name]["anomaly_maps"].append(anomaly_map.detach().cpu())


def score_and_map(
    model,
    prompt_learner,
    image_proj,
    patch_proj,
    prompt_proj,
    image_cache,
    patch_cache,
    item,
    image,
    device,
    adapter,
    snapshot=False,
):
    outputs_pack = visual_outputs(model, image, adapter=adapter, grad=False, snapshot_mode="true_clone" if snapshot else None)
    outputs = outputs_pack["outputs"] if snapshot else outputs_pack
    image_feature = normalized_image_feature(outputs)
    image_post, _, _, _ = image_memory_scores(image_feature, image_cache["keys"], image_cache["values"], image_proj)
    cls_memory = image_post[:, 1]
    patch_projection = average_neighbor(outputs[3][3])
    patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
    patch_bias = average_neighbor(outputs[1][3])
    patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)
    _, patch_bias_tokens, _, _ = compute_patch_socre(
        patch_bias,
        patch_cache["keys"],
        patch_cache["values"],
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
    maps = []
    for pidx, patch_feature in enumerate(outputs[3]):
        if pidx >= 3:
            patch_feature = average_neighbor(patch_feature)
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            sim = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
            smap = AnomalyCLIP_lib.get_similarity_map(sim, 518)
            maps.append((smap[..., 1] + 1 - smap[..., 0]) / 2.0)
    amap = torch.stack(maps)
    amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=4)) for x in amap.detach().cpu()], dim=0).sum(dim=0)
    topk = torch.topk(amap.flatten(), max(1, int(amap.numel() * 0.01))).values.mean()
    final_score = 0.3 * topk.to(device) + 0.7 * cls_memory
    return final_score, amap, outputs_pack


def map64(amap):
    x = amap.float()
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    return F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False).reshape(-1).cpu().numpy()


def safe_corr(a, b, kind):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or b.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    try:
        if kind == "pearson":
            return float(pearsonr(a, b).statistic)
        return float(spearmanr(a, b).statistic)
    except Exception:
        return float("nan")


def safe_cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def metric_row(direction, fold, variant, seed, corruption, metrics, images, seconds):
    return {
        "direction": direction,
        "fold": fold,
        "variant": variant,
        "seed": seed,
        "corruption": corruption,
        "I-AUROC": metric_value(metrics, "I-AUROC"),
        "I-AP": metric_value(metrics, "I-AP"),
        "P-AUROC": metric_value(metrics, "P-AUROC"),
        "PRO": metric_value(metrics, "PRO"),
        "images": images,
        "seconds": seconds,
    }


def build_cache(model, dataset, train_indices, dataset_name, device, adapter):
    if adapter is None:
        image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None)
    else:
        image_cache, patch_cache = build_adapted_task_cache(model, dataset, train_indices, dataset_name, device, adapter=adapter)
    return to_device_cache(image_cache, device), to_device_cache(patch_cache, device)


def load_clean_lookup():
    rows = read_csv(XA_OUT / "source_checkpoint_readout.csv")
    lookup = {}
    for r in rows:
        key = (r["direction"], int(r["fold"]), r["variant"], int(r["seed"]))
        lookup[key] = r
    return lookup


def evaluate_model_corruption(
    direction,
    fold,
    row,
    model,
    prompt_learner,
    image_proj,
    patch_proj,
    prompt_proj,
    dataset,
    train_indices,
    val_indices,
    dataset_name,
    device,
    corruption,
    adapter,
    image_cache,
    patch_cache,
    clean_predictions=None,
):
    obj_list = sorted({dataset[i]["cls_name"] for i in val_indices})
    results = init_results(obj_list)
    per_image = []
    maps = {}
    start = time.time()
    with torch.no_grad():
        for idx in val_indices:
            item = dataset[idx]
            img = corrupt_image(item["img"], corruption, device, idx=idx).unsqueeze(0)
            score, amap, _ = score_and_map(
                model,
                prompt_learner,
                image_proj,
                patch_proj,
                prompt_proj,
                image_cache,
                patch_cache,
                item,
                img,
                device,
                adapter,
                snapshot=False,
            )
            append_result(results, item["cls_name"], item, score.detach().cpu(), amap)
            per_image.append(
                {
                    "idx": idx,
                    "category": item["cls_name"],
                    "label": int(item["anomaly"]),
                    "score": float(score.detach().cpu()),
                }
            )
            maps[idx] = map64(amap)
    metrics = evaluate_results(results, obj_list, compute_pro=True)
    summary = metric_row(direction, fold, row["variant"], int(row["seed"]), corruption, metrics, len(val_indices), time.time() - start)
    per_category = []
    for category, vals in metrics.get("per_class", {}).items():
        per_category.append(
            {
                "direction": direction,
                "fold": fold,
                "variant": row["variant"],
                "seed": row["seed"],
                "corruption": corruption,
                "category": category,
                "I-AUROC": metric_value({"mean": vals}, "I-AUROC"),
                "I-AP": metric_value({"mean": vals}, "I-AP"),
                "P-AUROC": metric_value({"mean": vals}, "P-AUROC"),
                "PRO": metric_value({"mean": vals}, "PRO"),
            }
        )
    consistency = None
    if clean_predictions is not None and corruption != "clean":
        clean_scores = [clean_predictions["scores"][r["idx"]] for r in per_image]
        corr_scores = [r["score"] for r in per_image]
        map_cos = [safe_cos(clean_predictions["maps"][r["idx"]], maps[r["idx"]]) for r in per_image]
        map_spear = [safe_corr(clean_predictions["maps"][r["idx"]], maps[r["idx"]], "spearman") for r in per_image]
        consistency = {
            "direction": direction,
            "fold": fold,
            "variant": row["variant"],
            "seed": row["seed"],
            "corruption": corruption,
            "image_score_pearson": safe_corr(clean_scores, corr_scores, "pearson"),
            "image_score_spearman": safe_corr(clean_scores, corr_scores, "spearman"),
            "pixel_map_cosine_mean": float(np.nanmean(map_cos)),
            "pixel_map_spearman_mean": float(np.nanmean(map_spear)),
            "images": len(per_image),
        }
    return summary, per_category, {"scores": {r["idx"]: r["score"] for r in per_image}, "maps": maps}, consistency


def flatten_feature(x):
    if isinstance(x, list):
        x = x[0]
    return x.detach().float().reshape(-1)


def feature_diag(
    direction,
    fold,
    row,
    model,
    prompt_learner,
    image_proj,
    patch_proj,
    prompt_proj,
    dataset,
    indices,
    device,
    corruption,
    adapter,
):
    if corruption == "clean":
        return [], []
    feature_dist = defaultdict(list)
    residual_cos, residual_delta = [], []
    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]
            clean_img = corrupt_image(item["img"], "clean", device, idx=idx).unsqueeze(0)
            corr_img = corrupt_image(item["img"], corruption, device, idx=idx).unsqueeze(0)
            base_clean = visual_outputs(model, clean_img, adapter=None, grad=False, snapshot_mode="true_clone")
            base_corr = visual_outputs(model, corr_img, adapter=None, grad=False, snapshot_mode="true_clone")
            if adapter is None:
                ad_clean = base_clean
                ad_corr = base_corr
            else:
                ad_clean = visual_outputs(model, clean_img, adapter=adapter, grad=False, snapshot_mode="true_clone")
                ad_corr = visual_outputs(model, corr_img, adapter=adapter, grad=False, snapshot_mode="true_clone")
            pairs = {
                "block18_branch0": (
                    flatten_feature(base_clean["snapshots"]["true_block18"][0]),
                    flatten_feature(base_corr["snapshots"]["true_block18"][0]),
                    flatten_feature(ad_clean["snapshots"]["true_block18"][0]),
                    flatten_feature(ad_corr["snapshots"]["true_block18"][0]),
                ),
                "block18_branch1": (
                    flatten_feature(base_clean["snapshots"]["true_block18"][1]),
                    flatten_feature(base_corr["snapshots"]["true_block18"][1]),
                    flatten_feature(ad_clean["snapshots"]["true_block18"][1]),
                    flatten_feature(ad_corr["snapshots"]["true_block18"][1]),
                ),
                "final_CLS": (
                    flatten_feature(base_clean["outputs"][0]),
                    flatten_feature(base_corr["outputs"][0]),
                    flatten_feature(ad_clean["outputs"][0]),
                    flatten_feature(ad_corr["outputs"][0]),
                ),
                "final_patch": (
                    flatten_feature(base_clean["outputs"][1][3]),
                    flatten_feature(base_corr["outputs"][1][3]),
                    flatten_feature(ad_clean["outputs"][1][3]),
                    flatten_feature(ad_corr["outputs"][1][3]),
                ),
            }
            for name, (bc, bx, ac, ax) in pairs.items():
                base_d = 1.0 - float(F.cosine_similarity(bc, bx, dim=0).cpu())
                adapt_d = 1.0 - float(F.cosine_similarity(ac, ax, dim=0).cpu())
                feature_dist[(name, "base_distance")].append(base_d)
                feature_dist[(name, "adapted_distance")].append(adapt_d)
                feature_dist[(name, "amplification")].append(adapt_d / (base_d + EPS))
            if adapter is not None:
                rc = flatten_feature(ad_clean["snapshots"]["true_block18"][0]) - flatten_feature(base_clean["snapshots"]["true_block18"][0])
                rx = flatten_feature(ad_corr["snapshots"]["true_block18"][0]) - flatten_feature(base_corr["snapshots"]["true_block18"][0])
                residual_cos.append(float(F.cosine_similarity(rc, rx, dim=0).cpu()))
                residual_delta.append(float((rx - rc).norm().cpu()))
    frows = []
    for (feature, stat), vals in sorted(feature_dist.items()):
        frows.append(
            {
                "direction": direction,
                "fold": fold,
                "variant": row["variant"],
                "seed": row["seed"],
                "corruption": corruption,
                "feature": feature,
                stat: float(np.nanmean(vals)),
                "samples": len(vals),
            }
        )
    merged = {}
    for r in frows:
        key = (r["direction"], r["fold"], r["variant"], r["seed"], r["corruption"], r["feature"])
        merged.setdefault(key, {k: r[k] for k in ["direction", "fold", "variant", "seed", "corruption", "feature", "samples"]})
        for stat in ["base_distance", "adapted_distance", "amplification"]:
            if stat in r:
                merged[key][stat] = r[stat]
    arows = []
    if adapter is not None:
        arows.append(
            {
                "direction": direction,
                "fold": fold,
                "variant": row["variant"],
                "seed": row["seed"],
                "corruption": corruption,
                "adapter_residual_cosine": float(np.nanmean(residual_cos)),
                "adapter_residual_delta_norm": float(np.nanmean(residual_delta)),
                "samples": len(residual_cos),
            }
        )
    return list(merged.values()), arows


def balanced(row):
    return 0.25 * (float(row["I-AUROC"]) + float(row["I-AP"]) + float(row["P-AUROC"]) + float(row["PRO"]))


def compute_summaries(clean_rows, corruption_rows, percat_rows, consistency_rows, feature_rows, adapter_rows):
    clean_lookup = {(r["direction"], int(r["fold"]), r["variant"], int(r["seed"])): r for r in clean_rows}
    metric_rows = clean_rows + corruption_rows
    metric_lookup = {(r["direction"], int(r["fold"]), r["variant"], int(r["seed"]), r["corruption"]): r for r in metric_rows}
    drop_rows = []
    for r in corruption_rows:
        key = (r["direction"], int(r["fold"]), r["variant"], int(r["seed"]))
        c = clean_lookup[key]
        di = float(c["I-AUROC"]) - float(r["I-AUROC"])
        dap = float(c["I-AP"]) - float(r["I-AP"])
        dp = float(c["P-AUROC"]) - float(r["P-AUROC"])
        dpro = float(c["PRO"]) - float(r["PRO"])
        image_drop = 0.5 * di + 0.5 * dap
        pixel_drop = 0.5 * dp + 0.5 * dpro
        drop_rows.append(
            {
                "direction": r["direction"],
                "fold": r["fold"],
                "variant": r["variant"],
                "seed": r["seed"],
                "corruption": r["corruption"],
                "Drop_I-AUROC": di,
                "Drop_I-AP": dap,
                "Drop_P-AUROC": dp,
                "Drop_PRO": dpro,
                "ImageDrop": image_drop,
                "PixelDrop": pixel_drop,
                "BalancedDrop": 0.5 * image_drop + 0.5 * pixel_drop,
            }
        )
    drop_lookup = {(r["direction"], int(r["fold"]), r["variant"], int(r["seed"]), r["corruption"]): r for r in drop_rows}
    frag_rows = []
    for r in drop_rows:
        if r["variant"] == "XA0_OfficialMRADCLIP":
            continue
        b = drop_lookup.get((r["direction"], int(r["fold"]), "XA0_OfficialMRADCLIP", 0, r["corruption"]))
        if not b:
            continue
        frag_rows.append(
            {
                "direction": r["direction"],
                "fold": r["fold"],
                "variant": r["variant"],
                "seed": r["seed"],
                "corruption": r["corruption"],
                "ExcessFragility_image": r["ImageDrop"] - b["ImageDrop"],
                "ExcessFragility_pixel": r["PixelDrop"] - b["PixelDrop"],
                "ExcessFragility_balanced": r["BalancedDrop"] - b["BalancedDrop"],
            }
        )
    rel_by_model = []
    grouped = defaultdict(list)
    for r in frag_rows:
        grouped[(r["direction"], int(r["fold"]), r["variant"])].append(float(r["ExcessFragility_balanced"]))
    for key, vals in grouped.items():
        arr = np.asarray(vals, dtype=np.float64)
        rel_by_model.append(
            {
                "direction": key[0],
                "fold": key[1],
                "variant": key[2],
                "mean_excess_balanced": float(arr.mean()),
                "median_excess_balanced": float(np.median(arr)),
                "worst_excess_balanced": float(arr.max()),
                "std_excess_balanced": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "positive_corruption_fraction": float((arr > 0).mean()),
                "corruptions": int(arr.size),
            }
        )
    relation_rows = []
    for r in rel_by_model:
        key = (r["direction"], int(r["fold"]), r["variant"], 0)
        base = clean_lookup[(r["direction"], int(r["fold"]), "XA0_OfficialMRADCLIP", 0)]
        cur = clean_lookup[key]
        cg = balanced(cur) - balanced(base)
        relation_rows.append({**r, "CleanGain_vs_XA0": cg, "RobustnessPenalty": r["mean_excess_balanced"]})
    if len(relation_rows) >= 3:
        cg = [r["CleanGain_vs_XA0"] for r in relation_rows]
        rp = [r["RobustnessPenalty"] for r in relation_rows]
        relation_rows.append(
            {
                "direction": "ALL",
                "fold": "ALL",
                "variant": "CORRELATION",
                "CleanGain_vs_XA0": "",
                "RobustnessPenalty": "",
                "pearson": safe_corr(cg, rp, "pearson"),
                "spearman": safe_corr(cg, rp, "spearman"),
            }
        )
    per_corr = []
    by_corr = defaultdict(list)
    for r in frag_rows:
        by_corr[r["corruption"]].append(float(r["ExcessFragility_balanced"]))
    for corr, vals in sorted(by_corr.items()):
        arr = np.asarray(vals, dtype=np.float64)
        per_corr.append(
            {
                "corruption": corr,
                "mean_excess_balanced": float(arr.mean()),
                "median_excess_balanced": float(np.median(arr)),
                "max_excess_balanced": float(arr.max()),
                "positive_fraction": float((arr > 0).mean()),
                "rows": int(arr.size),
            }
        )
    return drop_rows, frag_rows, rel_by_model, relation_rows, per_corr


def decide(rel_by_model, feature_rows, adapter_rows, consistency_rows, clean_rows):
    xa4 = [r for r in rel_by_model if r["variant"] == "XA4_DualTeacherFullMemory"]
    by_direction = defaultdict(list)
    for r in xa4:
        by_direction[r["direction"]].append(r)
    condition_a = False
    for direction, rows in by_direction.items():
        if any(float(r["mean_excess_balanced"]) >= 0.30 for r in rows):
            same_sign = sum(float(r["mean_excess_balanced"]) > 0 for r in rows) > (len(rows) / 2.0)
            condition_a = condition_a or same_sign
    amp = [float(r.get("amplification", float("nan"))) for r in feature_rows if r.get("variant") == "XA4_DualTeacherFullMemory"]
    max_amp = float(np.nanmax(amp)) if amp else float("nan")
    residual_delta = [float(r["adapter_residual_delta_norm"]) for r in adapter_rows if r["variant"] == "XA4_DualTeacherFullMemory"]
    mean_delta = float(np.nanmean(residual_delta)) if residual_delta else float("nan")
    cons = [r for r in consistency_rows if r["variant"] == "XA4_DualTeacherFullMemory"]
    off = [r for r in consistency_rows if r["variant"] == "XA0_OfficialMRADCLIP"]
    pred_lower = False
    if cons and off:
        xa4_c = np.nanmean([float(r["image_score_pearson"]) for r in cons])
        off_c = np.nanmean([float(r["image_score_pearson"]) for r in off])
        xa4_m = np.nanmean([float(r["pixel_map_cosine_mean"]) for r in cons])
        off_m = np.nanmean([float(r["pixel_map_cosine_mean"]) for r in off])
        pred_lower = (xa4_c + 0.02 < off_c) or (xa4_m + 0.02 < off_m)
    mechanism = (max_amp > 1.20) or (np.isfinite(mean_delta) and mean_delta > 1.0) or pred_lower
    clean_lookup = {(r["direction"], int(r["fold"]), r["variant"]): balanced(r) for r in clean_rows}
    xa4_clean_gain_any = any(
        clean_lookup.get((r["direction"], int(r["fold"]), "XA4_DualTeacherFullMemory"), -1e9)
        > clean_lookup.get((r["direction"], int(r["fold"]), "XA0_OfficialMRADCLIP"), 1e9)
        for r in xa4
    )
    if condition_a and mechanism:
        return "PASS_TO_PHASE_XVI_B_DOMAIN_GENERALIZATION", {"condition_a": condition_a, "mechanism": mechanism, "max_feature_amplification": max_amp, "mean_adapter_delta": mean_delta, "xa4_clean_gain_any": xa4_clean_gain_any}
    if xa4 and all(float(r["mean_excess_balanced"]) <= 0 for r in xa4):
        return "XVI_A_NO_SOURCE_DOMAIN_FRAGILITY", {"condition_a": condition_a, "mechanism": mechanism, "max_feature_amplification": max_amp, "mean_adapter_delta": mean_delta, "xa4_clean_gain_any": xa4_clean_gain_any}
    if xa4:
        pos_corr = defaultdict(int)
        for r in xa4:
            if float(r["mean_excess_balanced"]) > 0:
                pos_corr[r["direction"]] += 1
        if not condition_a:
            return "XVI_A_DOMAIN_SHIFT_HYPOTHESIS_UNSUPPORTED", {"condition_a": condition_a, "mechanism": mechanism, "max_feature_amplification": max_amp, "mean_adapter_delta": mean_delta, "xa4_clean_gain_any": xa4_clean_gain_any}
    return "XVI_A_IMPLEMENTATION_FAILURE", {"condition_a": condition_a, "mechanism": mechanism, "max_feature_amplification": max_amp, "mean_adapter_delta": mean_delta, "xa4_clean_gain_any": xa4_clean_gain_any}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clean_lookup_existing = load_clean_lookup()
    clean_rows = read_csv(OUT / "clean_metrics.csv")
    corruption_rows = read_csv(OUT / "corruption_metrics.csv")
    percat_rows = read_csv(OUT / "per_category_summary.csv")
    consistency_rows = read_csv(OUT / "prediction_consistency.csv")
    feature_rows = read_csv(OUT / "feature_fragility.csv")
    adapter_rows = read_csv(OUT / "adapter_fragility.csv")
    progress = read_csv(OUT / "progress.csv")
    completed_clean = {(r["direction"], int(r["fold"]), r["variant"], int(r["seed"])) for r in clean_rows}
    completed_corruption = {(r["direction"], int(r["fold"]), r["variant"], int(r["seed"]), r["corruption"]) for r in corruption_rows}
    fold_plan = [("D1", 0), ("D1", 1), ("D1", 2), ("D2", 0)]
    for direction, fold in fold_plan:
        dataset_name, data_path, checkpoint = direction_cfg(direction)
        model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
        dataset = load_dataset(dataset_name, data_path)
        rotations = deterministic_source_folds(dataset_name)["rotations"]
        if fold >= len(rotations):
            continue
        rotation = rotations[fold]
        train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
        val_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
        rows = selected_rows(direction, fold, VARIANTS, [0])
        available = {r["variant"]: r for r in rows}
        if "XA0_OfficialMRADCLIP" not in available:
            rows.append({"direction": direction, "fold": str(fold), "variant": "XA0_OfficialMRADCLIP", "seed": "0", "status": "COMPLETE", "checkpoint": ""})
        rows = [r for r in rows if r["variant"] in VARIANTS and int(r["seed"]) == 0]
        rows.sort(key=lambda r: VARIANTS.index(r["variant"]))
        for row in rows:
            model_key = (direction, fold, row["variant"], int(row["seed"]))
            required = {(direction, fold, row["variant"], int(row["seed"]), c) for c in CORRUPTIONS if c != "clean"}
            if model_key in completed_clean and required.issubset(completed_corruption):
                progress.append({"direction": direction, "fold": fold, "variant": row["variant"], "corruption": "SKIP_COMPLETE", "seconds": 0.0})
                write_csv(OUT / "progress.csv", progress)
                continue
            adapter, resolved = load_adapter(row, device)
            image_cache, patch_cache = build_cache(model, dataset, train_indices, dataset_name, device, adapter)
            clean_key = (direction, fold, row["variant"], int(row["seed"]))
            clean_pred = None
            for corruption in CORRUPTIONS:
                corr_key = (direction, fold, row["variant"], int(row["seed"]), corruption)
                if corruption != "clean" and corr_key in completed_corruption:
                    continue
                summary, percat, pred, cons = evaluate_model_corruption(
                    direction,
                    fold,
                    row,
                    model,
                    prompt_learner,
                    image_proj,
                    patch_proj,
                    prompt_proj,
                    dataset,
                    train_indices,
                    val_indices,
                    dataset_name,
                    device,
                    corruption,
                    adapter,
                    image_cache,
                    patch_cache,
                    clean_predictions=clean_pred,
                )
                summary["resolved_checkpoint"] = resolved
                if corruption == "clean":
                    old = clean_lookup_existing.get(clean_key)
                    if old:
                        for k in ["I-AUROC", "I-AP", "P-AUROC", "PRO"]:
                            summary[f"existing_{k}"] = old[k]
                            summary[f"delta_vs_existing_{k}"] = float(summary[k]) - float(old[k])
                    if model_key not in completed_clean:
                        clean_rows.append(summary)
                        completed_clean.add(model_key)
                    clean_pred = pred
                else:
                    corruption_rows.append(summary)
                    completed_corruption.add(corr_key)
                    if cons:
                        consistency_rows.append(cons)
                    sample_indices = val_indices[:FEATURE_SAMPLE_LIMIT]
                    fr, ar = feature_diag(
                        direction,
                        fold,
                        row,
                        model,
                        prompt_learner,
                        image_proj,
                        patch_proj,
                        prompt_proj,
                        dataset,
                        sample_indices,
                        device,
                        corruption,
                        adapter,
                    )
                    feature_rows.extend(fr)
                    adapter_rows.extend(ar)
                percat_rows.extend(percat)
                progress.append({"direction": direction, "fold": fold, "variant": row["variant"], "corruption": corruption, "seconds": summary["seconds"]})
                write_csv(OUT / "progress.csv", progress)
                write_csv(OUT / "clean_metrics.csv", clean_rows)
                write_csv(OUT / "corruption_metrics.csv", corruption_rows)
                write_csv(OUT / "prediction_consistency.csv", consistency_rows)
                write_csv(OUT / "feature_fragility.csv", feature_rows)
                write_csv(OUT / "adapter_fragility.csv", adapter_rows)
                write_csv(OUT / "per_category_summary.csv", percat_rows)
    drop_rows, frag_rows, rel_by_model, relation_rows, per_corr = compute_summaries(clean_rows, corruption_rows, percat_rows, consistency_rows, feature_rows, adapter_rows)
    outcome, gate = decide(rel_by_model, feature_rows, adapter_rows, consistency_rows, clean_rows)
    write_csv(OUT / "robustness_drop.csv", drop_rows)
    write_csv(OUT / "relative_fragility.csv", frag_rows)
    write_csv(OUT / "clean_gain_robustness_relation.csv", relation_rows)
    write_csv(OUT / "per_corruption_summary.csv", per_corr)
    report_lines = [
        "# Phase XVI-A Source Domain-Shift Fragility Audit",
        "",
        f"Outcome: **{outcome}**",
        "",
        "No training was run. No target dataset access was performed.",
        "",
        "## Models",
        "",
        "- XA0 / Official MRAD-CLIP",
        "- XA2 / CLIP-preserved adapter",
        "- XA4 / real Memory Teacher adapter",
        "",
        "## XA4 Mean Excess Fragility",
        "",
        "| Direction | Fold | Mean Excess Balanced | Median | Worst | Positive Fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rel_by_model:
        if r["variant"] == "XA4_DualTeacherFullMemory":
            report_lines.append(f"| {r['direction']} | {r['fold']} | {r['mean_excess_balanced']:.4f} | {r['median_excess_balanced']:.4f} | {r['worst_excess_balanced']:.4f} | {r['positive_corruption_fraction']:.3f} |")
    report_lines.extend(["", "## Largest Corruption Families", "", "| Corruption | Mean Excess Balanced | Max | Positive Fraction |", "|---|---:|---:|---:|"])
    for r in sorted(per_corr, key=lambda x: float(x["mean_excess_balanced"]), reverse=True)[:8]:
        report_lines.append(f"| {r['corruption']} | {r['mean_excess_balanced']:.4f} | {r['max_excess_balanced']:.4f} | {r['positive_fraction']:.3f} |")
    corr_row = next((r for r in relation_rows if r.get("variant") == "CORRELATION"), {})
    report_lines.extend(
        [
            "",
            "## Clean-Gain / Fragility Relation",
            "",
            f"- Pearson: `{corr_row.get('pearson', '')}`",
            f"- Spearman: `{corr_row.get('spearman', '')}`",
            "",
            "## Mechanism Gate",
            "",
            f"- condition_a: `{gate.get('condition_a')}`",
            f"- mechanism: `{gate.get('mechanism')}`",
            f"- max_feature_amplification: `{gate.get('max_feature_amplification')}`",
            f"- mean_adapter_delta: `{gate.get('mean_adapter_delta')}`",
            "",
            "Mandatory stop: Phase XVI-B was not started.",
        ]
    )
    (OUT / "DOMAIN_SHIFT_AUDIT_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    payload = {
        "phase": "XVI-A",
        "outcome": outcome,
        "gate": gate,
        "target_accessed": False,
        "trained": False,
        "next": "STOP. Do not train Phase XVI-B automatically.",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write("\n\n## Phase XVI-A\n")
        f.write("- Command: python experiments/phase_xvi_domain_shift_audit.py\n")
        f.write("- Existing checkpoints only; target_accessed=False; trained=False.\n")
        f.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "DOMAIN_SHIFT_AUDIT_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
