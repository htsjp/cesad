import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc
from skimage import measure
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import AnomalyCLIP_lib
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer
from models.mlp import average_neighbor
from utils.dataset import Dataset
from utils.metrics import image_level_metrics, pixel_level_metrics
from utils.transforms import get_transform


TEXT_TOKENIZER = SimpleTokenizer()


def parse_args(description: str):
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--source_dataset", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit per split.")
    return parser.parse_args()


def load_config(path: str, overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = overrides or {}
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def project_root() -> Path:
    return ROOT_DIR


def dataset_path(cfg: Dict, name: str) -> str:
    paths = cfg.get("data_paths", {})
    if name not in paths:
        raise KeyError(f"Missing data path for dataset '{name}' in config.data_paths")
    return paths[name]


def default_source_dataset(target: str) -> str:
    return "visa" if target == "mvtec" else "mvtec"


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "not-a-git-repo"


def save_run_metadata(output_dir: Path, cfg: Dict, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    metadata = {
        "command": " ".join(sys.argv),
        "git_hash": git_hash(),
        "seed": seed,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def copy_log(output_dir: Path, lines: Sequence[str]):
    with open(output_dir / "log.txt", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def load_model(cfg: Dict, device: str):
    params = {
        "Prompt_length": cfg.get("n_ctx", 12),
        "learnabel_text_embedding_depth": cfg.get("depth", 9),
        "learnabel_text_embedding_length": cfg.get("t_n_ctx", 4),
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=cfg.get("dpam_layer", 24))
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_loader(cfg: Dict, dataset_name: str, mode: str, shuffle=False):
    class Args:
        pass

    args = Args()
    args.image_size = cfg.get("image_size", 518)
    args.data_path = dataset_path(cfg, dataset_name)
    args.dataset = dataset_name
    preprocess, target_transform = get_transform(args)
    data = Dataset(
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=dataset_name,
        mode=mode,
    )
    limit = cfg.get("limit")
    if limit:
        data.data_all = data.data_all[: int(limit)]
        data.length = len(data.data_all)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=cfg.get("batch_size", 1),
        shuffle=shuffle,
        num_workers=cfg.get("num_workers", 0),
    )
    return data, loader


def layer_index(features_list: Sequence[int], layer: int) -> int:
    if layer not in features_list:
        raise ValueError(f"Layer {layer} is not in features_list={features_list}")
    return list(features_list).index(layer)


@torch.no_grad()
def extract_layer_features(model, images, features_list, layer, device, smooth=True):
    """Extract one visual layer without using AnomalyCLIP's mutable out_tokens list.

    The upstream AnomalyCLIP transformer stores intermediate tensors by reference while
    later DPAM blocks update them in-place. For a layer audit we need cloned snapshots,
    so this function mirrors VisionTransformer.forward and clones the requested layer.
    """
    visual = model.visual
    x = visual.conv1(images.to(device).type(model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls_init = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_init, x], dim=1)

    side = int((visual.positional_embedding.shape[0] - 1) ** 0.5)
    new_side = int((x.shape[1] - 1) ** 0.5)
    if side != new_side:
        new_pos = visual.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
        new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode="bilinear")
        new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
        visual.positional_embedding.data = torch.cat([visual.positional_embedding[:1, :], new_pos[0]], 0)

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)

    value_tokens = {}
    handles = []
    block = visual.transformer.resblocks[layer - 1]
    if hasattr(block.attn, "qkv"):
        def capture_value(_module, _inputs, output):
            bsz, tokens, channels3 = output.shape
            channels = channels3 // 3
            heads = block.attn.num_heads
            value = output.reshape(bsz, tokens, 3, heads, channels // heads)[:, :, 2]
            value_tokens[layer] = value.reshape(bsz, tokens, channels).detach().clone()

        handles.append(block.attn.qkv.register_forward_hook(capture_value))

    layer_token = None
    x_ori = None
    for idx, r in enumerate(visual.transformer.resblocks, start=1):
        x = r(x, ffn=False)
        current = x[0] if isinstance(x, list) else x
        if idx == layer:
            layer_token = current.detach().clone()
        if isinstance(x, list):
            x_ori = x[1]
        else:
            x_ori = x

    for handle in handles:
        handle.remove()
    if layer_token is None:
        raise ValueError(f"Layer {layer} was not reached by visual transformer")

    token_nld = layer_token.permute(1, 0, 2)
    projected = visual.ln_post(token_nld) @ visual.proj
    patch = projected[:, 1:, :].float()
    cls = (layer_token[0, :, :] @ visual.proj).float()

    value = value_tokens.get(layer)
    if value is not None:
        value = value[:, 1:, :].float() @ visual.proj.float()
    else:
        value = patch.clone()
    if smooth:
        patch = average_neighbor(patch)
        value = average_neighbor(value)
    cls = F.normalize(cls, dim=-1)
    patch = F.normalize(patch, dim=-1)
    value = F.normalize(value, dim=-1)
    return cls, patch, value


def one_hot(labels: torch.Tensor, num_classes=2):
    return F.one_hot(labels.to(torch.int64), num_classes=num_classes).float()


def resize_mask(mask: torch.Tensor, side: int):
    mask = mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    resized = F.interpolate(mask, size=(side, side), mode="bilinear", align_corners=False)
    resized = resized.squeeze(1)
    return (resized > 0.5)


def append_region_mean(features, labels, patch, mask, label):
    selected = patch[mask.reshape(-1)]
    if selected.numel() == 0:
        return
    feat = selected.mean(dim=0, keepdim=True)
    features.append(F.normalize(feat, dim=-1))
    labels.append(torch.tensor([label], device=patch.device))


def build_layer_memory(model, loader, cfg: Dict, layer: int, device: str):
    image_features, image_labels = [], []
    patch_features, patch_labels = [], []
    features_list = cfg["features_list"]
    with torch.no_grad():
        for items in tqdm(loader, desc=f"build layer {layer} memory"):
            images = items["img"].to(device)
            labels = torch.as_tensor(items["anomaly"], device=device)
            gt = items["img_mask"].to(device)
            gt = (gt > 0.5).float()
            cls, patches, _ = extract_layer_features(
                model, images, features_list, layer, device, smooth=cfg.get("average_neighbor", True)
            )
            image_features.append(cls)
            image_labels.append(labels)
            side = int(math.sqrt(patches.shape[1]))
            gt_resized = resize_mask(gt, side)
            for b in range(images.shape[0]):
                patch = patches[b]
                mask = gt_resized[b]
                append_region_mean(patch_features, patch_labels, patch, mask, 1)
                append_region_mean(patch_features, patch_labels, patch, ~mask, 0)
    return {
        "image_keys": torch.cat(image_features, dim=0),
        "image_values": one_hot(torch.cat(image_labels, dim=0)).to(device),
        "patch_keys": torch.cat(patch_features, dim=0),
        "patch_values": one_hot(torch.cat(patch_labels, dim=0)).to(device),
    }


def fps_select(x: torch.Tensor, k: int):
    if x.shape[0] <= k:
        return x
    selected = [0]
    dist = 1 - (x @ x[0:1].T).squeeze(-1)
    for _ in range(1, k):
        idx = torch.argmax(dist).item()
        selected.append(idx)
        dist = torch.minimum(dist, 1 - (x @ x[idx : idx + 1].T).squeeze(-1))
    return x[selected]


def kmeans_select(x: torch.Tensor, k: int, seed: int):
    if x.shape[0] <= k:
        return x
    from sklearn.cluster import KMeans

    x_np = x.detach().float().cpu().numpy()
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(x_np)
    centers = torch.from_numpy(km.cluster_centers_).to(device=x.device, dtype=x.dtype)
    return F.normalize(centers, dim=-1)


def region_variance_feature(x: torch.Tensor):
    if x.shape[0] <= 1:
        return torch.zeros((1, x.shape[1]), device=x.device, dtype=x.dtype)
    return F.normalize(x.var(dim=0, unbiased=False, keepdim=True), dim=-1)


def build_prototype_memory(model, loader, cfg: Dict, memory_type: str, device: str):
    layer = cfg.get("layer", 24)
    k = int(cfg.get("k", 2))
    seed = int(cfg.get("seed", 0))
    image_features, image_labels = [], []
    patch_features, patch_labels = [], []
    features_list = cfg["features_list"]
    with torch.no_grad():
        for items in tqdm(loader, desc=f"build {memory_type} memory"):
            images = items["img"].to(device)
            labels = torch.as_tensor(items["anomaly"], device=device)
            gt = (items["img_mask"].to(device) > 0.5).float()
            cls, patches, _ = extract_layer_features(
                model, images, features_list, layer, device, smooth=cfg.get("average_neighbor", True)
            )
            image_features.append(cls)
            image_labels.append(labels)
            side = int(math.sqrt(patches.shape[1]))
            gt_resized = resize_mask(gt, side)
            for b in range(images.shape[0]):
                patch = patches[b]
                masks = [(gt_resized[b], 1), (~gt_resized[b], 0)]
                for mask, label in masks:
                    selected = patch[mask.reshape(-1)]
                    if selected.numel() == 0:
                        continue
                    if memory_type == "mean":
                        entries = selected.mean(dim=0, keepdim=True)
                    elif memory_type.startswith("kmeans"):
                        entries = kmeans_select(selected, k, seed)
                    elif memory_type.startswith("fps"):
                        entries = fps_select(selected, k)
                    elif memory_type == "token":
                        entries = selected
                    elif memory_type == "statistics":
                        entries = torch.cat([selected.mean(dim=0, keepdim=True), region_variance_feature(selected)], dim=0)
                    else:
                        raise ValueError(f"Unknown memory_type={memory_type}")
                    entries = F.normalize(entries, dim=-1)
                    patch_features.append(entries)
                    patch_labels.append(torch.full((entries.shape[0],), label, device=device, dtype=torch.long))
    return {
        "image_keys": torch.cat(image_features, dim=0),
        "image_values": one_hot(torch.cat(image_labels, dim=0)).to(device),
        "patch_keys": torch.cat(patch_features, dim=0),
        "patch_values": one_hot(torch.cat(patch_labels, dim=0)).to(device),
    }


def retrieval_logits(query, keys, values, temperature=1.0):
    sim = torch.matmul(query, keys.T)
    weights = F.softmax(sim / temperature, dim=-1)
    logits = torch.matmul(weights, values.float())
    return logits, sim, weights


def update_retrieval_stats(stats: Dict, sim: torch.Tensor, values: torch.Tensor, prefix: str, top_k=5):
    if sim.ndim == 3:
        flat = sim.reshape(-1, sim.shape[-1])
    else:
        flat = sim.reshape(-1, sim.shape[-1])
    k = min(top_k, flat.shape[-1])
    vals, idx = torch.topk(flat, k=k, dim=-1)
    probs = F.softmax(flat, dim=-1)
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
    margin = vals[:, 0] - vals[:, 1] if k > 1 else vals[:, 0]
    labels = values.argmax(dim=-1).detach().cpu().numpy()
    top_idx = idx[:, 0].detach().cpu().numpy()
    stats[f"{prefix}_entropy"].extend(entropy.detach().cpu().tolist())
    stats[f"{prefix}_margin"].extend(margin.detach().cpu().tolist())
    stats[f"{prefix}_top1_label"].extend([int(labels[i]) for i in top_idx])
    stats[f"{prefix}_top1_index"].extend([int(i) for i in top_idx])


def summarize_retrieval_stats(stats: Dict, memory_size: Dict):
    summary = {"memory_size": memory_size}
    for key, values in stats.items():
        if key.endswith("_index"):
            count = Counter(values)
            prefix = key[: -len("_top1_index")]
            summary[f"{prefix}_usage_unique"] = len(count)
            summary[f"{prefix}_coverage"] = len(count) / max(1, memory_size.get(prefix, len(count)))
            summary[f"{prefix}_usage_top10"] = count.most_common(10)
        elif key.endswith("_label"):
            summary[key] = dict(Counter(values))
        else:
            arr = np.asarray(values, dtype=np.float64)
            summary[key] = {
                "mean": float(arr.mean()) if arr.size else None,
                "std": float(arr.std()) if arr.size else None,
                "p25": float(np.percentile(arr, 25)) if arr.size else None,
                "p50": float(np.percentile(arr, 50)) if arr.size else None,
                "p75": float(np.percentile(arr, 75)) if arr.size else None,
            }
    return summary


def init_results(obj_list):
    results = {}
    for obj in obj_list:
        results[obj] = {
            "gt_sp": [],
            "pr_sp": [],
            "imgs_masks": [],
            "anomaly_maps": [],
            "specie": [],
        }
    return results


def evaluate_results(results, obj_list, compute_pro=True):
    per_class, means = {}, {}
    image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = [], [], [], [], []

    def safe_metric(fn, *args):
        try:
            return float(fn(*args))
        except Exception:
            return float("nan")

    def append_if_finite(dst, value):
        if np.isfinite(value):
            dst.append(value)

    for obj in obj_list:
        if len(results[obj]["gt_sp"]) == 0:
            continue
        results[obj]["imgs_masks"] = torch.cat(results[obj]["imgs_masks"]).detach().cpu()
        results[obj]["anomaly_maps"] = torch.cat(results[obj]["anomaly_maps"]).detach().cpu().numpy()
        metrics = {
            "I-AUROC": safe_metric(image_level_metrics, results, obj, "image-auroc"),
            "I-AP": safe_metric(image_level_metrics, results, obj, "image-ap"),
            "P-AUROC": safe_metric(fast_pixel_auroc, results, obj),
        }
        if compute_pro:
            metrics["PRO"] = safe_metric(fast_pixel_aupro, results, obj)
        per_class[obj] = {k: float(v) for k, v in metrics.items()}
        append_if_finite(image_auroc, metrics["I-AUROC"])
        append_if_finite(image_ap, metrics["I-AP"])
        append_if_finite(pixel_auroc, metrics["P-AUROC"])
        if compute_pro:
            append_if_finite(pixel_pro, metrics["PRO"])
    means["I-AUROC"] = float(np.mean(image_auroc)) if image_auroc else float("nan")
    means["I-AP"] = float(np.mean(image_ap)) if image_ap else float("nan")
    means["P-AUROC"] = float(np.mean(pixel_auroc)) if pixel_auroc else float("nan")
    if compute_pro:
        means["PRO"] = float(np.mean(pixel_pro)) if pixel_pro else float("nan")
    return {"mean": means, "per_class": per_class}


def fast_cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    masks = np.asarray(masks)
    amaps = np.asarray(amaps)
    if masks.ndim == 4:
        masks = np.squeeze(masks, axis=1)
    if amaps.ndim == 4:
        amaps = np.squeeze(amaps, axis=1)
    masks = masks > 0.5
    amaps = amaps.astype(np.float32, copy=False)
    min_th, max_th = float(np.min(amaps)), float(np.max(amaps))
    if not np.isfinite(min_th) or not np.isfinite(max_th) or max_th <= min_th:
        return float("nan")

    region_labels, region_areas = [], []
    for mask in masks:
        labels = measure.label(mask.astype(np.uint8))
        if labels.max() > 0:
            areas = np.bincount(labels.ravel())[1:].astype(np.float64)
        else:
            areas = np.zeros(0, dtype=np.float64)
        region_labels.append(labels)
        region_areas.append(areas)

    if sum(len(x) for x in region_areas) == 0:
        return float("nan")

    inverse_masks = ~masks
    inverse_count = int(inverse_masks.sum())
    if inverse_count == 0:
        return float("nan")

    thresholds = np.linspace(min_th, max_th, num=max_step, endpoint=False, dtype=np.float32)
    pros, fprs = [], []
    for th in thresholds:
        binary = amaps > th
        fp_pixels = np.logical_and(inverse_masks, binary).sum()
        fprs.append(float(fp_pixels) / inverse_count)

        per_region = []
        for binary_amap, labels, areas in zip(binary, region_labels, region_areas):
            if areas.size == 0:
                continue
            hits = np.bincount(labels[binary_amap].ravel(), minlength=areas.size + 1)[1:].astype(np.float64)
            per_region.extend((hits / areas).tolist())
        pros.append(float(np.mean(per_region)) if per_region else float("nan"))

    pros = np.asarray(pros, dtype=np.float64)
    fprs = np.asarray(fprs, dtype=np.float64)
    idxes = np.isfinite(pros) & np.isfinite(fprs) & (fprs < expect_fpr)
    if idxes.sum() < 2:
        return float("nan")
    fprs_sel = fprs[idxes]
    pros_sel = pros[idxes]
    order = np.argsort(fprs_sel)
    fprs_sel = fprs_sel[order]
    pros_sel = pros_sel[order]
    denom = fprs_sel.max() - fprs_sel.min()
    if denom <= 0:
        return float("nan")
    fprs_norm = (fprs_sel - fprs_sel.min()) / denom
    return float(auc(fprs_norm, pros_sel))


def fast_pixel_aupro(results, obj):
    gt = np.asarray(results[obj]["imgs_masks"])
    pr = np.asarray(results[obj]["anomaly_maps"])
    return fast_cal_pro_score(gt, pr)


def fast_pixel_auroc(results, obj, max_exact_pixels=20_000_000, bins=65536):
    gt = np.asarray(results[obj]["imgs_masks"])
    pr = np.asarray(results[obj]["anomaly_maps"])
    if gt.ndim == 4:
        gt = np.squeeze(gt, axis=1)
    if pr.ndim == 4:
        pr = np.squeeze(pr, axis=1)
    labels = (gt > 0.5).ravel()
    scores = pr.astype(np.float32, copy=False).ravel()
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    if labels.size <= max_exact_pixels:
        return exact_binary_auc(labels, scores)
    return histogram_binary_auc(labels, scores, bins=bins)


def exact_binary_auc(labels, scores):
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    n = sorted_scores.size
    pos = int(sorted_labels.sum())
    neg = n - pos
    rank_sum = 0.0
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        pos_count = int(sorted_labels[start:end].sum())
        if pos_count:
            avg_rank = (start + 1 + end) / 2.0
            rank_sum += pos_count * avg_rank
        start = end
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def histogram_binary_auc(labels, scores, bins=65536):
    min_s, max_s = float(scores.min()), float(scores.max())
    if not np.isfinite(min_s) or not np.isfinite(max_s) or max_s <= min_s:
        return float("nan")
    pos_scores = scores[labels]
    neg_scores = scores[~labels]
    hist_pos, edges = np.histogram(pos_scores, bins=bins, range=(min_s, max_s))
    hist_neg, _ = np.histogram(neg_scores, bins=edges)
    tp = np.cumsum(hist_pos[::-1])
    fp = np.cumsum(hist_neg[::-1])
    tpr = tp / max(1, pos_scores.size)
    fpr = fp / max(1, neg_scores.size)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(auc(fpr, tpr))


@torch.no_grad()
def run_visual_retrieval(model, memory, loader, cfg: Dict, layer: int, device: str):
    features_list = cfg["features_list"]
    image_size = cfg.get("image_size", 518)
    sigma = cfg.get("sigma", 4)
    fusion_k = cfg.get("fusion_k", 0.7)
    topk_ratio = cfg.get("topk_ratio", 0.01)
    compute_pro = cfg.get("compute_pixel_aupro", True)
    stats = defaultdict(list)
    results = init_results(loader.dataset.obj_list)
    anomaly_maps_to_save = cfg.get("save_anomaly_maps", False)
    saved_maps = {}

    for items in tqdm(loader, desc=f"eval layer {layer}"):
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        if "specie_name" in items:
            results[cls_name]["specie"].append(items["specie_name"])
        cls, patches, _ = extract_layer_features(
            model, image, features_list, layer, device, smooth=cfg.get("average_neighbor", True)
        )
        image_logits, image_sim, _ = retrieval_logits(
            cls, memory["image_keys"].to(device), memory["image_values"].to(device)
        )
        patch_logits, patch_sim, _ = retrieval_logits(
            patches, memory["patch_keys"].to(device), memory["patch_values"].to(device)
        )
        update_retrieval_stats(stats, image_sim, memory["image_values"], "image", cfg.get("retrieval_top_k", 5))
        update_retrieval_stats(stats, patch_sim, memory["patch_values"], "patch", cfg.get("retrieval_top_k", 5))
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack(
            [torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()],
            dim=0,
        )
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1)
        score = (1 - fusion_k) * topk_mean.to(device) + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        if anomaly_maps_to_save:
            saved_maps[items["img_path"][0]] = amap[0].numpy()

    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=compute_pro)
    memory_size = {"image": int(memory["image_keys"].shape[0]), "patch": int(memory["patch_keys"].shape[0])}
    return metrics, summarize_retrieval_stats(stats, memory_size), saved_maps


def save_metrics(output_dir: Path, metrics: Dict, retrieval: Optional[Dict] = None):
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if retrieval is not None:
        with open(output_dir / "retrieval_stats.json", "w", encoding="utf-8") as f:
            json.dump(retrieval, f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_layer_figures(output_dir: Path, rows: Sequence[Dict]):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return
    layers = [r["Layer"] for r in rows]
    metrics = ["I-AUROC", "I-AP", "P-AUROC", "PRO"]
    plt.figure(figsize=(8, 5))
    for metric in metrics:
        plt.plot(layers, [r[metric] for r in rows], marker="o", label=metric)
    plt.xlabel("CLIP ViT layer")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "layer_vs_metric.png", dpi=200)
    plt.close()
    data = np.array([[r[m] for m in metrics] for r in rows])
    plt.figure(figsize=(7, 4))
    sns.heatmap(data, annot=True, fmt=".3f", xticklabels=metrics, yticklabels=layers, cmap="viridis")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(output_dir / "layer_heatmap.png", dpi=200)
    plt.close()


def tokenize_texts(texts: Sequence[str], context_length=77):
    sot_token = TEXT_TOKENIZER.encoder["<|startoftext|>"]
    eot_token = TEXT_TOKENIZER.encoder["<|endoftext|>"]
    result = torch.zeros(len(texts), context_length, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = [sot_token] + TEXT_TOKENIZER.encode(text) + [eot_token]
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        result[i, : len(tokens)] = torch.tensor(tokens)
    return result


def encode_plain_text(model, tokens: torch.Tensor) -> torch.Tensor:
    x = model.token_embedding(tokens).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)
    for block in model.transformer.resblocks:
        x = x + block.attention(block.ln_1(x))
        x = x + block.mlp(block.ln_2(x))
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    return x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)] @ model.text_projection


@torch.no_grad()
def build_text_memory(model, cfg: Dict, device: str):
    normal = cfg["text_prompts"]["normal"]
    anomaly = cfg["text_prompts"]["anomaly"]
    texts = normal + anomaly
    tokens = tokenize_texts(texts).to(device)
    feats = encode_plain_text(model, tokens).float()
    feats = F.normalize(feats, dim=-1)
    labels = torch.tensor([0] * len(normal) + [1] * len(anomaly), device=device)
    return {
        "texts": texts,
        "keys": feats,
        "values": one_hot(labels).to(device),
        "normal_count": len(normal),
        "anomaly_count": len(anomaly),
    }


def save_anomaly_maps(output_dir: Path, maps: Dict[str, np.ndarray]):
    if not maps:
        return
    maps_dir = output_dir / "anomaly_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for idx, (path, amap) in enumerate(maps.items()):
        name = f"{idx:06d}.npy"
        np.save(maps_dir / name, amap)
        manifest[name] = path
    with open(maps_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
