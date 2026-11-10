import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from models.mlp import AnomalyMLP, MLP, Projector, average_neighbor
from models.prompt_learner import AnomalyCLIP_PromptLearner
from mrad import build_cache_model, build_patch_cache_model, compute_patch_socre
from phase_x.gradient_visual_forward import _safe_resblock_forwards, forward_visual_with_module, freeze_official_modules
from phase_x.insertion_modules import IdentityInsertion, ZeroResidualModule
from phase_x.parity_tests import tensor_stats
from utils.dataset import Dataset
from utils.metrics import image_level_metrics, pixel_level_metrics
from utils.transforms import get_transform


OUT = Path("results/phase_x0a_alias_parity")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_components(checkpoint_path, device):
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params).to(device).eval()
    image_proj = MLP().to(device).eval()
    patch_proj = Projector(1024, 768, length=2).to(device).eval()
    prompt_proj = AnomalyMLP().to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    image_proj.load_state_dict(ckpt["image_proj"])
    patch_proj.load_state_dict(ckpt["patch_proj"])
    prompt_learner.load_state_dict(ckpt["prompt_learner"])
    prompt_proj.load_state_dict(ckpt["prompt_proj"])
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=24)
    freeze_official_modules(model, prompt_learner, image_proj, patch_proj, prompt_proj)
    return model, prompt_learner, image_proj, patch_proj, prompt_proj


def safe_visual_eval(model, image, features_list):
    wrapped = getattr(type(model.visual).forward, "__wrapped__")
    with _safe_resblock_forwards(model.visual):
        with torch.no_grad():
            return wrapped(
                model.visual,
                image.type(model.dtype),
                features_list,
                ori_patch=False,
                proj_use=True,
                DPAM_layer=24,
                ffn=False,
            )


def load_caches(cache_dir, cache_name, device):
    image_cache = torch.load(Path(cache_dir) / f"cache_model_{cache_name}.pt", map_location=device)
    patch_cache = torch.load(Path(cache_dir) / f"cache_patch_model_{cache_name}.pt", map_location=device)
    return (
        image_cache["keys"].to(device),
        image_cache["values"].to(device),
        patch_cache["keys"].to(device),
        patch_cache["values"].to(device),
    )


def evaluate_safe_identity(direction, dataset_name, data_path, checkpoint_path, cache_dir, cache_name, device, limit=None):
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint_path, device)
    cache_key, cache_value, cache_keys_patch, cache_values_patch = load_caches(cache_dir, cache_name, device)
    args = argparse.Namespace(image_size=518)
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=data_path, transform=preprocess, target_transform=target_transform, dataset_name=dataset_name)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    obj_list = data.obj_list
    results = {obj: {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []} for obj in obj_list}
    per_sample = []
    start = time.time()
    peak = 0
    for idx, items in enumerate(tqdm(loader, desc=f"safe-eval {direction}")):
        if limit is not None and idx >= limit:
            break
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        cls_id = items["cls_id"]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask.squeeze(0))
        results[cls_name]["gt_sp"].extend(items["anomaly"].detach().cpu())
        with torch.no_grad():
            image_features, patch_features, _, patch_projections = safe_visual_eval(model, image, [6, 12, 18, 24])
            patch_projection = average_neighbor(patch_projections[3])
            patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
            patch_bias = average_neighbor(patch_features[3])
            patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)
            seg_logit, patch_bias_tokens, _, _ = compute_patch_socre(
                patch_bias,
                cache_keys_patch,
                cache_values_patch,
                device=device,
                proj=patch_proj,
                need_mask=False,
                patch_projection=patch_projection,
                gt_mask=items["img_mask"],
                is_mradft=False,
                use_proj=True,
            )
            bias = prompt_proj(patch_bias_tokens[:, 0, :], patch_bias_tokens[:, 1, :])
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_proj, cache_keys_proj = image_proj(image_features, cache_key)
            cache_logits = torch.softmax(image_features_proj @ cache_keys_proj.t(), dim=-1) @ cache_value.float()
            cls_memory = cache_logits[:, 1]
            anomaly_map_list = []
            for pidx, patch_feature in enumerate(patch_projections):
                if pidx >= 3:
                    patch_feature = average_neighbor(patch_feature)
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                    sim_map = AnomalyCLIP_lib.get_similarity_map(similarity, 518)
                    anomaly_map_list.append((sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0)
            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=4)) for i in anomaly_map.detach().cpu()], dim=0).sum(dim=0)
            topk_mean = torch.topk(anomaly_map.flatten(), int(anomaly_map.numel() * 0.01)).values.mean()
            final_score = 0.3 * topk_mean + 0.7 * cls_memory
            results[cls_name]["pr_sp"].extend(final_score.detach().cpu())
            results[cls_name]["anomaly_maps"].append(anomaly_map)
            per_sample.append({
                "direction": direction,
                "idx": idx,
                "class": cls_name,
                "label": int(items["anomaly"]),
                "cls_memory": float(cls_memory.detach().cpu()),
                "prompt_top1": float(topk_mean.detach().cpu()),
                "final_score": float(final_score.detach().cpu()),
            })
        if torch.cuda.is_available():
            peak = max(peak, torch.cuda.max_memory_allocated())
    metrics = {}
    table_rows = []
    for obj in obj_list:
        if len(results[obj]["gt_sp"]) == 0:
            continue
        row = {"direction": direction, "object": obj}
        row["pixel_auroc"] = pixel_level_metrics(results, obj, "pixel-auroc") * 100
        row["pixel_aupro"] = pixel_level_metrics(results, obj, "pixel-aupro") * 100
        row["image_auroc"] = image_level_metrics(results, obj, "image-auroc") * 100
        row["image_ap"] = image_level_metrics(results, obj, "image-ap") * 100
        table_rows.append(row)
    mean = {
        "direction": direction,
        "object": "mean",
        "pixel_auroc": float(np.mean([r["pixel_auroc"] for r in table_rows])),
        "pixel_aupro": float(np.mean([r["pixel_aupro"] for r in table_rows])),
        "image_auroc": float(np.mean([r["image_auroc"] for r in table_rows])),
        "image_ap": float(np.mean([r["image_ap"] for r in table_rows])),
    }
    table_rows.append(mean)
    return table_rows, per_sample, {"direction": direction, "seconds": time.time() - start, "peak_gpu_bytes": peak, "images": len(per_sample), "batch_size": 1, "precision": "official fp32 tensors"}


def rebuild_image_cache_safe(dataset_name, data_path, checkpoint_path, out_path, device):
    if out_path.exists() and out_path.stat().st_size > 0:
        return torch.load(out_path, map_location="cpu")
    model, *_ = load_components(checkpoint_path, device)
    args = argparse.Namespace(image_size=518)
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=data_path, transform=preprocess, target_transform=target_transform, dataset_name=dataset_name)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    keys, labels = [], []
    for items in tqdm(loader, desc=f"rebuild-image-cache {dataset_name}"):
        image = items["img"].to(device)
        with torch.no_grad():
            image_features = safe_visual_eval(model, image, [6, 12, 18, 24])[0]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            keys.append(image_features.detach().cpu())
            labels.append(items["anomaly"].long())
    raw = torch.cat(labels).long()
    d = {"keys": torch.cat(keys, dim=0), "values": F.one_hot(raw, num_classes=2).float()}
    torch.save(d, out_path)
    return d


def rebuild_patch_cache_safe(dataset_name, data_path, checkpoint_path, out_path, device):
    if out_path.exists() and out_path.stat().st_size > 0:
        return torch.load(out_path, map_location="cpu")
    model, *_ = load_components(checkpoint_path, device)
    args = argparse.Namespace(image_size=518)
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=data_path, transform=preprocess, target_transform=target_transform, dataset_name=dataset_name)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    feats, labels = [], []
    for items in tqdm(loader, desc=f"rebuild-patch-cache {dataset_name}"):
        image = items["img"].to(device)
        gt = items["img_mask"].squeeze().to(device)
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0
        with torch.no_grad():
            _, patch_features, _, _ = safe_visual_eval(model, image, [6, 12, 18, 24])
            patch = average_neighbor(patch_features[3]).reshape(37, 37, -1)
            mask = F.interpolate(gt.unsqueeze(0).unsqueeze(0), size=(37, 37), mode="bilinear", align_corners=False).squeeze()
            for value, lab in [((mask == 1), 1), ((mask == 0), 0)]:
                if value.sum() > 0:
                    f = patch[value].mean(dim=0, keepdim=True)
                    f = f / f.norm(dim=-1, keepdim=True)
                    feats.append(f.detach().cpu())
                    labels.append(torch.tensor([lab]))
    raw = torch.cat(labels).long()
    d = {"keys": torch.cat(feats, dim=0), "values": F.one_hot(raw, num_classes=2).float()}
    torch.save(d, out_path)
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows, samples, costs = [], [], []
    configs = [
        ("D1", "visa", "data/VISA", "checkpoints/test_on_visa.pth", "cache", "mvtec", [85.20, 88.00, 95.80, 88.00]),
        ("D2", "mvtec", "data/mvtec", "checkpoints/test_on_mvtec.pth", "cache", "visa", [93.90, 97.30, 92.90, 86.80]),
    ]
    for cfg in configs:
        m, s, c = evaluate_safe_identity(*cfg[:6], device)
        ref = cfg[6]
        mean = m[-1]
        mean["ref_image_auroc"] = ref[0]
        mean["ref_image_ap"] = ref[1]
        mean["ref_pixel_auroc"] = ref[2]
        mean["ref_pixel_aupro"] = ref[3]
        mean["delta_image_auroc"] = mean["image_auroc"] - ref[0]
        mean["delta_image_ap"] = mean["image_ap"] - ref[1]
        mean["delta_pixel_auroc"] = mean["pixel_auroc"] - ref[2]
        mean["delta_pixel_aupro"] = mean["pixel_aupro"] - ref[3]
        rows.extend(m)
        samples.extend(s)
        costs.append(c)
        write_csv(OUT / "metric_parity_safe_identity_running.csv", rows)
    write_csv(OUT / "metric_parity.csv", rows)
    write_csv(OUT / "safe_identity_sample_scores.csv", samples)
    write_csv(OUT / "computational_costs.csv", costs)
    # Cache rebuilds are saved for both source datasets.
    cache_root = OUT / "safe_rebuilt_cache"
    cache_root.mkdir(exist_ok=True)
    rebuild_rows = []
    for dataset_name, data_path, ckpt, cache_name in [
        ("mvtec", "data/mvtec", "checkpoints/test_on_visa.pth", "mvtec"),
        ("visa", "data/VISA", "checkpoints/test_on_mvtec.pth", "visa"),
    ]:
        img = rebuild_image_cache_safe(dataset_name, data_path, ckpt, cache_root / f"cache_model_{cache_name}.pt", device)
        pat = rebuild_patch_cache_safe(dataset_name, data_path, ckpt, cache_root / f"cache_patch_model_{cache_name}.pt", device)
        rebuild_rows.append({
            "dataset": dataset_name,
            "image_keys_shape": "x".join(map(str, img["keys"].shape)),
            "image_values_shape": "x".join(map(str, img["values"].shape)),
            "patch_keys_shape": "x".join(map(str, pat["keys"].shape)),
            "patch_values_shape": "x".join(map(str, pat["values"].shape)),
            "image_key_norm_mean": float(img["keys"].norm(dim=-1).mean()),
            "patch_key_norm_mean": float(pat["keys"].norm(dim=-1).mean()),
        })
    write_csv(OUT / "cache_rebuild_shapes.csv", rebuild_rows)
    # Evaluate rebuilt caches.
    cache_rows = []
    for cfg in [
        ("D1_rebuilt_cache", "visa", "data/VISA", "checkpoints/test_on_visa.pth", str(cache_root), "mvtec", [85.20, 88.00, 95.80, 88.00]),
        ("D2_rebuilt_cache", "mvtec", "data/mvtec", "checkpoints/test_on_mvtec.pth", str(cache_root), "visa", [93.90, 97.30, 92.90, 86.80]),
    ]:
        m, _, c = evaluate_safe_identity(*cfg[:6], device)
        mean = m[-1]
        ref = cfg[6]
        mean["ref_image_auroc"] = ref[0]
        mean["ref_image_ap"] = ref[1]
        mean["ref_pixel_auroc"] = ref[2]
        mean["ref_pixel_aupro"] = ref[3]
        mean["delta_image_auroc"] = mean["image_auroc"] - ref[0]
        mean["delta_image_ap"] = mean["image_ap"] - ref[1]
        mean["delta_pixel_auroc"] = mean["pixel_auroc"] - ref[2]
        mean["delta_pixel_aupro"] = mean["pixel_aupro"] - ref[3]
        cache_rows.append(mean)
        costs.append(c)
        write_csv(OUT / "cache_parity_running.csv", cache_rows)
    write_csv(OUT / "cache_parity.csv", cache_rows)
    write_csv(OUT / "computational_costs.csv", costs)


if __name__ == "__main__":
    main()
