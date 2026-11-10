import math
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from base_residual_memory.canonical import canonical_fuse
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from memory_strategies import STRATEGIES
from mrad_audit_utils import one_hot, resize_mask


def sample_id(items):
    if "img_path" in items:
        value = items["img_path"][0] if isinstance(items["img_path"], (list, tuple)) else items["img_path"]
        return str(value)
    return f"{items['cls_name'][0]}_{int(torch.as_tensor(items['anomaly'])[0])}"


def branch_features(outputs, branch):
    sem = F.normalize(outputs[24]["T"][:, 1:, :].float(), dim=-1)
    loc = F.normalize((F.normalize(outputs[12]["V"][:, 1:, :].float(), dim=-1) + F.normalize(outputs[18]["V"][:, 1:, :].float(), dim=-1)) / 2.0, dim=-1)
    if branch == "semantic":
        return sem
    if branch == "local":
        return loc
    if branch == "early":
        return canonical_fuse(outputs[24]["T"][:, 1:, :], outputs[12]["V"][:, 1:, :], outputs[18]["V"][:, 1:, :], alpha=0.5)
    raise ValueError(branch)


@torch.no_grad()
def build_branch_memory(model, loader, cfg, branch, seed, device):
    image_keys, image_labels, patch_keys, patch_labels, entries = [], [], [], [], []
    k = int(cfg.get("k", 4))
    region_id = memory_id = 0
    start = time.perf_counter()
    for items in tqdm(loader, desc=f"build {branch} memory seed {seed}"):
        image = items["img"].to(device)
        gt = (items["img_mask"].to(device) > 0.5).float()
        outputs = extract_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        patches = branch_features(outputs, branch)
        image_keys.append(cls.detach().cpu())
        image_labels.append(torch.as_tensor(items["anomaly"]).detach().cpu())
        side = int(math.sqrt(patches.shape[1]))
        gt_resized = resize_mask(gt, side)
        sid = sample_id(items)
        for b in range(image.shape[0]):
            for mask, region_type, label in ((gt_resized[b], "anomaly", 1), (~gt_resized[b], "normal", 0)):
                tokens = patches[b][mask.reshape(-1)]
                if tokens.numel() == 0:
                    continue
                selected = STRATEGIES["FPS_K4"](tokens, k, seed * 1_000_003 + region_id).detach().cpu()
                selected = F.normalize(selected, dim=-1)
                patch_keys.append(selected)
                patch_labels.append(torch.full((selected.shape[0],), label, dtype=torch.long))
                for local_j in range(selected.shape[0]):
                    entries.append({"memory_id": memory_id, "source_id": sid, "region_id": region_id, "region_type": region_type, "label": label, "local_j": local_j})
                    memory_id += 1
                region_id += 1
    return {
        "image_keys": torch.cat(image_keys, dim=0),
        "image_values": one_hot(torch.cat(image_labels, dim=0)),
        "patch_keys": torch.cat(patch_keys, dim=0),
        "patch_values": one_hot(torch.cat(patch_labels, dim=0)),
        "entries": entries,
        "branch": branch,
        "build_time": time.perf_counter() - start,
    }


def score_standardize(scores, stats, method):
    if method == "none":
        return scores
    if method == "zscore":
        return (scores - stats["mean"]) / max(stats["std"], 1e-6)
    if method == "robust":
        return (scores - stats["median"]) / max(stats["mad"], 1e-6)
    if method == "temperature":
        return scores / max(stats["temperature"], 1e-6)
    raise ValueError(method)

