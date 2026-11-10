import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_core_specialist_complementarity_oracle import (
    VAR_FB,
    VAR_VIS,
    load_e0_maps,
    pair_slug,
    resize_to_mask,
)
from experiments.phase_defect_query_transfer_pilot import light_fold, make_mvec_target
from experiments.phase_liv_memory_support_modes import external_cell
from experiments.phase_median_spatial_consensus_generalization import HOLDOUTS, MetaDataset
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import fnum, item_at, item_meta, mask_np, summarize_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score

OUT = ROOT / "results" / "phase_mva_mrad_same_protocol_visa_factorization" / "formal"
MAP_CACHE = OUT / "MAP_CACHE"

PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
)
VISA_FACTOR_PAIR = ("Full MVTec", "VisA")

E0 = "E0_Core_NTR_FixedLRP"
M1 = "M1_MRAD_TF"
M2 = "M2_MRAD_FT"
M3 = "M3_MRAD_CLIP"
MRAD_VARIANTS = (M1, M2, M3)
ALL_VARIANTS = (E0, M1, M2, M3)
LOCKED_LRP_METRICS = ROOT / "results" / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "FULL_RES_METRICS.csv"
LOCKED_LRP_PERCAT = ROOT / "results" / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "PER_CATEGORY_FULL_RES.csv"
LOCKED_LRP_VARIANT = "E2_GlobalLocalZoomPyramid"

OFFICIAL_DEFAULTS = {
    "backbone": "ViT-L/14@336px",
    "image_size": 518,
    "features_list": [6, 12, 18, 24],
    "feature_map_layer": [0, 1, 2, 3],
    "DPAM_layer": 24,
    "sigma": 4,
    "k": 0.7,
    "n_ctx": 12,
    "depth": 9,
    "t_n_ctx": 4,
}


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_config(source):
    if source == "Full MVTec":
        return {
            "source": source,
            "cache_name": "mvtec",
            "checkpoint": ROOT / "checkpoints" / "test_on_visa.pth",
            "checkpoint_note": "official checkpoint named test_on_visa.pth, used with MVTec source cache for strict MVTec-source evaluation",
        }
    if source == "Full VisA":
        return {
            "source": source,
            "cache_name": "visa",
            "checkpoint": ROOT / "checkpoints" / "test_on_mvtec.pth",
            "checkpoint_note": "official checkpoint named test_on_mvtec.pth, used with VisA source cache for strict VisA-source evaluation",
        }
    raise KeyError(source)


def metas_from_cell(target, dataset, refs, kind, smoke):
    n = min(24, len(refs)) if smoke else len(refs)
    metas = []
    for pos, ref in enumerate(refs[:n]):
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        metas.append({"category": cat, "label": label, "img_path": path, "mask": mask_np(item)})
    return metas


def load_metas_direct(source, target, device, smoke=False):
    del device
    if target == "BTAD":
        _cell_id, t, dataset, rows, _patch, kind = external_cell("BTAD", smoke)
        return metas_from_cell(t, dataset, rows, kind, smoke)
    if target == "MPDD":
        root, _domain, _kind = HOLDOUTS["MPDD"]
        dataset = MetaDataset(root, mode="test")
        refs = list(range(len(dataset)))
        return metas_from_cell("MPDD", dataset, refs, "source_dataset", smoke)
    if source == "Full MVTec" and target == "VisA":
        _dataset_name, dataset, train_indices, held_indices = light_fold("D2", 0)
        refs = list(train_indices) + list(held_indices)
        return metas_from_cell("VisA", dataset, refs, "source_dataset", smoke)
    if source == "Full VisA" and target == "MVTec":
        dataset, indices, _patch, _vocab = make_mvec_target(smoke)
        return metas_from_cell("MVTec", dataset, indices, "source_dataset", smoke)
    raise KeyError(f"unsupported MVA pair {source}->{target}")


def official_args(device):
    return SimpleNamespace(
        image_size=OFFICIAL_DEFAULTS["image_size"],
        features_list=list(OFFICIAL_DEFAULTS["features_list"]),
        depth=OFFICIAL_DEFAULTS["depth"],
        n_ctx=OFFICIAL_DEFAULTS["n_ctx"],
        t_n_ctx=OFFICIAL_DEFAULTS["t_n_ctx"],
        feature_map_layer=list(OFFICIAL_DEFAULTS["feature_map_layer"]),
        sigma=OFFICIAL_DEFAULTS["sigma"],
        k=OFFICIAL_DEFAULTS["k"],
        device=device,
    )


def locked_e0_rows(source, target):
    raw = []
    percat = []
    if LOCKED_LRP_METRICS.exists():
        for r in read_csv(LOCKED_LRP_METRICS):
            if r.get("source") == source and r.get("dataset") == target and r.get("variant") == LOCKED_LRP_VARIANT:
                row = dict(r)
                row["variant"] = E0
                row["aggregation"] = "CATEGORY_MACRO"
                raw.append(row)
    if LOCKED_LRP_PERCAT.exists():
        for r in read_csv(LOCKED_LRP_PERCAT):
            if r.get("source") == source and r.get("dataset") == target and r.get("variant") == LOCKED_LRP_VARIANT:
                row = dict(r)
                row["variant"] = E0
                row.setdefault("protocol", "M_FR")
                percat.append(row)
    return (raw[0] if raw else None), percat


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_official_modules():
    import AnomalyCLIP_lib
    from AnomalyCLIP_lib import compute_similarity, get_similarity_map
    from models.mlp import AnomalyMLP, MLP, Projector, average_neighbor
    from models.prompt_learner import AnomalyCLIP_PromptLearner
    from mrad import build_cache_model, build_patch_cache_model, compute_patch_socre, compute_socre
    from utils.transforms import get_transform

    return {
        "AnomalyCLIP_lib": AnomalyCLIP_lib,
        "compute_similarity": compute_similarity,
        "get_similarity_map": get_similarity_map,
        "AnomalyMLP": AnomalyMLP,
        "MLP": MLP,
        "Projector": Projector,
        "average_neighbor": average_neighbor,
        "AnomalyCLIP_PromptLearner": AnomalyCLIP_PromptLearner,
        "build_cache_model": build_cache_model,
        "build_patch_cache_model": build_patch_cache_model,
        "compute_patch_socre": compute_patch_socre,
        "compute_socre": compute_socre,
        "get_transform": get_transform,
    }


def freeze_module(m):
    if m is None:
        return None
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_mrad_bundle(source, variant, device):
    official = import_official_modules()
    args = official_args(device)
    cfg = source_config(source)
    params = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    print(f"[MVA] loading official MRAD {source} {variant}", flush=True)
    model, _ = official["AnomalyCLIP_lib"].load(
        OFFICIAL_DEFAULTS["backbone"], device=device, design_details=params
    )
    model.eval()

    prompt_learner = official["AnomalyCLIP_PromptLearner"](model.to("cpu"), params)
    image_proj = official["MLP"]()
    patch_proj = official["Projector"](1024, 768, length=2)
    prompt_proj = official["AnomalyMLP"]()

    checkpoint = None
    if variant in (M2, M3):
        checkpoint = torch.load(cfg["checkpoint"], map_location=device)
        image_proj.load_state_dict(checkpoint["image_proj"])
        patch_proj.load_state_dict(checkpoint["patch_proj"])
        image_proj.to(device)
        patch_proj.to(device)
    if variant == M3:
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        prompt_proj.load_state_dict(checkpoint["prompt_proj"])
        prompt_learner.to(device)
        prompt_proj.to(device)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=OFFICIAL_DEFAULTS["DPAM_layer"])

    cache_key, cache_value = official["build_cache_model"](
        load_cache=True,
        clip_model=model,
        train_loader_cache=None,
        device=device,
        dir=str(ROOT / "cache" / f"cache_model_{cfg['cache_name']}.pt"),
    )
    cache_keys_patch, cache_values_patch = official["build_patch_cache_model"](
        load_cache=True,
        clip_model=model,
        train_loader_cache=None,
        device=device,
        dir=str(ROOT / "cache" / f"cache_patch_model_{cfg['cache_name']}.pt"),
    )
    preprocess, _target_transform = official["get_transform"](args)

    modules = {
        "model": freeze_module(model),
        "prompt_learner": freeze_module(prompt_learner),
        "image_proj": freeze_module(image_proj.to(device)),
        "patch_proj": freeze_module(patch_proj.to(device)),
        "prompt_proj": freeze_module(prompt_proj.to(device)),
        "cache_key": cache_key,
        "cache_value": cache_value,
        "cache_keys_patch": cache_keys_patch,
        "cache_values_patch": cache_values_patch,
        "preprocess": preprocess,
        "args": args,
        "official": official,
        "source_cfg": cfg,
        "model_type": {M1: "mrad-tf", M2: "mrad-ft", M3: "mrad-clip"}[variant],
        "variant": variant,
    }
    return modules


def norm_torch(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def mrad_infer_one(bundle, img_path, device):
    official = bundle["official"]
    args = bundle["args"]
    model_type = bundle["model_type"]
    image = bundle["preprocess"](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    model = bundle["model"]

    image_features, patch_features, _all_cls_tokens, patch_projections = model.encode_image(
        image, args.features_list, DPAM_layer=OFFICIAL_DEFAULTS["DPAM_layer"]
    )

    patch_projection = official["average_neighbor"](patch_projections[3])
    patch_projection = norm_torch(patch_projection)
    patch_f_bia = official["average_neighbor"](patch_features[3])
    patch_f_bia = norm_torch(patch_f_bia)
    raw_patch_feature_np = patch_f_bia.detach().cpu().numpy()[0].astype(np.float32)

    use_proj = model_type != "mrad-tf"
    seg_logit, prompt_patch_features, ori_weight, ft_weight = official["compute_patch_socre"](
        patch_f_bia,
        bundle["cache_keys_patch"],
        bundle["cache_values_patch"],
        device=device,
        proj=bundle["patch_proj"] if use_proj else None,
        need_mask=False,
        patch_projection=patch_projection,
        gt_mask=None,
        is_mradft=(model_type != "mrad-clip"),
        use_proj=use_proj,
    )
    seg_similarity_map = official["get_similarity_map"](seg_logit, args.image_size)

    text_features = None
    if model_type == "mrad-clip":
        bias = bundle["prompt_proj"](prompt_patch_features[:, 0, :], prompt_patch_features[:, 1, :])
        prompts, tokenized_prompts, compound_prompts_text = bundle["prompt_learner"](cls_id=None, bias=bias)
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = norm_torch(text_features)

    image_features = norm_torch(image_features)
    cache_logits, _ = official["compute_socre"](
        image_features,
        bundle["cache_key"],
        bundle["cache_value"],
        device,
        proj=bundle["image_proj"] if use_proj else None,
        use_proj=use_proj,
    )
    image_prob = cache_logits[:, 1]

    anomaly_map_list = []
    for idx, patch_feature in enumerate(patch_projections):
        if idx >= args.feature_map_layer[3]:
            patch_feature = official["average_neighbor"](patch_feature)
            patch_feature = norm_torch(patch_feature)
            if model_type == "mrad-clip":
                similarity = official["compute_similarity"](patch_feature, text_features)
                similarity_map = official["get_similarity_map"](similarity, args.image_size)
                anomaly_map = (similarity_map[..., 1] + 1.0 - similarity_map[..., 0]) / 2.0
            else:
                anomaly_map = seg_similarity_map[..., 1]
            anomaly_map_list.append(anomaly_map)

    anomaly_map = torch.stack(anomaly_map_list).sum(dim=0).detach().cpu().numpy()[0]
    anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma).astype(np.float32)
    topk_values = np.partition(anomaly_map.reshape(-1), -max(1, int(anomaly_map.size * 0.01)))[-max(1, int(anomaly_map.size * 0.01)) :]
    image_score = (1.0 - args.k) * float(np.mean(topk_values)) + args.k * float(image_prob.detach().cpu().numpy()[0])

    retrieval = retrieval_stats(ori_weight, ft_weight, bundle["cache_values_patch"])
    del image, image_features, patch_features, patch_projections, patch_projection, patch_f_bia
    return anomaly_map, image_score, raw_patch_feature_np, retrieval


def retrieval_stats(ori_weight, ft_weight, cache_values_patch):
    row = {}
    for name, logits in (("ori", ori_weight), ("ft", ft_weight)):
        if logits is None or not torch.is_tensor(logits):
            continue
        probs = torch.softmax(logits.float(), dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        top = probs.argmax(dim=-1)
        labels = cache_values_patch.argmax(dim=-1)
        top_labels = labels[top.reshape(-1)].reshape(top.shape)
        row[f"{name}_retrieval_entropy_mean"] = float(ent.mean().detach().cpu())
        row[f"{name}_retrieval_top1_anomaly_fraction"] = float((top_labels == 1).float().mean().detach().cpu())
        row[f"{name}_retrieval_effective_neighbors"] = float(torch.exp(ent).mean().detach().cpu())
    return row


def mrad_cache_path(source, target, variant):
    return MAP_CACHE / f"{pair_slug(source, target)}__{variant}.npz"


def save_mrad_cache(source, target, variant, maps, image_scores, feature_means, retrieval_rows):
    path = mrad_cache_path(source, target, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        maps=np.asarray(maps, dtype=np.float16),
        image_scores=np.asarray(image_scores, dtype=np.float32),
        feature_means=np.asarray(feature_means, dtype=np.float32),
        retrieval_rows=np.asarray(retrieval_rows, dtype=object),
    )


def load_cached_mrad(source, target, variant):
    path = mrad_cache_path(source, target, variant)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "maps": np.asarray(data["maps"], dtype=np.float32),
        "image_scores": np.asarray(data["image_scores"], dtype=np.float32),
        "feature_means": np.asarray(data["feature_means"], dtype=np.float32),
        "retrieval_rows": list(data["retrieval_rows"]) if "retrieval_rows" in data.files else [],
    }


def build_mrad_maps(source, target, variant, metas, device, smoke=False, force=False):
    cached = None if force or smoke else load_cached_mrad(source, target, variant)
    if cached is not None and cached["maps"].shape[0] >= len(metas):
        return cached

    bundle = load_mrad_bundle(source, variant, device)
    n = min(12, len(metas)) if smoke else len(metas)
    maps, image_scores, feature_means, retrieval_rows = [], [], [], []
    start = time.time()
    for i, meta in enumerate(metas[:n]):
        if i % 50 == 0:
            elapsed = time.time() - start
            rate = elapsed / max(i, 1)
            eta = rate * (n - i)
            print(f"[MVA:{variant}] {source}->{target} {i}/{n}, eta {eta/60:.1f} min", flush=True)
        amap, img_score, proj, retr = mrad_infer_one(bundle, meta["img_path"], device)
        maps.append(amap)
        image_scores.append(img_score)
        feature_means.append(proj.mean(axis=0))
        rr = {"source": source, "dataset": target, "variant": variant, "image_index": i, "category": meta["category"], **retr}
        retrieval_rows.append(rr)
    if not smoke:
        save_mrad_cache(source, target, variant, maps, image_scores, feature_means, retrieval_rows)
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "maps": np.asarray(maps, dtype=np.float32),
        "image_scores": np.asarray(image_scores, dtype=np.float32),
        "feature_means": np.asarray(feature_means, dtype=np.float32),
        "retrieval_rows": retrieval_rows,
    }


def e0_bundle(source, target, metas):
    maps = load_e0_maps(source, target, metas)
    image_scores = np.asarray([top1_image_score(m) for m in maps], dtype=np.float32)
    return {"maps": maps.astype(np.float32), "image_scores": image_scores}


def evaluate_variant_maps(source, target, variant, maps, image_scores, metas):
    by_cat = {}
    n = min(len(metas), len(maps), len(image_scores))
    for i, meta in enumerate(metas[:n]):
        d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "scores": []})
        d["labels"].append(int(meta["label"]))
        d["image_scores"].append(float(image_scores[i]))
        d["masks"].append(np.asarray(meta["mask"], dtype=np.uint8))
        d["scores"].append(np.asarray(maps[i], dtype=np.float32))
    cat_rows = []
    for cat, vals in sorted(by_cat.items()):
        resized = [resize_to_mask(s, m, align_corners=True) for s, m in zip(vals["scores"], vals["masks"])]
        met = summarize_category(vals["labels"], vals["image_scores"], vals["masks"], resized)
        row = {
            "source": source,
            "dataset": target,
            "category": cat,
            "variant": variant,
            "protocol": "M_FR",
            "score_resolution": f"{vals['scores'][0].shape[0]}x{vals['scores'][0].shape[1]}",
        }
        for k, v in met.items():
            row[f"M_FR_{k.replace('-', '_')}"] = v
        cat_rows.append(row)
    raw = {
        "source": source,
        "dataset": target,
        "variant": variant,
        "aggregation": "CATEGORY_MACRO",
        "score_resolution": f"{maps[0].shape[0]}x{maps[0].shape[1]}",
    }
    for met in ("I_AUROC", "I_AP", "I_F1max", "P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
        vals = np.asarray([fnum(r, f"M_FR_{met}") for r in cat_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        raw[f"M_FR_{met}"] = float(vals.mean()) if vals.size else float("nan")
    return raw, cat_rows


def official_audit_rows():
    rows = []
    for source in ("Full MVTec", "Full VisA"):
        cfg = source_config(source)
        for variant, model_type in ((M1, "mrad-tf"), (M2, "mrad-ft"), (M3, "mrad-clip")):
            rows.append(
                {
                    "source": source,
                    "variant": variant,
                    "model_type": model_type,
                    "official_backbone": OFFICIAL_DEFAULTS["backbone"],
                    "input_resolution": OFFICIAL_DEFAULTS["image_size"],
                    "features_list": "6,12,18,24",
                    "score_layer_rule": "official feature_map_layer[3]=3, so final 24-layer map is used",
                    "memory_cache": f"cache_model_{cfg['cache_name']}.pt + cache_patch_model_{cfg['cache_name']}.pt",
                    "checkpoint": str(cfg["checkpoint"]) if variant in (M2, M3) else "none",
                    "checkpoint_sha256": sha256_file(cfg["checkpoint"]) if variant in (M2, M3) and cfg["checkpoint"].exists() else "",
                    "uses_patch_projection": variant in (M2, M3),
                    "uses_image_projection": variant in (M2, M3),
                    "uses_prompt_branch": variant == M3,
                    "prompt": "class-agnostic prompt learner with dynamic bias from patch normal/anomaly tokens" if variant == M3 else "none",
                    "gaussian_smoothing_sigma": OFFICIAL_DEFAULTS["sigma"],
                    "image_score_fusion_k": OFFICIAL_DEFAULTS["k"],
                    "strict_protocol_change": "explicit source cache selected by source; no target cache/statistics/category names",
                }
            )
    return rows


def otsu_threshold(gray):
    hist, edges = np.histogram(gray.reshape(-1), bins=128, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.5
    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1))[::-1]
    var_between = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[1:]) ** 2
    if var_between.size == 0:
        return 0.5
    return float(centers[:-1][np.argmax(var_between)])


def image_complexity(img_path):
    arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    edge = np.sqrt(gx * gx + gy * gy)
    edge_density = float(np.mean(edge > np.percentile(edge, 85)))
    entropy_hist, _ = np.histogram(gray.reshape(-1), bins=64, range=(0, 1), density=True)
    p = entropy_hist / max(float(entropy_hist.sum()), 1e-12)
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum() / math.log(64))
    thr = otsu_threshold(gray)
    fg1 = gray > thr
    fg2 = gray <= thr
    border = np.concatenate([fg1[0], fg1[-1], fg1[:, 0], fg1[:, -1]])
    foreground = fg1 if border.mean() < 0.5 else fg2
    foreground = ndimage.binary_opening(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    labels, n = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    return {
        "edge_density_proxy": edge_density,
        "gray_entropy_proxy": entropy,
        "foreground_area_ratio_proxy": float(foreground.mean()),
        "visible_object_instances_proxy": int(n),
    }


def mask_morphology(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {
            "anomaly_pixel_ratio": 0.0,
            "component_count": 0,
            "mean_component_area": 0.0,
            "median_component_area": 0.0,
            "boundary_to_area_ratio": 0.0,
            "thinness": 0.0,
        }
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.asarray([(labels == i).sum() for i in range(1, n + 1)], dtype=np.float64)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=np.uint8))
    boundary = mask & (~eroded)
    return {
        "anomaly_pixel_ratio": float(mask.mean()),
        "component_count": int(n),
        "mean_component_area": float(areas.mean()) if areas.size else 0.0,
        "median_component_area": float(np.median(areas)) if areas.size else 0.0,
        "boundary_to_area_ratio": float(boundary.sum() / max(mask.sum(), 1)),
        "thinness": float(boundary.sum() / max(mask.sum(), 1)),
    }


def pixel_arrays_for_category(maps, metas, category):
    ys, ss = [], []
    for i, m in enumerate(metas):
        if m["category"] != category:
            continue
        mask = np.asarray(m["mask"], dtype=np.uint8)
        score = resize_to_mask(maps[i], mask, align_corners=True)
        ys.append(mask.reshape(-1))
        ss.append(score.reshape(-1))
    if not ys:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float32)
    return np.concatenate(ys).astype(np.uint8), np.concatenate(ss).astype(np.float32)


def safe_auc(y, s):
    from experiments.standard_zsad_metrics import safe_auc as _safe_auc

    return _safe_auc(y, s)


def safe_ap(y, s):
    from experiments.standard_zsad_metrics import safe_ap as _safe_ap

    return _safe_ap(y, s)


def f1_threshold(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    if y.size == 0 or y.max() == y.min():
        return float(np.nanpercentile(s, 95))
    vals = np.unique(np.quantile(s, np.linspace(0, 1, 256)))
    best_t, best_f = float(vals[-1]), -1.0
    for t in vals:
        pred = s >= t
        tp = float(np.sum(pred & (y > 0)))
        fp = float(np.sum(pred & (y == 0)))
        fn = float(np.sum((~pred) & (y > 0)))
        f = 2 * tp / max(2 * tp + fp + fn, 1e-12)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def source_patch_centroid(source):
    cfg = source_config(source)
    data = torch.load(ROOT / "cache" / f"cache_patch_model_{cfg['cache_name']}.pt", map_location="cpu")
    keys = np.asarray(data["keys"], dtype=np.float32)
    vals = np.asarray(data["values"].argmax(dim=-1), dtype=np.int64)
    all_centroid = keys.mean(axis=0)
    normal = keys[vals == 0].mean(axis=0) if np.any(vals == 0) else all_centroid
    anomaly = keys[vals == 1].mean(axis=0) if np.any(vals == 1) else all_centroid
    def unit(x):
        return x / max(float(np.linalg.norm(x)), 1e-12)
    return unit(all_centroid), unit(normal), unit(anomaly)


def cosine_np(a, b):
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def load_external_specialist_maps(source, target, specialist_variant):
    path = ROOT / "results" / "phase_core_specialist_complementarity_oracle" / "formal" / "MAP_CACHE" / f"{pair_slug(source, target)}__{specialist_variant}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["maps"], dtype=np.float32)


def visa_category_audit(maps_by_variant, metas, mrad_feature_means):
    source, target = VISA_FACTOR_PAIR
    source_all, source_normal, source_anom = source_patch_centroid(source)
    cats = sorted({m["category"] for m in metas})
    rows = []
    for cat in cats:
        cat_metas = [m for m in metas if m["category"] == cat]
        morph_rows = [mask_morphology(m["mask"]) for m in cat_metas if np.asarray(m["mask"]).sum() > 0]
        comp_rows = [image_complexity(m["img_path"]) for m in cat_metas[: min(20, len(cat_metas))]]
        row = {"source": source, "dataset": target, "category": cat}
        for key in (
            "anomaly_pixel_ratio",
            "component_count",
            "mean_component_area",
            "median_component_area",
            "boundary_to_area_ratio",
            "thinness",
        ):
            vals = np.asarray([r[key] for r in morph_rows], dtype=np.float64)
            row[f"{key}_mean"] = float(vals.mean()) if vals.size else 0.0
        for key in ("edge_density_proxy", "gray_entropy_proxy", "foreground_area_ratio_proxy", "visible_object_instances_proxy"):
            vals = np.asarray([r[key] for r in comp_rows], dtype=np.float64)
            row[f"{key}_mean"] = float(vals.mean()) if vals.size else float("nan")
        if row.get("foreground_area_ratio_proxy_mean", 0) and row["foreground_area_ratio_proxy_mean"] > 0:
            row["anomaly_to_foreground_area_ratio_proxy"] = row["anomaly_pixel_ratio_mean"] / row["foreground_area_ratio_proxy_mean"]
        else:
            row["anomaly_to_foreground_area_ratio_proxy"] = float("nan")

        paps = {}
        for variant, maps in maps_by_variant.items():
            y, s = pixel_arrays_for_category(maps, metas, cat)
            row[f"{variant}_P_AUROC"] = safe_auc(y, s)
            row[f"{variant}_P_AP"] = safe_ap(y, s)
            paps[variant] = row[f"{variant}_P_AP"]
        if E0 in paps:
            for variant in maps_by_variant:
                if variant != E0:
                    row[f"Delta_P_AP_{variant}_minus_E0"] = paps[variant] - paps[E0]

        if E0 in maps_by_variant:
            y, s = pixel_arrays_for_category(maps_by_variant[E0], metas, cat)
            thr = f1_threshold(y, s)
            fn = (y > 0) & (s < thr)
            row["E0_FN_ratio_at_category_f1_threshold"] = float(fn.sum() / max(np.sum(y > 0), 1))

        feats = []
        for i, m in enumerate(metas):
            if m["category"] == cat and i < len(mrad_feature_means):
                feats.append(mrad_feature_means[i])
        if feats:
            feat = np.asarray(feats, dtype=np.float32).mean(axis=0)
            feat = feat / max(float(np.linalg.norm(feat)), 1e-12)
            row["clip_shift_distance_to_source_all_centroid"] = 1.0 - cosine_np(feat, source_all)
            row["clip_shift_distance_to_source_normal_centroid"] = 1.0 - cosine_np(feat, source_normal)
            row["clip_shift_distance_to_source_anomaly_centroid"] = 1.0 - cosine_np(feat, source_anom)
        rows.append(row)
    return rows


def corr_rows(category_rows):
    factors = [
        "anomaly_pixel_ratio_mean",
        "component_count_mean",
        "mean_component_area_mean",
        "median_component_area_mean",
        "boundary_to_area_ratio_mean",
        "thinness_mean",
        "edge_density_proxy_mean",
        "gray_entropy_proxy_mean",
        "foreground_area_ratio_proxy_mean",
        "visible_object_instances_proxy_mean",
        "anomaly_to_foreground_area_ratio_proxy",
        "E0_FN_ratio_at_category_f1_threshold",
        "clip_shift_distance_to_source_all_centroid",
        "clip_shift_distance_to_source_normal_centroid",
    ]
    metrics = ["E0_Core_NTR_FixedLRP_P_AP"]
    for k in category_rows[0].keys() if category_rows else []:
        if k.startswith("Delta_P_AP_"):
            metrics.append(k)
    rows = []
    for metric in metrics:
        y = np.asarray([fnum(r, metric) for r in category_rows], dtype=np.float64)
        for factor in factors:
            x = np.asarray([fnum(r, factor) for r in category_rows], dtype=np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3 or np.nanstd(x[ok]) < 1e-12 or np.nanstd(y[ok]) < 1e-12:
                pr = sp = float("nan")
            else:
                pr = float(pearsonr(x[ok], y[ok]).statistic)
                sp = float(spearmanr(x[ok], y[ok]).statistic)
            rows.append({"source": "Full MVTec", "dataset": "VisA", "metric": metric, "factor": factor, "pearson": pr, "spearman": sp, "n_categories": int(ok.sum())})
    return rows


def write_official_audit(rows):
    lines = [
        "# Official MRAD Implementation Audit",
        "",
        "Local official code was inspected in `README.md`, `run.sh`, `train.py`, `test.py`, `mrad.py`, and `models/mlp.py`.",
        "",
        "Key implementation facts:",
        "",
        "- Backbone: official `AnomalyCLIP_lib.load(\"ViT-L/14@336px\")` with `DPAM_layer=24`.",
        "- Input resolution: 518x518, official bicubic resize plus center crop.",
        "- Feature layers requested: `[6, 12, 18, 24]`; official test-time `feature_map_layer=[0,1,2,3]` means only the final layer index 3 contributes to the anomaly map.",
        "- MRAD-TF uses raw CLIP memory retrieval; MRAD-FT uses learned image/patch projection; MRAD-CLIP additionally uses prompt learner and prompt projection.",
        "- Patch score is retrieved from source patch memory with softmax over source memory labels, then resized to 518x518 via official `get_similarity_map`.",
        "- Official inference applies Gaussian smoothing with `sigma=4`.",
        "- The original official `test.py` auto-selects source cache from target name; this audit overrides that with explicit source cache to satisfy strict source-to-unseen-target protocol.",
        "",
        "| Source | Variant | Cache | Checkpoint | Prompt | Sigma |",
        "|---|---|---|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['source']} | {r['variant']} | {r['memory_cache']} | {Path(str(r['checkpoint'])).name if r['checkpoint'] != 'none' else 'none'} | {r['uses_prompt_branch']} | {r['gaussian_smoothing_sigma']} |"
        )
    (OUT / "OFFICIAL_MRAD_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_decision(raw_rows, visa_rows):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw_rows}
    mrad_deltas = []
    for source, target in PAIRS:
        e0 = by.get((source, target, E0))
        if not e0:
            continue
        best = None
        for v in MRAD_VARIANTS:
            r = by.get((source, target, v))
            if r is None:
                continue
            delta = fnum(r, "M_FR_P_AP") - fnum(e0, "M_FR_P_AP")
            if best is None or delta > best:
                best = delta
        if best is not None:
            mrad_deltas.append(best)
    visa_e0 = [fnum(r, f"{E0}_P_AP") for r in visa_rows]
    area = [fnum(r, "anomaly_pixel_ratio_mean") for r in visa_rows]
    thin = [fnum(r, "thinness_mean") for r in visa_rows]
    shift = [fnum(r, "clip_shift_distance_to_source_all_centroid") for r in visa_rows]
    def spcorr(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return float("nan")
        return float(spearmanr(x[ok], y[ok]).statistic)
    corr_area = spcorr(area, visa_e0)
    corr_thin = spcorr(thin, visa_e0)
    corr_shift = spcorr(shift, visa_e0)
    mean_best_mrad = float(np.nanmean(mrad_deltas)) if mrad_deltas else float("nan")

    if np.isfinite(mean_best_mrad) and mean_best_mrad >= 2.0:
        outcome = "MRAD_PRESERVES_COMPOSITIONAL_INFORMATION_MISSING_FROM_TWO_STATE"
    elif np.isfinite(corr_shift) and corr_shift <= -0.4:
        outcome = "VISA_FAILURE_IS_DOMINATED_BY_REPRESENTATION_DOMAIN_SHIFT"
    elif (np.isfinite(corr_area) and corr_area >= 0.35) or (np.isfinite(corr_thin) and corr_thin <= -0.35):
        outcome = "VISA_REQUIRES_INSTANCE_OR_STRUCTURE_AWARE_REPRESENTATION"
    elif mrad_deltas and any(np.isfinite(v) for v in mrad_deltas):
        outcome = "VISA_FAILURE_IS_MULTIFACTORIAL"
    else:
        outcome = "VISA_IS_A_GENERAL_FINE_GRAINED_DOMAIN_SHIFT_BOTTLENECK"
    return outcome, {
        "mean_best_mrad_delta_pap": mean_best_mrad,
        "visa_spearman_E0_PAP_vs_area": corr_area,
        "visa_spearman_E0_PAP_vs_thinness": corr_thin,
        "visa_spearman_E0_PAP_vs_clip_shift": corr_shift,
    }


def write_report(raw_rows, visa_rows, corr, outcome, summary, runtime):
    lines = [
        "# Phase MVA: MRAD Same-Protocol + VisA Failure Factorization Audit",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "This audit did not train or tune a new method. Official MRAD code/checkpoints/caches were evaluated through the current full-resolution evaluator, with source cache explicitly fixed by the requested source protocol.",
        "",
        "## Same-Protocol Metrics",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in raw_rows:
        lines.append(
            f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |"
        )

    lines.extend([
        "",
        "## VisA Category Factorization",
        "",
        "| Category | E0 P-AP | Area | Thinness | FN Ratio | CLIP Shift | Best Specialist Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in sorted(visa_rows, key=lambda x: fnum(x, f"{E0}_P_AP")):
        deltas = [fnum(r, k) for k in r.keys() if k.startswith("Delta_P_AP_")]
        best_delta = np.nanmax(deltas) if deltas else float("nan")
        lines.append(
            f"| {r['category']} | {fnum(r, f'{E0}_P_AP'):.4f} | {fnum(r,'anomaly_pixel_ratio_mean'):.5f} | {fnum(r,'thinness_mean'):.4f} | {fnum(r,'E0_FN_ratio_at_category_f1_threshold'):.4f} | {fnum(r,'clip_shift_distance_to_source_all_centroid'):.4f} | {best_delta:.4f} |"
        )

    top_corr = sorted(
        [r for r in corr if r["metric"] == f"{E0}_P_AP" and np.isfinite(fnum(r, "spearman"))],
        key=lambda x: abs(fnum(x, "spearman")),
        reverse=True,
    )[:8]
    lines.extend(["", "## Correlation Highlights", "", "| Metric | Factor | Pearson | Spearman |", "|---|---|---:|---:|"])
    for r in top_corr:
        lines.append(f"| {r['metric']} | {r['factor']} | {fnum(r,'pearson'):.4f} | {fnum(r,'spearman'):.4f} |")

    lines.extend([
        "",
        "## Decision Signals",
        "",
        f"- Mean best MRAD delta P-AP over evaluated transfers: `{summary['mean_best_mrad_delta_pap']:.4f}`.",
        f"- VisA category Spearman E0 P-AP vs anomaly area: `{summary['visa_spearman_E0_PAP_vs_area']:.4f}`.",
        f"- VisA category Spearman E0 P-AP vs thinness: `{summary['visa_spearman_E0_PAP_vs_thinness']:.4f}`.",
        f"- VisA category Spearman E0 P-AP vs CLIP feature shift: `{summary['visa_spearman_E0_PAP_vs_clip_shift']:.4f}`.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    setup_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    audit = official_audit_rows()
    write_csv(OUT / "OFFICIAL_MRAD_AUDIT.csv", audit)
    write_official_audit(audit)

    raw_rows, percat_rows, retrieval_rows = [], [], []
    metas_cache = {}
    maps_for_visa_factor = {}
    mrad_clip_feature_means = None

    selected_pairs = PAIRS[:2] if args.smoke else PAIRS
    selected_variants = (M1,) if args.smoke else MRAD_VARIANTS

    for source, target in selected_pairs:
        print(f"[MVA] loading metas {source}->{target}", flush=True)
        metas = load_metas_direct(source, target, device, smoke=args.smoke)
        metas_cache[(source, target)] = metas
        e0 = e0_bundle(source, target, metas)
        n = min(len(metas), len(e0["maps"]))
        locked_raw, locked_cat = (None, []) if args.smoke else locked_e0_rows(source, target)
        if locked_raw is not None:
            raw_rows.append(locked_raw)
            percat_rows.extend(locked_cat)
        else:
            raw, cat = evaluate_variant_maps(source, target, E0, e0["maps"][:n], e0["image_scores"][:n], metas[:n])
            raw_rows.append(raw)
            percat_rows.extend(cat)
        if (source, target) == VISA_FACTOR_PAIR:
            maps_for_visa_factor[E0] = e0["maps"][:n]

        for variant in selected_variants:
            data = build_mrad_maps(source, target, variant, metas, device, smoke=args.smoke, force=args.force)
            n = min(len(metas), len(data["maps"]))
            raw, cat = evaluate_variant_maps(source, target, variant, data["maps"][:n], data["image_scores"][:n], metas[:n])
            raw_rows.append(raw)
            percat_rows.extend(cat)
            retrieval_rows.extend(data.get("retrieval_rows", []))
            if (source, target) == VISA_FACTOR_PAIR:
                maps_for_visa_factor[variant] = data["maps"][:n]
                if variant == M3:
                    mrad_clip_feature_means = data["feature_means"][:n]

    write_csv(OUT / "MRAD_SAME_PROTOCOL.csv", raw_rows)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat_rows)
    write_csv(OUT / "MRAD_RETRIEVAL_DIAGNOSTICS.csv", retrieval_rows)

    visa_metas = metas_cache.get(VISA_FACTOR_PAIR)
    if visa_metas is not None:
        vis_maps = load_external_specialist_maps(*VISA_FACTOR_PAIR, VAR_VIS)
        fb_maps = load_external_specialist_maps(*VISA_FACTOR_PAIR, VAR_FB)
        if vis_maps is not None:
            maps_for_visa_factor[VAR_VIS] = vis_maps[: len(visa_metas)]
        if fb_maps is not None:
            maps_for_visa_factor[VAR_FB] = fb_maps[: len(visa_metas)]
        if mrad_clip_feature_means is None:
            cache = load_cached_mrad(*VISA_FACTOR_PAIR, M3)
            if cache is not None:
                mrad_clip_feature_means = cache["feature_means"][: len(visa_metas)]
        if mrad_clip_feature_means is None:
            # Any official projected feature mean is enough for source-target shift; use zeros only as an explicit missing-data sentinel.
            mrad_clip_feature_means = np.zeros((len(visa_metas), 1024), dtype=np.float32)
        factor_n = min([len(visa_metas), len(mrad_clip_feature_means)] + [len(v) for v in maps_for_visa_factor.values()])
        visa_factor_metas = visa_metas[:factor_n]
        visa_factor_maps = {k: v[:factor_n] for k, v in maps_for_visa_factor.items()}
        visa_rows = visa_category_audit(visa_factor_maps, visa_factor_metas, mrad_clip_feature_means[:factor_n])
        corr = corr_rows(visa_rows)
    else:
        visa_rows, corr = [], []

    write_csv(OUT / "VISA_CATEGORY_AUDIT.csv", visa_rows)
    write_csv(OUT / "CORRELATION_AUDIT.csv", corr)

    outcome, summary = summarize_decision(raw_rows, visa_rows)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, **summary, "runtime_sec": time.time() - start})
    write_report(raw_rows, visa_rows, corr, outcome, summary, time.time() - start)
    print(f"[MVA] complete: {outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
