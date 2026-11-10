import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    PROTOCOLS,
    METRICS,
    item_at,
    item_meta,
    mask_np,
    summarize_category,
    torch_resize,
)
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score


OUT = Path("results/phase_visualad_exact_protocol_audit/formal")
OFFICIAL_ROOT = ROOT / "_external" / "VisualAD"
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "MVTec"),
)
FEATURES_LIST = [6, 12, 18, 24]
IMAGE_SIZE = 518
BACKBONE = "ViT-L/14@336px"
SIGMA = 4
VARIANTS = (
    "E1a_TokensOnly",
    "E1b_Tokens_SCA",
    "E1c_FullVisualAD",
)


def parse_pairs(pair_args):
    if not pair_args:
        return PILOT_PAIRS
    pairs = []
    for item in pair_args:
        if "->" not in item:
            raise ValueError(f"pair must be Source->Target, got {item}")
        source, target = item.split("->", 1)
        pairs.append((source.strip(), target.strip()))
    return tuple(pairs)


def parse_variants(variant_args):
    if not variant_args:
        return VARIANTS
    valid = set(VARIANTS)
    out = []
    for item in variant_args:
        if item not in valid:
            raise ValueError(f"unknown VisualAD variant {item}; valid={sorted(valid)}")
        out.append(item)
    return tuple(out)
PRIMARY_PROTOCOL = "M"
BASELINE_VARIANT = "E0_MRAD_Core_NTR_FixedLRP"
OFFICIAL_REPO_URL = "https://github.com/7HHHHH/VisualAD"
OFFICIAL_ARXIV_URL = "https://arxiv.org/abs/2603.07952"


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


def fnum(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def ensure_official_imports():
    if not OFFICIAL_ROOT.exists():
        raise FileNotFoundError(f"Official VisualAD repository is missing: {OFFICIAL_ROOT}")
    official = str(OFFICIAL_ROOT)
    if official not in sys.path:
        sys.path.insert(0, official)
    try:
        import pkg_resources
        import packaging
        if not hasattr(pkg_resources, "packaging"):
            pkg_resources.packaging = packaging
    except Exception:
        pass
    # Official VisualAD imports its own top-level "utils" package. MRAD modules above are
    # already imported, so clear later imports to avoid accidentally binding MRAD utils.
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    import VisualAD_lib  # type: ignore
    from utils.feature_transform import create_feature_transform  # type: ignore
    from utils.spatial_cross_attention import build_layer_adaptive_cross_attention  # type: ignore
    from utils.transforms import get_transform  # type: ignore
    from utils.loss import FocalLoss, BinaryDiceLoss, ContrastiveLoss  # type: ignore
    from utils.scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO  # type: ignore

    return {
        "VisualAD_lib": VisualAD_lib,
        "create_feature_transform": create_feature_transform,
        "build_layer_adaptive_cross_attention": build_layer_adaptive_cross_attention,
        "get_transform": get_transform,
        "FocalLoss": FocalLoss,
        "BinaryDiceLoss": BinaryDiceLoss,
        "ContrastiveLoss": ContrastiveLoss,
        "reduce_anomaly_map": reduce_anomaly_map,
        "DEFAULT_TOPK_RATIO": DEFAULT_TOPK_RATIO,
    }


def checkpoint_path(source):
    folder = "train_on_mvtec" if source == "Full MVTec" else "train_on_visa"
    return OFFICIAL_ROOT / "weight" / folder / "CLIP.pth"


def sha256_tensors(named_tensors):
    h = hashlib.sha256()
    for name, tensor in named_tensors:
        h.update(name.encode("utf-8"))
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def frozen_backbone_checksum(model):
    trainable_names = {
        "visual.anomaly_token",
        "visual.normal_token",
        "visual.anomaly_pos",
        "visual.normal_pos",
        "visual.ln_post.weight",
        "visual.ln_post.bias",
    }
    frozen = [(n, p) for n, p in model.named_parameters() if n not in trainable_names and n.startswith("visual.")]
    return sha256_tensors(frozen)


def load_visualad_bundle(source, variant, official, device):
    ckpt_path = checkpoint_path(source)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Official VisualAD checkpoint missing: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    VisualAD_lib = official["VisualAD_lib"]
    model, _ = VisualAD_lib.load(checkpoint.get("backbone", BACKBONE), device=device)
    model.eval().to(device)

    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)
    if checkpoint.get("ln_post_weight") is not None:
        model.visual.ln_post.load_state_dict({
            "weight": checkpoint["ln_post_weight"].to(device),
            "bias": checkpoint["ln_post_bias"].to(device),
        })

    feature_dim = model.visual.embed_dim
    layer_transforms = nn.ModuleDict()
    if variant == "E1c_FullVisualAD":
        for layer_name, state_dict in checkpoint.get("layer_transforms", {}).items():
            hidden_dim = state_dict["mlp.0.weight"].shape[0]
            transform = official["create_feature_transform"](
                transform_type="mlp",
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0,
            ).to(device)
            transform.load_state_dict({k: v.to(device) for k, v in state_dict.items()}, strict=True)
            transform.eval()
            layer_transforms[layer_name] = transform

    cross_attn = None
    if variant in {"E1b_Tokens_SCA", "E1c_FullVisualAD"} and "cross_attn" in checkpoint:
        config = checkpoint.get("cross_attn_config", {})
        cross_attn = official["build_layer_adaptive_cross_attention"](
            layers=FEATURES_LIST,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            res_scale_init=config.get("res_scale_init", 0.01),
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"], strict=True)
        cross_attn.eval()

    for p in model.parameters():
        p.requires_grad_(False)
    for p in layer_transforms.parameters():
        p.requires_grad_(False)
    if cross_attn is not None:
        for p in cross_attn.parameters():
            p.requires_grad_(False)
    return model, layer_transforms, cross_attn, checkpoint


def official_transform(official):
    args = argparse.Namespace(image_size=IMAGE_SIZE)
    preprocess, _target_transform = official["get_transform"](args)
    return preprocess


def resolve_image_path(item, dataset, ref=None):
    candidates = []
    if isinstance(item, dict):
        p = item.get("img_path")
        if p:
            candidates.append(Path(str(p)))
    if ref is not None and isinstance(ref, dict):
        p = ref.get("img_path")
        if p:
            candidates.append(Path(str(p)))
    if hasattr(dataset, "root"):
        root = Path(getattr(dataset, "root"))
        for c in list(candidates):
            if not c.is_absolute():
                candidates.append(root / c)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot resolve image path from candidates={candidates}")


def generate_anomaly_map(anomaly_features, normal_features, patch_tokens, image_size):
    anomaly_features_norm = F.normalize(anomaly_features, dim=1, eps=1e-8)
    normal_features_norm = F.normalize(normal_features, dim=1, eps=1e-8)
    patch_tokens_norm = F.normalize(patch_tokens, dim=2, eps=1e-8)
    anomaly_sim = torch.cosine_similarity(patch_tokens_norm, anomaly_features_norm.unsqueeze(1), dim=2)
    normal_sim = torch.cosine_similarity(patch_tokens_norm, normal_features_norm.unsqueeze(1), dim=2)
    anomaly_score = anomaly_sim - normal_sim
    side = int(math.sqrt(anomaly_score.shape[1]))
    anomaly_map = anomaly_score.reshape(anomaly_score.shape[0], side, side)
    return F.interpolate(anomaly_map.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(1)


@torch.no_grad()
def sca_attention_stats(cross_attn, anomaly_features, normal_features, patch_features_list, layer_ids):
    if cross_attn is None:
        return []
    rows = []
    for layer_id, patch_tokens in zip(layer_ids, patch_features_list):
        module = cross_attn.layer_attentions[f"layer_{layer_id}"]
        b, n, c = patch_tokens.shape
        patch_with_pos = patch_tokens + module.pos_encoding[:, :n, :]
        k = module.k_norm(module.k_proj(patch_with_pos))
        scale = c ** -0.5
        anchor = module.anchor_queries.unsqueeze(0).expand(b, -1, -1)
        q = module.q_norm(module.q_proj(anchor))
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        flat = attn.reshape(-1, attn.shape[-1])
        entropy = (-(flat * flat.clamp_min(1e-12).log()).sum(dim=-1) / math.log(max(n, 2))).detach().cpu().numpy()
        top1 = flat.max(dim=-1).values.detach().cpu().numpy()
        for token_name in ("anomaly", "normal"):
            token = anomaly_features if token_name == "anomaly" else normal_features
            gate = module.gate(token).detach().cpu().numpy()
            rows.append({
                "layer": layer_id,
                "token": token_name,
                "attention_top1_mean": float(top1.mean()),
                "attention_top1_p95": float(np.percentile(top1, 95)),
                "attention_entropy_mean": float(entropy.mean()),
                "gate_mean": float(gate.mean()),
                "gate_p95": float(np.percentile(gate, 95)),
                "res_scale": float(module.res_scale.detach().cpu().item()),
            })
    return rows


@torch.no_grad()
def infer_batch(model, layer_transforms, cross_attn, images):
    vision_output = model.encode_image(images, FEATURES_LIST)
    anomaly_features = vision_output["anomaly_features"]
    normal_features = vision_output["normal_features"]
    patch_tokens = vision_output["patch_tokens"]
    patch_start_idx = vision_output["patch_start_idx"]
    patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
    if cross_attn is not None:
        adapted = cross_attn(anomaly_features, normal_features, patch_features_list, FEATURES_LIST)
        anomaly_features_list = [a["anomaly"] for a in adapted]
        normal_features_list = [a["normal"] for a in adapted]
    else:
        anomaly_features_list = [anomaly_features] * len(patch_tokens)
        normal_features_list = [normal_features] * len(patch_tokens)

    maps = []
    for idx, patch_feature in enumerate(patch_tokens):
        key = f"layer_{FEATURES_LIST[idx]}"
        if key in layer_transforms:
            b, n, d = patch_feature.shape
            patch_feature = layer_transforms[key](patch_feature.reshape(-1, d)).reshape(b, n, d)
        maps.append(generate_anomaly_map(
            anomaly_features_list[idx],
            normal_features_list[idx],
            patch_feature[:, patch_start_idx:, :],
            IMAGE_SIZE,
        ))
    final = torch.stack(maps, dim=0).sum(dim=0).detach().cpu().numpy().astype(np.float32)
    return final, sca_attention_stats(cross_attn, anomaly_features, normal_features, patch_features_list, FEATURES_LIST)


def find_cell(source, target, device, smoke):
    cells, _audits = protocol_cells(source, device, smoke)
    for cell_id, t, dataset, refs, patch, kind in cells:
        if t == target:
            return cell_id, t, dataset, refs, kind
    raise KeyError(f"No protocol cell for {source} -> {target}")


def evaluate_variant_maps(source, target, variant, maps, metas):
    raw_rows, cat_rows = [], []
    by_cat = {}
    for i, meta in enumerate(metas):
        d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "maps": []})
        d["labels"].append(meta["label"])
        d["image_scores"].append(top1_image_score(maps[i]))
        d["masks"].append(meta["mask"])
        d["maps"].append(np.asarray(maps[i], dtype=np.float32))
    fr_cat_by_protocol = {p: [] for p in PROTOCOLS}
    for cat, vals in sorted(by_cat.items()):
        for pkey, pcfg in PROTOCOLS.items():
            scores = [torch_resize(s, m.shape, align_corners=pcfg["align_corners"]) for s, m in zip(vals["maps"], vals["masks"])]
            fr = summarize_category(vals["labels"], vals["image_scores"], vals["masks"], scores)
            fr_row = {
                "source": source,
                "dataset": target,
                "category": cat,
                "variant": variant,
                "protocol": pkey,
                "protocol_name": pcfg["name"],
                "aggregation": "CATEGORY",
                "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}",
            }
            for met, val in fr.items():
                fr_row[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = val
            cat_rows.append(fr_row)
            fr_cat_by_protocol[pkey].append(fr_row)
    raw = {
        "source": source,
        "dataset": target,
        "variant": variant,
        "aggregation": "CATEGORY_MACRO",
        "score_resolution": f"{maps[0].shape[0]}x{maps[0].shape[1]}" if maps else "",
    }
    for pkey, pcfg in PROTOCOLS.items():
        for met in METRICS:
            vals = np.asarray([fnum(r, f"{pcfg['prefix']}_{met.replace('-', '_')}") for r in fr_cat_by_protocol[pkey]], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            raw[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = float(vals.mean()) if vals.size else float("nan")
    raw_rows.append(raw)
    return raw_rows, cat_rows


def baseline_rows_for_pilot():
    table = ROOT / "results" / "phase_paper_ready_final_validation" / "formal" / "INDUSTRIAL_MAIN_TABLE.csv"
    if not table.exists():
        raise FileNotFoundError(f"Missing frozen E0 baseline table: {table}")
    rows = []
    by = {(r["source"], r["dataset"]): r for r in read_csv(table)}
    for source, target in PILOT_PAIRS:
        r = by[(source, target)]
        row = {
            "source": source,
            "dataset": target,
            "variant": BASELINE_VARIANT,
            "aggregation": "CATEGORY_MACRO",
            "score_resolution": "formal_LRP_74x74",
        }
        for raw_key, out_key in (
            ("M_FR_P_AUROC", "M_FR_P_AUROC"),
            ("M_FR_P_AP", "M_FR_P_AP"),
            ("M_FR_P_F1max", "M_FR_P_F1max"),
            ("M_FR_PRO", "M_FR_PRO"),
            ("M_FR_G_pixel", "M_FR_G_pixel"),
            ("V_FR_P_AUROC", "V_FR_P_AUROC"),
            ("V_FR_P_AP", "V_FR_P_AP"),
            ("V_FR_P_F1max", "V_FR_P_F1max"),
            ("V_FR_PRO", "V_FR_PRO"),
            ("V_FR_G_pixel", "V_FR_G_pixel"),
        ):
            row[out_key] = fnum(r, raw_key)
        rows.append(row)
    return rows


def build_visualad_maps(source, target, variant, official, device, smoke, batch_size):
    model, layer_transforms, cross_attn, checkpoint = load_visualad_bundle(source, variant, official, device)
    preprocess = official_transform(official)
    cell_id, _t, dataset, refs, kind = find_cell(source, target, device, smoke)
    n = min(8, len(refs)) if smoke else len(refs)
    maps, metas, att_rows = [], [], []
    checksum_before = frozen_backbone_checksum(model)
    pending_images, pending_positions = [], []
    for pos, ref in enumerate(refs[:n]):
        if pos % 50 == 0:
            print(f"[VisualADAudit] {source} -> {target} {variant} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        meta_ref = ref if kind != "source_dataset" else None
        cat, label, _path_from_meta = item_meta(item, kind, ref, target)
        path = resolve_image_path(item, dataset, meta_ref)
        pending_images.append(preprocess(Image.open(path).convert("RGB")))
        pending_positions.append(pos)
        metas.append({"category": cat, "label": label, "mask": mask_np(item), "img_path": str(path)})
        if len(pending_images) == batch_size or pos == n - 1:
            batch = torch.stack(pending_images, dim=0).to(device)
            final, stats_rows = infer_batch(model, layer_transforms, cross_attn, batch)
            for bi, image_pos in enumerate(pending_positions):
                filtered = gaussian_filter(final[bi], sigma=SIGMA).astype(np.float32)
                maps.append(filtered)
            for sr in stats_rows:
                sr.update({"source": source, "dataset": target, "variant": variant, "image_index": ",".join(str(x) for x in pending_positions)})
                att_rows.append(sr)
            pending_images, pending_positions = [], []
    checksum_after = frozen_backbone_checksum(model)
    ck_rows = [{
        "source": source,
        "dataset": target,
        "variant": variant,
        "checkpoint": str(checkpoint_path(source)),
        "frozen_backbone_checksum_before": checksum_before,
        "frozen_backbone_checksum_after": checksum_after,
        "unchanged": checksum_before == checksum_after,
        "trainable_parameter_count_from_checkpoint": trainable_count(model, layer_transforms, cross_attn, variant),
        "features_list": ",".join(str(x) for x in checkpoint.get("features_list", FEATURES_LIST)),
        "image_size": checkpoint.get("image_size", IMAGE_SIZE),
    }]
    raw, cats = evaluate_variant_maps(source, target, variant, maps, metas)
    del model, layer_transforms, cross_attn
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return raw, cats, att_rows, ck_rows


def trainable_count(model, layer_transforms, cross_attn, variant):
    total = int(model.visual.anomaly_token.numel() + model.visual.normal_token.numel())
    total += int(model.visual.anomaly_pos.numel() + model.visual.normal_pos.numel())
    total += int(model.visual.ln_post.weight.numel() + model.visual.ln_post.bias.numel())
    if variant == "E1c_FullVisualAD":
        total += sum(p.numel() for p in layer_transforms.parameters())
    if variant in {"E1b_Tokens_SCA", "E1c_FullVisualAD"} and cross_attn is not None:
        total += sum(p.numel() for p in cross_attn.parameters())
    return int(total)


def delta_rows(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in PILOT_PAIRS:
        base = by.get((source, target, BASELINE_VARIANT))
        if base is None:
            continue
        for variant in VARIANTS:
            cur = by.get((source, target, variant))
            if cur is None:
                continue
            row = {"source": source, "dataset": target, "variant": variant}
            for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
                key = f"M_FR_{met}"
                row[f"Delta_{key}"] = fnum(cur, key) - fnum(base, key)
            rows.append(row)
    return rows


def decide(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    base_name = BASELINE_VARIANT
    best_variant, best_mean = None, -1e9
    summaries = []
    for variant in VARIANTS:
        vals = []
        for source, target in PILOT_PAIRS:
            cur = by.get((source, target, variant))
            base = by.get((source, target, base_name))
            if cur and base:
                vals.append(fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"))
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        mean = float(vals.mean()) if vals.size else float("nan")
        summaries.append({"variant": variant, "mean_delta_p_ap": mean, "median_delta_p_ap": float(np.median(vals)) if vals.size else float("nan"), "improved": int(np.sum(vals > 0)), "total": int(vals.size)})
        if np.isfinite(mean) and mean > best_mean:
            best_variant, best_mean = variant, mean
    full = next((s for s in summaries if s["variant"] == "E1c_FullVisualAD"), None)
    sca = next((s for s in summaries if s["variant"] == "E1b_Tokens_SCA"), None)
    tok = next((s for s in summaries if s["variant"] == "E1a_TokensOnly"), None)
    if full and full["mean_delta_p_ap"] >= 1.0 and full["improved"] >= 3:
        outcome = "VISUALAD_INTERACTIVE_STATE_HAS_STRONG_HEADROOM"
    elif best_variant and best_mean > 0.5 and (full is None or full["mean_delta_p_ap"] < 1.0):
        outcome = "VISUALAD_MODULE_HAS_TRANSFERABLE_VALUE"
    else:
        outcome = "VISUALAD_DOES_NOT_OUTPERFORM_CURRENT_CORE"
    details = {"best_variant": best_variant, "best_mean_delta_p_ap": best_mean, "summaries": summaries}
    if full and sca and tok:
        details["module_increment_sca_minus_tokens"] = sca["mean_delta_p_ap"] - tok["mean_delta_p_ap"]
        details["module_increment_saf_minus_sca"] = full["mean_delta_p_ap"] - sca["mean_delta_p_ap"]
    return outcome, details


def write_official_audit(out):
    lines = [
        "# VisualAD Official Implementation Audit",
        "",
        f"Sources: [official repository]({OFFICIAL_REPO_URL}) and [arXiv paper page]({OFFICIAL_ARXIV_URL}).",
        "",
        "## Official CLIP Path Observed In Code",
        "",
        "- Backbone: `ViT-L/14@336px`; README states CLIP uses ViT-L/14@336px.",
        "- Input image size: official training/testing uses `image_size=518` in CLI defaults, while the CLIP checkpoint positional embedding is interpolated when needed.",
        "- Feature layers: `[6, 12, 18, 24]` by default.",
        "- Token insertion: `anomaly_token`, `normal_token`, `class_embedding`, then patch tokens. The two VisualAD tokens are inside the frozen ViT token stream.",
        "- Trainable VisualAD token parameters: anomaly token, normal token, anomaly/normal positional embeddings created by the custom model. Official checkpoints explicitly store anomaly/normal tokens and `ln_post`.",
        "- SCA: `LayerAdaptiveCrossAttention` with independent per-layer `SpatialBottleneckAttention`, `num_anchors=4`, dropout `0.1`, `res_scale_init=0.01`; it enhances anomaly/normal tokens from patch tokens.",
        "- SAF in the released code path is implemented as per-layer MLP feature transforms from `setup_feature_transforms`, one `FeatureTransformMLP` per selected layer, applied to patch features before scoring.",
        "- Pixel score: per layer `cos(patch, anomaly_token) - cos(patch, normal_token)`, bilinear resized to 518, summed over four layers, then Gaussian filtered with `sigma=4` in official `test.py`.",
        "- Training losses in official `train.py`: segmentation focal + dice on sigmoid anomaly maps, image-level BCE over top-1% mean logits, and token contrastive loss. Optimizer is AdamW with token LR 0.001 and module LR 0.0001 under cosine schedule.",
        "- Official `scripts/CLIP.sh` uses MVTec->VisA for 2 epochs and VisA->MVTec for 1 epoch; `train.py` default is 15 epochs. This audit uses released source-specific official checkpoints when present.",
        "",
        "## MRAD Protocol Binding",
        "",
        "- Target data is used only for full-resolution evaluation.",
        "- No target normal reference, target statistics, target category/defect words, target tuning, NTR, LRP, Memory, Text, DINO, or hybrid module is added to VisualAD.",
        "- E1a/E1b/E1c use one official checkpoint per source and differ only by enabling tokens only, tokens+SCA, or tokens+SCA+SAF.",
    ]
    (out / "VISUALAD_OFFICIAL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(out, raw, deltas, checksum_rows, attention_rows, outcome, details, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Phase VisualAD Exact-Protocol Audit",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        f"Runtime: {runtime:.1f}s",
        "",
        "Official sources: [VisualAD GitHub](https://github.com/7HHHHH/VisualAD), [arXiv:2603.07952](https://arxiv.org/abs/2603.07952).",
        "",
        "## Pilot Results",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO | Delta P-AP vs E0 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for source, target in PILOT_PAIRS:
        base = by.get((source, target, BASELINE_VARIANT))
        variants = [BASELINE_VARIANT] + list(VARIANTS)
        for variant in variants:
            r = by.get((source, target, variant))
            if not r:
                continue
            delta = 0.0 if variant == BASELINE_VARIANT else fnum(r, "M_FR_P_AP") - fnum(base, "M_FR_P_AP")
            lines.append(
                f"| {source} | {target} | {variant} | {fnum(r, 'M_FR_P_AUROC'):.4f} | {fnum(r, 'M_FR_P_AP'):.4f} | {fnum(r, 'M_FR_P_F1max'):.4f} | {fnum(r, 'M_FR_PRO'):.4f} | {delta:+.4f} |"
            )
    lines += ["", "## Module Summary", "", "| Variant | Mean Delta P-AP | Median Delta P-AP | Improved / 4 |", "|---|---:|---:|---:|"]
    for s in details.get("summaries", []):
        lines.append(f"| {s['variant']} | {s['mean_delta_p_ap']:+.4f} | {s['median_delta_p_ap']:+.4f} | {s['improved']} / {s['total']} |")
    lines += [
        "",
        "## Implementation Checks",
        "",
        f"- Frozen visual backbone checksums unchanged in all evaluated rows: `{all(str(r.get('unchanged')).lower() == 'true' or r.get('unchanged') is True for r in checksum_rows)}`.",
        f"- SCA attention stats rows: {len(attention_rows)}. Native ViT attention weights are not exposed by the official `need_weights=False` forward path; SCA anchor-attention top1/entropy is recorded instead.",
        "- Official checkpoint files did not include per-epoch training curves, so `TRAINING_LOSS_CURVE.csv` records this as unavailable rather than fabricating curves.",
        "",
        "## Answers",
        "",
        "1. VisualAD is compared here as a source-specific official checkpoint under the same unseen-target evaluator, not combined with NTR/LRP.",
        "2. E1a/E1b/E1c isolate interactive tokens, SCA, and SAF within the released checkpoint path.",
        "3. The decision above is based on full-resolution category-macro P-AP deltas against the frozen MRAD Core+NTR+FixedLRP baseline.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    global PILOT_PAIRS, VARIANTS
    PILOT_PAIRS = parse_pairs(args.pairs)
    VARIANTS = parse_variants(args.variants)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    official = ensure_official_imports()
    write_official_audit(OUT)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    raw = baseline_rows_for_pilot()
    percat, attention_rows, checksum_rows = [], [], []
    for source, target in PILOT_PAIRS:
        for variant in VARIANTS:
            raw_rows, cat_rows, att_rows, ck_rows = build_visualad_maps(source, target, variant, official, device, args.smoke, args.batch_size)
            raw.extend(raw_rows)
            percat.extend(cat_rows)
            attention_rows.extend(att_rows)
            checksum_rows.extend(ck_rows)
            write_csv(OUT / "RAW_METRICS.partial.csv", raw)
            write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", percat)
            write_csv(OUT / "TOKEN_ATTENTION_STATS.partial.csv", attention_rows)
            write_csv(OUT / "FROZEN_BACKBONE_CHECKSUM.partial.csv", checksum_rows)
    deltas = delta_rows(raw)
    outcome, details = decide(raw)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", percat)
    write_csv(OUT / "MODULE_ABLATION.csv", deltas)
    write_csv(OUT / "TOKEN_ATTENTION_STATS.csv", attention_rows)
    write_csv(OUT / "FROZEN_BACKBONE_CHECKSUM.csv", checksum_rows)
    write_csv(OUT / "TRAINING_LOSS_CURVE.csv", [{
        "status": "UNAVAILABLE_IN_OFFICIAL_PRETRAINED_CHECKPOINTS",
        "note": "Official released checkpoints contain final trainable weights, not per-epoch loss curves. This audit did not retrain VisualAD.",
    }])
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, **details, "smoke": args.smoke})
    write_report(OUT, raw, deltas, checksum_rows, attention_rows, outcome, details, time.time() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pairs", nargs="*", default=None)
    p.add_argument("--variants", nargs="*", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
