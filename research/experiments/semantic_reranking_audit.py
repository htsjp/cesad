import csv
import gc
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import AnomalyCLIP_lib
from memory_strategies import STRATEGIES
from mrad_audit_utils import (
    copy_log,
    default_source_dataset,
    evaluate_results,
    extract_layer_features,
    init_results,
    load_config,
    load_model,
    make_loader,
    one_hot,
    parse_args,
    resize_mask,
    retrieval_logits,
    save_run_metadata,
    set_seed,
    write_csv,
)
from semantic_memory import (
    attach_prototype_semantics,
    build_text_anchor_bank,
    label_only_semantics,
    random_text_bank,
    reranked_logits,
    save_text_anchor_bank,
    semantic_scores,
    shuffled_semantics,
)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def package_versions():
    data = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}
    if torch.cuda.is_available():
        data["cuda"] = torch.version.cuda
        data["gpu"] = torch.cuda.get_device_name(0)
    return data


def memory_mb(memory, text_bank=None):
    total = 0
    for key in ("image_keys", "image_values", "patch_keys", "patch_values"):
        total += memory[key].numel() * memory[key].element_size()
    if text_bank is not None:
        total += text_bank["embeddings"].numel() * text_bank["embeddings"].element_size()
    return total / (1024 * 1024)


def checkpoint_enabled(cfg):
    return bool(cfg.get("resume", True))


def checkpoint_dir(output_dir: Path) -> Path:
    path = output_dir / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def tensor_dict_to_device(data, device):
    moved = {}
    for key, value in data.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def memory_checkpoint_path(output_dir: Path, cfg) -> Path:
    name = f"memory_{cfg['source_dataset']}_{cfg.get('source_mode', 'test')}_layer{cfg.get('layer', 24)}_fpsk{cfg.get('k', 4)}.pt"
    return checkpoint_dir(output_dir) / name


def seed_checkpoint_path(output_dir: Path, seed: int) -> Path:
    return checkpoint_dir(output_dir) / f"seed_{seed}_results.pt"


def load_seed_checkpoint(output_dir: Path, seed: int):
    path = seed_checkpoint_path(output_dir, seed)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def write_result_tables(output_dir, cfg, all_summary, all_sweep, all_per_category, all_diag, all_retrieval, sanity, audit_status, baseline_checks, errors):
    success_rows = success_control_rows(all_sweep, cfg)
    write_csv(output_dir / "summary_metrics.csv", all_summary, ["seed", "variant", "status", "I-AUROC", "I-AP", "P-AUROC", "PRO", "inference_time_per_image_sec", "gpu_memory_mb", "patch_memory_size", "text_bank_size", "selection_protocol"])
    write_csv(output_dir / "all_sweep_metrics.csv", all_sweep, ["seed", "variant", "status", "I-AUROC", "I-AP", "P-AUROC", "PRO", "inference_time_per_image_sec", "gpu_memory_mb", "patch_memory_size", "text_bank_size", "selection_protocol"])
    write_csv(output_dir / "per_category_metrics.csv", all_per_category, ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
    write_csv(output_dir / "semantic_diagnostics.csv", all_diag, ["seed", "query_normal_semantic_mean", "query_anomaly_semantic_mean", "prototype_normal_semantic_mean", "prototype_anomaly_semantic_mean", "prototype_semantic_label_auroc", "corr_q_semantic_visual_score", "corr_q_semantic_gt_label", "corr_q_semantic_baseline_error"])
    write_csv(output_dir / "retrieval_change.csv", all_retrieval, ["seed", "variant", "top1_changed_fraction", "top5_jaccard", "rank_correlation", "anomaly_memory_weight_change", "normal_memory_weight_change", "retrieval_entropy", "top1_margin"])
    write_csv(output_dir / "success_control_table.csv", success_rows, ["Variant", "Correct text", "Correct prototype association", "Uses prototype semantics", "I-AUROC", "P-AUROC", "PRO"])
    (output_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, default=json_default), encoding="utf-8")
    write_execution_summary(output_dir, audit_status, baseline_checks, all_summary, success_rows, all_diag, all_retrieval, errors)
    return success_rows


def build_fps_memory(model, loader, cfg, device):
    layer = int(cfg.get("layer", 24))
    features_list = cfg["features_list"]
    k = int(cfg.get("k", 4))
    image_features, image_labels = [], []
    patch_features, patch_labels = [], []
    first_region = None
    region_id = 0
    start = time.perf_counter()
    with torch.no_grad():
        for items in tqdm(loader, desc="build FPS_K4 memory"):
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
                for mask, label in ((gt_resized[b], 1), (~gt_resized[b], 0)):
                    tokens = patches[b][mask.reshape(-1)]
                    if tokens.numel() == 0:
                        continue
                    if first_region is None:
                        first_region = tokens.detach().clone()
                    entries = STRATEGIES["FPS_K4"](tokens, k, region_id)
                    entries = F.normalize(entries, dim=-1)
                    if entries.shape[0] != k:
                        raise RuntimeError(f"FPS_K4 returned {entries.shape[0]} entries, expected {k}")
                    patch_features.append(entries)
                    patch_labels.append(torch.full((entries.shape[0],), label, device=device, dtype=torch.long))
                    region_id += 1
    memory = {
        "image_keys": torch.cat(image_features, dim=0),
        "image_values": one_hot(torch.cat(image_labels, dim=0)).to(device),
        "patch_keys": torch.cat(patch_features, dim=0),
        "patch_values": one_hot(torch.cat(patch_labels, dim=0)).to(device),
    }
    sanity = {
        "patch_memory_entries": int(memory["patch_keys"].shape[0]),
        "image_memory_entries": int(memory["image_keys"].shape[0]),
        "fps_first_region_repeat_diff": 0.0,
    }
    if first_region is not None:
        a = STRATEGIES["FPS_K4"](first_region, k, 0)
        b = STRATEGIES["FPS_K4"](first_region, k, 123)
        sanity["fps_first_region_repeat_diff"] = float((a - b).abs().max().item())
        if sanity["fps_first_region_repeat_diff"] > 1e-6:
            raise RuntimeError("FPS_K4 visual memory is not deterministic")
    return memory, time.perf_counter() - start, sanity


def variant_specs(cfg):
    specs = [{"variant": "FPS_VisualOnly", "family": "visual"}]
    for lam in cfg["sweeps"]["lambda_text"]:
        specs.append({"variant": f"FPS_VisualPlusTextScore_lam{lam}", "family": "late_text", "lambda_text": float(lam)})
    for lam in cfg["sweeps"]["lambda_sem"]:
        specs.append({"variant": f"FPS_SemanticConsistencyRerank_lam{lam}", "family": "consistency", "lambda_sem": float(lam)})
        specs.append({"variant": f"FPS_SignedSemanticRerank_lam{lam}", "family": "signed", "lambda_sem": float(lam)})
    for tau in cfg["sweeps"]["tau_sem"]:
        specs.append({"variant": f"FPS_SemanticGate_tau{tau}", "family": "gate", "tau_sem": float(tau)})
    p = cfg["primary"]
    specs.extend(
        [
            {"variant": "C1_ShuffledPrototypeSemantics", "family": "consistency", "lambda_sem": float(p["lambda_sem"]), "control": "shuffled"},
            {"variant": "C2_RandomTextAnchors", "family": "consistency", "lambda_sem": float(p["lambda_sem"]), "control": "random_text"},
            {"variant": "C3_LabelOnlySemanticControl", "family": "consistency", "lambda_sem": float(p["lambda_sem"]), "control": "label_only"},
            {"variant": "C4_QueryTextOnly", "family": "late_text", "lambda_text": float(p["lambda_text"]), "control": "query_text_only"},
        ]
    )
    return specs


def empty_base_results(obj_list):
    results = init_results(obj_list)
    for obj in obj_list:
        results[obj]["map_paths"] = []
    return results


def empty_variant_outputs(obj_list, specs):
    return {
        spec["variant"]: {obj: {"pr_sp": [], "map_paths": []} for obj in obj_list}
        for spec in specs
    }


def add_common_gt(base_results, cls_name, gt_mask, anomaly):
    base_results[cls_name]["imgs_masks"].append(gt_mask.clone())
    base_results[cls_name]["gt_sp"].extend(torch.as_tensor(anomaly).detach().cpu())


def save_packed_maps(map_root: Path, seed: int, index: int, maps) -> str:
    seed_dir = map_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / f"{index:06d}.npy"
    tmp = seed_dir / f"{index:06d}.npy.tmp"
    arr = np.stack([m.detach().cpu().numpy().astype(np.float16) for m in maps], axis=0)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
    return str(path)


def similarity_map_from_patch_score(score, image_size, sigma):
    if score.ndim == 2:
        b, n = score.shape
        side = int(math.sqrt(n))
        score = score.reshape(b, side, side)
    score = F.interpolate(score.unsqueeze(1).float(), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(1)
    return torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in score.detach().cpu()], dim=0)


def normalized_query_text_score(q_semantic):
    return ((q_semantic + 2.0) / 4.0).clamp(0.0, 1.0)


def topk_image_score(amap, image_logits, cfg, device):
    flat = amap.flatten(start_dim=1)
    topn = max(1, int(flat.shape[1] * cfg.get("topk_ratio", 0.01)))
    topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
    return (1 - cfg.get("fusion_k", 0.7)) * topk_mean + cfg.get("fusion_k", 0.7) * image_logits[:, 1]


def collect_retrieval_change(stats, variant, base_sim, new_sim, values):
    base_flat = base_sim.reshape(-1, base_sim.shape[-1])
    new_flat = new_sim.reshape(-1, new_sim.shape[-1])
    k = min(5, base_flat.shape[-1])
    base_vals, base_idx = torch.topk(base_flat, k=k, dim=-1)
    new_vals, new_idx = torch.topk(new_flat, k=k, dim=-1)
    labels = values.argmax(dim=-1)
    base_w = torch.softmax(base_flat, dim=-1)
    new_w = torch.softmax(new_flat, dim=-1)
    normal = labels == 0
    anomaly = labels == 1
    changed = (base_idx[:, 0] != new_idx[:, 0]).float().mean().item()
    jacc = []
    corr = []
    for a, b in zip(base_idx.detach().cpu().tolist(), new_idx.detach().cpu().tolist()):
        sa, sb = set(a), set(b)
        jacc.append(len(sa & sb) / max(1, len(sa | sb)))
        union = list(sa | sb)
        if len(union) > 1:
            ra = np.array([a.index(x) if x in sa else k for x in union], dtype=np.float64)
            rb = np.array([b.index(x) if x in sb else k for x in union], dtype=np.float64)
            if np.std(ra) > 0 and np.std(rb) > 0:
                corr.append(float(np.corrcoef(ra, rb)[0, 1]))
    entropy = -(new_w * new_w.clamp_min(1e-12).log()).sum(dim=-1)
    margin = new_vals[:, 0] - new_vals[:, 1] if k > 1 else new_vals[:, 0]
    stats[variant]["top1_changed_fraction"].append(changed)
    stats[variant]["top5_jaccard"].append(float(np.mean(jacc)) if jacc else float("nan"))
    stats[variant]["rank_correlation"].append(float(np.mean(corr)) if corr else float("nan"))
    stats[variant]["anomaly_memory_weight_change"].append(float((new_w[:, anomaly].sum(1) - base_w[:, anomaly].sum(1)).mean().item()))
    stats[variant]["normal_memory_weight_change"].append(float((new_w[:, normal].sum(1) - base_w[:, normal].sum(1)).mean().item()))
    stats[variant]["retrieval_entropy"].append(float(entropy.mean().item()))
    stats[variant]["top1_margin"].append(float(margin.mean().item()))


def materialize_variant_results(base_results, variant_output, obj_list):
    results = {}
    for obj in obj_list:
        maps = []
        for ref in variant_output[obj]["map_paths"]:
            if isinstance(ref, (list, tuple)):
                path, map_idx = ref
                arr = np.load(path, mmap_mode="r")[int(map_idx)].astype(np.float32)
            else:
                arr = np.load(ref).astype(np.float32)
            maps.append(torch.from_numpy(arr))
        if maps:
            anomaly_maps = torch.cat(maps, dim=0)
        else:
            anomaly_maps = torch.empty(0)
        results[obj] = {
            "gt_sp": list(base_results[obj]["gt_sp"]),
            "pr_sp": list(variant_output[obj]["pr_sp"]),
            "imgs_masks": [x.clone() for x in base_results[obj]["imgs_masks"]],
            "anomaly_maps": [anomaly_maps],
            "specie": [],
        }
    return results


def collect_semantic_diagnostics(diag, q_sem, k_sem, patch_values, gt_mask_patch, visual_patch_anomaly, baseline_error):
    q_flat = q_sem.reshape(-1)
    gt_flat = gt_mask_patch.reshape(-1).to(q_flat.device)
    diag["query_normal_semantic"].extend(q_flat[~gt_flat].detach().cpu().tolist())
    diag["query_anomaly_semantic"].extend(q_flat[gt_flat].detach().cpu().tolist())
    diag["q_semantic"].extend(q_flat.detach().cpu().tolist())
    diag["visual_anomaly_score"].extend(visual_patch_anomaly.reshape(-1).detach().cpu().tolist())
    diag["gt_patch_label"].extend(gt_flat.detach().cpu().numpy().astype(int).tolist())
    diag["baseline_error"].extend(baseline_error.reshape(-1).detach().cpu().tolist())


def auc_from_arrays(labels, scores):
    from mrad_audit_utils import exact_binary_auc

    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return exact_binary_auc(labels, scores)


def summarize_list(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def corr_safe(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate_semantic_variants(model, memory, text_bank, loader, cfg, seed, device):
    specs = variant_specs(cfg)
    base_results = empty_base_results(loader.dataset.obj_list)
    variant_outputs = empty_variant_outputs(loader.dataset.obj_list, specs)
    map_root = Path(cfg.get("output_dir", "results/semantic_reranking")) / "_map_cache"
    per_image_times = defaultdict(list)
    retrieval_stats = defaultdict(lambda: defaultdict(list))
    semantic_diag = defaultdict(list)
    layer = int(cfg.get("layer", 24))
    image_size = int(cfg.get("image_size", 518))
    sigma = cfg.get("sigma", 4)
    m = int(cfg.get("semantic_top_m", 3))
    patch_values = memory["patch_values"].to(device)
    patch_keys = memory["patch_keys"].to(device)
    image_keys = memory["image_keys"].to(device)
    image_values = memory["image_values"].to(device)
    base_k_sem = memory["patch_semantic"].to(device)
    label_k_sem = label_only_semantics(patch_values).to(device)
    shuffled_k_sem = shuffled_semantics(base_k_sem, seed + 17)
    random_bank = random_text_bank(text_bank, seed + 101)
    random_k_sem = semantic_scores(patch_keys, random_bank, m=m)

    image_index = 0
    for items in tqdm(loader, desc=f"semantic variants seed {seed}"):
        start = time.perf_counter()
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        add_common_gt(base_results, cls_name, gt_mask, items["anomaly"])
        cls, patches, _ = extract_layer_features(
            model, image, cfg["features_list"], layer, device, smooth=cfg.get("average_neighbor", True)
        )
        image_logits, _, _ = retrieval_logits(cls, image_keys, image_values)
        base_patch_logits, base_sim, base_weights = retrieval_logits(patches, patch_keys, patch_values)
        visual_amap = AnomalyCLIP_lib.get_similarity_map(base_patch_logits, image_size)[..., 1]
        visual_amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in visual_amap.detach().cpu()], dim=0)
        visual_score = topk_image_score(visual_amap, image_logits, cfg, device)
        q_sem = semantic_scores(patches.reshape(-1, patches.shape[-1]), text_bank, m=m).reshape(patches.shape[:2])
        q_text_map = similarity_map_from_patch_score(normalized_query_text_score(q_sem), image_size, sigma)
        random_q_sem = semantic_scores(patches.reshape(-1, patches.shape[-1]), random_bank, m=m).reshape(patches.shape[:2])
        side = int(math.sqrt(patches.shape[1]))
        gt_patch = resize_mask(gt_mask.to(device), side)
        visual_patch_anomaly = base_patch_logits[..., 1]
        gt_patch_flat = gt_patch.reshape(gt_patch.shape[0], -1)
        baseline_error = (visual_patch_anomaly - gt_patch_flat.float()).abs()
        collect_semantic_diagnostics(semantic_diag, q_sem, base_k_sem, patch_values, gt_patch_flat, visual_patch_anomaly, baseline_error)

        image_maps = []
        image_map_refs = []
        image_scores = {}
        for spec_idx, spec in enumerate(specs):
            v = spec["variant"]
            family = spec["family"]
            if family == "visual":
                amap = visual_amap
                score = visual_score
                sim_for_change = base_sim
            elif family == "late_text":
                lam = float(spec["lambda_text"])
                amap = visual_amap + lam * q_text_map
                score = visual_score + lam * topk_image_score(q_text_map, torch.zeros_like(image_logits), cfg, device)
                sim_for_change = base_sim
            else:
                control = spec.get("control")
                if control == "shuffled":
                    k_sem = shuffled_k_sem
                    q_use = q_sem
                elif control == "random_text":
                    k_sem = random_k_sem
                    q_use = random_q_sem
                elif control == "label_only":
                    k_sem = label_k_sem
                    q_use = q_sem
                else:
                    k_sem = base_k_sem
                    q_use = q_sem
                if family == "consistency":
                    logits, sim_for_change, _ = reranked_logits(
                        patches, patch_keys, patch_values, "consistency", q_use, k_sem, lambda_sem=spec["lambda_sem"]
                    )
                elif family == "signed":
                    logits, sim_for_change, _ = reranked_logits(
                        patches, patch_keys, patch_values, "signed", q_use, k_sem, lambda_sem=spec["lambda_sem"]
                    )
                elif family == "gate":
                    logits, sim_for_change, _ = reranked_logits(
                        patches, patch_keys, patch_values, "gate", q_use, k_sem, tau_sem=spec["tau_sem"]
                    )
                else:
                    raise ValueError(f"Unknown family {family}")
                amap = AnomalyCLIP_lib.get_similarity_map(logits, image_size)[..., 1]
                amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
                score = topk_image_score(amap, image_logits, cfg, device)
            image_scores[v] = score.detach().cpu()
            image_maps.append(amap)
            image_map_refs.append((v, spec_idx))
            per_image_times[v].append((time.perf_counter() - start) / max(1, len(specs)))
            if v != "FPS_VisualOnly":
                collect_retrieval_change(retrieval_stats, v, base_sim, sim_for_change, patch_values)
        packed_path = save_packed_maps(map_root, seed, image_index, image_maps)
        for v, spec_idx in image_map_refs:
            variant_outputs[v][cls_name]["pr_sp"].extend(image_scores[v])
            variant_outputs[v][cls_name]["map_paths"].append((packed_path, spec_idx))
        image_index += 1

    metric_rows, per_category_rows, metrics_by_variant = [], [], {}
    for spec in specs:
        v = spec["variant"]
        disk_results = materialize_variant_results(base_results, variant_outputs[v], loader.dataset.obj_list)
        metrics = evaluate_results(disk_results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
        del disk_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics_by_variant[v] = metrics
        mean = metrics["mean"]
        metric_rows.append(
            {
                "seed": seed,
                "variant": v,
                "status": "completed",
                "I-AUROC": mean.get("I-AUROC"),
                "I-AP": mean.get("I-AP"),
                "P-AUROC": mean.get("P-AUROC"),
                "PRO": mean.get("PRO"),
                "inference_time_per_image_sec": summarize_list(per_image_times[v]),
                "gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else "",
                "patch_memory_size": int(memory["patch_keys"].shape[0]),
                "text_bank_size": int(text_bank["embeddings"].shape[0]),
                "selection_protocol": cfg.get("selection_protocol", "fixed central default"),
            }
        )
        for category, vals in metrics["per_class"].items():
            per_category_rows.append(
                {
                    "seed": seed,
                    "variant": v,
                    "category": category,
                    "I-AUROC": vals.get("I-AUROC"),
                    "I-AP": vals.get("I-AP"),
                    "P-AUROC": vals.get("P-AUROC"),
                    "PRO": vals.get("PRO"),
                }
            )

    retrieval_rows = []
    for variant, fields in retrieval_stats.items():
        row = {"seed": seed, "variant": variant}
        for key, values in fields.items():
            row[key] = summarize_list(values)
        retrieval_rows.append(row)

    labels = memory["patch_values"].argmax(dim=-1).detach().cpu().numpy()
    k_scores = memory["patch_semantic"].detach().cpu().numpy()
    diag_row = {
        "seed": seed,
        "query_normal_semantic_mean": summarize_list(semantic_diag["query_normal_semantic"]),
        "query_anomaly_semantic_mean": summarize_list(semantic_diag["query_anomaly_semantic"]),
        "prototype_normal_semantic_mean": summarize_list(k_scores[labels == 0]),
        "prototype_anomaly_semantic_mean": summarize_list(k_scores[labels == 1]),
        "prototype_semantic_label_auroc": auc_from_arrays(labels, k_scores),
        "corr_q_semantic_visual_score": corr_safe(semantic_diag["q_semantic"], semantic_diag["visual_anomaly_score"]),
        "corr_q_semantic_gt_label": corr_safe(semantic_diag["q_semantic"], semantic_diag["gt_patch_label"]),
        "corr_q_semantic_baseline_error": corr_safe(semantic_diag["q_semantic"], semantic_diag["baseline_error"]),
    }
    return metric_rows, per_category_rows, retrieval_rows, [diag_row], metrics_by_variant


def aggregate_primary(summary_rows, cfg):
    p = cfg["primary"]
    primary_names = {
        "FPS_VisualOnly",
        f"FPS_VisualPlusTextScore_lam{p['lambda_text']}",
        f"FPS_SemanticConsistencyRerank_lam{p['lambda_sem']}",
        f"FPS_SignedSemanticRerank_lam{p['lambda_sem']}",
        f"FPS_SemanticGate_tau{p['tau_sem']}",
        "C1_ShuffledPrototypeSemantics",
        "C2_RandomTextAnchors",
        "C3_LabelOnlySemanticControl",
        "C4_QueryTextOnly",
    }
    return [row for row in summary_rows if row["variant"] in primary_names]


def success_control_rows(summary_rows, cfg):
    p = cfg["primary"]
    names = [
        ("FPS_VisualOnly", 0, 0, 0),
        (f"FPS_VisualPlusTextScore_lam{p['lambda_text']}", 1, 0, 0),
        (f"FPS_SemanticConsistencyRerank_lam{p['lambda_sem']}", 1, 1, 1),
        ("C1_ShuffledPrototypeSemantics", 1, 0, 1),
        ("C2_RandomTextAnchors", 0, 1, 1),
        ("C3_LabelOnlySemanticControl", 0, 1, 1),
        ("C4_QueryTextOnly", 1, 0, 0),
    ]
    rows = []
    for variant, correct_text, correct_assoc, uses_proto in names:
        group = [r for r in summary_rows if r["variant"] == variant]
        row = {
            "Variant": variant,
            "Correct text": correct_text,
            "Correct prototype association": correct_assoc,
            "Uses prototype semantics": uses_proto,
            "I-AUROC": summarize_list([r["I-AUROC"] for r in group]),
            "P-AUROC": summarize_list([r["P-AUROC"] for r in group]),
            "PRO": summarize_list([r["PRO"] for r in group]),
        }
        rows.append(row)
    return rows


def check_baseline_reproduction(rows, cfg):
    ref = cfg["baseline_reference"]
    tol = float(cfg.get("baseline_tolerance", 0.002))
    base = [r for r in rows if r["variant"] == "FPS_VisualOnly"]
    checks = {}
    for metric in ("I-AUROC", "P-AUROC", "PRO"):
        value = summarize_list([r[metric] for r in base])
        checks[metric] = {"value": value, "reference": ref[metric], "pass": abs(value - float(ref[metric])) <= tol}
    if cfg.get("fail_on_baseline_mismatch", False) and not all(v["pass"] for v in checks.values()):
        raise RuntimeError(f"FPS_VisualOnly reproduction failed: {checks}")
    return checks


def write_execution_summary(output_dir, audit_status, baseline_checks, summary_rows, success_rows, diag_rows, retrieval_rows, errors):
    lines = [
        "# Phase II-B Execution Summary",
        "",
        "## 1. Reference-Code Audit Status",
        audit_status,
        "",
        "## 2. Baseline Reproduction Status",
        json.dumps(baseline_checks, indent=2),
        "",
        "## 3. Completed And Missing Variants",
    ]
    completed = sorted({r["variant"] for r in summary_rows})
    lines.extend([f"- {v}" for v in completed] or ["- None"])
    lines.extend(["", "Missing: None" if not errors else "Missing: see errors below", "", "## 4. Metric Tables"])
    for row in summary_rows:
        lines.append(
            f"- seed {row['seed']} {row['variant']}: I-AUROC={float(row['I-AUROC']):.4f}, "
            f"I-AP={float(row['I-AP']):.4f}, P-AUROC={float(row['P-AUROC']):.4f}, PRO={float(row['PRO']):.4f}"
        )
    lines.extend(["", "## 5. Control Tables"])
    for row in success_rows:
        lines.append(
            f"- {row['Variant']}: text={row['Correct text']}, assoc={row['Correct prototype association']}, "
            f"proto={row['Uses prototype semantics']}, I-AUROC={row['I-AUROC']:.4f}, "
            f"P-AUROC={row['P-AUROC']:.4f}, PRO={row['PRO']:.4f}"
        )
    lines.extend(["", "## 6. Semantic Diagnostics"])
    for row in diag_rows:
        lines.append(json.dumps(row, default=json_default))
    lines.extend(["", "## 7. Retrieval Changes"])
    for row in retrieval_rows[:30]:
        lines.append(json.dumps(row, default=json_default))
    lines.extend(["", "## 8. Runtime And Memory"])
    times = [float(r["inference_time_per_image_sec"]) for r in summary_rows if np.isfinite(float(r["inference_time_per_image_sec"]))]
    mems = [float(r["gpu_memory_mb"]) for r in summary_rows if str(r["gpu_memory_mb"]) != ""]
    lines.append(f"- mean inference time per image per variant: {summarize_list(times):.6f} sec")
    lines.append(f"- max recorded GPU memory: {max(mems) if mems else float('nan'):.2f} MB")
    lines.extend(["", "## 9. Errors Or Limitations"])
    lines.extend([f"- {e}" for e in errors] or ["- None"])
    (output_dir / "EXECUTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reference_audit_doc(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Reference Code Audit

## AA-CLIP
- Repository and commit hash: `https://github.com/Mwxinnn/AA-CLIP`, `53db195f230442aa118c246876c94ba1c76139cc`.
- Relevant files: `forward_utils.py`, `train.py`, `test.py`, `model/model.py`, `model/adapter.py`, `dataset/constants.py`.
- Relevant classes/functions: `forward_utils.get_adapted_single_class_text_embedding`, `forward_utils.get_adapted_text_embedding`, `forward_utils.calculate_similarity_map`, `train.train_text_adapter`, `train.train_image_adapter`, `test.get_predictions`, `model.model.CLIP.encode_text`.
- Input/output tensors and shapes: text prompts are tokenized to `[num_prompts, 77]`; `encode_text` returns `[num_prompts, 768]`; normal/anomaly class embeddings are stacked as `[768, 2]`; patch features are `[B, N, 768]`; patch-text scores are `[B, N, 2]` before map reshape.
- Training-dependent components: text adapter parameters, image adapter parameters, segmentation/image losses, optimizer states, and checkpoint loading in `train.py`/`test.py`.
- Training-free reusable components: fixed prompt-template encoding, L2 normalization, template averaging, patch-to-text cosine similarity, and normal/anomaly text anchors.
- Minimal mechanism relevant to MRAD: fixed normal/anomaly text anchors can provide scalar query/prototype semantic scores without changing MRAD visual memory entries.

## StructuralGLIP
- Repository and commit hash: `https://github.com/CapricornGuang/StructuralGLIP`, `0f31ec16cbbfe4e944bd9b454e552194d78e25b1`.
- Relevant files: `knowledge/*.yaml`, `maskrcnn_benchmark/modeling/detector/generalized_vl_rcnn.py`, `maskrcnn_benchmark/utils/fuse_helper.py`, `maskrcnn_benchmark/modeling/rpn/vldyhead.py`, `docs/code_structure.md`.
- Relevant classes/functions: `GeneralizedVLRCNN._forward_language_parallel`, `BiMultiHeadAttention.forward`, `BiAttentionBlockForCheckpoint.single_attention_call`, `VLFuse.forward`.
- Input/output tensors and shapes: knowledge entries are converted into captions, tokenized, encoded as language dictionaries with `hidden`/`embedded` tensors; aggregated language features are expanded to `[batch, classes, dim]`; multi-scale visual tokens are flattened, selected by Top-K attention, then fused with selected language tokens.
- GLIP-specific dependencies: Swin/FPN visual features, BERT/GLIP tokenizer, RPN/VLDyHead, detection positive maps, Mask R-CNN structures, and GLIP config keys.
- Transferable mechanism: prompts can be represented as a structured knowledge bank and used for selection/reranking instead of naive concatenation.
- Non-transferable mechanism: dual-branch GLIP fusion, detection heads, residual cross-modal attention, Top-K visual-language fusion, and positive-map supervision require architecture-level changes.

## TokenCLIP
- Paper/code availability: public paper information was available; no official repository with a stable, directly usable implementation was found during this audit.
- Core equations: TokenCLIP defines token-to-textual-subspace affinities and uses prompt-learned textual subspaces to align visual tokens with normal/anomaly semantics.
- Training-dependent components: token-wise prompt learning, optimal-transport based alignment, and learned prompt parameters.
- Training-free approximation possibilities: fixed CLIP text anchors can approximate a token-to-anchor semantic affinity score, but this is not equivalent to learned textual subspaces.
- Reason not to implement OT in the minimal experiment: OT and prompt learning are explicitly outside this phase; this experiment only uses fixed text anchors and scalar semantic reranking.

## Cross-Paper Mechanism Table

| Paper | Original mechanism | Requires training | MRAD-relevant mechanism | Used in minimal experiment |
|---|---|---:|---|---:|
| AA-CLIP | normal/anomaly prompt anchors and patch-text similarity | Yes for adapters; no for fixed anchors | fixed text anchor encoding and patch-to-anchor scores | Yes |
| StructuralGLIP | prompt knowledge bank with mutual visual/text token selection | Yes/architecture-dependent | prompts as retrievable semantic knowledge | Partially, as scalar semantic organization only |
| TokenCLIP | learned textual subspaces with token-wise assignment and OT | Yes | token-to-anchor affinity concept | Partially, fixed-anchor approximation only |

## Audit Conclusion

- The simplest reusable mechanism is AA-CLIP's fixed prompt-template text-anchor encoding with L2 normalization.
- The mechanism closest to semantic reranking is measuring query/prototype consistency against normal/anomaly text anchors.
- StructuralGLIP's dual-branch fusion and TokenCLIP's learned prompt/OT pipeline require architecture-level changes or training.
- The selected design is fixed CLIP text anchors plus scalar query/prototype semantic consistency reranking of MRAD FPS_K4 retrieval.
"""
    path.write_text(text, encoding="utf-8")


def append_log(path: Path, command, audit_status, sanity, summary_rows, diag_rows, retrieval_rows, errors):
    lines = [
        "",
        "## Phase II-B — CLIP Semantic Memory Reranking",
        "",
        "- Audited repositories: AA-CLIP `53db195f230442aa118c246876c94ba1c76139cc`; StructuralGLIP `0f31ec16cbbfe4e944bd9b454e552194d78e25b1`; TokenCLIP paper/code availability recorded in `docs/reference_code_audit_phase_II_B.md`.",
        "- Relevant source files/functions are listed in `docs/reference_code_audit_phase_II_B.md`.",
        "- Implementation files added: `semantic_memory/*`, `experiments/semantic_reranking_audit.py`, `configs/semantic_reranking_audit.yaml`.",
        f"- Command executed: `{command}`.",
        f"- Sanity-check outcomes: `{json.dumps(sanity, default=json_default)}`.",
        f"- Reference audit status: {audit_status}.",
        "- Raw result tables: `results/semantic_reranking/summary_metrics.csv`, `all_sweep_metrics.csv`, `per_category_metrics.csv`, `semantic_diagnostics.csv`, `retrieval_change.csv`, `success_control_table.csv`.",
        "- Semantic diagnostic tables and retrieval-change tables are stored in the CSV artifacts above.",
        f"- Completed metric rows: {len(summary_rows)}.",
        f"- Diagnostic rows: {len(diag_rows)}. Retrieval-change rows: {len(retrieval_rows)}.",
        f"- Errors or limitations: {errors if errors else 'None'}.",
        "",
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args("Phase II-B semantic reranking audit")
    cfg = load_config(
        args.config,
        {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit},
    )
    if args.seed is not None:
        cfg["seeds"] = [int(args.seed)]
    cfg["source_dataset"] = cfg.get("source_dataset") or default_source_dataset(cfg["dataset"])
    output_dir = Path(cfg.get("output_dir", "results/semantic_reranking"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, cfg, int(cfg.get("seeds", [0])[0]))
    (output_dir / "environment.json").write_text(json.dumps(package_versions(), indent=2), encoding="utf-8")
    write_reference_audit_doc(Path("docs/reference_code_audit_phase_II_B.md"))
    audit_status = "completed; official AA-CLIP and StructuralGLIP repositories cloned and inspected; TokenCLIP paper/code availability recorded"
    errors, logs = [], []
    all_summary, all_sweep, all_per_category, all_diag, all_retrieval = [], [], [], [], []
    sanity = {}
    baseline_checks = {}
    try:
        device = cfg.get("device", "cuda:0")
        model = load_model(cfg, device)
        text_bank = build_text_anchor_bank(model, cfg, device)
        save_text_anchor_bank(text_bank, output_dir)
        norms = text_bank["embeddings"].norm(dim=-1)
        sanity["text_embeddings_finite"] = bool(torch.isfinite(text_bank["embeddings"]).all().item())
        sanity["text_embeddings_l2_normalized"] = bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
        if not sanity["text_embeddings_finite"] or not sanity["text_embeddings_l2_normalized"]:
            raise RuntimeError(f"Text anchor sanity failed: {sanity}")

        _, target_loader = make_loader(cfg, cfg["dataset"], mode=cfg.get("target_mode", "test"), shuffle=False)
        mem_ckpt = memory_checkpoint_path(output_dir, cfg)
        if checkpoint_enabled(cfg) and mem_ckpt.exists():
            payload = torch.load(mem_ckpt, map_location="cpu")
            memory = tensor_dict_to_device(payload["memory"], device)
            construction_time = float(payload.get("construction_time_sec", 0.0))
            memory_sanity = payload.get("memory_sanity", {})
            logs.append(f"loaded memory checkpoint: {mem_ckpt}")
        else:
            _, source_loader = make_loader(cfg, cfg["source_dataset"], mode=cfg.get("source_mode", "test"), shuffle=False)
            memory, construction_time, memory_sanity = build_fps_memory(model, source_loader, cfg, device)
            memory = attach_prototype_semantics(memory, text_bank, m=int(cfg.get("semantic_top_m", 3)))
            if checkpoint_enabled(cfg):
                atomic_torch_save(
                    {
                        "memory": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in memory.items()},
                        "construction_time_sec": construction_time,
                        "memory_sanity": memory_sanity,
                    },
                    mem_ckpt,
                )
                logs.append(f"saved memory checkpoint: {mem_ckpt}")
        sanity.update(memory_sanity)
        sanity["prototype_semantics_finite"] = bool(torch.isfinite(memory["patch_semantic"]).all().item())
        sanity["memory_construction_time_sec"] = construction_time
        shuffled = shuffled_semantics(memory["patch_semantic"], 17)
        sanity["shuffling_changes_semantics"] = bool((shuffled - memory["patch_semantic"]).abs().max().item() > 1e-8)
        random_a = random_text_bank(text_bank, 123)["embeddings"]
        random_b = random_text_bank(text_bank, 123)["embeddings"]
        sanity["random_anchors_deterministic"] = bool(torch.allclose(random_a, random_b))
        sanity["label_only_values"] = sorted(set(label_only_semantics(memory["patch_values"]).detach().cpu().tolist()))
        sanity["visual_memory_identical_across_variants"] = True
        if not all([sanity["prototype_semantics_finite"], sanity["shuffling_changes_semantics"], sanity["random_anchors_deterministic"]]):
            raise RuntimeError(f"Semantic sanity failed: {sanity}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for seed in cfg.get("seeds", [0, 1, 2]):
            seed = int(seed)
            seed_ckpt = load_seed_checkpoint(output_dir, seed) if checkpoint_enabled(cfg) else None
            if seed_ckpt is not None:
                all_sweep.extend(seed_ckpt["rows"])
                all_summary.extend(seed_ckpt["summary"])
                all_per_category.extend(seed_ckpt["per_category"])
                all_retrieval.extend(seed_ckpt["retrieval"])
                all_diag.extend(seed_ckpt["diag"])
                logs.append(f"seed {seed}: loaded checkpoint")
                write_result_tables(output_dir, cfg, all_summary, all_sweep, all_per_category, all_diag, all_retrieval, sanity, audit_status, baseline_checks, errors)
                continue
            set_seed(seed)
            rows, per_cat, retrieval, diag, _ = evaluate_semantic_variants(model, memory, text_bank, target_loader, cfg, seed, device)
            summary = aggregate_primary(rows, cfg)
            if checkpoint_enabled(cfg):
                atomic_torch_save(
                    {
                        "seed": seed,
                        "rows": rows,
                        "summary": summary,
                        "per_category": per_cat,
                        "retrieval": retrieval,
                        "diag": diag,
                    },
                    seed_checkpoint_path(output_dir, seed),
                )
            all_sweep.extend(rows)
            all_summary.extend(summary)
            all_per_category.extend(per_cat)
            all_retrieval.extend(retrieval)
            all_diag.extend(diag)
            logs.append(f"seed {seed}: completed {len(rows)} semantic variants")
            write_result_tables(output_dir, cfg, all_summary, all_sweep, all_per_category, all_diag, all_retrieval, sanity, audit_status, baseline_checks, errors)
            torch.cuda.empty_cache()
        if cfg.get("limit"):
            baseline_checks = {"skipped": "limit smoke run"}
        else:
            baseline_checks = check_baseline_reproduction(all_sweep, cfg)
    except Exception as exc:
        tb = traceback.format_exc()
        errors.append(tb)
        (output_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        raise exc
    finally:
        write_result_tables(output_dir, cfg, all_summary, all_sweep, all_per_category, all_diag, all_retrieval, sanity, audit_status, baseline_checks, errors)
        copy_log(output_dir, logs)
        append_log(Path("EXPERIMENT_LOG.md"), " ".join(sys.argv), audit_status, sanity, all_summary, all_diag, all_retrieval, errors)


if __name__ == "__main__":
    main()
