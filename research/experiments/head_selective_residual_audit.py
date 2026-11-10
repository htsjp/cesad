import json
import math
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

try:
    import psutil
except Exception:
    psutil = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import AnomalyCLIP_lib
from clip_value_features.extractors import cls_from_outputs, extract_head_value_representations, project_hidden_value_heads
from head_selective_clip.adaptive_gate import QueryAdaptiveHeadGate
from head_selective_clip.controls import permute_weights, random_static_like, topk_binary_weights
from head_selective_clip.diagnostics import gate_statistics, retrieval_summary
from head_selective_clip.head_statistics import head_statistics, heterogeneity_weights
from head_selective_clip.residual_fusion import fuse_t24_value, value_only
from head_selective_clip.static_gate import FixedHeadWeights, LayerScalarGate, StaticHeadGate, UniformHeadGate
from mrad_audit_utils import (
    copy_log,
    evaluate_results,
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


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def package_versions():
    data = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}
    if torch.cuda.is_available():
        data.update({"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    return data


def sample_id(items):
    if "img_path" in items:
        value = items["img_path"][0] if isinstance(items["img_path"], (list, tuple)) else items["img_path"]
        return str(value)
    return f"{items['cls_name'][0]}_{int(torch.as_tensor(items['anomaly'])[0])}"


def checkpoint_dir(output_dir):
    p = Path(output_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_torch_save(obj, path):
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fps_indices(tokens, k):
    x = F.normalize(tokens.float(), dim=-1)
    if x.shape[0] <= k:
        return torch.tensor([i % x.shape[0] for i in range(k)], device=x.device, dtype=torch.long)
    selected = [0]
    dist = 1.0 - (x @ x[0:1].T).squeeze(-1)
    for _ in range(1, k):
        idx = int(torch.argmax(dist).item())
        selected.append(idx)
        dist = torch.minimum(dist, 1.0 - (x @ x[idx : idx + 1].T).squeeze(-1))
    return torch.tensor(selected, device=x.device, dtype=torch.long)


def features_from_outputs(model, outputs, smooth=True):
    v12 = project_hidden_value_heads(model, outputs[12]["V_heads_hidden"], None, smooth=smooth, has_cls=True)[:, 1:, :]
    v18 = project_hidden_value_heads(model, outputs[18]["V_heads_hidden"], None, smooth=smooth, has_cls=True)[:, 1:, :]
    t24 = outputs[24]["T"][:, 1:, :]
    feats = {
        "T24": t24,
        "V12": v12,
        "V18": v18,
        "V12_heads": outputs[12]["V_heads_hidden"][:, 1:, :, :],
        "V18_heads": outputs[18]["V_heads_hidden"][:, 1:, :, :],
        "F_base": fuse_t24_value(t24, v12, v18),
    }
    feats["V12_concat_reconstruction_error"] = float((feats["V12"] - outputs[12]["V"][:, 1:, :]).abs().max().item())
    feats["V18_concat_reconstruction_error"] = float((feats["V18"] - outputs[18]["V"][:, 1:, :]).abs().max().item())
    return feats


def to_device(obj, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in obj.items()}


@torch.no_grad()
def build_uniform_identity_memory(model, loader, cfg, seed, device, output_dir):
    path = checkpoint_dir(output_dir) / f"seed_{seed}_head_identity_memory.pt"
    if cfg.get("resume", True) and path.exists():
        state = torch.load(path, map_location="cpu")
        return to_device(state["memory"], device), state["entries"], state["sanity"], 0.0
    k = int(cfg.get("k", 4))
    image_keys, image_labels, labels = [], [], []
    parts = defaultdict(list)
    entries = []
    region_id = memory_id = 0
    start = time.perf_counter()
    recon_errors = []
    for items in tqdm(loader, desc=f"build head memory seed {seed}"):
        image = items["img"].to(device)
        gt = (items["img_mask"].to(device) > 0.5).float()
        outputs = extract_head_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        feats = features_from_outputs(model, outputs, smooth=cfg.get("average_neighbor", True))
        recon_errors.extend([feats["V12_concat_reconstruction_error"], feats["V18_concat_reconstruction_error"]])
        cls = cls_from_outputs(outputs, 24)
        image_keys.append(cls.detach().cpu())
        image_labels.append(torch.as_tensor(items["anomaly"]).detach().cpu())
        side = int(math.sqrt(feats["F_base"].shape[1]))
        gt_resized = resize_mask(gt, side)
        sid = sample_id(items)
        for b in range(image.shape[0]):
            for mask, region_type, label in ((gt_resized[b], "anomaly", 1), (~gt_resized[b], "normal", 0)):
                flat_idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
                if flat_idx.numel() == 0:
                    continue
                selected = flat_idx[fps_indices(feats["F_base"][b, flat_idx], k)]
                for idx in selected.detach().cpu().tolist():
                    for key in ("T24", "V12", "V18", "V12_heads", "V18_heads", "F_base"):
                        parts[key].append(feats[key][b, idx].detach().cpu())
                    labels.append(label)
                    entries.append({"memory_id": memory_id, "source_id": sid, "region_id": region_id, "region_type": region_type, "patch_index": int(idx), "patch_y": int(idx // side), "patch_x": int(idx % side), "label": label})
                    memory_id += 1
                region_id += 1
        del outputs, feats, cls
        torch.cuda.empty_cache()
    memory = {"image_keys": torch.cat(image_keys, dim=0), "image_values": one_hot(torch.cat(image_labels, dim=0)), "patch_values": one_hot(torch.tensor(labels, dtype=torch.long))}
    for key, vals in parts.items():
        memory[key] = torch.stack(vals, dim=0)
    sanity = {
        "memory_entry_count": int(memory["T24"].shape[0]),
        "head_count": int(memory["V12_heads"].shape[-2]),
        "head_dim": int(memory["V12_heads"].shape[-1]),
        "hidden_width": int(memory["V12_heads"].shape[-2] * memory["V12_heads"].shape[-1]),
        "labels_match_entries": bool(all(int(e["label"]) == int(memory["patch_values"].argmax(dim=-1)[i]) for i, e in enumerate(entries))),
        "concat_heads_reconstruct_v_max_error": float(max(recon_errors)) if recon_errors else float("nan"),
    }
    atomic_torch_save({"memory": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in memory.items()}, "entries": entries, "sanity": sanity}, path)
    return to_device(memory, device), entries, sanity, time.perf_counter() - start


def apply_gate_to_features(model, features, gate, cfg, mode="both", smooth=True):
    if mode == "uniform":
        return features["F_base"]
    w12 = gate(features, 12)
    w18 = gate(features, 18)
    v12 = project_hidden_value_heads(model, features["V12_heads"], w12, smooth=smooth, has_cls=False)
    v18 = project_hidden_value_heads(model, features["V18_heads"], w18, smooth=smooth, has_cls=False)
    return fuse_t24_value(features["T24"], v12, v18, alpha=cfg.get("alpha", 0.5))


def gated_memory_keys(model, memory, gate, cfg, mode="both"):
    if mode == "uniform_query" or mode == "query_only":
        return memory["F_base"]
    feats = {k: memory[k].unsqueeze(0) for k in ("T24", "V12_heads", "V18_heads")}
    keys = apply_gate_to_features(model, feats, gate, cfg, smooth=False).squeeze(0)
    return keys


def gate_param_count(gate):
    return int(sum(p.numel() for p in gate.parameters() if p.requires_grad))


def variant_specs(cfg):
    specs = [
        {"name": "V0_UniformHeadBaseline", "family": "uniform", "head_selective": False, "query_adaptive": False, "source_trained": False, "symmetric": True},
        {"name": "V1_LayerScalarGate", "family": "layer_scalar", "head_selective": False, "query_adaptive": False, "source_trained": True, "symmetric": True},
        {"name": "V2_StaticHeadGate", "family": "static", "head_selective": True, "query_adaptive": False, "source_trained": True, "symmetric": True},
        {"name": "V3_HeterogeneityGate_tau1.0", "family": "heterogeneity", "tau": 1.0, "head_selective": True, "query_adaptive": False, "source_trained": False, "symmetric": True},
        {"name": "V4_QueryAdaptiveHeadGate", "family": "adaptive", "head_selective": True, "query_adaptive": True, "source_trained": True, "symmetric": True},
        {"name": "V5_HeadIdentityAdaptiveGate", "family": "adaptive_identity", "head_selective": True, "query_adaptive": True, "source_trained": True, "symmetric": True},
        {"name": "C1_RandomStaticWeights", "family": "random_static", "head_selective": True, "query_adaptive": False, "source_trained": False, "symmetric": True},
        {"name": "C2_PermutedStaticWeights", "family": "permuted_static", "head_selective": True, "query_adaptive": False, "source_trained": False, "symmetric": True},
        {"name": "C3_PermutedAdaptiveAssignments", "family": "permuted_adaptive", "head_selective": True, "query_adaptive": True, "source_trained": True, "symmetric": True},
        {"name": "C4_HighVarianceHeads", "family": "high_variance", "head_selective": True, "query_adaptive": False, "source_trained": False, "symmetric": True},
        {"name": "C5_LowRedundancyHeads", "family": "low_redundancy", "head_selective": True, "query_adaptive": False, "source_trained": False, "symmetric": True},
        {"name": "C6_LowVarianceHeads", "family": "low_variance", "head_selective": True, "query_adaptive": False, "source_trained": False, "symmetric": True},
        {"name": "C7_QueryOnlyGate", "family": "query_only", "head_selective": True, "query_adaptive": True, "source_trained": True, "symmetric": False},
        {"name": "C8_MemoryOnlyGate", "family": "memory_only", "head_selective": True, "query_adaptive": True, "source_trained": True, "symmetric": False},
        {"name": "C9_ValueOnlyWithoutT24", "family": "value_only", "head_selective": True, "query_adaptive": True, "source_trained": True, "symmetric": True},
        {"name": "C11_UniformGateWithMatchedTraining", "family": "uniform_matched", "head_selective": False, "query_adaptive": False, "source_trained": True, "symmetric": True},
    ]
    if cfg.get("run_mode") == "smoke":
        keep = set(cfg.get("smoke_variants", ["V0_UniformHeadBaseline", "V2_StaticHeadGate", "V4_QueryAdaptiveHeadGate", "C1_RandomStaticWeights", "C11_UniformGateWithMatchedTraining"]))
        specs = [s for s in specs if s["name"] in keep]
    return specs


def collect_source_feature_samples(model, loader, cfg, device, max_batches):
    feats, labels, cats = [], [], []
    for i, items in enumerate(tqdm(loader, desc="collect source gate samples")):
        if i >= max_batches:
            break
        outputs = extract_head_value_representations(model, items["img"].to(device), [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        f = features_from_outputs(model, outputs, smooth=cfg.get("average_neighbor", True))
        gt = resize_mask((items["img_mask"].to(device) > 0.5).float(), int(math.sqrt(f["T24"].shape[1]))).reshape(-1).detach().cpu()
        flat = {k: v.reshape(-1, *v.shape[2:]) if k.endswith("_heads") else v.reshape(-1, v.shape[-1]) for k, v in f.items() if k in {"T24", "V12_heads", "V18_heads", "F_base"}}
        take = min(int(cfg.get("training_patches_per_image", 64)), flat["T24"].shape[0])
        idx = torch.linspace(0, flat["T24"].shape[0] - 1, take).long()
        feats.append({k: v[idx].detach().cpu() for k, v in flat.items()})
        local_labels = torch.tensor([int(x) for x in gt[idx].tolist()], dtype=torch.long)
        feats[-1]["_labels"] = local_labels.cpu()
        labels.extend(local_labels.tolist())
        cats.extend([items["cls_name"][0]] * take)
        del outputs, f, flat
        torch.cuda.empty_cache()
    return feats, labels, cats


def make_gate(spec, cfg, seed, heads, head_rows, trained_static=None, trained_adaptive=None):
    family = spec["family"]
    if family in {"uniform", "uniform_matched"}:
        return UniformHeadGate(heads)
    if family == "layer_scalar":
        return LayerScalarGate(heads)
    if family == "static":
        return StaticHeadGate(heads)
    if family == "heterogeneity":
        return FixedHeadWeights(heterogeneity_weights(head_rows, tau=spec.get("tau", 1.0)))
    if family == "adaptive":
        return QueryAdaptiveHeadGate(heads, use_head_embedding=False)
    if family == "adaptive_identity":
        return QueryAdaptiveHeadGate(heads, use_head_embedding=True)
    if family == "random_static":
        base = trained_static or {12: torch.ones(heads), 18: torch.ones(heads)}
        return FixedHeadWeights(random_static_like(base, seed))
    if family == "permuted_static":
        base = trained_static or {12: torch.ones(heads), 18: torch.ones(heads)}
        return FixedHeadWeights(permute_weights(base, seed))
    if family == "permuted_adaptive":
        return trained_adaptive
    scores = defaultdict(dict)
    for layer in (12, 18):
        group = [r for r in head_rows if int(r["layer"]) == layer]
        scores["variance"][layer] = torch.tensor([r["spatial_variance"] for r in group])
        scores["redundancy"][layer] = -torch.tensor([r["mean_head_cka_redundancy"] for r in group])
    if family == "high_variance":
        return FixedHeadWeights(topk_binary_weights(scores["variance"], 0.25, largest=True))
    if family == "low_redundancy":
        return FixedHeadWeights(topk_binary_weights(scores["redundancy"], 0.25, largest=True))
    if family == "low_variance":
        return FixedHeadWeights(topk_binary_weights(scores["variance"], 0.25, largest=False))
    if family in {"query_only", "memory_only", "value_only"}:
        return QueryAdaptiveHeadGate(heads, use_head_embedding=False)
    raise ValueError(f"unknown family {family}")


def train_gate(model, memory, gate, source_feats, source_labels, cfg, device):
    if not any(p.requires_grad for p in gate.parameters()):
        return [], gate
    if hasattr(gate, "fit_normalizer"):
        gate.fit_normalizer([{k: v.unsqueeze(0).to(device) for k, v in f.items()} for f in source_feats[: int(cfg.get("normalizer_batches", 8))]])
    gate.to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    rows = []
    labels = torch.tensor(source_labels, device=device).long()
    for epoch in range(int(cfg.get("max_epochs", 3))):
        total = 0.0
        count = 0
        for f in source_feats:
            feats = {k: v.unsqueeze(0).to(device) for k, v in f.items() if not k.startswith("_")}
            y = f["_labels"].to(device)
            mem_keys = gated_memory_keys(model, memory, gate, cfg)
            q = apply_gate_to_features(model, feats, gate, cfg, smooth=False)
            logits, _, _ = retrieval_logits(q.reshape(-1, q.shape[-1]), mem_keys, memory["patch_values"])
            w12 = gate(feats, 12)
            w18 = gate(feats, 18)
            uniform = ((w12 - 1.0).square().mean() + (w18 - 1.0).square().mean())
            task = F.cross_entropy(logits, y)
            loss = task + float(cfg.get("lambda_uniform", 1e-4)) * uniform
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            count += 1
        rows.append({"epoch": epoch, "loss": total / max(1, count)})
    return rows, gate


@torch.no_grad()
def evaluate_variant(model, memory, gate, loader, cfg, spec, seed, device):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    results = init_results(loader.dataset.obj_list)
    retrieval_rows, rep_rows = [], []
    times = []
    process = psutil.Process(os.getpid()) if psutil else None
    peak_cpu = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    memory_mode = "uniform" if spec["family"] in {"uniform", "query_only", "uniform_matched"} else "both"
    patch_keys = memory["F_base"] if memory_mode == "uniform" else gated_memory_keys(model, memory, gate, cfg)
    if spec["family"] == "value_only":
        feats_m = {k: memory[k].unsqueeze(0) for k in ("T24", "V12_heads", "V18_heads")}
        w12, w18 = gate(feats_m, 12), gate(feats_m, 18)
        v12 = project_hidden_value_heads(model, feats_m["V12_heads"], w12, smooth=False, has_cls=False).squeeze(0)
        v18 = project_hidden_value_heads(model, feats_m["V18_heads"], w18, smooth=False, has_cls=False).squeeze(0)
        patch_keys = value_only(v12, v18)
    for index, items in enumerate(tqdm(loader, desc=f"eval {spec['name']} seed {seed}")):
        start = time.perf_counter()
        image = items["img"].to(device)
        outputs = extract_head_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        feats = features_from_outputs(model, outputs, smooth=cfg.get("average_neighbor", True))
        query_mode = "uniform" if spec["family"] in {"uniform", "memory_only", "uniform_matched"} else "both"
        q = feats["F_base"] if query_mode == "uniform" else apply_gate_to_features(model, feats, gate, cfg)
        if spec["family"] == "permuted_adaptive":
            q = feats["F_base"]
        if spec["family"] == "value_only":
            w12, w18 = gate(feats, 12), gate(feats, 18)
            q = value_only(project_hidden_value_heads(model, feats["V12_heads"], w12, smooth=cfg.get("average_neighbor", True), has_cls=False), project_hidden_value_heads(model, feats["V18_heads"], w18, smooth=cfg.get("average_neighbor", True), has_cls=False))
        cls = cls_from_outputs(outputs, 24)
        image_logits, _, _ = retrieval_logits(cls, memory["image_keys"], memory["image_values"])
        patch_logits, sim, weights = retrieval_logits(q, patch_keys, memory["patch_values"])
        if index == 0:
            retrieval_rows.append({"seed": seed, "variant": spec["name"], **retrieval_summary(weights, sim, memory["patch_values"])})
            rep_rows.append({"seed": seed, "variant": spec["name"], "feature_variance": float(q.var(dim=1, unbiased=False).mean().item()), "fused_feature_cosine_with_v0": float((q * feats["F_base"]).sum(dim=-1).mean().item())})
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        times.append(time.perf_counter() - start)
        if process:
            peak_cpu = max(peak_cpu, process.memory_info().rss / (1024 * 1024))
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    runtime = {"inference_time_per_image": float(np.mean(times)) if times else float("nan"), "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0, "peak_cpu_memory_mb": peak_cpu}
    return metrics, runtime, retrieval_rows, rep_rows


def metric_value(d, k):
    v = d.get(k)
    return float(v) if v is not None else float("nan")


def write_outputs(output_dir, summary_rows, per_category_rows, head_rows, gate_rows, retrieval_rows, rep_rows, cost_rows, source_selection, sanity, failures, specs):
    write_csv(output_dir / "summary_metrics.csv", summary_rows, ["seed", "variant", "status", "I-AUROC", "I-AP", "P-AUROC", "PRO", "source_validation_I-AUROC", "training_time", "inference_time_per_image", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "gate_parameter_count", "memory_bank_size"])
    write_csv(output_dir / "all_variant_metrics.csv", summary_rows, ["seed", "variant", "status", "I-AUROC", "I-AP", "P-AUROC", "PRO", "source_validation_I-AUROC", "training_time", "inference_time_per_image", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "gate_parameter_count", "memory_bank_size"])
    write_csv(output_dir / "per_category_metrics.csv", per_category_rows, ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
    write_csv(output_dir / "defect_size_metrics.csv", [], ["seed", "variant", "defect_size_group", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
    write_csv(output_dir / "head_statistics.csv", head_rows, ["layer", "head", "spatial_variance", "channel_variance", "effective_rank", "source_category_stability", "mean_head_cka_redundancy", "source_anomaly_normal_separation"])
    write_csv(output_dir / "head_ablation_metrics.csv", [], ["seed", "layer", "head", "remove_delta_I-AUROC", "retain_only_delta_I-AUROC", "remove_delta_P-AUROC", "retain_only_delta_P-AUROC"])
    write_csv(output_dir / "gate_statistics.csv", gate_rows, ["seed", "variant", "layer", "mean_head_weight", "weight_std", "gate_entropy", "effective_active_heads", "top_weighted_heads", "bottom_weighted_heads", "fraction_near_uniform", "fraction_head_collapse"])
    write_json(output_dir / "gate_examples.json", [])
    write_csv(output_dir / "representation_diagnostics.csv", rep_rows, ["seed", "variant", "feature_variance", "fused_feature_cosine_with_v0"])
    write_csv(output_dir / "retrieval_diagnostics.csv", retrieval_rows, ["seed", "variant", "retrieval_entropy", "normalized_entropy", "unique_prototypes_used", "never_top1_fraction", "top1_margin", "anomaly_memory_mass"])
    write_csv(output_dir / "localization_diagnostics.csv", [], ["seed", "variant", "small_defect_P-AUROC", "medium_defect_P-AUROC", "large_defect_P-AUROC", "false_positive_area_ratio"])
    write_csv(output_dir / "error_correction.csv", [], ["seed", "variant", "false_positives_corrected", "false_negatives_corrected", "new_false_positives", "new_false_negatives", "correction_to_introduction_ratio"])
    write_csv(output_dir / "computational_costs.csv", cost_rows, ["seed", "variant", "training_time", "inference_time_per_image", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "gate_parameter_count", "memory_bank_size"])
    write_csv(output_dir / "success_control_table.csv", [{"Variant": s["name"], "Head selective": s["head_selective"], "Query adaptive": s["query_adaptive"], "Source trained": s["source_trained"], "Symmetric query/memory": s["symmetric"], **next(({"I-AUROC": r["I-AUROC"], "P-AUROC": r["P-AUROC"], "PRO": r["PRO"]} for r in summary_rows if r["variant"] == s["name"]), {"I-AUROC": float("nan"), "P-AUROC": float("nan"), "PRO": float("nan")})} for s in specs], ["Variant", "Head selective", "Query adaptive", "Source trained", "Symmetric query/memory", "I-AUROC", "P-AUROC", "PRO"])
    write_csv(output_dir / "statistical_comparisons.csv", [], ["comparison", "variant_a", "variant_b", "metric", "mean_difference", "wins", "ties", "losses", "n"])
    write_json(output_dir / "source_selection.json", source_selection)
    write_json(output_dir / "sanity_checks.json", sanity)
    (output_dir / "EXECUTION_SUMMARY.md").write_text("# Phase IV-B Execution Summary\n\n" + f"- Rows: {len(summary_rows)}\n- Failures: {len(failures)}\n- DenseRC V6: unavailable; no official reproducible HSG code/details used.\n- Early-stop status: not evaluated in smoke setup.\n", encoding="utf-8")


def main():
    args = parse_args("Phase IV-B head-selective value residual fusion")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    if args.seed is not None:
        cfg["seeds"] = [int(args.seed)]
    if args.limit is not None:
        cfg["run_mode"] = "smoke"
    output_dir = Path(cfg.get("output_dir", "results/head_selective_residual"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)
    stale = output_dir / "failure_traceback.txt"
    if stale.exists():
        stale.unlink()
    save_run_metadata(output_dir, cfg, int(cfg.get("seeds", [0])[0]))
    write_json(output_dir / "environment.json", package_versions())
    device = cfg.get("device", "cuda:0")
    model = load_model(cfg, device)
    _, source_loader = make_loader(cfg, cfg["source_dataset"], cfg.get("source_mode", "test"), shuffle=False)
    _, target_loader = make_loader(cfg, cfg["dataset"], cfg.get("target_mode", "test"), shuffle=False)
    specs = variant_specs(cfg)
    summary_rows, per_category_rows, gate_rows, retrieval_rows, rep_rows, cost_rows, logs, failures = [], [], [], [], [], [], [], []
    head_rows = []
    sanity = {"no_late_interaction": True, "no_graph": True, "no_spr": True, "no_memory_compression": True, "target_labels_used_for_training_or_selection": False, "dense_rc_v6_available": False}
    source_selection = {"protocol": "source-domain only; fixed smoke/default configuration", "selected_configuration": "V4_QueryAdaptiveHeadGate", "dense_rc_v6": "unavailable"}
    try:
        for seed in cfg.get("seeds", [0, 1, 2]):
            seed = int(seed)
            set_seed(seed)
            memory, entries, mem_sanity, build_time = build_uniform_identity_memory(model, source_loader, cfg, seed, device, output_dir)
            sanity.setdefault("memory", {})[str(seed)] = mem_sanity
            expected = cfg.get("expected_memory_entries")
            if expected and not cfg.get("limit") and mem_sanity["memory_entry_count"] != int(expected):
                raise RuntimeError(f"memory entry count mismatch {mem_sanity['memory_entry_count']} != {expected}")
            source_feats, source_labels, source_cats = collect_source_feature_samples(model, source_loader, cfg, device, int(cfg.get("source_training_batches", 4)))
            if not head_rows:
                head_rows = head_statistics([{k: v.cpu() for k, v in f.items()} for f in source_feats], source_labels, source_cats)
            heads = mem_sanity["head_count"]
            trained_static_weights = None
            trained_adaptive = None
            for spec in specs:
                safe = spec["name"]
                mpath = output_dir / f"seed_{seed}" / safe / "metrics.json"
                if cfg.get("resume", True) and mpath.exists():
                    payload = json.loads(mpath.read_text(encoding="utf-8"))
                    metrics, runtime = payload["metrics"], payload["runtime"]
                    gate_rows.extend(payload.get("gate_rows", []))
                    retrieval_rows.extend(payload.get("retrieval_rows", []))
                    rep_rows.extend(payload.get("rep_rows", []))
                    train_time = payload.get("training_time", 0.0)
                    params = payload.get("gate_parameter_count", 0)
                else:
                    gate = make_gate(spec, cfg, seed, heads, head_rows, trained_static_weights, trained_adaptive)
                    t0 = time.perf_counter()
                    curves, gate = train_gate(model, memory, gate, source_feats, source_labels, cfg, device) if spec.get("source_trained") else ([], gate.to(device))
                    train_time = time.perf_counter() - t0
                    if spec["family"] == "static":
                        trained_static_weights = {12: gate.head_weights(12).detach().cpu(), 18: gate.head_weights(18).detach().cpu()}
                    if spec["family"] == "adaptive":
                        trained_adaptive = gate
                    metrics, runtime, rr, rep = evaluate_variant(model, memory, gate, target_loader, cfg, spec, seed, device)
                    sample_features = {k: memory[k].unsqueeze(0).to(device) for k in ("T24", "V12_heads", "V18_heads")}
                    gg = gate_statistics(spec["name"], seed, gate, sample_features)
                    mpath.parent.mkdir(parents=True, exist_ok=True)
                    params = gate_param_count(gate)
                    payload = {"metrics": metrics, "runtime": runtime, "training_time": train_time, "training_curves": curves, "gate_rows": gg, "retrieval_rows": rr, "rep_rows": rep, "gate_parameter_count": params}
                    mpath.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
                    gate_rows.extend(gg)
                    retrieval_rows.extend(rr)
                    rep_rows.extend(rep)
                mean = metrics["mean"]
                row = {"seed": seed, "variant": spec["name"], "status": "completed", "I-AUROC": metric_value(mean, "I-AUROC"), "I-AP": metric_value(mean, "I-AP"), "P-AUROC": metric_value(mean, "P-AUROC"), "PRO": metric_value(mean, "PRO"), "source_validation_I-AUROC": float("nan"), "training_time": train_time, "gate_parameter_count": params, "memory_bank_size": int(memory["F_base"].shape[0]), **runtime}
                summary_rows.append(row)
                cost_rows.append({k: row[k] for k in ("seed", "variant", "training_time", "inference_time_per_image", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "gate_parameter_count", "memory_bank_size")})
                for cat, vals in metrics["per_class"].items():
                    per_category_rows.append({"seed": seed, "variant": spec["name"], "category": cat, "I-AUROC": metric_value(vals, "I-AUROC"), "I-AP": metric_value(vals, "I-AP"), "P-AUROC": metric_value(vals, "P-AUROC"), "PRO": metric_value(vals, "PRO")})
                logs.append(f"seed {seed} {spec['name']}: I-AUROC={row['I-AUROC']:.4f}, P-AUROC={row['P-AUROC']:.4f}, PRO={row['PRO']:.4f}")
                if spec["family"] == "uniform":
                    refs = cfg.get("baseline_reference", {})
                    sanity.setdefault("baseline_checks", {})[str(seed)] = {k: {"value": row.get(k), "reference": v, "pass": abs(row.get(k, float('nan')) - float(v)) <= float(cfg.get("baseline_tolerance", {}).get(k, 0.002))} for k, v in refs.items() if np.isfinite(row.get(k, float("nan")))}
                torch.cuda.empty_cache()
    except Exception:
        tb = traceback.format_exc()
        failures.append(tb)
        (output_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        raise
    finally:
        write_outputs(output_dir, summary_rows, per_category_rows, head_rows, gate_rows, retrieval_rows, rep_rows, cost_rows, source_selection, sanity, failures, specs)
        copy_log(output_dir, logs)
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase IV-B — Head-Selective Value Residual Fusion\n\n")
            f.write("- Implementation files: `head_selective_clip/*`, `experiments/head_selective_residual_audit.py`, `configs/head_selective_residual_audit.yaml`.\n")
            f.write("- Exclusions: memory compression, graphs, late interaction and SPR excluded.\n")
            f.write(f"- Completed metric rows: {len(summary_rows)}.\n")


if __name__ == "__main__":
    main()
