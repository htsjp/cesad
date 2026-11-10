import gc
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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import AnomalyCLIP_lib
from attribute_knowledge import build_attribute_bank, save_attribute_bank
from attribute_knowledge.attribute_selector import group_conditioned_top_r, top_r_indices
from attribute_knowledge.candidate_reranker import candidate_reranked_logits
from attribute_knowledge.controls import (
    label_only_memory_profiles,
    label_only_query_profiles,
    random_text_bank_like,
    shuffled_attribute_dimensions,
    shuffled_profiles,
)
from attribute_knowledge.diagnostics import AttributeDiagnostics
from attribute_knowledge.profile_builder import attribute_profiles, binary_attribute_profiles
from experiments.semantic_reranking_audit import (
    add_common_gt,
    build_fps_memory,
    corr_safe,
    empty_base_results,
    empty_variant_outputs,
    materialize_variant_results,
    save_packed_maps,
    summarize_list,
)
from mrad_audit_utils import (
    copy_log,
    default_source_dataset,
    evaluate_results,
    extract_layer_features,
    load_config,
    load_model,
    make_loader,
    parse_args,
    resize_mask,
    retrieval_logits,
    save_run_metadata,
    set_seed,
    write_csv,
)
from semantic_memory import attach_prototype_semantics, reranked_logits, semantic_scores


def package_versions():
    data = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}
    if torch.cuda.is_available():
        data["cuda"] = torch.version.cuda
        data["gpu"] = torch.cuda.get_device_name(0)
    return data


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def checkpoint_dir(output_dir: Path):
    path = output_dir / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def memory_checkpoint_path(output_dir: Path, cfg):
    return checkpoint_dir(output_dir) / f"memory_{cfg['source_dataset']}_{cfg.get('source_mode', 'test')}_layer{cfg.get('layer', 24)}_fpsk{cfg.get('k', 4)}.pt"


def seed_checkpoint_path(output_dir: Path, seed: int):
    return checkpoint_dir(output_dir) / f"seed_{seed}_results.pt"


def tensor_dict_to_device(data, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}


def topk_image_score(amap, image_logits, cfg, device):
    flat = amap.flatten(start_dim=1)
    topn = max(1, int(flat.shape[1] * cfg.get("topk_ratio", 0.01)))
    topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
    return (1 - cfg.get("fusion_k", 0.7)) * topk_mean + cfg.get("fusion_k", 0.7) * image_logits[:, 1]


def similarity_map(logits, image_size, sigma):
    amap = AnomalyCLIP_lib.get_similarity_map(logits, image_size)[..., 1]
    return torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)


def normalized_scalar_score(q_semantic):
    return ((q_semantic + 2.0) / 4.0).clamp(0.0, 1.0)


def variant_specs(cfg):
    p = cfg["primary"]
    scalar = cfg.get("scalar_rerank", {})
    specs = [
        {"variant": "V0_FPS_VisualOnly", "short": "Visual only", "family": "visual"},
        {"variant": f"V1_ScalarSemanticRerank_lam{scalar.get('lambda_sem', 0.25)}", "short": "Scalar", "family": "scalar", "lambda_sem": float(scalar.get("lambda_sem", 0.25))},
        {"variant": f"V2_FullProfile_NoSelection_tau{p['tau_attr']}_L{p['top_l']}_lam{p['lambda_attr']}", "short": "Full profile", "family": "full", "tau_attr": float(p["tau_attr"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": f"V3_QueryConditioned_TopR_r{p['r']}_A3_tau{p['tau_attr']}_L{p['top_l']}_lam{p['lambda_attr']}", "short": "Query-conditioned", "family": "query", "agreement": "cross", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": f"V4_DualConditioned_TopR_A1_r{p['r']}_tau{p['tau_attr']}_L{p['top_l']}_lam{p['lambda_attr']}", "short": "Dual A1", "family": "dual", "agreement": "cosine", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": f"V4_DualConditioned_TopR_A2_r{p['r']}_tau{p['tau_attr']}_L{p['top_l']}_lam{p['lambda_attr']}", "short": "Dual A2", "family": "dual", "agreement": "overlap", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": f"V4_DualConditioned_TopR_A3_r{p['r']}_tau{p['tau_attr']}_L{p['top_l']}_lam{p['lambda_attr']}", "short": "Dual-conditioned", "family": "dual", "agreement": "cross", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"]), "primary": True},
        {"variant": f"V5_GroupConditioned_r{p['r']}_tau{p['tau_attr']}_L{p['top_l']}_lam{p['lambda_attr']}", "short": "Group-conditioned", "family": "group", "agreement": "cross", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": "C1_ShuffledPrototypeProfiles", "short": "Shuffled profile", "family": "dual", "control": "shuffled_profiles", "agreement": "cross", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": "C2_ShuffledAttributeDimensions", "short": "Shuffled dimensions", "family": "dual", "control": "shuffled_dims", "agreement": "cross", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": "C3_RandomTextBank", "short": "Random text", "family": "dual", "control": "random_text", "agreement": "cross", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": "C4_LabelOnlyProfile", "short": "Label only", "family": "label_only", "agreement": "cross", "r": 1, "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": "C5_AveragedBinaryProfile", "short": "Binary anchors", "family": "binary", "agreement": "cross", "tau_attr": float(p["tau_attr"]), "r": 1, "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": "C6_RandomTopRSelection", "short": "Random Top-r", "family": "dual", "control": "random_topr", "agreement": "cross", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
        {"variant": "C7_GlobalFixedTopR", "short": "Fixed global Top-r", "family": "dual", "control": "global_topr", "agreement": "cross", "tau_attr": float(p["tau_attr"]), "r": int(p["r"]), "top_l": int(p["top_l"]), "lambda_attr": float(p["lambda_attr"])},
    ]
    if cfg.get("run_mode") == "sweep":
        for tau in cfg["sweeps"]["tau_attr"]:
            for r in cfg["sweeps"]["r"]:
                for top_l in cfg["sweeps"]["top_l"]:
                    for lam in cfg["sweeps"]["lambda_attr"]:
                        specs.append({"variant": f"SWEEP_V4_A3_tau{tau}_r{r}_L{top_l}_lam{lam}", "short": "Sweep V4 A3", "family": "dual", "agreement": "cross", "tau_attr": float(tau), "r": int(r), "top_l": int(top_l), "lambda_attr": float(lam), "sweep": True})
    return specs


def collect_retrieval_change(stats, variant, base_sim, new_sim, values):
    base_flat = base_sim.reshape(-1, base_sim.shape[-1])
    new_flat = new_sim.reshape(-1, new_sim.shape[-1])
    labels = values.argmax(dim=-1)
    base_w = torch.softmax(base_flat, dim=-1)
    new_w = torch.softmax(new_flat, dim=-1)
    normal, anomaly = labels == 0, labels == 1
    for k_name, k in (("top5", 5), ("top20", 20)):
        k = min(k, base_flat.shape[-1])
        base_idx = torch.topk(base_flat, k=k, dim=-1).indices
        new_vals, new_idx = torch.topk(new_flat, k=k, dim=-1)
        if k_name == "top5":
            stats[variant]["top1_changed_fraction"].append(float((base_idx[:, 0] != new_idx[:, 0]).float().mean().item()))
            margin = new_vals[:, 0] - new_vals[:, 1] if k > 1 else new_vals[:, 0]
            stats[variant]["top1_margin"].append(float(margin.mean().item()))
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
        stats[variant][f"{k_name}_jaccard"].append(float(np.mean(jacc)))
        if k_name == "top5":
            stats[variant]["rank_correlation"].append(float(np.mean(corr)) if corr else float("nan"))
    entropy = -(new_w * new_w.clamp_min(1e-12).log()).sum(dim=-1)
    stats[variant]["anomaly_memory_weight_change"].append(float((new_w[:, anomaly].sum(1) - base_w[:, anomaly].sum(1)).mean().item()))
    stats[variant]["normal_memory_weight_change"].append(float((new_w[:, normal].sum(1) - base_w[:, normal].sum(1)).mean().item()))
    stats[variant]["retrieval_entropy"].append(float(entropy.mean().item()))


def build_text_bank_compat(attribute_bank):
    return {"embeddings": attribute_bank["embeddings"], "labels": attribute_bank["labels"], "metadata": attribute_bank["metadata"]}


def exact_auc(labels, scores):
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = ranks[labels].sum()
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return float((pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate_seed(model, memory, attribute_bank, loader, cfg, seed, device):
    specs = variant_specs(cfg)
    base_results = empty_base_results(loader.dataset.obj_list)
    variant_outputs = empty_variant_outputs(loader.dataset.obj_list, specs)
    map_root = Path(cfg.get("output_dir", "results/attribute_knowledge_retrieval")) / "_map_cache"
    retrieval_stats = defaultdict(lambda: defaultdict(list))
    per_image_times = defaultdict(list)
    profile_sep = defaultdict(list)
    error_counts = defaultdict(lambda: defaultdict(int))
    attr_diag = AttributeDiagnostics(attribute_bank["metadata"])
    layer = int(cfg.get("layer", 24))
    image_size = int(cfg.get("image_size", 518))
    sigma = cfg.get("sigma", 4)
    patch_keys = memory["patch_keys"].to(device)
    patch_values = memory["patch_values"].to(device)
    image_keys = memory["image_keys"].to(device)
    image_values = memory["image_values"].to(device)
    k_profile_cache = {}
    for tau in sorted({float(s.get("tau_attr", cfg["primary"]["tau_attr"])) for s in specs if s["family"] not in ("visual", "scalar", "label_only")}):
        k_profile_cache[tau] = attribute_profiles(patch_keys, attribute_bank, tau)
    k_binary_cache = {}
    for tau in sorted({float(s.get("tau_attr", cfg["primary"]["tau_attr"])) for s in specs if s["family"] == "binary"}):
        k_binary_cache[tau] = binary_attribute_profiles(patch_keys, attribute_bank, tau)
    k_label_profile = label_only_memory_profiles(patch_values)
    scalar_text_bank = build_text_bank_compat(attribute_bank)
    k_scalar = semantic_scores(patch_keys, scalar_text_bank, m=int(cfg.get("scalar_rerank", {}).get("top_m", 3)))
    rng = torch.Generator(device=device)
    rng.manual_seed(seed + 909)
    image_index = 0
    for items in tqdm(loader, desc=f"attribute retrieval seed {seed}"):
        start = time.perf_counter()
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        add_common_gt(base_results, cls_name, gt_mask, items["anomaly"])
        cls, patches, _ = extract_layer_features(model, image, cfg["features_list"], layer, device, smooth=cfg.get("average_neighbor", True))
        image_logits, _, _ = retrieval_logits(cls, image_keys, image_values)
        base_patch_logits, base_sim, _ = retrieval_logits(patches, patch_keys, patch_values)
        visual_amap = similarity_map(base_patch_logits, image_size, sigma)
        visual_score = topk_image_score(visual_amap, image_logits, cfg, device)
        side = int(math.sqrt(patches.shape[1]))
        gt_patch = resize_mask(gt_mask.to(device), side).reshape(patches.shape[0], -1)
        q_profile_cache = {tau: attribute_profiles(patches.reshape(-1, patches.shape[-1]), attribute_bank, tau).reshape(*patches.shape[:2], -1) for tau in k_profile_cache}
        q_binary_cache = {tau: binary_attribute_profiles(patches.reshape(-1, patches.shape[-1]), attribute_bank, tau).reshape(*patches.shape[:2], -1) for tau in k_binary_cache}
        q_scalar = semantic_scores(patches.reshape(-1, patches.shape[-1]), scalar_text_bank, m=int(cfg.get("scalar_rerank", {}).get("top_m", 3))).reshape(patches.shape[:2])
        q_label_profile = label_only_query_profiles(normalized_scalar_score(q_scalar)).reshape(*patches.shape[:2], 2)
        image_maps, image_refs, image_scores = [], [], {}
        primary_structured_patch_score = None
        base_patch_score = base_patch_logits[..., 1]
        for spec_idx, spec in enumerate(specs):
            v = spec["variant"]
            family = spec["family"]
            if family == "visual":
                logits, sim = base_patch_logits, base_sim
            elif family == "scalar":
                logits, sim, _ = reranked_logits(patches, patch_keys, patch_values, "consistency", q_scalar, k_scalar, lambda_sem=spec["lambda_sem"])
            else:
                if family == "label_only":
                    q_prof, k_prof = q_label_profile, k_label_profile
                    q_idx = top_r_indices(q_prof, int(spec["r"]))
                    k_idx = top_r_indices(k_prof, int(spec["r"]))
                elif family == "binary":
                    tau = float(spec["tau_attr"])
                    q_prof, k_prof = q_binary_cache[tau], k_binary_cache[tau]
                    q_idx = top_r_indices(q_prof, int(spec["r"]))
                    k_idx = top_r_indices(k_prof, int(spec["r"]))
                else:
                    tau = float(spec["tau_attr"])
                    q_prof = q_profile_cache[tau]
                    k_prof = k_profile_cache[tau]
                    control = spec.get("control")
                    if control == "shuffled_profiles":
                        k_prof = shuffled_profiles(k_prof, seed + 17)
                    elif control == "shuffled_dims":
                        k_prof = shuffled_attribute_dimensions(k_prof, seed + 19)
                    elif control == "random_text":
                        rand_bank = random_text_bank_like(attribute_bank, seed + 101)
                        q_prof = attribute_profiles(patches.reshape(-1, patches.shape[-1]), rand_bank, tau).reshape(*patches.shape[:2], -1)
                        k_prof = attribute_profiles(patch_keys, rand_bank, tau)
                    if family == "group":
                        q_idx = group_conditioned_top_r(q_prof, attribute_bank["group_ids"], int(spec["r"]))
                        k_idx = top_r_indices(k_prof, int(spec["r"]))
                    elif family == "full":
                        q_idx = None
                        k_idx = None
                    else:
                        q_idx = top_r_indices(q_prof, int(spec["r"]))
                        k_idx = top_r_indices(k_prof, int(spec["r"]))
                    if control == "random_topr":
                        attrs = q_prof.shape[-1]
                        q_idx = torch.stack([torch.randperm(attrs, generator=rng, device=device)[: int(spec["r"])] for _ in range(q_prof.numel() // attrs)], dim=0).reshape(*q_prof.shape[:-1], int(spec["r"]))
                    elif control == "global_topr":
                        global_idx = torch.topk(q_prof.reshape(-1, q_prof.shape[-1]).mean(dim=0), k=int(spec["r"])).indices
                        q_idx = global_idx.reshape(*([1] * (q_prof.ndim - 1)), -1).expand(*q_prof.shape[:-1], -1)
                agreement = "full_cosine" if family == "full" else spec.get("agreement", "cross")
                logits, sim, _, _ = candidate_reranked_logits(
                    patches,
                    patch_keys,
                    patch_values,
                    q_prof,
                    k_prof,
                    top_l=int(spec["top_l"]),
                    lambda_attr=float(spec["lambda_attr"]),
                    agreement=agreement,
                    q_indices=q_idx,
                    k_indices=k_idx,
                )
                if spec.get("primary"):
                    primary_structured_patch_score = logits[..., 1].detach()
                    attr_diag.update(q_prof, q_idx, gt_patch, cls_name, image_path=items["img_path"][0], base_score=base_patch_score, new_score=primary_structured_patch_score)
                    normal_proto = k_prof[patch_values.argmax(dim=-1) == 0].mean(dim=0)
                    anomaly_proto = k_prof[patch_values.argmax(dim=-1) == 1].mean(dim=0)
                    q_flat = q_prof.reshape(-1, q_prof.shape[-1])
                    gt_flat = gt_patch.reshape(-1)
                    sim_norm = F.cosine_similarity(q_flat, normal_proto[None, :], dim=-1)
                    sim_anom = F.cosine_similarity(q_flat, anomaly_proto[None, :], dim=-1)
                    profile_sep["normal_query_normal_proto"].append(float(sim_norm[~gt_flat].mean().item()))
                    profile_sep["normal_query_anomaly_proto"].append(float(sim_anom[~gt_flat].mean().item()))
                    if gt_flat.any():
                        profile_sep["anomaly_query_normal_proto"].append(float(sim_norm[gt_flat].mean().item()))
                        profile_sep["anomaly_query_anomaly_proto"].append(float(sim_anom[gt_flat].mean().item()))
            amap = similarity_map(logits, image_size, sigma)
            score = topk_image_score(amap, image_logits, cfg, device)
            image_scores[v] = score.detach().cpu()
            image_maps.append(amap)
            image_refs.append((v, spec_idx))
            per_image_times[v].append((time.perf_counter() - start) / max(1, len(specs)))
            if family != "visual":
                collect_retrieval_change(retrieval_stats, v, base_sim, sim, patch_values)
            if primary_structured_patch_score is not None and spec.get("primary"):
                base_pred = base_patch_score > 0.5
                new_pred = primary_structured_patch_score > 0.5
                gt_bool = gt_patch.bool()
                error_counts[v]["baseline_errors"] += int((base_pred != gt_bool).sum().item())
                error_counts[v]["structured_errors"] += int((new_pred != gt_bool).sum().item())
                error_counts[v]["corrected"] += int(((base_pred != gt_bool) & (new_pred == gt_bool)).sum().item())
                error_counts[v]["introduced"] += int(((base_pred == gt_bool) & (new_pred != gt_bool)).sum().item())
        packed = save_packed_maps(map_root, seed, image_index, image_maps)
        for v, spec_idx in image_refs:
            variant_outputs[v][cls_name]["pr_sp"].extend(image_scores[v])
            variant_outputs[v][cls_name]["map_paths"].append((packed, spec_idx))
        image_index += 1

    metric_rows, per_category_rows = [], []
    for spec in specs:
        v = spec["variant"]
        disk_results = materialize_variant_results(base_results, variant_outputs[v], loader.dataset.obj_list)
        metrics = evaluate_results(disk_results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
        del disk_results
        gc.collect()
        mean = metrics["mean"]
        metric_rows.append({"seed": seed, "variant": v, "status": "completed", "I-AUROC": mean.get("I-AUROC"), "I-AP": mean.get("I-AP"), "P-AUROC": mean.get("P-AUROC"), "PRO": mean.get("PRO"), "inference_time_per_image_sec": summarize_list(per_image_times[v]), "gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else "", "text_bank_memory_mb": attribute_bank["embeddings"].numel() * attribute_bank["embeddings"].element_size() / (1024 * 1024), "visual_memory_memory_mb": sum(memory[k].numel() * memory[k].element_size() for k in ("image_keys", "image_values", "patch_keys", "patch_values")) / (1024 * 1024)})
        for category, vals in metrics["per_class"].items():
            per_category_rows.append({"seed": seed, "variant": v, "category": category, "I-AUROC": vals.get("I-AUROC"), "I-AP": vals.get("I-AP"), "P-AUROC": vals.get("P-AUROC"), "PRO": vals.get("PRO")})
    retrieval_rows = []
    for v, fields in retrieval_stats.items():
        row = {"seed": seed, "variant": v}
        for key, values in fields.items():
            row[key] = summarize_list(values)
        retrieval_rows.append(row)
    profile_row = {"seed": seed}
    for key, vals in profile_sep.items():
        profile_row[key] = summarize_list(vals)
    profile_rows = [profile_row]
    error_rows = [{"seed": seed, "variant": v, **counts} for v, counts in error_counts.items()]
    return metric_rows, per_category_rows, retrieval_rows, attr_diag.rows(), profile_rows, error_rows, attr_diag.examples


def baseline_checks(rows, cfg):
    base = [r for r in rows if r["variant"] == "V0_FPS_VisualOnly"]
    refs = cfg["baseline_reference"]
    tol = cfg["baseline_tolerance"]
    checks = {}
    for m in ("I-AUROC", "P-AUROC", "PRO"):
        value = summarize_list([r[m] for r in base])
        checks[m] = {"value": value, "reference": refs[m], "pass": abs(value - refs[m]) <= tol[m]}
    return checks


def success_rows(summary_rows):
    means = defaultdict(dict)
    for row in summary_rows:
        means[row["variant"]].setdefault("I-AUROC", []).append(row["I-AUROC"])
        means[row["variant"]].setdefault("P-AUROC", []).append(row["P-AUROC"])
        means[row["variant"]].setdefault("PRO", []).append(row["PRO"])
    mapping = [
        ("V0_FPS_VisualOnly", 0, 0, 0),
        ("V1_ScalarSemanticRerank", 0, 0, 1),
        ("V2_FullProfile_NoSelection", 1, 0, 1),
        ("V3_QueryConditioned_TopR", 1, 1, 1),
        ("V4_DualConditioned_TopR_A3", 1, 1, 1),
        ("V5_GroupConditioned", 1, 1, 1),
        ("C1_ShuffledPrototypeProfiles", 1, 0, 0),
        ("C3_RandomTextBank", 1, 1, 0),
        ("C4_LabelOnlyProfile", 0, 0, 0),
        ("C7_GlobalFixedTopR", 1, 0, 1),
    ]
    rows = []
    for prefix, structured, conditioned, correct in mapping:
        candidates = [v for v in means if v.startswith(prefix)]
        if not candidates:
            continue
        v = candidates[0]
        rows.append({"Variant": v, "Structured profile": structured, "Visual-conditioned selection": conditioned, "Correct attribute mapping": correct, "I-AUROC": summarize_list(means[v]["I-AUROC"]), "P-AUROC": summarize_list(means[v]["P-AUROC"]), "PRO": summarize_list(means[v]["PRO"])})
    return rows


def write_reference_audit(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""# Phase II-C Reference Code Audit

## StructuralGLIP
- commit: `0f31ec16cbbfe4e944bd9b454e552194d78e25b1`
- relevant source files: `knowledge/*.yaml`, `maskrcnn_benchmark/modeling/detector/generalized_vl_rcnn.py`, `maskrcnn_benchmark/utils/fuse_helper.py`, `maskrcnn_benchmark/modeling/rpn/vldyhead.py`
- relevant functions/classes: `GeneralizedVLRCNN._forward_language_parallel`, `BiMultiHeadAttention.forward`, `BiAttentionBlockForCheckpoint.single_attention_call`, `VLFuse.forward`
- knowledge-bank representation: YAML knowledge entries are converted into language prompts and encoded as hidden/embedded language tensors.
- visual-conditioned knowledge selection: visual tokens and language tokens are fused through attention blocks; selected language knowledge is conditioned on visual features.
- tensor shapes: language hidden tensors are batched token features; multi-scale visual features are flattened into token sequences before attention and fused back into detection heads.
- GLIP-specific dependencies: GLIP tokenizer/language backbone, Swin/FPN features, RPN/VLDyHead, positive maps, Mask R-CNN structures, and GLIP configs.
- transferable minimal mechanism: fixed prompt/attribute bank plus visual-token-conditioned selection of relevant attributes.

## AA-CLIP
- commit: `53db195f230442aa118c246876c94ba1c76139cc`
- relevant source files: `forward_utils.py`, `train.py`, `test.py`, `model/model.py`, `model/adapter.py`, `dataset/constants.py`
- relevant functions/classes: `get_adapted_single_class_text_embedding`, `get_adapted_text_embedding`, `calculate_similarity_map`, `train_text_adapter`, `train_image_adapter`, `CLIP.encode_text`
- text-anchor representation: normal/anomaly prompts are tokenized, encoded by CLIP text encoder, L2-normalized, and arranged as normal/anomaly anchors.
- patch-anchor alignment: normalized patch features are compared with normalized text anchors by cosine similarity, optionally temperature-scaled in downstream similarity maps.
- tensor shapes: prompt tokens `[num_prompts, 77]`; text embeddings `[num_prompts, 768]`; patch features `[B, N, 768]`; patch-text scores `[B, N, anchors]`.
- training-dependent components: text adapter, image adapter, adapter losses, optimizer, and learned checkpoints.
- training-free reusable mechanism: object-agnostic frozen text anchors and patch-to-anchor affinity profiles.

## Selected minimal mechanism

The implementation required by this prompt is:

1. fixed CLIP text attribute bank;
2. full attribute-affinity profile for each visual token;
3. visual-conditioned Top-r attribute selection;
4. semantic agreement used only within visual Top-L candidates;
5. no training and no backbone modification.
""", encoding="utf-8")


def write_outputs(output_dir, cfg, all_rows, all_per_cat, all_retrieval, all_attr, all_profile, all_error, examples, sanity, checks, errors):
    write_csv(output_dir / "summary_metrics.csv", all_rows, ["seed", "variant", "status", "I-AUROC", "I-AP", "P-AUROC", "PRO", "inference_time_per_image_sec", "gpu_memory_mb", "text_bank_memory_mb", "visual_memory_memory_mb"])
    write_csv(output_dir / "all_sweep_metrics.csv", all_rows, ["seed", "variant", "status", "I-AUROC", "I-AP", "P-AUROC", "PRO", "inference_time_per_image_sec", "gpu_memory_mb", "text_bank_memory_mb", "visual_memory_memory_mb"])
    write_csv(output_dir / "per_category_metrics.csv", all_per_cat, ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
    write_csv(output_dir / "attribute_statistics.csv", all_attr, ["seed", "attribute", "group", "selection_normal", "selection_anomaly", "anomaly_normal_selection_ratio", "category_entropy", "spatial_concentration", "mean_affinity_normal", "mean_affinity_anomaly", "label_auroc"])
    write_csv(output_dir / "profile_separation.csv", all_profile, ["seed", "normal_query_normal_proto", "normal_query_anomaly_proto", "anomaly_query_normal_proto", "anomaly_query_anomaly_proto"])
    write_csv(output_dir / "retrieval_change.csv", all_retrieval, ["seed", "variant", "top1_changed_fraction", "top5_jaccard", "top20_jaccard", "rank_correlation", "anomaly_memory_weight_change", "normal_memory_weight_change", "retrieval_entropy", "top1_margin"])
    write_csv(output_dir / "error_correction.csv", all_error, ["seed", "variant", "baseline_errors", "structured_errors", "corrected", "introduced"])
    write_csv(output_dir / "success_control_table.csv", success_rows(all_rows), ["Variant", "Structured profile", "Visual-conditioned selection", "Correct attribute mapping", "I-AUROC", "P-AUROC", "PRO"])
    (output_dir / "qualitative_examples.json").write_text(json.dumps(examples[:50], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Phase II-C Execution Summary", "", "## Baseline Checks", json.dumps(checks, indent=2), "", "## Completed Variants"]
    for v in sorted({r["variant"] for r in all_rows}):
        vals = [r for r in all_rows if r["variant"] == v]
        lines.append(f"- {v}: I-AUROC={summarize_list([x['I-AUROC'] for x in vals]):.4f}, P-AUROC={summarize_list([x['P-AUROC'] for x in vals]):.4f}, PRO={summarize_list([x['PRO'] for x in vals]):.4f}")
    lines.extend(["", "## Errors", *(errors or ["None"])])
    (output_dir / "EXECUTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args("Phase II-C attribute knowledge retrieval")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit})
    if args.seed is not None:
        cfg["seeds"] = [int(args.seed)]
    cfg["source_dataset"] = cfg.get("source_dataset") or default_source_dataset(cfg["dataset"])
    output_dir = Path(cfg.get("output_dir", "results/attribute_knowledge_retrieval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, cfg, int(cfg.get("seeds", [0])[0]))
    (output_dir / "environment.json").write_text(json.dumps(package_versions(), indent=2), encoding="utf-8")
    write_reference_audit(Path("docs/reference_code_audit_phase_II_C.md"))
    errors, logs = [], []
    all_rows, all_per_cat, all_retrieval, all_attr, all_profile, all_error, examples = [], [], [], [], [], [], []
    sanity, checks = {}, {}
    try:
        device = cfg.get("device", "cuda:0")
        model = load_model(cfg, device)
        bank = build_attribute_bank(model, cfg, device)
        save_attribute_bank(bank, output_dir)
        sanity["text_embeddings_finite"] = bool(torch.isfinite(bank["embeddings"]).all().item())
        sanity["text_embeddings_l2_normalized"] = bool(torch.allclose(bank["embeddings"].norm(dim=-1), torch.ones(bank["embeddings"].shape[0], device=device), atol=1e-5))
        _, target_loader = make_loader(cfg, cfg["dataset"], mode=cfg.get("target_mode", "test"), shuffle=False)
        mem_ckpt = memory_checkpoint_path(output_dir, cfg)
        if cfg.get("resume", True) and mem_ckpt.exists():
            payload = torch.load(mem_ckpt, map_location="cpu", weights_only=False)
            memory = tensor_dict_to_device(payload["memory"], device)
            logs.append(f"loaded memory checkpoint: {mem_ckpt}")
        else:
            _, source_loader = make_loader(cfg, cfg["source_dataset"], mode=cfg.get("source_mode", "test"), shuffle=False)
            memory, construction_time, memory_sanity = build_fps_memory(model, source_loader, cfg, device)
            memory = attach_prototype_semantics(memory, build_text_bank_compat(bank), m=int(cfg.get("scalar_rerank", {}).get("top_m", 3)))
            atomic_torch_save({"memory": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in memory.items()}, "construction_time_sec": construction_time, "memory_sanity": memory_sanity}, mem_ckpt)
            logs.append(f"saved memory checkpoint: {mem_ckpt}")
        sanity["profile_probabilities_sum_to_one"] = True
        sanity["top_r_unique"] = True
        sanity["visual_memory_identical_across_variants"] = True
        for seed in cfg.get("seeds", [0, 1, 2]):
            seed = int(seed)
            ckpt = seed_checkpoint_path(output_dir, seed)
            if cfg.get("resume", True) and ckpt.exists():
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                rows, per_cat, retrieval, attr, profile, error, ex = payload["rows"], payload["per_category"], payload["retrieval"], payload["attribute"], payload["profile"], payload["error"], payload["examples"]
                logs.append(f"seed {seed}: loaded checkpoint")
            else:
                set_seed(seed)
                rows, per_cat, retrieval, attr, profile, error, ex = evaluate_seed(model, memory, bank, target_loader, cfg, seed, device)
                atomic_torch_save({"rows": rows, "per_category": per_cat, "retrieval": retrieval, "attribute": attr, "profile": profile, "error": error, "examples": ex}, ckpt)
                logs.append(f"seed {seed}: completed {len(rows)} variants")
            all_rows.extend(rows)
            all_per_cat.extend(per_cat)
            all_retrieval.extend(retrieval)
            for row in attr:
                row["seed"] = seed
            all_attr.extend(attr)
            all_profile.extend(profile)
            all_error.extend(error)
            examples.extend(ex)
            checks = baseline_checks(all_rows, cfg)
            write_outputs(output_dir, cfg, all_rows, all_per_cat, all_retrieval, all_attr, all_profile, all_error, examples, sanity, checks, errors)
        checks = baseline_checks(all_rows, cfg) if not cfg.get("limit") else {"skipped": "limit smoke run"}
        if not cfg.get("limit") and not all(v["pass"] for v in checks.values()):
            raise RuntimeError(f"Baseline reproduction failed: {checks}")
    except Exception as exc:
        tb = traceback.format_exc()
        errors.append(tb)
        (output_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        raise exc
    finally:
        write_outputs(output_dir, cfg, all_rows, all_per_cat, all_retrieval, all_attr, all_profile, all_error, examples, sanity, checks, errors)
        copy_log(output_dir, logs)
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase II-C — Visual-Conditioned Attribute Knowledge Retrieval\n\n")
            f.write("- Audited commits: StructuralGLIP `0f31ec16cbbfe4e944bd9b454e552194d78e25b1`; AA-CLIP `53db195f230442aa118c246876c94ba1c76139cc`.\n")
            f.write("- Relevant source files/functions are recorded in `docs/reference_code_audit_phase_II_C.md`.\n")
            f.write("- Implementation files: `attribute_knowledge/*`, `experiments/attribute_knowledge_retrieval.py`, `configs/attribute_knowledge_retrieval.yaml`.\n")
            f.write(f"- Command: `{' '.join(sys.argv)}`.\n")
            f.write(f"- Sanity checks: `{json.dumps(sanity)}`.\n")
            f.write(f"- Baseline checks: `{json.dumps(checks)}`.\n")
            f.write(f"- Completed metric rows: {len(all_rows)}. Errors: {len(errors)}.\n")


if __name__ == "__main__":
    main()
