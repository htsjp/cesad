import json
import math
import os
import platform
import subprocess
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
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
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
from multiview_memory.alignment import build_alignment_entry, fps_indices, fused_baseline_view, summarize_alignment
from multiview_memory.diagnostics import (
    distribution_summary,
    memory_organization,
    retrieval_overlap_rows,
    view_geometry_rows,
)
from multiview_memory.late_interaction import (
    distribution_from_sims,
    early_concat,
    early_fusion_matched,
    logits_from_distribution,
    value_mean,
)
from multiview_memory.similarity import maxsim_scores, mismatch_similarities, view_similarities
from multiview_memory.tuple_memory import cpu_state, label_shuffle, memory_storage_mb, repeated_t24, shuffled_memory_views, to_device


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


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def checkpoint_dir(output_dir: Path) -> Path:
    p = output_dir / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def aligned_memory_path(output_dir: Path, seed: int) -> Path:
    return checkpoint_dir(output_dir) / f"seed_{seed}_aligned_memory.pt"


def metrics_path(output_dir: Path, seed: int, variant: str) -> Path:
    return output_dir / f"seed_{seed}" / variant / "metrics.json"


def variant_safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(",", "_")


def weights_name(prefix, weights):
    return f"{prefix}_w{weights[0]:g}_{weights[1]:g}_{weights[2]:g}"


def variant_specs(cfg):
    weight_sets = [tuple(float(x) for x in w) for w in cfg.get("weight_sets", [[1.0, 0.5, 0.5]])]
    concat_weights = [tuple(float(x) for x in w) for w in cfg.get("concat_weight_sets", [[1.0, 0.5, 0.5]])]
    c_values = [int(x) for x in cfg.get("rrf_c_values", [10, 30, 60])]
    temps = [float(x) for x in cfg.get("default_temperatures", [0.07, 0.07, 0.07])]
    specs = [
        {"name": "V0_EarlyFusionBaseline", "family": "v0", "multi_view_tuple": False, "correct_alignment": True, "late_interaction": False, "full_bank": True},
        {"name": "V1_T24Only", "family": "single_t24", "multi_view_tuple": False, "correct_alignment": True, "late_interaction": False, "full_bank": True},
        {"name": "V2_ValueMeanOnly", "family": "value_mean", "multi_view_tuple": False, "correct_alignment": True, "late_interaction": False, "full_bank": True},
    ]
    for w in concat_weights:
        specs.append({"name": weights_name("V3_EarlyConcatenation", w), "family": "early_concat", "weights": w, "multi_view_tuple": True, "correct_alignment": True, "late_interaction": False, "full_bank": True})
    for family, prefix in (("aligned_logit", "V4_AlignedLogitLateInteraction"), ("mixture", "V5_DistributionMixture"), ("product", "V6_DistributionProduct")):
        for w in weight_sets:
            specs.append({"name": weights_name(prefix, w), "family": family, "weights": w, "temperatures": temps, "multi_view_tuple": True, "correct_alignment": True, "late_interaction": True, "full_bank": True})
    for c in c_values:
        specs.append({"name": f"V7_RankReciprocalFusion_c{c}", "family": "rrf", "weights": (1.0, 0.5, 0.5), "c": c, "temperatures": temps, "multi_view_tuple": True, "correct_alignment": True, "late_interaction": True, "full_bank": True})
    specs.append({"name": "V8_AlignedMaxSim", "family": "maxsim", "multi_view_tuple": True, "correct_alignment": True, "late_interaction": True, "full_bank": True})
    default_w = tuple(float(x) for x in cfg.get("selected_weights", [1.0, 0.5, 0.5]))
    for family, prefix in (("aligned_logit", "C1_RepeatedT24Tuple"), ("mixture", "C1_RepeatedT24Tuple_Mixture"), ("product", "C1_RepeatedT24Tuple_Product")):
        specs.append({"name": prefix, "family": family, "control": "repeated_t24", "weights": default_w, "temperatures": temps, "multi_view_tuple": True, "correct_alignment": False, "late_interaction": True, "full_bank": True})
    specs.extend([
        {"name": "C2_ShuffledMemoryViewAlignment", "family": "aligned_logit", "control": "shuffle_memory_views", "weights": default_w, "temperatures": temps, "multi_view_tuple": True, "correct_alignment": False, "late_interaction": True, "full_bank": True},
        {"name": "C3_ShuffledQueryViewAlignment", "family": "aligned_logit", "control": "shuffle_query_views", "weights": default_w, "temperatures": temps, "multi_view_tuple": True, "correct_alignment": False, "late_interaction": True, "full_bank": True},
        {"name": "C4_RandomLayerTuple_T24_V6_V10", "family": "aligned_logit", "control": "random_layers", "weights": default_w, "temperatures": temps, "multi_view_tuple": True, "correct_alignment": True, "late_interaction": True, "full_bank": True},
        {"name": "C5_CrossViewMismatch", "family": "aligned_logit", "control": "cross_view_mismatch", "weights": default_w, "temperatures": temps, "multi_view_tuple": True, "correct_alignment": False, "late_interaction": True, "full_bank": True},
        {"name": "C6_IndependentBankAverage", "family": "independent_bank_average", "weights": default_w, "temperatures": temps, "multi_view_tuple": False, "correct_alignment": False, "late_interaction": False, "full_bank": True},
        {"name": "C7_LabelShuffle", "family": "aligned_logit", "control": "label_shuffle", "weights": default_w, "temperatures": temps, "multi_view_tuple": True, "correct_alignment": True, "late_interaction": True, "full_bank": True},
        {"name": "C8_EarlyFusionMatchedWeights", "family": "early_fusion_matched", "weights": default_w, "multi_view_tuple": True, "correct_alignment": True, "late_interaction": False, "full_bank": True},
    ])
    if cfg.get("run_mode") == "smoke":
        keep = set(cfg.get("smoke_variants", [
            "V0_EarlyFusionBaseline",
            "V4_AlignedLogitLateInteraction_w1_0.5_0.5",
            "C1_RepeatedT24Tuple",
            "C2_ShuffledMemoryViewAlignment",
            "C8_EarlyFusionMatchedWeights",
        ]))
        specs = [s for s in specs if s["name"] in keep]
    return specs


def extract_views(outputs):
    return {
        "T24": outputs[24]["T"][:, 1:, :].float(),
        "V12": outputs[12]["V"][:, 1:, :].float(),
        "V18": outputs[18]["V"][:, 1:, :].float(),
        "V6": outputs[6]["V"][:, 1:, :].float() if 6 in outputs else None,
        "V10": outputs[10]["V"][:, 1:, :].float() if 10 in outputs else None,
    }


@torch.no_grad()
def build_aligned_memory(model, loader, cfg, seed, device, output_dir):
    ckpt = aligned_memory_path(output_dir, seed)
    if cfg.get("resume", True) and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        return to_device(state["memory"], device), state["alignment_entries"], state["alignment_summary"], 0.0

    required_layers = sorted(set(int(x) for x in cfg.get("required_layers", [6, 10, 12, 18, 24])))
    k = int(cfg.get("k", 4))
    image_features, image_labels = [], []
    patch_values, entries = [], []
    view_parts = defaultdict(list)
    region_id, memory_id = 0, 0
    start = time.perf_counter()
    with torch.no_grad():
        for items in tqdm(loader, desc=f"build aligned memory seed {seed}"):
            images = items["img"].to(device)
            labels = torch.as_tensor(items["anomaly"], device=device)
            gt = (items["img_mask"].to(device) > 0.5).float()
            outputs = extract_value_representations(model, images, required_layers, device, smooth=cfg.get("average_neighbor", True))
            views = extract_views(outputs)
            cls = cls_from_outputs(outputs, 24)
            image_features.append(cls)
            image_labels.append(labels)
            fused = fused_baseline_view(views["T24"], views["V12"], views["V18"], alpha=cfg.get("baseline_alpha", 0.5))
            side = int(math.sqrt(fused.shape[1]))
            gt_resized = resize_mask(gt, side)
            sid = sample_id(items)
            for b in range(images.shape[0]):
                for mask, region_type, label in ((gt_resized[b], "anomaly", 1), (~gt_resized[b], "normal", 0)):
                    flat_idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
                    if flat_idx.numel() == 0:
                        continue
                    local_fused = fused[b, flat_idx]
                    local_select = fps_indices(local_fused, k)
                    selected_flat = flat_idx[local_select]
                    for idx in selected_flat.detach().cpu().tolist():
                        for key in ("T24", "V12", "V18", "V6", "V10"):
                            if views.get(key) is not None:
                                view_parts[key].append(F.normalize(views[key][b, idx], dim=-1))
                        patch_values.append(label)
                        entries.append(build_alignment_entry(memory_id, sid, region_id, region_type, int(idx), side, label))
                        memory_id += 1
                    region_id += 1

    memory = {
        "image_keys": torch.cat(image_features, dim=0),
        "image_values": one_hot(torch.cat(image_labels, dim=0)).to(device),
        "patch_values": one_hot(torch.tensor(patch_values, device=device, dtype=torch.long)).to(device),
    }
    for key, parts in view_parts.items():
        memory[key] = F.normalize(torch.stack(parts, dim=0), dim=-1).to(device)
    memory["F"] = fused_baseline_view(memory["T24"], memory["V12"], memory["V18"], alpha=cfg.get("baseline_alpha", 0.5))
    summary = summarize_alignment(memory, entries)
    atomic_torch_save({"memory": cpu_state(memory), "alignment_entries": entries, "alignment_summary": summary}, ckpt)
    return memory, entries, summary, time.perf_counter() - start


def query_views_from_outputs(outputs, cfg, control=None, seed=0):
    views = extract_views(outputs)
    if control == "random_layers":
        views = {"T24": views["T24"], "V12": views["V6"], "V18": views["V10"]}
    if control == "shuffle_query_views":
        generator = torch.Generator(device=views["T24"].device)
        for key, off in (("V12", 3), ("V18", 9)):
            generator.manual_seed(int(seed) + off)
            flat = views[key].reshape(-1, views[key].shape[-1])
            perm = torch.randperm(flat.shape[0], generator=generator, device=flat.device)
            views[key] = flat[perm].reshape_as(views[key])
    views["F"] = fused_baseline_view(views["T24"], views["V12"], views["V18"], alpha=cfg.get("baseline_alpha", 0.5))
    return views


def memory_for_spec(memory, spec, seed):
    control = spec.get("control")
    if control == "repeated_t24":
        return repeated_t24(memory)
    if control == "shuffle_memory_views":
        return shuffled_memory_views(memory, seed)
    if control == "label_shuffle":
        return label_shuffle(memory, seed)
    if control == "random_layers":
        out = dict(memory)
        out["V12"] = memory["V6"]
        out["V18"] = memory["V10"]
        return out
    return memory


def patch_probs(query_views, memory, spec):
    family = spec["family"]
    if family == "v0":
        return retrieval_logits(query_views["F"], memory["F"], memory["patch_values"])[2]
    if family == "single_t24":
        return retrieval_logits(query_views["T24"], memory["T24"], memory["patch_values"])[2]
    if family == "value_mean":
        sim = value_mean(query_views, memory)
        return F.softmax(sim, dim=-1)
    if family == "early_concat":
        sim = early_concat(query_views, memory, spec["weights"])
        return F.softmax(sim, dim=-1)
    if family == "early_fusion_matched":
        sim = early_fusion_matched(query_views, memory, spec["weights"])
        return F.softmax(sim, dim=-1)
    if family == "independent_bank_average":
        weights = tuple(float(x) for x in spec.get("weights", [1.0, 0.5, 0.5]))
        norm = sum(weights)
        probs = 0.0
        for key, w, tau in zip(("T24", "V12", "V18"), weights, spec.get("temperatures", [0.07, 0.07, 0.07])):
            sim = query_views[key] @ memory[key].T
            probs = probs + (w / norm) * F.softmax(sim / float(tau), dim=-1)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if family == "maxsim":
        sim = maxsim_scores(query_views, memory)
        return F.softmax(sim / 0.07, dim=-1)
    sims = mismatch_similarities(query_views, memory) if spec.get("control") == "cross_view_mismatch" else view_similarities(query_views, memory)
    return distribution_from_sims(sims, spec)


def defect_group(gt_mask):
    area = float((gt_mask > 0.5).float().mean().item())
    if area <= 0:
        return "normal"
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


@torch.no_grad()
def evaluate_variant(model, memory, loader, cfg, spec, seed, device):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    required_layers = sorted(set(int(x) for x in cfg.get("required_layers", [6, 10, 12, 18, 24])))
    results = init_results(loader.dataset.obj_list)
    groups = ["normal", "small", "medium", "large"]
    defect_results = init_results(groups)
    retrieval_rows, overlap_rows, view_rows, contribution_rows = [], [], [], []
    sim_time, per_image_times = [], []
    process = psutil.Process(os.getpid()) if psutil else None
    peak_cpu = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for index, items in enumerate(tqdm(loader, desc=f"eval {spec['name']}")):
        start = time.perf_counter()
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        outputs = extract_value_representations(model, image, required_layers, device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        query_views = query_views_from_outputs(outputs, cfg, control=spec.get("control"), seed=seed + index)
        if index == 0:
            view_rows.extend(view_geometry_rows(seed, "target_query", {k: query_views[k] for k in ("T24", "V12", "V18")}))
        sim_start = time.perf_counter()
        probs = patch_probs(query_views, memory, spec)
        sim_time.append(time.perf_counter() - sim_start)
        patch_logits = logits_from_distribution(probs, memory["patch_values"])
        image_logits, _, _ = retrieval_logits(cls, memory["image_keys"], memory["image_values"])
        if index < int(cfg.get("diagnostic_batches", 1)):
            sims = view_similarities(query_views, memory)
            overlap_rows.extend(retrieval_overlap_rows(seed, spec["name"], sims))
            retrieval_rows.append({"seed": seed, "variant": spec["name"], **distribution_summary(probs, memory["patch_values"])})
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        group = defect_group(gt_mask)
        defect_results[group]["pr_sp"].extend(score.detach().cpu())
        defect_results[group]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        defect_results[group]["imgs_masks"].append(gt_mask)
        defect_results[group]["anomaly_maps"].append(amap.detach().cpu())
        if spec["name"].startswith("V4_") and index < int(cfg.get("diagnostic_batches", 1)):
            contribution_rows.append({"seed": seed, "variant": spec["name"], "category": cls_name, "score_all": float(score.mean().item()), "best_supporting_view": "posthoc", "conflicting_view": "posthoc"})
        per_image_times.append(time.perf_counter() - start)
        if process:
            peak_cpu = max(peak_cpu, process.memory_info().rss / (1024 * 1024))
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    defect_metrics = evaluate_results(defect_results, groups, compute_pro=cfg.get("compute_pixel_aupro", False))
    runtime = {
        "inference_time_per_image": float(np.mean(per_image_times)) if per_image_times else float("nan"),
        "view_similarity_time_per_image": float(np.mean(sim_time)) if sim_time else float("nan"),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
        "peak_cpu_memory_mb": peak_cpu,
    }
    return metrics, defect_metrics, runtime, retrieval_rows, overlap_rows, view_rows, contribution_rows


def aggregate_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    out = []
    for variant, group in sorted(grouped.items()):
        dst = {"variant": variant}
        for key in ("I-AUROC", "I-AP", "P-AUROC", "PRO"):
            vals = np.asarray([float(r.get(key)) for r in group if r.get(key) is not None and np.isfinite(float(r.get(key)))], dtype=np.float64)
            dst[key] = float(vals.mean()) if vals.size else float("nan")
            dst[f"{key}_std"] = float(vals.std(ddof=0)) if vals.size else float("nan")
        out.append(dst)
    return out


def statistical_comparisons(summary_rows, per_category_rows):
    pairs = [
        ("V4 vs V0", "V4_AlignedLogitLateInteraction_w1_0.5_0.5", "V0_EarlyFusionBaseline"),
        ("V5 vs V0", "V5_DistributionMixture_w1_0.5_0.5", "V0_EarlyFusionBaseline"),
        ("V6 vs V0", "V6_DistributionProduct_w1_0.5_0.5", "V0_EarlyFusionBaseline"),
        ("V7 vs V0", "V7_RankReciprocalFusion_c60", "V0_EarlyFusionBaseline"),
        ("V4 vs C1", "V4_AlignedLogitLateInteraction_w1_0.5_0.5", "C1_RepeatedT24Tuple"),
        ("V4 vs C2", "V4_AlignedLogitLateInteraction_w1_0.5_0.5", "C2_ShuffledMemoryViewAlignment"),
        ("V4 vs C4", "V4_AlignedLogitLateInteraction_w1_0.5_0.5", "C4_RandomLayerTuple_T24_V6_V10"),
        ("V4 vs C5", "V4_AlignedLogitLateInteraction_w1_0.5_0.5", "C5_CrossViewMismatch"),
        ("V4 vs C6", "V4_AlignedLogitLateInteraction_w1_0.5_0.5", "C6_IndependentBankAverage"),
        ("V4 vs C8", "V4_AlignedLogitLateInteraction_w1_0.5_0.5", "C8_EarlyFusionMatchedWeights"),
    ]
    idx = {(r["seed"], r["variant"], r["category"]): r for r in per_category_rows}
    keys = sorted({(r["seed"], r["category"]) for r in per_category_rows})
    out = []
    for label, a, b in pairs:
        for metric in ("I-AUROC", "P-AUROC", "PRO"):
            diffs, wins, ties, losses = [], 0, 0, 0
            for seed, cat in keys:
                ra, rb = idx.get((seed, a, cat)), idx.get((seed, b, cat))
                if not ra or not rb:
                    continue
                va, vb = float(ra.get(metric, float("nan"))), float(rb.get(metric, float("nan")))
                if not np.isfinite(va) or not np.isfinite(vb):
                    continue
                diff = va - vb
                diffs.append(diff)
                wins += int(diff > 1e-6)
                ties += int(abs(diff) <= 1e-6)
                losses += int(diff < -1e-6)
            out.append({"comparison": label, "variant_a": a, "variant_b": b, "metric": metric, "mean_difference": float(np.mean(diffs)) if diffs else float("nan"), "wins": wins, "ties": ties, "losses": losses, "n": len(diffs)})
    return out


def success_control_rows(summary_rows, specs):
    by_variant = {r["variant"]: r for r in aggregate_summary(summary_rows)}
    out = []
    for spec in specs:
        row = {
            "Variant": spec["name"],
            "Multi-view tuple": bool(spec.get("multi_view_tuple")),
            "Correct alignment": bool(spec.get("correct_alignment")),
            "Late interaction": bool(spec.get("late_interaction")),
            "Full bank": bool(spec.get("full_bank")),
        }
        row.update({k: by_variant.get(spec["name"], {}).get(k, float("nan")) for k in ("I-AUROC", "P-AUROC", "PRO")})
        out.append(row)
    return out


def baseline_checks(summary_rows, cfg):
    row = next((r for r in summary_rows if r["variant"] == "V0_EarlyFusionBaseline"), None)
    if not row:
        return {"missing": "V0_EarlyFusionBaseline"}
    checks = {}
    for key, ref in cfg.get("baseline_reference", {}).items():
        tol = float(cfg.get("baseline_tolerance", {}).get(key, 0.002))
        value = float(row.get(key, float("nan")) if row.get(key) is not None else float("nan"))
        checks[key] = {"value": value, "reference": float(ref), "tolerance": tol, "pass": bool(np.isfinite(value) and abs(value - float(ref)) <= tol)}
    return checks


def metric_value(mean, key):
    value = mean.get(key)
    return float(value) if value is not None else float("nan")


def write_outputs(output_dir, summary_rows, per_category_rows, defect_rows, retrieval_rows, overlap_rows, view_rows, contribution_rows, memory_rows, cost_rows, specs, sanity, failures):
    write_csv(output_dir / "summary_metrics.csv", summary_rows, ["seed", "variant", "status", "I-AUROC", "I-AP", "P-AUROC", "PRO", "inference_time_per_image", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "memory_bank_storage_mb", "view_similarity_time_per_image"])
    write_csv(output_dir / "all_sweep_metrics.csv", summary_rows, ["seed", "variant", "status", "I-AUROC", "I-AP", "P-AUROC", "PRO", "inference_time_per_image", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "memory_bank_storage_mb", "view_similarity_time_per_image"])
    write_csv(output_dir / "per_category_metrics.csv", per_category_rows, ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
    write_csv(output_dir / "defect_size_metrics.csv", defect_rows, ["seed", "variant", "defect_size_group", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
    write_csv(output_dir / "view_geometry.csv", view_rows, ["seed", "split", "view", "effective_rank", "variance", "mean_norm", "pairwise_view_cosine_to_T24", "cka_to_T24"])
    write_csv(output_dir / "retrieval_overlap.csv", overlap_rows, ["seed", "variant", "view_pair", "top1_agreement", "top5_jaccard", "top20_jaccard", "top100_jaccard"])
    write_csv(output_dir / "view_contributions.csv", contribution_rows, ["seed", "variant", "category", "score_all", "best_supporting_view", "conflicting_view"])
    write_csv(output_dir / "memory_organization_diagnostics.csv", memory_rows, ["seed", "memory_entries", "total_multiview_vectors", "t24_duplicate_ratio", "v12_duplicate_ratio", "v18_duplicate_ratio", "memory_storage_mb", "compression_active", "graph_active", "spr_active"])
    write_csv(output_dir / "error_correction.csv", [], ["seed", "variant", "false_positives_corrected", "false_negatives_corrected", "new_false_positives", "new_false_negatives", "correction_to_introduction_ratio", "view_agreement", "rank_disagreement"])
    write_csv(output_dir / "retrieval_diagnostics.csv", retrieval_rows, ["seed", "variant", "retrieval_entropy", "normalized_entropy", "unique_prototypes_used", "never_top1_fraction", "anomaly_mass_mean"])
    write_csv(output_dir / "statistical_comparisons.csv", statistical_comparisons(summary_rows, per_category_rows), ["comparison", "variant_a", "variant_b", "metric", "mean_difference", "wins", "ties", "losses", "n"])
    write_csv(output_dir / "success_control_table.csv", success_control_rows(summary_rows, specs), ["Variant", "Multi-view tuple", "Correct alignment", "Late interaction", "Full bank", "I-AUROC", "P-AUROC", "PRO"])
    write_csv(output_dir / "computational_costs.csv", cost_rows, ["seed", "variant", "inference_time_per_image", "view_similarity_time_per_image", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "memory_bank_storage_mb"])
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_summary(summary_rows), ["variant", "I-AUROC", "I-AUROC_std", "I-AP", "I-AP_std", "P-AUROC", "P-AUROC_std", "PRO", "PRO_std"])
    write_json(output_dir / "sanity_checks.json", sanity)
    lines = [
        "# Phase IV-A Execution Summary",
        "",
        "## Run Status",
        f"- Rows: {len(summary_rows)}",
        f"- Failures: {len(failures)}",
        f"- Variants configured: {len(specs)}",
        "",
        "## Sanity",
        f"- Baseline checks: `{json.dumps(sanity.get('baseline_checks', {}), default=json_default)}`",
        f"- Alignment checks: `{json.dumps(sanity.get('alignment', {}), default=json_default)}`",
        "",
        "## Errors",
    ]
    lines.extend([f"- {x}" for x in failures] or ["- None"])
    (output_dir / "EXECUTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_selection(output_dir, cfg):
    data = {
        "protocol": "fixed defaults; no VisA validation split available in current Dataset wrapper",
        "candidate_settings": {
            "temperatures": cfg.get("temperature_grid", [0.03, 0.05, 0.07, 0.10]),
            "weights": cfg.get("weight_sets"),
            "rrf_c": cfg.get("rrf_c_values"),
        },
        "selected_configuration": {
            "variant": "V4_AlignedLogitLateInteraction",
            "temperatures": cfg.get("selected_temperatures", [0.07, 0.07, 0.07]),
            "weights": cfg.get("selected_weights", [1.0, 0.5, 0.5]),
        },
        "seeds": cfg.get("seeds", [0, 1, 2]),
        "fallback_usage": True,
        "target_sweeps": "post-hoc only",
    }
    write_json(output_dir / "source_selection.json", data)


def append_log(output_dir, cfg, sanity, summary_rows):
    text = "\n## Phase IV-A — Aligned Multi-View CLIP Memory with Late Interaction\n\n"
    text += "- Audited references: `docs/reference_code_audit_phase_IV_A.md`.\n"
    text += "- Tuple alignment: baseline fused FPS_K4 indices are selected once per source region, then T24/V12/V18 are gathered from the same spatial patch.\n"
    text += "- Implementation files: `multiview_memory/*`, `experiments/multiview_late_interaction_audit.py`, `configs/multiview_late_interaction_audit.yaml`.\n"
    text += f"- Command executed: `{' '.join(sys.argv)}`.\n"
    text += "- Source selection: fixed defaults; target sweeps marked post-hoc.\n"
    text += f"- Sanity checks: `{json.dumps(sanity, default=json_default)}`.\n"
    text += "- Exclusions: no compression, no graph propagation, no class split, no text prompts, no SPR.\n"
    text += f"- Completed metric rows: {len(summary_rows)}.\n"
    with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write(text)


def main():
    args = parse_args("Phase IV-A aligned multi-view late interaction audit")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    if args.seed is not None:
        cfg["seeds"] = [int(args.seed)]
    if args.limit is not None:
        cfg["run_mode"] = "smoke"
    output_dir = Path(cfg.get("output_dir", "results/multiview_late_interaction"))
    cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_failure = output_dir / "failure_traceback.txt"
    if stale_failure.exists():
        stale_failure.unlink()
    save_run_metadata(output_dir, cfg, int(cfg.get("seeds", [0])[0]))
    write_json(output_dir / "environment.json", package_versions())
    write_source_selection(output_dir, cfg)
    device = cfg.get("device", "cuda:0")
    model = load_model(cfg, device)
    _, source_loader = make_loader(cfg, cfg["source_dataset"], cfg.get("source_mode", "test"), shuffle=False)
    _, target_loader = make_loader(cfg, cfg["dataset"], cfg.get("target_mode", "test"), shuffle=False)
    specs = variant_specs(cfg)
    summary_rows, per_category_rows, defect_rows = [], [], []
    retrieval_rows, overlap_rows, view_rows, contribution_rows = [], [], [], []
    memory_rows, cost_rows, logs, failures = [], [], [], []
    sanity = {
        "no_training": True,
        "clip_frozen": True,
        "no_memory_graph": True,
        "no_class_split": True,
        "no_spr": True,
        "no_topk_support_truncation_primary": True,
        "target_labels_used_for_selection": False,
        "variants": [s["name"] for s in specs],
    }
    try:
        for seed in cfg.get("seeds", [0, 1, 2]):
            seed = int(seed)
            set_seed(seed)
            memory, entries, align_summary, construction_time = build_aligned_memory(model, source_loader, cfg, seed, device, output_dir)
            sanity.setdefault("alignment_by_seed", {})[str(seed)] = align_summary
            sanity["alignment"] = align_summary
            write_json(output_dir / f"memory_alignment_seed_{seed}.json", align_summary)
            write_json(output_dir / "memory_alignment.json", sanity["alignment_by_seed"])
            expected = cfg.get("expected_memory_entries")
            if expected and not cfg.get("limit") and int(memory["T24"].shape[0]) != int(expected):
                raise RuntimeError(f"Aligned memory entry count {memory['T24'].shape[0]} != expected {expected}")
            storage_mb = memory_storage_mb(memory)
            memory_rows.append(memory_organization(seed, memory, storage_mb))
            view_rows.extend(view_geometry_rows(seed, "source_memory", {k: memory[k] for k in ("T24", "V12", "V18")}))
            baseline_seen = False
            for spec in specs:
                safe = variant_safe_name(spec["name"])
                run_dir = output_dir / f"seed_{seed}" / safe
                run_dir.mkdir(parents=True, exist_ok=True)
                mpath = metrics_path(output_dir, seed, safe)
                if cfg.get("resume", True) and mpath.exists():
                    payload = json.loads(mpath.read_text(encoding="utf-8"))
                    metrics, defect_metrics, runtime = payload["metrics"], payload["defect_metrics"], payload["runtime"]
                    extra = payload.get("extra", {})
                    retrieval_rows.extend(extra.get("retrieval_rows", []))
                    overlap_rows.extend(extra.get("overlap_rows", []))
                    view_rows.extend(extra.get("view_rows", []))
                    contribution_rows.extend(extra.get("contribution_rows", []))
                else:
                    mem_spec = memory_for_spec(memory, spec, seed)
                    metrics, defect_metrics, runtime, rr, oo, vv, cc = evaluate_variant(model, mem_spec, target_loader, cfg, spec, seed, device)
                    payload = {"metrics": metrics, "defect_metrics": defect_metrics, "runtime": runtime, "extra": {"retrieval_rows": rr, "overlap_rows": oo, "view_rows": vv, "contribution_rows": cc}}
                    mpath.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
                    retrieval_rows.extend(rr)
                    overlap_rows.extend(oo)
                    view_rows.extend(vv)
                    contribution_rows.extend(cc)
                    del mem_spec
                mean = metrics["mean"]
                row = {
                    "seed": seed,
                    "variant": spec["name"],
                    "status": "completed",
                    "I-AUROC": metric_value(mean, "I-AUROC"),
                    "I-AP": metric_value(mean, "I-AP"),
                    "P-AUROC": metric_value(mean, "P-AUROC"),
                    "PRO": metric_value(mean, "PRO"),
                    "memory_bank_storage_mb": storage_mb,
                    **runtime,
                }
                summary_rows.append(row)
                cost_rows.append({"seed": seed, "variant": spec["name"], "memory_bank_storage_mb": storage_mb, **runtime})
                for category, vals in metrics["per_class"].items():
                    per_category_rows.append({"seed": seed, "variant": spec["name"], "category": category, "I-AUROC": metric_value(vals, "I-AUROC"), "I-AP": metric_value(vals, "I-AP"), "P-AUROC": metric_value(vals, "P-AUROC"), "PRO": metric_value(vals, "PRO")})
                for group, vals in defect_metrics["per_class"].items():
                    defect_rows.append({"seed": seed, "variant": spec["name"], "defect_size_group": group, "I-AUROC": metric_value(vals, "I-AUROC"), "I-AP": metric_value(vals, "I-AP"), "P-AUROC": metric_value(vals, "P-AUROC"), "PRO": metric_value(vals, "PRO")})
                logs.append(f"seed {seed} {spec['name']}: I-AUROC={metric_value(mean, 'I-AUROC'):.4f}, P-AUROC={metric_value(mean, 'P-AUROC'):.4f}, PRO={metric_value(mean, 'PRO'):.4f}")
                if spec["name"] == "V0_EarlyFusionBaseline" and not baseline_seen:
                    checks = baseline_checks([row], cfg)
                    sanity["baseline_checks"] = checks
                    failed = [k for k, v in checks.items() if isinstance(v, dict) and not v.get("pass")]
                    if failed and cfg.get("fail_on_baseline_mismatch", True) and not cfg.get("limit"):
                        raise RuntimeError(f"V0 baseline check failed: {checks}")
                    baseline_seen = True
                torch.cuda.empty_cache()
            del memory
            torch.cuda.empty_cache()
    except Exception:
        tb = traceback.format_exc()
        failures.append(tb)
        (output_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        raise
    finally:
        write_outputs(output_dir, summary_rows, per_category_rows, defect_rows, retrieval_rows, overlap_rows, view_rows, contribution_rows, memory_rows, cost_rows, specs, sanity, failures)
        copy_log(output_dir, logs)
        append_log(output_dir, cfg, sanity, summary_rows)


if __name__ == "__main__":
    main()
