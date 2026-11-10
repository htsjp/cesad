import json
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
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
from clip_value_features.cache import cache_size_mb, save_feature_cache, verify_cache
from clip_value_features.diagnostics import representation_geometry, retrieval_summary
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from clip_value_features.fusion import build_variant_specs, fuse_patch_features, required_layers
from memory_strategies import STRATEGIES
from memory_strategies.common import mechanism_metrics
from mrad_audit_utils import (
    copy_log,
    default_source_dataset,
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


def package_versions():
    data = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}
    if torch.cuda.is_available():
        data.update({"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    return data


def checkpoint_dir(output_dir: Path) -> Path:
    path = output_dir / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def tensor_dict_to_device(memory, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in memory.items()}


def memory_path(output_dir: Path, seed: int, variant: str) -> Path:
    return checkpoint_dir(output_dir) / f"seed_{seed}_{variant}_memory.pt"


def metrics_path(output_dir: Path, seed: int, variant: str) -> Path:
    return output_dir / f"seed_{seed}" / variant / "metrics.json"


def retrieval_path(output_dir: Path, seed: int, variant: str) -> Path:
    return output_dir / f"seed_{seed}" / variant / "retrieval_metrics.json"


def memory_mb(memory):
    total = 0
    for key in ("image_keys", "image_values", "patch_keys", "patch_values"):
        total += memory[key].numel() * memory[key].element_size()
    return total / (1024 * 1024)


def sample_id(items):
    if "img_path" in items:
        value = items["img_path"][0] if isinstance(items["img_path"], (list, tuple)) else items["img_path"]
        return str(value)
    return f"{items['cls_name'][0]}_{int(torch.as_tensor(items['anomaly'])[0])}"


def build_memory(model, loader, cfg, spec, layers, seed, device, cache_root: Path):
    ckpt = memory_path(Path(cfg["output_dir"]), seed, spec["name"])
    if cfg.get("resume", True) and ckpt.exists():
        return tensor_dict_to_device(torch.load(ckpt, map_location="cpu"), device), 0.0, [], []

    k = int(cfg.get("k", 4))
    image_features, image_labels, patch_features, patch_labels = [], [], [], []
    mechanism_rows, cache_checks = [], []
    region_id = 0
    start = time.perf_counter()
    with torch.no_grad():
        for index, items in enumerate(tqdm(loader, desc=f"build {spec['name']} memory")):
            images = items["img"].to(device)
            labels = torch.as_tensor(items["anomaly"], device=device)
            gt = (items["img_mask"].to(device) > 0.5).float()
            outputs = extract_value_representations(model, images, layers, device, smooth=cfg.get("average_neighbor", True))
            cls = cls_from_outputs(outputs, 24)
            patches = fuse_patch_features(outputs, spec)
            image_features.append(cls)
            image_labels.append(labels)
            if cfg.get("save_feature_cache", True) and index < int(cfg.get("cache_first_n", 8)):
                sid = sample_id(items)
                for layer, by_rep in outputs.items():
                    for rep in ("T", "V", "AV", "PAV"):
                        path = save_feature_cache(cache_root, cfg["source_dataset"], cfg.get("source_mode", "test"), sid, layer, rep, by_rep[rep][:, 1:, :], True)
                        cache_checks.append(verify_cache(path, by_rep[rep][:, 1:, :]))
            side = int(math.sqrt(patches.shape[1]))
            gt_resized = resize_mask(gt, side)
            for b in range(images.shape[0]):
                for mask, label_name, label_value in ((gt_resized[b], "anomaly", 1), (~gt_resized[b], "normal", 0)):
                    tokens = patches[b][mask.reshape(-1)]
                    if tokens.numel() == 0:
                        continue
                    entries = STRATEGIES["FPS_K4"](tokens, k, seed * 1_000_003 + region_id)
                    entries = F.normalize(entries, dim=-1)
                    patch_features.append(entries)
                    patch_labels.append(torch.full((entries.shape[0],), label_value, device=device, dtype=torch.long))
                    row = {"seed": seed, "variant": spec["name"], "region_type": label_name, "region_id": region_id, "region_tokens": int(tokens.shape[0])}
                    row.update(mechanism_metrics(tokens, entries))
                    mechanism_rows.append(row)
                    region_id += 1

    memory = {
        "image_keys": torch.cat(image_features, dim=0),
        "image_values": one_hot(torch.cat(image_labels, dim=0)).to(device),
        "patch_keys": torch.cat(patch_features, dim=0),
        "patch_values": one_hot(torch.cat(patch_labels, dim=0)).to(device),
    }
    atomic_torch_save({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in memory.items()}, ckpt)
    return memory, time.perf_counter() - start, cache_checks, mechanism_rows


@torch.no_grad()
def evaluate_variant(model, memory, loader, cfg, spec, layers, device):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    top_k = int(cfg.get("retrieval_top_k", 5))
    results = init_results(loader.dataset.obj_list)
    entropy_values, margin_values, top5_weight_values, per_image_times = [], [], [], []
    top1_counter, top1_label_counter = Counter(), Counter()
    patch_labels = memory["patch_values"].argmax(dim=-1).detach().cpu().numpy()
    geometry_rows = []

    for index, items in enumerate(tqdm(loader, desc=f"eval {spec['name']}")):
        start = time.perf_counter()
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        outputs = extract_value_representations(model, image, layers, device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        patches = fuse_patch_features(outputs, spec)
        if index == 0:
            geometry_rows.append(representation_geometry(patches, spec["name"]))
        image_logits, _, _ = retrieval_logits(cls, memory["image_keys"], memory["image_values"])
        patch_logits, patch_sim, patch_weights = retrieval_logits(patches, memory["patch_keys"], memory["patch_values"])
        flat_sim = patch_sim.reshape(-1, patch_sim.shape[-1])
        flat_weights = patch_weights.reshape(-1, patch_weights.shape[-1])
        k = min(top_k, flat_sim.shape[-1])
        vals, idx = torch.topk(flat_sim, k=k, dim=-1)
        entropy = -(flat_weights * flat_weights.clamp_min(1e-12).log()).sum(dim=-1)
        margin = vals[:, 0] - vals[:, 1] if k > 1 else vals[:, 0]
        entropy_values.extend(entropy.detach().cpu().tolist())
        margin_values.extend(margin.detach().cpu().tolist())
        top5_weight_values.extend(torch.gather(flat_weights, 1, idx).sum(dim=1).detach().cpu().tolist())
        top_idx = idx[:, 0].detach().cpu().numpy()
        top1_counter.update([int(x) for x in top_idx])
        top1_label_counter.update([int(patch_labels[x]) for x in top_idx])
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        per_image_times.append(time.perf_counter() - start)

    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    retrieval = retrieval_summary(entropy_values, margin_values, top5_weight_values, top1_counter, top1_label_counter, int(memory["patch_keys"].shape[0]))
    retrieval["mean_inference_time_per_image"] = float(np.mean(per_image_times)) if per_image_times else float("nan")
    return metrics, retrieval, geometry_rows


def aggregate_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["variant"], row.get("region_type", "all"))].append(row)
    out = []
    for (seed, variant, region_type), group in sorted(groups.items()):
        dst = {"seed": seed, "variant": variant, "region_type": region_type, "regions": len(group)}
        for key in keys:
            vals = [float(r[key]) for r in group if np.isfinite(float(r[key]))]
            dst[key] = float(np.mean(vals)) if vals else float("nan")
        out.append(dst)
    return out


def bootstrap_ci(values, rng_seed=0, samples=1000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    means = [rng.choice(values, size=values.size, replace=True).mean() for _ in range(samples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def best_variant(summary_rows, predicate, metric="P-AUROC"):
    candidates = [r for r in summary_rows if predicate(r) and np.isfinite(float(r.get(metric, float("nan"))))]
    if not candidates:
        return None
    means = defaultdict(list)
    for row in candidates:
        means[row["variant"]].append(float(row[metric]))
    return max(means, key=lambda name: float(np.mean(means[name])))


def statistical_comparisons(summary_rows, per_category_rows):
    index = {(r["seed"], r["variant"], r["category"]): r for r in per_category_rows}
    keys = sorted({(r["seed"], r["category"]) for r in per_category_rows})
    by_name = {r["variant"]: r for r in summary_rows}
    comparisons = []
    best_single_v = best_variant(summary_rows, lambda r: r["variant"].startswith("V") and "_Mean" not in r["variant"])
    best_single_av = best_variant(summary_rows, lambda r: r["variant"].startswith("AV") and "_Mean" not in r["variant"])
    best_v_mean = best_variant(summary_rows, lambda r: r["variant"].startswith("V") and r["variant"].endswith("_Mean"))
    best_av_mean = best_variant(summary_rows, lambda r: r["variant"].startswith("AV") and r["variant"].endswith("_Mean"))
    best_t_mean = best_variant(summary_rows, lambda r: r["variant"].startswith("T") and r["variant"].endswith("_Mean"))
    best_anchor = best_variant(summary_rows, lambda r: r["variant"].startswith("T24_plus_"))
    best_random = best_variant(summary_rows, lambda r: r["variant"].startswith("Random_"))
    best_single_t = best_variant(summary_rows, lambda r: r["variant"].startswith("T") and "_Mean" not in r["variant"])
    pairs = [
        ("best V2 vs V0", best_single_v, "T24_Baseline"),
        ("best V3 vs V0", best_single_av, "T24_Baseline"),
        ("best V5 vs matched V4", best_v_mean, best_t_mean),
        ("best V6 vs matched V4", best_av_mean, best_t_mean),
        ("best V7 vs V0", best_anchor, "T24_Baseline"),
        ("best V7 vs best V5/V6", best_anchor, best_v_mean or best_av_mean),
        ("targeted multi-layer vs random-layer control", best_v_mean or best_av_mean or best_t_mean, best_random),
        ("best value-path vs best single-layer block output", best_single_v or best_single_av or best_v_mean or best_av_mean, best_single_t),
    ]
    for label, a, b in pairs:
        if not a or not b:
            continue
        for metric in ("I-AUROC", "P-AUROC", "PRO"):
            diffs, wins, ties, losses = [], 0, 0, 0
            for seed, category in keys:
                ra = index.get((seed, a, category))
                rb = index.get((seed, b, category))
                if not ra or not rb:
                    continue
                va, vb = float(ra.get(metric, float("nan"))), float(rb.get(metric, float("nan")))
                if not np.isfinite(va) or not np.isfinite(vb):
                    continue
                diff = va - vb
                diffs.append(diff)
                if abs(diff) < 1e-6:
                    ties += 1
                elif diff > 0:
                    wins += 1
                else:
                    losses += 1
            lo, hi = bootstrap_ci(diffs)
            comparisons.append({"comparison": label, "variant_a": a, "variant_b": b, "metric": metric, "mean_difference": float(np.mean(diffs)) if diffs else float("nan"), "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi, "wins": wins, "ties": ties, "losses": losses, "n": len(diffs)})
    return comparisons


def success_rows(summary_rows):
    names = {
        "T24 baseline": "T24_Baseline",
        "best single T": best_variant(summary_rows, lambda r: r["variant"].startswith("T") and "_Mean" not in r["variant"]),
        "best single V": best_variant(summary_rows, lambda r: r["variant"].startswith("V") and "_Mean" not in r["variant"]),
        "best single AV": best_variant(summary_rows, lambda r: r["variant"].startswith("AV") and "_Mean" not in r["variant"]),
        "best block mean": best_variant(summary_rows, lambda r: r["variant"].startswith("T") and r["variant"].endswith("_Mean")),
        "best V mean": best_variant(summary_rows, lambda r: r["variant"].startswith("V") and r["variant"].endswith("_Mean")),
        "best AV mean": best_variant(summary_rows, lambda r: r["variant"].startswith("AV") and r["variant"].endswith("_Mean")),
        "best semantic-anchor residual": best_variant(summary_rows, lambda r: r["variant"].startswith("T24_plus_")),
        "best random-layer control": best_variant(summary_rows, lambda r: r["variant"].startswith("Random_")),
    }
    out = []
    for label, name in names.items():
        rows = [r for r in summary_rows if r["variant"] == name]
        if not rows:
            continue
        out.append({
            "selection": label,
            "Variant": name,
            "Uses Value Path": rows[0]["Uses Value Path"],
            "Uses Multiple Layers": rows[0]["Uses Multiple Layers"],
            "Uses T24 Semantic Anchor": rows[0]["Uses T24 Semantic Anchor"],
            "I-AUROC": float(np.nanmean([float(r["I-AUROC"]) for r in rows])),
            "P-AUROC": float(np.nanmean([float(r["P-AUROC"]) for r in rows])),
            "PRO": float(np.nanmean([float(r["PRO"]) for r in rows])),
        })
    return out


def write_outputs(output_dir, summary_rows, per_category_rows, retrieval_rows, geometry_rows, memory_rows, sanity, failures):
    write_csv(output_dir / "summary_metrics.csv", summary_rows, ["seed", "variant", "status", "Uses Value Path", "Uses Multiple Layers", "Uses T24 Semantic Anchor", "I-AUROC", "I-AP", "P-AUROC", "PRO", "inference_time_per_image_sec", "memory_construction_time_sec", "peak_gpu_memory_mb", "feature_cache_size_mb", "patch_memory_size"])
    write_csv(output_dir / "per_category_metrics.csv", per_category_rows, ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
    write_csv(output_dir / "retrieval_diagnostics.csv", retrieval_rows, ["seed", "variant", "retrieval_entropy", "normalized_entropy", "usage_gini", "never_top1_fraction", "top1_margin", "top5_cumulative_weight", "unique_prototypes_used", "normal_top1_usage", "anomaly_top1_usage", "mean_inference_time_per_image"])
    write_csv(output_dir / "geometry_diagnostics.csv", geometry_rows, ["seed", "variant", "mean_norm_before_normalization", "feature_variance", "effective_rank", "top_pc_energy", "mean_pairwise_cosine", "patch_spatial_smoothness"])
    write_csv(output_dir / "memory_diagnostics.csv", aggregate_rows(memory_rows, ["mean_reconstruction_distance", "maximum_coverage_radius", "pairwise_prototype_diversity", "region_variance_coverage", "prototype_duplication_ratio"]), ["seed", "variant", "region_type", "regions", "mean_reconstruction_distance", "maximum_coverage_radius", "pairwise_prototype_diversity", "region_variance_coverage", "prototype_duplication_ratio"])
    write_csv(output_dir / "statistical_comparisons.csv", statistical_comparisons(summary_rows, per_category_rows), ["comparison", "variant_a", "variant_b", "metric", "mean_difference", "bootstrap_ci95_low", "bootstrap_ci95_high", "wins", "ties", "losses", "n"])
    write_csv(output_dir / "success_table.csv", success_rows(summary_rows), ["selection", "Variant", "Uses Value Path", "Uses Multiple Layers", "Uses T24 Semantic Anchor", "I-AUROC", "P-AUROC", "PRO"])
    write_csv(output_dir / "head_diagnostics.csv", [], ["seed", "layer", "head", "feature_variance", "effective_rank", "mean_patch_spatial_smoothness", "anomaly_normal_separability_auroc", "correlation_with_final_anomaly_map", "cross_category_stability", "mean_redundancy_with_other_heads", "cosine_similarity_to_T24"])
    (output_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, default=json_default), encoding="utf-8")
    lines = ["# Phase II-D Execution Summary", "", "## Completed", f"- Rows: {len(summary_rows)}", f"- Failures: {len(failures)}", "", "## Errors"]
    lines.extend([f"- {x}" for x in failures] or ["- None"])
    (output_dir / "EXECUTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def baseline_checks(summary_rows, cfg):
    refs = cfg.get("baseline_reference", {})
    tols = cfg.get("baseline_tolerance", {})
    row = next((r for r in summary_rows if r["variant"] == "T24_Baseline"), None)
    checks = {}
    if not row:
        return {"missing": "T24_Baseline"}
    for metric, ref in refs.items():
        tol = float(tols.get(metric, 0.002))
        value = float(row.get(metric, float("nan")))
        checks[metric] = {"value": value, "reference": float(ref), "tolerance": tol, "pass": bool(np.isfinite(value) and abs(value - float(ref)) <= tol)}
    return checks


def append_log(cfg, output_dir, sanity, summary_rows):
    text = "\n## Phase II-D — Multi-Layer Value Representation Audit\n\n"
    text += "- Audited source files: `AnomalyCLIP_lib/AnomalyCLIP.py`, `AnomalyCLIP_lib/CLIP.py`, `experiments/mrad_audit_utils.py`.\n"
    text += "- Hook/extraction strategy: non-destructive mirrored visual forward in `clip_value_features/extractors.py`; no source patch applied.\n"
    text += "- Feature definitions: projected and normalized `T`, `V`, `AV`, `PAV` patch tokens for configured layers; image memory remains layer-24 CLS.\n"
    text += "- Code files added: `clip_value_features/*`, `experiments/value_representation_audit.py`, `configs/value_representation_audit.yaml`, `docs/clip_value_path_audit_phase_II_D.md`.\n"
    text += f"- Command executed: `{' '.join(sys.argv)}`.\n"
    text += f"- Sanity checks: `{json.dumps(sanity, default=json_default)}`.\n"
    text += f"- Raw metric tables: `{output_dir / 'summary_metrics.csv'}`, `per_category_metrics.csv`, `geometry_diagnostics.csv`, `memory_diagnostics.csv`, `retrieval_diagnostics.csv`.\n"
    if summary_rows:
        text += f"- Completed metric rows: {len(summary_rows)}.\n"
    with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write(text)


def main():
    args = parse_args("Phase II-D multi-layer value representation audit")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    cfg["source_dataset"] = cfg.get("source_dataset") or default_source_dataset(cfg["dataset"])
    if args.seed is not None:
        cfg["seeds"] = [int(args.seed)]
    if args.limit is not None:
        cfg["run_mode"] = "smoke"
    output_dir = Path(cfg.get("output_dir", "results/value_representation_audit"))
    cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cfg.get("cache_dir", "cache/value_representation_audit"))
    save_run_metadata(output_dir, cfg, int(cfg.get("seeds", [0])[0]))
    (output_dir / "environment.json").write_text(json.dumps(package_versions(), indent=2, default=json_default), encoding="utf-8")

    device = cfg.get("device", "cuda:0")
    model = load_model(cfg, device)
    variants = build_variant_specs(cfg)
    layers = required_layers(variants)
    _, source_loader = make_loader(cfg, cfg["source_dataset"], cfg.get("source_mode", "test"), shuffle=False)
    _, target_loader = make_loader(cfg, cfg["dataset"], cfg.get("target_mode", "test"), shuffle=False)

    summary_rows, per_category_rows, retrieval_rows, geometry_rows, memory_rows, failures, logs = [], [], [], [], [], [], []
    sanity = {"variants": [x["name"] for x in variants], "required_layers": layers, "hooks_do_not_patch_source": True}
    try:
        for seed in cfg.get("seeds", [0, 1, 2]):
            seed = int(seed)
            set_seed(seed)
            for spec in variants:
                run_dir = output_dir / f"seed_{seed}" / spec["name"]
                run_dir.mkdir(parents=True, exist_ok=True)
                mpath = metrics_path(output_dir, seed, spec["name"])
                rpath = retrieval_path(output_dir, seed, spec["name"])
                if cfg.get("resume", True) and mpath.exists() and rpath.exists():
                    metrics = json.loads(mpath.read_text(encoding="utf-8"))
                    retrieval = json.loads(rpath.read_text(encoding="utf-8"))
                    construction_time = 0.0
                    memory = tensor_dict_to_device(torch.load(memory_path(output_dir, seed, spec["name"]), map_location="cpu"), device)
                    cache_checks = []
                else:
                    memory, construction_time, cache_checks, mechanism_rows = build_memory(model, source_loader, cfg, spec, layers, seed, device, cache_root)
                    memory_rows.extend(mechanism_rows)
                    metrics, retrieval, geom = evaluate_variant(model, memory, target_loader, cfg, spec, layers, device)
                    mpath.write_text(json.dumps(metrics, indent=2, default=json_default), encoding="utf-8")
                    rpath.write_text(json.dumps(retrieval, indent=2, default=json_default), encoding="utf-8")
                    for row in geom:
                        geometry_rows.append({"seed": seed, **row})
                mean = metrics["mean"]
                if cache_checks:
                    sanity.setdefault("cache_checks", []).extend(cache_checks[:4])
                row = {
                    "seed": seed,
                    "variant": spec["name"],
                    "status": "completed",
                    "Uses Value Path": spec.get("rep") in {"V", "AV", "PAV"} or spec.get("source") in {"V", "AV"},
                    "Uses Multiple Layers": len(spec.get("layers", [])) > 1,
                    "Uses T24 Semantic Anchor": spec["family"] == "anchor",
                    "I-AUROC": mean.get("I-AUROC"),
                    "I-AP": mean.get("I-AP"),
                    "P-AUROC": mean.get("P-AUROC"),
                    "PRO": mean.get("PRO"),
                    "inference_time_per_image_sec": retrieval.get("mean_inference_time_per_image"),
                    "memory_construction_time_sec": construction_time,
                    "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
                    "feature_cache_size_mb": cache_size_mb(cache_root),
                    "patch_memory_size": int(memory["patch_keys"].shape[0]),
                }
                summary_rows.append(row)
                retrieval_rows.append({"seed": seed, "variant": spec["name"], **retrieval})
                for category, vals in metrics["per_class"].items():
                    per_category_rows.append({"seed": seed, "variant": spec["name"], "category": category, "I-AUROC": vals.get("I-AUROC"), "I-AP": vals.get("I-AP"), "P-AUROC": vals.get("P-AUROC"), "PRO": vals.get("PRO")})
                logs.append(f"seed {seed} {spec['name']}: I-AUROC={mean.get('I-AUROC'):.4f}, P-AUROC={mean.get('P-AUROC'):.4f}, PRO={mean.get('PRO'):.4f}")
                if spec["name"] == "T24_Baseline":
                    checks = baseline_checks([row], cfg)
                    sanity["baseline_checks"] = checks
                    failed = [k for k, v in checks.items() if isinstance(v, dict) and not v.get("pass")]
                    if failed and cfg.get("fail_on_baseline_mismatch", True) and not cfg.get("limit"):
                        raise RuntimeError(f"T24 baseline check failed: {checks}")
                del memory
                torch.cuda.empty_cache()
    except Exception:
        tb = traceback.format_exc()
        failures.append(tb)
        (output_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        raise
    finally:
        write_outputs(output_dir, summary_rows, per_category_rows, retrieval_rows, geometry_rows, memory_rows, sanity, failures)
        copy_log(output_dir, logs)
        append_log(cfg, output_dir, sanity, summary_rows)


if __name__ == "__main__":
    main()
