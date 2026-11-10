import json
import math
import os
import platform
import subprocess
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

os.environ.setdefault("OMP_NUM_THREADS", "6")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import AnomalyCLIP_lib
from memory_strategies import STRATEGIES
from memory_strategies.common import mechanism_metrics
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


K4_VARIANTS = ["DuplicateMean_K4", "Random_K4", "KMeans_K4", "FPS_K4", "GreedyCoreset_K4"]
COMPARISONS = [
    ("FPS_K4", "DuplicateMean_K4"),
    ("FPS_K4", "Random_K4"),
    ("GreedyCoreset_K4", "DuplicateMean_K4"),
    ("GreedyCoreset_K4", "Random_K4"),
    ("FPS_K4", "KMeans_K4"),
    ("GreedyCoreset_K4", "KMeans_K4"),
]


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def memory_mb(memory):
    total = 0
    for key in ("image_keys", "image_values", "patch_keys", "patch_values"):
        total += memory[key].numel() * memory[key].element_size()
    return total / (1024 * 1024)


def gini_from_counts(counts, size):
    arr = np.zeros(size, dtype=np.float64)
    for idx, value in counts.items():
        if 0 <= idx < size:
            arr[idx] = value
    if arr.sum() == 0:
        return float("nan")
    arr.sort()
    n = arr.size
    return float((2 * np.arange(1, n + 1).dot(arr) / (n * arr.sum())) - (n + 1) / n)


def mean_dict(rows, keys):
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def package_versions():
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        versions["cuda"] = torch.version.cuda
        versions["gpu"] = torch.cuda.get_device_name(0)
    return versions


def metrics_file_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mean = data.get("mean", {})
        for metric in ("I-AUROC", "I-AP", "P-AUROC", "PRO"):
            value = float(mean.get(metric, float("nan")))
            if not math.isfinite(value):
                return False
        for vals in data.get("per_class", {}).values():
            for metric in ("I-AUROC", "P-AUROC", "PRO"):
                value = float(vals.get(metric, float("nan")))
                if not math.isfinite(value):
                    return False
        return True
    except Exception:
        return False


def seed_is_complete(output_dir: Path, seed: int, variants) -> bool:
    seed_dir = output_dir / f"seed_{seed}"
    return all(metrics_file_is_valid(seed_dir / variant / "metrics.json") for variant in variants)


def estimated_memory_mb(image_entries, patch_entries):
    bytes_per_entry = (768 + 2) * 4
    return float((image_entries + patch_entries) * bytes_per_entry / (1024 * 1024))


def build_all_memories(model, loader, cfg, seed, device):
    variants = cfg["variants"]
    k = int(cfg.get("k", 4))
    layer = int(cfg.get("layer", 24))
    features_list = cfg["features_list"]
    image_features, image_labels = [], []
    patch_features = {variant: [] for variant in variants}
    patch_labels = {variant: [] for variant in variants}
    mechanism_rows = []
    first_region_tokens = None
    region_id = 0
    start = time.perf_counter()

    with torch.no_grad():
        for items in tqdm(loader, desc=f"build diversity memories seed {seed}"):
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
                for mask, label_name, label_value in ((gt_resized[b], "anomaly", 1), (~gt_resized[b], "normal", 0)):
                    tokens = patches[b][mask.reshape(-1)]
                    if tokens.numel() == 0:
                        continue
                    if first_region_tokens is None:
                        first_region_tokens = tokens.detach().clone()
                    for variant in variants:
                        local_k = 1 if variant == "RegionMean_K1" else k
                        variant_seed = int(seed) * 1_000_003 + region_id * 97
                        entries = STRATEGIES[variant](tokens, local_k, variant_seed)
                        entries = F.normalize(entries, dim=-1)
                        if variant != "RegionMean_K1" and entries.shape[0] != k:
                            raise RuntimeError(f"{variant} returned {entries.shape[0]} entries, expected {k}")
                        patch_features[variant].append(entries)
                        patch_labels[variant].append(
                            torch.full((entries.shape[0],), label_value, device=device, dtype=torch.long)
                        )
                        row = {
                            "seed": seed,
                            "variant": variant,
                            "region_type": label_name,
                            "region_id": region_id,
                            "region_tokens": int(tokens.shape[0]),
                        }
                        row.update(mechanism_metrics(tokens, entries))
                        mechanism_rows.append(row)
                    region_id += 1

    image_keys = torch.cat(image_features, dim=0)
    image_values = one_hot(torch.cat(image_labels, dim=0)).to(device)
    memories = {}
    for variant in variants:
        memories[variant] = {
            "image_keys": image_keys,
            "image_values": image_values,
            "patch_keys": torch.cat(patch_features[variant], dim=0),
            "patch_values": one_hot(torch.cat(patch_labels[variant], dim=0)).to(device),
        }

    construction_time = time.perf_counter() - start
    sanity = run_memory_sanity(memories, first_region_tokens, seed, k)
    return memories, mechanism_rows, construction_time, sanity


def run_memory_sanity(memories, first_region_tokens, seed, k):
    sanity = {}
    counts = {variant: int(memories[variant]["patch_keys"].shape[0]) for variant in K4_VARIANTS}
    sanity["k4_entry_counts"] = counts
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"K=4 capacity mismatch: {counts}")

    dup = memories["DuplicateMean_K4"]["patch_keys"].reshape(-1, k, memories["DuplicateMean_K4"]["patch_keys"].shape[-1])
    max_dup_diff = (dup - dup[:, :1, :]).abs().max().item()
    sanity["duplicate_mean_max_abs_diff"] = float(max_dup_diff)
    if max_dup_diff > 1e-6:
        raise RuntimeError(f"DuplicateMean_K4 is not exactly duplicated: max diff {max_dup_diff}")

    if first_region_tokens is not None:
        fps_a = STRATEGIES["FPS_K4"](first_region_tokens, k, seed)
        fps_b = STRATEGIES["FPS_K4"](first_region_tokens, k, seed)
        greedy_a = STRATEGIES["GreedyCoreset_K4"](first_region_tokens, k, seed)
        greedy_b = STRATEGIES["GreedyCoreset_K4"](first_region_tokens, k, seed)
        rand_a = STRATEGIES["Random_K4"](first_region_tokens, k, seed)
        rand_b = STRATEGIES["Random_K4"](first_region_tokens, k, seed + 1)
        sanity["fps_deterministic_max_abs_diff"] = float((fps_a - fps_b).abs().max().item())
        sanity["greedy_deterministic_max_abs_diff"] = float((greedy_a - greedy_b).abs().max().item())
        sanity["random_cross_seed_max_abs_diff"] = float((rand_a - rand_b).abs().max().item())
        if sanity["fps_deterministic_max_abs_diff"] > 1e-6:
            raise RuntimeError("FPS_K4 is not deterministic within a seed")
        if sanity["greedy_deterministic_max_abs_diff"] > 1e-6:
            raise RuntimeError("GreedyCoreset_K4 is not deterministic within a seed")
        if first_region_tokens.shape[0] > k and sanity["random_cross_seed_max_abs_diff"] <= 1e-6:
            raise RuntimeError("Random_K4 did not differ across sanity-check seeds")
    return sanity


@torch.no_grad()
def run_eval_with_diagnostics(model, memory, loader, cfg, variant, device):
    layer = int(cfg.get("layer", 24))
    features_list = cfg["features_list"]
    image_size = int(cfg.get("image_size", 518))
    sigma = cfg.get("sigma", 4)
    fusion_k = cfg.get("fusion_k", 0.7)
    topk_ratio = cfg.get("topk_ratio", 0.01)
    top_k = int(cfg.get("retrieval_top_k", 5))
    results = init_results(loader.dataset.obj_list)
    top1_counter = Counter()
    top1_label_counter = Counter()
    entropy_values, margin_values, top5_weight_values = [], [], []
    per_image_times = []
    patch_values = memory["patch_values"].to(device)
    patch_labels = patch_values.argmax(dim=-1).detach().cpu().numpy()
    patch_m = int(memory["patch_keys"].shape[0])

    for items in tqdm(loader, desc=variant):
        start = time.perf_counter()
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())

        cls, patches, _ = extract_layer_features(
            model, image, features_list, layer, device, smooth=cfg.get("average_neighbor", True)
        )
        image_logits, _, _ = retrieval_logits(
            cls, memory["image_keys"].to(device), memory["image_values"].to(device)
        )
        patch_logits, patch_sim, patch_weights = retrieval_logits(
            patches, memory["patch_keys"].to(device), patch_values
        )
        flat_sim = patch_sim.reshape(-1, patch_sim.shape[-1])
        flat_weights = patch_weights.reshape(-1, patch_weights.shape[-1])
        k = min(top_k, flat_sim.shape[-1])
        vals, idx = torch.topk(flat_sim, k=k, dim=-1)
        weight_vals = torch.gather(flat_weights, 1, idx)
        entropy = -(flat_weights * flat_weights.clamp_min(1e-12).log()).sum(dim=-1)
        margin = vals[:, 0] - vals[:, 1] if k > 1 else vals[:, 0]
        entropy_values.extend(entropy.detach().cpu().tolist())
        margin_values.extend(margin.detach().cpu().tolist())
        top5_weight_values.extend(weight_vals.sum(dim=1).detach().cpu().tolist())
        top_idx = idx[:, 0].detach().cpu().numpy()
        top1_counter.update([int(x) for x in top_idx])
        top1_label_counter.update([int(patch_labels[x]) for x in top_idx])

        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1)
        score = (1 - fusion_k) * topk_mean.to(device) + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        per_image_times.append(time.perf_counter() - start)

    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    unique_used = len(top1_counter)
    diagnostics = {
        "retrieval_entropy": float(np.mean(entropy_values)) if entropy_values else float("nan"),
        "normalized_retrieval_entropy": float(np.mean(entropy_values) / math.log(max(2, patch_m))) if entropy_values else float("nan"),
        "memory_usage_gini": gini_from_counts(top1_counter, patch_m),
        "fraction_memory_never_top1": float((patch_m - unique_used) / max(1, patch_m)),
        "top1_similarity_margin": float(np.mean(margin_values)) if margin_values else float("nan"),
        "top5_cumulative_weight": float(np.mean(top5_weight_values)) if top5_weight_values else float("nan"),
        "unique_memory_entries_used": int(unique_used),
        "normal_top1_usage": int(top1_label_counter.get(0, 0)),
        "anomaly_top1_usage": int(top1_label_counter.get(1, 0)),
    }
    return metrics, diagnostics, float(np.mean(per_image_times)) if per_image_times else float("nan")


def aggregate_mechanism_rows(rows):
    keys = [
        "mean_reconstruction_distance",
        "maximum_coverage_radius",
        "pairwise_prototype_diversity",
        "region_variance_coverage",
        "prototype_duplication_ratio",
    ]
    out = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["variant"], "all")].append(row)
        groups[(row["seed"], row["variant"], row["region_type"])].append(row)
    for (seed, variant, region_type), group in sorted(groups.items()):
        agg = {"seed": seed, "variant": variant, "region_type": region_type, "regions": len(group)}
        agg.update(mean_dict(group, keys))
        out.append(agg)
    return out


def bootstrap_ci(values, rng_seed=0, samples=2000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    means = [rng.choice(values, size=values.size, replace=True).mean() for _ in range(samples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def statistical_comparisons(per_category_rows):
    metrics = ["I-AUROC", "P-AUROC", "PRO"]
    rows = []
    index = {(r["seed"], r["variant"], r["category"]): r for r in per_category_rows}
    keys = sorted({(r["seed"], r["category"]) for r in per_category_rows})
    for a, b in COMPARISONS:
        for metric in metrics:
            diffs = []
            wins = ties = losses = 0
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
            ci_low, ci_high = bootstrap_ci(diffs)
            rows.append({
                "comparison": f"{a} vs {b}",
                "metric": metric,
                "mean_difference": float(np.mean(diffs)) if diffs else float("nan"),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "n": len(diffs),
            })
    return rows


def append_experiment_log(path, summary_text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(summary_text)


def write_execution_summary(output_dir, summary_rows, failures):
    path = output_dir / "EXECUTION_SUMMARY.md"
    lines = [
        "# Phase II-A Execution Summary",
        "",
        "## Completed Experiments",
    ]
    completed = [r for r in summary_rows if r.get("status") == "completed"]
    if completed:
        for row in completed:
            lines.append(f"- seed {row['seed']} / {row['variant']}")
    else:
        lines.append("- None")
    lines.extend(["", "## Missing Experiments"])
    missing = [r for r in summary_rows if r.get("status") != "completed"]
    lines.extend([f"- seed {r.get('seed')} / {r.get('variant')}: {r.get('status')}" for r in missing] or ["- None"])
    lines.extend(["", "## Errors"])
    lines.extend([f"- {failure}" for failure in failures] or ["- None"])
    lines.extend([
        "",
        "## Artifact Paths",
        f"- `{output_dir / 'summary_metrics.csv'}`",
        f"- `{output_dir / 'per_category_metrics.csv'}`",
        f"- `{output_dir / 'mechanism_metrics.csv'}`",
        f"- `{output_dir / 'retrieval_metrics.csv'}`",
        f"- `{output_dir / 'statistical_comparisons.csv'}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args("Phase II-A diversity-vs-capacity audit")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit})
    cfg["source_dataset"] = cfg.get("source_dataset") or default_source_dataset(cfg["dataset"])
    device = cfg.get("device", "cuda:0")
    output_dir = Path(cfg.get("output_dir", "results/diversity_capacity_audit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, cfg, int(cfg.get("seeds", [0])[0]))
    (output_dir / "environment.json").write_text(json.dumps(package_versions(), indent=2, default=json_default), encoding="utf-8")

    summary_rows, per_category_rows, mechanism_all, retrieval_rows, failures = [], [], [], [], []
    logs = []
    model = load_model(cfg, device)
    _, source_loader = make_loader(cfg, cfg["source_dataset"], mode=cfg.get("source_mode", "test"), shuffle=False)
    _, target_loader = make_loader(cfg, cfg["dataset"], mode=cfg.get("target_mode", "test"), shuffle=False)

    try:
        for seed in cfg.get("seeds", [0, 1, 2]):
            seed = int(seed)
            set_seed(seed)
            seed_dir = output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            if cfg.get("skip_complete_seeds", False) and cfg.get("resume_existing", True) and seed_is_complete(output_dir, seed, cfg["variants"]):
                logs.append(f"seed {seed}: skipped complete valid seed")
                sanity_path = seed_dir / "sanity_checks.json"
                sanity = json.loads(sanity_path.read_text(encoding="utf-8")) if sanity_path.exists() else {}
                k4_count = int(next(iter(sanity.get("k4_entry_counts", {"": 0}).values()), 0))
                region_mean_count = k4_count // int(cfg.get("k", 4)) if k4_count else ""
                image_entries = len(source_loader.dataset)
                for variant in cfg["variants"]:
                    metrics = json.loads((seed_dir / variant / "metrics.json").read_text(encoding="utf-8"))
                    diagnostics = json.loads((seed_dir / variant / "retrieval_metrics.json").read_text(encoding="utf-8"))
                    mean = metrics["mean"]
                    patch_entries = region_mean_count if variant == "RegionMean_K1" else k4_count
                    summary_rows.append({
                        "seed": seed,
                        "variant": variant,
                        "status": "completed",
                        "I-AUROC": mean.get("I-AUROC"),
                        "I-AP": mean.get("I-AP"),
                        "P-AUROC": mean.get("P-AUROC"),
                        "PRO": mean.get("PRO"),
                        "patch_memory_entries": patch_entries,
                        "memory_size_mb": estimated_memory_mb(image_entries, patch_entries) if patch_entries != "" else "",
                        "memory_construction_time_sec": "",
                        "mean_inference_time_per_image_sec": diagnostics.get("mean_inference_time_per_image"),
                        "start_timestamp": "",
                        "end_timestamp": "",
                    })
                    retrieval_rows.append({"seed": seed, "variant": variant, **diagnostics})
                    for category, vals in metrics["per_class"].items():
                        per_category_rows.append({
                            "seed": seed,
                            "variant": variant,
                            "category": category,
                            "I-AUROC": vals.get("I-AUROC"),
                            "P-AUROC": vals.get("P-AUROC"),
                            "PRO": vals.get("PRO"),
                        })
                continue
            start_ts = datetime.now().isoformat()
            memories, mechanism_rows, construction_time, sanity = build_all_memories(model, source_loader, cfg, seed, device)
            mechanism_all.extend(mechanism_rows)
            (seed_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, default=json_default), encoding="utf-8")

            for variant in cfg["variants"]:
                run_dir = seed_dir / variant
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "metrics.json"
                retrieval_path = run_dir / "retrieval_metrics.json"
                if metrics_file_is_valid(metrics_path) and retrieval_path.exists() and cfg.get("resume_existing", True):
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    diagnostics = json.loads(retrieval_path.read_text(encoding="utf-8"))
                    infer_time = diagnostics.get("mean_inference_time_per_image", float("nan"))
                else:
                    metrics, diagnostics, infer_time = run_eval_with_diagnostics(model, memories[variant], target_loader, cfg, variant, device)
                    diagnostics["mean_inference_time_per_image"] = infer_time
                    metrics_path.write_text(json.dumps(metrics, indent=2, default=json_default), encoding="utf-8")
                    retrieval_path.write_text(json.dumps(diagnostics, indent=2, default=json_default), encoding="utf-8")

                mean = metrics["mean"]
                patch_entries = int(memories[variant]["patch_keys"].shape[0])
                row = {
                    "seed": seed,
                    "variant": variant,
                    "status": "completed",
                    "I-AUROC": mean.get("I-AUROC"),
                    "I-AP": mean.get("I-AP"),
                    "P-AUROC": mean.get("P-AUROC"),
                    "PRO": mean.get("PRO"),
                    "patch_memory_entries": patch_entries,
                    "memory_size_mb": memory_mb(memories[variant]),
                    "memory_construction_time_sec": construction_time,
                    "mean_inference_time_per_image_sec": infer_time,
                    "start_timestamp": start_ts,
                    "end_timestamp": datetime.now().isoformat(),
                }
                summary_rows.append(row)
                retrieval_rows.append({"seed": seed, "variant": variant, **diagnostics})
                for category, vals in metrics["per_class"].items():
                    per_category_rows.append({
                        "seed": seed,
                        "variant": variant,
                        "category": category,
                        "I-AUROC": vals.get("I-AUROC"),
                        "P-AUROC": vals.get("P-AUROC"),
                        "PRO": vals.get("PRO"),
                    })
                logs.append(f"seed {seed} {variant}: I-AUROC={mean.get('I-AUROC'):.4f}, P-AUROC={mean.get('P-AUROC'):.4f}, PRO={mean.get('PRO'):.4f}")
                if (not cfg.get("limit")) and seed == int(cfg.get("seeds", [0])[0]) and variant == "RegionMean_K1":
                    ref = cfg.get("baseline_reference", {})
                    tol = float(cfg.get("baseline_tolerance", 0.002))
                    for metric in ("I-AUROC", "P-AUROC"):
                        if metric in ref and abs(float(mean[metric]) - float(ref[metric])) > tol:
                            raise RuntimeError(f"RegionMean_K1 sanity failed for {metric}: got {mean[metric]}, ref {ref[metric]}")
                torch.cuda.empty_cache()

            del memories
            torch.cuda.empty_cache()
    except Exception as exc:
        tb = traceback.format_exc()
        failures.append(tb)
        (output_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        raise exc
    finally:
        mechanism_summary = aggregate_mechanism_rows(mechanism_all)
        write_csv(output_dir / "summary_metrics.csv", summary_rows, ["seed", "variant", "status", "I-AUROC", "I-AP", "P-AUROC", "PRO", "patch_memory_entries", "memory_size_mb", "memory_construction_time_sec", "mean_inference_time_per_image_sec", "start_timestamp", "end_timestamp"])
        write_csv(output_dir / "per_category_metrics.csv", per_category_rows, ["seed", "variant", "category", "I-AUROC", "P-AUROC", "PRO"])
        write_csv(output_dir / "mechanism_metrics.csv", mechanism_summary, ["seed", "variant", "region_type", "regions", "mean_reconstruction_distance", "maximum_coverage_radius", "pairwise_prototype_diversity", "region_variance_coverage", "prototype_duplication_ratio"])
        write_csv(output_dir / "retrieval_metrics.csv", retrieval_rows, ["seed", "variant", "retrieval_entropy", "normalized_retrieval_entropy", "memory_usage_gini", "fraction_memory_never_top1", "top1_similarity_margin", "top5_cumulative_weight", "unique_memory_entries_used", "normal_top1_usage", "anomaly_top1_usage", "mean_inference_time_per_image"])
        comparison_rows = statistical_comparisons(per_category_rows)
        write_csv(output_dir / "statistical_comparisons.csv", comparison_rows, ["comparison", "metric", "mean_difference", "bootstrap_ci95_low", "bootstrap_ci95_high", "wins", "ties", "losses", "n"])
        write_execution_summary(output_dir, summary_rows, failures)
        copy_log(output_dir, logs)

        log_text = "\n## Phase II-A — Diversity vs Capacity Audit\n\n"
        log_text += "Purpose: determine whether FPS_K4 gains over RegionMean_K1 come from increased memory capacity or diversity-preserving coverage of intra-region feature modes.\n\n"
        log_text += "Code files added or modified: `experiments/diversity_capacity_audit.py`, `configs/diversity_capacity_audit.yaml`, `memory_strategies/*`, `experiments/mrad_audit_utils.py`.\n\n"
        log_text += f"Commands executed: `{' '.join(sys.argv)}`\n\n"
        log_text += f"Completed runs: {len([r for r in summary_rows if r.get('status') == 'completed'])}. Failures: {len(failures)}.\n\n"
        log_text += "Raw metric tables: `results/diversity_capacity_audit/summary_metrics.csv`, `per_category_metrics.csv`, `mechanism_metrics.csv`, `retrieval_metrics.csv`.\n\n"
        log_text += "### Factual Result Summary\n\n"
        if summary_rows:
            best = max(summary_rows, key=lambda r: float(r.get("P-AUROC") or float("-inf")))
            log_text += f"- Highest P-AUROC row: seed {best['seed']} / {best['variant']} = {best['P-AUROC']}.\n"
        log_text += "- Equal-capacity, diversity tracking, category changes, runtime, and memory cost are recorded in the CSV artifacts listed above.\n"
        append_experiment_log(Path("EXPERIMENT_LOG.md"), log_text)


if __name__ == "__main__":
    main()
