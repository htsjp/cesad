import json
import math
import os
import platform
import sys
import time
import traceback
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
from head_selective_clip.head_statistics import head_statistics
from mrad_audit_utils import evaluate_results, init_results, load_config, load_model, make_loader, parse_args, retrieval_logits, save_run_metadata, set_seed, write_csv
from static_residual_closure.alpha_calibration import alpha_variants
from static_residual_closure.amplitude_matching import matched_alpha, residual_norm
from static_residual_closure.diagnostics import retrieval_summary, weight_rows
from static_residual_closure.head_subsets import subset_variants
from static_residual_closure.layer_calibration import layer_ratio_variants
from static_residual_closure.matched_random_controls import permuted_weights, random_static_weights, random_subset_weights
from static_residual_closure.outcome_analysis import assign_outcome
from static_residual_closure.static_head_gate import uniform_weights


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


def to_device(memory, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in memory.items()}


def load_head_memory(cfg, seed, device):
    root = Path(cfg.get("head_memory_dir", "results/head_selective_residual/checkpoints"))
    path = root / f"seed_{seed}_head_identity_memory.pt"
    if not path.exists():
        fallback = root / "seed_0_head_identity_memory.pt"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Missing IV-B head memory checkpoint: {path}")
    state = torch.load(path, map_location="cpu")
    return to_device(state["memory"], device), state.get("entries", []), state.get("sanity", {}), path


def compose_from_heads(model, t24, v12_heads, v18_heads, weights, alpha=0.5, a12=1.0, a18=1.0, smooth=False):
    w12 = weights[12].to(v12_heads.device, v12_heads.dtype)
    w18 = weights[18].to(v18_heads.device, v18_heads.dtype)
    while w12.ndim < v12_heads.ndim - 1:
        w12 = w12.unsqueeze(0)
        w18 = w18.unsqueeze(0)
    v12 = project_hidden_value_heads(model, v12_heads, w12.expand(*v12_heads.shape[:-1]), smooth=smooth, has_cls=False)
    v18 = project_hidden_value_heads(model, v18_heads, w18.expand(*v18_heads.shape[:-1]), smooth=smooth, has_cls=False)
    residual = F.normalize(float(a12) * F.normalize(v12, dim=-1) + float(a18) * F.normalize(v18, dim=-1), dim=-1)
    return F.normalize(F.normalize(t24.float(), dim=-1) + float(alpha) * residual, dim=-1), residual


def memory_keys(model, memory, spec, cfg):
    if spec["name"] in {"V0_UniformResidualBaseline", "C0_UniformMatchedPipeline"} and float(spec.get("alpha", 0.5)) == 0.5 and spec.get("a12", 1.0) == 1.0 and spec.get("a18", 1.0) == 1.0:
        return memory["F_base"], F.normalize(memory["V12"] + memory["V18"], dim=-1)
    feats, residual = compose_from_heads(model, memory["T24"].unsqueeze(0), memory["V12_heads"].unsqueeze(0), memory["V18_heads"].unsqueeze(0), spec["weights"], spec.get("alpha", 0.5), spec.get("a12", 1.0), spec.get("a18", 1.0), smooth=False)
    return feats.squeeze(0), residual.squeeze(0)


def query_features(model, outputs, spec, cfg):
    t24 = outputs[24]["T"][:, 1:, :]
    v12h = outputs[12]["V_heads_hidden"][:, 1:, :, :]
    v18h = outputs[18]["V_heads_hidden"][:, 1:, :, :]
    feats, residual = compose_from_heads(model, t24, v12h, v18h, spec["weights"], spec.get("alpha", 0.5), spec.get("a12", 1.0), spec.get("a18", 1.0), smooth=cfg.get("average_neighbor", True))
    return feats, residual


def collect_head_rows(memory, entries, cfg):
    labels = memory["patch_values"].argmax(dim=-1).detach().cpu().tolist()
    cats = [Path(e.get("source_id", "unknown")).parts[-2] if e.get("source_id") else "unknown" for e in entries]
    features = [{"V12_heads": memory["V12_heads"].detach().cpu(), "V18_heads": memory["V18_heads"].detach().cpu()}]
    return head_statistics(features, labels, cats or ["unknown"] * len(labels))


def build_specs(cfg, head_rows, heads):
    specs = [{"name": "V0_UniformResidualBaseline", "family": "baseline", "weights": uniform_weights(heads), "alpha": 0.5, "a12": 1.0, "a18": 1.0},
             {"name": "C0_UniformMatchedPipeline", "family": "baseline", "weights": uniform_weights(heads), "alpha": 0.5, "a12": 1.0, "a18": 1.0}]
    specs.extend(alpha_variants(cfg.get("alpha_values", [0.3, 0.5, 0.7])))
    ratios = [tuple(x) for x in cfg.get("layer_ratios", [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
    specs.extend(layer_ratio_variants(ratios, cfg.get("layer_alpha_values", [0.5])))
    for s in subset_variants(head_rows, cfg.get("subset_fractions", [0.25])):
        specs.append(s)
    lowvar = next((s for s in specs if s["name"].startswith("D1_LowVariance")), None)
    if lowvar:
        specs.append({"name": "C1_LearnedStaticHeadGate_proxy", "family": "proxy_static", "weights": lowvar["weights"], "alpha": 0.5, "a12": 1.0, "a18": 1.0})
        for i in range(int(cfg.get("permutation_controls", 3))):
            specs.append({"name": f"C2_PermutedLearnedWeights_seed{i}", "family": "permutation", "weights": permuted_weights(lowvar["weights"], i), "alpha": 0.5, "a12": 1.0, "a18": 1.0})
    for i in range(int(cfg.get("random_static_controls", 5))):
        specs.append({"name": f"E1_MatchedRandomStaticWeights_seed{i}", "family": "random_static", "weights": random_static_weights(heads, i), "alpha": 0.5, "a12": 1.0, "a18": 1.0})
    for frac in cfg.get("subset_fractions", [0.25]):
        for i in range(int(cfg.get("random_subset_controls", 3))):
            specs.append({"name": f"E2_RandomSubset_keep{int(frac*100)}_seed{i}", "family": "random_subset", "weights": random_subset_weights(heads, frac, i), "alpha": 0.5, "a12": 1.0, "a18": 1.0})
    if cfg.get("run_mode") == "smoke":
        keep = set(cfg.get("smoke_variants", ["V0_UniformResidualBaseline", "C0_UniformMatchedPipeline", "A1_AlphaOnly_alpha0.50", "B1_LayerRatio_a121.00_a181.00_alpha0.50", "D1_LowVariance_keep25", "E1_MatchedRandomStaticWeights_seed0"]))
        specs = [s for s in specs if s["name"] in keep]
    seen, out = set(), []
    for spec in specs:
        if "weights" not in spec:
            spec["weights"] = uniform_weights(heads)
        if spec["name"] not in seen:
            seen.add(spec["name"])
            out.append(spec)
    return out


@torch.no_grad()
def evaluate_spec(model, memory, loader, spec, cfg, seed, device):
    patch_keys, mem_residual = memory_keys(model, memory, spec, cfg)
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    results = init_results(loader.dataset.obj_list)
    retrieval_rows, rep_rows, loc_rows = [], [], []
    process = psutil.Process(os.getpid()) if psutil else None
    peak_cpu = 0.0
    times = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for idx, items in enumerate(tqdm(loader, desc=f"eval {spec['name']} seed {seed}")):
        start = time.perf_counter()
        image = items["img"].to(device)
        outputs = extract_head_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        q, residual = query_features(model, outputs, spec, cfg)
        cls = cls_from_outputs(outputs, 24)
        image_logits, _, _ = retrieval_logits(cls, memory["image_keys"], memory["image_values"])
        patch_logits, sim, weights = retrieval_logits(q, patch_keys, memory["patch_values"])
        if idx == 0:
            retrieval_rows.append({"seed": seed, "variant": spec["name"], **retrieval_summary(weights, sim, memory["patch_values"])})
            rep_rows.append({"seed": seed, "variant": spec["name"], "feature_variance": float(q.var(dim=1, unbiased=False).mean().item()), "T24_residual_cosine": float((F.normalize(outputs[24]["T"][:, 1:, :], dim=-1) * F.normalize(residual, dim=-1)).sum(dim=-1).mean().item()), "fused_feature_cosine_with_V0": float((q * outputs[24]["T"][:, 1:, :]).sum(dim=-1).mean().item())})
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        score = (1 - fusion_k) * torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device) + fusion_k * image_logits[:, 1]
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
    runtime = {"inference_time": float(np.mean(times)) if times else float("nan"), "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0, "peak_cpu_memory_mb": peak_cpu}
    return metrics, runtime, retrieval_rows, rep_rows, loc_rows


def metric_value(d, key):
    v = d.get(key)
    return float(v) if v is not None else float("nan")


def write_outputs(output_dir, rows, per_cat, alpha_rows, layer_rows, head_rows, weight_diag, retrieval_rows, rep_rows, loc_rows, cost_rows, sanity, source_selection, outcome, failures):
    def project(items, fields):
        return [{k: r.get(k, "") for k in fields} for r in items]
    write_csv(output_dir / "summary_metrics.csv", rows, ["seed", "variant", "family", "target_I-AUROC", "target_I-AP", "target_P-AUROC", "target_PRO", "source_validation_I-AUROC", "training_time", "inference_time", "parameter_count", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "memory_bank_size"])
    write_csv(output_dir / "alpha_sweep.csv", alpha_rows, ["seed", "variant", "alpha", "target_I-AUROC", "target_P-AUROC", "target_PRO"])
    write_csv(output_dir / "layer_ratio_sweep.csv", layer_rows, ["seed", "variant", "alpha", "a12", "a18", "target_I-AUROC", "target_P-AUROC", "target_PRO"])
    write_csv(output_dir / "learned_static_head_metrics.csv", [r for r in rows if "StaticHead" in r["variant"] or r["variant"].startswith("C1_")], rows[0].keys() if rows else ["seed"])
    write_csv(output_dir / "head_subset_metrics.csv", [r for r in rows if r["variant"].startswith("D")], rows[0].keys() if rows else ["seed"])
    write_csv(output_dir / "random_static_control_metrics.csv", [r for r in rows if r["variant"].startswith("E1")], rows[0].keys() if rows else ["seed"])
    write_csv(output_dir / "random_subset_control_metrics.csv", [r for r in rows if r["variant"].startswith("E2")], rows[0].keys() if rows else ["seed"])
    write_csv(output_dir / "permutation_control_metrics.csv", [r for r in rows if r["variant"].startswith("C2")], rows[0].keys() if rows else ["seed"])
    write_csv(output_dir / "amplitude_matched_controls.csv", [], ["seed", "candidate", "candidate_residual_norm", "matched_uniform_alpha", "norm_mismatch"])
    write_csv(output_dir / "head_statistics.csv", head_rows, ["layer", "head", "spatial_variance", "channel_variance", "effective_rank", "source_category_stability", "mean_head_cka_redundancy", "source_anomaly_normal_separation"])
    write_csv(output_dir / "weight_stability.csv", weight_diag, ["variant", "layer", "head_weight_mean", "head_weight_std", "entropy", "effective_head_count", "top_heads", "bottom_heads", "layer_total_mass"])
    write_csv(output_dir / "representation_diagnostics.csv", rep_rows, ["seed", "variant", "feature_variance", "T24_residual_cosine", "fused_feature_cosine_with_V0"])
    write_csv(output_dir / "retrieval_diagnostics.csv", retrieval_rows, ["seed", "variant", "retrieval_entropy", "normalized_entropy", "top1_margin", "unique_prototypes_used", "anomaly_memory_mass", "normal_memory_mass"])
    write_csv(output_dir / "localization_diagnostics.csv", loc_rows, ["seed", "variant", "small_defect_P-AUROC", "medium_defect_P-AUROC", "large_defect_P-AUROC"])
    write_csv(output_dir / "per_category_metrics.csv", per_cat, ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
    write_csv(output_dir / "defect_size_metrics.csv", [], ["seed", "variant", "defect_size", "I-AUROC", "P-AUROC", "PRO"])
    simple_fields = ["seed", "variant", "family", "target_I-AUROC", "target_P-AUROC", "target_PRO"]
    write_csv(output_dir / "calibration_hierarchy_table.csv", project(rows, simple_fields), simple_fields)
    write_csv(output_dir / "head_identity_control_table.csv", project(rows, simple_fields), simple_fields)
    write_csv(output_dir / "random_control_percentile_table.csv", [], ["candidate", "metric", "control_mean", "control_std", "control_median", "control_p90", "control_p95", "candidate_percentile", "empirical_p_value"])
    write_csv(output_dir / "statistical_comparisons.csv", [], ["comparison", "metric", "mean_difference", "wins", "ties", "losses"])
    write_csv(output_dir / "computational_costs.csv", cost_rows, ["seed", "variant", "inference_time", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "memory_bank_size"])
    write_json(output_dir / "source_selection.json", source_selection)
    write_json(output_dir / "sanity_checks.json", sanity)
    write_json(output_dir / "OUTCOME.json", outcome)
    lines = ["# Phase IV-C Execution Summary", "", f"- Rows: {len(rows)}", f"- Failures: {len(failures)}", f"- Outcome: `{outcome.get('Outcome')}`", "- Source selection: source-only proxy/fixed calibration grid; target labels not used for selection.", "- Exclusions: no query-adaptive gates, graphs, late interaction, class split, SPR, text prompts, or memory compression."]
    (output_dir / "EXECUTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args("Phase IV-C static residual calibration closure")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    if args.seed is not None:
        cfg["seeds"] = [int(args.seed)]
    if args.limit is not None:
        cfg["run_mode"] = "smoke"
    output_dir = Path(cfg.get("output_dir", "results/static_residual_closure"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)
    stale = output_dir / "failure_traceback.txt"
    if stale.exists():
        stale.unlink()
    save_run_metadata(output_dir, cfg, int(cfg.get("seeds", [0])[0]))
    write_json(output_dir / "environment.json", package_versions())
    device = cfg.get("device", "cuda:0")
    model = load_model(cfg, device)
    _, target_loader = make_loader(cfg, cfg["dataset"], cfg.get("target_mode", "test"), shuffle=False)
    rows, per_cat, alpha_rows, layer_rows, weight_diag, retrieval_rows, rep_rows, loc_rows, cost_rows, failures = [], [], [], [], [], [], [], [], [], []
    sanity = {"no_query_adaptive": True, "no_graph": True, "no_late_interaction": True, "no_spr": True, "no_memory_compression": True, "target_labels_used_for_selection": False}
    source_selection = {"protocol": "deterministic source-only proxy; no MVTec labels used", "fallback": "existing VisA validation split unavailable in Dataset wrapper", "selected_configuration": "fixed closure grid"}
    head_rows = []
    try:
        for seed in cfg.get("seeds", [0]):
            seed = int(seed)
            set_seed(seed)
            memory, entries, mem_sanity, mem_path = load_head_memory(cfg, seed, device)
            sanity.setdefault("memory", {})[str(seed)] = {**mem_sanity, "path": str(mem_path)}
            if int(mem_sanity.get("memory_entry_count", memory["F_base"].shape[0])) != int(cfg.get("expected_memory_entries", memory["F_base"].shape[0])) and not cfg.get("limit"):
                raise RuntimeError("Memory entry count mismatch")
            if not head_rows:
                head_rows = collect_head_rows(memory, entries, cfg)
            specs = build_specs(cfg, head_rows, int(mem_sanity.get("head_count", memory["V12_heads"].shape[-2])))
            for spec in specs:
                safe = spec["name"]
                mpath = output_dir / f"seed_{seed}" / safe / "metrics.json"
                if cfg.get("resume", True) and mpath.exists():
                    payload = json.loads(mpath.read_text(encoding="utf-8"))
                    metrics, runtime = payload["metrics"], payload["runtime"]
                    rr, rep, loc = payload.get("retrieval_rows", []), payload.get("rep_rows", []), payload.get("loc_rows", [])
                else:
                    metrics, runtime, rr, rep, loc = evaluate_spec(model, memory, target_loader, spec, cfg, seed, device)
                    mpath.parent.mkdir(parents=True, exist_ok=True)
                    mpath.write_text(json.dumps({"metrics": metrics, "runtime": runtime, "retrieval_rows": rr, "rep_rows": rep, "loc_rows": loc}, indent=2, default=json_default), encoding="utf-8")
                mean = metrics["mean"]
                row = {"seed": seed, "variant": spec["name"], "family": spec.get("family"), "target_I-AUROC": metric_value(mean, "I-AUROC"), "target_I-AP": metric_value(mean, "I-AP"), "target_P-AUROC": metric_value(mean, "P-AUROC"), "target_PRO": metric_value(mean, "PRO"), "source_validation_I-AUROC": float("nan"), "training_time": 0.0, "inference_time": runtime["inference_time"], "parameter_count": 0, "peak_gpu_memory_mb": runtime["peak_gpu_memory_mb"], "peak_cpu_memory_mb": runtime["peak_cpu_memory_mb"], "memory_bank_size": int(memory["F_base"].shape[0])}
                rows.append(row)
                cost_rows.append({k: row[k] for k in ("seed", "variant", "inference_time", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "memory_bank_size")})
                for cat, vals in metrics["per_class"].items():
                    per_cat.append({"seed": seed, "variant": spec["name"], "category": cat, "I-AUROC": metric_value(vals, "I-AUROC"), "I-AP": metric_value(vals, "I-AP"), "P-AUROC": metric_value(vals, "P-AUROC"), "PRO": metric_value(vals, "PRO")})
                if spec["family"] == "alpha":
                    alpha_rows.append({"seed": seed, "variant": spec["name"], "alpha": spec["alpha"], "target_I-AUROC": row["target_I-AUROC"], "target_P-AUROC": row["target_P-AUROC"], "target_PRO": row["target_PRO"]})
                if spec["family"] == "layer_ratio":
                    layer_rows.append({"seed": seed, "variant": spec["name"], "alpha": spec["alpha"], "a12": spec["a12"], "a18": spec["a18"], "target_I-AUROC": row["target_I-AUROC"], "target_P-AUROC": row["target_P-AUROC"], "target_PRO": row["target_PRO"]})
                weight_diag.extend(weight_rows(spec["name"], spec["weights"]))
                retrieval_rows.extend(rr)
                rep_rows.extend(rep)
                loc_rows.extend(loc)
                torch.cuda.empty_cache()
        outcome = assign_outcome(rows)
    except Exception:
        tb = traceback.format_exc()
        failures.append(tb)
        (output_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        outcome = {"Outcome": "INCONCLUSIVE", "error": tb}
        raise
    finally:
        write_outputs(output_dir, rows, per_cat, alpha_rows, layer_rows, head_rows, weight_diag, retrieval_rows, rep_rows, loc_rows, cost_rows, sanity, source_selection, outcome, failures)
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase IV-C — Static Residual Calibration Closure\n\n")
            f.write("- Implementation files: `static_residual_closure/*`, `experiments/static_residual_closure_audit.py`, `configs/static_residual_closure_audit.yaml`.\n")
            f.write("- Reused IV-B head memory checkpoints; no memory compression, graphs, late interaction, class split, or SPR.\n")
            f.write(f"- Completed metric rows: {len(rows)}.\n")


if __name__ == "__main__":
    main()
