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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import AnomalyCLIP_lib
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from clip_value_features.fusion import fuse_patch_features
from memory_readout.adaptive_prototype import prototype_plus_evidence, query_adaptive_prototype_scores
from memory_readout.asymmetric_evidence import asymmetric_evidence_logit
from memory_readout.class_partition import matched_capacity_indices, partition_memory, random_label_partition, tensor_checksum
from memory_readout.controls import global_topk_then_split, shared_adaptive_prototype, static_centroid_logit, topk_label_voting
from memory_readout.diagnostics import append_basic_diag, false_area
from memory_readout.kernel_evidence import full_kernel_evidence, original_full_bank_voting, sigmoid_logits_to_probs, topk_kernel_evidence
from mrad_audit_utils import (
    copy_log,
    evaluate_results,
    init_results,
    load_config,
    load_model,
    make_loader,
    parse_args,
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


def checkpoint_dir(output_dir):
    path = Path(output_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_phase_iid_memory(cfg, seed, device):
    name = cfg["representation"]["name"]
    path = Path(cfg["phase_iid_result_dir"]) / "checkpoints" / f"seed_{seed}_{name}_memory.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase II-D memory checkpoint: {path}")
    memory = torch.load(path, map_location="cpu")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in memory.items()}, path


def representation_spec(cfg):
    spec = dict(cfg["representation"])
    spec["name"] = cfg["representation"]["name"]
    return spec


def variant_specs(cfg):
    d = cfg["source_selection"]
    primary = [
        {"name": "V0_OriginalFullBankVoting", "family": "v0", "tau": 1.0},
        {"name": f"V1_ClassSplit_FullKernelEvidence_tau{d['tau']}", "family": "v1", "tau": float(d["tau"])},
        {"name": f"V2_ClassSplit_TopKKernelEvidence_k{d['k']}_tau{d['tau']}", "family": "v2", "k": int(d["k"]), "tau": float(d["tau"])},
        {"name": f"V3_QueryAdaptive_ClassPrototype_k{d['k']}_tau{d['tau']}_taucls{d['tau_cls']}", "family": "v3", "k": int(d["k"]), "tau": float(d["tau"]), "tau_cls": float(d["tau_cls"])},
        {"name": f"V4_PrototypePlusEvidence_k{d['k']}_tau{d['tau']}_taucls{d['tau_cls']}_gamma{d['gamma']}", "family": "v4", "k": int(d["k"]), "tau": float(d["tau"]), "tau_cls": float(d["tau_cls"]), "gamma": float(d["gamma"])},
        {"name": f"V5_AsymmetricEvidence_k{d['k']}_tau{d['tau']}_aa1.0_an1.0", "family": "v5", "k": int(d["k"]), "tau": float(d["tau"]), "alpha_a": 1.0, "alpha_n": 1.0},
        {"name": "C1_StaticClassCentroids", "family": "c1", "tau_cls": float(d["tau_cls"])},
        {"name": f"C2_UniformTopKPrototype_k{d['k']}", "family": "c2", "k": int(d["k"]), "tau": float(d["tau"]), "tau_cls": float(d["tau_cls"])},
        {"name": f"C3_GlobalTopKThenSplit_k{d['k']}", "family": "c3", "k": int(d["k"]), "tau": float(d["tau"]), "tau_cls": float(d["tau_cls"])},
        {"name": f"C4_TopKLabelVoting_k{d['k']}", "family": "c4", "k": int(d["k"]), "tau": float(d["tau"])},
        {"name": "C5_RandomLabelPartition", "family": "c5", "k": int(d["k"]), "tau": float(d["tau"]), "tau_cls": float(d["tau_cls"])},
        {"name": "C6_MatchedCapacityPartition", "family": "c6", "k": int(d["k"]), "tau": float(d["tau"]), "tau_cls": float(d["tau_cls"])},
        {"name": f"C7_SharedAdaptivePrototype_k{d['k']}", "family": "c7", "k": int(d["k"]), "tau": float(d["tau"])},
        {"name": "C8_PrototypeLabelSwap", "family": "c8", "k": int(d["k"]), "tau": float(d["tau"]), "tau_cls": float(d["tau_cls"])},
    ]
    if cfg.get("limit") or cfg.get("run_mode") == "smoke":
        return primary[:6] + primary[6:10]
    if cfg.get("run_mode") != "sweep":
        return primary
    specs = list(primary)
    for tau in cfg["tau_values"]:
        specs.append({"name": f"V1_ClassSplit_FullKernelEvidence_tau{tau}", "family": "v1", "tau": float(tau)})
    for k in cfg["k_values"]:
        for tau in cfg["tau_values"]:
            specs.append({"name": f"V2_ClassSplit_TopKKernelEvidence_k{k}_tau{tau}", "family": "v2", "k": int(k), "tau": float(tau)})
            for tau_cls in cfg["tau_cls_values"]:
                specs.append({"name": f"V3_QueryAdaptive_ClassPrototype_k{k}_tau{tau}_taucls{tau_cls}", "family": "v3", "k": int(k), "tau": float(tau), "tau_cls": float(tau_cls)})
    for gamma in cfg["gamma_values"]:
        specs.append({"name": f"V4_PrototypePlusEvidence_k{d['k']}_tau{d['tau']}_taucls{d['tau_cls']}_gamma{gamma}", "family": "v4", "k": int(d["k"]), "tau": float(d["tau"]), "tau_cls": float(d["tau_cls"]), "gamma": float(gamma)})
    for aa, an in cfg["asymmetric_weights"]:
        specs.append({"name": f"V5_AsymmetricEvidence_k{d['k']}_tau{d['tau']}_aa{aa}_an{an}", "family": "v5", "k": int(d["k"]), "tau": float(d["tau"]), "alpha_a": float(aa), "alpha_n": float(an)})
    seen, unique = set(), []
    for spec in specs:
        if spec["name"] not in seen:
            seen.add(spec["name"])
            unique.append(spec)
    return unique


def source_stats_for_evidence(memory, part, k, tau):
    keys = memory["patch_keys"]
    sim_n = keys @ part["normal_keys"].T
    sim_a = keys @ part["anomaly_keys"].T
    en, _, _ = topk_kernel_evidence(sim_n, k, tau)
    ea, _, _ = topk_kernel_evidence(sim_a, k, tau)
    both = torch.cat([en, ea], dim=0).float()
    return {"evidence_mean": float(both.mean().item()), "evidence_std": float(both.std(unbiased=False).item())}


def readout_probs(query, memory, part, spec, seed, stats):
    fam = spec["family"]
    if fam == "v0":
        probs, _, _ = original_full_bank_voting(query, memory, spec.get("tau", 1.0))
        return probs, {}
    normal_keys, anomaly_keys = part["normal_keys"], part["anomaly_keys"]
    if fam == "v1":
        en = full_kernel_evidence(query @ normal_keys.T, spec["tau"])
        ea = full_kernel_evidence(query @ anomaly_keys.T, spec["tau"])
        return sigmoid_logits_to_probs(ea - en), {"en": en, "ea": ea}
    if fam == "v2":
        en, vals_n, idx_n = topk_kernel_evidence(query @ normal_keys.T, spec["k"], spec["tau"])
        ea, vals_a, idx_a = topk_kernel_evidence(query @ anomaly_keys.T, spec["k"], spec["tau"])
        return sigmoid_logits_to_probs(ea - en), {"en": en, "ea": ea, "idx_n": idx_n, "idx_a": idx_a}
    if fam == "v3":
        out = query_adaptive_prototype_scores(query, normal_keys, anomaly_keys, spec["k"], spec["tau"], spec["tau_cls"])
        return sigmoid_logits_to_probs(out["logit"]), out
    if fam == "v4":
        out = prototype_plus_evidence(query, normal_keys, anomaly_keys, spec["k"], spec["tau"], spec["tau_cls"], spec["gamma"], stats)
        return sigmoid_logits_to_probs(out["logit"]), out
    if fam == "v5":
        logit, en, ea, vals_n, vals_a, idx_n, idx_a = asymmetric_evidence_logit(query, normal_keys, anomaly_keys, spec["k"], spec["tau"], spec["alpha_a"], spec["alpha_n"])
        return sigmoid_logits_to_probs(logit), {"en": en, "ea": ea, "idx_n": idx_n, "idx_a": idx_a}
    if fam == "c1":
        logit, rn, ra = static_centroid_logit(query, normal_keys, anomaly_keys, spec.get("tau_cls", 1.0))
        return sigmoid_logits_to_probs(logit), {"rn": rn, "ra": ra}
    if fam == "c2":
        out = query_adaptive_prototype_scores(query, normal_keys, anomaly_keys, spec["k"], spec["tau"], spec["tau_cls"], uniform=True)
        return sigmoid_logits_to_probs(out["logit"]), out
    if fam == "c3":
        logit = global_topk_then_split(query, memory, spec["k"], spec["tau"], spec["tau_cls"])
        return sigmoid_logits_to_probs(logit), {}
    if fam == "c4":
        return topk_label_voting(query, memory, spec["k"], spec["tau"]), {}
    if fam == "c5":
        labels = random_label_partition(memory, seed)
        p = partition_memory(memory, labels_override=labels)
        out = query_adaptive_prototype_scores(query, p["normal_keys"], p["anomaly_keys"], spec["k"], spec["tau"], spec["tau_cls"])
        return sigmoid_logits_to_probs(out["logit"]), out
    if fam == "c6":
        n_sel, a_sel = matched_capacity_indices(part, seed)
        keys = memory["patch_keys"]
        out = query_adaptive_prototype_scores(query, keys[n_sel], keys[a_sel], spec["k"], spec["tau"], spec["tau_cls"])
        return sigmoid_logits_to_probs(out["logit"]), out
    if fam == "c7":
        p = shared_adaptive_prototype(query, memory, spec["k"], spec["tau"])
        return torch.stack([1.0 - p, p], dim=-1), {}
    if fam == "c8":
        out = query_adaptive_prototype_scores(query, normal_keys, anomaly_keys, spec["k"], spec["tau"], spec["tau_cls"])
        return sigmoid_logits_to_probs(-out["logit"]), out
    raise ValueError(f"Unknown readout family {fam}")


@torch.no_grad()
def evaluate_readout(model, memory, part, loader, cfg, spec, seed, device, evidence_stats):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    rep_spec = representation_spec(cfg)
    results = init_results(loader.dataset.obj_list)
    diag_rows, geom_rows, err_rows, loc_rows = [], [], [], []
    times = []
    baseline_patch_probs = None

    for index, items in enumerate(tqdm(loader, desc=spec["name"])):
        start = time.perf_counter()
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        outputs = extract_value_representations(model, image, rep_spec["layers"], device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        patches = fuse_patch_features(outputs, rep_spec)
        image_logits, _, _ = retrieval_logits(cls, memory["image_keys"], memory["image_values"])
        patch_probs, aux = readout_probs(patches, memory, part, spec, seed, evidence_stats)
        if spec["family"] != "v0":
            v0_probs, _ = readout_probs(patches, memory, part, {"family": "v0", "tau": 1.0}, seed, evidence_stats)
            baseline_patch_probs = v0_probs[..., 1]
        append_basic_diag(diag_rows, seed, spec["name"], cls_name, patch_probs[..., 1], baseline_patch_probs)
        amap = AnomalyCLIP_lib.get_similarity_map(patch_probs, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        fp, fn = false_area(gt_mask, amap)
        loc_rows.append({"seed": seed, "variant": spec["name"], "category": cls_name, "false_positive_area_ratio": fp, "false_negative_area_ratio": fn, "anomaly_map_smoothness": float(torch.diff(amap.flatten()).abs().mean().item())})
        times.append(time.perf_counter() - start)
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    return metrics, diag_rows, geom_rows, err_rows, loc_rows, float(np.mean(times)) if times else float("nan")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def append_log(output_dir, audit_status, sanity, rows):
    text = "\n## Phase III-A — Class-Split Query-Adaptive Memory Readout\n\n"
    text += f"- Audited repositories and commit hashes: `{json.dumps(audit_status)}`.\n"
    text += "- Implementation files: `memory_readout/*`, `experiments/class_split_readout_audit.py`, `configs/class_split_readout_audit.yaml`, `docs/reference_code_audit_phase_III_A.md`.\n"
    text += f"- Command executed: `{' '.join(sys.argv)}`.\n"
    text += "- Source-selection protocol: fixed defaults; no target labels used for hyperparameter selection.\n"
    text += f"- Sanity checks: `{json.dumps(sanity, default=json_default)}`.\n"
    text += f"- Raw result tables: `{output_dir / 'summary_metrics.csv'}`, `per_category_metrics.csv`, `retrieval_diagnostics.csv`, `adaptive_prototype_geometry.csv`, `error_correction.csv`, `localization_diagnostics.csv`.\n"
    text += f"- Completed metric rows: {len(rows)}.\n"
    with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write(text)


def main():
    args = parse_args("Phase III-A class-split query-adaptive memory readout")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    if args.seed is not None:
        cfg["seeds"] = [int(args.seed)]
    if args.limit is not None:
        cfg["run_mode"] = "smoke"
    output_dir = Path(cfg.get("output_dir", "results/class_split_readout"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, cfg, int(cfg.get("seeds", [0])[0]))
    write_json(output_dir / "environment.json", package_versions())
    write_json(output_dir / "source_selection.json", {"protocol": cfg["source_selection"]["protocol"], "selected_settings": cfg["source_selection"], "random_seeds": cfg.get("seeds", [0, 1, 2]), "spr": "SPR intentionally excluded to isolate class-conditioned memory readout."})

    device = cfg.get("device", "cuda:0")
    model = load_model(cfg, device)
    _, target_loader = make_loader(cfg, cfg["dataset"], cfg.get("target_mode", "test"), shuffle=False)
    specs = variant_specs(cfg)
    audit_status = {}
    for name in ("DMN", "Tip-Adapter", "TDA"):
        repo = Path("../reference_repos") / name
        if repo.exists():
            import subprocess
            audit_status[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    summary_rows, per_category_rows, diag_rows, geom_rows, err_rows, loc_rows, logs = [], [], [], [], [], [], []
    failures = []
    sanity = {"spr_excluded": True, "variants": [s["name"] for s in specs], "baseline_checks": {}}
    try:
        for seed in cfg.get("seeds", [0, 1, 2]):
            seed = int(seed)
            set_seed(seed)
            memory, mem_path = load_phase_iid_memory(cfg, seed, device)
            part = partition_memory(memory)
            evidence_stats = source_stats_for_evidence(memory, part, cfg["source_selection"]["k"], cfg["source_selection"]["tau"])
            partition_record = {k: v for k, v in part.items() if not isinstance(v, torch.Tensor)}
            partition_record.update({"memory_checkpoint": str(mem_path), "normal_count": part["normal_count"], "anomaly_count": part["anomaly_count"], "source_evidence_stats": evidence_stats})
            write_json(output_dir / f"memory_partition_seed_{seed}.json", partition_record)
            if seed == int(cfg.get("seeds", [0])[0]):
                write_json(output_dir / "memory_partition.json", partition_record)
            for spec in specs:
                run_dir = output_dir / f"seed_{seed}" / spec["name"]
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "metrics.json"
                if cfg.get("resume", True) and metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    infer_time = float(metrics.get("mean_inference_time_per_image", float("nan")))
                else:
                    metrics, drows, grows, erows, lrows, infer_time = evaluate_readout(model, memory, part, target_loader, cfg, spec, seed, device, evidence_stats)
                    metrics["mean_inference_time_per_image"] = infer_time
                    metrics_path.write_text(json.dumps(metrics, indent=2, default=json_default), encoding="utf-8")
                    diag_rows.extend(drows); geom_rows.extend(grows); err_rows.extend(erows); loc_rows.extend(lrows)
                mean = metrics["mean"]
                row = {"seed": seed, "variant": spec["name"], "status": "completed", "Class split": spec["family"] not in {"v0", "c4", "c7"}, "Top-k": "k" in spec, "Query-adaptive prototype": spec["family"] in {"v3", "v4", "c2", "c3", "c5", "c6", "c7", "c8"}, "Correct labels": spec["family"] != "c5", "I-AUROC": mean.get("I-AUROC"), "I-AP": mean.get("I-AP"), "P-AUROC": mean.get("P-AUROC"), "PRO": mean.get("PRO"), "inference_time_per_image_sec": infer_time, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0, "visual_memory_size": int(memory["patch_keys"].shape[0]), "additional_readout_memory": 0}
                summary_rows.append(row)
                for category, vals in metrics["per_class"].items():
                    per_category_rows.append({"seed": seed, "variant": spec["name"], "category": category, "I-AUROC": vals.get("I-AUROC"), "I-AP": vals.get("I-AP"), "P-AUROC": vals.get("P-AUROC"), "PRO": vals.get("PRO")})
                logs.append(f"seed {seed} {spec['name']}: I-AUROC={mean.get('I-AUROC'):.4f}, P-AUROC={mean.get('P-AUROC'):.4f}, PRO={mean.get('PRO'):.4f}")
                if spec["family"] == "v0":
                    refs = cfg.get("baseline_reference", {})
                    tols = cfg.get("baseline_tolerance", {})
                    sanity["baseline_checks"][seed] = {m: {"value": float(mean[m]), "reference": float(refs[m]), "pass": abs(float(mean[m]) - float(refs[m])) <= float(tols[m])} for m in refs}
                    if cfg.get("fail_on_baseline_mismatch", True) and not cfg.get("limit") and not all(v["pass"] for v in sanity["baseline_checks"][seed].values()):
                        raise RuntimeError(f"Phase II-D baseline mismatch for seed {seed}: {sanity['baseline_checks'][seed]}")
                torch.cuda.empty_cache()
    except Exception:
        tb = traceback.format_exc()
        failures.append(tb)
        (output_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        raise
    finally:
        write_csv(output_dir / "summary_metrics.csv", summary_rows, ["seed", "variant", "status", "Class split", "Top-k", "Query-adaptive prototype", "Correct labels", "I-AUROC", "I-AP", "P-AUROC", "PRO", "inference_time_per_image_sec", "peak_gpu_memory_mb", "visual_memory_size", "additional_readout_memory"])
        write_csv(output_dir / "per_category_metrics.csv", per_category_rows, ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
        write_csv(output_dir / "retrieval_diagnostics.csv", diag_rows, ["seed", "variant", "category", "score_mean", "score_std", "score_min", "score_max", "score_corr_vs_v0", "score_abs_change_vs_v0"])
        write_csv(output_dir / "adaptive_prototype_geometry.csv", geom_rows, ["seed", "variant"])
        write_csv(output_dir / "error_correction.csv", err_rows, ["seed", "variant"])
        write_csv(output_dir / "localization_diagnostics.csv", loc_rows, ["seed", "variant", "category", "false_positive_area_ratio", "false_negative_area_ratio", "anomaly_map_smoothness"])
        success = []
        for row in summary_rows:
            if row["seed"] == int(cfg.get("seeds", [0])[0]):
                success.append({"Variant": row["variant"], "Class split": row["Class split"], "Top-k": row["Top-k"], "Query-adaptive prototype": row["Query-adaptive prototype"], "Correct labels": row["Correct labels"], "I-AUROC": row["I-AUROC"], "P-AUROC": row["P-AUROC"], "PRO": row["PRO"]})
        write_csv(output_dir / "success_control_table.csv", success, ["Variant", "Class split", "Top-k", "Query-adaptive prototype", "Correct labels", "I-AUROC", "P-AUROC", "PRO"])
        write_json(output_dir / "sanity_checks.json", sanity)
        lines = ["# Phase III-A Execution Summary", "", f"- Rows: {len(summary_rows)}", f"- Failures: {len(failures)}", "", "## Errors"]
        lines.extend([f"- {x}" for x in failures] or ["- None"])
        (output_dir / "EXECUTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        copy_log(output_dir, logs)
        append_log(output_dir, audit_status, sanity, summary_rows)


if __name__ == "__main__":
    main()
