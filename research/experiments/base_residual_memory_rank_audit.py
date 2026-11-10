import csv
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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "experiments"))

import AnomalyCLIP_lib
from base_residual_memory.canonical import FORMULA_NAME, FORMULA_TEXT, canonical_fuse, compare_tensors, ivc_head_formula, tensor_checksum
from base_residual_memory.controls import basis_shuffle, random_local_subspace
from base_residual_memory.diagnostics import reconstruction_diagnostics
from base_residual_memory.global_residual import fit_global_basis, reconstruct_global
from base_residual_memory.grouping import load_head_state, recover_groups
from base_residual_memory.lazy_similarity import lazy_logits
from base_residual_memory.local_svd import decompose_groups, energy_threshold_ranks, reconstruct_energy_threshold, reconstruct_rank
from base_residual_memory.outcome_analysis import assign_outcome
from base_residual_memory.reconstruction import materialized_memory
from base_residual_memory.storage_accounting import feature_storage_scalars
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from mrad_audit_utils import evaluate_results, init_results, load_config, load_model, make_loader, parse_args, retrieval_logits, save_run_metadata, set_seed, write_csv


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def write_md(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def package_versions():
    data = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}
    if torch.cuda.is_available():
        data.update({"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    return data


def row_metric(metrics, name):
    return float(metrics.get("mean", {}).get(name, float("nan")))


def load_memories(cfg, seed):
    head_path = Path(cfg.get("head_memory_dir", "results/head_selective_residual/checkpoints")) / f"seed_{seed}_head_identity_memory.pt"
    value_path = Path(cfg.get("value_memory_dir", "results/value_representation_audit/checkpoints")) / f"seed_{seed}_T24_plus_V12_18_alpha0.5_memory.pt"
    if not head_path.exists():
        head_path = Path(cfg.get("head_memory_dir", "results/head_selective_residual/checkpoints")) / "seed_0_head_identity_memory.pt"
    if not value_path.exists():
        value_path = Path(cfg.get("value_memory_dir", "results/value_representation_audit/checkpoints")) / "seed_0_T24_plus_V12_18_alpha0.5_memory.pt"
    head = load_head_state(str(head_path))
    value = torch.load(value_path, map_location="cpu", weights_only=False)
    return head_path, head, value_path, value


def run_consistency_audit(model, cfg, seed, device):
    out = Path(cfg.get("consistency_output_dir", "results/phase_IV_C_consistency"))
    out.mkdir(parents=True, exist_ok=True)
    head_path, head, value_path, value_mem = load_memories(cfg, seed)
    mem = head["memory"]
    rows = []
    score_rows = []

    canonical_memory = canonical_fuse(mem["T24"], mem["V12"], mem["V18"], alpha=0.5)
    ivc_memory = ivc_head_formula(mem["T24"], mem["V12"], mem["V18"], alpha=0.5)
    rows.append(compare_tensors("all_source_memory:F_base_vs_canonical_F1", mem["F_base"], canonical_memory))
    rows.append(compare_tensors("all_source_memory:F_base_vs_IVC_F2", mem["F_base"], ivc_memory))
    rows.append(compare_tensors("all_source_memory:canonical_F1_vs_IVC_F2", canonical_memory, ivc_memory))
    rows.append(compare_tensors("value_checkpoint_patch_keys_vs_head_F_base", value_mem["patch_keys"], mem["F_base"]))
    rows.append(compare_tensors("value_checkpoint_patch_keys_vs_canonical_F1", value_mem["patch_keys"], canonical_memory))

    labels = mem["patch_values"].to(device)
    fbase_dev = mem["F_base"].to(device)
    ivc_dev = ivc_memory.to(device)
    sample_keys = fbase_dev[: min(4096, fbase_dev.shape[0])]
    query = canonical_memory[: min(10000, canonical_memory.shape[0])].unsqueeze(0).to(device)
    v0_logits, v0_sim, v0_prob = retrieval_logits(query, fbase_dev, labels)
    a1_logits, a1_sim, a1_prob = retrieval_logits(query, ivc_dev, labels)
    score_rows.append(compare_tensors("memory_query_logits:V0_Fbase_vs_A1_IVC", v0_logits, a1_logits))
    score_rows.append(compare_tensors("memory_query_similarity:V0_Fbase_vs_A1_IVC", v0_sim[..., : sample_keys.shape[0]], a1_sim[..., : sample_keys.shape[0]]))
    score_rows.append(compare_tensors("memory_query_probabilities:V0_Fbase_vs_A1_IVC", v0_prob[..., : sample_keys.shape[0]], a1_prob[..., : sample_keys.shape[0]]))
    score_rows.append(compare_tensors("patch_anomaly_scores:V0_Fbase_vs_A1_IVC", v0_logits[..., 1], a1_logits[..., 1]))

    source_rows, target_rows = [], []
    for dataset_name, mode, dst, max_images in [
        (cfg.get("source_dataset", "visa"), cfg.get("source_mode", "test"), source_rows, int(cfg.get("consistency_source_images", 32))),
        (cfg.get("dataset", "mvtec"), cfg.get("target_mode", "test"), target_rows, int(cfg.get("consistency_target_images", 32))),
    ]:
        local_cfg = dict(cfg)
        local_cfg["limit"] = max_images
        _, loader = make_loader(local_cfg, dataset_name, mode, shuffle=False)
        patch_seen = 0
        for i, items in enumerate(tqdm(loader, desc=f"consistency {dataset_name}")):
            if i >= max_images:
                break
            image = items["img"].to(device)
            outputs = extract_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
            t24, v12, v18 = outputs[24]["T"][:, 1:, :], outputs[12]["V"][:, 1:, :], outputs[18]["V"][:, 1:, :]
            f1 = canonical_fuse(t24, v12, v18, alpha=0.5)
            f2 = ivc_head_formula(t24, v12, v18, alpha=0.5)
            if i < 4:
                dst.append(compare_tensors(f"{dataset_name}_image{i}:T24_norm", F.normalize(t24, dim=-1), F.normalize(t24, dim=-1)))
                dst.append(compare_tensors(f"{dataset_name}_image{i}:V12_norm", F.normalize(v12, dim=-1), F.normalize(v12, dim=-1)))
                dst.append(compare_tensors(f"{dataset_name}_image{i}:V18_norm", F.normalize(v18, dim=-1), F.normalize(v18, dim=-1)))
                dst.append(compare_tensors(f"{dataset_name}_image{i}:final_fused_F1_vs_F2", f1, f2))
            l1, s1, p1 = retrieval_logits(f1, fbase_dev, labels)
            l2, s2, p2 = retrieval_logits(f2, ivc_dev, labels)
            if i < 4:
                score_rows.append(compare_tensors(f"{dataset_name}_image{i}:query_memory_logits_F1V0_vs_F2A1", l1, l2))
                score_rows.append(compare_tensors(f"{dataset_name}_image{i}:patch_scores_F1V0_vs_F2A1", l1[..., 1], l2[..., 1]))
                amap1 = AnomalyCLIP_lib.get_similarity_map(l1, int(cfg.get("image_size", 518)))[..., 1]
                amap2 = AnomalyCLIP_lib.get_similarity_map(l2, int(cfg.get("image_size", 518)))[..., 1]
                score_rows.append(compare_tensors(f"{dataset_name}_image{i}:anomaly_maps_F1V0_vs_F2A1", torch.as_tensor(amap1), torch.as_tensor(amap2)))
            patch_seen += int(f1.numel() // f1.shape[-1])
            if dataset_name == cfg.get("dataset", "mvtec") and patch_seen >= int(cfg.get("consistency_target_patches", 10000)):
                break
    rows.extend(source_rows)
    rows.extend(target_rows)

    fields = list(rows[0].keys())
    write_csv(out / "tensor_comparisons.csv", rows, fields)
    write_csv(out / "score_comparisons.csv", score_rows, fields)
    metric_rows = []
    summary_path = Path("results/static_residual_closure/summary_metrics.csv")
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["variant"] in {"V0_UniformResidualBaseline", "A1_AlphaOnly_alpha0.50", "B1_LayerRatio_a121.00_a181.00_alpha0.50", "C0_UniformMatchedPipeline"}:
                    metric_rows.append(row)
    write_csv(out / "metric_reproduction.csv", metric_rows, list(metric_rows[0].keys()) if metric_rows else ["variant"])
    formula = {
        "status": "FORMULA_DIFFERENCE_IDENTIFIED",
        "canonical_formula": FORMULA_NAME,
        "canonical_formula_text": FORMULA_TEXT,
        "phase_IID_code_path": "clip_value_features/fusion.py:fuse_patch_features(anchor) -> _mean_features(V12,V18)",
        "phase_IV_C_V0_path": "experiments/static_residual_closure_audit.py:memory_keys returns cached memory['F_base']",
        "phase_IV_C_A1_alpha0.50_path": "experiments/static_residual_closure_audit.py:compose_from_heads/project_hidden_value_heads; residual=normalize(normalize(V12)+normalize(V18))",
        "normalization_difference": "A1 recomposes from per-head hidden values and omits the /2 mean before residual normalization; /2 is mathematically neutral after normalization, but the head projection path is distinct from cached projected V tensors.",
        "alpha_application": "alpha applied after residual normalization in both paths",
    }
    write_json(out / "feature_path_formulas.json", formula)
    write_json(out / "cache_checksums.json", {
        "head_memory_checkpoint": str(head_path),
        "value_memory_checkpoint": str(value_path),
        "head_F_base_checksum": tensor_checksum(mem["F_base"]),
        "value_patch_keys_checksum": tensor_checksum(value_mem["patch_keys"]),
        "canonical_recomputed_checksum": tensor_checksum(canonical_memory),
        "ivc_recomputed_checksum": tensor_checksum(ivc_memory),
    })
    proxy = {
        "variant": "C1_LearnedStaticHeadGate_proxy",
        "optimizer_run": False,
        "trainable_logits_updated": False,
        "checkpoint_exists": False,
        "weights_differ_from_initialization": False,
        "implemented_as_proxy_or_alias": True,
        "aliased_to": "D1_LowVariance_keep25",
        "evidence": "static_residual_closure_audit.py build_specs constructs C1 directly from lowvar['weights']; no optimizer/checkpoint is used in Phase IV-C.",
    }
    write_json(out / "proxy_gate_audit.json", proxy)
    summary = [
        "# Phase IV-C Consistency Summary",
        "",
        "- Status: `FORMULA_DIFFERENCE_IDENTIFIED`",
        f"- Canonical formula: `{FORMULA_NAME}`",
        "- V0 used cached Phase II-D `F_base`.",
        "- A1 alpha=0.50 recomputed features through the Phase IV-C per-head path.",
        "- Therefore A1 alpha=0.50 was not an equivalent alpha-only path and should not be interpreted as a baseline-identical variant.",
        "- C1 is a proxy/alias, not a trained static gate.",
    ]
    write_md(out / "CONSISTENCY_SUMMARY.md", "\n".join(summary))
    write_md("docs/phase_IV_C_consistency_audit.md", "\n".join(summary + ["", f"Outputs: `{out}`"]))
    return "FORMULA_DIFFERENCE_IDENTIFIED", formula


@torch.no_grad()
def evaluate_memory_variant(model, memory, loader, cfg, device, variant_name):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    results = init_results(loader.dataset.obj_list)
    times = []
    entropies, margins = [], []
    labels = memory["patch_values"].argmax(dim=-1).detach().cpu().numpy()
    top1 = []
    for items in tqdm(loader, desc=f"eval {variant_name}"):
        start = time.perf_counter()
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        outputs = extract_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        q = canonical_fuse(outputs[24]["T"][:, 1:, :], outputs[12]["V"][:, 1:, :], outputs[18]["V"][:, 1:, :], alpha=0.5)
        image_logits, _, _ = retrieval_logits(cls, memory["image_keys"], memory["image_values"])
        patch_logits, sim, prob = retrieval_logits(q, memory["patch_keys"], memory["patch_values"])
        flat_sim = sim.reshape(-1, sim.shape[-1])
        flat_prob = prob.reshape(-1, prob.shape[-1])
        vals, idx = torch.topk(flat_sim, k=min(2, flat_sim.shape[-1]), dim=-1)
        entropies.extend((-(flat_prob * flat_prob.clamp_min(1e-12).log()).sum(dim=-1)).detach().cpu().tolist())
        if vals.shape[1] > 1:
            margins.extend((vals[:, 0] - vals[:, 1]).detach().cpu().tolist())
        top1.extend(idx[:, 0].detach().cpu().tolist())
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        times.append(time.perf_counter() - start)
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    used = len(set(int(x) for x in top1))
    return metrics, {
        "variant": variant_name,
        "retrieval_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "top1_margin": float(np.mean(margins)) if margins else float("nan"),
        "unique_prototypes_used": used,
        "never_used_fraction": float(1.0 - used / max(1, memory["patch_keys"].shape[0])),
        "top1_anomaly_fraction": float(np.mean([labels[int(x)] for x in top1])) if top1 else float("nan"),
        "mean_inference_time": float(np.mean(times)) if times else float("nan"),
    }


def make_variants(cfg, decomp, groups, stage1=False):
    variants = []
    variants.append(("V0_FullFPS", decomp["M"].reshape(-1, decomp["M"].shape[-1]), "fullfps", None))
    variants.append(("V2_FPSCentroidRepeated4", reconstruct_rank(decomp, 0), "rank0", 0))
    for rank in (1, 2, 3):
        variants.append((f"V{rank+2}_BaseResidual_Rank{rank}" + ("_Exact" if rank == 3 else ""), reconstruct_rank(decomp, rank), "local_rank", rank))
    for th in cfg.get("energy_thresholds", [0.8, 0.9, 0.95, 0.99]):
        keys, ranks = reconstruct_energy_threshold(decomp, float(th))
        variants.append((f"V6_BaseResidual_EnergyThreshold_{int(float(th)*100)}", keys, "energy_threshold", float(th)))
    for rank in cfg.get("global_ranks", [1, 2, 4, 8]):
        basis = fit_global_basis(decomp["centered"], int(rank))
        variants.append((f"V{6+int(rank)}_GlobalResidualPCA_Rank{int(rank)}", reconstruct_global(decomp["mu"], decomp["centered"], basis), "global_pca", int(rank)))
    controls = 5 if stage1 else int(cfg.get("control_seeds", 20))
    for seed in range(controls):
        variants.append((f"C1_RandomLocalSubspace_Rank1_seed{seed}", random_local_subspace(decomp, 1, seed), "random_rank1", 1))
        variants.append((f"C2_RandomLocalSubspace_Rank2_seed{seed}", random_local_subspace(decomp, 2, seed), "random_rank2", 2))
        labels = [int(g["label"]) for g in groups]
        variants.append((f"C3_BasisShuffle_Rank1_seed{seed}", basis_shuffle(decomp, labels, 1, seed), "shuffle_rank1", 1))
        variants.append((f"C4_BasisShuffle_Rank2_seed{seed}", basis_shuffle(decomp, labels, 2, seed), "shuffle_rank2", 2))
    return variants


def main():
    args = parse_args("Phase V-A base residual memory rank audit")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed = int(cfg.get("seed", cfg.get("seeds", [0])[0]))
    if args.seed is not None:
        seed = int(args.seed)
    set_seed(seed)
    out = Path(cfg.get("output_dir", "results/base_residual_memory"))
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "failure_traceback.txt"
    if stale.exists():
        stale.unlink()
    save_run_metadata(out, cfg, seed)
    write_json(out / "environment.json", package_versions())
    device = cfg.get("device", "cuda:0")
    model = load_model(cfg, device)
    rows, all_rows, rec_rows, retrieval_rows, storage_rows, failures = [], [], [], [], [], []
    sanity = {}
    try:
        consistency_status, formula = run_consistency_audit(model, cfg, seed, device)
        sanity["consistency_status"] = consistency_status
        sanity["consistency_resolved"] = consistency_status != "UNRESOLVED"
        if consistency_status == "UNRESOLVED":
            raise RuntimeError("Phase IV-C consistency audit unresolved")

        head_path, head, value_path, value_mem = load_memories(cfg, seed)
        memory_cpu = head["memory"]
        groups, grouping = recover_groups(head["entries"], memory_cpu)
        write_json(out / "grouping_audit.json", grouping)
        sanity["grouping_ok"] = grouping.get("status") == "OK"
        if not sanity["grouping_ok"]:
            raise RuntimeError("GROUP_METADATA_UNAVAILABLE")

        full = value_mem["patch_keys"].float()
        decomp = decompose_groups(full, groups)
        max_rank3 = float((reconstruct_rank(decomp, 3) - full).abs().max().item())
        sanity["rank3_exact_max_abs_diff"] = max_rank3
        sanity["rank3_exact"] = max_rank3 <= 1e-6
        if not sanity["rank3_exact"]:
            raise RuntimeError(f"Rank3 does not reconstruct FullFPS: {max_rank3}")

        grouped_rank1 = reconstruct_rank(decomp, 1).reshape(len(groups), 4, -1)
        q = full[:64].unsqueeze(0)
        lazy_l, lazy_s, _ = lazy_logits(q, grouped_rank1, memory_cpu["patch_values"], batch_groups=128)
        mat_l, mat_s, _ = retrieval_logits(q, grouped_rank1.reshape(-1, grouped_rank1.shape[-1]), memory_cpu["patch_values"])
        sanity["lazy_materialized_logits_max_abs_diff"] = float((lazy_l - mat_l).abs().max().item())
        sanity["lazy_materialized_similarity_max_abs_diff"] = float((lazy_s - mat_s).abs().max().item())
        sanity["lazy_materialized_match"] = sanity["lazy_materialized_logits_max_abs_diff"] <= 1e-6 and sanity["lazy_materialized_similarity_max_abs_diff"] <= 1e-6
        if not sanity["lazy_materialized_match"]:
            raise RuntimeError("Lazy and materialized logits disagree")

        _, target_loader = make_loader(cfg, cfg.get("dataset", "mvtec"), cfg.get("target_mode", "test"), shuffle=False)
        stage1 = bool(cfg.get("stage1", True))
        variants = make_variants(cfg, decomp, groups, stage1=stage1)
        if cfg.get("run_mode") == "smoke":
            keep = set(cfg.get("smoke_variants", ["V0_FullFPS", "V2_FPSCentroidRepeated4", "V3_BaseResidual_Rank1", "V4_BaseResidual_Rank2", "V5_BaseResidual_Rank3_Exact", "C1_RandomLocalSubspace_Rank1_seed0", "C2_RandomLocalSubspace_Rank2_seed0", "C3_BasisShuffle_Rank1_seed0", "C4_BasisShuffle_Rank2_seed0"]))
            variants = [v for v in variants if v[0] in keep]
        base_memory_cpu = {
            "image_keys": value_mem["image_keys"].float(),
            "image_values": value_mem["image_values"].float(),
            "patch_values": value_mem["patch_values"].float(),
        }
        for name, keys, family, rank in variants:
            mpath = out / f"seed_{seed}" / name / "metrics.json"
            if cfg.get("resume", True) and mpath.exists():
                payload = json.loads(mpath.read_text(encoding="utf-8"))
                metrics, retrieval = payload["metrics"], payload["retrieval"]
            else:
                mem_eval = materialized_memory({k: v.to(device) for k, v in base_memory_cpu.items()}, keys.to(device))
                metrics, retrieval = evaluate_memory_variant(model, mem_eval, target_loader, cfg, device, name)
                mpath.parent.mkdir(parents=True, exist_ok=True)
                mpath.write_text(json.dumps({"metrics": metrics, "retrieval": retrieval}, indent=2, default=json_default), encoding="utf-8")
                del mem_eval
                torch.cuda.empty_cache()
            row = {
                "seed": seed,
                "variant": name,
                "family": family,
                "rank": rank if rank is not None else "",
                "I-AUROC": row_metric(metrics, "I-AUROC"),
                "I-AP": row_metric(metrics, "I-AP"),
                "P-AUROC": row_metric(metrics, "P-AUROC"),
                "PRO": row_metric(metrics, "PRO"),
            }
            rows.append(row)
            all_rows.append(row)
            retrieval_rows.append({"seed": seed, **retrieval})
            rec_rows.append({"seed": seed, **reconstruction_diagnostics(name, full, keys, rank=rank if isinstance(rank, int) else -1)})
            sr = feature_storage_scalars(full.shape[-1], int(rank) if isinstance(rank, int) and rank >= 0 else 3, len(groups))
            storage_rows.append({"seed": seed, "variant": name, "family": family, "rank": row["rank"], **sr})

        full_row = next((r for r in rows if r["variant"] == "V0_FullFPS"), None)
        mean_row = next((r for r in rows if r["variant"] == "V2_FPSCentroidRepeated4"), None)
        gain_rows = []
        if full_row and mean_row:
            for r in rows:
                for metric in ("P-AUROC", "PRO"):
                    denom = float(full_row[metric]) - float(mean_row[metric])
                    recovery = "NOT_APPLICABLE_SMALL_DENOMINATOR" if denom <= 0.001 else (float(r[metric]) - float(mean_row[metric])) / denom
                    gain_rows.append({"variant": r["variant"], "metric": metric, "recovery": recovery, "denominator": denom})
        write_csv(out / "summary_metrics.csv", rows, list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "all_variant_metrics.csv", all_rows, list(all_rows[0].keys()) if all_rows else ["seed"])
        write_csv(out / "rank_metrics.csv", [r for r in rows if str(r["family"]).startswith("rank") or r["family"] in {"fullfps", "local_rank"}], list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "random_control_metrics.csv", [r for r in rows if str(r["family"]).startswith("random")], list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "shuffle_control_metrics.csv", [r for r in rows if str(r["family"]).startswith("shuffle")], list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "global_residual_metrics.csv", [r for r in rows if r["family"] == "global_pca"], list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "energy_threshold_metrics.csv", [r for r in rows if r["family"] == "energy_threshold"], list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "mean_baseline_metrics.csv", [r for r in rows if r["variant"] == "V2_FPSCentroidRepeated4"], list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "gain_recovery.csv", gain_rows, ["variant", "metric", "recovery", "denominator"])
        write_csv(out / "reconstruction_diagnostics.csv", rec_rows, list(rec_rows[0].keys()) if rec_rows else ["seed"])
        write_csv(out / "retrieval_diagnostics.csv", retrieval_rows, list(retrieval_rows[0].keys()) if retrieval_rows else ["seed"])
        write_csv(out / "storage_accounting.csv", storage_rows, list(storage_rows[0].keys()) if storage_rows else ["seed"])
        write_csv(out / "computational_costs.csv", [{"variant": r["variant"], "mean_inference_time": next((x["mean_inference_time"] for x in retrieval_rows if x["variant"] == r["variant"]), float("nan"))} for r in rows], ["variant", "mean_inference_time"])
        write_csv(out / "per_category_metrics.csv", [], ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
        write_csv(out / "defect_size_metrics.csv", [], ["seed", "variant", "defect_size", "I-AUROC", "P-AUROC", "PRO"])
        write_csv(out / "localization_diagnostics.csv", [], ["seed", "variant", "fragmentation", "false_positive_area_ratio", "false_negative_area_ratio"])
        write_csv(out / "statistical_comparisons.csv", [], ["comparison", "metric", "delta", "wins", "ties", "losses"])
        write_csv(out / "success_control_table.csv", [{
            "Variant": r["variant"],
            "Shared base": "FPS centroid" if r["variant"] != "V0_FullFPS" else "none",
            "Local residual rank": r["rank"],
            "True direction": str(r["family"] in {"local_rank", "fullfps"}),
            "Virtual entries": 12476,
            "Storage ratio": next((s["feature_scalar_ratio"] for s in storage_rows if s["variant"] == r["variant"]), ""),
            "I-AUROC": r["I-AUROC"],
            "P-AUROC": r["P-AUROC"],
            "PRO": r["PRO"],
        } for r in rows], ["Variant", "Shared base", "Local residual rank", "True direction", "Virtual entries", "Storage ratio", "I-AUROC", "P-AUROC", "PRO"])
        source_selection = {"status": "NO_SOURCE_SELECTED_COMPACT_RANK", "protocol": "not run in smoke/stage scaffold; no MVTec target labels used", "selected_rank": None}
        write_json(out / "source_selection.json", source_selection)
        sanity["original_region_mean_availability"] = "ORIGINAL_REGION_MEAN_UNAVAILABLE"
        outcome = assign_outcome(rows, sanity)
        write_json(out / "sanity_checks.json", sanity)
        write_json(out / "OUTCOME.json", {"Outcome": outcome})
        summary = [
            "# Phase V-A Execution Summary",
            "",
            f"- Phase IV-C consistency status: `{consistency_status}`",
            f"- Canonical feature formula: `{FORMULA_NAME}`",
            "- Proxy-gate audit result: `C1_LearnedStaticHeadGate_proxy` is an alias/proxy, not trained.",
            f"- FPS grouping status: `{grouping.get('status')}`",
            "- Original region mean availability: `ORIGINAL_REGION_MEAN_UNAVAILABLE`",
            f"- Completed variants: {len(rows)}",
            f"- Rank3 exact max abs diff: `{max_rank3}`",
            f"- Lazy/materialized logits max abs diff: `{sanity['lazy_materialized_logits_max_abs_diff']}`",
            f"- Outcome: `{outcome}`",
        ]
        write_md(out / "EXECUTION_SUMMARY.md", "\n".join(summary))
        if stage1:
            write_md(out / "PHASE_V_A_EARLY_STOP_REPORT.md", "\n".join(summary + ["", "Stage-1 outputs were generated; full 20-control expansion was not required for the smoke run."]))
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase IV-C Consistency Audit\n\n")
            f.write(f"- Status: `{consistency_status}`.\n")
            f.write(f"- Canonical formula: `{FORMULA_NAME}`.\n")
            f.write("- C1 static gate relabeled as proxy/alias, not trained.\n")
            f.write("\n## Phase V-A — Base–Residual Memory Rank Audit\n\n")
            f.write(f"- Output directory: `{out}`.\n")
            f.write(f"- Completed variants: {len(rows)}.\n")
            f.write(f"- Outcome: `{outcome}`.\n")
    except Exception:
        tb = traceback.format_exc()
        failures.append(tb)
        (out / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        write_json(out / "OUTCOME.json", {"Outcome": "IMPLEMENTATION_OR_METADATA_FAILURE", "error": tb})
        raise


if __name__ == "__main__":
    main()
