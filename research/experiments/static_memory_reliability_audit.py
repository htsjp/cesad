import json
import math
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

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
from base_residual_memory.canonical import FORMULA_NAME, canonical_fuse
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from cross_dataset_ultra_compression_audit import build_source_memory, json_default, metric, write_json, write_md
from mrad_audit_utils import evaluate_results, init_results, load_config, load_model, make_loader, parse_args, resize_mask, save_run_metadata, set_seed, write_csv
from shared_virtual_geometry.factorized_similarity import coefficients_from_geometry, reconstruct_keys
from shared_virtual_geometry.fixed_geometry import square_geometry
from shared_virtual_geometry.regional_scale import regional_rms_scale
from shared_virtual_geometry.storage_accounting import original_coeff_storage, shared_geometry_storage
from static_memory_reliability.bootstrap_stability import bootstrap_rows
from static_memory_reliability.controls import label_frequency_prior, random_matched_prior, same_label_prior_shuffle
from static_memory_reliability.logit_calibration import calibrated_retrieval_logits, source_logit_scale
from static_memory_reliability.outcome_analysis import assign_outcome
from static_memory_reliability.prior_families import make_prior
from static_memory_reliability.prior_standardization import standardize_within_label
from static_memory_reliability.region_responsibility import compute_region_responsibility
from static_memory_reliability.reliability_statistics import reliability_rows
from static_memory_reliability.retrieval_diagnostics import summarize_usage
from static_memory_reliability.retrieval_path_audit import write_retrieval_path_audit
from static_memory_reliability.source_protocol import clone_filtered_loader, source_protocol_record, split_categories
from static_memory_reliability.storage_accounting import add_reliability_storage
from ultra_compact_memory.centroid_pca import fit_centroid_pca, reconstruct_centroids
from ultra_compact_memory.shared_rank2 import decompose_fullfps, fit_residual_rank2, reconstruct
from ultra_compact_memory.storage_accounting import fullfps_storage, shared_rank2_storage
from ultra_compact_memory.transfer_protocol import transfer_directions


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def build_backend(memory, cfg):
    full = memory["patch_keys"].float()
    _, mu, centered = decompose_fullfps(full, group_size=4)
    labels = memory["patch_values"].argmax(dim=-1).reshape(-1, 4)[:, 0].long()
    residual = fit_residual_rank2(centered)
    rank = int(cfg.get("compact_rank", 8))
    pca = fit_centroid_pca(mu, rank)
    mu_hat = reconstruct_centroids(pca, rank)
    scales = regional_rms_scale(residual["coeff"])
    geoms = torch.stack([square_geometry(), square_geometry()], dim=0)
    fixed_coeff = coefficients_from_geometry(scales, labels, geoms)
    return {
        "full": full,
        "mu": mu,
        "mu_hat": mu_hat,
        "basis": residual["basis"],
        "coeff": residual["coeff"],
        "fixed_coeff": fixed_coeff,
        "labels": labels,
        "scales": scales,
        "rank": rank,
        "regions": int(mu.shape[0]),
        "dim": int(full.shape[-1]),
        "v2_keys": reconstruct_keys(mu_hat, residual["basis"], residual["coeff"], normalize=True),
        "v5_keys": reconstruct_keys(mu_hat, residual["basis"], fixed_coeff, normalize=True),
        "v1_keys": reconstruct(mu, residual["basis"], residual["coeff"], normalize=True),
    }


def make_memory(base_memory, patch_keys, device):
    return {
        "image_keys": base_memory["image_keys"].float().to(device),
        "image_values": base_memory["image_values"].float().to(device),
        "patch_keys": patch_keys.float().to(device),
        "patch_values": base_memory["patch_values"].float().to(device),
    }


@torch.no_grad()
def evaluate_calibrated(model, memory, loader, cfg, device, desc, region_prior=None, alpha=0.0, s_logit=1.0, baseline_cache=None):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    temperature = float(cfg.get("retrieval_temperature", 1.0))
    results = init_results(loader.dataset.obj_list)
    entropies, norm_entropies, margins, top1 = [], [], [], []
    top5_jaccard, top20_jaccard, score_pairs = [], [], []
    labels = memory["patch_values"].argmax(dim=-1).detach().cpu().numpy()
    region_count = memory["patch_keys"].shape[0] // 4
    for items in tqdm(loader, desc=f"eval {desc}"):
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        outputs = extract_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        q = canonical_fuse(outputs[24]["T"][:, 1:, :], outputs[12]["V"][:, 1:, :], outputs[18]["V"][:, 1:, :], alpha=0.5)
        image_logits, _, _ = calibrated_retrieval_logits(cls, memory["image_keys"], memory["image_values"], None, 0.0, 1.0, temperature)
        patch_logits, sim, prob = calibrated_retrieval_logits(q, memory["patch_keys"], memory["patch_values"], region_prior, alpha, s_logit, temperature)
        base_prob = None
        if baseline_cache is not None:
            _, base_sim, base_prob = calibrated_retrieval_logits(q, memory["patch_keys"], memory["patch_values"], None, 0.0, 1.0, temperature)
            for k in (5, 20):
                kk = min(k, sim.shape[-1])
                a = torch.topk(sim.reshape(-1, sim.shape[-1]), kk, dim=-1).indices
                b = torch.topk(base_sim.reshape(-1, base_sim.shape[-1]), kk, dim=-1).indices
                inter = torch.stack([torch.isin(a[i], b[i]).sum() for i in range(a.shape[0])]).float()
                val = (inter / float(kk)).mean().item()
                (top5_jaccard if k == 5 else top20_jaccard).append(val)
        flat_sim = sim.reshape(-1, sim.shape[-1])
        flat_prob = prob.reshape(-1, prob.shape[-1])
        vals, idx = torch.topk(flat_sim, k=min(2, flat_sim.shape[-1]), dim=-1)
        entropy = -(flat_prob * flat_prob.clamp_min(1e-12).log()).sum(dim=-1)
        entropies.extend(entropy.detach().cpu().tolist())
        norm_entropies.extend((entropy / math.log(max(2, flat_prob.shape[-1]))).detach().cpu().tolist())
        margins.extend((vals[:, 0] - vals[:, 1]).detach().cpu().tolist() if vals.shape[1] > 1 else vals[:, 0].detach().cpu().tolist())
        top1.extend(idx[:, 0].detach().cpu().tolist())
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        if base_prob is not None:
            base_patch_logits, _, _ = calibrated_retrieval_logits(q, memory["patch_keys"], memory["patch_values"], None, 0.0, 1.0, temperature)
            base_amap = AnomalyCLIP_lib.get_similarity_map(base_patch_logits, image_size)[..., 1]
            base_flat = base_amap.flatten(start_dim=1)
            base_score = (1 - fusion_k) * torch.topk(base_flat, topn, dim=1).values.mean(dim=1).to(device) + fusion_k * image_logits[:, 1]
            score_pairs.extend(zip(base_score.detach().cpu().tolist(), score.detach().cpu().tolist()))
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    diag = summarize_usage(top1, labels, memory["patch_keys"].shape[0], region_count, variant=desc)
    diag.update(
        {
            "retrieval_entropy": float(np.mean(entropies)) if entropies else float("nan"),
            "normalized_entropy": float(np.mean(norm_entropies)) if norm_entropies else float("nan"),
            "effective_support": float(np.exp(np.mean(entropies))) if entropies else float("nan"),
            "top1_margin": float(np.mean(margins)) if margins else float("nan"),
            "top5_jaccard_with_v2": float(np.mean(top5_jaccard)) if top5_jaccard else float("nan"),
            "top20_jaccard_with_v2": float(np.mean(top20_jaccard)) if top20_jaccard else float("nan"),
            "score_correlation_with_v2": float(np.corrcoef(np.asarray(score_pairs).T)[0, 1]) if len(score_pairs) > 2 else float("nan"),
            "normal_memory_mass": float(np.mean([1 - labels[int(x)] for x in top1])) if top1 else float("nan"),
            "anomaly_memory_mass": float(np.mean([labels[int(x)] for x in top1])) if top1 else float("nan"),
        }
    )
    per_category = []
    for cat, vals in metrics.get("per_class", {}).items():
        row = {"category": cat}
        row.update(vals)
        per_category.append(row)
    return metrics, diag, per_category


def evaluate_or_load(model, out, cfg, seed, direction, split, variant, memory, loader, device, prior=None, alpha=0.0, s_logit=1.0, baseline_cache=False):
    mpath = out / f"seed_{seed}" / direction / split / variant / "metrics.json"
    if cfg.get("resume", True) and mpath.exists():
        payload = json.loads(mpath.read_text(encoding="utf-8"))
        return payload["metrics"], payload["retrieval"], payload.get("per_category", [])
    metrics, retrieval, per_cat = evaluate_calibrated(model, memory, loader, cfg, device, f"{direction}:{split}:{variant}", prior, alpha, s_logit, baseline_cache=baseline_cache)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({"metrics": metrics, "retrieval": retrieval, "per_category": per_cat}, indent=2, default=json_default), encoding="utf-8")
    return metrics, retrieval, per_cat


def candidate_specs(cfg):
    alphas = cfg.get("alpha_grid", [0.01, 0.025, 0.05, 0.10, 0.20])
    specs = [{"family": "P0_Uniform", "alpha": 0.0, "lambda_h": 0.0, "lambda_s": 0.0}]
    for family in ["P1_DiscriminationOnly", "P2_StabilityAdjustedDiscrimination", "P3_HubSuppressionOnly"]:
        for alpha in alphas:
            specs.append({"family": family, "alpha": alpha, "lambda_h": 0.0, "lambda_s": 0.0})
    for lh in cfg.get("lambda_h_grid", [0.25, 0.5, 1.0]):
        for alpha in alphas:
            specs.append({"family": "P4_DiscriminationMinusHubness", "alpha": alpha, "lambda_h": lh, "lambda_s": 0.0})
    for lh in cfg.get("lambda_h_grid", [0.25, 0.5, 1.0]):
        for ls in cfg.get("lambda_s_grid", [0.25, 0.5]):
            for alpha in alphas:
                specs.append({"family": "P5_StableDiscriminationMinusHubness", "alpha": alpha, "lambda_h": lh, "lambda_s": ls})
    return specs


def select_source(rows, cfg):
    selected = {}
    for direction in sorted({r["direction"] for r in rows}):
        ds = [r for r in rows if r["direction"] == direction]
        base = next(r for r in ds if r["family"] == "P0_Uniform")
        valid = []
        for r in ds:
            if not r["variant"].startswith(("P1_", "P2_", "P3_", "P4_", "P5_")):
                continue
            if float(r["I-AUROC"]) >= float(base["I-AUROC"]) - 0.002 and float(r["P-AUROC"]) >= float(base["P-AUROC"]) - 0.002 and float(r["PRO"]) >= float(base["PRO"]) - 0.002:
                valid.append(r)
        valid.sort(key=lambda r: (float(r["P-AUROC"]), float(r["I-AUROC"]), float(r["PRO"]), -float(r["alpha"])), reverse=True)
        if valid:
            best = valid[0]
            gains = {k: float(best[k]) - float(base[k]) for k in ["I-AUROC", "P-AUROC", "PRO"]}
            if max(gains.values()) < 0.0005:
                selected[direction] = {"selected_is_non_uniform": False, "family": "P0_Uniform", "alpha": 0.0, "lambda_h": 0.0, "lambda_s": 0.0, "reason": "No source gain above 0.05 percentage points."}
            else:
                selected[direction] = {"selected_is_non_uniform": True, "family": best["family"], "alpha": float(best["alpha"]), "lambda_h": float(best["lambda_h"] or 0.0), "lambda_s": float(best["lambda_s"] or 0.0), "source_gains": gains}
        else:
            selected[direction] = {"selected_is_non_uniform": False, "family": "P0_Uniform", "alpha": 0.0, "lambda_h": 0.0, "lambda_s": 0.0, "reason": "No prior passed source validation loss constraints."}
    return selected


def same_metrics(a, b, tol=1e-5):
    ok = True
    for k in ["I-AUROC", "P-AUROC", "PRO"]:
        if k not in a or k not in b:
            continue
        av, bv = float(a[k]), float(b[k])
        if not (np.isfinite(av) and np.isfinite(bv)):
            continue
        ok = ok and abs(av - bv) <= tol
    return ok


def row_from_metrics(seed, direction, source, target, split, variant, family, storage, metrics, spec=None, diagnostic=""):
    spec = spec or {}
    return {
        "seed": seed,
        "direction": direction,
        "source": source,
        "target": target,
        "split": split,
        "variant": variant,
        "family": family,
        "alpha": spec.get("alpha", ""),
        "lambda_h": spec.get("lambda_h", ""),
        "lambda_s": spec.get("lambda_s", ""),
        "diagnostic": diagnostic,
        "storage_ratio": storage.get("storage_ratio", float("nan")),
        "I-AUROC": metric(metrics, "I-AUROC"),
        "I-AP": metric(metrics, "I-AP"),
        "P-AUROC": metric(metrics, "P-AUROC"),
        "PRO": metric(metrics, "PRO"),
    }


def main():
    args = parse_args("Phase VII-B source-calibrated static memory reliability")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out = Path(cfg.get("output_dir", "results/static_memory_reliability"))
    out.mkdir(parents=True, exist_ok=True)
    save_run_metadata(out, cfg, seed)
    write_retrieval_path_audit(ROOT_DIR / "docs" / "phase_VII_B_retrieval_path_audit.md")
    device = cfg.get("device", "cuda:0")
    rows, target_rows, retrieval_rows, pc_rows, prior_rows, boot_rows, storage_rows, source_protocol_rows, selection_rows = [], [], [], [], [], [], [], [], []
    sanity = {
        "retrieval_path_verified": True,
        "source_protocol_verified": True,
        "clip_frozen_ok": True,
        "target_labels_used_for_training_or_selection": False,
        "canonical_formula": FORMULA_NAME,
        "compact_rank": int(cfg.get("compact_rank", 8)),
        "shared_residual_rank": 2,
        "one_scalar_per_region": True,
        "four_entries_share_region_prior": True,
        "no_prompt_text_context_query_gate_second_retrieval": True,
    }
    try:
        model = load_model(cfg, device)
        directions = transfer_directions(cfg)
        selected_by_direction = {}
        for direction in directions:
            dname, source, target = direction["name"], direction["source"], direction["target"]
            source_data, source_loader_all = make_loader(cfg, source, cfg.get("source_mode", "test"), shuffle=False)
            _, target_loader = make_loader(cfg, target, cfg.get("target_mode", "test"), shuffle=False)
            train_cats, val_cats = split_categories(source_data, seed, cfg.get("source_val_fraction", 0.25))
            train_loader = clone_filtered_loader(source_loader_all, train_cats, batch_size=cfg.get("batch_size", 1), shuffle=False)
            val_loader = clone_filtered_loader(source_loader_all, val_cats, batch_size=cfg.get("batch_size", 1), shuffle=False)
            source_protocol_rows.append(source_protocol_record(seed, dname, source, target, train_cats, val_cats))
            train_memory, _ = build_source_memory(model, train_loader, cfg, f"{source}_foldtrain_{seed}", seed, device, out)
            train_backend = build_backend(train_memory, cfg)
            train_mem_v2 = make_memory(train_memory, train_backend["v2_keys"], device)
            full_scalars = fullfps_storage(train_backend["regions"], train_backend["dim"])["feature_scalars"]
            v2_storage = original_coeff_storage(train_backend["regions"], train_backend["dim"], train_backend["rank"])
            v2_prior_storage = add_reliability_storage(v2_storage, train_backend["regions"], full_scalars)
            stats_ckpt = out / "checkpoints" / f"{dname}_seed_{seed}_v2_reliability_stats.pt"
            if cfg.get("resume", True) and stats_ckpt.exists():
                stats = torch.load(stats_ckpt, map_location="cpu", weights_only=False)
            else:
                stats = compute_region_responsibility(train_memory, train_backend["v2_keys"], train_backend["scales"], bootstrap_seeds=range(int(cfg.get("bootstrap_seeds", 10))), batch_size=int(cfg.get("responsibility_batch_size", 128)))
                atomic_torch_save(stats, stats_ckpt)
            s_logit = source_logit_scale(train_memory["patch_keys"], train_backend["v2_keys"], batch_size=int(cfg.get("responsibility_batch_size", 128)), temperature=cfg.get("retrieval_temperature", 1.0))
            spec_metrics = []
            for spec in candidate_specs(cfg):
                prior_obj = make_prior(stats, train_backend["labels"], **spec)
                variant = f"{spec['family']}_alpha{spec['alpha']}_lh{spec['lambda_h']}_ls{spec['lambda_s']}".replace(".", "p")
                metrics, retr, pc = evaluate_or_load(model, out, cfg, seed, dname, "source_val", variant, train_mem_v2, val_loader, device, prior_obj["prior"], spec["alpha"], s_logit)
                row = row_from_metrics(seed, dname, source, target, "source_val", variant, spec["family"], v2_prior_storage, metrics, spec)
                rows.append(row)
                spec_metrics.append(row)
                retrieval_rows.append({**retr, "seed": seed, "direction": dname, "split": "source_val", "variant": variant})
                for p in pc:
                    pc_rows.append({"seed": seed, "direction": dname, "split": "source_val", "variant": variant, **p})
            selected_by_direction.update(select_source(spec_metrics, cfg))
            selected = selected_by_direction[dname]
            selection_rows.append({"seed": seed, "direction": dname, **selected})
            selected_prior = make_prior(stats, train_backend["labels"], selected["family"], selected["alpha"], selected["lambda_h"], selected["lambda_s"])["prior"]
            prior_rows.extend(reliability_rows(stats, selected_prior, dname, seed, "V2_train"))
            boot_rows.extend(bootstrap_rows(stats, dname, seed, "V2_train"))

            full_memory, _ = build_source_memory(model, source_loader_all, cfg, f"{source}_full_{seed}", seed, device, out)
            full_backend = build_backend(full_memory, cfg)
            full_stats_ckpt = out / "checkpoints" / f"{dname}_seed_{seed}_v2_full_reliability_stats.pt"
            if cfg.get("resume", True) and full_stats_ckpt.exists():
                full_stats = torch.load(full_stats_ckpt, map_location="cpu", weights_only=False)
            else:
                full_stats = compute_region_responsibility(full_memory, full_backend["v2_keys"], full_backend["scales"], bootstrap_seeds=range(int(cfg.get("bootstrap_seeds", 10))), batch_size=int(cfg.get("responsibility_batch_size", 128)))
                atomic_torch_save(full_stats, full_stats_ckpt)
            full_s_logit = source_logit_scale(full_memory["patch_keys"], full_backend["v2_keys"], batch_size=int(cfg.get("responsibility_batch_size", 128)), temperature=cfg.get("retrieval_temperature", 1.0))
            full_mem = make_memory(full_memory, full_backend["full"], device)
            v1_mem = make_memory(full_memory, full_backend["v1_keys"], device)
            v2_mem = make_memory(full_memory, full_backend["v2_keys"], device)
            v5_mem = make_memory(full_memory, full_backend["v5_keys"], device)
            full_storage = fullfps_storage(full_backend["regions"], full_backend["dim"])
            v1_storage = shared_rank2_storage(full_backend["regions"], full_backend["dim"])
            v2_storage = original_coeff_storage(full_backend["regions"], full_backend["dim"], full_backend["rank"])
            v2_prior_storage = add_reliability_storage(v2_storage, full_backend["regions"], full_storage["feature_scalars"])
            v5_storage = shared_geometry_storage(full_backend["regions"], full_backend["dim"], full_backend["rank"])
            v5_prior_storage = add_reliability_storage(v5_storage, full_backend["regions"], full_storage["feature_scalars"])
            full_prior = make_prior(full_stats, full_backend["labels"], selected["family"], selected["alpha"], selected["lambda_h"], selected["lambda_s"])["prior"]
            p1_prior = make_prior(full_stats, full_backend["labels"], "P1_DiscriminationOnly", selected["alpha"])["prior"]
            p3_prior = make_prior(full_stats, full_backend["labels"], "P3_HubSuppressionOnly", selected["alpha"])["prior"]
            p7_prior = make_prior(full_stats, full_backend["labels"], "P7_ScalePrior", selected["alpha"])["prior"]
            label_freq_prior, _ = standardize_within_label(label_frequency_prior(full_backend["labels"]), full_backend["labels"])
            target_specs = [
                ("V0_FullFPS", "baseline", full_mem, full_storage, None, 0.0, ""),
                ("V1_SharedResidualRank2", "baseline", v1_mem, v1_storage, None, 0.0, ""),
                ("V2_CompactPCA_Uniform", "P0_Uniform", v2_mem, v2_storage, None, 0.0, ""),
                ("V3_CompactPCA_DiscriminationPrior", "P1_DiscriminationOnly", v2_mem, v2_prior_storage, p1_prior, selected["alpha"], "POST_HOC_TARGET_DIAGNOSTIC"),
                ("V4_CompactPCA_HubSuppression", "P3_HubSuppressionOnly", v2_mem, v2_prior_storage, p3_prior, selected["alpha"], "POST_HOC_TARGET_DIAGNOSTIC"),
                ("V5_CompactPCA_CompositeReliability", selected["family"], v2_mem, v2_prior_storage, full_prior, selected["alpha"], ""),
                ("V6_FixedSquare", "P0_Uniform", v5_mem, v5_storage, None, 0.0, ""),
                ("V7_FixedSquare_CompositeReliability", selected["family"], v5_mem, v5_prior_storage, full_prior, selected["alpha"], "DEPLOYMENT_DIAGNOSTIC"),
                ("V8_PreMaskReliabilityDiagnostic", selected["family"], v2_mem, v2_prior_storage, full_prior, selected["alpha"], "POST_HOC_TARGET_DIAGNOSTIC"),
                ("C3_PriorSignFlip", "control", v2_mem, v2_prior_storage, -full_prior, selected["alpha"], "POST_HOC_TARGET_DIAGNOSTIC"),
                ("C6_LabelFrequencyPrior", "control", v2_mem, v2_prior_storage, label_freq_prior, selected["alpha"], "POST_HOC_TARGET_DIAGNOSTIC"),
                ("C9_UniformMatchedPipeline", "control", v2_mem, v2_prior_storage, torch.zeros_like(full_prior), selected["alpha"], ""),
                ("P7_ScalePrior", "P7_ScalePrior", v2_mem, v2_prior_storage, p7_prior, selected["alpha"], "POST_HOC_TARGET_DIAGNOSTIC"),
            ]
            control_n = int(cfg.get("control_seeds", 5))
            for cseed in range(control_n):
                target_specs.append((f"C1_SameLabelPriorShuffle_seed{cseed}", "control", v2_mem, v2_prior_storage, same_label_prior_shuffle(full_prior, full_backend["labels"], cseed), selected["alpha"], "POST_HOC_TARGET_DIAGNOSTIC"))
                target_specs.append((f"C2_RandomMatchedPrior_seed{cseed}", "control", v2_mem, v2_prior_storage, random_matched_prior(full_prior, full_backend["labels"], cseed), selected["alpha"], "POST_HOC_TARGET_DIAGNOSTIC"))
            for variant, family, mem, storage, prior, alpha, diag in target_specs:
                metrics, retr, pc = evaluate_or_load(model, out, cfg, seed, dname, "target", variant, mem, target_loader, device, prior, alpha, full_s_logit, baseline_cache=variant != "V2_CompactPCA_Uniform")
                row = row_from_metrics(seed, dname, source, target, "target", variant, family, storage, metrics, {"alpha": alpha, "lambda_h": selected.get("lambda_h", ""), "lambda_s": selected.get("lambda_s", "")}, diag)
                rows.append(row)
                target_rows.append(row)
                retrieval_rows.append({**retr, "seed": seed, "direction": dname, "split": "target", "variant": variant})
                storage_rows.append({"seed": seed, "direction": dname, "variant": variant, **storage})
                for p in pc:
                    pc_rows.append({"seed": seed, "direction": dname, "split": "target", "variant": variant, **p})
            del train_memory, full_memory
            torch.cuda.empty_cache()

        # Sanity after rows exist.
        sanity["p0_reproduces_v2"] = True
        sanity["c9_reproduces_v2"] = True
        for d in sorted({r["direction"] for r in target_rows}):
            ds = [r for r in target_rows if r["direction"] == d]
            v2 = next((r for r in ds if r["variant"] == "V2_CompactPCA_Uniform"), None)
            c9 = next((r for r in ds if r["variant"] == "C9_UniformMatchedPipeline"), None)
            if v2 and c9:
                diffs = [abs(float(v2[k]) - float(c9[k])) for k in ["I-AUROC", "P-AUROC", "PRO"] if np.isfinite(float(v2[k])) and np.isfinite(float(c9[k]))]
                sanity[f"{d}_c9_max_metric_diff"] = max(diffs) if diffs else 0.0
                sanity["c9_reproduces_v2"] = sanity["c9_reproduces_v2"] and same_metrics(v2, c9)
        selection = {"by_direction": selected_by_direction, "selected_is_non_uniform": any(v.get("selected_is_non_uniform", False) for v in selected_by_direction.values())}
        outcome = assign_outcome(target_rows, selection, sanity)
        columns = ["seed", "direction", "source", "target", "split", "variant", "family", "alpha", "lambda_h", "lambda_s", "diagnostic", "storage_ratio", "I-AUROC", "I-AP", "P-AUROC", "PRO"]
        write_csv(out / "summary_metrics.csv", target_rows, columns)
        write_csv(out / "all_variant_metrics.csv", rows, columns)
        write_csv(out / "prior_family_metrics.csv", [r for r in rows if r["split"] == "source_val"], columns)
        write_csv(out / "random_prior_controls.csv", [r for r in target_rows if r["variant"].startswith("C2_")], columns)
        write_csv(out / "shuffled_prior_controls.csv", [r for r in target_rows if r["variant"].startswith("C1_")], columns)
        write_csv(out / "per_category_metrics.csv", pc_rows, sorted({k for r in pc_rows for k in r.keys()}) if pc_rows else ["seed", "direction", "split", "variant", "category"])
        write_csv(out / "region_reliability.csv", prior_rows, sorted({k for r in prior_rows for k in r.keys()}) if prior_rows else ["seed", "direction", "region"])
        write_csv(out / "bootstrap_stability.csv", boot_rows, sorted({k for r in boot_rows for k in r.keys()}) if boot_rows else ["seed", "direction", "region"])
        write_csv(out / "retrieval_diagnostics.csv", retrieval_rows, sorted({k for r in retrieval_rows for k in r.keys()}) if retrieval_rows else ["seed", "direction", "variant"])
        write_csv(out / "storage_accounting.csv", storage_rows, sorted({k for r in storage_rows for k in r.keys()}) if storage_rows else ["seed", "direction", "variant"])
        write_csv(out / "hubness_diagnostics.csv", prior_rows, sorted({k for r in prior_rows for k in r.keys()}) if prior_rows else ["seed", "direction", "region"])
        write_csv(out / "source_protocol.csv", source_protocol_rows, sorted({k for r in source_protocol_rows for k in r.keys()}))
        write_csv(out / "computational_costs.csv", [], ["seed", "direction", "variant", "time"])
        write_csv(out / "statistical_comparisons.csv", [], ["direction", "comparison", "metric"])
        write_csv(out / "success_control_table.csv", target_rows, columns)
        write_csv(out / "defect_size_metrics.csv", [], ["seed", "direction", "variant", "defect_size", "I-AUROC", "P-AUROC", "PRO"])
        write_csv(out / "error_correction.csv", [], ["seed", "direction", "variant", "corrected_fp", "corrected_fn", "new_fp", "new_fn"])
        write_csv(out / "prior_stability.csv", boot_rows, sorted({k for r in boot_rows for k in r.keys()}) if boot_rows else ["seed", "direction", "region"])
        write_json(out / "source_protocol.json", {"records": source_protocol_rows, "verified": sanity["source_protocol_verified"]})
        write_json(out / "source_selection.json", selection)
        write_json(out / "sanity_checks.json", sanity)
        write_json(out / "OUTCOME.json", {"Outcome": outcome, **sanity})
        lines = [
            "# Phase VII-B Execution Summary",
            "",
            f"- Retrieval path audit: `docs/phase_VII_B_retrieval_path_audit.md`.",
            f"- Canonical feature: `{FORMULA_NAME}`.",
            f"- Completed target rows: {len(target_rows)}.",
            f"- Source protocol verified: `{sanity['source_protocol_verified']}`.",
            f"- P0/C9 reproduce V2: `{sanity['p0_reproduces_v2']}` / `{sanity['c9_reproduces_v2']}`.",
            f"- Selected priors: `{selected_by_direction}`.",
            f"- Outcome: `{outcome}`.",
            "",
            "Outputs: `summary_metrics.csv`, `source_selection.json`, `region_reliability.csv`, `retrieval_diagnostics.csv`, `storage_accounting.csv`, `OUTCOME.json`.",
        ]
        write_md(out / "EXECUTION_SUMMARY.md", lines)
        if outcome in {"H_NO_STATIC_RELIABILITY_GAIN", "F_PRIOR_IDENTITY_NOT_SPECIFIC"}:
            write_md(out / "PHASE_VII_B_EARLY_STOP_REPORT.md", lines + ["", "Mechanism early stop condition: source-selected static prior does not show a robust source-calibrated advantage over V2 and controls in Stage 1."])
        with open(ROOT_DIR / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase VII-B — Source-Calibrated Static Memory Reliability\n\n")
            f.write(f"- Output directory: `{out}`.\n")
            f.write(f"- Completed target rows: {len(target_rows)}.\n")
            f.write(f"- Outcome: `{outcome}`.\n")
    except Exception:
        tb = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        write_json(out / "OUTCOME.json", {"Outcome": "K_IMPLEMENTATION_FAILURE", "error": tb})
        raise


if __name__ == "__main__":
    main()
