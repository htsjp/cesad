import csv
import json
import math
import os
import sys
import time
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
from memory_strategies import STRATEGIES
from mrad_audit_utils import (
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
from ultra_compact_memory.centroid_codebook import (
    add_offset,
    fit_kmeans,
    fit_offset,
    random_assignment,
    random_codebook,
    reconstruct_codebook,
    utilization,
)
from ultra_compact_memory.centroid_pca import (
    fit_centroid_pca,
    random_subspace_reconstruct,
    reconstruct_centroids,
    same_label_shuffle_coeff,
)
from ultra_compact_memory.dataset_audit import build_dataset_audit
from ultra_compact_memory.diagnostics import centroid_diagnostics, control_summary, virtual_diagnostics
from ultra_compact_memory.outcome_analysis import assign_outcome
from ultra_compact_memory.shared_rank2 import centroid_repeated, decompose_fullfps, fit_residual_rank2, reconstruct
from ultra_compact_memory.storage_accounting import (
    codebook_storage,
    centroid_pca_storage,
    fullfps_storage,
    serialized_bytes,
    shared_rank2_storage,
    write_json_artifact,
)
from ultra_compact_memory.transfer_protocol import selection_note, transfer_directions


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


def write_md(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def metric(metrics, key):
    return float(metrics.get("mean", {}).get(key, float("nan")))


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def sample_id(items):
    if "img_path" in items:
        value = items["img_path"][0] if isinstance(items["img_path"], (list, tuple)) else items["img_path"]
        return str(value)
    return f"{items['cls_name'][0]}_{int(torch.as_tensor(items['anomaly'])[0])}"


@torch.no_grad()
def build_source_memory(model, loader, cfg, source, seed, device, out):
    ckpt = out / "checkpoints" / f"{source}_seed_{seed}_canonical_fpsk4.pt"
    if cfg.get("resume", True) and ckpt.exists():
        return torch.load(ckpt, map_location="cpu", weights_only=False), 0.0
    image_keys, image_labels, patch_keys, patch_labels, entries = [], [], [], [], []
    k = int(cfg.get("k", 4))
    region_id = memory_id = 0
    start = time.perf_counter()
    for items in tqdm(loader, desc=f"build source memory {source} seed {seed}"):
        image = items["img"].to(device)
        gt = (items["img_mask"].to(device) > 0.5).float()
        outputs = extract_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        patches = canonical_fuse(outputs[24]["T"][:, 1:, :], outputs[12]["V"][:, 1:, :], outputs[18]["V"][:, 1:, :], alpha=0.5)
        image_keys.append(cls.detach().cpu())
        image_labels.append(torch.as_tensor(items["anomaly"]).detach().cpu())
        side = int(math.sqrt(patches.shape[1]))
        gt_resized = resize_mask(gt, side)
        sid = sample_id(items)
        for b in range(image.shape[0]):
            for mask, region_type, label in ((gt_resized[b], "anomaly", 1), (~gt_resized[b], "normal", 0)):
                tokens = patches[b][mask.reshape(-1)]
                if tokens.numel() == 0:
                    continue
                selected = STRATEGIES["FPS_K4"](tokens, k, seed * 1_000_003 + region_id).detach().cpu()
                selected = F.normalize(selected, dim=-1)
                patch_keys.append(selected)
                patch_labels.append(torch.full((selected.shape[0],), label, dtype=torch.long))
                for local_j in range(selected.shape[0]):
                    entries.append(
                        {
                            "memory_id": memory_id,
                            "source_id": sid,
                            "region_id": region_id,
                            "region_type": region_type,
                            "label": label,
                            "local_j": local_j,
                        }
                    )
                    memory_id += 1
                region_id += 1
        del outputs, patches, cls
        torch.cuda.empty_cache()
    memory = {
        "image_keys": torch.cat(image_keys, dim=0),
        "image_values": one_hot(torch.cat(image_labels, dim=0)),
        "patch_keys": torch.cat(patch_keys, dim=0),
        "patch_values": one_hot(torch.cat(patch_labels, dim=0)),
        "entries": entries,
        "source": source,
        "seed": seed,
        "canonical_formula": FORMULA_NAME,
    }
    atomic_torch_save(memory, ckpt)
    return memory, time.perf_counter() - start


@torch.no_grad()
def evaluate_memory_variant(model, memory, loader, cfg, device, desc):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    results = init_results(loader.dataset.obj_list)
    entropies, margins, top1 = [], [], []
    labels = memory["patch_values"].argmax(dim=-1).detach().cpu().numpy()
    times = []
    for items in tqdm(loader, desc=f"eval {desc}"):
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
    retrieval = {
        "retrieval_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "top1_margin": float(np.mean(margins)) if margins else float("nan"),
        "unique_prototypes_used": used,
        "never_used_fraction": float(1.0 - used / max(1, memory["patch_keys"].shape[0])),
        "top1_anomaly_fraction": float(np.mean([labels[int(x)] for x in top1])) if top1 else float("nan"),
        "mean_inference_time": float(np.mean(times)) if times else float("nan"),
    }
    per_category = []
    for cat, vals in metrics.get("per_class", {}).items():
        row = {"category": cat}
        row.update(vals)
        per_category.append(row)
    return metrics, retrieval, per_category


def build_variant_specs(cfg, full, mu, centered, labels):
    regions, dim = mu.shape
    residual = fit_residual_rank2(centered)
    pca_ranks = cfg.get("centroid_pca_ranks", [2, 4, 8, 16, 32, 64, 128])
    codebook_ks = cfg.get("codebook_sizes", [16, 32, 64, 128, 256, 512])
    offset_pairs = [tuple(x) for x in cfg.get("codebook_offset_pairs", [[32, 4], [64, 4], [64, 8], [128, 4], [128, 8], [256, 4]])]
    if cfg.get("run_mode") == "stage1":
        pca_ranks = [r for r in pca_ranks if r in cfg.get("stage1_pca_ranks", [8, 16, 32, 64])]
        codebook_ks = [k for k in codebook_ks if k in cfg.get("stage1_codebook_sizes", [64, 128, 256])]
        offset_pairs = [p for p in offset_pairs if list(p) in cfg.get("stage1_offset_pairs", [[64, 4]]) or p in [tuple(x) for x in cfg.get("stage1_offset_pairs", [[64, 4]])]]
    max_pca = max(pca_ranks + [1])
    pca_fit = fit_centroid_pca(mu, max_pca)
    variants = [
        {"name": "V0_FullFPS", "family": "baseline", "patch_keys": full, "storage": fullfps_storage(regions, dim), "fit": None},
        {
            "name": "V1_CentroidRepeated4",
            "family": "baseline",
            "patch_keys": centroid_repeated(mu),
            "storage": {
                "feature_scalars": int(regions) * int(dim),
                "storage_ratio": 0.25,
                "fp32_bytes": int(regions) * int(dim) * 4,
                "fp16_bytes": int(regions) * int(dim) * 2,
            },
            "fit": None,
        },
        {"name": "V2_SharedResidualRank2_FullCentroid", "family": "shared_rank2", "patch_keys": reconstruct(mu, residual["basis"], residual["coeff"]), "storage": shared_rank2_storage(regions, dim), "fit": None},
    ]
    for rank in pca_ranks:
        mu_hat = reconstruct_centroids(pca_fit, rank)
        keys = reconstruct(mu_hat, residual["basis"], residual["coeff"])
        variants.append({"name": f"A_CentroidPCA_Rank{rank}", "family": "centroid_pca", "rank": rank, "patch_keys": keys, "storage": centroid_pca_storage(regions, dim, rank), "fit": {"mu_hat": mu_hat}})
    for k in codebook_ks:
        cb = fit_kmeans(mu, k, seed=int(cfg.get("kmeans_seed", 0)), tol=cfg.get("kmeans_tol", 1e-4), n_init=cfg.get("kmeans_n_init", 10))
        mu_hat = reconstruct_codebook(cb)
        keys = reconstruct(mu_hat, residual["basis"], residual["coeff"])
        variants.append({"name": f"B_Codebook_K{k}", "family": "codebook", "K": k, "patch_keys": keys, "storage": codebook_storage(regions, dim, k), "fit": {"mu_hat": mu_hat, "codebook": cb}})
        for kk, q in offset_pairs:
            if kk == k:
                off = fit_offset(mu, mu_hat, q)
                mu_off = add_offset(mu_hat, off, q)
                keys = reconstruct(mu_off, residual["basis"], residual["coeff"])
                variants.append({"name": f"B_Codebook_K{k}_OffsetRank{q}", "family": "codebook_offset", "K": k, "rank": q, "patch_keys": keys, "storage": codebook_storage(regions, dim, k, q), "fit": {"mu_hat": mu_off, "codebook": cb, "offset": off}})
    controls = int(cfg.get("stage1_control_seeds", 5) if cfg.get("run_mode") == "stage1" else cfg.get("control_seeds", 20))
    for seed in range(controls):
        for rank in cfg.get("random_pca_control_ranks", [8, 16, 32, 64]):
            if rank > max_pca:
                continue
            mu_rand, _, _ = random_subspace_reconstruct(mu, rank, seed, energy_match=False)
            variants.append({"name": f"C_A1_RandomCentroidSubspace_Rank{rank}_seed{seed}", "family": "random_pca", "rank": rank, "patch_keys": reconstruct(mu_rand, residual["basis"], residual["coeff"]), "storage": centroid_pca_storage(regions, dim, rank), "fit": {"mu_hat": mu_rand}})
            mu_energy, _, _ = random_subspace_reconstruct(mu, rank, seed, energy_match=True)
            variants.append({"name": f"C_A2_EnergyMatchedRandomCentroidSubspace_Rank{rank}_seed{seed}", "family": "energy_random_pca", "rank": rank, "patch_keys": reconstruct(mu_energy, residual["basis"], residual["coeff"]), "storage": centroid_pca_storage(regions, dim, rank), "fit": {"mu_hat": mu_energy}})
        for rank in cfg.get("coeff_shuffle_pca_ranks", [32]):
            if rank <= max_pca:
                shuf = same_label_shuffle_coeff(pca_fit["coeff"][:, :rank], labels, seed)
                mu_shuf = reconstruct_centroids(pca_fit, rank, coeff_override=shuf)
                variants.append({"name": f"C_A3_CentroidCoefficientShuffle_Rank{rank}_seed{seed}", "family": "pca_coeff_shuffle", "rank": rank, "patch_keys": reconstruct(mu_shuf, residual["basis"], residual["coeff"]), "storage": centroid_pca_storage(regions, dim, rank), "fit": {"mu_hat": mu_shuf}})
        for k in cfg.get("random_codebook_control_sizes", [64, 128, 256]):
            cb = fit_kmeans(mu, k, seed=int(cfg.get("kmeans_seed", 0)), tol=cfg.get("kmeans_tol", 1e-4), n_init=cfg.get("kmeans_n_init", 10))
            rand_assign = dict(cb)
            rand_assign["assignments"] = random_assignment(cb["assignments"], seed)
            mu_ra = reconstruct_codebook(rand_assign)
            variants.append({"name": f"C_B1_RandomAssignment_K{k}_seed{seed}", "family": "random_assignment", "K": k, "patch_keys": reconstruct(mu_ra, residual["basis"], residual["coeff"]), "storage": codebook_storage(regions, dim, k), "fit": {"mu_hat": mu_ra, "codebook": rand_assign}})
            rcb = random_codebook(mu, k, seed)
            mu_rc = reconstruct_codebook(rcb)
            variants.append({"name": f"C_B2_RandomCodebook_K{k}_seed{seed}", "family": "random_codebook", "K": k, "patch_keys": reconstruct(mu_rc, residual["basis"], residual["coeff"]), "storage": codebook_storage(regions, dim, k), "fit": {"mu_hat": mu_rc, "codebook": rcb}})
    variants.append({"name": "C_A5_GlobalMeanOnly", "family": "global_mean_only", "patch_keys": reconstruct(mu.mean(dim=0, keepdim=True).expand_as(mu), residual["basis"], residual["coeff"]), "storage": shared_rank2_storage(regions, dim), "fit": {"mu_hat": mu.mean(dim=0, keepdim=True).expand_as(mu)}})
    return variants, residual, pca_fit


def memory_for_variant(base_memory, keys, device):
    return {
        "image_keys": base_memory["image_keys"].float().to(device),
        "image_values": base_memory["image_values"].float().to(device),
        "patch_keys": keys.float().to(device),
        "patch_values": base_memory["patch_values"].float().to(device),
    }


def select_source_methods(rows):
    selected5, selected10 = {}, {}
    for direction in sorted(set(r["direction"] for r in rows)):
        selected5[direction] = None
        selected10[direction] = None
    return {
        "status": "NO_SOURCE_SELECTED_95_PERCENT_COMPRESSION",
        "selected_5pct_by_direction": selected5,
        "selected_10pct_by_direction": selected10,
        "reason": "Category-held-out source validation is not available in the current dataset wrapper; target metrics are recorded only as POST_HOC_TARGET_DIAGNOSTIC.",
        **selection_note(),
    }


def summarize_generalization(rows):
    out = []
    for direction in sorted(set(r["direction"] for r in rows)):
        drows = [r for r in rows if r["direction"] == direction]
        full = next((r for r in drows if r["variant"] == "V0_FullFPS"), None)
        if not full:
            continue
        for r in drows:
            out.append(
                {
                    "Method": r["variant"],
                    "Source": r["source"],
                    "Target": r["target"],
                    "Storage ratio": r["storage_ratio"],
                    "I-AUROC": r["I-AUROC"],
                    "dFull I": float(r["I-AUROC"]) - float(full["I-AUROC"]),
                    "P-AUROC": r["P-AUROC"],
                    "dFull P": float(r["P-AUROC"]) - float(full["P-AUROC"]),
                    "PRO": r["PRO"],
                    "dFull PRO": float(r["PRO"]) - float(full["PRO"]),
                }
            )
    return out


def main():
    args = parse_args("Phase V-C cross-dataset shared-basis generalization and ultra-compression audit")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    set_seed(seed)
    out = Path(cfg.get("output_dir", "results/ultra_compact_memory"))
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "failure_traceback.txt"
    if stale.exists():
        stale.unlink()
    save_run_metadata(out, cfg, seed)
    device = cfg.get("device", "cuda:0")
    summary_rows, all_rows, retrieval_rows, rep_rows, storage_rows, per_category_rows, costs = [], [], [], [], [], [], []
    sanity = {
        "canonical_formula": FORMULA_NAME,
        "target_labels_used_for_fitting_or_selection": False,
        "no_graph": True,
        "no_spr": True,
        "no_text_prompt": True,
        "full_bank_retrieval_unchanged": True,
    }
    try:
        dataset_audit = build_dataset_audit(cfg)
        write_json(out / "dataset_audit.json", dataset_audit)
        if dataset_audit["unavailable_datasets"]:
            raise RuntimeError(f"Missing required datasets: {dataset_audit['unavailable_datasets']}")
        model = load_model(cfg, device)
        directions = transfer_directions(cfg)
        sanity["both_transfer_directions_configured"] = {d["name"]: d for d in directions}
        for direction in directions:
            dname, source, target = direction["name"], direction["source"], direction["target"]
            _, source_loader = make_loader(cfg, source, cfg.get("source_mode", "test"), shuffle=False)
            _, target_loader = make_loader(cfg, target, cfg.get("target_mode", "test"), shuffle=False)
            source_memory, build_time = build_source_memory(model, source_loader, cfg, source, seed, device, out)
            full = source_memory["patch_keys"].float()
            grouped, mu, centered = decompose_fullfps(full, group_size=4)
            labels = source_memory["patch_values"].argmax(dim=-1).reshape(-1, 4)[:, 0].tolist()
            sanity[f"{dname}_group_count"] = int(grouped.shape[0])
            sanity[f"{dname}_all_groups_size_four"] = bool(full.shape[0] % 4 == 0)
            variants, residual, pca_fit = build_variant_specs(cfg, full, mu, centered, labels)
            sanity[f"{dname}_residual_rank2_shape"] = list(residual["basis"].shape)
            # Materialized reconstruction self-check.
            sr_keys = next(v["patch_keys"] for v in variants if v["name"] == "V2_SharedResidualRank2_FullCentroid")
            sanity[f"{dname}_shared_rank2_finite"] = bool(torch.isfinite(sr_keys).all().item())
            for spec in variants:
                name = spec["name"]
                mpath = out / f"seed_{seed}" / dname / name / "metrics.json"
                if cfg.get("resume", True) and mpath.exists():
                    payload = json.loads(mpath.read_text(encoding="utf-8"))
                    metrics, retrieval, per_cat = payload["metrics"], payload["retrieval"], payload.get("per_category", [])
                else:
                    eval_mem = memory_for_variant(source_memory, spec["patch_keys"], device)
                    metrics, retrieval, per_cat = evaluate_memory_variant(model, eval_mem, target_loader, cfg, device, f"{dname}:{name}")
                    mpath.parent.mkdir(parents=True, exist_ok=True)
                    mpath.write_text(json.dumps({"metrics": metrics, "retrieval": retrieval, "per_category": per_cat}, indent=2, default=json_default), encoding="utf-8")
                    del eval_mem
                    torch.cuda.empty_cache()
                row = {
                    "seed": seed,
                    "direction": dname,
                    "source": source,
                    "target": target,
                    "variant": name,
                    "family": spec["family"],
                    "rank": spec.get("rank", ""),
                    "K": spec.get("K", ""),
                    "storage_ratio": spec["storage"].get("storage_ratio", float("nan")),
                    "I-AUROC": metric(metrics, "I-AUROC"),
                    "I-AP": metric(metrics, "I-AP"),
                    "P-AUROC": metric(metrics, "P-AUROC"),
                    "PRO": metric(metrics, "PRO"),
                }
                summary_rows.append(row)
                all_rows.append(row)
                retrieval_rows.append({"seed": seed, "direction": dname, "variant": name, **retrieval})
                for pc in per_cat:
                    per_category_rows.append({"seed": seed, "direction": dname, "source": source, "target": target, "variant": name, **pc})
                mu_hat = spec.get("fit", {}).get("mu_hat") if spec.get("fit") else None
                if mu_hat is not None:
                    rep = {"seed": seed, "direction": dname, **centroid_diagnostics(name, mu, mu_hat, spec["family"], spec.get("rank", ""), spec.get("K", ""))}
                    rep.update(virtual_diagnostics(name, full, spec["patch_keys"]))
                    if spec.get("fit", {}).get("codebook"):
                        cb = spec["fit"]["codebook"]
                        rep.update(utilization(cb["assignments"], spec.get("K", cb["codebook"].shape[0])))
                    rep_rows.append(rep)
                artifact = out / f"seed_{seed}" / dname / name / "storage_payload.json"
                serialized = write_json_artifact(artifact, {"variant": name, "storage": spec["storage"]})
                storage_rows.append(
                    {
                        "seed": seed,
                        "direction": dname,
                        "variant": name,
                        "family": spec["family"],
                        **spec["storage"],
                        "serialized_file_bytes": serialized,
                        "materialized_virtual_memory_bytes": int(spec["patch_keys"].numel() * spec["patch_keys"].element_size()),
                    }
                )
                costs.append({"seed": seed, "direction": dname, "variant": name, "source_memory_build_time": build_time, "mean_inference_time": retrieval.get("mean_inference_time", float("nan"))})
            del source_memory
            torch.cuda.empty_cache()
        source_selection = select_source_methods(summary_rows)
        write_json(out / "source_selection.json", source_selection)
        cross = summarize_generalization(summary_rows)
        sr_ok = True
        for direction in sorted(set(r["direction"] for r in summary_rows)):
            drows = [r for r in summary_rows if r["direction"] == direction]
            full = next(r for r in drows if r["variant"] == "V0_FullFPS")
            sr = next(r for r in drows if r["variant"] == "V2_SharedResidualRank2_FullCentroid")
            sr_ok = sr_ok and float(sr["storage_ratio"]) <= 0.30
            sr_ok = sr_ok and float(sr["I-AUROC"]) >= float(full["I-AUROC"]) - 0.003
            sr_ok = sr_ok and float(sr["P-AUROC"]) >= float(full["P-AUROC"]) - 0.003
            sr_ok = sr_ok and float(sr["PRO"]) >= float(full["PRO"]) - 0.005
        sanity["both_transfer_directions_completed"] = len(set(r["direction"] for r in summary_rows)) == 2
        sanity["shared_rank2_generalization_supported"] = bool(sr_ok)
        sanity["five_pct_primary_success"] = False
        sanity["five_pct_partial_success"] = False
        sanity["ten_pct_compact_success"] = False
        sanity["one_direction_compression_success"] = False
        outcome = assign_outcome(summary_rows, source_selection, sanity)
        columns = ["seed", "direction", "source", "target", "variant", "family", "rank", "K", "storage_ratio", "I-AUROC", "I-AP", "P-AUROC", "PRO"]
        write_csv(out / "summary_metrics.csv", summary_rows, columns)
        write_csv(out / "all_variant_metrics.csv", all_rows, columns)
        write_csv(out / "centroid_pca_metrics.csv", [r for r in summary_rows if r["family"] == "centroid_pca"], columns)
        write_csv(out / "centroid_codebook_metrics.csv", [r for r in summary_rows if r["family"] == "codebook"], columns)
        write_csv(out / "centroid_offset_metrics.csv", [r for r in summary_rows if r["family"] == "codebook_offset"], columns)
        write_csv(out / "random_pca_controls.csv", [r for r in summary_rows if r["family"] in {"random_pca", "energy_random_pca", "pca_coeff_shuffle"}], columns)
        write_csv(out / "random_codebook_controls.csv", [r for r in summary_rows if r["family"] in {"random_assignment", "random_codebook"}], columns)
        write_csv(out / "cross_dataset_generalization.csv", cross, ["Method", "Source", "Target", "Storage ratio", "I-AUROC", "dFull I", "P-AUROC", "dFull P", "PRO", "dFull PRO"])
        write_csv(out / "transfer_summary.csv", cross, ["Method", "Source", "Target", "Storage ratio", "I-AUROC", "dFull I", "P-AUROC", "dFull P", "PRO", "dFull PRO"])
        if rep_rows:
            write_csv(out / "representation_diagnostics.csv", rep_rows, sorted({k for r in rep_rows for k in r.keys()}))
        else:
            write_csv(out / "representation_diagnostics.csv", [], ["seed", "direction", "variant"])
        write_csv(out / "retrieval_diagnostics.csv", retrieval_rows, sorted({k for r in retrieval_rows for k in r.keys()}))
        write_csv(out / "per_category_metrics.csv", per_category_rows, sorted({k for r in per_category_rows for k in r.keys()}) if per_category_rows else ["seed", "direction", "variant", "category"])
        write_csv(out / "storage_accounting.csv", storage_rows, sorted({k for r in storage_rows for k in r.keys()}))
        write_csv(out / "computational_costs.csv", costs, sorted({k for r in costs for k in r.keys()}))
        write_csv(out / "localization_diagnostics.csv", [], ["seed", "direction", "variant", "fragmentation", "false_positive_area_ratio", "false_negative_area_ratio"])
        write_csv(out / "defect_size_metrics.csv", [], ["seed", "direction", "variant", "defect_size", "P-AUROC", "PRO"])
        comps = []
        for direction in sorted(set(r["direction"] for r in summary_rows)):
            drows = [r for r in summary_rows if r["direction"] == direction]
            cand = next((r for r in drows if r["variant"] == "V2_SharedResidualRank2_FullCentroid"), None)
            ctrls = [r for r in drows if r["family"] == "random_pca"]
            if cand and ctrls:
                for key in ("I-AUROC", "P-AUROC", "PRO"):
                    comps.append({"direction": direction, "candidate": cand["variant"], **control_summary(ctrls, cand, key)})
        write_csv(out / "statistical_comparisons.csv", comps, sorted({k for r in comps for k in r.keys()}) if comps else ["direction", "candidate", "metric"])
        write_csv(out / "success_control_table.csv", cross, ["Method", "Source", "Target", "Storage ratio", "I-AUROC", "dFull I", "P-AUROC", "dFull P", "PRO", "dFull PRO"])
        write_json(out / "sanity_checks.json", sanity)
        write_json(out / "OUTCOME.json", {"Outcome": outcome, **{k: sanity[k] for k in ("shared_rank2_generalization_supported", "five_pct_primary_success", "five_pct_partial_success", "ten_pct_compact_success", "one_direction_compression_success")}})
        lines = [
            "# Phase V-C Execution Summary",
            "",
            f"- Dataset availability: VisA `{dataset_audit['datasets']['visa']['available']}`, MVTec `{dataset_audit['datasets']['mvtec']['available']}`.",
            f"- Transfer directions completed: `{sanity['both_transfer_directions_completed']}`.",
            f"- Canonical feature: `{FORMULA_NAME}`.",
            f"- Source selection status: `{source_selection['status']}`.",
            f"- Shared Residual Rank2 generalization supported: `{sanity['shared_rank2_generalization_supported']}`.",
            f"- Completed metric rows: {len(summary_rows)}.",
            f"- Outcome: `{outcome}`.",
            "",
            "Outputs: `summary_metrics.csv`, `cross_dataset_generalization.csv`, `storage_accounting.csv`, `representation_diagnostics.csv`, `retrieval_diagnostics.csv`, `source_selection.json`, `sanity_checks.json`.",
        ]
        write_md(out / "EXECUTION_SUMMARY.md", lines)
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase V-C — Cross-Dataset Generalization and Ultra-Compression Audit\n\n")
            f.write(f"- Output directory: `{out}`.\n")
            f.write(f"- Completed metric rows: {len(summary_rows)}.\n")
            f.write(f"- Outcome: `{outcome}`.\n")
    except Exception:
        tb = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        write_json(out / "OUTCOME.json", {"Outcome": "G_IMPLEMENTATION_OR_DATASET_FAILURE", "error": tb})
        raise


if __name__ == "__main__":
    main()
