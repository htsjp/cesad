import csv
import json
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
from base_residual_memory.canonical import FORMULA_NAME, canonical_fuse, tensor_checksum
from base_residual_memory.grouping import load_head_state, recover_groups
from base_residual_memory.lazy_similarity import lazy_logits
from base_residual_memory.local_svd import decompose_groups
from base_residual_memory.reconstruction import materialized_memory
from shared_residual_memory.covariance import check_orthonormal, covariance_pca
from shared_residual_memory.diagnostics import basis_rows, reconstruction_row
from shared_residual_memory.label_basis import fit_label_bases, reconstruct_label_conditioned
from shared_residual_memory.outcome_analysis import assign_outcome
from shared_residual_memory.random_controls import (
    energy_matched_random,
    fixed_simplex_coefficients,
    gaussian_matched_coefficients,
    random_basis,
    same_label_coefficient_shuffle,
)
from shared_residual_memory.shared_basis import coefficients, reconstruct_from_basis, reconstruct_grouped, retained_energy
from shared_residual_memory.storage_accounting import global_shared_storage
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


def write_md(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def metric(metrics, key):
    return float(metrics.get("mean", {}).get(key, float("nan")))


def load_inputs(cfg, seed):
    head_path = Path(cfg.get("head_memory_dir", "results/head_selective_residual/checkpoints")) / f"seed_{seed}_head_identity_memory.pt"
    value_path = Path(cfg.get("value_memory_dir", "results/value_representation_audit/checkpoints")) / f"seed_{seed}_T24_plus_V12_18_alpha0.5_memory.pt"
    if not head_path.exists():
        head_path = head_path.with_name("seed_0_head_identity_memory.pt")
    if not value_path.exists():
        value_path = value_path.with_name("seed_0_T24_plus_V12_18_alpha0.5_memory.pt")
    return load_head_state(str(head_path)), torch.load(value_path, map_location="cpu", weights_only=False), head_path, value_path


@torch.no_grad()
def evaluate_memory_variant(model, memory, loader, cfg, device, variant_name):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    results = init_results(loader.dataset.obj_list)
    times, entropies, margins, top1 = [], [], [], []
    labels = memory["patch_values"].argmax(dim=-1).detach().cpu().numpy()
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
    retrieval = {
        "variant": variant_name,
        "retrieval_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "top1_margin": float(np.mean(margins)) if margins else float("nan"),
        "unique_prototypes_used": used,
        "never_used_fraction": float(1.0 - used / max(1, memory["patch_keys"].shape[0])),
        "top1_anomaly_fraction": float(np.mean([labels[int(x)] for x in top1])) if top1 else float("nan"),
        "mean_inference_time": float(np.mean(times)) if times else float("nan"),
    }
    return metrics, retrieval


def build_variants(cfg, full, mu, centered, group_labels, global_basis, eigvals, label_bases):
    variants = []
    variants.append({"name": "V0_FullFPS", "family": "fullfps", "rank": "", "keys": full, "storage_ratio": 1.0, "true_basis": True})
    centroid = F.normalize(mu[:, None, :].expand(-1, 4, -1), dim=-1).reshape(-1, full.shape[-1])
    variants.append({"name": "V1_FPSCentroidRepeated4", "family": "centroid", "rank": 0, "keys": centroid, "storage_ratio": 0.25, "true_basis": False})
    max_rank = global_basis.shape[1]
    for rank in [1, 2, 4, 8, 16]:
        if rank <= max_rank:
            b = global_basis[:, :rank]
            c = coefficients(centered, b)
            keys = reconstruct_from_basis(mu, c, b)
            sr = global_shared_storage(len(group_labels), full.shape[-1], rank, label_conditioned=False)["feature_scalar_ratio"]
            variants.append({"name": f"V{rank+1}_GlobalSharedPCA_Rank{rank}", "family": "global_pca", "rank": rank, "keys": keys, "storage_ratio": sr, "coeff": c, "basis": b, "true_basis": True})
    for rank, name in [(1, "V7_LabelConditionedPCA_Rank1"), (2, "V8_LabelConditionedPCA_Rank2"), (4, "V9_LabelConditionedPCA_Rank4")]:
        keys = reconstruct_label_conditioned(mu, centered, group_labels, label_bases, rank)
        sr = global_shared_storage(len(group_labels), full.shape[-1], rank, label_conditioned=True)["feature_scalar_ratio"]
        variants.append({"name": name, "family": "label_pca", "rank": rank, "keys": keys, "storage_ratio": sr, "true_basis": True})
    controls = 5 if cfg.get("stage1", True) else int(cfg.get("control_seeds", 20))
    for seed in range(controls):
        for rank, cname in [(1, "C1_RandomGlobalBasis_Rank1"), (2, "C2_RandomGlobalBasis_Rank2"), (4, "C3_RandomGlobalBasis_Rank4")]:
            if rank in cfg.get("random_control_ranks", [2]):
                b = random_basis(full.shape[-1], rank, seed)
                c = coefficients(centered, b)
                keys = reconstruct_from_basis(mu, c, b)
                sr = global_shared_storage(len(group_labels), full.shape[-1], rank, label_conditioned=False)["feature_scalar_ratio"]
                variants.append({"name": f"{cname}_seed{seed}", "family": "random_basis", "rank": rank, "keys": keys, "storage_ratio": sr, "true_basis": False})
        for rank in cfg.get("energy_matched_ranks", [2]):
            b = global_basis[:, :rank]
            pc = coefficients(centered, b)
            keys, c, rb = energy_matched_random(mu, centered, pc, rank, seed)
            sr = global_shared_storage(len(group_labels), full.shape[-1], rank, label_conditioned=False)["feature_scalar_ratio"]
            variants.append({"name": f"C4_EnergyMatchedRandomGlobalBasis_Rank{rank}_seed{seed}", "family": "energy_random", "rank": rank, "keys": keys, "storage_ratio": sr, "true_basis": False})
        for rank in cfg.get("coefficient_shuffle_ranks", [2]):
            b = global_basis[:, :rank]
            pc = coefficients(centered, b)
            keys, c = same_label_coefficient_shuffle(mu, pc, b, group_labels, seed)
            sr = global_shared_storage(len(group_labels), full.shape[-1], rank, label_conditioned=False)["feature_scalar_ratio"]
            variants.append({"name": f"C5_SameLabelCoefficientShuffle_Rank{rank}_seed{seed}", "family": "coeff_shuffle", "rank": rank, "keys": keys, "storage_ratio": sr, "true_basis": True})
    for rank in cfg.get("fixed_simplex_ranks", [2]):
        b = global_basis[:, :rank]
        for mode in ("region_norm", "label_median"):
            keys, c = fixed_simplex_coefficients(mu, centered, b, mode=mode)
            sr = global_shared_storage(len(group_labels), full.shape[-1], rank, label_conditioned=False)["feature_scalar_ratio"]
            variants.append({"name": f"C8_FixedSimplexCoefficients_Rank{rank}_{mode}", "family": "fixed_simplex", "rank": rank, "keys": keys, "storage_ratio": sr, "true_basis": True})
    variants.append({"name": "C11_SingleCentroidPhysicalEntry", "family": "single_centroid", "rank": 0, "keys": F.normalize(mu, dim=-1), "storage_ratio": 0.25, "true_basis": False, "single_entry": True})
    variants.append({"name": "C13_FourDuplicateCentroidEntries", "family": "duplicate_centroid", "rank": 0, "keys": centroid, "storage_ratio": 0.25, "true_basis": False})
    return variants


def main():
    args = parse_args("Phase V-B shared residual basis memory organization")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    set_seed(seed)
    out = Path(cfg.get("output_dir", "results/shared_residual_basis"))
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "failure_traceback.txt"
    if stale.exists():
        stale.unlink()
    save_run_metadata(out, cfg, seed)
    device = cfg.get("device", "cuda:0")
    model = load_model(cfg, device)
    rows, all_rows, rec_rows, basis_diag, coeff_diag, retrieval_rows, storage_rows = [], [], [], [], [], [], []
    sanity = {"canonical_formula": FORMULA_NAME}
    try:
        head, value_mem, head_path, value_path = load_inputs(cfg, seed)
        groups, grouping = recover_groups(head["entries"], head["memory"])
        full = value_mem["patch_keys"].float()
        decomp = decompose_groups(full, groups)
        mu, centered = decomp["mu"], decomp["centered"]
        group_labels = [int(g["label"]) for g in groups]
        sanity["baseline_checksum"] = tensor_checksum(full)
        sanity["grouping_ok"] = grouping.get("status") == "OK"
        if not sanity["grouping_ok"]:
            raise RuntimeError("Grouping failed")
        residual_row_sum = centered.sum(dim=1).abs().max().item()
        residual_global_mean_norm = centered.reshape(-1, centered.shape[-1]).mean(dim=0).norm().item()
        grouping.update({"residual_row_sum_max_abs": residual_row_sum, "global_residual_mean_norm": residual_global_mean_norm, "canonical_value_checkpoint": str(value_path)})
        write_json(out / "grouping_and_input_audit.json", grouping)

        max_rank = int(cfg.get("max_basis_rank", 16))
        flat = centered.reshape(-1, centered.shape[-1])
        global_basis, top_eigs, all_eigs = covariance_pca(flat, max_rank)
        label_bases, label_eigs, label_counts = fit_label_bases(centered, group_labels, max_rank)
        sanity["global_basis_orthonormal_error"] = check_orthonormal(global_basis)
        sanity["eigenvalues_sorted_descending"] = bool(torch.all(all_eigs[:-1] >= all_eigs[1:]).item())
        sanity["basis_orthonormal"] = sanity["global_basis_orthonormal_error"] <= 1e-5
        if not sanity["basis_orthonormal"]:
            raise RuntimeError("Basis orthonormality failed")
        basis_diag.append(basis_rows("GlobalSharedPCA", all_eigs, max_rank, sanity["global_basis_orthonormal_error"], "global"))
        for label in (0, 1):
            err = check_orthonormal(label_bases[label])
            sanity[f"label_{label}_basis_orthonormal_error"] = err
            basis_diag.append(basis_rows("LabelConditionedPCA", label_eigs[label]["all"], max_rank, err, f"label_{label}"))
        write_json(out / "basis_fit_audit.json", {"global_residual_vectors": int(flat.shape[0]), "label_counts": label_counts})

        # Rotation invariance check on rank 2.
        b2 = global_basis[:, :2]
        q, _ = torch.linalg.qr(torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=torch.float32))
        c2 = coefficients(centered, b2)
        rec = reconstruct_from_basis(mu, c2, b2)
        brot = b2 @ q
        crot = coefficients(centered, brot)
        rec_rot = reconstruct_from_basis(mu, crot, brot)
        sanity["rotation_invariant_feature_max_abs_diff"] = float((rec - rec_rot).abs().max().item())
        if sanity["rotation_invariant_feature_max_abs_diff"] > 1e-6:
            raise RuntimeError("Rotation invariance failed")

        grouped = reconstruct_grouped(mu, c2, b2)
        qtest = full[:64].unsqueeze(0)
        vals = value_mem["patch_values"].float()
        l_lazy, s_lazy, _ = lazy_logits(qtest, grouped, vals, batch_groups=256)
        l_mat, s_mat, _ = retrieval_logits(qtest, grouped.reshape(-1, grouped.shape[-1]), vals)
        sanity["lazy_materialized_logits_max_abs_diff"] = float((l_lazy - l_mat).abs().max().item())
        sanity["lazy_materialized_similarity_max_abs_diff"] = float((s_lazy - s_mat).abs().max().item())
        sanity["lazy_materialized_match"] = sanity["lazy_materialized_logits_max_abs_diff"] <= 1e-6 and sanity["lazy_materialized_similarity_max_abs_diff"] <= 1e-6
        if not sanity["lazy_materialized_match"]:
            raise RuntimeError("Lazy/materialized mismatch")

        _, target_loader = make_loader(cfg, cfg.get("dataset", "mvtec"), cfg.get("target_mode", "test"), shuffle=False)
        variants = build_variants(cfg, full, mu, centered, group_labels, global_basis, all_eigs, label_bases)
        if cfg.get("run_mode") == "smoke":
            keep = set(cfg.get("smoke_variants", ["V0_FullFPS", "V1_FPSCentroidRepeated4", "V3_GlobalSharedPCA_Rank2", "V8_LabelConditionedPCA_Rank2", "C2_RandomGlobalBasis_Rank2_seed0", "C4_EnergyMatchedRandomGlobalBasis_Rank2_seed0", "C5_SameLabelCoefficientShuffle_Rank2_seed0", "C8_FixedSimplexCoefficients_Rank2_region_norm", "C13_FourDuplicateCentroidEntries"]))
            variants = [v for v in variants if v["name"] in keep]

        image_values = value_mem["image_values"].float()
        patch_values_full = value_mem["patch_values"].float()
        for spec in variants:
            name = spec["name"]
            mpath = out / f"seed_{seed}" / name / "metrics.json"
            if cfg.get("resume", True) and mpath.exists():
                payload = json.loads(mpath.read_text(encoding="utf-8"))
                metrics, retrieval = payload["metrics"], payload["retrieval"]
            else:
                if spec.get("single_entry"):
                    labels = torch.stack([patch_values_full[torch.tensor(g["positions"], dtype=torch.long)][0] for g in groups], dim=0)
                else:
                    labels = patch_values_full
                mem_eval = materialized_memory({"image_keys": value_mem["image_keys"].float().to(device), "image_values": image_values.to(device), "patch_values": labels.to(device)}, spec["keys"].to(device))
                metrics, retrieval = evaluate_memory_variant(model, mem_eval, target_loader, cfg, device, name)
                mpath.parent.mkdir(parents=True, exist_ok=True)
                mpath.write_text(json.dumps({"metrics": metrics, "retrieval": retrieval}, indent=2, default=json_default), encoding="utf-8")
                del mem_eval
                torch.cuda.empty_cache()
            row = {"seed": seed, "variant": name, "family": spec["family"], "rank": spec["rank"], "storage_ratio": spec["storage_ratio"], "I-AUROC": metric(metrics, "I-AUROC"), "I-AP": metric(metrics, "I-AP"), "P-AUROC": metric(metrics, "P-AUROC"), "PRO": metric(metrics, "PRO")}
            rows.append(row)
            all_rows.append(row)
            retrieval_rows.append({"seed": seed, **retrieval})
            energy = float(retained_energy(centered, spec["coeff"]).mean().item()) if "coeff" in spec else float("nan")
            rec_rows.append({"seed": seed, **reconstruction_row(name, full if not spec.get("single_entry") else F.normalize(mu, dim=-1), spec["keys"], spec["family"], spec["rank"], energy)})
            label_conditioned = spec["family"] == "label_pca"
            rank = int(spec["rank"]) if str(spec["rank"]).isdigit() else 0
            if spec["family"] == "fullfps":
                full_scalars = len(groups) * 4 * full.shape[-1]
                sr = {
                    "full_feature_scalars": full_scalars,
                    "compressed_feature_scalars": full_scalars,
                    "feature_scalar_ratio": 1.0,
                    "fp32_feature_bytes": full_scalars * 4,
                    "fp16_feature_bytes": full_scalars * 2,
                    "full_fp32_feature_bytes": full_scalars * 4,
                    "full_fp16_feature_bytes": full_scalars * 2,
                    "basis_count": 0,
                }
            elif spec["family"] in {"centroid", "duplicate_centroid", "single_centroid"}:
                full_scalars = len(groups) * 4 * full.shape[-1]
                compressed = len(groups) * full.shape[-1]
                sr = {
                    "full_feature_scalars": full_scalars,
                    "compressed_feature_scalars": compressed,
                    "feature_scalar_ratio": compressed / full_scalars,
                    "fp32_feature_bytes": compressed * 4,
                    "fp16_feature_bytes": compressed * 2,
                    "full_fp32_feature_bytes": full_scalars * 4,
                    "full_fp16_feature_bytes": full_scalars * 2,
                    "basis_count": 0,
                }
            else:
                sr = global_shared_storage(len(groups), full.shape[-1], rank, label_conditioned=label_conditioned)
            storage_rows.append({"seed": seed, "variant": name, "family": spec["family"], "rank": spec["rank"], **sr, "reported_storage_ratio": spec["storage_ratio"]})

        full_row = next(r for r in rows if r["variant"] == "V0_FullFPS")
        sanity["baseline_ok"] = abs(float(full_row["I-AUROC"]) - 0.941900) <= 0.002 and abs(float(full_row["P-AUROC"]) - 0.921767) <= 0.002 and abs(float(full_row["PRO"]) - 0.878239) <= 0.005
        write_csv(out / "summary_metrics.csv", rows, list(rows[0].keys()))
        write_csv(out / "all_variant_metrics.csv", all_rows, list(rows[0].keys()))
        write_csv(out / "global_basis_metrics.csv", [r for r in rows if r["family"] == "global_pca"], list(rows[0].keys()))
        write_csv(out / "label_basis_metrics.csv", [r for r in rows if r["family"] == "label_pca"], list(rows[0].keys()))
        write_csv(out / "random_basis_controls.csv", [r for r in rows if r["family"] == "random_basis"], list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "energy_matched_random_controls.csv", [r for r in rows if r["family"] == "energy_random"], list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "coefficient_shuffle_controls.csv", [r for r in rows if r["family"] == "coeff_shuffle"], list(rows[0].keys()) if rows else ["seed"])
        write_csv(out / "virtual_geometry_controls.csv", [r for r in rows if r["family"] in {"fixed_simplex", "single_centroid", "duplicate_centroid", "centroid"}], list(rows[0].keys()))
        write_csv(out / "label_basis_controls.csv", [], ["seed", "variant", "family", "rank", "I-AUROC", "P-AUROC", "PRO"])
        write_csv(out / "energy_adaptive_metrics.csv", [], ["seed", "variant", "threshold", "I-AUROC", "P-AUROC", "PRO"])
        write_csv(out / "reconstruction_diagnostics.csv", rec_rows, list(rec_rows[0].keys()))
        write_csv(out / "basis_diagnostics.csv", basis_diag, list(basis_diag[0].keys()))
        write_csv(out / "coefficient_diagnostics.csv", coeff_diag, ["seed", "variant", "coefficient_entropy", "covariance_trace"])
        write_csv(out / "retrieval_diagnostics.csv", retrieval_rows, list(retrieval_rows[0].keys()))
        write_csv(out / "storage_accounting.csv", storage_rows, list(storage_rows[0].keys()))
        write_csv(out / "computational_costs.csv", [{"variant": r["variant"], "mean_inference_time": next((x["mean_inference_time"] for x in retrieval_rows if x["variant"] == r["variant"]), float("nan"))} for r in rows], ["variant", "mean_inference_time"])
        write_csv(out / "per_category_metrics.csv", [], ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
        write_csv(out / "defect_size_metrics.csv", [], ["seed", "variant", "defect_size", "I-AUROC", "P-AUROC", "PRO"])
        write_csv(out / "localization_diagnostics.csv", [], ["seed", "variant", "fragmentation", "false_positive_area_ratio", "false_negative_area_ratio"])
        write_csv(out / "statistical_comparisons.csv", [], ["comparison", "metric", "delta", "wins", "ties", "losses"])
        write_csv(out / "success_control_table.csv", [{"Variant": r["variant"], "Rank": r["rank"], "Storage ratio": r["storage_ratio"], "I-AUROC": r["I-AUROC"], "P-AUROC": r["P-AUROC"], "PRO": r["PRO"]} for r in rows], ["Variant", "Rank", "Storage ratio", "I-AUROC", "P-AUROC", "PRO"])
        source_selection = {"status": "NO_SOURCE_SELECTED_SHARED_BASIS", "protocol": "source-validation fold implementation unavailable in current dataset wrapper; no MVTec target labels used", "selected_method": None}
        write_json(out / "source_selection.json", source_selection)
        outcome = assign_outcome(rows, sanity)
        write_json(out / "sanity_checks.json", sanity)
        write_json(out / "OUTCOME.json", {"Outcome": outcome})
        summary = ["# Phase V-B Execution Summary", "", f"- FullFPS reproduction: `{sanity['baseline_ok']}`", f"- Grouping verification: `{sanity['grouping_ok']}`", f"- Completed variants: {len(rows)}", f"- Lazy/materialized equivalence: `{sanity['lazy_materialized_match']}`", f"- Source selection: `{source_selection['status']}`", f"- Outcome: `{outcome}`"]
        write_md(out / "EXECUTION_SUMMARY.md", summary)
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase V-B — Shared Residual Basis Memory Organization\n\n")
            f.write(f"- Output directory: `{out}`.\n")
            f.write(f"- Completed variants: {len(rows)}.\n")
            f.write(f"- Outcome: `{outcome}`.\n")
    except Exception:
        tb = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        write_json(out / "OUTCOME.json", {"Outcome": "IMPLEMENTATION_OR_METADATA_FAILURE", "error": tb})
        raise


if __name__ == "__main__":
    main()
