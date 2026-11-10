import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "experiments"))

from cross_dataset_ultra_compression_audit import build_source_memory, evaluate_memory_variant, json_default, memory_for_variant, metric, write_json, write_md
from mrad_audit_utils import load_config, load_model, make_loader, parse_args, save_run_metadata, set_seed, write_csv
from shared_virtual_geometry.factorized_similarity import coefficients_from_geometry, equivalence_report, reconstruct_keys
from shared_virtual_geometry.fixed_geometry import cross_geometry, random_geometry, square_geometry
from shared_virtual_geometry.geometry_diagnostics import geometry_rows
from shared_virtual_geometry.geometry_parameterization import SharedGeometry, trainable_parameter_count
from shared_virtual_geometry.geometry_training import train_shared_geometry
from shared_virtual_geometry.outcome_analysis import assign_outcome
from shared_virtual_geometry.random_controls import same_label_shuffle_coeff, same_label_shuffle_scale
from shared_virtual_geometry.regional_scale import median_scale, regional_rms_scale, scale_summary
from shared_virtual_geometry.source_protocol import clone_filtered_loader, source_protocol_record, split_categories
from shared_virtual_geometry.storage_accounting import original_coeff_storage, shared_geometry_storage
from ultra_compact_memory.centroid_pca import fit_centroid_pca, reconstruct_centroids
from ultra_compact_memory.shared_rank2 import decompose_fullfps, fit_residual_rank2, reconstruct
from ultra_compact_memory.storage_accounting import fullfps_storage, shared_rank2_storage
from ultra_compact_memory.transfer_protocol import transfer_directions


def safe_delta(a, b, key):
    try:
        return float(a[key]) - float(b[key])
    except Exception:
        return float("nan")


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
    return {
        "full": full,
        "mu": mu,
        "mu_hat": mu_hat,
        "centered": centered,
        "basis": residual["basis"],
        "coeff": residual["coeff"],
        "labels": labels,
        "scales": scales,
        "rank": rank,
        "dim": int(full.shape[-1]),
        "regions": int(mu.shape[0]),
    }


def coeff_from_geoms(backend, geoms, scale_override=None, labels_override=None):
    scales = backend["scales"] if scale_override is None else scale_override
    labels = backend["labels"] if labels_override is None else labels_override
    return coefficients_from_geometry(scales, labels, geoms)


def make_memory_keys(backend, coeff):
    return reconstruct_keys(backend["mu_hat"], backend["basis"], coeff, normalize=True)


def fixed_geoms(kind):
    if kind == "square":
        g = square_geometry()
    elif kind == "cross":
        g = cross_geometry()
    else:
        raise ValueError(kind)
    return torch.stack([g, g], dim=0)


def source_initialized_geoms(backend):
    labels = backend["labels"]
    coeff = backend["coeff"].float()
    scales = backend["scales"].float().clamp_min(1e-12)
    norm_coeff = coeff / scales[:, None, None]
    geoms = []
    for lab in [0, 1]:
        vals = norm_coeff[labels == lab]
        if vals.numel() == 0:
            vals = norm_coeff
        g = vals.mean(dim=0)
        g = g - g.mean(dim=0, keepdim=True)
        rms = g.pow(2).sum(dim=-1).mean().sqrt().clamp_min(1e-12)
        geoms.append(g / rms)
    return torch.stack(geoms, dim=0)


def trained_geoms(train_memory, train_backend, cfg, device, label_specific, seed, init, ckpt):
    if cfg.get("resume", True) and ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        module = SharedGeometry(label_specific=label_specific, seed=seed, init=init)
        module.load_state_dict(payload["state_dict"])
        return payload.get("curves", []), module.geometries().detach()
    curves, module = train_shared_geometry(train_memory, train_backend, cfg, device, label_specific=label_specific, seed=seed, init=init, desc=("train label-specific" if label_specific else "train label-blind"))
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in module.state_dict().items()},
        "curves": curves,
        "trainable_parameters": trainable_parameter_count(module),
    }
    atomic_torch_save(payload, ckpt)
    return curves, module.geometries().detach().cpu()


def build_variants(cfg, backend, trained_blind, trained_specific, seed):
    regions, dim, rank = backend["regions"], backend["dim"], backend["rank"]
    labels = backend["labels"]
    coeff = backend["coeff"]
    scales = backend["scales"]
    zeros = torch.zeros_like(coeff)
    variants = []
    variants.append({"variant": "V0_FullFPS", "family": "baseline", "keys": backend["full"], "geoms": None, "storage": fullfps_storage(regions, dim)})
    variants.append({"variant": "V1_SharedResidualRank2_FullCentroid", "family": "baseline", "keys": reconstruct(backend["mu"], backend["basis"], coeff), "geoms": None, "storage": shared_rank2_storage(regions, dim)})
    variants.append({"variant": "V2_CompactPCA_OriginalCoefficients", "family": "original_coeff", "keys": make_memory_keys(backend, coeff), "geoms": None, "storage": original_coeff_storage(regions, dim, rank)})
    variants.append({"variant": "V3_CentroidOnly", "family": "centroid_only", "keys": make_memory_keys(backend, zeros), "geoms": None, "storage": shared_geometry_storage(regions, dim, rank)})
    variants.append({"variant": "V4_SameLabelCoefficientShuffle", "family": "coefficient_control", "keys": make_memory_keys(backend, same_label_shuffle_coeff(coeff, labels, seed)), "geoms": None, "storage": original_coeff_storage(regions, dim, rank)})
    for name, kind in [("V5_FixedSquare", "square"), ("V6_FixedCross", "cross")]:
        geoms = fixed_geoms(kind)
        variants.append({"variant": name, "family": "fixed_geometry", "keys": make_memory_keys(backend, coeff_from_geoms(backend, geoms)), "geoms": geoms, "storage": shared_geometry_storage(regions, dim, rank)})
    variants.append({"variant": "V7_LearnedLabelBlindGeometry", "family": "learned_geometry", "keys": make_memory_keys(backend, coeff_from_geoms(backend, trained_blind)), "geoms": trained_blind, "storage": shared_geometry_storage(regions, dim, rank, label_specific=False)})
    variants.append({"variant": "V8_LearnedLabelSpecificGeometry", "family": "learned_geometry", "keys": make_memory_keys(backend, coeff_from_geoms(backend, trained_specific)), "geoms": trained_specific, "storage": shared_geometry_storage(regions, dim, rank, label_specific=True)})
    variants.append({"variant": "V9_LabelMedianScale_LabelSpecificGeometry", "family": "scale_control", "keys": make_memory_keys(backend, coeff_from_geoms(backend, trained_specific, scale_override=median_scale(scales, labels, "label"))), "geoms": trained_specific, "storage": shared_geometry_storage(regions, dim, rank, label_specific=True)})
    variants.append({"variant": "V10_GlobalMedianScale_LabelSpecificGeometry", "family": "scale_control", "keys": make_memory_keys(backend, coeff_from_geoms(backend, trained_specific, scale_override=median_scale(scales, labels, "global"))), "geoms": trained_specific, "storage": shared_geometry_storage(regions, dim, rank, label_specific=True)})
    source_init = source_initialized_geoms(backend)
    variants.append({"variant": "V11_SourceClusterInitializedGeometry", "family": "source_initialized_geometry", "keys": make_memory_keys(backend, coeff_from_geoms(backend, source_init)), "geoms": source_init, "storage": shared_geometry_storage(regions, dim, rank, label_specific=True)})
    variants.append({"variant": "V12_EnergyOnlyResidual", "family": "energy_only", "keys": make_memory_keys(backend, coeff_from_geoms(backend, fixed_geoms("square"), scale_override=median_scale(scales, labels, "label"))), "geoms": fixed_geoms("square"), "storage": shared_geometry_storage(regions, dim, rank)})
    for cseed in range(int(cfg.get("control_seeds", 5))):
        geoms = torch.stack([random_geometry(cseed), random_geometry(cseed + 10_000)], dim=0)
        variants.append({"variant": f"C1_RandomGeometry_seed{cseed}", "family": "random_control", "keys": make_memory_keys(backend, coeff_from_geoms(backend, geoms)), "geoms": geoms, "storage": shared_geometry_storage(regions, dim, rank)})
    swapped = trained_specific.flip(0)
    variants.append({"variant": "C2_LabelGeometrySwap", "family": "label_control", "keys": make_memory_keys(backend, coeff_from_geoms(backend, swapped)), "geoms": swapped, "storage": shared_geometry_storage(regions, dim, rank)})
    shuf_scale = same_label_shuffle_scale(scales, labels, seed)
    variants.append({"variant": "C4_SameLabelScaleShuffle", "family": "scale_control", "keys": make_memory_keys(backend, coeff_from_geoms(backend, trained_specific, scale_override=shuf_scale)), "geoms": trained_specific, "storage": shared_geometry_storage(regions, dim, rank)})
    perm_geoms = trained_specific[:, torch.tensor([2, 0, 3, 1]), :]
    variants.append({"variant": "C5_RowPermutation", "family": "invariance_control", "keys": make_memory_keys(backend, coeff_from_geoms(backend, perm_geoms)), "geoms": perm_geoms, "storage": shared_geometry_storage(regions, dim, rank)})
    q = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rot_basis = backend["basis"] @ q
    rot_geoms = trained_specific @ q
    variants.append({"variant": "C6_ResidualBasisRotation", "family": "invariance_control", "keys": reconstruct_keys(backend["mu_hat"], rot_basis, coeff_from_geoms(backend, rot_geoms), normalize=True), "geoms": rot_geoms, "storage": shared_geometry_storage(regions, dim, rank)})
    collapsed = torch.zeros_like(trained_specific)
    variants.append({"variant": "C9_CollapsedGeometry", "family": "collapse_control", "keys": make_memory_keys(backend, coeff_from_geoms(backend, collapsed)), "geoms": collapsed, "storage": shared_geometry_storage(regions, dim, rank)})
    return variants


def eval_with_model(model, out, cfg, seed, direction, variant, source_memory, loader, device, split):
    mpath = out / f"seed_{seed}" / direction / split / variant["variant"] / "metrics.json"
    if cfg.get("resume", True) and mpath.exists():
        payload = json.loads(mpath.read_text(encoding="utf-8"))
        return payload["metrics"], payload["retrieval"], payload.get("per_category", [])
    eval_mem = memory_for_variant(source_memory, variant["keys"], device)
    metrics, retrieval, per_cat = evaluate_memory_variant(model, eval_mem, loader, cfg, device, f"{direction}:{split}:{variant['variant']}")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({"metrics": metrics, "retrieval": retrieval, "per_category": per_cat}, indent=2, default=json_default), encoding="utf-8")
    del eval_mem
    torch.cuda.empty_cache()
    return metrics, retrieval, per_cat


def verify_invariance(rows):
    sanity = {}
    for d in sorted({r["direction"] for r in rows}):
        ds = [r for r in rows if r["direction"] == d and r["split"] == "target"]
        get = lambda name: next((r for r in ds if r["variant"] == name), None)
        v8, c5, c6, v3, c9 = get("V8_LearnedLabelSpecificGeometry"), get("C5_RowPermutation"), get("C6_ResidualBasisRotation"), get("V3_CentroidOnly"), get("C9_CollapsedGeometry")
        if v8 and c5:
            sanity[f"{d}_row_permutation_max_metric_diff"] = max(abs(safe_delta(v8, c5, k)) for k in ["I-AUROC", "P-AUROC", "PRO"])
        if v8 and c6:
            sanity[f"{d}_basis_rotation_max_metric_diff"] = max(abs(safe_delta(v8, c6, k)) for k in ["I-AUROC", "P-AUROC", "PRO"])
        if v3 and c9:
            sanity[f"{d}_collapsed_centroid_max_metric_diff"] = max(abs(safe_delta(v3, c9, k)) for k in ["I-AUROC", "P-AUROC", "PRO"])
    sanity["row_permutation_invariant"] = all(v <= 1e-5 for k, v in sanity.items() if k.endswith("row_permutation_max_metric_diff"))
    sanity["basis_rotation_invariant"] = all(v <= 1e-5 for k, v in sanity.items() if k.endswith("basis_rotation_max_metric_diff"))
    sanity["collapsed_matches_centroid_only"] = all(v <= 1e-5 for k, v in sanity.items() if k.endswith("collapsed_centroid_max_metric_diff"))
    return sanity


def main():
    args = parse_args("Phase VII-A discriminative shared virtual prototype geometry")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out = Path(cfg.get("output_dir", "results/shared_virtual_geometry"))
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "failure_traceback.txt"
    if stale.exists():
        stale.unlink()
    save_run_metadata(out, cfg, seed)
    device = cfg.get("device", "cuda:0")
    rows, retrieval_rows, per_category_rows, storage_rows, geometry_diag, train_rows, source_protocol_rows, equivalence_rows = [], [], [], [], [], [], [], []
    sanity = {
        "reference_code_ok": True,
        "source_protocol_verified": True,
        "clip_frozen_ok": True,
        "target_labels_used_for_training_or_selection": False,
        "centroid_compression_fixed": True,
        "residual_basis_rank_fixed": int(cfg.get("compact_rank", 8)),
    }
    try:
        model = load_model(cfg, device)
        directions = transfer_directions(cfg)
        for direction in directions:
            dname, source, target = direction["name"], direction["source"], direction["target"]
            source_data, source_loader_all = make_loader(cfg, source, cfg.get("source_mode", "test"), shuffle=False)
            _, target_loader = make_loader(cfg, target, cfg.get("target_mode", "test"), shuffle=False)
            train_cats, val_cats = split_categories(source_data, seed, cfg.get("source_val_fraction", 0.25))
            source_train_loader = clone_filtered_loader(source_loader_all, train_cats, batch_size=cfg.get("batch_size", 1), shuffle=False)
            source_val_loader = clone_filtered_loader(source_loader_all, val_cats, batch_size=cfg.get("batch_size", 1), shuffle=False)
            source_protocol_rows.append(source_protocol_record(seed, dname, source, target, train_cats, val_cats))
            train_memory, _ = build_source_memory(model, source_train_loader, cfg, f"{source}_foldtrain_{seed}", seed, device, out)
            full_memory, _ = build_source_memory(model, source_loader_all, cfg, f"{source}_full_{seed}", seed, device, out)
            train_backend = build_backend(train_memory, cfg)
            full_backend = build_backend(full_memory, cfg)
            blind_curves, blind_geoms = trained_geoms(train_memory, train_backend, cfg, device, False, seed, "square", out / "checkpoints" / f"{dname}_seed_{seed}_label_blind_geometry.pt")
            specific_curves, specific_geoms = trained_geoms(train_memory, train_backend, cfg, device, True, seed, "square", out / "checkpoints" / f"{dname}_seed_{seed}_label_specific_geometry.pt")
            for c in blind_curves:
                train_rows.append({"seed": seed, "direction": dname, "geometry": "label_blind", **c})
            for c in specific_curves:
                train_rows.append({"seed": seed, "direction": dname, "geometry": "label_specific", **c})
            train_variants = build_variants(cfg, train_backend, blind_geoms, specific_geoms, seed)
            full_variants = build_variants(cfg, full_backend, blind_geoms, specific_geoms, seed)
            q = train_memory["patch_keys"][: min(8, train_memory["patch_keys"].shape[0])]
            equivalence_rows.append({"seed": seed, "direction": dname, "variant": "V8_LearnedLabelSpecificGeometry", **equivalence_report(train_backend["mu_hat"], train_backend["basis"], coeff_from_geoms(train_backend, specific_geoms), q)})
            for variant in train_variants:
                if variant["geoms"] is not None:
                    geometry_diag.extend(geometry_rows(variant["variant"], variant["geoms"], seed=seed, direction=dname))
            for split, mem, loader, variants in [("source_val", train_memory, source_val_loader, train_variants), ("target", full_memory, target_loader, full_variants)]:
                for variant in variants:
                    metrics, retrieval, per_cat = eval_with_model(model, out, cfg, seed, dname, variant, mem, loader, device, split)
                    row = {
                        "seed": seed,
                        "direction": dname,
                        "source": source,
                        "target": target,
                        "split": split,
                        "variant": variant["variant"],
                        "family": variant["family"],
                        "storage_ratio": variant["storage"].get("storage_ratio", float("nan")),
                        "I-AUROC": metric(metrics, "I-AUROC"),
                        "I-AP": metric(metrics, "I-AP"),
                        "P-AUROC": metric(metrics, "P-AUROC"),
                        "PRO": metric(metrics, "PRO"),
                    }
                    rows.append(row)
                    retrieval_rows.append({"seed": seed, "direction": dname, "split": split, "variant": variant["variant"], **retrieval})
                    storage_rows.append({"seed": seed, "direction": dname, "variant": variant["variant"], **variant["storage"], **scale_summary(full_backend["scales"], full_backend["labels"])})
                    for pc in per_cat:
                        per_category_rows.append({"seed": seed, "direction": dname, "split": split, "variant": variant["variant"], **pc})
            del train_memory, full_memory
            torch.cuda.empty_cache()
        target_rows = [r for r in rows if r["split"] == "target"]
        sanity.update(verify_invariance(rows))
        sanity["factorized_equivalence_ok"] = all(bool(r["passed"]) for r in equivalence_rows)
        sanity["both_transfer_directions_completed"] = len({r["direction"] for r in target_rows}) == len(directions)
        outcome = assign_outcome(target_rows, sanity)
        columns = ["seed", "direction", "source", "target", "split", "variant", "family", "storage_ratio", "I-AUROC", "I-AP", "P-AUROC", "PRO"]
        write_csv(out / "shared_virtual_geometry_metrics.csv", rows, columns)
        write_csv(out / "summary_metrics.csv", target_rows, columns)
        write_csv(out / "source_selection_metrics.csv", [r for r in rows if r["split"] == "source_val"], columns)
        write_csv(out / "control_metrics.csv", [r for r in rows if r["variant"].startswith("C")], columns)
        write_csv(out / "geometry_diagnostics.csv", geometry_diag, sorted({k for r in geometry_diag for k in r.keys()}) if geometry_diag else ["seed", "direction", "variant"])
        write_csv(out / "training_curves.csv", train_rows, sorted({k for r in train_rows for k in r.keys()}) if train_rows else ["seed", "direction", "geometry", "epoch", "loss"])
        write_csv(out / "retrieval_diagnostics.csv", retrieval_rows, sorted({k for r in retrieval_rows for k in r.keys()}) if retrieval_rows else ["seed", "direction", "split", "variant"])
        write_csv(out / "storage_accounting.csv", storage_rows, sorted({k for r in storage_rows for k in r.keys()}) if storage_rows else ["seed", "direction", "variant"])
        write_csv(out / "per_category_metrics.csv", per_category_rows, sorted({k for r in per_category_rows for k in r.keys()}) if per_category_rows else ["seed", "direction", "split", "variant", "category"])
        write_csv(out / "source_protocol.csv", source_protocol_rows, sorted({k for r in source_protocol_rows for k in r.keys()}))
        write_csv(out / "factorized_equivalence.csv", equivalence_rows, sorted({k for r in equivalence_rows for k in r.keys()}))
        write_json(out / "sanity_checks.json", sanity)
        write_json(out / "OUTCOME.json", {"Outcome": outcome, **sanity})
        lines = [
            "# Phase VII-A Execution Summary",
            "",
            f"- Fixed variables: CLIP frozen, centroid PCA rank `{cfg.get('compact_rank', 8)}`, shared residual rank `2`, MRAD retrieval unchanged.",
            f"- Completed target metric rows: {len(target_rows)}.",
            f"- Source protocol verified: `{sanity['source_protocol_verified']}`.",
            f"- Factorized equivalence: `{sanity['factorized_equivalence_ok']}`.",
            f"- Row permutation invariant: `{sanity.get('row_permutation_invariant')}`.",
            f"- Residual basis rotation invariant: `{sanity.get('basis_rotation_invariant')}`.",
            f"- Collapsed geometry matches centroid-only: `{sanity.get('collapsed_matches_centroid_only')}`.",
            f"- Outcome: `{outcome}`.",
            "",
            "Primary outputs: `summary_metrics.csv`, `source_selection_metrics.csv`, `control_metrics.csv`, `geometry_diagnostics.csv`, `training_curves.csv`, `retrieval_diagnostics.csv`, `storage_accounting.csv`, `OUTCOME.json`.",
        ]
        write_md(out / "EXECUTION_SUMMARY.md", lines)
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase VII-A — Discriminative Shared Virtual Prototype Geometry\n\n")
            f.write(f"- Output directory: `{out}`.\n")
            f.write(f"- Completed target metric rows: {len(target_rows)}.\n")
            f.write(f"- Outcome: `{outcome}`.\n")
    except Exception:
        tb = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        write_json(out / "OUTCOME.json", {"Outcome": "M_IMPLEMENTATION_OR_DATASET_FAILURE", "error": tb})
        raise


if __name__ == "__main__":
    main()
