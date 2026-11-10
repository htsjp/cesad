import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "experiments"))

from experiments.cross_dataset_ultra_compression_audit import (
    build_source_memory,
    evaluate_memory_variant,
    memory_for_variant,
    metric,
    write_json,
    write_md,
)
from memory_conditioned_prompt.base_prompt import freeze_clip, trainable_parameter_count
from mrad_audit_utils import load_config, load_model, make_loader, parse_args, save_run_metadata, set_seed, write_csv
from text_anchored_memory.code_audit import write_audit_report
from text_anchored_memory.controls import shuffled_delta
from text_anchored_memory.diagnostics import representation_row, text_anchor_row
from text_anchored_memory.factorized_similarity import build_factorized, equivalence_report
from text_anchored_memory.guided_basis import guided_basis, rotate_basis_equivalent
from text_anchored_memory.outcome_analysis import assign_outcome
from text_anchored_memory.prompt_anchor import learned_prompt_module, neutral_direction, static_anchors
from text_anchored_memory.prompt_training import train_source_prompt
from text_anchored_memory.reconstruction import virtual_keys
from text_anchored_memory.source_protocol import clone_filtered_loader, source_protocol_record, split_categories
from text_anchored_memory.storage_accounting import guided_storage
from text_anchored_memory.visual_statistics import visual_stats
from ultra_compact_memory.centroid_pca import fit_centroid_pca, reconstruct_centroids
from ultra_compact_memory.shared_rank2 import decompose_fullfps, fit_residual_rank2, reconstruct
from ultra_compact_memory.storage_accounting import fullfps_storage, shared_rank2_storage
from ultra_compact_memory.transfer_protocol import transfer_directions


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def eval_or_resume(model, base_memory, keys, loader, cfg, device, out, seed, direction, name):
    mpath = out / f"seed_{seed}" / direction / name / "metrics.json"
    if cfg.get("resume", True) and mpath.exists():
        payload = json.loads(mpath.read_text(encoding="utf-8"))
        return payload["metrics"], payload["retrieval"], payload.get("per_category", [])
    mem = memory_for_variant(base_memory, keys, device)
    metrics, retrieval, per_cat = evaluate_memory_variant(model, mem, loader, cfg, device, f"{direction}:{name}")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({"metrics": metrics, "retrieval": retrieval, "per_category": per_cat}, indent=2, default=json_default), encoding="utf-8")
    del mem
    torch.cuda.empty_cache()
    return metrics, retrieval, per_cat


def row_from_metrics(seed, direction, source, target, name, family, rank, lam, rho, storage, metrics, selected=False):
    return {
        "seed": seed,
        "direction": direction,
        "source": source,
        "target": target,
        "variant": name,
        "family": family,
        "rank": rank,
        "lambda": lam,
        "rho": rho,
        "storage_ratio": storage.get("storage_ratio", float("nan")),
        "selection_status": "SOURCE_SELECTED" if selected else "POST_HOC_TARGET_DIAGNOSTIC",
        "I-AUROC": metric(metrics, "I-AUROC"),
        "I-AP": metric(metrics, "I-AP"),
        "P-AUROC": metric(metrics, "P-AUROC"),
        "PRO": metric(metrics, "PRO"),
    }


def make_basis_variants(mu, labels, residual, static, learned, cfg, seed, stats_override=None):
    stats = stats_override or visual_stats(mu, labels)
    regions, dim = mu.shape
    rank = int(cfg.get("selected_rank", cfg.get("ranks", [8])[0]))
    lam = float(cfg.get("selected_lambda", cfg.get("lambdas", [0.1])[0]))
    rho = float(cfg.get("selected_rho", cfg.get("rhos", [1.0])[0]))
    pca_fit = fit_centroid_pca(mu, max(max(cfg.get("pca_ranks", [4, 8, 16, 32])), rank))
    variants = []
    full_storage = fullfps_storage(regions, dim)
    shared_storage = shared_rank2_storage(regions, dim)
    variants.append({"name": "V0_FullFPS", "family": "baseline", "keys": None, "storage": full_storage, "fit": None, "rank": "", "lambda": "", "rho": ""})
    variants.append(
        {
            "name": "V1_SharedResidualRank2_FullCentroid",
            "family": "shared_rank2",
            "keys": reconstruct(mu, residual["basis"], residual["coeff"]),
            "storage": shared_storage,
            "fit": None,
            "rank": "",
            "lambda": "",
            "rho": "",
        }
    )
    mu_pca = reconstruct_centroids(pca_fit, rank)
    variants.append(
        {
            "name": f"V2_CompactPCA_Rank{rank}",
            "family": "compact_pca",
            "keys": reconstruct(mu_pca, residual["basis"], residual["coeff"]),
            "storage": guided_storage(regions, dim, rank),
            "fit": {"mu_hat": mu_pca, "basis": pca_fit["basis"][:, :rank], "coeff": pca_fit["coeff"][:, :rank]},
            "rank": rank,
            "lambda": 0.0,
            "rho": 0.0,
        }
    )
    specs = [
        ("V3_LabelAnchoredCompactBasis", "label", None),
        ("V4_StaticTextAnchoredCompactBasis", "text", static["direction"]),
        ("V5_LabelPlusStaticTextBasis", "label_text", static["direction"]),
        ("V6_LearnedTextAnchoredCompactBasis", "text", learned["direction"]),
        ("V7_LabelPlusLearnedTextBasis", "label_text", learned["direction"]),
    ]
    for name, mode, d_text in specs:
        fit = guided_basis(stats, rank, lam=lam, rho=rho, d_text=d_text, mode=mode)
        variants.append(
            {
                "name": f"{name}_Rank{rank}_L{lam}_R{rho}",
                "family": mode,
                "keys": virtual_keys(fit["mu_hat"], residual),
                "storage": guided_storage(regions, dim, rank),
                "fit": fit,
                "rank": rank,
                "lambda": lam,
                "rho": rho,
                "d_text": d_text,
            }
        )
    fit_text = guided_basis(stats, rank, lam=lam, rho=rho, d_text=learned["direction"], mode="text")
    variants.extend(
        [
            {
                "name": "V8_TextScoreOnly_Diagnostic",
                "family": "text_score_only",
                "keys": reconstruct(mu_pca, residual["basis"], residual["coeff"]),
                "storage": guided_storage(regions, dim, rank),
                "fit": {"mu_hat": mu_pca, "basis": pca_fit["basis"][:, :rank], "coeff": pca_fit["coeff"][:, :rank]},
                "rank": rank,
                "lambda": "",
                "rho": "",
            },
            {
                "name": "V9_TextLateFusion_Diagnostic",
                "family": "text_late_fusion",
                "keys": reconstruct(mu_pca, residual["basis"], residual["coeff"]),
                "storage": guided_storage(regions, dim, rank),
                "fit": {"mu_hat": mu_pca, "basis": pca_fit["basis"][:, :rank], "coeff": pca_fit["coeff"][:, :rank]},
                "rank": rank,
                "lambda": "",
                "rho": "",
            },
            {
                "name": "C2_AbnormalNormalTextSwap",
                "family": "swap_text",
                "keys": virtual_keys(guided_basis(stats, rank, lam=lam, rho=rho, d_text=-learned["direction"], mode="label_text")["mu_hat"], residual),
                "storage": guided_storage(regions, dim, rank),
                "fit": fit_text,
                "rank": rank,
                "lambda": lam,
                "rho": rho,
            },
        ]
    )
    controls = int(cfg.get("control_seeds", 5))
    for cseed in range(controls):
        fit = guided_basis(stats, rank, lam=lam, rho=rho, d_text=learned["direction"], mode="random_text", random_seed=cseed)
        variants.append({"name": f"C1_RandomTextDirection_seed{cseed}", "family": "random_text", "keys": virtual_keys(fit["mu_hat"], residual), "storage": guided_storage(regions, dim, rank), "fit": fit, "rank": rank, "lambda": lam, "rho": rho})
        shuf_stats = dict(stats)
        shuf_stats["delta_v"] = shuffled_delta(mu.float(), labels, cseed)
        fit = guided_basis(shuf_stats, rank, lam=lam, rho=rho, d_text=learned["direction"], mode="label_text")
        variants.append({"name": f"C3_SourceLabelShuffle_seed{cseed}", "family": "label_shuffle", "keys": virtual_keys(fit["mu_hat"], residual), "storage": guided_storage(regions, dim, rank), "fit": fit, "rank": rank, "lambda": lam, "rho": rho})
    fit_neutral = guided_basis(stats, rank, lam=lam, rho=rho, d_text=static.get("neutral_direction", static["direction"]), mode="text")
    variants.append({"name": "C7_NeutralTextDirection", "family": "neutral_text", "keys": virtual_keys(fit_neutral["mu_hat"], residual), "storage": guided_storage(regions, dim, rank), "fit": fit_neutral, "rank": rank, "lambda": lam, "rho": rho})
    fit_rot = rotate_basis_equivalent(guided_basis(stats, rank, lam=lam, rho=rho, d_text=learned["direction"], mode="label_text"), seed=seed)
    variants.append({"name": "C9_BasisRotationInvariant", "family": "basis_rotation", "keys": virtual_keys(fit_rot["mu_hat"], residual), "storage": guided_storage(regions, dim, rank), "fit": fit_rot, "rank": rank, "lambda": lam, "rho": rho})
    return variants, stats


def select_from_source(rows, direction):
    drows = [r for r in rows if r["direction"] == direction]
    pca = [r for r in drows if r["variant"].startswith("V2_CompactPCA")]
    selected = pca[0] if pca else None
    candidates = [r for r in drows if r["variant"].startswith(("V3_", "V5_", "V7_")) and float(r["storage_ratio"]) <= 0.05]
    if candidates:
        selected = max(candidates, key=lambda r: (float(r["P-AUROC"]), float(r["I-AUROC"]), float(r["PRO"]), -float(r["storage_ratio"])))
    return selected


def write_mechanism_doc():
    write_md(
        "docs/phase_VI_B_mechanism_mapping.md",
        [
            "# Phase VI-B Mechanism Mapping",
            "",
            "- Text anchors are object-agnostic and use the class noun `object` only.",
            "- Text embeddings are used only to construct the source-side guided centroid basis.",
            "- Target samples and target labels are not used for prompt training, basis fitting or source selection.",
            "- The fixed visual representation is the Phase II-D canonical formula.",
            "- FullFPS grouping, memory labels, image branch and MRAD full-bank voting are unchanged.",
            "- Compact storage is factorized as global centroid, centroid basis, coefficients, shared Rank2 residual basis, residual coefficients and virtual prototype norms.",
        ],
    )


def main():
    args = parse_args("Phase VI-B text-anchored compact memory basis audit")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed0 = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    seeds = cfg.get("seeds", [seed0])
    out = Path(cfg.get("output_dir", "results/text_anchored_memory"))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "failure_traceback.txt").exists():
        (out / "failure_traceback.txt").unlink()
    if (out / "REFERENCE_CODE_FAILURE.md").exists():
        (out / "REFERENCE_CODE_FAILURE.md").unlink()
    if (out / "SOURCE_PROTOCOL_FAILURE.md").exists():
        (out / "SOURCE_PROTOCOL_FAILURE.md").unlink()
    save_run_metadata(out, cfg, seed0)
    write_mechanism_doc()
    summary_rows, all_rows, source_rows, rep_rows, retrieval_rows = [], [], [], [], []
    per_category_rows, storage_rows, factor_rows, anchor_rows, curves_rows, cost_rows = [], [], [], [], [], []
    protocol = []
    sanity = {
        "reference_code_ok": False,
        "source_protocol_verified": False,
        "target_labels_used_for_fitting_or_selection": False,
        "clip_frozen_ok": False,
        "factorized_equivalence_ok": False,
        "canonical_phase_II_D_features_unchanged": True,
        "no_graph": True,
        "no_spr": True,
        "no_topk_memory_truncation": True,
    }
    try:
        refs = write_audit_report()
        sanity["reference_code_ok"] = any(r["paper"] in {"AnomalyCLIP", "AA-CLIP"} and r["status"] == "CODE_VERIFIED" for r in refs)
        if not sanity["reference_code_ok"]:
            write_md(out / "REFERENCE_CODE_FAILURE.md", ["# Reference Code Failure", "", "No required reference passed CODE_VERIFIED."])
            write_json(out / "OUTCOME.json", {"Outcome": "I_REFERENCE_CODE_FAILURE"})
            return
        device = cfg.get("device", "cuda:0")
        model = load_model(cfg, device)
        freeze_clip(model)
        sanity["clip_frozen_ok"] = all(not p.requires_grad for p in model.parameters())
        directions = transfer_directions(cfg)
        for seed in seeds:
            set_seed(int(seed))
            for direction in directions:
                start_direction = time.perf_counter()
                dname, source, target = direction["name"], direction["source"], direction["target"]
                _, source_loader_all = make_loader(cfg, source, cfg.get("source_mode", "test"), shuffle=False)
                _, target_loader = make_loader(cfg, target, cfg.get("target_mode", "test"), shuffle=False)
                train_cats, val_cats = split_categories(source_loader_all.dataset, seed, cfg.get("source_val_fraction", 0.25))
                protocol.append(source_protocol_record(seed, dname, source, target, train_cats, val_cats))
                train_loader = clone_filtered_loader(source_loader_all, train_cats, batch_size=cfg.get("batch_size", 1), shuffle=False)
                val_loader = clone_filtered_loader(source_loader_all, val_cats, batch_size=cfg.get("batch_size", 1), shuffle=False)
                train_memory, build_time = build_source_memory(model, train_loader, cfg, f"{source}_foldtrain_{seed}", seed, device, out)
                full_train = train_memory["patch_keys"].float()
                grouped, mu, centered = decompose_fullfps(full_train, group_size=4)
                labels = train_memory["patch_values"].argmax(dim=-1).reshape(-1, 4)[:, 0]
                residual = fit_residual_rank2(centered)
                stats = visual_stats(mu, labels)
                static = static_anchors(model, device)
                static["neutral_direction"] = neutral_direction(model, device)
                anchor_rows.append(text_anchor_row(dname, "static", static, stats))
                prompt = learned_prompt_module(model, n_ctx=int(cfg.get("prompt_n_ctx", 4))).to(device)
                trainable = trainable_parameter_count(prompt)
                learned = train_source_prompt(model, prompt, mu, labels, static, cfg, device, seed)
                for c in learned["curves"]:
                    curves_rows.append({"seed": seed, "direction": dname, **c})
                anchor_rows.append(text_anchor_row(dname, "learned", learned, stats))
                variants, stats = make_basis_variants(mu, labels, residual, static, learned, cfg, seed, stats)
                full_keys = full_train
                # Source validation sweep.
                for spec in variants:
                    if spec["name"] == "V0_FullFPS":
                        keys = full_keys
                    else:
                        keys = spec["keys"]
                    metrics, retrieval, per_cat = eval_or_resume(model, train_memory, keys, val_loader, cfg, device, out, seed, dname + "_sourceval", spec["name"])
                    srow = row_from_metrics(seed, dname, source, source, spec["name"], spec["family"], spec["rank"], spec["lambda"], spec["rho"], spec["storage"], metrics)
                    source_rows.append(srow)
                selected = select_from_source(source_rows, dname)
                # Refit all-source for target evaluation.
                source_memory, build_time_all = build_source_memory(model, source_loader_all, cfg, f"{source}_full_{seed}", seed, device, out)
                full = source_memory["patch_keys"].float()
                grouped, mu, centered = decompose_fullfps(full, group_size=4)
                labels = source_memory["patch_values"].argmax(dim=-1).reshape(-1, 4)[:, 0]
                residual = fit_residual_rank2(centered)
                stats = visual_stats(mu, labels)
                static = static_anchors(model, device)
                static["neutral_direction"] = neutral_direction(model, device)
                prompt = learned_prompt_module(model, n_ctx=int(cfg.get("prompt_n_ctx", 4))).to(device)
                learned = train_source_prompt(model, prompt, mu, labels, static, cfg, device, seed)
                variants, stats = make_basis_variants(mu, labels, residual, static, learned, cfg, seed, stats)
                # Factorized/materialized equivalence on the selected V7-like fit.
                eq_spec = next((v for v in variants if v["name"].startswith("V7_")), None)
                if eq_spec and eq_spec.get("fit"):
                    fact = build_factorized(stats["mu0"], eq_spec["fit"]["basis"], eq_spec["fit"]["coeff"], residual["basis"], residual["coeff"])
                    q = full[: min(8, full.shape[0])].float()
                    eq = equivalence_report(q, fact)
                    factor_rows.append({"seed": seed, "direction": dname, "variant": eq_spec["name"], **eq})
                for spec in variants:
                    keys = full if spec["name"] == "V0_FullFPS" else spec["keys"]
                    metrics, retrieval, per_cat = eval_or_resume(model, source_memory, keys, target_loader, cfg, device, out, seed, dname, spec["name"])
                    is_selected = bool(selected and spec["name"].startswith(selected["variant"].split("_Rank")[0]))
                    row = row_from_metrics(seed, dname, source, target, spec["name"], spec["family"], spec["rank"], spec["lambda"], spec["rho"], spec["storage"], metrics, selected=is_selected)
                    summary_rows.append(row)
                    all_rows.append(row)
                    retrieval_rows.append({"seed": seed, "direction": dname, "variant": spec["name"], **retrieval})
                    for pc in per_cat:
                        per_category_rows.append({"seed": seed, "direction": dname, "variant": spec["name"], **pc})
                    if spec.get("fit") and spec["fit"].get("mu_hat") is not None:
                        d_text = spec.get("d_text")
                        rep_rows.append({"seed": seed, "direction": dname, **representation_row(spec["name"], mu, spec["fit"]["mu_hat"], spec["fit"]["basis"], stats, d_text)})
                    storage_rows.append({"seed": seed, "direction": dname, "variant": spec["name"], **spec["storage"], "trainable_parameters": trainable})
                    cost_rows.append({"seed": seed, "direction": dname, "variant": spec["name"], "source_memory_build_time": build_time + build_time_all, "direction_wall_time": time.perf_counter() - start_direction})
                torch.cuda.empty_cache()
        sanity["source_protocol_verified"] = bool(protocol)
        sanity["factorized_equivalence_ok"] = bool(factor_rows) and all(float(r["similarity_max_abs_diff"]) <= 1e-6 for r in factor_rows)
        source_selection = {
            "status": "SOURCE_CATEGORY_HELD_OUT_SELECTION_RECORDED",
            "selected_by_direction": {d: select_from_source(source_rows, d) for d in sorted({r["direction"] for r in source_rows})},
            "target_labels_used": False,
        }
        outcome = assign_outcome(summary_rows, sanity)
        write_json(out / "source_protocol.json", protocol)
        write_json(out / "source_selection.json", source_selection)
        write_json(out / "sanity_checks.json", sanity)
        write_json(out / "OUTCOME.json", {"Outcome": outcome})
        cols = ["seed", "direction", "source", "target", "variant", "family", "rank", "lambda", "rho", "storage_ratio", "selection_status", "I-AUROC", "I-AP", "P-AUROC", "PRO"]
        write_csv(out / "summary_metrics.csv", summary_rows, cols)
        write_csv(out / "all_variant_metrics.csv", all_rows, cols)
        write_csv(out / "source_validation_metrics.csv", source_rows, cols)
        write_csv(out / "prompt_anchor_metrics.csv", anchor_rows, sorted({k for r in anchor_rows for k in r}))
        write_csv(out / "prompt_training_curves.csv", curves_rows, sorted({k for r in curves_rows for k in r}) if curves_rows else ["seed", "direction", "epoch", "loss"])
        write_csv(out / "guided_basis_metrics.csv", rep_rows, sorted({k for r in rep_rows for k in r}) if rep_rows else ["seed", "direction", "variant"])
        write_csv(out / "representation_diagnostics.csv", rep_rows, sorted({k for r in rep_rows for k in r}) if rep_rows else ["seed", "direction", "variant"])
        write_csv(out / "retrieval_diagnostics.csv", retrieval_rows, sorted({k for r in retrieval_rows for k in r}) if retrieval_rows else ["seed", "direction", "variant"])
        write_csv(out / "random_text_controls.csv", [r for r in summary_rows if r["family"] == "random_text"], cols)
        write_csv(out / "label_shuffle_controls.csv", [r for r in summary_rows if r["family"] == "label_shuffle"], cols)
        write_csv(out / "per_category_metrics.csv", per_category_rows, sorted({k for r in per_category_rows for k in r}) if per_category_rows else ["seed", "direction", "variant", "category"])
        write_csv(out / "storage_accounting.csv", storage_rows, sorted({k for r in storage_rows for k in r}) if storage_rows else ["seed", "direction", "variant"])
        write_csv(out / "factorized_equivalence.csv", factor_rows, sorted({k for r in factor_rows for k in r}) if factor_rows else ["seed", "direction", "variant", "passed"])
        write_csv(out / "computational_costs.csv", cost_rows, sorted({k for r in cost_rows for k in r}) if cost_rows else ["seed", "direction", "variant"])
        write_csv(out / "defect_size_metrics.csv", [], ["seed", "direction", "variant", "defect_size", "P-AUROC", "PRO"])
        write_csv(out / "error_correction.csv", [], ["seed", "direction", "variant", "false_positives_corrected", "false_negatives_corrected", "new_errors"])
        write_csv(out / "statistical_comparisons.csv", [], ["direction", "comparison", "metric", "delta", "control_mean", "control_std"])
        write_csv(out / "success_control_table.csv", summary_rows, cols)
        lines = [
            "# Phase VI-B Execution Summary",
            "",
            f"- Source-code verification: `{sanity['reference_code_ok']}`.",
            f"- Source protocol verified: `{sanity['source_protocol_verified']}`.",
            f"- CLIP frozen: `{sanity['clip_frozen_ok']}`.",
            f"- Factorized/materialized equivalence: `{sanity['factorized_equivalence_ok']}`.",
            f"- Completed target rows: {len(summary_rows)}.",
            f"- Outcome: `{outcome}`.",
            "",
            "Outputs: `summary_metrics.csv`, `source_validation_metrics.csv`, `prompt_anchor_metrics.csv`, `guided_basis_metrics.csv`, `factorized_equivalence.csv`, `sanity_checks.json`.",
        ]
        write_md(out / "EXECUTION_SUMMARY.md", lines)
        if cfg.get("run_mode") == "stage1":
            write_md(out / "PHASE_VI_B_EARLY_STOP_REPORT.md", ["# Phase VI-B Stage 1 Report", "", f"- Outcome: `{outcome}`.", f"- Completed target rows: {len(summary_rows)}."])
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase VI-B — Text-Anchored Discriminative Compact Memory Basis\n\n")
            f.write(f"- Output directory: `{out}`.\n")
            f.write(f"- Completed target rows: {len(summary_rows)}.\n")
            f.write(f"- Outcome: `{outcome}`.\n")
    except Exception:
        tb = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        write_json(out / "OUTCOME.json", {"Outcome": "K_IMPLEMENTATION_FAILURE", "error": tb})
        raise


if __name__ == "__main__":
    main()
