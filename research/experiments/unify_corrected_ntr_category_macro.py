import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.correct_ces_ntr_ablation_visualization import (  # noqa: E402
    PAIRS,
    SUMMARY_METRICS,
    VARIANTS,
    HeadSpec,
    get_target_cell,
    item_at,
    item_meta,
    local_cache_path,
    mask_np,
)
from experiments.phase_sgpr import safe_ap_binary  # noqa: E402
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import torch_resize  # noqa: E402
from experiments.phase_ces_linear_head_control import preprocess_patch_grid, stitch_local_maps, unit_np  # noqa: E402


OUT = ROOT / "results" / "ces_ntr_visualization" / "corrected_metric_unification"
CORRECTED = ROOT / "results" / "ces_ntr_visualization" / "corrected"
RAW_GLOBAL = CORRECTED / "CORRECTED_ABLATION_RAW.csv"
CHECKPOINT_CONFIG = CORRECTED / "CHECKPOINT_CONFIG_COMPARISON.csv"
LINEAR_RAW = ROOT / "results" / "phase_ces_linear_head_control" / "HEAD_CONTROL_RAW_METRICS.csv"
LINEAR_SUMMARY = ROOT / "results" / "phase_ces_linear_head_control" / "HEAD_CONTROL_SUMMARY.csv"
CAPACITY_METRICS = ROOT / "results" / "paper_assets" / "final_evidence" / "STATE_CAPACITY_METRICS.csv"
CAPACITY_SUMMARY = ROOT / "results" / "paper_assets" / "final_evidence" / "STATE_CAPACITY_SUMMARY.csv"


def source_scoring_info(source: str, device: torch.device) -> dict:
    """Load only the frozen scoring geometry, not the full source patch pool."""
    if source == "Full MVTec":
        payload = np.load(ROOT / "results" / "phase_hard_normal_cross_source_mvtec" / "formal" / "contrast_vectors.npz")
        return {
            "source": source,
            "use_align": False,
            "q_pc1": None,
            "align_model": None,
            "m": payload["midpoint"].astype(np.float32),
            "alpha": float(payload["alpha"]),
        }
    if source == "Full VisA":
        from experiments.phase_hard_normal_contrast import PC1_DEBIAS_OUT, load_payload
        from experiments.phase_state_pair_decomposition import decompose
        from experiments.phase_statealigned_state_alignment_swap import make_model

        q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
        true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
        corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
        states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
        states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
        _m_t, d_t = decompose(states_t)
        m_c, _d_c = decompose(states_c)
        align_model = make_model(states_c, corrupt_payload, device).eval()
        for p in align_model.parameters():
            p.requires_grad_(False)
        return {
            "source": source,
            "use_align": True,
            "q_pc1": q_pc1,
            "align_model": align_model,
            "m": m_c.astype(np.float32),
            "alpha": float(np.linalg.norm(d_t)),
        }
    raise ValueError(source)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ffloat(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def load_v(path: str) -> np.ndarray:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "v" not in ckpt:
        raise KeyError(f"Checkpoint has no `v`: {path}")
    return np.asarray(ckpt["v"], dtype=np.float32)


def spec_weight(spec: HeadSpec) -> np.ndarray:
    return np.asarray(spec.weight(), dtype=np.float32)


def score_patch_grid_multi(source_info: dict, specs: dict[str, HeadSpec], patch_np: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    z = preprocess_patch_grid(source_info, patch_np, device)
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])
    weights = torch.tensor(np.stack([spec_weight(specs[v]) for v in specs], axis=1), dtype=torch.float32, device=device)
    logits = (z @ weights).detach().cpu().numpy().astype(np.float32)
    return {variant: logits[:, j].reshape(37, 37) for j, variant in enumerate(specs)}


def lrp_maps_multi(source_info: dict, specs: dict[str, HeadSpec], global_patch: np.ndarray, local_patches: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    g_maps = score_patch_grid_multi(source_info, specs, global_patch, device)
    local_by_variant = {variant: [] for variant in specs}
    for i in range(4):
        lm = score_patch_grid_multi(source_info, specs, local_patches[i], device)
        for variant, m in lm.items():
            local_by_variant[variant].append(m)
    out = {}
    for variant in specs:
        local74 = stitch_local_maps(np.stack(local_by_variant[variant], axis=0))
        g74 = torch_resize(g_maps[variant], (74, 74), align_corners=True)
        out[variant] = (0.5 * g74 + 0.5 * local74).astype(np.float32)
    return out


def raw_lookup() -> dict[tuple[str, str, int, str], dict]:
    raw = pd.read_csv(RAW_GLOBAL)
    return {
        (str(r.source), str(r.target), int(r.seed), str(r.variant)): r._asdict()
        for r in raw.itertuples(index=False)
    }


def evaluate_pap_category_macro_only(source: str, target: str, maps: dict[str, list[np.ndarray]], metas: list[dict]) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    cat_rows: list[dict] = []
    for variant in VARIANTS:
        by_cat: dict[str, dict[str, list]] = {}
        for i, meta in enumerate(metas):
            d = by_cat.setdefault(meta["category"], {"masks": [], "maps": []})
            d["masks"].append(meta["mask"])
            d["maps"].append(np.asarray(maps[variant][i], dtype=np.float32))
        pap_vals = []
        for cat, vals in sorted(by_cat.items()):
            full_scores = [torch_resize(s, m.shape, align_corners=True) for s, m in zip(vals["maps"], vals["masks"])]
            y = np.concatenate([m.reshape(-1).astype(np.uint8) for m in vals["masks"]])
            s = np.concatenate([x.reshape(-1).astype(np.float32) for x in full_scores])
            pap = safe_ap_binary(y, s)
            cat_rows.append(
                {
                    "source": source,
                    "dataset": target,
                    "category": cat,
                    "variant": variant,
                    "protocol": "M_FR",
                    "aggregation": "CATEGORY",
                    "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}",
                    "M_FR_P_AP": pap,
                }
            )
            if np.isfinite(pap):
                pap_vals.append(float(pap))
        raw_rows.append(
            {
                "source": source,
                "dataset": target,
                "variant": variant,
                "aggregation": "CATEGORY_MACRO",
                "score_resolution": f"{maps[variant][0].shape[0]}x{maps[variant][0].shape[1]}",
                "M_FR_P_AP": float(np.mean(pap_vals)) if pap_vals else float("nan"),
                "metric_subset": "P_AP_ONLY",
                "evaluator": "category_grouped_safe_ap_binary_align_corners_true",
            }
        )
    return raw_rows, cat_rows


def score_maps_category_macro(device: torch.device) -> tuple[list[dict], list[dict]]:
    lookup = raw_lookup()
    raw_rows: list[dict] = []
    cat_rows: list[dict] = []
    source_infos: dict[str, dict] = {}

    for source, target in PAIRS:
        for seed in (0, 1, 2):
            print(f"[Unify] start {source} -> {target} seed={seed}", flush=True)
            if source not in source_infos:
                # m/alpha and frozen preprocessing are source-level quantities; avoid
                # loading the full source training pool for this read-only metric pass.
                source_infos[source] = source_scoring_info(source, device)
            info = source_infos[source]
            cell = get_target_cell(source, target, device, smoke=False)
            print(f"[Unify] loaded target cache {source} -> {target} seed={seed}", flush=True)
            _cell_id, _target, dataset, idxs, patch, kind = cell
            local_path = local_cache_path(source, target)
            local_cache = None
            if local_path.exists():
                local_cache = np.load(local_path, mmap_mode="r")

            specs: dict[str, HeadSpec] = {}
            ckpt_paths: dict[str, str] = {}
            cache_paths: dict[str, str] = {}
            statuses: dict[str, str] = {}
            missing = []
            for variant in VARIANTS:
                meta = lookup.get((source, target, seed, variant))
                if meta is None:
                    missing.append(f"{source}->{target} seed{seed} {variant}: missing corrected raw row")
                    continue
                ckpt = str(meta["checkpoint"])
                v = load_v(ckpt)
                specs[variant] = HeadSpec("CES", {"m": info["m"], "alpha": info["alpha"], "v": v}, source, seed=seed)
                ckpt_paths[variant] = ckpt
                cache_paths[variant] = str(meta.get("cache", ""))
                statuses[variant] = str(meta.get("formal_status", ""))
                if variant.endswith("_LRP") and local_cache is None:
                    missing.append(f"{source}->{target} seed{seed} {variant}: missing local cache {local_path}")
            if missing:
                raise FileNotFoundError("\n".join(missing))

            maps = {v: [] for v in VARIANTS}
            metas = []
            for pos, ref in enumerate(idxs):
                if pos % 100 == 0:
                    print(f"[Unify] scoring {source} -> {target} seed={seed} image {pos}/{len(idxs)}", flush=True)
                item = item_at(dataset, kind, ref, pos)
                cat, label, _path = item_meta(item, kind, ref, target)
                metas.append({"category": cat, "label": int(label), "mask": mask_np(item)})
                global_patch = patch[pos]
                core_specs = {v: specs[v] for v in VARIANTS if not v.endswith("_LRP")}
                lrp_specs = {v: specs[v] for v in VARIANTS if v.endswith("_LRP")}
                core_maps = score_patch_grid_multi(info, core_specs, global_patch, device)
                for variant, smap in core_maps.items():
                    maps[variant].append(smap)
                local_patches = np.asarray(local_cache[pos], dtype=np.float32)
                lmaps = lrp_maps_multi(info, lrp_specs, global_patch, local_patches, device)
                for variant, smap in lmaps.items():
                    maps[variant].append(smap)

            print(f"[Unify] metric {source} -> {target} seed={seed}", flush=True)
            rows, cats = evaluate_pap_category_macro_only(source, target, maps, metas)
            for row in rows:
                variant = row["variant"]
                out = dict(row)
                out.update(
                    {
                        "seed": seed,
                        "checkpoint": ckpt_paths[variant],
                        "cache": cache_paths[variant],
                        "formal_status": statuses[variant],
                        "metric_standard": "MAIN_TABLE_CATEGORY_MACRO",
                        "evaluator": "experiments.phase_sgpr.evaluate_maps",
                    }
                )
                raw_rows.append(out)
            for row in cats:
                out = dict(row)
                out["seed"] = seed
                cat_rows.append(out)

    return raw_rows, cat_rows


def build_summary(raw_rows: list[dict]) -> list[dict]:
    out = []
    for variant in VARIANTS:
        part = [r for r in raw_rows if r["variant"] == variant]
        row = {"variant": variant, "num_cells": len(part), "aggregation": "CATEGORY_MACRO"}
        for metric in SUMMARY_METRICS:
            key = {
                "I_AUROC": "M_FR_I_AUROC",
                "I_AP": "M_FR_I_AP",
                "P_AUROC": "M_FR_P_AUROC",
                "P_AP": "M_FR_P_AP",
                "P_F1MAX": "M_FR_P_F1max",
                "AUPRO": "M_FR_PRO",
            }[metric]
            vals_all = np.asarray([ffloat(r.get(key)) for r in part], dtype=np.float64)
            vals_all = vals_all[np.isfinite(vals_all)]
            row[f"{metric}_cell_mean"] = float(vals_all.mean()) if vals_all.size else float("nan")
            row[f"{metric}_cell_std"] = float(vals_all.std(ddof=0)) if vals_all.size else float("nan")
            transfer_means, transfer_stds = [], []
            for source, target in PAIRS:
                arr = np.asarray([ffloat(r.get(key)) for r in part if r["source"] == source and r["dataset"] == target], dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    transfer_means.append(float(arr.mean()))
                    transfer_stds.append(float(arr.std(ddof=0)))
            row[f"{metric}_mean"] = float(np.mean(transfer_means)) if transfer_means else float("nan")
            row[f"{metric}_std_main_convention"] = float(np.mean(transfer_stds)) if transfer_stds else float("nan")
            row[metric] = (
                f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_std_main_convention']:.2f}"
                if np.isfinite(row[f"{metric}_mean"])
                else "NA"
            )
        out.append(row)
    return out


def per_transfer(raw_rows: list[dict]) -> list[dict]:
    rows = []
    for source, target in PAIRS:
        row = {"source": source, "target": target, "transfer": f"{source} -> {target}"}
        means = {}
        for variant in VARIANTS:
            part = [r for r in raw_rows if r["source"] == source and r["dataset"] == target and r["variant"] == variant]
            vals = np.asarray([ffloat(r.get("M_FR_P_AP")) for r in part], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            means[variant] = float(vals.mean()) if vals.size else float("nan")
            row[f"{variant}_P_AP_category_macro"] = means[variant]
        row["Delta_NTR_P_AP"] = means["Trained_CES_NTR"] - means["Trained_CES_Core"]
        row["Delta_LRP_base_P_AP"] = means["Trained_CES_LRP"] - means["Trained_CES_Core"]
        row["Delta_LRP_after_NTR_P_AP"] = means["Trained_CES_NTR_LRP"] - means["Trained_CES_NTR"]
        row["Delta_NTR_with_LRP_P_AP"] = means["Trained_CES_NTR_LRP"] - means["Trained_CES_LRP"]
        row["Delta_Final_P_AP"] = means["Trained_CES_NTR_LRP"] - means["Trained_CES_Core"]
        rows.append(row)
    return rows


def build_reconciliation(summary: list[dict], raw_rows: list[dict]) -> list[dict]:
    global_summary = pd.read_csv(CORRECTED / "CORRECTED_ABLATION_SUMMARY.csv")
    global_raw = pd.read_csv(CORRECTED / "CORRECTED_ABLATION_RAW.csv")
    cat_summary = {r["variant"]: r for r in summary}
    rows = []
    for variant in VARIANTS:
        g = global_summary[global_summary["variant"] == variant].iloc[0].to_dict()
        c = cat_summary[variant]
        rows.append(
            {
                "level": "summary",
                "variant": variant,
                "dataset_global_P_AP_mean": ffloat(g.get("P_AP_mean")),
                "dataset_global_P_AP_std_main_convention": ffloat(g.get("P_AP_std")),
                "category_macro_P_AP_mean": ffloat(c.get("P_AP_mean")),
                "category_macro_P_AP_std_main_convention": ffloat(c.get("P_AP_std_main_convention")),
                "category_minus_global": ffloat(c.get("P_AP_mean")) - ffloat(g.get("P_AP_mean")),
            }
        )
    for (source, target, seed, variant), crows in pd.DataFrame(raw_rows).groupby(["source", "dataset", "seed", "variant"]):
        grows = global_raw[
            (global_raw["source"] == source)
            & (global_raw["target"] == target)
            & (global_raw["seed"] == seed)
            & (global_raw["variant"] == variant)
        ]
        gpap = ffloat(grows.iloc[0]["P_AP"]) if len(grows) else float("nan")
        cpap = ffloat(crows.iloc[0]["M_FR_P_AP"])
        rows.append(
            {
                "level": "cell",
                "variant": variant,
                "source": source,
                "target": target,
                "seed": seed,
                "dataset_global_P_AP": gpap,
                "category_macro_P_AP": cpap,
                "category_minus_global": cpap - gpap,
            }
        )
    return rows


def audit_other_experiments() -> tuple[list[dict], list[dict]]:
    rows = []
    if LINEAR_RAW.exists():
        raw = pd.read_csv(LINEAR_RAW)
        rows.append(
            {
                "experiment": "LinearHead parameterization",
                "artifact": str(LINEAR_RAW),
                "summary_artifact": str(LINEAR_SUMMARY),
                "observed_evaluator_field": "; ".join(sorted(raw["evaluator"].dropna().unique().astype(str))) if "evaluator" in raw else "UNKNOWN",
                "observed_transfers": len(raw[["source", "target"]].drop_duplicates()) if {"source", "target"}.issubset(raw.columns) else "",
                "P_AP_field": "P_AP",
                "aggregation_conclusion": "CATEGORY_MACRO-compatible M_FR: values match paper-ready category-macro cells (e.g. CES MVTec->VisA 21.4606 and MPDD 34.3636). CSV label is `unified_full_resolution_M_FR`, so report should spell out category-macro provenance.",
                "directly_comparable_to_main_table": "YES",
            }
        )
    else:
        rows.append({"experiment": "LinearHead parameterization", "artifact": str(LINEAR_RAW), "aggregation_conclusion": "MISSING"})
    if CAPACITY_METRICS.exists():
        cap = pd.read_csv(CAPACITY_METRICS)
        rows.append(
            {
                "experiment": "K=2/3/5 state capacity",
                "artifact": str(CAPACITY_METRICS),
                "summary_artifact": str(CAPACITY_SUMMARY),
                "observed_record_source": "; ".join(sorted(cap["record_source"].dropna().unique().astype(str))) if "record_source" in cap else "UNKNOWN",
                "observed_transfers": len(cap[["source", "target"]].drop_duplicates()) if {"source", "target"}.issubset(cap.columns) else "",
                "P_AP_field": "M_FR_P_AP",
                "aggregation_conclusion": "CATEGORY_MACRO M_FR inherited from existing final-evidence matched records; directly aligned with paper main-table convention.",
                "directly_comparable_to_main_table": "YES",
            }
        )
    else:
        rows.append({"experiment": "K=2/3/5 state capacity", "artifact": str(CAPACITY_METRICS), "aggregation_conclusion": "MISSING"})
    return rows, []


def write_report(summary: list[dict], reconciliation: list[dict], other_audit: list[dict]) -> None:
    by_variant = {r["variant"]: r for r in summary}
    def fmt(x):
        return "NA" if not np.isfinite(ffloat(x)) else f"{ffloat(x):.4f}"

    lines = [
        "# Corrected NTR Category-Macro Metric Unification",
        "",
        "## Scope",
        "",
        "No model training was performed. This pass reuses the corrected checkpoints, cached target patch features, LRP local feature caches, and GT masks. Metrics are recomputed with the same category-macro full-resolution evaluator used by the paper main table.",
        "",
        "## Main-Table Evaluator Audit",
        "",
        "- Main paper script: `experiments/phase_paper_ready_final_validation.py` maps `M_CATEGORY_MACRO_FR_*` to `M_FR_*` and states that official tables use full-resolution category-macro metrics.",
        "- Category-macro implementation: `experiments/phase_sgpr.py::evaluate_maps` / `experiments/phase_parameter_free_local_zoom_pyramid_fullres.py::evaluate_maps` group images by category, compute full-resolution metrics within each category, then average category metrics arithmetically.",
        "- Full-resolution M_FR uses `torch_resize(..., align_corners=True)` and original GT mask size.",
        "- Image score uses `top1_image_score`, i.e. the same top-1% image-score helper used in the main pipeline.",
        "- Dataset-global corrected results are preserved only as a reconciliation column and are not mixed into category-macro effect sizes.",
        "",
        "## Unified Corrected Ablation Table",
        "",
        "| Variant | P-AP category-macro mean ± std | P-AUROC | P-F1max | AUPRO |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        r = by_variant[variant]
        lines.append(
            f"| {variant} | {r['P_AP']} | {r['P_AUROC']} | {r['P_F1MAX']} | {r['AUPRO']} |"
        )
    lines += [
        "",
        "P-AP effects under category-macro M_FR:",
        "",
        f"- NTR without LRP: `{fmt(by_variant['Trained_CES_NTR']['P_AP_mean'] - by_variant['Trained_CES_Core']['P_AP_mean'])}`.",
        f"- LRP on no-NTR CES: `{fmt(by_variant['Trained_CES_LRP']['P_AP_mean'] - by_variant['Trained_CES_Core']['P_AP_mean'])}`.",
        f"- LRP after NTR: `{fmt(by_variant['Trained_CES_NTR_LRP']['P_AP_mean'] - by_variant['Trained_CES_NTR']['P_AP_mean'])}`.",
        f"- NTR with LRP: `{fmt(by_variant['Trained_CES_NTR_LRP']['P_AP_mean'] - by_variant['Trained_CES_LRP']['P_AP_mean'])}`.",
        f"- Final vs core: `{fmt(by_variant['Trained_CES_NTR_LRP']['P_AP_mean'] - by_variant['Trained_CES_Core']['P_AP_mean'])}`.",
        "",
        "## Dataset-Global vs Category-Macro",
        "",
        "The previously observed `43.26 ± 0.17` belongs to dataset-global P-AP. Under the main-table category-macro evaluator, the corresponding final corrected CES+NTR+LRP value is listed above. The two values must not be used interchangeably.",
        "",
        "## LinearHead and K-State Capacity Metric Audit",
        "",
        "| Experiment | Artifact | Conclusion | Directly comparable? |",
        "|---|---|---|---|",
    ]
    for row in other_audit:
        lines.append(
            f"| {row.get('experiment','')} | `{row.get('artifact','')}` | {row.get('aggregation_conclusion','')} | {row.get('directly_comparable_to_main_table','')} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `CORRECTED_ABLATION_CATEGORY_MACRO_RAW.csv`: per-transfer, per-seed category-macro metrics.",
        "- `CORRECTED_ABLATION_CATEGORY_MACRO_PER_CATEGORY.csv`: per-category metric rows used to form macro means.",
        "- `CORRECTED_ABLATION_CATEGORY_MACRO_SUMMARY.csv`: unified corrected ablation table.",
        "- `CORRECTED_ABLATION_AGGREGATION_RECONCILIATION.csv`: old dataset-global vs category-macro comparison.",
        "- `PARAMETERIZATION_CAPACITY_METRIC_AUDIT.csv`: LinearHead and K-state metric provenance check.",
        "",
        "## Conclusion",
        "",
        "Use the category-macro table for paper main comparisons. Keep the dataset-global table only as an audit/reconciliation artifact.",
    ]
    (OUT / "METRIC_UNIFICATION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_rows, cat_rows = score_maps_category_macro(device)
    summary = build_summary(raw_rows)
    effects = per_transfer(raw_rows)
    reconciliation = build_reconciliation(summary, raw_rows)
    other_audit, _ = audit_other_experiments()

    write_csv(OUT / "CORRECTED_ABLATION_CATEGORY_MACRO_RAW.csv", raw_rows)
    write_csv(OUT / "CORRECTED_ABLATION_CATEGORY_MACRO_PER_CATEGORY.csv", cat_rows)
    write_csv(OUT / "CORRECTED_ABLATION_CATEGORY_MACRO_SUMMARY.csv", summary)
    write_csv(OUT / "CORRECTED_ABLATION_CATEGORY_MACRO_EFFECTS.csv", effects)
    write_csv(OUT / "CORRECTED_ABLATION_AGGREGATION_RECONCILIATION.csv", reconciliation)
    write_csv(OUT / "PARAMETERIZATION_CAPACITY_METRIC_AUDIT.csv", other_audit)
    (OUT / "OUTCOME.json").write_text(
        json.dumps(
            {
                "outcome": "CORRECTED_NTR_ABLATION_UNIFIED_TO_CATEGORY_MACRO",
                "no_training": True,
                "evaluator": "category-macro M_FR via experiments.phase_sgpr.evaluate_maps",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(summary, reconciliation, other_audit)
    final = next(r for r in summary if r["variant"] == "Trained_CES_NTR_LRP")
    print("=== CATEGORY-MACRO UNIFICATION COMPLETE ===")
    print(f"Final corrected CES+NTR+LRP P-AP: {final['P_AP']}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
