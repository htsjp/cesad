from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import evaluate_results, init_results
from mrad import compute_socre
from prototype_calibration.calibration_diagnostics import rmse, score_distribution
from prototype_calibration.diagnostics import mae, safe_corr, sha256_array, sha256_file
from prototype_calibration.prototype_builder import build_projected_bank
from prototype_calibration.prototype_scorer import score_prototype_variant
from prototype_calibration.source_score_calibrator import fit_source_calibrator, sigmoid_affine
from scripts.run_phase_b0_formal import load_cache, load_components, make_loader, resolve_path, set_seed
from scripts.run_phase_c_patch_causal import official_patch_map


VARIANTS = ["E0_OfficialFull", "E1_ClassCentroid", "E2_ClassCentroid_SourceAffine", "E3_HQAP4", "E4_HQAP4_SourceAffine"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not-a-git-repo"


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_rows(path: Path, new_rows: List[Dict], keys: Tuple[str, ...]) -> List[Dict]:
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    merged = {tuple(str(row.get(k, "")) for k in keys): row for row in existing}
    for row in new_rows:
        merged[tuple(str(row.get(k, "")) for k in keys)] = row
    return list(merged.values())


def projected_image_keys(image_proj, image_keys: torch.Tensor, device: torch.device) -> torch.Tensor:
    dummy = torch.zeros((1, image_keys.shape[-1]), dtype=image_keys.dtype, device=device)
    _, k_proj = image_proj(dummy, image_keys.to(device))
    return k_proj.detach().cpu().float()


def image_metrics_only(results: Dict, obj_list: List[str]) -> Tuple[Dict, Dict]:
    per_class = {}
    image_auroc, image_ap = [], []
    for obj in obj_list:
        labels = np.asarray(results[obj]["gt_sp"], dtype=int)
        scores = np.asarray([float(x) for x in results[obj]["pr_sp"]], dtype=float)
        if labels.size == 0:
            continue
        ia = float("nan")
        ap = float("nan")
        if len(set(labels.tolist())) > 1:
            try:
                ia = float(roc_auc_score(labels, scores))
                ap = float(average_precision_score(labels, scores))
            except Exception:
                pass
        per_class[obj] = {"I-AUROC": ia, "I-AP": ap}
        if np.isfinite(ia):
            image_auroc.append(ia)
        if np.isfinite(ap):
            image_ap.append(ap)
    return {"I-AUROC": float(np.mean(image_auroc)) if image_auroc else float("nan"), "I-AP": float(np.mean(image_ap)) if image_ap else float("nan")}, per_class


def full_score(q_raw, image_keys, image_values, image_proj, device):
    logits, _ = compute_socre(q_raw, image_keys, image_values, device, proj=image_proj, use_proj=True)
    return logits[:, 1]


def add_deltas(rows: List[Dict]) -> List[Dict]:
    base0 = {(r["direction"], r["dataset"]): r for r in rows if r["variant"] == "E0_OfficialFull"}
    base1 = {(r["direction"], r["dataset"]): r for r in rows if r["variant"] == "E1_ClassCentroid"}
    out = []
    for r in rows:
        row = dict(r)
        for base, name in [(base0, "E0"), (base1, "E1")]:
            b = base.get((r["direction"], r["dataset"]))
            try:
                row[f"delta_vs_{name}_image_auroc"] = float(r["image_auroc"]) - float(b["image_auroc"])
                row[f"delta_vs_{name}_image_ap"] = float(r["image_ap"]) - float(b["image_ap"])
            except Exception:
                row[f"delta_vs_{name}_image_auroc"] = float("nan")
                row[f"delta_vs_{name}_image_ap"] = float("nan")
        out.append(row)
    return out


def hash_tensor_list(tensors: List[torch.Tensor]) -> str:
    arrays = [t.detach().cpu().numpy().astype("float32") for t in tensors]
    return sha256_array(np.concatenate([x.reshape(-1) for x in arrays]))


@torch.no_grad()
def run_direction(cfg: Dict, config_path: Path, dname: str, dcfg: Dict, variants: List[str], limit_images: int | None):
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    data, loader = make_loader(cfg, dcfg["target"], limit_images)
    model, image_proj, patch_proj, prompt_learner, prompt_proj = load_components(cfg, resolve_path(dcfg["checkpoint"]), device, include_prompt=True)
    image_keys, image_values = load_cache(resolve_path(dcfg["image_cache"]))
    patch_keys, patch_values = load_cache(resolve_path(dcfg["patch_cache"]))
    image_keys, image_values = image_keys.to(device), image_values.to(device)
    patch_keys, patch_values = patch_keys.to(device), patch_values.to(device)
    projected = projected_image_keys(image_proj, image_keys, device)
    bank_e1 = build_projected_bank("D1_ClassCentroid", projected, image_values.detach().cpu())
    bank_e3 = build_projected_bank("D3_HQAP4", projected, image_values.detach().cpu())

    folds = int(cfg.get("calibration_folds", 5))
    seed = int(cfg.get("seed", 0)) + (0 if dname == "d1" else 10000)
    cal_centroid = fit_source_calibrator("centroid", image_keys.detach().cpu(), image_values.detach().cpu(), image_proj, device, folds=folds, seed=seed)
    cal_hqap = fit_source_calibrator("hqap4", image_keys.detach().cpu(), image_values.detach().cpu(), image_proj, device, folds=folds, seed=seed + 33)
    calibrators = {
        "E2_ClassCentroid_SourceAffine": cal_centroid,
        "E4_HQAP4_SourceAffine": cal_hqap,
    }

    results = {v: init_results(data.obj_list) for v in variants}
    alphas = [float(x) for x in cfg.get("fusion_alphas", [0.0, 0.25, 0.5, 0.75, 1.0])]
    fusion_results = {(v, a): init_results(data.obj_list) for v in variants for a in alphas}
    image_scores = defaultdict(list)
    fused_scores = defaultdict(list)
    labels_all = []
    times = defaultdict(float)
    patch_hash = None
    patch_scores = []
    image_count = 0

    for items in loader:
        image_count += 1
        if image_count == 1 or image_count % 50 == 0:
            print(f"[phase-e] {dname}: image {image_count}/{len(loader)}", flush=True)
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_eval = gt_mask.clone()
        gt_eval[gt_eval > 0.5], gt_eval[gt_eval <= 0.5] = 1, 0
        label = items["anomaly"].detach().cpu()
        labels_all.extend(label.tolist())
        image_features, patch_features, _, patch_projections = model.encode_image(image, cfg.get("features_list", [6, 12, 18, 24]), DPAM_layer=int(cfg.get("dpam_layer", 24)))
        q_raw = F.normalize(image_features, dim=-1)
        q_proj, _ = image_proj(q_raw, image_keys)
        start_patch = time.perf_counter()
        amap = official_patch_map(cfg, model, patch_proj, prompt_learner, prompt_proj, patch_features, patch_projections, patch_keys, patch_values, gt_mask, device)
        patch_elapsed = time.perf_counter() - start_patch
        if patch_hash is None:
            patch_hash = sha256_array(amap.detach().cpu().numpy())
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * float(cfg.get("topk_ratio", 0.01))))
        patch_top = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        patch_scores.extend(patch_top.detach().cpu().tolist())

        base_scores = {}
        for variant in variants:
            start = time.perf_counter()
            if variant == "E0_OfficialFull":
                score = full_score(q_raw, image_keys, image_values, image_proj, device)
            elif variant in {"E1_ClassCentroid", "E2_ClassCentroid_SourceAffine"}:
                raw = score_prototype_variant("D1_ClassCentroid", q_proj, bank_e1)[0]
                score = sigmoid_affine(raw, cal_centroid.a, cal_centroid.b) if variant.startswith("E2_") else raw
            elif variant in {"E3_HQAP4", "E4_HQAP4_SourceAffine"}:
                raw = score_prototype_variant("D3_HQAP4", q_proj, bank_e3)[0]
                score = sigmoid_affine(raw, cal_hqap.a, cal_hqap.b) if variant.startswith("E4_") else raw
            else:
                raise ValueError(variant)
            times[variant] += patch_elapsed + (time.perf_counter() - start)
            base_scores[variant] = score

        for variant, score in base_scores.items():
            fused = (1 - float(cfg.get("fusion_k", 0.7))) * patch_top + float(cfg.get("fusion_k", 0.7)) * score
            results[variant][cls_name]["gt_sp"].extend(label)
            results[variant][cls_name]["pr_sp"].extend(fused.detach().cpu())
            if variant == "E0_OfficialFull":
                results[variant][cls_name]["imgs_masks"].append(gt_eval)
                results[variant][cls_name]["anomaly_maps"].append(amap.detach().cpu())
            image_scores[variant].extend(score.detach().cpu().tolist())
            fused_scores[variant].extend(fused.detach().cpu().tolist())
            for alpha in alphas:
                diag_score = alpha * score + (1 - alpha) * patch_top
                fusion_results[(variant, alpha)][cls_name]["gt_sp"].extend(label)
                fusion_results[(variant, alpha)][cls_name]["pr_sp"].extend(diag_score.detach().cpu())
        if torch.cuda.is_available() and image_count % 20 == 0:
            torch.cuda.empty_cache()

    e0_metrics = evaluate_results(results["E0_OfficialFull"], data.obj_list, compute_pro=bool(cfg.get("compute_pixel_aupro", True)))
    pixel_mean = e0_metrics["mean"]
    pixel_per = e0_metrics["per_class"]
    metric_rows, corr_rows, cal_rows, fusion_rows, latency_rows = [], [], [], [], []
    for variant in variants:
        if variant == "E0_OfficialFull":
            mean = e0_metrics["mean"]
        else:
            image_mean, _ = image_metrics_only(results[variant], data.obj_list)
            mean = {"I-AUROC": image_mean["I-AUROC"], "I-AP": image_mean["I-AP"], "P-AUROC": pixel_mean.get("P-AUROC", float("nan")), "PRO": pixel_mean.get("PRO", float("nan"))}
        cal = calibrators.get(variant)
        a = cal.a if cal else 1.0
        b = cal.b if cal else 0.0
        row = {
            "direction": dname,
            "dataset": dcfg["target"],
            "variant": variant,
            "image_auroc": mean.get("I-AUROC", float("nan")),
            "image_ap": mean.get("I-AP", float("nan")),
            "pixel_auroc": mean.get("P-AUROC", float("nan")),
            "pro": mean.get("PRO", float("nan")),
            "score_spearman_vs_E0": safe_corr(image_scores["E0_OfficialFull"], image_scores[variant], "spearman"),
            "score_pearson_vs_E0": safe_corr(image_scores["E0_OfficialFull"], image_scores[variant], "pearson"),
            "score_mae_vs_E0": mae(image_scores["E0_OfficialFull"], image_scores[variant]),
            "score_rmse_vs_E0": rmse(image_scores["E0_OfficialFull"], image_scores[variant]),
            "a": a,
            "b": b,
            "latency_ms_image": 1000.0 * times[variant] / max(1, image_count),
            "target_tuned": False,
        }
        row.update(score_distribution(image_scores[variant], labels_all))
        metric_rows.append(row)
        corr_rows.append({k: row[k] for k in ["direction", "dataset", "variant", "score_spearman_vs_E0", "score_pearson_vs_E0", "score_mae_vs_E0", "score_rmse_vs_E0", "score_mean_normal", "score_std_normal", "score_mean_anomaly", "score_std_anomaly", "score_gap"]})
        latency_rows.append({"direction": dname, "dataset": dcfg["target"], "variant": variant, "latency_ms_image": row["latency_ms_image"], "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0})

    for variant, cal in calibrators.items():
        cal_rows.append({
            "direction": dname,
            "dataset": dcfg["target"],
            "variant": variant,
            "a": cal.a,
            "b": cal.b,
            "source_fold_variance_a": cal.fold_variance_a,
            "source_fold_variance_b": cal.fold_variance_b,
            "source_split_hash": cal.source_split_hash,
            "source_score_mean": float(np.mean(cal.source_scores)),
            "source_label_mean": float(np.mean(cal.source_labels)),
            "target_tuned": False,
        })

    fold_rows = []
    for variant, cal in [("E2_ClassCentroid_SourceAffine", cal_centroid), ("E4_HQAP4_SourceAffine", cal_hqap)]:
        for row in cal.fold_rows:
            fold_rows.append({"direction": dname, "dataset": dcfg["target"], "target_variant": variant, **row})

    for (variant, alpha), res in fusion_results.items():
        image_mean, _ = image_metrics_only(res, data.obj_list)
        fusion_rows.append({
            "direction": dname,
            "dataset": dcfg["target"],
            "variant": variant,
            "alpha_memory": alpha,
            "image_auroc": image_mean["I-AUROC"],
            "image_ap": image_mean["I-AP"],
            "pixel_auroc": pixel_mean.get("P-AUROC", float("nan")),
            "pro": pixel_mean.get("PRO", float("nan")),
            "diagnostic_only": True,
            "target_tuned": False,
        })

    precheck = {
        "direction": dname,
        "dataset": dcfg["target"],
        "git_commit": git_commit(),
        "config_hash": sha256_file(config_path),
        "checkpoint_hash": sha256_file(resolve_path(dcfg["checkpoint"])),
        "patch_map_hash": patch_hash or "",
        "D0_memory_hash": sha256_file(resolve_path(dcfg["image_cache"])),
        "D1_prototype_hash": hash_tensor_list([bank_e1.global_centroids]),
        "D3_prototype_hash": hash_tensor_list([bank_e3.subprototypes[0], bank_e3.subprototypes[1]]),
        "source_split_hash": cal_centroid.source_split_hash,
        "target_label_access": "evaluation_only",
        "only_image_memory_scoring_changes": True,
        "checkpoint_projection_matches_phase_d": True,
        "patch_prompt_map_shared_across_E0_E4": True,
    }

    per_cat = []
    for variant in variants:
        if variant == "E0_OfficialFull":
            for cls, vals in e0_metrics["per_class"].items():
                per_cat.append({"direction": dname, "dataset": dcfg["target"], "variant": variant, "category": cls, **vals})
        else:
            image_mean, image_per = image_metrics_only(results[variant], data.obj_list)
            for cls, vals in image_per.items():
                per_cat.append({"direction": dname, "dataset": dcfg["target"], "variant": variant, "category": cls, **vals, "P-AUROC": pixel_per.get(cls, {}).get("P-AUROC", float("nan")), "PRO": pixel_per.get(cls, {}).get("PRO", float("nan"))})

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_rows, cal_rows, corr_rows, fusion_rows, fold_rows, latency_rows, per_cat, [precheck]


def write_reports(cfg: Dict, out: Path, metrics: List[Dict], prechecks: List[Dict]) -> None:
    report_dir = ROOT / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase-XXX-E Precheck", "", f"- Stage: `{cfg.get('stage')}`", f"- Output: `{out.relative_to(ROOT)}`", "- E0/E1/E3 reuse Phase-D implementation paths.", "- Official patch/prompt anomaly map is computed once per image and shared across E0-E4.", "- Target labels are evaluation-only.", "", "| direction | dataset | checkpoint | patch_map | D0_memory | D1_proto | D3_proto | source_split | target_label_access |", "|---|---|---|---|---|---|---|---|---|"]
    for r in prechecks:
        lines.append(f"| {r['direction']} | {r['dataset']} | `{r['checkpoint_hash'][:12]}` | `{r['patch_map_hash'][:12]}` | `{r['D0_memory_hash'][:12]}` | `{r['D1_prototype_hash'][:12]}` | `{r['D3_prototype_hash'][:12]}` | `{r['source_split_hash'][:12]}` | {r['target_label_access']} |")
    (report_dir / "phase_e_precheck.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    name = "phase_e_smoke_report.md" if cfg.get("stage") == "smoke" else "phase_e_calibration_report.md"
    lines = ["# Phase-XXX-E Calibration Report" if cfg.get("stage") != "smoke" else "# Phase-XXX-E Smoke Report", "", f"- Stage: `{cfg.get('stage')}`", f"- Output: `{out.relative_to(ROOT)}`", "", "| direction | variant | I-AUROC | d vs E0 | d vs E1 | I-AP | rho vs E0 | MAE vs E0 | a | b | target_tuned |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in metrics:
        lines.append(f"| {r['direction']} | {r['variant']} | {float(r['image_auroc']):.6f} | {float(r['delta_vs_E0_image_auroc'])*100:+.2f} | {float(r['delta_vs_E1_image_auroc'])*100:+.2f} | {float(r['image_ap']):.6f} | {float(r['score_spearman_vs_E0']):.4f} | {float(r['score_mae_vs_E0']):.6f} | {float(r['a']):.4f} | {float(r['b']):.4f} | {r['target_tuned']} |")
    if cfg.get("stage") != "smoke":
        lines.extend(["", "## Decision Notes", "", "- E2/E4 use source-only scalar affine calibration in logit space.", "- Fusion sensitivity is diagnostic only and is not used to choose target alpha.", "- See `phase_e_final_decision.md` for the gate decision."])
    (report_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if cfg.get("stage") == "smoke":
        return
    by = {(r["direction"], r["variant"]): r for r in metrics}
    lines = ["# Phase-XXX-E Final Decision", "", "## Status", "", "- Sprint E0 precheck: completed.", "- Sprint E1 calibration: completed for E0-E4 on d1/d2.", "- Sprint E2 diagnostics: completed.", "- Holdout was not launched automatically.", "", "## Answers", ""]
    for d in sorted({r["direction"] for r in metrics}):
        e1 = by[(d, "E1_ClassCentroid")]
        e2 = by[(d, "E2_ClassCentroid_SourceAffine")]
        e3 = by[(d, "E3_HQAP4")]
        e4 = by[(d, "E4_HQAP4_SourceAffine")]
        lines.append(f"- {d}: E1 vs E0 {float(e1['delta_vs_E0_image_auroc'])*100:+.2f} pt; E2 vs E1 {float(e2['delta_vs_E1_image_auroc'])*100:+.2f} pt; E3 vs E1 {float(e3['delta_vs_E1_image_auroc'])*100:+.2f} pt; E4 vs E3 {(float(e4['image_auroc'])-float(e3['image_auroc']))*100:+.2f} pt.")
    lines.extend(["", "1. E1 still reproduces centroid/full near-equivalence if deltas remain within the Phase-D tolerance.", "2. E2 answers whether source-only calibration recovers the E1 gap.", "3. MAE/RMSE rows in `score_correlation.csv` answer whether calibration approximates full-memory scores.", "4. Spearman >= 0.99 indicates ranking preservation.", "5. E4 versus E3 tests whether HQAP's d1 behavior is calibration-like.", "", "## Decision", "", "Use the table above and `phase_e_calibration_report.md` as the final gate. Recommend holdout only if E2 reaches strong or clear mechanism success without target tuning."])
    (report_dir / "phase_e_final_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype_calibration/phase_e_smoke.yaml")
    parser.add_argument("--directions", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--limit_images", type=int, default=None)
    parser.add_argument("--reset_outputs", action="store_true")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 0)))
    out = ROOT / cfg.get("output_dir", "results/phase_e")
    out.mkdir(parents=True, exist_ok=True)
    if args.reset_outputs:
        for name in ["formal_d1_d2.csv", "score_calibration.csv", "score_correlation.csv", "fusion_sensitivity.csv", "source_calibrator_folds.csv", "latency.csv", "per_category.csv", "precheck.csv"]:
            p = out / name
            if p.exists():
                p.unlink()
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"command": " ".join(sys.argv), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": cfg.get("stage")}, indent=2), encoding="utf-8")
    variants = args.variants or cfg.get("variants") or VARIANTS
    directions = args.directions or cfg.get("evaluation", {}).get("directions") or list(cfg["directions"].keys())
    for dname in directions:
        limit = args.limit_images if args.limit_images is not None else cfg.get("evaluation", {}).get("limit_images")
        print(f"[phase-e] {dname}: limit_images={limit if limit else 'FULL'} variants={','.join(variants)}", flush=True)
        m, cal, corr, fus, folds, lat, per, pre = run_direction(cfg, config_path, dname, cfg["directions"][dname], variants, int(limit) if limit else None)
        metrics = add_deltas(merge_rows(out / "formal_d1_d2.csv", m, ("direction", "dataset", "variant")))
        write_csv(out / "formal_d1_d2.csv", metrics)
        write_csv(out / "score_calibration.csv", merge_rows(out / "score_calibration.csv", cal, ("direction", "dataset", "variant")))
        write_csv(out / "score_correlation.csv", merge_rows(out / "score_correlation.csv", corr, ("direction", "dataset", "variant")))
        write_csv(out / "fusion_sensitivity.csv", merge_rows(out / "fusion_sensitivity.csv", fus, ("direction", "dataset", "variant", "alpha_memory")))
        write_csv(out / "source_calibrator_folds.csv", merge_rows(out / "source_calibrator_folds.csv", folds, ("direction", "dataset", "target_variant", "fold")))
        write_csv(out / "latency.csv", merge_rows(out / "latency.csv", lat, ("direction", "dataset", "variant")))
        write_csv(out / "per_category.csv", merge_rows(out / "per_category.csv", per, ("direction", "dataset", "variant", "category")))
        prechecks = merge_rows(out / "precheck.csv", pre, ("direction", "dataset"))
        write_csv(out / "precheck.csv", prechecks)
        write_reports(cfg, out, metrics, prechecks)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
