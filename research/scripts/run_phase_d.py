from __future__ import annotations

import argparse
import csv
import json
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
from prototype_calibration.diagnostics import mae, safe_corr, sha256_array, sha256_file
from prototype_calibration.prototype_builder import build_projected_bank
from prototype_calibration.prototype_scorer import score_prototype_variant
from scripts.run_phase_b0_formal import load_cache, load_components, make_loader, resolve_path, set_seed
from scripts.run_phase_c_patch_causal import official_patch_map


PRIMARY_VARIANTS = ["D0_OfficialFull", "D1_ClassCentroid", "D2_HQAP2", "D3_HQAP4", "D4_CoreBoundary"]


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


def add_deltas(rows: List[Dict]) -> List[Dict]:
    base = {(r["direction"], r["dataset"]): r for r in rows if r["variant"] == "D0_OfficialFull"}
    out = []
    for r in rows:
        b = base.get((r["direction"], r["dataset"]))
        row = dict(r)
        for src, dst in [("image_auroc", "delta_image_auroc"), ("image_ap", "delta_image_ap"), ("pixel_auroc", "delta_pixel_auroc"), ("pro", "delta_pro")]:
            try:
                rv = float(row[src])
                bv = float(b[src]) if b else float("nan")
                row[dst] = rv - bv if np.isfinite(rv) and np.isfinite(bv) else float("nan")
            except Exception:
                row[dst] = float("nan")
        out.append(row)
    return out


def image_score_full(q_raw, image_keys, image_values, image_proj, device):
    logits, _ = compute_socre(q_raw, image_keys, image_values, device, proj=image_proj, use_proj=True)
    return logits[:, 1]


def image_score_centroid(q_proj, projected_keys, image_values):
    labels = image_values.argmax(dim=-1).long().cpu()
    centroids = []
    for cls in [0, 1]:
        idx = torch.where(labels == cls)[0]
        centroids.append(projected_keys[idx].mean(dim=0, keepdim=True))
    protos = torch.cat(centroids, dim=0).to(q_proj.device)
    values = F.one_hot(torch.tensor([0, 1], device=q_proj.device), num_classes=2).float()
    weights = torch.softmax(q_proj @ protos.T, dim=-1)
    return (weights @ values)[:, 1]


def build_banks(projected_keys: torch.Tensor, image_values: torch.Tensor, variants: List[str]) -> Dict[str, object]:
    banks = {}
    for variant in variants:
        if variant == "D0_OfficialFull":
            continue
        bank_variant = "D1_ClassCentroid" if variant == "D1_ClassCentroid" else variant
        banks[variant] = build_projected_bank(bank_variant, projected_keys, image_values)
    return banks


def blank_results(data, variants: List[str], alphas: List[float]):
    results = {v: init_results(data.obj_list) for v in variants}
    fusion_results = {(v, a): init_results(data.obj_list) for v in ["D0_OfficialFull", "D1_ClassCentroid", "D3_HQAP4", "D4_CoreBoundary"] if v in variants for a in alphas}
    return results, fusion_results


def image_metrics_for_results(results: Dict, obj_list: List[str]) -> Tuple[Dict, Dict]:
    per_class = {}
    image_auroc, image_ap = [], []
    for obj in obj_list:
        labels = np.asarray(results[obj]["gt_sp"], dtype=int)
        scores = np.asarray([float(x) for x in results[obj]["pr_sp"]], dtype=float)
        if labels.size == 0:
            continue
        try:
            ia = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else float("nan")
        except Exception:
            ia = float("nan")
        try:
            ap = float(average_precision_score(labels, scores)) if len(set(labels.tolist())) > 1 else float("nan")
        except Exception:
            ap = float("nan")
        per_class[obj] = {"I-AUROC": ia, "I-AP": ap}
        if np.isfinite(ia):
            image_auroc.append(ia)
        if np.isfinite(ap):
            image_ap.append(ap)
    mean = {
        "I-AUROC": float(np.mean(image_auroc)) if image_auroc else float("nan"),
        "I-AP": float(np.mean(image_ap)) if image_ap else float("nan"),
    }
    return mean, per_class


@torch.no_grad()
def run_direction(cfg: Dict, dname: str, dcfg: Dict, variants: List[str], limit_images: int | None):
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    data, loader = make_loader(cfg, dcfg["target"], limit_images)
    model, image_proj, patch_proj, prompt_learner, prompt_proj = load_components(cfg, resolve_path(dcfg["checkpoint"]), device, include_prompt=True)
    image_keys, image_values = load_cache(resolve_path(dcfg["image_cache"]))
    patch_keys, patch_values = load_cache(resolve_path(dcfg["patch_cache"]))
    image_keys, image_values = image_keys.to(device), image_values.to(device)
    patch_keys, patch_values = patch_keys.to(device), patch_values.to(device)
    projected_keys = projected_image_keys(image_proj, image_keys, device)
    banks = build_banks(projected_keys, image_values.detach().cpu(), variants)
    alphas = [float(x) for x in cfg.get("fusion_alphas", [0.0, 0.25, 0.5, 0.75, 1.0])]
    results, fusion_results = blank_results(data, variants, alphas)
    times = defaultdict(float)
    image_scores = defaultdict(list)
    patch_scores = []
    fused_scores = defaultdict(list)
    per_query_rows = []
    patch_hash = None
    patch_identity_ok = True
    image_count = 0

    for items in loader:
        image_count += 1
        if image_count == 1 or image_count % 50 == 0:
            print(f"[phase-d] {dname}: image {image_count}/{len(loader)}", flush=True)
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_eval = gt_mask.clone()
        gt_eval[gt_eval > 0.5], gt_eval[gt_eval <= 0.5] = 1, 0
        label = items["anomaly"].detach().cpu()
        image_features, patch_features, _, patch_projections = model.encode_image(image, cfg.get("features_list", [6, 12, 18, 24]), DPAM_layer=int(cfg.get("dpam_layer", 24)))
        q_raw = F.normalize(image_features, dim=-1)
        q_proj, _ = image_proj(q_raw, image_keys)
        start_patch = time.perf_counter()
        amap = official_patch_map(cfg, model, patch_proj, prompt_learner, prompt_proj, patch_features, patch_projections, patch_keys, patch_values, gt_mask, device)
        patch_elapsed = time.perf_counter() - start_patch
        this_hash = sha256_array(amap.detach().cpu().numpy())
        if patch_hash is None:
            patch_hash = this_hash
        patch_identity_ok = patch_identity_ok and (this_hash == sha256_array(amap.detach().cpu().numpy()))
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * float(cfg.get("topk_ratio", 0.01))))
        patch_topkmean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        patch_scores.extend(patch_topkmean.detach().cpu().tolist())

        score_map = {}
        diagnostics = {}
        for variant in variants:
            start = time.perf_counter()
            if variant == "D0_OfficialFull":
                score = image_score_full(q_raw, image_keys, image_values, image_proj, device)
                diag = {}
            elif variant == "D1_ClassCentroid":
                score = image_score_centroid(q_proj, projected_keys, image_values.detach().cpu())
                diag = {}
            else:
                score, diag = score_prototype_variant(variant, q_proj, banks[variant])
            times[variant] += patch_elapsed + (time.perf_counter() - start)
            score_map[variant] = score
            diagnostics[variant] = diag

        for variant, image_score in score_map.items():
            fused = (1 - float(cfg.get("fusion_k", 0.7))) * patch_topkmean + float(cfg.get("fusion_k", 0.7)) * image_score
            results[variant][cls_name]["gt_sp"].extend(label)
            results[variant][cls_name]["pr_sp"].extend(fused.detach().cpu())
            if variant == "D0_OfficialFull":
                results[variant][cls_name]["imgs_masks"].append(gt_eval)
                results[variant][cls_name]["anomaly_maps"].append(amap.detach().cpu())
            image_scores[variant].extend(image_score.detach().cpu().tolist())
            fused_scores[variant].extend(fused.detach().cpu().tolist())
            for alpha in alphas:
                key = (variant, alpha)
                if key not in fusion_results:
                    continue
                diag_fused = alpha * image_score + (1 - alpha) * patch_topkmean
                fusion_results[key][cls_name]["gt_sp"].extend(label)
                fusion_results[key][cls_name]["pr_sp"].extend(diag_fused.detach().cpu())

        per_query_rows.append({
            "direction": dname,
            "dataset": dcfg["target"],
            "image_index": image_count - 1,
            "category": cls_name,
            "label": int(label.item()),
            "patch_top1_score": float(patch_topkmean.detach().cpu().item()),
            "full_memory_score": float(score_map["D0_OfficialFull"].detach().cpu().item()),
            "centroid_score": float(score_map.get("D1_ClassCentroid", torch.tensor([float("nan")])).detach().cpu().item()),
            "D2_HQAP2_score": float(score_map.get("D2_HQAP2", torch.tensor([float("nan")])).detach().cpu().item()),
            "D3_HQAP4_score": float(score_map.get("D3_HQAP4", torch.tensor([float("nan")])).detach().cpu().item()),
            "D4_CoreBoundary_score": float(score_map.get("D4_CoreBoundary", torch.tensor([float("nan")])).detach().cpu().item()),
            "D3_normal_shift_norm": diagnostics.get("D3_HQAP4", {}).get("class_0_shift_norm", float("nan")),
            "D3_anomaly_shift_norm": diagnostics.get("D3_HQAP4", {}).get("class_1_shift_norm", float("nan")),
            "D4_boundary_gate": diagnostics.get("D4_CoreBoundary", {}).get("boundary_gate", float("nan")),
        })
        if torch.cuda.is_available() and image_count % 20 == 0:
            torch.cuda.empty_cache()

    metric_rows, per_category_rows, latency_rows = [], [], []
    d0_pixel_metrics = evaluate_results(results["D0_OfficialFull"], data.obj_list, compute_pro=bool(cfg.get("compute_pixel_aupro", True)))
    pixel_mean = d0_pixel_metrics["mean"]
    pixel_per_class = d0_pixel_metrics["per_class"]
    for variant in variants:
        if variant == "D0_OfficialFull":
            mean = dict(d0_pixel_metrics["mean"])
            per_class = d0_pixel_metrics["per_class"]
        else:
            image_mean, image_per_class = image_metrics_for_results(results[variant], data.obj_list)
            mean = {
                "I-AUROC": image_mean.get("I-AUROC", float("nan")),
                "I-AP": image_mean.get("I-AP", float("nan")),
                "P-AUROC": pixel_mean.get("P-AUROC", float("nan")),
                "PRO": pixel_mean.get("PRO", float("nan")),
            }
            per_class = {}
            for cls in data.obj_list:
                if cls in image_per_class:
                    per_class[cls] = {
                        **image_per_class[cls],
                        "P-AUROC": pixel_per_class.get(cls, {}).get("P-AUROC", float("nan")),
                        "PRO": pixel_per_class.get(cls, {}).get("PRO", float("nan")),
                    }
        row = {
            "direction": dname,
            "dataset": dcfg["target"],
            "variant": variant,
            "prototype_count": 0 if variant == "D0_OfficialFull" else int(banks[variant].prototype_count),
            "image_auroc": mean.get("I-AUROC", float("nan")),
            "image_ap": mean.get("I-AP", float("nan")),
            "pixel_auroc": mean.get("P-AUROC", float("nan")),
            "pro": mean.get("PRO", float("nan")),
            "score_spearman_vs_full": safe_corr(image_scores["D0_OfficialFull"], image_scores[variant], "spearman"),
            "score_pearson_vs_full": safe_corr(image_scores["D0_OfficialFull"], image_scores[variant], "pearson"),
            "score_mae_vs_full": mae(image_scores["D0_OfficialFull"], image_scores[variant]),
            "latency_ms_image": 1000.0 * times[variant] / max(1, image_count),
            "target_tuned": False,
        }
        metric_rows.append(row)
        latency_rows.append({
            "direction": dname,
            "dataset": dcfg["target"],
            "variant": variant,
            "latency_ms_image": row["latency_ms_image"],
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
        })
        for cls, vals in per_class.items():
            per_category_rows.append({"direction": dname, "dataset": dcfg["target"], "variant": variant, "category": cls, **vals})

    corr_rows = []
    for variant in variants:
        corr_rows.append({
            "direction": dname,
            "dataset": dcfg["target"],
            "variant": variant,
            "score_spearman_vs_full": safe_corr(image_scores["D0_OfficialFull"], image_scores[variant], "spearman"),
            "score_pearson_vs_full": safe_corr(image_scores["D0_OfficialFull"], image_scores[variant], "pearson"),
            "score_mae_vs_full": mae(image_scores["D0_OfficialFull"], image_scores[variant]),
            "score_mean": float(np.mean(image_scores[variant])),
            "score_std": float(np.std(image_scores[variant])),
        })

    stats_rows = []
    for variant, bank in banks.items():
        for row in bank.stats:
            stats_rows.append({"direction": dname, "dataset": dcfg["target"], "prototype_count": bank.prototype_count, **row})

    fusion_rows = []
    for (variant, alpha), res in fusion_results.items():
        image_mean, _ = image_metrics_for_results(res, data.obj_list)
        fusion_rows.append({
            "direction": dname,
            "dataset": dcfg["target"],
            "variant": variant,
            "alpha_memory": alpha,
            "image_auroc": image_mean.get("I-AUROC", float("nan")),
            "image_ap": image_mean.get("I-AP", float("nan")),
            "pixel_auroc": pixel_mean.get("P-AUROC", float("nan")),
            "pro": pixel_mean.get("PRO", float("nan")),
            "diagnostic_only": True,
            "target_tuned": False,
        })

    precheck = {
        "direction": dname,
        "dataset": dcfg["target"],
        "source": dcfg["source"],
        "checkpoint_hash": sha256_file(resolve_path(dcfg["checkpoint"])),
        "image_memory_hash": sha256_file(resolve_path(dcfg["image_cache"])),
        "patch_cache_hash": sha256_file(resolve_path(dcfg["patch_cache"])),
        "official_baseline_hash": sha256_file(ROOT / "scripts" / "run_phase_c_image_causal.py"),
        "patch_map_hash": patch_hash or "",
        "patch_map_identity_across_variants": patch_identity_ok,
        "prototype_space": "official projected image-memory key space: image_proj.mlp2(cache_keys)",
        "prototype_count": "D1=2; D2=4; D3=8; D4=4",
        "target_label_access": "evaluation_only",
        "only_image_memory_branch_changes": True,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_rows, corr_rows, stats_rows, per_category_rows, fusion_rows, latency_rows, per_query_rows, [precheck]


def write_precheck_report(cfg: Dict, out: Path, rows: List[Dict]) -> None:
    report_dir = ROOT / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase-XXX-D Precheck",
        "",
        f"- Stage: `{cfg.get('stage')}`",
        f"- Output: `{out.relative_to(ROOT)}`",
        "- D0/D1 reuse the Phase-C validated official image-memory and class-centroid projected-key paths.",
        "- Official patch/prompt anomaly map is computed once per image and reused unchanged across D0-D4.",
        "- Target labels are appended only after scores are produced, for metrics and diagnostics.",
        "",
        "| direction | dataset | checkpoint_hash | image_memory_hash | patch_map_hash | prototype_space | prototype_count | target_label_access |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['direction']} | {r['dataset']} | `{r['checkpoint_hash'][:12]}` | `{r['image_memory_hash'][:12]}` | "
            f"`{r['patch_map_hash'][:12]}` | {r['prototype_space']} | {r['prototype_count']} | {r['target_label_access']} |"
        )
    (report_dir / "phase_d_precheck.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_primary_report(cfg: Dict, out: Path, rows: List[Dict]) -> None:
    report_dir = ROOT / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    name = "phase_d_smoke_report.md" if cfg.get("stage") == "smoke" else "phase_d_primary_report.md"
    lines = [
        "# Phase-XXX-D Primary Prototype Report" if cfg.get("stage") != "smoke" else "# Phase-XXX-D Smoke Report",
        "",
        f"- Stage: `{cfg.get('stage')}`",
        f"- Output: `{out.relative_to(ROOT)}`",
        "- Scope: D0-D4 primary compact prototype audit only. D5-D8 and holdout remain conditional.",
        "",
        "| direction | variant | prototypes | image AUROC | d image AUROC | image AP | d image AP | pixel AUROC | PRO | Spearman vs D0 | MAE vs D0 | latency ms/image |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['direction']} | {r['variant']} | {int(r['prototype_count'])} | {float(r['image_auroc']):.6f} | "
            f"{float(r['delta_image_auroc']) * 100:+.2f} | {float(r['image_ap']):.6f} | {float(r['delta_image_ap']) * 100:+.2f} | "
            f"{float(r['pixel_auroc']):.6f} | {float(r.get('pro', float('nan'))):.6f} | {float(r['score_spearman_vs_full']):.4f} | "
            f"{float(r['score_mae_vs_full']):.6f} | {float(r['latency_ms_image']):.2f} |"
        )
    (report_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_decision(cfg: Dict, rows: List[Dict]) -> None:
    if cfg.get("stage") == "smoke":
        return
    report_dir = ROOT / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    by = {(r["direction"], r["variant"]): r for r in rows}
    lines = [
        "# Phase-XXX-D Final Decision",
        "",
        "## Status",
        "",
        "- Sprint D0 precheck: completed.",
        "- Sprint D1 primary compact prototype test: completed for D0-D4 on d1/d2.",
        "- Sprint D2 calibration, Sprint D3 text alignment, and Sprint D4 holdout were not run unless explicitly triggered after this gate.",
        "",
        "## Gate Readout",
        "",
    ]
    for d in sorted({r["direction"] for r in rows}):
        d1 = by.get((d, "D1_ClassCentroid"))
        d3 = by.get((d, "D3_HQAP4"))
        d4 = by.get((d, "D4_CoreBoundary"))
        if d1 and d3 and d4:
            lines.append(
                f"- {d}: D1 delta {float(d1['delta_image_auroc']) * 100:+.2f} pt; "
                f"D3 delta {float(d3['delta_image_auroc']) * 100:+.2f} pt; "
                f"D4 delta {float(d4['delta_image_auroc']) * 100:+.2f} pt."
            )
    lines.extend([
        "",
        "## Next",
        "",
        "Use `reports/phase_d_primary_report.md` as the decision card for whether D5/D6 source-only calibration is justified.",
    ])
    (report_dir / "phase_d_final_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype_calibration/phase_d_smoke.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--directions", nargs="*", default=None)
    parser.add_argument("--limit_images", type=int, default=None)
    parser.add_argument("--reset_outputs", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(resolve_path(args.config), "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 0)))
    out = ROOT / cfg.get("output_dir", "results/phase_d")
    out.mkdir(parents=True, exist_ok=True)
    if args.reset_outputs:
        for name in ["formal_d1_d2.csv", "score_correlation.csv", "prototype_statistics.csv", "per_category.csv", "fusion_sensitivity.csv", "holdout.csv", "latency.csv", "query_diagnostics.csv", "precheck.csv"]:
            p = out / name
            if p.exists():
                p.unlink()
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"command": " ".join(sys.argv), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": cfg.get("stage")}, indent=2), encoding="utf-8")
    variants = args.variants or cfg.get("variants") or PRIMARY_VARIANTS
    directions = args.directions or cfg.get("evaluation", {}).get("directions") or list(cfg["directions"].keys())
    for dname in directions:
        limit = args.limit_images if args.limit_images is not None else cfg.get("evaluation", {}).get("limit_images")
        print(f"[phase-d] {dname}: limit_images={limit if limit else 'FULL'} variants={','.join(variants)}", flush=True)
        m, c, s, p, f, l, q, pre = run_direction(cfg, dname, cfg["directions"][dname], variants, int(limit) if limit else None)
        metric_path = out / "formal_d1_d2.csv"
        corr_path = out / "score_correlation.csv"
        stats_path = out / "prototype_statistics.csv"
        per_cat_path = out / "per_category.csv"
        fusion_path = out / "fusion_sensitivity.csv"
        latency_path = out / "latency.csv"
        query_path = out / "query_diagnostics.csv"
        precheck_path = out / "precheck.csv"
        metrics = add_deltas(merge_rows(metric_path, m, ("direction", "dataset", "variant")))
        write_csv(metric_path, metrics)
        write_csv(corr_path, merge_rows(corr_path, c, ("direction", "dataset", "variant")))
        write_csv(stats_path, merge_rows(stats_path, s, ("direction", "dataset", "variant", "class_label", "prototype_kind", "prototype_id")))
        write_csv(per_cat_path, merge_rows(per_cat_path, p, ("direction", "dataset", "variant", "category")))
        write_csv(fusion_path, merge_rows(fusion_path, f, ("direction", "dataset", "variant", "alpha_memory")))
        write_csv(latency_path, merge_rows(latency_path, l, ("direction", "dataset", "variant")))
        write_csv(query_path, merge_rows(query_path, q, ("direction", "dataset", "image_index")))
        prechecks = merge_rows(precheck_path, pre, ("direction", "dataset"))
        write_csv(precheck_path, prechecks)
        write_precheck_report(cfg, out, prechecks)
        write_primary_report(cfg, out, metrics)
        write_final_decision(cfg, metrics)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
