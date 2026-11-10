from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import evaluate_results, init_results
from mrad import compute_socre
from scripts.run_phase_b0_formal import load_cache, load_components, make_loader, resolve_path, set_seed
from scripts.run_phase_c_patch_causal import official_patch_map


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


def image_score_variant(
    variant: str,
    q_raw: torch.Tensor,
    image_keys: torch.Tensor,
    image_values: torch.Tensor,
    image_proj,
    device: torch.device,
    *,
    shuffled_query: torch.Tensor | None = None,
    key_perm: torch.Tensor | None = None,
    value_perm: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict]:
    keys = image_keys
    values = image_values
    q = q_raw
    if variant == "I1_QueryShuffle" and shuffled_query is not None:
        q = shuffled_query
    if variant == "I2_MemoryKeyShuffle" and key_perm is not None:
        keys = keys[key_perm]
    if variant == "I3_ValueShuffle" and value_perm is not None:
        values = values[value_perm]

    q_proj, k_proj = image_proj(q, keys)
    if variant == "I0_OfficialFull" or variant in {"I1_QueryShuffle", "I2_MemoryKeyShuffle", "I3_ValueShuffle"}:
        logits = q_proj @ k_proj.to(device).T
        weights = torch.softmax(logits, dim=-1)
        score = (weights @ values.to(device).float())[:, 1]
    elif variant == "I5_NormNeutralizedKey":
        k_norm = k_proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        median = k_norm.median().detach()
        k_new = F.normalize(k_proj, dim=-1) * median
        logits = q_proj @ k_new.to(device).T
        weights = torch.softmax(logits, dim=-1)
        score = (weights @ values.to(device).float())[:, 1]
    elif variant == "I6_NormNeutralizedQuery":
        k_norm = k_proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        median = k_norm.median().detach()
        q_new = F.normalize(q_proj, dim=-1) * median
        logits = q_new @ k_proj.to(device).T
        weights = torch.softmax(logits, dim=-1)
        score = (weights @ values.to(device).float())[:, 1]
    elif variant == "I8_ClassCentroidProjected":
        labels = values.argmax(dim=-1).long()
        centroids, centroid_values = [], []
        for cls in [0, 1]:
            idx = torch.where(labels == cls)[0]
            centroids.append(k_proj[idx].mean(dim=0, keepdim=True))
            centroid_values.append(F.one_hot(torch.tensor([cls], device=device), num_classes=2).float())
        k_new = torch.cat(centroids, dim=0)
        v_new = torch.cat(centroid_values, dim=0)
        logits = q_proj @ k_new.to(device).T
        weights = torch.softmax(logits, dim=-1)
        score = (weights @ v_new)[:, 1]
    else:
        raise ValueError(f"Unknown image causal variant: {variant}")

    with torch.no_grad():
        top = torch.topk(weights.detach().cpu(), k=min(8, weights.shape[-1]), dim=-1)
        diag = {
            "query_proj_norm": float(q_proj.norm(dim=-1).mean().detach().cpu().item()),
            "key_proj_norm_mean": float(k_proj.norm(dim=-1).mean().detach().cpu().item()),
            "key_proj_norm_median": float(k_proj.norm(dim=-1).median().detach().cpu().item()),
            "projected_dot_mean": float(logits.detach().cpu().mean().item()),
            "attention_top1_mean": float(top.values[:, 0].mean().item()),
            "selected_memory_ids": top.indices[0].tolist(),
        }
    return score, diag


@torch.no_grad()
def collect_query_bank(cfg: Dict, dcfg: Dict, limit_images: int | None) -> torch.Tensor:
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    _, loader = make_loader(cfg, dcfg["target"], limit_images)
    model, _, _, _, _ = load_components(cfg, resolve_path(dcfg["checkpoint"]), device, include_prompt=True)
    queries = []
    for items in loader:
        image = items["img"].to(device)
        image_features, _, _, _ = model.encode_image(image, cfg.get("features_list", [6, 12, 18, 24]), DPAM_layer=int(cfg.get("dpam_layer", 24)))
        queries.append(F.normalize(image_features, dim=-1).detach().cpu())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(queries, dim=0)


@torch.no_grad()
def run_direction_variant(cfg: Dict, dname: str, dcfg: Dict, variant: str, limit_images: int | None, query_bank: torch.Tensor):
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    data, loader = make_loader(cfg, dcfg["target"], limit_images)
    model, image_proj, patch_proj, prompt_learner, prompt_proj = load_components(cfg, resolve_path(dcfg["checkpoint"]), device, include_prompt=True)
    image_keys, image_values = load_cache(resolve_path(dcfg["image_cache"]))
    patch_keys, patch_values = load_cache(resolve_path(dcfg["patch_cache"]))
    image_keys, image_values = image_keys.to(device), image_values.to(device)
    patch_keys, patch_values = patch_keys.to(device), patch_values.to(device)
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + (0 if dname == "d1" else 10000) + 123)
    q_perm = torch.randperm(query_bank.shape[0], generator=gen)
    key_perm = torch.randperm(image_keys.shape[0], generator=gen).to(device)
    value_perm = torch.randperm(image_values.shape[0], generator=gen).to(device)
    results = init_results(data.obj_list)
    image_scores, patch_scores, fused_scores = [], [], []
    norm_rows = []
    seconds = 0.0
    image_count = 0

    for items in loader:
        image_count += 1
        if image_count == 1 or image_count % 50 == 0:
            print(f"[phase-c-image] {dname} {variant}: image {image_count}/{len(loader)}", flush=True)
        idx = image_count - 1
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_eval = gt_mask.clone()
        gt_eval[gt_eval > 0.5], gt_eval[gt_eval <= 0.5] = 1, 0
        label = items["anomaly"].detach().cpu()
        image_features, patch_features, _, patch_projections = model.encode_image(image, cfg.get("features_list", [6, 12, 18, 24]), DPAM_layer=int(cfg.get("dpam_layer", 24)))
        q_raw = F.normalize(image_features, dim=-1)
        start = time.perf_counter()
        amap = official_patch_map(cfg, model, patch_proj, prompt_learner, prompt_proj, patch_features, patch_projections, patch_keys, patch_values, gt_mask, device)
        shuffled = query_bank[q_perm[idx:idx + 1]].to(device) if variant == "I1_QueryShuffle" else None
        image_score, diag = image_score_variant(
            variant,
            q_raw,
            image_keys,
            image_values,
            image_proj,
            device,
            shuffled_query=shuffled,
            key_perm=key_perm,
            value_perm=value_perm,
        )
        seconds += time.perf_counter() - start
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * float(cfg.get("topk_ratio", 0.01))))
        patch_topkmean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        fused = (1 - float(cfg.get("fusion_k", 0.7))) * patch_topkmean + float(cfg.get("fusion_k", 0.7)) * image_score
        results[cls_name]["imgs_masks"].append(gt_eval)
        results[cls_name]["gt_sp"].extend(label)
        results[cls_name]["pr_sp"].extend(fused.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        image_scores.extend(image_score.detach().cpu().tolist())
        patch_scores.extend(patch_topkmean.detach().cpu().tolist())
        fused_scores.extend(fused.detach().cpu().tolist())
        norm_rows.append({"direction": dname, "dataset": dcfg["target"], "variant": variant, "image_index": idx, **diag})
        if torch.cuda.is_available() and image_count % 20 == 0:
            torch.cuda.empty_cache()

    metrics = evaluate_results(results, data.obj_list, compute_pro=bool(cfg.get("compute_pixel_aupro", True)))
    mean = metrics["mean"]
    row = {
        "direction": dname,
        "dataset": dcfg["target"],
        "variant": variant,
        "backend": "official_image_memory_causal",
        "historical_or_new": "NEW",
        "I-AUROC": mean.get("I-AUROC", float("nan")),
        "I-AP": mean.get("I-AP", float("nan")),
        "P-AUROC": mean.get("P-AUROC", float("nan")),
        "PRO": mean.get("PRO", float("nan")),
        "latency_ms_image": 1000.0 * seconds / max(1, image_count),
        "image_score_mean": float(np.mean(image_scores)),
        "patch_score_mean": float(np.mean(patch_scores)),
        "fused_score_mean": float(np.mean(fused_scores)),
    }
    per_cat = [{"direction": dname, "dataset": dcfg["target"], "variant": variant, "category": cls, **vals} for cls, vals in metrics["per_class"].items()]
    latency = {"direction": dname, "dataset": dcfg["target"], "variant": variant, "latency_ms_image": row["latency_ms_image"], "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [row], norm_rows, per_cat, [latency]


def add_deltas(rows: List[Dict]) -> List[Dict]:
    base = {(r["direction"], r["dataset"]): r for r in rows if r["variant"] == "I0_OfficialFull"}
    out = []
    for r in rows:
        b = base.get((r["direction"], r["dataset"]))
        row = dict(r)
        for metric in ["I-AUROC", "I-AP", "P-AUROC", "PRO"]:
            bv = float(b[metric]) if b else float("nan")
            rv = float(row[metric])
            row[f"delta_{metric}"] = rv - bv if np.isfinite(rv) and np.isfinite(bv) else float("nan")
        out.append(row)
    return out


def write_report(cfg: Dict, out_dir: Path, rows: List[Dict]) -> None:
    report_dir = ROOT / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    stage = cfg.get("stage", "formal")
    name = "phase_c_image_smoke_report.md" if stage == "smoke" else "phase_c_image_report.md"
    lines = [
        "# Phase-XXX-C Image Causal Report" if stage != "smoke" else "# Phase-XXX-C Image Smoke Report",
        "",
        f"- Stage: `{stage}`",
        f"- Output: `{out_dir.relative_to(ROOT)}`",
        "- Backend: official MRAD-CLIP image-memory causal path; prompt maps are unchanged.",
        "",
        "| direction | variant | I-AUROC | I-AP | P-AUROC | PRO | dI | dAP | dP | dPRO | latency ms/image |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['direction']} | {r['variant']} | {float(r['I-AUROC']):.4f} | {float(r['I-AP']):.4f} | "
            f"{float(r['P-AUROC']):.4f} | {float(r.get('PRO', float('nan'))):.4f} | "
            f"{float(r['delta_I-AUROC']):+.4f} | {float(r['delta_I-AP']):+.4f} | "
            f"{float(r['delta_P-AUROC']):+.4f} | {float(r['delta_PRO']):+.4f} | {float(r['latency_ms_image']):.2f} |"
        )
    (report_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/memory_audit/phase_c_image_smoke.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--directions", nargs="*", default=None)
    parser.add_argument("--limit_images", type=int, default=None)
    parser.add_argument("--reset_outputs", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(resolve_path(args.config), "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 0)))
    out = ROOT / cfg.get("output_dir", "results/phase_c")
    out.mkdir(parents=True, exist_ok=True)
    if args.reset_outputs:
        for name in ["image_causal.csv", "image_norm_angle_stats.csv", "image_per_category.csv", "image_latency.csv"]:
            target = out / name
            if target.exists():
                target.unlink()
    (out / "image_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "image_run_metadata.json").write_text(json.dumps({"command": " ".join(sys.argv), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": cfg.get("stage")}, indent=2), encoding="utf-8")
    variants = args.variants or cfg.get("variants") or []
    directions = args.directions or cfg.get("evaluation", {}).get("directions") or list(cfg["directions"].keys())
    for dname in directions:
        limit = args.limit_images if args.limit_images is not None else cfg.get("evaluation", {}).get("limit_images")
        print(f"[phase-c-image] {dname}: collect query bank limit_images={limit if limit else 'FULL'}", flush=True)
        query_bank = collect_query_bank(cfg, cfg["directions"][dname], int(limit) if limit else None)
        for variant in variants:
            print(f"[phase-c-image] {dname}: variant={variant}", flush=True)
            m, n, p, l = run_direction_variant(cfg, dname, cfg["directions"][dname], variant, int(limit) if limit else None, query_bank)
            metrics_path = out / "image_causal.csv"
            norm_path = out / "image_norm_angle_stats.csv"
            per_cat_path = out / "image_per_category.csv"
            latency_path = out / "image_latency.csv"
            merged = merge_rows(metrics_path, m, ("direction", "dataset", "variant"))
            merged = add_deltas(merged)
            write_csv(metrics_path, merged)
            write_csv(norm_path, merge_rows(norm_path, n, ("direction", "dataset", "variant", "image_index")))
            write_csv(per_cat_path, merge_rows(per_cat_path, p, ("direction", "dataset", "variant", "category")))
            write_csv(latency_path, merge_rows(latency_path, l, ("direction", "dataset", "variant")))
            write_report(cfg, out, merged)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
