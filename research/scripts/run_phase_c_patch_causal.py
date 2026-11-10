from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import evaluate_results, init_results
from models.mlp import average_neighbor
from mrad import compute_patch_socre, compute_socre
from scripts.run_phase_b0_formal import load_cache, load_components, make_loader, resolve_path, set_seed


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


def parse_seed(name: str) -> int:
    if "_seed" not in name:
        return 0
    return int(name.rsplit("_seed", 1)[1])


def class_quota(labels: torch.Tensor, total: int, balanced: bool = False) -> Dict[int, int]:
    if balanced:
        return {0: total // 2, 1: total - total // 2}
    counts = {int(c): int((labels == c).sum().item()) for c in [0, 1]}
    n0 = max(1, int(round(total * counts[0] / max(1, counts[0] + counts[1]))))
    n1 = total - n0
    if n1 <= 0:
        n1, n0 = 1, total - 1
    return {0: n0, 1: n1}


def fps_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.shape[0] <= k:
        return torch.arange(x.shape[0], dtype=torch.long)
    selected = [0]
    dist = torch.cdist(x, x[0:1]).squeeze(1)
    for _ in range(1, k):
        idx = int(torch.argmax(dist).item())
        selected.append(idx)
        dist = torch.minimum(dist, torch.cdist(x, x[idx:idx + 1]).squeeze(1))
    return torch.tensor(selected, dtype=torch.long)


def projected_patch_keys(patch_keys: torch.Tensor, patch_proj, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        keys = F.normalize(patch_keys.to(device), dim=-1)
        proj = patch_proj(keys, 1)
    return proj.detach().cpu().float()


def build_variant_cache(
    name: str,
    patch_keys: torch.Tensor,
    patch_values: torch.Tensor,
    projected_keys: torch.Tensor,
    direction_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    labels = patch_values.argmax(dim=-1).long().cpu()
    keys = patch_keys.detach().cpu().float()
    values = patch_values.detach().cpu().float()
    info = {"variant": name, "historical_or_new": "NEW", "backend": "official_prompt_bias_patch_path"}

    if name == "P0_OfficialFull":
        info.update({"intervention": "none", "selection_space": "official_raw_patch_cache"})
        return keys, values, info

    if name == "P2_OfficialCentroidOnly":
        out_keys, out_values = [], []
        for cls in [0, 1]:
            idx = torch.where(labels == cls)[0]
            out_keys.append(F.normalize(keys[idx].mean(dim=0, keepdim=True), dim=-1))
            out_values.append(F.one_hot(torch.tensor([cls]), num_classes=2).float())
        info.update({"intervention": "binary_label_centroids", "selection_space": "official_raw_patch_cache"})
        return torch.cat(out_keys, dim=0), torch.cat(out_values, dim=0), info

    if name == "P3_SameLabelKeyRoll":
        # With one-hot binary values, any same-label permutation is mathematically
        # invariant for official softmax-weighted class logits. Keep this as a
        # sanity/coupling audit and report it explicitly.
        out_keys = keys.clone()
        for cls in [0, 1]:
            idx = torch.where(labels == cls)[0]
            out_keys[idx] = keys[idx.roll(1)]
        info.update({"intervention": "same_label_key_roll", "expected": "near_invariant"})
        return out_keys, values, info

    if name == "P4_GlobalValuePermutation":
        gen = torch.Generator().manual_seed(direction_seed + 404)
        perm = torch.randperm(values.shape[0], generator=gen)
        info.update({"intervention": "global_value_permutation_keys_fixed", "permutation_seed": direction_seed + 404})
        return keys, values[perm], info

    if name.startswith("P5_ClassStratifiedFPS100"):
        quota = class_quota(labels, 100, balanced=False)
        selected = []
        for cls in [0, 1]:
            idx = torch.where(labels == cls)[0]
            local = fps_indices(projected_keys[idx], min(quota[cls], idx.numel()))
            selected.append(idx[local])
        selected = torch.cat(selected, dim=0)
        info.update({"intervention": "class_stratified_fps100", "selection_space": "projected_patch_key_space", "quota": quota})
        return keys[selected], values[selected], info

    if name.startswith("P7_MatchedRandom100"):
        seed = parse_seed(name)
        quota = class_quota(labels, 100, balanced=False)
        gen = torch.Generator().manual_seed(direction_seed + 700 + seed)
        selected = []
        for cls in [0, 1]:
            idx = torch.where(labels == cls)[0]
            perm = idx[torch.randperm(idx.numel(), generator=gen)[: min(quota[cls], idx.numel())]]
            selected.append(perm)
        selected = torch.cat(selected, dim=0)
        info.update({"intervention": "matched_random100", "selection_space": "official_raw_patch_cache", "quota": quota, "random_seed": direction_seed + 700 + seed})
        return keys[selected], values[selected], info

    raise ValueError(f"Unknown Phase-C patch variant: {name}")


def memory_stats(direction: str, target: str, name: str, keys: torch.Tensor, values: torch.Tensor, info: Dict) -> Dict:
    labels = values.argmax(dim=-1).long()
    keys_f = keys.float()
    norms = keys_f.norm(dim=-1)
    sample = keys_f
    if sample.shape[0] > 1200:
        sample = sample[:1200]
    sim = F.normalize(sample, dim=-1) @ F.normalize(sample, dim=-1).T
    offdiag = sim[~torch.eye(sim.shape[0], dtype=torch.bool)]
    centered = keys_f - keys_f.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered)
        effective_rank = float(torch.exp(-(p := (s / s.sum()).clamp_min(1e-12)).mul(p.log()).sum()).item()) if s.sum() > 0 else 0.0
    except Exception:
        effective_rank = float("nan")
    row = {
        "direction": direction,
        "dataset": target,
        "variant": name,
        "backend": info.get("backend", "official_prompt_bias_patch_path"),
        "historical_or_new": info.get("historical_or_new", "NEW"),
        "intervention": info.get("intervention", ""),
        "num_keys": int(keys.shape[0]),
        "feature_dim": int(keys.shape[-1]),
        "num_normal": int((labels == 0).sum().item()),
        "num_anomaly": int((labels == 1).sum().item()),
        "key_norm_mean": float(norms.mean().item()),
        "key_norm_std": float(norms.std().item()) if norms.numel() > 1 else 0.0,
        "pairwise_similarity_mean": float(offdiag.mean().item()) if offdiag.numel() else float("nan"),
        "effective_rank": effective_rank,
        "source_category_coverage": "unavailable_in_official_cache",
        "source_image_coverage": "unavailable_in_official_cache",
    }
    for k, v in info.items():
        if k not in row:
            row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    return row


def official_patch_map(cfg: Dict, model, patch_proj, prompt_learner, prompt_proj, patch_features, patch_projections, keys, values, gt_mask, device):
    patch_projection = F.normalize(average_neighbor(patch_projections[3]), dim=-1)
    patch_feature_bias = F.normalize(average_neighbor(patch_features[3]), dim=-1)
    seg_logit, patch_feature_bias, _, _ = compute_patch_socre(
        patch_feature_bias,
        keys.to(device),
        values.to(device),
        device=device,
        proj=patch_proj,
        need_mask=False,
        patch_projection=patch_projection,
        gt_mask=gt_mask,
        is_mradft=False,
        use_proj=True,
    )
    bias = prompt_proj(patch_feature_bias[:, 0, :], patch_feature_bias[:, 1, :])
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = F.normalize(text_features, dim=-1)
    maps = []
    for idx, patch_feature in enumerate(patch_projections):
        if idx >= int(cfg.get("feature_map_layer", [0, 1, 2, 3])[3]):
            patch_feature = F.normalize(average_neighbor(patch_feature), dim=-1)
            similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity, int(cfg.get("image_size", 518)))
            maps.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
    amap = torch.stack(maps).sum(dim=0)
    amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=float(cfg.get("sigma", 4)))) for x in amap.detach().cpu()], dim=0)
    return amap


@torch.no_grad()
def run_direction(cfg: Dict, dname: str, dcfg: Dict, variants: List[str], limit_images: int | None):
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    data, loader = make_loader(cfg, dcfg["target"], limit_images)
    model, image_proj, patch_proj, prompt_learner, prompt_proj = load_components(cfg, resolve_path(dcfg["checkpoint"]), device, include_prompt=True)
    image_keys, image_values = load_cache(resolve_path(dcfg["image_cache"]))
    patch_keys, patch_values = load_cache(resolve_path(dcfg["patch_cache"]))
    image_keys, image_values = image_keys.to(device), image_values.to(device)
    projected = projected_patch_keys(patch_keys, patch_proj, device)
    built = {}
    stats_rows = []
    direction_seed = int(cfg.get("seed", 0)) + (0 if dname == "d1" else 10000)
    for name in variants:
        k, v, info = build_variant_cache(name, patch_keys, patch_values, projected, direction_seed)
        built[name] = (k, v, info)
        stats_rows.append(memory_stats(dname, dcfg["target"], name, k, v, info))

    results = {name: init_results(data.obj_list) for name in variants}
    image_scores_raw = {name: [] for name in variants}
    patch_scores_raw = {name: [] for name in variants}
    fused_scores_raw = {name: [] for name in variants}
    times = {name: 0.0 for name in variants}
    image_count = 0

    for items in loader:
        image_count += 1
        if image_count == 1 or image_count % 50 == 0:
            print(f"[phase-c-patch] {dname}: image {image_count}/{len(loader)}", flush=True)
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_eval = gt_mask.clone()
        gt_eval[gt_eval > 0.5], gt_eval[gt_eval <= 0.5] = 1, 0
        label = items["anomaly"].detach().cpu()
        image_features, patch_features, _, patch_projections = model.encode_image(image, cfg.get("features_list", [6, 12, 18, 24]), DPAM_layer=int(cfg.get("dpam_layer", 24)))
        image_features = F.normalize(image_features, dim=-1)
        image_logits, _ = compute_socre(image_features, image_keys, image_values, device, proj=image_proj, use_proj=True)
        image_score = image_logits[:, 1]

        for name in variants:
            keys, values, _ = built[name]
            start = time.perf_counter()
            amap = official_patch_map(cfg, model, patch_proj, prompt_learner, prompt_proj, patch_features, patch_projections, keys, values, gt_mask, device)
            times[name] += time.perf_counter() - start
            flat = amap.flatten(start_dim=1)
            topn = max(1, int(flat.shape[1] * float(cfg.get("topk_ratio", 0.01))))
            patch_topkmean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
            fused = (1 - float(cfg.get("fusion_k", 0.7))) * patch_topkmean + float(cfg.get("fusion_k", 0.7)) * image_score
            results[name][cls_name]["imgs_masks"].append(gt_eval)
            results[name][cls_name]["gt_sp"].extend(label)
            results[name][cls_name]["pr_sp"].extend(fused.detach().cpu())
            results[name][cls_name]["anomaly_maps"].append(amap.detach().cpu())
            image_scores_raw[name].extend(image_score.detach().cpu().tolist())
            patch_scores_raw[name].extend(patch_topkmean.detach().cpu().tolist())
            fused_scores_raw[name].extend(fused.detach().cpu().tolist())

        if torch.cuda.is_available() and image_count % 20 == 0:
            torch.cuda.empty_cache()

    metric_rows, per_category_rows, latency_rows = [], [], []
    for name in variants:
        metrics = evaluate_results(results[name], data.obj_list, compute_pro=bool(cfg.get("compute_pixel_aupro", True)))
        mean = metrics["mean"]
        row = {
            "direction": dname,
            "dataset": dcfg["target"],
            "variant": name,
            "backend": "official_prompt_bias_patch_path",
            "historical_or_new": built[name][2].get("historical_or_new", "NEW"),
            "I-AUROC": mean.get("I-AUROC", float("nan")),
            "I-AP": mean.get("I-AP", float("nan")),
            "P-AUROC": mean.get("P-AUROC", float("nan")),
            "PRO": mean.get("PRO", float("nan")),
            "latency_ms_image": 1000.0 * times[name] / max(1, image_count),
        }
        metric_rows.append(row)
        latency_rows.append({"direction": dname, "dataset": dcfg["target"], "variant": name, "latency_ms_image": row["latency_ms_image"], "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0})
        for cls, vals in metrics["per_class"].items():
            per_category_rows.append({"direction": dname, "dataset": dcfg["target"], "variant": name, "category": cls, **vals})
    return metric_rows, stats_rows, per_category_rows, latency_rows


def add_deltas(rows: List[Dict]) -> List[Dict]:
    base = {(r["direction"], r["dataset"]): r for r in rows if r["variant"] == "P0_OfficialFull"}
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
    name = "phase_c_patch_smoke_report.md" if stage == "smoke" else "phase_c_patch_report.md"
    lines = [
        "# Phase-XXX-C Patch Causal Report" if stage != "smoke" else "# Phase-XXX-C Patch Smoke Report",
        "",
        f"- Stage: `{stage}`",
        f"- Output: `{out_dir.relative_to(ROOT)}`",
        "- Backend: official MRAD-CLIP prompt-bias patch path.",
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
    parser.add_argument("--config", default="configs/memory_audit/phase_c_patch_smoke.yaml")
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
        for name in ["patch_causal.csv", "patch_memory_stats.csv", "per_category.csv", "latency.csv"]:
            target = out / name
            if target.exists():
                target.unlink()
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"command": " ".join(sys.argv), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": cfg.get("stage")}, indent=2), encoding="utf-8")
    variants = args.variants or cfg.get("variants") or []
    directions = args.directions or cfg.get("evaluation", {}).get("directions") or list(cfg["directions"].keys())
    for dname in directions:
        limit = args.limit_images if args.limit_images is not None else cfg.get("evaluation", {}).get("limit_images")
        for variant in variants:
            print(f"[phase-c-patch] {dname}: limit_images={limit if limit else 'FULL'} variant={variant}", flush=True)
            m, s, p, l = run_direction(cfg, dname, cfg["directions"][dname], [variant], int(limit) if limit else None)
            metrics_path = out / "patch_causal.csv"
            stats_path = out / "patch_memory_stats.csv"
            per_cat_path = out / "per_category.csv"
            latency_path = out / "latency.csv"
            merged_metrics = merge_rows(metrics_path, m, ("direction", "dataset", "variant"))
            merged_metrics = add_deltas(merged_metrics)
            write_csv(metrics_path, merged_metrics)
            write_csv(stats_path, merge_rows(stats_path, s, ("direction", "dataset", "variant")))
            write_csv(per_cat_path, merge_rows(per_cat_path, p, ("direction", "dataset", "variant", "category")))
            write_csv(latency_path, merge_rows(latency_path, l, ("direction", "dataset", "variant")))
            write_report(cfg, out, merged_metrics)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
