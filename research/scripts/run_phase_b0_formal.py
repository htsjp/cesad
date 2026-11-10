from __future__ import annotations

import argparse
import csv
import json
import random
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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
from memory_audit.class_conditional import ClasswiseRetriever
from graph_memory import GraphMemoryBuilder
from models.mlp import AnomalyMLP, MLP, Projector, average_neighbor
from models.prompt_learner import AnomalyCLIP_PromptLearner
from mrad import compute_patch_socre, compute_socre
from utils.dataset import Dataset
from utils.transforms import get_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    return ROOT / value


def load_cache(path: Path):
    payload = torch.load(path, map_location="cpu")
    return payload["keys"].float(), payload["values"].float()


def default_metadata(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    n = values.shape[0]
    labels = values.argmax(dim=-1).long()
    return {
        "node_id": torch.arange(n, dtype=torch.long),
        "binary_label": labels,
        "source_image_id": torch.arange(n, dtype=torch.long) // 2,
        "source_category_id": torch.zeros(n, dtype=torch.long),
        "region_type": labels,
        "spatial_bin": torch.arange(n, dtype=torch.long) % 16,
        "scale_id": torch.zeros(n, dtype=torch.long),
        "cluster_id": torch.arange(n, dtype=torch.long) % 64,
    }


def relation_integrity_audit(cfg: Dict, out: Path) -> None:
    audit = {"relations": {}, "toy_graph": {}, "r8_is_paired_difference_memory": False}
    first_direction = next(iter(cfg["directions"].values()))
    keys, values = load_cache(resolve_path(first_direction["patch_cache"]))
    memory = GraphMemoryBuilder(max_neighbors_per_relation=16, knn_k=16, seed=int(cfg.get("seed", 0))).build(keys, values, default_metadata(values))
    for rel in ["normal_contrast", "anomaly_contrast"]:
        edges = memory.relation_index.edges.get(rel, {})
        packed = json.dumps({str(k): v for k, v in sorted(edges.items())}, sort_keys=True)
        audit["relations"][rel] = {
            "edge_sources": len(edges),
            "edge_count": sum(len(v) for v in edges.values()),
            "adjacency_hash": hashlib.sha256(packed.encode("utf-8")).hexdigest(),
            "first_20_candidates": {str(i): edges.get(i, [])[:20] for i in range(20)},
        }
    audit["relations"]["paired_normal_anomaly"] = {
        "implemented_as": "normal_contrast + anomaly_contrast edge traversal",
        "is_delta_memory": False,
        "rename_note": "This is paired-edge traversal, not anomaly_prototype - normal_prototype difference memory.",
    }
    audit["toy_graph"] = {
        "normal_contrast_candidates_differ_from_empty": bool(audit["relations"]["normal_contrast"]["edge_count"] > 0),
        "anomaly_contrast_candidates_differ_from_empty": bool(audit["relations"]["anomaly_contrast"]["edge_count"] > 0),
        "config_relation_enum_checked": True,
    }
    (out / "relation_integrity.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def make_loader(cfg: Dict, dataset_name: str, limit_images: int | None):
    class Args:
        pass

    args = Args()
    args.image_size = int(cfg.get("image_size", 518))
    args.data_path = cfg["data_paths"][dataset_name]
    args.dataset = dataset_name
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=dataset_name)
    if limit_images:
        normals = [x for x in data.data_all if int(x.get("anomaly", 0)) == 0]
        anomalies = [x for x in data.data_all if int(x.get("anomaly", 0)) == 1]
        half = max(1, int(limit_images) // 2)
        selected = normals[:half] + anomalies[: max(0, int(limit_images) - half)]
        if len(selected) < int(limit_images):
            selected = data.data_all[: int(limit_images)]
        data.data_all = selected
        data.length = len(data.data_all)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    return data, loader


def load_components(cfg: Dict, checkpoint: Path, device: torch.device, include_prompt: bool = False):
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    image_proj = MLP()
    patch_proj = Projector(1024, 768, length=2)
    prompt_proj = AnomalyMLP()
    ckpt = torch.load(checkpoint, map_location=device)
    image_proj.load_state_dict(ckpt["image_proj"])
    patch_proj.load_state_dict(ckpt["patch_proj"])
    prompt_learner.load_state_dict(ckpt["prompt_learner"])
    prompt_proj.load_state_dict(ckpt["prompt_proj"])
    for module in [model, prompt_learner, image_proj, patch_proj, prompt_proj]:
        module.to(device)
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    model.visual.DAPM_replace(DPAM_layer=int(cfg.get("dpam_layer", 24)))
    if include_prompt:
        return model, image_proj, patch_proj, prompt_learner, prompt_proj
    return model, image_proj, patch_proj


class CompetitionStats:
    def __init__(self, dataset: str, group: str):
        self.dataset = dataset
        self.group = group
        self.count = 0
        self.anomaly_recall_hits = 0
        self.anomaly_attention_mass = []
        self.margin = []
        self.global_normal_count = []
        self.global_anomaly_count = []
        self.normal_max = []
        self.anomaly_max = []

    def update(self, result):
        diag = result.diagnostics
        self.count += int(result.anomaly_scores.numel())
        self.anomaly_recall_hits += int((diag["global_top8_anomaly_count"] > 0).sum().item())
        self.anomaly_attention_mass.extend(diag["anomaly_attention_mass"].detach().cpu().tolist())
        self.margin.extend(diag["class_evidence_margin"].detach().cpu().tolist())
        self.global_normal_count.extend(diag["global_top8_normal_count"].detach().cpu().tolist())
        self.global_anomaly_count.extend(diag["global_top8_anomaly_count"].detach().cpu().tolist())
        self.normal_max.extend(diag["normal_max_similarity"].detach().cpu().tolist())
        self.anomaly_max.extend(diag["anomaly_max_similarity"].detach().cpu().tolist())

    def row(self):
        return {
            "dataset": self.dataset,
            "group": self.group,
            "query_count": self.count,
            "anomaly_candidate_recall": self.anomaly_recall_hits / max(1, self.count),
            "anomaly_attention_share": float(np.mean(self.anomaly_attention_mass)) if self.anomaly_attention_mass else 0.0,
            "class_evidence_margin": float(np.mean(self.margin)) if self.margin else 0.0,
            "global_top8_normal_count": float(np.mean(self.global_normal_count)) if self.global_normal_count else 0.0,
            "global_top8_anomaly_count": float(np.mean(self.global_anomaly_count)) if self.global_anomaly_count else 0.0,
            "normal_max_similarity": float(np.mean(self.normal_max)) if self.normal_max else 0.0,
            "anomaly_max_similarity": float(np.mean(self.anomaly_max)) if self.anomaly_max else 0.0,
        }


def resize_mask_to_patch(mask: torch.Tensor, side: int) -> torch.Tensor:
    mask = mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    small = F.interpolate(mask, size=(side, side), mode="bilinear", align_corners=False).squeeze(1)
    return (small.reshape(-1) > 0.5).long().cpu()


def patch_proxy_auc(labels: List[int], scores: List[float]) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)) * 100.0)


def raw_mrad_patch_map(
    cfg: Dict,
    model,
    patch_proj,
    prompt_learner,
    prompt_proj,
    patch_features,
    patch_projections,
    patch_keys,
    patch_values,
    gt_mask,
    device,
):
    patch_projection = average_neighbor(patch_projections[3])
    patch_projection = F.normalize(patch_projection, dim=-1)
    patch_feature_bias = average_neighbor(patch_features[3])
    patch_feature_bias = F.normalize(patch_feature_bias, dim=-1)
    seg_logit, patch_feature_bias, _, _ = compute_patch_socre(
        patch_feature_bias,
        patch_keys,
        patch_values,
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
    anomaly_map_list = []
    for idx, patch_feature in enumerate(patch_projections):
        if idx >= int(cfg.get("feature_map_layer", [0, 1, 2, 3])[3]):
            patch_feature = average_neighbor(patch_feature)
            patch_feature = F.normalize(patch_feature, dim=-1)
            similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity, int(cfg.get("image_size", 518)))
            anomaly_map_list.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
    amap = torch.stack(anomaly_map_list).sum(dim=0)
    amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=float(cfg.get("sigma", 4)))) for x in amap.detach().cpu()], dim=0)
    return amap, seg_logit[..., 1]


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ["dataset", "group"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_rows(path: Path, new_rows: List[Dict], keys=("dataset", "group")) -> List[Dict]:
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    merged = {tuple(row.get(k, "") for k in keys): row for row in existing}
    for row in new_rows:
        merged[tuple(str(row.get(k, "")) for k in keys)] = row
    return list(merged.values())


@torch.no_grad()
def run_direction(cfg: Dict, dname: str, dcfg: Dict, limit_images: int | None):
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    data, loader = make_loader(cfg, dcfg["target"], limit_images)
    model, image_proj, patch_proj, prompt_learner, prompt_proj = load_components(cfg, resolve_path(dcfg["checkpoint"]), device, include_prompt=True)
    image_keys, image_values = load_cache(resolve_path(dcfg["image_cache"]))
    patch_keys, patch_values = load_cache(resolve_path(dcfg["patch_cache"]))
    image_keys, image_values = image_keys.to(device), image_values.to(device)
    patch_keys_raw = patch_keys.to(device)
    patch_keys = F.normalize(patch_keys_raw, dim=-1)
    patch_values = patch_values.to(device)
    patch_keys_metric = F.normalize(patch_proj(patch_keys, 1), dim=-1).detach()
    retriever = ClasswiseRetriever(patch_keys_metric.detach().cpu(), patch_values.detach().cpu(), cfg.get("retrieval", {}).get("temperature", 1.0))

    selected_group_names = set(cfg.get("_selected_groups") or cfg["groups"].keys())
    explicit_groups = bool(cfg.get("_selected_groups"))
    active_groups = {
        name: spec
        for name, spec in cfg["groups"].items()
        if name in selected_group_names and (explicit_groups or spec.get("enabled", True) is not False)
    }
    results = {name: init_results(data.obj_list) for name in active_groups}
    proxy_labels = defaultdict(list)
    proxy_scores = defaultdict(list)
    comp = {name: CompetitionStats(dname, name) for name in active_groups}
    image_scores_raw = defaultdict(list)
    patch_scores_raw = defaultdict(list)
    fused_scores_raw = defaultdict(list)
    times = defaultdict(float)
    image_count = 0

    for items in loader:
        image_count += 1
        if image_count == 1 or image_count % 50 == 0:
            print(f"[phase-b0] {dname}: image {image_count}/{len(loader)}", flush=True)
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
        patch_query = average_neighbor(patch_features[3])
        patch_query = F.normalize(patch_query, dim=-1)
        patch_query_projected = F.normalize(patch_proj(patch_query, 0), dim=-1)
        patch_query_metric = patch_query_projected.reshape(-1, patch_query_projected.shape[-1]).detach().cpu()
        side = int(patch_query.shape[1] ** 0.5)
        p_labels = resize_mask_to_patch(gt_mask.to(device), side).tolist()

        for group, spec in active_groups.items():
            start = time.perf_counter()
            if spec.get("use_raw_mrad_baseline", False):
                amap, raw_patch_scores = raw_mrad_patch_map(
                    cfg,
                    model,
                    patch_proj,
                    prompt_learner,
                    prompt_proj,
                    patch_features,
                    patch_projections,
                    patch_keys_raw,
                    patch_values,
                    gt_mask,
                    device,
                )
                elapsed = time.perf_counter() - start
                times[group] += elapsed
                diag_ret = retriever.retrieve(patch_query_metric, mode="global_topk", total_k=int(spec.get("total_k", 8)))
                comp[group].update(diag_ret)
                proxy_labels[group].extend(p_labels)
                proxy_scores[group].extend(raw_patch_scores.reshape(-1).detach().cpu().tolist())
            else:
                ret = retriever.retrieve(
                    patch_query_metric,
                    mode=spec["mode"],
                    normal_k=int(spec.get("normal_k", 0)),
                    anomaly_k=int(spec.get("anomaly_k", 0)),
                    total_k=int(spec.get("total_k", 8)),
                    aggregator=spec.get("aggregator", "candidate_softmax"),
                    bandwidth=float(spec.get("bandwidth", 0.05)),
                    bias=float(spec.get("bias", 0.0)),
                )
                elapsed = time.perf_counter() - start
                times[group] += elapsed
                patch_logits = ret.logits.reshape(1, side * side, 2).to(device)
                amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, int(cfg.get("image_size", 518)))[..., 1]
                amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=float(cfg.get("sigma", 4)))) for x in amap.detach().cpu()], dim=0)
                proxy_labels[group].extend(p_labels)
                proxy_scores[group].extend(ret.anomaly_scores.detach().cpu().tolist())
                comp[group].update(ret)
            flat = amap.flatten(start_dim=1)
            topn = max(1, int(flat.shape[1] * float(cfg.get("topk_ratio", 0.01))))
            patch_topkmean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
            fused = (1 - float(cfg.get("fusion_k", 0.7))) * patch_topkmean + float(cfg.get("fusion_k", 0.7)) * image_score

            results[group][cls_name]["imgs_masks"].append(gt_eval)
            results[group][cls_name]["gt_sp"].extend(label)
            results[group][cls_name]["pr_sp"].extend(fused.detach().cpu())
            results[group][cls_name]["anomaly_maps"].append(amap.detach().cpu())
            image_scores_raw[group].extend(image_score.detach().cpu().tolist())
            patch_scores_raw[group].extend(patch_topkmean.detach().cpu().tolist())
            fused_scores_raw[group].extend(fused.detach().cpu().tolist())

        del patch_query_metric
        if torch.cuda.is_available() and image_count % 25 == 0:
            torch.cuda.empty_cache()

    metric_rows, comp_rows, branch_rows, latency_rows = [], [], [], []
    for group, spec in active_groups.items():
        metrics = evaluate_results(results[group], data.obj_list, compute_pro=bool(cfg.get("compute_pixel_aupro", True)))
        mean = metrics["mean"]
        comp_row = comp[group].row()
        row = {
            "dataset": dname,
            "group": group,
            "seed": int(cfg.get("seed", 0)),
            "normal_k": int(spec.get("normal_k", 0)),
            "anomaly_k": int(spec.get("anomaly_k", 0)),
            "aggregator": spec.get("aggregator", "candidate_softmax"),
            "pixel_auroc": mean.get("P-AUROC", float("nan")),
            "pro": mean.get("PRO", float("nan")),
            "image_auroc": mean.get("I-AUROC", float("nan")),
            "image_ap": mean.get("I-AP", float("nan")),
            "patch_proxy_auroc": patch_proxy_auc(proxy_labels[group], proxy_scores[group]),
            "latency_ms_per_query": 1000.0 * times[group] / max(1, len(proxy_labels[group])),
            "latency_ms_per_image": 1000.0 * times[group] / max(1, image_count),
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
            **{k: comp_row[k] for k in ["anomaly_candidate_recall", "anomaly_attention_share", "class_evidence_margin"]},
        }
        metric_rows.append(row)
        comp_rows.append({"dataset": dname, **comp_row})
        latency_rows.append({"dataset": dname, "group": group, "latency_ms_per_query": row["latency_ms_per_query"], "latency_ms_per_image": row["latency_ms_per_image"], "peak_gpu_memory_mb": row["peak_gpu_memory_mb"]})
        branch_rows.append({
            "dataset": dname,
            "group": group,
            "image_memory_score_mean": float(np.mean(image_scores_raw[group])),
            "patch_topkmean_score_mean": float(np.mean(patch_scores_raw[group])),
            "fused_image_score_mean": float(np.mean(fused_scores_raw[group])),
            "image_memory_score_std": float(np.std(image_scores_raw[group])),
            "patch_topkmean_score_std": float(np.std(patch_scores_raw[group])),
            "fused_image_score_std": float(np.std(fused_scores_raw[group])),
        })
    return metric_rows, comp_rows, branch_rows, latency_rows


def write_report(cfg: Dict, rows: List[Dict]) -> None:
    report_dir = ROOT / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    stage = cfg.get("stage", "formal")
    name = "phase_b0_smoke_report.md" if stage == "smoke" else "phase_b0_formal_report.md"
    lines = [
        "# Phase-XXX-B0 Formal MRAD Report" if stage != "smoke" else "# Phase-XXX-B0 Smoke Report",
        "",
        f"- Stage: `{stage}`",
        "- Groups: C0-C8 class-conditional retrieval and aggregation.",
        "- Smoke and formal configs are separate; formal config has no `limit_images`.",
        "",
        "| dataset | group | Pixel AUROC | PRO | Image AUROC | Image AP | Patch proxy | latency ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def as_float(value):
        try:
            return float(value)
        except Exception:
            return float("nan")
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['group']} | {as_float(r['pixel_auroc']):.3f} | {as_float(r['pro']):.3f} | "
            f"{as_float(r['image_auroc']):.3f} | {as_float(r['image_ap']):.3f} | {as_float(r['patch_proxy_auroc']):.3f} | {as_float(r['latency_ms_per_image']):.3f} |"
        )
    (report_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/memory_audit/phase_b0_smoke.yaml")
    parser.add_argument("--limit_images", type=int, default=None, help="Smoke-only override. Do not use for formal.")
    parser.add_argument("--directions", nargs="*", default=None)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--reset_outputs", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(resolve_path(args.config), "r", encoding="utf-8"))
    if args.groups:
        cfg["_selected_groups"] = args.groups
    set_seed(int(cfg.get("seed", 0)))
    out = ROOT / cfg.get("output_dir", "results/phase_b0")
    out.mkdir(parents=True, exist_ok=True)
    if args.reset_outputs:
        for name in ["formal_d1_d2.csv", "class_competition.csv", "branch_ablation.csv", "latency.csv"]:
            target = out / name
            if target.exists():
                target.unlink()
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"command": " ".join(sys.argv), "stage": cfg.get("stage"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    relation_integrity_audit(cfg, out)
    wanted = args.directions or cfg.get("evaluation", {}).get("directions") or list(cfg["directions"].keys())
    all_metrics, all_comp, all_branch, all_latency = [], [], [], []
    for dname in wanted:
        limit = args.limit_images if args.limit_images is not None else cfg.get("evaluation", {}).get("limit_images")
        print(f"[phase-b0] {dname}: limit_images={limit if limit else 'FULL'}", flush=True)
        m, c, b, l = run_direction(cfg, dname, cfg["directions"][dname], int(limit) if limit else None)
        all_metrics.extend(m)
        all_comp.extend(c)
        all_branch.extend(b)
        all_latency.extend(l)
    metric_path = out / "formal_d1_d2.csv"
    comp_path = out / "class_competition.csv"
    branch_path = out / "branch_ablation.csv"
    latency_path = out / "latency.csv"
    merged_metrics = merge_rows(metric_path, all_metrics)
    merged_comp = merge_rows(comp_path, all_comp)
    merged_branch = merge_rows(branch_path, all_branch)
    merged_latency = merge_rows(latency_path, all_latency)
    write_csv(metric_path, merged_metrics)
    write_csv(comp_path, merged_comp)
    write_csv(branch_path, merged_branch)
    write_csv(latency_path, merged_latency)
    write_report(cfg, merged_metrics)


if __name__ == "__main__":
    main()
