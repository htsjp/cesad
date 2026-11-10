from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
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
from graph_memory import ActiveGraphRetriever, GraphMemoryBuilder, RetrievalDiagnostics
from models.mlp import AnomalyMLP, MLP, Projector, average_neighbor
from models.prompt_learner import AnomalyCLIP_PromptLearner
from mrad import compute_socre
from utils.dataset import Dataset
from utils.transforms import get_transform
from experiments.mrad_audit_utils import evaluate_results, init_results


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    payload = torch.load(path, map_location="cpu")
    return payload["keys"].float(), payload["values"].float()


def resolve_existing_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    rooted = ROOT / value
    if rooted.exists():
        return rooted
    return rooted


def default_metadata(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    n = values.shape[0]
    labels = values.argmax(dim=-1).long()
    pair_id = torch.arange(n, dtype=torch.long) // 2
    spatial = torch.arange(n, dtype=torch.long) % 16
    cluster = torch.arange(n, dtype=torch.long) % 64
    return {
        "node_id": torch.arange(n, dtype=torch.long),
        "binary_label": labels,
        "source_image_id": pair_id,
        "source_category_id": torch.zeros(n, dtype=torch.long),
        "region_type": labels,
        "spatial_bin": spatial,
        "scale_id": torch.zeros(n, dtype=torch.long),
        "cluster_id": cluster,
    }


def synthetic_queries(features: torch.Tensor, values: torch.Tensor, count: int, noise: float, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = features.shape[0]
    idx = torch.randperm(n, generator=gen)[: min(count, n)]
    eps = torch.randn((idx.numel(), features.shape[1]), generator=gen) * float(noise)
    q = F.normalize(features[idx] + eps, dim=-1)
    y = values[idx].argmax(dim=-1)
    return q, y, idx


def load_official_components(checkpoint: Path, device: torch.device):
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
    model.visual.DAPM_replace(DPAM_layer=24)
    return model, prompt_learner, image_proj, patch_proj, prompt_proj


def make_target_loader(cfg: Dict, dataset_name: str, limit: int):
    class Args:
        pass

    args = Args()
    args.image_size = int(cfg.get("image_size", 518))
    args.data_path = cfg["data_paths"][dataset_name]
    args.dataset = dataset_name
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=dataset_name)
    if limit:
        normals = [x for x in data.data_all if int(x.get("anomaly", 0)) == 0]
        anomalies = [x for x in data.data_all if int(x.get("anomaly", 0)) == 1]
        half = max(1, int(limit) // 2)
        selected = normals[:half] + anomalies[: max(0, int(limit) - half)]
        if len(selected) < int(limit):
            selected = data.data_all[: int(limit)]
        data.data_all = selected
        data.length = len(data.data_all)
    return torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0), data.obj_list


def tokens_from_patch_logits(logits: torch.Tensor, patch_projection: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    anomaly_probs = logits[:, :, 1]
    anomaly_area = (anomaly_probs > float(threshold)).float()
    normal_area = 1.0 - anomaly_area
    anomaly_mask = anomaly_area.unsqueeze(-1)
    normal_mask = normal_area.unsqueeze(-1)
    anomaly_count = anomaly_mask.sum(dim=1).clamp(min=1.0)
    normal_count = normal_mask.sum(dim=1).clamp(min=1.0)
    anomaly_token = (patch_projection * anomaly_mask).sum(dim=1) / anomaly_count
    normal_token = (patch_projection * normal_mask).sum(dim=1) / normal_count
    anomaly_token[anomaly_mask.sum(dim=1).squeeze(-1) == 0] = 0.0
    normal_token[normal_mask.sum(dim=1).squeeze(-1) == 0] = 0.0
    return torch.stack([normal_token, anomaly_token], dim=1)


def binary_metrics(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = labels.detach().cpu().numpy()
    s = scores.detach().cpu().numpy()
    out = {"image_ap": float(average_precision_score(y, s) * 100.0)}
    try:
        out["image_auroc"] = float(roc_auc_score(y, s) * 100.0)
    except ValueError:
        out["image_auroc"] = float("nan")
    out["pixel_auroc"] = float("nan")
    out["pro"] = float("nan")
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(out: Path, rows: List[Dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    groups = [r["group"] for r in rows]
    unique = [r["unique_selected_prototypes"] for r in rows]
    gini = [r["prototype_usage_gini"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(groups, unique, color="#4C78A8")
    ax1.set_ylabel("Unique selected prototypes")
    ax2 = ax1.twinx()
    ax2.plot(groups, gini, color="#F58518", marker="o")
    ax2.set_ylabel("Usage Gini")
    fig.tight_layout()
    fig.savefig(out / "retrieval_diagnostics.png", dpi=180)
    plt.close(fig)


def evaluate_real_group(cfg: Dict, dcfg: Dict, group: str, rcfg: Dict, memory, labels_cpu, g1_selected, args, components, image_cache, loader_bundle):
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model, prompt_learner, image_proj, _patch_proj, prompt_proj = components
    image_keys, image_values = image_cache
    loader, obj_list = loader_bundle
    retriever = ActiveGraphRetriever(rcfg)
    diagnostics = RetrievalDiagnostics(memory_size=memory.features.shape[0])
    results = init_results(obj_list)
    fusion_k = float(cfg.get("fusion_k", 0.7))
    sigma = float(cfg.get("sigma", 4))
    image_size = int(cfg.get("image_size", 518))
    features_list = cfg.get("features_list", [6, 12, 18, 24])
    started = time.perf_counter()
    image_count = 0

    with torch.no_grad():
        for items in loader:
            image_count += 1
            image = items["img"].to(device)
            cls_name = items["cls_name"][0]
            cls_id = items["cls_id"]
            gt_mask = items["img_mask"]
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results[cls_name]["imgs_masks"].append(gt_mask)
            results[cls_name]["gt_sp"].extend(items["anomaly"].detach().cpu())

            image_features, patch_features, _all_cls_tokens, patch_projections = model.encode_image(image, features_list, DPAM_layer=24)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cache_logits, _ = compute_socre(image_features, image_keys, image_values, device, proj=image_proj, use_proj=True)

            patch_projection = average_neighbor(patch_projections[3])
            patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
            patch_query = average_neighbor(patch_features[3])
            patch_query = patch_query / patch_query.norm(dim=-1, keepdim=True)
            flat_query = patch_query.reshape(-1, patch_query.shape[-1]).detach().cpu()
            output = retriever.retrieve(flat_query, memory)
            patch_logits = output.logits.reshape(1, patch_query.shape[1], 2).to(device)
            diagnostics.update(
                range(flat_query.shape[0]),
                output.selected_ids,
                output.scores,
                output.relations,
                labels=labels_cpu,
                baseline_selected=g1_selected,
                fallback_events=output.fallback_events,
            )

            bias_tokens = tokens_from_patch_logits(patch_logits, patch_projection)
            bias = prompt_proj(bias_tokens[:, 0, :], bias_tokens[:, 1, :])
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_projections):
                if idx >= 3:
                    patch_feature = average_neighbor(patch_feature)
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity, image_size)
                    anomaly_map_list.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in anomaly_map.detach().cpu()], dim=0).sum(dim=0)
            flat = anomaly_map.flatten()
            topn = max(1, int(flat.shape[0] * 0.01))
            topk_mean = torch.topk(flat, topn).values.mean().to(device)
            image_score = (1 - fusion_k) * topk_mean + fusion_k * cache_logits[:, 1]
            results[cls_name]["pr_sp"].extend(image_score.detach().cpu())
            results[cls_name]["anomaly_maps"].append(anomaly_map.detach().cpu())

    metrics = evaluate_results(results, obj_list, compute_pro=bool(cfg.get("compute_pixel_aupro", False)))
    mean = metrics["mean"]
    summary = {
        "group": group,
        "mode": rcfg["mode"],
        "image_count": image_count,
        "image_auroc": mean.get("I-AUROC", float("nan")),
        "image_ap": mean.get("I-AP", float("nan")),
        "pixel_auroc": mean.get("P-AUROC", float("nan")),
        "pro": mean.get("PRO", float("nan")),
        "mean_latency_per_image_sec": (time.perf_counter() - started) / max(1, image_count),
        "similarity_evaluations_per_query": int(memory.features.shape[0]),
        **diagnostics.summarize(),
        "_selected_nodes": diagnostics.previous_selected and output.selected_ids,
    }
    return summary


def run_direction(cfg: Dict, dname: str, dcfg: Dict, args) -> List[Dict]:
    out = Path(cfg["output_dir"]) / dname
    out.mkdir(parents=True, exist_ok=True)
    print(f"[graph-audit] {dname}: loading patch cache {dcfg['patch_cache']}", flush=True)
    features, values = load_cache(resolve_existing_path(dcfg["patch_cache"]))
    metadata = default_metadata(values)
    builder = GraphMemoryBuilder(
        max_neighbors_per_relation=cfg.get("graph", {}).get("max_neighbors_per_relation", 16),
        knn_k=cfg.get("graph", {}).get("knn_k", 16),
        seed=cfg.get("seed", 0),
    )
    started = time.perf_counter()
    print(f"[graph-audit] {dname}: building graph for {features.shape[0]} nodes", flush=True)
    memory = builder.build(features, values, metadata)
    graph_time = time.perf_counter() - started

    q, labels, query_ids = synthetic_queries(
        memory.features,
        memory.values,
        int(cfg.get("smoke", {}).get("synthetic_queries", args.synthetic_queries)),
        float(cfg.get("smoke", {}).get("noise", 0.03)),
        int(cfg.get("seed", 0)),
    )
    if args.limit:
        q, labels, query_ids = q[: args.limit], labels[: args.limit], query_ids[: args.limit]

    rows = []
    g1_selected = None
    groups = cfg.get("groups", {})
    selected_groups = args.groups or ["G0", "G1", "G2", "G3", "G4", "G5"]
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    print(f"[graph-audit] {dname}: loading checkpoint/model {dcfg['checkpoint']}", flush=True)
    components = load_official_components(resolve_existing_path(dcfg["checkpoint"]), device)
    image_cache = load_cache(resolve_existing_path(dcfg["image_cache"]))
    image_cache = (image_cache[0].to(device), image_cache[1].to(device))
    loader_bundle = make_target_loader(cfg, dcfg["target"], args.limit or int(cfg.get("smoke", {}).get("real_images", 4)))
    for group in selected_groups:
        print(f"[graph-audit] {dname}: running {group}", flush=True)
        checkpoint = out / f"{group}_checkpoint.json"
        if checkpoint.exists() and not args.force:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            rows.append(payload)
            if group == "G1":
                g1_selected = payload.get("_selected_nodes")
            continue
        rcfg = dict(cfg.get("retrieval", {}))
        rcfg.update(groups.get(group, {}))
        started = time.perf_counter()
        result = ActiveGraphRetriever(rcfg).retrieve(q, memory)
        latency = (time.perf_counter() - started) / max(1, q.shape[0])
        diagnostics = RetrievalDiagnostics(memory_size=memory.features.shape[0])
        diagnostics.update(
            query_ids.tolist(),
            result.selected_ids,
            result.scores,
            result.relations,
            labels=memory.metadata["binary_label"].tolist(),
            baseline_selected=g1_selected,
            fallback_events=result.fallback_events,
        )
        metric = binary_metrics(result.logits[:, 1], labels)
        summary = {
            "direction": dname,
            "group": group,
            "mode": rcfg["mode"],
            "graph_construction_time_sec": graph_time,
            "mean_latency_per_query_sec": latency,
            "similarity_evaluations_per_query": int(memory.features.shape[0]),
            **metric,
            **diagnostics.summarize(),
            "_selected_nodes": result.selected_ids,
        }
        checkpoint.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rows.append(summary)
        if group == "G1":
            g1_selected = result.selected_ids
    public_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (out / "results.json").write_text(json.dumps(public_rows, indent=2), encoding="utf-8")
    write_csv(out / "results.csv", public_rows)
    save_plots(out, public_rows)
    return public_rows


def run_real_direction(cfg: Dict, dname: str, dcfg: Dict, args) -> List[Dict]:
    out = Path(cfg["output_dir"]) / dname
    out.mkdir(parents=True, exist_ok=True)
    eval_name = "real-smoke" if cfg.get("stage") == "smoke" else "mechanism"
    print(f"[graph-audit] {dname}: loading {eval_name} patch cache {dcfg['patch_cache']}", flush=True)
    features, values = load_cache(resolve_existing_path(dcfg["patch_cache"]))
    metadata = default_metadata(values)
    builder = GraphMemoryBuilder(
        max_neighbors_per_relation=cfg.get("graph", {}).get("max_neighbors_per_relation", 16),
        knn_k=cfg.get("graph", {}).get("knn_k", 16),
        seed=cfg.get("seed", 0),
    )
    print(f"[graph-audit] {dname}: building graph for {features.shape[0]} nodes", flush=True)
    started_graph = time.perf_counter()
    memory = builder.build(features, values, metadata)
    graph_time = time.perf_counter() - started_graph
    labels_cpu = memory.metadata["binary_label"].tolist()
    rows = []
    g1_selected = None
    groups = cfg.get("groups", {})
    selected_groups = args.groups or ["G0", "G1", "G2", "G3", "G4", "G5"]
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    print(f"[graph-audit] {dname}: loading checkpoint/model {dcfg['checkpoint']}", flush=True)
    components = load_official_components(resolve_existing_path(dcfg["checkpoint"]), device)
    image_cache = load_cache(resolve_existing_path(dcfg["image_cache"]))
    image_cache = (image_cache[0].to(device), image_cache[1].to(device))
    if args.limit is not None:
        target_limit = int(args.limit)
    elif cfg.get("stage") == "smoke":
        target_limit = int(cfg.get("smoke", {}).get("real_images", 4))
    else:
        target_limit = 0
    loader_bundle = make_target_loader(cfg, dcfg["target"], target_limit)
    for group in selected_groups:
        print(f"[graph-audit] {dname}: {eval_name} running {group}", flush=True)
        checkpoint = out / f"{group}_real_checkpoint.json"
        if checkpoint.exists() and not args.force:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            rows.append(payload)
            if group == "G1":
                g1_selected = payload.get("_selected_nodes")
            continue
        rcfg = dict(cfg.get("retrieval", {}))
        rcfg.update(groups.get(group, {}))
        summary = evaluate_real_group(cfg, dcfg, group, rcfg, memory, labels_cpu, g1_selected, args, components, image_cache, loader_bundle)
        summary["direction"] = dname
        summary["graph_construction_time_sec"] = graph_time
        checkpoint.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rows.append(summary)
        if group == "G1":
            g1_selected = summary.get("_selected_nodes")
    public_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (out / "real_results.json").write_text(json.dumps(public_rows, indent=2), encoding="utf-8")
    write_csv(out / "real_results.csv", public_rows)
    save_plots(out, public_rows)
    return public_rows


def write_report(cfg: Dict, all_rows: List[Dict], args) -> None:
    out = Path(cfg["output_dir"])
    lines = [
        "# Active Graph Reconstruction Audit",
        "",
        "This smoke run validates graph construction, G0-G5 routing, fallback accounting, and machine-readable outputs. It does not claim full MRAD benchmark performance.",
        "",
        "## Commands",
        "",
        f"- Real smoke: `python scripts/run_graph_reconstruction_audit.py --config {args.config} --stage smoke --limit {args.limit or cfg.get('smoke', {}).get('real_images', 4)} --groups G0 G1 G2 G3 G4 G5`",
        f"- Synthetic developer check: `python scripts/run_graph_reconstruction_audit.py --config {args.config} --stage smoke --synthetic --limit 32 --groups G0 G1 G2 G3 G4 G5`",
        "- Mechanism test: `python scripts/run_graph_reconstruction_audit.py --config configs/graph_reconstruction.yaml --stage mechanism --groups G0 G1 G2 G3 G4 G5`",
        "",
        "## Environment",
        "",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Torch: `{torch.__version__}`",
        f"- CUDA available: `{torch.cuda.is_available()}`",
        "- Git branch: unavailable in this workspace because no `.git` directory exists.",
        "",
        "## Files Changed",
        "",
        "- `graph_memory/graph_builder.py`",
        "- `graph_memory/relation_index.py`",
        "- `graph_memory/active_router.py`",
        "- `graph_memory/prototype_reconstructor.py`",
        "- `graph_memory/diagnostics.py`",
        "- `configs/graph_reconstruction.yaml`",
        "- `scripts/run_graph_reconstruction_audit.py`",
        "- `scripts/compare_retrieval_groups.py`",
        "- `tests/test_graph_memory.py`",
        "",
        "## Results",
        "",
        "| direction | group | AUROC | AP | unique selected | gini | fallback rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['direction']} | {row['group']} | {row['image_auroc']:.3f} | {row['image_ap']:.3f} | "
            f"{row['unique_selected_prototypes']} | {row['prototype_usage_gini']:.4f} | {row['fallback_rate']:.4f} |"
        )
    lines += [
        "",
        "## Failures And Questions",
        "",
        "- Small-subset smoke metrics are only for shape/numerical/logging validation; they are not benchmark claims.",
        "- The available patch cache does not include original source image/category/spatial metadata, so smoke metadata is deterministic proxy metadata derived without target labels.",
        "- G6-G8 are configured but should be run after G5 clears the mechanism gate.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/graph_reconstruction.yaml")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--synthetic_queries", type=int, default=64)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run the old cache/synthetic query developer check instead of real target images.")
    parser.add_argument("--directions", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = resolve_existing_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    if args.stage:
        cfg["stage"] = args.stage
    set_seed(int(cfg.get("seed", 0)))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_metadata.json").write_text(json.dumps({"command": " ".join(sys.argv), "config": str(config_path), "stage": cfg.get("stage"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    all_rows = []
    directions = list(cfg["directions"].items())
    if args.directions:
        wanted = set(args.directions)
        directions = [(k, v) for k, v in directions if k in wanted]
    elif cfg.get("stage") == "smoke" and not args.synthetic:
        directions = directions[:1]
    for dname, dcfg in directions:
        if args.synthetic:
            all_rows.extend(run_direction(cfg, dname, dcfg, args))
        else:
            all_rows.extend(run_real_direction(cfg, dname, dcfg, args))
    write_csv(out / "all_results.csv", all_rows)
    write_report(cfg, all_rows, args)


if __name__ == "__main__":
    main()
