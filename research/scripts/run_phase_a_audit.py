from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from graph_memory import GraphMemoryBuilder
from memory_audit.phase_a import (
    RELATION_GROUPS,
    active_g5,
    adaptive_balanced_topk,
    balanced_topk,
    finite_ap,
    finite_auc,
    fallback_breakdown,
    relation_only,
    summarize_output,
    topk_direct,
)
from models.mlp import average_neighbor
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
    return torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0)


def load_visual_model(cfg: Dict, checkpoint: Path, device: torch.device):
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    # Load checkpoint to document the exact MRAD checkpoint without changing prompt branch.
    torch.load(checkpoint, map_location=device)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.visual.DAPM_replace(DPAM_layer=24)
    return model


class StreamingSummary:
    def __init__(self, dataset: str, group: str, seed: int, memory_budget: int):
        self.dataset = dataset
        self.group = group
        self.seed = seed
        self.memory_budget = memory_budget
        self.labels: List[int] = []
        self.scores: List[float] = []
        self.top1 = Counter()
        self.selected = Counter()
        self.fallbacks = Counter()
        self.normal_counts: List[int] = []
        self.anomaly_counts: List[int] = []
        self.jaccards: List[float] = []
        self.previous_selected = None
        self.total_seconds = 0.0
        self.patch_count = 0

    def update(self, labels: List[int], output, seconds: float):
        self.labels.extend(int(x) for x in labels)
        self.scores.extend(float(x) for x in output.scores.tolist())
        self.fallbacks.update(output.fallback_types)
        self.normal_counts.extend(output.normal_counts)
        self.anomaly_counts.extend(output.anomaly_counts)
        self.total_seconds += float(seconds)
        self.patch_count += len(labels)
        for row in output.selected_ids:
            if row:
                self.top1[int(row[0])] += 1
            self.selected.update(int(x) for x in row)
            cur = set(int(x) for x in row)
            if self.previous_selected is not None:
                self.jaccards.append(len(self.previous_selected & cur) / max(1, len(self.previous_selected | cur)))
            self.previous_selected = cur

    def row(self) -> Dict:
        selected_counts = np.asarray(list(self.selected.values()), dtype=np.float64)
        if selected_counts.size and selected_counts.sum():
            p = selected_counts / selected_counts.sum()
            entropy = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
            sorted_counts = np.sort(selected_counts)
            n = sorted_counts.size
            gini = float((2 * np.arange(1, n + 1).dot(sorted_counts) / sorted_counts.sum() - (n + 1)) / n)
        else:
            entropy = 0.0
            gini = 0.0
        return {
            "dataset": self.dataset,
            "group": self.group,
            "seed": self.seed,
            "memory_budget": self.memory_budget,
            "image_auroc": float("nan"),
            "image_ap": float("nan"),
            "pixel_auroc": finite_auc(self.labels, self.scores),
            "pro": float("nan"),
            "unique_top1": len(self.top1),
            "unique_selected": len(self.selected),
            "gini": gini,
            "entropy": entropy,
            "jaccard": float(np.mean(self.jaccards)) if self.jaccards else 0.0,
            "fallback_rate": float(sum(v for k, v in self.fallbacks.items() if k != "none") / max(1, self.patch_count)),
            "latency_ms": 1000.0 * self.total_seconds / max(1, self.patch_count),
            "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
            "patch_count": self.patch_count,
            "selected_normal_mean": float(np.mean(self.normal_counts)) if self.normal_counts else 0.0,
            "selected_anomaly_mean": float(np.mean(self.anomaly_counts)) if self.anomaly_counts else 0.0,
        }


class FallbackAccumulator:
    def __init__(self, dataset: str, group: str):
        self.dataset = dataset
        self.group = group
        self.rows = defaultdict(lambda: {
            "labels": [],
            "scores": [],
            "baseline_scores": [],
            "normal_counts": [],
            "anomaly_counts": [],
            "relations": Counter(),
        })

    def update(self, labels: List[int], output, baseline_scores: List[float]):
        for i, fb in enumerate(output.fallback_types):
            targets = [fb]
            if fb != "none":
                targets.append("fallback_all")
            for target in targets:
                row = self.rows[target]
                row["labels"].append(int(labels[i]))
                row["scores"].append(float(output.scores[i]))
                row["baseline_scores"].append(float(baseline_scores[i]))
                row["normal_counts"].append(output.normal_counts[i])
                row["anomaly_counts"].append(output.anomaly_counts[i])
                row["relations"].update(output.relations[i])

    def summarize(self) -> List[Dict]:
        out = []
        for fb in ["none", "missing_normal", "missing_anomaly", "missing_both", "numerical_fallback", "fallback_all"]:
            row = self.rows.get(fb)
            if not row or not row["labels"]:
                out.append({"dataset": self.dataset, "group": self.group, "fallback_type": fb, "query_count": 0})
                continue
            scores = np.asarray(row["scores"], dtype=np.float64)
            baseline = np.asarray(row["baseline_scores"], dtype=np.float64)
            out.append({
                "dataset": self.dataset,
                "group": self.group,
                "fallback_type": fb,
                "query_count": len(row["labels"]),
                "pixel_auroc_proxy": finite_auc(row["labels"], row["scores"]),
                "patch_ap": finite_ap(row["labels"], row["scores"]),
                "score_difference_vs_g1": float(np.mean(scores - baseline)),
                "selected_normal_count": float(np.mean(row["normal_counts"])),
                "selected_anomaly_count": float(np.mean(row["anomaly_counts"])),
                "relation_branch": dict(row["relations"]),
            })
        return out


@torch.no_grad()
def iter_patch_queries(model, loader, cfg: Dict, device: torch.device):
    features_list = cfg.get("features_list", [6, 12, 18, 24])
    for image_idx, items in enumerate(loader):
        image = items["img"].to(device)
        gt = items["img_mask"].to(device).float()
        _, patch_features, _, _ = model.encode_image(image, features_list, DPAM_layer=24)
        patch_query = average_neighbor(patch_features[3])
        patch_query = F.normalize(patch_query, dim=-1)
        side = int(patch_query.shape[1] ** 0.5)
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        gt_small = F.interpolate(gt, size=(side, side), mode="bilinear", align_corners=False).squeeze(1)
        patch_labels = (gt_small.reshape(-1) > 0.5).long().cpu()
        yield patch_query.reshape(-1, patch_query.shape[-1]).cpu(), patch_labels.tolist()


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_direction(cfg: Dict, dname: str, dcfg: Dict, out: Path, limit_override: int | None):
    seed = int(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    limit = limit_override
    if limit is None:
        limit = cfg.get("evaluation", {}).get("limit_images")
    print(f"[phase-a] {dname}: limit_images={limit if limit else 'FULL'}", flush=True)

    keys, values = load_cache(resolve_path(dcfg["patch_cache"]))
    memory = GraphMemoryBuilder(
        max_neighbors_per_relation=cfg.get("graph", {}).get("max_neighbors_per_relation", 16),
        knn_k=cfg.get("graph", {}).get("knn_k", 16),
        seed=seed,
    ).build(keys, values, default_metadata(values))
    model = load_visual_model(cfg, resolve_path(dcfg["checkpoint"]), device)
    loader = make_loader(cfg, dcfg["target"], int(limit) if limit else None)
    retrieval_cfg = dict(cfg.get("retrieval", {}))
    memory_budget = int(memory.features.shape[0])
    group_names = ["G1_top8", "G5_old", "B1_top8", "B2_n4_a4", "B3_n2_a6", "B4_n6_a2", "B5_adaptive"] + list(RELATION_GROUPS.keys())
    summaries = {name: StreamingSummary(dname, name, seed, memory_budget) for name in group_names}
    fallback_acc = FallbackAccumulator(dname, "G5_old")

    for image_idx, (queries, patch_labels) in enumerate(iter_patch_queries(model, loader, cfg, device), start=1):
        if image_idx == 1 or image_idx % 50 == 0:
            print(f"[phase-a] {dname}: streaming image {image_idx}", flush=True)

        start = time.perf_counter()
        g1_output = topk_direct(queries, memory, int(retrieval_cfg.get("k2", 8)), retrieval_cfg.get("temperature", 1.0))
        summaries["G1_top8"].update(patch_labels, g1_output, time.perf_counter() - start)

        start = time.perf_counter()
        g5_output = active_g5(queries, memory, {**retrieval_cfg, "mode": "active_graph"})
        summaries["G5_old"].update(patch_labels, g5_output, time.perf_counter() - start)
        fallback_acc.update(patch_labels, g5_output, g1_output.scores.tolist())

        specs = [
            ("B1_top8", lambda: topk_direct(queries, memory, 8, retrieval_cfg.get("temperature", 1.0))),
            ("B2_n4_a4", lambda: balanced_topk(queries, memory, 4, 4, retrieval_cfg.get("temperature", 1.0))),
            ("B3_n2_a6", lambda: balanced_topk(queries, memory, 2, 6, retrieval_cfg.get("temperature", 1.0))),
            ("B4_n6_a2", lambda: balanced_topk(queries, memory, 6, 2, retrieval_cfg.get("temperature", 1.0))),
            ("B5_adaptive", lambda: adaptive_balanced_topk(queries, memory, 8, retrieval_cfg.get("temperature", 1.0))),
        ]
        for name, fn in specs:
            start = time.perf_counter()
            summaries[name].update(patch_labels, fn(), time.perf_counter() - start)

        for name, relation in RELATION_GROUPS.items():
            start = time.perf_counter()
            summaries[name].update(patch_labels, relation_only(queries, memory, relation, retrieval_cfg), time.perf_counter() - start)

        del queries
        if torch.cuda.is_available() and image_idx % 25 == 0:
            torch.cuda.empty_cache()

    summary_rows = [summaries[name].row() for name in group_names]
    balanced_rows = [summaries[name].row() for name in ["B1_top8", "B2_n4_a4", "B3_n2_a6", "B4_n6_a2", "B5_adaptive"]]
    relation_rows = [summaries[name].row() for name in RELATION_GROUPS.keys()]
    fallback_rows = fallback_acc.summarize()
    return summary_rows, fallback_rows, balanced_rows, relation_rows


def write_report(cfg: Dict, out: Path, summary_rows: List[Dict], command: str) -> None:
    report_dir = ROOT / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    stage = cfg.get("stage", "unknown")
    lines = [
        "# Phase-XXX-A Sprint 1 Smoke Report" if stage == "smoke" else "# Phase-XXX-A Sprint 1 Report",
        "",
        f"- Stage: `{stage}`",
        f"- Command: `{command}`",
        "- Scope: fallback decomposition, balanced Top-k, and relation-only patch ranking proxy.",
        "- Formal config is separate from smoke config; no smoke image limit is present in `phase_a_formal.yaml`.",
        "",
        "| dataset | group | patch AUROC | fallback | unique selected | latency ms/query |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['dataset']} | {r['group']} | {r['pixel_auroc']:.3f} | {r['fallback_rate']:.4f} | "
            f"{r['unique_selected']} | {r['latency_ms']:.4f} |"
        )
    target = report_dir / ("phase_a_smoke.md" if stage == "smoke" else "phase_a_decision.md")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/memory_audit/phase_a_smoke.yaml")
    parser.add_argument("--limit_images", type=int, default=None, help="Smoke-only override. Do not use for formal runs.")
    parser.add_argument("--directions", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(resolve_path(args.config), "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 0)))
    out = ROOT / cfg.get("output_dir", "results/phase_a")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"command": " ".join(sys.argv), "stage": cfg.get("stage"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")

    wanted = args.directions or cfg.get("evaluation", {}).get("directions") or list(cfg["directions"].keys())
    summary_rows, fallback_rows, balanced_rows, relation_rows = [], [], [], []
    for dname in wanted:
        s, f, b, r = run_direction(cfg, dname, cfg["directions"][dname], out, args.limit_images)
        summary_rows.extend(s)
        fallback_rows.extend(f)
        balanced_rows.extend(b)
        relation_rows.extend(r)

    write_csv(out / "phase_a_summary.csv", summary_rows)
    write_csv(out / "fallback_breakdown.csv", fallback_rows)
    write_csv(out / "balanced_retrieval.csv", balanced_rows)
    write_csv(out / "relation_vs_random.csv", relation_rows)
    write_csv(out / "latency_profile.csv", summary_rows)
    write_report(cfg, out, summary_rows, " ".join(sys.argv))


if __name__ == "__main__":
    main()


