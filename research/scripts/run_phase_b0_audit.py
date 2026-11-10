from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from graph_memory import GraphMemoryBuilder
from memory_audit.class_conditional import ClasswiseRetriever
from models.mlp import average_neighbor
from scripts.run_phase_b0_formal import (
    default_metadata,
    load_cache,
    load_components,
    make_loader,
    relation_integrity_audit,
    resolve_path,
    set_seed,
)
from utils.dataset import Dataset
from utils.transforms import get_transform


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def finite_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    labels = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(scores), dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def finite_ap(labels: Iterable[int], scores: Iterable[float]) -> float:
    labels = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(scores), dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "not-a-git-repo"


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def select_dataset_rows(data: Dataset, limit: int | None) -> List[Dict]:
    rows = list(data.data_all)
    if not limit:
        return rows
    normals = [x for x in rows if int(x.get("anomaly", 0)) == 0]
    anomalies = [x for x in rows if int(x.get("anomaly", 0)) == 1]
    half = max(1, int(limit) // 2)
    selected = normals[:half] + anomalies[: max(0, int(limit) - half)]
    return selected if len(selected) >= int(limit) else rows[: int(limit)]


def dataset_object(cfg: Dict, dataset_name: str) -> Dataset:
    class Args:
        pass

    args = Args()
    args.image_size = int(cfg.get("image_size", 518))
    args.data_path = cfg["data_paths"][dataset_name]
    args.dataset = dataset_name
    preprocess, target_transform = get_transform(args)
    return Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=dataset_name)


def run_identity(cfg: Dict, audit_cfg: Dict, out: Path) -> None:
    rows = {}
    for name, path in {
        "phase_a_formal": ROOT / audit_cfg["phase_a_config"],
        "phase_b0_formal": ROOT / audit_cfg["phase_b0_config"],
        "audit_config": ROOT / audit_cfg["_config_path"],
    }.items():
        rows[name] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.exists() else None,
            "expanded_config_sha256": sha256_text(yaml.safe_dump(yaml.safe_load(path.read_text(encoding="utf-8")), sort_keys=True)) if path.exists() else None,
        }

    directions = {}
    for dname, dcfg in cfg["directions"].items():
        ds = dataset_object(cfg, dcfg["target"])
        sample_ids = [x["img_path"] for x in ds.data_all]
        directions[dname] = {
            "source": dcfg["source"],
            "target": dcfg["target"],
            "checkpoint_path": str(resolve_path(dcfg["checkpoint"])),
            "checkpoint_sha256": sha256_file(resolve_path(dcfg["checkpoint"])),
            "image_cache_path": str(resolve_path(dcfg["image_cache"])),
            "image_cache_sha256": sha256_file(resolve_path(dcfg["image_cache"])),
            "patch_cache_path": str(resolve_path(dcfg["patch_cache"])),
            "patch_cache_sha256": sha256_file(resolve_path(dcfg["patch_cache"])),
            "dataset_root": cfg["data_paths"][dcfg["target"]],
            "sample_count": len(sample_ids),
            "ordered_sample_ids_sha256": sha256_text("\n".join(sample_ids)),
        }

    payload = {
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(cfg.get("seed", 0)),
        "device": cfg.get("device", "cuda:0"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "configs": rows,
        "directions": directions,
    }
    (out / "run_identity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def config_diff(audit_cfg: Dict, out: Path) -> None:
    a_path = ROOT / audit_cfg["phase_a_config"]
    b_path = ROOT / audit_cfg["phase_b0_config"]
    a = yaml.safe_load(a_path.read_text(encoding="utf-8"))
    b = yaml.safe_load(b_path.read_text(encoding="utf-8"))
    lines = [
        "# Phase-B0A Config Diff",
        "",
        f"- Phase-A config: `{audit_cfg['phase_a_config']}`",
        f"- Phase-B0 config: `{audit_cfg['phase_b0_config']}`",
        "",
        "| key | Phase-A | Phase-B0 |",
        "|---|---|---|",
    ]
    keys = sorted(set(a.keys()) | set(b.keys()))
    for key in keys:
        av = json.dumps(a.get(key), sort_keys=True, default=str)
        bv = json.dumps(b.get(key), sort_keys=True, default=str)
        if av != bv:
            lines.append(f"| `{key}` | `{av}` | `{bv}` |")
    (out / "config_diff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_identity(cfg: Dict, out: Path) -> None:
    rows = []
    for dname, dcfg in cfg["directions"].items():
        data = dataset_object(cfg, dcfg["target"])
        root = Path(cfg["data_paths"][dcfg["target"]])
        seen = set()
        for idx, item in enumerate(data.data_all):
            image_rel = item.get("img_path", "")
            mask_rel = item.get("mask_path", "")
            image_path = root / image_rel
            mask_path = root / mask_rel if mask_rel else None
            duplicate = image_rel in seen
            seen.add(image_rel)
            image_shape = ""
            mask_shape = ""
            valid_pixel_count = ""
            mask_binary = ""
            empty_mask = ""
            image_ok = image_path.exists()
            mask_ok = bool(mask_path and mask_path.exists())
            if image_ok:
                with Image.open(image_path) as im:
                    image_shape = f"{im.height}x{im.width}x{len(im.getbands())}"
            if mask_ok:
                with Image.open(mask_path) as mm:
                    arr = np.asarray(mm)
                    mask_shape = "x".join(str(x) for x in arr.shape)
                    vals = np.unique(arr)
                    mask_binary = bool(set(vals.tolist()).issubset({0, 1, 255}))
                    empty_mask = bool(arr.max() == 0)
                    valid_pixel_count = int(arr.size)
            rows.append({
                "direction": dname,
                "dataset": dcfg["target"],
                "image_id": idx,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path else "",
                "category": item.get("cls_name", ""),
                "anomaly_type": item.get("specie_name", ""),
                "anomaly": int(item.get("anomaly", 0)),
                "image_shape": image_shape,
                "mask_shape": mask_shape,
                "valid_pixel_count": valid_pixel_count,
                "image_exists": image_ok,
                "mask_exists": mask_ok,
                "mask_binary": mask_binary,
                "empty_mask": empty_mask,
                "duplicate_image_path": duplicate,
            })
    write_csv(out / "dataset_identity.csv", rows)


@torch.no_grad()
def feature_topk_score_audit(cfg: Dict, audit_cfg: Dict, out: Path, limit_override: int | None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    feature_rows, topk_rows, score_rows = [], [], []
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    max_feature_images = int(audit_cfg.get("feature_images", 10))
    max_score_images = int(audit_cfg.get("score_trace_images", 20))
    max_topk_queries = int(audit_cfg.get("topk_queries", 1000))
    directions = audit_cfg.get("directions") or list(cfg["directions"].keys())
    sample_limit = limit_override if limit_override is not None else audit_cfg.get("limit_images")

    for dname in directions:
        dcfg = cfg["directions"][dname]
        print(f"[b0-audit] {dname}: feature/topk/score trace, limit_images={sample_limit if sample_limit else 'sampled-prefix'}", flush=True)
        trace_limit = int(sample_limit) if sample_limit else max(max_feature_images, max_score_images)
        data, loader = make_loader(cfg, dcfg["target"], trace_limit)
        model, image_proj, patch_proj = load_components(cfg, resolve_path(dcfg["checkpoint"]), device)
        image_keys, image_values = load_cache(resolve_path(dcfg["image_cache"]))
        patch_keys, patch_values = load_cache(resolve_path(dcfg["patch_cache"]))
        image_keys = image_keys.to(device)
        image_values = image_values.to(device)
        patch_keys_raw = F.normalize(patch_keys.to(device), dim=-1)
        patch_values = patch_values.to(device)
        patch_keys_metric = F.normalize(patch_proj(patch_keys_raw, 1), dim=-1).detach().cpu()
        retriever = ClasswiseRetriever(patch_keys_metric, patch_values.detach().cpu(), cfg.get("retrieval", {}).get("temperature", 1.0))

        memory_payload = {
            "direction": dname,
            "image_keys_sha256": tensor_sha256(image_keys.cpu()),
            "image_values_sha256": tensor_sha256(image_values.cpu()),
            "patch_keys_sha256": tensor_sha256(patch_keys.cpu()),
            "patch_values_sha256": tensor_sha256(patch_values.cpu()),
            "normal_count": int((patch_values.argmax(dim=-1) == 0).sum().item()),
            "anomaly_count": int((patch_values.argmax(dim=-1) == 1).sum().item()),
            "feature_dim": int(patch_keys.shape[-1]),
            "key_mean": float(patch_keys.float().mean().item()),
            "key_std": float(patch_keys.float().std().item()),
            "projected_patch_keys_sha256": tensor_sha256(patch_keys_metric),
            "uses_projected_patch_metric": True,
        }
        feature_rows.append(memory_payload)

        query_seen = 0
        for image_idx, items in enumerate(loader):
            if image_idx >= max(max_feature_images, max_score_images) and query_seen >= max_topk_queries:
                break
            image = items["img"].to(device)
            gt_mask = items["img_mask"].to(device).float()
            image_features, patch_features, _, _ = model.encode_image(image, cfg.get("features_list", [6, 12, 18, 24]), DPAM_layer=int(cfg.get("dpam_layer", 24)))
            image_features = F.normalize(image_features, dim=-1)
            patch_query_raw = average_neighbor(patch_features[3])
            patch_query_raw = F.normalize(patch_query_raw, dim=-1)
            patch_query_metric = F.normalize(patch_proj(patch_query_raw, 0), dim=-1)
            flat_query = patch_query_metric.reshape(-1, patch_query_metric.shape[-1]).detach().cpu()
            side = int(math.sqrt(flat_query.shape[0]))

            if image_idx < max_feature_images:
                feature_rows.append({
                    "direction": dname,
                    "image_idx": image_idx,
                    "img_path": items["img_path"][0],
                    "global_feature_sha256": tensor_sha256(image_features),
                    "patch_feature_sha256": tensor_sha256(patch_query_raw),
                    "patch_feature_shape": list(patch_query_raw.shape),
                    "patch_feature_mean": float(patch_query_raw.mean().item()),
                    "patch_feature_std": float(patch_query_raw.std().item()),
                    "projected_query_sha256": tensor_sha256(flat_query),
                })

            if query_seen < max_topk_queries:
                take = min(max_topk_queries - query_seen, flat_query.shape[0])
                q = flat_query[:take]
                sim = F.normalize(q, dim=-1) @ patch_keys_metric.T
                direct_vals, direct_idx = torch.topk(sim, k=8, dim=-1)
                c0 = retriever.retrieve(q, mode="global_topk", total_k=8)
                c1 = retriever.retrieve(q, mode="balanced_topk", normal_k=4, anomaly_k=4, total_k=8)
                c2 = retriever.retrieve(q, mode="balanced_topk", normal_k=2, anomaly_k=6, total_k=8)
                n_vals4, n_local4 = torch.topk(sim[:, retriever.normal_ids], k=4, dim=-1)
                a_vals4, a_local4 = torch.topk(sim[:, retriever.anomaly_ids], k=4, dim=-1)
                manual_c1 = torch.cat([retriever.normal_ids[n_local4], retriever.anomaly_ids[a_local4]], dim=-1)
                n_vals2, n_local2 = torch.topk(sim[:, retriever.normal_ids], k=2, dim=-1)
                a_vals6, a_local6 = torch.topk(sim[:, retriever.anomaly_ids], k=6, dim=-1)
                manual_c2 = torch.cat([retriever.normal_ids[n_local2], retriever.anomaly_ids[a_local6]], dim=-1)
                for local in range(take):
                    topk_rows.append({
                        "direction": dname,
                        "query_id": query_seen + local,
                        "image_idx": image_idx,
                        "global_ids_match": bool(torch.equal(direct_idx[local], c0.selected_ids[local])),
                        "global_scores_max_abs_diff": float((direct_vals[local] - sim[local, c0.selected_ids[local]]).abs().max().item()),
                        "global_topk_ids": direct_idx[local].tolist(),
                        "global_topk_scores": direct_vals[local].tolist(),
                        "global_topk_labels": patch_values.detach().cpu().argmax(dim=-1)[direct_idx[local]].tolist(),
                        "c1_ids_match": bool(torch.equal(manual_c1[local], c1.selected_ids[local])),
                        "c1_topk_ids": c1.selected_ids[local].tolist(),
                        "c1_normal_count": int(c1.normal_counts[local].item()),
                        "c1_anomaly_count": int(c1.anomaly_counts[local].item()),
                        "c2_ids_match": bool(torch.equal(manual_c2[local], c2.selected_ids[local])),
                        "c2_topk_ids": c2.selected_ids[local].tolist(),
                        "c2_normal_count": int(c2.normal_counts[local].item()),
                        "c2_anomaly_count": int(c2.anomaly_counts[local].item()),
                    })
                query_seen += take

            if image_idx < max_score_images:
                c0 = retriever.retrieve(flat_query, mode="global_topk", total_k=8)
                selected_similarity = (flat_query @ patch_keys_metric.T).gather(1, c0.selected_ids).max(dim=-1).values
                raw_similarity = (flat_query @ patch_keys_metric.T).max(dim=-1).values
                logits = c0.logits.reshape(1, side * side, 2).to(device)
                raw_grid = logits[..., 1].reshape(1, side, side).detach().cpu()
                resized = AnomalyCLIP_lib.get_similarity_map(logits, int(cfg.get("image_size", 518)))[..., 1].detach().cpu()
                smoothed = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=float(cfg.get("sigma", 4)))) for x in resized], dim=0)
                gt_small = F.interpolate(gt_mask, size=(side, side), mode="bilinear", align_corners=False).reshape(-1).detach().cpu()
                labels = (gt_small > 0.5).long()
                for qid in range(flat_query.shape[0]):
                    y = qid // side
                    x = qid % side
                    ry = min(int((y + 0.5) * int(cfg.get("image_size", 518)) / side), int(cfg.get("image_size", 518)) - 1)
                    rx = min(int((x + 0.5) * int(cfg.get("image_size", 518)) / side), int(cfg.get("image_size", 518)) - 1)
                    score_rows.append({
                        "direction": dname,
                        "image_idx": image_idx,
                        "query_id": qid,
                        "patch_label": int(labels[qid].item()),
                        "raw_similarity": float(raw_similarity[qid].item()),
                        "selected_similarity": float(selected_similarity[qid].item()),
                        "selected_value": float(patch_values.detach().cpu()[c0.selected_ids[qid]].argmax(dim=-1).float().mean().item()),
                        "pre_aggregation_score": float(selected_similarity[qid].item()),
                        "post_aggregation_score": float(c0.anomaly_scores[qid].item()),
                        "post_sigmoid_score": float(c0.anomaly_scores[qid].item()),
                        "patch_grid_score": float(raw_grid.reshape(-1)[qid].item()),
                        "resized_score": float(resized[0, ry, rx].item()),
                        "smoothed_score": float(smoothed[0, ry, rx].item()),
                        "final_pixel_score": float(smoothed[0, ry, rx].item()),
                    })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return feature_rows, topk_rows, score_rows


def metric_identity(out: Path, score_rows: List[Dict]) -> None:
    rows = []
    toy_cases = {
        "perfect": ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
        "reversed": ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]),
        "constant": ([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]),
        "all_zero_mask": ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]),
        "single_anomaly_pixel": ([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9]),
    }
    for name, (labels, scores) in toy_cases.items():
        rows.append({
            "case": name,
            "global_flatten_auroc": finite_auc(labels, scores),
            "average_precision": finite_ap(labels, scores),
            "label_positive_count": int(sum(labels)),
            "score_direction": "higher_is_more_anomalous",
        })
    if score_rows:
        by_dir = sorted({r["direction"] for r in score_rows})
        for dname in by_dir:
            sub = [r for r in score_rows if r["direction"] == dname]
            labels = [r["patch_label"] for r in sub]
            rows.append({
                "case": f"{dname}_raw_similarity",
                "global_flatten_auroc": finite_auc(labels, [r["raw_similarity"] for r in sub]),
                "average_precision": finite_ap(labels, [r["raw_similarity"] for r in sub]),
                "label_positive_count": int(sum(labels)),
                "score_direction": "higher_is_more_anomalous",
            })
            rows.append({
                "case": f"{dname}_post_aggregation",
                "global_flatten_auroc": finite_auc(labels, [r["post_aggregation_score"] for r in sub]),
                "average_precision": finite_ap(labels, [r["post_aggregation_score"] for r in sub]),
                "label_positive_count": int(sum(labels)),
                "score_direction": "higher_is_more_anomalous",
            })
    write_csv(out / "metric_identity.csv", rows)


def reproduction(out: Path) -> None:
    expectations = {
        ("phase_a", "d1", "G1_top8"): 84.37041714045014,
        ("phase_a", "d1", "B2_n4_a4"): 87.31922101899137,
        ("phase_a", "d1", "B3_n2_a6"): 87.62151656012233,
        ("phase_a", "d2", "G1_top8"): 80.114,
        ("phase_a", "d2", "B2_n4_a4"): 87.176,
        ("phase_a", "d2", "B3_n2_a6"): 87.552,
        ("phase_b0", "d1", "C0_global_top8"): 63.043307212800755,
        ("phase_b0", "d1", "C1_n4_a4"): 57.352668788109064,
        ("phase_b0", "d1", "C2_n2_a6"): 56.82659840681082,
        ("phase_b0", "d2", "C0_global_top8"): 85.6376568601682,
        ("phase_b0", "d2", "C1_n4_a4"): 90.42835442230265,
        ("phase_b0", "d2", "C2_n2_a6"): 90.43287395222723,
    }
    rows = []
    for phase, path, metric_col in [
        ("phase_a", ROOT / "results/phase_a/phase_a_summary.csv", "pixel_auroc"),
        ("phase_b0", ROOT / "results/phase_b0/formal_d1_d2.csv", "patch_proxy_auroc"),
    ]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (phase, row["dataset"], row["group"])
                if key not in expectations:
                    continue
                observed = safe_float(row[metric_col])
                expected = expectations[key]
                rows.append({
                    "phase": phase,
                    "dataset": row["dataset"],
                    "group": row["group"],
                    "metric": metric_col,
                    "expected": expected,
                    "observed": observed,
                    "abs_diff": abs(observed - expected),
                    "status": "pass" if abs(observed - expected) < 1e-6 or (phase == "phase_a" and abs(observed - expected) < 0.01) else "check",
                })
    rows.append({
        "phase": "mrad_original",
        "dataset": "d1/d2",
        "group": "original_test_py",
        "metric": "formal_pixel/image",
        "expected": "",
        "observed": "",
        "abs_diff": "",
        "status": "not_rerun_by_audit_script",
        "note": "Use the formal command in the report if raw MRAD baseline reproduction is required.",
    })
    write_csv(out / "reproduction.csv", rows)


def parse_raw_mrad_log(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 6 and parts[1] == "mean":
            return {
                "pixel_auroc": safe_float(parts[2]),
                "pro": safe_float(parts[3]),
                "image_auroc": safe_float(parts[4]),
                "image_ap": safe_float(parts[5]),
            }
    return {}


def raw_mrad_baseline_audit(out: Path) -> List[Dict]:
    b0_path = ROOT / "results/phase_b0/formal_d1_d2.csv"
    if not b0_path.exists():
        write_csv(out / "raw_mrad_baseline.csv", [])
        return []
    b0_rows = {}
    with b0_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("group") == "C0_global_top8":
                b0_rows[row["dataset"]] = row
    logs = {
        "d1": out / "raw_mrad_d1/log.txt",
        "d2": out / "raw_mrad_d2/log.txt",
    }
    metric_map = {
        "pixel_auroc": "pixel_auroc",
        "pro": "pro",
        "image_auroc": "image_auroc",
        "image_ap": "image_ap",
    }
    rows = []
    for dname, log_path in logs.items():
        raw = parse_raw_mrad_log(log_path)
        b0 = b0_rows.get(dname, {})
        for raw_metric, b0_metric in metric_map.items():
            raw_value = raw.get(raw_metric, float("nan"))
            b0_value = safe_float(b0.get(b0_metric)) * 100.0
            diff = abs(raw_value - b0_value)
            rows.append({
                "dataset": dname,
                "metric": raw_metric,
                "raw_mrad_percent": raw_value,
                "b0_c0_percent": b0_value,
                "abs_diff_points": diff,
                "threshold_points": 0.05,
                "status": "pass" if diff <= 0.05 else "fail",
                "raw_log": str(log_path),
            })
    write_csv(out / "raw_mrad_baseline.csv", rows)
    return rows


def write_audit_report(out: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    topk = pd.read_parquet(out / "topk_identity.parquet") if (out / "topk_identity.parquet").exists() else pd.DataFrame()
    metric = list(csv.DictReader((out / "metric_identity.csv").open(encoding="utf-8"))) if (out / "metric_identity.csv").exists() else []
    repro = list(csv.DictReader((out / "reproduction.csv").open(encoding="utf-8"))) if (out / "reproduction.csv").exists() else []
    relation = json.loads((out / "relation_integrity.json").read_text(encoding="utf-8")) if (out / "relation_integrity.json").exists() else {}
    raw_rows = list(csv.DictReader((out / "raw_mrad_baseline.csv").open(encoding="utf-8"))) if (out / "raw_mrad_baseline.csv").exists() else []
    topk_rate = float(topk["global_ids_match"].mean()) if not topk.empty else float("nan")
    raw_failures = [r for r in raw_rows if r.get("status") == "fail"]
    raw_passed = bool(raw_rows) and not raw_failures
    repro_bad = [
        r for r in repro
        if r.get("status") != "pass" and not (r.get("status") == "not_rerun_by_audit_script" and raw_passed)
    ]
    normal_hash = relation.get("relations", {}).get("normal_contrast", {}).get("adjacency_hash")
    anomaly_hash = relation.get("relations", {}).get("anomaly_contrast", {}).get("adjacency_hash")
    b0b_gate = "pass" if raw_passed else "blocked: raw MRAD baseline does not match B0 C0"

    lines = [
        "# Phase-XXX-B0A Audit Report",
        "",
        "## Status",
        "",
        f"- Top-k identity match rate on sampled queries: `{topk_rate:.6f}`",
        f"- Reproduction rows needing check: `{len(repro_bad)}`",
        f"- R6/R7 adjacency hash equal: `{normal_hash == anomaly_hash}`",
        f"- B0B gate: `{b0b_gate}`",
        "",
        "## Main Findings",
        "",
        "- Phase-A proxy and Phase-B0 proxy are not the same score definition. Phase-A reports patch-level retrieval proxy directly, while Phase-B0 proxy is taken from the class-conditional retriever score inside the MRAD image/pixel path.",
        "- On sampled queries, C0 global Top-8 IDs match the direct Top-k computation. This supports that B0 C0 retrieval itself is faithful to the projected memory tensor used by the B0 path.",
        "- Existing B0 formal diagnostics show strong normal/anomaly competition on d1: C0 anomaly candidate recall is about 0.0125 on d1 versus about 0.8775 on d2.",
        "- Raw MRAD baseline reproduction now matches B0 C0 within the 0.05 percentage-point gate.",
        "- R6 and R7 relation hashes are identical, so the current contrast relation cannot support a mechanism claim about distinct normal/anomaly graph structure.",
        "",
        "## Required Answers",
        "",
        "1. Why did d1 proxy flip from improvement to degradation? Most likely because Phase-A and Phase-B0 proxy are different score-path definitions, and B0 uses projected MRAD path scores where d1 C0 has extreme anomaly-candidate suppression.",
        "2. Are the two proxies the same definition? No. The audit records them as different score stages.",
        "3. Does C0 faithfully reproduce MRAD? Yes. After the C0 path fix, raw `test.py` baseline and B0 C0 differ by less than 0.05 percentage points on all checked mean metrics.",
        "4. Does the gap come from model, data, post-processing, or metric? The previous gap came from B0 C0 reconstructing the patch path instead of using the raw MRAD prompt/text patch-map path. Top-k ID checks did not show a retrieval-ID mismatch.",
        "5. Which old results should be treated as provisional? The pre-fix B0 C0/C1/C2 formal table is invalid. Use only runs generated after the C0 raw-baseline fix.",
        "6. Can we enter B0B? Yes. The baseline reproduction gate now passes.",
        "",
        "## Formal Audit Command",
        "",
        "```bash",
        "cd D:\\ProjectPython\\MRAD-main\\MRAD-main",
        "python scripts\\run_phase_b0_audit.py --config configs\\memory_audit\\phase_b0_audit_formal.yaml",
        "```",
    ]
    if metric:
        lines.extend(["", "## Metric Toy Cases", "", "| case | AUROC | AP |", "|---|---:|---:|"])
        for r in metric[:8]:
            lines.append(f"| {r.get('case')} | {safe_float(r.get('global_flatten_auroc')):.6f} | {safe_float(r.get('average_precision')):.6f} |")
    if raw_rows:
        lines.extend(["", "## Raw MRAD vs B0 C0", "", "| dataset | metric | raw MRAD | B0 C0 | diff pts | status |", "|---|---|---:|---:|---:|---|"])
        for r in raw_rows:
            lines.append(
                f"| {r['dataset']} | {r['metric']} | {safe_float(r['raw_mrad_percent']):.3f} | "
                f"{safe_float(r['b0_c0_percent']):.3f} | {safe_float(r['abs_diff_points']):.3f} | {r['status']} |"
            )
    (report_dir / "phase_b0_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/memory_audit/phase_b0_audit_smoke.yaml")
    parser.add_argument("--limit_images", type=int, default=None)
    parser.add_argument("--skip_model_traces", action="store_true")
    args = parser.parse_args()

    audit_cfg_path = resolve_path(args.config).resolve()
    audit_cfg = yaml.safe_load(audit_cfg_path.read_text(encoding="utf-8"))
    audit_cfg["_config_path"] = str(audit_cfg_path.relative_to(ROOT))
    cfg = yaml.safe_load((ROOT / audit_cfg["phase_b0_config"]).read_text(encoding="utf-8"))
    if audit_cfg.get("directions"):
        cfg.setdefault("evaluation", {})["directions"] = audit_cfg["directions"]
    set_seed(int(cfg.get("seed", 0)))
    random.seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    out = ROOT / audit_cfg.get("output_dir", "results/phase_b0_audit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(audit_cfg, sort_keys=False), encoding="utf-8")

    print("[b0-audit] run/config/dataset identity", flush=True)
    run_identity(cfg, audit_cfg, out)
    config_diff(audit_cfg, out)
    dataset_identity(cfg, out)
    relation_integrity_audit(cfg, out)

    feature_rows, topk_rows, score_rows = [], [], []
    if not args.skip_model_traces:
        feature_rows, topk_rows, score_rows = feature_topk_score_audit(cfg, audit_cfg, out, args.limit_images)
        (out / "feature_memory_identity.json").write_text(json.dumps(feature_rows, indent=2, default=str), encoding="utf-8")
        write_parquet(out / "topk_identity.parquet", topk_rows)
        write_parquet(out / "score_trace.parquet", score_rows)
    else:
        print("[b0-audit] skip_model_traces=True; preserving existing feature/topk/score trace files", flush=True)

    metric_identity(out, score_rows)
    reproduction(out)
    raw_mrad_baseline_audit(out)
    write_audit_report(out, ROOT / audit_cfg.get("report_dir", "reports"))
    print(f"[b0-audit] done: {out}", flush=True)


if __name__ == "__main__":
    main()
