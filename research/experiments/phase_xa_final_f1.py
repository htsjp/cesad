import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import evaluate_results, init_results
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import load_dataset, to_device_cache, train_variant, write_csv
from experiments.phase_xa_stage1 import all_indices_for_categories, read_existing
from models.mlp import average_neighbor
from mrad import compute_patch_socre
from phase_x.gradient_smoke_test import parameter_hashes
from phase_xa.adapted_cache_builder import build_adapted_task_cache
from phase_xa.cache_alignment import assert_cache_alignment
from phase_xa.config import SMOKE_CONFIG
from phase_xa.official_task_forward import image_memory_scores, normalized_image_feature, visual_outputs
from phase_xa.shared_adapter import SharedBlock18Adapter, trainable_parameter_count
from phase_xa.teacher_memory_builder import build_memory_from_indices


BASE = Path("results/phase_xa_memory_distillation")
OUT = BASE / "final_decision"
CKPT_DIR = OUT / "checkpoints"
CACHE_DIR = OUT / "adapted_source_caches"
METRICS = ["I-AUROC", "I-AP", "P-AUROC", "PRO"]
VARIANTS = ["XA0_OfficialMRADCLIP", "XA2_CLIPPreservedAdapter", "XA4_DualTeacherFullMemory"]


def direction_cfg(direction):
    if direction == "D1":
        return {
            "direction": "D1",
            "source_dataset": "mvtec",
            "source_path": "data/mvtec",
            "target_dataset": "visa",
            "target_path": "data/VISA",
            "checkpoint": "checkpoints/test_on_visa.pth",
        }
    return {
        "direction": "D2",
        "source_dataset": "visa",
        "source_path": "data/VISA",
        "target_dataset": "mvtec",
        "target_path": "data/mvtec",
        "checkpoint": "checkpoints/test_on_mvtec.pth",
    }


def f(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value, digits=4):
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def sha256_state_dict(state):
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        h.update(key.encode("utf-8"))
        if torch.is_tensor(tensor):
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(tensor).encode("utf-8"))
    return h.hexdigest()


def config_hash(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def actual_epoch_count(direction, variant):
    rows = []
    for path in sorted(BASE.glob(f"stage1_metrics_{direction}_fold*_{variant}_seed0.csv")):
        metrics = read_existing(path)
        if metrics:
            max_epoch = max(int(row["epoch"]) for row in metrics)
            rows.append(max_epoch + 1)
    if not rows:
        return 12
    rows = sorted(rows)
    return int(rows[len(rows) // 2])


def all_indices(dataset):
    return list(range(len(dataset)))


def save_protocol():
    protocol = {
        "phase": "X-A final decision F1",
        "source_side_status": {
            "D1": "SOURCE_MEMORY_SIGNAL_SUPPORTED_FOLD_SENSITIVE",
            "D2": "SOURCE_MEMORY_SIGNAL_SUPPORTED_EXISTING_FOLD",
            "overall": "XA_SOURCE_MEMORY_SIGNAL_POSITIVE",
        },
        "frozen_before_target": True,
        "primary_method": "XA4_DualTeacherFullMemory",
        "primary_control": "XA2_CLIPPreservedAdapter",
        "official_baseline": "XA0_OfficialMRADCLIP",
        "seed": 0,
        "epochs": {
            "D1": {
                "XA2_CLIPPreservedAdapter": actual_epoch_count("D1", "XA2_CLIPPreservedAdapter"),
                "XA4_DualTeacherFullMemory": actual_epoch_count("D1", "XA4_DualTeacherFullMemory"),
            },
            "D2": {
                "XA2_CLIPPreservedAdapter": actual_epoch_count("D2", "XA2_CLIPPreservedAdapter"),
                "XA4_DualTeacherFullMemory": actual_epoch_count("D2", "XA4_DualTeacherFullMemory"),
            },
        },
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "loss_weights": {
            "hard_label": 1.0,
            "clip_preserve": 0.1,
            "memory_distillation": 0.1,
        },
        "frozen_modules": ["CLIP", "prompt_learner", "image_proj", "patch_proj", "prompt_proj"],
        "adapter": "SharedBlock18Adapter",
        "all_source_categories": True,
        "target_tuning": False,
        "same_image_self_exclusion": True,
    }
    protocol["config_hash"] = config_hash(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_training_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def set_stage1_training_config(epochs):
    SMOKE_CONFIG["epochs"] = int(epochs)
    SMOKE_CONFIG["learning_rate"] = 1e-4
    SMOKE_CONFIG["cache_items_per_epoch"] = None
    SMOKE_CONFIG["teacher_items"] = None
    SMOKE_CONFIG["optimize_items_per_epoch"] = 10**9


def final_checkpoint_path(direction, variant):
    return CKPT_DIR / f"final_seed0_{variant}_{direction}.pt"


def train_final_one(cfg, variant, epochs, device):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = final_checkpoint_path(cfg["direction"], variant)
    if path.exists() and path.stat().st_size > 0:
        return {"direction": cfg["direction"], "variant": variant, "checkpoint": str(path), "status": "EXISTS"}

    set_stage1_training_config(epochs)
    torch.manual_seed(0)
    model, _, image_proj, patch_proj, _ = load_components(cfg["checkpoint"], device)
    dataset = load_dataset(cfg["source_dataset"], cfg["source_path"])
    source_indices = all_indices(dataset)
    teacher_image_cache, teacher_patch_cache = build_memory_from_indices(
        model, dataset, source_indices, cfg["source_dataset"], device, adapter=None
    )
    alignment = assert_cache_alignment(teacher_image_cache, teacher_patch_cache)
    teacher_image_cache = to_device_cache(teacher_image_cache, device)
    teacher_patch_cache = to_device_cache(teacher_patch_cache, device)
    before = parameter_hashes(model)
    start = time.time()
    metrics, diag, adapter = train_variant(
        cfg["direction"],
        variant,
        model,
        dataset,
        source_indices,
        source_indices[:2],
        image_proj,
        patch_proj,
        teacher_image_cache,
        teacher_patch_cache,
        device,
        return_adapter=True,
        random_seed=0,
    )
    after = parameter_hashes(model)
    payload = {
        "adapter": adapter.state_dict(),
        "direction": cfg["direction"],
        "variant": variant,
        "seed": 0,
        "source_dataset": cfg["source_dataset"],
        "target_dataset": cfg["target_dataset"],
        "epochs": epochs,
        "config": dict(SMOKE_CONFIG),
        "adapter_hash": sha256_state_dict(adapter.state_dict()),
        "official_hash_before": before,
        "official_hash_after": after,
    }
    torch.save(payload, path)
    write_csv(OUT / f"final_train_metrics_{cfg['direction']}_{variant}_seed0.csv", metrics)
    diag.update({
        "direction": cfg["direction"],
        "variant": variant,
        "checkpoint": str(path),
        "seconds": time.time() - start,
        "cache_alignment": alignment,
        "adapter_hash": payload["adapter_hash"],
    })
    return diag


def load_final_adapter(path, device):
    payload = torch.load(path, map_location=device)
    adapter = SharedBlock18Adapter().to(device).eval()
    adapter.load_state_dict(payload["adapter"], strict=True)
    return adapter, payload


def append_eval_result(results, cls_name, item, image_score, anomaly_map):
    mask = item["img_mask"].detach().cpu()
    mask = (mask > 0.5).float()
    results[cls_name]["imgs_masks"].append(mask)
    results[cls_name]["gt_sp"].append(int(item["anomaly"]))
    results[cls_name]["pr_sp"].append(float(image_score))
    results[cls_name]["anomaly_maps"].append(anomaly_map.detach().cpu())


def metric_value(metrics, key):
    value = metrics["mean"].get(key, float("nan"))
    return float(value) * 100.0 if np.isfinite(value) and value <= 1.5 else float(value)


def evaluate_target_one(cfg, variant, device):
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(cfg["checkpoint"], device)
    source_dataset = load_dataset(cfg["source_dataset"], cfg["source_path"])
    target_dataset = load_dataset(cfg["target_dataset"], cfg["target_path"])
    source_indices = all_indices(source_dataset)

    adapter = None
    resolved_checkpoint = ""
    adapter_hash = ""
    if not variant.startswith("XA0_"):
        resolved_checkpoint = str(final_checkpoint_path(cfg["direction"], variant))
        adapter, payload = load_final_adapter(resolved_checkpoint, device)
        adapter_hash = payload.get("adapter_hash", sha256_state_dict(adapter.state_dict()))

    cache_image_path = CACHE_DIR / f"{cfg['direction']}_{variant}_seed0_image_cache.pt"
    cache_patch_path = CACHE_DIR / f"{cfg['direction']}_{variant}_seed0_patch_cache.pt"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if adapter is None:
        image_cache, patch_cache = build_memory_from_indices(
            model, source_dataset, source_indices, cfg["source_dataset"], device, adapter=None
        )
    else:
        image_cache, patch_cache = build_adapted_task_cache(
            model, source_dataset, source_indices, cfg["source_dataset"], device, adapter=adapter
        )
    alignment = assert_cache_alignment(image_cache, patch_cache)
    torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in image_cache.items()}, cache_image_path)
    torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in patch_cache.items()}, cache_patch_path)
    image_cache = to_device_cache(image_cache, device)
    patch_cache = to_device_cache(patch_cache, device)

    obj_list = target_dataset.obj_list
    results = init_results(obj_list)
    per_image = []
    start = time.time()
    peak = 0
    for idx in range(len(target_dataset)):
        item = target_dataset[idx]
        cls_name = item["cls_name"]
        image = item["img"].unsqueeze(0).to(device)
        cls_id = torch.tensor([int(item["cls_id"])], device=device)
        with torch.no_grad():
            outputs = visual_outputs(model, image, adapter=adapter, grad=False)
            image_feature = normalized_image_feature(outputs)
            image_post, _, _, _ = image_memory_scores(image_feature, image_cache["keys"], image_cache["values"], image_proj)
            cls_memory = image_post[:, 1]

            patch_projection = average_neighbor(outputs[3][3])
            patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
            patch_bias = average_neighbor(outputs[1][3])
            patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)
            _, patch_bias_tokens, _, _ = compute_patch_socre(
                patch_bias,
                patch_cache["keys"],
                patch_cache["values"],
                device=device,
                proj=patch_proj,
                need_mask=False,
                patch_projection=patch_projection,
                gt_mask=item["img_mask"].unsqueeze(0),
                is_mradft=False,
                use_proj=True,
            )
            bias = prompt_proj(patch_bias_tokens[:, 0, :], patch_bias_tokens[:, 1, :])
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            anomaly_map_list = []
            for pidx, patch_feature in enumerate(outputs[3]):
                if pidx >= 3:
                    patch_feature = average_neighbor(patch_feature)
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                    sim_map = AnomalyCLIP_lib.get_similarity_map(similarity, 518)
                    anomaly_map_list.append((sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0)
            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=4)) for x in anomaly_map.detach().cpu()], dim=0).sum(dim=0)
            topk = torch.topk(anomaly_map.flatten(), max(1, int(anomaly_map.numel() * 0.01))).values.mean()
            final_score = 0.3 * topk.to(device) + 0.7 * cls_memory
            append_eval_result(results, cls_name, item, final_score.detach().cpu(), anomaly_map)
            per_image.append({
                "direction": cfg["direction"],
                "target_dataset": cfg["target_dataset"],
                "variant": variant,
                "seed": 0,
                "idx": idx,
                "category": cls_name,
                "label": int(item["anomaly"]),
                "score": float(final_score.detach().cpu()),
            })
        if torch.cuda.is_available():
            peak = max(peak, torch.cuda.max_memory_allocated())

    metrics = evaluate_results(results, obj_list, compute_pro=True)
    summary = {
        "direction": cfg["direction"],
        "source_dataset": cfg["source_dataset"],
        "target_dataset": cfg["target_dataset"],
        "variant": variant,
        "seed": 0,
        "I-AUROC": metric_value(metrics, "I-AUROC"),
        "I-AP": metric_value(metrics, "I-AP"),
        "P-AUROC": metric_value(metrics, "P-AUROC"),
        "PRO": metric_value(metrics, "PRO"),
        "images": len(target_dataset),
        "seconds": time.time() - start,
        "peak_gpu_bytes": peak,
        "resolved_checkpoint": resolved_checkpoint,
        "adapter_hash": adapter_hash,
        "image_cache": str(cache_image_path),
        "patch_cache": str(cache_patch_path),
    }
    per_category = []
    for category, vals in metrics.get("per_class", {}).items():
        per_category.append({
            "direction": cfg["direction"],
            "target_dataset": cfg["target_dataset"],
            "variant": variant,
            "seed": 0,
            "category": category,
            "I-AUROC": metric_value({"mean": vals}, "I-AUROC"),
            "I-AP": metric_value({"mean": vals}, "I-AP"),
            "P-AUROC": metric_value({"mean": vals}, "P-AUROC"),
            "PRO": metric_value({"mean": vals}, "PRO"),
        })
    diag = {
        "direction": cfg["direction"],
        "variant": variant,
        "seconds": summary["seconds"],
        "images": len(target_dataset),
        "peak_gpu_bytes": peak,
        "batch_size": 1,
        "precision": "fp32 evaluation tensors",
        "cache_alignment": alignment,
    }
    return summary, per_category, per_image, diag


def row_key(row):
    return (row["direction"], row["variant"])


def classify_direction(rows, direction):
    by_variant = {row["variant"]: row for row in rows if row["direction"] == direction}
    xa0 = by_variant["XA0_OfficialMRADCLIP"]
    xa2 = by_variant["XA2_CLIPPreservedAdapter"]
    xa4 = by_variant["XA4_DualTeacherFullMemory"]
    d0 = {m: f(xa4[m]) - f(xa0[m]) for m in METRICS}
    d2 = {m: f(xa4[m]) - f(xa2[m]) for m in METRICS}
    image0 = 0.5 * d0["I-AUROC"] + 0.5 * d0["I-AP"]
    pixel0 = 0.5 * d0["P-AUROC"] + 0.5 * d0["PRO"]
    image2 = 0.5 * d2["I-AUROC"] + 0.5 * d2["I-AP"]
    pixel2 = 0.5 * d2["P-AUROC"] + 0.5 * d2["PRO"]
    improved_groups = []
    if d0["I-AUROC"] >= 0.50 or d0["I-AP"] >= 0.40:
        improved_groups.append("image")
    if d0["P-AUROC"] >= 0.30 or d0["PRO"] >= 0.40:
        improved_groups.append("pixel")
    no_large_drop = all(delta >= -0.20 for delta in d0.values())
    xa4_exceeds_xa2_same = ("image" in improved_groups and image2 > 0) or ("pixel" in improved_groups and pixel2 > 0)
    if improved_groups and no_large_drop and xa4_exceeds_xa2_same:
        case = "F1-A"
        outcome = "TARGET_PERFORMANCE_PROMISING"
        reason = "XA4 improves official XA0 and exceeds XA2 on the improved metric group."
    elif any(delta < -0.20 for delta in d0.values()) and any(delta > 0 for delta in d0.values()):
        case = "F1-E"
        outcome = "TARGET_METRIC_TRADEOFF"
        reason = "XA4 improves at least one target metric but degrades another by >0.20 pp."
    elif image2 > 0.20 or pixel2 > 0.20:
        case = "F1-B"
        outcome = "TARGET_MEMORY_SIGNAL_BUT_NO_OFFICIAL_GAIN"
        reason = "XA4 clearly exceeds XA2 but does not sufficiently improve the official baseline."
    elif image2 <= 0 and pixel2 <= 0:
        case = "F1-C"
        outcome = "TARGET_MEMORY_NOT_NEEDED"
        reason = "XA2 matches or exceeds XA4 on the relevant groups."
    else:
        case = "F1-D"
        outcome = "SOURCE_ONLY_MEMORY_GAIN"
        reason = "XA4 does not meaningfully improve XA0 or XA2 on target."
    return {
        "direction": direction,
        "case": case,
        "outcome": outcome,
        "reason": reason,
        "deltas_vs_xa0": d0,
        "deltas_vs_xa2": d2,
        "ImageGain_vs_XA0": image0,
        "PixelGain_vs_XA0": pixel0,
        "ImageGain_vs_XA2": image2,
        "PixelGain_vs_XA2": pixel2,
    }


def category_deltas(per_category, direction):
    rows = [row for row in per_category if row["direction"] == direction]
    by = {(row["variant"], row["category"]): row for row in rows}
    cats = sorted({row["category"] for row in rows})
    out = []
    for metric in ["I-AUROC", "P-AUROC", "PRO"]:
        for comparison, base_variant in [("XA4-XA0", "XA0_OfficialMRADCLIP"), ("XA4-XA2", "XA2_CLIPPreservedAdapter")]:
            values = []
            for cat in cats:
                xa4 = by.get(("XA4_DualTeacherFullMemory", cat))
                base = by.get((base_variant, cat))
                if xa4 and base:
                    values.append((cat, f(xa4[metric]) - f(base[metric])))
            if not values:
                continue
            wins = sum(1 for _, x in values if x > 0.05)
            ties = sum(1 for _, x in values if abs(x) <= 0.05)
            losses = sum(1 for _, x in values if x < -0.05)
            deltas = [x for _, x in values]
            positive = [max(0.0, x) for _, x in values]
            total_pos = sum(positive)
            max_share = max(positive) / total_pos if total_pos > 0 else 0.0
            out.append({
                "direction": direction,
                "metric": metric,
                "comparison": comparison,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "median": sorted(deltas)[len(deltas) // 2],
                "worst": min(deltas),
                "best": max(deltas),
                "categories": len(deltas),
                "gain_spread": "CATEGORY_CONCENTRATED_GAIN" if max_share > 0.50 else "CATEGORY_DISTRIBUTED_GAIN",
                "max_positive_category_share": max_share,
            })
    return out


def report(rows, per_category, decisions, train_diags, eval_diags):
    lines = [
        "# Phase X-A Final Decision Batch F1",
        "",
        "Execution: F1 only. No F2 random-memory control was launched.",
        "",
        "## Target Metrics",
        "",
        "| Direction | Variant | I-AUROC | I-AP | P-AUROC | PRO |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['direction']} | {row['variant']} | {fmt(row['I-AUROC'], 6)} | {fmt(row['I-AP'], 6)} | "
            f"{fmt(row['P-AUROC'], 6)} | {fmt(row['PRO'], 6)} |"
        )
    lines.extend(["", "## F1 Decisions", ""])
    for decision in decisions:
        lines.extend([
            f"### {decision['direction']}",
            "",
            f"- Case: `{decision['case']}`",
            f"- Outcome: `{decision['outcome']}`",
            f"- Reason: {decision['reason']}",
            f"- ImageGain_vs_XA0: `{fmt(decision['ImageGain_vs_XA0'])}` pp",
            f"- PixelGain_vs_XA0: `{fmt(decision['PixelGain_vs_XA0'])}` pp",
            f"- ImageGain_vs_XA2: `{fmt(decision['ImageGain_vs_XA2'])}` pp",
            f"- PixelGain_vs_XA2: `{fmt(decision['PixelGain_vs_XA2'])}` pp",
            "",
            "| Metric | XA4-XA0 | XA4-XA2 |",
            "|---|---:|---:|",
        ])
        for metric in METRICS:
            lines.append(f"| {metric} | {fmt(decision['deltas_vs_xa0'][metric])} | {fmt(decision['deltas_vs_xa2'][metric])} |")
        lines.append("")
    cat_rows = []
    for direction in ["D1", "D2"]:
        cat_rows.extend(category_deltas(per_category, direction))
    write_csv(OUT / "final_per_category_metrics.csv", cat_rows)
    lines.extend([
        "## Per-Category Delta Summary",
        "",
        "| Direction | Metric | Comparison | W/T/L | Median | Worst | Best | Spread |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in cat_rows:
        lines.append(
            f"| {row['direction']} | {row['metric']} | {row['comparison']} | {row['wins']}/{row['ties']}/{row['losses']} | "
            f"{fmt(row['median'])} | {fmt(row['worst'])} | {fmt(row['best'])} | {row['gain_spread']} |"
        )
    lines.extend([
        "",
        "## Sanity",
        "",
        f"- Final checkpoints trained/evaluated: `{len(train_diags)}` train diagnostics, `{len(eval_diags)}` eval diagnostics.",
        "- Adapter-only update and official hash checks are recorded in `final_sanity_checks.json` and `final_source_model_hashes.json`.",
        "",
        "Mandatory stop: Batch F1 is complete. F2 is not executed automatically.",
    ])
    (OUT / "FINAL_BATCH_F1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_experiment_log(protocol, decisions, rows):
    log = Path("EXPERIMENT_LOG.md")
    lines = [
        "",
        "## Phase X-A Final Decision F1",
        "",
        f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- Command: `python -u experiments\\phase_xa_final_f1.py`",
        f"- Config hash: `{protocol['config_hash']}`",
        "- Branches completed: F1 only.",
        "- Branches skipped: F2/F3/F4 skipped by mandatory stop after F1.",
        "",
        "| Direction | Variant | I-AUROC | I-AP | P-AUROC | PRO |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['direction']} | {row['variant']} | {fmt(row['I-AUROC'], 6)} | {fmt(row['I-AP'], 6)} | "
            f"{fmt(row['P-AUROC'], 6)} | {fmt(row['PRO'], 6)} |"
        )
    lines.extend(["", "Direction decisions:"])
    for decision in decisions:
        lines.append(f"- {decision['direction']}: `{decision['case']}` / `{decision['outcome']}`")
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    protocol = save_protocol()
    train_diags = []
    model_hashes = []
    for direction in ["D1", "D2"]:
        cfg = direction_cfg(direction)
        for variant in ["XA2_CLIPPreservedAdapter", "XA4_DualTeacherFullMemory"]:
            epochs = protocol["epochs"][direction][variant]
            diag = train_final_one(cfg, variant, epochs, device)
            train_diags.append(diag)
            path = final_checkpoint_path(direction, variant)
            payload = torch.load(path, map_location="cpu")
            model_hashes.append({
                "direction": direction,
                "variant": variant,
                "checkpoint": str(path),
                "adapter_hash": payload.get("adapter_hash"),
                "official_hash_before": payload.get("official_hash_before"),
                "official_hash_after": payload.get("official_hash_after"),
            })
            (OUT / "final_train_diagnostics_running.json").write_text(json.dumps(train_diags, indent=2), encoding="utf-8")
    (OUT / "final_source_model_hashes.json").write_text(json.dumps(model_hashes, indent=2), encoding="utf-8")

    summaries, per_category, per_image, eval_diags = [], [], [], []
    for direction in ["D1", "D2"]:
        cfg = direction_cfg(direction)
        for variant in VARIANTS:
            summary, cats, images, diag = evaluate_target_one(cfg, variant, device)
            summaries.append(summary)
            per_category.extend(cats)
            per_image.extend(images)
            eval_diags.append(diag)
            write_csv(OUT / "final_f1_target_metrics_running.csv", summaries)
    write_csv(OUT / "final_f1_target_metrics.csv", summaries)
    write_csv(OUT / "final_f1_per_category.csv", per_category)
    write_csv(OUT / "final_f1_per_image_scores.csv", per_image)

    decisions = [classify_direction(summaries, "D1"), classify_direction(summaries, "D2")]
    sanity = {
        "f1_only": True,
        "target_tuning": False,
        "final_checkpoints": len(model_hashes),
        "eval_rows": len(summaries),
        "official_hashes_unchanged": all(m["official_hash_before"] == m["official_hash_after"] for m in model_hashes),
        "no_nan_inf_train": all(d.get("no_nan_inf", True) for d in train_diags),
        "no_feature_collapse": all(f(d.get("rank_collapse_fraction", 0.0)) <= 0.30 for d in train_diags),
        "adapter_trainable_parameters": trainable_parameter_count(SharedBlock18Adapter()),
        "eval_diagnostics": eval_diags,
    }
    (OUT / "final_sanity_checks.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    report(summaries, per_category, decisions, train_diags, eval_diags)
    outcome = {
        "phase": "Phase X-A F1",
        "mandatory_stop_after_f1": True,
        "direction_decisions": decisions,
        "next_batch": "F2 only for directions classified F1-A; not launched automatically.",
    }
    (OUT / "FINAL_OUTCOME.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    summary_lines = [
        "# Final Execution Summary",
        "",
        "Batch F1 completed and stopped.",
        "",
    ]
    for decision in decisions:
        summary_lines.append(f"- {decision['direction']}: `{decision['case']}` / `{decision['outcome']}`")
    (OUT / "FINAL_EXECUTION_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    append_experiment_log(protocol, decisions, summaries)
    print(json.dumps({"report": str(OUT / "FINAL_BATCH_F1_REPORT.md"), "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
