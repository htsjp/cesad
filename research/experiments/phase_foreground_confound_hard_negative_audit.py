import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_defect_query_transfer_pilot import CHECKPOINT, SEED, SIDE, resize_binary_mask, safe_ap, safe_auc
from experiments.phase_spatial_supervision_transfer_audit import evaluate_all, score_query_on_target, target_cells
from experiments.phase_type_factorized_consensus_query import load_patch_proj
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_foreground_confound_hard_negative_audit")
PREV = Path("results/phase_anomaly_content_vs_spatial_bias/formal")
DILATION = 3


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def load_queries():
    paths = {
        "B0_LocalContrast": PREV / "B0_LocalContrast.npy",
        "E1_SpatialForeground": PREV / "E1_SpatialForegroundControl.npy",
        "E2_ContentResidual": PREV / "E2_ContentResidual.npy",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing fixed directions: " + ", ".join(missing))
    return {k: np.load(v).astype(np.float32) for k, v in paths.items()}


def local_mask(mask37):
    gt = np.asarray(mask37, dtype=bool).reshape(SIDE, SIDE)
    if not gt.any():
        return None, None
    dil = binary_dilation(gt, iterations=DILATION)
    ring = np.logical_and(dil, ~gt)
    keep = np.logical_or(gt, ring)
    if not ring.any() or np.unique(gt[keep].astype(np.uint8)).size < 2:
        return None, None
    return keep.reshape(-1), gt.reshape(-1).astype(np.uint8)


def local_eval_one(dataset, idx_or_rows, kind, image_scores, maps):
    by_cat = defaultdict(lambda: {"iy": [], "is": [], "ly": [], "ls": [], "local_images": 0})
    all_y, all_s = [], []
    for pos, ref in enumerate(idx_or_rows):
        item = dataset[int(ref)] if kind == "source_dataset" else dataset[pos]
        if kind == "source_dataset":
            cat = str(item["cls_name"])
            label = int(item["anomaly"])
            mask_available = True
        else:
            row = dict(ref)
            cat = str(row.get("category", item.get("cls_name", "unknown")))
            label = int(row.get("label", item.get("anomaly", 0)))
            mask_available = bool(row.get("mask_available", True))
        by_cat[cat]["iy"].append(label)
        by_cat[cat]["is"].append(float(image_scores[pos]))
        if label != 1 or not mask_available:
            continue
        mask = resize_binary_mask(item)
        keep, y = local_mask(mask)
        if keep is None:
            continue
        s = np.asarray(maps[pos], dtype=np.float32).reshape(-1)
        yy, ss = y[keep], s[keep]
        by_cat[cat]["ly"].append(yy)
        by_cat[cat]["ls"].append(ss)
        by_cat[cat]["local_images"] += 1
        all_y.append(yy)
        all_s.append(ss)
    per = {}
    for cat, vals in sorted(by_cat.items()):
        y = np.concatenate(vals["ly"]) if vals["ly"] else np.asarray([], dtype=np.uint8)
        s = np.concatenate(vals["ls"]) if vals["ls"] else np.asarray([], dtype=np.float32)
        row = {
            "Local_P-AUROC": safe_auc(y, s),
            "Local_P-AP": safe_ap(y, s),
            "local_images": vals["local_images"],
            "dilation_patch_cells": DILATION,
        }
        row["Local_G"] = 0.5 * (row["Local_P-AUROC"] + row["Local_P-AP"]) if np.isfinite(row["Local_P-AUROC"]) and np.isfinite(row["Local_P-AP"]) else float("nan")
        per[cat] = row
    y_all = np.concatenate(all_y) if all_y else np.asarray([], dtype=np.uint8)
    s_all = np.concatenate(all_s) if all_s else np.asarray([], dtype=np.float32)
    mean = {
        "Local_P-AUROC": safe_auc(y_all, s_all),
        "Local_P-AP": safe_ap(y_all, s_all),
        "local_images": len(all_y),
        "dilation_patch_cells": DILATION,
    }
    mean["Local_G"] = 0.5 * (mean["Local_P-AUROC"] + mean["Local_P-AP"]) if np.isfinite(mean["Local_P-AUROC"]) and np.isfinite(mean["Local_P-AP"]) else float("nan")
    return mean, per


def evaluate_local(queries, patch_proj, device, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, q in queries.items():
            print(f"[HardNegative] local eval {variant} on {cell_id}", flush=True)
            image, maps = score_query_on_target(q, patch, patch_proj, device, smoke)
            metric, per = local_eval_one(dataset, idx_use, kind, image, maps)
            metric.update({"dataset": cell_id, "target": target, "variant": variant})
            raw.append(metric)
            for cat, vals in per.items():
                percat.append({"dataset": cell_id, "target": target, "category": cat, "variant": variant, **vals})
    return raw, percat


def merge_per_category(std_per, local_per):
    rows = []
    by_local = {(r["dataset"], r["category"], r["variant"]): r for r in local_per}
    for r in std_per:
        key = (r["dataset"], r["category"], r["variant"])
        out = dict(r)
        out.update({k: v for k, v in by_local.get(key, {}).items() if k not in {"dataset", "target", "category", "variant"}})
        rows.append(out)
    return rows


def summarize(std, local):
    sby = {(r["dataset"], r["variant"]): r for r in std}
    lby = {(r["dataset"], r["variant"]): r for r in local}
    rows = []
    for ds in sorted({r["dataset"] for r in std}):
        b0s, e1s, e2s = sby[(ds, "B0_LocalContrast")], sby[(ds, "E1_SpatialForeground")], sby[(ds, "E2_ContentResidual")]
        b0l, e1l, e2l = lby[(ds, "B0_LocalContrast")], lby[(ds, "E1_SpatialForeground")], lby[(ds, "E2_ContentResidual")]
        rows.append({
            "dataset": ds,
            "standard_B0_G": float(b0s["G_pixel"]),
            "standard_E1_G": float(e1s["G_pixel"]),
            "standard_E2_G": float(e2s["G_pixel"]),
            "local_B0_G": float(b0l["Local_G"]),
            "local_E1_G": float(e1l["Local_G"]),
            "local_E2_G": float(e2l["Local_G"]),
            "local_E1_minus_B0_G": float(e1l["Local_G"]) - float(b0l["Local_G"]),
            "local_E1_minus_E2_G": float(e1l["Local_G"]) - float(e2l["Local_G"]),
            "B0_local_minus_standard_G": float(b0l["Local_G"]) - float(b0s["G_pixel"]),
            "E1_local_minus_standard_G": float(e1l["Local_G"]) - float(e1s["G_pixel"]),
            "E2_local_minus_standard_G": float(e2l["Local_G"]) - float(e2s["G_pixel"]),
        })
    return rows


def decide(rows):
    e1_b0 = float(np.mean([r["local_E1_minus_B0_G"] for r in rows]))
    e1_e2 = float(np.mean([r["local_E1_minus_E2_G"] for r in rows]))
    drops = {
        "B0": float(np.mean([r["B0_local_minus_standard_G"] for r in rows])),
        "E1": float(np.mean([r["E1_local_minus_standard_G"] for r in rows])),
        "E2": float(np.mean([r["E2_local_minus_standard_G"] for r in rows])),
    }
    local_vals = np.array([[r["local_B0_G"], r["local_E1_G"], r["local_E2_G"]] for r in rows], dtype=np.float64)
    mean_local_range = float(np.mean(np.max(local_vals, axis=1) - np.min(local_vals, axis=1)))
    if e1_b0 <= -3.0 and e1_e2 <= -3.0:
        outcome = "FOREGROUND_METRIC_CONFOUND_CONFIRMED"
    elif all(v <= -10.0 for v in drops.values()) and mean_local_range < 1.0:
        outcome = "CURRENT_SIGNAL_LARGELY_FOREGROUND_DRIVEN"
    elif e1_b0 <= -3.0 and e1_e2 <= -3.0:
        outcome = "ANOMALY_CONTENT_SURVIVES_FOREGROUND_CONTROL"
    elif abs(e1_b0) < 1.0 and abs(e1_e2) < 1.0:
        outcome = "FOREGROUND_CONTROL_INSUFFICIENT"
    else:
        outcome = "FOREGROUND_CONTROL_INSUFFICIENT"
    return outcome, {"mean_local_E1_minus_B0_G": e1_b0, "mean_local_E1_minus_E2_G": e1_e2, "mean_local_minus_standard_G": drops, "mean_local_variant_range": mean_local_range}


def write_report(out, outcome, gate, std, local, summary):
    lines = [
        "# Foreground Confound Hard-Negative Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "## Protocol",
        "",
        f"No training. The three fixed directions are loaded from the prior content-vs-spatial audit. Local hard-negative evaluation uses target GT only for final evaluation, with fixed dilation={DILATION} patch cells.",
        "",
        "## Standard Metrics",
        "",
        "| Dataset | Variant | P-AUROC | P-AP | G |",
        "|---|---|---:|---:|---:|",
    ]
    for r in std:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} |")
    lines.extend(["", "## Local Hard-Negative Metrics", "", "| Dataset | Variant | Local P-AUROC | Local P-AP | Local G | Images |", "|---|---|---:|---:|---:|---:|"])
    for r in local:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['Local_P-AUROC']):.4f} | {float(r['Local_P-AP']):.4f} | {float(r['Local_G']):.4f} | {int(r['local_images'])} |")
    lines.extend(["", "## Summary", "", "| Dataset | E1-B0 Local G | E1-E2 Local G | B0 Local-Standard | E1 Local-Standard | E2 Local-Standard |", "|---|---:|---:|---:|---:|---:|"])
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['local_E1_minus_B0_G']:+.4f} | {r['local_E1_minus_E2_G']:+.4f} | {r['B0_local_minus_standard_G']:+.4f} | {r['E1_local_minus_standard_G']:+.4f} | {r['E2_local_minus_standard_G']:+.4f} |")
    lines.extend([
        "",
        "## Gate",
        "",
        json.dumps(gate, indent=2, default=json_default),
        "",
        "Per-category standard and local results are saved in `PER_CATEGORY.csv`.",
        "",
        "STOP. No new direction, training, Memory, or dilation tuning was started.",
    ])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    out = OUT / ("smoke" if args.smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[HardNegative] start smoke={args.smoke} device={device}", flush=True)
    queries = load_queries()
    patch_proj = load_patch_proj(device)
    std_raw, std_per = evaluate_all(queries, patch_proj, device, args.smoke)
    local_raw, local_per = evaluate_local(queries, patch_proj, device, args.smoke)
    percat = merge_per_category(std_per, local_per)
    summary = summarize(std_raw, local_raw)
    outcome, gate = decide(summary)
    write_csv(out / "RAW_METRICS.csv", std_raw)
    write_csv(out / "LOCAL_HARD_NEGATIVE_METRICS.csv", local_raw + summary)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_json(out / "OUTCOME.json", {
        "phase": "foreground_confound_hard_negative_audit",
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "protocol": "strict_source_only_cross_dataset_zsad",
        "fixed_directions_reused_from": str(PREV),
        "variants": list(queries.keys()),
        "dilation_patch_cells": DILATION,
        "outcome": outcome,
        "gate": gate,
        "smoke": args.smoke,
        "runtime_sec": time.time() - start,
        "checkpoint": str(CHECKPOINT),
        "seed": SEED,
        "training_used": False,
        "target_information_used_for_training": False,
        "memory_used": False,
        "prompt_used": False,
        "adapter_used": False,
        "dino_used": False,
    })
    write_report(out, outcome, gate, std_raw, local_raw, summary)
    print(f"[HardNegative] outcome={outcome} runtime_sec={time.time() - start:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
