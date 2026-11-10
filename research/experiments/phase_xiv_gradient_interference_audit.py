import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import load_dataset, patch_mask_37, to_device_cache
from phase_xa.adapted_cache_builder import build_adapted_task_cache
from phase_xa.distillation_losses import hard_label_loss
from phase_xa.official_task_forward import (
    image_memory_scores,
    normalized_image_feature,
    normalized_patch_feature,
    patch_memory_scores,
    visual_outputs,
)
from phase_xa.shared_adapter import SharedBlock18Adapter, trainable_parameter_count
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xiv_dsls/gradient_audit")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def all_indices_for_categories(dataset, categories):
    cats = set(categories)
    return [idx for idx in range(len(dataset)) if dataset[idx]["cls_name"] in cats]


def balanced_order(dataset, indices, batches=32):
    normals, anomalies = [], []
    for idx in indices:
        item = dataset[idx]
        if int(item["anomaly"]) == 1:
            anomalies.append(idx)
        else:
            normals.append(idx)
    if not normals or not anomalies:
        raise RuntimeError("XIV-A needs both normal and anomalous source images.")
    ordered = []
    for i in range(batches):
        ordered.append(anomalies[i % len(anomalies)] if i % 2 == 0 else normals[i % len(normals)])
    return ordered, len(normals), len(anomalies)


def patch_mask(item, device, side):
    mask = item["img_mask"].squeeze().to(device)
    mask = (mask > 0.5).float()
    return F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(side, side), mode="bilinear", align_corners=False).reshape(1, -1)


def split_losses(model, adapter, image, item, image_proj, patch_proj, image_cache, patch_cache, device):
    out = visual_outputs(model, image, adapter=adapter, grad=True)
    image_feat = normalized_image_feature(out)
    patch_feat = normalized_patch_feature(out)
    image_post, _, _, _ = image_memory_scores(image_feat, image_cache["keys"], image_cache["values"], image_proj)
    patch_post, _, _, _ = patch_memory_scores(patch_feat, patch_cache["keys"], patch_cache["values"], patch_proj)
    side = int(math.sqrt(patch_post.shape[1]))
    pmask = patch_mask(item, device, side)
    image_target = torch.tensor([int(item["anomaly"])], device=device)
    image_loss = F.nll_loss(torch.log(image_post.clamp_min(1e-6)), image_target)
    pixel_loss = F.binary_cross_entropy(patch_post[..., 1].clamp(1e-6, 1 - 1e-6), pmask)
    combined = hard_label_loss(image_post, patch_post, item["anomaly"], patch_mask_37(item, device))
    return image_loss, pixel_loss, combined, image_post, patch_post


def param_rows(adapter, grads_a, grads_b):
    rows = []
    for (name, param), ga, gb in zip(adapter.named_parameters(), grads_a, grads_b):
        va = torch.zeros_like(param).reshape(-1) if ga is None else ga.detach().float().reshape(-1)
        vb = torch.zeros_like(param).reshape(-1) if gb is None else gb.detach().float().reshape(-1)
        denom = va.norm() * vb.norm()
        cos = float((va @ vb / denom).cpu()) if denom > 0 else 0.0
        rows.append((name, cos, float(va.norm().cpu()), float(vb.norm().cpu())))
    return rows


def flat_grad(grads, params):
    parts = []
    for grad, param in zip(grads, params):
        if grad is None:
            parts.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
        else:
            parts.append(grad.detach().float().reshape(-1))
    return torch.cat(parts)


def cosine(a, b):
    denom = a.norm() * b.norm()
    return float((a @ b / denom).cpu()) if denom > 0 else 0.0


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "frac_negative": float((arr < 0).mean()) if arr.size else 0.0,
        "frac_lt_minus_0p1": float((arr < -0.1).mean()) if arr.size else 0.0,
    }


def run_direction(direction, dataset_name, data_path, checkpoint, device):
    start = time.time()
    torch.manual_seed(1400 + (1 if direction == "D1" else 2))
    model, _, image_proj, patch_proj, _ = load_components(checkpoint, device)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][0]
    train_indices = all_indices_for_categories(dataset, rotation["train_categories"])
    ordered, normal_count, anomaly_count = balanced_order(dataset, train_indices, batches=32)
    image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None)
    image_cache = to_device_cache(image_cache, device)
    patch_cache = to_device_cache(patch_cache, device)
    adapter = SharedBlock18Adapter().to(device)
    params = [p for p in adapter.parameters() if p.requires_grad]

    batch_rows = []
    group_rows = []
    for batch_id, idx in enumerate(tqdm(ordered, desc=f"XIV-A {direction} gradient audit")):
        item = dataset[idx]
        image = item["img"].unsqueeze(0).to(device)

        image_loss, _, combined, image_post, patch_post = split_losses(
            model, adapter, image, item, image_proj, patch_proj, image_cache, patch_cache, device
        )
        g_image = torch.autograd.grad(image_loss, params, retain_graph=False, allow_unused=True)

        pixel_loss, patch_side = None, None
        image_loss_2, pixel_loss, combined_2, _, patch_post_2 = split_losses(
            model, adapter, image, item, image_proj, patch_proj, image_cache, patch_cache, device
        )
        patch_side = int(math.sqrt(patch_post_2.shape[1]))
        g_pixel = torch.autograd.grad(pixel_loss, params, retain_graph=False, allow_unused=True)

        vi = flat_grad(g_image, params)
        vp = flat_grad(g_pixel, params)
        c = cosine(vi, vp)
        batch_rows.append({
            "direction": direction,
            "batch_id": batch_id,
            "sample_idx": idx,
            "category": item["cls_name"],
            "image_label": int(item["anomaly"]),
            "image_loss": float(image_loss.detach().cpu()),
            "pixel_loss": float(pixel_loss.detach().cpu()),
            "combined_hard_label_loss": float(combined.detach().cpu()),
            "cos_grad": c,
            "image_grad_norm": float(vi.norm().cpu()),
            "pixel_grad_norm": float(vp.norm().cpu()),
            "patch_side": patch_side,
            "trainable_parameters": trainable_parameter_count(adapter),
            "target_accessed": False,
            "optimizer_step": False,
        })
        for name, pc, ni, npix in param_rows(adapter, g_image, g_pixel):
            group_rows.append({
                "direction": direction,
                "batch_id": batch_id,
                "parameter_group": name,
                "cos_grad": pc,
                "image_grad_norm": ni,
                "pixel_grad_norm": npix,
            })
    seconds = time.time() - start
    return batch_rows, group_rows, {
        "direction": direction,
        "dataset": dataset_name,
        "source_train_images": len(train_indices),
        "source_train_normals": normal_count,
        "source_train_anomalies": anomaly_count,
        "batches": len(batch_rows),
        "seconds": seconds,
        "trainable_parameters": trainable_parameter_count(adapter),
        "target_accessed": False,
        "optimizer_steps": 0,
    }


def outcome_from_summaries(summaries):
    passes = []
    strong = []
    no_interference = []
    partial_ok = []
    for direction, s in summaries.items():
        passes.append(s["mean"] <= 0.10 and s["frac_negative"] >= 0.30)
        strong.append(s["mean"] < 0.0 and s["frac_negative"] >= 0.30)
        no_interference.append(s["mean"] > 0.30 and s["frac_negative"] < 0.10)
        partial_ok.append(s["mean"] <= 0.20)
    if any(passes):
        return "PASS_TO_XIV_B"
    if all(no_interference):
        return "XIV_A_NO_MEANINGFUL_TASK_INTERFERENCE"
    if any(strong) and all(partial_ok):
        return "XIV_A_PARTIAL_INTERFERENCE"
    return "XIV_A_NO_MEANINGFUL_TASK_INTERFERENCE"


def write_report(batch_rows, group_rows, costs, outcome):
    by_dir = {}
    for direction in sorted({r["direction"] for r in batch_rows}):
        vals = [float(r["cos_grad"]) for r in batch_rows if r["direction"] == direction]
        s = summarize(vals)
        by_dir[direction] = s
    group_summary = []
    for direction in sorted({r["direction"] for r in group_rows}):
        groups = sorted({r["parameter_group"] for r in group_rows if r["direction"] == direction})
        for group in groups:
            vals = [float(r["cos_grad"]) for r in group_rows if r["direction"] == direction and r["parameter_group"] == group]
            s = summarize(vals)
            group_summary.append({"direction": direction, "parameter_group": group, **s})

    lines = [
        "# Phase XIV-A Gradient Interference Audit",
        "",
        f"Outcome: **{outcome}**",
        "",
        "No adapters were trained. No optimizer step was executed. Target data was not accessed.",
        "",
        "## Direction Summary",
        "",
        "| Direction | Mean Cos | Std | Median | Frac < 0 | Frac < -0.1 | Batches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for direction, s in by_dir.items():
        n = sum(1 for r in batch_rows if r["direction"] == direction)
        lines.append(f"| {direction} | {s['mean']:.4f} | {s['std']:.4f} | {s['median']:.4f} | {s['frac_negative']:.3f} | {s['frac_lt_minus_0p1']:.3f} | {n} |")
    lines.extend([
        "",
        "## Parameter Groups",
        "",
        "| Direction | Group | Mean Cos | Std | Median | Frac < 0 | Frac < -0.1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for r in group_summary:
        lines.append(f"| {r['direction']} | {r['parameter_group']} | {r['mean']:.4f} | {r['std']:.4f} | {r['median']:.4f} | {r['frac_negative']:.3f} | {r['frac_lt_minus_0p1']:.3f} |")
    lines.extend([
        "",
        "## Required Answers",
        "",
        "Q1. Image and pixel objective conflict is determined by the shared adapter gradient cosine above.",
        "Q2. Direction-level consistency is shown in the Direction Summary table.",
        "Q3. The strongest conflicting adapter parameter groups are the rows with lowest mean cosine in Parameter Groups.",
        "Q4. Systematic conflict requires a high negative-batch fraction; otherwise it is driven by a minority of batches.",
        "Q5. Semantic/local decoupling is justified only by PASS_TO_XIV_B or XIV_A_PARTIAL_INTERFERENCE.",
        "",
        "## Runtime",
        "",
        "| Direction | Dataset | Batches | Seconds | Source Train Images | Target Accessed | Optimizer Steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for c in costs:
        lines.append(f"| {c['direction']} | {c['dataset']} | {c['batches']} | {c['seconds']:.2f} | {c['source_train_images']} | {c['target_accessed']} | {c['optimizer_steps']} |")
    lines.extend([
        "",
        "Mandatory stop: Phase XIV-B training was not started.",
    ])
    (OUT / "GRADIENT_INTERFERENCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return by_dir, group_summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    directions = {
        "D1": ("mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"),
        "D2": ("visa", "data/VISA", "checkpoints/test_on_mvtec.pth"),
    }
    all_batch_rows, all_group_rows, costs = [], [], []
    try:
        for direction, spec in directions.items():
            batch_rows, group_rows, cost = run_direction(direction, *spec, device=device)
            all_batch_rows.extend(batch_rows)
            all_group_rows.extend(group_rows)
            costs.append(cost)
        summaries = {
            d: summarize([float(r["cos_grad"]) for r in all_batch_rows if r["direction"] == d])
            for d in directions
        }
        outcome = outcome_from_summaries(summaries)
        by_dir, group_summary = write_report(all_batch_rows, all_group_rows, costs, outcome)
        write_csv(OUT / "batch_gradients.csv", all_batch_rows)
        write_csv(OUT / "parameter_group_gradients.csv", all_group_rows)
        payload = {
            "outcome": outcome,
            "direction_summaries": by_dir,
            "parameter_group_summary": group_summary,
            "costs": costs,
            "target_accessed": False,
            "optimizer_steps": 0,
            "next": "STOP. Do not start XIV-B automatically unless the gate outcome explicitly permits it.",
        }
    except Exception as exc:
        outcome = "XIV_J_IMPLEMENTATION_FAILURE"
        payload = {
            "outcome": outcome,
            "error": repr(exc),
            "target_accessed": False,
            "optimizer_steps": 0,
            "next": "STOP. Fix implementation before any XIV-B training.",
        }
        (OUT / "GRADIENT_INTERFERENCE_REPORT.md").write_text(
            "# Phase XIV-A Gradient Interference Audit\n\n"
            f"Outcome: **{outcome}**\n\n"
            f"Error: `{repr(exc)}`\n\n"
            "No Phase XIV-B training was started.\n",
            encoding="utf-8",
        )
        raise
    finally:
        (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "GRADIENT_INTERFERENCE_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
