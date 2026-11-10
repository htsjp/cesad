import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import load_dataset, to_device_cache
from experiments.phase_xiv_b_source_smoke import BranchModule, build_cache, write_csv
from phase_xa.official_task_forward import (
    image_memory_scores,
    normalized_image_feature,
    normalized_patch_feature,
    patch_memory_scores,
    visual_outputs,
)
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xx_pixel_gradient_quality/audit")
XIV_CKPT = Path("results/phase_xiv_dsls/source_smoke/checkpoints/D1_fold0_XIV1_SharedAdapter_seed0.pt")
XIX_D2_CKPT = Path("results/phase_xix_diverse_anomaly_exposure/batch_a/checkpoints/D2_fold0_XIX1_REAL_ONLY_seed0.pt")
REGIONS = ["R1_ANOMALY_CORE", "R2_BOUNDARY", "R3_NEAR_BACKGROUND", "R4_FAR_BACKGROUND", "R5_NORMAL_IMAGE"]
CONFS = ["LOW_CONF", "MID_CONF", "HIGH_CONF"]


def fast_indices_for_categories(dataset, categories):
    cats = set(categories)
    return [idx for idx, row in enumerate(dataset.data_all) if row["cls_name"] in cats]


def select_balanced_indices(dataset, source_indices, count=32):
    normals = [i for i in source_indices if int(dataset.data_all[i]["anomaly"]) == 0]
    anoms = [i for i in source_indices if int(dataset.data_all[i]["anomaly"]) == 1]
    selected = []
    for i in range(count // 2):
        selected.append(normals[i % len(normals)])
        selected.append(anoms[i % len(anoms)])
    return selected[:count]


def params_and_names(adapter):
    named = [(n, p) for n, p in adapter.named_parameters() if p.requires_grad]
    return [p for _, p in named], named


def grad_vec(grads, params):
    parts = []
    for g, p in zip(grads, params):
        parts.append(torch.zeros_like(p, dtype=torch.float32).reshape(-1) if g is None else g.detach().float().reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(1)


def grad_of(loss, params, retain_graph=True):
    if loss is None:
        return torch.zeros(sum(p.numel() for p in params), device=params[0].device)
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return grad_vec(grads, params)


def cos_dot(a, b):
    dot = float((a @ b).detach().cpu())
    na = float(a.norm().detach().cpu())
    nb = float(b.norm().detach().cpu())
    cos = None if na < 1e-12 or nb < 1e-12 else dot / (na * nb)
    nim = max(0.0, -dot)
    return cos, dot, na, nb, nim


def group_name(name):
    if name.endswith("gate_raw"):
        return "gate"
    if ".pre_norm." in name or name.startswith("pre_norm."):
        return "LayerNorm"
    if name.endswith("down.weight"):
        return "down.weight"
    if name.endswith("down.bias"):
        return "down.bias"
    if name.endswith("up.weight"):
        return "up.weight"
    if name.endswith("up.bias"):
        return "up.bias"
    return name


def group_slice(named):
    out = {}
    start = 0
    for n, p in named:
        end = start + p.numel()
        out.setdefault(group_name(n), []).append((start, end))
        start = end
    return out


def masked_mean(values, mask):
    if int(mask.sum().item()) == 0:
        return None
    return values[mask].mean()


def dice_loss(prob, target):
    eps = 1e-6
    inter = (prob * target).sum()
    return 1.0 - (2.0 * inter + eps) / (prob.sum() + target.sum() + eps)


def patch_labels(item, device, side):
    mask = item["img_mask"].squeeze().to(device)
    mask = (mask > 0.5).float()
    small = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(side, side), mode="nearest").reshape(-1)
    return small


def region_masks(label):
    arr = (label.detach().cpu().numpy().reshape(37, 37) > 0.5)
    masks = {}
    if arr.any():
        er = binary_erosion(arr, iterations=1, border_value=0)
        d1 = binary_dilation(arr, iterations=1)
        d2 = binary_dilation(arr, iterations=2)
        masks["R1_ANOMALY_CORE"] = er
        masks["R2_BOUNDARY"] = d1 & (~er)
        masks["R3_NEAR_BACKGROUND"] = d2 & (~d1)
        masks["R4_FAR_BACKGROUND"] = ~d2
        masks["R5_NORMAL_IMAGE"] = np.zeros_like(arr, dtype=bool)
    else:
        masks["R1_ANOMALY_CORE"] = np.zeros_like(arr, dtype=bool)
        masks["R2_BOUNDARY"] = np.zeros_like(arr, dtype=bool)
        masks["R3_NEAR_BACKGROUND"] = np.zeros_like(arr, dtype=bool)
        masks["R4_FAR_BACKGROUND"] = np.zeros_like(arr, dtype=bool)
        masks["R5_NORMAL_IMAGE"] = np.ones_like(arr, dtype=bool)
    return {k: torch.from_numpy(v.reshape(-1)).bool() for k, v in masks.items()}


def confidence_masks(official_prob):
    conf = (2.0 * official_prob.detach().cpu().reshape(-1) - 1.0).abs()
    return {
        "LOW_CONF": conf < 0.25,
        "MID_CONF": (conf >= 0.25) & (conf < 0.60),
        "HIGH_CONF": conf >= 0.60,
    }


def official_patch_prob(model, item, patch_proj, patch_cache, device):
    image = item["img"].unsqueeze(0).to(device)
    with torch.no_grad():
        out = visual_outputs(model, image, adapter=None, grad=False)
        patch = normalized_patch_feature(out)
        patch_post, _, _, _ = patch_memory_scores(patch, patch_cache["keys"], patch_cache["values"], patch_proj)
    return patch_post[..., 1].reshape(-1).detach().cpu()


def forward_losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device):
    image = item["img"].unsqueeze(0).to(device)
    out = visual_outputs(model, image, adapter=adapter, grad=True)
    img = normalized_image_feature(out)
    patch = normalized_patch_feature(out)
    image_post, _, _, _ = image_memory_scores(img, image_cache["keys"], image_cache["values"], image_proj)
    patch_post, _, _, _ = patch_memory_scores(patch, patch_cache["keys"], patch_cache["values"], patch_proj)
    target_img = torch.tensor([int(item["anomaly"])], device=device)
    l_image = F.nll_loss(torch.log(image_post.clamp_min(1e-6)), target_img)
    prob = patch_post[..., 1].reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    label = patch_labels(item, device, int(math.sqrt(prob.numel()))).to(device)
    bce = F.binary_cross_entropy(prob, label, reduction="none")
    l_focal = bce.mean()
    l_dice = dice_loss(prob, label)
    return l_image, l_focal, l_dice, bce, prob, label


def summarize(rows, keys):
    grouped = {}
    for r in rows:
        grouped.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, xs in grouped.items():
        row = {k: v for k, v in zip(keys, key)}
        for metric in ["cos", "norm", "nim", "nim_per_pixel", "pixel_count"]:
            vals = [float(x[metric]) for x in xs if x.get(metric) not in (None, "")]
            arr = np.asarray(vals, dtype=np.float64)
            row[f"{metric}_mean"] = float(arr.mean()) if arr.size else ""
            row[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else (0.0 if arr.size else "")
            row[f"{metric}_median"] = float(np.median(arr)) if arr.size else ""
        cos_vals = [x["cos"] for x in xs if x.get("cos") is not None]
        dot_vals = [float(x["dot"]) for x in xs if x.get("dot") not in (None, "")]
        row["negative_fraction"] = float(np.mean([v < 0 for v in dot_vals])) if dot_vals else ""
        row["batches"] = len(xs)
        out.append(row)
    return out


def load_adapter(direction, state, device):
    adapter = BranchModule("XIV1_SharedAdapter").to(device)
    if state == "S1":
        ckpt = XIV_CKPT if direction == "D1" else XIX_D2_CKPT
        payload = torch.load(ckpt, map_location=device)
        adapter.load_state_dict(payload["adapter"])
    adapter.eval()
    return adapter


def run_state(direction, state, model, dataset, source_indices, sample_indices, image_proj, patch_proj, dataset_name, official_patch_cache, device):
    adapter = load_adapter(direction, state, device)
    image_cache, patch_cache = build_cache(model, dataset, source_indices, dataset_name, device, adapter)
    params, named = params_and_names(adapter)
    slices = group_slice(named)
    region_rows, conf_rows, fd_rows, param_rows, sample_rows, ablation_rows = [], [], [], [], [], []
    for batch_id, idx in enumerate(tqdm(sample_indices, desc=f"{direction} {state} XX-A gradients")):
        item = dataset[idx]
        official_prob = official_patch_prob(model, item, patch_proj, official_patch_cache, device)
        l_image, l_focal, l_dice, bce, prob, label = forward_losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device)
        g_image = grad_of(l_image, params, retain_graph=True)
        g_focal = grad_of(l_focal, params, retain_graph=True)
        g_dice = grad_of(l_dice, params, retain_graph=True)
        cos_f, dot_f, ni, nf, nim_f = cos_dot(g_image, g_focal)
        cos_d, dot_d, _, nd, nim_d = cos_dot(g_image, g_dice)
        fd_rows.extend([
            {"direction": direction, "state": state, "batch_id": batch_id, "sample_idx": idx, "loss_component": "BCE_CLASSIFICATION_FULL", "cos": cos_f, "dot": dot_f, "image_norm": ni, "component_norm": nf, "nim": nim_f},
            {"direction": direction, "state": state, "batch_id": batch_id, "sample_idx": idx, "loss_component": "DICE_FULL_DIAGNOSTIC_NON_ADDITIVE", "cos": cos_d, "dot": dot_d, "image_norm": ni, "component_norm": nd, "nim": nim_d},
        ])
        label_cpu = label.detach().cpu()
        region_mask = region_masks(label_cpu)
        conf_mask = confidence_masks(official_prob)
        component_grads = {"A0_FULL_PIXEL": g_focal}
        per_component = []
        for region, mask_cpu in region_mask.items():
            mask = mask_cpu.to(device)
            loss = masked_mean(bce, mask)
            g = grad_of(loss, params, retain_graph=True) if loss is not None else torch.zeros_like(g_image)
            cos, dot, _, norm, nim = cos_dot(g_image, g)
            pix = int(mask.sum().item())
            row = {"direction": direction, "state": state, "batch_id": batch_id, "sample_idx": idx, "category": item["cls_name"], "label": int(item["anomaly"]), "region": region, "cos": cos, "dot": dot, "norm": norm, "nim": nim, "pixel_count": pix, "nim_per_pixel": nim / max(1, pix)}
            region_rows.append(row)
            per_component.append((region, g, nim, pix))
            if region == "R2_BOUNDARY":
                component_grads["A1_REMOVE_BOUNDARY"] = g
            if region in {"R3_NEAR_BACKGROUND", "R4_FAR_BACKGROUND"}:
                component_grads.setdefault("A2_REMOVE_BACKGROUND", torch.zeros_like(g))
                component_grads["A2_REMOVE_BACKGROUND"] = component_grads["A2_REMOVE_BACKGROUND"] + g
            for pg, spans in slices.items():
                sub_i = torch.cat([g_image[s:e] for s, e in spans])
                sub_g = torch.cat([g[s:e] for s, e in spans])
                c, d, _, no, nm = cos_dot(sub_i, sub_g)
                param_rows.append({"direction": direction, "state": state, "batch_id": batch_id, "sample_idx": idx, "component_type": "region", "component": region, "parameter_group": pg, "cos": c, "dot": d, "norm": no, "nim": nm})
        for conf, mask_cpu in conf_mask.items():
            mask = mask_cpu.to(device)
            loss = masked_mean(bce, mask)
            g = grad_of(loss, params, retain_graph=True) if loss is not None else torch.zeros_like(g_image)
            cos, dot, _, norm, nim = cos_dot(g_image, g)
            pix = int(mask.sum().item())
            conf_rows.append({"direction": direction, "state": state, "batch_id": batch_id, "sample_idx": idx, "category": item["cls_name"], "label": int(item["anomaly"]), "confidence_group": conf, "cos": cos, "dot": dot, "norm": norm, "nim": nim, "pixel_count": pix, "nim_per_pixel": nim / max(1, pix)})
            if conf == "LOW_CONF":
                component_grads["A3_REMOVE_LOW_CONF"] = g
            for pg, spans in slices.items():
                sub_i = torch.cat([g_image[s:e] for s, e in spans])
                sub_g = torch.cat([g[s:e] for s, e in spans])
                c, d, _, no, nm = cos_dot(sub_i, sub_g)
                param_rows.append({"direction": direction, "state": state, "batch_id": batch_id, "sample_idx": idx, "component_type": "confidence", "component": conf, "parameter_group": pg, "cos": c, "dot": d, "norm": no, "nim": nm})
        total_nim = sum(x[2] for x in per_component)
        for region, _g, nim, pix in per_component:
            sample_rows.append({"direction": direction, "state": state, "batch_id": batch_id, "component": region, "nim": nim, "nim_share": nim / (total_nim + 1e-12), "pixel_count": pix})
        for name, remove_g in [
            ("A0_FULL_PIXEL", torch.zeros_like(g_image)),
            ("A1_REMOVE_BOUNDARY", component_grads.get("A1_REMOVE_BOUNDARY", torch.zeros_like(g_image))),
            ("A2_REMOVE_BACKGROUND", component_grads.get("A2_REMOVE_BACKGROUND", torch.zeros_like(g_image))),
            ("A3_REMOVE_LOW_CONF", component_grads.get("A3_REMOVE_LOW_CONF", torch.zeros_like(g_image))),
        ]:
            remaining = g_focal if name == "A0_FULL_PIXEL" else (g_focal - remove_g)
            virtual = g_image + remaining
            cos_img, _dot, _nvi, nv, _nim = cos_dot(virtual, g_image)
            cos_pix, _dot2, _nvp, _npix, _nim2 = cos_dot(virtual, g_focal)
            retained = float(remaining.norm().detach().cpu()) / (float(g_focal.norm().detach().cpu()) + 1e-12)
            ablation_rows.append({"direction": direction, "state": state, "batch_id": batch_id, "ablation": name, "cos_virtual_image": cos_img, "cos_virtual_full_pixel": cos_pix, "virtual_norm": nv, "retained_pixel_norm_ratio": retained})
    return region_rows, conf_rows, fd_rows, param_rows, sample_rows, ablation_rows


def by_direction_summary(region_rows, conf_rows, ablation_rows):
    rows = []
    for direction in ["D1", "D2"]:
        r_s1 = [r for r in region_rows if r["direction"] == direction and r["state"] == "S1"]
        c_s1 = [r for r in conf_rows if r["direction"] == direction and r["state"] == "S1"]
        a_s1 = [r for r in ablation_rows if r["direction"] == direction and r["state"] == "S1"]
        def mean_region(region, metric):
            vals = [float(r[metric]) for r in r_s1 if r["region"] == region]
            return float(np.mean(vals)) if vals else 0.0
        def sum_region(regions, metric):
            vals = [float(r[metric]) for r in r_s1 if r["region"] in regions]
            return float(np.sum(vals)) if vals else 0.0
        def mean_conf(conf, metric):
            vals = [float(r[metric]) for r in c_s1 if r["confidence_group"] == conf]
            return float(np.mean(vals)) if vals else 0.0
        base = np.mean([float(r["cos_virtual_image"]) for r in a_s1 if r["ablation"] == "A0_FULL_PIXEL"])
        def abl(name):
            xs = [r for r in a_s1 if r["ablation"] == name]
            return (
                float(np.mean([float(r["cos_virtual_image"]) for r in xs])) if xs else 0.0,
                float(np.mean([float(r["retained_pixel_norm_ratio"]) for r in xs])) if xs else 0.0,
            )
        a1, r1 = abl("A1_REMOVE_BOUNDARY")
        a2, r2 = abl("A2_REMOVE_BACKGROUND")
        a3, r3 = abl("A3_REMOVE_LOW_CONF")
        total_nim = sum_region(REGIONS, "nim")
        bg_nim = sum_region(["R3_NEAR_BACKGROUND", "R4_FAR_BACKGROUND"], "nim")
        low_nim = sum(float(r["nim"]) for r in c_s1 if r["confidence_group"] == "LOW_CONF")
        conf_total = sum(float(r["nim"]) for r in c_s1)
        rows.append({
            "direction": direction,
            "state": "S1",
            "boundary_per_pixel_over_core": mean_region("R2_BOUNDARY", "nim_per_pixel") / (mean_region("R1_ANOMALY_CORE", "nim_per_pixel") + 1e-12),
            "background_nim_share": bg_nim / (total_nim + 1e-12),
            "low_conf_nim_share": low_nim / (conf_total + 1e-12),
            "low_over_high_nim_per_pixel": mean_conf("LOW_CONF", "nim_per_pixel") / (mean_conf("HIGH_CONF", "nim_per_pixel") + 1e-12),
            "a1_boundary_alignment_improvement": a1 - base,
            "a1_retained_pixel_norm": r1,
            "a2_background_alignment_improvement": a2 - base,
            "a2_retained_pixel_norm": r2,
            "a3_low_conf_alignment_improvement": a3 - base,
            "a3_retained_pixel_norm": r3,
        })
    return rows


def decide(summary_rows):
    by = {r["direction"]: r for r in summary_rows}
    h1 = all(by[d]["boundary_per_pixel_over_core"] >= 1.5 and by[d]["a1_boundary_alignment_improvement"] >= 0.10 and by[d]["a1_retained_pixel_norm"] >= 0.70 for d in by)
    h2 = all(by[d]["background_nim_share"] >= 0.50 and by[d]["a2_background_alignment_improvement"] >= 0.10 and by[d]["a2_retained_pixel_norm"] >= 0.70 for d in by)
    h3 = all(by[d]["low_conf_nim_share"] >= 0.40 and by[d]["low_over_high_nim_per_pixel"] >= 1.5 and by[d]["a3_low_conf_alignment_improvement"] >= 0.10 and by[d]["a3_retained_pixel_norm"] >= 0.70 for d in by)
    if h1 or h2 or h3:
        return "PASS_TO_PHASE_XX_B"
    mechanisms = []
    for d, r in by.items():
        candidates = {
            "boundary": r["boundary_per_pixel_over_core"],
            "background": r["background_nim_share"],
            "low_conf": r["low_conf_nim_share"],
        }
        mechanisms.append(max(candidates, key=candidates.get))
    if len(set(mechanisms)) > 1:
        return "XX_A_DIRECTION_SPECIFIC_PIXEL_NOISE"
    any_heavy = any(r["boundary_per_pixel_over_core"] >= 1.5 or r["background_nim_share"] >= 0.50 or r["low_conf_nim_share"] >= 0.40 for r in by.values())
    retained_fail = any_heavy and any(
        (r["a1_retained_pixel_norm"] < 0.70 and r["boundary_per_pixel_over_core"] >= 1.5)
        or (r["a2_retained_pixel_norm"] < 0.70 and r["background_nim_share"] >= 0.50)
        or (r["a3_retained_pixel_norm"] < 0.70 and r["low_conf_nim_share"] >= 0.40)
        for r in by.values()
    )
    if retained_fail:
        return "XX_A_NO_SELECTIVE_REMOVAL_HEADROOM"
    return "XX_A_DIFFUSE_PIXEL_CONFLICT"


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    all_region, all_conf, all_fd, all_param, all_share, all_ablate = [], [], [], [], [], []
    for direction, dataset_name, data_path, checkpoint in [
        ("D1", "mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"),
        ("D2", "visa", "data/visa", "checkpoints/test_on_mvtec.pth"),
    ]:
        model, _prompt_learner, image_proj, patch_proj, _prompt_proj = load_components(checkpoint, device)
        dataset = load_dataset(dataset_name, data_path)
        rot = deterministic_source_folds(dataset_name)["rotations"][0]
        source_indices = fast_indices_for_categories(dataset, rot["train_categories"])
        sample_indices = select_balanced_indices(dataset, source_indices, count=32)
        official_adapter = BranchModule("XIV1_SharedAdapter").to(device)
        official_image_cache, official_patch_cache = build_cache(model, dataset, source_indices, dataset_name, device, official_adapter)
        for state in ["S0", "S1"]:
            rows = run_state(direction, state, model, dataset, source_indices, sample_indices, image_proj, patch_proj, dataset_name, official_patch_cache, device)
            region, conf, fd, param, share, ablate = rows
            all_region.extend(region)
            all_conf.extend(conf)
            all_fd.extend(fd)
            all_param.extend(param)
            all_share.extend(share)
            all_ablate.extend(ablate)
            write_csv(OUT / "REGION_GRADIENT_METRICS.csv", summarize(all_region, ["direction", "state", "region"]))
            write_csv(OUT / "CONFIDENCE_GRADIENT_METRICS.csv", summarize(all_conf, ["direction", "state", "confidence_group"]))
            write_csv(OUT / "FOCAL_DICE_GRADIENTS.csv", summarize(all_fd, ["direction", "state", "loss_component"]))
            write_csv(OUT / "PARAMETER_GROUP_GRADIENTS.csv", summarize(all_param, ["direction", "state", "component_type", "component", "parameter_group"]))
            write_csv(OUT / "NEGATIVE_INTERACTION_SHARE.csv", all_share)
            write_csv(OUT / "VIRTUAL_ABLATION.csv", summarize(all_ablate, ["direction", "state", "ablation"]))
    region_summary = summarize(all_region, ["direction", "state", "region"])
    conf_summary = summarize(all_conf, ["direction", "state", "confidence_group"])
    ablate_summary = summarize(all_ablate, ["direction", "state", "ablation"])
    per_direction = by_direction_summary(all_region, all_conf, all_ablate)
    outcome = decide(per_direction)
    write_csv(OUT / "PER_DIRECTION_SUMMARY.csv", per_direction)
    lines = [
        "# Phase XX-A Spatial Pixel-Gradient Quality Audit",
        "",
        f"Outcome: **{outcome}**",
        "",
        "No model training and no target access were performed. Primary conclusions use trained state S1; S0 is diagnostic only.",
        "",
        "## S1 Per-Direction Gate Summary",
        "",
        "| Direction | Boundary/Core per-pixel | Background NIM share | Low-conf NIM share | A1 dCos/Retain | A2 dCos/Retain | A3 dCos/Retain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in per_direction:
        lines.append(f"| {r['direction']} | {r['boundary_per_pixel_over_core']:.4f} | {r['background_nim_share']:.4f} | {r['low_conf_nim_share']:.4f} | {r['a1_boundary_alignment_improvement']:.4f}/{r['a1_retained_pixel_norm']:.4f} | {r['a2_background_alignment_improvement']:.4f}/{r['a2_retained_pixel_norm']:.4f} | {r['a3_low_conf_alignment_improvement']:.4f}/{r['a3_retained_pixel_norm']:.4f} |")
    lines.extend([
        "",
        "## Answers",
        "",
        "- Q1: Negative interaction source is reported in `REGION_GRADIENT_METRICS.csv`, `CONFIDENCE_GRADIENT_METRICS.csv`, and summarized above.",
        "- Q2: Boundary support requires boundary/core per-pixel >=1.5 in both directions plus A1 improvement/retention gates.",
        "- Q3: Background support requires near+far background >=50% NIM plus A2 improvement/retention gates.",
        "- Q4: Low-confidence support requires >=40% NIM and low/high per-pixel >=1.5 plus A3 gates.",
        "- Q5: Same mechanism is required cross-directionally for PASS.",
        "- Q6: Retained pixel norm ratios are shown for A1/A2/A3.",
        "",
        "Mandatory stop: Phase XX-B was not started.",
    ])
    (OUT / "XX_A_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {"phase": "XX-A", "outcome": outcome, "target_accessed": False, "trained": False, "runtime_sec": time.time() - start, "next": "STOP"}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write("\n\n## Phase XX-A\n")
        f.write("- Scope: source-only gradient quality audit on D1 MVTec fold0 and D2 VisA fold0; 32 deterministic balanced source-training samples per direction; states S0/S1.\n")
        f.write("- Pixel classification attribution used current XIV1 BCE pixel term; Dice was reported separately as non-additive diagnostic.\n")
        f.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XX_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
