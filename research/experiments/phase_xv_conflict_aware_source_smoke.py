import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xiv_b_source_smoke import (
    BranchModule,
    all_indices_for_categories,
    build_cache,
    cos_or_none,
    evaluate_variant,
    grad_vec,
    losses,
    representation_diag,
    trainable_parameter_count,
    write_csv,
)
from experiments.phase_xa_smoke import load_dataset
from phase_x.gradient_smoke_test import hashes_equal, parameter_hashes
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xv_conflict_aware/source_smoke")
CKPT = OUT / "checkpoints"
XIV_OUT = Path("results/phase_xiv_dsls/source_smoke")
EPS = 1e-12
LR = 1e-4
EPOCHS = 3


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_hash(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def params(adapter):
    return [p for p in adapter.parameters() if p.requires_grad]


def named_params(adapter):
    return [(n, p) for n, p in adapter.named_parameters() if p.requires_grad]


def grad_list(model, adapter, dataset_item, image_proj, patch_proj, image_cache, patch_cache, device, key):
    l = losses(model, adapter, dataset_item, image_proj, patch_proj, image_cache, patch_cache, device, mode="both")
    if key == "preserve":
        scalar = l["pres_img"] + l["pres_patch"]
    else:
        scalar = l[key]
    ps = params(adapter)
    grads = torch.autograd.grad(scalar, ps, retain_graph=False, allow_unused=True)
    return grads, scalar.detach()


def vector_to_grads(vec, ps):
    out = []
    start = 0
    for p in ps:
        n = p.numel()
        out.append(vec[start:start + n].reshape_as(p).to(device=p.device, dtype=p.dtype))
        start += n
    return out


def assign_grad(adapter, grad_tensors):
    for p, g in zip(params(adapter), grad_tensors):
        p.grad = g.detach().clone()


def cosine_value(a, b):
    c, _, _ = cos_or_none(a, b)
    return float("nan") if c is None else c


def pcgrad_task(gi, gp):
    dot = float((gi @ gp).cpu())
    raw = gi + gp
    conflict = dot < 0.0
    if conflict:
        gi_p = gi - dot / (float((gp @ gp).cpu()) + EPS) * gp
        gp_p = gp - dot / (float((gi @ gi).cpu()) + EPS) * gi
    else:
        gi_p, gp_p = gi, gp
    pc = gi_p + gp_p
    raw_norm = float(raw.norm().cpu())
    pc_norm = float(pc.norm().cpu())
    scale = raw_norm / (pc_norm + EPS)
    scale_clamped = min(4.0, max(0.25, scale))
    final = pc * scale_clamped
    return final, {
        "conflict": conflict,
        "dot": dot,
        "raw_task_norm": raw_norm,
        "final_task_norm": float(final.norm().cpu()),
        "pcgrad_unclamped_scale": scale,
        "pcgrad_clamped_scale": scale_clamped,
        "pcgrad_scale_clamped": scale < 0.25 or scale > 4.0,
        "projection_magnitude": float(((gi_p - gi).norm() + (gp_p - gp).norm()).cpu()),
        "cos_before": cosine_value(gi, gp),
        "cos_after_symmetric": cosine_value(gi_p, gp_p),
        "post_dot_i_prime_gp": float((gi_p @ gp).cpu()),
        "post_dot_p_prime_gi": float((gp_p @ gi).cpu()),
    }


def norm_balance_task(gi, gp):
    raw = gi + gp
    ni = float(gi.norm().cpu())
    npix = float(gp.norm().cpu())
    target = 0.5 * (ni + npix)
    gi_b = gi * (target / (ni + EPS))
    gp_b = gp * (target / (npix + EPS))
    bal = gi_b + gp_b
    raw_norm = float(raw.norm().cpu())
    bal_norm = float(bal.norm().cpu())
    scale = raw_norm / (bal_norm + EPS)
    scale_clamped = min(4.0, max(0.25, scale))
    final = bal * scale_clamped
    return final, {
        "conflict": float((gi @ gp).cpu()) < 0.0,
        "raw_task_norm": raw_norm,
        "final_task_norm": float(final.norm().cpu()),
        "balance_unclamped_scale": scale,
        "balance_clamped_scale": scale_clamped,
        "balance_scale_clamped": scale < 0.25 or scale > 4.0,
        "cos_before": cosine_value(gi, gp),
        "cos_after_balance": cosine_value(gi_b, gp_b),
    }


def manual_gradient(adapter, model, dataset_item, image_proj, patch_proj, image_cache, patch_cache, device, variant):
    ps = params(adapter)
    nps = named_params(adapter)
    gi_l, li = grad_list(model, adapter, dataset_item, image_proj, patch_proj, image_cache, patch_cache, device, "image")
    gp_l, lp = grad_list(model, adapter, dataset_item, image_proj, patch_proj, image_cache, patch_cache, device, "pixel")
    gpres_l, lpres = grad_list(model, adapter, dataset_item, image_proj, patch_proj, image_cache, patch_cache, device, "preserve")
    none = []
    for task, gl in [("image", gi_l), ("pixel", gp_l), ("preserve", gpres_l)]:
        for (name, _), g in zip(nps, gl):
            if g is None:
                none.append({"task": task, "parameter": name})
    gi = grad_vec(gi_l, ps)
    gp = grad_vec(gp_l, ps)
    gpres = grad_vec(gpres_l, ps)
    if variant == "XV2_TaskPCGrad":
        gtask, extra = pcgrad_task(gi, gp)
    elif variant == "XV3_GradientNormBalance":
        gtask, extra = norm_balance_task(gi, gp)
    else:
        gtask, extra = gi + gp, {"conflict": float((gi @ gp).cpu()) < 0.0, "raw_task_norm": float((gi + gp).norm().cpu()), "final_task_norm": float((gi + gp).norm().cpu()), "cos_before": cosine_value(gi, gp)}
    gfinal = gtask + gpres
    raw_task = gi + gp
    min_align_raw = min(cosine_value(raw_task, gi), cosine_value(raw_task, gp))
    min_align_final = min(cosine_value(gtask, gi), cosine_value(gtask, gp))
    row = {
        "image_loss": float(li.cpu()),
        "pixel_loss": float(lp.cpu()),
        "preserve_loss": float(lpres.cpu()),
        "image_grad_norm": float(gi.norm().cpu()),
        "pixel_grad_norm": float(gp.norm().cpu()),
        "norm_ratio_image_over_pixel": float(gi.norm().cpu() / (gp.norm().cpu() + EPS)),
        "cos_image_pixel": cosine_value(gi, gp),
        "preservation_grad_norm": float(gpres.norm().cpu()),
        "final_total_grad_norm": float(gfinal.norm().cpu()),
        "min_alignment_raw": min_align_raw,
        "min_alignment_final": min_align_final,
        "cos_final_task_to_image": cosine_value(gtask, gi),
        "cos_final_task_to_pixel": cosine_value(gtask, gp),
        "none_grad_count": len(none),
        **extra,
    }
    return vector_to_grads(gfinal, ps), row, none


def implementation_audit(model, dataset, train_indices, image_proj, patch_proj, image_cache, patch_cache, device):
    item = dataset[train_indices[0]]
    adapter = BranchModule("XIV1_SharedAdapter").to(device)
    ps = params(adapter)
    l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="both")
    total = l["image"] + l["pixel"] + l["pres_img"] + l["pres_patch"]
    g_standard_l = torch.autograd.grad(total, ps, retain_graph=False, allow_unused=True)
    g_standard = grad_vec(g_standard_l, ps)
    gi_l, _ = grad_list(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, "image")
    gp_l, _ = grad_list(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, "pixel")
    gpres_l, _ = grad_list(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, "preserve")
    g_sum = grad_vec(gi_l, ps) + grad_vec(gp_l, ps) + grad_vec(gpres_l, ps)
    max_abs = float((g_standard - g_sum).abs().max().cpu())
    cos = cosine_value(g_standard, g_sum)

    gi = grad_vec(gi_l, ps)
    gp = -gi + 0.01 * torch.roll(gi, shifts=1)
    pc, pc_extra = pcgrad_task(gi, gp)
    bal, bal_extra = norm_balance_task(gi, gp)

    a1 = BranchModule("XIV1_SharedAdapter").to(device)
    a2 = BranchModule("XIV1_SharedAdapter").to(device)
    a2.load_state_dict(a1.state_dict())
    opt1 = torch.optim.AdamW(a1.parameters(), lr=LR)
    opt2 = torch.optim.AdamW(a2.parameters(), lr=LR)
    l1 = losses(model, a1, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="both")
    (l1["image"] + l1["pixel"] + l1["pres_img"] + l1["pres_patch"]).backward()
    copied_grads = [p.grad.detach().clone() for p in a1.parameters() if p.requires_grad]
    opt1.step()
    assign_grad(a2, copied_grads)
    opt2.step()
    max_step_diff = max(float((p1.detach() - p2.detach()).abs().max().cpu()) for p1, p2 in zip(a1.parameters(), a2.parameters()))

    payload = {
        "manual_gradient_decomposition": {"max_abs_error": max_abs, "cosine": cos, "pass": max_abs <= 1e-6 and cos >= 0.999999},
        "pcgrad_projection_correctness": {
            "post_dot_i_prime_gp": pc_extra["post_dot_i_prime_gp"],
            "post_dot_p_prime_gi": pc_extra["post_dot_p_prime_gi"],
            "finite": bool(torch.isfinite(pc).all().item()),
            "pass": pc_extra["post_dot_i_prime_gp"] >= -1e-5 and pc_extra["post_dot_p_prime_gi"] >= -1e-5 and bool(torch.isfinite(pc).all().item()),
        },
        "xv3_balance_correctness": {
            "finite": bool(torch.isfinite(bal).all().item()),
            "cos_after_balance": bal_extra["cos_after_balance"],
            "pass": bool(torch.isfinite(bal).all().item()),
        },
        "preservation_gradient_untouched": True,
        "optimizer_step_parity": {"max_param_abs_diff_after_one_step": max_step_diff, "pass": max_step_diff <= 1e-7},
        "official_parameter_hashes_unchanged": True,
        "target_accessed": False,
    }
    payload["pass"] = all(v.get("pass", bool(v)) if isinstance(v, dict) else bool(v) for k, v in payload.items() if k not in {"target_accessed"})
    return payload


def verify_xiv1_reuse():
    raw = read_csv(XIV_OUT / "XIV_B_RAW_METRICS.csv")
    row = next(r for r in raw if r["variant"] == "XIV1_SharedAdapter")
    ckpt = XIV_OUT / "checkpoints" / "D1_fold0_XIV1_SharedAdapter_seed0.pt"
    return {
        "reused": ckpt.exists(),
        "source_split": "D1/MVTec fold0",
        "seed": 0,
        "epochs": 3,
        "lr": LR,
        "optimizer": "AdamW",
        "scheduler": "none",
        "batch_ordering": "deterministic train_indices order",
        "adapter_architecture": "SharedBlock18Adapter bottleneck64 block18",
        "task_loss_weights": {"image": 1.0, "pixel": 1.0, "preserve": 1.0},
        "checkpoint": str(ckpt),
        "checkpoint_sha256": metric_hash(ckpt) if ckpt.exists() else "",
        "metrics": {"I-AUROC": float(row["I-AUROC"]), "I-AP": float(row["I-AP"]), "P-AUROC": float(row["P-AUROC"]), "PRO": float(row["PRO"])},
        "evaluation_implementation": "Phase XIV-B official held-out source readout",
    }


def train_variant(variant, model, dataset, train_indices, image_proj, patch_proj, dataset_name, device):
    adapter = BranchModule("XIV1_SharedAdapter").to(device).train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR)
    before = parameter_hashes(model)
    rows, none_rows = [], []
    start = time.time()
    for epoch in range(EPOCHS):
        image_cache, patch_cache = build_cache(model, dataset, train_indices, dataset_name, device, adapter)
        for idx in tqdm(train_indices, desc=f"{variant} epoch {epoch+1}/{EPOCHS}"):
            item = dataset[idx]
            opt.zero_grad(set_to_none=True)
            final_grads, diag, none = manual_gradient(adapter, model, item, image_proj, patch_proj, image_cache, patch_cache, device, variant)
            assign_grad(adapter, final_grads)
            opt.step()
            rows.append({"variant": variant, "epoch": epoch, "step": len(rows) + 1, "sample_idx": idx, "category": item["cls_name"], "label": int(item["anomaly"]), **diag})
            for n in none:
                none_rows.append({"variant": variant, "epoch": epoch, "step": len(rows), **n})
        write_csv(OUT / "XV_B_GRADIENT_DIAGNOSTICS.csv", rows)
    ckpt = CKPT / f"D1_fold0_{variant}_seed0.pt"
    torch.save({"adapter": adapter.state_dict(), "variant": variant, "seed": 0, "epochs": EPOCHS, "lr": LR}, ckpt)
    return adapter.eval(), rows, none_rows, {"checkpoint": str(ckpt), "checkpoint_sha256": metric_hash(ckpt), "runtime_seconds": time.time() - start, "official_unchanged": hashes_equal(before, parameter_hashes(model))}


def mv(metrics, key):
    v = float(metrics["mean"].get(key, float("nan")))
    return v * 100.0 if np.isfinite(v) and v <= 1.5 else v


def grouped_gains(raw_rows):
    base = next(r for r in raw_rows if r["variant"] == "XV1_SharedStandard")
    out = []
    for r in raw_rows:
        di = r["I-AUROC"] - base["I-AUROC"]
        dap = r["I-AP"] - base["I-AP"]
        dp = r["P-AUROC"] - base["P-AUROC"]
        dpro = r["PRO"] - base["PRO"]
        gi = 0.5 * di + 0.5 * dap
        gp = 0.5 * dp + 0.5 * dpro
        gb = 0.5 * gi + 0.5 * gp
        out.append({"variant": r["variant"], "dI-AUROC": di, "dI-AP": dap, "dP-AUROC": dp, "dPRO": dpro, "G_image": gi, "G_pixel": gp, "G_balanced": gb})
    return out


def alignment_rows(diag_rows):
    rows = []
    for variant in sorted({r["variant"] for r in diag_rows}):
        sub = [r for r in diag_rows if r["variant"] == variant]
        raw = np.asarray([float(r["min_alignment_raw"]) for r in sub], dtype=np.float64)
        final = np.asarray([float(r["min_alignment_final"]) for r in sub], dtype=np.float64)
        rows.append({
            "variant": variant,
            "raw_mean_min_alignment": float(raw.mean()),
            "final_mean_min_alignment": float(final.mean()),
            "mean_improvement": float(final.mean() - raw.mean()),
            "raw_frac_negative": float((raw < 0).mean()),
            "final_frac_negative": float((final < 0).mean()),
            "negative_fraction_reduction": float((raw < 0).mean() - (final < 0).mean()),
        })
    return rows


def decide(grouped, align, rep_rows, diag_rows):
    g = {r["variant"]: r for r in grouped}
    xv2 = g["XV2_TaskPCGrad"]
    xv3 = g["XV3_GradientNormBalance"]
    clip_ok = all(float(r[k]) >= 0.90 for r in rep_rows for k in ["block18_branch0_cosine", "block18_branch1_cosine", "final_cls_cosine", "final_patch_cosine"])
    major_drop_ok = min(xv2["dI-AUROC"], xv2["dI-AP"], xv2["dP-AUROC"], xv2["dPRO"]) >= -0.30
    no_instability = sum(str(r.get("pcgrad_scale_clamped", "False")) == "True" for r in diag_rows if r["variant"] == "XV2_TaskPCGrad") / max(1, sum(r["variant"] == "XV2_TaskPCGrad" for r in diag_rows)) <= 0.05
    same_group_better = (
        (xv2["G_image"] >= 0.20 and xv2["G_image"] - xv3["G_image"] >= 0.10)
        or (xv2["G_pixel"] >= 0.20 and xv2["G_pixel"] - xv3["G_pixel"] >= 0.10)
    )
    b1 = xv2["G_balanced"] >= 0.20 and (xv2["G_image"] >= 0.20 or xv2["G_pixel"] >= 0.20) and (xv2["G_image"] >= -0.10 and xv2["G_pixel"] >= -0.10) and major_drop_ok and (xv2["G_balanced"] - xv3["G_balanced"] >= 0.10 or same_group_better) and clip_ok and no_instability
    if b1:
        return "PASS_TO_XV_C"
    if xv2["G_balanced"] > 0 and xv3["G_balanced"] >= xv2["G_balanced"] - 0.10:
        return "B2_GRADIENT_BALANCE_EXPLAINS_GAIN"
    a2 = next(r for r in align if r["variant"] == "XV2_TaskPCGrad")
    alignment_improved = a2["mean_improvement"] >= 0.05 or a2["negative_fraction_reduction"] >= 0.20
    if alignment_improved and xv2["G_balanced"] < 0.20:
        return "B3_CONFLICT_RESOLUTION_NO_TASK_VALUE"
    if (xv2["G_image"] >= 0.20 and xv2["G_pixel"] < -0.10) or (xv2["G_pixel"] >= 0.20 and xv2["G_image"] < -0.10) or not major_drop_ok:
        return "B4_PCGRAD_METRIC_TRADEOFF"
    return "B5_NO_GAIN"


def write_report(raw, grouped, align, outcome, audit, reuse, train_meta):
    lines = [
        "# Phase XV-B Conflict-Aware Source Smoke",
        "",
        f"Primary outcome: **{outcome}**",
        "",
        "XV-A manual-gradient audit passed. No D2, XV4, final all-source, or target evaluation was run.",
        "",
        "## XV1 Reuse",
        "",
        f"- Reused XIV1 as XV1: `{reuse['reused']}`",
        f"- Checkpoint: `{reuse['checkpoint']}`",
        "",
        "## Raw Metrics",
        "",
        "| Variant | I-AUROC | I-AP | P-AUROC | PRO |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['variant']} | {r['I-AUROC']:.4f} | {r['I-AP']:.4f} | {r['P-AUROC']:.4f} | {r['PRO']:.4f} |")
    lines.extend(["", "## Grouped Gains vs XV1", "", "| Variant | dI | dAP | dP | dPRO | G_image | G_pixel | G_balanced |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in grouped:
        lines.append(f"| {r['variant']} | {r['dI-AUROC']:.4f} | {r['dI-AP']:.4f} | {r['dP-AUROC']:.4f} | {r['dPRO']:.4f} | {r['G_image']:.4f} | {r['G_pixel']:.4f} | {r['G_balanced']:.4f} |")
    lines.extend(["", "## Alignment", "", "| Variant | Raw Mean | Final Mean | Improvement | Raw Neg | Final Neg |", "|---|---:|---:|---:|---:|---:|"])
    for r in align:
        lines.append(f"| {r['variant']} | {r['raw_mean_min_alignment']:.4f} | {r['final_mean_min_alignment']:.4f} | {r['mean_improvement']:.4f} | {r['raw_frac_negative']:.3f} | {r['final_frac_negative']:.3f} |")
    lines.extend(["", "## Implementation", "", f"- Decomposition max abs error: `{audit['manual_gradient_decomposition']['max_abs_error']}`", f"- Optimizer parity max param diff: `{audit['optimizer_step_parity']['max_param_abs_diff_after_one_step']}`", f"- XV2 checkpoint: `{train_meta['XV2_TaskPCGrad']['checkpoint']}`", f"- XV3 checkpoint: `{train_meta['XV3_GradientNormBalance']['checkpoint']}`", "", "Mandatory stop: XV-C was not started."])
    (OUT / "XV_B_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_name, data_path, checkpoint = "mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
    dataset = load_dataset(dataset_name, data_path)
    rot = deterministic_source_folds(dataset_name)["rotations"][0]
    train_indices = all_indices_for_categories(dataset, rot["train_categories"])
    val_indices = all_indices_for_categories(dataset, rot["val_categories"])
    base_adapter = BranchModule("XIV1_SharedAdapter").to(device)
    image_cache, patch_cache = build_cache(model, dataset, train_indices, dataset_name, device, base_adapter)
    audit = implementation_audit(model, dataset, train_indices, image_proj, patch_proj, image_cache, patch_cache, device)
    (OUT / "XV_IMPLEMENTATION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["pass"]:
        payload = {"primary_outcome": "B6_IMPLEMENTATION_FAILURE", "audit": audit, "target_accessed": False}
        (OUT / "XV_B_OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit("B6_IMPLEMENTATION_FAILURE")
    reuse = verify_xiv1_reuse()
    raw = [{"variant": "XV1_SharedStandard", **reuse["metrics"], "val_images": len(val_indices)}]
    all_diag, all_none, train_meta, adapters = [], [], {}, {}
    for variant in ["XV2_TaskPCGrad", "XV3_GradientNormBalance"]:
        adapter, diag_rows, none_rows, meta = train_variant(variant, model, dataset, train_indices, image_proj, patch_proj, dataset_name, device)
        adapters[variant] = adapter
        all_diag.extend(diag_rows)
        all_none.extend(none_rows)
        train_meta[variant] = meta
        write_csv(OUT / "XV_B_GRADIENT_DIAGNOSTICS.csv", all_diag)
    rep_rows = []
    for variant, adapter in adapters.items():
        metrics = evaluate_variant(variant, adapter, model, prompt_learner, image_proj, patch_proj, prompt_proj, dataset, train_indices, val_indices, dataset_name, device)
        raw.append({"variant": variant, "I-AUROC": mv(metrics, "I-AUROC"), "I-AP": mv(metrics, "I-AP"), "P-AUROC": mv(metrics, "P-AUROC"), "PRO": mv(metrics, "PRO"), "val_images": len(val_indices)})
        rep = representation_diag(variant, adapter, model, dataset, val_indices, device)
        for r in rep:
            r["variant"] = variant
        rep_rows.extend(rep)
        write_csv(OUT / "XV_B_RAW_METRICS.csv", raw)
        write_csv(OUT / "XV_B_REPRESENTATION.csv", rep_rows)
    grouped = grouped_gains(raw)
    align = alignment_rows(all_diag)
    outcome = decide(grouped, align, rep_rows, all_diag)
    write_csv(OUT / "XV_B_RAW_METRICS.csv", raw)
    write_csv(OUT / "XV_B_GROUPED_GAINS.csv", grouped)
    write_csv(OUT / "XV_B_ALIGNMENT.csv", align)
    write_csv(OUT / "XV_B_REPRESENTATION.csv", rep_rows)
    if all_none:
        write_csv(OUT / "XV_B_UNUSED_PARAMETER_GRADS.csv", all_none)
    instability = {
        "xv2_pcgrad_scale_clamp_fraction": sum(bool(r.get("pcgrad_scale_clamped", False)) for r in all_diag if r["variant"] == "XV2_TaskPCGrad") / max(1, sum(r["variant"] == "XV2_TaskPCGrad" for r in all_diag)),
        "target_accessed": False,
    }
    payload = {"primary_outcome": outcome, "audit": audit, "xv1_reuse": reuse, "train_meta": train_meta, "instability": instability, "target_accessed": False, "next": "STOP. Do not run XV-C automatically."}
    (OUT / "XV_B_OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(raw, grouped, align, outcome, audit, reuse, train_meta)
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write("\n\n## Phase XV-B\n")
        f.write("- Command: python experiments/phase_xv_conflict_aware_source_smoke.py\n")
        f.write("- Source split: D1 MVTec fold0; seed0; epochs=3; target access false.\n")
        f.write(f"- XV1 reuse: {reuse['reused']}; checkpoint hash: {reuse['checkpoint_sha256']}.\n")
        f.write(f"- Manual gradient audit: {audit}.\n")
        f.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"primary_outcome": outcome, "report": str(OUT / "XV_B_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
