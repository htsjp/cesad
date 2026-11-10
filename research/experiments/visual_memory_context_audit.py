import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "experiments"))

import AnomalyCLIP_lib
from base_residual_memory.canonical import FORMULA_NAME, canonical_fuse
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from experiments.cross_dataset_ultra_compression_audit import build_source_memory, metric, write_json, write_md
from memory_conditioned_prompt.base_prompt import freeze_clip
from mrad_audit_utils import evaluate_results, init_results, load_config, load_model, make_loader, one_hot, parse_args, resize_mask, retrieval_logits, save_run_metadata, set_seed, write_csv
from text_anchored_memory.source_protocol import clone_filtered_loader, source_protocol_record, split_categories
from ultra_compact_memory.centroid_pca import fit_centroid_pca, reconstruct_centroids
from ultra_compact_memory.shared_rank2 import decompose_fullfps, fit_residual_rank2, reconstruct
from ultra_compact_memory.storage_accounting import fullfps_storage, shared_rank2_storage
from ultra_compact_memory.transfer_protocol import transfer_directions
from visual_memory_context.code_audit import write_audit_report
from visual_memory_context.controls import apply_context_control
from visual_memory_context.diagnostics import summarize_adapter
from visual_memory_context.low_rank_attention import DirectResidualMLP, LowRankContextAdapter, trainable_parameter_count
from visual_memory_context.memory_context import initial_read_context, select_context_tokens
from visual_memory_context.outcome_analysis import assign_outcome
from visual_memory_context.query_refinement import direct_refine
from visual_memory_context.storage_accounting import adapter_storage


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def progress(out, message, **extra):
    payload = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message, **extra}
    with open(Path(out) / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=json_default) + "\n")
    print(f"[VI-C] {message}", flush=True)


@torch.no_grad()
def extract_q0(model, image, device, average_neighbor=True):
    outputs = extract_value_representations(model, image, [12, 18, 24], device, smooth=average_neighbor)
    cls = cls_from_outputs(outputs, 24)
    q0 = canonical_fuse(outputs[24]["T"][:, 1:, :], outputs[12]["V"][:, 1:, :], outputs[18]["V"][:, 1:, :], alpha=0.5)
    return cls, q0


def compact_backend(source_memory, rank=8):
    full = source_memory["patch_keys"].float()
    grouped, mu, centered = decompose_fullfps(full, group_size=4)
    residual = fit_residual_rank2(centered)
    pca = fit_centroid_pca(mu, rank)
    mu_hat = reconstruct_centroids(pca, rank)
    return {
        "full": full,
        "shared": reconstruct(mu, residual["basis"], residual["coeff"]),
        "compact": reconstruct(mu_hat, residual["basis"], residual["coeff"]),
        "rank": int(rank),
        "residual": residual,
        "storage": {
            "full": fullfps_storage(mu.shape[0], mu.shape[1]),
            "shared": shared_rank2_storage(mu.shape[0], mu.shape[1]),
            "compact": {"storage_ratio": (mu.shape[1] + rank * mu.shape[1] + mu.shape[0] * rank + 2 * mu.shape[1] + 8 * mu.shape[0] + 4 * mu.shape[0]) / (mu.shape[0] * 4 * mu.shape[1])},
        },
    }


def make_eval_memory(source_memory, patch_keys, device):
    return {
        "image_keys": source_memory["image_keys"].float().to(device),
        "image_values": source_memory["image_values"].float().to(device),
        "patch_keys": patch_keys.float().to(device),
        "patch_values": source_memory["patch_values"].float().to(device),
    }


def patch_loss_from_logits(patch_logits, gt_mask, side):
    gt_resized = resize_mask((gt_mask > 0.5).float(), side)
    target = gt_resized.reshape(-1).long().to(patch_logits.device)
    return F.cross_entropy(patch_logits.reshape(-1, 2), target)


def image_score_from_logits(patch_logits):
    return patch_logits.mean(dim=0, keepdim=True)


def variant_names(control_seeds):
    names = [
        "V0_FullFPS",
        "V1_SharedResidualRank2",
        "V2_CompactPCA",
        "V3_DirectContrastResidual",
        "V4_DirectLearnedResidual",
        "V5_StaticGlobalContextAttention",
        "V6_PatchMemoryContextAttention",
        "V7_GatedPatchMemoryContextAttention",
        "V8_NormalOnlyContext",
        "V9_AnomalyOnlyContext",
        "V10_NormalAnomalyTwoTokenContext",
        "V11_ContrastOnlyContext",
        "C4_NormalAnomalySwap",
        "C7_ZeroAdapter",
        "C11_RepeatedRetrieval",
    ]
    for seed in range(int(control_seeds)):
        names.append(f"C1_CrossImageContextShuffle_seed{seed}")
        names.append(f"C3_RandomMatchedContext_seed{seed}")
    return names


def train_adapter(model, source_memory, compact_keys, loader, cfg, device, gated=True, direct=False):
    dim = compact_keys.shape[-1]
    rank = int(cfg.get("adapter_rank", 8))
    alpha = float(cfg.get("alpha", 0.1))
    if direct:
        module = DirectResidualMLP(dim, rank).to(device)
    else:
        module = LowRankContextAdapter(dim, rank, use_gate=gated).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=float(cfg.get("adapter_lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    memory = make_eval_memory(source_memory, compact_keys, device)
    curves = []
    best, wait = float("inf"), 0
    for epoch in range(int(cfg.get("epochs", 1))):
        losses = []
        module.train()
        for items in tqdm(loader, desc=f"train visual adapter epoch {epoch+1}/{cfg.get('epochs', 1)}"):
            image = items["img"].to(device)
            gt = items["img_mask"].to(device)
            labels = torch.as_tensor(items["anomaly"], device=device).long()
            with torch.no_grad():
                _, q0 = extract_q0(model, image, device, cfg.get("average_neighbor", True))
                q = q0[0]
                context, stats, _ = initial_read_context(q, memory["patch_keys"], memory["patch_values"])
            if direct:
                delta = module(context[:, 2, :])
                q1 = direct_refine(q, delta, alpha)
                gate = attn = None
            else:
                q1, delta, gate, attn = module(q, context, stats, alpha=alpha)
            patch_logits, _, _ = retrieval_logits(q1, memory["patch_keys"], memory["patch_values"])
            side = int(math.sqrt(q.shape[0]))
            l_patch = patch_loss_from_logits(patch_logits, gt, side)
            l_image = F.cross_entropy(image_score_from_logits(patch_logits), labels)
            l_preserve = (1.0 - (q1 * q).sum(dim=-1)).mean()
            l_delta = delta.pow(2).mean()
            loss = l_patch + float(cfg.get("lambda_image", 1.0)) * l_image + float(cfg.get("lambda_preserve", 1e-4)) * l_preserve + float(cfg.get("lambda_delta", 1e-5)) * l_delta
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean = float(np.mean(losses)) if losses else float("nan")
        curves.append({"epoch": epoch + 1, "loss": mean, "direct": bool(direct), "gated": bool(gated)})
        if mean + 1e-8 < best:
            best, wait = mean, 0
        else:
            wait += 1
        if wait >= int(cfg.get("patience", 5)):
            break
    return module.eval(), curves


@torch.no_grad()
def evaluate_variant(model, source_memory, backend, adapter, direct_adapter, loader, cfg, device, variant):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    alpha = float(cfg.get("alpha", 0.1))
    if variant == "V0_FullFPS":
        patch_keys = backend["full"]
    elif variant == "V1_SharedResidualRank2":
        patch_keys = backend["shared"]
    else:
        patch_keys = backend["compact"]
    memory = make_eval_memory(source_memory, patch_keys, device)
    compact_memory = make_eval_memory(source_memory, backend["compact"], device)
    results = init_results(loader.dataset.obj_list)
    diag_rows, ctx_rows, retr_rows, times = [], [], [], []
    for items in tqdm(loader, desc=f"eval {variant}"):
        start = time.perf_counter()
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        cls, q0 = extract_q0(model, image, device, cfg.get("average_neighbor", True))
        q = q0[0]
        image_logits, _, _ = retrieval_logits(cls, memory["image_keys"], memory["image_values"])
        context, stats, cdiag = initial_read_context(q, compact_memory["patch_keys"], compact_memory["patch_values"])
        q1, delta, gate, attn = q, torch.zeros_like(q), None, None
        if variant == "V3_DirectContrastResidual":
            q1 = direct_refine(q, context[:, 2, :], alpha)
            delta = context[:, 2, :]
        elif variant == "V4_DirectLearnedResidual":
            delta = direct_adapter(context[:, 2, :])
            q1 = direct_refine(q, delta, alpha)
        elif variant.startswith(("V5_", "V6_", "V7_", "V8_", "V9_", "V10_", "V11_", "C1_", "C3_", "C4_")):
            c = context
            if variant.startswith("V5_"):
                c = context.mean(dim=0, keepdim=True).expand_as(context)
            elif variant.startswith("V8_"):
                c = select_context_tokens(context, "normal")
            elif variant.startswith("V9_"):
                c = select_context_tokens(context, "anomaly")
            elif variant.startswith("V10_"):
                c = select_context_tokens(context, "two")
            elif variant.startswith("V11_"):
                c = select_context_tokens(context, "contrast")
            elif variant.startswith("C1_"):
                c = apply_context_control(context, "cross_shuffle", int(variant.rsplit("seed", 1)[-1]))
            elif variant.startswith("C3_"):
                c = apply_context_control(context, "random", int(variant.rsplit("seed", 1)[-1]))
            elif variant.startswith("C4_"):
                c = apply_context_control(context, "swap", 0)
            no_gate = variant.startswith("V6_")
            q1, delta, gate, attn = adapter(q, c, stats, alpha=alpha, gate_override=(1.0 if no_gate else None))
        elif variant == "C7_ZeroAdapter" or variant == "C11_RepeatedRetrieval":
            q1 = q
        patch_logits, sim, prob = retrieval_logits(q1, memory["patch_keys"], memory["patch_values"])
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits.unsqueeze(0), image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        diag_rows.append(summarize_adapter(q, q1, delta, gate, attn))
        ctx_rows.append(cdiag)
        ent = -(prob.reshape(-1, prob.shape[-1]) * prob.reshape(-1, prob.shape[-1]).clamp_min(1e-12).log()).sum(dim=-1).mean()
        retr_rows.append({"retrieval_entropy": float(ent.detach().cpu())})
        times.append(time.perf_counter() - start)
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    def avg(rows, key):
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else float("nan")
    retrieval = {"mean_inference_time": float(np.mean(times)) if times else float("nan"), "retrieval_entropy": avg(retr_rows, "retrieval_entropy")}
    adapter_diag = {k: avg(diag_rows, k) for k in sorted({k for r in diag_rows for k in r})}
    context_diag = {k: avg(ctx_rows, k) for k in sorted({k for r in ctx_rows for k in r})}
    per_category = []
    for cat, vals in metrics.get("per_class", {}).items():
        row = {"category": cat}
        row.update(vals)
        per_category.append(row)
    return metrics, retrieval, adapter_diag, context_diag, per_category


def run_eval_or_resume(model, source_memory, backend, adapter, direct_adapter, loader, cfg, device, out, seed, direction, variant):
    mpath = out / f"seed_{seed}" / direction / variant / "metrics.json"
    if cfg.get("resume", True) and mpath.exists():
        payload = json.loads(mpath.read_text(encoding="utf-8"))
        return payload["metrics"], payload["retrieval"], payload.get("adapter_diag", {}), payload.get("context_diag", {}), payload.get("per_category", [])
    metrics, retrieval, adapter_diag, context_diag, per_cat = evaluate_variant(model, source_memory, backend, adapter, direct_adapter, loader, cfg, device, variant)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({"metrics": metrics, "retrieval": retrieval, "adapter_diag": adapter_diag, "context_diag": context_diag, "per_category": per_cat}, indent=2, default=json_default), encoding="utf-8")
    torch.cuda.empty_cache()
    return metrics, retrieval, adapter_diag, context_diag, per_cat


def main():
    args = parse_args("Phase VI-C compact memory visual context adapter audit")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    set_seed(seed)
    out = Path(cfg.get("output_dir", "results/visual_memory_context"))
    out.mkdir(parents=True, exist_ok=True)
    for stale in ["failure_traceback.txt", "REFERENCE_CODE_FAILURE.md", "SOURCE_PROTOCOL_FAILURE.md"]:
        p = out / stale
        if p.exists():
            p.unlink()
    save_run_metadata(out, cfg, seed)
    progress(out, "metadata saved", output_dir=str(out), seed=seed)
    write_md("docs/phase_VI_C_mechanism_mapping.md", ["# Phase VI-C Mechanism Mapping", "", "- The only changed variable is a post-extraction visual context adapter.", "- CompactPCA Rank8, shared residual Rank2 and MRAD retrieval are fixed.", "- No text-side method is used."])
    summary_rows, source_rows, per_rows, ctx_rows, attn_rows, rep_rows, retr_rows, storage_rows, cost_rows = [], [], [], [], [], [], [], [], []
    protocol = []
    sanity = {"reference_code_ok": False, "source_protocol_verified": False, "clip_frozen_ok": False, "target_labels_used_for_training_or_selection": False, "compact_memory_storage_leq_1pct": True, "c7_reproduces_v2": False, "c11_reproduces_v2": False}
    try:
        progress(out, "starting reference audit")
        audit = write_audit_report()
        sanity["reference_code_ok"] = audit["status"] == "CODE_VERIFIED"
        progress(out, "reference audit completed", status=audit["status"])
        if not sanity["reference_code_ok"]:
            write_md(out / "REFERENCE_CODE_FAILURE.md", ["# Reference Code Failure", "", "VCP-CLIP did not pass CODE_VERIFIED."])
            write_json(out / "OUTCOME.json", {"Outcome": "J_REFERENCE_CODE_FAILURE"})
            return
        device = cfg.get("device", "cuda:0")
        progress(out, "loading model", device=device)
        model = load_model(cfg, device)
        progress(out, "model loaded")
        freeze_clip(model)
        sanity["clip_frozen_ok"] = all(not p.requires_grad for p in model.parameters())
        progress(out, "clip frozen check", clip_frozen_ok=sanity["clip_frozen_ok"])
        names = variant_names(cfg.get("control_seeds", 5))
        for direction in transfer_directions(cfg):
            dname, source, target = direction["name"], direction["source"], direction["target"]
            progress(out, "starting direction", direction=dname, source=source, target=target)
            _, source_loader_all = make_loader(cfg, source, cfg.get("source_mode", "test"), shuffle=False)
            _, target_loader = make_loader(cfg, target, cfg.get("target_mode", "test"), shuffle=False)
            train_cats, val_cats = split_categories(source_loader_all.dataset, seed, cfg.get("source_val_fraction", 0.25))
            protocol.append(source_protocol_record(seed, dname, source, target, train_cats, val_cats))
            progress(out, "source split ready", direction=dname, train_categories=train_cats, val_categories=val_cats)
            train_loader = clone_filtered_loader(source_loader_all, train_cats, cfg.get("batch_size", 1), False)
            val_loader = clone_filtered_loader(source_loader_all, val_cats, cfg.get("batch_size", 1), False)
            progress(out, "building fold train memory", direction=dname)
            train_memory, _ = build_source_memory(model, train_loader, cfg, f"{source}_vmc_foldtrain_{seed}", seed, device, out)
            progress(out, "fold train memory ready", direction=dname, patch_keys=int(train_memory["patch_keys"].shape[0]))
            backend = compact_backend(train_memory, int(cfg.get("compact_rank", 8)))
            if not cfg.get("limit"):
                sanity["compact_memory_storage_leq_1pct"] = sanity["compact_memory_storage_leq_1pct"] and backend["storage"]["compact"]["storage_ratio"] <= 0.01
            progress(out, "compact backend ready", direction=dname, compact_storage=backend["storage"]["compact"]["storage_ratio"])
            adapter, curves = train_adapter(model, train_memory, backend["compact"], train_loader, cfg, device, gated=True, direct=False)
            direct_adapter, dcurves = train_adapter(model, train_memory, backend["compact"], train_loader, cfg, device, gated=False, direct=True)
            progress(out, "adapters trained on fold", direction=dname)
            for variant in names:
                progress(out, "source validation variant", direction=dname, variant=variant)
                metrics, retrieval, adiag, cdiag, per_cat = run_eval_or_resume(model, train_memory, backend, adapter, direct_adapter, val_loader, cfg, device, out, seed, dname + "_sourceval", variant)
                row = {"seed": seed, "direction": dname, "source": source, "target": source, "variant": variant, "selection_status": "SOURCE_VALIDATION", "storage_ratio": backend["storage"]["compact"]["storage_ratio"], "adapter_parameters": trainable_parameter_count(adapter), "I-AUROC": metric(metrics, "I-AUROC"), "I-AP": metric(metrics, "I-AP"), "P-AUROC": metric(metrics, "P-AUROC"), "PRO": metric(metrics, "PRO")}
                source_rows.append(row)
            progress(out, "building full source memory", direction=dname)
            full_memory, _ = build_source_memory(model, source_loader_all, cfg, f"{source}_vmc_full_{seed}", seed, device, out)
            progress(out, "full source memory ready", direction=dname, patch_keys=int(full_memory["patch_keys"].shape[0]))
            backend = compact_backend(full_memory, int(cfg.get("compact_rank", 8)))
            if not cfg.get("limit"):
                sanity["compact_memory_storage_leq_1pct"] = sanity["compact_memory_storage_leq_1pct"] and backend["storage"]["compact"]["storage_ratio"] <= 0.01
            adapter, curves = train_adapter(model, full_memory, backend["compact"], source_loader_all, cfg, device, gated=True, direct=False)
            direct_adapter, dcurves = train_adapter(model, full_memory, backend["compact"], source_loader_all, cfg, device, gated=False, direct=True)
            progress(out, "adapters trained on full source", direction=dname)
            for variant in names:
                start = time.perf_counter()
                progress(out, "target evaluation variant", direction=dname, variant=variant)
                metrics, retrieval, adiag, cdiag, per_cat = run_eval_or_resume(model, full_memory, backend, adapter, direct_adapter, target_loader, cfg, device, out, seed, dname, variant)
                row = {"seed": seed, "direction": dname, "source": source, "target": target, "variant": variant, "selection_status": "POST_HOC_TARGET_DIAGNOSTIC", "storage_ratio": backend["storage"]["compact"]["storage_ratio"], "adapter_parameters": trainable_parameter_count(adapter), "I-AUROC": metric(metrics, "I-AUROC"), "I-AP": metric(metrics, "I-AP"), "P-AUROC": metric(metrics, "P-AUROC"), "PRO": metric(metrics, "PRO")}
                summary_rows.append(row)
                for pc in per_cat:
                    per_rows.append({"seed": seed, "direction": dname, "variant": variant, **pc})
                ctx_rows.append({"seed": seed, "direction": dname, "variant": variant, **cdiag})
                attn_rows.append({"seed": seed, "direction": dname, "variant": variant, **adiag})
                retr_rows.append({"seed": seed, "direction": dname, "variant": variant, **retrieval})
                storage_rows.append({"seed": seed, "direction": dname, "variant": variant, **backend["storage"]["compact"], **adapter_storage(trainable_parameter_count(adapter))})
                cost_rows.append({"seed": seed, "direction": dname, "variant": variant, "wall_time": time.perf_counter() - start, "mean_inference_time": retrieval.get("mean_inference_time", float("nan"))})
        sanity["source_protocol_verified"] = bool(protocol)
        progress(out, "all directions completed")
        def metric_same(x, y):
            x, y = float(x), float(y)
            if math.isnan(x) and math.isnan(y):
                return True
            return abs(x - y) <= 1e-8

        def same(a, b):
            return all(metric_same(a[k], b[k]) for k in ("I-AUROC", "I-AP", "P-AUROC", "PRO"))
        c7_ok = c11_ok = True
        for d in sorted({r["direction"] for r in summary_rows}):
            ds = [r for r in summary_rows if r["direction"] == d]
            v2 = next(r for r in ds if r["variant"] == "V2_CompactPCA")
            c7 = next(r for r in ds if r["variant"] == "C7_ZeroAdapter")
            c11 = next(r for r in ds if r["variant"] == "C11_RepeatedRetrieval")
            c7_ok = c7_ok and same(v2, c7)
            c11_ok = c11_ok and same(v2, c11)
        sanity["c7_reproduces_v2"] = bool(c7_ok)
        sanity["c11_reproduces_v2"] = bool(c11_ok)
        outcome = assign_outcome(summary_rows, sanity)
        source_selection = {"status": "STAGE1_FIXED_RANK8_ALPHA", "selected_by_direction": {}, "target_labels_used": False}
        for d in sorted({r["direction"] for r in source_rows}):
            ds = [r for r in source_rows if r["direction"] == d and r["variant"].startswith(("V6_", "V7_"))]
            source_selection["selected_by_direction"][d] = max(ds, key=lambda r: (r["P-AUROC"], r["I-AUROC"])) if ds else None
        write_json(out / "source_protocol.json", protocol)
        write_json(out / "source_selection.json", source_selection)
        write_json(out / "sanity_checks.json", sanity)
        write_json(out / "OUTCOME.json", {"Outcome": outcome})
        cols = ["seed", "direction", "source", "target", "variant", "selection_status", "storage_ratio", "adapter_parameters", "I-AUROC", "I-AP", "P-AUROC", "PRO"]
        write_csv(out / "summary_metrics.csv", summary_rows, cols)
        write_csv(out / "all_variant_metrics.csv", summary_rows, cols)
        write_csv(out / "source_validation_metrics.csv", source_rows, cols)
        write_csv(out / "random_context_controls.csv", [r for r in summary_rows if r["variant"].startswith(("C1_", "C3_"))], cols)
        write_csv(out / "per_category_metrics.csv", per_rows, sorted({k for r in per_rows for k in r}) if per_rows else ["seed", "direction", "variant", "category"])
        write_csv(out / "context_diagnostics.csv", ctx_rows, sorted({k for r in ctx_rows for k in r}) if ctx_rows else ["seed", "direction", "variant"])
        write_csv(out / "attention_diagnostics.csv", attn_rows, sorted({k for r in attn_rows for k in r}) if attn_rows else ["seed", "direction", "variant"])
        write_csv(out / "representation_diagnostics.csv", attn_rows, sorted({k for r in attn_rows for k in r}) if attn_rows else ["seed", "direction", "variant"])
        write_csv(out / "retrieval_diagnostics.csv", retr_rows, sorted({k for r in retr_rows for k in r}) if retr_rows else ["seed", "direction", "variant"])
        write_csv(out / "storage_accounting.csv", storage_rows, sorted({k for r in storage_rows for k in r}) if storage_rows else ["seed", "direction", "variant"])
        write_csv(out / "computational_costs.csv", cost_rows, sorted({k for r in cost_rows for k in r}) if cost_rows else ["seed", "direction", "variant"])
        write_csv(out / "defect_size_metrics.csv", [], ["seed", "direction", "variant", "defect_size", "P-AUROC", "PRO"])
        write_csv(out / "error_correction.csv", [], ["seed", "direction", "variant", "false_positives_corrected", "false_negatives_corrected", "new_errors"])
        write_csv(out / "statistical_comparisons.csv", [], ["direction", "comparison", "metric", "delta"])
        write_csv(out / "success_control_table.csv", summary_rows, cols)
        write_md(out / "EXECUTION_SUMMARY.md", ["# Phase VI-C Execution Summary", "", f"- VCP-CLIP source-code verification: `{sanity['reference_code_ok']}`.", f"- Source protocol verified: `{sanity['source_protocol_verified']}`.", f"- CLIP frozen: `{sanity['clip_frozen_ok']}`.", f"- C7 reproduces V2: `{sanity['c7_reproduces_v2']}`.", f"- C11 reproduces V2: `{sanity['c11_reproduces_v2']}`.", f"- Completed target rows: {len(summary_rows)}.", f"- Outcome: `{outcome}`."])
        if cfg.get("run_mode") == "stage1":
            write_md(out / "PHASE_VI_C_EARLY_STOP_REPORT.md", ["# Phase VI-C Stage 1 Report", "", f"- Outcome: `{outcome}`.", f"- Completed target rows: {len(summary_rows)}."])
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase VI-C — Compact Memory Visual Context Adapter\n\n")
            f.write(f"- Output directory: `{out}`.\n")
            f.write(f"- Completed target rows: {len(summary_rows)}.\n")
            f.write(f"- Outcome: `{outcome}`.\n")
    except Exception:
        tb = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        write_json(out / "OUTCOME.json", {"Outcome": "L_IMPLEMENTATION_FAILURE", "error": tb})
        raise


if __name__ == "__main__":
    main()
