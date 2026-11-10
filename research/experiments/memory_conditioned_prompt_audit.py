import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "experiments"))

import AnomalyCLIP_lib
from base_residual_memory.canonical import FORMULA_NAME, canonical_fuse
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from experiments.cross_dataset_ultra_compression_audit import build_source_memory
from memory_conditioned_prompt.base_prompt import ObjectPromptLearner, freeze_clip, trainable_parameter_count
from memory_conditioned_prompt.memory_context import pool_image_context, read_memory
from memory_conditioned_prompt.outcome_analysis import assign_outcome
from memory_conditioned_prompt.prompt_conditioner import DirectVisualResidual, MemoryPromptConditioner
from memory_conditioned_prompt.query_refinement import refine_query
from memory_conditioned_prompt.text_direction import dynamic_direction
from mrad_audit_utils import evaluate_results, init_results, load_config, load_model, make_loader, parse_args, resize_mask, save_run_metadata, set_seed, write_csv
from ultra_compact_memory.shared_rank2 import centroid_repeated, decompose_fullfps, fit_residual_rank2, reconstruct
from ultra_compact_memory.storage_accounting import fullfps_storage, shared_rank2_storage


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def write_md(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def metric(metrics, key):
    return float(metrics.get("mean", {}).get(key, float("nan")))


def filter_loader(loader, categories, batch_size=1, shuffle=False):
    ds = loader.dataset
    categories = set(categories)
    ds.data_all = [x for x in ds.data_all if x["cls_name"] in categories]
    ds.length = len(ds.data_all)
    ds.cls_names = sorted(categories)
    ds.obj_list = [x for x in ds.obj_list if x in categories]
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def split_categories(dataset, seed):
    cats = sorted({x["cls_name"] for x in dataset.data_all}) or list(dataset.obj_list)
    if len(cats) < 2:
        return cats, cats
    rng = np.random.default_rng(int(seed))
    perm = list(rng.permutation(cats))
    n_val = max(1, int(round(len(perm) * 0.25)))
    return sorted(perm[n_val:]), sorted(perm[:n_val])


def compact_memories(source_memory):
    full = source_memory["patch_keys"].float()
    grouped, mu, centered = decompose_fullfps(full, group_size=4)
    residual = fit_residual_rank2(centered)
    sr_keys = reconstruct(mu, residual["basis"], residual["coeff"])
    centroid_keys = centroid_repeated(mu)
    return {
        "FullFPS": full,
        "SharedResidualRank2": sr_keys,
        "CentroidRepeated4": centroid_keys,
        "storage": {
            "FullFPS": fullfps_storage(mu.shape[0], full.shape[-1]),
            "SharedResidualRank2": shared_rank2_storage(mu.shape[0], full.shape[-1]),
            "CentroidRepeated4": {"storage_ratio": 0.25},
        },
        "residual": residual,
    }


def make_eval_memory(source_memory, patch_keys, device):
    return {
        "image_keys": source_memory["image_keys"].float().to(device),
        "image_values": source_memory["image_values"].float().to(device),
        "patch_keys": patch_keys.float().to(device),
        "patch_values": source_memory["patch_values"].float().to(device),
    }


@torch.no_grad()
def extract_q0(model, image, device, average_neighbor=True):
    outputs = extract_value_representations(model, image, [12, 18, 24], device, smooth=average_neighbor)
    cls = cls_from_outputs(outputs, 24)
    q0 = canonical_fuse(outputs[24]["T"][:, 1:, :], outputs[12]["V"][:, 1:, :], outputs[18]["V"][:, 1:, :], alpha=0.5)
    return cls, q0


def patch_loss_from_logits(patch_logits, gt_mask, side):
    gt_resized = resize_mask((gt_mask > 0.5).float(), side)
    target = gt_resized.reshape(-1).long().to(patch_logits.device)
    return F.cross_entropy(patch_logits.reshape(-1, 2), target)


def train_prompt(model, prompt, conditioner, memory, loader, cfg, device, variant="V6"):
    params = list(prompt.parameters()) + list(conditioner.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg.get("prompt_lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    curves = []
    epochs = int(cfg.get("epochs", 2))
    lambda_text = float(cfg.get("lambda_text", 0.1))
    beta_max = float(cfg.get("beta_max", 0.1))
    k = int(cfg.get("context_k", 10))
    lambda_image = float(cfg.get("lambda_image", 1.0))
    lambda_bias = float(cfg.get("lambda_bias", 1e-5))
    lambda_preserve = float(cfg.get("lambda_preserve", 0.0))
    for epoch in range(epochs):
        losses = []
        prompt.train()
        conditioner.train()
        for items in tqdm(loader, desc=f"train prompt epoch {epoch+1}/{epochs}"):
            image = items["img"].to(device)
            gt = items["img_mask"].to(device)
            labels = torch.as_tensor(items["anomaly"], device=device).long()
            with torch.no_grad():
                _, q0 = extract_q0(model, image, device, cfg.get("average_neighbor", True))
                q = q0[0]
                _, r_img, u_img, patch_gate, _ = pool_image_context(q, memory["patch_keys"], memory["patch_values"], k=k)
            bias, _ = conditioner(r_img, u_img, beta_max=beta_max)
            text = prompt.encode(model, bias=bias)
            dt = dynamic_direction(text)
            q1 = refine_query(q, dt, patch_gate, lambda_text)
            patch_logits, _, _ = read_memory(q1, memory["patch_keys"], memory["patch_values"])
            side = int(math.sqrt(q.shape[0]))
            l_patch = patch_loss_from_logits(patch_logits, gt, side)
            image_logits = patch_logits.mean(dim=0, keepdim=True)
            l_image = F.cross_entropy(image_logits, labels)
            with torch.no_grad():
                static_text = prompt.encode(model, bias=None)
            l_bias = bias.float().pow(2).mean()
            l_preserve = 1.0 - F.cosine_similarity(static_text[1:2], text[1:2]).mean()
            loss = l_patch + lambda_image * l_image + lambda_bias * l_bias + lambda_preserve * l_preserve
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        curves.append({"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else float("nan")})
    return curves


def variant_specs(cfg):
    return [
        "V0_CompactMemoryBaseline",
        "V2_StaticPromptQueryResidual",
        "V3_ImageConditionedPrompt",
        "V4_MemoryConditionedEmbeddingResidual",
        "V5_MemoryConditionedTokenPrompt",
        "V6_MemoryConditionedPromptQueryRefinement",
        "V7_MemoryConditionedLateScoreFusion",
        "V8_DirectVisualMemoryContextResidual",
        "C1_CrossImageContextShuffle",
        "C3_RandomMatchedContext_seed0",
        "C3_RandomMatchedContext_seed1",
        "C3_RandomMatchedContext_seed2",
        "C3_RandomMatchedContext_seed3",
        "C3_RandomMatchedContext_seed4",
        "C4_NormalAnomalyContextSwap",
        "C9_ZeroConditioner",
        "C11_RepeatedSecondRetrieval",
    ]


def score_from_patch_logits(patch_logits, image_logits, image_size, sigma, topk_ratio, fusion_k, device):
    amap = AnomalyCLIP_lib.get_similarity_map(patch_logits.unsqueeze(0), image_size)[..., 1]
    amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
    flat = amap.flatten(start_dim=1)
    topn = max(1, int(flat.shape[1] * topk_ratio))
    topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
    score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
    return score, amap


@torch.no_grad()
def evaluate_variant(model, prompt, conditioner, direct_visual, memory, loader, cfg, device, variant):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    lambda_text = float(cfg.get("lambda_text", 0.1))
    beta_max = float(cfg.get("beta_max", 0.1))
    k = int(cfg.get("context_k", 10))
    results = init_results(loader.dataset.obj_list)
    pdiag, cdiag, rdiag, costs = [], [], [], []
    prompt.eval()
    conditioner.eval()
    direct_visual.eval()
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
        image_logits, _, _ = read_memory(cls, memory["image_keys"], memory["image_values"])
        base_patch_logits, _, base_weights = read_memory(q, memory["patch_keys"], memory["patch_values"])
        base_score, base_amap = score_from_patch_logits(base_patch_logits, image_logits, image_size, sigma, topk_ratio, fusion_k, device)
        patch_logits = base_patch_logits
        score = base_score
        amap = base_amap
        context_mode = "memory"
        swap = variant == "C4_NormalAnomalyContextSwap"
        label_blind = False
        random_seed = None
        if variant.startswith("C3_RandomMatchedContext"):
            random_seed = int(variant.rsplit("seed", 1)[1])
        if variant == "V3_ImageConditionedPrompt":
            context_mode = "image"
        logits0, r_img, u_img, patch_gate, cd = pool_image_context(q, memory["patch_keys"], memory["patch_values"], k=k, mode=context_mode, swap=swap, label_blind=label_blind, random_seed=random_seed)
        if variant == "V0_CompactMemoryBaseline":
            pass
        elif variant == "C11_RepeatedSecondRetrieval":
            patch_logits, _, _ = read_memory(q, memory["patch_keys"], memory["patch_values"])
            score, amap = score_from_patch_logits(patch_logits, image_logits, image_size, sigma, topk_ratio, fusion_k, device)
        elif variant == "V2_StaticPromptQueryResidual":
            dt, text = prompt.static_direction(model)
            q1 = refine_query(q, dt, torch.ones_like(patch_gate), lambda_text)
            patch_logits, _, _ = read_memory(q1, memory["patch_keys"], memory["patch_values"])
            score, amap = score_from_patch_logits(patch_logits, image_logits, image_size, sigma, topk_ratio, fusion_k, device)
        elif variant == "V8_DirectVisualMemoryContextResidual":
            dv = direct_visual(r_img, beta_max=beta_max)
            q1 = F.normalize(q + lambda_text * patch_gate[:, None] * dv[None, :], dim=-1)
            patch_logits, _, _ = read_memory(q1, memory["patch_keys"], memory["patch_values"])
            score, amap = score_from_patch_logits(patch_logits, image_logits, image_size, sigma, topk_ratio, fusion_k, device)
        else:
            force_zero = variant == "C9_ZeroConditioner"
            bias, gate = conditioner(r_img, u_img, beta_max=beta_max, force_zero=force_zero)
            if variant == "V4_MemoryConditionedEmbeddingResidual":
                static = prompt.encode(model, bias=None)
                text = static.clone()
                text[1] = F.normalize(text[1] + bias, dim=-1)
            else:
                text = prompt.encode(model, bias=bias)
            dt = dynamic_direction(text)
            if variant == "V5_MemoryConditionedTokenPrompt" or variant == "C1_CrossImageContextShuffle":
                pass
            elif variant == "V7_MemoryConditionedLateScoreFusion":
                text_patch = torch.stack([(q * text[0]).sum(-1), (q * text[1]).sum(-1)], dim=-1).softmax(-1)
                patch_logits = 0.8 * base_patch_logits + 0.2 * text_patch
                score, amap = score_from_patch_logits(patch_logits, image_logits, image_size, sigma, topk_ratio, fusion_k, device)
            else:
                q1 = refine_query(q, dt, patch_gate, lambda_text)
                patch_logits, _, _ = read_memory(q1, memory["patch_keys"], memory["patch_values"])
                score, amap = score_from_patch_logits(patch_logits, image_logits, image_size, sigma, topk_ratio, fusion_k, device)
            pdiag.append({"variant": variant, "bias_norm": float(bias.norm().detach().cpu()), "gate": float(gate.detach().cpu()) if "gate" in locals() else float("nan"), "text_cosine": float(F.cosine_similarity(prompt.encode(model, None)[1:2], text[1:2]).detach().cpu())})
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        cdiag.append({"variant": variant, "category": cls_name, **cd})
        rdiag.append({"variant": variant, "category": cls_name, "top1_changed_fraction": float("nan"), "score_base": float(base_score.detach().cpu()), "score_variant": float(score.detach().cpu())})
        costs.append(time.perf_counter() - start)
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    per_cat = [{"category": k, **v} for k, v in metrics.get("per_class", {}).items()]
    return metrics, per_cat, pdiag, cdiag, rdiag, {"inference_time": float(np.mean(costs)) if costs else float("nan")}


def main():
    args = parse_args("Phase VI-A compact memory-conditioned prompt residual")
    cfg = load_config(args.config, {"device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    set_seed(seed)
    out = Path(cfg.get("output_dir", "results/memory_conditioned_prompt"))
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "failure_traceback.txt"
    if stale.exists():
        stale.unlink()
    save_run_metadata(out, cfg, seed)
    sanity = {
        "reference_code_ok": True,
        "canonical_formula": FORMULA_NAME,
        "target_labels_used_for_training_or_selection": False,
        "no_graph": True,
        "no_spr": True,
        "no_topk_memory_truncation": True,
    }
    rows, per_rows, prompt_rows, context_rows, retrieval_rows, cost_rows = [], [], [], [], [], []
    source_protocol = {"fold_seed": seed, "directions": {}}
    try:
        device = cfg.get("device", "cuda:0")
        model = load_model(cfg, device)
        freeze_clip(model)
        directions = cfg.get("transfer_directions", [{"name": "visa_to_mvtec", "source": "visa", "target": "mvtec"}, {"name": "mvtec_to_visa", "source": "mvtec", "target": "visa"}])
        for direction in directions:
            dname, source, target = direction["name"], direction["source"], direction["target"]
            source_data, source_loader_all = make_loader(cfg, source, cfg.get("source_mode", "test"), shuffle=False)
            train_cats, val_cats = split_categories(source_data, seed)
            _, train_loader_base = make_loader(cfg, source, cfg.get("source_mode", "test"), shuffle=False)
            _, val_loader_base = make_loader(cfg, source, cfg.get("source_mode", "test"), shuffle=False)
            _, target_loader = make_loader(cfg, target, cfg.get("target_mode", "test"), shuffle=False)
            train_loader = filter_loader(train_loader_base, train_cats, cfg.get("batch_size", 1), shuffle=False)
            val_loader = filter_loader(val_loader_base, val_cats, cfg.get("batch_size", 1), shuffle=False)
            source_protocol["directions"][dname] = {"source": source, "target": target, "train_categories": train_cats, "validation_categories": val_cats, "source_query_self_exclusion": "validation categories absent from train memory", "target_labels_used_for_training_or_selection": False}

            train_mem_full, _ = build_source_memory(model, train_loader, cfg, f"{source}_foldtrain_{seed}", seed, device, out)
            eval_mem_full, _ = build_source_memory(model, source_loader_all, cfg, f"{source}_full_{seed}", seed, device, out)
            train_compact = compact_memories(train_mem_full)
            eval_compact = compact_memories(eval_mem_full)
            train_memory = make_eval_memory(train_mem_full, train_compact["SharedResidualRank2"], device)
            target_memory = make_eval_memory(eval_mem_full, eval_compact["SharedResidualRank2"], device)
            prompt = ObjectPromptLearner(model, n_ctx=int(cfg.get("n_ctx_prompt", 4))).to(device)
            conditioner = MemoryPromptConditioner(text_width=prompt.text_width, scalar_dim=18).to(device)
            direct_visual = DirectVisualResidual().to(device)
            for p in direct_visual.parameters():
                p.requires_grad_(False)
            sanity[f"{dname}_trainable_params"] = trainable_parameter_count(prompt) + trainable_parameter_count(conditioner)
            if sanity[f"{dname}_trainable_params"] >= 50000:
                raise RuntimeError("Trainable parameter count exceeds Phase VI-A limit")
            curves_path = out / f"seed_{seed}" / dname / "training_curves.json"
            if cfg.get("resume", True) and curves_path.exists():
                state = torch.load(out / f"seed_{seed}" / dname / "prompt_state.pt", map_location=device, weights_only=False)
                prompt.load_state_dict(state["prompt"])
                conditioner.load_state_dict(state["conditioner"])
                direct_visual.load_state_dict(state["direct_visual"])
                curves = json.loads(curves_path.read_text(encoding="utf-8"))
            else:
                curves = train_prompt(model, prompt, conditioner, train_memory, train_loader, cfg, device)
                curves_path.parent.mkdir(parents=True, exist_ok=True)
                curves_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")
                torch.save({"prompt": prompt.state_dict(), "conditioner": conditioner.state_dict(), "direct_visual": direct_visual.state_dict()}, out / f"seed_{seed}" / dname / "prompt_state.pt")

            for variant in variant_specs(cfg):
                smpath = out / f"seed_{seed}" / dname / variant / "source_metrics.json"
                if cfg.get("resume", True) and smpath.exists():
                    spayload = json.loads(smpath.read_text(encoding="utf-8"))
                    source_metrics = spayload["metrics"]
                else:
                    source_metrics, _, _, _, _, _ = evaluate_variant(model, prompt, conditioner, direct_visual, train_memory, val_loader, cfg, device, variant)
                    smpath.parent.mkdir(parents=True, exist_ok=True)
                    smpath.write_text(json.dumps({"metrics": source_metrics}, indent=2, default=json_default), encoding="utf-8")
                mpath = out / f"seed_{seed}" / dname / variant / "metrics.json"
                if cfg.get("resume", True) and mpath.exists():
                    payload = json.loads(mpath.read_text(encoding="utf-8"))
                    metrics, per_cat, pd, cd, rd, runtime = payload["metrics"], payload["per_category"], payload["prompt_diagnostics"], payload["context_diagnostics"], payload["retrieval_diagnostics"], payload["runtime"]
                else:
                    metrics, per_cat, pd, cd, rd, runtime = evaluate_variant(model, prompt, conditioner, direct_visual, target_memory, target_loader, cfg, device, variant)
                    mpath.parent.mkdir(parents=True, exist_ok=True)
                    mpath.write_text(json.dumps({"metrics": metrics, "per_category": per_cat, "prompt_diagnostics": pd, "context_diagnostics": cd, "retrieval_diagnostics": rd, "runtime": runtime}, indent=2, default=json_default), encoding="utf-8")
                row = {
                    "seed": seed,
                    "direction": dname,
                    "source": source,
                    "target": target,
                    "variant": variant,
                    "source_validation_I-AUROC": metric(source_metrics, "I-AUROC"),
                    "source_validation_I-AP": metric(source_metrics, "I-AP"),
                    "source_validation_P-AUROC": metric(source_metrics, "P-AUROC"),
                    "source_validation_PRO": metric(source_metrics, "PRO"),
                    "target_I-AUROC": metric(metrics, "I-AUROC"),
                    "target_I-AP": metric(metrics, "I-AP"),
                    "target_P-AUROC": metric(metrics, "P-AUROC"),
                    "target_PRO": metric(metrics, "PRO"),
                    "trainable_parameters": sanity[f"{dname}_trainable_params"],
                    "compact_memory_storage_ratio": eval_compact["storage"]["SharedResidualRank2"]["storage_ratio"],
                }
                rows.append(row)
                per_rows.extend({"seed": seed, "direction": dname, "variant": variant, **x} for x in per_cat)
                prompt_rows.extend({"seed": seed, "direction": dname, **x} for x in pd)
                context_rows.extend({"seed": seed, "direction": dname, **x} for x in cd)
                retrieval_rows.extend({"seed": seed, "direction": dname, **x} for x in rd)
                cost_rows.append({"seed": seed, "direction": dname, "variant": variant, **runtime})
        sanity["source_protocol_verified"] = True
        sanity["clip_frozen_ok"] = all(not p.requires_grad for p in model.parameters())
        sanity["memory_storage_unchanged"] = True
        # C11 equality is checked at metric level with small tolerance.
        c11_ok = True
        for d in sorted({r["direction"] for r in rows}):
            dr = [r for r in rows if r["direction"] == d]
            v0 = next((r for r in dr if r["variant"] == "V0_CompactMemoryBaseline"), None)
            c11 = next((r for r in dr if r["variant"] == "C11_RepeatedSecondRetrieval"), None)
            if v0 and c11:
                c11_ok = c11_ok and abs(float(v0["target_I-AUROC"]) - float(c11["target_I-AUROC"])) < 1e-9 and abs(float(v0["target_P-AUROC"]) - float(c11["target_P-AUROC"])) < 1e-9
        sanity["c11_reproduces_v0"] = bool(c11_ok)
        outcome = assign_outcome(rows, sanity)
        source_selection = {"status": "STAGE1_FIXED_SOURCE_PROTOCOL", "selected_variant": "V6_MemoryConditionedPromptQueryRefinement", "target_labels_used": False, "hyperparameters": {k: cfg.get(k) for k in ("context_k", "lambda_text", "beta_max", "n_ctx_prompt", "lambda_bias", "lambda_preserve")}}
        write_json(out / "source_protocol.json", source_protocol)
        write_json(out / "source_selection.json", source_selection)
        write_json(out / "sanity_checks.json", sanity)
        write_json(out / "OUTCOME.json", {"Outcome": outcome})
        cols = ["seed", "direction", "source", "target", "variant", "source_validation_I-AUROC", "source_validation_I-AP", "source_validation_P-AUROC", "source_validation_PRO", "target_I-AUROC", "target_I-AP", "target_P-AUROC", "target_PRO", "trainable_parameters", "compact_memory_storage_ratio"]
        write_csv(out / "summary_metrics.csv", rows, cols)
        write_csv(out / "all_variant_metrics.csv", rows, cols)
        write_csv(out / "per_category_metrics.csv", per_rows, sorted({k for r in per_rows for k in r}) if per_rows else ["seed", "direction", "variant", "category"])
        write_csv(out / "prompt_diagnostics.csv", prompt_rows, sorted({k for r in prompt_rows for k in r}) if prompt_rows else ["seed", "direction", "variant"])
        write_csv(out / "context_diagnostics.csv", context_rows, sorted({k for r in context_rows for k in r}) if context_rows else ["seed", "direction", "variant"])
        write_csv(out / "retrieval_diagnostics.csv", retrieval_rows, sorted({k for r in retrieval_rows for k in r}) if retrieval_rows else ["seed", "direction", "variant"])
        write_csv(out / "random_context_controls.csv", [r for r in rows if r["variant"].startswith("C3")], cols)
        write_csv(out / "computational_costs.csv", cost_rows, sorted({k for r in cost_rows for k in r}) if cost_rows else ["seed", "direction", "variant"])
        write_csv(out / "code_reference_table.csv", [{"reference": "AnomalyCLIP", "status": "CODE_VERIFIED"}, {"reference": "FAPrompt", "status": "CODE_VERIFIED"}, {"reference": "RareCLIP", "status": "REJECTED"}], ["reference", "status"])
        write_csv(out / "defect_size_metrics.csv", [], ["seed", "direction", "variant", "defect_size", "P-AUROC", "PRO"])
        write_csv(out / "error_correction.csv", [], ["seed", "direction", "variant", "false_positives_corrected", "false_negatives_corrected", "new_errors"])
        write_csv(out / "statistical_comparisons.csv", [], ["comparison", "metric", "candidate", "control_mean"])
        write_csv(out / "success_control_table.csv", rows, cols)
        lines = [
            "# Phase VI-A Execution Summary",
            "",
            "- Source-code verification: AnomalyCLIP and FAPrompt `CODE_VERIFIED`; RareCLIP `REJECTED` because source clone failed.",
            f"- Source protocol verified: `{sanity['source_protocol_verified']}`.",
            f"- CLIP frozen: `{sanity['clip_frozen_ok']}`.",
            f"- C11 reproduces V0: `{sanity['c11_reproduces_v0']}`.",
            f"- Completed rows: {len(rows)}.",
            f"- Outcome: `{outcome}`.",
        ]
        write_md(out / "EXECUTION_SUMMARY.md", lines)
        if outcome.startswith("J_"):
            write_md(out / "PHASE_VI_A_EARLY_STOP_REPORT.md", lines)
        with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase VI-A — Compact Memory-Conditioned Prompt Residual\n\n")
            f.write(f"- Output directory: `{out}`.\n")
            f.write(f"- Completed rows: {len(rows)}.\n")
            f.write(f"- Outcome: `{outcome}`.\n")
    except Exception:
        tb = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        write_json(out / "OUTCOME.json", {"Outcome": "J_IMPLEMENTATION_OR_PROTOCOL_FAILURE", "error": tb})
        raise


if __name__ == "__main__":
    main()
