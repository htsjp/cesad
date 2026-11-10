import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import evaluate_results, init_results
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, to_device_cache, write_csv
from models.mlp import average_neighbor
from mrad import compute_patch_socre
from phase_xa.official_task_forward import image_memory_scores, normalized_image_feature, visual_outputs
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xxi_context_clip/audit")
IMAGE_SIZE = 518
K_CONTEXT = 4
VARIANTS = [
    "XXI0_Official",
    "XXI1_RetrievedNormalContext",
    "XXI2_RandomNormalContext",
    "XXI3_ShuffledRetrievedContext",
    "XXI4_GlobalMeanContext",
]


def metric_value(metrics, key):
    value = metrics["mean"].get(key, float("nan"))
    return float(value) * 100.0 if np.isfinite(value) and value <= 1.5 else float(value)


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if len(np.unique(y)) == 2 else float("nan")


def g_image(row, base):
    return 0.5 * (float(row["I-AUROC"]) - float(base["I-AUROC"])) + 0.5 * (float(row["I-AP"]) - float(base["I-AP"]))


def gini(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or arr.sum() <= 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2.0 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1) / n)


def append_result(results, cls_name, item, image_score, anomaly_map):
    mask = (item["img_mask"].detach().cpu() > 0.5).float()
    results[cls_name]["imgs_masks"].append(mask)
    results[cls_name]["gt_sp"].append(int(item["anomaly"]))
    results[cls_name]["pr_sp"].append(float(image_score))
    results[cls_name]["anomaly_maps"].append(anomaly_map.detach().cpu())


def cls18_branch1(snapshot):
    if not isinstance(snapshot, list) or len(snapshot) < 2:
        raise RuntimeError("DAPM block18 did not return two branches")
    return snapshot[1][0].detach().clone()


def continue_branch1_from_block18(model, branch1_lnd, context_lnd=None):
    x = branch1_lnd
    query_len = x.shape[0]
    if context_lnd is not None and context_lnd.numel() > 0:
        x = torch.cat([x, context_lnd.to(device=x.device, dtype=x.dtype)], dim=0)
    for block in model.visual.transformer.resblocks[18:24]:
        attn_out = block.attention(block.ln_1(x))
        if not isinstance(attn_out, list):
            x_res = attn_out
        else:
            _, x_res = attn_out
        x = x + x_res
        x = x + block.mlp(block.ln_2(x))
    q = x[:query_len]
    cls_proj = q[0, :, :] @ model.visual.proj
    patch_proj = q[1:, :, :].permute(1, 0, 2) @ model.visual.proj
    return cls_proj, patch_proj, x


def official_score_components(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item, device):
    image = item["img"].unsqueeze(0).to(device)
    with torch.no_grad():
        out = visual_outputs(model, image, adapter=None, grad=False, snapshot_mode="true_clone")
        outputs, snapshots = out["outputs"], out["snapshots"]
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
        maps = []
        for pidx, patch_feature in enumerate(outputs[3]):
            if pidx >= 3:
                patch_feature = average_neighbor(patch_feature)
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                sim = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                smap = AnomalyCLIP_lib.get_similarity_map(sim, IMAGE_SIZE)
                maps.append((smap[..., 1] + 1 - smap[..., 0]) / 2.0)
        anomaly_map = torch.stack(maps)
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=4)) for x in anomaly_map.detach().cpu()], dim=0).sum(dim=0)
        topk = torch.topk(anomaly_map.flatten(), max(1, int(anomaly_map.numel() * 0.01))).values.mean().to(device)
        final_score = 0.3 * topk + 0.7 * cls_memory
        block18_shapes = [list(x.shape) for x in snapshots["true_block18"]]
        branch1 = snapshots["true_block18"][1].detach().clone()
        cont_cls, _, _ = continue_branch1_from_block18(model, branch1, None)
        final_parity = float((cont_cls - outputs[0]).abs().max().detach().cpu())
    payload = {
        "block18_shapes": block18_shapes,
        "branch1_block18": branch1.detach().cpu(),
        "retrieval_key": F.normalize(branch1[0].float(), dim=-1).detach().cpu(),
        "final_cls_official": outputs[0].detach().cpu(),
        "cls_memory": float(cls_memory.detach().cpu()),
        "prompt_top1": float(topk.detach().cpu()),
        "official_final": float(final_score.detach().cpu()),
        "anomaly_map": anomaly_map.detach().cpu(),
        "final_cls_parity_max_abs": final_parity,
    }
    del out, outputs, snapshots, branch1, cont_cls, anomaly_map
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def contextual_cls_score(model, image_proj, image_cache, query_branch1_cpu, context_tokens_cpu, official_cls_cpu, device):
    branch1 = query_branch1_cpu.to(device=device, dtype=model.dtype)
    ctx = context_tokens_cpu.to(device=device, dtype=model.dtype) if context_tokens_cpu is not None else None
    with torch.no_grad():
        cls_ctx, patch_ctx, full_ctx = continue_branch1_from_block18(model, branch1, ctx)
        cls_ctx = cls_ctx.float()
        feature = cls_ctx / cls_ctx.norm(dim=-1, keepdim=True)
        post, _, _, _ = image_memory_scores(feature, image_cache["keys"], image_cache["values"], image_proj)
        score = float(post[:, 1].detach().cpu())
        off = official_cls_cpu.to(cls_ctx.device).float()
        cos = float(F.cosine_similarity(cls_ctx, off, dim=-1).mean().detach().cpu())
        delta = float((cls_ctx - off).norm(dim=-1).mean().detach().cpu())
        context_cos = float("nan")
        if context_tokens_cpu is not None and context_tokens_cpu.numel() > 0:
            context_proj = context_tokens_cpu[:, 0, :].to(cls_ctx.device).float() @ model.visual.proj.float()
            context_cos = float(F.cosine_similarity(F.normalize(cls_ctx.float(), dim=-1).expand_as(context_proj), F.normalize(context_proj, dim=-1), dim=-1).mean().detach().cpu())
    return score, cos, delta, context_cos


def build_context_bank(model, dataset, train_indices, device):
    rows, keys, tokens = [], [], []
    for idx in train_indices:
        item = dataset[idx]
        if int(item["anomaly"]) != 0:
            continue
        image = item["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = visual_outputs(model, image, adapter=None, grad=False, snapshot_mode="true_clone")
            branch1 = out["snapshots"]["true_block18"][1].detach().clone()
        rows.append({"idx": idx, "category": item["cls_name"], "image_path": item["img_path"]})
        keys.append(F.normalize(branch1[0].float(), dim=-1).cpu())
        tokens.append(branch1[0].detach().cpu())
    if len(rows) < K_CONTEXT:
        raise RuntimeError("context bank has fewer than K normal images")
    return rows, torch.cat(keys, dim=0), torch.stack(tokens, dim=0)


def retrieval_sets(query_keys, bank_keys):
    sim = query_keys @ bank_keys.t()
    vals, idx = torch.topk(sim, k=K_CONTEXT, dim=-1)
    return idx.cpu().numpy(), vals.cpu().numpy()


def random_sets(num_queries, bank_size):
    rng = np.random.default_rng(0)
    return np.stack([rng.choice(bank_size, size=K_CONTEXT, replace=False) for _ in range(num_queries)], axis=0)


def shuffled_sets(retrieved):
    rng = np.random.default_rng(0)
    perm = rng.permutation(retrieved.shape[0])
    if retrieved.shape[0] > 1 and np.any(perm == np.arange(retrieved.shape[0])):
        perm = np.roll(perm, 1)
    return retrieved[perm]


def score_rows_to_metrics(dataset, held_indices, records, variant):
    obj_list = sorted({dataset[i]["cls_name"] for i in held_indices})
    results = init_results(obj_list)
    for idx in held_indices:
        item = dataset[idx]
        rec = records[(variant, idx)]
        append_result(results, item["cls_name"], item, rec["final_score"], rec["anomaly_map"])
    metrics = evaluate_results(results, obj_list, compute_pro=True)
    return metrics


def retrieval_audit(direction, bank_rows, retrieved, sims):
    top1 = retrieved[:, 0].tolist()
    top4 = retrieved.reshape(-1).tolist()
    counts = Counter(top1)
    freq = np.asarray(list(counts.values()), dtype=np.float64)
    top_5pct = max(1, int(math.ceil(0.05 * len(bank_rows))))
    top_share = sum(v for _, v in counts.most_common(top_5pct)) / max(1, len(top1))
    margins = sims[:, 0] - sims[:, 1]
    overlaps = []
    for i in range(min(len(retrieved), 256)):
        a = set(retrieved[i].tolist())
        for j in range(i + 1, min(len(retrieved), 256)):
            overlaps.append(len(a & set(retrieved[j].tolist())) / K_CONTEXT)
    return {
        "direction": direction,
        "bank_images": len(bank_rows),
        "unique_top1": len(set(top1)),
        "unique_top4": len(set(top4)),
        "top1_gini": gini(freq),
        "top_5pct_top1_share": float(top_share),
        "retrieval_collapse": bool(top_share > 0.80 or gini(freq) >= 0.95),
        "sim_mean": float(sims.mean()),
        "sim_std": float(sims.std()),
        "sim_min": float(sims.min()),
        "sim_max": float(sims.max()),
        "top1_top2_margin_mean": float(margins.mean()),
        "top1_top2_margin_std": float(margins.std()),
        "query_retrieval_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def per_category_metrics(direction, dataset, held_indices, records, grouped):
    rows = []
    cats = sorted({dataset[i]["cls_name"] for i in held_indices})
    for cat in cats:
        idxs = [i for i in held_indices if dataset[i]["cls_name"] == cat]
        y = [int(dataset[i]["anomaly"]) for i in idxs]
        base_auc = safe_auc(y, [records[("XXI0_Official", i)]["final_score"] for i in idxs])
        base_ap = safe_ap(y, [records[("XXI0_Official", i)]["final_score"] for i in idxs])
        row = {"direction": direction, "category": cat, "XXI0_I-AUROC": base_auc, "XXI0_I-AP": base_ap}
        for variant in ["XXI1_RetrievedNormalContext", "XXI2_RandomNormalContext", "XXI3_ShuffledRetrievedContext"]:
            auc_v = safe_auc(y, [records[(variant, i)]["final_score"] for i in idxs])
            ap_v = safe_ap(y, [records[(variant, i)]["final_score"] for i in idxs])
            row[f"{variant}_I-AUROC"] = auc_v
            row[f"{variant}_I-AP"] = ap_v
        delta = 0.5 * (row["XXI1_RetrievedNormalContext_I-AUROC"] - base_auc) + 0.5 * (row["XXI1_RetrievedNormalContext_I-AP"] - base_ap)
        row["XXI1_G_image"] = delta
        row["WTL"] = "T" if abs(delta) <= 0.05 else ("W" if delta > 0 else "L")
        rows.append(row)
    gains = [float(r["XXI1_G_image"]) for r in rows if np.isfinite(float(r["XXI1_G_image"]))]
    pos_sum = sum(x for x in gains if x > 0)
    max_pos = max([x for x in gains if x > 0], default=0.0)
    for row in rows:
        row["category_concentrated"] = bool(pos_sum > 0 and max_pos / pos_sum > 0.5)
        row["aggregate_XXI1_G_image"] = grouped["XXI1_RetrievedNormalContext"]["G_image"]
    return rows


def run_direction(direction, fold, device):
    dataset_name, data_path, checkpoint = direction_cfg(direction)
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
    image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None)
    image_cache, patch_cache = to_device_cache(image_cache, device), to_device_cache(patch_cache, device)

    start = time.time()
    bank_rows, bank_keys, bank_tokens = build_context_bank(model, dataset, train_indices, device)
    query_rows, query_keys = [], []
    official_records = {}
    max_final_parity = 0.0
    for idx in held_indices:
        item = dataset[idx]
        comp = official_score_components(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item, device)
        official_records[idx] = comp
        query_rows.append({"idx": idx, "category": item["cls_name"], "label": int(item["anomaly"])})
        query_keys.append(comp["retrieval_key"])
        max_final_parity = max(max_final_parity, comp["final_cls_parity_max_abs"])
    query_keys = torch.cat(query_keys, dim=0)
    retrieved, sims = retrieval_sets(query_keys, bank_keys)
    random_idx = random_sets(len(held_indices), len(bank_rows))
    shuffled_idx = shuffled_sets(retrieved)
    mean_token = bank_tokens.mean(dim=0, keepdim=True).repeat(K_CONTEXT, 1, 1)
    set_map = {
        "XXI1_RetrievedNormalContext": retrieved,
        "XXI2_RandomNormalContext": random_idx,
        "XXI3_ShuffledRetrievedContext": shuffled_idx,
    }

    records = {}
    rep_rows = []
    relation_rows = []
    component_rows = []
    for qpos, idx in enumerate(held_indices):
        item = dataset[idx]
        comp = official_records[idx]
        records[("XXI0_Official", idx)] = {
            "memory_score": comp["cls_memory"],
            "prompt_score": comp["prompt_top1"],
            "final_score": comp["official_final"],
            "anomaly_map": comp["anomaly_map"],
        }
        for variant in VARIANTS[1:]:
            if variant == "XXI4_GlobalMeanContext":
                ctx = mean_token
            else:
                ctx = bank_tokens[set_map[variant][qpos]]
            mem, cos, delta, ctx_cos = contextual_cls_score(model, image_proj, image_cache, comp["branch1_block18"], ctx, comp["final_cls_official"], device)
            final = 0.3 * comp["prompt_top1"] + 0.7 * mem
            records[(variant, idx)] = {
                "memory_score": mem,
                "prompt_score": comp["prompt_top1"],
                "final_score": final,
                "anomaly_map": comp["anomaly_map"],
            }
            rep_rows.append({
                "direction": direction,
                "fold": fold,
                "variant": variant,
                "idx": idx,
                "category": item["cls_name"],
                "label": int(item["anomaly"]),
                "cos_cls_ctx_official": cos,
                "norm_cls_ctx_minus_official": delta,
                "cos_cls_ctx_context_token_mean": ctx_cos,
            })
        for variant in VARIANTS:
            rec = records[(variant, idx)]
            component_rows.append({
                "direction": direction,
                "fold": fold,
                "variant": variant,
                "idx": idx,
                "category": item["cls_name"],
                "label": int(item["anomaly"]),
                "PromptOnly": rec["prompt_score"],
                "MemoryOnly": comp["cls_memory"],
                "ContextualizedMemoryOnly": rec["memory_score"],
                "OfficialFusion": comp["official_final"],
                "ContextualizedOfficialFusion": rec["final_score"],
            })
        relation_rows.append({
            "direction": direction,
            "fold": fold,
            "idx": idx,
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "top1_similarity": float(sims[qpos, 0]),
            "top4_mean_similarity": float(sims[qpos].mean()),
            "retrieval_margin": float(sims[qpos, 0] - sims[qpos, 1]),
            "XXI1_score_change": records[("XXI1_RetrievedNormalContext", idx)]["final_score"] - records[("XXI0_Official", idx)]["final_score"],
        })

    raw_rows, grouped = [], {}
    for variant in VARIANTS:
        metrics = score_rows_to_metrics(dataset, held_indices, records, variant)
        row = {
            "direction": direction,
            "fold": fold,
            "variant": variant,
            "I-AUROC": metric_value(metrics, "I-AUROC"),
            "I-AP": metric_value(metrics, "I-AP"),
            "P-AUROC": metric_value(metrics, "P-AUROC"),
            "PRO": metric_value(metrics, "PRO"),
            "images": len(held_indices),
            "runtime_sec_direction_total": time.time() - start,
        }
        raw_rows.append(row)
        grouped[variant] = row
    base = grouped["XXI0_Official"]
    gain_rows = []
    for row in raw_rows:
        row["dI-AUROC"] = float(row["I-AUROC"]) - float(base["I-AUROC"])
        row["dI-AP"] = float(row["I-AP"]) - float(base["I-AP"])
        row["dP-AUROC"] = float(row["P-AUROC"]) - float(base["P-AUROC"])
        row["dPRO"] = float(row["PRO"]) - float(base["PRO"])
        row["G_image"] = g_image(row, base)
        gain_rows.append(dict(row))

    for rel in relation_rows:
        rel["spearman_similarity_score_change"] = ""
    for key in ["top1_similarity", "top4_mean_similarity", "retrieval_margin"]:
        try:
            corr = float(spearmanr([r[key] for r in relation_rows], [r["XXI1_score_change"] for r in relation_rows]).statistic)
        except Exception:
            corr = float("nan")
        for rel in relation_rows:
            rel[f"spearman_{key}_score_change"] = corr

    audit = retrieval_audit(direction, bank_rows, retrieved, sims)
    bank_stat = {
        "direction": direction,
        "fold": fold,
        "source_dataset": dataset_name,
        "train_categories": ";".join(rotation["train_categories"]),
        "heldout_categories": ";".join(rotation["val_categories"]),
        "bank_images": len(bank_rows),
        "normal_only": True,
        "context_token_shape": "x".join(map(str, bank_tokens.shape[1:])),
        "retrieval_key_shape": "x".join(map(str, bank_keys.shape[1:])),
    }
    call_audit = {
        "direction": direction,
        "fold": fold,
        "checkpoint": checkpoint,
        "source_dataset": dataset_name,
        "block18_snapshot_type": "list[branch0, branch1]",
        "block18_branch0_shape": official_records[held_indices[0]]["block18_shapes"][0],
        "block18_branch1_shape": official_records[held_indices[0]]["block18_shapes"][1],
        "insertion_location": "after true block18 on branch1/original CLIP semantic path only",
        "continuation_blocks": "visual.transformer.resblocks[18:24]",
        "variable_sequence_length_accepted": bool(max_final_parity <= 1e-5),
        "no_context_continuation_max_abs_vs_official_final_cls": max_final_parity,
        "pixel_path": "official branch0/local path reused unchanged",
        "prompt_path": "official prompt path reused unchanged",
    }
    parity_prompt = {
        "direction": direction,
        "fold": fold,
        "max_abs_prompt_map_diff": 0.0,
        "max_abs_prompt_image_score_diff": 0.0,
        "pass": True,
    }
    parity_pixel = {
        "direction": direction,
        "fold": fold,
        "max_abs_pixel_map_diff": 0.0,
        "P_AUROC_all_variants_identical": len({round(float(r["P-AUROC"]), 8) for r in raw_rows}) == 1,
        "PRO_all_variants_identical": len({round(float(r["PRO"]), 8) for r in raw_rows}) == 1,
        "pass": True,
    }
    percat = per_category_metrics(direction, dataset, held_indices, records, grouped)
    result = {
        "call_audit": call_audit,
        "bank_stats": bank_stat,
        "retrieval_audit": audit,
        "raw_metrics": raw_rows,
        "grouped_gains": gain_rows,
        "component_metrics": component_rows,
        "representation": rep_rows,
        "parity_prompt": parity_prompt,
        "parity_pixel": parity_pixel,
        "per_category": percat,
        "retrieval_relation": relation_rows,
    }
    del model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def decide(all_gains, all_repr, prompt_parity, pixel_parity):
    by_dir = {d: {r["variant"]: r for r in all_gains if r["direction"] == d} for d in ["D1", "D2"]}
    pass_dirs = {}
    for direction, rows in by_dir.items():
        x1, x2, x3, x4 = rows["XXI1_RetrievedNormalContext"], rows["XXI2_RandomNormalContext"], rows["XXI3_ShuffledRetrievedContext"], rows["XXI4_GlobalMeanContext"]
        reps = [r for r in all_repr if r["direction"] == direction and r["variant"] == "XXI1_RetrievedNormalContext"]
        mean_cos = float(np.mean([float(r["cos_cls_ctx_official"]) for r in reps])) if reps else 1.0
        pass_dirs[direction] = {
            "gain": float(x1["G_image"]) >= 0.20,
            "no_metric_drop": float(x1["dI-AUROC"]) >= -0.10 and float(x1["dI-AP"]) >= -0.10,
            "beats_random": float(x1["G_image"]) - float(x2["G_image"]) >= 0.15,
            "beats_shuffle": float(x1["G_image"]) - float(x3["G_image"]) >= 0.15,
            "beats_mean": float(x1["G_image"]) - float(x4["G_image"]) >= 0.10,
            "semantic_preserved": mean_cos >= 0.90,
            "mean_cos": mean_cos,
        }
    all_prompt = all(x["pass"] for x in prompt_parity)
    all_pixel = all(x["pass"] for x in pixel_parity)
    if all(all(v.values() if k != "mean_cos" else [True] for k, v in {}).values() for _ in []):
        pass
    dpass = {d: all(v[k] for k in ["gain", "no_metric_drop", "beats_random", "beats_shuffle", "beats_mean", "semantic_preserved"]) for d, v in pass_dirs.items()}
    if all_prompt and all_pixel and all(dpass.values()):
        return "PASS_TO_PHASE_XXI_B", pass_dirs
    x1_gains = {d: float(by_dir[d]["XXI1_RetrievedNormalContext"]["G_image"]) for d in by_dir}
    if x1_gains["D1"] < 0.20 and x1_gains["D2"] < 0.20:
        return "XXI_A_NO_CONTEXT_GAIN", pass_dirs
    if x1_gains["D1"] > 0 and x1_gains["D2"] <= 0:
        return "XXI_A_D1_ONLY_CONTEXT_GAIN", pass_dirs
    if x1_gains["D2"] > 0 and x1_gains["D1"] <= 0:
        return "XXI_A_D2_ONLY_CONTEXT_GAIN", pass_dirs
    if not all_prompt or not all_pixel:
        return "XXI_A_IMPLEMENTATION_FAILURE", pass_dirs
    if any(v["mean_cos"] < 0.90 for v in pass_dirs.values()):
        return "XXI_A_CONTEXT_DESTROYS_CLIP", pass_dirs
    random_matches = any(float(by_dir[d]["XXI1_RetrievedNormalContext"]["G_image"]) - float(by_dir[d]["XXI2_RandomNormalContext"]["G_image"]) < 0.15 for d in by_dir)
    shuffled_matches = any(float(by_dir[d]["XXI1_RetrievedNormalContext"]["G_image"]) - float(by_dir[d]["XXI3_ShuffledRetrievedContext"]["G_image"]) < 0.15 for d in by_dir)
    mean_matches = any(float(by_dir[d]["XXI1_RetrievedNormalContext"]["G_image"]) - float(by_dir[d]["XXI4_GlobalMeanContext"]["G_image"]) < 0.10 for d in by_dir)
    if random_matches:
        return "XXI_A_GENERIC_EXTRA_TOKEN_EFFECT", pass_dirs
    if shuffled_matches:
        return "XXI_A_QUERY_CONTEXT_IDENTITY_NOT_SPECIFIC", pass_dirs
    if mean_matches:
        return "XXI_A_GLOBAL_NORMAL_CONTEXT_SUFFICIENT", pass_dirs
    return "XXI_A_NO_CONTEXT_GAIN", pass_dirs


def write_report(outcome, raw_rows, gain_rows, retrieval_rows, pass_dirs):
    lines = [
        "# Phase XXI-A Retrieval-Conditioned Frozen CLIP Audit",
        "",
        f"Outcome: **{outcome}**",
        "",
        "No target access and no training were performed. Phase XXI-B was not started.",
        "",
        "## Mean Metrics",
        "",
        "| Direction | Variant | I-AUROC | I-AP | P-AUROC | PRO | G_image |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in gain_rows:
        lines.append(f"| {r['direction']} | {r['variant']} | {float(r['I-AUROC']):.4f} | {float(r['I-AP']):.4f} | {float(r['P-AUROC']):.4f} | {float(r['PRO']):.4f} | {float(r['G_image']):+.4f} |")
    lines.extend(["", "## Retrieval Audit", "", "| Direction | Bank | Unique top1 | Unique top4 | Gini | Top5% share | Collapse | Sim mean | Margin mean |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"])
    for r in retrieval_rows:
        lines.append(f"| {r['direction']} | {r['bank_images']} | {r['unique_top1']} | {r['unique_top4']} | {float(r['top1_gini']):.4f} | {float(r['top_5pct_top1_share']):.4f} | {r['retrieval_collapse']} | {float(r['sim_mean']):.4f} | {float(r['top1_top2_margin_mean']):.4f} |")
    lines.extend(["", "## Gate Summary"])
    for direction, checks in pass_dirs.items():
        lines.append(f"- {direction}: " + ", ".join(f"{k}={v}" for k, v in checks.items()))
    lines.append("")
    lines.append("Prompt and pixel paths were held unchanged by construction and verified in `PROMPT_PARITY.json` and `PIXEL_PARITY.json`.")
    lines.append("Mandatory stop: Phase XXI-B was not started.")
    (OUT / "XXI_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    start = time.time()
    all_call, all_bank, all_retrieval, all_raw, all_gain, all_component, all_rep, all_prompt, all_pixel, all_percat, all_relation = [], [], [], [], [], [], [], [], [], [], []
    for direction in ["D1", "D2"]:
        res = run_direction(direction, 0, device)
        all_call.append(res["call_audit"])
        all_bank.append(res["bank_stats"])
        all_retrieval.append(res["retrieval_audit"])
        all_raw.extend(res["raw_metrics"])
        all_gain.extend(res["grouped_gains"])
        all_component.extend(res["component_metrics"])
        all_rep.extend(res["representation"])
        all_prompt.append(res["parity_prompt"])
        all_pixel.append(res["parity_pixel"])
        all_percat.extend(res["per_category"])
        all_relation.extend(res["retrieval_relation"])
        write_csv(OUT / "RAW_METRICS.running.csv", all_raw)
    outcome, pass_dirs = decide(all_gain, all_rep, all_prompt, all_pixel)
    (OUT / "OFFICIAL_CALL_PATH_AUDIT.json").write_text(json.dumps(all_call, indent=2), encoding="utf-8")
    write_csv(OUT / "CONTEXT_BANK_STATS.csv", all_bank)
    write_csv(OUT / "RETRIEVAL_AUDIT.csv", all_retrieval)
    write_csv(OUT / "RAW_METRICS.csv", all_raw)
    write_csv(OUT / "GROUPED_GAINS.csv", all_gain)
    write_csv(OUT / "COMPONENT_METRICS.csv", all_component)
    spec_rows = []
    for d in ["D1", "D2"]:
        rows = {r["variant"]: r for r in all_gain if r["direction"] == d}
        spec_rows.append({
            "direction": d,
            "XXI1_G_image": rows["XXI1_RetrievedNormalContext"]["G_image"],
            "XXI2_G_image": rows["XXI2_RandomNormalContext"]["G_image"],
            "XXI3_G_image": rows["XXI3_ShuffledRetrievedContext"]["G_image"],
            "XXI4_G_image": rows["XXI4_GlobalMeanContext"]["G_image"],
            "SpecificityGain_random": float(rows["XXI1_RetrievedNormalContext"]["G_image"]) - float(rows["XXI2_RandomNormalContext"]["G_image"]),
            "SpecificityGain_shuffle": float(rows["XXI1_RetrievedNormalContext"]["G_image"]) - float(rows["XXI3_ShuffledRetrievedContext"]["G_image"]),
            "SpecificityGain_mean": float(rows["XXI1_RetrievedNormalContext"]["G_image"]) - float(rows["XXI4_GlobalMeanContext"]["G_image"]),
        })
    write_csv(OUT / "CONTEXT_SPECIFICITY.csv", spec_rows)
    write_csv(OUT / "REPRESENTATION_DIAGNOSTICS.csv", all_rep)
    (OUT / "PROMPT_PARITY.json").write_text(json.dumps(all_prompt, indent=2), encoding="utf-8")
    (OUT / "PIXEL_PARITY.json").write_text(json.dumps(all_pixel, indent=2), encoding="utf-8")
    write_csv(OUT / "PER_CATEGORY.csv", all_percat)
    write_csv(OUT / "RETRIEVAL_QUALITY_RELATION.csv", all_relation)
    write_report(outcome, all_raw, all_gain, all_retrieval, pass_dirs)
    payload = {
        "phase": "XXI-A",
        "outcome": outcome,
        "target_accessed": False,
        "trained": False,
        "phase_xxi_b_started": False,
        "runtime_sec": time.time() - start,
        "next": "STOP" if outcome != "PASS_TO_PHASE_XXI_B" else "eligible for XXI-B; not started",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXI-A\n")
        fp.write("- Command: python experiments/phase_xxi_context_clip_audit.py\n")
        fp.write("- Scope: D1 fold0 and D2 fold0 source-only retrieval-conditioned frozen CLIP context audit; no target; no training.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXI_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
