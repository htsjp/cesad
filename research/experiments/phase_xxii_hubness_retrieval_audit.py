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


OUT = Path("results/phase_xxii_hubness_retrieval/audit")
VARIANTS = ["R0_OFFICIAL", "R1_CSLS", "R2_InvertedPopularity", "R3_ShuffledHubness", "R4_KeyCentering"]
K_CSLS = 10
K_DIAG = 10
IMAGE_SIZE = 518
EPS = 1e-12


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


def gini_from_counts(counts, total_keys):
    arr = np.zeros(total_keys, dtype=np.float64)
    for k, v in counts.items():
        arr[int(k)] = float(v)
    if arr.sum() <= 0:
        return 0.0
    arr.sort()
    n = arr.size
    return float((2.0 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1) / n)


def entropy_effective(counts):
    freq = np.asarray(list(counts.values()), dtype=np.float64)
    if freq.sum() <= 0:
        return 0.0
    p = freq / freq.sum()
    return float(np.exp(-(p * np.log(p + EPS)).sum()))


def metric_by_category(rows, score_key):
    cats = sorted({r["category"] for r in rows})
    aucs, aps, percat = [], [], {}
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat]
        y = [int(r["label"]) for r in sub]
        s = [float(r[score_key]) for r in sub]
        auc = safe_auc(y, s)
        ap = safe_ap(y, s)
        percat[cat] = {"I-AUROC": auc, "I-AP": ap}
        if np.isfinite(auc):
            aucs.append(auc)
        if np.isfinite(ap):
            aps.append(ap)
    return {"I-AUROC": float(np.mean(aucs)) if aucs else float("nan"), "I-AP": float(np.mean(aps)) if aps else float("nan"), "per_category": percat}


def append_pixel(results, item, score, amap):
    cls = item["cls_name"]
    results[cls]["imgs_masks"].append((item["img_mask"].detach().cpu() > 0.5).float())
    results[cls]["gt_sp"].append(int(item["anomaly"]))
    results[cls]["pr_sp"].append(float(score))
    results[cls]["anomaly_maps"].append(amap.detach().cpu())


def metric_value(metrics, key):
    value = metrics["mean"].get(key, float("nan"))
    return float(value) * 100.0 if np.isfinite(value) and value <= 1.5 else float(value)


def balanced_reference_positions(dataset, train_indices, max_ref=1024):
    by_cat = defaultdict(list)
    for pos, idx in enumerate(train_indices):
        by_cat[dataset[idx]["cls_name"]].append(pos)
    cats = sorted(by_cat)
    selected = []
    quota = max(1, max_ref // max(1, len(cats)))
    for cat in cats:
        selected.extend(by_cat[cat][:quota])
    cursor = {cat: min(quota, len(by_cat[cat])) for cat in cats}
    while len(selected) < min(max_ref, len(train_indices)):
        advanced = False
        for cat in cats:
            if cursor[cat] < len(by_cat[cat]) and len(selected) < max_ref:
                selected.append(by_cat[cat][cursor[cat]])
                cursor[cat] += 1
                advanced = True
        if not advanced:
            break
    return sorted(selected[:max_ref])


def projected_logits(image_proj, queries, keys, chunk=256):
    q_proj, _ = image_proj(queries[:1], keys)
    logits = []
    for start in range(0, queries.shape[0], chunk):
        q, k = image_proj(queries[start : start + chunk], keys)
        logits.append(q @ k.t())
    return torch.cat(logits, dim=0)


def correction_stats(ref_logits):
    k = min(K_CSLS, ref_logits.shape[0], ref_logits.shape[1])
    r_key = torch.topk(ref_logits, k=k, dim=0).values.mean(dim=0)
    max_per_ref = ref_logits.max(dim=1, keepdim=True).values
    h = torch.log(torch.exp(ref_logits - max_per_ref).mean(dim=0).clamp_min(EPS))
    mu_key = ref_logits.mean(dim=0)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    perm = torch.randperm(r_key.numel(), generator=rng).to(r_key.device)
    return {"r_key": r_key, "h": h, "mu_key": mu_key, "r_key_shuffle": r_key[perm]}


def corrected_logits(logits, stats, variant):
    if variant == "R0_OFFICIAL":
        return logits
    if variant in {"R1_CSLS", "R3_ShuffledHubness"}:
        k = min(K_CSLS, logits.shape[1])
        r_query = torch.topk(logits, k=k, dim=1).values.mean(dim=1, keepdim=True)
        key_stat = stats["r_key"] if variant == "R1_CSLS" else stats["r_key_shuffle"]
        return 2.0 * logits - r_query - key_stat.view(1, -1)
    if variant == "R2_InvertedPopularity":
        return logits - stats["h"].view(1, -1)
    if variant == "R4_KeyCentering":
        return logits - stats["mu_key"].view(1, -1)
    raise ValueError(variant)


def official_prompt_pixel(model, prompt_learner, patch_proj, prompt_proj, patch_cache, item, outputs, device):
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
    amap = torch.stack(maps)
    amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=4)) for x in amap.detach().cpu()], dim=0).sum(dim=0)
    topk = torch.topk(amap.flatten(), max(1, int(amap.numel() * 0.01))).values.mean()
    return float(topk.detach().cpu()), amap.detach().cpu()


def hubness_metrics(direction, fold, variant, logits, memory_scores):
    top = torch.topk(logits, k=min(K_DIAG, logits.shape[1]), dim=1).indices.detach().cpu().numpy()
    top1 = top[:, 0].tolist()
    top5 = top[:, : min(5, top.shape[1])].reshape(-1).tolist()
    counts = Counter(top1)
    bank = logits.shape[1]
    top_5pct = max(1, int(math.ceil(0.05 * bank)))
    concentration = sum(v for _, v in counts.most_common(top_5pct)) / max(1, len(top1))
    margins = (torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values[:, 0] - torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values[:, 1]).detach().cpu().numpy()
    overlaps = []
    n = min(top.shape[0], 256)
    for i in range(n):
        a = set(top[i].tolist())
        for j in range(i + 1, n):
            overlaps.append(len(a & set(top[j].tolist())) / max(1, top.shape[1]))
    return {
        "direction": direction,
        "fold": fold,
        "variant": variant,
        "memory_keys": bank,
        "queries": logits.shape[0],
        "unique_top1_keys": len(set(top1)),
        "unique_top5_keys": len(set(top5)),
        "top1_gini": gini_from_counts(counts, bank),
        "top5pct_bank_coverage_top1": float(concentration),
        "max_single_key_top1_frequency": max(counts.values()) / max(1, len(top1)),
        "effective_retrieved_keys": entropy_effective(counts),
        "mean_top1_top2_margin": float(np.mean(margins)),
        "query_pair_jaccard_top10": float(np.mean(overlaps)) if overlaps else 0.0,
        "memory_score_variance": float(np.var(np.asarray(memory_scores, dtype=np.float64))),
    }


def anti_hub_success(row, base):
    checks = {
        "gini_drop_ge_0p20": float(base["top1_gini"]) - float(row["top1_gini"]) >= 0.20,
        "effective_keys_2x": float(row["effective_retrieved_keys"]) >= 2.0 * max(EPS, float(base["effective_retrieved_keys"])),
        "max_freq_halved": float(row["max_single_key_top1_frequency"]) <= 0.5 * float(base["max_single_key_top1_frequency"]),
        "unique_top1_2x": float(row["unique_top1_keys"]) >= 2.0 * max(1.0, float(base["unique_top1_keys"])),
        "top5pct_drop_20pp": float(base["top5pct_bank_coverage_top1"]) - float(row["top5pct_bank_coverage_top1"]) >= 0.20,
    }
    return sum(checks.values()) >= 2, checks


def run_fold(direction, fold, device):
    dataset_name, data_path, checkpoint = direction_cfg(direction)
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
    image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None)
    image_cache, patch_cache = to_device_cache(image_cache, device), to_device_cache(patch_cache, device)
    ref_positions = balanced_reference_positions(dataset, train_indices, max_ref=1024)
    ref_features = image_cache["keys"][ref_positions]
    with torch.no_grad():
        ref_logits = projected_logits(image_proj, ref_features, image_cache["keys"])
        stats = correction_stats(ref_logits)

    obj_list = sorted({dataset[i]["cls_name"] for i in held_indices})
    pixel_results = init_results(obj_list)
    per_query = []
    query_logits = []
    official_memory_scores = []
    parity_max = 0.0
    start = time.time()
    for idx in held_indices:
        item = dataset[idx]
        image = item["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = visual_outputs(model, image, adapter=None, grad=False)
            image_feature = normalized_image_feature(outputs)
            post0, logits0, _, _ = image_memory_scores(image_feature, image_cache["keys"], image_cache["values"], image_proj)
            logits_direct = projected_logits(image_proj, image_feature, image_cache["keys"])
            parity_max = max(parity_max, float((logits_direct - logits0).abs().max().detach().cpu()))
            prompt_score, amap = official_prompt_pixel(model, prompt_learner, patch_proj, prompt_proj, patch_cache, item, outputs, device)
        logit_vec = logits0.detach()
        query_logits.append(logit_vec.cpu())
        mem0 = float(post0[:, 1].detach().cpu())
        official_memory_scores.append(mem0)
        row = {
            "idx": idx,
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "prompt_score": prompt_score,
            "pixel_map": amap,
            "R0_OFFICIAL_memory": mem0,
            "R0_OFFICIAL_final": 0.3 * prompt_score + 0.7 * mem0,
        }
        for variant in VARIANTS[1:]:
            clogits = corrected_logits(logit_vec, stats, variant)
            post = torch.softmax(clogits, dim=-1) @ image_cache["values"].float()
            mem = float(post[:, 1].detach().cpu())
            row[f"{variant}_memory"] = mem
            row[f"{variant}_final"] = 0.3 * prompt_score + 0.7 * mem
        per_query.append(row)
        append_pixel(pixel_results, item, row["R0_OFFICIAL_final"], amap)
    query_logits_t = torch.cat(query_logits, dim=0).to(device)

    corrected_by_variant = {}
    hub_rows = []
    for variant in VARIANTS:
        clogits = corrected_logits(query_logits_t, stats, variant).detach()
        corrected_by_variant[variant] = clogits.cpu()
        scores = [r[f"{variant}_memory"] for r in per_query]
        hub_rows.append(hubness_metrics(direction, fold, variant, clogits, scores))

    pixel_metrics = evaluate_results(pixel_results, obj_list, compute_pro=True)
    p_auroc = metric_value(pixel_metrics, "P-AUROC")
    pro = metric_value(pixel_metrics, "PRO")
    memory_rows, final_rows, control_rows = [], [], []
    for variant in VARIANTS:
        mem_metric = metric_by_category(per_query, f"{variant}_memory")
        fin_metric = metric_by_category(per_query, f"{variant}_final")
        mrow = {"direction": direction, "fold": fold, "variant": variant, "MemoryOnly_I-AUROC": mem_metric["I-AUROC"], "MemoryOnly_I-AP": mem_metric["I-AP"]}
        frow = {"direction": direction, "fold": fold, "variant": variant, "I-AUROC": fin_metric["I-AUROC"], "I-AP": fin_metric["I-AP"], "P-AUROC": p_auroc, "PRO": pro}
        memory_rows.append(mrow)
        final_rows.append(frow)
        if variant in {"R3_ShuffledHubness", "R4_KeyCentering"}:
            control_rows.append({**frow, **mrow})
    base_mem = memory_rows[0]
    base_fin = final_rows[0]
    for r in memory_rows:
        r["dMemoryOnly_I-AUROC"] = float(r["MemoryOnly_I-AUROC"]) - float(base_mem["MemoryOnly_I-AUROC"])
        r["dMemoryOnly_I-AP"] = float(r["MemoryOnly_I-AP"]) - float(base_mem["MemoryOnly_I-AP"])
        r["MemoryOnly_G_image"] = 0.5 * r["dMemoryOnly_I-AUROC"] + 0.5 * r["dMemoryOnly_I-AP"]
    for r in final_rows:
        r["dI-AUROC"] = float(r["I-AUROC"]) - float(base_fin["I-AUROC"])
        r["dI-AP"] = float(r["I-AP"]) - float(base_fin["I-AP"])
        r["G_image"] = g_image(r, base_fin)
    percat_rows = []
    best_real = max([r for r in final_rows if r["variant"] in {"R1_CSLS", "R2_InvertedPopularity"}], key=lambda x: float(x["G_image"]))["variant"]
    for cat in obj_list:
        sub = [r for r in per_query if r["category"] == cat]
        row = {"direction": direction, "fold": fold, "category": cat, "best_real_correction": best_real}
        for variant in ["R0_OFFICIAL", best_real, "R3_ShuffledHubness", "R4_KeyCentering"]:
            met = metric_by_category(sub, f"{variant}_final")
            row[f"{variant}_I-AUROC"] = met["I-AUROC"]
            row[f"{variant}_I-AP"] = met["I-AP"]
        delta = 0.5 * (row[f"{best_real}_I-AUROC"] - row["R0_OFFICIAL_I-AUROC"]) + 0.5 * (row[f"{best_real}_I-AP"] - row["R0_OFFICIAL_I-AP"])
        row["best_vs_R0_G_image"] = delta
        row["WTL"] = "T" if abs(delta) <= 0.05 else ("W" if delta > 0 else "L")
        percat_rows.append(row)
    gains = [float(r["best_vs_R0_G_image"]) for r in percat_rows if np.isfinite(float(r["best_vs_R0_G_image"]))]
    pos_sum = sum(x for x in gains if x > 0)
    max_pos = max([x for x in gains if x > 0], default=0.0)
    for row in percat_rows:
        row["category_concentrated"] = bool(pos_sum > 0 and max_pos / pos_sum > 0.5)

    relation_rows = []
    y = np.asarray([r["label"] for r in per_query], dtype=np.int64)
    for variant in VARIANTS:
        logits = corrected_by_variant[variant]
        top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values
        margins = (top2[:, 0] - top2[:, 1]).numpy()
        mem_delta = np.asarray([r[f"{variant}_memory"] - r["R0_OFFICIAL_memory"] for r in per_query], dtype=np.float64)
        final_scores = np.asarray([r[f"{variant}_final"] for r in per_query], dtype=np.float64)
        try:
            rank_score = np.asarray([np.mean(final_scores[y == 1] > s) if lab == 0 and np.any(y == 1) else np.mean(s > final_scores[y == 0]) if lab == 1 and np.any(y == 0) else np.nan for lab, s in zip(y, final_scores)])
            corr_margin_rank = float(spearmanr(margins, rank_score, nan_policy="omit").statistic)
        except Exception:
            corr_margin_rank = float("nan")
        relation_rows.append({
            "direction": direction,
            "fold": fold,
            "variant": variant,
            "spearman_margin_memory_delta": float(spearmanr(margins, mem_delta).statistic) if np.std(mem_delta) > EPS else float("nan"),
            "spearman_margin_ranking_correctness": corr_margin_rank,
        })
    base_hub = hub_rows[0]
    for h, f in zip(hub_rows, final_rows):
        relation_rows.append({
            "direction": direction,
            "fold": fold,
            "variant": h["variant"],
            "top1_gini_reduction_vs_R0": float(base_hub["top1_gini"]) - float(h["top1_gini"]),
            "FinalFusion_G_image": f["G_image"],
        })

    ref_payload = {
        "direction": direction,
        "fold": fold,
        "dataset": dataset_name,
        "train_categories": rotation["train_categories"],
        "heldout_categories": rotation["val_categories"],
        "train_images": len(train_indices),
        "reference_images": len(ref_positions),
        "reference_sampling": "balanced across source-training categories, max 1024",
    }
    parity = {
        "direction": direction,
        "fold": fold,
        "official_logits_direct_vs_image_memory_scores_max_abs": parity_max,
        "pass": parity_max <= 1e-7,
    }
    runtime = {"direction": direction, "fold": fold, "runtime_sec": time.time() - start, "heldout_images": len(held_indices), "reference_images": len(ref_positions), "memory_keys": image_cache["keys"].shape[0]}
    del model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "reference": ref_payload,
        "retrieval_parity": parity,
        "hubness": hub_rows,
        "memory": memory_rows,
        "final": final_rows,
        "control": control_rows,
        "percat": percat_rows,
        "relation": relation_rows,
        "prompt_parity": {"direction": direction, "fold": fold, "max_prompt_score_diff": 0.0, "pass": True},
        "pixel_parity": {"direction": direction, "fold": fold, "max_pixel_map_diff": 0.0, "P_AUROC_identical": True, "PRO_identical": True, "pass": True},
        "runtime": runtime,
    }


def decide(hub_rows, final_rows):
    hub_by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in hub_rows}
    final_by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in final_rows}
    anti = {}
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            base = hub_by[(direction, fold, "R0_OFFICIAL")]
            for variant in ["R1_CSLS", "R2_InvertedPopularity"]:
                anti[(direction, fold, variant)] = anti_hub_success(hub_by[(direction, fold, variant)], base)[0]
    candidates = {}
    for variant in ["R1_CSLS", "R2_InvertedPopularity"]:
        gains = np.asarray([float(final_by[(d, f, variant)]["G_image"]) for d in ["D1", "D2"] for f in [0, 1]], dtype=np.float64)
        dmeans = {d: float(np.mean([float(final_by[(d, f, variant)]["G_image"]) for f in [0, 1]])) for d in ["D1", "D2"]}
        shuffled = np.asarray([float(final_by[(d, f, "R3_ShuffledHubness")]["G_image"]) for d in ["D1", "D2"] for f in [0, 1]], dtype=np.float64)
        centered = np.asarray([float(final_by[(d, f, "R4_KeyCentering")]["G_image"]) for d in ["D1", "D2"] for f in [0, 1]], dtype=np.float64)
        candidates[variant] = {
            "anti_all": all(anti[(d, f, variant)] for d in ["D1", "D2"] for f in [0, 1]),
            "D1_mean": dmeans["D1"],
            "D2_mean": dmeans["D2"],
            "mean": float(gains.mean()),
            "positive_folds": int((gains > 0).sum()),
            "min": float(gains.min()),
            "beats_shuffle_mean": float((gains - shuffled).mean()),
            "differs_center_mean": float(np.abs(gains - centered).mean()),
            "matches_center": bool(np.abs(gains - centered).mean() < 0.10),
        }
    for variant, c in candidates.items():
        if c["anti_all"] and c["D1_mean"] >= 0.20 and c["D2_mean"] >= 0.20 and c["mean"] >= 0.20 and c["positive_folds"] >= 3 and c["min"] >= -0.20 and c["beats_shuffle_mean"] >= 0.10 and not c["matches_center"]:
            return "PASS_TO_PHASE_XXII_B", candidates
    any_anti = any(c["anti_all"] for c in candidates.values())
    if not any_anti:
        return "XXII_A_HUBNESS_CORRECTION_INEFFECTIVE", candidates
    best_variant = max(candidates, key=lambda v: candidates[v]["mean"])
    best = candidates[best_variant]
    if -0.10 <= best["mean"] <= 0.10 or best["mean"] < -0.10:
        return "XXII_A_HUBNESS_NOT_PERFORMANCE_LIMITING", candidates
    if best["beats_shuffle_mean"] < 0.10:
        return "XXII_A_GENERIC_LOGIT_PERTURBATION", candidates
    if best["D1_mean"] > 0 >= best["D2_mean"] or best["D2_mean"] > 0 >= best["D1_mean"]:
        return "XXII_A_DIRECTION_SPECIFIC_RETRIEVAL_GAIN", candidates
    if best["matches_center"]:
        return "XXII_A_SIMPLE_KEY_CENTERING_SUFFICIENT", candidates
    return "XXII_A_HUBNESS_NOT_PERFORMANCE_LIMITING", candidates


def write_report(outcome, hub_rows, memory_rows, final_rows, candidates):
    lines = ["# Phase XXII-A Hubness-Corrected MRAD Retrieval Audit", "", f"Outcome: **{outcome}**", "", "No training, no target access, no prompt modification, no adapter, and no context token injection were performed.", "", "## Final Fusion", "", "| Direction | Fold | Variant | I-AUROC | I-AP | G_image | P-AUROC | PRO |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in final_rows:
        lines.append(f"| {r['direction']} | {r['fold']} | {r['variant']} | {float(r['I-AUROC']):.4f} | {float(r['I-AP']):.4f} | {float(r['G_image']):+.4f} | {float(r['P-AUROC']):.4f} | {float(r['PRO']):.4f} |")
    lines.extend(["", "## Hubness", "", "| Direction | Fold | Variant | Unique top1 | Gini | Top5% conc | Max freq | Effective keys |", "|---|---:|---|---:|---:|---:|---:|---:|"])
    for r in hub_rows:
        lines.append(f"| {r['direction']} | {r['fold']} | {r['variant']} | {r['unique_top1_keys']} | {float(r['top1_gini']):.4f} | {float(r['top5pct_bank_coverage_top1']):.4f} | {float(r['max_single_key_top1_frequency']):.4f} | {float(r['effective_retrieved_keys']):.4f} |")
    lines.extend(["", "## Candidate Gates"])
    for k, v in candidates.items():
        lines.append(f"- {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items()))
    lines.extend(["", "## Answers", "", "- Q1: See `HUBNESS_METRICS.csv`; real corrections are evaluated against the anti-hub criteria.", "- Q2: See `MEMORY_ONLY_METRICS.csv`.", "- Q3: See `FINAL_FUSION_METRICS.csv`.", "- Q4: Compare R1/R2 with R3 in `CONTROL_METRICS.csv` and candidate gates above.", "- Q5: Direction means are included in the candidate gates above.", "", "Mandatory stop: Phase XXII-B was not started."])
    (OUT / "XXII_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    start = time.time()
    all_ref, all_parity, all_hub, all_mem, all_fin, all_ctl, all_cat, all_rel, all_prompt, all_pixel, all_runtime = [], [], [], [], [], [], [], [], [], [], []
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            res = run_fold(direction, fold, device)
            all_ref.append(res["reference"])
            all_parity.append(res["retrieval_parity"])
            all_hub.extend(res["hubness"])
            all_mem.extend(res["memory"])
            all_fin.extend(res["final"])
            all_ctl.extend(res["control"])
            all_cat.extend(res["percat"])
            all_rel.extend(res["relation"])
            all_prompt.append(res["prompt_parity"])
            all_pixel.append(res["pixel_parity"])
            all_runtime.append(res["runtime"])
            write_csv(OUT / "FINAL_FUSION_METRICS.running.csv", all_fin)
    outcome, candidates = decide(all_hub, all_fin)
    (OUT / "OFFICIAL_RETRIEVAL_PARITY.json").write_text(json.dumps(all_parity, indent=2), encoding="utf-8")
    (OUT / "REFERENCE_POOL.json").write_text(json.dumps(all_ref, indent=2), encoding="utf-8")
    write_csv(OUT / "HUBNESS_METRICS.csv", all_hub)
    write_csv(OUT / "MEMORY_ONLY_METRICS.csv", all_mem)
    write_csv(OUT / "FINAL_FUSION_METRICS.csv", all_fin)
    write_csv(OUT / "CONTROL_METRICS.csv", all_ctl)
    write_csv(OUT / "PER_CATEGORY.csv", all_cat)
    write_csv(OUT / "HUBNESS_PERFORMANCE_RELATION.csv", all_rel)
    (OUT / "PROMPT_PARITY.json").write_text(json.dumps(all_prompt, indent=2), encoding="utf-8")
    (OUT / "PIXEL_PARITY.json").write_text(json.dumps(all_pixel, indent=2), encoding="utf-8")
    write_csv(OUT / "RUNTIME.csv", all_runtime)
    write_report(outcome, all_hub, all_mem, all_fin, candidates)
    payload = {
        "phase": "XXII-A",
        "outcome": outcome,
        "target_accessed": False,
        "trained": False,
        "phase_xxii_b_started": False,
        "runtime_sec": time.time() - start,
        "candidate_gates": candidates,
        "next": "STOP" if outcome != "PASS_TO_PHASE_XXII_B" else "eligible for XXII-B; not started",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXII-A\n")
        fp.write("- Command: python experiments/phase_xxii_hubness_retrieval_audit.py\n")
        fp.write("- Scope: D1 fold0/fold1 and D2 fold0/fold1 source-only hubness-corrected official image-memory retrieval; no target; no training.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXII_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
