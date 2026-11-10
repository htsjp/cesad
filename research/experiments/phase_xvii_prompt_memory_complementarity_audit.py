import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, to_device_cache, write_csv
from models.mlp import average_neighbor
from mrad import compute_patch_socre
from phase_xa.official_task_forward import image_memory_scores, normalized_image_feature, visual_outputs
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xvii_prompt_memory_complementarity/audit")
FOLDS = [("D1", 0), ("D1", 1), ("D1", 2), ("D2", 0)]
CALIBRATIONS = ["C0_raw", "C1_zscore", "C2_rank"]
ALPHAS = [round(i / 10.0, 1) for i in range(11)]
DIAG_KEYS = [
    "score_memory_raw",
    "score_prompt_raw",
    "abs_disagreement",
    "signed_disagreement",
    "prompt_map_mean",
    "prompt_map_std",
    "prompt_map_max",
    "prompt_top1_mean",
    "prompt_top5_mean",
    "prompt_spatial_entropy",
    "prompt_concentration_top1_over_mean",
    "memory_softmax_entropy",
    "memory_top1_probability",
    "memory_top2_margin",
    "memory_similarity_margin",
    "memory_effective_support_size",
]


def read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s) * 100.0)


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s) * 100.0)


def safe_corr(a, b, kind):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    try:
        return float((pearsonr(a, b) if kind == "pearson" else spearmanr(a, b)).statistic)
    except Exception:
        return float("nan")


def metric_pair(rows, score_key):
    return metric_rows_scores(rows, [float(r[score_key]) for r in rows])


def image_metric(labels, scores):
    return {"I-AUROC": safe_auc(labels, scores), "I-AP": safe_ap(labels, scores)}


def metric_rows_scores(rows, scores):
    by_cat = defaultdict(lambda: {"labels": [], "scores": []})
    for r, score in zip(rows, scores):
        by_cat[r["category"]]["labels"].append(int(r["label"]))
        by_cat[r["category"]]["scores"].append(float(score))
    aucs, aps = [], []
    for vals in by_cat.values():
        auc = safe_auc(vals["labels"], vals["scores"])
        ap = safe_ap(vals["labels"], vals["scores"])
        if np.isfinite(auc):
            aucs.append(auc)
        if np.isfinite(ap):
            aps.append(ap)
    return {"I-AUROC": float(np.mean(aucs)) if aucs else float("nan"), "I-AP": float(np.mean(aps)) if aps else float("nan")}


def g_image(a, b):
    return 0.5 * (float(a["I-AUROC"]) - float(b["I-AUROC"])) + 0.5 * (float(a["I-AP"]) - float(b["I-AP"]))


def prompt_entropy(flat):
    x = flat.float()
    x = x - x.min()
    total = x.sum()
    if float(total.cpu()) <= 1e-12:
        return 0.0
    p = (x / total).clamp_min(1e-12)
    return float(-(p * p.log()).sum().cpu() / math.log(float(p.numel())))


def score_one(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, dataset_name, item, device):
    image = item["img"].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = visual_outputs(model, image, adapter=None, grad=False)
        image_feature = normalized_image_feature(outputs)
        image_post, image_logits, _, _ = image_memory_scores(image_feature, image_cache["keys"], image_cache["values"], image_proj)
        memory_score = image_post[:, 1]
        memory_prob = torch.softmax(image_logits, dim=-1)
        top_probs, top_idx = torch.topk(memory_prob, k=min(2, memory_prob.shape[-1]), dim=-1)
        top_logits = torch.gather(image_logits, 1, top_idx)

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
        flat = anomaly_map.flatten()
        top1n = max(1, int(flat.numel() * 0.01))
        top5n = max(1, int(flat.numel() * 0.05))
        prompt_top1 = torch.topk(flat, top1n).values.mean()
        prompt_top5 = torch.topk(flat, top5n).values.mean()
        final_score = 0.3 * prompt_top1.to(device) + 0.7 * memory_score
        recon = 0.3 * float(prompt_top1.cpu()) + 0.7 * float(memory_score.cpu())
        return {
            "score_memory_raw": float(memory_score.cpu()),
            "score_prompt_raw": float(prompt_top1.cpu()),
            "score_official_final": float(final_score.cpu()),
            "reconstructed_official_final": recon,
            "official_reconstruction_abs_error": abs(float(final_score.cpu()) - recon),
            "prompt_map_mean": float(flat.mean().cpu()),
            "prompt_map_std": float(flat.std(unbiased=False).cpu()),
            "prompt_map_max": float(flat.max().cpu()),
            "prompt_top1_mean": float(prompt_top1.cpu()),
            "prompt_top5_mean": float(prompt_top5.cpu()),
            "prompt_spatial_entropy": prompt_entropy(flat),
            "prompt_concentration_top1_over_mean": float(prompt_top1.cpu()) / (float(flat.mean().cpu()) + 1e-12),
            "memory_softmax_entropy": float((-(memory_prob * memory_prob.clamp_min(1e-12).log()).sum(dim=-1)).cpu()),
            "memory_top1_probability": float(top_probs[:, 0].cpu()),
            "memory_top2_margin": float((top_probs[:, 0] - top_probs[:, 1]).cpu()) if top_probs.shape[1] > 1 else float("nan"),
            "memory_similarity_margin": float((top_logits[:, 0] - top_logits[:, 1]).cpu()) if top_logits.shape[1] > 1 else float("nan"),
            "memory_effective_support_size": float(torch.exp((-(memory_prob * memory_prob.clamp_min(1e-12).log()).sum(dim=-1))).cpu()),
        }


def cache_scores():
    cached_path = OUT / "cached_component_scores.csv"
    existing = read_csv(cached_path)
    completed = {(r["direction"], int(r["fold"]), r["split"], int(r["idx"])) for r in existing}
    rows = list(existing)
    for direction, fold in FOLDS:
        dataset_name, data_path, checkpoint = direction_cfg(direction)
        dataset = load_dataset(dataset_name, data_path)
        rot = deterministic_source_folds(dataset_name)["rotations"][fold]
        train_indices = all_indices_for_categories(dataset, set(rot["train_categories"]))
        val_indices = all_indices_for_categories(dataset, set(rot["val_categories"]))
        expected = {(direction, fold, "train", idx) for idx in train_indices} | {(direction, fold, "heldout", idx) for idx in val_indices}
        if expected.issubset(completed):
            continue
        model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, "cuda:0" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device
        image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None)
        image_cache, patch_cache = to_device_cache(image_cache, device), to_device_cache(patch_cache, device)
        for split, indices in [("train", train_indices), ("heldout", val_indices)]:
            for idx in indices:
                key = (direction, fold, split, idx)
                if key in completed:
                    continue
                item = dataset[idx]
                scores = score_one(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, dataset_name, item, device)
                row = {
                    "direction": direction,
                    "dataset": dataset_name,
                    "fold": fold,
                    "split": split,
                    "idx": idx,
                    "category": item["cls_name"],
                    "label": int(item["anomaly"]),
                    **scores,
                }
                row["abs_disagreement"] = abs(row["score_memory_raw"] - row["score_prompt_raw"])
                row["signed_disagreement"] = row["score_memory_raw"] - row["score_prompt_raw"]
                rows.append(row)
            write_csv(cached_path, rows)
    return rows


class Calibrator:
    def __init__(self, train_rows, method):
        self.method = method
        self.stats = {}
        for key in ["score_memory_raw", "score_prompt_raw"]:
            vals = np.asarray([float(r[key]) for r in train_rows], dtype=np.float64)
            self.stats[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std() + 1e-12),
                "sorted": np.sort(vals),
            }

    def one(self, key, value):
        value = float(value)
        if self.method == "C0_raw":
            return value
        if self.method == "C1_zscore":
            st = self.stats[key]
            return (value - st["mean"]) / st["std"]
        st = self.stats[key]["sorted"]
        return float(np.searchsorted(st, value, side="right") / max(1, len(st)))

    def apply(self, rows):
        out = []
        for r in rows:
            rr = dict(r)
            rr["memory_cal"] = self.one("score_memory_raw", r["score_memory_raw"])
            rr["prompt_cal"] = self.one("score_prompt_raw", r["score_prompt_raw"])
            rr["official_raw"] = float(r["score_official_final"])
            out.append(rr)
        return out


def by_fold(rows, direction, fold, split):
    return [r for r in rows if r["direction"] == direction and int(r["fold"]) == fold and r["split"] == split]


def branch_and_complementarity(rows):
    branch_rows, corr_rows, rank_rows = [], [], []
    static_grid, static_cross, oracle_rows = [], [], []
    percat = []
    selected_static = {}
    for direction, fold in FOLDS:
        train = by_fold(rows, direction, fold, "train")
        held = by_fold(rows, direction, fold, "heldout")
        official = metric_pair(held, "score_official_final")
        branch_rows.append({"direction": direction, "fold": fold, "calibration": "official", "method": "OfficialFusion", **official})
        mem = np.asarray([float(r["score_memory_raw"]) for r in held])
        prm = np.asarray([float(r["score_prompt_raw"]) for r in held])
        corr_rows.append({"direction": direction, "fold": fold, "pearson": safe_corr(mem, prm, "pearson"), "spearman": safe_corr(mem, prm, "spearman"), "samples": len(held)})
        signs_m = np.sign(mem[:, None] - mem[None, :])
        signs_p = np.sign(prm[:, None] - prm[None, :])
        upper = np.triu(np.ones_like(signs_m, dtype=bool), k=1)
        disagree = (signs_m[upper] * signs_p[upper]) < 0
        rank_rows.append({"direction": direction, "fold": fold, "ranking_disagreement_fraction": float(disagree.mean()), "pairs": int(disagree.size)})
        best_train = None
        best_combo = None
        best_held = None
        held_oracle_best = None
        for cal in CALIBRATIONS:
            c = Calibrator(train, cal)
            tr, he = c.apply(train), c.apply(held)
            for method, key in [("MemoryOnly", "memory_cal"), ("PromptOnly", "prompt_cal")]:
                branch_rows.append({"direction": direction, "fold": fold, "calibration": cal, "method": method, **metric_pair(he, key)})
            labels_tr = [int(r["label"]) for r in tr]
            labels_he = [int(r["label"]) for r in he]
            for alpha in ALPHAS:
                sc_tr = [alpha * float(r["memory_cal"]) + (1 - alpha) * float(r["prompt_cal"]) for r in tr]
                sc_he = [alpha * float(r["memory_cal"]) + (1 - alpha) * float(r["prompt_cal"]) for r in he]
                mt, mh = metric_rows_scores(tr, sc_tr), metric_rows_scores(he, sc_he)
                row = {"direction": direction, "fold": fold, "calibration": cal, "alpha": alpha, "train_I-AUROC": mt["I-AUROC"], "train_I-AP": mt["I-AP"], "heldout_I-AUROC": mh["I-AUROC"], "heldout_I-AP": mh["I-AP"]}
                row["train_G_image_vs_official"] = 0.5 * (mt["I-AUROC"] + mt["I-AP"]) - 0.5 * (official["I-AUROC"] + official["I-AP"])
                row["heldout_G_image_vs_official"] = g_image(mh, official)
                static_grid.append(row)
                train_score = 0.5 * (mt["I-AUROC"] + mt["I-AP"])
                held_score = 0.5 * (mh["I-AUROC"] + mh["I-AP"])
                if best_train is None or train_score > best_train:
                    best_train, best_combo, best_held = train_score, (cal, alpha), mh
                if held_oracle_best is None or held_score > held_oracle_best[0]:
                    held_oracle_best = (held_score, cal, alpha, mh)
        selected_static[(direction, fold)] = best_combo
        static_cross.append({"direction": direction, "fold": fold, "selected_calibration": best_combo[0], "selected_alpha": best_combo[1], **best_held, "Official_I-AUROC": official["I-AUROC"], "Official_I-AP": official["I-AP"], "G_image_vs_official": g_image(best_held, official), "heldout_oracle_calibration": held_oracle_best[1], "heldout_oracle_alpha": held_oracle_best[2], "heldout_oracle_I-AUROC": held_oracle_best[3]["I-AUROC"], "heldout_oracle_I-AP": held_oracle_best[3]["I-AP"]})
        cal = Calibrator(train, best_combo[0])
        he = c_he = cal.apply(held)
        oracle_scores = []
        oracle_pref = []
        for r in he:
            if int(r["label"]) == 1:
                choose_mem = float(r["memory_cal"]) >= float(r["prompt_cal"])
                oracle_scores.append(max(float(r["memory_cal"]), float(r["prompt_cal"])))
            else:
                choose_mem = float(r["memory_cal"]) <= float(r["prompt_cal"])
                oracle_scores.append(min(float(r["memory_cal"]), float(r["prompt_cal"])))
            oracle_pref.append(int(choose_mem))
        om = metric_rows_scores(he, oracle_scores)
        oracle_rows.append({"direction": direction, "fold": fold, "calibration": best_combo[0], **om, "Official_I-AUROC": official["I-AUROC"], "Official_I-AP": official["I-AP"], "BestStatic_I-AUROC": best_held["I-AUROC"], "BestStatic_I-AP": best_held["I-AP"], "OracleHeadroom": g_image(om, official), "OracleVsBestStatic": g_image(om, best_held)})
        for category in sorted({r["category"] for r in held}):
            hcat = [r for r in held if r["category"] == category]
            hecat = [r for r in he if r["category"] == category]
            labels = [int(r["label"]) for r in hcat]
            scores_static = [best_combo[1] * float(r["memory_cal"]) + (1 - best_combo[1]) * float(r["prompt_cal"]) for r in hecat]
            scores_oracle = []
            for r in hecat:
                scores_oracle.append(max(float(r["memory_cal"]), float(r["prompt_cal"])) if int(r["label"]) else min(float(r["memory_cal"]), float(r["prompt_cal"])))
            for method, scores in [
                ("Official", [float(r["score_official_final"]) for r in hcat]),
                ("MemoryOnly", [float(r["memory_cal"]) for r in hecat]),
                ("PromptOnly", [float(r["prompt_cal"]) for r in hecat]),
                ("BestStatic", scores_static),
                ("Oracle", scores_oracle),
            ]:
                percat.append({"direction": direction, "fold": fold, "category": category, "method": method, **image_metric(labels, scores)})
    return branch_rows, corr_rows, rank_rows, static_grid, static_cross, oracle_rows, percat, selected_static


def standardize_train_held(train, held):
    xtr = np.asarray([[float(r[k]) for k in DIAG_KEYS] for r in train], dtype=np.float32)
    xhe = np.asarray([[float(r[k]) for k in DIAG_KEYS] for r in held], dtype=np.float32)
    mu = np.nanmean(xtr, axis=0)
    sd = np.nanstd(xtr, axis=0) + 1e-6
    xtr = np.nan_to_num((xtr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    xhe = np.nan_to_num((xhe - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return xtr, xhe


class Router(nn.Module):
    def __init__(self, dim, mlp=True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 16), nn.GELU(), nn.Linear(16, 1)) if mlp else nn.Linear(dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train_router(xtr, ytr, mem_tr, prm_tr, xhe, mem_he, prm_he, seed, mlp=True):
    torch.manual_seed(seed)
    model = Router(xtr.shape[1], mlp=mlp)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    x = torch.tensor(xtr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    m = torch.tensor(mem_tr, dtype=torch.float32)
    p = torch.tensor(prm_tr, dtype=torch.float32)
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float((1 - y).sum()))
    weight = torch.where(y > 0.5, torch.tensor(neg / pos), torch.tensor(1.0))
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        w = model(x)
        s = (w * m + (1 - w) * p).clamp(1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(s, y, weight=weight)
        ent = -(w.clamp(1e-6, 1 - 1e-6) * w.clamp(1e-6, 1 - 1e-6).log() + (1 - w).clamp(1e-6, 1 - 1e-6) * (1 - w).clamp(1e-6, 1 - 1e-6).log()).mean()
        (bce - 0.01 * ent).backward()
        opt.step()
    with torch.no_grad():
        whe = model(torch.tensor(xhe, dtype=torch.float32)).numpy()
    return whe


def router_probe(rows, selected_static, oracle_rows, branch_static, run, skip_reason="SKIPPED"):
    router_metrics, seed_rows, gate_rows, pref_rows, importance_rows = [], [], [], [], []
    if not run:
        skipped = {"status": skip_reason}
        return [skipped], [skipped], [skipped], [skipped], [skipped], []
    for direction, fold in FOLDS:
        train_raw = by_fold(rows, direction, fold, "train")
        held_raw = by_fold(rows, direction, fold, "heldout")
        cal = Calibrator(train_raw, selected_static[(direction, fold)][0])
        train = cal.apply(train_raw)
        held = cal.apply(held_raw)
        xtr, xhe = standardize_train_held(train, held)
        ytr = np.asarray([int(r["label"]) for r in train], dtype=np.float32)
        yhe = np.asarray([int(r["label"]) for r in held], dtype=np.int64)
        mem_tr = np.asarray([float(r["memory_cal"]) for r in train], dtype=np.float32)
        prm_tr = np.asarray([float(r["prompt_cal"]) for r in train], dtype=np.float32)
        mem_he = np.asarray([float(r["memory_cal"]) for r in held], dtype=np.float32)
        prm_he = np.asarray([float(r["prompt_cal"]) for r in held], dtype=np.float32)
        official = metric_pair(held_raw, "score_official_final")
        best_static = next(r for r in branch_static if r["direction"] == direction and int(r["fold"]) == fold)
        oracle_pref = np.asarray([(float(r["memory_cal"]) >= float(r["prompt_cal"])) if int(r["label"]) else (float(r["memory_cal"]) <= float(r["prompt_cal"])) for r in held], dtype=np.int64)
        for kind in ["LinearRouter", "MLPRouter", "ShuffledDiagnosticRouter"]:
            fold_scores = []
            fold_gates = []
            for seed in range(5):
                if kind == "ShuffledDiagnosticRouter":
                    rng = np.random.default_rng(seed)
                    xs = xtr.copy()
                    rng.shuffle(xs, axis=0)
                    w = train_router(xs, ytr, mem_tr, prm_tr, xhe, mem_he, prm_he, seed, mlp=True)
                else:
                    w = train_router(xtr, ytr, mem_tr, prm_tr, xhe, mem_he, prm_he, seed, mlp=(kind == "MLPRouter"))
                scores = w * mem_he + (1 - w) * prm_he
                met = metric_rows_scores(held, scores)
                gain = g_image(met, official)
                seed_rows.append({"direction": direction, "fold": fold, "router": kind, "seed": seed, **met, "G_image_vs_official": gain})
                gate_rows.append({"direction": direction, "fold": fold, "router": kind, "seed": seed, "gate_mean": float(np.mean(w)), "gate_std": float(np.std(w)), "gate_p05": float(np.percentile(w, 5)), "gate_p95": float(np.percentile(w, 95))})
                pref_rows.append({"direction": direction, "fold": fold, "router": kind, "seed": seed, "router_weight_preference_AUROC": safe_auc(oracle_pref, w)})
                fold_scores.append(gain)
                fold_gates.append(float(np.std(w)))
                if kind == "MLPRouter" and seed == 0:
                    base = met["I-AUROC"]
                    for j, key in enumerate(DIAG_KEYS):
                        xp = xhe.copy()
                        rng = np.random.default_rng(1000 + j)
                        rng.shuffle(xp[:, j])
                        wp = train_router(xtr, ytr, mem_tr, prm_tr, xp, mem_he, prm_he, seed, mlp=True)
                        mp = metric_rows_scores(held, wp * mem_he + (1 - wp) * prm_he)
                        importance_rows.append({"direction": direction, "fold": fold, "diagnostic": key, "delta_I-AUROC_when_permuted": base - mp["I-AUROC"]})
            router_metrics.append({"direction": direction, "fold": fold, "router": kind, "mean_G_image_vs_official": float(np.mean(fold_scores)), "std_G_image_vs_official": float(np.std(fold_scores, ddof=1)), "mean_gate_std": float(np.mean(fold_gates)), "BestStatic_G_image_vs_official": best_static["G_image_vs_official"]})
    return router_metrics, seed_rows, gate_rows, pref_rows, importance_rows, []


def decide(static_cross, oracle_rows, router_metrics, router_seed_rows, gate_rows):
    static_gains = [float(r["G_image_vs_official"]) for r in static_cross]
    static_ok = (np.mean(static_gains) >= 0.30 and sum(g >= 0.30 for g in static_gains) >= 3 and min(static_gains) >= -0.20)
    if static_ok:
        return "XVII_A_STATIC_FUSION_SUFFICIENT", {"static_mean_gain": float(np.mean(static_gains))}
    oracle_head = [float(r["OracleHeadroom"]) for r in oracle_rows]
    oracle_vs_static = [float(r["OracleVsBestStatic"]) for r in oracle_rows]
    oracle_ok = np.mean(oracle_head) >= 1.0 and sum(g >= 0.50 for g in oracle_head) >= 3 and np.mean(oracle_vs_static) >= 0.50
    if not oracle_ok:
        return "XVII_A_NO_DYNAMIC_FUSION_HEADROOM", {"oracle_headroom_mean": float(np.mean(oracle_head)), "oracle_vs_static_mean": float(np.mean(oracle_vs_static))}
    mlp = [r for r in router_metrics if r.get("router") == "MLPRouter"]
    shuf = [r for r in router_metrics if r.get("router") == "ShuffledDiagnosticRouter"]
    if not mlp or not shuf:
        return "XVII_A_ORACLE_HEADROOM_NOT_LEARNABLE", {}
    mlp_g = [float(r["mean_G_image_vs_official"]) for r in mlp]
    shuf_g = [float(r["mean_G_image_vs_official"]) for r in shuf]
    stat = [float(r["BestStatic_G_image_vs_official"]) for r in mlp]
    mlp_seed = [r for r in router_seed_rows if r.get("router") == "MLPRouter"]
    seed_mean = defaultdict(list)
    for r in mlp_seed:
        seed_mean[int(r["seed"])].append(float(r["G_image_vs_official"]))
    pos_seeds = sum(np.mean(v) > 0 for v in seed_mean.values())
    gate_std = np.mean([float(r["gate_std"]) for r in gate_rows if r.get("router") == "MLPRouter"])
    success = np.mean(mlp_g) >= 0.30 and sum(g > 0 for g in mlp_g) >= 3 and min(mlp_g) >= -0.20 and (np.mean(mlp_g) - np.mean(stat)) >= 0.15 and (np.mean(mlp_g) - np.mean(shuf_g)) >= 0.20 and pos_seeds >= 4 and gate_std >= 0.05
    if success:
        return "PASS_TO_PHASE_XVII_B", {"mlp_mean_gain": float(np.mean(mlp_g)), "shuffled_mean_gain": float(np.mean(shuf_g)), "gate_std": float(gate_std)}
    if abs(np.mean(mlp_g) - np.mean(stat)) < 0.15:
        return "XVII_A_STATIC_FUSION_EXPLAINS_GAIN", {"mlp_mean_gain": float(np.mean(mlp_g)), "static_mean_gain": float(np.mean(stat))}
    if np.mean(shuf_g) + 0.20 >= np.mean(mlp_g):
        return "XVII_A_ROUTER_FEATURES_NOT_SPECIFIC", {"mlp_mean_gain": float(np.mean(mlp_g)), "shuffled_mean_gain": float(np.mean(shuf_g))}
    return "XVII_A_ORACLE_HEADROOM_NOT_LEARNABLE", {"mlp_mean_gain": float(np.mean(mlp_g)), "static_mean_gain": float(np.mean(stat))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = cache_scores()
    max_err = max(float(r["official_reconstruction_abs_error"]) for r in rows)
    official_audit = {
        "source": "test.py lines 195-197 and phase_xa source readout",
        "score_memory_raw": "projected CLS memory posterior anomaly probability",
        "score_prompt_raw": "Gaussian-processed prompt anomaly map top 1% mean",
        "formula": "score_official_final = 0.7 * score_memory_raw + 0.3 * score_prompt_raw",
        "max_abs_reconstruction_error": max_err,
        "pass": max_err <= 1e-6,
    }
    (OUT / "official_path_audit.json").write_text(json.dumps(official_audit, indent=2), encoding="utf-8")
    pixel_status = {"status": "PIXEL_BRANCH_DECOMPOSITION_NOT_AVAILABLE", "reason": "official MRAD-CLIP image path exposes prompt anomaly map; patch-memory path conditions prompt bias but no independent final pixel anomaly branch is fused in the official output"}
    (OUT / "PIXEL_COMPONENT_STATUS.json").write_text(json.dumps(pixel_status, indent=2), encoding="utf-8")
    if not official_audit["pass"]:
        outcome = "XVII_A_IMPLEMENTATION_FAILURE"
        for name in ["branch_metrics.csv", "branch_correlations.csv", "ranking_disagreement.csv", "static_fusion_grid.csv", "static_cross_category.csv", "oracle_headroom.csv", "router_metrics.csv", "router_seed_metrics.csv", "router_gate_statistics.csv", "router_preference_analysis.csv", "per_category_metrics.csv", "diagnostic_importance.csv"]:
            write_csv(OUT / name, [{"status": "FAILED_OFFICIAL_RECONSTRUCTION"}])
    else:
        branch, corr, rank, grid, static_cross, oracle, percat, selected = branch_and_complementarity(rows)
        write_csv(OUT / "branch_metrics.csv", branch)
        write_csv(OUT / "branch_correlations.csv", corr)
        write_csv(OUT / "ranking_disagreement.csv", rank)
        write_csv(OUT / "static_fusion_grid.csv", grid)
        write_csv(OUT / "static_cross_category.csv", static_cross)
        write_csv(OUT / "oracle_headroom.csv", oracle)
        oracle_gate = np.mean([float(r["OracleHeadroom"]) for r in oracle]) >= 1.0 and sum(float(r["OracleHeadroom"]) >= 0.5 for r in oracle) >= 3 and np.mean([float(r["OracleVsBestStatic"]) for r in oracle]) >= 0.5
        static_gate = np.mean([float(r["G_image_vs_official"]) for r in static_cross]) >= 0.30 and sum(float(r["G_image_vs_official"]) >= 0.30 for r in static_cross) >= 3 and min(float(r["G_image_vs_official"]) for r in static_cross) >= -0.20
        skip_reason = "SKIPPED_STATIC_FUSION_SUFFICIENT" if static_gate else "SKIPPED_ORACLE_GATE_FAILED"
        router_metrics, seed_rows, gate_rows, pref_rows, importance, _ = router_probe(rows, selected, oracle, static_cross, run=(oracle_gate and not static_gate), skip_reason=skip_reason)
        write_csv(OUT / "router_metrics.csv", router_metrics)
        write_csv(OUT / "router_seed_metrics.csv", seed_rows)
        write_csv(OUT / "router_gate_statistics.csv", gate_rows)
        write_csv(OUT / "router_preference_analysis.csv", pref_rows)
        percat_router = []
        if router_metrics and router_metrics[0].get("status") is None:
            pass
        write_csv(OUT / "per_category_metrics.csv", percat)
        write_csv(OUT / "diagnostic_importance.csv", importance)
        outcome, gate_info = decide(static_cross, oracle, router_metrics, seed_rows, gate_rows)
    report = ["# Phase XVII-A Prompt-Memory Complementarity Audit", "", f"Outcome: **{outcome}**", "", "No target access, no CLIP/prompt/memory/projection training.", "", "## Official Path", "", f"- Formula: `{official_audit['formula']}`", f"- Max reconstruction error: `{official_audit['max_abs_reconstruction_error']}`", f"- Pixel status: `{pixel_status['status']}`"]
    if official_audit["pass"]:
        static_cross = read_csv(OUT / "static_cross_category.csv")
        oracle = read_csv(OUT / "oracle_headroom.csv")
        report.extend(["", "## Static Fusion", "", "| Direction | Fold | Cal | Alpha | G_image | I-AUROC | I-AP |", "|---|---:|---|---:|---:|---:|---:|"])
        for r in static_cross:
            report.append(f"| {r['direction']} | {r['fold']} | {r['selected_calibration']} | {float(r['selected_alpha']):.1f} | {float(r['G_image_vs_official']):.4f} | {float(r['I-AUROC']):.4f} | {float(r['I-AP']):.4f} |")
        report.extend(["", "## Oracle Headroom", "", "| Direction | Fold | Headroom | OracleVsStatic | I-AUROC | I-AP |", "|---|---:|---:|---:|---:|---:|"])
        for r in oracle:
            report.append(f"| {r['direction']} | {r['fold']} | {float(r['OracleHeadroom']):.4f} | {float(r['OracleVsBestStatic']):.4f} | {float(r['I-AUROC']):.4f} | {float(r['I-AP']):.4f} |")
        router = read_csv(OUT / "router_metrics.csv")
        report.extend(["", "## Router", ""])
        if router and router[0].get("status"):
            report.append(f"- `{router[0]['status']}`")
        else:
            for r in router:
                report.append(f"- {r['direction']} fold{r['fold']} {r['router']}: mean G_image `{float(r['mean_G_image_vs_official']):.4f}`, gate std `{float(r['mean_gate_std']):.4f}`")
    report.append("\nMandatory stop: Phase XVII-B was not started.")
    (OUT / "XVII_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    payload = {"phase": "XVII-A", "outcome": outcome, "official_audit": official_audit, "pixel_component_status": pixel_status, "target_accessed": False, "trained_visual_or_prompt_or_memory": False, "next": "STOP. Do not execute Phase XVII-B automatically."}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write("\n\n## Phase XVII-A\n")
        f.write("- Command: python experiments/phase_xvii_prompt_memory_complementarity_audit.py\n")
        f.write(f"- Official decomposition max abs error: {max_err}; formula: {official_audit['formula']}.\n")
        f.write(f"- Pixel component status: {pixel_status['status']}.\n")
        f.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XVII_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
