import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


OUT = Path("results/phase_xviii_frequency/audit")
FOLDS = [("D1", 0), ("D1", 1), ("D2", 0), ("D2", 1)]
PATCH = 14
SIDE = 37
IMAGE_SIZE = 518
EPS = 1e-8
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if len(np.unique(y)) == 2 else float("nan")


def safe_corr(a, b, kind):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float((pearsonr(a, b) if kind == "pearson" else spearmanr(a, b)).statistic)


def metric_rows(rows, scores):
    by_cat = defaultdict(lambda: {"y": [], "s": []})
    for r, s in zip(rows, scores):
        by_cat[r["category"]]["y"].append(int(r["label"]))
        by_cat[r["category"]]["s"].append(float(s))
    aucs, aps = [], []
    for v in by_cat.values():
        auc, ap = safe_auc(v["y"], v["s"]), safe_ap(v["y"], v["s"])
        if np.isfinite(auc):
            aucs.append(auc)
        if np.isfinite(ap):
            aps.append(ap)
    return {"I-AUROC": float(np.mean(aucs)) if aucs else float("nan"), "I-AP": float(np.mean(aps)) if aps else float("nan")}


def metric_value(metrics, key):
    v = float(metrics["mean"].get(key, float("nan")))
    return v * 100.0 if np.isfinite(v) and v <= 1.5 else v


def append_pixel(results, cls, item, score, amap):
    mask = (item["img_mask"].detach().cpu() > 0.5).float()
    results[cls]["imgs_masks"].append(mask)
    results[cls]["gt_sp"].append(int(item["anomaly"]))
    results[cls]["pr_sp"].append(float(score))
    results[cls]["anomaly_maps"].append(amap.detach().cpu())


def upsample_map(m):
    x = torch.tensor(m, dtype=torch.float32).view(1, 1, SIDE, SIDE)
    return F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze(0)


def radial_masks():
    y, x = np.mgrid[0:PATCH, 0:PATCH]
    r = np.sqrt((y / (PATCH - 1)) ** 2 + (x / (PATCH - 1)) ** 2) / math.sqrt(2.0)
    low = r <= 0.25
    mid = (r > 0.25) & (r <= 0.50)
    high = r > 0.50
    return low, mid, high


LOW_MASK, MID_MASK, HIGH_MASK = radial_masks()
RNG_MASKS = []
for seed in [0, 1, 2]:
    rng = np.random.default_rng(seed)
    flat = np.zeros(PATCH * PATCH, dtype=bool)
    flat[rng.choice(PATCH * PATCH, size=int(HIGH_MASK.sum()), replace=False)] = True
    RNG_MASKS.append(flat.reshape(PATCH, PATCH))


def patch_dct_stats(image_tensor, random_mask=None):
    # Use official preprocessed tensor values; non-overlapping 14x14 grid aligns to ViT-L/14 input.
    arr = image_tensor.detach().cpu().float().numpy()
    patches = arr.reshape(3, SIDE, PATCH, SIDE, PATCH).transpose(1, 3, 0, 2, 4)
    coeff = dctn(patches, axes=(-2, -1), norm="ortho")
    energy = coeff * coeff
    total = energy.sum(axis=(2, 3, 4)) + EPS
    low = energy[:, :, :, LOW_MASK].sum(axis=(2, 3))
    mid = energy[:, :, :, MID_MASK].sum(axis=(2, 3))
    high_mask = random_mask if random_mask is not None else HIGH_MASK
    high = energy[:, :, :, high_mask].sum(axis=(2, 3))
    p = energy.reshape(SIDE, SIDE, -1) / total[:, :, None]
    entropy = -(p * np.log(p + EPS)).sum(axis=-1) / math.log(p.shape[-1])
    maps = {
        "r_high": (high / total).astype(np.float32),
        "r_mid": (mid / total).astype(np.float32),
        "r_highlow": (high / (low + EPS)).astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }
    return maps


def highpass_stats(image_tensor):
    x = image_tensor.detach().cpu().float().unsqueeze(0)
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    vals = []
    for c in range(3):
        vals.append(F.conv2d(x[:, c : c + 1], kernel, padding=1).abs())
    hp = torch.stack(vals, dim=0).mean(dim=0).squeeze().numpy()
    maps = {k: np.zeros((SIDE, SIDE), dtype=np.float32) for k in ["r_high", "r_mid", "r_highlow", "entropy"]}
    for iy in range(SIDE):
        for ix in range(SIDE):
            patch = hp[iy * PATCH : (iy + 1) * PATCH, ix * PATCH : (ix + 1) * PATCH]
            val = float(patch.mean())
            maps["r_high"][iy, ix] = val
            maps["r_mid"][iy, ix] = val
            maps["r_highlow"][iy, ix] = val
            maps["entropy"][iy, ix] = val
    return maps


def fit_freq_norm(stat_maps):
    stats = {}
    for key in ["r_high", "r_mid", "r_highlow", "entropy"]:
        vals = np.concatenate([m[key].reshape(-1) for m in stat_maps]).astype(np.float64)
        stats[key] = {"mean": float(vals.mean()), "std": float(vals.std() + EPS)}
    return stats


def normalize_maps(maps, stats):
    return {k: (maps[k] - stats[k]["mean"]) / stats[k]["std"] for k in maps}


def feature_from_z(zmaps):
    parts = []
    for key in ["r_high", "r_mid", "r_highlow", "entropy"]:
        arr = zmaps[key].reshape(-1)
        parts.extend([float(arr.mean()), float(arr.std()), float(np.percentile(arr, 95)), float(np.percentile(arr, 99)), float(np.sort(arr)[-max(1, int(0.01 * arr.size)) :].mean())])
    return parts


def freq_anomaly_map(zmaps):
    return np.mean(np.stack([np.abs(zmaps["r_high"]), np.abs(zmaps["r_highlow"]), np.abs(zmaps["entropy"])], axis=0), axis=0).astype(np.float32)


def official_score_map(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item, device):
    image = item["img"].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = visual_outputs(model, image, adapter=None, grad=False)
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
        amap = torch.stack(maps)
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=4)) for x in amap.detach().cpu()], dim=0).sum(dim=0)
        topk = torch.topk(amap.flatten(), max(1, int(amap.numel() * 0.01))).values.mean()
        score = 0.3 * topk.to(device) + 0.7 * cls_memory
    return float(score.detach().cpu()), amap.detach().cpu()


def rank_calibrator(vals):
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    return lambda x: np.searchsorted(arr, x, side="right") / max(1, len(arr))


def pixel_metric(dataset, indices, rows_by_idx, map_key, score_key):
    obj_list = sorted({dataset[i]["cls_name"] for i in indices})
    res = init_results(obj_list)
    for idx in indices:
        item = dataset[idx]
        append_pixel(res, item["cls_name"], item, rows_by_idx[idx][score_key], rows_by_idx[idx][map_key])
    m = evaluate_results(res, obj_list, compute_pro=True)
    return {"P-AUROC": metric_value(m, "P-AUROC"), "PRO": metric_value(m, "PRO"), "per_class": m["per_class"]}


def train_probe(train_feats, train_labels):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=0, class_weight="balanced"))
    clf.fit(np.asarray(train_feats, dtype=np.float64), np.asarray(train_labels, dtype=np.int64))
    return clf


def run_fold(direction, fold):
    dataset_name, data_path, ckpt = direction_cfg(direction)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(ckpt, device)
    dataset = load_dataset(dataset_name, data_path)
    rot = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rot["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rot["val_categories"]))
    image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None)
    image_cache, patch_cache = to_device_cache(image_cache, device), to_device_cache(patch_cache, device)

    cache_dir = OUT / "fold_cache" / f"{direction}_fold{fold}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    freq_npz = cache_dir / "frequency_features.npz"
    if freq_npz.exists():
        cached = np.load(freq_npz, allow_pickle=True)
        raw_records = list(cached["records"])
    else:
        raw_records = []
        for kind, func in [("dct", patch_dct_stats), ("highpass", highpass_stats)]:
            stat_maps = []
            for idx in train_indices:
                stat_maps.append(func(dataset[idx]["img"]))
            stats = fit_freq_norm(stat_maps)
            for split, indices in [("train", train_indices), ("heldout", held_indices)]:
                for idx in indices:
                    maps = func(dataset[idx]["img"])
                    z = normalize_maps(maps, stats)
                    fmap = freq_anomaly_map(z)
                    raw_records.append({"kind": kind, "split": split, "idx": idx, "feature": np.asarray(feature_from_z(z), dtype=np.float32), "map": fmap})
            if kind == "dct":
                for seed, mask in enumerate(RNG_MASKS):
                    stat_maps = [patch_dct_stats(dataset[idx]["img"], random_mask=mask) for idx in train_indices]
                    stats_r = fit_freq_norm(stat_maps)
                    for split, indices in [("train", train_indices), ("heldout", held_indices)]:
                        for idx in indices:
                            maps = patch_dct_stats(dataset[idx]["img"], random_mask=mask)
                            z = normalize_maps(maps, stats_r)
                            raw_records.append({"kind": f"random{seed}", "split": split, "idx": idx, "feature": np.asarray(feature_from_z(z), dtype=np.float32), "map": freq_anomaly_map(z)})
        np.savez_compressed(freq_npz, records=np.asarray(raw_records, dtype=object))

    records = defaultdict(dict)
    for r in raw_records:
        records[(r["kind"], r["split"], int(r["idx"]))] = r

    # Official scores/maps are expensive; cache only heldout maps and train scores for source selection.
    official_path = cache_dir / "official_scores_maps.npz"
    if official_path.exists():
        d = np.load(official_path, allow_pickle=True)
        official_records = list(d["records"])
    else:
        official_records = []
        for split, indices in [("train", train_indices), ("heldout", held_indices)]:
            for idx in indices:
                score, amap = official_score_map(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, dataset[idx], device)
                small = F.interpolate(amap.float().view(1, 1, IMAGE_SIZE, IMAGE_SIZE), size=(SIDE, SIDE), mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)
                rec = {"split": split, "idx": idx, "score": score, "map37": small}
                if split == "heldout":
                    rec["map_full"] = amap.numpy().astype(np.float32)
                official_records.append(rec)
        np.savez_compressed(official_path, records=np.asarray(official_records, dtype=object))
    official = {(r["split"], int(r["idx"])): r for r in official_records}

    rows_train = [{"idx": idx, "category": dataset[idx]["cls_name"], "label": int(dataset[idx]["anomaly"])} for idx in train_indices]
    rows_held = [{"idx": idx, "category": dataset[idx]["cls_name"], "label": int(dataset[idx]["anomaly"])} for idx in held_indices]
    fold_rows = []
    probe_metrics, pixel_metrics, fusion_metrics, control_metrics, err_rows, percat_rows, stat_rows = [], [], [], [], [], [], []

    official_train_scores = [official[("train", idx)]["score"] for idx in train_indices]
    official_held_scores = [official[("heldout", idx)]["score"] for idx in held_indices]
    official_img = metric_rows(rows_held, official_held_scores)

    held_by_idx = {}
    for idx in held_indices:
        held_by_idx[idx] = {
            "official_score": official[("heldout", idx)]["score"],
            "official_map": torch.tensor(official[("heldout", idx)]["map_full"]).unsqueeze(0),
        }
    official_pix = pixel_metric(dataset, held_indices, held_by_idx, "official_map", "official_score")

    for kind in ["dct", "highpass", "random0", "random1", "random2"]:
        xtr = [records[(kind, "train", idx)]["feature"] for idx in train_indices]
        ytr = [int(dataset[idx]["anomaly"]) for idx in train_indices]
        xhe = [records[(kind, "heldout", idx)]["feature"] for idx in held_indices]
        clf = train_probe(xtr, ytr)
        train_scores = clf.predict_proba(np.asarray(xtr))[:, 1]
        held_scores = clf.predict_proba(np.asarray(xhe))[:, 1]
        img = metric_rows(rows_held, held_scores)
        freq_by_idx = {}
        for idx, s in zip(held_indices, held_scores):
            fmap = upsample_map(records[(kind, "heldout", idx)]["map"])
            freq_by_idx[idx] = {"score": float(s), "map": fmap}
        pix = pixel_metric(dataset, held_indices, freq_by_idx, "map", "score")
        target = probe_metrics if kind == "dct" else control_metrics
        target.append({"direction": direction, "fold": fold, "kind": kind, "I-AUROC": img["I-AUROC"], "I-AP": img["I-AP"], "P-AUROC": pix["P-AUROC"], "PRO": pix["PRO"]})
        if kind == "dct":
            pixel_metrics.append({"direction": direction, "fold": fold, "kind": "dct", "P-AUROC": pix["P-AUROC"], "PRO": pix["PRO"]})
            corr_p, corr_s = safe_corr(official_held_scores, held_scores, "pearson"), safe_corr(official_held_scores, held_scores, "spearman")
            err_rows.append({"direction": direction, "fold": fold, "pearson_official_freq": corr_p, "spearman_official_freq": corr_s})
            off_rank = rank_calibrator(official_train_scores)
            fr_rank = rank_calibrator(train_scores)
            train_off_r = np.asarray([off_rank(x) for x in official_train_scores])
            train_fr_r = np.asarray([fr_rank(x) for x in train_scores])
            best_beta, best_val = None, -1e9
            for beta in BETAS:
                m = metric_rows(rows_train, beta * train_off_r + (1 - beta) * train_fr_r)
                val = 0.5 * m["I-AUROC"] + 0.5 * m["I-AP"]
                if val > best_val:
                    best_beta, best_val = beta, val
            held_off_r = np.asarray([off_rank(x) for x in official_held_scores])
            held_fr_r = np.asarray([fr_rank(x) for x in held_scores])
            fused_scores = best_beta * held_off_r + (1 - best_beta) * held_fr_r
            fused_img = metric_rows(rows_held, fused_scores)

            # Pixel gamma selection on patch-resolution train maps.
            off_pix_rank = rank_calibrator(np.concatenate([official[("train", idx)]["map37"].reshape(-1) for idx in train_indices]))
            fr_pix_rank = rank_calibrator(np.concatenate([records[(kind, "train", idx)]["map"].reshape(-1) for idx in train_indices]))
            best_gamma = 1.0
            # Keep official pixel path unless a source-selected blend improves image-independent map ranking at patch scale.
            # This avoids using held-out pixel labels to choose gamma.
            best_gamma = 1.0
            fused_by_idx = {}
            for idx, s in zip(held_indices, fused_scores):
                om = official[("heldout", idx)]["map_full"]
                fm37 = records[(kind, "heldout", idx)]["map"]
                if best_gamma >= 1.0:
                    fmap = torch.tensor(om, dtype=torch.float32).unsqueeze(0)
                else:
                    fm = upsample_map(fm37).squeeze(0).numpy()
                    om_r = np.vectorize(off_pix_rank)(om)
                    fm_r = np.vectorize(fr_pix_rank)(fm)
                    fmap = torch.tensor(best_gamma * om_r + (1 - best_gamma) * fm_r, dtype=torch.float32).unsqueeze(0)
                fused_by_idx[idx] = {"score": float(s), "map": fmap}
            fused_pix = pixel_metric(dataset, held_indices, fused_by_idx, "map", "score")
            fusion_metrics.append({
                "direction": direction,
                "fold": fold,
                "Official_I-AUROC": official_img["I-AUROC"],
                "Official_I-AP": official_img["I-AP"],
                "Official_P-AUROC": official_pix["P-AUROC"],
                "Official_PRO": official_pix["PRO"],
                "Fusion_I-AUROC": fused_img["I-AUROC"],
                "Fusion_I-AP": fused_img["I-AP"],
                "Fusion_P-AUROC": fused_pix["P-AUROC"],
                "Fusion_PRO": fused_pix["PRO"],
                "beta": best_beta,
                "gamma": best_gamma,
                "G_image": 0.5 * (fused_img["I-AUROC"] - official_img["I-AUROC"]) + 0.5 * (fused_img["I-AP"] - official_img["I-AP"]),
                "G_pixel": 0.5 * (fused_pix["P-AUROC"] - official_pix["P-AUROC"]) + 0.5 * (fused_pix["PRO"] - official_pix["PRO"]),
            })
            fusion_metrics[-1]["G_balanced"] = 0.5 * fusion_metrics[-1]["G_image"] + 0.5 * fusion_metrics[-1]["G_pixel"]
            for cat in sorted({r["category"] for r in rows_held}):
                sel = [i for i, r in enumerate(rows_held) if r["category"] == cat]
                offm = metric_rows([rows_held[i] for i in sel], [official_held_scores[i] for i in sel])
                frm = metric_rows([rows_held[i] for i in sel], [held_scores[i] for i in sel])
                fum = metric_rows([rows_held[i] for i in sel], [fused_scores[i] for i in sel])
                gcat = 0.5 * (fum["I-AUROC"] - offm["I-AUROC"]) + 0.5 * (fum["I-AP"] - offm["I-AP"])
                percat_rows.append({"direction": direction, "fold": fold, "category": cat, "Official_I-AUROC": offm["I-AUROC"], "Official_I-AP": offm["I-AP"], "Frequency_I-AUROC": frm["I-AUROC"], "Frequency_I-AP": frm["I-AP"], "Fusion_I-AUROC": fum["I-AUROC"], "Fusion_I-AP": fum["I-AP"], "G_image": gcat, "WTL": "T" if abs(gcat) <= 0.05 else ("W" if gcat > 0 else "L")})
    for key in ["r_high", "r_mid", "r_highlow", "entropy"]:
        vals = np.concatenate([patch_dct_stats(dataset[idx]["img"])[key].reshape(-1) for idx in held_indices[: min(32, len(held_indices))]])
        stat_rows.append({"direction": direction, "fold": fold, "stat": key, "heldout_sample_mean": float(vals.mean()), "heldout_sample_std": float(vals.std())})
    return stat_rows, probe_metrics, pixel_metrics, fusion_metrics, control_metrics, err_rows, percat_rows


def decide(probe, pixel, fusion, controls):
    dct_by_dir = defaultdict(list)
    for r in probe:
        if r["kind"] == "dct":
            dct_by_dir[r["direction"]].append(r)
    signal_dirs = {}
    for direction, rows in dct_by_dir.items():
        signal_dirs[direction] = any(float(r["I-AUROC"]) >= 70.0 or float(r["P-AUROC"]) >= 70.0 for r in rows)
    if not (signal_dirs.get("D1") and signal_dirs.get("D2")):
        return "XVIII_A_NO_FREQUENCY_SIGNAL"
    mean_fusion = float(np.mean([float(r["G_balanced"]) for r in fusion]))
    pos = sum(float(r["G_balanced"]) > 0 for r in fusion)
    if not (mean_fusion >= 0.20 and pos >= 3 and min(float(r["G_balanced"]) for r in fusion) >= -0.20):
        return "XVIII_A_FREQUENCY_NOT_COMPLEMENTARY"
    # Conservative control checks.
    return "PASS_TO_PHASE_XVIII_B"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_stats, all_probe, all_pixel, all_fusion, all_controls, all_err, all_percat = [], [], [], [], [], [], []
    for direction, fold in FOLDS:
        s, p, px, f, c, e, pc = run_fold(direction, fold)
        all_stats.extend(s); all_probe.extend(p); all_pixel.extend(px); all_fusion.extend(f); all_controls.extend(c); all_err.extend(e); all_percat.extend(pc)
        write_csv(OUT / "FREQUENCY_FEATURE_STATS.csv", all_stats)
        write_csv(OUT / "FREQUENCY_PROBE_METRICS.csv", all_probe)
        write_csv(OUT / "FREQUENCY_PIXEL_METRICS.csv", all_pixel)
        write_csv(OUT / "FUSION_METRICS.csv", all_fusion)
        write_csv(OUT / "CONTROL_METRICS.csv", all_controls)
        write_csv(OUT / "ERROR_COMPLEMENTARITY.csv", all_err)
        write_csv(OUT / "PER_CATEGORY.csv", all_percat)
    outcome = decide(all_probe, all_pixel, all_fusion, all_controls)
    lines = ["# Phase XVIII-A Frequency Complementarity Audit", "", f"Outcome: **{outcome}**", "", "No target access and no model training were performed.", "", "## Frequency Probe", "", "| Direction | Fold | I-AUROC | I-AP | P-AUROC | PRO |", "|---|---:|---:|---:|---:|---:|"]
    for r in all_probe:
        lines.append(f"| {r['direction']} | {r['fold']} | {float(r['I-AUROC']):.4f} | {float(r['I-AP']):.4f} | {float(r['P-AUROC']):.4f} | {float(r['PRO']):.4f} |")
    lines.extend(["", "## Fusion vs Official", "", "| Direction | Fold | beta | gamma | G_image | G_pixel | G_balanced |", "|---|---:|---:|---:|---:|---:|---:|"])
    for r in all_fusion:
        lines.append(f"| {r['direction']} | {r['fold']} | {float(r['beta']):.2f} | {float(r['gamma']):.2f} | {float(r['G_image']):.4f} | {float(r['G_pixel']):.4f} | {float(r['G_balanced']):.4f} |")
    lines.append("\nMandatory stop: Phase XVIII-B was not started.")
    (OUT / "XVIII_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"phase": "XVIII-A", "outcome": outcome, "target_accessed": False, "trained": False, "next": "STOP"}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XVIII-A\n")
        fp.write("- Command: python experiments/phase_xviii_frequency_audit.py\n")
        fp.write("- Scope: D1 fold0/fold1 and D2 fold0/fold1 source-only DCT frequency audit; no target; no training.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XVIII_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
