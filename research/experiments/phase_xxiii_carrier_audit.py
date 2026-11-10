import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import evaluate_results, init_results
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from phase_xa.official_task_forward import normalized_image_feature, normalized_patch_feature, visual_outputs
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xxiii_carrier_audit")
SIDE = 37
IMAGE_SIZE = 518
BANK_CAP = 20_000
BETAS = [0.0, 0.25, 0.50, 0.75, 1.0]
EPS = 1e-12


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if len(np.unique(y)) == 2 else float("nan")


def metric_rows(rows, scores):
    by_cat = defaultdict(lambda: {"y": [], "s": []})
    for r, s in zip(rows, scores):
        by_cat[r["category"]]["y"].append(int(r["label"]))
        by_cat[r["category"]]["s"].append(float(s))
    aucs, aps, percat = [], [], {}
    for cat, v in by_cat.items():
        auc = safe_auc(v["y"], v["s"])
        ap = safe_ap(v["y"], v["s"])
        percat[cat] = {"I-AUROC": auc, "I-AP": ap}
        if np.isfinite(auc):
            aucs.append(auc)
        if np.isfinite(ap):
            aps.append(ap)
    return {"I-AUROC": float(np.mean(aucs)) if aucs else float("nan"), "I-AP": float(np.mean(aps)) if aps else float("nan"), "per_category": percat}


def g_image(candidate, base):
    return 0.5 * (float(candidate["I-AUROC"]) - float(base["I-AUROC"])) + 0.5 * (float(candidate["I-AP"]) - float(base["I-AP"]))


def append_pixel(results, item, score, map37):
    cls = item["cls_name"]
    amap = torch.tensor(map37, dtype=torch.float32).view(1, 1, SIDE, SIDE)
    amap = F.interpolate(amap, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze(0)
    amap = torch.from_numpy(gaussian_filter(amap.squeeze(0).numpy(), sigma=4)).unsqueeze(0)
    results[cls]["imgs_masks"].append((item["img_mask"].detach().cpu() > 0.5).float())
    results[cls]["gt_sp"].append(int(item["anomaly"]))
    results[cls]["pr_sp"].append(float(score))
    results[cls]["anomaly_maps"].append(amap)


def pixel_metrics(dataset, indices, scores, maps):
    obj_list = sorted({dataset[i]["cls_name"] for i in indices})
    results = init_results(obj_list)
    for pos, idx in enumerate(indices):
        append_pixel(results, dataset[idx], float(scores[pos]), maps[pos])
    m = evaluate_results(results, obj_list, compute_pro=True)
    out = {}
    for k in ["P-AUROC", "PRO"]:
        v = float(m["mean"].get(k, float("nan")))
        out[k] = v * 100.0 if np.isfinite(v) and v <= 1.5 else v
    return out, m.get("per_class", {})


def rank_calibrator(vals):
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    return lambda x: np.searchsorted(arr, x, side="right") / max(1, len(arr))


def apply_rank(vals, calibrator):
    return np.asarray([calibrator(v) for v in vals], dtype=np.float64)


def train_probe(x, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", class_weight="balanced", random_state=0))
    clf.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64))
    return clf


def probe_scores(train_x, train_y, held_x):
    clf = train_probe(train_x, train_y)
    return clf.predict_proba(np.asarray(train_x, dtype=np.float64))[:, 1], clf.predict_proba(np.asarray(held_x, dtype=np.float64))[:, 1]


def dino_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def extract_dino_one(model, path, transform, device):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    out = model.forward_features(x)
    cls = out["x_norm_clstoken"].float()
    patch = out["x_norm_patchtokens"].float()
    cls = F.normalize(cls, dim=-1)
    patch = F.normalize(patch, dim=-1)
    return cls.squeeze(0).cpu().numpy().astype(np.float32), patch.squeeze(0).cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_clip_one(model, item, device):
    image = item["img"].unsqueeze(0).to(device)
    out = visual_outputs(model, image, adapter=None, grad=False)
    cls = normalized_image_feature(out).squeeze(0).float().cpu().numpy().astype(np.float32)
    patch = normalized_patch_feature(out).squeeze(0).float().cpu().numpy().astype(np.float32)
    return cls, patch


def balanced_normal_patch_bank(dataset, train_indices, extractor, dim, seed=0):
    rng = np.random.default_rng(seed)
    normal_by_cat = defaultdict(list)
    for idx in train_indices:
        item = dataset[idx]
        if int(item["anomaly"]) == 0:
            normal_by_cat[item["cls_name"]].append(idx)
    cats = sorted(normal_by_cat)
    per_cat = max(1, BANK_CAP // max(1, len(cats)))
    parts = []
    for cat in cats:
        need = per_cat
        cat_feats = []
        for idx in normal_by_cat[cat]:
            patch = extractor(idx)
            if patch.shape[0] <= need:
                cat_feats.append(patch)
                need -= patch.shape[0]
            else:
                sel = rng.choice(patch.shape[0], size=need, replace=False)
                cat_feats.append(patch[sel])
                need = 0
            if need <= 0:
                break
        if cat_feats:
            parts.append(np.concatenate(cat_feats, axis=0))
    bank = np.concatenate(parts, axis=0).astype(np.float32) if parts else np.zeros((0, dim), dtype=np.float32)
    if bank.shape[0] > BANK_CAP:
        sel = rng.choice(bank.shape[0], size=BANK_CAP, replace=False)
        bank = bank[sel]
    bank /= np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), EPS)
    return bank


def normal_distance_map(patch, bank, device, chunk=256):
    p = torch.tensor(patch, dtype=torch.float32, device=device)
    b = torch.tensor(bank, dtype=torch.float32, device=device)
    sims = []
    with torch.no_grad():
        for start in range(0, p.shape[0], chunk):
            sims.append((p[start : start + chunk] @ b.t()).max(dim=1).values.detach().cpu())
    score = 1.0 - torch.cat(sims).numpy()
    return score.reshape(SIDE, SIDE).astype(np.float32)


def map_score(map37):
    flat = map37.reshape(-1)
    k = max(1, int(flat.size * 0.01))
    return float(np.sort(flat)[-k:].mean())


def random_projection_matrix(in_dim, out_dim, seed=0):
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    q, _ = np.linalg.qr(mat)
    return q[:, :out_dim].astype(np.float32)


def effective_rank(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    arr = arr - arr.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(arr, compute_uv=False)
    except Exception:
        return float("nan")
    p = s / max(EPS, s.sum())
    return float(np.exp(-(p * np.log(p + EPS)).sum()))


def geometry_rows(direction, fold, carrier, train_cls, train_patch_samples, train_rows):
    rows = []
    normal_cls = np.asarray([x for x, r in zip(train_cls, train_rows) if int(r["label"]) == 0], dtype=np.float32)
    anomaly_cls = np.asarray([x for x, r in zip(train_cls, train_rows) if int(r["label"]) == 1], dtype=np.float32)
    def pair_cos(a, b=None, max_pairs=4096):
        if a.size == 0:
            return float("nan"), float("nan")
        rng = np.random.default_rng(0)
        if b is None:
            if len(a) < 2:
                return float("nan"), float("nan")
            i = rng.integers(0, len(a), size=max_pairs)
            j = rng.integers(0, len(a), size=max_pairs)
            ok = i != j
            vals = (a[i[ok]] * a[j[ok]]).sum(axis=1)
        else:
            if b.size == 0:
                return float("nan"), float("nan")
            i = rng.integers(0, len(a), size=max_pairs)
            j = rng.integers(0, len(b), size=max_pairs)
            vals = (a[i] * b[j]).sum(axis=1)
        return float(vals.mean()), float(vals.std())
    nn_m, nn_s = pair_cos(normal_cls)
    na_m, na_s = pair_cos(normal_cls, anomaly_cls)
    smooth_vals, var_vals = [], []
    for p in train_patch_samples[: min(len(train_patch_samples), 128)]:
        p = p.astype(np.float32)
        grid = p.reshape(SIDE, SIDE, -1)
        smooth_vals.append(float(np.mean((grid[:, :-1] * grid[:, 1:]).sum(axis=-1))))
        smooth_vals.append(float(np.mean((grid[:-1, :] * grid[1:, :]).sum(axis=-1))))
        var_vals.append(float(np.mean(np.var(p, axis=0))))
    centroids = []
    for cat in sorted({r["category"] for r in train_rows}):
        vals = np.asarray([x for x, r in zip(train_cls, train_rows) if r["category"] == cat], dtype=np.float32)
        if len(vals):
            c = vals.mean(axis=0)
            c /= max(EPS, np.linalg.norm(c))
            centroids.append(c)
    cd_m, cd_s = pair_cos(np.asarray(centroids, dtype=np.float32)) if len(centroids) >= 2 else (float("nan"), float("nan"))
    rows.append({
        "direction": direction,
        "fold": fold,
        "carrier": carrier,
        "normal_normal_cos_mean": nn_m,
        "normal_normal_cos_std": nn_s,
        "normal_anomaly_cos_mean": na_m,
        "normal_anomaly_cos_std": na_s,
        "effective_rank_cls": effective_rank(train_cls),
        "patch_spatial_smoothness": float(np.mean(smooth_vals)) if smooth_vals else float("nan"),
        "within_image_patch_variance": float(np.mean(var_vals)) if var_vals else float("nan"),
        "between_category_centroid_cos_mean": cd_m,
        "between_category_centroid_cos_std": cd_s,
        "between_category_centroid_distance": float(1.0 - cd_m) if np.isfinite(cd_m) else float("nan"),
    })
    return rows


def run_fold(direction, fold, device):
    dataset_name, data_path, checkpoint = direction_cfg(direction)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
    train_rows = [{"idx": idx, "category": dataset[idx]["cls_name"], "label": int(dataset[idx]["anomaly"])} for idx in train_indices]
    held_rows = [{"idx": idx, "category": dataset[idx]["cls_name"], "label": int(dataset[idx]["anomaly"])} for idx in held_indices]

    clip_cache = OUT / "feature_cache" / f"{direction}_fold{fold}_clip.npz"
    dino_cache = OUT / "feature_cache" / f"{direction}_fold{fold}_dino.npz"
    clip_cache.parent.mkdir(parents=True, exist_ok=True)

    if clip_cache.exists():
        c = np.load(clip_cache, allow_pickle=True)
        clip_train_cls, clip_held_cls = c["train_cls"], c["held_cls"]
        clip_train_patch, clip_held_patch = c["train_patch"], c["held_patch"]
    else:
        model, *_ = load_components(checkpoint, device)
        clip_train_cls, clip_train_patch = [], []
        for idx in train_indices:
            cls, patch = extract_clip_one(model, dataset[idx], device)
            clip_train_cls.append(cls); clip_train_patch.append(patch.astype(np.float16))
        clip_held_cls, clip_held_patch = [], []
        for idx in held_indices:
            cls, patch = extract_clip_one(model, dataset[idx], device)
            clip_held_cls.append(cls); clip_held_patch.append(patch.astype(np.float16))
        clip_train_cls, clip_held_cls = np.stack(clip_train_cls), np.stack(clip_held_cls)
        clip_train_patch, clip_held_patch = np.stack(clip_train_patch), np.stack(clip_held_patch)
        np.savez_compressed(clip_cache, train_cls=clip_train_cls, held_cls=clip_held_cls, train_patch=clip_train_patch, held_patch=clip_held_patch)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dino_cache.exists():
        d = np.load(dino_cache, allow_pickle=True)
        dino_train_cls, dino_held_cls = d["train_cls"], d["held_cls"]
        dino_train_patch, dino_held_patch = d["train_patch"], d["held_patch"]
    else:
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True).to(device).eval()
        for p in dino.parameters():
            p.requires_grad_(False)
        tfm = dino_transform()
        dino_train_cls, dino_train_patch = [], []
        for idx in train_indices:
            cls, patch = extract_dino_one(dino, dataset[idx]["img_path"], tfm, device)
            dino_train_cls.append(cls); dino_train_patch.append(patch.astype(np.float16))
        dino_held_cls, dino_held_patch = [], []
        for idx in held_indices:
            cls, patch = extract_dino_one(dino, dataset[idx]["img_path"], tfm, device)
            dino_held_cls.append(cls); dino_held_patch.append(patch.astype(np.float16))
        dino_train_cls, dino_held_cls = np.stack(dino_train_cls), np.stack(dino_held_cls)
        dino_train_patch, dino_held_patch = np.stack(dino_train_patch), np.stack(dino_held_patch)
        np.savez_compressed(dino_cache, train_cls=dino_train_cls, held_cls=dino_held_cls, train_patch=dino_train_patch, held_patch=dino_held_patch)
        del dino
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Keep cached patch tensors in float16 and cast one image/chunk at a time.
    # D2 folds are large enough that eager float32 copies of CLIP+DINO patches
    # can terminate the process before an OUTCOME is written.

    y_train = np.asarray([r["label"] for r in train_rows], dtype=np.int64)
    y_held = np.asarray([r["label"] for r in held_rows], dtype=np.int64)
    clip_train_score, clip_held_score = probe_scores(clip_train_cls, y_train, clip_held_cls)
    dino_train_score, dino_held_score = probe_scores(dino_train_cls, y_train, dino_held_cls)
    proj = random_projection_matrix(dino_train_cls.shape[1], clip_train_cls.shape[1], seed=0)
    dino_rp_train_score, dino_rp_held_score = probe_scores(dino_train_cls @ proj, y_train, dino_held_cls @ proj)

    image_rows = []
    for name, tr, he in [("CLIP_CLS", clip_train_score, clip_held_score), ("DINO_CLS", dino_train_score, dino_held_score), ("DINO_RandomOrtho", dino_rp_train_score, dino_rp_held_score)]:
        met = metric_rows(held_rows, he)
        image_rows.append({"direction": direction, "fold": fold, "representation": name, "I-AUROC": met["I-AUROC"], "I-AP": met["I-AP"]})

    clip_bank = balanced_normal_patch_bank(dataset, train_indices, lambda idx: clip_train_patch[train_indices.index(idx)], clip_train_patch.shape[-1], seed=0)
    dino_bank = balanced_normal_patch_bank(dataset, train_indices, lambda idx: dino_train_patch[train_indices.index(idx)], dino_train_patch.shape[-1], seed=0)
    def project_patch(patch):
        out = patch.astype(np.float32) @ proj
        out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), EPS)
        return out.astype(np.float32)

    dino_rp_bank = balanced_normal_patch_bank(dataset, train_indices, lambda idx: project_patch(dino_train_patch[train_indices.index(idx)]), clip_train_cls.shape[-1], seed=0)

    clip_maps = np.stack([normal_distance_map(p.astype(np.float32), clip_bank, device) for p in clip_held_patch])
    dino_maps = np.stack([normal_distance_map(p.astype(np.float32), dino_bank, device) for p in dino_held_patch])
    dino_rp_maps = np.stack([normal_distance_map(project_patch(p), dino_rp_bank, device) for p in dino_held_patch])
    clip_pix_scores = np.asarray([map_score(m) for m in clip_maps])
    dino_pix_scores = np.asarray([map_score(m) for m in dino_maps])
    dino_rp_pix_scores = np.asarray([map_score(m) for m in dino_rp_maps])

    pixel_rows, pixel_perclass = [], {}
    for name, scores, maps in [("CLIP_patch", clip_pix_scores, clip_maps), ("DINO_patch", dino_pix_scores, dino_maps), ("DINO_RandomOrtho_patch", dino_rp_pix_scores, dino_rp_maps)]:
        pix, per = pixel_metrics(dataset, held_indices, scores, maps)
        pixel_rows.append({"direction": direction, "fold": fold, "representation": name, "P-AUROC": pix["P-AUROC"], "PRO": pix["PRO"]})
        pixel_perclass[name] = per

    # Source-selected rank fusion.
    clip_rank = rank_calibrator(clip_train_score)
    dino_rank = rank_calibrator(dino_train_score)
    clip_train_r, dino_train_r = apply_rank(clip_train_score, clip_rank), apply_rank(dino_train_score, dino_rank)
    clip_held_r, dino_held_r = apply_rank(clip_held_score, clip_rank), apply_rank(dino_held_score, dino_rank)
    best_beta, best_val = 0.0, -1e9
    for beta in BETAS:
        m = metric_rows(train_rows, beta * clip_train_r + (1 - beta) * dino_train_r)
        val = 0.5 * m["I-AUROC"] + 0.5 * m["I-AP"]
        if val > best_val:
            best_beta, best_val = beta, val
    fused_image_scores = best_beta * clip_held_r + (1 - best_beta) * dino_held_r
    fused_img = metric_rows(held_rows, fused_image_scores)

    clip_train_maps = np.stack([normal_distance_map(p.astype(np.float32), clip_bank, device) for p in clip_train_patch])
    dino_train_maps = np.stack([normal_distance_map(p.astype(np.float32), dino_bank, device) for p in dino_train_patch])
    clip_map_rank = rank_calibrator(clip_train_maps.reshape(-1))
    dino_map_rank = rank_calibrator(dino_train_maps.reshape(-1))
    clip_train_pix_r = np.asarray([map_score(apply_rank(m.reshape(-1), clip_map_rank).reshape(SIDE, SIDE)) for m in clip_train_maps])
    dino_train_pix_r = np.asarray([map_score(apply_rank(m.reshape(-1), dino_map_rank).reshape(SIDE, SIDE)) for m in dino_train_maps])
    best_pbeta, best_pval = 1.0, -1e9
    for beta in BETAS:
        m = metric_rows(train_rows, beta * clip_train_pix_r + (1 - beta) * dino_train_pix_r)
        val = 0.5 * m["I-AUROC"] + 0.5 * m["I-AP"]
        if val > best_pval:
            best_pbeta, best_pval = beta, val
    clip_held_maps_r = np.asarray([apply_rank(m.reshape(-1), clip_map_rank).reshape(SIDE, SIDE) for m in clip_maps], dtype=np.float32)
    dino_held_maps_r = np.asarray([apply_rank(m.reshape(-1), dino_map_rank).reshape(SIDE, SIDE) for m in dino_maps], dtype=np.float32)
    fused_maps = best_pbeta * clip_held_maps_r + (1 - best_pbeta) * dino_held_maps_r
    fused_pix_scores = np.asarray([map_score(m) for m in fused_maps])
    fused_pix, _ = pixel_metrics(dataset, held_indices, fused_pix_scores, fused_maps)

    fusion_rows = [
        {"direction": direction, "fold": fold, "task": "image", "selected_beta_clip": best_beta, "I-AUROC": fused_img["I-AUROC"], "I-AP": fused_img["I-AP"], "source_train_selection_metric": best_val},
        {"direction": direction, "fold": fold, "task": "pixel", "selected_beta_clip": best_pbeta, "P-AUROC": fused_pix["P-AUROC"], "PRO": fused_pix["PRO"], "source_train_selection_metric": best_pval},
    ]

    rng = np.random.default_rng(0)
    shuf = rng.permutation(len(dino_held_score))
    shuffled_img = metric_rows(held_rows, best_beta * clip_held_r + (1 - best_beta) * dino_held_r[shuf])
    shuf_map_idx = rng.permutation(len(dino_held_maps_r))
    shuffled_maps = best_pbeta * clip_held_maps_r + (1 - best_pbeta) * dino_held_maps_r[shuf_map_idx]
    shuffled_pix_scores = np.asarray([map_score(m) for m in shuffled_maps])
    shuffled_pix, _ = pixel_metrics(dataset, held_indices, shuffled_pix_scores, shuffled_maps)
    control_rows = [
        {"direction": direction, "fold": fold, "control": "C1_DINO_RandomOrtho_image", "I-AUROC": image_rows[2]["I-AUROC"], "I-AP": image_rows[2]["I-AP"]},
        {"direction": direction, "fold": fold, "control": "C1_DINO_RandomOrtho_pixel", "P-AUROC": pixel_rows[2]["P-AUROC"], "PRO": pixel_rows[2]["PRO"]},
        {"direction": direction, "fold": fold, "control": "C2_ShuffledDINO_image_fusion", "I-AUROC": shuffled_img["I-AUROC"], "I-AP": shuffled_img["I-AP"]},
        {"direction": direction, "fold": fold, "control": "C3_ShuffledDINO_patch_fusion", "P-AUROC": shuffled_pix["P-AUROC"], "PRO": shuffled_pix["PRO"]},
    ]

    img_corr_p = float(pearsonr(clip_held_score, dino_held_score).statistic) if len(clip_held_score) > 2 else float("nan")
    img_corr_s = float(spearmanr(clip_held_score, dino_held_score).statistic) if len(clip_held_score) > 2 else float("nan")
    clip_pred = (clip_held_score >= np.median(clip_train_score)).astype(np.int64)
    dino_pred = (dino_held_score >= np.median(dino_train_score)).astype(np.int64)
    img_disagree = float(np.mean((clip_pred == y_held) != (dino_pred == y_held)))
    oracle_img = np.maximum(
        np.where(y_held == 1, clip_held_score, -clip_held_score),
        np.where(y_held == 1, dino_held_score, -dino_held_score),
    )
    oracle_img_met = metric_rows(held_rows, oracle_img)
    best_img = max(image_rows[:2], key=lambda r: 0.5 * float(r["I-AUROC"]) + 0.5 * float(r["I-AP"]))
    oracle_headroom = 0.5 * (oracle_img_met["I-AUROC"] - best_img["I-AUROC"]) + 0.5 * (oracle_img_met["I-AP"] - best_img["I-AP"])

    map_cos, map_sp, pix_dis = [], [], []
    for cm, dm in zip(clip_maps, dino_maps):
        c, d = cm.reshape(-1), dm.reshape(-1)
        map_cos.append(float(np.dot(c, d) / max(EPS, np.linalg.norm(c) * np.linalg.norm(d))))
        map_sp.append(float(spearmanr(c, d).statistic))
        pix_dis.append(float(np.mean((c > np.percentile(c, 95)) != (d > np.percentile(d, 95)))))
    complement_rows = [{
        "direction": direction,
        "fold": fold,
        "image_pearson": img_corr_p,
        "image_spearman": img_corr_s,
        "image_error_disagreement": img_disagree,
        "OracleHeadroom_image": oracle_headroom,
        "pixel_map_cosine": float(np.nanmean(map_cos)),
        "pixel_map_spearman": float(np.nanmean(map_sp)),
        "pixel_error_disagreement": float(np.nanmean(pix_dis)),
    }]

    geometry = []
    geometry.extend(geometry_rows(direction, fold, "CLIP", clip_train_cls, clip_train_patch, train_rows))
    geometry.extend(geometry_rows(direction, fold, "DINO", dino_train_cls, dino_train_patch, train_rows))

    percat = []
    clip_img_pc = metric_rows(held_rows, clip_held_score)["per_category"]
    dino_img_pc = metric_rows(held_rows, dino_held_score)["per_category"]
    fuse_img_pc = fused_img["per_category"]
    for cat in sorted({r["category"] for r in held_rows}):
        row = {"direction": direction, "fold": fold, "category": cat}
        row["CLIP_I-AUROC"] = clip_img_pc[cat]["I-AUROC"]; row["CLIP_I-AP"] = clip_img_pc[cat]["I-AP"]
        row["DINO_I-AUROC"] = dino_img_pc[cat]["I-AUROC"]; row["DINO_I-AP"] = dino_img_pc[cat]["I-AP"]
        row["Fusion_I-AUROC"] = fuse_img_pc[cat]["I-AUROC"]; row["Fusion_I-AP"] = fuse_img_pc[cat]["I-AP"]
        delta = 0.5 * (row["Fusion_I-AUROC"] - row["CLIP_I-AUROC"]) + 0.5 * (row["Fusion_I-AP"] - row["CLIP_I-AP"])
        row["Fusion_vs_CLIP_G_image"] = delta
        row["WTL"] = "T" if abs(delta) <= 0.05 else ("W" if delta > 0 else "L")
        percat.append(row)

    return {
        "image": image_rows,
        "pixel": pixel_rows,
        "fusion": fusion_rows,
        "control": control_rows,
        "complement": complement_rows,
        "geometry": geometry,
        "percat": percat,
        "split": {"direction": direction, "fold": fold, "dataset": dataset_name, "train_categories": rotation["train_categories"], "heldout_categories": rotation["val_categories"], "train_images": len(train_indices), "heldout_images": len(held_indices), "clip_dim": int(clip_train_cls.shape[1]), "dino_dim": int(dino_train_cls.shape[1]), "patch_tokens": int(clip_train_patch.shape[1])},
    }


def decide(image_rows, pixel_rows, fusion_rows, control_rows):
    img = {(r["direction"], int(r["fold"]), r["representation"]): r for r in image_rows}
    pix = {(r["direction"], int(r["fold"]), r["representation"]): r for r in pixel_rows}
    fus = {(r["direction"], int(r["fold"]), r["task"]): r for r in fusion_rows}
    ctrl = defaultdict(dict)
    for r in control_rows:
        ctrl[(r["direction"], int(r["fold"]))][r["control"]] = r
    dir_signal = {}
    details = {}
    for d in ["D1", "D2"]:
        img_g, pix_g, fus_img_g, fus_pix_g, shuf_img_gap, shuf_pix_gap = [], [], [], [], [], []
        for f in [0, 1]:
            clip_i, dino_i = img[(d, f, "CLIP_CLS")], img[(d, f, "DINO_CLS")]
            clip_p, dino_p = pix[(d, f, "CLIP_patch")], pix[(d, f, "DINO_patch")]
            ig = 0.5 * (float(dino_i["I-AUROC"]) - float(clip_i["I-AUROC"])) + 0.5 * (float(dino_i["I-AP"]) - float(clip_i["I-AP"]))
            pg = 0.5 * (float(dino_p["P-AUROC"]) - float(clip_p["P-AUROC"])) + 0.5 * (float(dino_p["PRO"]) - float(clip_p["PRO"]))
            fi = fus[(d, f, "image")]
            fp = fus[(d, f, "pixel")]
            fig = 0.5 * (float(fi["I-AUROC"]) - float(clip_i["I-AUROC"])) + 0.5 * (float(fi["I-AP"]) - float(clip_i["I-AP"]))
            fpg = 0.5 * (float(fp["P-AUROC"]) - float(clip_p["P-AUROC"])) + 0.5 * (float(fp["PRO"]) - float(clip_p["PRO"]))
            si = ctrl[(d, f)]["C2_ShuffledDINO_image_fusion"]
            sp = ctrl[(d, f)]["C3_ShuffledDINO_patch_fusion"]
            shuf_img_gap.append(fig - (0.5 * (float(si["I-AUROC"]) - float(clip_i["I-AUROC"])) + 0.5 * (float(si["I-AP"]) - float(clip_i["I-AP"]))))
            shuf_pix_gap.append(fpg - (0.5 * (float(sp["P-AUROC"]) - float(clip_p["P-AUROC"])) + 0.5 * (float(sp["PRO"]) - float(clip_p["PRO"]))))
            img_g.append(ig); pix_g.append(pg); fus_img_g.append(fig); fus_pix_g.append(fpg)
        details[d] = {
            "DINO_image_gain_mean": float(np.mean(img_g)),
            "DINO_pixel_gain_mean": float(np.mean(pix_g)),
            "fusion_image_gain_mean": float(np.mean(fus_img_g)),
            "fusion_pixel_gain_mean": float(np.mean(fus_pix_g)),
            "fusion_image_real_minus_shuffled": float(np.mean(shuf_img_gap)),
            "fusion_pixel_real_minus_shuffled": float(np.mean(shuf_pix_gap)),
        }
        useful = (
            details[d]["DINO_image_gain_mean"] >= 0.50
            or details[d]["DINO_pixel_gain_mean"] >= 0.50
            or (
                (details[d]["fusion_image_gain_mean"] >= 0.30 and details[d]["fusion_pixel_gain_mean"] >= -0.10 and details[d]["fusion_image_real_minus_shuffled"] >= 0.20)
                or (details[d]["fusion_pixel_gain_mean"] >= 0.30 and details[d]["fusion_image_gain_mean"] >= -0.10 and details[d]["fusion_pixel_real_minus_shuffled"] >= 0.20)
            )
        )
        dir_signal[d] = useful
    if dir_signal["D1"] and dir_signal["D2"]:
        return "PASS_TO_PHASE_XXIII_B", details
    if dir_signal["D1"] != dir_signal["D2"]:
        return "XXIII_A_DIRECTION_SPECIFIC_CARRIER_GAIN", details
    any_small = any(max(v["DINO_image_gain_mean"], v["DINO_pixel_gain_mean"], v["fusion_image_gain_mean"], v["fusion_pixel_gain_mean"]) > 0 for v in details.values())
    if any_small:
        return "XXIII_A_COMPLEMENTARY_BUT_NOT_STRONG_ENOUGH", details
    clip_dominates = all(v["DINO_image_gain_mean"] < 0 and v["DINO_pixel_gain_mean"] < 0 for v in details.values())
    return "XXIII_A_CLIP_ALREADY_DOMINATES" if clip_dominates else "XXIII_A_NO_DINO_ANOMALY_SIGNAL", details


def write_report(outcome, details, image_rows, pixel_rows, fusion_rows):
    lines = ["# Phase XXIII-A Frozen Representation Carrier Audit", "", f"Outcome: **{outcome}**", "", "DINOv2 ViT-L/14 was loaded exactly via torch hub. No MRAD training, target access, adapters, or pipeline replacement were performed.", "", "## Image Probes", "", "| Direction | Fold | Representation | I-AUROC | I-AP |", "|---|---:|---|---:|---:|"]
    for r in image_rows:
        lines.append(f"| {r['direction']} | {r['fold']} | {r['representation']} | {float(r['I-AUROC']):.4f} | {float(r['I-AP']):.4f} |")
    lines.extend(["", "## Pixel Distance", "", "| Direction | Fold | Representation | P-AUROC | PRO |", "|---|---:|---|---:|---:|"])
    for r in pixel_rows:
        lines.append(f"| {r['direction']} | {r['fold']} | {r['representation']} | {float(r['P-AUROC']):.4f} | {float(r['PRO']):.4f} |")
    lines.extend(["", "## Fusion", "", "| Direction | Fold | Task | beta CLIP | Metric 1 | Metric 2 |", "|---|---:|---|---:|---:|---:|"])
    for r in fusion_rows:
        if r["task"] == "image":
            lines.append(f"| {r['direction']} | {r['fold']} | image | {float(r['selected_beta_clip']):.2f} | {float(r['I-AUROC']):.4f} | {float(r['I-AP']):.4f} |")
        else:
            lines.append(f"| {r['direction']} | {r['fold']} | pixel | {float(r['selected_beta_clip']):.2f} | {float(r['P-AUROC']):.4f} | {float(r['PRO']):.4f} |")
    lines.extend(["", "## Gate Details"])
    for d, v in details.items():
        lines.append(f"- {d}: " + ", ".join(f"{k}={val:.4f}" for k, val in v.items()))
    lines.extend(["", "Answers Q1-Q5 are reflected in the tables and gate details above. Mandatory stop: Phase XXIII-B was not started."])
    (OUT / "XXIII_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_audit = {"requested_model": "DINOv2 ViT-L/14", "torch_hub_entry": "facebookresearch/dinov2:dinov2_vitl14"}
    try:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True).to(device).eval()
        model_audit.update({"loaded": True, "class": m.__class__.__name__, "has_forward_features": hasattr(m, "forward_features")})
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        model_audit.update({"loaded": False, "error": repr(e)})
        (OUT / "MODEL_AUDIT.json").write_text(json.dumps(model_audit, indent=2), encoding="utf-8")
        outcome = "XXIII_A_IMPLEMENTATION_FAILURE"
        (OUT / "OUTCOME.json").write_text(json.dumps({"phase": "XXIII-A", "outcome": outcome, "target_accessed": False, "trained": False}, indent=2), encoding="utf-8")
        print(json.dumps({"outcome": outcome, "reason": "DINOv2 ViT-L/14 unavailable"}))
        return
    all_image, all_pixel, all_comp, all_fusion, all_control, all_geom, all_cat, splits = [], [], [], [], [], [], [], []
    for d in ["D1", "D2"]:
        for f in [0, 1]:
            res = run_fold(d, f, device)
            all_image.extend(res["image"]); all_pixel.extend(res["pixel"]); all_comp.extend(res["complement"])
            all_fusion.extend(res["fusion"]); all_control.extend(res["control"]); all_geom.extend(res["geometry"]); all_cat.extend(res["percat"]); splits.append(res["split"])
            write_csv(OUT / "IMAGE_PROBE_METRICS.running.csv", all_image)
            write_csv(OUT / "PIXEL_DISTANCE_METRICS.running.csv", all_pixel)
    outcome, details = decide(all_image, all_pixel, all_fusion, all_control)
    model_audit.update({"device": device, "runtime_sec_total": time.time() - start, "splits": splits})
    (OUT / "MODEL_AUDIT.json").write_text(json.dumps(model_audit, indent=2), encoding="utf-8")
    write_csv(OUT / "IMAGE_PROBE_METRICS.csv", all_image)
    write_csv(OUT / "PIXEL_DISTANCE_METRICS.csv", all_pixel)
    write_csv(OUT / "COMPLEMENTARITY.csv", all_comp)
    write_csv(OUT / "FUSION_METRICS.csv", all_fusion)
    write_csv(OUT / "CONTROL_METRICS.csv", all_control)
    write_csv(OUT / "REPRESENTATION_GEOMETRY.csv", all_geom)
    write_csv(OUT / "PER_CATEGORY.csv", all_cat)
    write_report(outcome, details, all_image, all_pixel, all_fusion)
    payload = {"phase": "XXIII-A", "outcome": outcome, "target_accessed": False, "trained_mrad": False, "visual_models_frozen": True, "phase_xxiii_b_started": False, "runtime_sec": time.time() - start, "gate_details": details, "next": "STOP" if outcome != "PASS_TO_PHASE_XXIII_B" else "eligible for XXIII-B; not started"}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXIII-A\n")
        fp.write("- Command: python experiments/phase_xxiii_carrier_audit.py\n")
        fp.write("- Scope: D1 fold0/fold1 and D2 fold0/fold1 source-only frozen CLIP vs DINOv2 carrier audit; no target; no MRAD training.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXIII_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
