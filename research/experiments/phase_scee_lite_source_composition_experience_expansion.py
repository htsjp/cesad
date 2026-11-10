import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cmab_compact_multi_anomaly_basis import load_mvtec_groups, load_visa_groups
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_defect_query_transfer_pilot import light_fold
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC, balanced_category_batch
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_median_spatial_consensus_generalization import HOLDOUTS, MetaDataset
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np, torch_resize
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_sgpr import evaluate_maps, fixed_quadrant_boxes, stitch_quadrant74
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, logits_from_z
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import ExternalDataset, TARGETS
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, LR, SIDE, norm_np
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_scee_lite/formal")
SYN_OUT = OUT / "synthetic_preview"
VARIANTS = ("E0_OriginalSource", "E1_RandomTransplant", "E2_CoverageBiasedTransplant")
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "DTD-Synthetic"),
)
HARD_PAIRS = (("Full MVTec", "VisA"), ("Full VisA", "MPDD"))
STABILITY_PAIR = ("Full MVTec", "BTAD")
EPOCHS = 10
MAX_SYN_PER_SOURCE = 240
PATCH_SAMPLE_SYN = 512


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


def finite_mean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def safe_auc_np(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    n = len(sorted_s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    pos = y.astype(bool)
    n_pos = float(pos.sum())
    n_neg = float(y.size - pos.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1.0) / 2.0) / max(n_pos * n_neg, 1.0) * 100.0)


def load_runtime(device):
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    model = load_components(str(CHECKPOINT), device)[0]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return preprocess, model


def component_rows(mask, source, dataset, category, image_id, domain):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return []
    labels, ncomp = ndimage.label(m, structure=np.ones((3, 3), dtype=np.int8))
    h, w = m.shape
    rows = []
    for rid in range(1, ncomp + 1):
        comp = labels == rid
        area = int(comp.sum())
        if area <= 0:
            continue
        ys, xs = np.where(comp)
        bh, bw = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
        boundary = comp & (~ndimage.binary_erosion(comp, structure=np.ones((3, 3), dtype=bool), border_value=0))
        t = torch.tensor(comp.astype(np.float32)).view(1, 1, h, w)
        patch = F.interpolate(t, size=(SIDE, SIDE), mode="nearest").view(SIDE, SIDE).numpy() > 0.5
        rows.append({
            "domain": domain,
            "source": source,
            "dataset": dataset,
            "category": category,
            "image_id": image_id,
            "component_id": rid,
            "area_ratio": float(area / max(h * w, 1)),
            "thinness": float(boundary.sum() / max(area, 1)),
            "boundary_to_area_ratio": float(boundary.sum() / max(area, 1)),
            "aspect_ratio": float(max(bw, bh) / max(min(bw, bh), 1)),
            "component_count_image": int(ncomp),
            "sparsity": float(ncomp / max(area, 1)),
            "patch_area": int(patch.sum()),
            "patch_ratio": float(patch.mean()),
            "bbox_w": bw,
            "bbox_h": bh,
        })
    return rows


def source_dataset(source, smoke):
    if source == "Full VisA":
        _name, dataset, indices, _cls, patch, _y, _cat = load_full_visa_source_lazy(smoke)
        return dataset, indices, patch
    _name, dataset, indices, patch = load_full_mvtec_source(smoke)
    return dataset, indices, patch


def resolve_dataset_path(dataset, rel):
    if not rel:
        return ""
    p = Path(str(rel))
    if p.is_absolute():
        return str(p)
    return str(Path(dataset.root) / p)


def mask_from_meta(dataset, rec):
    mask_path = str(rec.get("mask_path", ""))
    if not mask_path:
        return None
    path = resolve_dataset_path(dataset, mask_path)
    if not Path(path).exists():
        return None
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def target_meta_from_ref(dataset, kind, ref, pos, target):
    if kind == "source_dataset":
        rec = dataset.data_all[int(ref)]
    elif hasattr(dataset, "data_all"):
        rec = dataset.data_all[int(pos)]
    else:
        return None, target, 0
    cat = str(rec.get("category", rec.get("cls_name", target)))
    label = int(rec.get("label", rec.get("anomaly", 0)))
    return rec, cat, label


def mask_from_target_meta(dataset, rec):
    if rec is None:
        return None
    direct = mask_from_meta(dataset, rec)
    if direct is not None:
        return direct
    if hasattr(dataset, "resolve_mask"):
        resolved = dataset.resolve_mask(rec)
        if resolved is not None and Path(resolved).exists():
            return (np.asarray(Image.open(resolved).convert("L")) > 0).astype(np.uint8)
    img_rel = Path(str(rec.get("img_path", "")))
    if int(rec.get("label", rec.get("anomaly", 0))) and len(img_rel.parts) >= 3 and "test" in img_rel.parts:
        parts = list(img_rel.parts)
        ti = parts.index("test")
        parts[ti] = "ground_truth"
        parts[-1] = f"{Path(parts[-1]).stem}_mask.png"
        cand = Path(dataset.root).joinpath(*parts)
        if cand.exists():
            return (np.asarray(Image.open(cand).convert("L")) > 0).astype(np.uint8)
    return None


def collect_source_morph(source, smoke):
    dataset, indices, _patch = source_dataset(source, smoke)
    rows = []
    for pos, idx in enumerate(indices):
        rec = dataset.data_all[int(idx)]
        if int(rec.get("anomaly", 0)) != 1:
            continue
        mask = mask_from_meta(dataset, rec)
        if mask is None:
            continue
        rows.extend(component_rows(mask, source, source.replace("Full ", ""), str(rec.get("cls_name", source)), pos, "source"))
    return rows


def light_target_records(target, smoke):
    if target == "VisA":
        _name, dataset, train_indices, held_indices = light_fold("D2", 0)
        refs = list(train_indices) + list(held_indices)
        if smoke:
            refs = refs[: min(96, len(refs))]
        return dataset, refs
    if target == "MVTec":
        _name, dataset, indices, _patch = load_full_mvtec_source(smoke)
        return dataset, indices
    if target in TARGETS:
        dataset = ExternalDataset(TARGETS[target], target, mode="test")
        refs = list(range(len(dataset)))
        if smoke:
            refs = refs[: min(96, len(refs))]
        return dataset, refs
    if target in HOLDOUTS:
        root, _domain, _kind = HOLDOUTS[target]
        dataset = MetaDataset(root, mode="test")
        refs = list(range(len(dataset)))
        if smoke:
            refs = refs[: min(96, len(refs))]
        return dataset, refs
    return None, []


def collect_target_morph(source, target, _device, smoke):
    dataset, refs = light_target_records(target, smoke)
    if dataset is None:
        return []
    rows = []
    for pos, ref in enumerate(refs):
        rec = dataset.data_all[int(ref)]
        cat = str(rec.get("category", rec.get("cls_name", target)))
        label = int(rec.get("label", rec.get("anomaly", 0)))
        if int(label) != 1:
            continue
        mask = mask_from_target_meta(dataset, rec)
        if mask is None:
            item = dataset[int(ref)]
            mask = mask_np(item)
        rows.extend(component_rows(mask, source, target, cat, pos, "target"))
    return rows


def summarize_components(rows, prefix):
    out = {}
    for key in ("area_ratio", "thinness", "aspect_ratio", "component_count_image", "boundary_to_area_ratio", "patch_area", "patch_ratio"):
        vals = [fnum(r, key) for r in rows]
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        for stat_name, fun in (("mean", np.mean), ("median", np.median), ("p05", lambda x: np.percentile(x, 5)), ("p95", lambda x: np.percentile(x, 95))):
            out[f"{prefix}_{key}_{stat_name}"] = float(fun(arr)) if arr.size else float("nan")
    out[f"{prefix}_components"] = int(len(rows))
    return out


def coverage_score(source_rows, target_rows):
    metrics = ("area_ratio", "thinness", "aspect_ratio", "component_count_image", "boundary_to_area_ratio", "patch_area")
    src = {m: np.asarray([fnum(r, m) for r in source_rows], dtype=np.float64) for m in metrics}
    score_parts = []
    tail_parts = []
    detail = {}
    for m in metrics:
        s = src[m][np.isfinite(src[m])]
        t = np.asarray([fnum(r, m) for r in target_rows], dtype=np.float64)
        t = t[np.isfinite(t)]
        if s.size < 4 or t.size == 0:
            detail[f"{m}_coverage"] = float("nan")
            detail[f"{m}_target_tail_fraction"] = float("nan")
            continue
        lo, hi = np.percentile(s, [5, 95])
        cov = float(np.mean((t >= lo) & (t <= hi)))
        tail = float(1.0 - cov)
        detail[f"{m}_source_p05"] = float(lo)
        detail[f"{m}_source_p95"] = float(hi)
        detail[f"{m}_target_median"] = float(np.median(t))
        detail[f"{m}_coverage"] = cov
        detail[f"{m}_target_tail_fraction"] = tail
        score_parts.append(cov)
        tail_parts.append(tail)
    detail["coverage_score"] = finite_mean(score_parts)
    detail["tail_fraction_mean"] = finite_mean(tail_parts)
    return detail


def baseline_pap(source, target):
    paths = [
        Path("results/phase_paper_ready_final_validation/formal/INDUSTRIAL_MAIN_TABLE.csv"),
        Path("results/phase_final_cross_dataset_benchmark/formal/RAW_METRICS.csv"),
        Path("results/phase_parameter_free_local_zoom_pyramid_fullres/formal/RAW_METRICS.csv"),
    ]
    variants = ("Ours", "M2_Ours_HardNormal_Median", "E2_GlobalLocalZoomPyramid", "E0_FixedLRP_Baseline")
    for path in paths:
        if not path.exists():
            continue
        for r in read_csv(path):
            if r.get("source") == source and r.get("dataset") == target and r.get("variant") in variants:
                for key in ("M_FR_P_AP", "P-AP", "P_AP"):
                    v = fnum(r, key)
                    if np.isfinite(v):
                        return v
    return float("nan")


def gate1_coverage(device, smoke):
    src_cache = {}
    all_components, summary_rows = [], []
    for source, target in PILOT_PAIRS:
        if source not in src_cache:
            src_cache[source] = collect_source_morph(source, smoke)
        source_rows = src_cache[source]
        target_rows = collect_target_morph(source, target, device, smoke)
        all_components.extend(source_rows)
        all_components.extend(target_rows)
        row = {"source": source, "target": target}
        row.update(summarize_components(source_rows, "source"))
        row.update(summarize_components(target_rows, "target"))
        row.update(coverage_score(source_rows, target_rows))
        row["baseline_P_AP"] = baseline_pap(source, target)
        summary_rows.append(row)
    cov = np.asarray([fnum(r, "coverage_score") for r in summary_rows], dtype=np.float64)
    pap = np.asarray([fnum(r, "baseline_P_AP") for r in summary_rows], dtype=np.float64)
    ok = np.isfinite(cov) & np.isfinite(pap)
    corr = float(stats.spearmanr(cov[ok], pap[ok]).statistic) if ok.sum() >= 3 else float("nan")
    hard_tail = [r for r in summary_rows if (r["source"], r["target"]) in HARD_PAIRS and fnum(r, "tail_fraction_mean") >= 0.25]
    supported = bool((np.isfinite(corr) and corr >= 0.25 and len(hard_tail) >= 1) or (len(hard_tail) == len(HARD_PAIRS)))
    audit = {
        "gate": "Gate1_MorphologySupportCoverage",
        "spearman_coverage_vs_baseline_P_AP": corr,
        "hard_pair_tail_gap_count": len(hard_tail),
        "supported": supported,
    }
    return supported, audit, all_components, summary_rows


def choose_component(mask, biased, rng):
    labels, ncomp = ndimage.label(mask > 0, structure=np.ones((3, 3), dtype=np.int8))
    comps = []
    h, w = mask.shape
    for rid in range(1, ncomp + 1):
        comp = labels == rid
        if not np.any(comp):
            continue
        rows = component_rows(comp, "", "", "", 0, "")
        if not rows:
            continue
        r = rows[0]
        score = (1.0 / max(fnum(r, "patch_area"), 1.0)) + fnum(r, "thinness") + min(fnum(r, "aspect_ratio"), 10.0) * 0.05
        comps.append((score, comp))
    if not comps:
        return None
    if biased:
        comps.sort(key=lambda x: x[0], reverse=True)
        top = comps[: max(1, min(5, len(comps)))]
        return top[int(rng.integers(0, len(top)))][1]
    return comps[int(rng.integers(0, len(comps)))][1]


def paste_component(donor_img, donor_mask, recv_img, rng):
    donor = np.asarray(donor_img.convert("RGB"))
    recv = np.asarray(recv_img.convert("RGB")).copy()
    ys, xs = np.where(donor_mask > 0)
    if ys.size == 0:
        return recv_img, np.zeros(recv.shape[:2], dtype=np.uint8)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    patch = Image.fromarray(donor[y1:y2, x1:x2])
    mpatch = Image.fromarray((donor_mask[y1:y2, x1:x2].astype(np.uint8) * 255))
    rw, rh = recv_img.size
    scale = float(rng.uniform(0.85, 1.15))
    nw = max(4, min(rw, int(round(patch.size[0] * scale))))
    nh = max(4, min(rh, int(round(patch.size[1] * scale))))
    patch = patch.resize((nw, nh), Image.BILINEAR)
    mpatch = mpatch.resize((nw, nh), Image.NEAREST)
    angle = float(rng.uniform(-10, 10))
    patch = patch.rotate(angle, resample=Image.BILINEAR, expand=True)
    mpatch = mpatch.rotate(angle, resample=Image.NEAREST, expand=True)
    nw, nh = patch.size
    if nw >= rw or nh >= rh:
        nw, nh = min(nw, rw), min(nh, rh)
        patch = patch.resize((nw, nh), Image.BILINEAR)
        mpatch = mpatch.resize((nw, nh), Image.NEAREST)
    px = int(rng.integers(0, max(rw - nw + 1, 1)))
    py = int(rng.integers(0, max(rh - nh + 1, 1)))
    alpha = mpatch.filter(ImageFilter.GaussianBlur(radius=1.0))
    out = recv_img.copy()
    out.paste(patch, (px, py), alpha)
    full_mask = np.zeros((rh, rw), dtype=np.uint8)
    full_mask[py:py + nh, px:px + nw] = (np.asarray(mpatch) > 0).astype(np.uint8)
    return out, full_mask


def collect_image_records(source, smoke):
    dataset, indices, _patch = source_dataset(source, smoke)
    normal, anomaly = [], []
    for pos, idx in enumerate(indices):
        item = dataset.data_all[int(idx)]
        cat = str(item.get("cls_name", source))
        path = resolve_dataset_path(dataset, item.get("img_path", ""))
        lab = int(item.get("anomaly", 0))
        if not path:
            continue
        rec = {"pos": pos, "idx": int(idx), "category": cat, "path": path, "label": lab, "dataset": dataset, "meta": item}
        if lab == 1:
            anomaly.append(rec)
        else:
            normal.append(rec)
    return dataset, indices, normal, anomaly


def make_synthetic_samples(source, mode, smoke):
    rng = np.random.default_rng(SEED + (17 if mode == "biased" else 3))
    random.seed(SEED)
    _dataset, _indices, normals, anomalies = collect_image_records(source, smoke)
    if smoke:
        max_n = min(24, MAX_SYN_PER_SOURCE)
    else:
        max_n = MAX_SYN_PER_SOURCE
    chosen_anom = list(anomalies)
    rng.shuffle(chosen_anom)
    rows, samples = [], []
    SYN_OUT.mkdir(parents=True, exist_ok=True)
    for count, donor in enumerate(chosen_anom[:max_n]):
        donor_img = Image.open(donor["path"]).convert("RGB")
        donor_mask = mask_from_meta(donor["dataset"], donor["meta"])
        if donor_mask is None:
            donor_img.close()
            continue
        comp = choose_component(donor_mask, mode == "biased", rng)
        if comp is None:
            donor_img.close()
            continue
        candidates = [r for r in normals if r["category"] != donor["category"]] or normals
        recv = candidates[int(rng.integers(0, len(candidates)))]
        recv_img = Image.open(recv["path"]).convert("RGB")
        synth_img, synth_mask = paste_component(donor_img, comp.astype(np.uint8), recv_img, rng)
        if count < (8 if not smoke else 4):
            img_path = SYN_OUT / f"{source.replace(' ', '_')}_{mode}_{count:04d}.png"
            mask_path = SYN_OUT / f"{source.replace(' ', '_')}_{mode}_{count:04d}_mask.png"
            synth_img.save(img_path)
            Image.fromarray((synth_mask * 255).astype(np.uint8)).save(mask_path)
        else:
            img_path = ""
            mask_path = ""
        samples.append({"image": synth_img, "mask": synth_mask, "category": recv["category"], "donor_category": donor["category"], "receiver_category": recv["category"]})
        rows.append({
            "source": source,
            "variant": "E2_CoverageBiasedTransplant" if mode == "biased" else "E1_RandomTransplant",
            "sample_index": len(samples) - 1,
            "donor_category": donor["category"],
            "receiver_category": recv["category"],
            "cross_category": donor["category"] != recv["category"],
            "mask_area_ratio": float(np.mean(synth_mask > 0)),
            "preview_image": str(img_path),
            "preview_mask": str(mask_path),
        })
        donor_img.close()
        recv_img.close()
    return samples, rows


@torch.no_grad()
def encode_synth_sample(model_info, clip_model, preprocess, sample, device):
    batch = preprocess(sample["image"]).unsqueeze(0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patch = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)[0]
    if model_info["use_align"]:
        z = aligned_grid(patch, model_info["q_pc1"], model_info["align_model"], device).detach().cpu().numpy().reshape(SIDE * SIDE, -1)
    else:
        z = norm_np(np.asarray(patch, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    t = torch.tensor(sample["mask"].astype(np.float32)).view(1, 1, *sample["mask"].shape)
    y = (F.interpolate(t, size=(SIDE, SIDE), mode="nearest").view(-1).numpy() > 0.5).astype(np.float32)
    return z.astype(np.float32), y


def score_np(model_info, z, device, v=None):
    m_t = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v_t = torch.tensor(model_info["v1"] if v is None else v, dtype=torch.float32, device=device)
    zt = torch.tensor(z, dtype=torch.float32, device=device)
    return contrast_logits(zt, m_t, model_info["alpha"], v_t).detach().cpu().numpy().astype(np.float32)


def unpack_groups(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def synthetic_validity(source, model_info, clip_model, preprocess, samples, mode, device, smoke):
    rows = []
    real_groups = unpack_groups(load_visa_groups(model_info, device, smoke) if source == "Full VisA" else load_mvtec_groups(smoke))
    real_anom = np.concatenate([g["x"][g["y"] > 0.5] for g in real_groups.values() if np.any(g["y"] > 0.5)], axis=0)
    real_norm = np.concatenate([g["x"][g["y"] < 0.5] for g in real_groups.values() if np.any(g["y"] < 0.5)], axis=0)
    rng = np.random.default_rng(SEED)
    if real_anom.shape[0] > 2048:
        real_anom = real_anom[rng.choice(real_anom.shape[0], 2048, replace=False)]
    if real_norm.shape[0] > 2048:
        real_norm = real_norm[rng.choice(real_norm.shape[0], 2048, replace=False)]
    ra = real_anom / np.maximum(np.linalg.norm(real_anom, axis=1, keepdims=True), 1e-12)
    rn = real_norm / np.maximum(np.linalg.norm(real_norm, axis=1, keepdims=True), 1e-12)
    feature_pack = []
    for i, sample in enumerate(samples):
        z, y = encode_synth_sample(model_info, clip_model, preprocess, sample, device)
        s = score_np(model_info, z, device)
        pos = y > 0.5
        neg = ~pos
        if not pos.any() or not neg.any():
            continue
        zpos = z[pos]
        zpos = zpos / np.maximum(np.linalg.norm(zpos, axis=1, keepdims=True), 1e-12)
        sim_a = np.max(zpos @ ra.T, axis=1)
        sim_n = np.max(zpos @ rn.T, axis=1)
        mask2 = y.reshape(SIDE, SIDE).astype(bool)
        boundary = mask2 & (~ndimage.binary_erosion(mask2, structure=np.ones((3, 3), dtype=bool), border_value=0))
        interior = mask2 & ~boundary
        sg = s.reshape(SIDE, SIDE)
        row = {
            "source": source,
            "variant": "E2_CoverageBiasedTransplant" if mode == "biased" else "E1_RandomTransplant",
            "sample_index": i,
            "anomaly_score_mean": float(s[pos].mean()),
            "recipient_background_score_mean": float(s[neg].mean()),
            "score_region_minus_background": float(s[pos].mean() - s[neg].mean()),
            "nn_real_anomaly_cos_mean": float(sim_a.mean()),
            "nn_real_normal_cos_mean": float(sim_n.mean()),
            "anomaly_closer_than_normal_rate": float(np.mean(sim_a > sim_n)),
            "boundary_score_mean": float(sg[boundary].mean()) if boundary.any() else float("nan"),
            "interior_score_mean": float(sg[interior].mean()) if interior.any() else float("nan"),
            "boundary_minus_interior": float(sg[boundary].mean() - sg[interior].mean()) if boundary.any() and interior.any() else float("nan"),
            "paste_edge_shortcut_flag": bool(boundary.any() and interior.any() and (sg[boundary].mean() - sg[interior].mean()) > 0.25),
            "positive_patches": int(pos.sum()),
        }
        rows.append(row)
        feature_pack.append((sample["category"], z, y, mode))
    return rows, feature_pack


def append_synth_to_groups(base_groups, feature_pack, variant):
    groups = {c: {"x": g["x"].copy(), "y": g["y"].copy(), "origin": g["origin"].copy()} for c, g in base_groups.items()}
    rng = np.random.default_rng(SEED)
    for cat, z, y, mode in feature_pack:
        if cat not in groups:
            cat = sorted(groups)[int(rng.integers(0, len(groups)))]
        pos = np.flatnonzero(y > 0.5)
        neg = np.flatnonzero(y < 0.5)
        if pos.size == 0 or neg.size == 0:
            continue
        p_take = min(PATCH_SAMPLE_SYN // 2, pos.size)
        n_take = min(PATCH_SAMPLE_SYN // 2, neg.size)
        ids = np.concatenate([
            rng.choice(pos, p_take, replace=pos.size < p_take),
            rng.choice(neg, n_take, replace=neg.size < n_take),
        ])
        yy = y[ids].astype(np.float32)
        xx = z[ids].astype(np.float32)
        oo = np.asarray([f"synthetic_{mode}"] * len(ids), dtype=object)
        groups[cat]["x"] = np.concatenate([groups[cat]["x"], xx], axis=0)
        groups[cat]["y"] = np.concatenate([groups[cat]["y"], yy], axis=0)
        groups[cat]["origin"] = np.concatenate([groups[cat]["origin"], oo], axis=0)
    return groups


def train_contrast(source, variant, groups, model_info, device, smoke):
    if variant == "E0_OriginalSource":
        return np.asarray(model_info["v1"], dtype=np.float32), [], True
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{source.replace(' ', '_')}_{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["curves"], True
    print(f"[SCEE] train {source} {variant}", flush=True)
    raw_v = torch.nn.Parameter(torch.tensor(model_info["v1"], dtype=torch.float32, device=device))
    m = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([raw_v], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, int(math.ceil(total / BATCH_SIZE)))
    if smoke:
        steps = min(steps, 2)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        synth_frac = []
        for _step in range(steps):
            xb, yb, origins, _cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(raw_v, dim=-1)
            logits = contrast_logits(z, m, float(model_info["alpha"]), v)
            task, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(math.ceil(float(neg_idx.numel()) * TOP_NEG_FRAC)))
            selected = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
            hard = F.softplus(logits[selected]).mean()
            loss = task + hard
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task.detach().cpu()), float(hard.detach().cpu()), float(bce.cpu()), float(dice.cpu())))
            synth_frac.append(float(np.mean(np.char.startswith(origins.astype(str), "synthetic_"))))
        arr = np.asarray(vals, dtype=np.float64)
        curves.append({
            "source": source,
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "state_loss": float(arr[:, 1].mean()),
            "ntr_loss": float(arr[:, 2].mean()),
            "bce": float(arr[:, 3].mean()),
            "dice": float(arr[:, 4].mean()),
            "synthetic_sampling_ratio": finite_mean(synth_frac),
        })
    v = F.normalize(raw_v, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "curves": curves}, ckpt)
    return v, curves, False


@torch.no_grad()
def score_feature_grid(model_info, patch_np, device, v):
    m_t = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    if model_info["use_align"]:
        z = aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).unsqueeze(0)
        return logits_from_z(z, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    z = norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    zt = torch.tensor(z, dtype=torch.float32, device=device)
    return contrast_logits(zt, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


@torch.no_grad()
def local_zoom_map(model_info, clip_model, preprocess, img, device, v):
    boxes = fixed_quadrant_boxes(*img.size)
    batch = torch.stack([preprocess(img.crop(tuple(b))) for b in boxes], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patches = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)
    scores = [score_feature_grid(model_info, patches[i], device, v) for i in range(4)]
    return stitch_quadrant74(scores)


@torch.no_grad()
def scorer_z_grid(model_info, patch_np, device):
    if model_info["use_align"]:
        return aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).detach()
    z = norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    return torch.tensor(z, dtype=torch.float32, device=device).reshape(SIDE, SIDE, -1)


@torch.no_grad()
def score_z_grid(model_info, z_grid, device, v):
    m_t = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    if model_info["use_align"]:
        return logits_from_z(z_grid.unsqueeze(0), m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    zt = z_grid.reshape(-1, z_grid.shape[-1])
    return contrast_logits(zt, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


@torch.no_grad()
def crop_z_grids(model_info, clip_model, preprocess, img, device):
    boxes = fixed_quadrant_boxes(*img.size)
    batch = torch.stack([preprocess(img.crop(tuple(b))) for b in boxes], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patches = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)
    return [scorer_z_grid(model_info, patches[i], device) for i in range(4)]


def evaluate_variant_maps(model_infos, vectors_by_source, clip_model, preprocess, pairs, device, smoke):
    raw, cats = [], []
    for source, target in pairs:
        info = model_infos[source]
        cells, _ = protocol_cells(source, device, smoke)
        match = [c for c in cells if c[1] == target][0]
        _cid, _target, dataset, refs, patch, kind = match
        if smoke:
            refs = refs[: min(24, len(refs))]
            patch = patch[: min(24, len(patch))]
        maps = {v: [] for v in VARIANTS}
        metas = []
        for pos, ref in enumerate(refs):
            if pos % 25 == 0:
                print(f"[SCEE] eval {source}->{target} image {pos}/{len(refs)}", flush=True)
            item = item_at(dataset, kind, ref, pos)
            cat, label, path = item_meta(item, kind, ref, target)
            img = Image.open(path).convert("RGB")
            global_z = scorer_z_grid(info, patch[pos], device)
            local_z = crop_z_grids(info, clip_model, preprocess, img, device)
            for variant in VARIANTS:
                v = vectors_by_source[source][variant]
                g37 = score_z_grid(info, global_z, device, v)
                l74 = stitch_quadrant74([score_z_grid(info, z, device, v) for z in local_z])
                g74 = torch_resize(g37, (SIDE * 2, SIDE * 2), align_corners=True)
                maps[variant].append((0.5 * g74 + 0.5 * l74).astype(np.float32))
            metas.append({"category": cat, "label": int(label), "mask": mask_np(item), "img_path": path})
            img.close()
        rr, cc = evaluate_maps(source, target, refs, dataset, kind, maps, metas, variants=VARIANTS)
        raw.extend(rr)
        cats.extend(cc)
        write_csv(OUT / "MICRO_TRAIN_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", cats)
    return raw, cats


def delta_rows(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in sorted({(r["source"], r["dataset"]) for r in raw}):
        base = by.get((source, target, "E0_OriginalSource"))
        for variant in VARIANTS[1:]:
            cur = by.get((source, target, variant))
            if base and cur:
                rows.append({
                    "source": source,
                    "dataset": target,
                    "variant": variant,
                    "Delta_M_FR_P_AUROC": fnum(cur, "M_FR_P_AUROC") - fnum(base, "M_FR_P_AUROC"),
                    "Delta_M_FR_P_AP": fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"),
                    "Delta_M_FR_P_F1max": fnum(cur, "M_FR_P_F1max") - fnum(base, "M_FR_P_F1max"),
                    "Delta_M_FR_PRO": fnum(cur, "M_FR_PRO") - fnum(base, "M_FR_PRO"),
                })
    for variant in VARIANTS[1:]:
        vals = [fnum(r, "Delta_M_FR_P_AP") for r in rows if r["variant"] == variant]
        rows.append({"source": "SUMMARY", "dataset": "MICRO", "variant": variant, "mean_delta_M_FR_P_AP": finite_mean(vals), "median_delta_M_FR_P_AP": finite_median(vals), "improved": int(sum(v > 0 for v in vals if np.isfinite(v)))})
    return rows


def direction_geometry(model_infos, vectors_by_source):
    rows = []
    for source, variants in vectors_by_source.items():
        base = np.asarray(model_infos[source]["v1"], dtype=np.float32)
        for variant, v in variants.items():
            dot = float(np.dot(unit_np(base), unit_np(v)))
            rows.append({
                "source": source,
                "variant": variant,
                "cos_d_baseline": dot,
                "angle_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, dot))))),
                "state_direction_norm": float(np.linalg.norm(v)),
                "state_alpha": float(model_infos[source]["alpha"]),
            })
    return rows


def write_report(outcome, gate1, coverage_rows, synth_rows, validity_rows, raw, deltas, geom, runtime):
    lines = [
        "# Phase SCEE-Lite: Source-Composition Experience Expansion",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "This phase keeps CLIP, Two-State/NTR, and Fixed LRP fixed unless Gate 1/2 pass. Synthetic samples use only source images and source masks.",
        "",
        "## Gate 1 Coverage",
        "",
        f"- Spearman coverage vs baseline P-AP: `{gate1.get('spearman_coverage_vs_baseline_P_AP', float('nan')):.4f}`",
        f"- Hard-pair tail-gap count: `{gate1.get('hard_pair_tail_gap_count', 0)}`",
        f"- Gate 1 supported: `{gate1.get('supported')}`",
        "",
        "| Source | Target | Coverage | Tail Fraction | Baseline P-AP |",
        "|---|---|---:|---:|---:|",
    ]
    for r in coverage_rows:
        lines.append(f"| {r['source']} | {r['target']} | {fnum(r,'coverage_score'):.4f} | {fnum(r,'tail_fraction_mean'):.4f} | {fnum(r,'baseline_P_AP'):.4f} |")
    if synth_rows:
        lines += ["", "## Synthetic Validity", "", "| Source | Variant | samples | score region-bg | anomaly closer rate | edge shortcut rate |", "|---|---|---:|---:|---:|---:|"]
        for source in sorted({r["source"] for r in validity_rows}):
            for variant in sorted({r["variant"] for r in validity_rows if r["source"] == source}):
                part = [r for r in validity_rows if r["source"] == source and r["variant"] == variant]
                lines.append(f"| {source} | {variant} | {len(part)} | {finite_mean([fnum(r,'score_region_minus_background') for r in part]):+.4f} | {finite_mean([fnum(r,'anomaly_closer_than_normal_rate') for r in part]):.4f} | {finite_mean([float(r.get('paste_edge_shortcut_flag', False)) for r in part]):.4f} |")
    if raw:
        lines += ["", "## Micro Training Metrics", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |", "|---|---|---|---:|---:|---:|---:|"]
        for r in raw:
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
        lines += ["", "## Delta vs E0", "", "| Source | Target | Variant | Delta P-AP |", "|---|---|---|---:|"]
        for r in deltas:
            if r["source"] != "SUMMARY":
                lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} |")
        lines += ["", "## Direction Geometry", "", "| Source | Variant | cos(d_base,d) | angle |", "|---|---|---:|---:|"]
        for r in geom:
            lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'cos_d_baseline'):.4f} | {fnum(r,'angle_deg'):.4f} |")
    lines.append(f"\nRuntime: {runtime:.1f}s.")
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SCEE] device={device}", flush=True)
    preprocess, clip_model = load_runtime(device)
    gate_supported, gate_audit, components, coverage_rows = gate1_coverage(device, args.smoke)
    write_csv(OUT / "MORPHOLOGY_COMPONENTS.csv", components)
    write_csv(OUT / "MORPHOLOGY_COVERAGE.csv", coverage_rows)
    write_json(OUT / "GATE1.json", gate_audit)
    if not gate_supported:
        outcome = "SOURCE_MORPHOLOGY_COVERAGE_GAP_NOT_SUPPORTED"
        runtime = time.time() - start
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "gate1": gate_audit, "runtime_sec": runtime})
        write_report(outcome, gate_audit, coverage_rows, [], [], [], [], [], runtime)
        print(f"[SCEE] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)
        return

    infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    synth_rows, validity_rows, packs = [], [], {s: {"random": [], "biased": []} for s in infos}
    artifact = False
    for source, info in infos.items():
        for mode in ("random", "biased"):
            samples, rows = make_synthetic_samples(source, mode, args.smoke)
            synth_rows.extend(rows)
            valid, pack = synthetic_validity(source, info, clip_model, preprocess, samples, mode, device, args.smoke)
            validity_rows.extend(valid)
            packs[source][mode] = pack
            edge_rate = finite_mean([float(r["paste_edge_shortcut_flag"]) for r in valid])
            close_rate = finite_mean([fnum(r, "anomaly_closer_than_normal_rate") for r in valid])
            score_gain = finite_mean([fnum(r, "score_region_minus_background") for r in valid])
            if (np.isfinite(edge_rate) and edge_rate > 0.35) or (np.isfinite(close_rate) and close_rate < 0.50) or (np.isfinite(score_gain) and score_gain <= 0.0):
                artifact = True
    write_csv(OUT / "SYNTHETIC_SAMPLE_AUDIT.csv", synth_rows)
    write_csv(OUT / "SYNTHETIC_VALIDITY.csv", validity_rows)
    if artifact:
        outcome = "SYNTHETIC_COMPOSITION_HAS_ARTIFACT_SHORTCUT"
        runtime = time.time() - start
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "gate1": gate_audit, "runtime_sec": runtime})
        write_report(outcome, gate_audit, coverage_rows, synth_rows, validity_rows, [], [], [], runtime)
        print(f"[SCEE] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)
        return

    base_groups = {
        "Full VisA": unpack_groups(load_visa_groups(infos["Full VisA"], device, args.smoke)),
        "Full MVTec": unpack_groups(load_mvtec_groups(args.smoke)),
    }
    vectors, train_rows = {}, []
    for source in infos:
        vectors[source] = {}
        for variant in VARIANTS:
            if variant == "E0_OriginalSource":
                groups = base_groups[source]
            elif variant == "E1_RandomTransplant":
                groups = append_synth_to_groups(base_groups[source], packs[source]["random"], variant)
            else:
                groups = append_synth_to_groups(base_groups[source], packs[source]["biased"], variant)
            v, curves, _cached = train_contrast(source, variant, groups, infos[source], device, args.smoke)
            vectors[source][variant] = v
            train_rows.extend(curves)
    write_csv(OUT / "SOURCE_TRAINING.csv", train_rows)
    eval_pairs = list(HARD_PAIRS)
    raw, cats = evaluate_variant_maps(infos, vectors, clip_model, preprocess, eval_pairs, device, args.smoke)
    deltas = delta_rows(raw)
    e2_hard = [r for r in deltas if r["variant"] == "E2_CoverageBiasedTransplant" and r["source"] != "SUMMARY"]
    e2_vals = [fnum(r, "Delta_M_FR_P_AP") for r in e2_hard]
    geom = direction_geometry(infos, vectors)
    e2_pass = finite_mean(e2_vals) >= 1.0 and any(v >= 1.0 for v in e2_vals if np.isfinite(v)) and all(v >= -0.2 for v in e2_vals if np.isfinite(v))
    abnormal_rotation = any(r["variant"] == "E2_CoverageBiasedTransplant" and fnum(r, "cos_d_baseline") < 0.95 for r in geom)
    if e2_pass and not abnormal_rotation and not args.smoke:
        raw2, cats2 = evaluate_variant_maps(infos, vectors, clip_model, preprocess, [STABILITY_PAIR], device, args.smoke)
        raw.extend(raw2)
        cats.extend(cats2)
        deltas = delta_rows(raw)
        stab = [r for r in deltas if r["variant"] == "E2_CoverageBiasedTransplant" and (r["source"], r["dataset"]) == STABILITY_PAIR]
        outcome = "SCEE_LIGHTWEIGHT_VALIDATION_PASSED" if stab and fnum(stab[0], "Delta_M_FR_P_AP") >= -0.2 else "SCEE_HAS_WEAK_SIGNAL_BUT_NOT_ENOUGH_FOR_EXPANSION"
    elif e2_pass and not abnormal_rotation:
        outcome = "SCEE_HAS_WEAK_SIGNAL_BUT_NOT_ENOUGH_FOR_EXPANSION"
    else:
        outcome = "SOURCE_COMPOSITION_DOES_NOT_IMPROVE_TRANSFER"
    runtime = time.time() - start
    write_csv(OUT / "MICRO_TRAIN_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "MICRO_DELTA_SUMMARY.csv", deltas)
    write_csv(OUT / "STATE_GEOMETRY.csv", geom)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "gate1": gate_audit, "runtime_sec": runtime, "max_synthetic_per_source": MAX_SYN_PER_SOURCE})
    write_report(outcome, gate_audit, coverage_rows, synth_rows, validity_rows, raw, deltas, geom, runtime)
    print(f"[SCEE] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
