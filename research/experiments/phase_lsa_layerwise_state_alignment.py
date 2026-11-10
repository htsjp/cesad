import argparse
import csv
import hashlib
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
from PIL import Image
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    METRICS,
    PROTOCOLS,
    evaluate_maps,
    fnum,
    item_at,
    item_meta,
    mask_np,
    quadrant_crops,
    torch_resize,
)
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, logits_from_z
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np, patch_labels
from models.mlp import average_neighbor
from phase_xa.official_task_forward import visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_layerwise_state_alignment/formal")
LAYERS = (6, 12, 18)
ALL_LAYERS = (6, 12, 18, 24)
EPOCHS = 10
LR = 0.001
PATCH_CAP_PER_CLASS_CATEGORY = 512
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "DTD-Synthetic"),
)
VARIANT_LAYERS = {
    "E1_L18_SAF": (18,),
    "E2_L12_L18_SAF": (12, 18),
    "E3_L6_L12_L18_SAF": (6, 12, 18),
}
BASELINE = "E0_Core_NTR_FixedLRP"
OFFICIAL_REPO_URL = "https://github.com/7HHHHH/VisualAD"


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class SAFMLP(nn.Module):
    """VisualAD release SAF analogue: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, dim=1024, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)


def sha256_tensors(named_tensors):
    h = hashlib.sha256()
    for name, tensor in named_tensors:
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def clip_checksum(model):
    return sha256_tensors([(n, p) for n, p in model.visual.named_parameters()])


def state_checksum(info):
    vals = [
        ("m", torch.tensor(info["m"], dtype=torch.float32)),
        ("alpha", torch.tensor([float(info["alpha"])], dtype=torch.float32)),
        ("v1", torch.tensor(info["v1"], dtype=torch.float32)),
    ]
    return sha256_tensors(vals)


def snapshot_patch(outputs_pack, layer):
    outputs = outputs_pack["outputs"]
    if layer == 24:
        patch = average_neighbor(outputs[1][3]).float()
        return F.normalize(patch, dim=-1)
    snap = outputs_pack["snapshots"][f"true_block{layer}"]
    branch = snap[0] if isinstance(snap, list) else snap
    patch = branch.permute(1, 0, 2)[:, 1:, :].float()
    patch = average_neighbor(patch)
    return F.normalize(patch, dim=-1)


@torch.no_grad()
def image_layer_features(clip_model, image_batch):
    pack = visual_outputs(clip_model, image_batch, grad=False, snapshot_mode="true_clone")
    return {layer: snapshot_patch(pack, layer).detach() for layer in ALL_LAYERS}


def source_dataset(source, smoke):
    if source == "Full VisA":
        _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    else:
        _name, dataset, indices, patch = load_full_mvtec_source(False)
    if smoke:
        by_cat = defaultdict(lambda: {0: [], 1: []})
        for pos, idx in enumerate(indices):
            meta = dataset.data_all[int(idx)]
            by_cat[str(meta["cls_name"])][int(meta["anomaly"])].append(pos)
        keep = []
        for cat in sorted(by_cat):
            keep.extend(by_cat[cat][0][:2])
            keep.extend(by_cat[cat][1][:2])
        indices = [indices[i] for i in keep]
        patch = np.stack([np.asarray(patch[i], dtype=np.float32) for i in keep], axis=0)
    return dataset, list(indices), patch


def load_source_info(source, device):
    return load_visa_source(device) if source == "Full VisA" else load_mvtec_source()


def collect_source_pool(source_info, clip_model, device, smoke):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / f"{source_info['source'].replace(' ', '_')}_LSA_SOURCE_POOL_{suffix}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        groups = {}
        for cat in data["categories"].tolist():
            groups[cat] = {
                "y": data[f"{cat}_y"].astype(np.float32),
                "origin": data[f"{cat}_origin"].astype(str),
                "x": {layer: data[f"{cat}_L{layer}"].astype(np.float32) for layer in LAYERS},
            }
        return groups, data["audit"].tolist()

    dataset, indices, _patch = source_dataset(source_info["source"], smoke)
    by_cat = defaultdict(list)
    for pos, idx in enumerate(indices):
        by_cat[str(dataset.data_all[int(idx)]["cls_name"])].append(pos)
    rng = np.random.default_rng(SEED)
    groups, audit_rows = {}, []
    for cat, positions in sorted(by_cat.items()):
        parts = {layer: [] for layer in LAYERS}
        ys, origins = [], []
        used = defaultdict(int)
        for pos in positions:
            item = dataset[int(indices[pos])]
            y_grid = patch_labels(item).astype(np.int64)
            if int(item.get("anomaly", 0)) == 1:
                pools = [("anomaly_mask_out", 0, np.flatnonzero(y_grid == 0)), ("anomaly_mask", 1, np.flatnonzero(y_grid == 1))]
            else:
                pools = [("normal_image", 0, np.flatnonzero(y_grid == 0))]
            chosen = []
            for origin, cls, ids in pools:
                room = PATCH_CAP_PER_CLASS_CATEGORY - used[origin]
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                used[origin] += int(ids.size)
                chosen.append((origin, cls, ids))
            if not chosen:
                continue
            image = item["img"].unsqueeze(0).to(device)
            feats = image_layer_features(clip_model, image)
            for origin, cls, ids in chosen:
                for layer in LAYERS:
                    parts[layer].append(feats[layer][0, ids].detach().cpu().numpy().astype(np.float32))
                ys.extend([cls] * len(ids))
                origins.extend([origin] * len(ids))
        if len(set(ys)) < 2:
            continue
        groups[cat] = {
            "y": np.asarray(ys, dtype=np.float32),
            "origin": np.asarray(origins, dtype=str),
            "x": {layer: np.concatenate(parts[layer], axis=0).astype(np.float32) for layer in LAYERS},
        }
        audit_rows.append({
            "source": source_info["source"],
            "category": cat,
            "samples": int(len(ys)),
            "positive_samples": int(np.sum(np.asarray(ys) > 0.5)),
            "negative_samples": int(np.sum(np.asarray(ys) < 0.5)),
            "normal_image_negatives": int(np.sum(groups[cat]["origin"][groups[cat]["y"] < 0.5] == "normal_image")),
            "mask_out_negatives": int(np.sum(groups[cat]["origin"][groups[cat]["y"] < 0.5] == "anomaly_mask_out")),
        })
        print(f"[LSA] source pool {source_info['source']} {cat} samples={len(ys)}", flush=True)
    save = {"categories": np.asarray(list(groups.keys()), dtype=object), "audit": np.asarray(audit_rows, dtype=object)}
    for cat, data in groups.items():
        save[f"{cat}_y"] = data["y"]
        save[f"{cat}_origin"] = data["origin"]
        for layer in LAYERS:
            save[f"{cat}_L{layer}"] = data["x"][layer].astype(np.float16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **save)
    return groups, audit_rows


def state_logits(z, source_info, device):
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    v = torch.tensor(source_info["v1"], dtype=torch.float32, device=device)
    return contrast_logits(z.reshape(-1, z.shape[-1]), m, float(source_info["alpha"]), v).reshape(z.shape[:-1])


def train_layer_saf(source_info, groups, layer, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "checkpoints" / f"{source_info['source'].replace(' ', '_')}_SAF_L{layer}_{suffix}.pt"
    if ckpt.exists():
        saf = SAFMLP().to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        saf.load_state_dict(payload["model"], strict=True)
        saf.eval()
        return saf, payload.get("curves", []), True

    torch.manual_seed(SEED + layer)
    rng = np.random.default_rng(SEED + layer)
    saf = SAFMLP().to(device)
    opt = torch.optim.AdamW(saf.parameters(), lr=LR, weight_decay=0.01)
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, int(math.ceil(total / BATCH_SIZE)))
    if smoke:
        steps = min(steps, 3)
    curves = []
    cats = list(groups)
    per_cat = max(2, BATCH_SIZE // max(1, len(cats)))
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _step in range(steps):
            xs, ys = [], []
            for cat in cats:
                data = groups[cat]
                per_cls = max(1, per_cat // 2)
                for cls in (0, 1):
                    ids = np.flatnonzero(data["y"] == cls)
                    choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
                    xs.append(data["x"][layer][choice])
                    ys.append(data["y"][choice])
            x_np = np.concatenate(xs, axis=0).astype(np.float32)
            y_np = np.concatenate(ys, axis=0).astype(np.float32)
            order = rng.permutation(len(y_np))
            x = torch.tensor(x_np[order], dtype=torch.float32, device=device)
            y = torch.tensor(y_np[order], dtype=torch.float32, device=device)
            z = saf(x)
            logits = state_logits(z, source_info, device)
            loss, bce, dice = balanced_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                out_norm = z.norm(dim=-1)
                in_norm = x.norm(dim=-1)
                cos = F.cosine_similarity(F.normalize(x, dim=-1), z, dim=-1)
                margin_pos = logits[y > 0.5].mean() if torch.any(y > 0.5) else torch.tensor(float("nan"), device=device)
                margin_neg = logits[y < 0.5].mean() if torch.any(y < 0.5) else torch.tensor(float("nan"), device=device)
            vals.append((float(loss.detach().cpu()), float(bce.cpu()), float(dice.cpu()), float(in_norm.mean().cpu()), float(out_norm.mean().cpu()), float(cos.mean().cpu()), float(margin_pos.cpu()), float(margin_neg.cpu())))
        arr = np.asarray(vals, dtype=np.float64)
        row = {
            "source": source_info["source"],
            "layer": layer,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "bce": float(arr[:, 1].mean()),
            "dice": float(arr[:, 2].mean()),
            "input_norm_mean": float(arr[:, 3].mean()),
            "output_norm_mean": float(arr[:, 4].mean()),
            "cos_raw_aligned_mean": float(arr[:, 5].mean()),
            "positive_margin_mean": float(arr[:, 6].mean()),
            "negative_margin_mean": float(arr[:, 7].mean()),
        }
        curves.append(row)
        print(f"[LSA] train {source_info['source']} L{layer} epoch={epoch}/{EPOCHS} loss={row['loss']:.4f}", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": saf.state_dict(), "curves": curves}, ckpt)
    saf.eval()
    return saf, curves, False


@torch.no_grad()
def base_global_map(source_info, patch_np, device):
    if source_info["use_align"]:
        z = aligned_grid(patch_np, source_info["q_pc1"], source_info["align_model"], device).reshape(-1, 1024)
        return state_logits(z, source_info, device).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    z = torch.tensor(norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024)), dtype=torch.float32, device=device)
    return state_logits(z, source_info, device).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


@torch.no_grad()
def base_local_map(source_info, clip_model, preprocess, img_path, device):
    crops = quadrant_crops(img_path)
    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    pack = visual_outputs(clip_model, batch, grad=False)
    patch = average_neighbor(pack[1][3]).float()
    patch = F.normalize(patch, dim=-1).detach().cpu().numpy().astype(np.float32)
    maps = [base_global_map(source_info, patch[i], device) for i in range(4)]
    out = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    out[:SIDE, :SIDE] = maps[0]
    out[:SIDE, SIDE:] = maps[1]
    out[SIDE:, :SIDE] = maps[2]
    out[SIDE:, SIDE:] = maps[3]
    return out


@torch.no_grad()
def saf_score_map(saf, feat, source_info, device):
    b, n, d = feat.shape
    z = saf(feat.reshape(-1, d).to(device))
    return state_logits(z, source_info, device).reshape(b, SIDE, SIDE).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def lsa_maps_for_image(source_info, saf_by_layer, clip_model, preprocess, item, patch_np, img_path, device):
    image = item["img"].unsqueeze(0).to(device)
    feats = image_layer_features(clip_model, image)
    base37 = base_global_map(source_info, patch_np, device)
    local74 = base_local_map(source_info, clip_model, preprocess, img_path, device)
    base74 = 0.5 * torch_resize(base37, (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * local74
    layer_lrp = {}
    layer_global = {}
    for layer, saf in saf_by_layer.items():
        g37 = saf_score_map(saf, feats[layer], source_info, device)[0]
        layer_global[layer] = g37
    crops = quadrant_crops(img_path)
    crop_batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    crop_feats = image_layer_features(clip_model, crop_batch)
    for layer, saf in saf_by_layer.items():
        q = saf_score_map(saf, crop_feats[layer], source_info, device)
        stitched = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
        stitched[:SIDE, :SIDE] = q[0]
        stitched[:SIDE, SIDE:] = q[1]
        stitched[SIDE:, :SIDE] = q[2]
        stitched[SIDE:, SIDE:] = q[3]
        layer_lrp[layer] = 0.5 * torch_resize(layer_global[layer], (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * stitched
    out = {BASELINE: base74}
    for variant, layers in VARIANT_LAYERS.items():
        out[variant] = np.mean([base74] + [layer_lrp[l] for l in layers], axis=0).astype(np.float32)
    return out, layer_global


def resolve_path(item, dataset, ref, kind):
    candidates = []
    if isinstance(item, dict) and item.get("img_path"):
        candidates.append(Path(str(item["img_path"])))
    if kind != "source_dataset" and isinstance(ref, dict) and ref.get("img_path"):
        candidates.append(Path(str(ref["img_path"])))
    if hasattr(dataset, "root"):
        root = Path(getattr(dataset, "root"))
        candidates.extend([root / c for c in list(candidates) if not c.is_absolute()])
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(candidates)


def selected_eval(source, target, maps, metas, variants):
    old = tuple(lzp.VARIANTS)
    try:
        lzp.VARIANTS = tuple(variants)
        return evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old


def evaluate_pilot(source_info, saf_by_layer, clip_model, preprocess, device, smoke):
    suffix = "smoke" if smoke else "formal"
    raw, percat, corr_rows = [], [], []
    variants = (BASELINE,) + tuple(VARIANT_LAYERS)
    cells, _audits = protocol_cells(source_info["source"], device, smoke)
    for _cid, target, dataset, refs, patch, kind in cells:
        if (source_info["source"], target) not in PILOT_PAIRS:
            continue
        n = min(8, len(refs)) if smoke else len(refs)
        cache = OUT / f"{source_info['source'].replace(' ', '_')}_to_{target}_LSA_MAPS_{suffix}.npz"
        metas = []
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            maps = {v: [x.astype(np.float32) for x in data[v]] for v in variants}
            corr_rows.extend(data["corr_rows"].tolist())
            metas = data["metas"].tolist()
        else:
            maps = {v: [] for v in variants}
            for pos, ref in enumerate(refs[:n]):
                if pos % 25 == 0:
                    print(f"[LSA] eval {source_info['source']} -> {target} image={pos}/{n}", flush=True)
                item = item_at(dataset, kind, ref, pos)
                cat, label, _p = item_meta(item, kind, ref, target)
                path = resolve_path(item, dataset, ref, kind)
                m = {"category": cat, "label": label, "mask": mask_np(item), "img_path": path}
                metas.append(m)
                per_variant, layer_global = lsa_maps_for_image(source_info, saf_by_layer, clip_model, preprocess, item, patch[pos], path, device)
                for v in variants:
                    maps[v].append(per_variant[v])
                base_flat = maps[BASELINE][-1].reshape(-1)
                for layer in LAYERS:
                    layer74 = torch_resize(layer_global[layer], (SIDE * 2, SIDE * 2), align_corners=True)
                    lf = layer74.reshape(-1)
                    corr_rows.append({
                        "source": source_info["source"],
                        "dataset": target,
                        "image_index": pos,
                        "layer": layer,
                        "pearson_with_final_lrp": float(pearsonr(base_flat, lf).statistic),
                        "spearman_with_final_lrp": float(spearmanr(base_flat, lf).statistic),
                    })
            np.savez_compressed(cache, **{v: np.stack(maps[v]).astype(np.float16) for v in variants}, metas=np.asarray(metas, dtype=object), corr_rows=np.asarray(corr_rows, dtype=object))
        rr, cc = selected_eval(source_info["source"], target, maps, metas, variants)
        raw.extend(rr)
        percat.extend(cc)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.partial.csv", percat)
    return raw, percat, corr_rows


def delta_summary(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in PILOT_PAIRS:
        b = by.get((source, target, BASELINE))
        if not b:
            continue
        for variant in VARIANT_LAYERS:
            e = by.get((source, target, variant))
            if not e:
                continue
            row = {"source": source, "dataset": target, "variant": variant}
            for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
                key = f"M_FR_{met}"
                row[f"Delta_{key}"] = fnum(e, key) - fnum(b, key)
            rows.append(row)
    return rows


def old_multilayer_rows():
    rows = []
    for path, label in (
        (ROOT / "results" / "phase_multilayer_shared_contrast_scoring" / "formal" / "REPORT.md", "Raw Multi-layer shared scoring"),
        (ROOT / "results" / "phase_cross_layer_robust_state_learning" / "formal" / "REPORT.md", "Raw/robust cross-layer CLRS"),
    ):
        if path.exists():
            rows.append({"reference": label, "path": str(path), "available": True})
        else:
            rows.append({"reference": label, "path": str(path), "available": False})
    return rows


def decide(rows):
    best, best_mean = None, -1e9
    summary = []
    for variant in VARIANT_LAYERS:
        vals = np.asarray([fnum(r, "Delta_M_FR_P_AP") for r in rows if r["variant"] == variant], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        mean = float(vals.mean()) if vals.size else float("nan")
        rec = {"variant": variant, "mean_delta_p_ap": mean, "median_delta_p_ap": float(np.median(vals)) if vals.size else float("nan"), "improved": int(np.sum(vals > 0)), "non_drop_ge_neg_0p2": int(np.sum(vals >= -0.2)), "total": int(vals.size)}
        summary.append(rec)
        if np.isfinite(mean) and mean > best_mean:
            best, best_mean = variant, mean
    best_rec = next((s for s in summary if s["variant"] == best), None)
    if best_rec and best_rec["mean_delta_p_ap"] >= 1.0 and best_rec["improved"] >= 3:
        outcome = "LAYERWISE_STATE_ALIGNMENT_SIGNIFICANTLY_IMPROVES_TRANSFER"
    elif best_rec and best_rec["mean_delta_p_ap"] > 0 and best_rec["improved"] >= 2:
        outcome = "MULTILAYER_SIGNAL_EXISTS_BUT_ALIGNMENT_GAIN_IS_WEAK"
    else:
        outcome = "LAYERWISE_STATE_ALIGNMENT_HAS_NO_COMPETITIVE_VALUE"
    return outcome, {"best_variant": best, "best_mean_delta_p_ap": best_mean, "summary": summary}


def write_report(raw, delta_rows, train_rows, diag_rows, checksum_rows, outcome, details, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Phase LSA: Layer-wise State Alignment",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        f"Runtime: {runtime:.1f}s",
        "",
        f"Official SAF reference: [VisualAD GitHub]({OFFICIAL_REPO_URL}). The released CLIP path uses independent per-layer `FeatureTransformMLP`: Linear -> GELU -> Dropout -> Linear -> Dropout, no residual branch.",
        "",
        "Protocol: CLIP frozen, canonical states frozen, no VisualAD tokens, no SCA, no text, no memory, no DINO, no target statistics/tuning. SAF is trained source-only and evaluated with the same fixed 2x2 LRP observation path.",
        "",
        "## Pilot Results",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO | Delta P-AP |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for source, target in PILOT_PAIRS:
        b = by.get((source, target, BASELINE))
        for variant in (BASELINE,) + tuple(VARIANT_LAYERS):
            r = by.get((source, target, variant))
            if not r:
                continue
            delta = 0.0 if variant == BASELINE else fnum(r, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            lines.append(f"| {source} | {target} | {variant} | {fnum(r, 'M_FR_P_AUROC'):.4f} | {fnum(r, 'M_FR_P_AP'):.4f} | {fnum(r, 'M_FR_P_F1max'):.4f} | {fnum(r, 'M_FR_PRO'):.4f} | {delta:+.4f} |")
    lines += ["", "## Summary", "", "| Variant | Mean Delta P-AP | Median | Improved | Non-drop >= -0.2 |", "|---|---:|---:|---:|---:|"]
    for s in details["summary"]:
        lines.append(f"| {s['variant']} | {s['mean_delta_p_ap']:+.4f} | {s['median_delta_p_ap']:+.4f} | {s['improved']}/{s['total']} | {s['non_drop_ge_neg_0p2']}/{s['total']} |")
    lines += [
        "",
        "## Raw Multi-Layer Reference",
        "",
        "- `phase_multilayer_shared_contrast_scoring`: Raw multi-layer shared scoring was domain-dependent; VisA-source mean E1-B0 G was -3.2802.",
        "- `phase_cross_layer_robust_state_learning`: Cross-layer robust state learning E3 mean P-AP delta over 8 industrial transfers was -2.6133.",
        "",
        "## Integrity",
        "",
        f"- Frozen CLIP checksum unchanged: `{all(r.get('clip_unchanged') for r in checksum_rows)}`.",
        f"- Frozen Two-State checksum unchanged: `{all(r.get('state_unchanged') for r in checksum_rows)}`.",
        f"- SAF training rows: {len(train_rows)}; layer diagnostic rows: {len(diag_rows)}.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    clip_model, _prompt, _image_proj, _patch_proj, _prompt_proj = load_components(CHECKPOINT, device)
    clip_model.eval()
    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_before = clip_checksum(clip_model)
    raw_all, percat_all, train_all, diag_all, source_audit, corr_all, checksum_rows = [], [], [], [], [], [], []
    saf_by_source = {}
    for source in ("Full MVTec", "Full VisA"):
        info = load_source_info(source, device)
        state_before = state_checksum(info)
        groups, audit = collect_source_pool(info, clip_model, device, args.smoke)
        source_audit.extend(audit)
        saf_by_layer = {}
        for layer in LAYERS:
            saf, curves, _cached = train_layer_saf(info, groups, layer, device, args.smoke)
            saf_by_layer[layer] = saf
            train_all.extend(curves)
        saf_by_source[source] = saf_by_layer
        raw, percat, corr = evaluate_pilot(info, saf_by_layer, clip_model, preprocess, device, args.smoke)
        raw_all.extend(raw)
        percat_all.extend(percat)
        corr_all.extend(corr)
        checksum_rows.append({
            "source": source,
            "clip_checksum_before": clip_before,
            "clip_checksum_after": clip_checksum(clip_model),
            "clip_unchanged": clip_before == clip_checksum(clip_model),
            "state_checksum_before": state_before,
            "state_checksum_after": state_checksum(info),
            "state_unchanged": state_before == state_checksum(info),
            "saf_parameter_count_per_layer": sum(p.numel() for p in next(iter(saf_by_layer.values())).parameters()),
        })
    deltas = delta_summary(raw_all)
    outcome, details = decide(deltas)
    write_csv(OUT / "RAW_METRICS.csv", raw_all)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat_all)
    write_csv(OUT / "SAF_TRAINING_DIAGNOSTICS.csv", train_all)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", source_audit)
    write_csv(OUT / "LAYER_MAP_CORRELATION.csv", corr_all)
    write_csv(OUT / "CHECKSUMS.csv", checksum_rows)
    write_csv(OUT / "PILOT_DELTAS.csv", deltas)
    write_csv(OUT / "RAW_MULTILAYER_REFERENCE.csv", old_multilayer_rows())
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, **details, "smoke": args.smoke})
    write_report(raw_all, deltas, train_all, corr_all, checksum_rows, outcome, details, time.time() - t0)
    print(f"[LSA] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
