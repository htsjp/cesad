import argparse
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
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells, sota_reference
from experiments.phase_hard_normal_contrast import PC1_DEBIAS_OUT, load_payload
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_paper_ready_final_validation import FULL_RES_METRIC_MAP, existing_seed0_rows
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import evaluate_maps, fnum, json_default, write_json
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, LR, SIDE, norm_np, patch_labels
from models.mlp import average_neighbor
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_cross_layer_robust_state_learning/formal")
LAYERS = (6, 12, 18, 24)
TARGETS = {"MVTec", "BTAD", "MPDD", "DTD-Synthetic", "VisA"}
PAIRS = {
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
}
VARIANTS = ("E1_MultiLayerInferenceOnly", "E2_SharedMultiLayerMeanTraining", "E3_CrossLayerRobustStateLearning")
EPOCHS = 10
TOP_NEG_FRAC = 0.10


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_report(lines):
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_variants_eval(source, target, maps, metas, variants):
    old = tuple(lzp.VARIANTS)
    try:
        lzp.VARIANTS = tuple(variants)
        return evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old


def layer_identity_audit(model):
    n_blocks = len(model.visual.transformer.resblocks)
    rows = []
    for layer in LAYERS:
        rows.append({
            "requested_transformer_block": layer,
            "python_module_index_0_based": layer - 1,
            "exists": 1 <= layer <= n_blocks,
            "note": "AnomalyCLIP feature_map_layer=[0,1,2,3] indexes the extracted 6/12/18/24 feature list, not transformer blocks 0/1/2/3.",
        })
    rows.append({"requested_transformer_block": "__num_blocks__", "python_module_index_0_based": "", "exists": True, "note": n_blocks})
    return rows


def snapshot_patch(outputs_pack, layer):
    outputs = outputs_pack["outputs"]
    if layer == 24:
        return normalized_patch_feature(outputs).float()[0]
    snap = outputs_pack["snapshots"][f"true_block{layer}"]
    branch = snap[0] if isinstance(snap, list) else snap
    patch = branch.permute(1, 0, 2)[:, 1:, :].float()
    patch = average_neighbor(patch)
    return F.normalize(patch, dim=-1)[0]


@torch.no_grad()
def image_layer_features(clip_model, item, device):
    image = item["img"].unsqueeze(0).to(device)
    pack = visual_outputs(clip_model, image, grad=False, snapshot_mode="true_clone")
    return {layer: snapshot_patch(pack, layer).detach().cpu().numpy().astype(np.float32) for layer in LAYERS}


def preprocess_layer(z, source_info, device):
    arr = np.asarray(z, dtype=np.float32).reshape(-1, np.asarray(z).shape[-1])
    if source_info["use_align"]:
        arr = remove_pc1(arr, source_info["q_pc1"])
        arr = norm_np(arr)
        with torch.no_grad():
            out = source_info["align_model"].adapt(torch.tensor(arr, dtype=torch.float32, device=device))
        return out.detach().cpu().numpy().astype(np.float32)
    return norm_np(arr).astype(np.float32)


def source_positions_by_category(dataset, indices):
    cats = sorted({str(dataset.data_all[int(idx)]["cls_name"]) for idx in indices})
    return {cat: [p for p, idx in enumerate(indices) if str(dataset.data_all[int(idx)]["cls_name"]) == cat] for cat in cats}


def select_source_pool(source_name, smoke):
    if source_name == "Full VisA":
        _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    else:
        _name, dataset, indices, patch = load_full_mvtec_source(False)
    slots = source_positions_by_category(dataset, indices)
    if smoke:
        trimmed = {}
        for cat, positions in slots.items():
            keep, n0, n1 = [], 0, 0
            for pos in positions:
                item = dataset[int(indices[int(pos)])]
                label = int(item.get("anomaly", 0))
                if label == 0 and n0 < 2:
                    keep.append(pos)
                    n0 += 1
                if label == 1 and n1 < 2:
                    keep.append(pos)
                    n1 += 1
                if n0 >= 2 and n1 >= 2:
                    break
            if keep:
                trimmed[cat] = keep
        slots = trimmed
    return dataset, list(indices), patch, slots


def build_source_layer_pool(source_info, clip_model, device, smoke):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / f"{source_info['source'].replace(' ', '_')}_SOURCE_LAYER_POOL_{suffix}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        groups = {}
        for cat in data["categories"].tolist():
            idx = data[f"{cat}_idx"]
            groups[cat] = {
                "y": data[f"{cat}_y"].astype(np.float32),
                "origin": data[f"{cat}_origin"].astype(str),
                "x": {layer: data[f"{cat}_L{layer}"].astype(np.float32) for layer in LAYERS},
            }
        return groups, data["audit"].tolist()

    dataset, indices, patch_final, slots = select_source_pool(source_info["source"], smoke)
    rng = np.random.default_rng(0)
    groups, audit_rows = {}, []
    for cat, positions in slots.items():
        parts = {layer: [] for layer in LAYERS}
        ys, origins = [], []
        used_by_origin = defaultdict(int)
        for pos in positions:
            item = dataset[int(indices[int(pos)])]
            y_grid = patch_labels(item).astype(np.int64)
            is_anom = bool(y_grid.max() > 0)
            pools = [("normal_image", 0, np.flatnonzero(y_grid == 0))]
            if is_anom:
                pools = [("anomaly_mask_out", 0, np.flatnonzero(y_grid == 0)), ("anomaly_mask", 1, np.flatnonzero(y_grid == 1))]
            chosen = []
            for origin, cls, ids in pools:
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - used_by_origin[origin]
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                used_by_origin[origin] += int(ids.size)
                chosen.append((origin, cls, ids))
            if not chosen:
                continue
            layer_feats = image_layer_features(clip_model, item, device)
            for origin, cls, ids in chosen:
                for layer in LAYERS:
                    parts[layer].append(preprocess_layer(layer_feats[layer][ids], source_info, device))
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
        print(f"[CLRSL] source pool {source_info['source']} {cat} samples={len(ys)}", flush=True)

    save = {"categories": np.asarray(list(groups.keys()), dtype=object), "audit": np.asarray(audit_rows, dtype=object)}
    for cat, data in groups.items():
        save[f"{cat}_idx"] = np.arange(len(data["y"]))
        save[f"{cat}_y"] = data["y"]
        save[f"{cat}_origin"] = data["origin"]
        for layer in LAYERS:
            save[f"{cat}_L{layer}"] = data["x"][layer].astype(np.float16)
    np.savez_compressed(cache, **save)
    return groups, audit_rows


def balanced_layer_batch(groups, rng):
    cats = list(groups)
    per_cat = max(2, BATCH_SIZE // max(1, len(cats)))
    xs = {layer: [] for layer in LAYERS}
    ys = []
    for cat in cats:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            for layer in LAYERS:
                xs[layer].append(data["x"][layer][choice])
            ys.append(data["y"][choice])
    y = np.concatenate(ys).astype(np.float32)
    order = rng.permutation(len(y))
    return {layer: np.concatenate(xs[layer], axis=0).astype(np.float32)[order] for layer in LAYERS}, y[order]


def layer_loss(logits, y):
    task, _bce, _dice = balanced_loss(logits, y)
    neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    k = max(1, int(math.ceil(float(len(neg)) * TOP_NEG_FRAC)))
    idx = neg[torch.topk(logits.detach()[neg], k=k, largest=True).indices]
    return task + F.softplus(logits[idx]).mean(), idx


def train_variant(source_info, groups, variant, mode, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "checkpoints" / f"{source_info['source'].replace(' ', '_')}_{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        return payload["v"].astype(np.float32), payload.get("diagnostics", []), True
    param = torch.nn.Parameter(torch.tensor(source_info["v0"], dtype=torch.float32, device=device))
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([param], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(0)
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, int(math.ceil(total / BATCH_SIZE)))
    if smoke:
        steps = min(steps, 3)
    diagnostics = []
    for epoch in range(1, EPOCHS + 1):
        epoch_losses, epoch_layer_losses = [], []
        overlap = defaultdict(list)
        for _step in range(steps):
            xb, yb = balanced_layer_batch(groups, rng)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(param, dim=-1)
            losses, selected = [], {}
            for layer in LAYERS:
                z = torch.tensor(xb[layer], dtype=torch.float32, device=device)
                logits = contrast_logits(z, m, source_info["alpha"], v)
                loss_l, idx = layer_loss(logits, y)
                losses.append(loss_l)
                selected[layer] = set(idx.detach().cpu().numpy().tolist())
            stack = torch.stack(losses)
            mean = stack.mean()
            if mode == "mean":
                loss = mean
            else:
                loss = mean + torch.sqrt(torch.mean((stack - mean) ** 2) + 1e-12)
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals = [float(x.detach().cpu()) for x in losses]
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_layer_losses.append(vals)
            for i, a in enumerate(LAYERS):
                for b in LAYERS[i + 1:]:
                    union = len(selected[a] | selected[b])
                    inter = len(selected[a] & selected[b])
                    overlap[(a, b)].append(float(inter / union) if union else float("nan"))
        arr = np.asarray(epoch_layer_losses, dtype=np.float64)
        row = {
            "source": source_info["source"],
            "variant": variant,
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)),
            "L_mean": float(arr.mean()),
            "L_std": float(arr.std(axis=1).mean()),
            "L_max_minus_min": float((arr.max(axis=1) - arr.min(axis=1)).mean()),
        }
        for i, layer in enumerate(LAYERS):
            row[f"L{layer}"] = float(arr[:, i].mean())
        for (a, b), vals in overlap.items():
            row[f"hard_jaccard_L{a}_L{b}"] = float(np.nanmean(vals))
        diagnostics.append(row)
        print(f"[CLRSL] train {source_info['source']} {variant} epoch={epoch}/{EPOCHS} loss={row['loss']:.4f} layer_std={row['L_std']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(param, dim=-1).detach().cpu().numpy().astype(np.float32)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "diagnostics": diagnostics}, ckpt)
    return v, diagnostics, False


def score_layers(layer_feats, source_info, vectors, device):
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    out = {name: [] for name in vectors}
    layer_scores_for_corr = {layer: None for layer in LAYERS}
    for layer in LAYERS:
        z = preprocess_layer(layer_feats[layer], source_info, device)
        zt = torch.tensor(z, dtype=torch.float32, device=device)
        for name, v in vectors.items():
            vt = torch.tensor(v, dtype=torch.float32, device=device)
            s = contrast_logits(zt, m, source_info["alpha"], vt).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            out[name].append(s)
            if name == "E3_CrossLayerRobustStateLearning":
                layer_scores_for_corr[layer] = s
    return {name: np.mean(vals, axis=0).astype(np.float32) for name, vals in out.items()}, layer_scores_for_corr


def target_item(dataset, idx_or_rows, kind, pos):
    return dataset[int(idx_or_rows[pos])] if kind == "source_dataset" else dataset[pos]


def target_meta(dataset, idx_or_rows, kind, target, pos):
    item = target_item(dataset, idx_or_rows, kind, pos)
    if kind == "source_dataset":
        cat = str(item.get("cls_name", target))
        label = int(item.get("anomaly", 0))
    else:
        row = dict(idx_or_rows[pos])
        cat = str(row.get("category", item.get("cls_name", target)))
        label = int(row.get("label", item.get("anomaly", 0)))
    mask = (np.squeeze(item["img_mask"].detach().cpu().numpy()) > 0.5).astype(np.uint8)
    return {"category": cat, "label": label, "mask": mask}


def evaluate_source_targets(source_info, clip_model, vectors, device, smoke):
    suffix = "smoke" if smoke else "formal"
    raw, percat, corr_rows = [], [], []
    cells, _audits = protocol_cells(source_info["source"], device, smoke)
    for _cid, target, dataset, idx_or_rows, _patch, kind in cells:
        if (source_info["source"], target) not in PAIRS:
            continue
        n = min(24, len(idx_or_rows)) if smoke else len(idx_or_rows)
        cache = OUT / f"{source_info['source'].replace(' ', '_')}_to_{target}_maps_{suffix}.npz"
        metas = [target_meta(dataset, idx_or_rows, kind, target, i) for i in range(n)]
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            maps = {name: [x.astype(np.float32) for x in data[name]] for name in VARIANTS}
            corr_rows.extend(data["corr_rows"].tolist())
        else:
            maps = {name: [] for name in VARIANTS}
            layer_corr_acc = []
            for pos in range(n):
                item = target_item(dataset, idx_or_rows, kind, pos)
                feats = image_layer_features(clip_model, item, device)
                fused, layer_scores = score_layers(feats, source_info, vectors, device)
                for name in VARIANTS:
                    maps[name].append(fused[name])
                flat = {layer: layer_scores[layer].reshape(-1) for layer in LAYERS}
                for i, a in enumerate(LAYERS):
                    for b in LAYERS[i + 1:]:
                        layer_corr_acc.append({
                            "source": source_info["source"],
                            "dataset": target,
                            "image_index": pos,
                            "layer_a": a,
                            "layer_b": b,
                            "pearson": float(pearsonr(flat[a], flat[b]).statistic),
                            "spearman": float(spearmanr(flat[a], flat[b]).statistic),
                        })
                if pos % 25 == 0:
                    print(f"[CLRSL] eval maps {source_info['source']} -> {target} image {pos}/{n}", flush=True)
            corr_rows.extend(layer_corr_acc)
            np.savez_compressed(cache, **{name: np.stack(maps[name]).astype(np.float16) for name in VARIANTS}, corr_rows=np.asarray(layer_corr_acc, dtype=object))
            maps = {name: [np.asarray(x, dtype=np.float32) for x in maps[name]] for name in VARIANTS}
        rr, cc = selected_variants_eval(source_info["source"], target, maps, metas, VARIANTS)
        raw.extend(rr)
        percat.extend(cc)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", percat)
    return raw, percat, corr_rows


def b0_rows():
    rows = []
    seen = set()
    for r in existing_seed0_rows():
        if (r.get("source"), r.get("dataset")) not in PAIRS:
            continue
        if r.get("seed0_method") == "StateAligned":
            continue
        if r.get("variant") != "B0_HardNormal":
            continue
        if not np.isfinite(fnum(r, "M_FR_P_AP")):
            continue
        key = (r.get("source"), r.get("dataset"))
        if key in seen:
            continue
        seen.add(key)
        row = {"source": key[0], "dataset": key[1], "variant": "B0_FinalLayerCanonicalNTR", "aggregation": "CATEGORY_MACRO", "score_resolution": r.get("score_resolution", "37x37")}
        for k, v in r.items():
            if k.startswith(("M_FR_", "V_FR_", "PG_")):
                row[k] = v
        rows.append(row)
    return rows


def summarize(raw):
    rows = []
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    for source, dataset in sorted({(r["source"], r["dataset"]) for r in raw}):
        b0 = by.get((source, dataset, "B0_FinalLayerCanonicalNTR"))
        if not b0:
            continue
        row = {"source": source, "dataset": dataset, "B0_M_FR_P_AP": fnum(b0, "M_FR_P_AP")}
        for variant in VARIANTS:
            cur = by.get((source, dataset, variant))
            if not cur:
                continue
            row[f"{variant}_M_FR_P_AP"] = fnum(cur, "M_FR_P_AP")
            row[f"{variant}_delta_P_AP_vs_B0"] = fnum(cur, "M_FR_P_AP") - fnum(b0, "M_FR_P_AP")
            row[f"{variant}_delta_G_vs_B0"] = fnum(cur, "M_FR_G_pixel") - fnum(b0, "M_FR_G_pixel")
        rows.append(row)
    return rows


def balance_metrics(raw):
    refs = {(r["training_source"], r["target"], r["paper"]): r for r in sota_reference()}
    rows = []
    for variant in ("B0_FinalLayerCanonicalNTR",) + VARIANTS:
        gaps = []
        for r in raw:
            if r.get("variant") != variant:
                continue
            candidates = [ref for (src, tgt, _paper), ref in refs.items() if src == r["source"] and tgt == r["dataset"] and ref.get("comparison_type") == "DIRECT_COMPARABLE"]
            if not candidates:
                continue
            ref = candidates[0]
            pap = fnum(ref, "P_AP")
            if np.isfinite(pap):
                gap = fnum(r, "M_FR_P_AP") - pap
                gaps.append(gap)
                rows.append({"variant": variant, "source": r["source"], "dataset": r["dataset"], "reference": ref["paper"], "gap_P_AP": gap})
        if gaps:
            arr = np.asarray(gaps, dtype=np.float64)
            rows.append({"variant": variant, "source": "__summary__", "dataset": "__direct__", "reference": "__direct__", "mean_gap": float(arr.mean()), "std_gap": float(arr.std()), "worst_gap": float(arr.min())})
    return rows


def sota_gap(raw):
    rows = []
    for ref in sota_reference():
        for r in raw:
            if r.get("source") == ref.get("training_source") and r.get("dataset") == ref.get("target") and r.get("variant") == "E3_CrossLayerRobustStateLearning":
                rows.append({
                    "paper": ref["paper"],
                    "comparison_type": ref["comparison_type"],
                    "source": r["source"],
                    "dataset": r["dataset"],
                    "variant": r["variant"],
                    "ours_M_FR_P_AUROC": fnum(r, "M_FR_P_AUROC"),
                    "ref_P_AUROC": ref.get("P_AUROC"),
                    "Delta_P_AUROC": fnum(r, "M_FR_P_AUROC") - fnum(ref, "P_AUROC"),
                    "ours_M_FR_P_AP": fnum(r, "M_FR_P_AP"),
                    "ref_P_AP": ref.get("P_AP"),
                    "Delta_P_AP": fnum(r, "M_FR_P_AP") - fnum(ref, "P_AP"),
                    "ours_M_FR_G_pixel": fnum(r, "M_FR_G_pixel"),
                    "ref_G_pixel": ref.get("G_pixel"),
                    "Delta_G_pixel": fnum(r, "M_FR_G_pixel") - fnum(ref, "G_pixel"),
                })
    return rows


def decide(summary_rows, balance_rows):
    e3 = { (r["source"], r["dataset"]): r for r in summary_rows }
    def d(src, ds):
        return float(e3[(src, ds)]["E3_CrossLayerRobustStateLearning_delta_P_AP_vs_B0"])
    deltas = np.asarray([d(src, ds) for src, ds in sorted(PAIRS)], dtype=np.float64)
    finite = deltas[np.isfinite(deltas)]
    ok = (
        d("Full MVTec", "VisA") >= 3.0
        and d("Full VisA", "MVTec") >= 1.0
        and d("Full VisA", "BTAD") >= -1.0
        and d("Full MVTec", "BTAD") >= -1.0
        and all(d(src, ds) >= -0.5 for src, ds in PAIRS if ds in {"MPDD", "DTD-Synthetic"})
        and int(np.sum(deltas >= -0.3)) >= 7
        and finite.size == len(deltas)
        and float(finite.mean()) >= 1.5
    )
    if ok:
        outcome = "CROSS_LAYER_RISK_EQUALIZATION_IMPROVES_BALANCED_TRANSFER"
    else:
        vis_improve = d("Full MVTec", "VisA") > 0
        if not vis_improve:
            outcome = "MULTILAYER_TRAINING_DOES_NOT_FIX_TRANSFER_IMBALANCE"
        else:
            outcome = "SOURCE_LAYER_ROBUSTNESS_NOT_TARGET_ROBUSTNESS"
    return outcome, {"mean_delta_P_AP": float(finite.mean()) if finite.size else float("nan"), "finite_targets": int(finite.size), "non_drop_targets": int(np.sum(deltas >= -0.3)), "deltas": {f"{s}->{t}": d(s, t) for s, t in sorted(PAIRS)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    write_csv(OUT / "LAYER_ID_AUDIT.csv", layer_identity_audit(clip_model))

    source_infos = [load_visa_source(device), load_mvtec_source()]
    all_raw = b0_rows()
    all_percat, all_diag, all_corr, pool_rows = [], [], [], []
    for source_info in source_infos:
        groups, audit = build_source_layer_pool(source_info, clip_model, device, args.smoke)
        pool_rows.extend(audit)
        v_e2, diag2, _ = train_variant(source_info, groups, "E2_SharedMultiLayerMeanTraining", "mean", device, args.smoke)
        v_e3, diag3, _ = train_variant(source_info, groups, "E3_CrossLayerRobustStateLearning", "robust", device, args.smoke)
        all_diag.extend(diag2 + diag3)
        vectors = {
            "E1_MultiLayerInferenceOnly": source_info["v1"],
            "E2_SharedMultiLayerMeanTraining": v_e2,
            "E3_CrossLayerRobustStateLearning": v_e3,
        }
        raw, percat, corr = evaluate_source_targets(source_info, clip_model, vectors, device, args.smoke)
        all_raw.extend(raw)
        all_percat.extend(percat)
        all_corr.extend(corr)

    summary = summarize(all_raw)
    balance = balance_metrics(all_raw)
    gaps = sota_gap(all_raw)
    outcome, stats = decide(summary, balance)
    runtime = time.time() - start

    write_csv(OUT / "LAYER_SOURCE_DIAGNOSTICS.csv", all_diag + pool_rows)
    write_csv(OUT / "LAYER_SCORE_CORRELATION.csv", all_corr)
    write_csv(OUT / "RAW_METRICS.csv", all_raw)
    write_csv(OUT / "FULL_RES_METRICS.csv", all_raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", all_percat)
    write_csv(OUT / "BALANCE_METRICS.csv", balance)
    write_csv(OUT / "SOTA_GAP.csv", gaps)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": stats, "runtime_sec": runtime, "device": device, "strict_source_only": True, "target_statistics": False, "lr": LR, "epochs": EPOCHS, "layers": LAYERS, "lrp_after_gate_run": False})
    lines = [
        "# Cross-Layer Robust State Learning",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Protocol: Full VisA / Full MVTec source-only; CLIP frozen; fixed layers 6/12/18/24; one shared canonical state contrast and the existing shared alignment/interface are used across all layers. No target training, target statistics, target-specific layer selection, routing, prompt, or memory was used.",
        "",
        "| Source | Target | B0 P-AP | E1 delta | E2 delta | E3 delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['source']} | {r['dataset']} | {float(r['B0_M_FR_P_AP']):.4f} | {float(r.get('E1_MultiLayerInferenceOnly_delta_P_AP_vs_B0', float('nan'))):+.4f} | {float(r.get('E2_SharedMultiLayerMeanTraining_delta_P_AP_vs_B0', float('nan'))):+.4f} | {float(r.get('E3_CrossLayerRobustStateLearning_delta_P_AP_vs_B0', float('nan'))):+.4f} |")
    lines.extend(["", f"E3 mean P-AP delta over 8 industrial transfers: {stats['mean_delta_P_AP']:+.4f}.", f"E3 non-drop targets (delta >= -0.3): {stats['non_drop_targets']}/8.", "", "E3 failed the balanced gate unless the outcome states the risk equalization support condition; fixed 2x2 LRP was therefore not applied inside this phase.", "", f"Runtime: {runtime:.1f}s."])
    write_report(lines)
    print(f"[CLRSL] OUTCOME={outcome}", flush=True)


if __name__ == "__main__":
    main()
