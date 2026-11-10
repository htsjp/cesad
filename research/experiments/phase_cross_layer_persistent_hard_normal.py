import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import extract_layer_features
from experiments.phase_hard_normal_contrast import (
    CACHE_VERSION,
    PC1_DEBIAS_OUT,
    TOP_NEG_FRAC,
    balanced_category_batch,
    final_negative_distribution,
    json_default,
    load_payload,
)
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np


OUT = Path("results/phase_cross_layer_persistent_hard_normal/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
EPOCHS = 10
LAYERS = [6, 12, 18, 24]
MOECLIP = {
    "MVTec": {"P-AUROC": 92.5, "P-AP": 45.7, "G_pixel": 69.10},
    "BTAD": {"P-AUROC": 96.8, "P-AP": 50.4, "G_pixel": 73.60},
    "CVC-ClinicDB": {"P-AUROC": 89.7, "P-AP": 49.9, "G_pixel": 69.80},
}
VISUALAD = {
    "MVTec": {"P-AUROC": 90.8, "P-AP": 41.2, "G_pixel": 66.00},
    "BTAD": {"P-AUROC": 91.1, "P-AP": 43.1, "G_pixel": 67.10},
    "CVC-ClinicDB": {"P-AUROC": 85.2, "P-AP": 37.8, "G_pixel": 61.50},
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def coerce_metric_row(row, variant):
    out = dict(row)
    out["variant"] = variant
    for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
        try:
            out[key] = float(out[key])
        except Exception:
            out[key] = float("nan")
    for key in ("mask_images",):
        if key in out:
            out[key] = int(float(out[key]))
    return out


def load_b0_hardnormal():
    raw = []
    for row in read_csv(HN_OUT / "RAW_METRICS.csv"):
        if row.get("variant") == "E1_HardNormalContrast":
            raw.append(coerce_metric_row(row, "B0_FinalLayerHardNormal"))
    percat = []
    for row in read_csv(HN_OUT / "PER_CATEGORY.csv"):
        if row.get("variant") == "E1_HardNormalContrast":
            percat.append(coerce_metric_row(row, "B0_FinalLayerHardNormal"))
    return raw, percat


def load_b0_hardnormal_vector():
    path = HN_OUT / "contrast_vectors.npz"
    if not path.exists():
        raise RuntimeError(f"Missing HardNormal contrast vector file: {path}")
    vecs = np.load(path)
    return unit_np(np.asarray(vecs["E1"], dtype=np.float32))


def collect_source_pool_with_records(slots, labels_by_pos, patch, align_model, q_pc1, device):
    rng = np.random.default_rng(SEED)
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    by_key = {}
    audit = []
    records = []
    sid = 0
    for cat, positions in slots.items():
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        ys = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        origins = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        cats = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        ids_by_origin = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
            image_is_anomaly = bool(y.max() > 0)
            z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
            z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
            with torch.no_grad():
                z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
            pools = []
            if image_is_anomaly:
                pools.append(("anomaly_mask_out", 0, np.flatnonzero(y == 0)))
                pools.append(("anomaly_mask", 1, np.flatnonzero(y == 1)))
            else:
                pools.append(("normal_image", 0, np.flatnonzero(y == 0)))
            for origin_name, cls, ids in pools:
                if ids.size == 0:
                    continue
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(part.shape[0] for part in parts[origin_name])
                if room <= 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin_name].append(z[ids])
                ys[origin_name].extend([cls] * len(ids))
                origins[origin_name].extend([origin_name] * len(ids))
                cats[origin_name].extend([cat] * len(ids))
                local_sids = []
                for patch_id in ids.tolist():
                    records.append({"sample_id": sid, "source_pos": int(pos), "patch_id": int(patch_id), "category": cat, "label": cls, "origin": origin_name})
                    local_sids.append(sid)
                    sid += 1
                ids_by_origin[origin_name].extend(local_sids)
        xs, y_all, origin_all, cat_all, sid_all = [], [], [], [], []
        for origin_name in ("normal_image", "anomaly_mask_out", "anomaly_mask"):
            if not parts[origin_name]:
                continue
            arr = np.concatenate(parts[origin_name], axis=0).astype(np.float32)
            xs.append(arr)
            y_all.extend(ys[origin_name])
            origin_all.extend(origins[origin_name])
            cat_all.extend(cats[origin_name])
            sid_all.extend(ids_by_origin[origin_name])
        if not xs or len(set(y_all)) < 2:
            raise RuntimeError(f"{cat}: insufficient source patch samples.")
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.asarray(y_all, dtype=np.float32)
        origin = np.asarray(origin_all, dtype=object)
        by_key[cat] = {"x": x, "y": y, "origin": origin, "sid": np.asarray(sid_all, dtype=np.int64), "cat": np.asarray(cat_all, dtype=object)}
        audit.append({
            "variant": "source_pool",
            "record_type": "pool",
            "category": cat,
            "samples": int(x.shape[0]),
            "positive_samples": int(y.sum()),
            "negative_samples": int((y == 0).sum()),
            "normal_image_negatives": int(np.sum(origin[y == 0] == "normal_image")),
            "anomaly_mask_out_negatives": int(np.sum(origin[y == 0] == "anomaly_mask_out")),
        })
    return by_key, records, audit


def batch_with_ids(groups, rng, total_batch):
    names = list(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, origins, cats, sids = [], [], [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            origins.extend(data["origin"][choice].tolist())
            cats.extend(data["cat"][choice].tolist())
            sids.extend(data["sid"][choice].tolist())
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], np.asarray(origins, dtype=object)[order], np.asarray(cats, dtype=object)[order], np.asarray(sids, dtype=np.int64)[order]


def layer_feature_cache(dataset, indices, records, smoke, device):
    suffix = "smoke" if smoke else "formal"
    cache_path = OUT / f"SOURCE_LAYER_HARDNESS_{suffix}.npz"
    meta_path = OUT / f"SOURCE_LAYER_HARDNESS_{suffix}.json"
    if cache_path.exists() and meta_path.exists():
        return np.load(cache_path), json.loads(meta_path.read_text(encoding="utf-8"))

    print("[CrossLayerHN] extract source layer features for selected training patches", flush=True)
    model, *_ = load_components("checkpoints/test_on_mvtec.pth", device)
    by_pos = {}
    labels = np.zeros((len(records),), dtype=np.float32)
    origins = np.empty((len(records),), dtype=object)
    for rec in records:
        by_pos.setdefault(int(rec["source_pos"]), []).append(rec)
        labels[int(rec["sample_id"])] = float(rec["label"])
        origins[int(rec["sample_id"])] = rec["origin"]

    feats = {layer: np.zeros((len(records), 768), dtype=np.float16) for layer in LAYERS}
    start = time.time()
    for n, (pos, recs) in enumerate(sorted(by_pos.items()), start=1):
        item = dataset[int(indices[int(pos)])]
        image = item["img"].unsqueeze(0).to(device)
        for layer in LAYERS:
            _cls, patch_layer, _value = extract_layer_features(model, image, LAYERS, layer, device, smooth=True)
            arr = patch_layer.detach().cpu().numpy()[0].astype(np.float32)
            for rec in recs:
                feats[layer][int(rec["sample_id"])] = arr[int(rec["patch_id"])].astype(np.float16)
        if n % 50 == 0:
            print(f"[CrossLayerHN] layer cache images {n}/{len(by_pos)}", flush=True)

    scores = {}
    for layer in LAYERS:
        x = feats[layer].astype(np.float32)
        x = norm_np(x)
        pos = labels > 0.5
        q = norm_np(x[pos].mean(axis=0, keepdims=True) - x[~pos].mean(axis=0, keepdims=True))[0]
        scores[f"score_{layer}"] = (x @ q).astype(np.float32)
    np.savez_compressed(cache_path, labels=labels, origins=origins, **scores)
    meta = {
        "layers": LAYERS,
        "sample_count": len(records),
        "source_image_count": len(by_pos),
        "feature_dim": 768,
        "hardness_direction": "source layer mean(anomaly_mask_patch)-mean(negative_patch)",
        "target_information_used": False,
        "seconds": time.time() - start,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=json_default), encoding="utf-8")
    return np.load(cache_path), meta


def percentile_rank(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size <= 1:
        return np.ones_like(x, dtype=np.float32)
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(np.float32)
    return order / max(float(x.size - 1), 1.0)


def selected_by_persistent_rank(cache, batch_sids, neg_idx, mode, rng):
    ids = batch_sids[neg_idx.detach().cpu().numpy()]
    ranks = []
    for layer in LAYERS:
        vals = np.asarray(cache[f"score_{layer}"][ids], dtype=np.float32).copy()
        if mode == "shuffle" and layer in (12, 18, 24):
            vals = vals[rng.permutation(vals.size)]
        ranks.append(percentile_rank(vals))
    hard = np.stack(ranks, axis=1).mean(axis=1)
    k = max(1, int(np.ceil(float(len(ids)) * TOP_NEG_FRAC)))
    chosen_np = np.argsort(hard)[-k:]
    return torch.tensor(chosen_np, dtype=torch.long, device=neg_idx.device), hard, np.stack(ranks, axis=1)


def train_variant(variant, mode, groups, layer_cache, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}_{CACHE_VERSION}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["hard_rows"], True
    print(f"[CrossLayerHN] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 2)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    curves = []
    hard_rows = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin, _cats, sids = batch_with_ids(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            selected_local, persist, ranks = selected_by_persistent_rank(layer_cache, sids, neg_idx, mode, rng)
            selected_idx = neg_idx[selected_local]
            extra_loss = F.softplus(logits[selected_idx]).mean()
            loss = task_loss + extra_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sel_np = selected_idx.detach().cpu().numpy()
            selected_origins = origin[sel_np]
            hard_rows.append({
                "variant": variant,
                "record_type": "train_selection",
                "epoch": epoch,
                "step": step + 1,
                "mode": mode,
                "selected_count": int(len(sel_np)),
                "persistent_rank_mean": float(np.mean(persist[selected_local.detach().cpu().numpy()])),
                "persistent_rank_std": float(np.std(persist[selected_local.detach().cpu().numpy()])),
                "normal_image_ratio": float(np.mean(selected_origins == "normal_image")),
                "anomaly_mask_out_ratio": float(np.mean(selected_origins == "anomaly_mask_out")),
                "selected_score_mean": float(logits.detach()[selected_idx].cpu().numpy().mean()),
                "all_negative_score_mean": float(logits.detach()[neg_idx].cpu().numpy().mean()),
            })
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(extra_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "extra_normal_loss": float(arr[:, 2].mean()),
            "bce": float(arr[:, 3].mean()),
            "dice": float(arr[:, 4].mean()),
        }
        curves.append(row)
        print(f"[CrossLayerHN] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} extra={row['extra_normal_loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "hard_rows": hard_rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, hard_rows, False


def layer_diagnostics(cache):
    scores = {layer: np.asarray(cache[f"score_{layer}"], dtype=np.float32) for layer in LAYERS}
    labels = np.asarray(cache["labels"], dtype=np.float32)
    neg = labels < 0.5
    rows = []
    for i, a in enumerate(LAYERS):
        for b in LAYERS[i + 1 :]:
            rho = spearmanr(scores[a][neg], scores[b][neg]).statistic
            rows.append({"record_type": "layer_spearman", "layer_a": a, "layer_b": b, "spearman": float(rho)})
    top_masks = {}
    for layer in LAYERS:
        vals = scores[layer][neg]
        k = max(1, int(np.ceil(vals.size * TOP_NEG_FRAC)))
        ids = np.argsort(vals)[-k:]
        mask = np.zeros((vals.size,), dtype=bool)
        mask[ids] = True
        top_masks[layer] = mask
    overlap = np.stack([top_masks[layer] for layer in LAYERS], axis=1).sum(axis=1)
    for n in (2, 3, 4):
        rows.append({"record_type": "top10_overlap", "layers_in_top10_at_least": n, "patch_ratio": float(np.mean(overlap >= n))})
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_FinalLayerHardNormal")]
        e1 = by[(ds, "E1_PersistentHardNormal")]
        e2 = by[(ds, "E2_ShuffledLayerControl")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
            "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    mt_gap_shrink = float(e1_b0[0])
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and mt_gap_shrink >= 1.0:
        outcome = "CROSS_LAYER_PERSISTENT_HARD_NORMAL_SUPPORTED"
    elif e1_b0[1] > 0 and e1_b0[0] <= 0.5 and e1_b0[2] <= 0.5:
        outcome = "PERSISTENT_HARDNESS_DOMAIN_DEPENDENT"
    else:
        outcome = "CROSS_LAYER_PERSISTENT_HARD_NORMAL_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
        "MVTec_MoECLIP_gap_shrink_pp": mt_gap_shrink,
    }


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_PersistentHardNormal"}
    for ref_name, ref_map in (("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD), ("MoECLIP pixel", MOECLIP)):
        for ds, ref in ref_map.items():
            row = current[ds]
            p_auroc = float(row["P-AUROC"])
            p_ap = float(row["P-AP"])
            g = float(row["G_pixel"])
            rows.append({
                "dataset": ds,
                "current_method": "E1_PersistentHardNormal",
                "current_P-AUROC": p_auroc,
                "current_P-AP": p_ap,
                "current_G": g,
                "reference": ref_name,
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": p_auroc - ref["P-AUROC"],
                "Delta_P_AP": p_ap - ref["P-AP"],
                "Delta_G": g - ref["G_pixel"],
                "comparison_status": "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS",
            })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(summary_rows, outcome, stats, geom, runtime, sota_rows, layer_meta):
    lines = [
        "# Cross-Layer Persistent Hard-Normal",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. B0 reuses the validated final-layer HardNormal result. E1/E2 train only one global contrast; multi-layer signals are used only for source negative patch selection, and target inference adds no module.",
        "",
        "| Dataset | B0 Final-Layer HN G | E1 Persistent HN G | E2 Shuffled-Layer G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        f"MVTec MoECLIP gap shrink proxy equals E1-B0 on MVTec: {stats['MVTec_MoECLIP_gap_shrink_pp']:+.4f} pp.",
        f"Contrast cosines: B0/E1={geom[0]['cos_v_B0_v_E1']:.6f}, B0/E2={geom[0]['cos_v_B0_v_E2']:.6f}, E1/E2={geom[0]['cos_v_E1_v_E2']:.6f}.",
        f"Layer cache: {layer_meta['sample_count']} source patches from {layer_meta['source_image_count']} source images; target information used: {layer_meta['target_information_used']}.",
        "",
        "## External SOTA Position",
        "",
        "VisualAD(CLIP) and MoECLIP rows are VisA-source pixel metric references. They are directly comparable only at the level of source/target split and pixel P-AUROC/P-AP/G; exact implementation details and preprocessing remain protocol caveats.",
        "",
        "| Reference | Dataset | Current G | Reference G | Delta G | Status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for r in sota_rows:
        lines.append(f"| {r['reference']} | {r['dataset']} | {float(r['current_G']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend([
        "",
        "Stop-loss: this round is evaluated against the prior supported HardNormal. If it fails to improve mean G by >= +1pp and fails to shrink the MVTec MoECLIP gap by >=1pp, do not add more cross-layer selection complexity.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CrossLayerHN] device={device}", flush=True)
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, records, pool_audit = collect_source_pool_with_records(slots, labels_by_pos, patch, align_model, q_pc1, device)
    layer_cache, layer_meta = layer_feature_cache(dataset, indices, records, args.smoke, device)

    v_e1, hard_e1, reused_e1 = train_variant("E1_PersistentHardNormal", "persistent", groups, layer_cache, m_c, alpha, init_v, device, args.smoke)
    v_e2, hard_e2, reused_e2 = train_variant("E2_ShuffledLayerControl", "shuffle", groups, layer_cache, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_PersistentHardNormal": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_ShuffledLayerControl": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_hardnormal()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    hard_rows = pool_audit + layer_diagnostics(layer_cache) + hard_e1 + hard_e2
    for variant, v in (("E1_PersistentHardNormal", v_e1), ("E2_ShuffledLayerControl", v_e2)):
        hard_rows.extend(final_negative_distribution(variant, groups, v, m_c, alpha))
    v_b0 = load_b0_hardnormal_vector()
    geom = contrast_geometry(v_b0, v_e1, v_e2)
    outcome, stats = decide(summary_rows)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "PERSISTENT_HARDNESS_DIAGNOSTICS.csv", hard_rows)
    np.savez_compressed(OUT / "contrast_vectors.npz", B0=v_b0.astype(np.float32), E1=v_e1.astype(np.float32), E2=v_e2.astype(np.float32))
    sota_rows = write_sota_gap(raw)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "layers": LAYERS,
        "alpha": alpha,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, stats, geom, runtime, sota_rows, layer_meta)
    print(f"[CrossLayerHN] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
