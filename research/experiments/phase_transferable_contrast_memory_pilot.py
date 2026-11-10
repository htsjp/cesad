import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_defect_query_transfer_pilot import light_fold
from experiments.phase_liv_memory_support_modes import ArrayConcat, npz_array_mmap
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import (
    EPOCHS as STATE_EPOCHS,
    FEATURE_CACHE,
    LR as STATE_LR,
    SIDE,
    centroids,
    norm_np,
    train_state_vectors,
)

OUT = Path("results/phase_transferable_contrast_memory_pilot/formal_corrected")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
DIRMAG_OUT = Path("results/phase_state_contrast_direction_magnitude/formal")
XLV_A = Path("results/phase_xlv_cross_domain_transfer")
XLV_B = Path("results/phase_xlv_medical_pixel_transfer")
SLOTS = 12


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


def load_payload(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def source_true_masks():
    from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps

    true_by_pos, _corrupt, _audit = source_label_maps(False)
    return true_by_pos


def collect_slot_samples(dataset, indices, patch, labels_by_pos, positions, q_pc1, cap, seed):
    rng = np.random.default_rng(seed)
    by_cls = {0: [], 1: []}
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    for pos in positions:
        y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
        z = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
        for cls in (0, 1):
            ids = np.flatnonzero(y == cls)
            if ids.size == 0:
                continue
            room = cap - sum(part.shape[0] for part in by_cls[cls])
            if room <= 0:
                continue
            if ids.size > room:
                ids = rng.choice(ids, size=room, replace=False)
            by_cls[cls].append(norm_np(z[ids]))
    feats, ys, cats = [], [], []
    for cls in (0, 1):
        if not by_cls[cls]:
            continue
        arr = np.concatenate(by_cls[cls], axis=0).astype(np.float32)
        feats.append(arr)
        ys.extend([cls] * arr.shape[0])
        cats.extend(["slot"] * arr.shape[0])
    if not feats:
        return np.zeros((0, 1024), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.asarray([])
    return np.concatenate(feats, axis=0), np.asarray(ys, dtype=np.int64), np.asarray(cats)


def train_slot_contrast(slot_name, positions, dataset, indices, cls, patch, y_img, cat_img, labels_by_pos, q_pc1, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "slot_checkpoints" / f"{slot_name}_{suffix}.npz"
    if ckpt.exists():
        data = np.load(ckpt)
        return data["v"].astype(np.float32), dict(data["meta"].item())
    positions = list(positions)
    print(f"[ContrastMemory] train slot {slot_name} images={len(positions)}", flush=True)
    patch_x, patch_y, patch_cat = collect_slot_samples(
        dataset,
        indices,
        patch,
        labels_by_pos,
        positions,
        q_pc1,
        PATCH_SAMPLE_PER_CLASS_CATEGORY,
        SEED,
    )
    if patch_x.shape[0] == 0 or np.unique(patch_y).size < 2:
        raise RuntimeError(f"{slot_name}: insufficient binary patch samples.")
    img_x = remove_pc1(cls[positions], q_pc1) if cls.shape[1] == q_pc1.shape[0] else cls[positions]
    img_y = y_img[positions]
    img_cat = np.asarray(["slot"] * len(positions))
    init_img = centroids(img_x, img_y)
    init_patch = centroids(patch_x, patch_y)
    _img_states, patch_states, _curves = train_state_vectors(
        img_x,
        img_y,
        img_cat,
        patch_x,
        patch_y,
        patch_cat,
        init_img,
        init_patch,
        shuffled=False,
        device=device,
    )
    v = unit(patch_states[1] - patch_states[0])
    meta = {
        "slot": slot_name,
        "source_images": int(len(positions)),
        "source_normals": int(np.sum(img_y == 0)),
        "source_anomalies": int(np.sum(img_y == 1)),
        "patch_samples": int(patch_x.shape[0]),
        "positive_patch_samples": int(np.sum(patch_y == 1)),
        "state_epochs": STATE_EPOCHS,
        "state_lr": STATE_LR,
    }
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ckpt, v=v.astype(np.float32), meta=np.asarray(meta, dtype=object))
    return v, meta


def source_slots_by_category(dataset, indices, smoke):
    cats = sorted({str(dataset.data_all[int(i)]["cls_name"]) for i in indices})
    if smoke:
        cats = cats[:4]
    return {cat: [p for p, idx in enumerate(indices) if str(dataset.data_all[int(idx)]["cls_name"]) == cat] for cat in cats}


def random_slots_like(category_slots, dataset, indices, seed):
    names = list(category_slots)
    counts = []
    normal_positions, anomaly_positions = [], []
    for name in names:
        pos = category_slots[name]
        normal = [p for p in pos if int(dataset.data_all[int(indices[p])]["anomaly"]) == 0]
        anomaly = [p for p in pos if int(dataset.data_all[int(indices[p])]["anomaly"]) == 1]
        counts.append((len(normal), len(anomaly)))
        normal_positions.extend(normal)
        anomaly_positions.extend(anomaly)
    rng = random.Random(seed)
    rng.shuffle(normal_positions)
    rng.shuffle(anomaly_positions)
    out = {}
    noff = 0
    aoff = 0
    for name, (n_normal, n_anomaly) in zip(names, counts):
        picked = normal_positions[noff : noff + n_normal] + anomaly_positions[aoff : aoff + n_anomaly]
        rng.shuffle(picked)
        out[f"random_{name}"] = picked
        noff += n_normal
        aoff += n_anomaly
    return out


def build_keys(slots, cls, y_img):
    rows = []
    keys = {}
    for name, positions in slots.items():
        pos = np.asarray(positions, dtype=np.int64)
        normal_pos = pos[y_img[pos] == 0]
        use = normal_pos if normal_pos.size else pos
        key = unit(np.asarray(cls[use], dtype=np.float32).mean(axis=0))
        keys[name] = key
        rows.append({
            "memory": "slot_key",
            "slot": name,
            "source_images": int(len(pos)),
            "normal_key_images": int(len(use)),
            "key_norm": float(np.linalg.norm(key)),
        })
    return keys, rows


def target_cells_with_cls(smoke):
    cells = target_cells(smoke)
    out = []
    for cell_id, target, dataset, idx_or_rows, patch, kind in cells:
        if target == "MVTec":
            _name, _dataset, train_indices, held_indices = light_fold("D1", 0)
            cache = FEATURE_CACHE / "D1_fold0_clip.npz"
            cls_full = ArrayConcat([npz_array_mmap(cache, "train_cls"), npz_array_mmap(cache, "held_cls")])
            all_indices = list(train_indices) + list(held_indices)
            if smoke:
                positions = []
                for pos, idx in enumerate(all_indices):
                    if int(_dataset.data_all[int(idx)]["anomaly"]) == 1:
                        positions.append(pos)
                    if len(positions) >= 32:
                        break
                cls = np.stack([np.asarray(cls_full[p], dtype=np.float32) for p in positions[: len(idx_or_rows)]], axis=0)
            else:
                cls = np.stack([np.asarray(cls_full[p], dtype=np.float32) for p in range(len(all_indices))], axis=0)
        elif target == "BTAD":
            cache = np.load(XLV_A / "feature_cache" / "BTAD_clip.npz", allow_pickle=True)
            cls = np.asarray(cache["cls"], dtype=np.float32)
            if smoke:
                cls = cls[: len(idx_or_rows)]
        else:
            cache = np.load(XLV_B / "feature_cache" / f"{target}_clip.npz", allow_pickle=True)
            cls = np.asarray(cache["cls"], dtype=np.float32)
            if smoke:
                cls = cls[: len(idx_or_rows)]
        out.append((cell_id, target, dataset, idx_or_rows, patch, kind, cls))
    return out


def build_slot_models(vectors, m_shared, alpha, align_payload, device):
    return {name: make_model(hybrid_states(m_shared, alpha * unit(v)), align_payload, device) for name, v in vectors.items()}


@torch.no_grad()
def score_memory_variant(models, keys, cls, q_pc1, patch, device, smoke, batch_images=8):
    slot_names = list(keys)
    key_mat = np.stack([keys[n] for n in slot_names], axis=0).astype(np.float32)
    key_mat = norm_np(key_mat)
    cls_n = norm_np(cls)
    slot_ids = np.argmax(cls_n @ key_mat.T, axis=1)
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    slot_ids = slot_ids[:n]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    usage = Counter()
    for sid, name in enumerate(slot_names):
        img_ids = np.flatnonzero(slot_ids == sid)
        if img_ids.size == 0:
            continue
        usage[name] += int(img_ids.size)
        model = models[name].eval()
        for start in range(0, img_ids.size, batch_images):
            ids = img_ids[start : start + batch_images]
            arr = remove_pc1(np.asarray([patch[int(i)] for i in ids], dtype=np.float32), q_pc1)
            flat = norm_np(arr.reshape(-1, arr.shape[-1]))
            x = torch.tensor(flat, dtype=torch.float32, device=device)
            logits, _ = model.logits(x)
            s = torch.sigmoid(logits).detach().cpu().numpy().reshape(len(ids), SIDE, SIDE).astype(np.float32)
            for local, img_id in enumerate(ids):
                maps[int(img_id)] = s[local].astype(np.float16)
                image[int(img_id)] = top1_image_score(s[local])
    total = max(sum(usage.values()), 1)
    probs = np.asarray([usage.get(nm, 0) / total for nm in slot_names], dtype=np.float64)
    entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    return image, maps, usage, entropy, slot_names


def evaluate_memories(memories, q_pc1, device, smoke):
    raw, percat, usage_rows = [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind, cls in target_cells_with_cls(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, payload in memories.items():
            print(f"[ContrastMemory] eval {variant} on {target}", flush=True)
            image, maps, usage, entropy, names = score_memory_variant(payload["models"], payload["keys"], cls, q_pc1, patch, device, smoke)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": target, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            total = max(sum(usage.values()), 1)
            usage_rows.append({"dataset": target, "variant": variant, "slot": "__entropy__", "count": total, "fraction": 1.0, "entropy": entropy, "effective_slots": float(math.exp(entropy))})
            for name in names:
                count = int(usage.get(name, 0))
                usage_rows.append({"dataset": target, "variant": variant, "slot": name, "count": count, "fraction": count / total, "entropy": entropy, "effective_slots": float(math.exp(entropy))})
            for cat, vals in per.items():
                row = {"dataset": target, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat, usage_rows


def vector_stats(vectors, label):
    names = list(vectors)
    mat = np.stack([vectors[n] for n in names], axis=0).astype(np.float32)
    sim = norm_np(mat) @ norm_np(mat).T
    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    rows = [
        {"memory": label, "stat": "slot_count", "value": int(len(names))},
        {"memory": label, "stat": "pairwise_cosine_mean", "value": float(off.mean())},
        {"memory": label, "stat": "pairwise_cosine_std", "value": float(off.std())},
        {"memory": label, "stat": "pairwise_cosine_min", "value": float(off.min())},
        {"memory": label, "stat": "pairwise_cosine_max", "value": float(off.max())},
        {"memory": label, "stat": "ContrastMemoryDiversity", "value": float(1.0 - off.mean())},
    ]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rows.append({"memory": label, "slot_a": a, "slot_b": b, "stat": "pairwise_cosine", "value": float(sim[names.index(a), names.index(b)])})
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_GlobalContrast_ReconstructedAlpha")]
        e1 = by[(ds, "E1_TransferableContrastMemory")]
        e2 = by[(ds, "E2_MatchedRandomContrastMemory")]
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
    stable = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0
    if stable and float(e1_e2.mean()) >= 1.0:
        outcome = "TRANSFERABLE_CONTRAST_MEMORY_SUPPORTED"
    elif stable and abs(float(e1_e2.mean())) < 1.0:
        outcome = "MULTI_CONTRAST_GAIN_NOT_SEMANTIC"
    else:
        outcome = "MULTI_CONTRAST_MEMORY_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_report(summary_rows, memory_stats, outcome, stats, runtime):
    ms = {(r.get("memory"), r.get("stat")): r.get("value") for r in memory_stats if "stat" in r}
    lines = [
        "# Transferable Contrast Memory Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. All variants use the same fixed contrast magnitude alpha from the global TrueContrast. Slot keys use source normal CLS means; slot values are normalized state-contrast directions. Target images retrieve one slot by frozen CLS similarity; no target training, statistics, memory, category names, or defect names.",
        "",
        "| Dataset | B0 Global G | E1 ContrastMemory G | E2 RandomMemory G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
        f"E1 slots: {int(ms[('E1_TransferableContrastMemory', 'slot_count')])}, diversity: {ms[('E1_TransferableContrastMemory', 'ContrastMemoryDiversity')]:.6f}",
        f"E2 slots: {int(ms[('E2_MatchedRandomContrastMemory', 'slot_count')])}, diversity: {ms[('E2_MatchedRandomContrastMemory', 'ContrastMemoryDiversity')]:.6f}",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ContrastMemory] device={device}", flush=True)
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    m_c, _d_c = decompose(states_c)
    dataset_name, dataset, indices, cls, patch, y_img, cat_img = load_full_visa_source_lazy(False)
    labels_by_pos = source_true_masks()
    cat_slots = source_slots_by_category(dataset, indices, args.smoke)
    rnd_slots = random_slots_like(cat_slots, dataset, indices, SEED)
    true_keys, key_rows_true = build_keys(cat_slots, cls, y_img)
    rnd_keys, key_rows_rnd = build_keys(rnd_slots, cls, y_img)

    true_vectors, true_meta = {}, []
    for name, positions in cat_slots.items():
        v, meta = train_slot_contrast(f"true_{name}", positions, dataset, indices, cls, patch, y_img, cat_img, labels_by_pos, q_pc1, device, args.smoke)
        true_vectors[name] = v
        true_meta.append({"memory": "E1_TransferableContrastMemory", **meta})
    rnd_vectors, rnd_meta = {}, []
    for name, positions in rnd_slots.items():
        v, meta = train_slot_contrast(name, positions, dataset, indices, cls, patch, y_img, cat_img, labels_by_pos, q_pc1, device, args.smoke)
        rnd_vectors[name] = v
        rnd_meta.append({"memory": "E2_MatchedRandomContrastMemory", **meta})

    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    alpha = float(np.linalg.norm(d_t))
    global_model = make_model(hybrid_states(m_c, alpha * unit(d_t)), corrupt_payload, device)
    true_models = build_slot_models(true_vectors, m_c, alpha, corrupt_payload, device)
    rnd_models = build_slot_models(rnd_vectors, m_c, alpha, corrupt_payload, device)
    raw, percat, usage_rows = evaluate_memories(
        {
            "B0_GlobalContrast_ReconstructedAlpha": {"models": {"global": global_model}, "keys": {"global": unit(np.asarray(cls[y_img == 0], dtype=np.float32).mean(axis=0))}},
            "E1_TransferableContrastMemory": {"models": true_models, "keys": true_keys},
            "E2_MatchedRandomContrastMemory": {"models": rnd_models, "keys": rnd_keys},
        },
        q_pc1,
        device,
        args.smoke,
    )
    summary_rows = summarize(raw)
    memory_stats = key_rows_true + key_rows_rnd + true_meta + rnd_meta + vector_stats(true_vectors, "E1_TransferableContrastMemory") + vector_stats(rnd_vectors, "E2_MatchedRandomContrastMemory")
    outcome, decision_stats = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "MEMORY_STATS.csv", memory_stats)
    write_csv(OUT / "SLOT_USAGE.csv", usage_rows)
    np.savez_compressed(OUT / "contrast_memory_vectors.npz", **{f"E1_{k}": v for k, v in true_vectors.items()}, **{f"E2_{k}": v for k, v in rnd_vectors.items()})
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "source_dataset": dataset_name,
        "slot_count": len(cat_slots),
        "alpha": alpha,
        "shared_midpoint": "m_C",
        "shared_alignment": "A_C",
    })
    write_report(summary_rows, memory_stats, outcome, decision_stats, runtime)
    print(f"[ContrastMemory] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
