import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_spatial_supervision_transfer_audit import (
    corrupt_masks,
    fast_patch_grid_eval,
    target_cells,
    union_source_records,
)
from experiments.phase_statealigned_supervision_necessity import (
    load_full_visa_source_lazy,
    score_adapter_batch,
)
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import (
    BATCH_SIZE,
    EPOCHS as ADAPTER_EPOCHS,
    LR as ADAPTER_LR,
    PATCH_SAMPLE_PER_CLASS_CATEGORY,
    SEED,
    WEIGHT_DECAY,
    balanced_loss,
)
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import (
    EPOCHS as STATE_EPOCHS,
    LR as STATE_LR,
    SIDE,
    centroids,
    norm_np,
    train_state_vectors,
)

OUT = Path("results/phase_statealigned_pc1_debias/formal")
SUPERVISION_OUT = Path("results/phase_statealigned_supervision_necessity/formal")


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


def remove_pc1(x, q):
    x = np.asarray(x, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    return x - (x @ q)[..., None] * q


def source_labels(smoke, corrupted):
    dataset_u, indices_u, _patch_u, records = union_source_records(smoke)
    if corrupted:
        masks, audit = corrupt_masks(records)
    else:
        masks = [np.asarray(r["mask"], dtype=np.float32) for r in records]
        audit = []
    labels_by_pos = {}
    for i, rec in enumerate(records):
        if bool(rec["normal"]):
            labels_by_pos[int(rec["pos"])] = np.zeros((SIDE * SIDE,), dtype=np.int64)
        else:
            labels_by_pos[int(rec["pos"])] = (np.asarray(masks[i], dtype=np.float32) > 0.5).astype(np.int64)
    return dataset_u, indices_u, records, labels_by_pos, audit


def compute_statealigned_pc1(dataset, indices, patch, smoke):
    suffix = "smoke" if smoke else "formal"
    q_path = OUT / f"q_pc1_statealigned1024_{suffix}.npy"
    spectrum_path = OUT / f"PC1_SPATIAL_SPECTRUM_{suffix}.csv"
    if q_path.exists() and spectrum_path.exists():
        q = np.load(q_path).astype(np.float32)
        return q / max(float(np.linalg.norm(q)), 1e-12)

    print("[PC1Debias] compute normal-only spatial PC1 in 1024-d StateAligned feature space", flush=True)
    fmap_sum = np.zeros((SIDE * SIDE, 1024), dtype=np.float64)
    n_normal = 0
    source_positions = range(len(indices))
    if smoke:
        source_positions = range(min(96, len(indices)))
    for pos in source_positions:
        item = dataset.data_all[int(indices[pos])]
        if int(item["anomaly"]) != 0:
            continue
        fmap_sum += np.asarray(patch[pos], dtype=np.float32).astype(np.float64)
        n_normal += 1
    if n_normal <= 0:
        raise RuntimeError("No source normal images available for PC1 computation.")
    fmap = (fmap_sum / n_normal).astype(np.float64)
    x = fmap - fmap.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    q = vt[0].astype(np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    total = float(np.sum(s * s))
    rows = []
    for i, sv in enumerate(s):
        rows.append(
            {
                "rank": i + 1,
                "singular_value": float(sv),
                "explained_variance_ratio": float((sv * sv) / max(total, 1e-12)),
                "cumulative_explained_variance_ratio": float(np.sum(s[: i + 1] * s[: i + 1]) / max(total, 1e-12)),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(q_path, q)
    write_csv(spectrum_path, rows)
    return q


def feature_variance(dataset, indices, patch, q, smoke):
    rows = []
    source_positions = list(range(len(indices)))
    if smoke:
        source_positions = source_positions[:96]
    groups = {
        "all_source_patches": source_positions,
        "normal_source_patches": [p for p in source_positions if int(dataset.data_all[int(indices[p])]["anomaly"]) == 0],
    }
    for group, positions in groups.items():
        count = 0
        sum_x = np.zeros((1024,), dtype=np.float64)
        sum_x2 = np.zeros((1024,), dtype=np.float64)
        sum_proj = 0.0
        sum_proj2 = 0.0
        for pos in positions:
            arr = np.asarray(patch[pos], dtype=np.float32).reshape(-1, 1024).astype(np.float64)
            proj = arr @ q.astype(np.float64)
            count += arr.shape[0]
            sum_x += arr.sum(axis=0)
            sum_x2 += (arr * arr).sum(axis=0)
            sum_proj += float(proj.sum())
            sum_proj2 += float((proj * proj).sum())
        mean = sum_x / max(count, 1)
        total_var_before = float(np.maximum(sum_x2 / max(count, 1) - mean * mean, 0).sum())
        proj_mean = sum_proj / max(count, 1)
        pc1_var = float(max(sum_proj2 / max(count, 1) - proj_mean * proj_mean, 0.0))
        total_var_after = max(total_var_before - pc1_var, 0.0)
        rows.append(
            {
                "group": group,
                "patch_count": int(count),
                "total_variance_before": total_var_before,
                "pc1_axis_variance": pc1_var,
                "total_variance_after": total_var_after,
                "removed_variance_ratio": pc1_var / max(total_var_before, 1e-12),
                "remaining_variance_ratio": total_var_after / max(total_var_before, 1e-12),
            }
        )
    return rows


def collect_pc1_removed_patch_samples(dataset, indices, patches, labels_by_pos, source_positions, q, cap, seed):
    rng = np.random.default_rng(seed)
    by_key = defaultdict(list)
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    for source_pos in source_positions:
        idx = indices[int(source_pos)]
        item = dataset.data_all[int(idx)]
        cat = str(item["cls_name"])
        y = np.asarray(labels_by_pos.get(int(source_pos), zero), dtype=np.int64)
        x = remove_pc1(np.asarray(patches[int(source_pos)], dtype=np.float32), q)
        for cls in (0, 1):
            ids = np.flatnonzero(y == cls)
            if ids.size == 0:
                continue
            key = (cat, cls)
            room = cap - sum(part.shape[0] for part in by_key[key])
            if room <= 0:
                continue
            if ids.size > room:
                ids = rng.choice(ids, size=room, replace=False)
            by_key[key].append(norm_np(x[ids]))
    feats, ys, cats = [], [], []
    for (cat, cls), parts in sorted(by_key.items()):
        if not parts:
            continue
        arr = np.concatenate(parts, axis=0).astype(np.float32)
        feats.append(arr)
        ys.extend([cls] * arr.shape[0])
        cats.extend([cat] * arr.shape[0])
    if not feats:
        return np.zeros((0, 1024), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.asarray([])
    return np.concatenate(feats, axis=0), np.asarray(ys, dtype=np.int64), np.asarray(cats)


def train_pc1_removed_statealigned(variant, corrupted, q, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    dataset_name, dataset, indices, cls, patch, y_img, cat_img = load_full_visa_source_lazy(False)
    _dataset_u, indices_u, _records, labels_by_pos, audit = source_labels(smoke, corrupted)
    source_positions = sorted(labels_by_pos) if smoke else list(range(len(indices)))
    if any(int(indices_u[p]) != int(indices[p]) for p in source_positions):
        raise RuntimeError(f"{variant}: source order mismatch.")
    if smoke:
        cls_train = cls[source_positions]
        y_img_train = y_img[source_positions]
        cat_img_train = cat_img[source_positions]
    else:
        cls_train = cls
        y_img_train = y_img
        cat_img_train = cat_img

    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        states = np.asarray(payload["patch_states"], dtype=np.float32)
        model = AdapterToState(1024, states).to(device).eval()
        model.load_state_dict(payload["model"], strict=True)
        return states, model, payload.get("curves", []), audit, True

    print(f"[PC1Debias] train {variant} states", flush=True)
    patch_x, patch_y, patch_cat = collect_pc1_removed_patch_samples(
        dataset,
        indices,
        patch,
        labels_by_pos,
        source_positions,
        q,
        PATCH_SAMPLE_PER_CLASS_CATEGORY,
        SEED,
    )
    init_img = centroids(cls_train, y_img_train)
    init_patch = centroids(patch_x, patch_y)
    img_states, patch_states, state_curves = train_state_vectors(
        cls_train,
        y_img_train,
        cat_img_train,
        patch_x,
        patch_y,
        patch_cat,
        init_img,
        init_patch,
        shuffled=False,
        device=device,
    )

    print(f"[PC1Debias] train {variant} adapter", flush=True)
    model = AdapterToState(1024, patch_states).to(device)
    x_all = norm_np(patch_x)
    y_all = patch_y.astype(np.float32)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=ADAPTER_LR, weight_decay=WEIGHT_DECAY)
    adapter_curves = []
    for epoch in range(1, ADAPTER_EPOCHS + 1):
        order = rng.permutation(len(y_all))
        vals = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            logits, _ = model.logits(x)
            loss, bce, dice = balanced_loss(logits, y)
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "stage": "adapter",
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "bce": float(arr[:, 1].mean()),
            "dice": float(arr[:, 2].mean()),
        }
        adapter_curves.append(row)
        print(f"[PC1Debias] {variant} adapter epoch {epoch}/{ADAPTER_EPOCHS} loss={row['loss']:.4f}", flush=True)
    curves = []
    for row in state_curves:
        rr = dict(row)
        rr["stage"] = "state"
        rr["variant"] = variant
        curves.append(rr)
    curves.extend(adapter_curves)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.eval().state_dict(),
            "img_states": img_states,
            "patch_states": patch_states,
            "curves": curves,
            "variant": variant,
            "source_dataset": dataset_name,
            "pc1_removed": True,
            "corrupted_masks": bool(corrupted),
            "state_epochs": STATE_EPOCHS,
            "state_lr": STATE_LR,
            "adapter_epochs": ADAPTER_EPOCHS,
            "adapter_lr": ADAPTER_LR,
            "adapter_weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        ckpt,
    )
    return patch_states, model.eval(), curves, audit, False


@torch.no_grad()
def score_pc1_removed_adapter_batch(patch, model, q, device, smoke=False, batch_images=8):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    model.eval()
    for start in range(0, n, batch_images):
        arr = remove_pc1(np.asarray(patch[start : start + batch_images], dtype=np.float32), q)
        b = arr.shape[0]
        flat = norm_np(arr.reshape(-1, arr.shape[-1]))
        x = torch.tensor(flat, dtype=torch.float32, device=device)
        logits, _ = model.logits(x)
        s = torch.sigmoid(logits).detach().cpu().numpy().reshape(b, SIDE, SIDE).astype(np.float32)
        maps[start : start + b] = s.astype(np.float16)
        image[start : start + b] = np.asarray([top1_image_score(m) for m in s], dtype=np.float32)
    return image, maps


def evaluate_pc1_removed(models, q, device, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, model in models.items():
            print(f"[PC1Debias] eval {variant} on {cell_id}", flush=True)
            image, maps = score_pc1_removed_adapter_batch(patch, model, q, device, smoke=smoke)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat


def load_b0_reuse():
    raw, percat = [], []
    with (SUPERVISION_OUT / "RAW_METRICS.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] != "B0_TrueMask-StateAligned":
                continue
            out = dict(row)
            out["variant"] = "B0_OriginalTrueMask"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            raw.append(out)
    with (SUPERVISION_OUT / "PER_CATEGORY.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] != "B0_TrueMask-StateAligned":
                continue
            out = dict(row)
            out["variant"] = "B0_OriginalTrueMask"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            percat.append(out)
    return raw, percat


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_OriginalTrueMask")]
        e1 = by[(ds, "E1_PC1Removed-TrueMask")]
        e2 = by[(ds, "E2_PC1Removed-CorruptedMask")]
        rows.append(
            {
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
            }
        )
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    mean_e1_b0 = float(np.mean(e1_b0))
    mean_e1_e2 = float(np.mean(e1_e2))
    e1_e2_wins = int(np.sum(e1_e2 > 0))
    if mean_e1_e2 >= 3.0 and e1_e2_wins >= 2:
        outcome = "ANOMALY_SIGNAL_SURVIVES_PC1_REMOVAL"
    elif abs(mean_e1_e2) < 1.0:
        outcome = "STATEALIGNED_SIGNAL_PC1_ENTANGLED"
    elif mean_e1_e2 > 0.0 and mean_e1_b0 <= -10.0:
        outcome = "ANOMALY_SIGNAL_EXISTS_BUT_FOREGROUND_MODE_CRITICAL"
    else:
        outcome = "PC1_DEBIAS_INCONCLUSIVE"
    return outcome, {
        "mean_E1_minus_B0_G": mean_e1_b0,
        "mean_E1_minus_E2_G": mean_e1_e2,
        "E1_wins_over_E2": e1_e2_wins,
    }


def write_report(summary_rows, variance_rows, outcome, stats, runtime_sec, reused):
    lines = [
        "# StateAligned PC1 Debias Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source only. The PC1 direction is recomputed in the 1024-d StateAligned patch-feature space from source normal images only. Target data are evaluation-only.",
        "",
        "| Dataset | B0 Original G | E1 PC1Removed TrueMask G | E2 PC1Removed CorruptedMask G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend(
        [
            "",
            f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
            f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
            "",
            "| Feature group | removed variance ratio | remaining variance ratio |",
            "|---|---:|---:|",
        ]
    )
    for r in variance_rows:
        lines.append(f"| {r['group']} | {r['removed_variance_ratio']:.6f} | {r['remaining_variance_ratio']:.6f} |")
    lines.extend(
        [
            "",
            f"Runtime: {runtime_sec:.1f}s. Checkpoints reused: {reused}.",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PC1Debias] device={device}", flush=True)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    q_pc1 = compute_statealigned_pc1(dataset, indices, patch, args.smoke)
    variance_rows = feature_variance(dataset, indices, patch, q_pc1, args.smoke)

    _s1, m1, c1, audit_true, reused1 = train_pc1_removed_statealigned("E1_PC1Removed-TrueMask", False, q_pc1, device, args.smoke)
    _s2, m2, c2, audit_corrupt, reused2 = train_pc1_removed_statealigned("E2_PC1Removed-CorruptedMask", True, q_pc1, device, args.smoke)
    raw, percat = evaluate_pc1_removed(
        {
            "E1_PC1Removed-TrueMask": m1,
            "E2_PC1Removed-CorruptedMask": m2,
        },
        q_pc1,
        device,
        args.smoke,
    )
    b0_raw, b0_percat = load_b0_reuse()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    outcome, decision_stats = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "FEATURE_VARIANCE.csv", variance_rows)
    write_csv(OUT / "E1_TRAINING_CURVES.csv", c1)
    write_csv(OUT / "E2_TRAINING_CURVES.csv", c2)
    write_csv(OUT / "MASK_CORRUPTION_AUDIT.csv", audit_corrupt)
    np.save(OUT / "q_pc1_statealigned1024.npy", q_pc1.astype(np.float32))
    runtime_sec = time.time() - start
    write_json(
        OUT / "OUTCOME.json",
        {
            "outcome": outcome,
            "decision_stats": decision_stats,
            "runtime_sec": runtime_sec,
            "device": device,
            "strict_source_only": True,
            "target_training": False,
            "target_statistics": False,
            "memory_used": False,
            "pc1_source": "Full VisA normal source spatial mean map in 1024-d StateAligned patch feature space",
            "checkpoints_reused": {"E1": bool(reused1), "E2": bool(reused2)},
        },
    )
    write_report(summary_rows, variance_rows, outcome, decision_stats, runtime_sec, {"E1": bool(reused1), "E2": bool(reused2)})
    print(f"[PC1Debias] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
