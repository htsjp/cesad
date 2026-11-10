import argparse
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

from experiments.phase_li_asymmetric_state import load_full_visa_adapter
from experiments.phase_spatial_supervision_transfer_audit import (
    corrupt_masks,
    fast_patch_grid_eval,
    target_cells,
    union_source_records,
)
from experiments.phase_defect_query_transfer_pilot import light_fold
from experiments.phase_liv_memory_support_modes import ArrayConcat, npz_array_mmap
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
    FEATURE_CACHE,
    LR as STATE_LR,
    SIDE,
    centroids,
    norm_np,
    train_state_vectors,
)

OUT = Path("results/phase_statealigned_supervision_necessity/formal")
PC1_OUT = Path("results/phase_pc1_anomaly_orientation/formal")


def load_full_visa_source_lazy(smoke):
    dataset_name, dataset, train_indices, held_indices = light_fold("D2", 0)
    cache = FEATURE_CACHE / "D2_fold0_clip.npz"
    train_cls = npz_array_mmap(cache, "train_cls")
    held_cls = npz_array_mmap(cache, "held_cls")
    cls = np.concatenate([np.asarray(train_cls), np.asarray(held_cls)], axis=0).astype(np.float32)
    patch = ArrayConcat([npz_array_mmap(cache, "train_patch"), npz_array_mmap(cache, "held_patch")])
    indices = list(train_indices) + list(held_indices)
    y_img = np.asarray([int(dataset.data_all[int(i)]["anomaly"]) for i in indices], dtype=np.int64)
    cat_img = np.asarray([str(dataset.data_all[int(i)]["cls_name"]) for i in indices])
    if smoke:
        keep = min(96, len(indices))
        return dataset_name, dataset, indices[:keep], cls[:keep], patch, y_img[:keep], cat_img[:keep]
    return dataset_name, dataset, indices, cls, patch, y_img, cat_img


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def collect_corrupted_patch_samples(dataset, indices, patches, labels_by_pos, source_positions, cap_per_class_per_category, seed):
    rng = np.random.default_rng(seed)
    by_key = defaultdict(list)
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    for source_pos in source_positions:
        idx = indices[int(source_pos)]
        item = dataset[int(idx)]
        cat = str(item["cls_name"])
        y = np.asarray(labels_by_pos.get(int(source_pos), zero), dtype=np.int64)
        x = np.asarray(patches[int(source_pos)], dtype=np.float32)
        for cls in (0, 1):
            ids = np.flatnonzero(y == cls)
            if ids.size == 0:
                continue
            key = (cat, cls)
            room = cap_per_class_per_category - sum(part.shape[0] for part in by_key[key])
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
        dim = int(patches.shape[-1])
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.asarray([])
    return np.concatenate(feats, axis=0), np.asarray(ys, dtype=np.int64), np.asarray(cats)


def corrupted_labels_for_source(smoke):
    dataset_u, indices_u, _patch_u, records = union_source_records(smoke)
    masks, audit = corrupt_masks(records)
    labels_by_pos = {}
    for i, rec in enumerate(records):
        if bool(rec["normal"]):
            labels_by_pos[int(rec["pos"])] = np.zeros((SIDE * SIDE,), dtype=np.int64)
        else:
            labels_by_pos[int(rec["pos"])] = (np.asarray(masks[i], dtype=np.float32) > 0.5).astype(np.int64)
    return dataset_u, indices_u, records, labels_by_pos, audit


def train_corrupted_statealigned(device, smoke=False):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"e1_corrupted_statealigned_{suffix}.pt"
    curves_path = OUT / f"E1_CORRUPTED_TRAINING_CURVES_{suffix}.csv"
    dataset_name, dataset, indices, cls, patch, y_img, cat_img = load_full_visa_source_lazy(False)
    dataset_u, indices_u, records, labels_by_pos, audit = corrupted_labels_for_source(smoke)
    source_positions = sorted(labels_by_pos) if smoke else list(range(len(indices)))
    if any(int(indices_u[p]) != int(indices[p]) for p in source_positions):
        raise RuntimeError("Corrupted-mask source order does not match Full VisA source cache order.")
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

    print("[SupervisionNecessity] train E1 corrupted-mask StateAligned states", flush=True)
    patch_x, patch_y, patch_cat = collect_corrupted_patch_samples(
        dataset,
        indices,
        patch,
        labels_by_pos,
        source_positions,
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

    print("[SupervisionNecessity] train E1 corrupted-mask rank4 adapter", flush=True)
    model = AdapterToState(patch_x.shape[1], patch_states).to(device)
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
            "variant": "E1_CorruptedMask_StateAligned",
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "bce": float(arr[:, 1].mean()),
            "dice": float(arr[:, 2].mean()),
        }
        adapter_curves.append(row)
        print(f"[SupervisionNecessity] adapter epoch {epoch}/{ADAPTER_EPOCHS} loss={row['loss']:.4f}", flush=True)
    model = model.eval()

    curves = []
    for row in state_curves:
        rr = dict(row)
        rr["stage"] = "state"
        rr["variant"] = "E1_CorruptedMask_StateAligned"
        curves.append(rr)
    curves.extend(adapter_curves)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(curves_path, curves)
    torch.save(
        {
            "model": model.state_dict(),
            "img_states": img_states,
            "patch_states": patch_states,
            "curves": curves,
            "source_dataset": dataset_name,
            "smoke": bool(smoke),
            "state_epochs": STATE_EPOCHS,
            "state_lr": STATE_LR,
            "adapter_epochs": ADAPTER_EPOCHS,
            "adapter_lr": ADAPTER_LR,
            "adapter_weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        ckpt,
    )
    return patch_states, model, curves, audit, False


@torch.no_grad()
def score_adapter_batch(patch, model, device, smoke=False, batch_images=8):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    model.eval()
    for start in range(0, n, batch_images):
        arr = np.asarray(patch[start : start + batch_images], dtype=np.float32)
        b = arr.shape[0]
        flat = norm_np(arr.reshape(-1, arr.shape[-1]))
        x = torch.tensor(flat, dtype=torch.float32, device=device)
        logits, _ = model.logits(x)
        s = torch.sigmoid(logits).detach().cpu().numpy().reshape(b, SIDE, SIDE).astype(np.float32)
        maps[start : start + b] = s.astype(np.float16)
        image[start : start + b] = np.asarray([top1_image_score(m) for m in s], dtype=np.float32)
    return image, maps


def evaluate_statealigned_variants(models, device, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, model in models.items():
            print(f"[SupervisionNecessity] eval {variant} on {cell_id}", flush=True)
            image, maps = score_adapter_batch(patch, model, device, smoke=smoke)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update(
                {
                    "dataset": cell_id,
                    "variant": variant,
                    "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"])),
                }
            )
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = (
                    0.5 * (float(row["P-AUROC"]) + float(row["P-AP"]))
                    if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"]))
                    else float("nan")
                )
                percat.append(row)
    return raw, percat


def load_pc1_reuse():
    raw = []
    with (PC1_OUT / "RAW_METRICS.csv").open("r", encoding="utf-8", newline="") as fp:
        import csv

        for row in csv.DictReader(fp):
            if row["variant"] != "E2_NormalOnlyOrientedPC1":
                continue
            out = dict(row)
            out["variant"] = "E2_Normal-PC1"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            raw.append(out)
    percat = []
    with (PC1_OUT / "PER_CATEGORY.csv").open("r", encoding="utf-8", newline="") as fp:
        import csv

        for row in csv.DictReader(fp):
            if row["variant"] != "E2_NormalOnlyOrientedPC1":
                continue
            out = dict(row)
            out["variant"] = "E2_Normal-PC1"
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
        b0 = by[(ds, "B0_TrueMask-StateAligned")]
        e1 = by[(ds, "E1_CorruptedMask-StateAligned")]
        e2 = by[(ds, "E2_Normal-PC1")]
        rows.append(
            {
                "dataset": ds,
                "B0_P-AUROC": float(b0["P-AUROC"]),
                "B0_P-AP": float(b0["P-AP"]),
                "B0_G": float(b0["G_pixel"]),
                "E1_P-AUROC": float(e1["P-AUROC"]),
                "E1_P-AP": float(e1["P-AP"]),
                "E1_G": float(e1["G_pixel"]),
                "E2_P-AUROC": float(e2["P-AUROC"]),
                "E2_P-AP": float(e2["P-AP"]),
                "E2_G": float(e2["G_pixel"]),
                "B0_minus_E1_G": float(b0["G_pixel"]) - float(e1["G_pixel"]),
                "B0_minus_E2_G": float(b0["G_pixel"]) - float(e2["G_pixel"]),
                "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            }
        )
    return rows


def decide(summary_rows):
    b0_e1 = np.asarray([r["B0_minus_E1_G"] for r in summary_rows], dtype=np.float32)
    b0_e2 = np.asarray([r["B0_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    mean_b0_e1 = float(np.mean(b0_e1))
    b0_wins = int(np.sum(b0_e1 > 0))
    close_b0_e1 = abs(mean_b0_e1) < 1.0
    close_to_pc1 = float(np.mean(np.abs(b0_e2))) < 3.0 and float(np.mean(np.abs(e1_e2))) < 3.0
    above_pc1 = float(np.mean(b0_e2)) >= 3.0 and float(np.mean(e1_e2)) >= 3.0
    if b0_wins >= 2 and mean_b0_e1 >= 1.0:
        outcome = "STATEALIGNED_TRUE_ANOMALY_SUPERVISION_SUPPORTED"
    elif close_b0_e1 and close_to_pc1:
        outcome = "STATEALIGNED_FOREGROUND_CONFOUND_SUSPECTED"
    elif close_b0_e1 and above_pc1:
        outcome = "ANOMALY_IMAGE_SIGNAL_WITHOUT_SPATIAL_CORRESPONDENCE"
    else:
        outcome = "STATEALIGNED_SUPERVISION_CAUSALITY_INCONCLUSIVE"
    return outcome, {
        "mean_B0_minus_E1_G": mean_b0_e1,
        "B0_wins_over_E1": b0_wins,
        "mean_B0_minus_E2_G": float(np.mean(b0_e2)),
        "mean_E1_minus_E2_G": float(np.mean(e1_e2)),
        "close_threshold_pp": 1.0,
        "pc1_clear_gap_threshold_pp": 3.0,
    }


def write_report(path, summary_rows, outcome, decision_stats, runtime_sec, reused_e1):
    lines = [
        "# StateAligned Anomaly-Supervision Necessity Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source only; targets MVTec, BTAD, and CVC-ClinicDB are evaluation-only. B0 reuses the formal TrueMask StateAligned checkpoint; E1 keeps the same StateAligned architecture/training protocol but replaces source anomaly masks with the fixed area-matched relocated masks; E2 reuses the completed normal-only PC1 baseline.",
        "",
        "| Dataset | B0 TrueMask G | E1 CorruptedMask G | E2 Normal-PC1 G | B0-E1 | B0-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['B0_minus_E1_G']:+.4f} | {r['B0_minus_E2_G']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Mean B0-E1 G: {decision_stats['mean_B0_minus_E1_G']:+.4f} pp across 3 targets.",
            f"Mean B0-E2 G: {decision_stats['mean_B0_minus_E2_G']:+.4f} pp; mean E1-E2 G: {decision_stats['mean_E1_minus_E2_G']:+.4f} pp.",
            "",
            "Interpretation: the decision is made only from the three fixed variants requested. No Memory, extra adapter, target statistics, or post-hoc tuning was used.",
            "",
            f"Runtime: {runtime_sec:.1f}s. E1 checkpoint reused: {bool(reused_e1)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SupervisionNecessity] device={device}", flush=True)

    _states_b0, b0_model = load_full_visa_adapter(device)
    _states_e1, e1_model, curves, corruption_audit, reused_e1 = train_corrupted_statealigned(device, smoke=args.smoke)
    raw, percat = evaluate_statealigned_variants(
        {
            "B0_TrueMask-StateAligned": b0_model,
            "E1_CorruptedMask-StateAligned": e1_model,
        },
        device,
        args.smoke,
    )
    pc1_raw, pc1_percat = load_pc1_reuse()
    if args.smoke:
        pc1_raw = pc1_raw[:3]
        pc1_percat = pc1_percat[:3]
    raw.extend(pc1_raw)
    percat.extend(pc1_percat)
    summary_rows = summarize(raw)
    outcome, decision_stats = decide(summary_rows)

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "MASK_CORRUPTION_AUDIT.csv", corruption_audit)
    if curves:
        write_csv(OUT / "E1_CORRUPTED_TRAINING_CURVES.csv", curves)
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
            "variants": [
                "B0_TrueMask-StateAligned",
                "E1_CorruptedMask-StateAligned",
                "E2_Normal-PC1",
            ],
            "pc1_reuse_dir": str(PC1_OUT),
            "e1_checkpoint_reused": bool(reused_e1),
        },
    )
    write_report(OUT / "REPORT.md", summary_rows, outcome, decision_stats, runtime_sec, reused_e1)
    print(f"[SupervisionNecessity] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
