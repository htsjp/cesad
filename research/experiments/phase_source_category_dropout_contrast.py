import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import random_slots_like, source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np

OUT = Path("results/phase_source_category_dropout_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
TCM_OUT = Path("results/phase_transferable_contrast_memory_pilot/formal_corrected")
EPOCHS = 10


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


def load_b0_rows():
    raw, percat = [], []
    for name, dest in (("RAW_METRICS.csv", raw), ("PER_CATEGORY.csv", percat)):
        with (TCM_OUT / name).open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                if row["variant"] != "B0_GlobalContrast_ReconstructedAlpha":
                    continue
                out = dict(row)
                out["variant"] = "B0_GlobalContrast"
                for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                    try:
                        out[key] = float(out[key])
                    except ValueError:
                        out[key] = float("nan")
                out["mask_images"] = int(float(out["mask_images"]))
                dest.append(out)
    return raw, percat


def collect_group_samples(slots, labels_by_pos, patch, align_model, q_pc1, device):
    rng = np.random.default_rng(SEED)
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    out = {}
    rows = []
    for slot, positions in slots.items():
        by_cls = {0: [], 1: []}
        image_normals = 0
        image_anomalies = 0
        for pos in positions:
            y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
            if y.max() > 0:
                image_anomalies += 1
            else:
                image_normals += 1
            z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
            z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
            with torch.no_grad():
                z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
            for cls in (0, 1):
                ids = np.flatnonzero(y == cls)
                if ids.size == 0:
                    continue
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(part.shape[0] for part in by_cls[cls])
                if room <= 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                by_cls[cls].append(z[ids])
        xs, ys = [], []
        for cls in (0, 1):
            if not by_cls[cls]:
                continue
            arr = np.concatenate(by_cls[cls], axis=0).astype(np.float32)
            xs.append(arr)
            ys.extend([cls] * arr.shape[0])
        if not xs or len(set(ys)) < 2:
            raise RuntimeError(f"{slot}: insufficient binary samples.")
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.asarray(ys, dtype=np.float32)
        out[slot] = {"x": x, "y": y}
        rows.append({
            "partition": "group_samples",
            "group": slot,
            "source_images": int(len(positions)),
            "source_normal_images": int(image_normals),
            "source_anomaly_images": int(image_anomalies),
            "patch_samples": int(x.shape[0]),
            "positive_patch_samples": int(y.sum()),
            "negative_patch_samples": int((y == 0).sum()),
        })
    return out, rows


def sample_batch(groups, allowed, rng, total_batch):
    per_group = max(1, total_batch // max(1, len(allowed)))
    xs, ys = [], []
    for group in allowed:
        data = groups[group]
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            take = max(1, per_group // 2)
            choice = rng.choice(ids, size=take, replace=ids.size < take)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order]


def train_dropout_variant(variant, slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["dropout_rows"], payload["source_validation_rows"], True
    print(f"[CategoryDropout] prepare samples for {variant}", flush=True)
    groups, sample_rows = collect_group_samples(slots, labels_by_pos, patch, align_model, q_pc1, device)
    names = list(groups)
    total_samples = sum(len(groups[n]["y"]) for n in names)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    dropout_counts = defaultdict(int)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _step in range(steps_per_epoch):
            held = names[int(rng.integers(0, len(names)))]
            dropout_counts[held] += 1
            allowed = [n for n in names if n != held]
            xb, yb = sample_batch(groups, allowed, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            loss, bce, dice = balanced_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {"variant": variant, "epoch": epoch, "loss": float(arr[:, 0].mean()), "bce": float(arr[:, 1].mean()), "dice": float(arr[:, 2].mean())}
        curves.append(row)
        print(f"[CategoryDropout] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    dropout_rows = []
    for row in sample_rows:
        group = row["group"]
        out = dict(row)
        out.update({"variant": variant, "dropout_steps": int(dropout_counts[group]), "total_optimizer_steps": int(EPOCHS * steps_per_epoch)})
        dropout_rows.append(out)
    source_rows = source_validation(variant, groups, v, m_c, alpha)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "dropout_rows": dropout_rows, "source_validation_rows": source_rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, dropout_rows, source_rows, False


def source_validation(variant, groups, v, m_c, alpha):
    rows = []
    m = torch.tensor(m_c, dtype=torch.float32)
    vt = torch.tensor(v, dtype=torch.float32)
    for group, data in groups.items():
        z = torch.tensor(data["x"], dtype=torch.float32)
        with torch.no_grad():
            score = contrast_logits(z, m, alpha, vt).cpu().numpy()
        y = data["y"].astype(np.int64)
        pauroc = float(roc_auc_score(y, score) * 100.0) if np.unique(y).size == 2 else float("nan")
        pap = float(average_precision_score(y, score) * 100.0) if np.unique(y).size == 2 else float("nan")
        rows.append({"variant": variant, "source_group": group, "P-AUROC": pauroc, "P-AP": pap, "G_pixel": 0.5 * (pauroc + pap), "samples": int(len(y)), "positive_samples": int(y.sum())})
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_GlobalContrast")]
        e1 = by[(ds, "E1_TrueCategoryDropout")]
        e2 = by[(ds, "E2_MatchedRandomGroupDropout")]
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


def contrast_geometry(v_b0, v_e1, v_e2):
    return [{
        "cos_v_B0_v_E1": float(unit_np(v_b0) @ unit_np(v_e1)),
        "cos_v_B0_v_E2": float(unit_np(v_b0) @ unit_np(v_e2)),
        "cos_v_E1_v_E2": float(unit_np(v_e1) @ unit_np(v_e2)),
        "norm_v_B0": float(np.linalg.norm(v_b0)),
        "norm_v_E1": float(np.linalg.norm(v_e1)),
        "norm_v_E2": float(np.linalg.norm(v_e2)),
    }]


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    stable = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0
    if stable and float(e1_e2.mean()) >= 0.5:
        outcome = "SOURCE_CATEGORY_DROPOUT_SUPPORTED"
    elif stable and float(e1_e2.mean()) < 0.5:
        outcome = "GROUP_DROPOUT_REGULARIZATION_NOT_SEMANTIC"
    else:
        outcome = "SOURCE_CATEGORY_DROPOUT_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_report(summary_rows, outcome, stats, geom, runtime):
    lines = [
        "# Source-Category Dropout Contrast Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Strict source-only Full VisA -> MVTec / BTAD / CVC-ClinicDB. E1/E2 train one shared contrast only; target inference has no memory, routing, prompt, DINO, or target statistics.",
        "",
        "| Dataset | B0 G | E1 True Dropout G | E2 Random Dropout G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        f"Contrast cosines: B0/E1={geom[0]['cos_v_B0_v_E1']:.6f}, B0/E2={geom[0]['cos_v_B0_v_E2']:.6f}, E1/E2={geom[0]['cos_v_E1_v_E2']:.6f}.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CategoryDropout] device={device}", flush=True)
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
    cat_slots = source_slots_by_category(dataset, indices, args.smoke)
    random_slots = random_slots_like(cat_slots, dataset, indices, SEED)

    v_e1, drop_e1, source_e1, reused_e1 = train_dropout_variant("E1_TrueCategoryDropout", cat_slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, args.smoke)
    v_e2, drop_e2, source_e2, reused_e2 = train_dropout_variant("E2_MatchedRandomGroupDropout", random_slots, labels_by_pos, patch, align_model, m_c, alpha, init_v, q_pc1, device, args.smoke)

    models = {
        "E1_TrueCategoryDropout": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_MatchedRandomGroupDropout": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    geom = contrast_geometry(init_v, v_e1, v_e2)
    outcome, stats = decide(summary_rows)
    runtime = time.time() - start

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "DROPOUT_AUDIT.csv", drop_e1 + drop_e2)
    write_csv(OUT / "SOURCE_PER_CATEGORY_VALIDATION.csv", source_e1 + source_e2)
    np.savez_compressed(OUT / "contrast_vectors.npz", B0=init_v.astype(np.float32), E1=v_e1.astype(np.float32), E2=v_e2.astype(np.float32))
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
        "alpha": alpha,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, stats, geom, runtime)
    print(f"[CategoryDropout] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
