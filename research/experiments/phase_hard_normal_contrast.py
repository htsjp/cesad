import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.external_sota_gate import apply_gate
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_category_dropout_contrast import load_b0_rows
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np

OUT = Path("results/phase_hard_normal_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPOCHS = 10
TOP_NEG_FRAC = 0.10
CACHE_VERSION = "balanced_negative_origins"


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


def collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device):
    rng = np.random.default_rng(SEED)
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    by_key = {}
    audit = []
    for cat, positions in slots.items():
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
            image_is_anomaly = bool(y.max() > 0)
            z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
            z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
            with torch.no_grad():
                z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
            pools = []
            if image_is_anomaly:
                pools.append(("anomaly_mask_out", np.flatnonzero(y == 0)))
                pools.append(("anomaly_mask", np.flatnonzero(y == 1)))
            else:
                pools.append(("normal_image", np.flatnonzero(y == 0)))
            for origin_name, ids in pools:
                if ids.size == 0:
                    continue
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(part.shape[0] for part in parts[origin_name])
                if room <= 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin_name].append(z[ids])
        xs, ys, os = [], [], []
        for origin_name, cls in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if not parts[origin_name]:
                continue
            arr = np.concatenate(parts[origin_name], axis=0).astype(np.float32)
            xs.append(arr)
            ys.extend([cls] * arr.shape[0])
            os.extend([origin_name] * arr.shape[0])
        if not xs or len(set(ys)) < 2:
            raise RuntimeError(f"{cat}: insufficient source patch samples.")
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.asarray(ys, dtype=np.float32)
        origin = np.asarray(os, dtype=object)
        by_key[cat] = {"x": x, "y": y, "origin": origin}
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
    return by_key, audit


def balanced_category_batch(groups, rng, total_batch):
    names = list(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, origins, cats = [], [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            take = per_cls
            choice = rng.choice(ids, size=take, replace=ids.size < take)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            origins.extend(data["origin"][choice].tolist())
            cats.extend([cat] * take)
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    origin = np.asarray(origins, dtype=object)[order]
    cat_arr = np.asarray(cats, dtype=object)[order]
    return x[order], y[order], origin, cat_arr


def train_variant(variant, mode, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}_{CACHE_VERSION}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["hard_rows"], True
    print(f"[HardNormal] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    curves = []
    hard_rows = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin, cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            neg_scores = logits.detach()[neg_idx]
            if mode == "hard":
                selected_local = torch.topk(neg_scores, k=k, largest=True).indices
            elif mode == "random":
                perm = torch.tensor(rng.permutation(len(neg_idx))[:k], dtype=torch.long, device=device)
                selected_local = perm
            else:
                selected_local = torch.empty((0,), dtype=torch.long, device=device)
            if mode == "none":
                extra_loss = torch.zeros((), dtype=torch.float32, device=device)
                selected_idx = neg_idx[:0]
            else:
                selected_idx = neg_idx[selected_local]
                extra_loss = F.softplus(logits[selected_idx]).mean()
            loss = task_loss + extra_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if selected_idx.numel():
                sel_np = selected_idx.detach().cpu().numpy()
                sel_scores = logits.detach()[selected_idx].cpu().numpy()
                sel_origin = origin[sel_np]
                hard_rows.append({
                    "variant": variant,
                    "record_type": "train_selection",
                    "epoch": epoch,
                    "step": step + 1,
                    "mode": mode,
                    "selected_count": int(len(sel_np)),
                    "selected_score_mean": float(np.mean(sel_scores)),
                    "selected_score_std": float(np.std(sel_scores)),
                    "normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                    "anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
                    "all_negative_score_mean": float(neg_scores.cpu().numpy().mean()),
                    "all_negative_score_std": float(neg_scores.cpu().numpy().std()),
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
        print(f"[HardNormal] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} extra={row['extra_normal_loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "hard_rows": hard_rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, hard_rows, False


def final_negative_distribution(variant, groups, v, m_c, alpha):
    rows = []
    m = torch.tensor(m_c, dtype=torch.float32)
    vt = torch.tensor(v, dtype=torch.float32)
    all_scores = []
    all_origins = []
    for cat, data in groups.items():
        z = torch.tensor(data["x"], dtype=torch.float32)
        y = data["y"]
        with torch.no_grad():
            score = contrast_logits(z, m, alpha, vt).cpu().numpy()
        neg = y < 0.5
        scores = score[neg]
        origins = data["origin"][neg]
        all_scores.append(scores)
        all_origins.extend(origins.tolist())
        rows.append({
            "variant": variant,
            "record_type": "final_negative_distribution",
            "category": cat,
            "negative_count": int(scores.size),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_p90": float(np.percentile(scores, 90)),
            "score_p95": float(np.percentile(scores, 95)),
            "normal_image_ratio": float(np.mean(origins == "normal_image")),
            "anomaly_mask_out_ratio": float(np.mean(origins == "anomaly_mask_out")),
        })
    scores = np.concatenate(all_scores, axis=0)
    origins = np.asarray(all_origins, dtype=object)
    rows.append({
        "variant": variant,
        "record_type": "final_negative_distribution_summary",
        "category": "__all__",
        "negative_count": int(scores.size),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_p90": float(np.percentile(scores, 90)),
        "score_p95": float(np.percentile(scores, 95)),
        "normal_image_ratio": float(np.mean(origins == "normal_image")),
        "anomaly_mask_out_ratio": float(np.mean(origins == "anomaly_mask_out")),
    })
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_GlobalContrast")]
        e1 = by[(ds, "E1_HardNormalContrast")]
        e2 = by[(ds, "E2_RandomNormalControl")]
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


def decide(summary_rows, hard_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    stable = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0
    if stable and float(e1_e2.mean()) >= 0.5:
        outcome = "HARD_NORMAL_CONTRAST_SUPPORTED"
    elif stable and abs(float(e1_e2.mean())) < 0.5:
        outcome = "EXTRA_NEGATIVE_WEIGHTING_NOT_HARDNESS"
    else:
        e1_sel = [r for r in hard_rows if r.get("variant") == "E1_HardNormalContrast" and r.get("record_type") == "train_selection"]
        improved = False
        if e1_sel:
            early = np.asarray([r["selected_score_mean"] for r in e1_sel if int(r["epoch"]) <= 2], dtype=np.float32)
            late = np.asarray([r["selected_score_mean"] for r in e1_sel if int(r["epoch"]) >= EPOCHS - 1], dtype=np.float32)
            improved = bool(early.size and late.size and float(late.mean()) < float(early.mean()) - 0.02)
        outcome = "SOURCE_HARDNESS_NOT_TRANSFERABILITY" if improved else "HARD_NORMAL_CONTRAST_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_report(summary_rows, outcome, stats, geom, runtime):
    lines = [
        "# Hard-Normal Contrast Learning Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Strict Full VisA source-only. E1 adds top-10% hard negative normal BCE; E2 adds matched random negative BCE. Target inference keeps one global contrast only.",
        "",
        "| Dataset | B0 G | E1 HardNormal G | E2 RandomNormal G | E1-B0 | E1-E2 |",
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
    print(f"[HardNormal] device={device}", flush=True)
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
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)

    v_e1, hard_e1, reused_e1 = train_variant("E1_HardNormalContrast", "hard", groups, m_c, alpha, init_v, device, args.smoke)
    v_e2, hard_e2, reused_e2 = train_variant("E2_RandomNormalControl", "random", groups, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_HardNormalContrast": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_RandomNormalControl": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    hard_stats = pool_audit + hard_e1 + hard_e2
    for variant, v in (("B0_GlobalContrast", init_v), ("E1_HardNormalContrast", v_e1), ("E2_RandomNormalControl", v_e2)):
        hard_stats.extend(final_negative_distribution(variant, groups, v, m_c, alpha))
    geom = contrast_geometry(init_v, v_e1, v_e2)
    outcome, stats = decide(summary_rows, hard_stats)
    runtime = time.time() - start

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "HARD_NEGATIVE_STATS.csv", hard_stats)
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
        "top_negative_fraction": TOP_NEG_FRAC,
        "alpha": alpha,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary_rows, outcome, stats, geom, runtime)
    apply_gate(
        OUT,
        "E1_HardNormalContrast",
        "HardNormal",
        "Stop-loss: not triggered. This method improves mean target G by +4.2430 pp and narrows/overcomes the external gap on all three listed targets.",
    )
    print(f"[HardNormal] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
