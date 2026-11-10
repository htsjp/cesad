import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.external_sota_gate import apply_gate
from experiments.phase_hard_normal_contrast import collect_source_patch_pool
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import OUT as PC1_DEBIAS_OUT, evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_source_episodic_meta_generalization/formal")
EPOCHS = 10
TOP_NEG_FRAC = 0.10
META_ALPHA = 0.001


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


def load_hardnormal_b0_rows():
    import pandas as pd

    base = Path("results/phase_hard_normal_contrast/formal")
    raw = pd.read_csv(base / "RAW_METRICS.csv")
    per = pd.read_csv(base / "PER_CATEGORY.csv")
    raw = raw[raw["variant"] == "E1_HardNormalContrast"].copy()
    per = per[per["variant"] == "E1_HardNormalContrast"].copy()
    raw["variant"] = "B0_HardNormal"
    per["variant"] = "B0_HardNormal"
    return raw.to_dict("records"), per.to_dict("records")


def gp(row):
    return 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"]))


def sample_batch(groups, names, rng, total_batch):
    per_group = max(2, total_batch // max(1, len(names)))
    xs, ys = [], []
    for name in names:
        data = groups[name]
        per_cls = max(1, per_group // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order]


def random_group_slots_like(slots, groups, seed):
    names = list(groups)
    counts = {n: len(groups[n]["y"]) for n in names}
    all_x = np.concatenate([groups[n]["x"] for n in names], axis=0)
    all_y = np.concatenate([groups[n]["y"] for n in names], axis=0)
    rng = np.random.default_rng(seed)
    pos_ids = rng.permutation(np.flatnonzero(all_y == 1))
    neg_ids = rng.permutation(np.flatnonzero(all_y == 0))
    out = {}
    pos_off = 0
    neg_off = 0
    for name in names:
        y = groups[name]["y"]
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        ids = np.concatenate([pos_ids[pos_off : pos_off + n_pos], neg_ids[neg_off : neg_off + n_neg]])
        rng.shuffle(ids)
        out[f"random_{name}"] = {"x": all_x[ids], "y": all_y[ids]}
        pos_off += n_pos
        neg_off += n_neg
    return out


def hard_indices_from_logits(logits, y):
    neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    if neg_idx.numel() == 0:
        return neg_idx
    k = max(1, int(math.ceil(float(neg_idx.numel()) * TOP_NEG_FRAC)))
    return neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]


def objective_parts(x, y, raw_v, m, alpha):
    v = F.normalize(raw_v, dim=-1)
    logits = contrast_logits(x, m, alpha, v)
    task, bce, dice = balanced_loss(logits, y)
    hard_idx = hard_indices_from_logits(logits, y)
    hard = F.softplus(logits[hard_idx]).mean() if hard_idx.numel() else torch.zeros((), dtype=torch.float32, device=x.device)
    return task, hard, task + hard, bce, dice, logits


def grad_cos(loss_a, loss_b, raw_v):
    ga = torch.autograd.grad(loss_a, raw_v, retain_graph=True, create_graph=False)[0]
    gb = torch.autograd.grad(loss_b, raw_v, retain_graph=True, create_graph=False)[0]
    return float(F.cosine_similarity(ga.detach(), gb.detach(), dim=0).cpu()), float(torch.linalg.norm(ga.detach()).cpu()), float(torch.linalg.norm(gb.detach()).cpu())


def tensor_batch(xb, yb, device):
    return torch.tensor(xb, dtype=torch.float32, device=device), torch.tensor(yb, dtype=torch.float32, device=device)


def meta_step_category(groups, names, step_idx, rng, raw_v, m, alpha, device, mode):
    if mode == "category":
        held = names[step_idx % len(names)]
        train_names = [n for n in names if n != held]
        val_names = [held]
        xtr, ytr = sample_batch(groups, train_names, rng, BATCH_SIZE)
        xv, yv = sample_batch(groups, val_names, rng, max(512, BATCH_SIZE // len(names)))
    elif mode == "random":
        all_x, all_y = sample_batch(groups, names, rng, BATCH_SIZE)
        val_n = max(512, all_x.shape[0] // max(2, len(names)))
        order = rng.permutation(all_x.shape[0])
        val_ids = order[:val_n]
        tr_ids = order[val_n:]
        xtr, ytr = all_x[tr_ids], all_y[tr_ids]
        xv, yv = all_x[val_ids], all_y[val_ids]
        held = "random_meta_val"
    else:
        raise ValueError(mode)
    xtr_t, ytr_t = tensor_batch(xtr, ytr, device)
    xv_t, yv_t = tensor_batch(xv, yv, device)
    task_tr, hard_tr, ltr, _bce_tr, _dice_tr, _logits_tr = objective_parts(xtr_t, ytr_t, raw_v, m, alpha)
    g_tr = torch.autograd.grad(ltr, raw_v, create_graph=True, retain_graph=True)[0]
    raw_v_prime = raw_v - META_ALPHA * g_tr
    task_val_before, hard_val_before, lval_before, _bce_b, _dice_b, _logits_b = objective_parts(xv_t, yv_t, raw_v, m, alpha)
    _task_val, _hard_val, lmeta, _bce_v, _dice_v, _logits_v = objective_parts(xv_t, yv_t, raw_v_prime, m, alpha)
    with torch.no_grad():
        delta_val = float((lval_before - lmeta).detach().cpu())
    g_val = torch.autograd.grad(lval_before, raw_v, retain_graph=True, create_graph=False)[0]
    cos_tr_val = float(F.cosine_similarity(g_tr.detach(), g_val.detach(), dim=0).cpu())
    return lmeta, {
        "heldout_meta_category": held,
        "meta_train_loss": float(ltr.detach().cpu()),
        "meta_val_loss_before": float(lval_before.detach().cpu()),
        "meta_val_loss_virtual": float(lmeta.detach().cpu()),
        "virtual_update_val_improvement": delta_val,
        "cos_g_train_g_val": cos_tr_val,
        "meta_train_task_loss": float(task_tr.detach().cpu()),
        "meta_train_hard_loss": float(hard_tr.detach().cpu()),
        "meta_val_task_loss_before": float(task_val_before.detach().cpu()),
        "meta_val_hard_loss_before": float(hard_val_before.detach().cpu()),
    }


def train_meta_variant(variant, mode, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["curves"], payload["diag"], True
    print(f"[MetaGeneralization] train {variant}", flush=True)
    names = sorted(groups)
    total_samples = sum(len(groups[n]["y"]) for n in names)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    raw_v = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([raw_v], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    curves = []
    diag = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb = sample_batch(groups, names, rng, BATCH_SIZE)
            x, y = tensor_batch(xb, yb, device)
            task, hard, l0_all, bce, dice, _logits = objective_parts(x, y, raw_v, m, alpha)
            lmeta, meta_row = meta_step_category(groups, names, (epoch - 1) * steps_per_epoch + step, rng, raw_v, m, alpha, device, mode)
            task_cos, task_norm, hard_norm = grad_cos(task, hard, raw_v)
            total = l0_all + lmeta
            opt.zero_grad()
            total.backward()
            opt.step()
            meta_row.update({
                "variant": variant,
                "epoch": epoch,
                "step": step + 1,
                "L0_all": float(l0_all.detach().cpu()),
                "L_total": float(total.detach().cpu()),
                "task_loss": float(task.detach().cpu()),
                "hard_loss": float(hard.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "dice": float(dice.detach().cpu()),
                "cos_task_grad_hard_grad": task_cos,
                "task_grad_norm": task_norm,
                "hard_grad_norm": hard_norm,
            })
            diag.append(meta_row)
            vals.append((meta_row["L_total"], meta_row["L0_all"], meta_row["meta_val_loss_virtual"], meta_row["virtual_update_val_improvement"], meta_row["cos_g_train_g_val"]))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "L0_all": float(arr[:, 1].mean()),
            "meta_val_loss": float(arr[:, 2].mean()),
            "virtual_update_val_improvement": float(arr[:, 3].mean()),
            "cos_g_train_g_val": float(arr[:, 4].mean()),
        }
        curves.append(row)
        print(f"[MetaGeneralization] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} meta={row['meta_val_loss']:.4f} dval={row['virtual_update_val_improvement']:+.6f}", flush=True)
    with torch.no_grad():
        v = F.normalize(raw_v, dim=-1).detach().cpu().numpy().astype(np.float32)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "curves": curves, "diag": diag}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    write_csv(OUT / f"{variant}_META_DIAGNOSTICS.csv", diag)
    return v, curves, diag, False


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_CategoryMeta")]
        e2 = by[(ds, "E2_RandomMeta")]
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
    by = {r["dataset"]: r for r in summary_rows}
    improved = int(np.sum(e1_b0 > 0))
    mean_b0 = float(e1_b0.mean())
    mean_e2 = float(e1_e2.mean())
    mt_or_clinic = by["MVTec"]["E1_minus_B0_G"] >= 1.5 or by["CVC-ClinicDB"]["E1_minus_B0_G"] >= 1.5
    btad_ok = by["BTAD"]["E1_minus_B0_G"] >= -1.0
    e2_b0 = np.asarray([r["E2_G"] - r["B0_G"] for r in summary_rows], dtype=np.float32)
    if improved >= 2 and mean_b0 >= 1.0 and mean_e2 >= 0.5 and mt_or_clinic and btad_ok:
        outcome = "SOURCE_EPISODIC_META_GENERALIZATION_SUPPORTED"
    elif int(np.sum(e2_b0 > 0)) >= 2 and float(e2_b0.mean()) >= 1.0 and mean_e2 < 0.5:
        outcome = "GENERIC_LOOKAHEAD_OPTIMIZATION_SUPPORTED_NOT_CATEGORY_SPECIFIC"
    elif by["BTAD"]["E1_minus_B0_G"] > 1.0 and by["MVTec"]["E1_minus_B0_G"] <= 0.5 and by["CVC-ClinicDB"]["E1_minus_B0_G"] <= 0.5:
        outcome = "META_GENERALIZATION_DOMAIN_DEPENDENT"
    elif mean_e2 > 0 and mean_b0 <= 0:
        outcome = "META_CATEGORY_SIGNAL_NO_TARGET_GAIN"
    else:
        outcome = "SOURCE_EPISODIC_META_GENERALIZATION_NO_SIGNAL"
    return outcome, {
        "targets_improved_E1_over_B0": improved,
        "mean_E1_minus_B0_G": mean_b0,
        "mean_E1_minus_E2_G": mean_e2,
        "MVTec_E1_minus_B0_G": float(by["MVTec"]["E1_minus_B0_G"]),
        "BTAD_E1_minus_B0_G": float(by["BTAD"]["E1_minus_B0_G"]),
        "ClinicDB_E1_minus_B0_G": float(by["CVC-ClinicDB"]["E1_minus_B0_G"]),
        "BTAD_drop_ok": bool(btad_ok),
    }


def sota_gap(raw, variant):
    refs = {
        "MVTec": {"P-AUROC": 92.5, "P-AP": 45.7, "G_pixel": 69.10},
        "BTAD": {"P-AUROC": 96.8, "P-AP": 50.4, "G_pixel": 73.60},
        "CVC-ClinicDB": {"P-AUROC": 89.7, "P-AP": 49.9, "G_pixel": 69.80},
    }
    rows = []
    for r in raw:
        if r["variant"] != variant or r["dataset"] not in refs:
            continue
        ref = refs[r["dataset"]]
        rows.append({
            "dataset": r["dataset"],
            "variant": variant,
            "current_P-AUROC": float(r["P-AUROC"]),
            "current_P-AP": float(r["P-AP"]),
            "current_G": float(r["G_pixel"]),
            "reference": "MoECLIP OpenCLIP ViT-L/14-336 518x518",
            "reference_P-AUROC": ref["P-AUROC"],
            "reference_P-AP": ref["P-AP"],
            "reference_G": ref["G_pixel"],
            "Delta_P_AUROC": float(r["P-AUROC"]) - ref["P-AUROC"],
            "Delta_P_AP": float(r["P-AP"]) - ref["P-AP"],
            "Delta_G": float(r["G_pixel"]) - ref["G_pixel"],
            "comparison_status": "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS",
        })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(outcome, stats, summary_rows, geom, runtime):
    lines = [
        "# Source-Episodic Meta-Generalized HardNormal",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA -> MVTec / BTAD / CVC-ClinicDB. Frozen CLIP path and existing StateAligned/HardNormal protocol are kept; E1/E2 add second-order virtual meta-validation loss with alpha=0.001. Target data are used only for final evaluation.",
        "",
        "| Dataset | B0 HardNormal G | E1 CategoryMeta G | E2 RandomMeta G | E1-B0 | E1-E2 | E1-B0 P-AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} | {r['E1_minus_B0_P-AP']:+.4f} |")
    lines.extend([
        "",
        "## Gate",
        f"- targets improved E1 over B0: {stats['targets_improved_E1_over_B0']}/3",
        f"- mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f}",
        f"- mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f}",
        f"- MVTec E1-B0 G: {stats['MVTec_E1_minus_B0_G']:+.4f}",
        f"- BTAD E1-B0 G: {stats['BTAD_E1_minus_B0_G']:+.4f}",
        f"- ClinicDB E1-B0 G: {stats['ClinicDB_E1_minus_B0_G']:+.4f}",
        "",
        "## Contrast Geometry",
        f"- cos(d_E1,d_B0): {geom[0]['cos_v_B0_v_E1']:.6f}",
        f"- cos(d_E2,d_B0): {geom[0]['cos_v_B0_v_E2']:.6f}",
        f"- cos(d_E1,d_E2): {geom[0]['cos_v_E1_v_E2']:.6f}",
        "",
        "## External SOTA Position",
        "- `SOTA_GAP.csv` compares E1 to MoECLIP reference rows. Treat as directly comparable with caution because exact implementation/evaluation details may differ.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MetaGeneralization] device={device}", flush=True)
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(np.load("results/phase_hard_normal_contrast/formal/contrast_vectors.npz")["E1"].astype(np.float32))
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    if args.smoke:
        groups = {k: groups[k] for k in sorted(groups)[:4]}
    random_groups = random_group_slots_like(slots, groups, SEED)
    v_e1, curves_e1, diag_e1, reused_e1 = train_meta_variant("E1_CategoryMeta", "category", groups, m_c, alpha, init_v, device, args.smoke)
    v_e2, curves_e2, diag_e2, reused_e2 = train_meta_variant("E2_RandomMeta", "random", random_groups, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_CategoryMeta": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_RandomMeta": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_per = load_hardnormal_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_per = b0_per[:3]
    raw = b0_raw + raw
    percat = b0_per + percat
    summary_rows = summarize(raw)
    outcome, stats = decide(summary_rows)
    geom = contrast_geometry(init_v, v_e1, v_e2)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_audit)
    write_csv(OUT / "META_DIAGNOSTICS.csv", [{"variant": "E1_CategoryMeta", **r} for r in diag_e1] + [{"variant": "E2_RandomMeta", **r} for r in diag_e2])
    sota_gap(raw, "E1_CategoryMeta")
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
        "meta_alpha": META_ALPHA,
        "top_negative_fraction": TOP_NEG_FRAC,
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2)},
        "cross_source_replication_started": outcome == "SOURCE_EPISODIC_META_GENERALIZATION_SUPPORTED",
    })
    write_report(outcome, stats, summary_rows, geom, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)
    if outcome == "SOURCE_EPISODIC_META_GENERALIZATION_SUPPORTED":
        print("[MetaGeneralization] First stage passed; cross-source replication is required but not implemented in this script.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
