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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import random_slots_like, source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import (
    BATCH_SIZE,
    PATCH_SAMPLE_PER_CLASS_CATEGORY,
    SEED,
    balanced_loss,
)
from experiments.phase_xxxviii_visual_state_representation import EPOCHS, LR, SIDE, norm_np

OUT = Path("results/phase_private_residual_contrast_memory/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
TCM_OUT = Path("results/phase_transferable_contrast_memory_pilot/formal_corrected")


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


def unit_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def load_b0_rows():
    raw, percat = [], []
    with (TCM_OUT / "RAW_METRICS.csv").open("r", encoding="utf-8", newline="") as fp:
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
            raw.append(out)
    with (TCM_OUT / "PER_CATEGORY.csv").open("r", encoding="utf-8", newline="") as fp:
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
            percat.append(out)
    return raw, percat


def sample_aligned_training_data(slots, labels_by_pos, dataset, indices, patch, align_model, q_pc1, device, smoke):
    rng = np.random.default_rng(SEED)
    slot_names = list(slots)
    xs, ys, gids = [], [], []
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    for gid, slot in enumerate(slot_names):
        by_cls = defaultdict(list)
        for pos in slots[slot]:
            y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
            z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
            z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
            with torch.no_grad():
                x = torch.tensor(z0, dtype=torch.float32, device=device)
                z = align_model.adapt(x).detach().cpu().numpy().astype(np.float32)
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
        for cls in (0, 1):
            if not by_cls[cls]:
                continue
            arr = np.concatenate(by_cls[cls], axis=0).astype(np.float32)
            xs.append(arr)
            ys.extend([cls] * arr.shape[0])
            gids.extend([gid] * arr.shape[0])
    if not xs:
        raise RuntimeError("No source training samples collected.")
    x_all = np.concatenate(xs, axis=0).astype(np.float32)
    y_all = np.asarray(ys, dtype=np.float32)
    g_all = np.asarray(gids, dtype=np.int64)
    return x_all, y_all, g_all, slot_names


class PrivateResidualContrast(torch.nn.Module):
    def __init__(self, init_v, slot_count):
        super().__init__()
        self.v_raw = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32))
        self.u = torch.nn.Parameter(torch.zeros(slot_count, init_v.shape[0], dtype=torch.float32))

    def shared(self):
        return F.normalize(self.v_raw, dim=-1)

    def private(self):
        vg = self.shared()
        r = self.u - (self.u @ vg).unsqueeze(-1) * vg.unsqueeze(0)
        vc = F.normalize(vg.unsqueeze(0) + r, dim=-1)
        return vc, r


def contrast_logits(z, m, alpha, v):
    s_a = F.normalize(m + alpha * v, dim=-1)
    s_n = F.normalize(m - alpha * v, dim=-1)
    if s_a.ndim == 2:
        return (z * s_a).sum(dim=-1) - (z * s_n).sum(dim=-1)
    return z @ s_a - z @ s_n


def train_private_model(variant, slots, labels_by_pos, dataset, indices, patch, align_model, m_c, alpha, init_v, q_pc1, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v_shared"].astype(np.float32), payload["slot_names"], payload.get("diag", []), True
    print(f"[PrivateResidual] prepare source aligned samples for {variant}", flush=True)
    x_all, y_all, g_all, slot_names = sample_aligned_training_data(slots, labels_by_pos, dataset, indices, patch, align_model, q_pc1, device, smoke)
    model = PrivateResidualContrast(init_v, len(slot_names)).to(device)
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all))
        vals = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            z = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[idx], dtype=torch.float32, device=device)
            gid = torch.tensor(g_all[idx], dtype=torch.long, device=device)
            vg = model.shared()
            vc, _r = model.private()
            logits_g = contrast_logits(z, m, alpha, vg)
            logits_p = contrast_logits(z, m, alpha, vc[gid])
            loss_g, bce_g, dice_g = balanced_loss(logits_g, y)
            loss_p, bce_p, dice_p = balanced_loss(logits_p, y)
            loss = loss_g + loss_p
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(loss_g.detach().cpu()), float(loss_p.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {"variant": variant, "epoch": epoch, "loss": float(arr[:, 0].mean()), "global_loss": float(arr[:, 1].mean()), "private_loss": float(arr[:, 2].mean())}
        curves.append(row)
        print(f"[PrivateResidual] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f}", flush=True)
    with torch.no_grad():
        v_shared = model.shared().detach().cpu().numpy().astype(np.float32)
        vc, r = model.private()
        diag = private_diagnostics(variant, v_shared, vc.detach().cpu().numpy(), r.detach().cpu().numpy(), slot_names)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v_shared": v_shared, "slot_names": slot_names, "curves": curves, "diag": diag}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v_shared, slot_names, diag, False


def private_diagnostics(variant, v_shared, vc, r, slot_names):
    rows = []
    r_norm = np.linalg.norm(r, axis=1)
    rn = norm_np(r)
    sim = rn @ rn.T
    off = sim[~np.eye(sim.shape[0], dtype=bool)] if sim.shape[0] > 1 else np.asarray([], dtype=np.float32)
    for i, slot in enumerate(slot_names):
        rows.append({
            "variant": variant,
            "record_type": "slot",
            "slot": slot,
            "slot_b": "",
            "residual_norm": float(r_norm[i]),
            "cos_vc_vG": float(unit_np(vc[i]) @ unit_np(v_shared)),
            "pairwise_residual_cosine_mean": float(off.mean()) if off.size else float("nan"),
            "private_residual_diversity": float(1.0 - off.mean()) if off.size else float("nan"),
            "pairwise_residual_cosine": float("nan"),
        })
    for i in range(len(slot_names)):
        for j in range(i + 1, len(slot_names)):
            rows.append({
                "variant": variant,
                "record_type": "pair",
                "slot": slot_names[i],
                "slot_b": slot_names[j],
                "residual_norm": float("nan"),
                "cos_vc_vG": float("nan"),
                "pairwise_residual_cosine_mean": float(off.mean()) if off.size else float("nan"),
                "private_residual_diversity": float(1.0 - off.mean()) if off.size else float("nan"),
                "pairwise_residual_cosine": float(sim[i, j]),
            })
    if off.size:
        rows.append({"variant": variant, "record_type": "summary", "slot": "__summary__", "slot_b": "", "residual_norm": float(r_norm.mean()), "cos_vc_vG": float(np.mean([unit_np(v) @ unit_np(v_shared) for v in vc])), "pairwise_residual_cosine_mean": float(off.mean()), "private_residual_diversity": float(1.0 - off.mean()), "pairwise_residual_cosine": float("nan")})
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_GlobalContrast")]
        e1 = by[(ds, "E1_TrueCategoryPrivateResidualMemory")]
        e2 = by[(ds, "E2_MatchedRandomPrivateResidualMemory")]
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
        outcome = "PRIVATE_RESIDUAL_MEMORY_SUPPORTED"
    elif stable and abs(float(e1_e2.mean())) < 1.0:
        outcome = "PRIVATE_MEMORY_GAIN_NOT_SEMANTIC"
    else:
        outcome = "PRIVATE_RESIDUAL_MEMORY_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
    }


def write_report(summary_rows, diag, outcome, stats, runtime):
    summary_diag = {r["variant"]: r for r in diag if r["slot"] == "__summary__"}
    lines = [
        "# Private Residual Contrast Memory Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. Private residual slots are used only during training; target inference keeps only the learned shared contrast v_G with fixed m_C, alpha, and A_C.",
        "",
        "| Dataset | B0 Global G | E1 True Private G | E2 Random Private G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
    ])
    for variant, row in summary_diag.items():
        lines.append(f"{variant}: mean ||r_c||={row['residual_norm']:.6f}, mean cos(v_c,v_G)={row['cos_vc_vG']:.6f}, diversity={row['private_residual_diversity']:.6f}.")
    lines.append("")
    lines.append(f"Runtime: {runtime:.1f}s.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PrivateResidual] device={device}", flush=True)
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = unit_np(q_pc1)
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    align_model = make_model(states_c, corrupt_payload, device)
    dataset_name, dataset, indices, cls, patch, y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    cat_slots = source_slots_by_category(dataset, indices, args.smoke)
    rnd_slots = random_slots_like(cat_slots, dataset, indices, SEED)

    v_true, names_true, diag_true, reused_true = train_private_model("E1_TrueCategoryPrivateResidualMemory", cat_slots, labels_by_pos, dataset, indices, patch, align_model, m_c, alpha, init_v, q_pc1, device, args.smoke)
    v_random, names_random, diag_random, reused_random = train_private_model("E2_MatchedRandomPrivateResidualMemory", rnd_slots, labels_by_pos, dataset, indices, patch, align_model, m_c, alpha, init_v, q_pc1, device, args.smoke)
    models = {
        "E1_TrueCategoryPrivateResidualMemory": make_model(hybrid_states(m_c, alpha * v_true), corrupt_payload, device),
        "E2_MatchedRandomPrivateResidualMemory": make_model(hybrid_states(m_c, alpha * v_random), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    diag = diag_true + diag_random
    outcome, decision = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "PRIVATE_RESIDUAL_GEOMETRY.csv", diag)
    np.savez_compressed(OUT / "shared_contrast_vectors.npz", E1=v_true.astype(np.float32), E2=v_random.astype(np.float32), init=init_v.astype(np.float32))
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "private_slots_used_at_inference": False,
        "source_dataset": dataset_name,
        "epochs": EPOCHS,
        "lr": LR,
        "alpha": alpha,
        "checkpoints_reused": {"E1": bool(reused_true), "E2": bool(reused_random)},
    })
    write_report(summary_rows, diag, outcome, decision, runtime)
    print(f"[PrivateResidual] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
