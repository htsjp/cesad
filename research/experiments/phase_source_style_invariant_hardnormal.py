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

from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np


OUT = Path("results/phase_source_style_invariant_hardnormal/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPOCHS = 10
EPS = 1e-6

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
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def load_payload(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def load_b0_rows():
    raw, percat = [], []
    for name, dest in (("RAW_METRICS.csv", raw), ("PER_CATEGORY.csv", percat)):
        for row in read_csv(HN_OUT / name):
            if row["variant"] != "E1_HardNormalContrast":
                continue
            out = dict(row)
            out["variant"] = "B0_SymmetricHardNormal"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                if key in out:
                    try:
                        out[key] = float(out[key])
                    except ValueError:
                        out[key] = float("nan")
            if "mask_images" in out:
                out["mask_images"] = int(float(out["mask_images"]))
            dest.append(out)
    return raw, percat


def load_hardnormal_context(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    align_model = make_model(states_c, corrupt_payload, device).eval()
    hn_vec = np.load(HN_OUT / "contrast_vectors.npz")["E1"].astype(np.float32)
    return q_pc1, corrupt_payload, align_model, m_c.astype(np.float32), alpha, init_v.astype(np.float32), unit_np(hn_vec)


def source_image_bank(smoke):
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    slots = source_slots_by_category(dataset, indices, smoke)
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    labels_by_pos, _corrupt, _audit = __import__(
        "experiments.phase_pc1_orthogonal_anomaly_sufficiency",
        fromlist=["source_label_maps"],
    ).source_label_maps(False)
    bank = {}
    for cat, positions in slots.items():
        normals, anomalies = [], []
        for pos in positions:
            y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
            entry = {"pos": int(pos), "label": y.astype(np.float32), "category": cat, "is_anomaly": bool(y.max() > 0)}
            if entry["is_anomaly"]:
                anomalies.append(entry)
            else:
                normals.append(entry)
        if normals and anomalies:
            bank[cat] = {"normal": normals, "anomaly": anomalies}
    if len(bank) < 2:
        raise RuntimeError("Need at least two source categories for style mixing.")
    return dataset, indices, patch, bank


def transform_image(pos, patch, align_model, q_pc1, device):
    z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
    z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
    with torch.no_grad():
        z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
    return z


def style_mix(z, donor):
    mu_i = z.mean(axis=0, keepdims=True)
    sig_i = z.std(axis=0, keepdims=True)
    mu_j = donor.mean(axis=0, keepdims=True)
    sig_j = donor.std(axis=0, keepdims=True)
    return (sig_j * (z - mu_i) / (sig_i + EPS) + mu_j).astype(np.float32)


def pick_donor(bank, cat, mode, rng):
    if mode == "cross":
        choices = [c for c in bank if c != cat]
        donor_cat = str(rng.choice(choices))
    elif mode == "same":
        donor_cat = cat
    else:
        raise ValueError(mode)
    pool = bank[donor_cat]["normal"] + bank[donor_cat]["anomaly"]
    return rng.choice(pool), donor_cat


def sample_source_batch(bank, patch, align_model, q_pc1, device, rng, donor_mode, total_batch):
    cats = list(bank)
    per_image = max(8, total_batch // max(1, len(cats) * 2))
    xs, xs_style, ys, origins, src_cats, donor_cats = [], [], [], [], [], []
    for cat in cats:
        for kind in ("normal", "anomaly"):
            entry = rng.choice(bank[cat][kind])
            donor, donor_cat = pick_donor(bank, cat, donor_mode, rng)
            z = transform_image(entry["pos"], patch, align_model, q_pc1, device)
            z_donor = transform_image(donor["pos"], patch, align_model, q_pc1, device)
            zm = style_mix(z, z_donor)
            y = entry["label"]
            if kind == "anomaly":
                pos_ids = np.flatnonzero(y > 0.5)
                neg_ids = np.flatnonzero(y < 0.5)
                n_pos = max(1, min(pos_ids.size, per_image // 2))
                n_neg = max(1, per_image - n_pos)
                ids = np.concatenate([
                    rng.choice(pos_ids, size=n_pos, replace=pos_ids.size < n_pos),
                    rng.choice(neg_ids, size=n_neg, replace=neg_ids.size < n_neg),
                ])
                origin = np.where(y[ids] > 0.5, "anomaly_mask", "anomaly_mask_out")
            else:
                neg_ids = np.flatnonzero(y < 0.5)
                ids = rng.choice(neg_ids, size=per_image, replace=neg_ids.size < per_image)
                origin = np.asarray(["normal_image"] * len(ids), dtype=object)
            xs.append(z[ids])
            xs_style.append(zm[ids])
            ys.append(y[ids].astype(np.float32))
            origins.extend(origin.tolist())
            src_cats.extend([cat] * len(ids))
            donor_cats.extend([donor_cat] * len(ids))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    xm = np.concatenate(xs_style, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], xm[order], y[order], np.asarray(origins, dtype=object)[order], np.asarray(src_cats, dtype=object)[order], np.asarray(donor_cats, dtype=object)[order]


def hardnormal_loss(logits, y, rng, mode="hard"):
    task, bce, dice = balanced_loss(logits, y)
    neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
    neg_scores = logits.detach()[neg_idx]
    if mode == "hard":
        local = torch.topk(neg_scores, k=k, largest=True).indices
    else:
        local = torch.tensor(rng.permutation(len(neg_idx))[:k], dtype=torch.long, device=logits.device)
    selected = neg_idx[local]
    extra = F.softplus(logits[selected]).mean()
    return task + extra, task, extra, bce, dice, selected


def train_style_variant(variant, donor_mode, bank, patch, align_model, q_pc1, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["stats"], True
    print(f"[StyleHN] train {variant}", flush=True)
    total_images = sum(len(v["normal"]) + len(v["anomaly"]) for v in bank.values())
    steps_per_epoch = max(1, int(np.ceil(total_images / max(1, len(bank) * 2))))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 2)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    rows, curves = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, xb_style, yb, origin, cats, donor_cats = sample_source_batch(bank, patch, align_model, q_pc1, device, rng, donor_mode, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            zs = torch.tensor(xb_style, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            logits_style = contrast_logits(zs, m, alpha, v)
            loss_o, task_o, hard_o, bce_o, dice_o, selected = hardnormal_loss(logits, y, rng, "hard")
            loss_s, task_s, hard_s, bce_s, dice_s, _selected_s = hardnormal_loss(logits_style, y, rng, "hard")
            cons = F.mse_loss(torch.sigmoid(logits), torch.sigmoid(logits_style))
            loss = loss_o + loss_s + cons
            opt.zero_grad()
            loss.backward()
            opt.step()
            sel_np = selected.detach().cpu().numpy()
            rows.append({
                "variant": variant,
                "record_type": "train_step",
                "epoch": epoch,
                "step": step + 1,
                "donor_mode": donor_mode,
                "loss": float(loss.detach().cpu()),
                "original_loss": float(loss_o.detach().cpu()),
                "style_loss": float(loss_s.detach().cpu()),
                "consistency_loss": float(cons.detach().cpu()),
                "original_task_loss": float(task_o.detach().cpu()),
                "style_task_loss": float(task_s.detach().cpu()),
                "original_hard_loss": float(hard_o.detach().cpu()),
                "style_hard_loss": float(hard_s.detach().cpu()),
                "selected_count": int(len(sel_np)),
                "hard_normal_score_mean": float(logits.detach()[selected].cpu().numpy().mean()),
                "normal_image_ratio": float(np.mean(origin[sel_np] == "normal_image")),
                "anomaly_mask_out_ratio": float(np.mean(origin[sel_np] == "anomaly_mask_out")),
                "cross_category_donor_ratio": float(np.mean(cats != donor_cats)),
            })
            vals.append([float(loss.detach().cpu()), float(loss_o.detach().cpu()), float(loss_s.detach().cpu()), float(cons.detach().cpu())])
        arr = np.asarray(vals, dtype=np.float32)
        curve = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "original_loss": float(arr[:, 1].mean()),
            "style_loss": float(arr[:, 2].mean()),
            "consistency_loss": float(arr[:, 3].mean()),
        }
        curves.append(curve)
        print(f"[StyleHN] {variant} epoch {epoch}/{EPOCHS} loss={curve['loss']:.4f} cons={curve['consistency_loss']:.5f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "stats": rows, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, rows, False


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_SymmetricHardNormal")]
        e1 = by[(ds, "E1_CrossCategoryStyleMixHardNormal")]
        e2 = by[(ds, "E2_SameCategoryStyleMixControl")]
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


def decide(summary):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary], dtype=np.float32)
    mt_or_clinic = max([r["E1_minus_B0_G"] for r in summary if r["dataset"] in ("MVTec", "CVC-ClinicDB")])
    btad = [r["E1_minus_B0_G"] for r in summary if r["dataset"] == "BTAD"][0]
    success = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and mt_or_clinic >= 1.5 and btad >= -1.0
    if success:
        return "SOURCE_STYLE_INVARIANT_HARDNORMAL_SUPPORTED", {
            "run_mvtec_source_next": True,
            "mean_E1_minus_B0_G": float(e1_b0.mean()),
            "E1_targets_improved": int(np.sum(e1_b0 > 0)),
            "max_MVTec_or_ClinicDB_gain": float(mt_or_clinic),
            "BTAD_gain": float(btad),
        }
    return "SOURCE_STYLE_INVARIANT_HARDNORMAL_NO_SIGNAL", {
        "run_mvtec_source_next": False,
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "E1_targets_improved": int(np.sum(e1_b0 > 0)),
        "max_MVTec_or_ClinicDB_gain": float(mt_or_clinic),
        "BTAD_gain": float(btad),
    }


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_CrossCategoryStyleMixHardNormal"}
    for ds, row in current.items():
        for ref_name, table, status in (
            ("MoECLIP OpenCLIP ViT-L/14-336 518x518", MOECLIP, "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS"),
            ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD, "REFERENCE_ONLY_OR_PROTOCOL_CAUTION"),
        ):
            ref = table[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_CrossCategoryStyleMixHardNormal",
                "current_P-AUROC": float(row["P-AUROC"]),
                "current_P-AP": float(row["P-AP"]),
                "current_G": float(row["G_pixel"]),
                "reference": ref_name,
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": float(row["P-AUROC"]) - ref["P-AUROC"],
                "Delta_P_AP": float(row["P-AP"]) - ref["P-AP"],
                "Delta_G": float(row["G_pixel"]) - ref["G_pixel"],
                "comparison_status": status,
            })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(summary, outcome, stats, sota_rows, runtime):
    lines = [
        "# Source Style-Invariant HardNormal",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA -> MVTec / BTAD / CVC-ClinicDB. Frozen CLIP/StateAligned feature path; target data are used only for final pixel evaluation. B0 reuses validated Symmetric HardNormal.",
        "",
        "| Dataset | B0 G | E1 Cross-Category StyleMix G | E2 Same-Category StyleMix G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"E1 improved targets: {stats['E1_targets_improved']}/3.",
        f"MVTec/ClinicDB best gain: {stats['max_MVTec_or_ClinicDB_gain']:+.4f} pp; BTAD gain: {stats['BTAD_gain']:+.4f} pp.",
        "",
        "## External SOTA Position",
        "",
        "MoECLIP is the primary reference when source/target/backbone/resolution match closely; VisualAD(CLIP) is retained as secondary reference. These comparisons still carry implementation and preprocessing caveats.",
        "",
        "| Dataset | Current G | Reference | Reference G | Delta G | Status |",
        "|---|---:|---|---:|---:|---|",
    ])
    for r in sota_rows:
        lines.append(f"| {r['dataset']} | {float(r['current_G']):.4f} | {r['reference']} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend([
        "",
        "Decision rule: only if E1 passes the VisA-source gate should the MVTec-source replication be started.",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[StyleHN] device={device}", flush=True)
    q_pc1, corrupt_payload, align_model, m_c, alpha, init_v, hn_vec = load_hardnormal_context(device)
    _dataset, _indices, patch, bank = source_image_bank(args.smoke)
    e1, stats_e1, reused_e1 = train_style_variant("E1_CrossCategoryStyleMixHardNormal", "cross", bank, patch, align_model, q_pc1, m_c, alpha, init_v, device, args.smoke)
    e2, stats_e2, reused_e2 = train_style_variant("E2_SameCategoryStyleMixControl", "same", bank, patch, align_model, q_pc1, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_CrossCategoryStyleMixHardNormal": make_model(hybrid_states(m_c, alpha * e1), corrupt_payload, device),
        "E2_SameCategoryStyleMixControl": make_model(hybrid_states(m_c, alpha * e2), corrupt_payload, device),
    }
    raw_new, percat_new = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    raw_b0, percat_b0 = load_b0_rows()
    if args.smoke:
        raw_b0 = raw_b0[:3]
        percat_b0 = percat_b0[:3]
    raw = raw_b0 + raw_new
    percat = percat_b0 + percat_new
    summary = summarize(raw)
    outcome, decision_stats = decide(summary)
    sota_rows = write_sota_gap(raw)
    geom = contrast_geometry(hn_vec, e1, e2)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary)
    write_csv(OUT / "STYLE_MIX_TRAINING_STATS.csv", stats_e1 + stats_e2)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    np.savez_compressed(OUT / "contrast_vectors.npz", B0=hn_vec.astype(np.float32), E1=e1.astype(np.float32), E2=e2.astype(np.float32))
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "style_mix": "feature-wise per-image patch-token mean/std swap",
        "checkpoints_reused": {"E1": bool(reused_e1), "E2": bool(reused_e2), "B0": True},
    })
    write_report(summary, outcome, decision_stats, sota_rows, runtime)
    print(f"[StyleHN] OUTCOME={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
