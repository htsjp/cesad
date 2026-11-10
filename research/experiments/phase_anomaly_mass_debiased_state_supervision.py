import argparse
import csv
import json
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
from experiments.phase_hard_normal_contrast import CACHE_VERSION, EPOCHS, TOP_NEG_FRAC
from experiments.phase_patch_occupancy_supervision_audit import current_binary_label, true_patch_occupancy
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_state_pair_decomposition import decompose
from experiments.phase_state_pair_decomposition import hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np


OUT = Path("results/phase_anomaly_mass_debiased_state_supervision/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPS = 1e-12
PATCH_GRID = SIDE * SIDE
LOCAL_CACHE_VERSION = f"{CACHE_VERSION}_from_truecontrast_v2"


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


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    y = y.float()
    return 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)


def decomposed_task_loss(logits, y_bin, q_occ, image_ids, mode):
    y_bin = y_bin.float()
    q_occ = q_occ.float()
    image_ids = image_ids.long()
    per_pos_bin = F.softplus(-logits)
    per_neg = F.softplus(logits)
    neg = y_bin < 0.5
    if bool(neg.any()):
        neg_term = 0.5 * per_neg[neg].mean()
    else:
        neg_term = torch.zeros((), dtype=torch.float32, device=logits.device)

    if mode in ("binary_mass", "binary_instance"):
        y_for_dice = y_bin
        pos_weight = y_bin
    elif mode in ("occupancy_mass", "occupancy_instance"):
        y_for_dice = q_occ
        pos_weight = q_occ
    else:
        raise ValueError(mode)

    positive = pos_weight > 0
    if not bool(positive.any()):
        pos_term = torch.zeros((), dtype=torch.float32, device=logits.device)
    elif mode in ("binary_mass", "occupancy_mass"):
        pos_term = 0.5 * (pos_weight[positive] * per_pos_bin[positive]).sum() / pos_weight[positive].sum().clamp_min(1e-8)
    else:
        image_terms = []
        for image_id in torch.unique(image_ids[positive]):
            sel = positive & (image_ids == image_id)
            denom = pos_weight[sel].sum().clamp_min(1e-8)
            image_terms.append((pos_weight[sel] * per_pos_bin[sel]).sum() / denom)
        pos_term = 0.5 * torch.stack(image_terms).mean() if image_terms else torch.zeros((), dtype=torch.float32, device=logits.device)
    dl = dice_loss(logits, y_for_dice)
    return pos_term + neg_term + dl, pos_term.detach(), neg_term.detach(), dl.detach()


def reconstruction_gate(device):
    rng = np.random.default_rng(SEED)
    n = 4096
    logits = torch.tensor(rng.normal(size=n).astype(np.float32), device=device)
    y = torch.tensor((rng.random(n) > 0.72).astype(np.float32), device=device)
    q = y.clone()
    image_ids = torch.tensor(np.repeat(np.arange(128), int(np.ceil(n / 128)))[:n], dtype=torch.long, device=device)
    reconstructed, _p, _n, _d = decomposed_task_loss(logits, y, q, image_ids, "binary_mass")
    original, _bce, _dice = balanced_loss(logits, y)
    diff = float(torch.abs(reconstructed - original).detach().cpu())
    row = {
        "check": "B0_reconstructed_decomposed_task_loss",
        "L_original": float(original.detach().cpu()),
        "L_reconstructed": float(reconstructed.detach().cpu()),
        "abs_diff": diff,
        "pass": bool(diff < 1e-6),
        "positive_term_definition": "0.5 * mean positive BCE under original binary patch-mass aggregation",
        "negative_term_definition": "0.5 * mean negative BCE, unchanged",
        "dice_definition": "original Dice with matching target tensor",
    }
    return row


def load_training_context(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = torch.load(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", map_location=device, weights_only=False)
    corrupt_payload = torch.load(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", map_location=device, weights_only=False)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    align_model = make_model(states_c, corrupt_payload, device).eval()
    return q_pc1, align_model, m_c.astype(np.float32), alpha, init_v.astype(np.float32), corrupt_payload


def load_hardnormal_b0(smoke):
    raw = []
    percat = []
    with (HN_OUT / "RAW_METRICS.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("variant") == "E1_HardNormalContrast":
                r = dict(row)
                r["variant"] = "B0_CurrentHardNormal"
                raw.append(r)
    with (HN_OUT / "PER_CATEGORY.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("variant") == "E1_HardNormalContrast":
                r = dict(row)
                r["variant"] = "B0_CurrentHardNormal"
                percat.append(r)
    if smoke:
        return raw[:3], percat[:3]
    return raw, percat


def collect_mass_groups(dataset, indices, patch, align_model, q_pc1, device, smoke):
    rng = np.random.default_rng(SEED)
    slots = source_slots_by_category(dataset, indices, smoke)
    by_cat = {}
    audit = []
    zero_rows = []
    size_rows = []
    image_serial = 0
    for cat, positions in slots.items():
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        ys, qs, origins, image_ids, patch_ids = [], [], [], [], []
        local_image_records = []
        for pos in positions:
            item = dataset[int(indices[int(pos)])]
            is_anom = int(item["anomaly"]) == 1
            if is_anom:
                q, _method = true_patch_occupancy(item)
                y = (q > 0.5).astype(np.float32)
                current_y = current_binary_label(item).astype(np.float32)
            else:
                q = np.zeros((PATCH_GRID,), dtype=np.float32)
                y = np.zeros((PATCH_GRID,), dtype=np.float32)
                current_y = y
            z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
            z0 = norm_np(z0.reshape(PATCH_GRID, 1024))
            with torch.no_grad():
                z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
            if is_anom:
                m = int(y.sum())
                a = float(q.sum())
                size_rows.append({
                    "category": cat,
                    "position": int(pos),
                    "image_serial": image_serial,
                    "binary_positive_patch_count": m,
                    "occupancy_mass": a,
                    "zero_binary_positive": bool(m == 0),
                    "current_label_mismatch_patches": int(np.sum(current_y != y)),
                })
                if m == 0 and a > 0:
                    zero_rows.append({
                        "category": cat,
                        "position": int(pos),
                        "image_serial": image_serial,
                        "B0_positive_contribution": False,
                        "E1_positive_contribution": False,
                        "E2_positive_contribution": True,
                        "E3_positive_contribution": True,
                        "occupancy_mass": a,
                    })
            pools = []
            if is_anom:
                pools.append(("anomaly_mask_out", np.flatnonzero(y == 0)))
                pools.append(("anomaly_mask", np.flatnonzero(y == 1)))
                occ_pos = np.flatnonzero((q > 0) & (y == 0))
                if occ_pos.size:
                    pools.append(("anomaly_mask", occ_pos))
            else:
                pools.append(("normal_image", np.arange(PATCH_GRID)))
            for origin_name, ids in pools:
                if ids.size == 0:
                    continue
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(part.shape[0] for part in parts[origin_name])
                if room <= 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                arr = z[ids]
                parts[origin_name].append(arr)
                ys.extend(y[ids].tolist())
                qs.extend(q[ids].tolist())
                origins.extend([origin_name] * len(ids))
                image_ids.extend([image_serial] * len(ids))
                patch_ids.extend(ids.tolist())
            local_image_records.append(image_serial)
            image_serial += 1
        xs = []
        for origin_name in ("normal_image", "anomaly_mask_out", "anomaly_mask"):
            if parts[origin_name]:
                xs.append(np.concatenate(parts[origin_name], axis=0).astype(np.float32))
        if not xs:
            continue
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y_arr = np.asarray(ys, dtype=np.float32)
        q_arr = np.asarray(qs, dtype=np.float32)
        origin_arr = np.asarray(origins, dtype=object)
        by_cat[cat] = {
            "x": x,
            "y": y_arr,
            "q": q_arr,
            "origin": origin_arr,
            "image_id": np.asarray(image_ids, dtype=np.int64),
            "patch_id": np.asarray(patch_ids, dtype=np.int64),
        }
        audit.append({
            "record_type": "pool",
            "category": cat,
            "samples": int(x.shape[0]),
            "binary_positive_samples": int(np.sum(y_arr > 0.5)),
            "occupancy_positive_samples": int(np.sum(q_arr > 0)),
            "normal_image_negatives": int(np.sum(origin_arr == "normal_image")),
            "anomaly_mask_out_negatives": int(np.sum(origin_arr == "anomaly_mask_out")),
            "zero_binary_positive_images": int(sum(1 for r in size_rows if r["category"] == cat and r["zero_binary_positive"])),
        })
    return by_cat, audit, zero_rows, size_rows


def mass_batch(groups, rng, total_batch):
    names = list(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, qs, origins, cats, image_ids = [], [], [], [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        neg_ids = np.flatnonzero(data["y"] < 0.5)
        pos_ids = np.flatnonzero(data["q"] > 0)
        for ids, take in ((neg_ids, per_cls), (pos_ids, per_cls)):
            if ids.size == 0:
                continue
            choice = rng.choice(ids, size=take, replace=ids.size < take)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            qs.append(data["q"][choice])
            origins.extend(data["origin"][choice].tolist())
            cats.extend([cat] * take)
            image_ids.extend(data["image_id"][choice].tolist())
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    q = np.concatenate(qs, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], q[order], np.asarray(origins, dtype=object)[order], np.asarray(cats, dtype=object)[order], np.asarray(image_ids, dtype=np.int64)[order]


def train_variant(variant, mode, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}_{LOCAL_CACHE_VERSION}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload.get("curves", []), payload.get("gradient_mass", []), True
    print(f"[MassDebias] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    curves = []
    gradient_mass = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, qb, origin, _cats, image_ids = mass_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            q = torch.tensor(qb, dtype=torch.float32, device=device)
            image = torch.tensor(image_ids, dtype=torch.long, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            task_loss, pos_term, neg_term, dice = decomposed_task_loss(logits, y, q, image, mode)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            selected_idx = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices] if neg_idx.numel() else neg_idx
            hard_loss = F.softplus(logits[selected_idx]).mean() if selected_idx.numel() else torch.zeros((), dtype=torch.float32, device=device)
            loss = task_loss + hard_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(pos_term.cpu()), float(neg_term.cpu()), float(dice.cpu()), float(hard_loss.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        curves.append({
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "positive_bce_term": float(arr[:, 2].mean()),
            "negative_bce_term": float(arr[:, 3].mean()),
            "dice": float(arr[:, 4].mean()),
            "hard_loss": float(arr[:, 5].mean()),
        })
        if epoch in (1, 5, 10):
            gradient_mass.append(gradient_mass_snapshot(variant, epoch, groups, model, m_c, alpha, mode, device))
        print(f"[MassDebias] {variant} epoch {epoch}/{EPOCHS} loss={curves[-1]['loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "curves": curves, "gradient_mass": gradient_mass}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, curves, gradient_mass, False


def gradient_mass_snapshot(variant, epoch, groups, model, m_c, alpha, mode, device):
    rows = []
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    for cat, data in groups.items():
        for image_id in np.unique(data["image_id"]):
            ids = np.flatnonzero((data["image_id"] == image_id) & (data["q"] > 0))
            if ids.size == 0:
                continue
            z = torch.tensor(data["x"][ids], dtype=torch.float32, device=device)
            y = torch.tensor(data["y"][ids], dtype=torch.float32, device=device)
            q = torch.tensor(data["q"][ids], dtype=torch.float32, device=device)
            image = torch.tensor(data["image_id"][ids], dtype=torch.long, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            loss, _p, _n, _d = decomposed_task_loss(logits, y, q, image, mode)
            g = torch.autograd.grad(loss, model, retain_graph=False)[0].detach().cpu().numpy()
            rows.append({"patch_mass": float(np.sum(data["y"][ids] > 0.5)), "grad_norm": float(np.linalg.norm(g))})
    if not rows:
        return {"variant": variant, "epoch": epoch, "top10_largest_mask_grad_mass_fraction": float("nan")}
    masses = np.asarray([r["patch_mass"] for r in rows], dtype=np.float64)
    norms = np.asarray([r["grad_norm"] for r in rows], dtype=np.float64)
    order = np.argsort(-masses)
    k = max(1, int(np.ceil(len(order) * 0.10)))
    return {
        "variant": variant,
        "epoch": epoch,
        "image_count": int(len(rows)),
        "top10_largest_mask_grad_mass_fraction": float(norms[order[:k]].sum() / max(EPS, norms.sum())),
    }


def source_size_diagnostics(variant, groups, v_np, m_c, alpha, size_rows, device):
    thresholds = {}
    mvals = np.asarray([r["binary_positive_patch_count"] for r in size_rows if r["binary_positive_patch_count"] > 0], dtype=np.float64)
    thresholds["p25"] = float(np.percentile(mvals, 25))
    thresholds["p75"] = float(np.percentile(mvals, 75))
    buckets = defaultdict(lambda: {"scores": [], "losses": [], "count": 0})
    row_by_image = {int(r["image_serial"]): r for r in size_rows}
    mt = torch.tensor(m_c, dtype=torch.float32, device=device)
    vt = torch.tensor(unit_np(v_np), dtype=torch.float32, device=device)
    for cat, data in groups.items():
        for image_id in np.unique(data["image_id"]):
            meta = row_by_image.get(int(image_id))
            if not meta:
                continue
            ids = np.flatnonzero((data["image_id"] == image_id) & (data["q"] > 0))
            if ids.size == 0:
                continue
            mass = int(meta["binary_positive_patch_count"])
            if mass <= thresholds["p25"]:
                bucket = "small_bottom25"
            elif mass >= thresholds["p75"]:
                bucket = "large_top25"
            else:
                bucket = "medium"
            with torch.no_grad():
                z = torch.tensor(data["x"][ids], dtype=torch.float32, device=device)
                score = contrast_logits(z, mt, alpha, vt).detach().cpu().numpy()
            loss = np.logaddexp(0, -score)
            buckets[bucket]["scores"].extend(score.tolist())
            buckets[bucket]["losses"].extend(loss.tolist())
            buckets[bucket]["count"] += 1
    rows = []
    for bucket in ("small_bottom25", "medium", "large_top25"):
        scores = np.asarray(buckets[bucket]["scores"], dtype=np.float64)
        losses = np.asarray(buckets[bucket]["losses"], dtype=np.float64)
        rows.append({
            "variant": variant,
            "bucket": bucket,
            "image_count": buckets[bucket]["count"],
            "patch_count": int(scores.size),
            "positive_score_mean": float(scores.mean()) if scores.size else float("nan"),
            "positive_score_std": float(scores.std()) if scores.size else float("nan"),
            "positive_loss_mean": float(losses.mean()) if losses.size else float("nan"),
        })
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    variants = ["B0_CurrentHardNormal", "E1_InstanceBalancedBinary", "E2_GlobalOccupancy", "E3_InstanceBalancedOccupancy"]
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        base = by[(ds, "B0_CurrentHardNormal")]
        row = {"dataset": ds}
        for v in variants:
            row[f"{v}_P-AUROC"] = float(by[(ds, v)]["P-AUROC"])
            row[f"{v}_P-AP"] = float(by[(ds, v)]["P-AP"])
            row[f"{v}_G"] = float(by[(ds, v)]["G_pixel"])
        for v in variants[1:]:
            row[f"{v}_minus_B0_G"] = row[f"{v}_G"] - float(base["G_pixel"])
        row["E3_minus_E1_G"] = row["E3_InstanceBalancedOccupancy_G"] - row["E1_InstanceBalancedBinary_G"]
        row["E3_minus_E2_G"] = row["E3_InstanceBalancedOccupancy_G"] - row["E2_GlobalOccupancy_G"]
        rows.append(row)
    return rows


def decide(summary):
    e1 = np.asarray([r["E1_InstanceBalancedBinary_minus_B0_G"] for r in summary], dtype=np.float64)
    e2 = np.asarray([r["E2_GlobalOccupancy_minus_B0_G"] for r in summary], dtype=np.float64)
    e3 = np.asarray([r["E3_InstanceBalancedOccupancy_minus_B0_G"] for r in summary], dtype=np.float64)
    e3e1 = np.asarray([r["E3_minus_E1_G"] for r in summary], dtype=np.float64)
    e3e2 = np.asarray([r["E3_minus_E2_G"] for r in summary], dtype=np.float64)
    e3_success = int(np.sum(e3 > 0)) >= 2 and float(e3.mean()) >= 1.0 and (e3[0] >= 1.5 or e3[2] >= 1.5) and e3[1] >= -1.0 and (float(e3e1.mean()) >= 0.3 or float(e3e2.mean()) >= 0.3)
    e1_strong = int(np.sum(e1 > 0)) >= 2 and float(e1.mean()) >= 1.0
    e2_strong = int(np.sum(e2 > 0)) >= 2 and float(e2.mean()) >= 1.0
    if e3_success:
        if e1_strong and e2_strong and float(e3.mean()) > max(float(e1.mean()), float(e2.mean())):
            outcome = "INSTANCE_BALANCE_AND_OCCUPANCY_COMPLEMENTARY"
        else:
            outcome = "ANOMALY_MASS_DEBIASED_STATE_SUPPORTED"
    elif e1_strong and float(e3.mean()) <= float(e1.mean()) + 0.3:
        outcome = "INSTANCE_BALANCED_STATE_SUPPORTED_OCCUPANCY_UNNECESSARY"
    elif e2_strong and not e1_strong:
        outcome = "OCCUPANCY_RESCUE_SUPPORTED_MASS_BIAS_NOT_METHOD_RELEVANT"
    elif e3[1] > 1.0 and e3[0] < 0.5 and e3[2] < 0.5:
        outcome = "ANOMALY_MASS_DEBIAS_DOMAIN_DEPENDENT"
    else:
        outcome = "ANOMALY_MASS_DEBIASED_STATE_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1.mean()),
        "mean_E2_minus_B0_G": float(e2.mean()),
        "mean_E3_minus_B0_G": float(e3.mean()),
        "mean_E3_minus_E1_G": float(e3e1.mean()),
        "mean_E3_minus_E2_G": float(e3e2.mean()),
        "E3_targets_improved": int(np.sum(e3 > 0)),
    }


def write_report(outcome, stats, summary, recon, geom, runtime, smoke):
    lines = [
        "# Anomaly-Mass Debiased State Supervision",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Strict Full VisA source-only training. CLIP is frozen; target data is used only for final evaluation. HardNormal selection and hard-normal loss remain binary and unchanged.",
        "",
        "## Loss Implementation Check",
        "",
        f"- abs(L_reconstructed - L_original): {recon['abs_diff']:.10f}; pass={recon['pass']}.",
        "- B0 decomposition exactly reconstructs current `balanced_loss` as positive BCE mass mean + negative BCE mass mean + Dice.",
        "",
        "## Target Metrics",
        "",
        "| Dataset | B0 G | E1 Binary Instance G | E2 Occupancy Mass G | E3 Occupancy Instance G | E1-B0 | E2-B0 | E3-B0 | E3-E1 | E3-E2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['dataset']} | {r['B0_CurrentHardNormal_G']:.4f} | {r['E1_InstanceBalancedBinary_G']:.4f} | "
            f"{r['E2_GlobalOccupancy_G']:.4f} | {r['E3_InstanceBalancedOccupancy_G']:.4f} | "
            f"{r['E1_InstanceBalancedBinary_minus_B0_G']:+.4f} | {r['E2_GlobalOccupancy_minus_B0_G']:+.4f} | "
            f"{r['E3_InstanceBalancedOccupancy_minus_B0_G']:+.4f} | {r['E3_minus_E1_G']:+.4f} | {r['E3_minus_E2_G']:+.4f} |"
        )
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E2-B0 G: {stats['mean_E2_minus_B0_G']:+.4f} pp.",
        f"Mean E3-B0 G: {stats['mean_E3_minus_B0_G']:+.4f} pp.",
        f"Mean E3-E1 G: {stats['mean_E3_minus_E1_G']:+.4f} pp.",
        f"Mean E3-E2 G: {stats['mean_E3_minus_E2_G']:+.4f} pp.",
        "",
        "## Contrast Geometry",
        "",
        f"- cos(d_B0,d_E1): {geom[0]['cos_v_B0_v_E1']:.6f}.",
        f"- cos(d_B0,d_E2): {geom[1]['cos_v_B0_v_E2']:.6f}.",
        f"- cos(d_B0,d_E3): {geom[2]['cos_v_B0_v_E3']:.6f}.",
        "",
        f"Runtime: {runtime:.1f}s. Smoke: {smoke}.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MassDebias] device={device}", flush=True)
    recon = reconstruction_gate(device)
    write_csv(OUT / "LOSS_IMPLEMENTATION_CHECK.csv", [recon])
    if not recon["pass"]:
        write_json(OUT / "OUTCOME.json", {"outcome": "LOSS_RECONSTRUCTION_FAILURE", "loss_check": recon, "target_training": False})
        (OUT / "REPORT.md").write_text("# Anomaly-Mass Debiased State Supervision\n\nOutcome: `LOSS_RECONSTRUCTION_FAILURE`\n\nSTOP: reconstructed B0 loss did not match current formal loss.\n", encoding="utf-8")
        return
    q_pc1, align_model, m_c, alpha, init_v, corrupt_payload = load_training_context(device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    groups, pool_audit, zero_rows, size_rows = collect_mass_groups(dataset, indices, patch, align_model, q_pc1, device, args.smoke)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_audit)
    write_csv(OUT / "ZERO_BINARY_POSITIVE_IMAGES.csv", zero_rows)
    write_csv(OUT / "SOURCE_ANOMALY_SIZE_GROUPS.csv", size_rows)
    specs = [
        ("E1_InstanceBalancedBinary", "binary_instance"),
        ("E2_GlobalOccupancy", "occupancy_mass"),
        ("E3_InstanceBalancedOccupancy", "occupancy_instance"),
    ]
    vectors = {"B0": init_v.astype(np.float32)}
    gradient_rows = []
    curve_rows = []
    for variant, mode in specs:
        v, curves, gm, reused = train_variant(variant, mode, groups, m_c, alpha, init_v, device, args.smoke)
        vectors[variant] = v.astype(np.float32)
        curve_rows.extend(curves)
        for row in gm:
            rr = dict(row)
            rr["reused_checkpoint"] = bool(reused)
            gradient_rows.append(rr)
    write_csv(OUT / "TRAINING_CURVES.csv", curve_rows)
    write_csv(OUT / "GRADIENT_MASS_CONCENTRATION.csv", gradient_rows)
    models = {variant: make_model(hybrid_states(m_c, alpha * v), corrupt_payload, device) for variant, v in vectors.items() if variant != "B0"}
    model_variants = {
        "E1_InstanceBalancedBinary": models["E1_InstanceBalancedBinary"],
        "E2_GlobalOccupancy": models["E2_GlobalOccupancy"],
        "E3_InstanceBalancedOccupancy": models["E3_InstanceBalancedOccupancy"],
    }
    raw_new, percat_new = evaluate_pc1_removed(model_variants, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_hardnormal_b0(args.smoke)
    raw = b0_raw + raw_new
    percat = b0_percat + percat_new
    summary = summarize(raw)
    geom = [
        {"pair": "B0_E1", "cos_v_B0_v_E1": float(np.dot(vectors["B0"], vectors["E1_InstanceBalancedBinary"]) / max(EPS, np.linalg.norm(vectors["B0"]) * np.linalg.norm(vectors["E1_InstanceBalancedBinary"])))},
        {"pair": "B0_E2", "cos_v_B0_v_E2": float(np.dot(vectors["B0"], vectors["E2_GlobalOccupancy"]) / max(EPS, np.linalg.norm(vectors["B0"]) * np.linalg.norm(vectors["E2_GlobalOccupancy"])))},
        {"pair": "B0_E3", "cos_v_B0_v_E3": float(np.dot(vectors["B0"], vectors["E3_InstanceBalancedOccupancy"]) / max(EPS, np.linalg.norm(vectors["B0"]) * np.linalg.norm(vectors["E3_InstanceBalancedOccupancy"])))},
    ]
    source_diag = []
    for variant, v in vectors.items():
        name = "B0_CurrentHardNormal" if variant == "B0" else variant
        source_diag.extend(source_size_diagnostics(name, groups, v, m_c, alpha, size_rows, device))
    outcome, stats = decide(summary)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    write_csv(OUT / "SOURCE_SIZE_DIAGNOSTIC.csv", source_diag)
    np.savez_compressed(OUT / "contrast_vectors.npz", **vectors)
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
        "loss_reconstruction": recon,
    })
    write_report(outcome, stats, summary, recon, geom, runtime, bool(args.smoke))
    apply_gate(OUT, "E3_InstanceBalancedOccupancy", "E3 Instance-Balanced Occupancy")
    print(f"[MassDebias] OUTCOME={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
