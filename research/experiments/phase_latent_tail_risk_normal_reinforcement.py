import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_hard_normal_contrast import collect_source_patch_pool
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_category_dropout_contrast import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import OUT as PC1_DEBIAS_OUT, evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_latent_tail_risk_normal_reinforcement/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
EPOCHS = 10
TOP_FRAC = 0.10
K = 8
MIN_GROUP_NORMALS = 10
EPS = 1e-12


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


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def as_float_rows(rows):
    for row in rows:
        for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
            if key in row:
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    row[key] = float("nan")
        if "mask_images" in row:
            try:
                row["mask_images"] = int(float(row["mask_images"]))
            except (TypeError, ValueError):
                pass
    return rows


def load_b0_rows(smoke=False):
    raw = [r for r in as_float_rows(read_csv_rows(HN_OUT / "RAW_METRICS.csv")) if r["variant"] == "E1_HardNormalContrast"]
    per = [r for r in as_float_rows(read_csv_rows(HN_OUT / "PER_CATEGORY.csv")) if r["variant"] == "E1_HardNormalContrast"]
    for row in raw + per:
        row["variant"] = "B0_HardNormal"
    if smoke:
        raw = raw[:3]
        per = per[:3]
    return raw, per


def topk_loss(losses, frac=TOP_FRAC):
    if losses.numel() == 0:
        return losses
    k = max(1, int(math.ceil(float(losses.numel()) * frac)))
    return torch.topk(losses, k=k, largest=True).values


def source_context(device, smoke):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = torch.load(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", map_location=device, weights_only=False)
    corrupt_payload = torch.load(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", map_location=device, weights_only=False)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    # Historical formal variants are initialized from the reconstructed global contrast.
    init_v = unit_np(np.load(HN_OUT / "contrast_vectors.npz")["B0"].astype(np.float32))
    discover_v = unit_np(np.load(HN_OUT / "contrast_vectors.npz")["E1"].astype(np.float32))
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, smoke)
    groups, pool_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    return q_pc1, corrupt_payload, groups, pool_audit, m_c, alpha, init_v, discover_v


def add_latent_groups(groups, discover_v, smoke):
    cache = OUT / ("VISA_LATENT_GROUP_ASSIGNMENTS_smoke.npz" if smoke else "VISA_LATENT_GROUP_ASSIGNMENTS.npz")
    names = sorted(groups)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        latent_all = z["latent_group"].astype(np.int64)
        random_all = z["random_group"].astype(np.int64)
        offset = 0
        rows = []
        for name in names:
            n = len(groups[name]["y"])
            groups[name]["latent_group"] = latent_all[offset : offset + n]
            groups[name]["random_group"] = random_all[offset : offset + n]
            offset += n
        for k in range(K):
            rows.append({
                "source": "VisA",
                "group": k,
                "normal_count": int(np.sum(latent_all == k)),
                "random_normal_count": int(np.sum(random_all == k)),
            })
        return rows, True

    normal_x, normal_cat, refs = [], [], []
    for name in names:
        y = groups[name]["y"]
        neg = y < 0.5
        mu = groups[name]["x"][neg].mean(axis=0).astype(np.float32)
        group_ids = np.full(len(y), -1, dtype=np.int64)
        random_ids = np.full(len(y), -1, dtype=np.int64)
        groups[name]["latent_group"] = group_ids
        groups[name]["random_group"] = random_ids
        x = groups[name]["x"][neg].astype(np.float32)
        normal_x.append(x - mu[None, :])
        normal_cat.extend([name] * x.shape[0])
        refs.extend([(name, int(i)) for i in np.flatnonzero(neg)])
    r = np.concatenate(normal_x, axis=0)
    d = unit_np(discover_v)
    r = r - (r @ d)[:, None] * d[None, :]
    r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), EPS)
    labels = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(r).labels_.astype(np.int64)
    rng = np.random.default_rng(SEED)
    random_labels = labels.copy()
    rng.shuffle(random_labels)
    for (name, idx), gid, rgid in zip(refs, labels, random_labels):
        groups[name]["latent_group"][idx] = int(gid)
        groups[name]["random_group"][idx] = int(rgid)
    latent_all = np.concatenate([groups[n]["latent_group"] for n in names], axis=0)
    random_all = np.concatenate([groups[n]["random_group"] for n in names], axis=0)
    cat_all = np.concatenate([np.asarray([n] * len(groups[n]["y"]), dtype=object) for n in names], axis=0)
    y_all = np.concatenate([groups[n]["y"] for n in names], axis=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, latent_group=latent_all, random_group=random_all, category=cat_all, y=y_all)
    rows = []
    for k in range(K):
        rows.append({
            "source": "VisA",
            "group": k,
            "normal_count": int(np.sum(labels == k)),
            "random_normal_count": int(np.sum(random_labels == k)),
            "category_entropy": float(entropy(np.bincount([names.index(c) for c in np.asarray(normal_cat)[labels == k]], minlength=len(names)) + EPS)),
        })
    return rows, False


def sample_batch(groups, rng, total_batch):
    names = sorted(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, cats, latent, random_gid = [], [], [], [], []
    for name in names:
        data = groups[name]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            cats.extend([name] * len(choice))
            latent.extend(data["latent_group"][choice].tolist())
            random_gid.extend(data["random_group"][choice].tolist())
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return (
        x[order],
        y[order],
        np.asarray(cats, dtype=object)[order],
        np.asarray(latent, dtype=np.int64)[order],
        np.asarray(random_gid, dtype=np.int64)[order],
    )


def losses_for_batch(x, y, raw_v, m, alpha):
    v = F.normalize(raw_v, dim=-1)
    logits = contrast_logits(x, m, alpha, v)
    task, bce, dice = balanced_loss(logits, y)
    neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    neg_loss = F.softplus(logits[neg])
    global_vals = topk_loss(neg_loss, TOP_FRAC)
    global_hard = global_vals.mean() if global_vals.numel() else torch.zeros((), device=x.device)
    return logits, task, global_hard, bce, dice, neg, neg_loss


def grouped_cvar_loss(neg_loss, neg_groups, valid_values=range(K)):
    losses = []
    stats = []
    device = neg_loss.device
    neg_groups_t = torch.tensor(neg_groups, dtype=torch.long, device=device)
    for gid in valid_values:
        mask = neg_groups_t == int(gid)
        count = int(mask.sum().detach().cpu())
        if count < MIN_GROUP_NORMALS:
            stats.append((int(gid), count, 0, float("nan"), float("nan"), True))
            continue
        vals = neg_loss[mask]
        hard_vals = topk_loss(vals, TOP_FRAC)
        losses.append(hard_vals.mean())
        stats.append((int(gid), count, int(hard_vals.numel()), float(vals.detach().mean().cpu()), float(hard_vals.detach().mean().cpu()), False))
    if losses:
        return torch.stack(losses).mean(), stats
    return torch.zeros((), dtype=torch.float32, device=device), stats


def category_cvar_loss(neg_loss, neg_cats):
    cats = sorted(set(neg_cats.tolist()))
    cat_to_id = {c: i for i, c in enumerate(cats)}
    ids = np.asarray([cat_to_id[c] for c in neg_cats.tolist()], dtype=np.int64)
    loss, stats = grouped_cvar_loss(neg_loss, ids, range(len(cats)))
    out = []
    for gid, count, hard_count, mean_bce, cvar_bce, skipped in stats:
        out.append((cats[gid], count, hard_count, mean_bce, cvar_bce, skipped))
    return loss, out


def diagnostic_gradients(x, y, cats, latent, random_gid, raw_v, m, alpha):
    logits, task, global_hard, _bce, _dice, neg, neg_loss = losses_for_batch(x, y, raw_v, m, alpha)
    neg_np = neg.detach().cpu().numpy()
    lg, _ = grouped_cvar_loss(neg_loss, latent[neg_np], range(K))
    rg, _ = grouped_cvar_loss(neg_loss, random_gid[neg_np], range(K))
    cg, _ = category_cvar_loss(neg_loss, cats[neg_np])
    losses = {"global": global_hard, "latent": lg, "random": rg, "category": cg, "task": task}
    grads = {}
    for key, loss in losses.items():
        grads[key] = torch.autograd.grad(loss, raw_v, retain_graph=True, create_graph=False, allow_unused=False)[0].detach()
    def cos(a, b):
        return float(F.cosine_similarity(grads[a], grads[b], dim=0).cpu())
    return {
        "cos_g_Global_g_LG": cos("global", "latent"),
        "cos_g_LG_g_Random": cos("latent", "random"),
        "cos_g_LG_g_Category": cos("latent", "category"),
        "cos_g_Task_g_Global": cos("task", "global"),
        "norm_g_Global": float(torch.linalg.norm(grads["global"]).cpu()),
        "norm_g_LG": float(torch.linalg.norm(grads["latent"]).cpu()),
        "norm_g_Random": float(torch.linalg.norm(grads["random"]).cpu()),
        "norm_g_Category": float(torch.linalg.norm(grads["category"]).cpu()),
    }


def train_variant(variant, mode, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["train_rows"], payload["grad_rows"], True
    print(f"[LTRN] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    raw_v = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([raw_v], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    train_rows, grad_rows = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        epoch_group_rows = []
        for step in range(1, steps_per_epoch + 1):
            xb, yb, cats, latent, random_gid = sample_batch(groups, rng, BATCH_SIZE)
            x = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            logits, task, global_hard, bce, dice, neg, neg_loss = losses_for_batch(x, y, raw_v, m, alpha)
            neg_np = neg.detach().cpu().numpy()
            extra = global_hard
            group_loss = torch.zeros((), dtype=torch.float32, device=device)
            group_stats = []
            if mode == "duplicate_global":
                group_loss = global_hard
            elif mode == "latent":
                group_loss, group_stats = grouped_cvar_loss(neg_loss, latent[neg_np], range(K))
            elif mode == "random":
                group_loss, group_stats = grouped_cvar_loss(neg_loss, random_gid[neg_np], range(K))
            elif mode == "category":
                group_loss, group_stats = category_cvar_loss(neg_loss, cats[neg_np])
            else:
                raise ValueError(mode)
            total = task + extra + group_loss
            opt.zero_grad()
            total.backward()
            opt.step()
            with torch.no_grad():
                neg_scores = logits.detach()[neg].cpu().numpy()
                gh_count = max(1, int(math.ceil(len(neg_scores) * TOP_FRAC)))
                gh_idx_local = np.argpartition(neg_scores, len(neg_scores) - gh_count)[len(neg_scores) - gh_count :]
                gh_latent = latent[neg_np][gh_idx_local]
                gh_p = np.bincount(gh_latent[gh_latent >= 0], minlength=K).astype(np.float64)
                gh_p = gh_p / max(1.0, gh_p.sum())
                eff = float(np.exp(entropy(gh_p + EPS)))
                vals.append((float(total.detach().cpu()), float(task.detach().cpu()), float(global_hard.detach().cpu()), float(group_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu()), eff))
                if group_stats:
                    if mode == "category":
                        for gid, count, hard_count, mean_bce, cvar_bce, skipped in group_stats:
                            epoch_group_rows.append({"variant": variant, "epoch": epoch, "step": step, "group_type": "category", "group": gid, "normal_count": count, "hard_count": hard_count, "mean_BCE": mean_bce, "CVaR_BCE": cvar_bce, "skipped": bool(skipped)})
                    else:
                        for gid, count, hard_count, mean_bce, cvar_bce, skipped in group_stats:
                            epoch_group_rows.append({"variant": variant, "epoch": epoch, "step": step, "group_type": mode, "group": gid, "normal_count": count, "hard_count": hard_count, "mean_BCE": mean_bce, "CVaR_BCE": cvar_bce, "skipped": bool(skipped)})
        arr = np.asarray(vals, dtype=np.float32)
        skip_ratio = float(np.mean([bool(r["skipped"]) for r in epoch_group_rows])) if epoch_group_rows else 0.0
        train_rows.append({
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "global_hard_loss": float(arr[:, 2].mean()),
            "extra_group_loss": float(arr[:, 3].mean()),
            "bce": float(arr[:, 4].mean()),
            "dice": float(arr[:, 5].mean()),
            "effective_group_number_global_hard": float(arr[:, 6].mean()),
            "group_skip_ratio": skip_ratio,
        })
        train_rows.extend(epoch_group_rows)
        xb, yb, cats, latent, random_gid = sample_batch(groups, rng, BATCH_SIZE)
        x = torch.tensor(xb, dtype=torch.float32, device=device)
        y = torch.tensor(yb, dtype=torch.float32, device=device)
        g_row = diagnostic_gradients(x, y, cats, latent, random_gid, raw_v, m, alpha)
        g_row.update({"variant": variant, "epoch": epoch})
        grad_rows.append(g_row)
        print(f"[LTRN] {variant} epoch {epoch}/{EPOCHS} loss={train_rows[-len(epoch_group_rows)-1]['loss']:.4f} group={train_rows[-len(epoch_group_rows)-1]['extra_group_loss']:.4f} skip={skip_ratio:.3f}", flush=True)
    with torch.no_grad():
        v = F.normalize(raw_v, dim=-1).detach().cpu().numpy().astype(np.float32)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "train_rows": train_rows, "grad_rows": grad_rows}, ckpt)
    return v, train_rows, grad_rows, False


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e0 = by[(ds, "E0_DuplicateGlobalTail")]
        e1 = by[(ds, "E1_LatentGroupCVaR")]
        e2 = by[(ds, "E2_RandomGroupCVaR")]
        e3 = by[(ds, "E3_CategoryCVaR")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E0_G": float(e0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E3_G": float(e3["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E0_G": float(e1["G_pixel"]) - float(e0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_E3_G": float(e1["G_pixel"]) - float(e3["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
        })
    return rows


def decide(summary_rows, grad_rows):
    by = {r["dataset"]: r for r in summary_rows}
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e0 = np.asarray([r["E1_minus_E0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    e1_e3 = np.asarray([r["E1_minus_E3_G"] for r in summary_rows], dtype=np.float32)
    e0_b0 = np.asarray([r["E0_G"] - r["B0_G"] for r in summary_rows], dtype=np.float32)
    improved = int(np.sum(e1_b0 > 0))
    mt_or_clinic = by["MVTec"]["E1_minus_B0_G"] >= 1.5 or by["CVC-ClinicDB"]["E1_minus_B0_G"] >= 1.5
    btad_ok = by["BTAD"]["E1_minus_B0_G"] >= -1.0
    structural = False
    e1_grad = [r for r in grad_rows if r["variant"] == "E1_LatentGroupCVaR"]
    if e1_grad:
        structural = float(np.nanmean([r["cos_g_Global_g_LG"] for r in e1_grad])) < 0.95
    if improved >= 2 and float(e1_b0.mean()) >= 1.0 and mt_or_clinic and float(e1_e0.mean()) >= 0.5 and float(e1_e2.mean()) >= 0.5 and float(e1_e3.mean()) >= 0.3 and btad_ok:
        outcome = "LATENT_TAIL_RISK_NORMAL_REINFORCEMENT_SUPPORTED"
    elif int(np.sum(e1_b0 > 0)) >= 2 and int(np.sum(e0_b0 > 0)) >= 2 and float(e1_e0.mean()) < 0.5:
        outcome = "EXTRA_NORMAL_TAIL_WEIGHT_SUPPORTED_NOT_LATENT_GROUPS"
    elif abs(float(e1_e2.mean())) < 0.5:
        outcome = "GROUP_CVAR_GAIN_NOT_LATENT_SPECIFIC"
    elif abs(float(e1_e3.mean())) < 0.3:
        outcome = "LATENT_GROUPS_NO_BETTER_THAN_SOURCE_CATEGORY"
    elif by["BTAD"]["E1_minus_B0_G"] > 1.0 and by["MVTec"]["E1_minus_B0_G"] <= 0.5 and by["CVC-ClinicDB"]["E1_minus_B0_G"] <= 0.5:
        outcome = "LATENT_TAIL_RISK_DOMAIN_DEPENDENT"
    elif structural and float(e1_b0.mean()) <= 0:
        outcome = "LATENT_TAIL_STRUCTURE_NO_TARGET_GAIN"
    else:
        outcome = "LATENT_TAIL_RISK_NO_TARGET_SIGNAL"
    return outcome, {
        "targets_improved_E1_over_B0": improved,
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E0_G": float(e1_e0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "mean_E1_minus_E3_G": float(e1_e3.mean()),
        "MVTec_E1_minus_B0_G": float(by["MVTec"]["E1_minus_B0_G"]),
        "BTAD_E1_minus_B0_G": float(by["BTAD"]["E1_minus_B0_G"]),
        "ClinicDB_E1_minus_B0_G": float(by["CVC-ClinicDB"]["E1_minus_B0_G"]),
        "BTAD_drop_ok": bool(btad_ok),
        "latent_structural_gradient_difference": bool(structural),
    }


def sota_gap(raw, variant):
    refs = {
        "MVTec": {"P-AUROC": 92.5, "P-AP": 45.7, "G_pixel": 69.10},
        "BTAD": {"P-AUROC": 96.8, "P-AP": 50.4, "G_pixel": 73.60},
        "CVC-ClinicDB": {"P-AUROC": 89.7, "P-AP": 49.9, "G_pixel": 69.80},
    }
    rows = []
    for row in raw:
        if row["variant"] != variant or row["dataset"] not in refs:
            continue
        ref = refs[row["dataset"]]
        rows.append({
            "dataset": row["dataset"],
            "variant": variant,
            "current_P-AUROC": float(row["P-AUROC"]),
            "current_P-AP": float(row["P-AP"]),
            "current_G": float(row["G_pixel"]),
            "reference": "MoECLIP OpenCLIP ViT-L/14-336 518x518",
            "comparison_status": "REFERENCE_ONLY unless backbone/preprocess/eval are exactly matched",
            "reference_P-AUROC": ref["P-AUROC"],
            "reference_P-AP": ref["P-AP"],
            "reference_G": ref["G_pixel"],
            "Delta_P_AUROC": float(row["P-AUROC"]) - ref["P-AUROC"],
            "Delta_P_AP": float(row["P-AP"]) - ref["P-AP"],
            "Delta_G": float(row["G_pixel"]) - ref["G_pixel"],
        })
    return rows


def write_report(outcome, stats, summary, geom, runtime):
    lines = [
        "# Latent Tail-Risk Normal Reinforcement",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source to MVTec / BTAD / CVC-ClinicDB. Latent groups are fixed once from source normal patches using the previous source-only audit recipe. Target data are only used for final evaluation.",
        "",
        "| Dataset | B0 HardNormal | E0 DuplicateGlobal | E1 LTRN | E2 RandomGroup | E3 CategoryCVaR | E1-B0 | E1-E0 | E1-E2 | E1-E3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E3_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} | {r['E1_minus_E3_G']:+.4f} |")
    lines.extend([
        "",
        "## Gate",
        f"- targets improved E1 over B0: {stats['targets_improved_E1_over_B0']}/3",
        f"- mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f}",
        f"- mean E1-E0 G: {stats['mean_E1_minus_E0_G']:+.4f}",
        f"- mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f}",
        f"- mean E1-E3 G: {stats['mean_E1_minus_E3_G']:+.4f}",
        f"- BTAD drop ok: {stats['BTAD_drop_ok']}",
        "",
        "## Contrast Geometry",
        f"- cos(init,E1): {geom[0]['cos_v_B0_v_E1']:.6f}",
        f"- cos(init,E2): {geom[0]['cos_v_B0_v_E2']:.6f}",
        f"- cos(E1,E2): {geom[0]['cos_v_E1_v_E2']:.6f}",
        "",
        "## External SOTA Position",
        "MoECLIP rows in `SOTA_GAP.csv` are reference-only unless backbone, resolution, preprocessing, and evaluator are exactly matched. The main internal decision is E1 against B0/E0/E2/E3 under the same local protocol.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LTRN] device={device}", flush=True)
    q_pc1, corrupt_payload, groups, pool_audit, m_c, alpha, init_v, discover_v = source_context(device, args.smoke)
    group_rows, groups_reused = add_latent_groups(groups, discover_v, args.smoke)
    variants = [
        ("E0_DuplicateGlobalTail", "duplicate_global"),
        ("E1_LatentGroupCVaR", "latent"),
        ("E2_RandomGroupCVaR", "random"),
        ("E3_CategoryCVaR", "category"),
    ]
    vectors, train_rows, grad_rows, reused = {}, [], [], {}
    for variant, mode in variants:
        v, tr, gr, was_reused = train_variant(variant, mode, groups, m_c, alpha, init_v, device, args.smoke)
        vectors[variant] = v
        train_rows.extend(tr)
        grad_rows.extend(gr)
        reused[variant] = bool(was_reused)
    models = {variant: make_model(hybrid_states(m_c, alpha * vectors[variant]), corrupt_payload, device) for variant, _mode in variants}
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_per = load_b0_rows(args.smoke)
    raw = b0_raw + raw
    percat = b0_per + percat
    summary = summarize(raw)
    outcome, stats = decide(summary, grad_rows)
    geom = contrast_geometry(init_v, vectors["E1_LatentGroupCVaR"], vectors["E2_RandomGroupCVaR"])
    controls = []
    for row in summary:
        controls.append(dict(row))
    sota_rows = sota_gap(raw, "E1_LatentGroupCVaR")
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_TRAIN_DIAGNOSTICS.csv", train_rows)
    write_csv(OUT / "GRADIENT_GEOMETRY.csv", grad_rows)
    write_csv(OUT / "CONTROL_COMPARISON.csv", controls)
    write_csv(OUT / "SOTA_GAP.csv", sota_rows)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_audit)
    write_csv(OUT / "LATENT_GROUP_AUDIT.csv", group_rows)
    np.savez_compressed(
        OUT / "contrast_vectors.npz",
        init=init_v.astype(np.float32),
        E0=vectors["E0_DuplicateGlobalTail"].astype(np.float32),
        E1=vectors["E1_LatentGroupCVaR"].astype(np.float32),
        E2=vectors["E2_RandomGroupCVaR"].astype(np.float32),
        E3=vectors["E3_CategoryCVaR"].astype(np.float32),
    )
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
        "K": K,
        "top_fraction": TOP_FRAC,
        "min_group_normals": MIN_GROUP_NORMALS,
        "group_assignments_reused": bool(groups_reused),
        "checkpoints_reused": reused,
        "cross_source_replication_started": outcome == "LATENT_TAIL_RISK_NORMAL_REINFORCEMENT_SUPPORTED",
    })
    write_report(outcome, stats, summary, geom, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)
    if outcome == "LATENT_TAIL_RISK_NORMAL_REINFORCEMENT_SUPPORTED":
        print("[LTRN] First stage passed; MVTec-source cross-replication should be run next with the same script extension.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
