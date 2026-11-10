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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_category_dropout_contrast import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import OUT as PC1_DEBIAS_OUT, evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np


OUT = Path("results/phase_on_manifold_hard_normal_reinforcement/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
EPOCHS = 10
TOP_FRAC = 0.10
KNN = 32
RANK = 4
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


def as_metric_rows(rows):
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
    raw = [r for r in as_metric_rows(read_csv_rows(HN_OUT / "RAW_METRICS.csv")) if r["variant"] == "E1_HardNormalContrast"]
    per = [r for r in as_metric_rows(read_csv_rows(HN_OUT / "PER_CATEGORY.csv")) if r["variant"] == "E1_HardNormalContrast"]
    for row in raw + per:
        row["variant"] = "B0_HardNormal"
    if smoke:
        raw = raw[:3]
        per = per[:3]
    return raw, per


def source_context(device, smoke):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = torch.load(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", map_location=device, weights_only=False)
    corrupt_payload = torch.load(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", map_location=device, weights_only=False)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(np.load(HN_OUT / "contrast_vectors.npz")["B0"].astype(np.float32))
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, smoke)
    groups, pool_rows = collect_source_patch_pool_with_ids(slots, labels_by_pos, patch, align_model, q_pc1, device)
    return q_pc1, corrupt_payload, groups, pool_rows, m_c, alpha, init_v


def collect_source_patch_pool_with_ids(slots, labels_by_pos, patch, align_model, q_pc1, device):
    rng = np.random.default_rng(SEED)
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    out = {}
    rows = []
    for cat, positions in slots.items():
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        pids = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        image_ids = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
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
            for origin, ids in pools:
                if ids.size == 0:
                    continue
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(part.shape[0] for part in parts[origin])
                if room <= 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin].append(z[ids])
                pids[origin].append(ids.astype(np.int64))
                image_ids[origin].append(np.full((ids.size,), int(pos), dtype=np.int64))
        xs, ys, origins, patch_ids, image_pos = [], [], [], [], []
        for origin, cls in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if not parts[origin]:
                continue
            arr = np.concatenate(parts[origin], axis=0).astype(np.float32)
            ids = np.concatenate(pids[origin], axis=0).astype(np.int64)
            img = np.concatenate(image_ids[origin], axis=0).astype(np.int64)
            xs.append(arr)
            ys.extend([cls] * arr.shape[0])
            origins.extend([origin] * arr.shape[0])
            patch_ids.extend(ids.tolist())
            image_pos.extend(img.tolist())
        if not xs or len(set(ys)) < 2:
            raise RuntimeError(f"{cat}: insufficient source patch samples.")
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.asarray(ys, dtype=np.float32)
        origin = np.asarray(origins, dtype=object)
        out[cat] = {
            "x": x,
            "y": y,
            "origin": origin,
            "patch_id": np.asarray(patch_ids, dtype=np.int64),
            "image_pos": np.asarray(image_pos, dtype=np.int64),
        }
        rows.append({
            "variant": "source_pool",
            "category": cat,
            "samples": int(x.shape[0]),
            "positive_samples": int(y.sum()),
            "negative_samples": int((y == 0).sum()),
            "normal_image_negatives": int(np.sum(origin[y == 0] == "normal_image")),
            "anomaly_mask_out_negatives": int(np.sum(origin[y == 0] == "anomaly_mask_out")),
        })
    return out, rows


def orth_tangent_basis(vectors, z, fallback_seed):
    u = np.asarray(vectors, dtype=np.float32).copy()
    z = unit_np(z.astype(np.float32))
    u = u - z[None, :] * (u @ z)[:, None]
    q, _r = np.linalg.qr(u.T)
    q = q[:, :RANK].astype(np.float32)
    if q.shape[1] < RANK or not np.isfinite(q).all():
        rng = np.random.default_rng(fallback_seed)
        vec = rng.normal(size=(1024, RANK)).astype(np.float32)
        vec = vec - z[:, None] * (z @ vec)[None, :]
        q, _r = np.linalg.qr(vec)
        q = q[:, :RANK].astype(np.float32)
    return q


def precompute_manifold(groups, smoke):
    cache = OUT / ("MANIFOLD_CACHE_smoke.npz" if smoke else "MANIFOLD_CACHE.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        rows = [dict(metric="cache_reused", value=1.0, category="__all__")]
        return {k: z[k] for k in z.files}, rows
    print("[OMHR] precompute local normal manifolds", flush=True)
    names = sorted(groups)
    xs, cats, imgpos, local_refs = [], [], [], []
    for name in names:
        neg = groups[name]["y"] < 0.5
        local_idx = np.flatnonzero(neg)
        groups[name]["normal_manifold_index"] = np.full(len(groups[name]["y"]), -1, dtype=np.int64)
        start = len(xs)
        xs.extend(groups[name]["x"][local_idx].astype(np.float32))
        cats.extend([name] * len(local_idx))
        imgpos.extend(groups[name]["image_pos"][local_idx].tolist())
        for n, idx in enumerate(local_idx):
            groups[name]["normal_manifold_index"][idx] = start + n
            local_refs.append((name, int(idx)))
    x = np.asarray(xs, dtype=np.float32)
    cats = np.asarray(cats, dtype=object)
    imgpos = np.asarray(imgpos, dtype=np.int64)
    means = {name: x[cats == name].mean(axis=0).astype(np.float32) for name in names}
    r = x - np.stack([means[c] for c in cats.tolist()], axis=0)
    r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), EPS)
    n = x.shape[0]
    neighbors = np.zeros((n, KNN), dtype=np.int64)
    batch = 512 if not smoke else 128
    rt = torch.tensor(r, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    for start in range(0, n, batch):
        end = min(n, start + batch)
        sim = rt[start:end] @ rt.T
        same = torch.tensor(imgpos[start:end, None] == imgpos[None, :], device=sim.device)
        sim = sim.masked_fill(same, -1e9)
        take = torch.topk(sim, k=min(KNN, n - 1), largest=True).indices.detach().cpu().numpy()
        if take.shape[1] < KNN:
            take = np.pad(take, ((0, 0), (0, KNN - take.shape[1])), mode="edge")
        neighbors[start:end] = take[:, :KNN]
    U = np.zeros((n, 1024, RANK), dtype=np.float16)
    U_rand = np.zeros((n, 1024, RANK), dtype=np.float16)
    alpha = np.zeros((n,), dtype=np.float32)
    rows = []
    rng = np.random.default_rng(SEED)
    for i in range(n):
        nb = neighbors[i]
        diff = (x[nb] - x[i]).astype(np.float32)
        _u, _s, vt = np.linalg.svd(diff, full_matrices=False)
        basis = orth_tangent_basis(vt[:RANK], x[i], i)
        U[i] = basis.astype(np.float16)
        rand = rng.normal(size=(1024, RANK)).astype(np.float32)
        rand = rand - unit_np(x[i])[:, None] * (unit_np(x[i]) @ rand)[None, :]
        q, _ = np.linalg.qr(rand)
        U_rand[i] = q[:, :RANK].astype(np.float16)
        dots = np.clip(x[nb] @ unit_np(x[i]), -1.0, 1.0)
        alpha[i] = float(np.median(np.arccos(dots)))
        if i % 2000 == 0:
            print(f"[OMHR] manifold {i}/{n}", flush=True)
    for name in names:
        idx = cats == name
        rows.append({
            "category": name,
            "normal_count": int(np.sum(idx)),
            "alpha_mean": float(np.mean(alpha[idx])),
            "alpha_median": float(np.median(alpha[idx])),
            "alpha_p95": float(np.percentile(alpha[idx], 95)),
        })
    rows.append({"category": "__all__", "normal_count": int(n), "alpha_mean": float(np.mean(alpha)), "alpha_median": float(np.median(alpha)), "alpha_p95": float(np.percentile(alpha, 95))})
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x.astype(np.float16), category=cats, image_pos=imgpos, U=U, U_random=U_rand, alpha=alpha, neighbors=neighbors)
    return {"x": x.astype(np.float16), "category": cats, "image_pos": imgpos, "U": U, "U_random": U_rand, "alpha": alpha, "neighbors": neighbors}, rows


def attach_manifold_indices(groups, manifold, smoke):
    names = sorted(groups)
    if all("normal_manifold_index" in groups[n] for n in names):
        return
    offset = 0
    for name in names:
        neg = groups[name]["y"] < 0.5
        count = int(np.sum(neg))
        ids = np.full(len(groups[name]["y"]), -1, dtype=np.int64)
        ids[np.flatnonzero(neg)] = np.arange(offset, offset + count, dtype=np.int64)
        groups[name]["normal_manifold_index"] = ids
        offset += count


def sample_batch(groups, rng, total_batch):
    names = sorted(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, cats, mids = [], [], [], []
    for name in names:
        data = groups[name]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            cats.extend([name] * len(choice))
            mids.extend(data["normal_manifold_index"][choice].tolist())
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], np.asarray(cats, dtype=object)[order], np.asarray(mids, dtype=np.int64)[order]


def hard_indices(logits, y):
    neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    k = max(1, int(math.ceil(float(neg.numel()) * TOP_FRAC)))
    return neg[torch.topk(logits.detach()[neg], k=k, largest=True).indices]


def manifold_adv_loss(z, logits, hard_idx, mids, manifold, m, alpha_state, raw_v, mode):
    valid_np = mids[hard_idx.detach().cpu().numpy()]
    keep = valid_np >= 0
    if not np.any(keep):
        zero = torch.zeros((), dtype=torch.float32, device=z.device)
        return zero, {}
    hard_idx = hard_idx[torch.tensor(keep, dtype=torch.bool, device=z.device)]
    mid_np = valid_np[keep].astype(np.int64)
    z_h = z[hard_idx].detach().clone().requires_grad_(True)
    v = F.normalize(raw_v, dim=-1).detach()
    logits_h = contrast_logits(z_h, m.detach(), alpha_state, v)
    loss_h = F.softplus(logits_h).sum()
    g = torch.autograd.grad(loss_h, z_h, retain_graph=False, create_graph=False)[0].detach()
    if mode == "unconstrained":
        direction = g - (g * z_h.detach()).sum(dim=1, keepdim=True) * z_h.detach()
    else:
        key = "U" if mode == "manifold" else "U_random"
        U = torch.tensor(manifold[key][mid_np].astype(np.float32), dtype=torch.float32, device=z.device)
        proj = torch.bmm(U, torch.bmm(U.transpose(1, 2), g.unsqueeze(-1))).squeeze(-1)
        direction = proj - (proj * z_h.detach()).sum(dim=1, keepdim=True) * z_h.detach()
    direction = F.normalize(direction, dim=1, eps=1e-8).detach()
    alpha_i = torch.tensor(manifold["alpha"][mid_np], dtype=torch.float32, device=z.device).unsqueeze(1)
    z_adv = torch.cos(alpha_i) * z_h.detach() + torch.sin(alpha_i) * direction
    z_adv = F.normalize(z_adv, dim=1)
    adv_logits = contrast_logits(z_adv, m, alpha_state, F.normalize(raw_v, dim=-1))
    adv_loss = F.softplus(adv_logits).mean()
    with torch.no_grad():
        base_score = logits.detach()[hard_idx]
        if mode == "unconstrained":
            proj_ratio = torch.ones_like(base_score)
        else:
            proj_ratio = torch.linalg.norm(proj, dim=1) / torch.clamp(torch.linalg.norm(g, dim=1), min=1e-8)
        normal_x = torch.tensor(manifold["x"].astype(np.float32), dtype=torch.float32, device=z.device)
        sim = z_adv @ normal_x.T
        nn_dist = (1.0 - sim.max(dim=1).values).detach().cpu().numpy()
        stats = {
            "selected_count": int(len(mid_np)),
            "projection_ratio_mean": float(proj_ratio.mean().detach().cpu()),
            "projection_ratio_std": float(proj_ratio.std().detach().cpu()),
            "score_increase_mean": float((adv_logits.detach() - base_score).mean().cpu()),
            "score_increase_std": float((adv_logits.detach() - base_score).std().cpu()),
            "nn_distance_mean": float(np.mean(nn_dist)),
            "nn_distance_p95": float(np.percentile(nn_dist, 95)),
            "alpha_mean": float(np.mean(manifold["alpha"][mid_np])),
        }
    return adv_loss, stats


def batch_losses(x, y, raw_v, m, alpha):
    v = F.normalize(raw_v, dim=-1)
    logits = contrast_logits(x, m, alpha, v)
    task, bce, dice = balanced_loss(logits, y)
    hidx = hard_indices(logits, y)
    hard = F.softplus(logits[hidx]).mean()
    return logits, task, hard, bce, dice, hidx


def diagnostic_gradients(x, y, mids, raw_v, m, alpha, manifold):
    logits, task, hard, _bce, _dice, hidx = batch_losses(x, y, raw_v, m, alpha)
    omhr, _ = manifold_adv_loss(x, logits, hidx, mids, manifold, m, alpha, raw_v, "manifold")
    rand, _ = manifold_adv_loss(x, logits, hidx, mids, manifold, m, alpha, raw_v, "random")
    unc, _ = manifold_adv_loss(x, logits, hidx, mids, manifold, m, alpha, raw_v, "unconstrained")
    losses = {"B0": task + hard, "Hard": hard, "OMHR": omhr, "Random": rand, "Unconstrained": unc}
    grads = {}
    for key, loss in losses.items():
        grads[key] = torch.autograd.grad(loss, raw_v, retain_graph=True, create_graph=False)[0].detach()
    def cos(a, b):
        return float(F.cosine_similarity(grads[a], grads[b], dim=0).cpu())
    return {
        "cos_g_B0_g_OMHR": cos("B0", "OMHR"),
        "cos_g_Hard_g_OMHR": cos("Hard", "OMHR"),
        "cos_g_OMHR_g_Random": cos("OMHR", "Random"),
        "cos_g_OMHR_g_Unconstrained": cos("OMHR", "Unconstrained"),
        "norm_g_B0": float(torch.linalg.norm(grads["B0"]).cpu()),
        "norm_g_OMHR": float(torch.linalg.norm(grads["OMHR"]).cpu()),
        "OMHR_GRADIENT_REWEIGHT_COLLAPSE": bool(cos("Hard", "OMHR") >= 0.995),
    }


def train_variant(variant, mode, groups, manifold, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["stats"], payload["grad_rows"], True
    print(f"[OMHR] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    raw_v = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([raw_v], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    stats, grad_rows = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        adv_stats = []
        for step in range(1, steps_per_epoch + 1):
            xb, yb, cats, mids = sample_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            logits, task, hard, bce, dice, hidx = batch_losses(z, y, raw_v, m, alpha)
            if mode == "duplicate":
                extra = hard
                st = {"selected_count": int(hidx.numel())}
            else:
                extra, st = manifold_adv_loss(z, logits, hidx, mids, manifold, m, alpha, raw_v, mode)
            loss = task + hard + extra
            opt.zero_grad()
            loss.backward()
            opt.step()
            st.update({"variant": variant, "epoch": epoch, "step": step, "mode": mode})
            adv_stats.append(st)
            vals.append((float(loss.detach().cpu()), float(task.detach().cpu()), float(hard.detach().cpu()), float(extra.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "global_hard_loss": float(arr[:, 2].mean()),
            "extra_loss": float(arr[:, 3].mean()),
            "bce": float(arr[:, 4].mean()),
            "dice": float(arr[:, 5].mean()),
        }
        stats.append(row)
        stats.extend(adv_stats)
        xb, yb, cats, mids = sample_batch(groups, rng, BATCH_SIZE)
        x = torch.tensor(xb, dtype=torch.float32, device=device)
        y = torch.tensor(yb, dtype=torch.float32, device=device)
        g_row = diagnostic_gradients(x, y, mids, raw_v, m, alpha, manifold)
        g_row.update({"variant": variant, "epoch": epoch})
        grad_rows.append(g_row)
        print(f"[OMHR] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} extra={row['extra_loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(raw_v, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "stats": stats, "grad_rows": grad_rows}, ckpt)
    return v, stats, grad_rows, False


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e0 = by[(ds, "E0_DuplicateHard")]
        e1 = by[(ds, "E1_OnManifoldHardNormal")]
        e2 = by[(ds, "E2_RandomManifold")]
        e3 = by[(ds, "E3_UnconstrainedAdv")]
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


def decide(summary, grad_rows):
    by = {r["dataset"]: r for r in summary}
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary], dtype=np.float32)
    e1_e0 = np.asarray([r["E1_minus_E0_G"] for r in summary], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary], dtype=np.float32)
    e1_e3 = np.asarray([r["E1_minus_E3_G"] for r in summary], dtype=np.float32)
    improved = int(np.sum(e1_b0 > 0))
    btad_ok = by["BTAD"]["E1_minus_B0_G"] >= -1.0
    mt_or_clinic = by["MVTec"]["E1_minus_B0_G"] >= 1.5 or by["CVC-ClinicDB"]["E1_minus_B0_G"] >= 1.5
    e0_close = abs(float(e1_e0.mean())) < 0.5
    e2_close = abs(float(e1_e2.mean())) < 0.5
    e3_better = np.mean([r["E3_G"] - r["E1_G"] for r in summary]) >= 0.0
    structural = False
    e1_grad = [r for r in grad_rows if r["variant"] == "E1_OnManifoldHardNormal"]
    if e1_grad:
        structural = float(np.mean([r["cos_g_Hard_g_OMHR"] for r in e1_grad])) < 0.995
    if improved >= 2 and float(e1_b0.mean()) >= 1.0 and mt_or_clinic and float(e1_e0.mean()) >= 0.5 and float(e1_e2.mean()) >= 0.5 and float(e1_e3.mean()) >= 0.3 and btad_ok:
        outcome = "ON_MANIFOLD_HARD_NORMAL_REINFORCEMENT_SUPPORTED"
    elif e0_close:
        outcome = "OMHR_COLLAPSES_TO_EXTRA_HARDNORMAL_WEIGHT"
    elif e2_close:
        outcome = "LOCAL_NORMAL_MANIFOLD_NOT_SPECIFIC"
    elif e3_better:
        outcome = "UNCONSTRAINED_FEATURE_ROBUSTNESS_SUFFICIENT"
    elif structural and float(e1_b0.mean()) <= 0:
        outcome = "ON_MANIFOLD_HARD_STRUCTURE_NO_TARGET_GAIN"
    elif by["BTAD"]["E1_minus_B0_G"] > 1.0 and by["MVTec"]["E1_minus_B0_G"] <= 0.5 and by["CVC-ClinicDB"]["E1_minus_B0_G"] <= 0.5:
        outcome = "ON_MANIFOLD_HARDNORMAL_DOMAIN_DEPENDENT"
    else:
        outcome = "ON_MANIFOLD_HARD_NORMAL_NO_SIGNAL"
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
        "structural_gradient_difference": bool(structural),
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
        "# On-Manifold Hard-Normal Reinforcement",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source to MVTec / BTAD / CVC-ClinicDB. Local manifolds are precomputed from source binary-normal patches only; target data are used only for final evaluation.",
        "",
        "| Dataset | B0 HardNormal | E0 DuplicateHard | E1 OMHR | E2 RandomManifold | E3 Unconstrained | E1-B0 | E1-E0 | E1-E2 | E1-E3 |",
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
        "MoECLIP rows in `SOTA_GAP.csv` are reference-only unless backbone, resolution, preprocessing, and evaluator are exactly matched. The internal decision is E1 against B0/E0/E2/E3 under the same local protocol.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OMHR] device={device}", flush=True)
    q_pc1, corrupt_payload, groups, pool_rows, m_c, alpha, init_v = source_context(device, args.smoke)
    manifold, manifold_rows = precompute_manifold(groups, args.smoke)
    attach_manifold_indices(groups, manifold, args.smoke)
    variants = [
        ("E0_DuplicateHard", "duplicate"),
        ("E1_OnManifoldHardNormal", "manifold"),
        ("E2_RandomManifold", "random"),
        ("E3_UnconstrainedAdv", "unconstrained"),
    ]
    vectors, stats_rows, grad_rows, reused = {}, [], [], {}
    for variant, mode in variants:
        v, st, gr, was_reused = train_variant(variant, mode, groups, manifold, m_c, alpha, init_v, device, args.smoke)
        vectors[variant] = v
        stats_rows.extend(st)
        grad_rows.extend(gr)
        reused[variant] = bool(was_reused)
    models = {variant: make_model(hybrid_states(m_c, alpha * vectors[variant]), corrupt_payload, device) for variant, _mode in variants}
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_per = load_b0_rows(args.smoke)
    raw = b0_raw + raw
    percat = b0_per + percat
    summary = summarize(raw)
    outcome, decision_stats = decide(summary, grad_rows)
    geom = contrast_geometry(init_v, vectors["E1_OnManifoldHardNormal"], vectors["E2_RandomManifold"])
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "MANIFOLD_STATS.csv", manifold_rows + pool_rows)
    write_csv(OUT / "ADV_PLAUSIBILITY.csv", stats_rows)
    write_csv(OUT / "GRADIENT_GEOMETRY.csv", grad_rows)
    write_csv(OUT / "CONTROL_COMPARISON.csv", summary)
    write_csv(OUT / "SOTA_GAP.csv", sota_gap(raw, "E1_OnManifoldHardNormal"))
    np.savez_compressed(
        OUT / "contrast_vectors.npz",
        init=init_v.astype(np.float32),
        E0=vectors["E0_DuplicateHard"].astype(np.float32),
        E1=vectors["E1_OnManifoldHardNormal"].astype(np.float32),
        E2=vectors["E2_RandomManifold"].astype(np.float32),
        E3=vectors["E3_UnconstrainedAdv"].astype(np.float32),
    )
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_fraction": TOP_FRAC,
        "KNN": KNN,
        "rank": RANK,
        "checkpoints_reused": reused,
        "cross_source_replication_started": outcome == "ON_MANIFOLD_HARD_NORMAL_REINFORCEMENT_SUPPORTED",
    })
    write_report(outcome, decision_stats, summary, geom, runtime)
    print(json.dumps({"outcome": outcome, **decision_stats, "runtime_sec": runtime}, indent=2), flush=True)
    if outcome == "ON_MANIFOLD_HARD_NORMAL_REINFORCEMENT_SUPPORTED":
        print("[OMHR] First stage passed; MVTec-source cross-replication should be run next with the same extension.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
