import argparse
import csv
import inspect
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
from experiments.phase_state_pair_decomposition import decompose
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import SIDE, patch_labels, norm_np


OUT = Path("results/phase_patch_occupancy_supervision_audit/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPS = 1e-12
GRAD_BATCHES = 100
PATCHES_PER_IMAGE = 96


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


def load_context(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = torch.load(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", map_location=device, weights_only=False)
    corrupt_payload = torch.load(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", map_location=device, weights_only=False)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    v_hn = unit_np(np.load(HN_OUT / "contrast_vectors.npz")["E1"].astype(np.float32))
    align_model = make_model(states_c, corrupt_payload, device).eval()
    return q_pc1, align_model, m_c.astype(np.float32), alpha, v_hn.astype(np.float32)


def current_mask_audit():
    src_patch = inspect.getsource(patch_labels)
    src_loss = inspect.getsource(balanced_loss)
    return {
        "resize_function": "experiments.phase_xxxviii_visual_state_representation.patch_labels",
        "resize方式": "torch.nn.functional.interpolate from full-resolution binary mask to 37x37 patch grid",
        "interpolation方式": "bilinear, align_corners=False",
        "final_patch_target_strictly_binary": True,
        "binary_threshold": "> 0.5 after bilinear resize",
        "loss_function": "experiments.phase_xliii_patch_specialized_adaptation.balanced_loss imported from phase_xlii_patch_specialized_adaptation",
        "loss_receives_float_tensor": True,
        "loss_currently_receives_continuous_target_in_formal_pipeline": False,
        "loss_continuous_ready_caveat": "BCE/Dice accept float targets, but class weighting and positive/negative selection use y > 0.5 semantics; formal labels are binary.",
        "patch_labels_source": src_patch,
        "balanced_loss_source": src_loss,
    }


def mask_tensor(item):
    m = item["img_mask"].detach().cpu().float()
    if m.ndim == 3:
        m = m.squeeze(0)
    if m.ndim != 2:
        m = m.reshape(m.shape[-2], m.shape[-1])
    return (m > 0.5).float()


def true_patch_occupancy(item):
    m = mask_tensor(item).numpy().astype(np.float32)
    h, w = m.shape
    if h == SIDE * 14 and w == SIDE * 14:
        q = m.reshape(SIDE, 14, SIDE, 14).mean(axis=(1, 3)).reshape(-1)
        return q.astype(np.float32), "exact_14x14_patch_fraction"
    t = torch.tensor(m, dtype=torch.float32).view(1, 1, h, w)
    avg = F.interpolate(t, size=(SIDE, SIDE), mode="area").reshape(-1).numpy().astype(np.float32)
    return avg, "area_interpolation_fraction"


def occupancy_stats(dataset, indices):
    rows = []
    hist_bins = np.linspace(0, 1, 21)
    all_q = []
    anomaly_q = []
    by_cat = {}
    per_image = []
    methods = set()
    for pos, idx in enumerate(indices):
        item = dataset[int(idx)]
        if int(item["anomaly"]) != 1:
            continue
        cat = str(item["cls_name"])
        q, method = true_patch_occupancy(item)
        methods.add(method)
        all_q.append(q)
        pos_q = q[q > 0]
        if pos_q.size:
            anomaly_q.append(pos_q)
        mixed = (q > 0.05) & (q < 0.95)
        mixed_anom = (q > 0.05) & (q < 0.95) & (q > 0)
        per_image.append({
            "record_type": "per_image",
            "category": cat,
            "position": int(pos),
            "index": int(idx),
            "anomaly_patch_count_q_gt0": int(np.sum(q > 0)),
            "mixed_patch_count": int(np.sum(mixed)),
            "mixed_occupancy_ratio_all_patches": float(np.mean(mixed)),
            "mixed_occupancy_ratio_anomaly_related": float(np.sum(mixed_anom) / max(1, np.sum(q > 0))),
            "mean_q_positive": float(pos_q.mean()) if pos_q.size else 0.0,
            "max_q": float(q.max()),
        })
        d = by_cat.setdefault(cat, [])
        d.append(q)
    q_all = np.concatenate(all_q) if all_q else np.asarray([], dtype=np.float32)
    q_pos = np.concatenate(anomaly_q) if anomaly_q else np.asarray([], dtype=np.float32)
    mixed_all = (q_all > 0.05) & (q_all < 0.95)
    mixed_pos = (q_pos > 0.05) & (q_pos < 0.95)
    rows.append({
        "record_type": "summary",
        "category": "__all__",
        "anomaly_images": len(all_q),
        "patch_count": int(q_all.size),
        "q_gt0_patch_count": int(q_pos.size),
        "mixed_patch_count": int(np.sum(mixed_all)),
        "mixed_occupancy_ratio_all_patches": float(np.mean(mixed_all)) if q_all.size else 0.0,
        "mixed_occupancy_ratio_anomaly_related": float(np.mean(mixed_pos)) if q_pos.size else 0.0,
        "mean_q": float(q_all.mean()) if q_all.size else 0.0,
        "mean_q_positive": float(q_pos.mean()) if q_pos.size else 0.0,
        "occupancy_method": ";".join(sorted(methods)),
    })
    hist, edges = np.histogram(q_all, bins=hist_bins)
    for lo, hi, count in zip(edges[:-1], edges[1:], hist):
        rows.append({"record_type": "histogram", "category": "__all__", "bin_left": float(lo), "bin_right": float(hi), "count": int(count), "ratio": float(count / max(1, q_all.size))})
    for cat, qs in sorted(by_cat.items()):
        arr = np.concatenate(qs)
        pos = arr[arr > 0]
        mix = (arr > 0.05) & (arr < 0.95)
        mix_pos = (pos > 0.05) & (pos < 0.95)
        rows.append({
            "record_type": "per_category",
            "category": cat,
            "anomaly_images": len(qs),
            "patch_count": int(arr.size),
            "q_gt0_patch_count": int(pos.size),
            "mixed_patch_count": int(np.sum(mix)),
            "mixed_occupancy_ratio_all_patches": float(np.mean(mix)),
            "mixed_occupancy_ratio_anomaly_related": float(np.mean(mix_pos)) if pos.size else 0.0,
            "mean_q": float(arr.mean()),
            "mean_q_positive": float(pos.mean()) if pos.size else 0.0,
        })
    rows.extend(per_image)
    return rows


def transform_image(pos, patch, align_model, q_pc1, device, cache):
    pos = int(pos)
    if pos in cache:
        return cache[pos]
    z0 = remove_pc1(np.asarray(patch[pos], dtype=np.float32), q_pc1)
    z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
    with torch.no_grad():
        z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
    cache[pos] = z
    return z


def current_binary_label(item):
    return patch_labels(item).astype(np.float32)


def sample_batch(records, rng, max_images=86):
    chosen = rng.choice(len(records), size=max_images, replace=len(records) < max_images)
    xs, yb, yo = [], [], []
    for ridx in chosen:
        rec = records[int(ridx)]
        q = rec["q"]
        y = rec["y"]
        pos = np.flatnonzero(y > 0.5)
        neg = np.flatnonzero(y < 0.5)
        per = max(2, PATCHES_PER_IMAGE)
        p_take = max(1, min(pos.size, per // 2)) if pos.size else 0
        n_take = per - p_take
        ids = []
        if p_take:
            ids.append(rng.choice(pos, size=p_take, replace=pos.size < p_take))
        if neg.size:
            ids.append(rng.choice(neg, size=n_take, replace=neg.size < n_take))
        ids = np.concatenate(ids) if ids else rng.choice(len(y), size=per, replace=len(y) < per)
        xs.append(rec["z"][ids])
        yb.append(y[ids])
        yo.append(q[ids])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y_bin = np.concatenate(yb, axis=0).astype(np.float32)
    y_occ = np.concatenate(yo, axis=0).astype(np.float32)
    if x.shape[0] > BATCH_SIZE:
        ids = rng.choice(x.shape[0], size=BATCH_SIZE, replace=False)
        x, y_bin, y_occ = x[ids], y_bin[ids], y_occ[ids]
    return x, y_bin, y_occ


def hardnormal_total_loss(logits, y):
    task, _bce, _dice = balanced_loss(logits, y)
    neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    if neg_idx.numel() == 0:
        return task
    k = max(1, int(np.ceil(float(neg_idx.numel()) * TOP_NEG_FRAC)))
    selected = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
    return task + F.softplus(logits[selected]).mean()


def grad_for_batch(x_np, y_np, m_c, alpha, init_v, device):
    z = torch.tensor(x_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    v = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    logits = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
    loss = hardnormal_total_loss(logits, y)
    loss.backward()
    g = v.grad.detach().flatten()
    return g.detach().cpu().numpy().astype(np.float64), float(loss.detach().cpu())


def gradient_equivalence(dataset, indices, patch, q_pc1, align_model, m_c, alpha, init_v, device, smoke):
    rng = np.random.default_rng(SEED)
    records = []
    cache = {}
    for pos, idx in enumerate(indices):
        item = dataset[int(idx)]
        if int(item["anomaly"]) != 1:
            continue
        q, _method = true_patch_occupancy(item)
        y = current_binary_label(item)
        z = transform_image(pos, patch, align_model, q_pc1, device, cache)
        records.append({"pos": int(pos), "index": int(idx), "category": str(item["cls_name"]), "z": z, "y": y, "q": q})
    batches = 10 if smoke else GRAD_BATCHES
    rows = []
    for b in range(batches):
        x, yb, yo = sample_batch(records, rng)
        gb, lb = grad_for_batch(x, yb, m_c, alpha, init_v, device)
        go, lo = grad_for_batch(x, yo, m_c, alpha, init_v, device)
        nb = float(np.linalg.norm(gb))
        no = float(np.linalg.norm(go))
        cos = float(gb @ go / max(EPS, nb * no))
        rows.append({
            "record_type": "batch",
            "batch": b,
            "patches": int(x.shape[0]),
            "positive_binary_ratio": float(np.mean(yb > 0.5)),
            "positive_occupancy_mean": float(np.mean(yo)),
            "mixed_occupancy_ratio": float(np.mean((yo > 0.05) & (yo < 0.95))),
            "cos_g_bin_g_occ": cos,
            "norm_g_bin": nb,
            "norm_g_occ": no,
            "norm_ratio_occ_over_bin": float(no / max(EPS, nb)),
            "loss_binary": lb,
            "loss_occupancy": lo,
        })
    arr_cos = np.asarray([r["cos_g_bin_g_occ"] for r in rows], dtype=np.float64)
    arr_nb = np.asarray([r["norm_g_bin"] for r in rows], dtype=np.float64)
    arr_no = np.asarray([r["norm_g_occ"] for r in rows], dtype=np.float64)
    arr_ratio = np.asarray([r["norm_ratio_occ_over_bin"] for r in rows], dtype=np.float64)
    rows.insert(0, {
        "record_type": "summary",
        "batch": "__all__",
        "batches": batches,
        "cos_mean": float(arr_cos.mean()),
        "cos_std": float(arr_cos.std()),
        "cos_p5": float(np.percentile(arr_cos, 5)),
        "cos_p95": float(np.percentile(arr_cos, 95)),
        "norm_g_bin_mean": float(arr_nb.mean()),
        "norm_g_occ_mean": float(arr_no.mean()),
        "norm_ratio_mean": float(arr_ratio.mean()),
        "norm_ratio_std": float(arr_ratio.std()),
    })
    return rows


def decide(mask_audit, occ_rows, grad_rows):
    if mask_audit["loss_currently_receives_continuous_target_in_formal_pipeline"]:
        return "OCCUPANCY_ALREADY_IMPLEMENTED", {"reason": "formal pipeline already uses continuous patch target"}
    occ_summary = next(r for r in occ_rows if r["record_type"] == "summary")
    grad_summary = next(r for r in grad_rows if r["record_type"] == "summary")
    mixed = float(occ_summary["mixed_occupancy_ratio_anomaly_related"])
    cos_mean = float(grad_summary["cos_mean"])
    if mixed < 0.05 or cos_mean >= 0.995:
        return "OCCUPANCY_SUPERVISION_COLLAPSES_TO_BASELINE", {
            "mixed_anomaly_related_occupancy": mixed,
            "mean_gradient_cosine": cos_mean,
            "collapse_by_mixed_ratio_lt_5pct": bool(mixed < 0.05),
            "collapse_by_gradient_cos_ge_0.995": bool(cos_mean >= 0.995),
        }
    if mixed >= 0.05 and cos_mean < 0.99:
        return "OCCUPANCY_SUPERVISION_HAS_NEW_INFORMATION", {
            "mixed_anomaly_related_occupancy": mixed,
            "mean_gradient_cosine": cos_mean,
        }
    return "OCCUPANCY_SUPERVISION_COLLAPSE_INCONCLUSIVE", {
        "mixed_anomaly_related_occupancy": mixed,
        "mean_gradient_cosine": cos_mean,
    }


def write_report(mask_audit, occ_rows, grad_rows, outcome, decision, runtime):
    occ = next(r for r in occ_rows if r["record_type"] == "summary")
    grad = next(r for r in grad_rows if r["record_type"] == "summary")
    lines = [
        "# Patch Occupancy Supervision Collapse Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "No training or optimizer step was run. The audit uses Full VisA source and the current formal StateAligned/HardNormal checkpoint.",
        "",
        "## Current Mask Pipeline",
        "",
        f"- Resize: {mask_audit['resize方式']}.",
        f"- Interpolation: {mask_audit['interpolation方式']}.",
        f"- Final patch target strictly binary: {mask_audit['final_patch_target_strictly_binary']}.",
        f"- Binary threshold: `{mask_audit['binary_threshold']}`.",
        f"- Loss receives continuous target in formal pipeline: {mask_audit['loss_currently_receives_continuous_target_in_formal_pipeline']}.",
        f"- Caveat: {mask_audit['loss_continuous_ready_caveat']}",
        "",
        "## Occupancy Statistics",
        "",
        f"- Anomaly images: {occ['anomaly_images']}.",
        f"- q>0 anomaly-related patch count: {occ['q_gt0_patch_count']}.",
        f"- Mixed occupancy ratio over all patches: {float(occ['mixed_occupancy_ratio_all_patches']) * 100:.4f}%.",
        f"- Mixed occupancy ratio among q>0 patches: {float(occ['mixed_occupancy_ratio_anomaly_related']) * 100:.4f}%.",
        f"- Mean positive q: {float(occ['mean_q_positive']):.6f}.",
        "",
        "## Gradient Equivalence",
        "",
        f"- Batches: {grad['batches']}.",
        f"- cos(g_bin, g_occ): mean {float(grad['cos_mean']):.6f}, std {float(grad['cos_std']):.6f}, P5 {float(grad['cos_p5']):.6f}, P95 {float(grad['cos_p95']):.6f}.",
        f"- ||g_bin|| mean: {float(grad['norm_g_bin_mean']):.6f}.",
        f"- ||g_occ|| mean: {float(grad['norm_g_occ_mean']):.6f}.",
        f"- norm ratio occ/bin mean: {float(grad['norm_ratio_mean']):.6f}.",
        "",
        "## Decision",
        "",
        json.dumps(decision, indent=2),
        "",
        "STOP: this phase does not start occupancy training.",
        f"Runtime: {runtime:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OccupancyAudit] device={device}", flush=True)
    mask_audit = current_mask_audit()
    q_pc1, align_model, m_c, alpha, v_hn = load_context(device)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    occ_rows = occupancy_stats(dataset, indices)
    if mask_audit["loss_currently_receives_continuous_target_in_formal_pipeline"]:
        grad_rows = [{"record_type": "summary", "batch": "__all__", "batches": 0, "cos_mean": float("nan"), "cos_std": float("nan"), "cos_p5": float("nan"), "cos_p95": float("nan"), "norm_g_bin_mean": float("nan"), "norm_g_occ_mean": float("nan"), "norm_ratio_mean": float("nan"), "norm_ratio_std": float("nan")}]
    else:
        grad_rows = gradient_equivalence(dataset, indices, patch, q_pc1, align_model, m_c, alpha, v_hn, device, args.smoke)
    outcome, decision = decide(mask_audit, occ_rows, grad_rows)
    write_csv(OUT / "OCCUPANCY_STATS.csv", occ_rows)
    write_csv(OUT / "GRADIENT_EQUIVALENCE.csv", grad_rows)
    write_json(OUT / "MASK_PIPELINE_AUDIT.json", mask_audit)
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision": decision,
        "runtime_sec": runtime,
        "device": device,
        "no_training": True,
        "optimizer_step": False,
        "source": "Full VisA",
        "checkpoint": "current formal StateAligned/HardNormal",
        "gradient_batches": 10 if args.smoke else GRAD_BATCHES,
    })
    write_report(mask_audit, occ_rows, grad_rows, outcome, decision, runtime)
    print(f"[OccupancyAudit] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
