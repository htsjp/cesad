import argparse
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

from experiments.phase_patch_occupancy_supervision_audit import current_binary_label, load_context, mask_tensor, true_patch_occupancy
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import balanced_loss
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np, patch_labels


OUT = Path("results/phase_source_anomaly_gradient_mass_audit/formal")
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


def safe_corr(x, y, kind):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    if kind == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pct_stats(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {k: float("nan") for k in ("mean", "std", "P5", "P25", "P50", "P75", "P95", "min", "max")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "P5": float(np.percentile(x, 5)),
        "P25": float(np.percentile(x, 25)),
        "P50": float(np.percentile(x, 50)),
        "P75": float(np.percentile(x, 75)),
        "P95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def implementation_audit():
    src_loss = inspect.getsource(balanced_loss)
    src_labels = inspect.getsource(patch_labels)
    already_balanced = False
    rows = [
        {"item": "positive_loss_reduction", "value": "patch-level BCE/Dice reduction inside balanced_loss; not per-anomaly-image positive averaging"},
        {"item": "patch_aggregation", "value": "training samples patches into a batch and applies loss over all sampled patches"},
        {"item": "per_image_positive_normalization", "value": "not implemented in current formal HardNormal path"},
        {"item": "mask_area_affects_loss_weight", "value": "implicit through number of positive patches sampled/contributed within the positive pool"},
        {"item": "already_instance_balanced", "value": already_balanced},
        {"item": "patch_labels_source", "value": src_labels},
        {"item": "balanced_loss_source", "value": src_loss},
    ]
    return rows, already_balanced


def transform_image(pos, patch, q_pc1, align_model, device):
    z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
    z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
    with torch.no_grad():
        z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device))
    return z.detach().cpu().numpy().astype(np.float32)


def positive_grad(z_np, ids, m_c, alpha, init_v, device, reduction):
    if len(ids) == 0:
        return np.zeros((init_v.size,), dtype=np.float64), 0.0
    z = torch.tensor(z_np[np.asarray(ids, dtype=np.int64)], dtype=torch.float32, device=device)
    v = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    logits = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
    per_patch = F.softplus(-logits)
    loss = per_patch.sum() if reduction == "sum" else per_patch.mean()
    loss.backward()
    g = v.grad.detach().flatten().cpu().numpy().astype(np.float64)
    return g, float(loss.detach().cpu())


def total_hardnormal_grad(records, m_c, alpha, init_v, device):
    xs, ys = [], []
    for rec in records:
        xs.append(rec["z"])
        ys.append(rec["y"])
    z = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32, device=device)
    y = torch.tensor(np.concatenate(ys, axis=0), dtype=torch.float32, device=device)
    v = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    logits = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
    task, _bce, _dice = balanced_loss(logits, y)
    neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    k = max(1, int(np.ceil(float(neg.numel()) * 0.10)))
    selected = neg[torch.topk(logits.detach()[neg], k=k, largest=True).indices]
    loss = task + F.softplus(logits[selected]).mean()
    loss.backward()
    g = v.grad.detach().flatten().cpu().numpy().astype(np.float64)
    return g, float(loss.detach().cpu()), int(z.shape[0])


def collect_records(dataset, indices, patch, q_pc1, align_model, device, smoke):
    records = []
    area_rows = []
    max_images = 80 if smoke else len(indices)
    for pos, idx in enumerate(indices[:max_images]):
        item = dataset[int(idx)]
        if int(item["anomaly"]) != 1:
            continue
        q, occ_method = true_patch_occupancy(item)
        y = (q > 0.5).astype(np.float32)
        current_y = current_binary_label(item)
        z = transform_image(pos, patch, q_pc1, align_model, device)
        m = int(y.sum())
        mask = mask_tensor(item).numpy().astype(np.float32)
        area = float(mask.mean())
        records.append({
            "position": int(pos),
            "index": int(idx),
            "category": str(item["cls_name"]),
            "z": z,
            "y": y,
            "q": q.astype(np.float32),
            "positive_ids": np.flatnonzero(y > 0.5).astype(np.int64),
            "area": area,
            "current_label_mismatch": int(np.sum(current_y.astype(np.float32) != y)),
        })
        area_rows.append({
            "record_type": "per_image",
            "position": int(pos),
            "index": int(idx),
            "category": str(item["cls_name"]),
            "binary_positive_patch_count": m,
            "gt_pixel_area_ratio": area,
            "positive_patch_fraction": float(m / float(SIDE * SIDE)),
            "occupancy_method": occ_method,
            "current_label_mismatch_patches": int(np.sum(current_y.astype(np.float32) != y)),
        })
    return records, area_rows


def mask_area_stats(area_rows):
    m = np.asarray([r["binary_positive_patch_count"] for r in area_rows], dtype=np.float64)
    a = np.asarray([r["gt_pixel_area_ratio"] for r in area_rows], dtype=np.float64)
    rows = []
    for name, values in (("binary_positive_patch_count", m), ("gt_pixel_area_ratio", a)):
        st = pct_stats(values)
        row = {"record_type": "summary", "stat": name, **st}
        if name == "binary_positive_patch_count":
            pos = values[values > 0]
            row["max_min_ratio"] = float(pos.max() / max(EPS, pos.min())) if pos.size else float("nan")
            row["P95_P50_ratio_all_images"] = float(np.percentile(values, 95) / max(EPS, np.percentile(values, 50))) if values.size else float("nan")
            row["P95_P50_ratio_positive_images"] = float(np.percentile(pos, 95) / max(EPS, np.percentile(pos, 50))) if pos.size else float("nan")
            row["zero_positive_patch_images"] = int(np.sum(values <= 0))
        rows.append(row)
    rows.extend(area_rows)
    return rows


def image_gradient_contribution(records, m_c, alpha, init_v, device):
    rows = []
    grads = []
    for rec in records:
        g, loss = positive_grad(rec["z"], rec["positive_ids"], m_c, alpha, init_v, device, "sum")
        norm = float(np.linalg.norm(g))
        grads.append(g)
        rows.append({
            "position": rec["position"],
            "index": rec["index"],
            "category": rec["category"],
            "binary_positive_patch_count": int(len(rec["positive_ids"])),
            "gt_pixel_area_ratio": rec["area"],
            "positive_loss_sum": loss,
            "grad_norm": norm,
            "current_label_mismatch_patches": rec["current_label_mismatch"],
        })
    m = np.asarray([r["binary_positive_patch_count"] for r in rows], dtype=np.float64)
    a = np.asarray([r["gt_pixel_area_ratio"] for r in rows], dtype=np.float64)
    n = np.asarray([r["grad_norm"] for r in rows], dtype=np.float64)
    summary = {
        "position": "__summary__",
        "index": "__all__",
        "category": "__all__",
        "images": len(rows),
        "corr_m_grad_norm_pearson": safe_corr(m, n, "pearson"),
        "corr_m_grad_norm_spearman": safe_corr(m, n, "spearman"),
        "corr_area_grad_norm_pearson": safe_corr(a, n, "pearson"),
        "corr_area_grad_norm_spearman": safe_corr(a, n, "spearman"),
    }
    return [summary] + rows, np.asarray(grads, dtype=np.float64)


def concentration_rows(rows):
    data = [r for r in rows if r.get("position") != "__summary__"]
    m = np.asarray([r["binary_positive_patch_count"] for r in data], dtype=np.float64)
    g = np.asarray([r["grad_norm"] for r in data], dtype=np.float64)
    out = []
    for sort_by, values in (("mask_patch_count_desc", m), ("gradient_norm_desc", g)):
        order = np.argsort(-values)
        total = float(g.sum())
        for frac in (0.10, 0.20, 0.50):
            k = max(1, int(np.ceil(len(order) * frac)))
            ids = order[:k]
            out.append({
                "sort_by": sort_by,
                "top_fraction": frac,
                "image_count": int(k),
                "summed_positive_grad_norm": float(g[ids].sum()),
                "fraction_of_total_grad_norm": float(g[ids].sum() / max(EPS, total)),
                "mean_mask_patch_count": float(m[ids].mean()),
                "mean_grad_norm": float(g[ids].mean()),
            })
    return out


def mean_normalized(grads, ids):
    if len(ids) == 0:
        return np.zeros((grads.shape[1],), dtype=np.float64)
    arr = grads[np.asarray(ids, dtype=np.int64)]
    arr = arr / np.maximum(EPS, np.linalg.norm(arr, axis=1, keepdims=True))
    return arr.mean(axis=0)


def small_large_geometry(records, grad_rows, grads, g_total):
    data = [r for r in grad_rows if r.get("position") != "__summary__"]
    m = np.asarray([r["binary_positive_patch_count"] for r in data], dtype=np.float64)
    valid = np.flatnonzero(m > 0)
    mv = m[valid]
    q25 = np.percentile(mv, 25)
    q75 = np.percentile(mv, 75)
    small = valid[np.flatnonzero(mv <= q25)]
    large = valid[np.flatnonzero(mv >= q75)]
    g_small = mean_normalized(grads, small)
    g_large = mean_normalized(grads, large)
    nt = float(np.linalg.norm(g_total))
    ns = float(np.linalg.norm(g_small))
    nl = float(np.linalg.norm(g_large))
    return [{
        "small_threshold_patch_count_p25": float(q25),
        "large_threshold_patch_count_p75": float(q75),
        "small_image_count": int(len(small)),
        "large_image_count": int(len(large)),
        "zero_positive_patch_images_excluded": int(np.sum(m <= 0)),
        "cos_g_small_g_large": float(g_small @ g_large / max(EPS, ns * nl)),
        "cos_g_small_g_B0_total": float(g_small @ g_total / max(EPS, ns * nt)),
        "cos_g_large_g_B0_total": float(g_large @ g_total / max(EPS, nl * nt)),
        "norm_g_small_mean_normalized": ns,
        "norm_g_large_mean_normalized": nl,
    }]


def balanced_equivalence(records, m_c, alpha, init_v, g_hn_total, device):
    all_z = []
    all_ids = []
    starts = []
    labels = []
    for rec in records:
        ids = rec["positive_ids"]
        if ids.size == 0:
            continue
        starts.append((len(all_ids), len(all_ids) + ids.size))
        all_ids.extend(ids.tolist())
        all_z.append(rec["z"][ids])
        labels.append(rec["y"])
    z_np = np.concatenate(all_z, axis=0).astype(np.float32)
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    v = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    logits = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
    loss_original = F.softplus(-logits).mean()
    loss_original.backward()
    g_original = v.grad.detach().flatten().cpu().numpy().astype(np.float64)

    v2 = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    logits2 = contrast_logits(z, m, alpha, F.normalize(v2, dim=-1))
    per = F.softplus(-logits2)
    image_means = [per[s:e].mean() for s, e in starts]
    loss_bal = torch.stack(image_means).mean()
    loss_bal.backward()
    g_bal = v2.grad.detach().flatten().cpu().numpy().astype(np.float64)

    no = float(np.linalg.norm(g_original))
    nb = float(np.linalg.norm(g_bal))
    nh = float(np.linalg.norm(g_hn_total))
    return [{
        "positive_images": len(starts),
        "positive_patches": int(z_np.shape[0]),
        "loss_original_positive_patch_mean": float(loss_original.detach().cpu()),
        "loss_instance_balanced_image_mean": float(loss_bal.detach().cpu()),
        "cos_g_original_g_bal": float(g_original @ g_bal / max(EPS, no * nb)),
        "norm_g_original": no,
        "norm_g_bal": nb,
        "norm_ratio_g_bal_over_original": float(nb / max(EPS, no)),
        "cos_g_original_g_hardnormal_total": float(g_original @ g_hn_total / max(EPS, no * nh)),
        "cos_g_bal_g_hardnormal_total": float(g_bal @ g_hn_total / max(EPS, nb * nh)),
    }]


def decide(mask_rows, grad_rows, conc_rows, geom_rows, bal_rows, already_balanced):
    if already_balanced:
        return "ANOMALY_MASS_ALREADY_BALANCED", {}
    m_summary = next(r for r in mask_rows if r["record_type"] == "summary" and r["stat"] == "binary_positive_patch_count")
    g_summary = next(r for r in grad_rows if r.get("position") == "__summary__")
    p95_p50 = float(m_summary["P95_P50_ratio_positive_images"])
    corr = float(g_summary["corr_m_grad_norm_pearson"])
    top10 = next(r for r in conc_rows if r["sort_by"] == "mask_patch_count_desc" and abs(r["top_fraction"] - 0.10) < 1e-9)
    geom = geom_rows[0]
    bal = bal_rows[0]
    long_tail = p95_p50 >= 2.0
    mass_link = (np.isfinite(corr) and corr >= 0.4) or float(top10["fraction_of_total_grad_norm"]) >= 0.20
    directional_diff = float(geom["cos_g_small_g_large"]) <= 0.95
    new_direction = float(bal["cos_g_original_g_bal"]) <= 0.98
    if long_tail and mass_link and directional_diff and new_direction:
        outcome = "SOURCE_ANOMALY_GRADIENT_MASS_BIAS_CONFIRMED"
    elif mass_link and float(geom["cos_g_small_g_large"]) > 0.98 and float(bal["cos_g_original_g_bal"]) > 0.995:
        outcome = "AREA_WEIGHTING_COLLAPSES_TO_GRADIENT_RESCALING"
    else:
        outcome = "SOURCE_ANOMALY_MASS_BIAS_NO_SIGNAL"
    stats = {
        "P95_P50_patch_count_ratio_positive_images": p95_p50,
        "P95_P50_patch_count_ratio_all_images": float(m_summary["P95_P50_ratio_all_images"]),
        "corr_m_grad_norm_pearson": corr,
        "corr_area_grad_norm_pearson": float(g_summary["corr_area_grad_norm_pearson"]),
        "top10_largest_mask_grad_fraction": float(top10["fraction_of_total_grad_norm"]),
        "cos_g_small_g_large": float(geom["cos_g_small_g_large"]),
        "cos_g_original_g_bal": float(bal["cos_g_original_g_bal"]),
        "long_tail": long_tail,
        "mass_link": mass_link,
        "directional_diff": directional_diff,
        "new_direction": new_direction,
    }
    return outcome, stats


def write_report(outcome, impl_rows, stats, mask_rows, grad_rows, conc_rows, geom_rows, bal_rows, runtime, smoke):
    mask_summary = {r["stat"]: r for r in mask_rows if r["record_type"] == "summary"}
    grad_summary = next(r for r in grad_rows if r.get("position") == "__summary__")
    lines = [
        "# Source Anomaly Gradient-Mass Bias Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN audit on Full VisA source with the current formal Symmetric HardNormal checkpoint/scorer. No optimizer step was executed.",
        "",
        "## Implementation Audit",
        "",
        "- Current positive supervision is computed over sampled patch tensors with `balanced_loss`; it is not explicitly averaged per anomaly image after dividing by that image's positive patch count.",
        "- `balanced_loss` uses weighted BCE plus Dice on the whole supplied patch tensor. Mask area can therefore influence image-level contribution through the number of positive patches represented in the positive pool.",
        "",
        "## Mask-Mass Distribution",
        "",
        f"- anomaly images evaluated: {grad_summary['images']}.",
        f"- positive patch count P50/P95: {mask_summary['binary_positive_patch_count']['P50']:.4f} / {mask_summary['binary_positive_patch_count']['P95']:.4f}.",
        f"- positive patch count P95/P50 ratio: {mask_summary['binary_positive_patch_count']['P95_P50_ratio_all_images']:.4f} over all anomaly images, {mask_summary['binary_positive_patch_count']['P95_P50_ratio_positive_images']:.4f} over images with m_j>0.",
        f"- max/min positive patch count ratio: {mask_summary['binary_positive_patch_count']['max_min_ratio']:.4f}.",
        f"- zero-positive-patch anomaly images at current binary grid: {int(mask_summary['binary_positive_patch_count']['zero_positive_patch_images'])}.",
        f"- GT pixel area P50/P95: {mask_summary['gt_pixel_area_ratio']['P50']:.6f} / {mask_summary['gt_pixel_area_ratio']['P95']:.6f}.",
        "",
        "## Gradient Mass",
        "",
        f"- corr(m_j, ||G_j||): Pearson {grad_summary['corr_m_grad_norm_pearson']:.6f}, Spearman {grad_summary['corr_m_grad_norm_spearman']:.6f}.",
        f"- corr(area_j, ||G_j||): Pearson {grad_summary['corr_area_grad_norm_pearson']:.6f}, Spearman {grad_summary['corr_area_grad_norm_spearman']:.6f}.",
        f"- top 10% largest-mask images contribute {next(r for r in conc_rows if r['sort_by']=='mask_patch_count_desc' and abs(r['top_fraction']-0.10)<1e-9)['fraction_of_total_grad_norm']*100.0:.4f}% of summed positive gradient norm.",
        f"- top 20% largest-mask images contribute {next(r for r in conc_rows if r['sort_by']=='mask_patch_count_desc' and abs(r['top_fraction']-0.20)<1e-9)['fraction_of_total_grad_norm']*100.0:.4f}%.",
        f"- top 50% largest-mask images contribute {next(r for r in conc_rows if r['sort_by']=='mask_patch_count_desc' and abs(r['top_fraction']-0.50)<1e-9)['fraction_of_total_grad_norm']*100.0:.4f}%.",
        "",
        "## Small vs Large Geometry",
        "",
        f"- cos(g_small, g_large): {geom_rows[0]['cos_g_small_g_large']:.6f} computed after excluding {geom_rows[0]['zero_positive_patch_images_excluded']} zero-positive-patch anomaly images.",
        f"- cos(g_small, g_B0_total): {geom_rows[0]['cos_g_small_g_B0_total']:.6f}.",
        f"- cos(g_large, g_B0_total): {geom_rows[0]['cos_g_large_g_B0_total']:.6f}.",
        "",
        "## Instance-Balanced Counterfactual",
        "",
        f"- cos(g_original, g_bal): {bal_rows[0]['cos_g_original_g_bal']:.6f}.",
        f"- ||g_bal|| / ||g_original||: {bal_rows[0]['norm_ratio_g_bal_over_original']:.6f}.",
        f"- cos(g_bal, current HardNormal total gradient): {bal_rows[0]['cos_g_bal_g_hardnormal_total']:.6f}.",
        "",
        "## Gate",
        "",
        f"- long-tail P95/P50>=2: {stats.get('long_tail')}.",
        f"- mass-gradient link: {stats.get('mass_link')}.",
        f"- small/large directional difference cos<=0.95: {stats.get('directional_diff')}.",
        f"- instance-balanced new direction cos<=0.98: {stats.get('new_direction')}.",
        "",
        f"Runtime: {runtime:.1f}s. Smoke: {smoke}.",
        "",
        "STOP: no training was started.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    impl_rows, already_balanced = implementation_audit()
    write_csv(OUT / "IMPLEMENTATION_AUDIT.csv", impl_rows)
    if already_balanced:
        outcome, stats = decide([], [], [], [], [], True)
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "no_training": True, "optimizer_step": False, "runtime_sec": time.time() - start})
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MassBias] device={device}", flush=True)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    q_pc1, align_model, m_c, alpha, v_hn = load_context(device)
    records, area_rows = collect_records(dataset, indices, patch, q_pc1, align_model, device, args.smoke)
    mask_rows = mask_area_stats(area_rows)
    write_csv(OUT / "MASK_AREA_STATS.csv", mask_rows)
    grad_rows, grads = image_gradient_contribution(records, m_c, alpha, v_hn, device)
    conc = concentration_rows(grad_rows)
    for r in conc:
        r["record_type"] = "concentration"
    grad_rows_full = grad_rows[:1] + conc + grad_rows[1:]
    write_csv(OUT / "IMAGE_GRADIENT_CONTRIBUTION.csv", grad_rows_full)
    g_total, total_loss, total_patches = total_hardnormal_grad(records, m_c, alpha, v_hn, device)
    geom = small_large_geometry(records, grad_rows, grads, g_total)
    write_csv(OUT / "SMALL_LARGE_GEOMETRY.csv", geom)
    bal = balanced_equivalence(records, m_c, alpha, v_hn, g_total, device)
    bal[0]["hardnormal_total_loss_on_anomaly_records"] = total_loss
    bal[0]["hardnormal_total_patches_on_anomaly_records"] = total_patches
    write_csv(OUT / "BALANCED_GRADIENT_EQUIVALENCE.csv", bal)
    outcome, stats = decide(mask_rows, grad_rows, conc, geom, bal, already_balanced)
    runtime = time.time() - start
    payload = {
        "outcome": outcome,
        "runtime_sec": runtime,
        "device": str(device),
        "no_training": True,
        "optimizer_step": False,
        "source": "Full VisA",
        "checkpoint": "current formal Symmetric HardNormal scorer",
        "smoke": bool(args.smoke),
        **stats,
    }
    write_json(OUT / "OUTCOME.json", payload)
    write_report(outcome, impl_rows, stats, mask_rows, grad_rows, conc, geom, bal, runtime, bool(args.smoke))
    print(f"[MassBias] OUTCOME={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
