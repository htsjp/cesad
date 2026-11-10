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

from experiments.phase_cross_layer_persistent_hard_normal import MOECLIP, VISUALAD
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC
from experiments.phase_hard_normal_rejection_state import load_b0_rows, load_hardnormal_components
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import SEED
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np


OUT = Path("results/phase_hard_normal_nuisance_subspace/formal")
EPS = 1e-8
RANK = 4
SOURCE_BATCH = 8
TARGET_BATCH = 16


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


def zscore(x, mean, std):
    return (x - mean) / max(float(std), EPS)


def patch_mask(item):
    mask = item["img_mask"].float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    small = F.interpolate(mask, size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape(-1)
    return small.numpy() > 0.5


def label_stack(labels_by_pos, start, end):
    zero = np.zeros((SIDE * SIDE,), dtype=np.float32)
    return np.stack([np.asarray(labels_by_pos.get(int(i), zero), dtype=np.float32) for i in range(start, end)], axis=0) > 0.5


@torch.no_grad()
def aligned_and_base(raw_patch, align_model, q_pc1, device, batch_images=SOURCE_BATCH):
    n = raw_patch.shape[0]
    feats = []
    scores = []
    align_model.eval()
    for start in range(0, n, batch_images):
        arr = remove_pc1(np.asarray(raw_patch[start : start + batch_images], dtype=np.float32), q_pc1)
        b = arr.shape[0]
        flat = norm_np(arr.reshape(-1, arr.shape[-1]))
        x = torch.tensor(flat, dtype=torch.float32, device=device)
        z = align_model.adapt(x)
        logits, _ = align_model.logits(x)
        s = torch.sigmoid(logits).detach().cpu().numpy().reshape(b, SIDE * SIDE).astype(np.float32)
        feats.append(z.detach().cpu().numpy().reshape(b, SIDE * SIDE, -1).astype(np.float32))
        scores.append(s)
    return np.concatenate(feats, axis=0), np.concatenate(scores, axis=0)


def source_stats(smoke, align_model, q_pc1, device):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / f"SOURCE_NEGATIVE_STATS_{suffix}.npz"
    if cache.exists():
        payload = np.load(cache, allow_pickle=True)
        return {
            "mu_N": payload["mu_N"].astype(np.float32),
            "threshold": float(payload["threshold"]),
            "negative_count": int(payload["negative_count"]),
            "hard_count": int(payload["hard_count"]),
            "random_ids": payload["random_ids"].astype(np.int64),
            "score_mean": float(payload["score_mean"]),
            "score_std": float(payload["score_std"]),
            "score_p90": float(payload["score_p90"]),
            "score_p95": float(payload["score_p95"]),
        }
    print("[NuisanceSubspace] source pass 1: negative score distribution and mu_N", flush=True)
    _name, _dataset, _indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    true_labels, _corrupt, _audit = source_label_maps(False)
    n = min(96, patch.shape[0]) if smoke else patch.shape[0]
    sum_n = np.zeros((1024,), dtype=np.float64)
    scores = []
    neg_count = 0
    for start in range(0, n, SOURCE_BATCH):
        end = min(start + SOURCE_BATCH, n)
        z, s = aligned_and_base(np.asarray(patch[start:end], dtype=np.float32), align_model, q_pc1, device)
        labels = label_stack(true_labels, start, end)
        neg = ~labels
        zneg = z[neg]
        sneg = s[neg]
        sum_n += zneg.sum(axis=0, dtype=np.float64)
        neg_count += int(zneg.shape[0])
        scores.append(sneg.astype(np.float32))
        if (start // SOURCE_BATCH + 1) % 50 == 0:
            print(f"[NuisanceSubspace] source pass1 images {end}/{n}", flush=True)
    all_scores = np.concatenate(scores, axis=0).astype(np.float32)
    threshold = float(np.quantile(all_scores, 1.0 - TOP_NEG_FRAC))
    hard_count = int(np.sum(all_scores >= threshold))
    rng = np.random.default_rng(SEED)
    random_ids = np.sort(rng.choice(np.arange(neg_count, dtype=np.int64), size=hard_count, replace=False))
    mu_n = (sum_n / max(neg_count, 1)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        mu_N=mu_n,
        threshold=np.asarray(threshold),
        negative_count=np.asarray(neg_count),
        hard_count=np.asarray(hard_count),
        random_ids=random_ids,
        score_mean=np.asarray(float(all_scores.mean())),
        score_std=np.asarray(float(all_scores.std() + EPS)),
        score_p90=np.asarray(float(np.percentile(all_scores, 90))),
        score_p95=np.asarray(float(np.percentile(all_scores, 95))),
    )
    return {
        "mu_N": mu_n,
        "threshold": threshold,
        "negative_count": neg_count,
        "hard_count": hard_count,
        "random_ids": random_ids,
        "score_mean": float(all_scores.mean()),
        "score_std": float(all_scores.std() + EPS),
        "score_p90": float(np.percentile(all_scores, 90)),
        "score_p95": float(np.percentile(all_scores, 95)),
    }


def accumulate_covariances(smoke, align_model, q_pc1, stats, device):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / f"NUISANCE_SUBSPACES_{suffix}.npz"
    if cache.exists():
        payload = np.load(cache)
        return {k: payload[k] for k in payload.files}
    print("[NuisanceSubspace] source pass 2: hard/random covariance", flush=True)
    _name, _dataset, _indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    true_labels, _corrupt, _audit = source_label_maps(False)
    n = min(96, patch.shape[0]) if smoke else patch.shape[0]
    mu = stats["mu_N"].astype(np.float64)
    c_h = np.zeros((1024, 1024), dtype=np.float64)
    c_r = np.zeros((1024, 1024), dtype=np.float64)
    count_h = 0
    count_r = 0
    neg_seen = 0
    random_ids = stats["random_ids"]
    random_cursor = 0
    origin_counts = {"hard_normal_image": 0, "hard_anomaly_mask_out": 0, "random_normal_image": 0, "random_anomaly_mask_out": 0}
    for start in range(0, n, SOURCE_BATCH):
        end = min(start + SOURCE_BATCH, n)
        z, s = aligned_and_base(np.asarray(patch[start:end], dtype=np.float32), align_model, q_pc1, device)
        labels = label_stack(true_labels, start, end)
        neg = ~labels
        zneg = z[neg].astype(np.float64)
        sneg = s[neg]
        local_neg_count = int(zneg.shape[0])
        hard = sneg >= stats["threshold"]
        if np.any(hard):
            xh = zneg[hard] - mu
            c_h += xh.T @ xh
            count_h += int(xh.shape[0])
        image_ids = np.repeat(np.arange(start, end), SIDE * SIDE).reshape(end - start, SIDE * SIDE)[neg]
        for selected, prefix in ((hard, "hard"),):
            if np.any(selected):
                ids = image_ids[selected]
                for pos in ids:
                    item = _dataset.data_all[int(_indices[int(pos)])]
                    if int(item["anomaly"]) == 0:
                        origin_counts[f"{prefix}_normal_image"] += 1
                    else:
                        origin_counts[f"{prefix}_anomaly_mask_out"] += 1
        lo = neg_seen
        hi = neg_seen + local_neg_count
        take = []
        while random_cursor < random_ids.size and lo <= int(random_ids[random_cursor]) < hi:
            take.append(int(random_ids[random_cursor]) - lo)
            random_cursor += 1
        if take:
            zr = zneg[np.asarray(take, dtype=np.int64)] - mu
            c_r += zr.T @ zr
            count_r += int(zr.shape[0])
            ids = image_ids[np.asarray(take, dtype=np.int64)]
            for pos in ids:
                item = _dataset.data_all[int(_indices[int(pos)])]
                if int(item["anomaly"]) == 0:
                    origin_counts["random_normal_image"] += 1
                else:
                    origin_counts["random_anomaly_mask_out"] += 1
        neg_seen = hi
        if (start // SOURCE_BATCH + 1) % 50 == 0:
            print(f"[NuisanceSubspace] source pass2 images {end}/{n}", flush=True)
    vals_h, vecs_h = np.linalg.eigh(c_h / max(count_h - 1, 1))
    vals_r, vecs_r = np.linalg.eigh(c_r / max(count_r - 1, 1))
    order_h = np.argsort(vals_h)[::-1]
    order_r = np.argsort(vals_r)[::-1]
    vals_h = vals_h[order_h]
    vals_r = vals_r[order_r]
    u_h = vecs_h[:, order_h[:RANK]].astype(np.float32)
    u_r = vecs_r[:, order_r[:RANK]].astype(np.float32)
    np.savez_compressed(
        cache,
        U_H=u_h,
        U_R=u_r,
        eig_H=vals_h.astype(np.float64),
        eig_R=vals_r.astype(np.float64),
        count_H=np.asarray(count_h),
        count_R=np.asarray(count_r),
        origin_counts=np.asarray(json.dumps(origin_counts)),
    )
    return {"U_H": u_h, "U_R": u_r, "eig_H": vals_h, "eig_R": vals_r, "count_H": np.asarray(count_h), "count_R": np.asarray(count_r), "origin_counts": np.asarray(json.dumps(origin_counts))}


def source_calibration(smoke, align_model, q_pc1, stats, subspaces, device):
    suffix = "smoke" if smoke else "formal"
    path = OUT / f"SOURCE_CALIBRATION_{suffix}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print("[NuisanceSubspace] source pass 3: source-only z-score calibration", flush=True)
    _name, _dataset, _indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    n = min(96, patch.shape[0]) if smoke else patch.shape[0]
    mu = stats["mu_N"].astype(np.float32)
    u_h = subspaces["U_H"].astype(np.float32)
    u_r = subspaces["U_R"].astype(np.float32)
    acc = {k: [0.0, 0.0] for k in ("base", "hard_energy", "random_energy")}
    count = 0
    for start in range(0, n, SOURCE_BATCH):
        end = min(start + SOURCE_BATCH, n)
        z, s = aligned_and_base(np.asarray(patch[start:end], dtype=np.float32), align_model, q_pc1, device)
        flat = z.reshape(-1, z.shape[-1]).astype(np.float32)
        x = flat - mu[None, :]
        eh = np.linalg.norm(x @ u_h, axis=1).astype(np.float32)
        er = np.linalg.norm(x @ u_r, axis=1).astype(np.float32)
        for name, vals in (("base", s.reshape(-1)), ("hard_energy", eh), ("random_energy", er)):
            v = vals.astype(np.float64)
            acc[name][0] += float(v.sum())
            acc[name][1] += float((v * v).sum())
        count += int(flat.shape[0])
        if (start // SOURCE_BATCH + 1) % 50 == 0:
            print(f"[NuisanceSubspace] calibration images {end}/{n}", flush=True)
    rows = []
    for name, (ss, ss2) in acc.items():
        mean = ss / max(count, 1)
        var = max(ss2 / max(count, 1) - mean * mean, 0.0)
        rows.append({"score": name, "mean": float(mean), "std": float(np.sqrt(var) + EPS)})
    payload = {"target_information_used": False, "source_images": int(n), "patch_count": int(count), "rows": rows}
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "SOURCE_CALIBRATION.csv", rows)
    return payload


def cal_params(cal):
    return {r["score"]: (float(r["mean"]), float(r["std"])) for r in cal["rows"]}


def score_maps_for_patch(patch_arr, align_model, q_pc1, stats, subspaces, cal, device):
    z, base = aligned_and_base(patch_arr, align_model, q_pc1, device, batch_images=8)
    mu = stats["mu_N"].astype(np.float32)
    x = z.reshape(-1, z.shape[-1]).astype(np.float32) - mu[None, :]
    eh = np.linalg.norm(x @ subspaces["U_H"].astype(np.float32), axis=1).reshape(z.shape[0], SIDE, SIDE)
    er = np.linalg.norm(x @ subspaces["U_R"].astype(np.float32), axis=1).reshape(z.shape[0], SIDE, SIDE)
    params = cal_params(cal)
    zb = zscore(base.reshape(z.shape[0], SIDE, SIDE), *params["base"])
    mh = zscore(eh, *params["hard_energy"])
    mr = zscore(er, *params["random_energy"])
    return base.reshape(z.shape[0], SIDE, SIDE), eh, er, zb - mh, zb - mr


def evaluate(smoke, align_model, q_pc1, stats, subspaces, cal, device):
    raw, percat, diag = [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        if cell_id not in ("MVTec", "BTAD", "CVC-ClinicDB"):
            continue
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        print(f"[NuisanceSubspace] score {cell_id} n={n}", flush=True)
        base = np.zeros((n, SIDE, SIDE), dtype=np.float32)
        eh = np.zeros((n, SIDE, SIDE), dtype=np.float32)
        er = np.zeros((n, SIDE, SIDE), dtype=np.float32)
        maps = {
            "E1_HardNormalNuisanceSubspace": np.zeros((n, SIDE, SIDE), dtype=np.float32),
            "E2_RandomNormalSubspaceControl": np.zeros((n, SIDE, SIDE), dtype=np.float32),
        }
        for start in range(0, n, TARGET_BATCH):
            end = min(start + TARGET_BATCH, n)
            arr = np.stack([np.asarray(patch[i], dtype=np.float32) for i in range(start, end)], axis=0)
            b, h, r, mh, mr = score_maps_for_patch(arr, align_model, q_pc1, stats, subspaces, cal, device)
            base[start:end] = b
            eh[start:end] = h
            er[start:end] = r
            maps["E1_HardNormalNuisanceSubspace"][start:end] = mh
            maps["E2_RandomNormalSubspaceControl"][start:end] = mr
            if (start // TARGET_BATCH + 1) % 25 == 0:
                print(f"[NuisanceSubspace] {cell_id} scored {end}/{n}", flush=True)
        idx_use = idx_or_rows[:n]
        for variant, m in maps.items():
            image = np.asarray([top1_image_score(x) for x in m], dtype=np.float32)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, m)
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
        for variant, energy, final_map in (
            ("E1_HardNormalNuisanceSubspace", eh, maps["E1_HardNormalNuisanceSubspace"]),
            ("E2_RandomNormalSubspaceControl", er, maps["E2_RandomNormalSubspaceControl"]),
        ):
            pos_vals, neg_vals, base_neg, final_neg = [], [], [], []
            for pos in range(n):
                item = dataset[int(idx_use[pos])] if kind == "source_dataset" else dataset[pos]
                gt = patch_mask(item)
                vals = energy[pos].reshape(-1)
                if gt.any():
                    pos_vals.append(vals[gt])
                neg_vals.append(vals[~gt])
                base_neg.append(base[pos].reshape(-1)[~gt])
                final_neg.append(final_map[pos].reshape(-1)[~gt])
            p = np.concatenate(pos_vals).astype(np.float32) if pos_vals else np.asarray([], dtype=np.float32)
            q = np.concatenate(neg_vals).astype(np.float32)
            bn = np.concatenate(base_neg).astype(np.float32)
            fn = np.concatenate(final_neg).astype(np.float32)
            diag.append({
                "dataset": cell_id,
                "variant": variant,
                "anomaly_pixel_count": int(p.size),
                "normal_pixel_count": int(q.size),
                "anomaly_energy_mean": float(p.mean()) if p.size else float("nan"),
                "normal_energy_mean": float(q.mean()),
                "anomaly_energy_p95": float(np.percentile(p, 95)) if p.size else float("nan"),
                "normal_energy_p95": float(np.percentile(q, 95)),
                "energy_mean_gap": float(p.mean() - q.mean()) if p.size else float("nan"),
                "base_normal_score_mean": float(bn.mean()),
                "base_normal_score_p95": float(np.percentile(bn, 95)),
                "final_normal_score_mean": float(fn.mean()),
                "final_normal_score_p95": float(np.percentile(fn, 95)),
            })
    return raw, percat, diag


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_HardNormalNuisanceSubspace")]
        e2 = by[(ds, "E2_RandomNormalSubspaceControl")]
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


def subspace_rows(stats, subspaces):
    eig_h = np.asarray(subspaces["eig_H"], dtype=np.float64)
    eig_r = np.asarray(subspaces["eig_R"], dtype=np.float64)
    uh = np.asarray(subspaces["U_H"], dtype=np.float32)
    ur = np.asarray(subspaces["U_R"], dtype=np.float32)
    s = np.linalg.svd(uh.T @ ur, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
    origin_counts = json.loads(str(np.asarray(subspaces["origin_counts"]).item()))
    rows = [
        {
            "variant": "E1_HardNormalNuisanceSubspace",
            "selected_count": int(np.asarray(subspaces["count_H"]).item()),
            "top4_explained_variance": float(eig_h[:RANK].sum() / max(eig_h.sum(), EPS)),
            "eig1": float(eig_h[0]),
            "eig2": float(eig_h[1]),
            "eig3": float(eig_h[2]),
            "eig4": float(eig_h[3]),
            "normal_image_ratio": origin_counts["hard_normal_image"] / max(int(np.asarray(subspaces["count_H"]).item()), 1),
            "anomaly_mask_out_ratio": origin_counts["hard_anomaly_mask_out"] / max(int(np.asarray(subspaces["count_H"]).item()), 1),
            "source_negative_threshold": stats["threshold"],
        },
        {
            "variant": "E2_RandomNormalSubspaceControl",
            "selected_count": int(np.asarray(subspaces["count_R"]).item()),
            "top4_explained_variance": float(eig_r[:RANK].sum() / max(eig_r.sum(), EPS)),
            "eig1": float(eig_r[0]),
            "eig2": float(eig_r[1]),
            "eig3": float(eig_r[2]),
            "eig4": float(eig_r[3]),
            "normal_image_ratio": origin_counts["random_normal_image"] / max(int(np.asarray(subspaces["count_R"]).item()), 1),
            "anomaly_mask_out_ratio": origin_counts["random_anomaly_mask_out"] / max(int(np.asarray(subspaces["count_R"]).item()), 1),
            "source_negative_threshold": "",
        },
    ]
    angle_rows = [{"component": i + 1, "principal_angle_deg": float(a), "cosine": float(s[i])} for i, a in enumerate(angles)]
    return rows, angle_rows


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_HardNormalNuisanceSubspace"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_HardNormalNuisanceSubspace",
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
                "comparison_status": "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS",
            })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and (e1_b0[0] >= 1.5 or e1_b0[2] >= 1.5) and e1_b0[1] >= -1.0:
        outcome = "HARD_NORMAL_NUISANCE_SUBSPACE_SUPPORTED"
    elif float(e1_e2.mean()) > 0.5 and not (int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0):
        outcome = "HARD_NORMAL_SUBSPACE_HAS_STRUCTURE_NO_METHOD_GAIN"
    else:
        outcome = "HARD_NORMAL_NUISANCE_SUBSPACE_NO_SIGNAL"
    return outcome, {"mean_E1_minus_B0_G": float(e1_b0.mean()), "mean_E1_minus_E2_G": float(e1_e2.mean()), "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0))}


def write_report(summary_rows, outcome, stats, sota_rows, runtime):
    lines = [
        "# Hard-Normal Nuisance Subspace Calibration",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. Final-Layer HardNormal is frozen and reused. E1 subtracts rank4 hard-normal nuisance energy after source-only z-score calibration; E2 uses a matched random-normal rank4 subspace.",
        "",
        "| Dataset | B0 HardNormal G | E1 HardSubspace G | E2 RandomSubspace G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
        "## External SOTA Position",
        "",
        "MoECLIP is the primary external gate; VisualAD(CLIP) is retained as a secondary direct reference. The comparison is limited to VisA-source pixel P-AUROC/P-AP/G with implementation caveats.",
        "",
        "| Reference | Dataset | Current G | Reference G | Delta G | Status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for r in sota_rows:
        lines.append(f"| {r['reference']} | {r['dataset']} | {float(r['current_G']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend(["", "Stop-loss: if hard-normal subspace subtraction cannot beat frozen HardNormal and matched random subspace, do not add rank sweeps, Mahalanobis variants, learned weights, or target calibration.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NuisanceSubspace] device={device}", flush=True)
    q_pc1, _payload, align_model, _states_hn, _m_c, _alpha, _v_hn = load_hardnormal_components(device)
    stats = source_stats(args.smoke, align_model, q_pc1, device)
    subspaces = accumulate_covariances(args.smoke, align_model, q_pc1, stats, device)
    cal = source_calibration(args.smoke, align_model, q_pc1, stats, subspaces, device)
    raw_new, percat_new, diag = evaluate(args.smoke, align_model, q_pc1, stats, subspaces, cal, device)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw_new
    percat = b0_percat + percat_new
    summary_rows = summarize(raw)
    subspace_summary, angle_rows = subspace_rows(stats, subspaces)
    outcome, decision_stats = decide(summary_rows)
    sota_rows = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "SUBSPACE_STATS.csv", subspace_summary)
    write_csv(OUT / "PRINCIPAL_ANGLES.csv", angle_rows)
    write_csv(OUT / "NUISANCE_ENERGY_DIAGNOSTIC.csv", diag)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "network_retrained": False,
        "rank": RANK,
        "top_negative_fraction": TOP_NEG_FRAC,
        "source_negative_count": stats["negative_count"],
        "hard_negative_count": stats["hard_count"],
    })
    write_report(summary_rows, outcome, decision_stats, sota_rows, runtime)
    print(f"[NuisanceSubspace] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
