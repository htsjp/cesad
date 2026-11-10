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

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells, sota_reference
import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_hard_normal_contrast import PC1_DEBIAS_OUT, load_payload
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_liv_memory_support_modes import ArrayConcat
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    build_maps,
    evaluate_maps,
    fnum,
    json_default,
    write_json,
)
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, logits_from_z
from experiments.phase_spatial_supervision_transfer_audit import target_cells as visa_target_cells
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, LR, SIDE, norm_np, patch_labels
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_paper_ready_final_validation/formal")
SEEDS = (0, 1, 2)
EPOCHS = 10
TOP_NEG_FRAC = 0.10
INDUSTRIAL_FOCUS = {
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "MPDD"),
}
MEDICAL_FOCUS = {
    ("Full VisA", "CVC-ClinicDB"),
    ("Full MVTec", "CVC-ClinicDB"),
}
MULTISEED_PAIRS = INDUSTRIAL_FOCUS | MEDICAL_FOCUS
INDUSTRIAL_TABLE_DATASETS = {"MVTec", "VisA", "BTAD", "MPDD", "DTD-Synthetic"}
MEDICAL_TABLE_DATASETS = {"CVC-ClinicDB", "CVC-ColonDB", "Kvasir"}
METRIC_KEYS = ("M_FR_P_AUROC", "M_FR_P_AP", "M_FR_P_F1max", "M_FR_PRO", "M_FR_G_pixel", "V_FR_P_AUROC", "V_FR_P_AP", "V_FR_P_F1max", "V_FR_PRO", "V_FR_G_pixel")
FULL_RES_METRIC_MAP = {
    "M_CATEGORY_MACRO_FR_I_AUROC": "M_FR_I_AUROC",
    "M_CATEGORY_MACRO_FR_I_AP": "M_FR_I_AP",
    "M_CATEGORY_MACRO_FR_I_F1max": "M_FR_I_F1max",
    "M_CATEGORY_MACRO_FR_P_AUROC": "M_FR_P_AUROC",
    "M_CATEGORY_MACRO_FR_P_AP": "M_FR_P_AP",
    "M_CATEGORY_MACRO_FR_P_F1max": "M_FR_P_F1max",
    "M_CATEGORY_MACRO_FR_PRO": "M_FR_PRO",
    "M_CATEGORY_MACRO_FR_G_pixel": "M_FR_G_pixel",
    "V_CATEGORY_MACRO_FR_I_AUROC": "V_FR_I_AUROC",
    "V_CATEGORY_MACRO_FR_I_AP": "V_FR_I_AP",
    "V_CATEGORY_MACRO_FR_I_F1max": "V_FR_I_F1max",
    "V_CATEGORY_MACRO_FR_P_AUROC": "V_FR_P_AUROC",
    "V_CATEGORY_MACRO_FR_P_AP": "V_FR_P_AP",
    "V_CATEGORY_MACRO_FR_P_F1max": "V_FR_P_F1max",
    "V_CATEGORY_MACRO_FR_PRO": "V_FR_PRO",
    "V_CATEGORY_MACRO_FR_G_pixel": "V_FR_G_pixel",
}


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_report(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_slots_by_category_visa(dataset, indices):
    cats = sorted({str(dataset.data_all[int(idx)]["cls_name"]) for idx in indices})
    return {cat: [p for p, idx in enumerate(indices) if str(dataset.data_all[int(idx)]["cls_name"]) == cat] for cat in cats}


def collect_visa_pool(seed, device, smoke):
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
    for p in align_model.parameters():
        p.requires_grad_(False)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    if smoke:
        chosen = []
        seen = defaultdict(lambda: {0: 0, 1: 0})
        for pos, idx in enumerate(indices):
            row = dataset.data_all[int(idx)]
            cat = str(row["cls_name"])
            lab = int(row["anomaly"])
            if seen[cat][lab] < 2:
                chosen.append((pos, idx))
                seen[cat][lab] += 1
            if len(chosen) >= 48:
                break
        patch = np.stack([np.asarray(patch[p], dtype=np.float32) for p, _idx in chosen], axis=0)
        indices = [idx for _p, idx in chosen]
    slots = source_slots_by_category_visa(dataset, indices)
    labels_by_pos = {}
    for pos, idx in enumerate(indices):
        labels_by_pos[pos] = patch_labels(dataset[int(idx)]).astype(np.int64)
    rng = np.random.default_rng(seed)
    groups = {}
    for cat, positions in slots.items():
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            y = labels_by_pos[int(pos)]
            z0 = remove_pc1(np.asarray(patch[int(pos)], dtype=np.float32), q_pc1)
            z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
            with torch.no_grad():
                z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
            pools = [("normal_image", np.flatnonzero(y == 0))] if y.max() == 0 else [("anomaly_mask_out", np.flatnonzero(y == 0)), ("anomaly_mask", np.flatnonzero(y == 1))]
            for origin, ids in pools:
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(x.shape[0] for x in parts[origin])
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin].append(z[ids])
        xs, ys = [], []
        for origin, cls in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if parts[origin]:
                arr = np.concatenate(parts[origin], axis=0).astype(np.float32)
                xs.append(arr)
                ys.extend([cls] * arr.shape[0])
        if xs and len(set(ys)) == 2:
            groups[cat] = {"x": np.concatenate(xs, axis=0), "y": np.asarray(ys, dtype=np.float32)}
    return {"source": "Full VisA", "use_align": True, "q_pc1": q_pc1, "align_model": align_model, "m": m_c.astype(np.float32), "alpha": alpha, "init_v": init_v.astype(np.float32), "groups": groups}


def collect_mvtec_pool(seed, smoke):
    dataset_name, dataset, indices, patch = load_full_mvtec_source(smoke)
    cats = sorted({str(dataset.data_all[int(idx)]["cls_name"]) for idx in indices})
    slots = {cat: [p for p, idx in enumerate(indices) if str(dataset.data_all[int(idx)]["cls_name"]) == cat] for cat in cats}
    rng = np.random.default_rng(seed)
    groups = {}
    for cat, positions in slots.items():
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            idx = int(indices[pos])
            item = dataset[idx]
            y = patch_labels(item).astype(np.int64)
            z = norm_np(np.asarray(patch[pos], dtype=np.float32).reshape(SIDE * SIDE, 1024))
            pools = [("normal_image", np.flatnonzero(y == 0))] if int(item["anomaly"]) == 0 else [("anomaly_mask_out", np.flatnonzero(y == 0)), ("anomaly_mask", np.flatnonzero(y == 1))]
            for origin, ids in pools:
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(x.shape[0] for x in parts[origin])
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin].append(z[ids])
        xs, ys = [], []
        for origin, cls in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if parts[origin]:
                arr = np.concatenate(parts[origin], axis=0).astype(np.float32)
                xs.append(arr)
                ys.extend([cls] * arr.shape[0])
        if xs and len(set(ys)) == 2:
            groups[cat] = {"x": np.concatenate(xs, axis=0), "y": np.asarray(ys, dtype=np.float32)}
    pos = np.concatenate([g["x"][g["y"] > 0.5] for g in groups.values()], axis=0)
    neg = np.concatenate([g["x"][g["y"] < 0.5] for g in groups.values()], axis=0)
    pos_mu, neg_mu = unit_np(pos.mean(axis=0)), unit_np(neg.mean(axis=0))
    m = (0.5 * (pos_mu + neg_mu)).astype(np.float32)
    d = (0.5 * (pos_mu - neg_mu)).astype(np.float32)
    alpha = float(np.linalg.norm(d))
    return {"source": "Full MVTec", "use_align": False, "q_pc1": None, "align_model": None, "m": m, "alpha": alpha, "init_v": unit_np(d).astype(np.float32), "groups": groups}


def balanced_batch(groups, rng):
    names = list(groups)
    per_cat = max(2, BATCH_SIZE // max(1, len(names)))
    xs, ys = [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order]


def train_contrast(source_info, seed, mode, device, smoke):
    ckpt = OUT / "checkpoints" / f"{source_info['source'].replace(' ', '_')}_seed{seed}_{mode}.pt"
    if ckpt.exists() and not smoke:
        return torch.load(ckpt, map_location="cpu", weights_only=False)["v"].astype(np.float32), True
    rng = np.random.default_rng(seed)
    groups = source_info["groups"]
    total_samples = sum(len(g["y"]) for g in groups.values())
    steps_per_epoch = min(3, max(1, int(np.ceil(total_samples / BATCH_SIZE)))) if smoke else max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    param = torch.nn.Parameter(torch.tensor(source_info["init_v"], dtype=torch.float32, device=device))
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([param], lr=LR, weight_decay=0.0)
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for _step in range(steps_per_epoch):
            xb, yb = balanced_batch(groups, rng)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(param, dim=-1)
            logits = contrast_logits(z, m, source_info["alpha"], v)
            task, _bce, _dice = balanced_loss(logits, y)
            extra = torch.zeros((), device=device)
            if mode == "ntr":
                neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
                k = max(1, int(np.ceil(float(len(neg)) * TOP_NEG_FRAC)))
                idx = neg[torch.topk(logits.detach()[neg], k=k, largest=True).indices]
                extra = F.softplus(logits[idx]).mean()
            loss = task + extra
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[PaperReady] train {source_info['source']} seed={seed} mode={mode} epoch={epoch}/{EPOCHS} loss={np.mean(losses):.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(param, dim=-1).detach().cpu().numpy().astype(np.float32)
    if not smoke:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"v": v, "seed": seed, "mode": mode, "source": source_info["source"]}, ckpt)
    return v, False


def score_feature_grid(model_info, v, patch_np, device):
    m_t = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    if model_info["use_align"]:
        z = aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).unsqueeze(0)
        return logits_from_z(z, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    z = norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    zt = torch.tensor(z, dtype=torch.float32, device=device)
    return contrast_logits(zt, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


def evaluate_selected_maps(source, target, maps, metas, variants):
    old = tuple(lzp.VARIANTS)
    try:
        lzp.VARIANTS = tuple(variants)
        return evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old


def eval_core_variant(model_info, v, cells, device, smoke):
    rows = []
    for _cell_id, target, dataset, idx_or_rows, patch, kind in cells:
        if (model_info["source"], target) not in MULTISEED_PAIRS:
            continue
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        refs = idx_or_rows[:n]
        print(f"[PaperReady] eval core {model_info['source']} -> {target} n={n}", flush=True)
        shim = dict(model_info)
        shim["v1"] = v
        maps = {"B0_HardNormal": []}
        metas = []
        for pos, ref in enumerate(refs):
            item = dataset[int(ref)] if kind == "source_dataset" else dataset[pos]
            cat = str(item.get("cls_name", target)) if kind == "source_dataset" else str(dict(ref).get("category", item.get("cls_name", target)))
            label = int(item.get("anomaly", 0)) if kind == "source_dataset" else int(dict(ref).get("label", item.get("anomaly", 0)))
            gt = (np.squeeze(item["img_mask"].detach().cpu().numpy()) > 0.5).astype(np.uint8)
            maps["B0_HardNormal"].append(score_feature_grid(shim, v, patch[pos], device))
            metas.append({"category": cat, "label": label, "mask": gt})
        rr, _cc = evaluate_selected_maps(model_info["source"], target, maps, metas, ["B0_HardNormal"])
        r = next(x for x in rr if x["variant"] == "B0_HardNormal")
        rows.append(r)
    return rows


def existing_seed0_rows():
    rows = []
    full_res = Path("results/phase_full_resolution_benchmark_revalidation/formal/FULL_RES_RAW_METRICS.csv")
    if full_res.exists():
        variant_map = {"M0_StateAligned": "StateAligned", "M1_HardNormal": "StateAligned+NTR"}
        for r in read_csv(full_res):
            if r.get("variant") not in variant_map:
                continue
            out = {
                "source": r.get("source", ""),
                "dataset": r.get("dataset", ""),
                "variant": "B0_HardNormal",
                "aggregation": "CATEGORY_MACRO",
                "score_resolution": r.get("score_source_resolution", "37x37"),
                "seed0_method": variant_map[r.get("variant")],
            }
            for src, dst in FULL_RES_METRIC_MAP.items():
                out[dst] = r.get(src, "")
            rows.append(out)
    for path in (
        Path("results/phase_parameter_free_local_zoom_pyramid_fullres/formal/RAW_METRICS.csv"),
        Path("results/phase_locked_local_reencoding_holdout_validation/formal/HOLDOUT_FULLRES_METRICS.csv"),
    ):
        if path.exists():
            rows.extend(read_csv(path))
    return rows


def seed0_multiseed_rows(seed0_rows):
    rows = []
    seen = set()
    for r in seed0_rows:
        key_pair = (r.get("source"), r.get("dataset"))
        if key_pair not in MULTISEED_PAIRS:
            continue
        method = r.get("seed0_method")
        if method in {"StateAligned", "StateAligned+NTR"}:
            key = (0, method, r.get("source"), r.get("dataset"))
            if key not in seen:
                rows.append({"seed": 0, "method": method, **r})
                seen.add(key)
    for r in seed0_rows:
        key_pair = (r.get("source"), r.get("dataset"))
        if key_pair in INDUSTRIAL_FOCUS and r.get("variant") == "E2_GlobalLocalZoomPyramid":
            key = (0, "StateAligned+NTR+LRP", r.get("source"), r.get("dataset"))
            if key not in seen:
                rows.append({"seed": 0, "method": "StateAligned+NTR+LRP", **r})
                seen.add(key)
    return rows


def summarize_multiseed(rows):
    out = []
    keys = sorted({(r["source"], r["dataset"], r["method"]) for r in rows})
    for source, dataset, method in keys:
        part = [r for r in rows if (r["source"], r["dataset"], r["method"]) == (source, dataset, method)]
        row = {"source": source, "dataset": dataset, "method": method, "seeds": len(part)}
        for k in ("M_FR_P_AUROC", "M_FR_P_AP", "M_FR_P_F1max", "M_FR_PRO"):
            vals = np.asarray([fnum(r, k) for r in part], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
            row[f"{k}_std"] = float(vals.std(ddof=0)) if vals.size else float("nan")
        out.append(row)
    return out


def build_tables(seed0_rows):
    ind, med, sens = [], [], []
    med_seen = set()
    for r in seed0_rows:
        source, ds, var = r.get("source"), r.get("dataset"), r.get("variant")
        if ds in INDUSTRIAL_TABLE_DATASETS and var == "E2_GlobalLocalZoomPyramid":
            row = {"source": source, "dataset": ds, "official_method": "Core+LRP", **{k: r.get(k, "") for k in METRIC_KEYS}}
            ind.append(row)
        if ds in MEDICAL_TABLE_DATASETS and var == "B0_HardNormal" and r.get("seed0_method") != "StateAligned" and np.isfinite(fnum(r, "M_FR_P_AP")):
            med_key = (source, ds)
            if med_key in med_seen:
                continue
            med_seen.add(med_key)
            row = {"source": source, "dataset": ds, "official_method": "Core only", **{k: r.get(k, "") for k in METRIC_KEYS}}
            med.append(row)
    sens.append({"domain": "Industrial", "LRP_mean_delta_M_FR_P_AP": 4.2464, "decision": "use Core+LRP"})
    sens.append({"domain": "Medical", "LRP_mean_delta_M_FR_P_AP": -6.5313, "decision": "use Core only"})
    return ind, med, sens


def measure_efficiency(seed0_rows, device, smoke):
    repeats = 3 if smoke else 30
    warmup = 1 if smoke else 10
    views = {"Core": 1, "Core+LRP": 5}
    try:
        preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
        clip_model, *_ = load_components(str(CHECKPOINT), device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        total_params = int(sum(p.numel() for p in clip_model.parameters()) + 2048)
        trainable_params = 2048
        cells, _audits = protocol_cells("Full VisA", device, True)
        _cid, target, dataset, idx_or_rows, _patch, kind = next(c for c in cells if c[1] in {"MVTec", "BTAD"})
        ref = idx_or_rows[0]
        item = dataset[int(ref)] if kind == "source_dataset" else dataset[0]
        img = item.get("img", item.get("image"))
        if isinstance(img, torch.Tensor):
            tensor = img.unsqueeze(0).to(device, non_blocking=True)
        else:
            tensor = preprocess(img).unsqueeze(0).to(device, non_blocking=True)

        def forward_once(n_views):
            batch = tensor.repeat(n_views, 1, 1, 1)
            with torch.no_grad():
                out = visual_outputs(clip_model, batch, grad=False)
                _ = normalized_patch_feature(out).float().mean()
            if device == "cuda":
                torch.cuda.synchronize()

        rows = []
        for method, n_views in views.items():
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            for _ in range(warmup):
                forward_once(n_views)
            t0 = time.perf_counter()
            for _ in range(repeats):
                forward_once(n_views)
            elapsed = time.perf_counter() - t0
            latency = 1000.0 * elapsed / repeats
            peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else float("nan")
            rows.append({
                "method": method,
                "latency_ms_per_image": latency,
                "FPS": 1000.0 / latency if latency > 0 else float("nan"),
                "peak_gpu_memory_mb": peak,
                "views": n_views,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "device": device,
                "benchmark_target": target,
                "batch": 1,
                "warmup": warmup,
                "repeats": repeats,
            })
        return rows
    except Exception as exc:
        return [{"method": "Core", "latency_ms_per_image": "ERROR", "FPS": "ERROR", "peak_gpu_memory_mb": "ERROR", "views": 1, "total_params": "ERROR", "trainable_params": 2048, "error": repr(exc)},
                {"method": "Core+LRP", "latency_ms_per_image": "ERROR", "FPS": "ERROR", "peak_gpu_memory_mb": "ERROR", "views": 5, "total_params": "ERROR", "trainable_params": 2048, "error": repr(exc)}]


def build_sota_gap(official_rows):
    by_pair = {(r.get("source"), r.get("dataset")): r for r in official_rows}
    gaps = []
    for ref in sota_reference():
        pair = (ref.get("training_source"), ref.get("target"))
        ours = by_pair.get(pair)
        if not ours:
            continue
        row = {
            "paper": ref.get("paper"),
            "comparison_type": ref.get("comparison_type", "REFERENCE_ONLY"),
            "source": pair[0],
            "dataset": pair[1],
            "ours_method": ours.get("official_method", ""),
        }
        for ours_key, ref_key, out_key in (
            ("M_FR_P_AUROC", "P_AUROC", "Delta_P_AUROC"),
            ("M_FR_P_AP", "P_AP", "Delta_P_AP"),
            ("M_FR_P_F1max", "P_F1max", "Delta_P_F1max"),
            ("M_FR_PRO", "PRO", "Delta_PRO"),
            ("M_FR_G_pixel", "G_pixel", "Delta_G_pixel"),
            ("M_FR_I_AUROC", "I_AUROC", "Delta_I_AUROC"),
            ("M_FR_I_AP", "I_AP", "Delta_I_AP"),
            ("M_FR_I_F1max", "I_F1max", "Delta_I_F1max"),
        ):
            ov = fnum(ours, ours_key)
            rv = ref.get(ref_key)
            try:
                rvf = float(rv)
            except Exception:
                rvf = float("nan")
            row[f"ours_{ours_key}"] = ov
            row[f"ref_{ref_key}"] = rv
            row[out_key] = (ov - rvf) if np.isfinite(ov) and np.isfinite(rvf) else "NA"
        gaps.append(row)
    return gaps


def mechanism_table():
    specs = [
        ("State / contrast / direction decomposition", "State contrast direction dominated; magnitude/common component controls were separated in prior formal audits.", "phase_state_pair_decomposition / phase_state_contrast_direction_magnitude"),
        ("HardNormal = normal tail-risk evidence", "Symmetric HardNormal/NTR improved cross-dataset G and replicated across VisA/MVTec source.", "phase_hard_normal_contrast / phase_hard_normal_cross_source_mvtec"),
        ("Prediction-equivalent parameterization causal study", "Canonical parameterization reproduced B0 before causal decomposition, avoiding gauge artifacts.", "phase_prediction_null_state_gauge_causal_test / phase_canonical_joint_contrast_causal_decomposition"),
        ("Small/sparse bottleneck correlation", "Full-res bottleneck audit found P-AP tracks anomaly patch occupancy and thinness.", "phase_final_bottleneck_attribution / phase_full_resolution_benchmark_revalidation"),
        ("Local re-encoding vs resolution control", "Locked LRP holdout: industrial AP improves; medical AP drops; E2-E3 mean abs AP 0.7041.", "phase_parameter_free_local_zoom_pyramid_fullres / phase_locked_local_reencoding_holdout_validation"),
    ]
    return [{"mechanism": a, "finding": b, "source_phase": c} for a, b, c in specs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed0 = existing_seed0_rows()
    multiseed = seed0_multiseed_rows(seed0)
    partial_path = OUT / "MULTISEED_METRICS.partial.csv"
    if args.skip_train and partial_path.exists():
        partial_rows = read_csv(partial_path)
        if len(partial_rows) > len(multiseed):
            multiseed = partial_rows
    if not args.skip_train:
        preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
        clip_model, *_ = load_components(str(CHECKPOINT), device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        for seed in (1, 2):
            for collect in (lambda: collect_visa_pool(seed, device, args.smoke), lambda: collect_mvtec_pool(seed, args.smoke)):
                src = collect()
                v_state, _ = train_contrast(src, seed, "statealigned", device, args.smoke)
                v_ntr, _ = train_contrast(src, seed, "ntr", device, args.smoke)
                src["v1"] = v_ntr
                cells, _audits = protocol_cells(src["source"], device, args.smoke)
                core_rows = eval_core_variant(src, v_state, cells, device, args.smoke)
                for r in core_rows:
                    multiseed.append({"seed": seed, "method": "StateAligned", **r})
                ntr_rows = eval_core_variant(src, v_ntr, cells, device, args.smoke)
                for r in ntr_rows:
                    multiseed.append({"seed": seed, "method": "StateAligned+NTR", **r})
                for cell in cells:
                    _cid, target, dataset, idx_or_rows, patch, kind = cell
                    if (src["source"], target) not in INDUSTRIAL_FOCUS:
                        continue
                    n = min(24, patch.shape[0]) if args.smoke else patch.shape[0]
                    maps, metas, _scale = build_maps(src, clip_model, preprocess, patch[:n], idx_or_rows[:n], dataset, kind, target, device)
                    rr, _cc = evaluate_selected_maps(src["source"], target, maps, metas, ["E2_GlobalLocalZoomPyramid"])
                    e2 = next(r for r in rr if r["variant"] == "E2_GlobalLocalZoomPyramid")
                    multiseed.append({"seed": seed, "method": "StateAligned+NTR+LRP", **e2})
                write_csv(OUT / "MULTISEED_METRICS.partial.csv", multiseed)
    seed0_rows = existing_seed0_rows()
    ind, med, sens = build_tables(seed0_rows)
    summary = summarize_multiseed(multiseed)
    mech = mechanism_table()
    eff = measure_efficiency(seed0_rows, device, args.smoke)
    gaps = build_sota_gap(ind + med)
    write_csv(OUT / "MULTISEED_METRICS.csv", multiseed)
    write_csv(OUT / "MULTISEED_SUMMARY.csv", summary)
    write_csv(OUT / "EFFICIENCY.csv", eff)
    write_csv(OUT / "INDUSTRIAL_MAIN_TABLE.csv", ind)
    write_csv(OUT / "MEDICAL_GENERALIZATION_TABLE.csv", med)
    write_csv(OUT / "DOMAIN_SENSITIVITY_TABLE.csv", sens)
    write_csv(OUT / "MECHANISM_ABLATION_TABLE.csv", mech)
    write_csv(OUT / "SOTA_GAP_FINAL.csv", gaps)
    write_json(OUT / "OUTCOME.json", {"outcome": "PAPER_READY_FINAL_VALIDATION_COMPLETE", "runtime_sec": time.time() - start, "device": device, "multiseed_rows": len(multiseed)})
    write_report(OUT / "PAPER_RESULT_SUMMARY.md", [
        "# Paper-Ready Final Validation",
        "",
        "OUTCOME: `PAPER_READY_FINAL_VALIDATION_COMPLETE`",
        "",
        "Final method is frozen and domain-conditional without target tuning:",
        "",
        "- Industrial: Canonical Two-State + Normal Tail-Risk Reinforcement + fixed 2x2 Local Re-encoding Pyramid.",
        "- Medical: Canonical Two-State + Normal Tail-Risk Reinforcement only.",
        "",
        "All official tables use full-resolution, category-macro metrics (`M_FR_*`). Patch-grid metrics are retained only as analysis fields in the raw multiseed CSV.",
        "",
        f"Rows in `MULTISEED_METRICS.csv`: {len(multiseed)}",
        f"Rows in `INDUSTRIAL_MAIN_TABLE.csv`: {len(ind)}",
        f"Rows in `MEDICAL_GENERALIZATION_TABLE.csv`: {len(med)}",
        "",
        "Domain sensitivity is carried forward from the locked validation:",
        "",
        "- Industrial LRP mean P-AP delta: +4.2464.",
        "- Medical LRP mean P-AP delta: -6.5313; medical main table therefore reports Core only.",
        "",
        f"Runtime: {time.time() - start:.1f}s",
    ])
    print("[PaperReady] OUTCOME=PAPER_READY_FINAL_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
