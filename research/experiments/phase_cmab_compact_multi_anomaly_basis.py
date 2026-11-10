import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_contrast import (
    PC1_DEBIAS_OUT,
    collect_source_patch_pool as collect_visa_pool,
    load_payload,
)
from experiments.phase_hard_normal_cross_source_mvtec import (
    collect_source_pool as collect_mvtec_pool,
    load_full_mvtec_source,
    source_slots_by_category as mvtec_slots_by_category,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    METRICS,
    PROTOCOLS,
    item_at,
    item_meta,
    mask_np,
    pg_metrics,
    quadrant_crops,
    summarize_category,
    torch_resize,
)
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category as visa_slots_by_category
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, LR, SIDE, norm_np
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_cmab_compact_multi_anomaly_basis/formal")
EPOCHS = 10
TOP_NEG_FRAC = 0.10
DIV_WEIGHT = 1.0
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "DTD-Synthetic"),
)
FULL_PAIRS = (
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
)
K_VARIANTS = {
    "E0_K1_Baseline": 1,
    "E1_K2_CMAB": 2,
    "E2_K4_CMAB": 4,
    "E3_K8_CMAB": 8,
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
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def state_pair_from_model_info(model_info):
    m = np.asarray(model_info["m"], dtype=np.float32)
    alpha = float(model_info["alpha"])
    v = np.asarray(model_info["v1"], dtype=np.float32)
    s_n = m - alpha * v
    s_a = m + alpha * v
    return s_n.astype(np.float32), s_a.astype(np.float32)


def multi_state_logits(z, s_n_raw, s_a_raw):
    z = F.normalize(z, dim=-1)
    s_n = F.normalize(s_n_raw, dim=-1)
    s_a = F.normalize(s_a_raw, dim=-1)
    l_n = z @ s_n
    l_a = z @ s_a.T
    return torch.logsumexp(l_a, dim=-1) - l_n


def diversity_loss(s_a_raw):
    if s_a_raw.shape[0] <= 1:
        return torch.zeros((), dtype=s_a_raw.dtype, device=s_a_raw.device)
    s = F.normalize(s_a_raw, dim=-1)
    sim = s @ s.T
    k = sim.shape[0]
    off = sim[~torch.eye(k, dtype=torch.bool, device=sim.device)]
    return off.mean()


def balanced_category_batch(groups, rng, total_batch):
    names = list(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, origins = [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            if ids.size == 0:
                continue
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            origins.extend(data["origin"][choice].tolist())
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], np.asarray(origins, dtype=object)[order]


def init_anomaly_basis(s_a, s_n, k, seed):
    if k == 1:
        return s_a[None].astype(np.float32)
    rng = np.random.default_rng(seed)
    base = s_a.astype(np.float32)
    direction = base - s_n.astype(np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    basis = []
    for i in range(k):
        noise = rng.normal(size=base.shape).astype(np.float32)
        noise = noise - float(noise @ direction) * direction
        noise = noise / max(float(np.linalg.norm(noise)), 1e-12)
        scale = 0.01 * max(float(np.linalg.norm(base)), 1e-6)
        basis.append(base + scale * noise)
    return np.stack(basis, axis=0).astype(np.float32)


def train_cmab_variant(source_name, variant, k, groups, s_n0, s_a0, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{source_name.replace(' ', '_')}_{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["s_n"].astype(np.float32), payload["s_a"].astype(np.float32), payload["curves"], True
    if k == 1:
        return s_n0.astype(np.float32), s_a0[None].astype(np.float32), [], True

    print(f"[CMAB] train {source_name} {variant} K={k}", flush=True)
    torch.manual_seed(SEED + k)
    s_n = torch.nn.Parameter(torch.tensor(s_n0, dtype=torch.float32, device=device))
    s_a = torch.nn.Parameter(torch.tensor(init_anomaly_basis(s_a0, s_n0, k, SEED + k), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([s_n, s_a], lr=LR, weight_decay=0.0)
    total_samples = sum(len(g["y"]) for g in groups.values())
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 2)
    rng = np.random.default_rng(SEED + k)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, _origin = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            logits = multi_state_logits(z, s_n, s_a)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            kk = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            selected = neg_idx[torch.topk(logits.detach()[neg_idx], k=kk, largest=True).indices]
            hard_loss = F.softplus(logits[selected]).mean()
            div = diversity_loss(s_a)
            loss = task_loss + hard_loss + DIV_WEIGHT * div
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(div.detach().cpu()), float(bce.cpu()), float(dice.cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "source": source_name,
            "variant": variant,
            "K": k,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "task_loss": float(arr[:, 1].mean()),
            "hard_normal_loss": float(arr[:, 2].mean()),
            "diversity_loss": float(arr[:, 3].mean()),
            "bce": float(arr[:, 4].mean()),
            "dice": float(arr[:, 5].mean()),
        }
        curves.append(row)
        print(f"[CMAB] {source_name} {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} div={row['diversity_loss']:.4f}", flush=True)
    payload = {
        "s_n": s_n.detach().cpu().numpy().astype(np.float32),
        "s_a": s_a.detach().cpu().numpy().astype(np.float32),
        "curves": curves,
    }
    torch.save(payload, ckpt)
    return payload["s_n"], payload["s_a"], curves, False


def load_visa_groups(model_info, device, smoke):
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    align_model = make_model(np.stack(state_pair_from_model_info(model_info), axis=0), corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = visa_slots_by_category(dataset, indices, smoke)
    groups, audit = collect_visa_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    return groups, audit


def load_mvtec_groups(smoke):
    _dataset_name, dataset, indices, patch = load_full_mvtec_source(smoke)
    slots = mvtec_slots_by_category(dataset, indices)
    return collect_mvtec_pool(dataset, indices, patch, slots)


@torch.no_grad()
def feature_grid(model_info, patch_np, device):
    if model_info["use_align"]:
        return aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).reshape(SIDE * SIDE, 1024)
    return torch.tensor(norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024)), dtype=torch.float32, device=device)


@torch.no_grad()
def score_from_grid(z, states, device):
    s_n_np, s_a_np = states
    s_n = torch.tensor(s_n_np, dtype=torch.float32, device=device)
    s_a = torch.tensor(s_a_np, dtype=torch.float32, device=device)
    score = multi_state_logits(z, s_n, s_a).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    return score


@torch.no_grad()
def local_zoom_features(model_info, clip_model, preprocess, img_path, device):
    crops = quadrant_crops(img_path)
    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patches = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)
    return [feature_grid(model_info, patches[i], device) for i in range(4)]


@torch.no_grad()
def local_zoom_map_from_features(crop_features, states, device):
    qs = [score_from_grid(crop_features[i], states, device) for i in range(4)]
    s = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    s[:SIDE, :SIDE] = qs[0]
    s[:SIDE, SIDE:] = qs[1]
    s[SIDE:, :SIDE] = qs[2]
    s[SIDE:, SIDE:] = qs[3]
    return s


def evaluate_maps(source, target, maps, metas, variants):
    raw_rows, cat_rows = [], []
    for variant in variants:
        print(f"[CMAB] metrics {source} -> {target} {variant}", flush=True)
        by_cat = {}
        for i, meta in enumerate(metas):
            d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "maps": []})
            d["labels"].append(meta["label"])
            d["image_scores"].append(top1_image_score(maps[variant][i]))
            d["masks"].append(meta["mask"])
            d["maps"].append(np.asarray(maps[variant][i], dtype=np.float32))
        pg_cat_rows = []
        fr_cat_by_protocol = {p: [] for p in PROTOCOLS}
        for cat, vals in sorted(by_cat.items()):
            pg = pg_metrics(vals["labels"], vals["image_scores"], vals["masks"], vals["maps"])
            pg_row = {"source": source, "dataset": target, "category": cat, "variant": variant, "protocol": "PG", "aggregation": "CATEGORY", "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}"}
            for met, val in pg.items():
                pg_row[f"PG_{met.replace('-', '_')}"] = val
            cat_rows.append(pg_row)
            pg_cat_rows.append(pg_row)
            for pkey, pcfg in PROTOCOLS.items():
                scores = [torch_resize(s, m.shape, align_corners=pcfg["align_corners"]) for s, m in zip(vals["maps"], vals["masks"])]
                fr = summarize_category(vals["labels"], vals["image_scores"], vals["masks"], scores)
                fr_row = {"source": source, "dataset": target, "category": cat, "variant": variant, "protocol": pkey, "protocol_name": pcfg["name"], "aggregation": "CATEGORY", "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}"}
                for met, val in fr.items():
                    fr_row[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = val
                cat_rows.append(fr_row)
                fr_cat_by_protocol[pkey].append(fr_row)
        raw = {"source": source, "dataset": target, "variant": variant, "aggregation": "CATEGORY_MACRO", "score_resolution": f"{maps[variant][0].shape[0]}x{maps[variant][0].shape[1]}"}
        for met in METRICS:
            vals = np.asarray([fnum(r, f"PG_{met.replace('-', '_')}") for r in pg_cat_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            raw[f"PG_{met.replace('-', '_')}"] = float(vals.mean()) if vals.size else float("nan")
        for pkey, pcfg in PROTOCOLS.items():
            for met in METRICS:
                vals = np.asarray([fnum(r, f"{pcfg['prefix']}_{met.replace('-', '_')}") for r in fr_cat_by_protocol[pkey]], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                raw[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = float(vals.mean()) if vals.size else float("nan")
        raw_rows.append(raw)
    return raw_rows, cat_rows


def build_pair_maps(model_info, states_by_variant, clip_model, preprocess, patch, refs, dataset, kind, target, device, smoke):
    variants = list(states_by_variant)
    n = min(24, len(refs)) if smoke else len(refs)
    maps = {v: [] for v in variants}
    metas = []
    for pos, ref in enumerate(refs[:n]):
        if pos % 25 == 0:
            print(f"[CMAB] {model_info['source']} -> {target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        if not path:
            raise ValueError(f"missing img_path for {model_info['source']} -> {target} pos={pos}")
        metas.append({"category": cat, "label": label, "mask": mask_np(item), "img_path": path})
        global_z = feature_grid(model_info, patch[pos], device)
        crop_z = local_zoom_features(model_info, clip_model, preprocess, path, device)
        for variant, states in states_by_variant.items():
            global37 = score_from_grid(global_z, states, device)
            local74 = local_zoom_map_from_features(crop_z, states, device)
            global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
            final74 = 0.5 * global74 + 0.5 * local74
            maps[variant].append(final74)
    return maps, metas


def basis_rows(source, variant, states, groups=None):
    s_n, s_a = states
    rows = []
    s = s_a / np.maximum(np.linalg.norm(s_a, axis=1, keepdims=True), 1e-12)
    sim = s @ s.T
    for i in range(sim.shape[0]):
        for j in range(i + 1, sim.shape[0]):
            rows.append({"source": source, "variant": variant, "record_type": "pairwise_cosine", "state_i": i, "state_j": j, "cosine": float(sim[i, j])})
    if groups is not None:
        counts = np.zeros((s_a.shape[0],), dtype=np.int64)
        total = 0
        with torch.no_grad():
            tn = torch.tensor(s_n, dtype=torch.float32)
            ta = torch.tensor(s_a, dtype=torch.float32)
            for data in groups.values():
                z = torch.tensor(data["x"], dtype=torch.float32)
                logits = F.normalize(z, dim=-1) @ F.normalize(ta, dim=-1).T
                idx = torch.argmax(logits, dim=-1).cpu().numpy()
                counts += np.bincount(idx, minlength=s_a.shape[0])
                total += int(idx.size)
        frac = counts / max(total, 1)
        eff = float(1.0 / np.sum(np.square(frac))) if frac.size and np.sum(frac) > 0 else float("nan")
        for i, f in enumerate(frac):
            rows.append({"source": source, "variant": variant, "record_type": "argmax_usage", "state_i": i, "patch_fraction": float(f), "effective_used_states": eff})
    return rows


def delta_summary(raw, pairs):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    variants = [v for v in K_VARIANTS if v != "E0_K1_Baseline"]
    for variant in variants:
        vals = []
        for source, target in pairs:
            b = by.get((source, target, "E0_K1_Baseline"))
            e = by.get((source, target, variant))
            if not (b and e):
                continue
            d = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            vals.append(d)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": d,
                "Delta_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
        arr = np.asarray(vals, dtype=np.float64)
        rows.append({
            "source": "SUMMARY",
            "dataset": "PILOT_OR_FULL",
            "variant": variant,
            "mean_delta_M_FR_P_AP": float(arr.mean()) if arr.size else float("nan"),
            "median_delta_M_FR_P_AP": float(np.median(arr)) if arr.size else float("nan"),
            "improved": int(np.sum(arr > 0)) if arr.size else 0,
            "total": int(arr.size),
        })
    return rows


def best_pilot_variant(summary_rows):
    candidates = [r for r in summary_rows if r.get("source") == "SUMMARY"]
    if not candidates:
        return None, float("nan"), 0, 0
    best = max(candidates, key=lambda r: fnum(r, "mean_delta_M_FR_P_AP"))
    return best["variant"], fnum(best, "mean_delta_M_FR_P_AP"), int(float(best.get("improved", 0))), int(float(best.get("total", 0)))


def states_collapsed(rows, source, variant):
    usage = [r for r in rows if r.get("source") == source and r.get("variant") == variant and r.get("record_type") == "argmax_usage"]
    pair = [r for r in rows if r.get("source") == source and r.get("variant") == variant and r.get("record_type") == "pairwise_cosine"]
    eff = max([fnum(r, "effective_used_states") for r in usage], default=1.0)
    max_cos = max([fnum(r, "cosine") for r in pair], default=1.0)
    return eff < 1.5 or max_cos > 0.995


def decide(pilot_summary, basis_diag, run_full, full_summary=None):
    best, mean_delta, improved, total = best_pilot_variant(pilot_summary)
    sources = {s for s, _t in PILOT_PAIRS}
    collapsed = best is None or any(states_collapsed(basis_diag, s, best) for s in sources)
    pilot_pass = best is not None and mean_delta >= 1.0 and improved >= 3 and not collapsed
    if not pilot_pass:
        if collapsed:
            return "MULTI_ANOMALY_STATES_COLLAPSE_OR_ARE_REDUNDANT", {"best_pilot_variant": best, "pilot_mean_delta_P_AP": mean_delta, "pilot_improved": improved, "pilot_total": total, "pilot_pass": False}
        return "MULTI_ANOMALY_BASIS_DOES_NOT_IMPROVE_TRANSFER", {"best_pilot_variant": best, "pilot_mean_delta_P_AP": mean_delta, "pilot_improved": improved, "pilot_total": total, "pilot_pass": False}
    if not run_full or full_summary is None:
        return "PILOT_PASSED_FULL_EVALUATION_PENDING", {"best_pilot_variant": best, "pilot_mean_delta_P_AP": mean_delta, "pilot_improved": improved, "pilot_total": total, "pilot_pass": True}
    full_rows = [r for r in full_summary if r.get("source") == "SUMMARY" and r.get("variant") == best]
    if not full_rows:
        return "PILOT_PASSED_FULL_EVALUATION_PENDING", {"best_pilot_variant": best, "pilot_mean_delta_P_AP": mean_delta, "pilot_improved": improved, "pilot_total": total, "pilot_pass": True}
    fr = full_rows[0]
    full_mean = fnum(fr, "mean_delta_M_FR_P_AP")
    full_improved = int(float(fr.get("improved", 0)))
    if full_mean >= 1.5 and full_improved >= 6:
        out = "MULTI_ANOMALY_BASIS_SIGNIFICANTLY_IMPROVES_TRANSFER"
    elif full_mean >= 1.0 and full_improved >= 6:
        out = "MULTI_ANOMALY_BASIS_HAS_WEAK_BUT_REAL_GAIN"
    else:
        out = "MULTI_ANOMALY_BASIS_DOES_NOT_IMPROVE_TRANSFER"
    return out, {"best_pilot_variant": best, "pilot_mean_delta_P_AP": mean_delta, "pilot_improved": improved, "pilot_total": total, "pilot_pass": True, "full_mean_delta_P_AP": full_mean, "full_improved": full_improved}


def write_report(raw, deltas, basis_diag, outcome, stats, runtime):
    lines = [
        "# Phase CMAB: Compact Multi-Anomaly Basis",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR + fixed 2x2 Local Re-encoding Pyramid. CLIP visual backbone, alignment/interface, NTR rule, and Fixed LRP evaluation are unchanged.",
        "",
        "CMAB differs from prompt/prototype/memory methods by using only a compact latent anomaly-state basis: no text prompts, no ViT visual tokens, no category-conditioned prototypes, no exemplar retrieval, and no patch router/expert gating.",
        "",
        "## Pilot Metrics",
        "",
        "| Source | Target | Variant | K | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in raw:
        if (r["source"], r["dataset"]) not in PILOT_PAIRS:
            continue
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {K_VARIANTS[r['variant']]} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    lines.extend(["", "## P-AP Delta vs K=1", "", "| Source | Target | Variant | Delta P-AP |", "|---|---|---|---:|"])
    for r in deltas:
        if r.get("source") == "SUMMARY":
            continue
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} |")
    lines.extend(["", "## Summary", "", "| Variant | mean Delta P-AP | median | improved/total |", "|---|---:|---:|---:|"])
    for r in deltas:
        if r.get("source") != "SUMMARY":
            continue
        lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {int(float(r.get('improved',0)))}/{int(float(r.get('total',0)))} |")
    lines.extend(["", "## Basis Diagnostics", "", "| Source | Variant | effective used states | max pairwise cosine |", "|---|---|---:|---:|"])
    for source in sorted({r.get("source") for r in basis_diag if r.get("record_type") == "argmax_usage"}):
        for variant in K_VARIANTS:
            usage = [r for r in basis_diag if r.get("source") == source and r.get("variant") == variant and r.get("record_type") == "argmax_usage"]
            pair = [r for r in basis_diag if r.get("source") == source and r.get("variant") == variant and r.get("record_type") == "pairwise_cosine"]
            if not usage:
                continue
            eff = max(fnum(r, "effective_used_states") for r in usage)
            max_cos = max([fnum(r, "cosine") for r in pair], default=float("nan"))
            lines.append(f"| {source} | {variant} | {eff:.4f} | {max_cos:.6f} |")
    lines.extend(["", f"Decision stats: `{json.dumps(stats, default=json_default)}`", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CMAB] device={device}", flush=True)
    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    model_infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    groups_by_source = {}
    pool_audit = []
    states_by_source_variant = {}
    curves_all = []
    basis_diag = []
    for source, model_info in model_infos.items():
        print(f"[CMAB] prepare source pool {source}", flush=True)
        groups, audit = load_visa_groups(model_info, device, args.smoke) if source == "Full VisA" else load_mvtec_groups(args.smoke)
        groups_by_source[source] = groups
        pool_audit.extend({"source": source, **r} for r in audit)
        s_n0, s_a0 = state_pair_from_model_info(model_info)
        states_by_source_variant[source] = {}
        for variant, k in K_VARIANTS.items():
            s_n, s_a, curves, reused = train_cmab_variant(source, variant, k, groups, s_n0, s_a0, device, args.smoke)
            states_by_source_variant[source][variant] = (s_n, s_a)
            curves_all.extend(curves)
            for row in basis_rows(source, variant, (s_n, s_a), groups):
                row["checkpoint_reused"] = bool(reused)
                basis_diag.append(row)

    raw, percat = [], []
    pairs_to_run = list(FULL_PAIRS if args.full else PILOT_PAIRS)
    for source, target in pairs_to_run:
        model_info = model_infos[source]
        cells, _audits = protocol_cells(source, device, args.smoke)
        matches = [c for c in cells if c[1] == target]
        if not matches:
            raise RuntimeError(f"No target cell found for {source} -> {target}")
        _cell_id, _target, dataset, idx_or_rows, patch, kind = matches[0]
        n = min(24, len(idx_or_rows)) if args.smoke else len(idx_or_rows)
        print(f"[CMAB] evaluate {source} -> {target} n={n}", flush=True)
        maps, metas = build_pair_maps(model_info, states_by_source_variant[source], clip_model, preprocess, patch[:n], idx_or_rows[:n], dataset, kind, target, device, args.smoke)
        rr, cc = evaluate_maps(source, target, maps, metas, list(K_VARIANTS))
        raw.extend(rr)
        percat.extend(cc)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.partial.csv", percat)

    pilot_raw = [r for r in raw if (r["source"], r["dataset"]) in PILOT_PAIRS]
    pilot_delta = delta_summary(pilot_raw, PILOT_PAIRS)
    full_delta = delta_summary(raw, pairs_to_run) if args.full else None
    outcome, stats = decide(pilot_delta, basis_diag, args.full, full_delta)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", pilot_delta)
    if full_delta is not None:
        write_csv(OUT / "FULL_DELTA_SUMMARY.csv", full_delta)
    write_csv(OUT / "BASIS_USAGE.csv", basis_diag)
    write_csv(OUT / "TRAINING_CURVES.csv", curves_all)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_audit)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_adaptation": False,
        "clip_frozen": True,
        "fixed_lrp": True,
        "variants": K_VARIANTS,
        "diversity_regularization": "mean_pairwise_cosine",
        "diversity_weight": DIV_WEIGHT,
        "pilot_pairs": PILOT_PAIRS,
        "full_evaluation": bool(args.full),
    })
    write_report(raw, pilot_delta if full_delta is None else full_delta, basis_diag, outcome, stats, runtime)
    print(f"[CMAB] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true", help="Run the 8-transfer formal stage. Default only runs the 4-transfer pilot.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
