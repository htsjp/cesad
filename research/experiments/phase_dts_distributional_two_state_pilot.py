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

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    evaluate_maps,
    item_at,
    item_meta,
    mask_np,
    torch_resize,
)
import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import SEED
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np, patch_labels
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT_A = Path("results/phase_dts_a_symmetric_distributional_two_state/formal")
OUT_B = Path("results/phase_dts_b_asymmetric_normal_variation_geometry/formal")
PILOT = [("Full MVTec", "VisA"), ("Full VisA", "MPDD"), ("Full MVTec", "BTAD")]
MAX_RANK = 32
EXPLAINED = 0.90
GAMMA = 1.0
TOP_NEG_FRAC = 0.10


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


def fnum(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def source_dataset(source_name, smoke):
    if source_name == "Full VisA":
        _name, dataset, indices, _cls, patch, _y_img, cat_img = load_full_visa_source_lazy(smoke)
        labels_by_pos, _corrupt, _audit = source_label_maps(smoke)
        labels = [np.asarray(labels_by_pos.get(i, np.zeros((SIDE * SIDE,), dtype=np.int64)), dtype=np.float32) for i in range(len(indices))]
        cats = [str(c) for c in cat_img]
        return dataset, indices, patch, labels, cats
    _name, dataset, indices, patch = load_full_mvtec_source(smoke)
    labels, cats = [], []
    for idx in indices:
        item = dataset[int(idx)]
        labels.append(patch_labels(item).astype(np.float32))
        cats.append(str(item["cls_name"]))
    return dataset, indices, patch, labels, cats


@torch.no_grad()
def state_z(model_info, patch_np, device):
    if model_info["use_align"]:
        return aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).reshape(SIDE * SIDE, -1)
    z = norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    return torch.tensor(z, dtype=torch.float32, device=device)


def baseline_states_np(model_info):
    m = np.asarray(model_info["m"], dtype=np.float32)
    v = np.asarray(model_info["v1"], dtype=np.float32)
    alpha = float(model_info["alpha"])
    s_a = norm_np(m + alpha * v)
    s_n = norm_np(m - alpha * v)
    return s_n.astype(np.float32), s_a.astype(np.float32)


@torch.no_grad()
def q0_np(model_info, z, device):
    m = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v = torch.tensor(model_info["v1"], dtype=torch.float32, device=device)
    return contrast_logits(z, m, float(model_info["alpha"]), v).detach().cpu().numpy().astype(np.float32)


def residual_np(z, s):
    return (z - (z @ s).reshape(-1, 1) * s.reshape(1, -1)).astype(np.float32)


def select_source_residuals(source_name, model_info, device, smoke):
    cache = (OUT_A / f"{source_name.replace(' ', '_')}_source_residuals_{'smoke' if smoke else 'formal'}.npz")
    if cache.exists():
        z = np.load(cache)
        return {k: z[k].astype(np.float32) for k in z.files if k != "cats"}
    dataset, indices, patch, labels, cats = source_dataset(source_name, smoke)
    s_n, s_a = baseline_states_np(model_info)
    rng = np.random.default_rng(SEED)
    parts = {"N": [], "A": [], "N_q0": []}
    cap_per_cat_label = 4096 if not smoke else 512
    for cat in sorted(set(cats)):
        positions = [i for i, c in enumerate(cats) if c == cat]
        used = {0: 0, 1: 0}
        for pos in positions:
            y = labels[pos].reshape(-1)
            zt = state_z(model_info, patch[pos], device)
            z = zt.detach().cpu().numpy().astype(np.float32)
            q = q0_np(model_info, zt, device)
            for cls, key, state in ((0, "N", s_n), (1, "A", s_a)):
                ids = np.flatnonzero(y == cls)
                room = cap_per_cat_label - used[cls]
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[key].append(residual_np(z[ids], state))
                if cls == 0:
                    parts["N_q0"].append(q[ids])
                used[cls] += ids.size
    out = {}
    for key in ("N", "A", "N_q0"):
        out[key] = np.concatenate(parts[key], axis=0).astype(np.float32) if parts[key] else np.zeros((0, 1024), dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def fit_diag(residuals):
    var = np.var(residuals, axis=0).astype(np.float32) + 1e-6
    sigma2 = float(np.mean(var))
    invdiff = (1.0 / var - 1.0 / sigma2).astype(np.float32)
    return {"kind": "diag", "sigma2": sigma2, "invdiff": invdiff, "rank": int(var.size), "explained": 1.0, "residual_energy": float(np.mean(np.sum(residuals * residuals, axis=1)))}


def fit_lowrank(residuals, weights=None):
    x = residuals.astype(np.float64)
    if weights is None:
        cov = (x.T @ x) / max(x.shape[0], 1)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        cov = (x.T * w.reshape(1, -1)) @ x / max(float(w.sum()), 1e-12)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    total = float(vals.sum())
    csum = np.cumsum(vals) / max(total, 1e-12)
    rank = int(min(MAX_RANK, max(1, np.searchsorted(csum, EXPLAINED) + 1)))
    sigma2 = float(np.mean(vals[rank:])) if rank < vals.size else float(max(vals[-1], 1e-6))
    sigma2 = max(sigma2, 1e-6)
    lam = np.maximum(vals[:rank] - sigma2, 0.0).astype(np.float32)
    return {
        "kind": "lowrank",
        "sigma2": sigma2,
        "U": vecs[:, :rank].astype(np.float32),
        "lambda": lam,
        "rank": rank,
        "explained": float(csum[rank - 1]),
        "residual_energy": float(np.mean(np.sum(residuals * residuals, axis=1))),
    }


def delta_from_model(r, model):
    if model["kind"] == "zero":
        return np.zeros((r.shape[0],), dtype=np.float32)
    if model["kind"] == "diag":
        return (-0.5 * np.sum((r * r) * model["invdiff"].reshape(1, -1), axis=1)).astype(np.float32)
    coeff = r @ model["U"]
    invdiff = (1.0 / (model["sigma2"] + model["lambda"]) - 1.0 / model["sigma2"]).reshape(1, -1)
    return (-0.5 * np.sum((coeff * coeff) * invdiff, axis=1)).astype(np.float32)


def fit_models_for_source(source_name, model_info, device, smoke):
    cache = OUT_A / f"{source_name.replace(' ', '_')}_covariance_models_{'smoke' if smoke else 'formal'}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)
    res = select_source_residuals(source_name, model_info, device, smoke)
    qn = res["N_q0"].reshape(-1)
    weights = np.ones_like(qn, dtype=np.float32)
    if qn.size:
        k = max(1, int(math.ceil(qn.size * TOP_NEG_FRAC)))
        thr = np.partition(qn, qn.size - k)[qn.size - k]
        weights[qn >= thr] = 2.0
    models = {
        "N_diag": fit_diag(res["N"]),
        "A_diag": fit_diag(res["A"]),
        "N_low": fit_lowrank(res["N"]),
        "A_low": fit_lowrank(res["A"]),
        "N_weighted": fit_lowrank(res["N"], weights=weights),
    }
    zero = {"kind": "zero", "sigma2": 1.0, "rank": 0, "explained": 0.0, "residual_energy": 0.0}
    z = res["N"][: min(128, res["N"].shape[0])]
    models["degeneracy_max_error"] = float(np.max(np.abs(delta_from_model(z, zero)))) if z.size else 0.0
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(models, cache)
    return models


def score_distributional(model_info, cov_models, patch_np, phase, variant, device):
    zt = state_z(model_info, patch_np, device)
    z = zt.detach().cpu().numpy().astype(np.float32)
    base = q0_np(model_info, zt, device)
    s_n, s_a = baseline_states_np(model_info)
    if phase == "A":
        if variant == "E0_Baseline_LRP":
            return base, {}
        if variant == "E1_Diagonal_DTS":
            dn = delta_from_model(residual_np(z, s_n), cov_models["N_diag"])
            da = delta_from_model(residual_np(z, s_a), cov_models["A_diag"])
        else:
            dn = delta_from_model(residual_np(z, s_n), cov_models["N_low"])
            da = delta_from_model(residual_np(z, s_a), cov_models["A_low"])
        score = base + GAMMA * (da - dn)
        return score.astype(np.float32), {"delta_N_mean": float(dn.mean()), "delta_A_mean": float(da.mean()), "corr_q0_qdist": corr(base, score)}
    if variant == "E0_Baseline_LRP":
        return base, {}
    nmodel = cov_models["N_low"] if variant == "E1_NormalLowRankGeometry" else cov_models["N_weighted"]
    dn = delta_from_model(residual_np(z, s_n), nmodel)
    score = base - GAMMA * dn
    return score.astype(np.float32), {"delta_N_mean": float(dn.mean()), "corr_q0_qdist": corr(base, score)}


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def local_distributional_maps(model_info, cov_models, clip_model, preprocess, img_path, phase, variants, device):
    crops = lzp.quadrant_crops(img_path)
    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patches = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)
    local = {v: np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32) for v in variants}
    slices = [(slice(0, SIDE), slice(0, SIDE)), (slice(0, SIDE), slice(SIDE, SIDE * 2)), (slice(SIDE, SIDE * 2), slice(0, SIDE)), (slice(SIDE, SIDE * 2), slice(SIDE, SIDE * 2))]
    for ci, sl in enumerate(slices):
        for v in variants:
            q, _diag = score_distributional(model_info, cov_models, patches[ci], phase, v, device)
            local[v][sl] = q.reshape(SIDE, SIDE)
    return local


def eval_phase(phase, out_dir, models, cov_by_source, clip_model, smoke):
    old_variants = lzp.VARIANTS
    variants = ("E0_Baseline_LRP", "E1_Diagonal_DTS", "E2_LowRank_DTS") if phase == "A" else ("E0_Baseline_LRP", "E1_NormalLowRankGeometry", "E2_NTRWeightedNormalGeometry")
    lzp.VARIANTS = variants
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw, percat, diag = [], [], []
    try:
        for source, target in PILOT:
            cells, _audits = protocol_cells(source, device, smoke)
            for _cell_id, tname, dataset, idx_or_rows, patch, kind in cells:
                if tname != target:
                    continue
                n = min(24, len(idx_or_rows)) if smoke else len(idx_or_rows)
                idx_use = idx_or_rows[:n]
                maps = {v: [] for v in variants}
                metas = []
                for pos, ref in enumerate(idx_use):
                    if pos % 25 == 0:
                        print(f"[DTS-{phase}] {source} -> {target} image {pos}/{n}", flush=True)
                    item = item_at(dataset, kind, ref, pos)
                    cat, label, path = item_meta(item, kind, ref, target)
                    local = local_distributional_maps(models[source], cov_by_source[source], clip_model, preprocess, path, phase, variants, device)
                    for v in variants:
                        q37, drow = score_distributional(models[source], cov_by_source[source], patch[pos], phase, v, device)
                        q74 = torch_resize(q37.reshape(SIDE, SIDE), (SIDE * 2, SIDE * 2), align_corners=True)
                        final = 0.5 * q74 + 0.5 * local[v]
                        maps[v].append(final.astype(np.float32))
                        if drow:
                            d = dict(drow)
                            d.update({"phase": phase, "source": source, "dataset": target, "variant": v, "category": cat, "image_index": pos})
                            diag.append(d)
                    metas.append({"category": cat, "label": label, "mask": mask_np(item), "img_path": path})
                rr, cc = evaluate_maps(source, target, maps, metas)
                raw.extend(rr)
                percat.extend(cc)
    finally:
        lzp.VARIANTS = old_variants
    write_csv(out_dir / "RAW_METRICS.csv", raw)
    write_csv(out_dir / "PER_CATEGORY_STANDARD_METRICS.csv", percat)
    write_csv(out_dir / "DISTRIBUTION_DIAGNOSTICS.csv", diag)
    return raw, percat, diag


def model_rows(cov_by_source):
    rows = []
    for source, mods in cov_by_source.items():
        for name, m in mods.items():
            if not isinstance(m, dict):
                continue
            rows.append({
                "source": source,
                "model": name,
                "kind": m.get("kind"),
                "rank": m.get("rank"),
                "sigma2": m.get("sigma2"),
                "explained": m.get("explained"),
                "residual_energy": m.get("residual_energy"),
                "degeneracy_max_error": mods.get("degeneracy_max_error", float("nan")),
            })
    return rows


def summarize(raw, phase):
    variants = ("E1_Diagonal_DTS", "E2_LowRank_DTS") if phase == "A" else ("E1_NormalLowRankGeometry", "E2_NTRWeightedNormalGeometry")
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in PILOT:
        b = by.get((source, target, "E0_Baseline_LRP"))
        if not b:
            continue
        for v in variants:
            r = by.get((source, target, v))
            if not r:
                continue
            rows.append({
                "phase": phase,
                "source": source,
                "dataset": target,
                "variant": v,
                "E0_P_AP": fnum(b, "M_FR_P_AP"),
                "variant_P_AP": fnum(r, "M_FR_P_AP"),
                "Delta_P_AP": fnum(r, "M_FR_P_AP") - fnum(b, "M_FR_P_AP"),
                "P_AUROC": fnum(r, "M_FR_P_AUROC"),
                "P_F1max": fnum(r, "M_FR_P_F1max"),
                "PRO": fnum(r, "M_FR_PRO"),
            })
    return rows


def decide(summary, phase):
    variants = ("E1_Diagonal_DTS", "E2_LowRank_DTS") if phase == "A" else ("E1_NormalLowRankGeometry", "E2_NTRWeightedNormalGeometry")
    stats = {}
    for v in variants:
        vals = np.asarray([fnum(r, "Delta_P_AP") for r in summary if r["variant"] == v], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        stats[v] = {
            "mean_delta_P_AP": float(vals.mean()) if vals.size else float("nan"),
            "median_delta_P_AP": float(np.median(vals)) if vals.size else float("nan"),
            "improved": int(np.sum(vals > 0)),
            "non_drop_ge_minus_0_2": int(np.sum(vals >= -0.2)),
        }
    if phase == "A":
        s = stats["E2_LowRank_DTS"]
        outcome = "DISTRIBUTIONAL_TWO_STATE_HAS_POSITIVE_SIGNAL" if s["improved"] >= 2 and s["mean_delta_P_AP"] > 0 else "SYMMETRIC_DISTRIBUTIONAL_STATE_HAS_NO_USEFUL_GAIN"
    else:
        e1 = stats["E1_NormalLowRankGeometry"]["mean_delta_P_AP"]
        e2 = stats["E2_NTRWeightedNormalGeometry"]["mean_delta_P_AP"]
        if e2 > 0 and e2 > e1:
            outcome = "NTR_WEIGHTED_GEOMETRY_IMPROVES_NORMAL_MODELING"
        elif e1 > 0:
            outcome = "NORMAL_VARIATION_GEOMETRY_HAS_TRANSFERABLE_VALUE"
        else:
            outcome = "ASYMMETRIC_DISTRIBUTIONAL_STATE_HAS_NO_USEFUL_GAIN"
    return outcome, stats


def write_report(path, title, outcome, stats, summary, model_diag, runtime, phase):
    lines = [
        f"# {title}",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is frozen Canonical Two-State + NTR + Fixed LRP. CLIP, states, NTR, and LRP are unchanged; only source-fitted residual covariance corrections are applied at scoring time. Gamma is fixed at 1.0; no target tuning is used.",
        "",
        "## Formula",
    ]
    if phase == "A":
        lines += [
            "`q(z) = q0(z) + [Delta_A(z) - Delta_N(z)]`, where `Delta_c = -0.5 r_c^T(C_c^-1 - sigma_c^-2 I)r_c` and `r_c=(I-s_c s_c^T)z`.",
            "If all low-rank eigenvalues are zero, the correction is exactly zero and the scorer degenerates to `q0`.",
        ]
    else:
        lines += [
            "`Delta_N` is the log-likelihood gain of the source-normal residual covariance over an isotropic baseline. Larger `Delta_N` means the patch lies along preserved normal-variation directions, so the deployed score is `q(z)=q0(z)-Delta_N(z)`. Deviations outside modeled normal variation are not rewarded.",
            "If `Lambda_N=0`, `Delta_N=0` and the scorer degenerates to `q0`.",
        ]
    lines += [
        "",
        "## Covariance Diagnostics",
        "",
        "| Source | Model | Rank | Explained | Residual Energy | Degeneracy Error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in model_diag:
        if r["model"] in {"N_low", "A_low", "N_weighted"}:
            lines.append(f"| {r['source']} | {r['model']} | {int(r['rank'])} | {float(r['explained']):.4f} | {float(r['residual_energy']):.4f} | {float(r['degeneracy_max_error']):.2e} |")
    lines += ["", "## Pilot Metrics", "", "| Source | Target | Variant | E0 P-AP | Variant P-AP | Delta P-AP | P-AUROC | P-F1max | PRO |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'E0_P_AP'):.4f} | {fnum(r,'variant_P_AP'):.4f} | {fnum(r,'Delta_P_AP'):+.4f} | {fnum(r,'P_AUROC'):.4f} | {fnum(r,'P_F1max'):.4f} | {fnum(r,'PRO'):.4f} |")
    lines += ["", "## Gate Summary"]
    for v, s in stats.items():
        lines.append(f"- {v}: mean Delta P-AP {s['mean_delta_P_AP']:+.4f}, median {s['median_delta_P_AP']:+.4f}, improved {s['improved']}/3, non-drop {s['non_drop_ge_minus_0_2']}/3.")
    lines += ["", f"Runtime: {runtime:.1f}s."]
    (path / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT_A.mkdir(parents=True, exist_ok=True)
    OUT_B.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"[DTS] device={device}", flush=True)
    clip_model, _prompt, _ip, _pp, _tpp = load_components(CHECKPOINT, device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    models = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    cov_by_source = {}
    for source in ("Full VisA", "Full MVTec"):
        print(f"[DTS] fit/source load covariance {source}", flush=True)
        cov_by_source[source] = fit_models_for_source(source, models[source], device, args.smoke)
    model_diag = model_rows(cov_by_source)
    write_csv(OUT_A / "COVARIANCE_DIAGNOSTICS.csv", model_diag)
    write_csv(OUT_B / "COVARIANCE_DIAGNOSTICS.csv", model_diag)
    raw_a, _cat_a, _diag_a = eval_phase("A", OUT_A, models, cov_by_source, clip_model, args.smoke)
    summ_a = summarize(raw_a, "A")
    out_a, stats_a = decide(summ_a, "A")
    write_csv(OUT_A / "CONTROL_COMPARISON.csv", summ_a)
    write_json(OUT_A / "OUTCOME.json", {"outcome": out_a, "stats": stats_a, "degeneracy_test_max_error": max(float(r["degeneracy_max_error"]) for r in model_diag if np.isfinite(float(r["degeneracy_max_error"]))), "gamma": GAMMA, "rank_rule": f"{EXPLAINED:.0%}_explained_max_{MAX_RANK}"})
    raw_b, _cat_b, _diag_b = eval_phase("B", OUT_B, models, cov_by_source, clip_model, args.smoke)
    summ_b = summarize(raw_b, "B")
    out_b, stats_b = decide(summ_b, "B")
    runtime = time.time() - start
    write_csv(OUT_B / "CONTROL_COMPARISON.csv", summ_b)
    write_json(OUT_B / "OUTCOME.json", {"outcome": out_b, "stats": stats_b, "degeneracy_test_max_error": max(float(r["degeneracy_max_error"]) for r in model_diag if np.isfinite(float(r["degeneracy_max_error"]))), "gamma": GAMMA, "rank_rule": f"{EXPLAINED:.0%}_explained_max_{MAX_RANK}"})
    write_report(OUT_A, "Phase DTS-A: Symmetric Distributional Two-State Pilot", out_a, stats_a, summ_a, model_diag, runtime, "A")
    write_report(OUT_B, "Phase DTS-B: Asymmetric Normal Variation Geometry", out_b, stats_b, summ_b, model_diag, runtime, "B")
    print(json.dumps({"DTS_A": out_a, "DTS_B": out_b, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
