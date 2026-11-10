import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import load_model
from experiments.phase_patch_occupancy_supervision_audit import load_context
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np
from models.mlp import average_neighbor
from phase_x.gradient_visual_forward import forward_visual_with_module


OUT = Path("results/phase_feature_evolution_complementarity_audit/formal")
LAYERS = list(range(1, 25))
HIST_LAYERS = [6, 12, 18, 24]
EPS = 1e-12
PATCH_GRID = SIDE * SIDE


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


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s) * 100.0)


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s) * 100.0)


def metric_row(name, y, s):
    pa = safe_auc(y, s)
    pap = safe_ap(y, s)
    return {"scope": "source", "score": name, "P-AUROC": pa, "P-AP": pap, "G": 0.5 * (pa + pap)}


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "P50": float("nan"), "P90": float("nan"), "P95": float("nan")}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "P50": float(np.percentile(x, 50)),
        "P90": float(np.percentile(x, 90)),
        "P95": float(np.percentile(x, 95)),
    }


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if a.size < 2 or a.std() < EPS or b.std() < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def standardize(train_x, test_x):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)
    return (train_x - mu) / sd, (test_x - mu) / sd


def fit_probe(train_x, train_y, test_x):
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=0)
    clf.fit(train_x, train_y)
    return clf.predict_proba(test_x)[:, 1]


def snapshot_to_projected_patch(model, snap):
    branch = snap[0] if isinstance(snap, list) else snap
    token = branch.permute(1, 0, 2)
    if token.shape[-1] != 1024:
        token = model.visual.ln_post(token) @ model.visual.proj
    patch = average_neighbor(token[:, 1:, :].float())
    return F.normalize(patch, dim=-1)


def score_projected_patch(patch, q_pc1, align_model, m_c, alpha, v_hn, device):
    patch_np = patch.detach().float().cpu().numpy()[0].astype(np.float32)
    z0 = remove_pc1(patch_np, q_pc1)
    z0 = norm_np(z0.reshape(PATCH_GRID, 1024))
    with torch.no_grad():
        z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device))
        m = torch.tensor(m_c, dtype=torch.float32, device=device)
        v = torch.tensor(v_hn, dtype=torch.float32, device=device)
        score = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
    return score.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def forward_feature_evolution(model, image, q_pc1, align_model, m_c, alpha, v_hn, device):
    pack = forward_visual_with_module(
        model.visual,
        image.type(model.dtype),
        insertion_layer=18,
        inserted_module=None,
        return_layers=LAYERS,
        dpam_layer=24,
        ori_patch=False,
        proj_use=True,
        ffn=False,
        snapshot_mode="true_clone",
    )
    patches = [snapshot_to_projected_patch(model, pack["snapshots"][f"true_block{layer}"])[0] for layer in LAYERS]
    stack = torch.stack(patches, dim=0).float()
    cos_adj = (stack[:-1] * stack[1:]).sum(dim=-1).clamp(-1.0, 1.0)
    transition = (1.0 - cos_adj).detach().cpu().numpy().astype(np.float32)
    path = transition.mean(axis=0)
    late = transition[17:23].mean(axis=0)

    d = torch.tensor(v_hn, dtype=torch.float32, device=stack.device)
    d = F.normalize(d, dim=0)
    delta = stack[1:] - stack[:-1]
    parallel = (delta @ d).pow(2).mean(dim=0)
    total = delta.pow(2).sum(dim=-1).mean(dim=0)
    orthogonal = torch.clamp(total - parallel, min=0.0)
    ortho_ratio = orthogonal / torch.clamp(parallel + orthogonal, min=EPS)

    score_maps = {}
    for layer in HIST_LAYERS:
        score_maps[layer] = score_projected_patch(patches[layer - 1].unsqueeze(0), q_pc1, align_model, m_c, alpha, v_hn, device)

    rng_order = torch.randperm(len(LAYERS), generator=torch.Generator(device="cpu").manual_seed(0)).to(stack.device)
    shuffled = stack[rng_order]
    cos_rand = (shuffled[:-1] * shuffled[1:]).sum(dim=-1).clamp(-1.0, 1.0)
    path_random = (1.0 - cos_rand).mean(dim=0).detach().cpu().numpy().astype(np.float32)

    return {
        "PathLength": path.astype(np.float32),
        "LatePath": late.astype(np.float32),
        "ParallelEnergy": parallel.detach().cpu().numpy().astype(np.float32),
        "OrthogonalEnergy": orthogonal.detach().cpu().numpy().astype(np.float32),
        "OrthogonalRatio": ortho_ratio.detach().cpu().numpy().astype(np.float32),
        "PathLength_random": path_random,
        "s6": score_maps[6],
        "s12": score_maps[12],
        "s18": score_maps[18],
        "s24": score_maps[24],
    }


def collect_source(smoke, limit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"dpam_layer": 24, "n_ctx": 12, "depth": 9, "t_n_ctx": 4}
    model = load_model(cfg, str(device))
    q_pc1, align_model, m_c, alpha, v_hn = load_context(device)
    _name, dataset, indices, _cls, _patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    positions = list(range(len(indices)))
    if smoke:
        positions = positions[:64]
    if limit is not None:
        positions = positions[: int(limit)]

    cats = []
    cat_to_id = {}
    arrays = {k: [] for k in ["y", "PathLength", "LatePath", "ParallelEnergy", "OrthogonalEnergy", "OrthogonalRatio", "PathLength_random", "s6", "s12", "s18", "s24"]}
    cat_ids = []
    pos_ids = []
    for n, pos in enumerate(positions, 1):
        if n == 1 or n % 50 == 0:
            print(f"[FeatureEvolution] image {n}/{len(positions)}", flush=True)
        item = dataset[int(indices[int(pos)])]
        cat = str(item["cls_name"])
        if cat not in cat_to_id:
            cat_to_id[cat] = len(cats)
            cats.append(cat)
        image = item["img"].unsqueeze(0).to(device)
        scalars = forward_feature_evolution(model, image, q_pc1, align_model, m_c, alpha, v_hn, device)
        y = np.asarray(labels_by_pos.get(int(pos), np.zeros((PATCH_GRID,), dtype=np.float32)), dtype=np.uint8).reshape(-1)
        arrays["y"].append(y)
        for key, val in scalars.items():
            arrays[key].append(np.asarray(val, dtype=np.float32).reshape(-1))
        cat_ids.append(np.full((PATCH_GRID,), cat_to_id[cat], dtype=np.int16))
        pos_ids.append(np.full((PATCH_GRID,), int(pos), dtype=np.int32))

    data = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    data["category_id"] = np.concatenate(cat_ids, axis=0)
    data["position"] = np.concatenate(pos_ids, axis=0)
    data["categories"] = np.asarray(cats, dtype=object)
    data["S_H"] = data["s24"].astype(np.float32)
    return data, str(device)


def transition_stats(data):
    rows = []
    y = data["y"]
    for score in ["PathLength", "LatePath", "OrthogonalEnergy"]:
        for group, mask in [("normal", y == 0), ("anomaly", y > 0)]:
            rows.append({"score": score, "group": group, **stats(data[score][mask])})
        rows.append(metric_row(score, y, data[score]))
    rows.append(metric_row("HardNormal_S_H", y, data["S_H"]))
    rows.append({"scope": "source", "score": "corr_S_H_PathLength", "G": corr(data["S_H"], data["PathLength"])})
    return rows


def orthogonal_energy_rows(data):
    rows = []
    y = data["y"]
    for group, mask in [("normal", y == 0), ("anomaly", y > 0), ("all", np.ones_like(y, dtype=bool))]:
        rows.append({
            "group": group,
            **stats(data["OrthogonalRatio"][mask]),
            "parallel_mean": float(np.mean(data["ParallelEnergy"][mask])),
            "orthogonal_mean": float(np.mean(data["OrthogonalEnergy"][mask])),
        })
    return rows


def conditional_table(data):
    order = np.argsort(data["S_H"], kind="mergesort")
    bins = np.array_split(order, 20)
    rows = []
    aucs = []
    for i, ids in enumerate(bins):
        y = data["y"][ids]
        s = data["PathLength"][ids]
        auc = safe_auc(y, s)
        ap = safe_ap(y, s)
        if np.isfinite(auc):
            aucs.append(auc)
        rows.append({
            "bin": i,
            "count": int(ids.size),
            "S_H_min": float(data["S_H"][ids].min()),
            "S_H_max": float(data["S_H"][ids].max()),
            "anomaly_ratio": float(y.mean()),
            "PathLength_anomaly_minus_normal": float(s[y > 0].mean() - s[y == 0].mean()) if np.unique(y).size == 2 else float("nan"),
            "PathLength_AUROC": auc,
            "PathLength_AP": ap,
        })
    arr = np.asarray(aucs, dtype=np.float64)
    rows.insert(0, {"bin": "__summary__", "valid_bins": int(arr.size), "PathLength_AUROC": float(arr.mean()) if arr.size else float("nan"), "PathLength_AUROC_std": float(arr.std()) if arr.size else float("nan")})
    return rows


def loo_probe(data):
    cats = data["categories"].tolist()
    rows = []
    for cid, cat in enumerate(cats):
        test = data["category_id"] == cid
        train = ~test
        y_train = data["y"][train]
        y_test = data["y"][test]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        variants = {
            "P0_S_H": ([data["S_H"][train]], [data["S_H"][test]]),
            "P_old_S_H_plus_s6_s12_s18_s24": ([data["S_H"][train], data["s6"][train], data["s12"][train], data["s18"][train], data["s24"][train]], [data["S_H"][test], data["s6"][test], data["s12"][test], data["s18"][test], data["s24"][test]]),
            "P_new_S_H_plus_PathLength": ([data["S_H"][train], data["PathLength"][train]], [data["S_H"][test], data["PathLength"][test]]),
            "P_full_old_plus_PathLength": ([data["S_H"][train], data["s6"][train], data["s12"][train], data["s18"][train], data["s24"][train], data["PathLength"][train]], [data["S_H"][test], data["s6"][test], data["s12"][test], data["s18"][test], data["s24"][test], data["PathLength"][test]]),
            "P_rand_S_H_plus_PathLengthRandom": ([data["S_H"][train], data["PathLength_random"][train]], [data["S_H"][test], data["PathLength_random"][test]]),
        }
        metrics = {}
        for name, (tr_cols, te_cols) in variants.items():
            tr = np.stack(tr_cols, axis=1).astype(np.float64)
            te = np.stack(te_cols, axis=1).astype(np.float64)
            tr, te = standardize(tr, te)
            pred = fit_probe(tr, y_train, te)
            pa = safe_auc(y_test, pred)
            pap = safe_ap(y_test, pred)
            metrics[name] = {"P-AUROC": pa, "P-AP": pap, "G": 0.5 * (pa + pap)}
            rows.append({"heldout_category": cat, "variant": name, **metrics[name]})
        deltas = [
            ("DELTA_P_new_minus_P0", "P_new_S_H_plus_PathLength", "P0_S_H"),
            ("DELTA_P_full_minus_P_old", "P_full_old_plus_PathLength", "P_old_S_H_plus_s6_s12_s18_s24"),
            ("DELTA_P_new_minus_P_rand", "P_new_S_H_plus_PathLength", "P_rand_S_H_plus_PathLengthRandom"),
        ]
        for name, a, b in deltas:
            rows.append({
                "heldout_category": cat,
                "variant": name,
                "P-AUROC": metrics[a]["P-AUROC"] - metrics[b]["P-AUROC"],
                "P-AP": metrics[a]["P-AP"] - metrics[b]["P-AP"],
                "G": metrics[a]["G"] - metrics[b]["G"],
            })
    summary = {"heldout_category": "__summary__", "variant": "summary"}
    for name in ["DELTA_P_new_minus_P0", "DELTA_P_full_minus_P_old", "DELTA_P_new_minus_P_rand"]:
        vals = np.asarray([r["G"] for r in rows if r["variant"] == name], dtype=np.float64)
        summary[f"{name}_categories"] = int(vals.size)
        summary[f"{name}_improved"] = int(np.sum(vals > 0))
        summary[f"{name}_mean_G"] = float(vals.mean()) if vals.size else float("nan")
        summary[f"{name}_median_G"] = float(np.median(vals)) if vals.size else float("nan")
    return [summary] + rows


def error_analysis(data):
    y = data["y"]
    s = data["S_H"]
    pl = data["PathLength"]
    normal_s = s[y == 0]
    anomaly_s = s[y > 0]
    n95 = np.percentile(normal_s, 95)
    a20 = np.percentile(anomaly_s, 20)
    masks = {
        "FP_H_normal_S_H_gt_normal_P95": (y == 0) & (s > n95),
        "correct_normal": (y == 0) & (s <= n95),
        "FN_H_anomaly_S_H_lt_anomaly_P20": (y > 0) & (s < a20),
        "correct_anomaly": (y > 0) & (s >= a20),
    }
    rows = [{"group": name, **stats(pl[mask])} for name, mask in masks.items()]
    rows.append({"group": "delta_FP_minus_correct_normal", "mean": rows[0]["mean"] - rows[1]["mean"]})
    rows.append({"group": "delta_FN_minus_correct_anomaly", "mean": rows[2]["mean"] - rows[3]["mean"]})
    return rows


def decide(residual_rows, ortho_rows, cond_rows, loo_rows):
    all_ortho = next(r for r in ortho_rows if r["group"] == "all")
    ortho_ratio = float(all_ortho["mean"])
    corr_row = next(r for r in residual_rows if r.get("score") == "corr_S_H_PathLength")
    corr_sh_path = float(corr_row["G"])
    cond_summary = next(r for r in cond_rows if r["bin"] == "__summary__")
    loo_summary = next(r for r in loo_rows if r["heldout_category"] == "__summary__")
    pnew_p0_improved = float(loo_summary["DELTA_P_new_minus_P0_improved"])
    pnew_p0_cats = max(float(loo_summary["DELTA_P_new_minus_P0_categories"]), 1.0)
    pnew_p0_mean = float(loo_summary["DELTA_P_new_minus_P0_mean_G"])
    pfull_old_mean = float(loo_summary["DELTA_P_full_minus_P_old_mean_G"])
    pnew_prand_mean = float(loo_summary["DELTA_P_new_minus_P_rand_mean_G"])

    if (
        ortho_ratio > 0.30
        and abs(corr_sh_path) < 0.80
        and pnew_p0_improved / pnew_p0_cats >= 2.0 / 3.0
        and pnew_p0_mean >= 0.5
        and pfull_old_mean >= 0.3
        and pnew_prand_mean >= 0.3
    ):
        outcome = "FEATURE_EVOLUTION_HAS_COMPLEMENTARY_INFORMATION"
    elif abs(pnew_p0_mean) < 0.3:
        outcome = "FEATURE_EVOLUTION_REDUNDANT"
    elif pnew_p0_mean > 0.0 and abs(pfull_old_mean) < 0.3:
        outcome = "FEATURE_EVOLUTION_COLLAPSES_TO_EXISTING_MULTILAYER_INFORMATION"
    elif abs(pnew_prand_mean) < 0.3:
        outcome = "FEATURE_EVOLUTION_ORDER_NOT_SPECIFIC"
    else:
        outcome = "FEATURE_EVOLUTION_WEAK_SIGNAL"
    return outcome, {
        "orthogonal_transition_energy_ratio_mean": ortho_ratio,
        "corr_S_H_PathLength": corr_sh_path,
        "conditional_PathLength_AUROC_mean": float(cond_summary["PathLength_AUROC"]),
        "P_new_improve_fraction": pnew_p0_improved / pnew_p0_cats,
        "mean_delta_G_P_new_minus_P0": pnew_p0_mean,
        "mean_delta_G_P_full_minus_P_old": pfull_old_mean,
        "mean_delta_G_P_new_minus_P_rand": pnew_prand_mean,
    }


def write_report(outcome, decision, trans_rows, loo_rows, runtime, smoke):
    hard = next(r for r in trans_rows if r.get("score") == "HardNormal_S_H")
    pl = next(r for r in trans_rows if r.get("scope") == "source" and r.get("score") == "PathLength")
    late = next(r for r in trans_rows if r.get("scope") == "source" and r.get("score") == "LatePath")
    orth = next(r for r in trans_rows if r.get("scope") == "source" and r.get("score") == "OrthogonalEnergy")
    summary = next(r for r in loo_rows if r["heldout_category"] == "__summary__")
    lines = [
        "# Transformer Feature-Evolution Complementarity Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN / NO-TARGET. Full VisA source only; frozen current formal Symmetric HardNormal checkpoint and scorer.",
        "",
        "## Source Signal",
        "",
        f"- HardNormal source G: {float(hard['G']):.4f}.",
        f"- PathLength source AUROC/AP/G: {float(pl['P-AUROC']):.4f} / {float(pl['P-AP']):.4f} / {float(pl['G']):.4f}.",
        f"- LatePath source AUROC/AP/G: {float(late['P-AUROC']):.4f} / {float(late['P-AP']):.4f} / {float(late['G']):.4f}.",
        f"- OrthogonalEnergy source AUROC/AP/G: {float(orth['P-AUROC']):.4f} / {float(orth['P-AP']):.4f} / {float(orth['G']):.4f}.",
        f"- Orthogonal transition energy ratio mean: {decision['orthogonal_transition_energy_ratio_mean']:.4f}.",
        f"- corr(S_H, PathLength): {decision['corr_S_H_PathLength']:.4f}.",
        "",
        "## Complementarity",
        "",
        f"- Conditional PathLength AUROC within S_H bins: {decision['conditional_PathLength_AUROC_mean']:.4f}.",
        f"- P_new=[S_H,PathLength] improves P0=[S_H] in {int(summary['DELTA_P_new_minus_P0_improved'])}/{int(summary['DELTA_P_new_minus_P0_categories'])} categories.",
        f"- mean Delta G(P_new-P0): {decision['mean_delta_G_P_new_minus_P0']:+.4f}.",
        f"- mean Delta G(P_full-P_old): {decision['mean_delta_G_P_full_minus_P_old']:+.4f}.",
        f"- mean Delta G(P_new-P_rand): {decision['mean_delta_G_P_new_minus_P_rand']:+.4f}.",
        "",
        "STOP: no target experiment, trainable transition head, layer sweep, or fusion tuning was started.",
        "",
        f"Runtime: {runtime:.1f}s. Smoke: {smoke}.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ("source_scalars_smoke.npz" if args.smoke else "source_scalars_formal.npz")
    meta_path = OUT / ("source_scalars_smoke_meta.json" if args.smoke else "source_scalars_formal_meta.json")
    if cache.exists() and not args.no_cache:
        z = np.load(cache, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        data["categories"] = data["categories"].astype(object)
        device = "cache"
    else:
        data, device = collect_source(args.smoke, args.limit)
        np.savez_compressed(cache, **data)
        write_json(meta_path, {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": args.smoke, "limit": args.limit, "device": device, "patch_count": int(data["y"].size), "categories": data["categories"].tolist()})

    trans_rows = transition_stats(data)
    ortho_rows = orthogonal_energy_rows(data)
    cond_rows = conditional_table(data)
    loo_rows = loo_probe(data)
    err_rows = error_analysis(data)
    outcome, decision = decide(trans_rows, ortho_rows, cond_rows, loo_rows)

    write_csv(OUT / "TRANSITION_STATS.csv", trans_rows)
    write_csv(OUT / "ORTHOGONAL_ENERGY.csv", ortho_rows)
    write_csv(OUT / "CONDITIONAL_COMPLEMENTARITY.csv", cond_rows)
    write_csv(OUT / "LOO_PROBE.csv", loo_rows)
    write_csv(OUT / "ERROR_ANALYSIS.csv", err_rows)
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision,
        "runtime_sec": runtime,
        "device": device,
        "no_training": True,
        "target_data_used": False,
        "source": "Full VisA",
        "backbone": "current formal ViT-L/14 visual path",
        "layers": LAYERS,
        "historical_score_layers": HIST_LAYERS,
        "smoke": args.smoke,
        "limit": args.limit,
    })
    write_report(outcome, decision, trans_rows, loo_rows, runtime, args.smoke)
    print(f"[FeatureEvolution] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
