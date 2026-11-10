import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cross_foundation_stable_hardness_audit import build_clip_radio_pool, initialize_clip_context
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source, source_slots_by_category as mvtec_slots_by_category
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import PATCH_SAMPLE_PER_CLASS_CATEGORY
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np, patch_labels


OUT = Path("results/phase_latent_group_cvar_hardnormal_audit/formal")
HN_MVTEC_OUT = Path("results/phase_hard_normal_cross_source_mvtec/formal")
TOP_FRAC = 0.10
K = 8
SEED = 0
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


def entropy_eff(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    h = -np.sum(p * np.log(p + EPS))
    return float(h), float(np.exp(h))


def top_indices(scores, frac=TOP_FRAC):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    k = max(1, int(math.ceil(scores.size * frac)))
    return np.argpartition(scores, scores.size - k)[scores.size - k :]


def cos_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return float("nan")
    return float(a @ b / (na * nb))


def load_visa_context(device, smoke):
    q_pc1, m, alpha, v, _align_model = initialize_clip_context(device)
    pool = build_clip_radio_pool(device, smoke=smoke)
    return {
        "source": "VisA",
        "x": np.asarray(pool["x"], dtype=np.float32),
        "y": np.asarray(pool["y"], dtype=np.float32),
        "category": np.asarray(pool["category"], dtype=object),
        "origin": np.asarray(pool["origin"], dtype=object),
        "patch_id": np.asarray(pool["patch_id"], dtype=np.int64),
        "m": np.asarray(m, dtype=np.float32),
        "alpha": float(alpha),
        "v": unit_np(np.asarray(v, dtype=np.float32)),
    }


def load_mvtec_context(device, smoke):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / "cache" / f"mvtec_source_pool_{suffix}.npz"
    vecs = np.load(HN_MVTEC_OUT / "contrast_vectors.npz")
    m = np.asarray(vecs["midpoint"], dtype=np.float32)
    alpha = float(np.asarray(vecs["alpha"]))
    v = unit_np(np.asarray(vecs["E1"], dtype=np.float32))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {
            "source": "MVTec",
            "x": np.asarray(z["x"], dtype=np.float32),
            "y": np.asarray(z["y"], dtype=np.float32),
            "category": np.asarray(z["category"], dtype=object),
            "origin": np.asarray(z["origin"], dtype=object),
            "patch_id": np.asarray(z["patch_id"], dtype=np.int64),
            "m": m,
            "alpha": alpha,
            "v": v,
        }
    _name, dataset, indices, patch = load_full_mvtec_source(smoke)
    slots = mvtec_slots_by_category(dataset, indices)
    rng = np.random.default_rng(SEED)
    xs, ys, cats, origins, pids = [], [], [], [], []
    for cat, positions in slots.items():
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        part_ids = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            item = dataset[int(indices[pos])]
            y_grid = patch_labels(item).astype(np.int64)
            z = norm_np(np.asarray(patch[pos], dtype=np.float32).reshape(SIDE * SIDE, 1024))
            if int(item["anomaly"]) == 1:
                pools = [("anomaly_mask_out", np.flatnonzero(y_grid == 0)), ("anomaly_mask", np.flatnonzero(y_grid == 1))]
            else:
                pools = [("normal_image", np.flatnonzero(y_grid == 0))]
            for origin, ids in pools:
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(arr.shape[0] for arr in parts[origin])
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin].append(z[ids])
                part_ids[origin].append(ids.astype(np.int64))
        for origin, cls in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if not parts[origin]:
                continue
            arr = np.concatenate(parts[origin], axis=0).astype(np.float32)
            ids = np.concatenate(part_ids[origin], axis=0).astype(np.int64)
            xs.append(arr)
            ys.append(np.full((arr.shape[0],), cls, dtype=np.float32))
            cats.extend([cat] * arr.shape[0])
            origins.extend([origin] * arr.shape[0])
            pids.extend(ids.tolist())
    payload = {
        "x": np.concatenate(xs, axis=0).astype(np.float16),
        "y": np.concatenate(ys, axis=0).astype(np.float32),
        "category": np.asarray(cats, dtype=object),
        "origin": np.asarray(origins, dtype=object),
        "patch_id": np.asarray(pids, dtype=np.int64),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    return {"source": "MVTec", **{k: np.asarray(vv) for k, vv in payload.items()}, "m": m, "alpha": alpha, "v": v}


def logits_np(ctx):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        out = []
        m = torch.tensor(ctx["m"], dtype=torch.float32, device=device)
        v = torch.tensor(ctx["v"], dtype=torch.float32, device=device)
        for start in range(0, len(ctx["y"]), 8192):
            x = torch.tensor(ctx["x"][start : start + 8192], dtype=torch.float32, device=device)
            out.append(contrast_logits(x, m, ctx["alpha"], v).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def prepare_latent_groups(ctx, logits, smoke=False):
    normal = ctx["y"] < 0.5
    x_n = np.asarray(ctx["x"][normal], dtype=np.float32)
    cat_n = np.asarray(ctx["category"][normal], dtype=object)
    origin_n = np.asarray(ctx["origin"][normal], dtype=object)
    pid_n = np.asarray(ctx["patch_id"][normal], dtype=np.int64)
    logit_n = logits[normal]
    loss_n = np.logaddexp(0.0, logit_n).astype(np.float32)
    means = {}
    for cat in sorted(set(cat_n.tolist())):
        means[cat] = x_n[cat_n == cat].mean(axis=0).astype(np.float32)
    d = unit_np(ctx["v"])
    r = x_n - np.stack([means[c] for c in cat_n.tolist()], axis=0)
    r = r - (r @ d)[:, None] * d[None, :]
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    r_perp = r / np.maximum(r_norm, EPS)
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(r_perp)
    group = km.labels_.astype(np.int64)
    return {"x": x_n, "category": cat_n, "origin": origin_n, "patch_id": pid_n, "logit": logit_n, "loss": loss_n, "group": group, "r_perp": r_perp.astype(np.float32)}


def cvar_loss(indices, x, m_np, alpha, v_param):
    if len(indices) == 0:
        return torch.zeros((), dtype=torch.float32, device=v_param.device)
    z = torch.tensor(x[indices], dtype=torch.float32, device=v_param.device)
    m = torch.tensor(m_np, dtype=torch.float32, device=v_param.device)
    logits = contrast_logits(z, m, alpha, F.normalize(v_param, dim=-1))
    loss = F.softplus(logits)
    take = top_indices(loss.detach().cpu().numpy(), TOP_FRAC)
    return loss[torch.tensor(take, dtype=torch.long, device=v_param.device)].mean()


def grad_vec(loss, v_param):
    g = torch.autograd.grad(loss, v_param, retain_graph=False, create_graph=False)[0]
    return g.detach().cpu().numpy().astype(np.float64)


def grad_for_index_sets(index_sets, latent, ctx, device):
    v_param = torch.nn.Parameter(torch.tensor(ctx["v"], dtype=torch.float32, device=device))
    losses = [cvar_loss(ids, latent["x"], ctx["m"], ctx["alpha"], v_param) for ids in index_sets if len(ids) > 0]
    loss = torch.stack(losses).mean() if losses else torch.zeros((), dtype=torch.float32, device=device)
    g = grad_vec(loss, v_param)
    return g, float(loss.detach().cpu())


def distribution_stats(latent):
    group = latent["group"]
    all_counts = np.bincount(group, minlength=K).astype(np.float64)
    all_p = all_counts / max(1.0, all_counts.sum())
    h_global = top_indices(latent["loss"], TOP_FRAC)
    hard_counts = np.bincount(group[h_global], minlength=K).astype(np.float64)
    hard_p = hard_counts / max(1.0, hard_counts.sum())
    _h_all, eff_all = entropy_eff(all_p)
    _h_hard, eff_hard = entropy_eff(hard_p)
    js = float(jensenshannon(all_p + EPS, hard_p + EPS, base=np.e) ** 2)
    kl = float(np.sum(hard_p * (np.log(hard_p + EPS) - np.log(all_p + EPS))))
    rows = []
    for k in range(K):
        mask = group == k
        hard_mask = group[h_global] == k
        ys = latent["patch_id"][mask] // 37
        xs = latent["patch_id"][mask] % 37
        rows.append({
            "group": k,
            "size": int(np.sum(mask)),
            "ratio_all": float(all_p[k]),
            "ratio_hard": float(hard_p[k]),
            "hard_count": int(np.sum(hard_mask)),
            "hard_enrichment": float(hard_p[k] / max(all_p[k], EPS)),
            "mean_hardness": float(np.mean(latent["logit"][mask])) if np.any(mask) else float("nan"),
            "P90_hardness": float(np.percentile(latent["logit"][mask], 90)) if np.any(mask) else float("nan"),
            "normal_BCE_mean": float(np.mean(latent["loss"][mask])) if np.any(mask) else float("nan"),
            "patch_x_mean": float(np.mean(xs)) if np.any(mask) else float("nan"),
            "patch_y_mean": float(np.mean(ys)) if np.any(mask) else float("nan"),
            "patch_x_var": float(np.var(xs)) if np.any(mask) else float("nan"),
            "patch_y_var": float(np.var(ys)) if np.any(mask) else float("nan"),
        })
    return rows, {
        "effective_group_number_all": eff_all,
        "effective_group_number_hard": eff_hard,
        "EGR": float(eff_hard / max(eff_all, EPS)),
        "JS_divergence": js,
        "KL_hard_to_all": kl,
        "hard_count": int(len(h_global)),
    }


def gradient_audit(ctx, latent, device):
    all_ids = np.arange(len(latent["loss"]), dtype=np.int64)
    global_ids = [all_ids]
    group_sets = [np.flatnonzero(latent["group"] == k) for k in range(K)]
    cats = sorted(set(latent["category"].tolist()))
    cat_sets = [np.flatnonzero(latent["category"] == c) for c in cats]
    rng = np.random.default_rng(SEED)
    counts = [len(s) for s in group_sets]
    perm = rng.permutation(len(all_ids))
    random_group = np.empty(len(all_ids), dtype=np.int64)
    off = 0
    for k, c in enumerate(counts):
        random_group[perm[off : off + c]] = k
        off += c
    random_sets = [np.flatnonzero(random_group == k) for k in range(K)]
    g_global, l_global = grad_for_index_sets(global_ids, latent, ctx, device)
    g_lg, l_lg = grad_for_index_sets(group_sets, latent, ctx, device)
    g_cat, l_cat = grad_for_index_sets(cat_sets, latent, ctx, device)
    g_rand, l_rand = grad_for_index_sets(random_sets, latent, ctx, device)
    g_rows = [
        {"metric": "cos_g_global_g_LG", "value": cos_np(g_global, g_lg)},
        {"metric": "norm_g_LG_over_global", "value": float(np.linalg.norm(g_lg) / max(np.linalg.norm(g_global), EPS))},
        {"metric": "loss_global", "value": l_global},
        {"metric": "loss_LG", "value": l_lg},
        {"metric": "cos_g_LG_g_category", "value": cos_np(g_lg, g_cat)},
        {"metric": "cos_g_global_g_category", "value": cos_np(g_global, g_cat)},
        {"metric": "loss_category", "value": l_cat},
        {"metric": "cos_g_LG_g_random", "value": cos_np(g_lg, g_rand)},
        {"metric": "cos_g_global_g_random", "value": cos_np(g_global, g_rand)},
        {"metric": "loss_random", "value": l_rand},
    ]
    group_grad_rows = []
    group_grads = []
    for k, ids in enumerate(group_sets):
        gk, lk = grad_for_index_sets([ids], latent, ctx, device)
        group_grads.append(gk)
        group_grad_rows.append({"row_group": k, "col_group": k, "cosine": 1.0, "loss": lk})
    cosines = []
    for i, j in combinations(range(K), 2):
        c = cos_np(group_grads[i], group_grads[j])
        cosines.append(c)
        group_grad_rows.append({"row_group": i, "col_group": j, "cosine": c, "loss": float("nan")})
        group_grad_rows.append({"row_group": j, "col_group": i, "cosine": c, "loss": float("nan")})
    g_rows.extend([
        {"metric": "group_gradient_pairwise_cosine_mean", "value": float(np.mean(cosines))},
        {"metric": "group_gradient_pairwise_cosine_std", "value": float(np.std(cosines))},
        {"metric": "group_gradient_pairwise_cosine_min", "value": float(np.min(cosines))},
        {"metric": "group_gradient_pairwise_cosine_max", "value": float(np.max(cosines))},
    ])
    return g_rows, group_grad_rows, random_group


def cvar_equivalence(latent):
    top_score = set(top_indices(latent["logit"], TOP_FRAC).tolist())
    top_bce = set(top_indices(latent["loss"], TOP_FRAC).tolist())
    inter = len(top_score & top_bce)
    union = len(top_score | top_bce)
    return {
        "top10_anomaly_score_count": len(top_score),
        "top10_normal_bce_count": len(top_bce),
        "intersection": inter,
        "jaccard": float(inter / max(1, union)),
        "overlap_over_score_top": float(inter / max(1, len(top_score))),
    }


def run_source(ctx, device):
    logits = logits_np(ctx)
    latent = prepare_latent_groups(ctx, logits)
    group_rows, conc = distribution_stats(latent)
    for r in group_rows:
        r["source"] = ctx["source"]
    g_rows, group_grad_rows, random_group = gradient_audit(ctx, latent, device)
    for r in g_rows:
        r["source"] = ctx["source"]
    for r in group_grad_rows:
        r["source"] = ctx["source"]
    control_rows = [
        {"source": ctx["source"], "metric": "NMI_group_category", "value": float(normalized_mutual_info_score(latent["category"], latent["group"]))},
        {"source": ctx["source"], "metric": "NMI_group_origin", "value": float(normalized_mutual_info_score(latent["origin"], latent["group"]))},
    ]
    for k, v in conc.items():
        control_rows.append({"source": ctx["source"], "metric": k, "value": v})
    for k, v in cvar_equivalence(latent).items():
        control_rows.append({"source": ctx["source"], "metric": k, "value": v})
    hard_rows = []
    all_counts = np.bincount(latent["group"], minlength=K).astype(np.float64)
    hard = top_indices(latent["loss"], TOP_FRAC)
    hard_counts = np.bincount(latent["group"][hard], minlength=K).astype(np.float64)
    random_hard = top_indices(latent["loss"], TOP_FRAC)
    random_counts = np.bincount(random_group[random_hard], minlength=K).astype(np.float64)
    for k in range(K):
        hard_rows.append({
            "source": ctx["source"],
            "group": k,
            "normal_count": int(all_counts[k]),
            "global_hard_count": int(hard_counts[k]),
            "random_group_hard_count": int(random_counts[k]),
            "P_group_all": float(all_counts[k] / max(1, all_counts.sum())),
            "P_group_hard": float(hard_counts[k] / max(1, hard_counts.sum())),
            "P_random_group_hard": float(random_counts[k] / max(1, random_counts.sum())),
        })
    summary = {
        "source": ctx["source"],
        "NMI_category": float(normalized_mutual_info_score(latent["category"], latent["group"])),
        "NMI_origin": float(normalized_mutual_info_score(latent["origin"], latent["group"])),
        "EGR": conc["EGR"],
        "JS_divergence": conc["JS_divergence"],
        "cos_global_LG": next(r["value"] for r in g_rows if r["metric"] == "cos_g_global_g_LG"),
        "cos_LG_category": next(r["value"] for r in g_rows if r["metric"] == "cos_g_LG_g_category"),
        "cos_LG_random": next(r["value"] for r in g_rows if r["metric"] == "cos_g_LG_g_random"),
        "group_grad_mean_cos": next(r["value"] for r in g_rows if r["metric"] == "group_gradient_pairwise_cosine_mean"),
    }
    return summary, group_rows, hard_rows, g_rows, group_grad_rows, control_rows


def source_pass(s):
    return {
        "A_not_category_copy": s["NMI_category"] <= 0.50,
        "B_not_origin_copy": s["NMI_origin"] <= 0.30,
        "C_tail_concentration": (s["EGR"] <= 0.75 or s["JS_divergence"] >= 0.10),
        "D_group_grad_different": s["group_grad_mean_cos"] <= 0.80,
        "E_not_global_rescale": s["cos_global_LG"] <= 0.95,
        "F_not_category_cvar": s["cos_LG_category"] <= 0.95,
        "G_not_random_group": s["cos_LG_random"] <= 0.95,
    }


def decide(summaries):
    passes = {s["source"]: source_pass(s) for s in summaries}
    full_pass = {src: all(v.values()) for src, v in passes.items()}
    if all(full_pass.values()):
        outcome = "LATENT_NORMAL_TAIL_IMBALANCE_REPLICATED"
    elif any(s["NMI_category"] > 0.50 for s in summaries):
        outcome = "LATENT_GROUPS_COLLAPSE_TO_SOURCE_CATEGORY"
    elif all(not source_pass(s)["C_tail_concentration"] for s in summaries):
        outcome = "GLOBAL_HARDNORMAL_ALREADY_GROUP_DIVERSE"
    elif all(s["cos_global_LG"] > 0.95 for s in summaries):
        outcome = "GROUP_CVAR_COLLAPSES_TO_GLOBAL_HARDNORMAL"
    elif sum(full_pass.values()) == 1:
        outcome = "LATENT_TAIL_IMBALANCE_SOURCE_DEPENDENT"
    else:
        outcome = "LATENT_GROUP_TAIL_RISK_NO_SIGNAL"
    return outcome, passes


def write_report(outcome, summaries, passes, runtime):
    lines = [
        "# Latent Group-CVaR HardNormal Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN / NO-TARGET. Full VisA and Full MVTec are audited independently. KMeans K=8 is fit on source normal residuals after removing category mean and the current contrast direction.",
        "",
        "| Source | NMI cat | NMI origin | EGR | JS | cos(global,LG) | cos(LG,cat) | cos(LG,random) | group grad mean cos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(f"| {s['source']} | {s['NMI_category']:.4f} | {s['NMI_origin']:.4f} | {s['EGR']:.4f} | {s['JS_divergence']:.4f} | {s['cos_global_LG']:.4f} | {s['cos_LG_category']:.4f} | {s['cos_LG_random']:.4f} | {s['group_grad_mean_cos']:.4f} |")
    lines.extend(["", "## Gate"])
    for src, gate in passes.items():
        good = sum(bool(v) for v in gate.values())
        lines.append(f"- {src}: {good}/7 gates passed; " + ", ".join(f"{k}={v}" for k, v in gate.items()))
    lines.extend(["", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LatentCVaR] device={device}", flush=True)
    contexts = [load_visa_context(device, args.smoke), load_mvtec_context(device, args.smoke)]
    summaries, visa_rows, mvtec_rows, hard_rows, grad_rows, group_grad_rows, control_rows = [], [], [], [], [], [], []
    for ctx in contexts:
        print(f"[LatentCVaR] audit {ctx['source']} patches={len(ctx['y'])}", flush=True)
        summary, group_rows, h_rows, g_rows, gg_rows, c_rows = run_source(ctx, device)
        summaries.append(summary)
        if ctx["source"] == "VisA":
            visa_rows.extend(group_rows)
        else:
            mvtec_rows.extend(group_rows)
        hard_rows.extend(h_rows)
        grad_rows.extend(g_rows)
        group_grad_rows.extend(gg_rows)
        control_rows.extend(c_rows)
    outcome, passes = decide(summaries)
    runtime = time.time() - start
    write_csv(OUT / "VISA_GROUP_STATS.csv", visa_rows)
    write_csv(OUT / "MVTEC_GROUP_STATS.csv", mvtec_rows)
    write_csv(OUT / "HARD_TAIL_CONCENTRATION.csv", hard_rows)
    write_csv(OUT / "GRADIENT_GEOMETRY.csv", grad_rows)
    write_csv(OUT / "GROUP_GRADIENT_COSINE.csv", group_grad_rows)
    write_csv(OUT / "CONTROL_COMPARISON.csv", control_rows)
    write_csv(OUT / "GLOBAL_CVAR_EQUIVALENCE.csv", [dict(source=s["source"], **{r["metric"]: r["value"] for r in control_rows if r["source"] == s["source"] and r["metric"] in ("jaccard", "overlap_over_score_top")}) for s in summaries])
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "source_summaries": summaries, "gates": passes, "runtime_sec": runtime, "device": device, "no_train": True, "no_target": True, "optimizer_step": False, "K": K, "top_fraction": TOP_FRAC})
    write_report(outcome, summaries, passes, runtime)
    print(json.dumps({"outcome": outcome, "summaries": summaries, "runtime_sec": runtime}, indent=2, default=json_default), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
