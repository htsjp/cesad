import argparse
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

from experiments.phase_hard_normal_contrast import collect_source_patch_pool as collect_visa_pool
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC
from experiments.phase_hard_normal_cross_source_mvtec import (
    collect_source_pool as collect_mvtec_pool,
    init_midpoint_contrast,
    load_full_mvtec_source,
    source_slots_by_category as mvtec_slots_by_category,
)
from experiments.phase_patch_occupancy_supervision_audit import load_context
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category as visa_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED


OUT = Path("results/phase_hardnormal_origin_heterogeneity_audit/formal")
HN_VISA = Path("results/phase_hard_normal_contrast/formal")
HN_MVTEC = Path("results/phase_hard_normal_cross_source_mvtec/formal")
EPS = 1e-12
GRAD_BATCHES = 100


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


def percentile_stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "P90": float("nan"), "P95": float("nan"), "P99": float("nan")}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "P90": float(np.percentile(x, 90)),
        "P95": float(np.percentile(x, 95)),
        "P99": float(np.percentile(x, 99)),
    }


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / max(EPS, np.linalg.norm(a) * np.linalg.norm(b)))


def load_visa_source(device, smoke):
    q_pc1, align_model, m_c, alpha, v_hn = load_context(device)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = visa_slots_by_category(dataset, indices, smoke)
    groups, _pool = collect_visa_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    return {
        "source": "Full VisA",
        "groups": groups,
        "midpoint": m_c.astype(np.float32),
        "alpha": float(alpha),
        "v": v_hn.astype(np.float32),
    }


def load_mvtec_source(device, smoke):
    _name, dataset, indices, patch = load_full_mvtec_source(smoke)
    slots = mvtec_slots_by_category(dataset, indices)
    groups, _pool = collect_mvtec_pool(dataset, indices, patch, slots)
    midpoint, _init_v, alpha = init_midpoint_contrast(groups)
    payload = torch.load(HN_MVTEC / ("E1_HardNormal_smoke.pt" if smoke else "E1_HardNormal_formal.pt"), map_location=device, weights_only=False)
    return {
        "source": "Full MVTec",
        "groups": groups,
        "midpoint": midpoint.astype(np.float32),
        "alpha": float(alpha),
        "v": np.asarray(payload["v"], dtype=np.float32),
    }


def source_scores(groups, midpoint, alpha, v, device):
    out = {}
    mt = torch.tensor(midpoint, dtype=torch.float32, device=device)
    vt = torch.tensor(v, dtype=torch.float32, device=device)
    for cat, data in groups.items():
        with torch.no_grad():
            z = torch.tensor(data["x"], dtype=torch.float32, device=device)
            score = contrast_logits(z, mt, alpha, F.normalize(vt, dim=-1)).detach().cpu().numpy().astype(np.float32)
        out[cat] = score
    return out


def origin_pool_stats(source, groups, scores):
    rows = []
    all_scores = {"N_clean": [], "N_context": []}
    for cat, data in groups.items():
        neg = data["y"] < 0.5
        origins = data["origin"]
        for origin_label, origin_name in (("N_clean", "normal_image"), ("N_context", "anomaly_mask_out")):
            mask = neg & (origins == origin_name)
            vals = scores[cat][mask]
            all_scores[origin_label].append(vals)
            st = percentile_stats(vals)
            rows.append({"source": source, "category": cat, "origin": origin_label, "pool_ratio_within_category_negatives": float(mask.sum() / max(1, neg.sum())), **st})
    for origin_label in ("N_clean", "N_context"):
        vals = np.concatenate([x for x in all_scores[origin_label] if x.size], axis=0) if any(x.size for x in all_scores[origin_label]) else np.asarray([])
        total_neg = sum(len(x) for xs in all_scores.values() for x in xs)
        st = percentile_stats(vals)
        rows.append({"source": source, "category": "__all__", "origin": origin_label, "pool_ratio_within_category_negatives": float(vals.size / max(1, total_neg)), **st})
    return rows


def hard_selection_stats(source, groups, scores):
    rows = []
    all_clean = []
    all_context = []
    for cat, data in groups.items():
        neg = data["y"] < 0.5
        origins = data["origin"]
        neg_ids = np.flatnonzero(neg)
        k = max(1, int(np.ceil(len(neg_ids) * TOP_NEG_FRAC)))
        local = np.argpartition(scores[cat][neg_ids], -k)[-k:]
        hard_ids = neg_ids[local]
        clean = neg & (origins == "normal_image")
        context = neg & (origins == "anomaly_mask_out")
        hard = np.zeros_like(neg, dtype=bool)
        hard[hard_ids] = True
        h_clean = hard & clean
        h_context = hard & context
        hard_rate_clean = float(h_clean.sum() / max(1, clean.sum()))
        hard_rate_context = float(h_context.sum() / max(1, context.sum()))
        rows.append({
            "source": source,
            "category": cat,
            "N_clean": int(clean.sum()),
            "N_context": int(context.sum()),
            "H_clean": int(h_clean.sum()),
            "H_context": int(h_context.sum()),
            "hard_clean_ratio": float(h_clean.sum() / max(1, hard.sum())),
            "hard_context_ratio": float(h_context.sum() / max(1, hard.sum())),
            "hard_rate_clean": hard_rate_clean,
            "hard_rate_context": hard_rate_context,
            "enrichment_context": float(hard_rate_context / max(EPS, hard_rate_clean)),
        })
        all_clean.append((clean, h_clean, scores[cat], data))
        all_context.append((context, h_context, scores[cat], data))
    clean_n = sum(int(x[0].sum()) for x in all_clean)
    context_n = sum(int(x[0].sum()) for x in all_context)
    clean_h = sum(int(x[1].sum()) for x in all_clean)
    context_h = sum(int(x[1].sum()) for x in all_context)
    hard_total = clean_h + context_h
    hard_rate_clean = clean_h / max(1, clean_n)
    hard_rate_context = context_h / max(1, context_n)
    rows.append({
        "source": source,
        "category": "__all__",
        "N_clean": clean_n,
        "N_context": context_n,
        "H_clean": clean_h,
        "H_context": context_h,
        "hard_clean_ratio": clean_h / max(1, hard_total),
        "hard_context_ratio": context_h / max(1, hard_total),
        "hard_rate_clean": hard_rate_clean,
        "hard_rate_context": hard_rate_context,
        "enrichment_context": hard_rate_context / max(EPS, hard_rate_clean),
    })
    return rows


def sample_batch(groups, rng):
    names = list(groups)
    per_cat = max(2, BATCH_SIZE // max(1, len(names)))
    xs, ys, origins, cats = [], [], [], []
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
            cats.extend([cat] * len(choice))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], np.asarray(origins, dtype=object)[order], np.asarray(cats, dtype=object)[order]


def grad_from_loss(z_np, y_np, origin_np, midpoint, alpha, v_np, device, mode):
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    v = torch.nn.Parameter(torch.tensor(v_np, dtype=torch.float32, device=device))
    m = torch.tensor(midpoint, dtype=torch.float32, device=device)
    logits = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
    neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    if neg_idx.numel() == 0:
        return None, float("nan"), {"H_clean": 0, "H_context": 0}
    k = max(1, int(np.ceil(float(neg_idx.numel()) * TOP_NEG_FRAC)))
    selected = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
    sel_origin = origin_np[selected.detach().cpu().numpy()]
    clean_local = selected[torch.tensor(sel_origin == "normal_image", dtype=torch.bool, device=device)]
    context_local = selected[torch.tensor(sel_origin == "anomaly_mask_out", dtype=torch.bool, device=device)]
    if mode == "clean":
        if clean_local.numel() == 0:
            return None, float("nan"), {"H_clean": 0, "H_context": int(context_local.numel())}
        loss = F.softplus(logits[clean_local]).mean()
    elif mode == "context":
        if context_local.numel() == 0:
            return None, float("nan"), {"H_clean": int(clean_local.numel()), "H_context": 0}
        loss = F.softplus(logits[context_local]).mean()
    elif mode == "current":
        loss = F.softplus(logits[selected]).mean()
    elif mode == "balanced":
        terms = []
        if clean_local.numel():
            terms.append(F.softplus(logits[clean_local]).mean())
        if context_local.numel():
            terms.append(F.softplus(logits[context_local]).mean())
        if len(terms) < 2:
            return None, float("nan"), {"H_clean": int(clean_local.numel()), "H_context": int(context_local.numel())}
        loss = 0.5 * terms[0] + 0.5 * terms[1]
    else:
        raise ValueError(mode)
    loss.backward()
    grad = v.grad.detach().flatten().cpu().numpy().astype(np.float64)
    return grad, float(loss.detach().cpu()), {"H_clean": int(clean_local.numel()), "H_context": int(context_local.numel())}


def gradient_geometry(source, groups, midpoint, alpha, v, device, smoke):
    rng = np.random.default_rng(SEED)
    batches = 10 if smoke else GRAD_BATCHES
    rows = []
    clean_grads = []
    context_grads = []
    for b in range(batches):
        x, y, origin, _cat = sample_batch(groups, rng)
        gc, lc, hc = grad_from_loss(x, y, origin, midpoint, alpha, v, device, "clean")
        gx, lx, hx = grad_from_loss(x, y, origin, midpoint, alpha, v, device, "context")
        gcur, lcur, hcur = grad_from_loss(x, y, origin, midpoint, alpha, v, device, "current")
        gbal, lbal, hbal = grad_from_loss(x, y, origin, midpoint, alpha, v, device, "balanced")
        if gc is None or gx is None or gcur is None or gbal is None:
            continue
        clean_grads.append(gc / max(EPS, np.linalg.norm(gc)))
        context_grads.append(gx / max(EPS, np.linalg.norm(gx)))
        rows.append({
            "source": source,
            "record_type": "batch",
            "batch": b,
            "H_clean": hc["H_clean"],
            "H_context": hx["H_context"],
            "cos_g_clean_g_context": cosine(gc, gx),
            "norm_g_clean": float(np.linalg.norm(gc)),
            "norm_g_context": float(np.linalg.norm(gx)),
            "loss_clean": lc,
            "loss_context": lx,
            "cos_g_current_g_balanced": cosine(gcur, gbal),
            "norm_ratio_balanced_over_current": float(np.linalg.norm(gbal) / max(EPS, np.linalg.norm(gcur))),
            "loss_current": lcur,
            "loss_balanced": lbal,
        })
    if rows:
        batch_rows = list(rows)
        summary_rows = []
        for key in ("cos_g_clean_g_context", "norm_g_clean", "norm_g_context", "cos_g_current_g_balanced", "norm_ratio_balanced_over_current"):
            vals = np.asarray([r[key] for r in batch_rows], dtype=np.float64)
            summary_rows.append({
                "source": source,
                "record_type": f"summary_{key}",
                "batch": "__all__",
                "valid_batches": len(vals),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "P5": float(np.percentile(vals, 5)),
                "P95": float(np.percentile(vals, 95)),
            })
        rows = summary_rows + batch_rows
    g_clean = np.mean(clean_grads, axis=0) if clean_grads else np.zeros_like(v, dtype=np.float64)
    g_context = np.mean(context_grads, axis=0) if context_grads else np.zeros_like(v, dtype=np.float64)
    return rows, g_clean, g_context


def category_origin_stats(source, groups, scores, midpoint, alpha, v, device):
    rows = []
    for cat, data in groups.items():
        neg = data["y"] < 0.5
        neg_ids = np.flatnonzero(neg)
        if neg_ids.size == 0:
            continue
        k = max(1, int(np.ceil(neg_ids.size * TOP_NEG_FRAC)))
        hard_ids = neg_ids[np.argpartition(scores[cat][neg_ids], -k)[-k:]]
        origin = data["origin"]
        clean = hard_ids[origin[hard_ids] == "normal_image"]
        context = hard_ids[origin[hard_ids] == "anomaly_mask_out"]
        base = {
            "source": source,
            "category": cat,
            "N_clean": int(np.sum(neg & (origin == "normal_image"))),
            "N_context": int(np.sum(neg & (origin == "anomaly_mask_out"))),
            "H_clean": int(clean.size),
            "H_context": int(context.size),
            "hard_context_ratio": float(context.size / max(1, hard_ids.size)),
        }
        if clean.size and context.size:
            gc, _lc, _ = grad_for_ids(data["x"], clean, midpoint, alpha, v, device)
            gx, _lx, _ = grad_for_ids(data["x"], context, midpoint, alpha, v, device)
            base["cos_g_clean_g_context"] = cosine(gc, gx)
            base["norm_g_clean"] = float(np.linalg.norm(gc))
            base["norm_g_context"] = float(np.linalg.norm(gx))
        else:
            base["cos_g_clean_g_context"] = float("nan")
            base["norm_g_clean"] = float("nan")
            base["norm_g_context"] = float("nan")
        rows.append(base)
    return rows


def grad_for_ids(x_np, ids, midpoint, alpha, v_np, device):
    z = torch.tensor(x_np[np.asarray(ids, dtype=np.int64)], dtype=torch.float32, device=device)
    v = torch.nn.Parameter(torch.tensor(v_np, dtype=torch.float32, device=device))
    m = torch.tensor(midpoint, dtype=torch.float32, device=device)
    logits = contrast_logits(z, m, alpha, F.normalize(v, dim=-1))
    loss = F.softplus(logits).mean()
    loss.backward()
    g = v.grad.detach().flatten().cpu().numpy().astype(np.float64)
    return g, float(loss.detach().cpu()), int(len(ids))


def decide(grad_rows, cross_rows):
    def mean_for(source, record_type):
        vals = [r.get("mean") for r in grad_rows if r.get("source") == source and r.get("record_type") == record_type]
        return float(vals[0]) if vals else float("nan")
    cos_gc = np.asarray([mean_for(s, "summary_cos_g_clean_g_context") for s in ("Full VisA", "Full MVTec")], dtype=np.float64)
    cos_bal = np.asarray([mean_for(s, "summary_cos_g_current_g_balanced") for s in ("Full VisA", "Full MVTec")], dtype=np.float64)
    mean_clean_context = float(np.nanmean(cos_gc))
    mean_current_balanced = float(np.nanmean(cos_bal))
    cross = cross_rows[0]
    if mean_clean_context >= 0.98 and mean_current_balanced >= 0.995:
        outcome = "NORMAL_ORIGIN_COLLAPSES_TO_SAME_HARDNORMAL_SIGNAL"
    elif mean_clean_context <= 0.95 and mean_current_balanced <= 0.98:
        if float(cross["cos_g_clean_V_g_clean_M"]) > float(cross["cos_g_context_V_g_context_M"]) + 0.05:
            outcome = "CLEAN_NORMAL_ORIGIN_MORE_TRANSFERABLE"
        else:
            outcome = "NORMAL_ORIGIN_HETEROGENEITY_CONFIRMED"
    else:
        outcome = "NORMAL_ORIGIN_HETEROGENEITY_WEAK"
    return outcome, {
        "mean_cos_g_clean_g_context": mean_clean_context,
        "mean_cos_g_current_g_balanced": mean_current_balanced,
        "cos_g_clean_V_g_clean_M": float(cross["cos_g_clean_V_g_clean_M"]),
        "cos_g_context_V_g_context_M": float(cross["cos_g_context_V_g_context_M"]),
    }


def write_report(outcome, stats, pool_rows, hard_rows, runtime, smoke):
    def row(rows, source, cat, origin=None):
        for r in rows:
            if r.get("source") == source and r.get("category") == cat and (origin is None or r.get("origin") == origin):
                return r
        return {}
    lines = [
        "# Hard-Normal Origin Heterogeneity Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "NO-TRAIN audit. Full VisA and Full MVTec source pools reuse their current formal Symmetric HardNormal checkpoint/scorer. No optimizer step was executed.",
        "",
        "## Origin Pools",
        "",
        "| Source | Origin | Count | Ratio | Score mean | P95 | P99 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for source in ("Full VisA", "Full MVTec"):
        for origin in ("N_clean", "N_context"):
            r = row(pool_rows, source, "__all__", origin)
            lines.append(f"| {source} | {origin} | {int(r.get('count', 0))} | {float(r.get('pool_ratio_within_category_negatives', np.nan)):.4f} | {float(r.get('mean', np.nan)):.6f} | {float(r.get('P95', np.nan)):.6f} | {float(r.get('P99', np.nan)):.6f} |")
    lines.extend([
        "",
        "## Hard Selection",
        "",
        "| Source | H_clean | H_context | hard_rate_clean | hard_rate_context | enrichment_context |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for source in ("Full VisA", "Full MVTec"):
        r = row(hard_rows, source, "__all__")
        lines.append(f"| {source} | {int(r.get('H_clean', 0))} | {int(r.get('H_context', 0))} | {float(r.get('hard_rate_clean', np.nan)):.6f} | {float(r.get('hard_rate_context', np.nan)):.6f} | {float(r.get('enrichment_context', np.nan)):.6f} |")
    lines.extend([
        "",
        "## Gradient Geometry",
        "",
        f"- mean cos(g_clean,g_context): {stats['mean_cos_g_clean_g_context']:.6f}.",
        f"- mean cos(g_current,g_origin_balanced): {stats['mean_cos_g_current_g_balanced']:.6f}.",
        f"- cross-source cos clean: {stats['cos_g_clean_V_g_clean_M']:.6f}.",
        f"- cross-source cos context: {stats['cos_g_context_V_g_context_M']:.6f}.",
        "",
        f"Runtime: {runtime:.1f}s. Smoke: {smoke}.",
        "",
        "STOP: no training was started.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OriginAudit] device={device}", flush=True)
    contexts = [load_visa_source(device, args.smoke), load_mvtec_source(device, args.smoke)]
    pool_rows, hard_rows, grad_rows, cat_rows = [], [], [], []
    source_grads = {}
    for ctx in contexts:
        source = ctx["source"]
        print(f"[OriginAudit] source={source}", flush=True)
        scores = source_scores(ctx["groups"], ctx["midpoint"], ctx["alpha"], ctx["v"], device)
        pool_rows.extend(origin_pool_stats(source, ctx["groups"], scores))
        hard_rows.extend(hard_selection_stats(source, ctx["groups"], scores))
        rows, g_clean, g_context = gradient_geometry(source, ctx["groups"], ctx["midpoint"], ctx["alpha"], ctx["v"], device, args.smoke)
        grad_rows.extend(rows)
        cat_rows.extend(category_origin_stats(source, ctx["groups"], scores, ctx["midpoint"], ctx["alpha"], ctx["v"], device))
        source_grads[source] = {"clean": g_clean, "context": g_context}
    cross_rows = [{
        "cos_g_clean_V_g_clean_M": cosine(source_grads["Full VisA"]["clean"], source_grads["Full MVTec"]["clean"]),
        "cos_g_context_V_g_context_M": cosine(source_grads["Full VisA"]["context"], source_grads["Full MVTec"]["context"]),
        "cos_g_clean_V_g_context_M": cosine(source_grads["Full VisA"]["clean"], source_grads["Full MVTec"]["context"]),
        "cos_g_context_V_g_clean_M": cosine(source_grads["Full VisA"]["context"], source_grads["Full MVTec"]["clean"]),
    }]
    outcome, stats = decide(grad_rows, cross_rows)
    runtime = time.time() - start
    write_csv(OUT / "ORIGIN_POOL_STATS.csv", pool_rows)
    write_csv(OUT / "ORIGIN_HARD_STATS.csv", hard_rows)
    write_csv(OUT / "ORIGIN_GRADIENT_GEOMETRY.csv", grad_rows)
    write_csv(OUT / "CROSS_SOURCE_ORIGIN_GEOMETRY.csv", cross_rows)
    write_csv(OUT / "CATEGORY_ORIGIN_STATS.csv", cat_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "no_training": True,
        "optimizer_step": False,
        "sources": ["Full VisA", "Full MVTec"],
        "gradient_batches_per_source": 10 if args.smoke else GRAD_BATCHES,
        "top_negative_fraction": TOP_NEG_FRAC,
        "smoke": bool(args.smoke),
    })
    write_report(outcome, stats, pool_rows, hard_rows, runtime, bool(args.smoke))
    print(f"[OriginAudit] OUTCOME={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
