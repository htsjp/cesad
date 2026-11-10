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

from experiments.phase_cmab_compact_multi_anomaly_basis import load_mvtec_groups, load_visa_groups
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC
from experiments.phase_snir_source_nuisance_invariant_state_regularization import (
    contrast_direction_np,
    evaluate_pairs,
    fnum,
    finite_mean,
    finite_median,
    source_state_pair,
)
from experiments.phase_ssec_state_space_experience_consolidation import two_state_logits
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE
from utils.transforms import get_transform


OUT = Path("results/phase_tail_ranked_state_learning/formal")
VARIANTS = ("E0_Baseline", "E1_RandomPairRanking", "E2_CrossCategoryHardRanking")
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
)
EPOCHS = 10
LAMBDA_RANK = 1.0
RANK_MARGIN = 0.05


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_source_groups(source, model_info, device, smoke):
    if source == "Full VisA":
        return load_visa_groups(model_info, device, smoke)
    return load_mvtec_groups(smoke)


def category_balanced_batch(groups, rng, total_batch):
    names = sorted(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, cats = [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(np.asarray(data["y"]) == cls)
            if ids.size == 0:
                continue
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(np.asarray(data["x"], dtype=np.float32)[choice])
            ys.append(np.asarray(data["y"], dtype=np.float32)[choice])
            cats.extend([cat] * len(choice))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    cat = np.asarray(cats, dtype=object)
    order = rng.permutation(len(y))
    return x[order], y[order], cat[order]


def hardnormal_loss(logits, y):
    neg = torch.nonzero(y.reshape(-1) < 0.5, as_tuple=False).flatten()
    if neg.numel() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device), 0
    flat = logits.reshape(-1)
    k = max(1, int(math.ceil(float(neg.numel()) * TOP_NEG_FRAC)))
    selected = neg[torch.topk(flat.detach()[neg], k=k, largest=True).indices]
    return F.softplus(flat[selected]).mean(), int(k)


def diagnostic_pairs(logits, y, cats, rng):
    y_np = y.detach().cpu().numpy().reshape(-1)
    q = logits.detach().cpu().numpy().reshape(-1)
    cat = np.asarray(cats, dtype=object)
    pos = np.flatnonzero(y_np > 0.5)
    neg = np.flatnonzero(y_np < 0.5)
    out = {}
    if pos.size == 0 or neg.size == 0:
        for name in ("random", "hard", "crosshard"):
            out[f"{name}_ordering_acc"] = float("nan")
            out[f"{name}_margin"] = float("nan")
        return out
    sample = min(pos.size, 2048)
    pos_s = rng.choice(pos, size=sample, replace=pos.size < sample)
    rand_n = rng.choice(neg, size=sample, replace=neg.size < sample)
    hard_pool = neg[np.argsort(q[neg])[-min(sample, neg.size):]]
    hard_n = rng.choice(hard_pool, size=sample, replace=hard_pool.size < sample)
    cross_n = []
    sorted_neg = neg[np.argsort(q[neg])[::-1]]
    for p in pos_s:
        cand = sorted_neg[cat[sorted_neg] != cat[p]]
        if cand.size == 0:
            cand = sorted_neg
        cross_n.append(cand[0])
    cross_n = np.asarray(cross_n, dtype=np.int64)
    for name, nn in (("random", rand_n), ("hard", hard_n), ("crosshard", cross_n)):
        margin = q[pos_s] - q[nn]
        out[f"{name}_ordering_acc"] = float(np.mean(margin > 0.0))
        out[f"{name}_margin"] = float(np.mean(margin))
    return out


def rank_loss(logits, y, cats, mode, rng):
    y_np = y.detach().cpu().numpy().reshape(-1)
    cat = np.asarray(cats, dtype=object)
    pos = np.flatnonzero(y_np > 0.5)
    neg = np.flatnonzero(y_np < 0.5)
    if pos.size == 0 or neg.size == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device), 0
    pair_count = min(pos.size, neg.size, 4096)
    pos_s = rng.choice(pos, size=pair_count, replace=pos.size < pair_count)
    if mode == "random":
        neg_s = rng.choice(neg, size=pair_count, replace=neg.size < pair_count)
    else:
        q_np = logits.detach().cpu().numpy().reshape(-1)
        sorted_neg = neg[np.argsort(q_np[neg])[::-1]]
        chosen = []
        for p in pos_s:
            cand = sorted_neg[cat[sorted_neg] != cat[p]]
            if cand.size == 0:
                cand = sorted_neg
            chosen.append(cand[int(rng.integers(0, min(32, cand.size)))])
        neg_s = np.asarray(chosen, dtype=np.int64)
    p_t = torch.tensor(pos_s, dtype=torch.long, device=logits.device)
    n_t = torch.tensor(neg_s, dtype=torch.long, device=logits.device)
    flat = logits.reshape(-1)
    return F.softplus(flat[n_t] - flat[p_t] + RANK_MARGIN).mean(), int(pair_count)


def train_variant(source, groups, init_states, variant, device, smoke):
    if variant == "E0_Baseline":
        return init_states[0].astype(np.float32), init_states[1].astype(np.float32), [], True
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "checkpoints" / f"{source.replace(' ', '_')}_{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["s_n"].astype(np.float32), payload["s_a"].astype(np.float32), payload["curves"], True
    print(f"[TRSL] train {source} {variant}", flush=True)
    s_n0, s_a0 = init_states
    s_n = torch.nn.Parameter(torch.tensor(s_n0, dtype=torch.float32, device=device))
    s_a = torch.nn.Parameter(torch.tensor(s_a0, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([s_n, s_a], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    diag_rng = np.random.default_rng(SEED + 17)
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, int(math.ceil(total / BATCH_SIZE)))
    if smoke:
        steps = min(steps, 2)
    rank_mode = "random" if variant == "E1_RandomPairRanking" else "crosshard"
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _ in range(steps):
            xb, yb, cb = category_balanced_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            logits = two_state_logits(z, s_n, s_a)
            state_loss, bce, dice = balanced_loss(logits, y)
            ntr, hard_count = hardnormal_loss(logits, y)
            rloss, pair_count = rank_loss(logits, y, cb, rank_mode, rng)
            loss = state_loss + ntr + LAMBDA_RANK * rloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                d = diagnostic_pairs(logits, y, cb, diag_rng)
            vals.append({
                "loss": float(loss.detach().cpu()),
                "state_loss": float(state_loss.detach().cpu()),
                "ntr_loss": float(ntr.detach().cpu()),
                "rank_loss": float(rloss.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "dice": float(dice.detach().cpu()),
                "hard_patch_count": hard_count,
                "rank_pair_count": pair_count,
                **d,
            })
        row = {"source": source, "variant": variant, "epoch": epoch}
        for key in vals[0]:
            row[key] = finite_mean([v[key] for v in vals])
        curves.append(row)
        print(f"[TRSL] {source} {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} rank={row['rank_loss']:.4f} cross_acc={row['crosshard_ordering_acc']:.3f}", flush=True)
    payload = {"s_n": s_n.detach().cpu().numpy().astype(np.float32), "s_a": s_a.detach().cpu().numpy().astype(np.float32), "curves": curves}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt)
    return payload["s_n"], payload["s_a"], curves, False


def state_geometry(states_by_source_variant):
    rows = []
    for source, variants in states_by_source_variant.items():
        d0 = contrast_direction_np(*variants["E0_Baseline"])
        for variant, states in variants.items():
            d = contrast_direction_np(*states)
            cos = float(np.dot(d0, d))
            rows.append({
                "source": source,
                "variant": variant,
                "cos_d_E0": cos,
                "state_direction_angle": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
                "sN_norm": float(np.linalg.norm(states[0])),
                "sA_norm": float(np.linalg.norm(states[1])),
            })
    return rows


def delta_summary(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in VARIANTS[1:]:
        vals = []
        for source, target in PILOT_PAIRS:
            b = by.get((source, target, "E0_Baseline"))
            e = by.get((source, target, variant))
            if not b or not e:
                continue
            dap = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            vals.append(dap)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": dap,
                "Delta_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
        rows.append({
            "source": "SUMMARY",
            "dataset": "PILOT",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(vals),
            "median_delta_M_FR_P_AP": finite_median(vals),
            "improved": int(sum(v > 0 for v in vals if np.isfinite(v))),
            "non_drop_ge_minus_0_2": int(sum(v >= -0.2 for v in vals if np.isfinite(v))),
            "total": int(sum(np.isfinite(v) for v in vals)),
        })
    return rows


def decide(delta):
    summaries = [r for r in delta if r.get("source") == "SUMMARY"]
    e1 = next((r for r in summaries if r["variant"] == "E1_RandomPairRanking"), None)
    e2 = next((r for r in summaries if r["variant"] == "E2_CrossCategoryHardRanking"), None)
    if not e2:
        return "TAIL_RANKED_STATE_LEARNING_HAS_NO_COMPETITIVE_VALUE", {}
    vals = [r for r in delta if r.get("variant") == "E2_CrossCategoryHardRanking" and r.get("source") != "SUMMARY"]
    mean_delta = fnum(e2, "mean_delta_M_FR_P_AP")
    improved = int(fnum(e2, "improved", 0))
    non_drop = int(fnum(e2, "non_drop_ge_minus_0_2", 0))
    focus = max([fnum(r, "Delta_M_FR_P_AP") for r in vals if (r["source"], r["dataset"]) in {("Full MVTec", "VisA"), ("Full VisA", "MPDD")}], default=float("nan"))
    catastrophic = any(fnum(r, "Delta_M_FR_P_AP") < -5.0 for r in vals)
    e2_minus_e1 = mean_delta - (fnum(e1, "mean_delta_M_FR_P_AP") if e1 else float("nan"))
    stats = {
        "E2_mean_delta_P_AP": mean_delta,
        "E2_improved": improved,
        "E2_non_drop_ge_minus_0_2": non_drop,
        "focus_best_delta_P_AP": focus,
        "catastrophic_drop": catastrophic,
        "E2_minus_E1_mean_delta_P_AP": e2_minus_e1,
    }
    if mean_delta >= 1.0 and improved >= 2 and focus >= 1.0 and e2_minus_e1 >= 0.5 and not catastrophic:
        return "CROSS_CATEGORY_HARD_RANKING_SIGNIFICANTLY_IMPROVES_TRANSFER", stats
    if e1 and fnum(e1, "mean_delta_M_FR_P_AP") > 0 and abs(e2_minus_e1) < 0.5:
        return "RANKING_LOSS_HELPS_BUT_CROSS_CATEGORY_HARDNESS_HAS_NO_ADDED_VALUE", stats
    if mean_delta > 0 and improved >= 1:
        return "HARD_RANKING_HAS_WEAK_VALUE", stats
    return "TAIL_RANKED_STATE_LEARNING_HAS_NO_COMPETITIVE_VALUE", stats


def write_report(raw, delta, geometry, train_rows, outcome, stats, runtime):
    lines = [
        "# Phase TRSL: Tail-Ranked State Learning",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR + fixed 2x2 Local Re-encoding Pyramid. CLIP, feature stream, state count, and Fixed LRP inference are unchanged; ranking only modifies source-side state training.",
        "",
        f"Fixed ranking margin: `{RANK_MARGIN}`. Fixed lambda_rank: `{LAMBDA_RANK}`.",
        "",
        "## Pilot Metrics",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    lines += ["", "## Delta vs E0", "", "| Source | Target | Variant | Delta P-AP | Delta PRO |", "|---|---|---|---:|---:|"]
    for r in delta:
        if r.get("source") != "SUMMARY":
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} | {fnum(r,'Delta_M_FR_PRO'):+.4f} |")
    lines += ["", "## Summary", "", "| Variant | mean Delta P-AP | median | improved | non-drop |", "|---|---:|---:|---:|---:|"]
    for r in delta:
        if r.get("source") == "SUMMARY":
            lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {int(fnum(r,'improved',0))}/{int(fnum(r,'total',0))} | {int(fnum(r,'non_drop_ge_minus_0_2',0))}/{int(fnum(r,'total',0))} |")
    lines += ["", "## State Geometry", "", "| Source | Variant | cos(d,E0) | angle | sN norm | sA norm |", "|---|---|---:|---:|---:|---:|"]
    for r in geometry:
        lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'cos_d_E0'):.4f} | {fnum(r,'state_direction_angle'):.3f} | {fnum(r,'sN_norm'):.4f} | {fnum(r,'sA_norm'):.4f} |")
    last = {}
    for r in train_rows:
        last[(r["source"], r["variant"])] = r
    if last:
        lines += ["", "## Source Ordering Diagnostics", "", "| Source | Variant | random acc | hard acc | cross-hard acc | random margin | hard margin | cross-hard margin |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for (_source, _variant), r in sorted(last.items()):
            lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'random_ordering_acc'):.4f} | {fnum(r,'hard_ordering_acc'):.4f} | {fnum(r,'crosshard_ordering_acc'):.4f} | {fnum(r,'random_margin'):+.4f} | {fnum(r,'hard_margin'):+.4f} | {fnum(r,'crosshard_margin'):+.4f} |")
    lines += ["", f"Decision stats: `{json.dumps(stats, default=json_default)}`", "", f"Runtime: {runtime:.1f}s."]
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRSL] device={device}", flush=True)
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    model_infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    groups_by_source = {}
    pool_rows = []
    for source, info in model_infos.items():
        print(f"[TRSL] load source groups {source}", flush=True)
        groups, audit = load_source_groups(source, info, device, args.smoke)
        groups_by_source[source] = groups
        pool_rows.extend({"source": source, **r} for r in audit)
    states_by_source_variant = {}
    train_rows = []
    for source, info in model_infos.items():
        init = source_state_pair(info)
        states_by_source_variant[source] = {}
        for variant in VARIANTS:
            s_n, s_a, curves, reused = train_variant(source, groups_by_source[source], init, variant, device, args.smoke)
            states_by_source_variant[source][variant] = (s_n, s_a)
            for r in curves:
                r["checkpoint_reused"] = bool(reused)
                train_rows.append(r)
    raw, cats = evaluate_pairs(model_infos, states_by_source_variant, clip_model, preprocess, PILOT_PAIRS, device, args.smoke, VARIANTS)
    delta = delta_summary(raw)
    geometry = state_geometry(states_by_source_variant)
    outcome, stats = decide(delta)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", delta)
    write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
    write_csv(OUT / "SOURCE_TRAINING_DIAGNOSTICS.csv", train_rows)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "margin": RANK_MARGIN,
        "lambda_rank": LAMBDA_RANK,
        "epochs": EPOCHS,
        "lr": LR,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_tuning": False,
        "fixed_lrp": True,
    })
    write_report(raw, delta, geometry, train_rows, outcome, stats, runtime)
    print(f"[TRSL] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
