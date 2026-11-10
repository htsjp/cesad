import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cross_foundation_stable_hardness_audit import build_clip_radio_pool
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_cross_foundation_relative_ranking_pilot/formal")
SEED = 0
EPOCHS = 10
BATCH_SIZE = 4096
MAX_PAIRS_PER_IMAGE_LABEL = 256
CLAMP = 1e-5


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


def sigmoid_np(x):
    return np.clip(1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32))), CLAMP, 1.0 - CLAMP).astype(np.float32)


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def metric_row(cat, variant, y, logits):
    score = sigmoid_np(logits)
    auc = safe_auc(y, score)
    ap = safe_ap(y, score)
    return {
        "heldout_category": cat,
        "variant": variant,
        "P-AUROC": auc,
        "P-AP": ap,
        "G_pixel": 0.5 * (auc + ap) if np.isfinite(auc) and np.isfinite(ap) else float("nan"),
        "patches": int(len(y)),
        "positives": int(np.sum(np.asarray(y) > 0.5)),
    }


def source_arrays(pool, smoke):
    n = len(pool["y"])
    ids = np.arange(n, dtype=np.int64)
    if smoke:
        keep = []
        counts = defaultdict(int)
        for i, c in enumerate(pool["category"].tolist()):
            if counts[c] < 900:
                keep.append(i)
                counts[c] += 1
        ids = np.asarray(keep, dtype=np.int64)
    x = np.asarray(pool["x"][ids], dtype=np.float32)
    y = np.asarray(pool["y"][ids], dtype=np.float32)
    q_c = np.asarray(pool["S_C"][ids], dtype=np.float32)
    q_r = np.asarray(pool["S_R"][ids], dtype=np.float32)
    p_c = sigmoid_np(q_c)
    p_r = sigmoid_np(q_r)
    ce_c = -(y * np.log(p_c) + (1.0 - y) * np.log(1.0 - p_c))
    ce_r = -(y * np.log(p_r) + (1.0 - y) * np.log(1.0 - p_r))
    return {
        "x": x,
        "y": y,
        "qC": q_c,
        "qR": q_r,
        "pC": p_c,
        "pR": p_r,
        "adv": (ce_r < ce_c),
        "category": np.asarray(pool["category"][ids], dtype=object),
        "position": np.asarray(pool["position"][ids], dtype=np.int64),
        "patch_id": np.asarray(pool["patch_id"][ids], dtype=np.int64),
    }


def sample_pairwise_stats(qc, qr, rng, max_pairs=200000):
    n = len(qc)
    if n < 2:
        return {"kendall_tau": float("nan"), "spearman": float("nan"), "disagreement_ratio": float("nan"), "pairs_sampled": 0}
    a = rng.integers(0, n, size=max_pairs)
    b = rng.integers(0, n, size=max_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    dc = np.sign(qc[a] - qc[b])
    dr = np.sign(qr[a] - qr[b])
    valid = (dc != 0) & (dr != 0)
    dis = float(np.mean(dc[valid] != dr[valid])) if np.any(valid) else float("nan")
    return {
        "kendall_tau": float(kendalltau(qc, qr).statistic) if n > 2 else float("nan"),
        "spearman": float(spearmanr(qc, qr).statistic) if n > 2 else float("nan"),
        "disagreement_ratio": dis,
        "pairs_sampled": int(np.sum(valid)),
    }


def phase0_ranking_gate(data):
    rng = np.random.default_rng(SEED)
    rows = []
    redundant = True
    for label, name in [(0, "normal_normal"), (1, "anomaly_anomaly")]:
        ids = np.flatnonzero(data["y"] == label)
        row = {"label_group": name, "patches": int(ids.size), **sample_pairwise_stats(data["qC"][ids], data["qR"][ids], rng)}
        rows.append(row)
        kt = row["kendall_tau"]
        dis = row["disagreement_ratio"]
        if not ((np.isfinite(kt) and kt >= 0.9) or (np.isfinite(dis) and dis < 0.10)):
            redundant = False
    return rows, redundant


def construct_pairs(data, train_mask, rng, mode):
    train_ids = np.flatnonzero(train_mask & data["adv"])
    by_pos_label = defaultdict(list)
    for idx in train_ids:
        by_pos_label[(int(data["position"][idx]), int(data["y"][idx]))].append(int(idx))
    left, right, target, weight, label = [], [], [], [], []
    for (_pos, lab), ids in by_pos_label.items():
        if len(ids) < 2:
            continue
        ids = np.asarray(ids, dtype=np.int64)
        max_possible = len(ids) * (len(ids) - 1) // 2
        n_pairs = min(MAX_PAIRS_PER_IMAGE_LABEL, max_possible)
        pairs = set()
        tries = 0
        while len(pairs) < n_pairs and tries < n_pairs * 20:
            i, j = rng.choice(ids, size=2, replace=False)
            a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            pairs.add((a, b))
            tries += 1
        for i, j in pairs:
            if mode == "radio":
                t = np.sign(data["qR"][i] - data["qR"][j])
                w = abs(data["pR"][i] - data["pR"][j])
            elif mode == "clip":
                t = np.sign(data["qC"][i] - data["qC"][j])
                w = abs(data["pR"][i] - data["pR"][j])
            elif mode == "random":
                t = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32))
                w = abs(data["pR"][i] - data["pR"][j])
            else:
                raise ValueError(mode)
            if t == 0:
                continue
            left.append(i)
            right.append(j)
            target.append(float(t))
            weight.append(float(w))
            label.append(lab)
    return {
        "i": np.asarray(left, dtype=np.int64),
        "j": np.asarray(right, dtype=np.int64),
        "t": np.asarray(target, dtype=np.float32),
        "w": np.asarray(weight, dtype=np.float32),
        "label": np.asarray(label, dtype=np.int64),
    }


class RankHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, 1, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x))).squeeze(-1)


def train_head(variant, cat, data, pairs, init_state, device, smoke):
    ckpt = OUT / "checkpoints" / f"{variant}_{cat}_lr{LR:g}.pt"
    if ckpt.exists() and not smoke:
        head = RankHead(data["x"].shape[1]).to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        head.load_state_dict(payload["state_dict"])
        return head, payload.get("curves", []), True
    head = RankHead(data["x"].shape[1]).to(device)
    head.load_state_dict(init_state)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)
    epochs = 2 if smoke else EPOCHS
    n = len(pairs["i"])
    rng = np.random.default_rng(SEED)
    curves = []
    x_all = data["x"]
    q_c = data["qC"]
    p_c = data["pC"]
    if n == 0:
        raise RuntimeError(f"No pairs for {variant}/{cat}")
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        losses, rank_losses, anchor_losses = [], [], []
        for start in range(0, n, BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            li = pairs["i"][ids]
            rj = pairs["j"][ids]
            needed = np.unique(np.concatenate([li, rj]))
            local = {int(v): k for k, v in enumerate(needed.tolist())}
            xb = torch.tensor(x_all[needed], dtype=torch.float32, device=device)
            delta_needed = head(xb)
            di = delta_needed[torch.tensor([local[int(v)] for v in li], dtype=torch.long, device=device)]
            dj = delta_needed[torch.tensor([local[int(v)] for v in rj], dtype=torch.long, device=device)]
            qi = torch.tensor(q_c[li], dtype=torch.float32, device=device) + di
            qj = torch.tensor(q_c[rj], dtype=torch.float32, device=device) + dj
            t = torch.tensor(pairs["t"][ids], dtype=torch.float32, device=device)
            w = torch.tensor(pairs["w"][ids], dtype=torch.float32, device=device)
            rank_loss = (w * F.softplus(-t * (qi - qj))).mean()
            p_final = torch.sigmoid(torch.tensor(q_c[needed], dtype=torch.float32, device=device) + delta_needed)
            p_base = torch.tensor(p_c[needed], dtype=torch.float32, device=device)
            anchor = F.mse_loss(p_final, p_base)
            loss = rank_loss + anchor
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            rank_losses.append(float(rank_loss.detach().cpu()))
            anchor_losses.append(float(anchor.detach().cpu()))
        row = {"variant": variant, "heldout_category": cat, "epoch": epoch, "loss": float(np.mean(losses)), "rank_loss": float(np.mean(rank_losses)), "anchor_loss": float(np.mean(anchor_losses))}
        curves.append(row)
        print(f"[RankDistill] {variant}/{cat} epoch {epoch}/{epochs} loss={row['loss']:.6f}", flush=True)
    if not smoke:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "curves": curves}, ckpt)
    return head, curves, False


@torch.no_grad()
def predict_delta(head, x, device):
    out = []
    head.eval()
    for start in range(0, x.shape[0], BATCH_SIZE):
        xb = torch.tensor(x[start : start + BATCH_SIZE], dtype=torch.float32, device=device)
        out.append(head(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def linear_r2(x, delta, max_rows=20000):
    if len(delta) > max_rows:
        rng = np.random.default_rng(SEED)
        ids = rng.choice(np.arange(len(delta)), size=max_rows, replace=False)
        x = x[ids]
        delta = delta[ids]
    x_aug = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    coef, *_ = np.linalg.lstsq(x_aug, delta.astype(np.float64), rcond=1e-4)
    pred = x_aug @ coef
    ss_res = float(np.sum((delta - pred) ** 2))
    ss_tot = float(np.sum((delta - np.mean(delta)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def eval_tau(a, b):
    if len(a) < 3:
        return float("nan")
    return float(kendalltau(a, b).statistic)


def run_loo(data, device, smoke):
    cats = sorted(set(data["category"].tolist()))
    if smoke:
        cats = cats[:4]
    torch.manual_seed(SEED)
    init = RankHead(data["x"].shape[1])
    init_state = {k: v.detach().clone() for k, v in init.state_dict().items()}
    raw_rows, diag_rows, pair_rows, curve_rows = [], [], [], []
    for fold, cat in enumerate(cats):
        print(f"[RankDistill] LOO {fold + 1}/{len(cats)} heldout={cat}", flush=True)
        train = data["category"] != cat
        test = data["category"] == cat
        raw_rows.append(metric_row(cat, "B0_CLIP_HardNormal", data["y"][test], data["qC"][test]))
        raw_rows.append(metric_row(cat, "RADIO_Teacher_Reference", data["y"][test], data["qR"][test]))
        rng = np.random.default_rng(SEED + fold)
        pairsets = {
            "E1_RADIO_SameLabelRank": construct_pairs(data, train, rng, "radio"),
            "E2_CLIP_SelfRank": construct_pairs(data, train, rng, "clip"),
            "E3_RandomRank": construct_pairs(data, train, rng, "random"),
        }
        for variant, pairs in pairsets.items():
            pair_rows.append({
                "heldout_category": cat,
                "variant": variant,
                "pairs": int(len(pairs["i"])),
                "normal_pairs": int(np.sum(pairs["label"] == 0)),
                "anomaly_pairs": int(np.sum(pairs["label"] == 1)),
                "mean_weight": float(np.mean(pairs["w"])) if len(pairs["w"]) else float("nan"),
            })
            head, curves, reused = train_head(variant, cat, data, pairs, init_state, device, smoke)
            curve_rows.extend(curves)
            delta = predict_delta(head, data["x"][test], device)
            q_final = data["qC"][test] + delta
            raw_rows.append(metric_row(cat, variant, data["y"][test], q_final))
            tb = data["adv"][test]
            diag_rows.append({
                "heldout_category": cat,
                "variant": variant,
                "reused": bool(reused),
                "kendall_qfinal_qR": eval_tau(q_final, data["qR"][test]),
                "kendall_qC_qR": eval_tau(data["qC"][test], data["qR"][test]),
                "delta_kendall_vs_B0": eval_tau(q_final, data["qR"][test]) - eval_tau(data["qC"][test], data["qR"][test]),
                "delta_linear_R2": linear_r2(data["x"][test], delta),
                "mean_abs_delta": float(np.mean(np.abs(delta))),
                "corr_delta_qC": corr(delta, data["qC"][test]),
                "corr_delta_qR_minus_qC": corr(delta, data["qR"][test] - data["qC"][test]),
                "mean_abs_delta_teacher_better": float(np.mean(np.abs(delta[tb]))) if np.any(tb) else float("nan"),
                "mean_abs_delta_teacher_worse": float(np.mean(np.abs(delta[~tb]))) if np.any(~tb) else float("nan"),
            })
    return raw_rows, diag_rows, pair_rows, curve_rows


def summarize(raw_rows, diag_rows):
    variants = ["B0_CLIP_HardNormal", "RADIO_Teacher_Reference", "E1_RADIO_SameLabelRank", "E2_CLIP_SelfRank", "E3_RandomRank"]
    summary = []
    for variant in variants:
        rows = [r for r in raw_rows if r["variant"] == variant]
        summary.append({
            "variant": variant,
            "heldout_categories": len(rows),
            "mean_P-AUROC": float(np.mean([r["P-AUROC"] for r in rows])),
            "mean_P-AP": float(np.mean([r["P-AP"] for r in rows])),
            "mean_G": float(np.mean([r["G_pixel"] for r in rows])),
            "median_G": float(np.median([r["G_pixel"] for r in rows])),
        })
    by = {(r["heldout_category"], r["variant"]): r for r in raw_rows}
    cats = sorted({r["heldout_category"] for r in raw_rows})
    deltas = []
    for cat in cats:
        b0 = by[(cat, "B0_CLIP_HardNormal")]["G_pixel"]
        e1 = by[(cat, "E1_RADIO_SameLabelRank")]["G_pixel"]
        e2 = by[(cat, "E2_CLIP_SelfRank")]["G_pixel"]
        e3 = by[(cat, "E3_RandomRank")]["G_pixel"]
        deltas.append({"heldout_category": cat, "E1_minus_B0_G": e1 - b0, "E1_minus_E2_G": e1 - e2, "E1_minus_E3_G": e1 - e3})
    e1_r2 = [r["delta_linear_R2"] for r in diag_rows if r["variant"] == "E1_RADIO_SameLabelRank" and np.isfinite(r["delta_linear_R2"])]
    stats = {
        "categories": len(cats),
        "E1_improved_over_B0": int(sum(d["E1_minus_B0_G"] > 0 for d in deltas)),
        "mean_E1_minus_B0_G": float(np.mean([d["E1_minus_B0_G"] for d in deltas])),
        "median_E1_minus_B0_G": float(np.median([d["E1_minus_B0_G"] for d in deltas])),
        "mean_E1_minus_E2_G": float(np.mean([d["E1_minus_E2_G"] for d in deltas])),
        "mean_E1_minus_E3_G": float(np.mean([d["E1_minus_E3_G"] for d in deltas])),
        "mean_E1_delta_linear_R2": float(np.mean(e1_r2)) if e1_r2 else float("nan"),
    }
    return summary, deltas, stats


def decide(stats):
    if (
        stats["E1_improved_over_B0"] >= 8
        and stats["mean_E1_minus_B0_G"] >= 1.0
        and stats["median_E1_minus_B0_G"] > 0
        and stats["mean_E1_minus_E2_G"] >= 0.5
        and stats["mean_E1_minus_E3_G"] >= 0.5
        and stats["mean_E1_delta_linear_R2"] < 0.95
    ):
        return "CROSS_FOUNDATION_RELATIVE_RANKING_HAS_SIGNAL"
    if abs(stats["mean_E1_minus_E2_G"]) < 0.5:
        return "PAIRWISE_TRAINING_SIGNAL_NOT_TEACHER_SPECIFIC"
    if abs(stats["mean_E1_minus_E3_G"]) < 0.5:
        return "RADIO_RANKING_NOT_SPECIFIC"
    if stats["mean_E1_minus_B0_G"] > 0 and stats["mean_E1_delta_linear_R2"] >= 0.95:
        return "RANK_DISTILLATION_COLLAPSES_TO_LINEAR_CONTRAST"
    return "CROSS_FOUNDATION_RANKING_NO_SIGNAL"


def write_report(outcome, rank_rows, summary_rows, stats, runtime):
    lines = [
        "# Cross-Foundation Same-Label Ranking Distillation Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "SOURCE-ONLY pilot. CLIP Symmetric HardNormal and RADIO StateAligned are frozen. The only trainable module is a tiny correction head; RADIO is used only to define same-label ranking targets in VisA LOO.",
        "",
        "## Phase 0",
        "| Label group | Kendall tau | Spearman | Disagreement |",
        "|---|---:|---:|---:|",
    ]
    for r in rank_rows:
        lines.append(f"| {r['label_group']} | {r['kendall_tau']:.4f} | {r['spearman']:.4f} | {r['disagreement_ratio']:.4f} |")
    lines.extend(["", "## LOO Mean Metrics", "| Variant | Mean P-AUROC | Mean P-AP | Mean G | Median G |", "|---|---:|---:|---:|---:|"])
    for r in summary_rows:
        lines.append(f"| {r['variant']} | {r['mean_P-AUROC']:.4f} | {r['mean_P-AP']:.4f} | {r['mean_G']:.4f} | {r['median_G']:.4f} |")
    lines.extend([
        "",
        "## Gate",
        f"- E1 improved over B0: {stats['E1_improved_over_B0']}/{stats['categories']}",
        f"- mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f}",
        f"- median E1-B0 G: {stats['median_E1_minus_B0_G']:+.4f}",
        f"- mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f}",
        f"- mean E1-E3 G: {stats['mean_E1_minus_E3_G']:+.4f}",
        f"- mean E1 delta linear-fit R2: {stats['mean_E1_delta_linear_R2']:.6f}",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RankDistill] device={device}", flush=True)
    pool = build_clip_radio_pool(device, smoke=args.smoke)
    data = source_arrays(pool, args.smoke)
    rank_rows, redundant = phase0_ranking_gate(data)
    write_csv(OUT / "RANK_ALIGNMENT.csv", rank_rows)
    if redundant and not args.force_train:
        outcome = "CROSS_FOUNDATION_RANKING_ALREADY_REDUNDANT"
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "phase0_redundant": True, "target_used": False})
        write_report(outcome, rank_rows, [], {"E1_improved_over_B0": 0, "categories": 0, "mean_E1_minus_B0_G": float("nan"), "median_E1_minus_B0_G": float("nan"), "mean_E1_minus_E2_G": float("nan"), "mean_E1_minus_E3_G": float("nan"), "mean_E1_delta_linear_R2": float("nan")}, time.time() - start)
        print(json.dumps({"outcome": outcome}, indent=2), flush=True)
        return
    raw_rows, diag_rows, pair_rows, curve_rows = run_loo(data, device, args.smoke)
    summary_rows, delta_rows, stats = summarize(raw_rows, diag_rows)
    outcome = decide(stats)
    runtime = time.time() - start
    write_csv(OUT / "LOO_METRICS.csv", raw_rows)
    write_csv(OUT / "SUMMARY_METRICS.csv", summary_rows)
    write_csv(OUT / "DELTA_METRICS.csv", delta_rows)
    write_csv(OUT / "PAIR_DIAGNOSTICS.csv", pair_rows)
    write_csv(OUT / "CORRECTION_DIAGNOSTICS.csv", diag_rows)
    write_csv(OUT / "LINEAR_COLLAPSE.csv", [r for r in diag_rows if r["variant"] == "E1_RADIO_SameLabelRank"])
    write_csv(OUT / "TRAINING_CURVES.csv", curve_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "phase0_redundant": False,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_used": False,
        "epochs": 2 if args.smoke else EPOCHS,
        "lr": LR,
        "head": "Linear(D,32)-GELU-Linear(32,1,bias=False)",
    })
    write_report(outcome, rank_rows, summary_rows, stats, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force-train", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
