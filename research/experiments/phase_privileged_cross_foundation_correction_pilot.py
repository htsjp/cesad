import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cross_foundation_stable_hardness_audit import build_clip_radio_pool
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import SEED
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_privileged_cross_foundation_correction_pilot/formal")
EPOCHS = 10
BATCH_SIZE = 4096
HIDDEN = 16
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


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def metric_row(category, variant, y, score):
    auc = safe_auc(y, score)
    ap = safe_ap(y, score)
    return {
        "heldout_category": category,
        "variant": variant,
        "P-AUROC": auc,
        "P-AP": ap,
        "G_pixel": 0.5 * (auc + ap) if np.isfinite(auc) and np.isfinite(ap) else float("nan"),
        "patches": int(len(y)),
        "positives": int(np.sum(np.asarray(y) > 0.5)),
    }


class CorrectionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def prob_from_logits(logits):
    return np.clip(1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32))), CLAMP, 1.0 - CLAMP).astype(np.float32)


def source_arrays(pool, smoke):
    n = len(pool["y"])
    ids = np.arange(n, dtype=np.int64)
    if smoke:
        keep = []
        counts = {}
        for i, c in enumerate(pool["category"].tolist()):
            counts.setdefault(c, 0)
            if counts[c] < 800:
                keep.append(i)
                counts[c] += 1
        ids = np.asarray(keep, dtype=np.int64)
    x = np.asarray(pool["x"][ids], dtype=np.float32)
    y = np.asarray(pool["y"][ids], dtype=np.float32)
    cats = np.asarray(pool["category"][ids], dtype=object)
    p_c = prob_from_logits(pool["S_C"][ids])
    p_r = prob_from_logits(pool["S_R"][ids])
    q_c = np.log(p_c / (1.0 - p_c)).astype(np.float32)
    ce_c = -(y * np.log(p_c) + (1.0 - y) * np.log(1.0 - p_c))
    ce_r = -(y * np.log(p_r) + (1.0 - y) * np.log(1.0 - p_r))
    w = (ce_r < ce_c).astype(np.float32)
    return {"x": x, "y": y, "category": cats, "pC": p_c, "pR": p_r, "qC": q_c, "w": w, "ceC": ce_c, "ceR": ce_r}


def teacher_advantage_rows(data):
    rows = []
    scopes = [("__all__", np.ones_like(data["y"], dtype=bool)), ("normal", data["y"] < 0.5), ("anomaly", data["y"] > 0.5)]
    scopes.extend((str(c), data["category"] == c) for c in sorted(set(data["category"].tolist())))
    for name, mask in scopes:
        if not np.any(mask):
            continue
        rows.append({
            "scope": name,
            "patches": int(np.sum(mask)),
            "teacher_advantage_patches": int(np.sum(data["w"][mask] > 0.5)),
            "teacher_advantage_ratio": float(np.mean(data["w"][mask])),
            "mean_CE_CLIP": float(np.mean(data["ceC"][mask])),
            "mean_CE_RADIO": float(np.mean(data["ceR"][mask])),
        })
    return rows


def random_privileged_target(y, p_c, p_r, w, rng):
    out = p_c.copy()
    for label in (0.0, 1.0):
        ids = np.flatnonzero(y == label)
        if ids.size == 0:
            continue
        k = int(np.sum(w[ids] > 0.5))
        if k == 0:
            continue
        chosen = rng.choice(ids, size=k, replace=False)
        shuffled_values = p_r[ids].copy()
        rng.shuffle(shuffled_values)
        out[chosen] = shuffled_values[:k]
    return out.astype(np.float32)


def train_head(name, x, q_c, target, init_state, device, smoke):
    ckpt = OUT / "checkpoints" / f"{name}.pt"
    if ckpt.exists() and not smoke:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        head = CorrectionHead(x.shape[1]).to(device)
        head.load_state_dict(payload["state_dict"])
        return head, payload.get("curves", []), True
    head = CorrectionHead(x.shape[1]).to(device)
    head.load_state_dict(init_state)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    n = x.shape[0]
    epochs = 2 if smoke else EPOCHS
    curves = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            xb = torch.tensor(x[ids], dtype=torch.float32, device=device)
            base = torch.tensor(q_c[ids], dtype=torch.float32, device=device)
            tgt = torch.tensor(target[ids], dtype=torch.float32, device=device)
            delta = head(xb)
            loss = F.binary_cross_entropy_with_logits(base + delta, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"model": name, "epoch": epoch, "loss": float(np.mean(losses))}
        curves.append(row)
        print(f"[PrivCorrection] {name} epoch {epoch}/{epochs} loss={row['loss']:.5f}", flush=True)
    if not smoke:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "curves": curves}, ckpt)
    return head, curves, False


@torch.no_grad()
def predict_head(head, x, q_c, device):
    outs = []
    deltas = []
    head.eval()
    for start in range(0, x.shape[0], BATCH_SIZE):
        xb = torch.tensor(x[start : start + BATCH_SIZE], dtype=torch.float32, device=device)
        d = head(xb).detach().cpu().numpy().astype(np.float32)
        deltas.append(d)
        outs.append(1.0 / (1.0 + np.exp(-(q_c[start : start + BATCH_SIZE] + d))))
    return np.concatenate(outs).astype(np.float32), np.concatenate(deltas).astype(np.float32)


def linear_r2(x, delta, max_rows=20000):
    if x.shape[0] > max_rows:
        rng = np.random.default_rng(SEED)
        ids = rng.choice(np.arange(x.shape[0]), size=max_rows, replace=False)
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


def run_loo(data, device, smoke):
    cats = sorted(set(data["category"].tolist()))
    if smoke:
        cats = cats[:4]
    raw_rows, diag_rows, curve_rows = [], [], []
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    init_head = CorrectionHead(data["x"].shape[1])
    init_state = {k: v.detach().clone() for k, v in init_head.state_dict().items()}
    for fold, cat in enumerate(cats):
        print(f"[PrivCorrection] LOO {fold + 1}/{len(cats)} heldout={cat}", flush=True)
        train = data["category"] != cat
        test = data["category"] == cat
        xtr, ytr, qtr = data["x"][train], data["y"][train], data["qC"][train]
        pctr, prtr, wtr = data["pC"][train], data["pR"][train], data["w"][train]
        p_priv = (wtr * prtr + (1.0 - wtr) * pctr).astype(np.float32)
        p_rand = random_privileged_target(ytr, pctr, prtr, wtr, rng)
        targets = {
            "E1_PrivilegedCorrection": p_priv,
            "E2_GTOnlyCorrection": ytr.astype(np.float32),
            "E3_RandomPrivilegedCorrection": p_rand,
        }
        heads = {}
        for variant, target in targets.items():
            head, curves, reused = train_head(f"{variant}_{cat}", xtr, qtr, target, init_state, device, smoke)
            heads[variant] = head
            for row in curves:
                curve_rows.append({"heldout_category": cat, "variant": variant, "reused": bool(reused), **row})
        yte = data["y"][test]
        raw_rows.extend([
            metric_row(cat, "B0_CLIP_HardNormal", yte, data["pC"][test]),
            metric_row(cat, "RADIO_Teacher_Reference", yte, data["pR"][test]),
        ])
        for variant, head in heads.items():
            score, delta = predict_head(head, data["x"][test], data["qC"][test], device)
            raw_rows.append(metric_row(cat, variant, yte, score))
            tb = data["w"][test] > 0.5
            tw = ~tb
            diag_rows.append({
                "heldout_category": cat,
                "variant": variant,
                "delta_linear_R2": linear_r2(data["x"][test], delta),
                "corr_delta_pC": corr(delta, data["pC"][test]),
                "corr_delta_pR_minus_pC": corr(delta, data["pR"][test] - data["pC"][test]),
                "mean_abs_delta_teacher_better": float(np.mean(np.abs(delta[tb]))) if np.any(tb) else float("nan"),
                "mean_abs_delta_teacher_worse": float(np.mean(np.abs(delta[tw]))) if np.any(tw) else float("nan"),
                "teacher_better_patches": int(np.sum(tb)),
                "teacher_worse_patches": int(np.sum(tw)),
            })
    return raw_rows, diag_rows, curve_rows


def summarize(raw_rows):
    variants = ["B0_CLIP_HardNormal", "RADIO_Teacher_Reference", "E1_PrivilegedCorrection", "E2_GTOnlyCorrection", "E3_RandomPrivilegedCorrection"]
    rows = []
    for variant in variants:
        vals = [r["G_pixel"] for r in raw_rows if r["variant"] == variant and np.isfinite(r["G_pixel"])]
        rows.append({
            "variant": variant,
            "heldout_categories": len(vals),
            "mean_P-AUROC": float(np.mean([r["P-AUROC"] for r in raw_rows if r["variant"] == variant])),
            "mean_P-AP": float(np.mean([r["P-AP"] for r in raw_rows if r["variant"] == variant])),
            "mean_G": float(np.mean(vals)) if vals else float("nan"),
            "median_G": float(np.median(vals)) if vals else float("nan"),
        })
    by = {(r["heldout_category"], r["variant"]): r for r in raw_rows}
    cats = sorted({r["heldout_category"] for r in raw_rows})
    deltas = []
    for cat in cats:
        b0 = by[(cat, "B0_CLIP_HardNormal")]["G_pixel"]
        e1 = by[(cat, "E1_PrivilegedCorrection")]["G_pixel"]
        e2 = by[(cat, "E2_GTOnlyCorrection")]["G_pixel"]
        e3 = by[(cat, "E3_RandomPrivilegedCorrection")]["G_pixel"]
        deltas.append({"cat": cat, "E1_B0": e1 - b0, "E1_E2": e1 - e2, "E1_E3": e1 - e3})
    stats = {
        "categories": len(cats),
        "E1_improved_over_B0": int(sum(d["E1_B0"] > 0 for d in deltas)),
        "mean_E1_minus_B0_G": float(np.mean([d["E1_B0"] for d in deltas])),
        "median_E1_minus_B0_G": float(np.median([d["E1_B0"] for d in deltas])),
        "mean_E1_minus_E2_G": float(np.mean([d["E1_E2"] for d in deltas])),
        "mean_E1_minus_E3_G": float(np.mean([d["E1_E3"] for d in deltas])),
    }
    return rows, deltas, stats


def decide(stats, diag_rows):
    e1_r2 = [r["delta_linear_R2"] for r in diag_rows if r["variant"] == "E1_PrivilegedCorrection" and np.isfinite(r["delta_linear_R2"])]
    mean_r2 = float(np.mean(e1_r2)) if e1_r2 else float("nan")
    if (
        stats["E1_improved_over_B0"] >= 8
        and stats["mean_E1_minus_B0_G"] >= 1.0
        and stats["mean_E1_minus_E2_G"] >= 0.5
        and stats["mean_E1_minus_E3_G"] >= 0.5
        and stats["median_E1_minus_B0_G"] > 0
        and mean_r2 < 0.95
    ):
        return "PRIVILEGED_CROSS_FOUNDATION_CORRECTION_HAS_SIGNAL", mean_r2
    if abs(stats["mean_E1_minus_E2_G"]) < 0.5 and stats["mean_E1_minus_B0_G"] > 0:
        return "GENERIC_NONLINEAR_CORRECTION_HEAD_SIGNAL", mean_r2
    if abs(stats["mean_E1_minus_E3_G"]) < 0.5:
        return "PRIVILEGED_TEACHER_INFORMATION_NOT_SPECIFIC", mean_r2
    if stats["mean_E1_minus_B0_G"] > 0 and mean_r2 >= 0.95:
        return "PRIVILEGED_CORRECTION_COLLAPSES_TO_NEW_LINEAR_CONTRAST", mean_r2
    return "PRIVILEGED_CROSS_FOUNDATION_CORRECTION_NO_SIGNAL", mean_r2


def write_report(outcome, summary_rows, stats, mean_r2, runtime):
    by = {r["variant"]: r for r in summary_rows}
    lines = [
        "# Privileged Cross-Foundation Residual Distillation Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "SOURCE-ONLY / NO-TARGET. CLIP Symmetric HardNormal and RADIO StateAligned are frozen. Only a Linear(1024,16)-GELU-Linear(16,1) correction head is trained inside 12-category VisA LOO.",
        "",
        "| Variant | Mean P-AUROC | Mean P-AP | Mean G | Median G |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ["B0_CLIP_HardNormal", "RADIO_Teacher_Reference", "E1_PrivilegedCorrection", "E2_GTOnlyCorrection", "E3_RandomPrivilegedCorrection"]:
        r = by[variant]
        lines.append(f"| {variant} | {r['mean_P-AUROC']:.4f} | {r['mean_P-AP']:.4f} | {r['mean_G']:.4f} | {r['median_G']:.4f} |")
    lines.extend([
        "",
        "## Gate",
        f"- E1 improved over B0: {stats['E1_improved_over_B0']}/{stats['categories']}",
        f"- mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f}",
        f"- mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f}",
        f"- mean E1-E3 G: {stats['mean_E1_minus_E3_G']:+.4f}",
        f"- median E1-B0 G: {stats['median_E1_minus_B0_G']:+.4f}",
        f"- mean E1 delta linear-fit R2: {mean_r2:.6f}",
        "",
        "The correction target uses RADIO only when source GT CE shows RADIO beats CLIP on that patch. Target data are not used.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PrivCorrection] device={device}", flush=True)
    pool = build_clip_radio_pool(device, smoke=args.smoke)
    data = source_arrays(pool, args.smoke)
    teacher_rows = teacher_advantage_rows(data)
    raw_rows, diag_rows, curve_rows = run_loo(data, device, args.smoke)
    summary_rows, delta_rows, stats = summarize(raw_rows)
    outcome, mean_r2 = decide(stats, diag_rows)
    runtime = time.time() - start
    write_csv(OUT / "TEACHER_ADVANTAGE.csv", teacher_rows)
    write_csv(OUT / "LOO_METRICS.csv", raw_rows)
    write_csv(OUT / "SUMMARY_METRICS.csv", summary_rows)
    write_csv(OUT / "DELTA_METRICS.csv", delta_rows)
    write_csv(OUT / "CORRECTION_DIAGNOSTICS.csv", diag_rows)
    write_csv(OUT / "TRAINING_CURVES.csv", curve_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "mean_E1_delta_linear_R2": mean_r2,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_used": False,
        "frozen_clip_hardnormal": True,
        "frozen_radio_statealigned": True,
        "epochs": 2 if args.smoke else EPOCHS,
        "lr": LR,
        "hidden": HIDDEN,
    })
    write_report(outcome, summary_rows, stats, mean_r2, runtime)
    print(json.dumps({"outcome": outcome, **stats, "mean_E1_delta_linear_R2": mean_r2, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
