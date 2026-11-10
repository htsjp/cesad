import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    RANK,
    SEED,
    WEIGHT_DECAY,
    UniformLoRA,
    balanced_loss,
    load_train_samples,
    train_uniform,
)
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, load_fold, norm_np, state_scores
from experiments.phase_xxxix_local_multistate import evaluate_variant, score_patch_single_all


OUT = Path("results/phase_xliii_adapter_head_decomposition")
XXXVIII_OUT = Path("results/phase_xxxviii_visual_state_representation")
XLII_OUT = Path("results/phase_xlii_patch_specialized_adaptation")
PILOT = [("D1", 0), ("D2", 0)]


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
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gp(row):
    return 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"]))


class HeadOnly(nn.Module):
    def __init__(self, dim):
        super().__init__()
        torch.manual_seed(SEED)
        self.head = nn.Parameter(torch.randn(2, dim) * 0.02)

    def logits(self, x):
        z = F.normalize(x, dim=-1)
        h = F.normalize(self.head, dim=-1)
        return z @ h[1] - z @ h[0], z


class AdapterToState(nn.Module):
    def __init__(self, dim, states):
        super().__init__()
        torch.manual_seed(SEED)
        self.a = nn.Linear(dim, RANK, bias=False)
        self.b = nn.Linear(RANK, dim, bias=False)
        nn.init.normal_(self.a.weight, std=0.02)
        nn.init.zeros_(self.b.weight)
        self.register_buffer("states", torch.tensor(norm_np(states), dtype=torch.float32))

    def adapt(self, x):
        return F.normalize(x + self.b(self.a(x)), dim=-1)

    def logits(self, x):
        z = self.adapt(x)
        s = F.normalize(self.states, dim=-1)
        return z @ s[1] - z @ s[0], z


def train_head_only(x_np, y_np, device):
    model = HeadOnly(x_np.shape[1]).to(device)
    x_all = norm_np(x_np)
    y_all = y_np.astype(np.float32)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all))
        vals = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            logits, _ = model.logits(x)
            loss, bce, dice = balanced_loss(logits, y)
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(bce.cpu()), float(dice.cpu())))
        arr = np.asarray(vals)
        curves.append({"variant": "T1_HEAD_ONLY", "epoch": epoch, "loss": float(arr[:, 0].mean()), "bce": float(arr[:, 1].mean()), "dice": float(arr[:, 2].mean())})
    return model.eval(), curves


def train_adapter_to_state(x_np, y_np, states, device):
    model = AdapterToState(x_np.shape[1], states).to(device)
    x_all = norm_np(x_np)
    y_all = y_np.astype(np.float32)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all))
        vals = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            logits, _ = model.logits(x)
            loss, bce, dice = balanced_loss(logits, y)
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(bce.cpu()), float(dice.cpu())))
        arr = np.asarray(vals)
        curves.append({"variant": "T2_ADAPTER_ONLY_FROZEN_STATES", "epoch": epoch, "loss": float(arr[:, 0].mean()), "bce": float(arr[:, 1].mean()), "dice": float(arr[:, 2].mean())})
    return model.eval(), curves


def score_model(held_patch, model, device, states=None):
    maps = np.zeros((held_patch.shape[0], 37, 37), dtype=np.float16)
    cos_xz, margin_x, margin_z = [], [], []
    state_t = torch.tensor(norm_np(states), dtype=torch.float32, device=device) if states is not None else None
    with torch.no_grad():
        for i in range(held_patch.shape[0]):
            x = torch.tensor(norm_np(held_patch[i].astype(np.float32)), dtype=torch.float32, device=device)
            logits, z = model.logits(x)
            maps[i] = torch.sigmoid(logits).detach().cpu().numpy().reshape(37, 37).astype(np.float16)
            cos_xz.append((x * z).sum(dim=-1).detach().cpu().numpy())
            if state_t is not None:
                margin_x.append((x @ state_t[1] - x @ state_t[0]).detach().cpu().numpy())
                margin_z.append((z @ state_t[1] - z @ state_t[0]).detach().cpu().numpy())
    return maps, np.concatenate(cos_xz), (np.concatenate(margin_x) if margin_x else None), (np.concatenate(margin_z) if margin_z else None)


def head_geometry(model, states, variant, direction, fold):
    if not hasattr(model, "head"):
        return []
    h = norm_np(model.head.detach().cpu().numpy())
    s = norm_np(states)
    return [{
        "direction": direction,
        "fold": fold,
        "variant": variant,
        "cos_wN_wA": float(h[0] @ h[1]),
        "cos_wN_sN": float(h[0] @ s[0]),
        "cos_wN_sA": float(h[0] @ s[1]),
        "cos_wA_sN": float(h[1] @ s[0]),
        "cos_wA_sA": float(h[1] @ s[1]),
    }]


def geom_rows(direction, fold, variant, cos_xz):
    return [{"direction": direction, "fold": fold, "variant": variant, "mean_cos_x_z": float(np.mean(cos_xz)), "median_cos_x_z": float(np.median(cos_xz)), "q10_cos_x_z": float(np.quantile(cos_xz, 0.10)), "q90_cos_x_z": float(np.quantile(cos_xz, 0.90))}]


def margin_rows(direction, fold, variant, mx, mz):
    if mx is None or mz is None:
        return []
    shift = mz - mx
    return [{"direction": direction, "fold": fold, "variant": variant, "mean_margin_x": float(np.mean(mx)), "mean_margin_z": float(np.mean(mz)), "mean_margin_shift": float(np.mean(shift)), "median_margin_shift": float(np.median(shift)), "q10_shift": float(np.quantile(shift, 0.10)), "q90_shift": float(np.quantile(shift, 0.90))}]


def add_fold(rows, direction, fold):
    out = []
    for r in rows:
        rr = dict(r)
        rr["direction"] = direction
        rr["fold"] = fold
        out.append(rr)
    return out


def run_fold(direction, fold, device):
    start = time.time()
    dataset_name, dataset, _rot, train_indices, held_indices = load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_patch, held_patch = cache["train_patch"], cache["held_patch"]
    states38 = np.load(XXXVIII_OUT / "states" / f"{direction}_fold{fold}_states.npz")
    patch_states = states38["K2_patch"]
    zeros = np.zeros((len(held_indices),), dtype=np.float32)
    x_train, y_train, _cats = load_train_samples(dataset, train_indices, train_patch)

    rows, curves, feat_geom, margin_shift, head_geom = [], [], [], [], []
    print(f"[XLIII-A] {direction} fold{fold} T0", flush=True)
    t0_maps, _ = score_patch_single_all(held_patch, patch_states)
    t0 = evaluate_variant(dataset, held_indices, zeros, t0_maps)
    rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "T0_SINGLE_STATE", **{k: t0["mean"].get(k, float("nan")) for k in ["P-AUROC", "P-AP", "PRO"]}})

    print(f"[XLIII-A] {direction} fold{fold} T1 train/score", flush=True)
    t1_model, c = train_head_only(x_train, y_train, device)
    curves.extend(add_fold(c, direction, fold))
    maps, cosxz, mx, mz = score_model(held_patch, t1_model, device, states=patch_states)
    t1 = evaluate_variant(dataset, held_indices, zeros, maps)
    rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "T1_HEAD_ONLY", **{k: t1["mean"].get(k, float("nan")) for k in ["P-AUROC", "P-AP", "PRO"]}})
    feat_geom.extend(geom_rows(direction, fold, "T1_HEAD_ONLY", cosxz))
    head_geom.extend(head_geometry(t1_model, patch_states, "T1_HEAD_ONLY", direction, fold))

    print(f"[XLIII-A] {direction} fold{fold} T2 train/score", flush=True)
    t2_model, c = train_adapter_to_state(x_train, y_train, patch_states, device)
    curves.extend(add_fold(c, direction, fold))
    maps, cosxz, mx, mz = score_model(held_patch, t2_model, device, states=patch_states)
    t2 = evaluate_variant(dataset, held_indices, zeros, maps)
    rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "T2_ADAPTER_ONLY_FROZEN_STATES", **{k: t2["mean"].get(k, float("nan")) for k in ["P-AUROC", "P-AP", "PRO"]}})
    feat_geom.extend(geom_rows(direction, fold, "T2_ADAPTER_ONLY_FROZEN_STATES", cosxz))
    margin_shift.extend(margin_rows(direction, fold, "T2_ADAPTER_ONLY_FROZEN_STATES", mx, mz))

    print(f"[XLIII-A] {direction} fold{fold} T3 train/score", flush=True)
    t3_model, c = train_uniform(x_train, y_train, device)
    curves.extend(add_fold([dict(r, variant="T3_ADAPTER_PLUS_HEAD") for r in c], direction, fold))
    maps, cosxz, mx, mz = score_model(held_patch, t3_model, device, states=patch_states)
    t3 = evaluate_variant(dataset, held_indices, zeros, maps)
    rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "T3_ADAPTER_PLUS_HEAD", **{k: t3["mean"].get(k, float("nan")) for k in ["P-AUROC", "P-AP", "PRO"]}})
    feat_geom.extend(geom_rows(direction, fold, "T3_ADAPTER_PLUS_HEAD", cosxz))
    margin_shift.extend(margin_rows(direction, fold, "T3_ADAPTER_PLUS_HEAD", mx, mz))
    head_geom.extend(head_geometry(t3_model, patch_states, "T3_ADAPTER_PLUS_HEAD", direction, fold))

    return {"metrics": rows, "curves": curves, "feature_geometry": feat_geom, "margin_shift": margin_shift, "head_geometry": head_geom, "protocol": {"direction": direction, "fold": fold, "dataset": dataset_name, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "rank": RANK, "seed": SEED, "runtime_sec": time.time() - start}}


def aggregate(payloads, start):
    metrics = [r for p in payloads for r in p["metrics"]]
    curves = [r for p in payloads for r in p["curves"]]
    feature = [r for p in payloads for r in p["feature_geometry"]]
    margins = [r for p in payloads for r in p["margin_shift"]]
    heads = [r for p in payloads for r in p["head_geometry"]]
    protocols = [p["protocol"] for p in payloads]
    write_json(OUT / "PARITY_AUDIT.json", {"T3_recomputed_from_same_seed": True, "backbone_forward": False, "folds": PILOT})
    write_json(OUT / "XLIII_PROTOCOL.json", {"folds": protocols, "source": "cached frozen patch features", "target_accessed": False})
    write_csv(OUT / "RAW_METRICS.csv", metrics)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)
    write_csv(OUT / "FEATURE_GEOMETRY.csv", feature)
    write_csv(OUT / "STATE_MARGIN_SHIFT.csv", margins)
    write_csv(OUT / "HEAD_GEOMETRY.csv", heads)
    write_csv(OUT / "TRAINABLE_PARAMETER_AUDIT.csv", [
        {"variant": "T0_SINGLE_STATE", "trainable_parameters": 0, "adapter_parameters": 0, "head_parameters": 0},
        {"variant": "T1_HEAD_ONLY", "trainable_parameters": 2048, "adapter_parameters": 0, "head_parameters": 2048},
        {"variant": "T2_ADAPTER_ONLY_FROZEN_STATES", "trainable_parameters": 8192, "adapter_parameters": 8192, "head_parameters": 0},
        {"variant": "T3_ADAPTER_PLUS_HEAD", "trainable_parameters": 10240, "adapter_parameters": 8192, "head_parameters": 2048},
    ])

    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in metrics}
    gains, recovery = [], []
    t3_repro_ok = True
    xlii_rows = {}
    import csv
    with open(XLII_OUT / "PILOT_METRICS.csv", newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            if r["variant"] == "R1_UNIFORM_LORA":
                xlii_rows[(r["direction"], int(r["fold"]))] = r
    for d, f in PILOT:
        t0 = by[(d, f, "T0_SINGLE_STATE")]
        t1 = by[(d, f, "T1_HEAD_ONLY")]
        t2 = by[(d, f, "T2_ADAPTER_ONLY_FROZEN_STATES")]
        t3 = by[(d, f, "T3_ADAPTER_PLUS_HEAD")]
        full = gp(t3) - gp(t0)
        hr = (gp(t1) - gp(t0)) / full if full > 0 else float("nan")
        ar = (gp(t2) - gp(t0)) / full if full > 0 else float("nan")
        for v, r in [("T1_HEAD_ONLY", t1), ("T2_ADAPTER_ONLY_FROZEN_STATES", t2), ("T3_ADAPTER_PLUS_HEAD", t3)]:
            gains.append({"direction": d, "fold": f, "variant": v, "G_pixel_vs_T0": gp(r) - gp(t0), "G_pixel_vs_T1": gp(r) - gp(t1), "G_pixel_vs_T2": gp(r) - gp(t2), "G_pixel_vs_T3": gp(r) - gp(t3)})
        recovery.append({"direction": d, "fold": f, "Gain_full_T3_minus_T0": full, "HeadRecovery": hr, "AdapterStateRecovery": ar})
        ref = xlii_rows[(d, f)]
        if abs(float(ref["P-AUROC"]) - float(t3["P-AUROC"])) > 1e-4 or abs(float(ref["P-AP"]) - float(t3["P-AP"])) > 1e-4:
            t3_repro_ok = False
    write_csv(OUT / "CAUSAL_GAINS.csv", gains)
    write_csv(OUT / "GAIN_RECOVERY.csv", recovery)

    d1_t1 = next(g["G_pixel_vs_T0"] for g in gains if g["direction"] == "D1" and g["variant"] == "T1_HEAD_ONLY")
    d2_t1 = next(g["G_pixel_vs_T0"] for g in gains if g["direction"] == "D2" and g["variant"] == "T1_HEAD_ONLY")
    d1_t2 = next(g["G_pixel_vs_T0"] for g in gains if g["direction"] == "D1" and g["variant"] == "T2_ADAPTER_ONLY_FROZEN_STATES")
    d2_t2 = next(g["G_pixel_vs_T0"] for g in gains if g["direction"] == "D2" and g["variant"] == "T2_ADAPTER_ONLY_FROZEN_STATES")
    d1_t3_t1 = next(g["G_pixel_vs_T1"] for g in gains if g["direction"] == "D1" and g["variant"] == "T3_ADAPTER_PLUS_HEAD")
    d2_t3_t1 = next(g["G_pixel_vs_T1"] for g in gains if g["direction"] == "D2" and g["variant"] == "T3_ADAPTER_PLUS_HEAD")
    d1_t2_t1 = next(g["G_pixel_vs_T1"] for g in gains if g["direction"] == "D1" and g["variant"] == "T2_ADAPTER_ONLY_FROZEN_STATES")
    d2_t2_t1 = next(g["G_pixel_vs_T1"] for g in gains if g["direction"] == "D2" and g["variant"] == "T2_ADAPTER_ONLY_FROZEN_STATES")
    rec_by = {(r["direction"], r["fold"]): r for r in recovery}
    head_dom = d1_t1 >= 1 and d2_t1 >= 1 and rec_by[("D1", 0)]["HeadRecovery"] >= 0.75 and rec_by[("D2", 0)]["HeadRecovery"] >= 0.75 and d1_t3_t1 < 0.5 and d2_t3_t1 < 0.5
    adapter_state = d1_t2 >= 1 and d2_t2 >= 1 and rec_by[("D1", 0)]["AdapterStateRecovery"] >= 0.5 and rec_by[("D2", 0)]["AdapterStateRecovery"] >= 0.5 and d1_t2_t1 >= 0.5 and d2_t2_t1 >= 0.5
    synergy = d1_t1 >= 1 and d2_t1 >= 1 and d1_t2 >= 1 and d2_t2 >= 1 and min(d1_t3_t1, d2_t3_t1) >= 0.75 and min(-next(g["G_pixel_vs_T2"] for g in gains if g["direction"] == "D1" and g["variant"] == "T3_ADAPTER_PLUS_HEAD"), -next(g["G_pixel_vs_T2"] for g in gains if g["direction"] == "D2" and g["variant"] == "T3_ADAPTER_PLUS_HEAD")) >= 0.75
    if not t3_repro_ok:
        outcome = "XLIII_A_R1_GAIN_NOT_REPRODUCIBLE"
    elif head_dom:
        outcome = "XLIII_A_BINARY_HEAD_DOMINATES"
    elif adapter_state:
        outcome = "XLIII_A_ADAPTER_TO_STATE_SUPPORTED"
    elif synergy:
        outcome = "XLIII_A_ADAPTER_HEAD_SYNERGY"
    else:
        outcome = "XLIII_A_DISCRIMINATIVE_SCORER_SIGNAL_MECHANISM_MIXED"

    report = ["# Phase XLIII-A Adapter/Head Causal Decomposition", "", f"Outcome: **{outcome}**", "", f"T3 reproduces XLII R1: `{t3_repro_ok}`.", "", "## Pilot Metrics", ""]
    for r in metrics:
        report.append(f"- {r['direction']} fold{r['fold']} {r['variant']}: P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, PRO {float(r['PRO']):.4f}.")
    report.extend(["", "## Causal Gains", ""])
    for r in recovery:
        report.append(f"- {r['direction']} fold{r['fold']}: full gain {r['Gain_full_T3_minus_T0']:+.4f}, HeadRecovery {r['HeadRecovery']:.4f}, AdapterStateRecovery {r['AdapterStateRecovery']:.4f}.")
    report.extend(["", "Stop: Phase XLIII-A ended here.", f"Runtime: {time.time() - start:.1f}s."])
    (OUT / "XLIII_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {"phase": "XLIII-A", "outcome": outcome, "t3_reproducible": t3_repro_ok, "head_dominant": head_dom, "adapter_to_state_supported": adapter_state, "synergy": synergy, "runtime_sec": time.time() - start})
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLIII-A Adapter/Head Causal Decomposition\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- T3 reproducible: {t3_repro_ok}.\n")
        fp.write(f"- Recovery: {json.dumps(recovery, default=json_default)}.\n")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    payloads = []
    for direction, fold in PILOT:
        path = ckpt / f"{direction}_fold{fold}.json"
        if path.exists():
            print(f"[XLIII-A] resume {direction} fold{fold}", flush=True)
            payloads.append(read_json(path))
            continue
        payload = run_fold(direction, fold, device)
        write_json(path, payload)
        payloads.append(payload)
    aggregate(payloads, start)
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
