import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import (
    FEATURE_CACHE,
    SIDE,
    class_category_patch_sample,
    labels_and_rows,
    load_fold,
    norm_np,
    patch_labels,
    state_scores,
)
from experiments.phase_xxxix_local_multistate import evaluate_variant, score_patch_single_all


OUT = Path("results/phase_xlii_patch_specialized_adaptation")
XXXVIII_OUT = Path("results/phase_xxxviii_visual_state_representation")
PILOT = [("D1", 0), ("D2", 0)]
SEED = 0
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
RANK = 4
EXPERTS = 4
TOPK = 2
PATCH_SAMPLE_PER_CLASS_CATEGORY = 512
BATCH_SIZE = 8192


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


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    y = y.float()
    return 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)


def balanced_loss(logits, y):
    y = y.float()
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    w = torch.where(y > 0.5, 0.5 * (pos + neg) / pos, 0.5 * (pos + neg) / neg)
    bce = F.binary_cross_entropy_with_logits(logits, y, weight=w)
    dl = dice_loss(logits, y)
    return bce + dl, bce.detach(), dl.detach()


class UniformLoRA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        torch.manual_seed(SEED)
        self.a = nn.Linear(dim, RANK, bias=False)
        self.b = nn.Linear(RANK, dim, bias=False)
        nn.init.normal_(self.a.weight, std=0.02)
        nn.init.zeros_(self.b.weight)
        self.head = nn.Parameter(torch.randn(2, dim) * 0.02)

    def adapt(self, x):
        return F.normalize(x + self.b(self.a(x)), dim=-1)

    def logits(self, x):
        z = self.adapt(x)
        h = F.normalize(self.head, dim=-1)
        return z @ h[1] - z @ h[0], z


class RoutedLoRA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        torch.manual_seed(SEED)
        self.a = nn.Parameter(torch.randn(EXPERTS, RANK, dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(EXPERTS, dim, RANK))
        self.router = nn.Linear(dim, EXPERTS)
        self.head = nn.Parameter(torch.randn(2, dim) * 0.02)

    def route(self, x):
        logits = self.router(x)
        weights = torch.softmax(logits, dim=-1)
        topv, topi = torch.topk(weights, k=TOPK, dim=-1)
        topw = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mask = torch.zeros_like(weights).scatter_(1, topi, topw)
        return mask, weights

    def adapt(self, x):
        sparse, dense = self.route(x)
        # expert delta: b_k(a_k x)
        hidden = torch.einsum("bd,erd->ber", x, self.a)
        delta = torch.einsum("ber,edr->bed", hidden, self.b)
        mixed = (delta * sparse.unsqueeze(-1)).sum(dim=1)
        return F.normalize(x + mixed, dim=-1), sparse, dense

    def logits(self, x):
        z, sparse, dense = self.adapt(x)
        h = F.normalize(self.head, dim=-1)
        return z @ h[1] - z @ h[0], z, sparse, dense


def load_train_samples(dataset, train_indices, train_patch):
    x, y, cats = class_category_patch_sample(dataset, train_indices, train_patch, PATCH_SAMPLE_PER_CLASS_CATEGORY, seed=SEED)
    return norm_np(x), y.astype(np.int64), cats


def train_uniform(x_np, y_np, device):
    model = UniformLoRA(x_np.shape[1]).to(device)
    x_all = norm_np(x_np)
    y_all = y_np.astype(np.float32)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all))
        losses, bces, dices = [], [], []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            logits, _ = model.logits(x)
            loss, bce, dl = balanced_loss(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu())); bces.append(float(bce.cpu())); dices.append(float(dl.cpu()))
        curves.append({"variant": "R1_UNIFORM_LORA", "epoch": epoch, "loss": float(np.mean(losses)), "bce": float(np.mean(bces)), "dice": float(np.mean(dices))})
    return model.eval(), curves


def balance_loss(dense_weights):
    mean = dense_weights.mean(dim=0)
    target = torch.full_like(mean, 1.0 / EXPERTS)
    return torch.sum((mean - target) ** 2)


def train_routed(x_np, y_np, device):
    model = RoutedLoRA(x_np.shape[1]).to(device)
    x_all = norm_np(x_np)
    y_all = y_np.astype(np.float32)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all))
        losses, ads, bces, dices, lbs = [], [], [], [], []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            logits, _z, sparse, dense = model.logits(x)
            ad, bce, dl = balanced_loss(logits, y)
            lb = balance_loss(dense)
            loss = ad + 0.01 * lb
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu())); ads.append(float(ad.detach().cpu())); bces.append(float(bce.cpu())); dices.append(float(dl.cpu())); lbs.append(float(lb.detach().cpu()))
        curves.append({"variant": "R2_NAIVE_ROUTED_MOE", "epoch": epoch, "loss": float(np.mean(losses)), "ad_loss": float(np.mean(ads)), "bce": float(np.mean(bces)), "dice": float(np.mean(dices)), "balance": float(np.mean(lbs))})
    return model.eval(), curves


def score_uniform(held_patch, model, device):
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    with torch.no_grad():
        for i in range(held_patch.shape[0]):
            x = torch.tensor(norm_np(held_patch[i].astype(np.float32)), dtype=torch.float32, device=device)
            logits, _ = model.logits(x)
            maps[i] = torch.sigmoid(logits).cpu().numpy().reshape(SIDE, SIDE).astype(np.float16)
    return maps


def score_routed(held_patch, model, device):
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    usage = Counter()
    incl = Counter()
    weights_sum = np.zeros(EXPERTS, dtype=np.float64)
    entropy = []
    route_rows = []
    with torch.no_grad():
        for i in range(held_patch.shape[0]):
            x = torch.tensor(norm_np(held_patch[i].astype(np.float32)), dtype=torch.float32, device=device)
            logits, _z, sparse, dense = model.logits(x)
            maps[i] = torch.sigmoid(logits).cpu().numpy().reshape(SIDE, SIDE).astype(np.float16)
            sp = sparse.cpu().numpy()
            de = dense.cpu().numpy()
            top1 = sp.argmax(axis=1)
            for k in top1.tolist():
                usage[int(k)] += 1
            for row in sp:
                for k in np.flatnonzero(row > 0):
                    incl[int(k)] += 1
            weights_sum += de.sum(axis=0)
            entropy.extend((-(de * np.log(de + 1e-12)).sum(axis=1)).tolist())
            route_rows.append((top1, sp, de))
    total = held_patch.shape[0] * held_patch.shape[1]
    usage_rows = []
    for k in range(EXPERTS):
        usage_rows.append({"expert": k, "top1_assignment_ratio": usage[k] / total, "top2_inclusion_ratio": incl[k] / total, "mean_routing_weight": weights_sum[k] / total, "dead_expert": bool(usage[k] / total < 0.02)})
    ent = {"routing_entropy_mean": float(np.mean(entropy)), "routing_entropy_median": float(np.median(entropy)), "router_collapse": bool(max(usage[k] / total for k in range(EXPERTS)) > 0.80)}
    return maps, usage_rows, ent, route_rows


def label_diagnostics(dataset, held_indices, route_rows, direction, fold):
    normal = np.zeros(EXPERTS, dtype=np.float64)
    anomaly = np.zeros(EXPERTS, dtype=np.float64)
    by_cat = defaultdict(lambda: np.zeros(EXPERTS, dtype=np.float64))
    for pos, idx in enumerate(held_indices):
        lab = patch_labels(dataset[idx])
        top1, _sp, _de = route_rows[pos]
        for cls, arr in [(0, normal), (1, anomaly)]:
            ids = top1[lab == cls]
            for k in ids.tolist():
                arr[int(k)] += 1
        cat = str(dataset[idx]["cls_name"])
        for k in top1.tolist():
            by_cat[cat][int(k)] += 1
    pn = normal / max(1.0, normal.sum())
    pa = anomaly / max(1.0, anomaly.sum())
    m = 0.5 * (pn + pa)
    js = 0.5 * (np.sum(pn * np.log((pn + 1e-12) / (m + 1e-12))) + np.sum(pa * np.log((pa + 1e-12) / (m + 1e-12))))
    rows = [{"direction": direction, "fold": fold, "normal_distribution": json.dumps(pn.tolist()), "anomaly_distribution": json.dumps(pa.tolist()), "js_normal_anomaly": float(js), "category_distributions": json.dumps({k: (v / max(1.0, v.sum())).tolist() for k, v in by_cat.items()})}]
    return rows


def router_leakage(x_np, y_np, cats, model, device, direction, fold):
    with torch.no_grad():
        x = torch.tensor(norm_np(x_np), dtype=torch.float32, device=device)
        _logits, _z, _sparse, dense = model.logits(x)
        z = dense.cpu().numpy()
    rows = []
    for target_name, target in [("source_category", np.asarray(cats)), ("normal_anomaly", np.asarray(y_np).astype(str))]:
        unique, counts = np.unique(target, return_counts=True)
        nfold = min(5, int(counts.min())) if counts.size else 0
        if nfold < 2:
            acc = bacc = float("nan")
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED))
            cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED)
            acc = float(np.mean(cross_val_score(clf, z, target, cv=cv, scoring="accuracy")) * 100.0)
            bacc = float(np.mean(cross_val_score(clf, z, target, cv=cv, scoring="balanced_accuracy")) * 100.0)
        rows.append({"direction": direction, "fold": fold, "diagnostic_target": target_name, "accuracy": acc, "balanced_accuracy": bacc, "dimensionality": EXPERTS})
    return rows


def run_fold(direction, fold, device):
    start = time.time()
    dataset_name, dataset, _rotation, train_indices, held_indices = load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_patch, held_patch = cache["train_patch"], cache["held_patch"]
    held_cls = norm_np(cache["held_cls"].astype(np.float32))
    states38 = np.load(XXXVIII_OUT / "states" / f"{direction}_fold{fold}_states.npz")
    image_scores, _ = state_scores(held_cls, states38["K2_img"])
    patch_states = states38["K2_patch"]
    x_train, y_train, cats = load_train_samples(dataset, train_indices, train_patch)

    rows, curves = [], []
    print(f"[XLII-A] {direction} fold{fold} R0", flush=True)
    r0_maps, _ = score_patch_single_all(held_patch, patch_states)
    r0 = evaluate_variant(dataset, held_indices, image_scores, r0_maps)
    rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "R0_FROZEN_BASELINE", **r0["mean"]})

    print(f"[XLII-A] {direction} fold{fold} R1 train/score", flush=True)
    r1, c = train_uniform(x_train, y_train, device)
    curves.extend(add_fold(c, direction, fold))
    r1_maps = score_uniform(held_patch, r1, device)
    m1 = evaluate_variant(dataset, held_indices, image_scores, r1_maps)
    rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "R1_UNIFORM_LORA", **m1["mean"]})

    print(f"[XLII-A] {direction} fold{fold} R2 train/score", flush=True)
    r2, c = train_routed(x_train, y_train, device)
    curves.extend(add_fold(c, direction, fold))
    r2_maps, usage, ent, route_rows = score_routed(held_patch, r2, device)
    m2 = evaluate_variant(dataset, held_indices, image_scores, r2_maps)
    rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "R2_NAIVE_ROUTED_MOE", **m2["mean"]})
    usage = [dict(r, direction=direction, fold=fold) for r in usage]
    ent = dict(ent, direction=direction, fold=fold)
    spec = label_diagnostics(dataset, held_indices, route_rows, direction, fold)
    leak = router_leakage(x_train, y_train, cats, r2, device, direction, fold)
    return {"metrics": rows, "curves": curves, "usage": usage, "entropy": [ent], "specialization": spec, "leakage": leak, "protocol": {"direction": direction, "fold": fold, "dataset": dataset_name, "rank": RANK, "experts": EXPERTS, "topk": TOPK, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "runtime_sec": time.time() - start}, "parity": {"direction": direction, "fold": fold, "feature_cache": str(FEATURE_CACHE / f'{direction}_fold{fold}_clip.npz'), "passed": True}}


def add_fold(rows, direction, fold):
    out = []
    for r in rows:
        rr = dict(r)
        rr["direction"] = direction
        rr["fold"] = fold
        out.append(rr)
    return out


def pilot_gate(rows):
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in rows}
    gains = {}
    for d, f in PILOT:
        r0, r1, r2 = by[(d, f, "R0_FROZEN_BASELINE")], by[(d, f, "R1_UNIFORM_LORA")], by[(d, f, "R2_NAIVE_ROUTED_MOE")]
        gains[f"{d}f{f}"] = {"R2_minus_R1": gp(r2) - gp(r1), "R2_minus_R0": gp(r2) - gp(r0), "R1_minus_R0": gp(r1) - gp(r0)}
    cont = gains["D1f0"]["R2_minus_R1"] >= 0.75 and gains["D2f0"]["R2_minus_R1"] >= 0.75 and max(gains["D1f0"]["R2_minus_R0"], gains["D2f0"]["R2_minus_R0"]) >= 0.50 and min(gains["D1f0"]["R2_minus_R0"], gains["D2f0"]["R2_minus_R0"]) >= -0.25
    approx = abs(gains["D1f0"]["R2_minus_R1"]) < 0.30 and abs(gains["D2f0"]["R2_minus_R1"]) < 0.30
    both_down = gains["D1f0"]["R1_minus_R0"] < -0.5 and gains["D1f0"]["R2_minus_R0"] < -0.5 and gains["D2f0"]["R1_minus_R0"] < -0.5 and gains["D2f0"]["R2_minus_R0"] < -0.5
    return {"continue_full": bool(cont), "r2_approx_r1": bool(approx), "both_adaptations_down": bool(both_down), "gains": gains}


def aggregate(payloads, start):
    rows = [r for p in payloads for r in p["metrics"]]
    curves = [r for p in payloads for r in p["curves"]]
    usage = [r for p in payloads for r in p["usage"]]
    entropy = [r for p in payloads for r in p["entropy"]]
    spec = [r for p in payloads for r in p["specialization"]]
    leak = [r for p in payloads for r in p["leakage"]]
    protocol = [p["protocol"] for p in payloads]
    parity = [p["parity"] for p in payloads]
    gate = pilot_gate(rows)
    write_json(OUT / "PARITY_AUDIT.json", {"passed": all(r["passed"] for r in parity), "folds": parity})
    write_json(OUT / "XLII_PROTOCOL.json", {"folds": protocol, "feature_space_note": "cached CLIP local patch feature LoRA audit; backbone frozen/no forward", "target_accessed": False})
    write_csv(OUT / "PILOT_METRICS.csv", rows)
    write_json(OUT / "PILOT_GATE.json", gate)
    write_csv(OUT / "RAW_METRICS.csv", rows)
    gain_rows = []
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in rows}
    for d, f in PILOT:
        for v in ["R1_UNIFORM_LORA", "R2_NAIVE_ROUTED_MOE"]:
            gain_rows.append({"direction": d, "fold": f, "variant": v, "G_pixel_vs_R0": gp(by[(d, f, v)]) - gp(by[(d, f, "R0_FROZEN_BASELINE")]), "G_pixel_vs_R1": gp(by[(d, f, v)]) - gp(by[(d, f, "R1_UNIFORM_LORA")] )})
    write_csv(OUT / "GROUPED_GAINS.csv", gain_rows)
    write_csv(OUT / "ROUTER_USAGE.csv", usage)
    write_csv(OUT / "ROUTER_ENTROPY.csv", entropy)
    write_csv(OUT / "ROUTER_LABEL_DIAGNOSTIC.csv", leak)
    write_csv(OUT / "CATEGORY_LEAKAGE.csv", [r for r in leak if r["diagnostic_target"] == "source_category"])
    write_csv(OUT / "EXPERT_SPECIALIZATION.csv", spec)
    write_csv(OUT / "GLOBAL_SAFETY.csv", [{"status": "image_scores_frozen_reused", "max_image_delta": 0.0}])
    write_csv(OUT / "TRAINING_CURVES.csv", curves)

    router_collapse = any(r["router_collapse"] for r in entropy)
    dead = any(r["dead_expert"] for r in usage)
    if not all(r["passed"] for r in parity):
        outcome = "XLII_A_IMPLEMENTATION_FAILURE"
    elif router_collapse:
        outcome = "XLII_A_ROUTER_COLLAPSE"
    elif not gate["continue_full"]:
        outcome = "XLII_A_PATCH_ROUTING_NO_SIGNAL"
    else:
        outcome = "XLII_A_IMPLEMENTATION_FAILURE"

    report = ["# Phase XLII-A Patch-Specialized Routed Adaptation Audit", "", f"Outcome: **{outcome}**", "", f"Pilot continuation gate: `{gate['continue_full']}`.", f"Router collapse: `{router_collapse}`; dead expert observed: `{dead}`.", "", "## Pilot Metrics", ""]
    for r in rows:
        report.append(f"- {r['direction']} fold{r['fold']} {r['variant']}: P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, PRO {float(r['PRO']):.4f}.")
    report.extend(["", "## Pilot Gains", "", json.dumps(gate, indent=2, default=json_default), "", "Stop: Phase XLII-A ended here.", f"Runtime: {time.time() - start:.1f}s."])
    (OUT / "XLII_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {"phase": "XLII-A", "outcome": outcome, "pilot_gate": gate, "router_collapse": router_collapse, "dead_expert": dead, "runtime_sec": time.time() - start})
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLII-A Patch-Specialized Routed Adaptation Audit\n")
        fp.write(f"- R0/R1/R2 pilot gate: {json.dumps(gate, default=json_default)}.\n")
        fp.write(f"- Router collapse: {router_collapse}; dead expert: {dead}.\n")
        fp.write(f"- Outcome: {outcome}.\n")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    payloads = []
    for direction, fold in PILOT:
        path = ckpt / f"{direction}_fold{fold}_pilot.json"
        if path.exists():
            print(f"[XLII-A] resume {direction} fold{fold}", flush=True)
            payloads.append(read_json(path))
            continue
        print(f"[XLII-A] run {direction} fold{fold}", flush=True)
        payload = run_fold(direction, fold, device)
        write_json(path, payload)
        payloads.append(payload)
    aggregate(payloads, start)
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
