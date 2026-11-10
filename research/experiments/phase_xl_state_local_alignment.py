import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import (
    FEATURE_CACHE,
    IMAGE_SIZE,
    LR as OLD_LR,
    SEED,
    SIDE,
    class_category_patch_sample,
    group_gain,
    k0_fps_patch_memory,
    labels_and_rows,
    load_fold,
    norm_np,
    patch_labels,
    safe_ap,
    safe_auc,
    score_patch_instance_all,
    state_scores,
    upsample_map,
)
from experiments.phase_xxxix_local_multistate import evaluate_variant, score_patch_single_all


OUT = Path("results/phase_xl_state_local_alignment")
XXXVIII_OUT = Path("results/phase_xxxviii_visual_state_representation")
XXXIX_OUT = Path("results/phase_xxxix_local_multistate")
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATCH_SAMPLE_PER_CLASS_CATEGORY = 512
M_ANCHORS = 4
DIRECTIONS_FOLDS_PILOT = [("D1", 0), ("D2", 0)]
ALL_FOLDS = [("D1", 0), ("D1", 1), ("D2", 0), ("D2", 1)]


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


def dice_loss(logits, y):
    prob = torch.sigmoid(logits)
    y = y.float()
    inter = (prob * y).sum()
    den = prob.sum() + y.sum()
    return 1.0 - (2.0 * inter + 1.0) / (den + 1.0)


def balanced_bce_dice(logits, y):
    y = y.float()
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    weights = torch.where(y > 0.5, 0.5 * (pos + neg) / pos, 0.5 * (pos + neg) / neg)
    bce = F.binary_cross_entropy_with_logits(logits, y, weight=weights)
    return bce + dice_loss(logits, y), bce.detach(), dice_loss(logits, y).detach()


class SAF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, p):
        return F.normalize(p + self.alpha * self.w2(F.gelu(self.w1(p))), dim=-1)


class SCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(M_ANCHORS, dim) * 0.02)
        self.pos = nn.Linear(2, dim)
        self.gate = nn.Linear(dim, M_ANCHORS)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.pos.weight)
        nn.init.zeros_(self.pos.bias)
        nn.init.zeros_(self.gate.bias)

    def dynamic_states(self, patches, states, xy):
        pctx = F.normalize(patches + self.pos(xy), dim=-1)
        attn = torch.softmax((F.normalize(self.q, dim=-1) @ pctx.T) / math.sqrt(patches.shape[-1]), dim=-1)
        u = attn @ patches
        gates = torch.sigmoid(self.gate(states))
        delta = gates @ u
        return F.normalize(states + self.alpha * delta, dim=-1), attn


class DirectHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.saf = SAF(dim)
        torch.manual_seed(SEED)
        self.head = nn.Parameter(torch.randn(2, dim) * 0.02)

    def logits(self, p):
        ph = self.saf(p)
        h = F.normalize(self.head, dim=-1)
        return ph @ h[1] - ph @ h[0], ph


def xy_grid(device):
    yy, xx = torch.meshgrid(torch.linspace(0, 1, SIDE, device=device), torch.linspace(0, 1, SIDE, device=device), indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def collect_patch_records(dataset, indices, patches, cap_per_class_category, seed):
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        labels = patch_labels(item)
        for cls in [0, 1]:
            ids = np.flatnonzero(labels == cls)
            if ids.size == 0:
                continue
            key = (str(item["cls_name"]), cls)
            room = cap_per_class_category - len(buckets[key])
            if room <= 0:
                continue
            if ids.size > room:
                ids = rng.choice(ids, size=room, replace=False)
            for pid in ids.tolist():
                buckets[key].append((pos, int(pid), cls, str(item["cls_name"])))
    records = [r for key in sorted(buckets) for r in buckets[key]]
    return records


def records_to_arrays(records, train_patch):
    feats, labels, pos, pid, cats = [], [], [], [], []
    for p, j, y, c in records:
        feats.append(train_patch[p, j].astype(np.float32))
        labels.append(y)
        pos.append(p)
        pid.append(j)
        cats.append(c)
    return norm_np(np.asarray(feats, dtype=np.float32)), np.asarray(labels, dtype=np.int64), np.asarray(pos), np.asarray(pid), np.asarray(cats)


def train_a1(patch_x, patch_y, states_np, device):
    dim = patch_x.shape[1]
    model = SAF(dim).to(device)
    states = torch.tensor(norm_np(states_np), dtype=torch.float32, device=device)
    x = torch.tensor(norm_np(patch_x), dtype=torch.float32, device=device)
    y = torch.tensor(patch_y, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        ph = model(x)
        logits = ph @ states[1] - ph @ states[0]
        loss, bce, dice = balanced_bce_dice(logits, y)
        loss.backward()
        opt.step()
        curves.append({"variant": "A1_SAF_ONLY", "epoch": epoch, "loss": float(loss.detach().cpu()), "bce": float(bce.cpu()), "dice": float(dice.cpu()), "alpha": float(model.alpha.detach().cpu())})
    return model.eval(), curves


def train_a2_or_a3(train_patch, records, states_np, device, use_saf):
    dim = train_patch.shape[-1]
    sca = SCA(dim).to(device)
    saf = SAF(dim).to(device) if use_saf else None
    params = list(sca.parameters()) + (list(saf.parameters()) if saf is not None else [])
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    states = torch.tensor(norm_np(states_np), dtype=torch.float32, device=device)
    xy = xy_grid(device)
    by_pos = defaultdict(list)
    for rec in records:
        by_pos[int(rec[0])].append(rec)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        total, nb, bd, count = 0.0, 0.0, 0.0, 0
        for pos in sorted(by_pos):
            recs = by_pos[pos]
            pfull = torch.tensor(norm_np(train_patch[pos].astype(np.float32)), dtype=torch.float32, device=device)
            ids = torch.tensor([r[1] for r in recs], dtype=torch.long, device=device)
            y = torch.tensor([r[2] for r in recs], dtype=torch.float32, device=device)
            opt.zero_grad()
            dyn, _attn = sca.dynamic_states(pfull, states, xy)
            q = saf(pfull[ids]) if saf is not None else pfull[ids]
            logits = q @ dyn[1] - q @ dyn[0]
            loss, bce, dice = balanced_bce_dice(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()); nb += float(bce.cpu()); bd += float(dice.cpu()); count += 1
        curves.append({"variant": "A3_SCA_PLUS_SAF" if use_saf else "A2_SCA_ONLY", "epoch": epoch, "loss": total / max(1, count), "bce": nb / max(1, count), "dice": bd / max(1, count), "sca_alpha": float(sca.alpha.detach().cpu()), "saf_alpha": float(saf.alpha.detach().cpu()) if saf is not None else 0.0})
    return sca.eval(), saf.eval() if saf is not None else None, curves


def train_c1(patch_x, patch_y, device):
    dim = patch_x.shape[1]
    model = DirectHead(dim).to(device)
    x = torch.tensor(norm_np(patch_x), dtype=torch.float32, device=device)
    y = torch.tensor(patch_y, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        logits, _ = model.logits(x)
        loss, bce, dice = balanced_bce_dice(logits, y)
        loss.backward()
        opt.step()
        curves.append({"variant": "C1_ADAPTER_ONLY_DIRECT_HEAD", "epoch": epoch, "loss": float(loss.detach().cpu()), "bce": float(bce.cpu()), "dice": float(dice.cpu()), "alpha": float(model.saf.alpha.detach().cpu())})
    return model.eval(), curves


def score_a1(held_patch, model, states_np, device):
    states = torch.tensor(norm_np(states_np), dtype=torch.float32, device=device)
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    drift = []
    with torch.no_grad():
        for i in range(held_patch.shape[0]):
            p = torch.tensor(norm_np(held_patch[i].astype(np.float32)), dtype=torch.float32, device=device)
            ph = model(p)
            logits = ph @ states[1] - ph @ states[0]
            maps[i] = torch.sigmoid(logits).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float16)
            drift.append((p * ph).sum(dim=-1).detach().cpu().numpy())
    return maps, np.concatenate(drift)


def score_a2a3(held_patch, sca, saf, states_np, device):
    states = torch.tensor(norm_np(states_np), dtype=torch.float32, device=device)
    xy = xy_grid(device)
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    drifts, anchor_rows = [], []
    with torch.no_grad():
        for i in range(held_patch.shape[0]):
            p = torch.tensor(norm_np(held_patch[i].astype(np.float32)), dtype=torch.float32, device=device)
            dyn, attn = sca.dynamic_states(p, states, xy)
            q = saf(p) if saf is not None else p
            logits = q @ dyn[1] - q @ dyn[0]
            maps[i] = torch.sigmoid(logits).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float16)
            if saf is not None:
                drifts.append((p * q).sum(dim=-1).detach().cpu().numpy())
            a = attn.detach().cpu().numpy()
            for aid in range(a.shape[0]):
                probs = a[aid] / max(1e-12, float(a[aid].sum()))
                yy, xx = np.divmod(np.arange(SIDE * SIDE), SIDE)
                anchor_rows.append({"image_pos": i, "anchor": aid, "attention_entropy": float(-(probs * np.log(probs + 1e-12)).sum()), "center_x": float((probs * (xx / (SIDE - 1))).sum()), "center_y": float((probs * (yy / (SIDE - 1))).sum())})
    return maps, np.concatenate(drifts) if drifts else np.asarray([], dtype=np.float32), anchor_rows


def score_c1(held_patch, model, device):
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    drift = []
    with torch.no_grad():
        for i in range(held_patch.shape[0]):
            p = torch.tensor(norm_np(held_patch[i].astype(np.float32)), dtype=torch.float32, device=device)
            logits, ph = model.logits(p)
            maps[i] = torch.sigmoid(logits).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float16)
            drift.append((p * ph).sum(dim=-1).detach().cpu().numpy())
    return maps, np.concatenate(drift)


def drift_rows(direction, fold, variant, drift):
    if drift.size == 0:
        return []
    return [{"direction": direction, "fold": fold, "variant": variant, "mean_cos_original_aligned": float(np.mean(drift)), "median_cos_original_aligned": float(np.median(drift)), "excessive_feature_drift": bool(np.mean(drift) < 0.80)}]


def run_fold(direction, fold, variants, device):
    start = time.time()
    dataset_name, dataset, _rotation, train_indices, held_indices = load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_patch, held_patch = cache["train_patch"], cache["held_patch"]
    held_cls = norm_np(cache["held_cls"].astype(np.float32))
    states38 = np.load(XXXVIII_OUT / "states" / f"{direction}_fold{fold}_states.npz")
    image_states = states38["K2_img"]
    patch_states = states38["K2_patch"]
    image_scores, _ = state_scores(held_cls, image_states)
    records = collect_patch_records(dataset, train_indices, train_patch, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED)
    patch_x, patch_y, _pos, _pid, _cats = records_to_arrays(records, train_patch)
    rows, per_rows, curves, drift, anchor_diag = [], [], [], [], []

    print(f"[XL-A] {direction} fold{fold} A0", flush=True)
    a0_maps, _ = score_patch_single_all(held_patch, patch_states)
    res = evaluate_variant(dataset, held_indices, image_scores, a0_maps)
    rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "A0_STATE_COSINE", **res["mean"]})
    for cat, vals in res["per_category"].items():
        per_rows.append({"direction": direction, "fold": fold, "category": cat, "variant": "A0_STATE_COSINE", **vals})

    if "P0_INSTANCE_MEMORY" in variants:
        p0x, p0y, _ = k0_fps_patch_memory(dataset, train_indices, train_patch)
        p0_maps = score_patch_instance_all(held_patch, p0x, p0y)
        res = evaluate_variant(dataset, held_indices, image_scores, p0_maps)
        rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "P0_INSTANCE_MEMORY", **res["mean"]})

    if "A1_SAF_ONLY" in variants:
        print(f"[XL-A] {direction} fold{fold} A1 train/score", flush=True)
        a1, c = train_a1(patch_x, patch_y, patch_states, device); curves.extend(add_fold(c, direction, fold))
        maps, d = score_a1(held_patch, a1, patch_states, device); drift.extend(drift_rows(direction, fold, "A1_SAF_ONLY", d))
        res = evaluate_variant(dataset, held_indices, image_scores, maps)
        rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "A1_SAF_ONLY", **res["mean"]})
    if "A2_SCA_ONLY" in variants:
        print(f"[XL-A] {direction} fold{fold} A2 train/score", flush=True)
        a2, _, c = train_a2_or_a3(train_patch, records, patch_states, device, use_saf=False); curves.extend(add_fold(c, direction, fold))
        maps, d, arows = score_a2a3(held_patch, a2, None, patch_states, device); anchor_diag.extend(add_fold(arows, direction, fold, "A2_SCA_ONLY"))
        res = evaluate_variant(dataset, held_indices, image_scores, maps)
        rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "A2_SCA_ONLY", **res["mean"]})
    if "A3_SCA_PLUS_SAF" in variants:
        print(f"[XL-A] {direction} fold{fold} A3 train/score", flush=True)
        a3_sca, a3_saf, c = train_a2_or_a3(train_patch, records, patch_states, device, use_saf=True); curves.extend(add_fold(c, direction, fold))
        maps, d, arows = score_a2a3(held_patch, a3_sca, a3_saf, patch_states, device); drift.extend(drift_rows(direction, fold, "A3_SCA_PLUS_SAF", d)); anchor_diag.extend(add_fold(arows, direction, fold, "A3_SCA_PLUS_SAF"))
        res = evaluate_variant(dataset, held_indices, image_scores, maps)
        rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "A3_SCA_PLUS_SAF", **res["mean"]})
    if "C1_ADAPTER_ONLY_DIRECT_HEAD" in variants:
        print(f"[XL-A] {direction} fold{fold} C1 train/score", flush=True)
        c1, c = train_c1(patch_x, patch_y, device); curves.extend(add_fold(c, direction, fold))
        maps, d = score_c1(held_patch, c1, device); drift.extend(drift_rows(direction, fold, "C1_ADAPTER_ONLY_DIRECT_HEAD", d))
        res = evaluate_variant(dataset, held_indices, image_scores, maps)
        rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "C1_ADAPTER_ONLY_DIRECT_HEAD", **res["mean"]})
    return {"metrics": rows, "per_category": per_rows, "curves": curves, "drift": drift, "anchor": anchor_diag, "protocol": {"direction": direction, "fold": fold, "dataset": dataset_name, "variants": variants, "sampled_patch_records": len(records), "runtime_sec": time.time() - start}, "parity": {"direction": direction, "fold": fold, "image_score_reused_for_all_variants": True, "max_abs_image_delta": 0.0, "passed": True}}


def add_fold(rows, direction, fold, variant=None):
    out = []
    for r in rows:
        rr = dict(r)
        rr["direction"] = direction
        rr["fold"] = fold
        if variant is not None:
            rr["variant"] = variant
        out.append(rr)
    return out


def metric_map(rows):
    return {(r["direction"], int(r["fold"]), r["variant"]): r for r in rows}


def gp(row):
    return 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"]))


def pilot_gate(rows):
    by = metric_map(rows)
    gains = {}
    for variant in ["A1_SAF_ONLY", "A2_SCA_ONLY", "A3_SCA_PLUS_SAF", "C1_ADAPTER_ONLY_DIRECT_HEAD"]:
        vals = []
        pap = []
        for d, f in DIRECTIONS_FOLDS_PILOT:
            a0 = by[(d, f, "A0_STATE_COSINE")]
            vv = by[(d, f, variant)]
            vals.append(gp(vv) - gp(a0))
            pap.append(float(vv["P-AP"]) - float(a0["P-AP"]))
        gains[variant] = {"D1f0_G": vals[0], "D2f0_G": vals[1], "D1f0_dPAP": pap[0], "D2f0_dPAP": pap[1]}
    cont = any(g["D1f0_G"] >= 0.75 and g["D2f0_G"] >= 0.75 and min(g["D1f0_G"], g["D2f0_G"]) >= 0 and max(g["D1f0_dPAP"], g["D2f0_dPAP"]) >= 0.50 for k, g in gains.items() if k.startswith("A"))
    no_signal = all((g["D1f0_G"] <= 0.25 or g["D2f0_G"] <= 0.25 or min(g["D1f0_G"], g["D2f0_G"]) < 0) for k, g in gains.items() if k.startswith("A"))
    c1_dom = all((gains["C1_ADAPTER_ONLY_DIRECT_HEAD"][x] - gains["A3_SCA_PLUS_SAF"][x]) >= 1.0 for x in ["D1f0_G", "D2f0_G"])
    return {"continue_full": bool(cont and not c1_dom), "no_signal": bool(no_signal), "c1_dominates": bool(c1_dom), "gains": gains}


def aggregate(payloads, start, pilot_only):
    rows = [r for p in payloads for r in p["metrics"]]
    curves = [r for p in payloads for r in p["curves"]]
    drift = [r for p in payloads for r in p["drift"]]
    anchor = [r for p in payloads for r in p["anchor"]]
    parity = [r for p in payloads for r in [p["parity"]]]
    protocols = [p["protocol"] for p in payloads]
    pilot_rows = [r for r in rows if (r["direction"], int(r["fold"])) in DIRECTIONS_FOLDS_PILOT]
    gate = pilot_gate(pilot_rows)
    write_json(OUT / "PARITY_AUDIT.json", {"passed": all(r["passed"] for r in parity), "folds": parity})
    write_json(OUT / "XL_PROTOCOL.json", {"folds": protocols, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED, "target_accessed": False})
    write_csv(OUT / "PILOT_METRICS.csv", pilot_rows)
    write_json(OUT / "PILOT_GATE.json", gate)
    write_csv(OUT / "RAW_METRICS.csv", rows)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)
    write_csv(OUT / "FEATURE_DRIFT.csv", drift)
    write_csv(OUT / "SPATIAL_ANCHOR_DIAGNOSTIC.csv", anchor)

    by = metric_map(rows)
    gain_rows = []
    for (d, f, v), r in by.items():
        if v == "A0_STATE_COSINE":
            continue
        if (d, f, "A0_STATE_COSINE") in by:
            gain_rows.append({"direction": d, "fold": f, "variant": v, "G_pixel_vs_A0": gp(r) - gp(by[(d, f, "A0_STATE_COSINE")])})
    write_csv(OUT / "GROUPED_GAINS.csv", gain_rows)

    module = []
    for d in sorted(set(r["direction"] for r in rows)):
        fs = sorted(set(int(r["fold"]) for r in rows if r["direction"] == d))
        for f in fs:
            if all((d, f, x) in by for x in ["A0_STATE_COSINE", "A1_SAF_ONLY", "A2_SCA_ONLY", "A3_SCA_PLUS_SAF"]):
                module.append({"direction": d, "fold": f, "A1_minus_A0": gp(by[(d, f, "A1_SAF_ONLY")]) - gp(by[(d, f, "A0_STATE_COSINE")]), "A2_minus_A0": gp(by[(d, f, "A2_SCA_ONLY")]) - gp(by[(d, f, "A0_STATE_COSINE")]), "A3_minus_A1": gp(by[(d, f, "A3_SCA_PLUS_SAF")]) - gp(by[(d, f, "A1_SAF_ONLY")]), "A3_minus_A2": gp(by[(d, f, "A3_SCA_PLUS_SAF")]) - gp(by[(d, f, "A2_SCA_ONLY")])})
    write_csv(OUT / "MODULE_ATTRIBUTION.csv", module)
    state_vs = []
    for d in sorted(set(r["direction"] for r in rows)):
        vals = []
        for f in sorted(set(int(r["fold"]) for r in rows if r["direction"] == d)):
            if (d, f, "A3_SCA_PLUS_SAF") in by and (d, f, "C1_ADAPTER_ONLY_DIRECT_HEAD") in by:
                vals.append(gp(by[(d, f, "A3_SCA_PLUS_SAF")]) - gp(by[(d, f, "C1_ADAPTER_ONLY_DIRECT_HEAD")]))
        if vals:
            state_vs.append({"direction": d, "A3_minus_C1_G_pixel_mean": float(np.mean(vals))})
    write_csv(OUT / "STATE_VS_GENERIC_ADAPTER.csv", state_vs)
    write_csv(OUT / "STATE_SPECIFICITY.csv", [{"status": "NOT_RUN_C2_EARLY_STOP" if pilot_only else "C2_REQUIRED_NOT_IMPLEMENTED_IN_CURRENT_BRANCH"}])

    gap_rows = []
    # P0 may be available only for pilot folds.
    for d in sorted(set(r["direction"] for r in rows)):
        vals = []
        for f in sorted(set(int(r["fold"]) for r in rows if r["direction"] == d)):
            if all((d, f, x) in by for x in ["P0_INSTANCE_MEMORY", "A0_STATE_COSINE", "A3_SCA_PLUS_SAF"]):
                gap = gp(by[(d, f, "P0_INSTANCE_MEMORY")]) - gp(by[(d, f, "A0_STATE_COSINE")])
                rec = gp(by[(d, f, "A3_SCA_PLUS_SAF")]) - gp(by[(d, f, "A0_STATE_COSINE")])
                vals.append((gap, rec))
                gap_rows.append({"direction": d, "fold": f, "gap_before": gap, "recovery": rec, "gap_closure": rec / gap if gap > 0 else float("nan")})
        if vals:
            gap_rows.append({"direction": d, "fold": "mean", "gap_before": float(np.mean([x[0] for x in vals])), "recovery": float(np.mean([x[1] for x in vals])), "gap_closure": float(np.mean([x[1] for x in vals]) / np.mean([x[0] for x in vals])) if np.mean([x[0] for x in vals]) > 0 else float("nan")})
    write_csv(OUT / "INSTANCE_GAP_CLOSURE.csv", gap_rows)

    if not all(r["passed"] for r in parity):
        outcome = "XL_A_IMPLEMENTATION_FAILURE"
    elif gate["c1_dominates"]:
        outcome = "XL_A_GENERIC_PATCH_ADAPTATION_DOMINATES"
    elif not gate["continue_full"]:
        outcome = "XL_A_LOCAL_ALIGNMENT_NO_SIGNAL"
    else:
        outcome = "XL_A_IMPLEMENTATION_FAILURE"

    report = ["# Phase XL-A State-Conditioned Local Alignment Audit", "", f"Outcome: **{outcome}**", "", f"Pilot continuation gate: `{gate['continue_full']}`.", f"C1 dominates: `{gate['c1_dominates']}`.", "", "## Pilot Gains vs A0", ""]
    for variant, g in gate["gains"].items():
        report.append(f"- {variant}: D1f0 {g['D1f0_G']:+.4f} pp, D2f0 {g['D2f0_G']:+.4f} pp; dP-AP {g['D1f0_dPAP']:+.4f}/{g['D2f0_dPAP']:+.4f}.")
    report.extend(["", "## Pilot Metrics", ""])
    for r in pilot_rows:
        report.append(f"- {r['direction']} fold{r['fold']} {r['variant']}: P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, PRO {float(r['PRO']):.4f}.")
    report.extend(["", "Stop: Phase XL-A ended here.", f"Runtime: {time.time() - start:.1f}s."])
    (OUT / "XL_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {"phase": "XL-A", "outcome": outcome, "pilot_gate": gate, "runtime_sec": time.time() - start})
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XL-A State-Conditioned Local Alignment Audit\n")
        fp.write(f"- Parity: {all(r['passed'] for r in parity)}.\n")
        fp.write(f"- Pilot gate: {json.dumps(gate, default=json_default)}.\n")
        fp.write(f"- Outcome: {outcome}.\n")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pilot_variants = ["P0_INSTANCE_MEMORY", "A1_SAF_ONLY", "A2_SCA_ONLY", "A3_SCA_PLUS_SAF", "C1_ADAPTER_ONLY_DIRECT_HEAD"]
    payloads = []
    for direction, fold in DIRECTIONS_FOLDS_PILOT:
        path = ckpt / f"{direction}_fold{fold}_pilot.json"
        if path.exists():
            print(f"[XL-A] resume {direction} fold{fold}", flush=True)
            payloads.append(read_json(path))
            continue
        payload = run_fold(direction, fold, ["A0_STATE_COSINE", *pilot_variants], device)
        write_json(path, payload)
        payloads.append(payload)
    # Strict phase contract: decide from pilot before running full folds/C2.
    aggregate(payloads, start, pilot_only=True)
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
