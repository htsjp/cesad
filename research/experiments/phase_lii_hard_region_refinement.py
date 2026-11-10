import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_li_asymmetric_state import (
    DATA_ROOT,
    OUT as LI_OUT,
    XLV_A,
    XLV_B,
    evaluate_source_maps,
    load_full_visa_adapter,
    load_statealigned_adapter,
)
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, EPOCHS, LR, SEED, WEIGHT_DECAY, balanced_loss, load_train_samples
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_xlv_cross_domain_transfer import ExternalDataset, evaluate_external, gp, load_full_visa_source, top1_image_score
from experiments.phase_xlv_medical_pixel_transfer import MedicalDataset, eval_scores
from experiments.phase_xlviii_dino_memory_distillation import metric_row, safe_float
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, norm_np


OUT = Path("results/phase_lii_hard_region_refinement")
RANK = 4


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


class ResidualRefiner(nn.Module):
    def __init__(self, dim=1024, rank=RANK):
        super().__init__()
        torch.manual_seed(SEED)
        self.a = nn.Linear(dim, rank, bias=False)
        self.b = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.a.weight, std=0.02)
        nn.init.zeros_(self.b.weight)

    def forward(self, z, direction):
        delta = self.b(self.a(z))
        return delta @ direction


@torch.no_grad()
def frozen_terms(base, x, states_t):
    z = base.adapt(x).detach()
    sim_n = (z @ states_t[0]).detach()
    sim_a = (z @ states_t[1]).detach()
    s0 = (sim_a - sim_n).detach()
    p_a = torch.exp(sim_a) / torch.clamp(torch.exp(sim_a) + torch.exp(sim_n), min=1e-12)
    gate = (4.0 * p_a * (1.0 - p_a)).detach()
    return z, s0, gate


def train_refiner(variant, x_np, y_np, states, base, use_gate, device, tag):
    ckpt = OUT / "checkpoints" / f"{tag}_{variant}.pt"
    states_t = torch.tensor(norm_np(states), dtype=torch.float32, device=device)
    direction = torch.nn.functional.normalize(states_t[1] - states_t[0], dim=0)
    refiner = ResidualRefiner(x_np.shape[1]).to(device)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        refiner.load_state_dict(payload["model"], strict=True)
        return refiner.eval(), payload.get("curves", [])
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    x_all = norm_np(x_np.astype(np.float32))
    y_all = y_np.astype(np.float32)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(refiner.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all))
        vals = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            x = torch.tensor(x_all[ids], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[ids], dtype=torch.float32, device=device)
            with torch.no_grad():
                z, s0, gate = frozen_terms(base, x, states_t)
            opt.zero_grad()
            r = refiner(z, direction)
            logits = s0 + (gate * r if use_gate else r)
            loss, bce, dice = balanced_loss(logits, y)
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float64)
        row = {
            "tag": tag,
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "bce": float(arr[:, 1].mean()),
            "dice": float(arr[:, 2].mean()),
        }
        curves.append(row)
        print(f"[LII] train {tag} {variant} epoch {epoch}/{EPOCHS} loss {row['loss']:.4f}", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": refiner.state_dict(), "curves": curves, "use_gate": use_gate, "tag": tag}, ckpt)
    return refiner.eval(), curves


@torch.no_grad()
def score_pair(patch, states, base, refiner, use_gate, device, cell, uncertainty_rows):
    states_t = torch.tensor(norm_np(states), dtype=torch.float32, device=device)
    direction = torch.nn.functional.normalize(states_t[1] - states_t[0], dim=0)
    maps0 = np.zeros((patch.shape[0], 37, 37), dtype=np.float16)
    maps1 = np.zeros((patch.shape[0], 37, 37), dtype=np.float16)
    image0 = np.zeros((patch.shape[0],), dtype=np.float32)
    image1 = np.zeros((patch.shape[0],), dtype=np.float32)
    for i in range(patch.shape[0]):
        x = torch.tensor(norm_np(np.asarray(patch[i], dtype=np.float32)), dtype=torch.float32, device=device)
        z, s0, gate = frozen_terms(base, x, states_t)
        r = refiner(z, direction)
        s1 = s0 + (gate * r if use_gate else r)
        m0 = s0.detach().cpu().numpy().reshape(37, 37).astype(np.float32)
        m1 = s1.detach().cpu().numpy().reshape(37, 37).astype(np.float32)
        maps0[i] = m0.astype(np.float16)
        maps1[i] = m1.astype(np.float16)
        image0[i] = top1_image_score(m0)
        image1[i] = top1_image_score(m1)
        gn = gate.detach().cpu().numpy()
        rn = r.detach().cpu().numpy()
        uncertainty_rows.append({
            "cell": cell,
            "image_pos": i,
            "variant": "E1_HARD_REGION_REFINEMENT" if use_gate else "E2_UNGATED_RESIDUAL",
            "gate_mean": float(np.mean(gn)),
            "gate_std": float(np.std(gn)),
            "gate_q10": float(np.quantile(gn, 0.10)),
            "gate_q50": float(np.quantile(gn, 0.50)),
            "gate_q90": float(np.quantile(gn, 0.90)),
            "residual_mean": float(np.mean(rn)),
            "residual_abs_mean": float(np.mean(np.abs(rn))),
            "delta_score_abs_mean": float(np.mean(np.abs(gn * rn if use_gate else rn))),
        })
    return (image0, maps0), (image1, maps1)


def source_train_bundle(direction, fold, device):
    dataset_name, dataset, held_indices, held_patch, states, base, _remat = load_statealigned_adapter(direction, fold, device)
    _dn, _ds, _rot, train_indices, _held = __import__("experiments.phase_xxxviii_visual_state_representation", fromlist=["load_fold"]).load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    x_train, y_train, _ = load_train_samples(dataset, train_indices, cache["train_patch"])
    return dataset_name, dataset, held_indices, held_patch, states, base, x_train, y_train


def full_train_bundle(device):
    states, base = load_full_visa_adapter(device)
    cache_path = OUT / "cache" / "full_visa_train_sample.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return states, base, d["x"], d["y"]
    _name, dataset, indices, _cls, patch, _y, _cats, _rows = load_full_visa_source()
    from experiments.phase_xxxviii_visual_state_representation import class_category_patch_sample, PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY
    x, y, _ = class_category_patch_sample(dataset, indices, patch, PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY, seed=SEED)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=norm_np(x).astype(np.float32), y=y.astype(np.int64))
    return states, base, norm_np(x), y.astype(np.int64)


def append_source_metrics(direction, fold, variant, image_scores, maps, dataset, held_indices, raw, per):
    dataset_name = "mvtec" if direction == "D1" else "visa"
    cell_group = "VisA_target" if direction == "D1" else "MVTec_target"
    cell = f"{direction}_fold{fold}_{dataset_name}_heldout"
    res = evaluate_source_maps(dataset, held_indices, image_scores, maps)
    row = {"cell": cell, "cell_group": cell_group, "direction": direction, "fold": fold, "target": dataset_name, "variant": variant, **res["mean"]}
    row["G_pixel"] = gp(row)
    raw.append(row)
    for cat, vals in res["per_category"].items():
        rr = {"cell": cell, "direction": direction, "fold": fold, "target": dataset_name, "category": cat, "variant": variant, **vals}
        rr["G_pixel"] = gp(rr)
        per.append(rr)


def append_external_metrics(name, variant, image_scores, maps, raw, per):
    if name == "BTAD":
        dataset = ExternalDataset(DATA_ROOT / "btad" / "BTech_Dataset_transformed", "BTAD", mode="test")
        cache = np.load(XLV_A / "feature_cache" / "BTAD_clip.npz", allow_pickle=True)
        mean, pc = evaluate_external(dataset, [dict(r) for r in cache["rows"]], image_scores, maps)
    else:
        dataset = MedicalDataset(DATA_ROOT / name, mode="test")
        cache = np.load(XLV_B / "feature_cache" / f"{name}_clip.npz", allow_pickle=True)
        mean, _pi, _pix = eval_scores(dataset, [dict(r) for r in cache["rows"]], image_scores, maps)
        pc = {"skin" if name == "ISBI" else "colon": mean}
    row = {"cell": f"FullVisA_to_{name}", "cell_group": "external", "direction": "FullVisA", "fold": "full", "target": name, "variant": variant, **mean}
    row["G_pixel"] = gp(row)
    raw.append(row)
    for cat, vals in pc.items():
        rr = {"cell": f"FullVisA_to_{name}", "direction": "FullVisA", "fold": "full", "target": name, "category": cat, "variant": variant, **vals}
        rr["G_pixel"] = gp(rr)
        per.append(rr)


def evaluate_external_refiner(name, states, base, refiner, use_gate, device, raw, per, uncertainty):
    cache_path = XLV_A / "feature_cache" / "BTAD_clip.npz" if name == "BTAD" else XLV_B / "feature_cache" / f"{name}_clip.npz"
    cache = np.load(cache_path, allow_pickle=True)
    print(f"[LII] evaluate FullVisA_to_{name} {'E1' if use_gate else 'E2'}", flush=True)
    (img0, maps0), (img1, maps1) = score_pair(cache["patch"], states, base, refiner, use_gate, device, f"FullVisA_to_{name}", uncertainty)
    if not any(r["cell"] == f"FullVisA_to_{name}" and r["variant"] == "E0_STATE_ALIGNED" for r in raw):
        append_external_metrics(name, "E0_STATE_ALIGNED", img0, maps0, raw, per)
    append_external_metrics(name, "E1_HARD_REGION_REFINEMENT" if use_gate else "E2_UNGATED_RESIDUAL", img1, maps1, raw, per)


def comparison_rows(raw, baseline="E0_STATE_ALIGNED", variant="E1_HARD_REGION_REFINEMENT"):
    by = {(r["cell"], r["variant"]): r for r in raw}
    out = []
    for cell in sorted({r["cell"] for r in raw}):
        if (cell, baseline) not in by or (cell, variant) not in by:
            continue
        e0, e1 = by[(cell, baseline)], by[(cell, variant)]
        out.append({
            "cell": cell,
            "cell_group": e0["cell_group"],
            "target": e0["target"],
            "direction": e0["direction"],
            "fold": e0["fold"],
            "comparison": f"{variant}_minus_{baseline}",
            "delta_P-AUROC": safe_float(e1["P-AUROC"]) - safe_float(e0["P-AUROC"]),
            "delta_P-AP": safe_float(e1["P-AP"]) - safe_float(e0["P-AP"]),
            "delta_G_pixel": gp(e1) - gp(e0),
        })
    return out


def group_rows(cell_rows):
    rows = []
    for name, pred in [
        ("VisA_target", lambda r: r["cell_group"] == "VisA_target"),
        ("MVTec_target", lambda r: r["cell_group"] == "MVTec_target"),
        ("external", lambda r: r["cell_group"] == "external"),
        ("overall", lambda r: True),
    ]:
        vals = [r["delta_G_pixel"] for r in cell_rows if pred(r) and np.isfinite(r["delta_G_pixel"])]
        rows.append({
            "group": name,
            "cells": len(vals),
            "mean_delta_G_pixel": float(np.mean(vals)) if vals else float("nan"),
            "positive_cells": int(sum(v > 0 for v in vals)),
            "drop_le_minus1_cells": int(sum(v <= -1.0 for v in vals)),
        })
    return rows


def e1_pass(cell_rows, groups, raw):
    by_group = {r["group"]: r for r in groups}
    by = {(r["target"], r["variant"]): r for r in raw}
    btad_ap = safe_float(by[("BTAD", "E1_HARD_REGION_REFINEMENT")]["P-AP"]) - safe_float(by[("BTAD", "E0_STATE_ALIGNED")]["P-AP"])
    return (
        sum(r["delta_G_pixel"] > 0 for r in cell_rows) >= 5
        and by_group["overall"]["mean_delta_G_pixel"] >= 0.5
        and by_group["VisA_target"]["mean_delta_G_pixel"] > 0
        and by_group["MVTec_target"]["mean_delta_G_pixel"] > 0
        and by_group["external"]["mean_delta_G_pixel"] > 0
        and all(r["delta_G_pixel"] > -1.0 for r in cell_rows)
        and btad_ap > 0
    )


def decide(cell_rows, groups, raw, e2_rows=None):
    if e1_pass(cell_rows, groups, raw):
        if not e2_rows:
            return "HARD_REGION_REFINEMENT_E2_REQUIRED"
        e1_mean = np.mean([r["delta_G_pixel"] for r in comparison_rows(raw, variant="E1_HARD_REGION_REFINEMENT")])
        e2_mean = np.mean([r["delta_G_pixel"] for r in comparison_rows(raw, variant="E2_UNGATED_RESIDUAL")])
        e1_neg = sum(r["delta_G_pixel"] < 0 for r in comparison_rows(raw, variant="E1_HARD_REGION_REFINEMENT"))
        e2_neg = sum(r["delta_G_pixel"] < 0 for r in comparison_rows(raw, variant="E2_UNGATED_RESIDUAL"))
        if e1_mean - e2_mean >= 0.5 and e1_neg < e2_neg:
            return "HARD_REGION_SELECTION_SUPPORTED"
        return "RESIDUAL_REFINEMENT_SUPPORTED_SELECTION_UNPROVEN"
    overall = next(r for r in groups if r["group"] == "overall")
    if overall["mean_delta_G_pixel"] <= 0 or overall["drop_le_minus1_cells"] >= 2:
        return "HARD_REGION_REFINEMENT_NO_SIGNAL"
    return "HARD_REGION_REFINEMENT_DOMAIN_DEPENDENT"


def run_variant(use_gate, raw, per, uncertainty, curves):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant = "E1_HARD_REGION_REFINEMENT" if use_gate else "E2_UNGATED_RESIDUAL"
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            dataset_name, dataset, held_indices, held_patch, states, base, x_train, y_train = source_train_bundle(direction, fold, device)
            tag = f"{direction}_fold{fold}"
            refiner, c = train_refiner(variant, x_train, y_train, states, base, use_gate, device, tag)
            curves.extend(c)
            print(f"[LII] evaluate {tag} {variant}", flush=True)
            (img0, maps0), (img1, maps1) = score_pair(held_patch, states, base, refiner, use_gate, device, f"{direction}_fold{fold}_{dataset_name}_heldout", uncertainty)
            if use_gate and not any(r["cell"] == f"{direction}_fold{fold}_{dataset_name}_heldout" and r["variant"] == "E0_STATE_ALIGNED" for r in raw):
                append_source_metrics(direction, fold, "E0_STATE_ALIGNED", img0, maps0, dataset, held_indices, raw, per)
            append_source_metrics(direction, fold, variant, img1, maps1, dataset, held_indices, raw, per)
    states, base, x_train, y_train = full_train_bundle(device)
    refiner, c = train_refiner(variant, x_train, y_train, states, base, use_gate, device, "FullVisA")
    curves.extend(c)
    for name in ["BTAD", "ISBI", "CVC-ClinicDB"]:
        evaluate_external_refiner(name, states, base, refiner, use_gate, device, raw, per, uncertainty)


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    raw, per, uncertainty, curves = [], [], [], []
    run_variant(True, raw, per, uncertainty, curves)
    e1_cells = comparison_rows(raw, variant="E1_HARD_REGION_REFINEMENT")
    groups = group_rows(e1_cells)
    write_json(OUT / "E1_GATE.json", {"e1_pass": e1_pass(e1_cells, groups, raw), "groups": groups, "cells": e1_cells})
    e2_comp = []
    if e1_pass(e1_cells, groups, raw):
        run_variant(False, raw, per, uncertainty, curves)
        e2_comp = comparison_rows(raw, variant="E2_UNGATED_RESIDUAL")
    else:
        for target in ["D1_fold0", "D1_fold1", "D2_fold0", "D2_fold1", "BTAD", "ISBI", "CVC-ClinicDB"]:
            e2_comp.append({"cell": target, "comparison": "E2_NOT_RUN_E1_GATE_FAILED"})
    outcome = decide(e1_cells, groups, raw, e2_comp if e1_pass(e1_cells, groups, raw) else None)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "GROUP_METRICS.csv", e1_cells + groups)
    write_csv(OUT / "PER_CATEGORY.csv", per)
    write_csv(OUT / "UNCERTAINTY_DIAGNOSTIC.csv", uncertainty)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)
    write_csv(OUT / "E1_VS_E2.csv", e2_comp)
    write_json(OUT / "OUTCOME.json", {"phase": "LII", "outcome": outcome, "e1_pass": e1_pass(e1_cells, groups, raw), "groups": groups, "runtime_sec": time.time() - start})
    report = ["# Phase LII Selective Hard-Region Residual Refinement", "", f"Outcome: **{outcome}**", "", "## E1 Cell Deltas", "", "| Cell | Target | dG | dP-AUROC | dP-AP |", "|---|---|---:|---:|---:|"]
    for r in e1_cells:
        report.append(f"| {r['cell']} | {r['target']} | {r['delta_G_pixel']:+.4f} | {r['delta_P-AUROC']:+.4f} | {r['delta_P-AP']:+.4f} |")
    report.extend(["", "## Group Means", "", "| Group | Cells | Mean dG | Positive cells | <=-1 cells |", "|---|---:|---:|---:|---:|"])
    for r in groups:
        report.append(f"| {r['group']} | {r['cells']} | {r['mean_delta_G_pixel']:+.4f} | {r['positive_cells']} | {r['drop_le_minus1_cells']} |")
    report.extend([
        "",
        f"E1 pass: `{e1_pass(e1_cells, groups, raw)}`. E2 was {'run' if e1_pass(e1_cells, groups, raw) else 'not run because E1 gate failed'}.",
        "Base StateAligned was frozen. Only the rank4 residual branch was trained on source pixel loss. No text, DINO, Memory, retrieval, synthetic anomaly, target adaptation, new states, classifier, or learned router was used.",
        f"Runtime: {time.time() - start:.1f}s.",
        "",
        "Stop: no rank/gate/loss rescue was started.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": outcome, "groups": groups}, indent=2), flush=True)


if __name__ == "__main__":
    main()
