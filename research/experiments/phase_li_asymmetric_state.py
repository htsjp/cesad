import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import fast_cal_pro_score
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import load_train_samples
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState, train_adapter_to_state
from experiments.phase_xliii_anchor_specificity import train_adapter as train_adapter_anchor
from experiments.phase_xlv_cross_domain_transfer import ExternalDataset, evaluate_external, gp, top1_image_score
from experiments.phase_xlv_cross_domain_transfer import load_full_visa_source
from experiments.phase_xlv_medical_pixel_transfer import MedicalDataset, eval_scores
from experiments.phase_xlviii_dino_memory_distillation import e0_rows, metric_row, safe_float
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, IMAGE_SIZE, SIDE, load_fold, norm_np, patch_labels, safe_ap, safe_auc


OUT = Path("results/phase_li_asymmetric_state")
XXXVIII = Path("results/phase_xxxviii_visual_state_representation")
XLV_A = Path("results/phase_xlv_cross_domain_transfer")
XLV_B = Path("results/phase_xlv_medical_pixel_transfer")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
SIGMA = 4


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


def upsample_map(map37):
    x = torch.tensor(np.asarray(map37, dtype=np.float32)).view(1, 1, SIDE, SIDE)
    full = torch.nn.functional.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze().numpy()
    return gaussian_filter(full.astype(np.float32), sigma=SIGMA).astype(np.float32)


def load_statealigned_adapter(direction, fold, device):
    dataset_name, dataset, _rot, train_indices, held_indices = load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    states = np.load(XXXVIII / "states" / f"{direction}_fold{fold}_states.npz")["K2_patch"].astype(np.float32)
    ckpt = OUT / "frozen_checkpoint_cache" / f"{direction}_fold{fold}_statealigned.pt"
    model = AdapterToState(1024, states).to(device).eval()
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        model.load_state_dict(payload["model"], strict=True)
        return dataset_name, dataset, held_indices, cache["held_patch"], states, model, payload.get("rematerialized", False)
    train_patch = cache["train_patch"]
    x_train, y_train, _ = load_train_samples(dataset, train_indices, train_patch)
    print(f"[LI] rematerialize frozen StateAligned adapter {direction} fold{fold}", flush=True)
    if fold == 0:
        model, curves = train_adapter_to_state(x_train, y_train, states, device)
    else:
        model, curves = train_adapter_anchor(x_train, y_train, states, device, "A0_LEARNED")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "curves": curves,
        "direction": direction,
        "fold": fold,
        "rematerialized": True,
        "reason": "historical XLIII/XLIV checkpoints stored metrics JSON but not adapter state_dict",
    }, ckpt)
    return dataset_name, dataset, held_indices, cache["held_patch"], states, model.eval(), True


def load_full_visa_adapter(device):
    states = np.load(XLV_A / "source_model" / "full_visa_states.npz")["patch_states"].astype(np.float32)
    payload = torch.load(XLV_A / "source_model" / "full_visa_adapter.pt", map_location=device)
    model = AdapterToState(1024, states).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return states, model


@torch.no_grad()
def score_symmetric_and_asymmetric(patch, model, states, device, cell_id, diagnostic_rows, alignment_rows=None, dataset=None, indices=None):
    states_t = torch.tensor(norm_np(states), dtype=torch.float32, device=device)
    maps0 = np.zeros((patch.shape[0], SIDE, SIDE), dtype=np.float16)
    maps1 = np.zeros((patch.shape[0], SIDE, SIDE), dtype=np.float16)
    image0 = np.zeros((patch.shape[0],), dtype=np.float32)
    image1 = np.zeros((patch.shape[0],), dtype=np.float32)
    for i in range(patch.shape[0]):
        x = torch.tensor(norm_np(np.asarray(patch[i], dtype=np.float32)), dtype=torch.float32, device=device)
        z = model.adapt(x)
        sim_n = z @ states_t[0]
        sim_a = z @ states_t[1]
        s0 = sim_a - sim_n
        w = torch.softmax(-s0, dim=0)
        nq = torch.nn.functional.normalize((w[:, None] * z).sum(dim=0), dim=0)
        sim_nq = z @ nq
        sim_n_asym = torch.maximum(sim_n, sim_nq)
        s1 = sim_a - sim_n_asym
        m0 = s0.detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
        m1 = s1.detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
        maps0[i] = m0.astype(np.float16)
        maps1[i] = m1.astype(np.float16)
        image0[i] = top1_image_score(m0)
        image1[i] = top1_image_score(m1)
        row = {
            "cell": cell_id,
            "image_pos": i,
            "cos_nq_sN": float((nq @ states_t[0]).detach().cpu()),
            "cos_nq_sA": float((nq @ states_t[1]).detach().cpu()),
            "mean_S0": float(s0.mean().detach().cpu()),
            "mean_S1": float(s1.mean().detach().cpu()),
        }
        if dataset is not None and indices is not None:
            item = dataset[indices[i]]
            row.update({"category": str(item["cls_name"]), "image_label": int(item["anomaly"])})
        diagnostic_rows.append(row)
        if alignment_rows is not None and dataset is not None and indices is not None:
            labels = patch_labels(dataset[indices[i]])
            znq = sim_nq.detach().cpu().numpy()
            zsn = sim_n.detach().cpu().numpy()
            for group, mask in [("normal_gt_patch", labels == 0), ("anomaly_gt_patch", labels == 1)]:
                if np.any(mask):
                    alignment_rows.append({
                        "cell": cell_id,
                        "image_pos": i,
                        "category": str(dataset[indices[i]]["cls_name"]),
                        "image_label": int(dataset[indices[i]]["anomaly"]),
                        "patch_group": group,
                        "count": int(mask.sum()),
                        "mean_cos_z_nq": float(np.mean(znq[mask])),
                        "mean_cos_z_sN": float(np.mean(zsn[mask])),
                        "delta_nq_minus_sN": float(np.mean(znq[mask]) - np.mean(zsn[mask])),
                    })
    return (image0, maps0), (image1, maps1)


def evaluate_source_cell(direction, fold, device, raw_rows, per_rows, q_rows, align_rows, protocol_rows):
    dataset_name, dataset, held_indices, held_patch, states, model, remat = load_statealigned_adapter(direction, fold, device)
    cell = f"{direction}_fold{fold}_{dataset_name}_heldout"
    print(f"[LI] evaluate {cell}", flush=True)
    (img0, maps0), (img1, maps1) = score_symmetric_and_asymmetric(
        held_patch, model, states, device, cell, q_rows, align_rows, dataset, held_indices
    )
    for variant, image_scores, maps in [("E0_STATE_ALIGNED", img0, maps0), ("E1_ASYMMETRIC_STATE", img1, maps1)]:
        res = evaluate_source_maps(dataset, held_indices, image_scores, maps)
        metric = {"cell": cell, "cell_group": "VisA_target" if direction == "D1" else "MVTec_target", "direction": direction, "fold": fold, "target": dataset_name, "variant": variant, **res["mean"]}
        metric["G_pixel"] = gp(metric)
        raw_rows.append(metric)
        for cat, vals in res["per_category"].items():
            row = {"cell": cell, "direction": direction, "fold": fold, "target": dataset_name, "category": cat, "variant": variant, **vals}
            row["G_pixel"] = gp(row)
            per_rows.append(row)
    protocol_rows.append({"cell": cell, "adapter_source": "rematerialized_historical_statealigned" if remat else "loaded_statealigned_pt", "checkpoint_training_for_li_method": False})


def evaluate_source_maps(dataset, held_indices, image_scores, patch_maps):
    by_cat = defaultdict(lambda: {"iy": [], "is": [], "py": [], "ps": []})
    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        cat = str(item["cls_name"])
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
        full = upsample_map(patch_maps[pos]).reshape(-1)
        by_cat[cat]["iy"].append(int(item["anomaly"]))
        by_cat[cat]["is"].append(float(image_scores[pos]))
        by_cat[cat]["py"].append(mask)
        by_cat[cat]["ps"].append(full)
    per, ia, iap, pa, pap, pro = {}, [], [], [], [], []
    for cat, vals in sorted(by_cat.items()):
        py = np.concatenate(vals["py"])
        ps = np.concatenate(vals["ps"]).astype(np.float32)
        masks = np.stack([x.reshape(IMAGE_SIZE, IMAGE_SIZE) for x in vals["py"]]).astype(np.uint8)
        maps = np.stack([x.reshape(IMAGE_SIZE, IMAGE_SIZE) for x in vals["ps"]]).astype(np.float32)
        row = {
            "I-AUROC": safe_auc(vals["iy"], vals["is"]),
            "I-AP": safe_ap(vals["iy"], vals["is"]),
            "P-AUROC": safe_auc(py, ps),
            "P-AP": safe_ap(py, ps),
            "PRO": float(fast_cal_pro_score(masks, maps) * 100.0),
        }
        per[cat] = row
        for key, dst in [("I-AUROC", ia), ("I-AP", iap), ("P-AUROC", pa), ("P-AP", pap), ("PRO", pro)]:
            if np.isfinite(row[key]):
                dst.append(row[key])
    return {"mean": {
        "I-AUROC": float(np.mean(ia)) if ia else float("nan"),
        "I-AP": float(np.mean(iap)) if iap else float("nan"),
        "P-AUROC": float(np.mean(pa)) if pa else float("nan"),
        "P-AP": float(np.mean(pap)) if pap else float("nan"),
        "PRO": float(np.mean(pro)) if pro else float("nan"),
    }, "per_category": per}


def evaluate_external_cell(name, cache_path, dataset, states, model, device, raw_rows, per_rows, q_rows):
    print(f"[LI] evaluate external {name}", flush=True)
    cache = np.load(cache_path, allow_pickle=True)
    patch = cache["patch"]
    rows = [dict(r) for r in cache["rows"]]
    (img0, maps0), (img1, maps1) = score_symmetric_and_asymmetric(patch, model, states, device, f"FullVisA_to_{name}", q_rows)
    for variant, image_scores, maps in [("E0_STATE_ALIGNED", img0, maps0), ("E1_ASYMMETRIC_STATE", img1, maps1)]:
        if name == "BTAD":
            mean, per = evaluate_external(dataset, rows, image_scores, maps)
        else:
            mean, _per_img, _pixel = eval_scores(dataset, rows, image_scores, maps)
            per = {"colon" if name == "CVC-ClinicDB" else "skin": mean}
        metric = {"cell": f"FullVisA_to_{name}", "cell_group": "external", "direction": "FullVisA", "fold": "full", "target": name, "variant": variant, **mean}
        metric["G_pixel"] = gp(metric)
        raw_rows.append(metric)
        for cat, vals in per.items():
            row = {"cell": f"FullVisA_to_{name}", "direction": "FullVisA", "fold": "full", "target": name, "category": cat, "variant": variant, **vals}
            row["G_pixel"] = gp(row)
            per_rows.append(row)


def group_metrics(raw_rows):
    by = defaultdict(dict)
    for r in raw_rows:
        by[r["cell"]][r["variant"]] = r
    cell_rows = []
    for cell, vals in sorted(by.items()):
        if "E0_STATE_ALIGNED" not in vals or "E1_ASYMMETRIC_STATE" not in vals:
            continue
        e0, e1 = vals["E0_STATE_ALIGNED"], vals["E1_ASYMMETRIC_STATE"]
        cell_rows.append({
            "cell": cell,
            "cell_group": e0["cell_group"],
            "target": e0["target"],
            "direction": e0["direction"],
            "fold": e0["fold"],
            "E0_G_pixel": gp(e0),
            "E1_G_pixel": gp(e1),
            "delta_G_pixel": gp(e1) - gp(e0),
            "delta_P-AUROC": safe_float(e1["P-AUROC"]) - safe_float(e0["P-AUROC"]),
            "delta_P-AP": safe_float(e1["P-AP"]) - safe_float(e0["P-AP"]),
        })
    groups = []
    for name, predicate in [
        ("VisA_target", lambda r: r["cell_group"] == "VisA_target"),
        ("MVTec_target", lambda r: r["cell_group"] == "MVTec_target"),
        ("external", lambda r: r["cell_group"] == "external"),
        ("overall", lambda r: True),
    ]:
        vals = [r["delta_G_pixel"] for r in cell_rows if predicate(r) and np.isfinite(r["delta_G_pixel"])]
        groups.append({
            "group": name,
            "cells": len(vals),
            "mean_delta_G_pixel": float(np.mean(vals)) if vals else float("nan"),
            "positive_0p5_cells": int(sum(v >= 0.5 for v in vals)),
            "drop_below_minus1_cells": int(sum(v < -1.0 for v in vals)),
        })
    return cell_rows, groups


def decide(cell_rows, group_rows):
    deltas = [r["delta_G_pixel"] for r in cell_rows]
    by_group = {r["group"]: r for r in group_rows}
    supported = (
        sum(d >= 0.5 for d in deltas) >= 5
        and by_group["VisA_target"]["mean_delta_G_pixel"] > 0
        and by_group["MVTec_target"]["mean_delta_G_pixel"] > 0
        and by_group["external"]["mean_delta_G_pixel"] > 0
        and all(d >= -1.0 for d in deltas)
        and by_group["overall"]["mean_delta_G_pixel"] >= 0.5
    )
    if supported:
        return "ASYMMETRIC_STATE_SUPPORTED"
    bad_groups = sum(by_group[g]["mean_delta_G_pixel"] <= 0 for g in ["VisA_target", "MVTec_target", "external"])
    if by_group["overall"]["mean_delta_G_pixel"] <= 0 or sum(d < -1.0 for d in deltas) >= 2 or bad_groups >= 2:
        return "ASYMMETRIC_STATE_NO_SIGNAL"
    return "ASYMMETRIC_STATE_DOMAIN_DEPENDENT"


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_rows, per_rows, q_rows, align_rows, protocol_rows = [], [], [], [], []
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            evaluate_source_cell(direction, fold, device, raw_rows, per_rows, q_rows, align_rows, protocol_rows)
    states, full_model = load_full_visa_adapter(device)
    evaluate_external_cell("BTAD", XLV_A / "feature_cache" / "BTAD_clip.npz", ExternalDataset(DATA_ROOT / "btad" / "BTech_Dataset_transformed", "BTAD", mode="test"), states, full_model, device, raw_rows, per_rows, q_rows)
    evaluate_external_cell("ISBI", XLV_B / "feature_cache" / "ISBI_clip.npz", MedicalDataset(DATA_ROOT / "ISBI", mode="test"), states, full_model, device, raw_rows, per_rows, q_rows)
    evaluate_external_cell("CVC-ClinicDB", XLV_B / "feature_cache" / "CVC-ClinicDB_clip.npz", MedicalDataset(DATA_ROOT / "CVC-ClinicDB", mode="test"), states, full_model, device, raw_rows, per_rows, q_rows)
    cell_rows, groups = group_metrics(raw_rows)
    outcome = decide(cell_rows, groups)
    write_csv(OUT / "RAW_METRICS.csv", raw_rows)
    write_csv(OUT / "GROUP_METRICS.csv", cell_rows + groups)
    write_csv(OUT / "PER_CATEGORY.csv", per_rows)
    write_csv(OUT / "QUERY_STATE_GEOMETRY.csv", q_rows)
    write_csv(OUT / "NORMAL_ALIGNMENT_DIAGNOSTIC.csv", align_rows)
    write_json(OUT / "PROTOCOL.json", {"no_new_method_training": True, "inference_only_variant": True, "temperature": "implicit_softmax_temperature_1", "checkpoint_notes": protocol_rows})
    write_json(OUT / "OUTCOME.json", {"phase": "LI", "outcome": outcome, "cell_count": len(cell_rows), "groups": groups, "runtime_sec": time.time() - start})
    report = ["# Phase LI Asymmetric State Modeling", "", f"Outcome: **{outcome}**", "", "## Cell Deltas", "", "| Cell | Target | dG | dP-AUROC | dP-AP |", "|---|---|---:|---:|---:|"]
    for r in cell_rows:
        report.append(f"| {r['cell']} | {r['target']} | {r['delta_G_pixel']:+.4f} | {r['delta_P-AUROC']:+.4f} | {r['delta_P-AP']:+.4f} |")
    report.extend(["", "## Group Means", "", "| Group | Cells | Mean dG | >=+0.5 cells | <-1 cells |", "|---|---:|---:|---:|---:|"])
    for r in groups:
        report.append(f"| {r['group']} | {r['cells']} | {r['mean_delta_G_pixel']:+.4f} | {r['positive_0p5_cells']} | {r['drop_below_minus1_cells']} |")
    report.extend([
        "",
        "Only inference scoring was changed for E1. No target training, target normal reference, test-time optimization, text, DINO, retrieval, memory, threshold tuning, or score calibration was used.",
        "D1/D2 adapters were rematerialized only when historical phases had saved metrics JSON without adapter state_dict; Full VisA external used the existing XLV `.pt` checkpoint.",
        f"Runtime: {time.time() - start:.1f}s.",
        "",
        "Stop: no rescue variant or parameter search was started.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": outcome, "groups": groups}, indent=2), flush=True)


if __name__ == "__main__":
    main()
