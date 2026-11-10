import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import load_train_samples
from experiments.phase_xliii_anchor_specificity import low_rank_rows, train_adapter
from experiments.phase_xxxviii_visual_state_representation import (
    FEATURE_CACHE,
    IMAGE_SIZE,
    PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY,
    SIDE,
    centroids,
    class_category_patch_sample,
    instance_memory_scores,
    labels_and_rows,
    load_fold,
    norm_np,
    patch_labels,
    state_scores,
    train_state_vectors,
    upsample_map,
)
from experiments.phase_xxxix_local_multistate import evaluate_variant
from phase_xa.official_task_forward import normalized_image_feature, normalized_patch_feature, visual_outputs
from utils.dataset import generate_class_info
from utils.metrics import image_level_metrics
from utils.transforms import get_transform


OUT = Path("results/phase_xlv_cross_domain_transfer")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
SOURCE_CACHE = FEATURE_CACHE / "D2_fold0_clip.npz"
CHECKPOINT = Path("checkpoints/test_on_mvtec.pth")
TARGETS = {
    "BTAD": DATA_ROOT / "btad" / "BTech_Dataset_transformed",
    "BrainMRI": DATA_ROOT / "BrainMRI",
}
VARIANTS = ["V0_SINGLE_STATE", "V1_STATE_ALIGNED", "V2_INSTANCE_MEMORY"]
SEED = 0


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


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def gp(row):
    return 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")


def top1_image_score(map37):
    flat = np.asarray(map37, dtype=np.float32).reshape(-1)
    k = max(1, int(flat.size * 0.01))
    return float(np.sort(flat)[-k:].mean())


class ExternalDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_name, mode="test"):
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.preprocess, self.target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))[mode]
        self.data_all = []
        for cls_name in sorted(meta):
            self.data_all.extend(meta[cls_name])
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)

    def __len__(self):
        return len(self.data_all)

    def _resolve_mask(self, data):
        mask_path = data.get("mask_path", "")
        if mask_path:
            p = self.root / mask_path
            return p if p.exists() and p.is_file() else None
        if self.dataset_name == "BTAD" and int(data.get("anomaly", 0)) == 1:
            img_rel = Path(data["img_path"])
            cand_dir = self.root / img_rel.parts[0] / "ground_truth" / "ko"
            stem = img_rel.stem
            for ext in [img_rel.suffix, ".png", ".bmp", ".jpg"]:
                cand = cand_dir / f"{stem}{ext}"
                if cand.exists() and cand.is_file():
                    return cand
        return None

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path = self.root / data["img_path"]
        img = Image.open(img_path).convert("RGB")
        anomaly = int(data["anomaly"])
        mask_file = self._resolve_mask(data)
        if anomaly == 1 and mask_file is not None:
            mask = Image.open(mask_file).convert("L")
            mask_available = True
        else:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
            mask_available = anomaly == 0
        img_t = self.preprocess(img)
        mask_t = self.target_transform(mask)
        cls_name = str(data["cls_name"])
        return {
            "img": img_t,
            "img_mask": mask_t,
            "cls_name": cls_name,
            "anomaly": anomaly,
            "img_path": str(img_path),
            "mask_available": mask_available,
            "cls_id": self.class_name_map_class_id[cls_name],
        }


@torch.no_grad()
def extract_clip_one(model, item, device):
    image = item["img"].unsqueeze(0).to(device)
    out = visual_outputs(model, image, adapter=None, grad=False)
    cls = normalized_image_feature(out).squeeze(0).float().cpu().numpy().astype(np.float32)
    patch = normalized_patch_feature(out).squeeze(0).float().cpu().numpy().astype(np.float32)
    return cls, patch


def target_rows(dataset):
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        rows.append({
            "position": i,
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "mask_available": bool(item["mask_available"]),
            "img_path": item["img_path"],
        })
    return rows


def ensure_target_cache(target_name, dataset, device):
    cache_path = OUT / "feature_cache" / f"{target_name}_clip.npz"
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model, *_ = load_components(str(CHECKPOINT), device)
    cls_list, patch_list = [], []
    rows = []
    start = time.time()
    for i in range(len(dataset)):
        if i % 50 == 0:
            print(f"[XLV-A] extract {target_name} {i}/{len(dataset)}", flush=True)
        item = dataset[i]
        cls, patch = extract_clip_one(model, item, device)
        cls_list.append(cls)
        patch_list.append(patch.astype(np.float16))
        rows.append({
            "position": i,
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "mask_available": bool(item["mask_available"]),
            "img_path": item["img_path"],
        })
    np.savez_compressed(
        cache_path,
        cls=np.stack(cls_list).astype(np.float32),
        patch=np.stack(patch_list).astype(np.float16),
        rows=np.asarray(rows, dtype=object),
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[XLV-A] extracted {target_name} in {time.time() - start:.1f}s", flush=True)
    return cache_path


def load_full_visa_source():
    dataset_name, dataset, _rot, train_indices, held_indices = load_fold("D2", 0)
    cache = np.load(SOURCE_CACHE, allow_pickle=True)
    cls = np.concatenate([cache["train_cls"], cache["held_cls"]], axis=0).astype(np.float32)
    patch = np.concatenate([cache["train_patch"], cache["held_patch"]], axis=0)
    indices = list(train_indices) + list(held_indices)
    y, cats, rows = labels_and_rows(dataset, indices)
    return dataset_name, dataset, indices, cls, patch, y, cats, rows


def train_full_source_model(device):
    state_path = OUT / "source_model" / "full_visa_states.npz"
    adapter_path = OUT / "source_model" / "full_visa_adapter.pt"
    audit_path = OUT / "SOURCE_MODEL_AUDIT.json"
    curve_rows = []
    dataset_name, dataset, indices, cls, patch, y_img, cat_img, _rows = load_full_visa_source()
    if state_path.exists():
        states_npz = np.load(state_path)
        img_states = states_npz["img_states"]
        patch_states = states_npz["patch_states"]
        state_curves = []
    else:
        print("[XLV-A] train full VisA K2 states", flush=True)
        patch_x, patch_y, patch_cat = class_category_patch_sample(
            dataset, indices, patch, PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY, seed=SEED
        )
        init_img = centroids(cls, y_img)
        init_patch = centroids(patch_x, patch_y)
        img_states, patch_states, state_curves = train_state_vectors(
            cls,
            y_img,
            cat_img,
            patch_x,
            patch_y,
            patch_cat,
            init_img,
            init_patch,
            shuffled=False,
            device=device,
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(state_path, img_states=img_states, patch_states=patch_states)
        curve_rows.extend(state_curves)
    x_train, y_train, _cats = load_train_samples(dataset, indices, patch)
    if adapter_path.exists():
        payload = torch.load(adapter_path, map_location=device)
        from experiments.phase_xliii_adapter_head_decomposition import AdapterToState

        adapter = AdapterToState(x_train.shape[1], patch_states).to(device).eval()
        adapter.load_state_dict(payload["model"], strict=True)
        adapter_curves = payload.get("curves", [])
    else:
        print("[XLV-A] train full VisA rank4 adapter", flush=True)
        adapter, adapter_curves = train_adapter(x_train, y_train, patch_states, device, "V1_STATE_ALIGNED")
        torch.save({"model": adapter.state_dict(), "curves": adapter_curves}, adapter_path)
    for r in adapter_curves:
        rr = dict(r)
        rr["stage"] = "adapter"
        curve_rows.append(rr)
    write_csv(OUT / "SOURCE_TRAINING_CURVES.csv", curve_rows)
    source_patch_mem_x, source_patch_mem_y, _ = source_memory(dataset, indices, patch)
    write_json(audit_path, {
        "source": "FULL VisA source-labeled test/meta protocol inherited from prior source-held-out experiments",
        "source_cache": str(SOURCE_CACHE),
        "source_images": int(len(indices)),
        "source_anomaly_images": int(np.sum(y_img)),
        "patch_state_vectors": 2,
        "adapter_trainable_parameters": 8192,
        "adapter_rank": 4,
        "adapter_epochs": 10,
        "seed": SEED,
        "instance_memory_patch_vectors": int(source_patch_mem_x.shape[0]),
        "target_data_used_for_training": False,
    })
    geom = []
    ps = norm_np(patch_states)
    ims = norm_np(img_states)
    geom.append({"branch": "patch", "cos_sN_sA": float(ps[0] @ ps[1]), "norm_sN": float(np.linalg.norm(ps[0])), "norm_sA": float(np.linalg.norm(ps[1]))})
    geom.append({"branch": "image", "cos_sN_sA": float(ims[0] @ ims[1]), "norm_sN": float(np.linalg.norm(ims[0])), "norm_sA": float(np.linalg.norm(ims[1]))})
    write_csv(OUT / "SOURCE_STATE_GEOMETRY.csv", geom)
    return patch_states, adapter, (source_patch_mem_x, source_patch_mem_y), {"source_images": len(indices)}


def source_memory(dataset, indices, patch):
    mem_path = OUT / "source_model" / "full_visa_instance_memory.npz"
    if mem_path.exists():
        m = np.load(mem_path)
        return m["x"], m["y"], None
    from experiments.phase_xxxviii_visual_state_representation import k0_fps_patch_memory

    x, y, cats = k0_fps_patch_memory(dataset, indices, patch)
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mem_path, x=x.astype(np.float32), y=y.astype(np.int64), cats=cats)
    return x, y, cats


def score_v0(patch, states):
    maps = np.zeros((patch.shape[0], SIDE, SIDE), dtype=np.float16)
    image_scores = np.zeros((patch.shape[0],), dtype=np.float32)
    for i in range(patch.shape[0]):
        s, _ = state_scores(np.asarray(patch[i], dtype=np.float32), states)
        m = s.reshape(SIDE, SIDE).astype(np.float32)
        maps[i] = m.astype(np.float16)
        image_scores[i] = top1_image_score(m)
    return image_scores, maps


def score_v1(patch, adapter, device):
    maps = np.zeros((patch.shape[0], SIDE, SIDE), dtype=np.float16)
    image_scores = np.zeros((patch.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for i in range(patch.shape[0]):
            x = torch.tensor(norm_np(np.asarray(patch[i], dtype=np.float32)), dtype=torch.float32, device=device)
            logits, _ = adapter.logits(x)
            m = torch.sigmoid(logits).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            maps[i] = m.astype(np.float16)
            image_scores[i] = top1_image_score(m)
    return image_scores, maps


def score_v2(patch, mem_x, mem_y):
    maps = np.zeros((patch.shape[0], SIDE, SIDE), dtype=np.float16)
    image_scores = np.zeros((patch.shape[0],), dtype=np.float32)
    for i in range(patch.shape[0]):
        s = instance_memory_scores(np.asarray(patch[i], dtype=np.float32), mem_x, mem_y).reshape(SIDE, SIDE).astype(np.float32)
        maps[i] = s.astype(np.float16)
        image_scores[i] = top1_image_score(s)
    return image_scores, maps


def evaluate_external(dataset, rows, image_scores, maps):
    by_cat = defaultdict(lambda: {"iy": [], "is": [], "py": [], "ps": [], "mask_images": 0})
    all_iy, all_is = [], []
    for pos, row in enumerate(rows):
        item = dataset[pos]
        cat = row["category"]
        label = int(row["label"])
        all_iy.append(label)
        all_is.append(float(image_scores[pos]))
        by_cat[cat]["iy"].append(label)
        by_cat[cat]["is"].append(float(image_scores[pos]))
        if bool(row["mask_available"]):
            mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
            full = upsample_map(np.asarray(maps[pos], dtype=np.float32)).reshape(-1)
            by_cat[cat]["py"].append(mask)
            by_cat[cat]["ps"].append(full)
            by_cat[cat]["mask_images"] += 1
    per, means = {}, defaultdict(list)
    py_all, ps_all = [], []
    for cat, vals in sorted(by_cat.items()):
        if vals["py"]:
            py = np.concatenate(vals["py"])
            ps = np.concatenate(vals["ps"])
            py_all.append(py)
            ps_all.append(ps)
            p_auc = safe_auc(py, ps)
            p_ap = safe_ap(py, ps)
        else:
            p_auc = float("nan")
            p_ap = float("nan")
        row = {
            "I-AUROC": safe_auc(vals["iy"], vals["is"]),
            "I-AP": safe_ap(vals["iy"], vals["is"]),
            "P-AUROC": p_auc,
            "P-AP": p_ap,
            "PRO": float("nan"),
            "mask_images": vals["mask_images"],
        }
        per[cat] = row
        for key in ["I-AUROC", "I-AP", "P-AUROC", "P-AP"]:
            if np.isfinite(row[key]):
                means[key].append(row[key])
    if py_all:
        all_pixel = {"P-AUROC": safe_auc(np.concatenate(py_all), np.concatenate(ps_all)), "P-AP": safe_ap(np.concatenate(py_all), np.concatenate(ps_all))}
    else:
        all_pixel = {"P-AUROC": float("nan"), "P-AP": float("nan")}
    mean = {
        "I-AUROC": safe_auc(all_iy, all_is),
        "I-AP": safe_ap(all_iy, all_is),
        "P-AUROC": all_pixel["P-AUROC"],
        "P-AP": all_pixel["P-AP"],
        "PRO": float("nan"),
    }
    return mean, per


def brain_score_diagnostic(rows, maps):
    out = []
    for label, name in [(0, "normal"), (1, "anomaly")]:
        vals = []
        for i, row in enumerate(rows):
            if int(row["label"]) == label:
                vals.extend(np.asarray(maps[i], dtype=np.float32).reshape(-1).tolist())
        arr = np.asarray(vals, dtype=np.float32)
        if arr.size:
            out.append({"group": name, "median": float(np.median(arr)), "q25": float(np.quantile(arr, 0.25)), "q75": float(np.quantile(arr, 0.75)), "q90": float(np.quantile(arr, 0.90)), "pixels": int(arr.size)})
    return out


def run_target(target_name, dataset, cache_path, states, adapter, memory, device):
    c = np.load(cache_path, allow_pickle=True)
    patch = c["patch"]
    rows = [dict(r) for r in c["rows"]]
    scored = {}
    print(f"[XLV-A] score {target_name} V0", flush=True)
    scored["V0_SINGLE_STATE"] = score_v0(patch, states)
    print(f"[XLV-A] score {target_name} V1", flush=True)
    scored["V1_STATE_ALIGNED"] = score_v1(patch, adapter, device)
    print(f"[XLV-A] score {target_name} V2", flush=True)
    scored["V2_INSTANCE_MEMORY"] = score_v2(patch, memory[0], memory[1])
    metric_rows, per_rows, image_rows, brain_diag = [], [], [], []
    for variant, (image_scores, maps) in scored.items():
        mean, per = evaluate_external(dataset, rows, image_scores, maps)
        mr = {"target": target_name, "variant": variant, **mean}
        mr["G_pixel"] = gp(mr)
        metric_rows.append(mr)
        for cat, vals in per.items():
            pr = {"target": target_name, "category": cat, "variant": variant, **vals}
            pr["G_pixel"] = gp(pr)
            per_rows.append(pr)
        image_rows.append({"target": target_name, "variant": variant, "I-AUROC": mean["I-AUROC"], "I-AP": mean["I-AP"]})
        if target_name == "BrainMRI" and variant == "V1_STATE_ALIGNED":
            brain_diag = brain_score_diagnostic(rows, maps)
    return metric_rows, per_rows, image_rows, brain_diag


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    datasets = {name: ExternalDataset(path, name, mode="test") for name, path in TARGETS.items()}
    protocol = []
    for name, ds in datasets.items():
        masks = sum(bool(ds[i]["mask_available"]) for i in range(len(ds)))
        anomalies = sum(int(ds[i]["anomaly"]) for i in range(len(ds)))
        protocol.append({"target": name, "root": str(TARGETS[name]), "images": len(ds), "anomaly_images": anomalies, "mask_available_images": masks, "test_only": True})
    write_json(OUT / "DATASET_PROTOCOL.json", {"source": "FULL VisA source-labeled meta via existing D2 fold0 cache", "targets": protocol, "target_statistics_used": False})
    states, adapter, memory, source_info = train_full_source_model(device)
    caches = {name: ensure_target_cache(name, ds, device) for name, ds in datasets.items()}

    all_metrics, all_per, image_rows, brain_diag = [], [], [], []
    for name, ds in datasets.items():
        m, p, irows, bdiag = run_target(name, ds, caches[name], states, adapter, memory, device)
        all_metrics.extend(m)
        all_per.extend(p)
        image_rows.extend(irows)
        brain_diag.extend([dict(r, variant="V1_STATE_ALIGNED") for r in bdiag])

    write_csv(OUT / "PILOT_METRICS.csv", all_metrics)
    write_csv(OUT / "IMAGE_METRICS_SECONDARY.csv", image_rows)
    write_csv(OUT / "BRAINMRI_SCORE_DIAGNOSTIC.csv", brain_diag)

    by = {(r["target"], r["variant"]): r for r in all_metrics}
    gains, im_comp = [], []
    for target in TARGETS:
        v0 = by[(target, "V0_SINGLE_STATE")]
        v1 = by[(target, "V1_STATE_ALIGNED")]
        v2 = by[(target, "V2_INSTANCE_MEMORY")]
        mask_ok = np.isfinite(float(v0["G_pixel"])) and np.isfinite(float(v1["G_pixel"]))
        gains.append({
            "target": target,
            "V1_minus_V0_P-AUROC": float(v1["P-AUROC"]) - float(v0["P-AUROC"]) if mask_ok else float("nan"),
            "V1_minus_V0_P-AP": float(v1["P-AP"]) - float(v0["P-AP"]) if mask_ok else float("nan"),
            "V1_minus_V0_G_pixel": gp(v1) - gp(v0) if mask_ok else float("nan"),
            "pixel_gate_evaluable": mask_ok,
        })
        im_comp.append({
            "target": target,
            "V1_minus_V2_P-AUROC": float(v1["P-AUROC"]) - float(v2["P-AUROC"]) if mask_ok and np.isfinite(float(v2["P-AUROC"])) else float("nan"),
            "V1_minus_V2_P-AP": float(v1["P-AP"]) - float(v2["P-AP"]) if mask_ok and np.isfinite(float(v2["P-AP"])) else float("nan"),
            "V1_minus_V2_G_pixel": gp(v1) - gp(v2) if mask_ok and np.isfinite(gp(v2)) else float("nan"),
            "pixel_gate_evaluable": mask_ok,
        })
    write_csv(OUT / "TRANSFER_GAIN.csv", gains)
    write_csv(OUT / "INSTANCE_MEMORY_COMPARISON.csv", im_comp)

    btad_rows = []
    per_by = {(r["target"], r["category"], r["variant"]): r for r in all_per}
    for cat in sorted({r["category"] for r in all_per if r["target"] == "BTAD"}):
        v0 = per_by.get(("BTAD", cat, "V0_SINGLE_STATE"))
        v1 = per_by.get(("BTAD", cat, "V1_STATE_ALIGNED"))
        v2 = per_by.get(("BTAD", cat, "V2_INSTANCE_MEMORY"))
        for v in [v0, v1, v2]:
            if v:
                btad_rows.append(v)
        if v0 and v1:
            btad_rows.append({"target": "BTAD", "category": cat, "variant": "DELTA_V1_MINUS_V0", "P-AUROC": float(v1["P-AUROC"]) - float(v0["P-AUROC"]) if np.isfinite(float(v1["P-AUROC"])) and np.isfinite(float(v0["P-AUROC"])) else float("nan"), "P-AP": float(v1["P-AP"]) - float(v0["P-AP"]) if np.isfinite(float(v1["P-AP"])) and np.isfinite(float(v0["P-AP"])) else float("nan"), "G_pixel": gp(v1) - gp(v0) if np.isfinite(gp(v1)) and np.isfinite(gp(v0)) else float("nan")})
    write_csv(OUT / "BTAD_PER_CATEGORY.csv", btad_rows)

    internal_mean_gain = np.mean([8.370431845186936, 12.881044059101228])
    for r in gains:
        r["internal_mean_gain_reference"] = float(internal_mean_gain)
        r["external_gain_retention"] = float(r["V1_minus_V0_G_pixel"] / internal_mean_gain) if np.isfinite(r["V1_minus_V0_G_pixel"]) else float("nan")
    write_csv(OUT / "TRANSFER_GAIN.csv", gains)

    btad_gain = next(r for r in gains if r["target"] == "BTAD")
    brain_gain = next(r for r in gains if r["target"] == "BrainMRI")
    btad_pass = bool(btad_gain["pixel_gate_evaluable"] and btad_gain["V1_minus_V0_G_pixel"] >= 1.0 and btad_gain["V1_minus_V0_P-AUROC"] >= 0.0)
    brain_pass = bool(brain_gain["pixel_gate_evaluable"] and brain_gain["V1_minus_V0_G_pixel"] >= 1.0 and brain_gain["V1_minus_V0_P-AUROC"] >= 0.0)
    if btad_pass and brain_pass:
        outcome = "XLV_A_CROSS_DOMAIN_TRANSFER_SIGNAL"
    elif btad_pass and not brain_pass:
        outcome = "XLV_A_INDUSTRIAL_ONLY_TRANSFER"
    elif brain_pass and not btad_pass:
        outcome = "XLV_A_DOMAIN_INCONSISTENT_TRANSFER"
    else:
        outcome = "XLV_A_EXTERNAL_TRANSFER_NOT_SUPPORTED"
    memory_competitive = all(r["pixel_gate_evaluable"] and r["V1_minus_V2_G_pixel"] >= -1.0 for r in im_comp if r["target"] == "BTAD")

    report = ["# Phase XLV-A Cross-Domain State Alignment Transfer Pilot", "", f"Outcome: **{outcome}**", "", "## Pilot Metrics", ""]
    for r in all_metrics:
        report.append(f"- {r['target']} {r['variant']}: I-AUROC {float(r['I-AUROC']):.4f}, I-AP {float(r['I-AP']):.4f}, P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, G {float(r['G_pixel']):.4f}.")
    report.extend(["", "## Transfer Gain", ""])
    for r in gains:
        report.append(f"- {r['target']}: V1-V0 G {r['V1_minus_V0_G_pixel']:+.4f}, P-AUROC {r['V1_minus_V0_P-AUROC']:+.4f}, P-AP {r['V1_minus_V0_P-AP']:+.4f}, evaluable `{r['pixel_gate_evaluable']}`.")
    report.extend(["", "## Notes", ""])
    report.append("- BrainMRI has no pixel masks in meta.json; pixel gate is not evaluable and no pseudo-mask was used.")
    report.append("- BTAD masks are evaluated from provided ground_truth files; category 01 masks were resolved by matching anomaly image basename to ground_truth/ko.")
    report.append("Stop: Phase XLV-A Batch 1 ended after BTAD and BrainMRI.")
    report.append(f"Runtime: {time.time() - start:.1f}s.")
    (OUT / "XLV_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {
        "phase": "XLV-A",
        "outcome": outcome,
        "BTAD_pass": btad_pass,
        "BrainMRI_pass": brain_pass,
        "BrainMRI_pixel_masks_available": False,
        "EXTERNAL_MEMORY_COMPETITIVE_on_evaluable_targets": memory_competitive,
        "runtime_sec": time.time() - start,
    })
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLV-A Cross-Domain State Alignment Transfer Pilot\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- Transfer gain: {json.dumps(gains, default=json_default)}.\n")
        fp.write(f"- InstanceMemory comparison: {json.dumps(im_comp, default=json_default)}.\n")
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
