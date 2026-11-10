import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_xlv_cross_domain_transfer import score_v0, score_v1, score_v2, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np, upsample_map
from phase_xa.official_task_forward import normalized_image_feature, normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


DATA_ROOT = Path(r"E:\WorkSpace\dataset")
XLV_A = Path("results/phase_xlv_cross_domain_transfer")
OUT = Path("results/phase_xlv_medical_pixel_transfer")
CHECKPOINT = Path("checkpoints/test_on_mvtec.pth")
EVAL_DATASETS = {
    "ISBI": DATA_ROOT / "ISBI",
    "CVC-ClinicDB": DATA_ROOT / "CVC-ClinicDB",
}
MEDICAL_CANDIDATES = ["BrainMRI", "Br35H", "HeadCT", "ISBI", "CVC-ClinicDB", "CVC-ColonDB", "Kvasir", "Endo"]


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
    if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])):
        return 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"]))
    return float("nan")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def audit_dataset(name, root):
    meta = root / "meta.json"
    if not meta.exists():
        return {"dataset": name, "local_path": str(root), "meta_exists": False, "loader_pixel_ready": False}
    m = json.loads(meta.read_text(encoding="utf-8"))
    rows = []
    for mode, cats in m.items():
        for cls, items in cats.items():
            for x in items:
                rows.append((mode, cls, x))
    image_count = len(rows)
    anomaly = sum(int(x.get("anomaly", 0)) for _mode, _cls, x in rows)
    normal = image_count - anomaly
    mask_entries = [x for _mode, _cls, x in rows if int(x.get("anomaly", 0)) and x.get("mask_path")]
    valid_masks = 0
    mask_resolution = ""
    for x in mask_entries:
        p = root / x["mask_path"]
        if p.exists() and p.is_file():
            valid_masks += 1
            if not mask_resolution:
                with Image.open(p) as im:
                    mask_resolution = f"{im.size[0]}x{im.size[1]}"
    return {
        "dataset": name,
        "local_path": str(root),
        "canonical_name": name,
        "meta_exists": True,
        "image_count": image_count,
        "normal_image_count": normal,
        "anomaly_image_count": anomaly,
        "mask_file_count": valid_masks,
        "mask_resolution": mask_resolution,
        "image_mask_correspondence": "valid" if valid_masks == len(mask_entries) and valid_masks > 0 else ("no_masks" if valid_masks == 0 else "partial"),
        "meta_fields": ",".join(sorted({k for _m, _c, x in rows for k in x.keys()})),
        "loader_returns_image_label_mask": bool(valid_masks > 0),
        "loader_pixel_ready": bool(valid_masks > 0),
    }


def brain_mask_audit():
    root = DATA_ROOT / "BrainMRI"
    candidates = []
    for p in root.rglob("*"):
        if p.is_dir() and any(s in p.name.lower() for s in ["mask", "gt", "ground", "seg", "annotation"]):
            candidates.append(str(p))
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    anom_masks = sum(bool(x.get("mask_path")) for v in meta["test"].values() for x in v if int(x.get("anomaly", 0)))
    if candidates and anom_masks == 0:
        conclusion = "BRAINMRI_MASK_MAPPING_INVALID"
    elif anom_masks > 0:
        conclusion = "BRAINMRI_MASK_AVAILABLE_AND_LOADER_MISSED"
    else:
        conclusion = "BRAINMRI_LOCAL_PACKAGE_NO_PIXEL_MASK"
    return {"local_path": str(root), "candidate_mask_dirs": candidates, "anomaly_mask_entries_in_meta": anom_masks, "conclusion": conclusion}


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="test"):
        self.root = Path(root)
        self.preprocess, self.target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))[mode]
        self.data_all = []
        for cls in sorted(meta):
            self.data_all.extend(meta[cls])

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, idx):
        x = self.data_all[idx]
        img = Image.open(self.root / x["img_path"]).convert("RGB")
        mask_path = x.get("mask_path", "")
        if mask_path and (self.root / mask_path).exists():
            mask = Image.open(self.root / mask_path).convert("L")
            mask_available = True
        else:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
            mask_available = int(x.get("anomaly", 0)) == 0
        return {
            "img": self.preprocess(img),
            "img_mask": self.target_transform(mask),
            "cls_name": str(x["cls_name"]),
            "anomaly": int(x["anomaly"]),
            "img_path": str(self.root / x["img_path"]),
            "mask_available": mask_available,
        }


@torch.no_grad()
def extract_clip_one(model, item, device):
    image = item["img"].unsqueeze(0).to(device)
    out = visual_outputs(model, image, adapter=None, grad=False)
    cls = normalized_image_feature(out).squeeze(0).float().cpu().numpy().astype(np.float32)
    patch = normalized_patch_feature(out).squeeze(0).float().cpu().numpy().astype(np.float32)
    return cls, patch


def ensure_cache(name, dataset, device):
    path = OUT / "feature_cache" / f"{name}_clip.npz"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    model, *_ = load_components(str(CHECKPOINT), device)
    cls_list, patch_list, rows = [], [], []
    for i in range(len(dataset)):
        if i % 50 == 0:
            print(f"[XLV-B] extract {name} {i}/{len(dataset)}", flush=True)
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
    np.savez_compressed(path, cls=np.stack(cls_list), patch=np.stack(patch_list), rows=np.asarray(rows, dtype=object))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def load_source_model(device):
    states_npz = np.load(XLV_A / "source_model" / "full_visa_states.npz")
    states = states_npz["patch_states"]
    adapter = AdapterToState(1024, states).to(device).eval()
    payload = torch.load(XLV_A / "source_model" / "full_visa_adapter.pt", map_location=device)
    adapter.load_state_dict(payload["model"], strict=True)
    mem = np.load(XLV_A / "source_model" / "full_visa_instance_memory.npz")
    return states, adapter, (mem["x"], mem["y"])


def eval_scores(dataset, rows, image_scores, maps):
    py_all, ps_all, iy, iscore = [], [], [], []
    per_image = []
    for i, row in enumerate(rows):
        item = dataset[i]
        iy.append(int(row["label"]))
        iscore.append(float(image_scores[i]))
        if bool(row["mask_available"]):
            mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
            full = upsample_map(np.asarray(maps[i], dtype=np.float32)).reshape(-1)
            py_all.append(mask)
            ps_all.append(full)
            if int(row["label"]) == 1 and np.unique(mask).size == 2:
                per_image.append({"P-AUROC": safe_auc(mask, full), "P-AP": safe_ap(mask, full)})
    py = np.concatenate(py_all) if py_all else np.asarray([], dtype=np.uint8)
    ps = np.concatenate(ps_all) if ps_all else np.asarray([], dtype=np.float32)
    return {
        "I-AUROC": safe_auc(iy, iscore),
        "I-AP": safe_ap(iy, iscore),
        "P-AUROC": safe_auc(py, ps),
        "P-AP": safe_ap(py, ps),
        "PRO": float("nan"),
        "mask_images": len(py_all),
    }, per_image, (py, ps)


def score_distribution(target, variant, py, ps):
    rows = []
    normal = ps[py == 0]
    anomaly = ps[py == 1]
    for name, vals in [("normal_pixel", normal), ("anomaly_pixel", anomaly)]:
        rows.append({
            "target": target,
            "variant": variant,
            "group": name,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, 0.25)),
            "q75": float(np.quantile(vals, 0.75)),
            "q90": float(np.quantile(vals, 0.90)),
            "pixels": int(vals.size),
        })
    pooled = np.sqrt(0.5 * (np.var(normal) + np.var(anomaly)))
    rows.append({
        "target": target,
        "variant": variant,
        "group": "separation",
        "mean": float((np.mean(anomaly) - np.mean(normal)) / max(pooled, 1e-12)),
        "median": float(ks_2samp(normal, anomaly).statistic),
        "q25": float("nan"),
        "q75": float("nan"),
        "q90": float("nan"),
        "pixels": int(normal.size + anomaly.size),
    })
    return rows


def run_target(name, dataset, cache_path, states, adapter, memory, device):
    c = np.load(cache_path, allow_pickle=True)
    patch = c["patch"]
    rows = [dict(r) for r in c["rows"]]
    scored = {}
    print(f"[XLV-B] score {name} V0", flush=True)
    scored["V0_SINGLE_STATE"] = score_v0(patch, states)
    print(f"[XLV-B] score {name} V1", flush=True)
    scored["V1_STATE_ALIGNED"] = score_v1(patch, adapter, device)
    print(f"[XLV-B] score {name} V2", flush=True)
    scored["V2_INSTANCE_MEMORY"] = score_v2(patch, memory[0], memory[1])
    metrics, dist_rows, per_image_rows = [], [], []
    per_image_by_variant = {}
    for variant, (image_scores, maps) in scored.items():
        metric, per_img, pixel_pair = eval_scores(dataset, rows, image_scores, maps)
        metric = {"target": name, "variant": variant, **metric}
        metric["G_pixel"] = gp(metric)
        metrics.append(metric)
        dist_rows.extend(score_distribution(name, variant, pixel_pair[0], pixel_pair[1]))
        per_image_by_variant[variant] = per_img
    v0 = per_image_by_variant["V0_SINGLE_STATE"]
    v1 = per_image_by_variant["V1_STATE_ALIGNED"]
    n = min(len(v0), len(v1))
    if n:
        delta_auc = np.asarray([v1[i]["P-AUROC"] - v0[i]["P-AUROC"] for i in range(n)])
        delta_ap = np.asarray([v1[i]["P-AP"] - v0[i]["P-AP"] for i in range(n)])
        per_image_rows.append({
            "target": name,
            "anomaly_image_count": n,
            "delta_P-AUROC_median": float(np.median(delta_auc)),
            "delta_P-AUROC_q25": float(np.quantile(delta_auc, 0.25)),
            "delta_P-AUROC_q75": float(np.quantile(delta_auc, 0.75)),
            "delta_P-AP_median": float(np.median(delta_ap)),
            "delta_P-AP_q25": float(np.quantile(delta_ap, 0.25)),
            "delta_P-AP_q75": float(np.quantile(delta_ap, 0.75)),
            "V1_gt_V0_P-AUROC_ratio": float(np.mean(delta_auc > 0)),
            "V1_gt_V0_P-AP_ratio": float(np.mean(delta_ap > 0)),
        })
    return metrics, dist_rows, per_image_rows


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    audits = [audit_dataset(name, DATA_ROOT / name) for name in MEDICAL_CANDIDATES if (DATA_ROOT / name).exists()]
    write_csv(OUT / "MEDICAL_DATASET_AUDIT.csv", audits)
    brain_audit = brain_mask_audit()
    write_json(OUT / "BRAINMRI_MASK_AUDIT.json", brain_audit)
    valid_names = [name for name in EVAL_DATASETS if any(r["dataset"] == name and r.get("loader_pixel_ready") for r in audits)]
    if not valid_names:
        outcome = "XLV_B_MEDICAL_DATA_INTEGRITY_FAILURE"
        write_json(OUT / "OUTCOME.json", {"phase": "XLV-B", "outcome": outcome})
        return
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    states, adapter, memory = load_source_model(device)
    model_files = [XLV_A / "source_model" / "full_visa_states.npz", XLV_A / "source_model" / "full_visa_adapter.pt", XLV_A / "source_model" / "full_visa_instance_memory.npz"]
    reuse = [{"artifact": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in model_files]
    write_json(OUT / "SOURCE_MODEL_REUSE_AUDIT.json", {"source": "XLV-A FULL VisA frozen source model", "artifacts": reuse, "target_data_used": False})
    datasets = {name: MedicalDataset(EVAL_DATASETS[name], mode="test") for name in valid_names}
    caches = {name: ensure_cache(name, ds, device) for name, ds in datasets.items()}
    raw, dist, per_image = [], [], []
    for name, ds in datasets.items():
        m, d, p = run_target(name, ds, caches[name], states, adapter, memory, device)
        raw.extend(m); dist.extend(d); per_image.extend(p)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PIXEL_SCORE_DISTRIBUTION.csv", dist)
    write_csv(OUT / "PER_IMAGE_ROBUSTNESS.csv", per_image)
    by = {(r["target"], r["variant"]): r for r in raw}
    gains, im = [], []
    for name in valid_names:
        v0 = by[(name, "V0_SINGLE_STATE")]
        v1 = by[(name, "V1_STATE_ALIGNED")]
        v2 = by[(name, "V2_INSTANCE_MEMORY")]
        gain = {
            "target": name,
            "V1_minus_V0_P-AUROC": float(v1["P-AUROC"]) - float(v0["P-AUROC"]),
            "V1_minus_V0_P-AP": float(v1["P-AP"]) - float(v0["P-AP"]),
            "V1_minus_V0_G_pixel": gp(v1) - gp(v0),
            "medical_pixel_transfer_signal": bool((gp(v1) - gp(v0)) >= 1.0 and (float(v1["P-AUROC"]) - float(v0["P-AUROC"])) >= 0 and (float(v1["P-AP"]) - float(v0["P-AP"])) > 0),
            "internal_gain_mean_reference": 10.625737952144082,
            "medical_gain_retention": float((gp(v1) - gp(v0)) / 10.625737952144082),
        }
        gains.append(gain)
        im.append({
            "target": name,
            "V1_minus_V2_P-AUROC": float(v1["P-AUROC"]) - float(v2["P-AUROC"]),
            "V1_minus_V2_P-AP": float(v1["P-AP"]) - float(v2["P-AP"]),
            "V1_minus_V2_G_pixel": gp(v1) - gp(v2),
        })
    write_csv(OUT / "MEDICAL_TRANSFER_GAIN.csv", gains)
    write_csv(OUT / "INSTANCE_MEMORY_COMPARISON.csv", im)
    pass_count = sum(bool(r["medical_pixel_transfer_signal"]) for r in gains)
    if len(gains) >= 2 and pass_count == len(gains):
        outcome = "XLV_B_CROSS_DOMAIN_PIXEL_TRANSFER_CONFIRMED"
    elif len(gains) == 1 and pass_count == 1:
        outcome = "XLV_B_MEDICAL_PIXEL_SIGNAL_PRELIMINARY"
    elif 0 < pass_count < len(gains):
        outcome = "XLV_B_MEDICAL_TRANSFER_DATASET_DEPENDENT"
    else:
        outcome = "XLV_B_MEDICAL_PIXEL_TRANSFER_NOT_SUPPORTED"
    report = ["# Phase XLV-B Medical Pixel Transfer Resolution", "", f"Outcome: **{outcome}**", "", "## Dataset Audit", ""]
    for r in audits:
        report.append(f"- {r['dataset']}: images {r.get('image_count')}, anomaly {r.get('anomaly_image_count')}, masks {r.get('mask_file_count')}, ready `{r.get('loader_pixel_ready')}`.")
    report.extend(["", "## Metrics", ""])
    for r in raw:
        report.append(f"- {r['target']} {r['variant']}: P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, G {float(r['G_pixel']):.4f}, I-AUROC {float(r['I-AUROC']):.4f}.")
    report.extend(["", "## Transfer Gain", ""])
    for r in gains:
        report.append(f"- {r['target']}: V1-V0 G {r['V1_minus_V0_G_pixel']:+.4f}, P-AUROC {r['V1_minus_V0_P-AUROC']:+.4f}, P-AP {r['V1_minus_V0_P-AP']:+.4f}, pass `{r['medical_pixel_transfer_signal']}`.")
    report.append("")
    report.append(f"BrainMRI audit: {brain_audit['conclusion']}.")
    report.append("Stop: XLV-B evaluated only existing legal medical pixel datasets and reused the XLV-A source model.")
    report.append(f"Runtime: {time.time() - start:.1f}s.")
    (OUT / "XLV_B_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {"phase": "XLV-B", "outcome": outcome, "evaluated_datasets": valid_names, "pass_count": pass_count, "runtime_sec": time.time() - start})
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLV-B Medical Pixel Transfer Resolution\n")
        fp.write(f"- BrainMRI mask audit: {json.dumps(brain_audit, default=json_default)}.\n")
        fp.write(f"- Evaluated datasets: {valid_names}.\n")
        fp.write(f"- Medical transfer gain: {json.dumps(gains, default=json_default)}.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
