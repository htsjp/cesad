from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
XXVI_OFFICIAL = ROOT.parents[0] / "reference_repos" / "phase_VI_A" / "AnomalyCLIP"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import packaging as _packaging
import packaging.version as _packaging_version
import pkg_resources as _pkg_resources

_packaging.version = _packaging_version
if not hasattr(_pkg_resources, "packaging"):
    _pkg_resources.packaging = _packaging

IMPORT_MODE = os.environ.get("XXVIII_IMPORT_MODE", "project").strip().lower()
if IMPORT_MODE == "official":
    sys.path.insert(0, str(XXVI_OFFICIAL))
    import AnomalyCLIP_lib  # noqa: E402
    from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402
    sys.path = [p for p in sys.path if p != str(XXVI_OFFICIAL)]
else:
    import AnomalyCLIP_lib  # noqa: E402
    AnomalyCLIP_PromptLearner = None

_anomaly_utils_spec = importlib.util.spec_from_file_location("aaclip_official_utils", XXVI_OFFICIAL / "utils.py")
_anomaly_utils = importlib.util.module_from_spec(_anomaly_utils_spec)
assert _anomaly_utils_spec.loader is not None
_anomaly_utils_spec.loader.exec_module(_anomaly_utils)
anomaly_get_transform = _anomaly_utils.get_transform

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import to_device_cache
from experiments.phase_xxiii_b_dual_carrier_local import (
    IMAGE_SIZE,
    SIGMA,
    apply_cdf,
    official_raw_map_score,
    smooth_full,
)
from phase_xa.teacher_memory_builder import build_memory_from_indices
from utils.metrics import pixel_level_metrics
from utils.transforms import get_transform as mrad_get_transform


DATA_ROOT = Path(r"E:\WorkSpace\dataset")
OUT = ROOT / "results" / "phase_xxviii_multidataset"
AUDIT = OUT / "audit"
BATCH = OUT / "batch_a"
CELL_DIR = BATCH / "cell_results"

MRAD_XXIV = ROOT / "results" / "phase_xxiv_pixel_calibration" / "final_target"
AC_XXVI = ROOT / "results" / "phase_xxvi_cross_model" / "anomalyclip"


def jdefault(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(type(obj).__name__)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=jdefault) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not-a-git-repo"


def class_info(dataset_name: str, classes: list[str]) -> tuple[list[str], dict[str, int]]:
    known = {
        "mvtec": ["carpet", "bottle", "hazelnut", "leather", "cable", "capsule", "grid", "pill", "transistor", "metal_nut", "screw", "toothbrush", "zipper", "tile", "wood"],
        "visa": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
        "MPDD": ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"],
        "BTAD": ["01", "02", "03"],
        "DTD": ["Woven_001", "Woven_127", "Woven_104", "Stratified_154", "Blotchy_099", "Woven_068", "Woven_125", "Marbled_078", "Perforated_037", "Mesh_114", "Fibrous_183", "Matted_069"],
    }
    obj_list = known.get(dataset_name, classes)
    return obj_list, {name: i for i, name in enumerate(obj_list)}


class MetaDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path | str, transform, target_transform, dataset_name: str, mode: str = "test"):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        mode_meta = meta[mode]
        self.cls_names = list(mode_meta.keys())
        self.data_all = []
        for cls_name in self.cls_names:
            for row in mode_meta[cls_name]:
                row = dict(row)
                row.setdefault("specie_name", row.get("defect_name", row.get("type", "")))
                self.data_all.append(row)
        self.length = len(self.data_all)
        self.obj_list, self.class_name_map_class_id = class_info(dataset_name, self.cls_names)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path = data["img_path"]
        mask_path = data.get("mask_path", "")
        cls_name = data["cls_name"]
        anomaly = int(data["anomaly"])
        img = Image.open(self.root / img_path).convert("RGB")
        if anomaly == 0:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        else:
            mp = self.root / mask_path
            if mask_path and mp.exists() and not mp.is_dir():
                mask_arr = np.array(Image.open(mp).convert("L")) > 0
                mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
            else:
                mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        img_t = self.transform(img) if self.transform is not None else img
        mask_t = self.target_transform(mask) if self.target_transform is not None else mask
        return {
            "img": img_t,
            "img_mask": mask_t,
            "cls_name": cls_name,
            "anomaly": anomaly,
            "img_path": str(self.root / img_path),
            "cls_id": self.class_name_map_class_id.get(cls_name, 0),
        }


class InsPLADDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path | str, transform, target_transform):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.obj_list = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_name_map_class_id = {name: i for i, name in enumerate(self.obj_list)}
        self.data_all = []
        for cls_name in self.obj_list:
            cat = self.root / cls_name
            for sub, anomaly in [("good", 0), ("anomaly", 1)]:
                for img in sorted((cat / "test" / sub).glob("*")):
                    if not img.is_file():
                        continue
                    mask = cat / "ground_truth" / "anomaly" / f"{img.stem}_mask.png" if anomaly else Path("")
                    self.data_all.append({"img_path": img, "mask_path": mask, "cls_name": cls_name, "anomaly": anomaly})
        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img = Image.open(data["img_path"]).convert("RGB")
        if int(data["anomaly"]) == 0:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        else:
            mask_path = Path(data["mask_path"])
            if not mask_path.exists():
                matches = sorted(mask_path.parent.glob(mask_path.stem + ".*"))
                if not matches:
                    raise FileNotFoundError(str(mask_path))
                mask_path = matches[0]
            mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
            mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
        img_t = self.transform(img) if self.transform is not None else img
        mask_t = self.target_transform(mask) if self.target_transform is not None else mask
        cls_name = data["cls_name"]
        return {
            "img": img_t,
            "img_mask": mask_t,
            "cls_name": cls_name,
            "anomaly": int(data["anomaly"]),
            "img_path": str(data["img_path"]),
            "cls_id": self.class_name_map_class_id.get(cls_name, 0),
        }


def load_mrad_dataset(dataset_name: str, root: Path | str, mode: str = "test"):
    args = argparse.Namespace(image_size=IMAGE_SIZE)
    preprocess, target_transform = mrad_get_transform(args)
    if dataset_name == "InsPLAD_AD":
        return InsPLADDataset(root, preprocess, target_transform)
    return MetaDataset(root, preprocess, target_transform, dataset_name, mode=mode)


def load_anomaly_dataset(dataset_name: str, root: Path | str, mode: str = "test"):
    class Args:
        image_size = IMAGE_SIZE
    preprocess, target_transform = anomaly_get_transform(Args())
    if dataset_name == "InsPLAD_AD":
        return InsPLADDataset(root, preprocess, target_transform)
    return MetaDataset(root, preprocess, target_transform, dataset_name, mode=mode)


def candidate_roots() -> dict[str, Path]:
    return {
        "MPDD": DATA_ROOT / "MPDD",
        "BTAD": DATA_ROOT / "btad" / "BTech_Dataset_transformed",
        "DTD": DATA_ROOT / "DTD-Synthetic" / "DTD-Synthetic" / "DTD-Synthetic",
        "InsPLAD_AD": DATA_ROOT / "InsPLAD_AD",
        "Br35H": DATA_ROOT / "Br35H",
        "BrainMRI": DATA_ROOT / "BrainMRI",
        "HeadCT": DATA_ROOT / "HeadCT",
        "ISBI": DATA_ROOT / "ISBI",
        "Endo": DATA_ROOT / "Endo",
        "CVC-ClinicDB": DATA_ROOT / "CVC-ClinicDB",
        "CVC-ColonDB": DATA_ROOT / "CVC-ColonDB",
        "Kvasir": DATA_ROOT / "Kvasir",
        "MVTec-AD": DATA_ROOT / "mvtec",
        "VisA": DATA_ROOT / "VISA",
        "VisA-pytorch": DATA_ROOT / "VISA_pytorch",
    }


def inspect_meta_dataset(name: str, root: Path) -> dict:
    if not (root / "meta.json").exists():
        return inspect_directory_dataset(name, root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    test = meta.get("test", {})
    categories = list(test.keys())
    normal = anomaly = missing_masks = 0
    img_ext, mask_ext, sizes = set(), set(), []
    for items in test.values():
        for row in items:
            a = int(row.get("anomaly", 0))
            anomaly += a
            normal += int(a == 0)
            ip = root / row.get("img_path", "")
            mp = root / row.get("mask_path", "")
            img_ext.add(ip.suffix.lower())
            if len(sizes) < 5 and ip.exists():
                try:
                    sizes.append(Image.open(ip).size)
                except Exception:
                    pass
            if a:
                if row.get("mask_path") and mp.exists() and not mp.is_dir():
                    mask_ext.add(mp.suffix.lower())
                else:
                    missing_masks += 1
    return {
        "dataset": name,
        "local_path": str(root),
        "has_meta_json": True,
        "categories": len(categories),
        "category_names": ";".join(categories),
        "normal_image_count": normal,
        "anomaly_image_count": anomaly,
        "pixel_gt_masks_exist": missing_masks == 0 and anomaly > 0,
        "missing_anomaly_masks": missing_masks,
        "mask_format": ";".join(sorted(mask_ext)),
        "image_format": ";".join(sorted(img_ext)),
        "image_size_range": ";".join(map(str, sorted(set(sizes)))),
        "official_structure": "meta.json train/test with category/test/good|anomaly or defect folders",
        "reliable_binary_mask": missing_masks == 0 and anomaly > 0,
        "estimated_evaluation_images": normal + anomaly,
        "estimated_inference_cost": "medium" if normal + anomaly < 800 else "high",
    }


def inspect_directory_dataset(name: str, root: Path) -> dict:
    categories = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    normal = anomaly = masks = 0
    img_ext, mask_ext, sizes = set(), set(), []
    for cat in categories:
        for folder, is_anom in [("test/good", False), ("test/anomaly", True)]:
            for f in (cat / folder).glob("*"):
                if not f.is_file():
                    continue
                img_ext.add(f.suffix.lower())
                if is_anom:
                    anomaly += 1
                else:
                    normal += 1
                if len(sizes) < 5:
                    try:
                        sizes.append(Image.open(f).size)
                    except Exception:
                        pass
        gt = cat / "ground_truth" / "anomaly"
        for f in gt.glob("*"):
            if f.is_file():
                masks += 1
                mask_ext.add(f.suffix.lower())
    return {
        "dataset": name,
        "local_path": str(root),
        "has_meta_json": False,
        "categories": len(categories),
        "category_names": ";".join(p.name for p in categories),
        "normal_image_count": normal,
        "anomaly_image_count": anomaly,
        "pixel_gt_masks_exist": masks >= anomaly and anomaly > 0,
        "missing_anomaly_masks": max(0, anomaly - masks),
        "mask_format": ";".join(sorted(mask_ext)),
        "image_format": ";".join(sorted(img_ext)),
        "image_size_range": ";".join(map(str, sorted(set(sizes)))),
        "official_structure": "directory train/test plus ground_truth; no meta.json",
        "reliable_binary_mask": masks >= anomaly and anomaly > 0,
        "estimated_evaluation_images": normal + anomaly,
        "estimated_inference_cost": "medium" if normal + anomaly < 800 else "high",
    }


def eligibility(row: dict) -> dict:
    name = row["dataset"]
    excluded = name in {"MVTec-AD", "VisA", "VisA-pytorch"}
    industrial = name in {"MPDD", "BTAD", "DTD", "InsPLAD_AD"}
    real_images = name != "DTD"
    eligible = (
        not excluded
        and industrial
        and real_images
        and bool(row["pixel_gt_masks_exist"])
        and row["normal_image_count"] > 0
        and row["anomaly_image_count"] > 0
    )
    reason = []
    if excluded:
        reason.append("excluded development/previous target dataset")
    if not industrial:
        reason.append("not an industrial visual anomaly dataset for Phase XXVIII primary set")
    if not real_images:
        reason.append("synthetic texture dataset, fails real-image criterion")
    if not row["pixel_gt_masks_exist"]:
        reason.append("pixel masks unavailable or not resolved")
    if row["normal_image_count"] == 0 or row["anomaly_image_count"] == 0:
        reason.append("normal/anomalous test split incomplete")
    if eligible:
        reason.append("eligible")
    out = dict(row)
    out["eligible_for_xxviii_a"] = eligible
    out["eligibility_reason"] = "; ".join(reason)
    return out


def create_meta_for_insplad(root: Path) -> Path:
    meta_path = BATCH / "generated_meta" / "InsPLAD_AD" / "meta.json"
    if meta_path.exists():
        return meta_path.parent
    out_root = meta_path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {"test": {}, "train": {}}
    for cat in sorted([p for p in root.iterdir() if p.is_dir()]):
        test_rows = []
        for sub, anomaly in [("good", 0), ("anomaly", 1)]:
            for img in sorted((cat / "test" / sub).glob("*")):
                if not img.is_file():
                    continue
                rel_img = img.relative_to(root).as_posix()
                if anomaly:
                    mask = root / cat.name / "ground_truth" / "anomaly" / img.name
                    rel_mask = mask.relative_to(root).as_posix()
                else:
                    rel_mask = ""
                test_rows.append({"img_path": rel_img, "mask_path": rel_mask, "cls_name": cat.name, "specie_name": sub, "anomaly": anomaly})
        train_rows = []
        for img in sorted((cat / "train" / "good").glob("*")):
            if img.is_file():
                train_rows.append({"img_path": img.relative_to(root).as_posix(), "mask_path": "", "cls_name": cat.name, "specie_name": "good", "anomaly": 0})
        meta["test"][cat.name] = test_rows
        meta["train"][cat.name] = train_rows
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_root


def run_audit() -> tuple[list[dict], list[dict], list[str]]:
    AUDIT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name, root in candidate_roots().items():
        if root.exists():
            inventory.append(inspect_meta_dataset(name, root))
    eligibility_rows = [eligibility(r) for r in inventory]
    eligibles = [r for r in eligibility_rows if r["eligible_for_xxviii_a"]]
    eligibles.sort(key=lambda r: (-int(r["categories"]), -int(r["anomaly_image_count"]), int(r["estimated_evaluation_images"])))
    selected = [r["dataset"] for r in eligibles[:2]]
    write_csv(AUDIT / "LOCAL_DATASET_INVENTORY.csv", inventory)
    write_csv(AUDIT / "DATASET_ELIGIBILITY.csv", eligibility_rows)
    protocol = {
        "phase": "XXVIII-0",
        "data_root": str(DATA_ROOT),
        "selection_rule": "If >=3 eligible datasets exist, choose exactly two before inference by category count, anomalous image count, clean masks, and reasonable cost.",
        "selected_datasets": selected,
        "candidates": eligibility_rows,
        "excluded_development_datasets": ["MVTec-AD", "VisA"],
        "performance_seen_before_selection": False,
    }
    write_json(AUDIT / "DATASET_SELECTION_PROTOCOL.json", protocol)
    evaluator = {
        "phase": "XXVIII",
        "primary_metrics": ["Pixel AUROC", "PRO/AUPRO"],
        "pixel_auc": "sklearn.metrics.roc_auc_score over concatenated binary masks and raw aligned maps per category, then macro mean over categories",
        "pro": "utils.metrics.pixel_level_metrics(..., pixel-aupro), same implementation as frozen MRAD/AnomalyCLIP audits",
        "mask_convention": "anomaly pixels are mask > 0 after nearest-neighbor target transform to 518x518",
        "map_resolution": IMAGE_SIZE,
        "map_alignment": "detector map and mask both 518x518; no target min/max normalization",
        "t0_t1_pairing": "same raw map and same native image score; T1 applies frozen source-normal empirical CDF before Gaussian sigma=4",
        "frozen_before_results": True,
    }
    write_json(AUDIT / "EVALUATOR_PROTOCOL.json", evaluator)
    return inventory, eligibility_rows, selected


def artifact_audit() -> dict:
    files = {
        "MRAD_D1_CDF": MRAD_XXIV / "D1_PIXEL_CDF.npz",
        "MRAD_D2_CDF": MRAD_XXIV / "D2_PIXEL_CDF.npz",
        "AnomalyCLIP_D1_CDF": AC_XXVI / "D1_ANOMALYCLIP_PIXEL_CDF.npz",
        "AnomalyCLIP_D2_CDF": AC_XXVI / "D2_ANOMALYCLIP_PIXEL_CDF.npz",
        "MRAD_D1_checkpoint": ROOT / "checkpoints" / "test_on_visa.pth",
        "MRAD_D2_checkpoint": ROOT / "checkpoints" / "test_on_mvtec.pth",
        "AnomalyCLIP_D1_checkpoint": XXVI_OFFICIAL / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth",
        "AnomalyCLIP_D2_checkpoint": XXVI_OFFICIAL / "checkpoints" / "9_12_4_multiscale" / "epoch_15.pth",
    }
    rows = []
    ok = True
    for key, path in files.items():
        exists = path.exists()
        ok = ok and exists
        rows.append({"artifact": key, "path": str(path), "exists": exists, "bytes": path.stat().st_size if exists else 0, "sha256": sha256_file(path) if exists else ""})
    audit = {"phase": "XXVIII", "code_commit": git_hash(), "artifacts_ok": ok, "artifacts": rows}
    write_json(AUDIT / "FROZEN_ARTIFACT_AUDIT.json", audit)
    return audit


def source_spec(system: str) -> dict:
    if system == "M-MVTec":
        return {"model": "MRAD", "source": "MVTec", "dataset": "mvtec", "path": ROOT / "data" / "mvtec", "checkpoint": ROOT / "checkpoints" / "test_on_visa.pth", "cdf": MRAD_XXIV / "D1_PIXEL_CDF.npz"}
    if system == "M-VisA":
        return {"model": "MRAD", "source": "VisA", "dataset": "visa", "path": ROOT / "data" / "VISA", "checkpoint": ROOT / "checkpoints" / "test_on_mvtec.pth", "cdf": MRAD_XXIV / "D2_PIXEL_CDF.npz"}
    if system == "A-MVTec":
        return {"model": "AnomalyCLIP", "source": "MVTec", "dataset": "mvtec", "path": ROOT / "data" / "mvtec", "checkpoint": XXVI_OFFICIAL / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth", "cdf": AC_XXVI / "D1_ANOMALYCLIP_PIXEL_CDF.npz"}
    if system == "A-VisA":
        return {"model": "AnomalyCLIP", "source": "VisA", "dataset": "visa", "path": ROOT / "data" / "VISA", "checkpoint": XXVI_OFFICIAL / "checkpoints" / "9_12_4_multiscale" / "epoch_15.pth", "cdf": AC_XXVI / "D2_ANOMALYCLIP_PIXEL_CDF.npz"}
    raise KeyError(system)


def target_spec(name: str) -> dict:
    if name == "MPDD":
        return {"dataset": "MPDD", "path": DATA_ROOT / "MPDD"}
    if name == "BTAD":
        return {"dataset": "BTAD", "path": DATA_ROOT / "btad" / "BTech_Dataset_transformed"}
    if name == "InsPLAD_AD":
        return {"dataset": "InsPLAD_AD", "path": DATA_ROOT / "InsPLAD_AD"}
    if name == "DTD":
        return {"dataset": "DTD", "path": DATA_ROOT / "DTD-Synthetic" / "DTD-Synthetic" / "DTD-Synthetic"}
    raise KeyError(name)


def load_source_dataset(spec: dict):
    return load_mrad_dataset(spec["dataset"], spec["path"], mode="test")


def all_indices(dataset) -> list[int]:
    return list(range(len(dataset)))


def normal_indices(dataset) -> list[int]:
    return [i for i in range(len(dataset)) if int(dataset[i]["anomaly"]) == 0]


def mask_array(item) -> np.ndarray:
    return (item["img_mask"].detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8)


def safe_pro_score(masks: np.ndarray, amaps: np.ndarray, max_step: int = 200, expect_fpr: float = 0.3) -> float:
    masks = np.asarray(masks)
    amaps = np.asarray(amaps, dtype=np.float32)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if amaps.ndim == 4:
        amaps = amaps.squeeze(1)
    min_th, max_th = float(np.min(amaps)), float(np.max(amaps))
    if not np.isfinite(min_th) or not np.isfinite(max_th) or max_th <= min_th:
        return 0.0
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    delta = (max_th - min_th) / max_step
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        denom = inverse_masks.sum()
        if denom <= 0 or not pro:
            continue
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        pros.append(float(np.mean(pro)))
        fprs.append(float(fp_pixels / denom))
    pros = np.asarray(pros, dtype=np.float32)
    fprs = np.asarray(fprs, dtype=np.float32)
    idxes = np.isfinite(fprs) & np.isfinite(pros) & (fprs < expect_fpr)
    if int(idxes.sum()) < 2:
        return 0.0
    f = fprs[idxes]
    p = pros[idxes]
    if float(f.max()) <= float(f.min()):
        return 0.0
    f = (f - f.min()) / (f.max() - f.min())
    return float(auc(f, p))


def evaluate_standard(dataset, indices, scores, maps_by_pos) -> dict:
    obj_list = sorted({dataset[i]["cls_name"] for i in indices})
    per = {}
    vals = {"I-AUROC": [], "I-AP": [], "P-AUROC": [], "PRO": []}
    for obj in obj_list:
        gt_sp, pr_sp, masks, maps = [], [], [], []
        for pos, idx in enumerate(indices):
            item = dataset[idx]
            if item["cls_name"] != obj:
                continue
            gt_sp.append(int(item["anomaly"]))
            pr_sp.append(float(scores[pos]))
            masks.append(mask_array(item))
            maps.append(maps_by_pos[pos].astype(np.float32))
        gt_sp = np.asarray(gt_sp)
        pr_sp = np.asarray(pr_sp)
        masks = np.asarray(masks)
        maps = np.asarray(maps)
        ia = float(roc_auc_score(gt_sp, pr_sp) * 100.0) if len(np.unique(gt_sp)) > 1 else float("nan")
        iap = float(average_precision_score(gt_sp, pr_sp) * 100.0) if len(np.unique(gt_sp)) > 1 else float("nan")
        pa = float(roc_auc_score(masks.ravel(), maps.ravel()) * 100.0) if len(np.unique(masks.ravel())) > 1 else float("nan")
        try:
            pro = float(pixel_level_metrics({"x": {"imgs_masks": masks, "anomaly_maps": maps}}, "x", "pixel-aupro") * 100.0)
        except ValueError:
            pro = float(safe_pro_score(masks, maps) * 100.0)
        per[obj] = {"I-AUROC": ia, "I-AP": iap, "P-AUROC": pa, "PRO": pro}
        for k, v in per[obj].items():
            if np.isfinite(v):
                vals[k].append(v)
    return {**{k: float(np.mean(v)) if v else float("nan") for k, v in vals.items()}, "per_class": per}


def support_shift(raw_values: list[np.ndarray], cdf: np.ndarray) -> dict:
    q01 = float(np.percentile(cdf, 1))
    q99 = float(np.percentile(cdf, 99))
    mn = float(cdf[0])
    mx = float(cdf[-1])
    total = below_min = above_max = below_q01 = above_q99 = 0
    pct_quantiles = []
    for arr in raw_values:
        a = np.asarray(arr, dtype=np.float32)
        total += a.size
        below_min += int(np.sum(a < mn))
        above_max += int(np.sum(a > mx))
        below_q01 += int(np.sum(a < q01))
        above_q99 += int(np.sum(a > q99))
        pct = apply_cdf(a, cdf)
        pct_quantiles.append(np.percentile(pct, [1, 5, 50, 95, 99]))
    pq = np.mean(np.asarray(pct_quantiles), axis=0) if pct_quantiles else np.full(5, np.nan)
    return {
        "below_source_min_fraction": below_min / max(1, total),
        "above_source_max_fraction": above_max / max(1, total),
        "below_source_q01_fraction": below_q01 / max(1, total),
        "above_source_q99_fraction": above_q99 / max(1, total),
        "percentile_q01": float(pq[0]),
        "percentile_q05": float(pq[1]),
        "percentile_q50": float(pq[2]),
        "percentile_q95": float(pq[3]),
        "percentile_q99": float(pq[4]),
    }


def load_anomaly_model(checkpoint: Path, device: str):
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params, download_root=str(ROOT / ".cache" / "clip"))
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    payload = torch.load(checkpoint, map_location="cpu")
    prompt_learner.load_state_dict(payload["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    with torch.no_grad():
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, text_features


@torch.no_grad()
def anomaly_raw_map_score(model, text_features, item, device: str):
    image = item["img"].unsqueeze(0).to(device)
    image_features, patch_features = model.encode_image(image, [24], DPAM_layer=20)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_probs = image_features @ text_features.permute(0, 2, 1)
    text_probs = (text_probs / 0.07).softmax(-1)
    image_score = float(text_probs[:, 0, 1].detach().cpu())
    maps = []
    for idx, patch_feature in enumerate(patch_features):
        if idx >= 0:
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], IMAGE_SIZE)
            anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
            maps.append(anomaly_map)
    raw = torch.stack(maps).sum(dim=0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return raw, image_score


def run_cell(system: str, target_name: str, device: str) -> dict:
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CELL_DIR / f"{system}_{target_name}.json"
    if ckpt.exists():
        return json.loads(ckpt.read_text(encoding="utf-8"))
    spec = source_spec(system)
    target = target_spec(target_name)
    cdf = np.load(spec["cdf"])["cdf"].astype(np.float32)
    if spec["model"] == "MRAD":
        target_data = load_mrad_dataset(target["dataset"], target["path"], mode="test")
        source_data = load_source_dataset(spec)
        model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(str(spec["checkpoint"]), device)
        image_cache, patch_cache = build_memory_from_indices(model, source_data, all_indices(source_data), spec["dataset"], device, adapter=None)
        image_cache, patch_cache = to_device_cache(image_cache, device), to_device_cache(patch_cache, device)
        producer = lambda item: official_raw_map_score(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item, device)
        cleanup = [model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache]
    else:
        target_data = load_anomaly_dataset(target["dataset"], target["path"], mode="test")
        model, text_features = load_anomaly_model(spec["checkpoint"], device)
        producer = lambda item: anomaly_raw_map_score(model, text_features, item, device)
        cleanup = [model, text_features]

    indices = all_indices(target_data)
    scores = np.zeros((len(indices),), dtype=np.float32)
    maps0, maps1, raw_samples = [], [], []
    start = time.time()
    for pos, idx in enumerate(indices):
        raw, score = producer(target_data[idx])
        scores[pos] = score
        maps0.append(smooth_full(raw).astype(np.float32))
        maps1.append(smooth_full(apply_cdf(raw, cdf)).astype(np.float32))
        if len(raw_samples) < 128:
            raw_samples.append(raw.astype(np.float32))
        if pos % 25 == 0:
            progress = {"system": system, "target": target_name, "done": pos + 1, "total": len(indices), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            write_json(BATCH / "progress.json", progress)
            print(f"[XXVIII] {system} -> {target_name}: {pos + 1}/{len(indices)}", flush=True)

    m0 = evaluate_standard(target_data, indices, scores, maps0)
    m1 = evaluate_standard(target_data, indices, scores, maps1)
    delta = {
        "model": spec["model"],
        "source": spec["source"],
        "system": system,
        "target": target_name,
        "dP-AUROC": m1["P-AUROC"] - m0["P-AUROC"],
        "dPRO": m1["PRO"] - m0["PRO"],
    }
    delta["G_pixel"] = 0.5 * delta["dP-AUROC"] + 0.5 * delta["dPRO"]
    image_parity = abs(m1["I-AUROC"] - m0["I-AUROC"]) <= 1e-7 and abs(m1["I-AP"] - m0["I-AP"]) <= 1e-7
    positive = (((delta["dPRO"] >= 0.50) or (delta["dP-AUROC"] >= 0.20)) and delta["dP-AUROC"] >= -0.15 and delta["dPRO"] >= -0.15 and delta["G_pixel"] > 0 and image_parity)
    strong = delta["dPRO"] >= 1.00 and delta["dP-AUROC"] >= -0.10 and delta["G_pixel"] >= 0.50
    percat = []
    positive_gain_total = sum(max(0.0, 0.5 * (m1["per_class"][c]["P-AUROC"] - m0["per_class"][c]["P-AUROC"]) + 0.5 * (m1["per_class"][c]["PRO"] - m0["per_class"][c]["PRO"])) for c in m0["per_class"])
    max_gain_share = 0.0
    for cat in sorted(m0["per_class"]):
        p0, p1 = m0["per_class"][cat], m1["per_class"][cat]
        g = 0.5 * (p1["P-AUROC"] - p0["P-AUROC"]) + 0.5 * (p1["PRO"] - p0["PRO"])
        max_gain_share = max(max_gain_share, max(0.0, g) / max(1e-12, positive_gain_total))
        percat.append({
            "model": spec["model"], "source": spec["source"], "system": system, "target": target_name, "category": cat,
            "T0_P-AUROC": p0["P-AUROC"], "T1_P-AUROC": p1["P-AUROC"], "T0_PRO": p0["PRO"], "T1_PRO": p1["PRO"],
            "dP-AUROC": p1["P-AUROC"] - p0["P-AUROC"], "dPRO": p1["PRO"] - p0["PRO"], "G_pixel": g,
            "WTL": "T" if abs(g) <= 0.05 else ("W" if g > 0 else "L"),
        })
    for row in percat:
        row["CATEGORY_CONCENTRATED"] = bool(max_gain_share > 0.5)
        row["max_positive_gain_share"] = max_gain_share
    result = {
        "system": system,
        "model": spec["model"],
        "source": spec["source"],
        "target": target_name,
        "metrics": [
            {"model": spec["model"], "source": spec["source"], "system": system, "target": target_name, "method": "T0_OFFICIAL", **{k: m0[k] for k in ["I-AUROC", "I-AP", "P-AUROC", "PRO"]}},
            {"model": spec["model"], "source": spec["source"], "system": system, "target": target_name, "method": "T1_SOURCE_NORMAL_PERCENTILE", **{k: m1[k] for k in ["I-AUROC", "I-AP", "P-AUROC", "PRO"]}},
        ],
        "delta": delta,
        "cell_outcome": {"positive": bool(positive), "strong_cell": bool(strong), "image_parity_pass": bool(image_parity), "max_abs_image_score_diff": 0.0},
        "support_shift": {"model": spec["model"], "source": spec["source"], "system": system, "target": target_name, **support_shift(raw_samples, cdf)},
        "per_category": percat,
        "runtime": {"seconds": time.time() - start, "images": len(indices), "device": device, "batch_size": 1, "precision": "torch default fp32/fp16 per loaded checkpoint"},
    }
    write_json(ckpt, result)
    del cleanup, maps0, maps1, scores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize(selected: list[str], cell_results: list[dict]) -> str:
    raw_metrics, deltas, outcomes, support, percat = [], [], [], [], []
    for r in cell_results:
        raw_metrics.extend(r["metrics"])
        deltas.append({**r["delta"], **r["cell_outcome"]})
        outcomes.append({**r["delta"], **r["cell_outcome"]})
        support.append(r["support_shift"])
        percat.extend(r["per_category"])
    write_csv(BATCH / "RAW_METRICS.csv", raw_metrics)
    write_csv(BATCH / "DELTAS.csv", deltas)
    write_csv(BATCH / "CELL_OUTCOMES.csv", outcomes)
    write_csv(BATCH / "CALIBRATION_SUPPORT_SHIFT.csv", support)
    write_csv(BATCH / "PER_CATEGORY.csv", percat)

    dataset_rows = []
    for target in selected:
        rows = [r for r in outcomes if r["target"] == target]
        positive = sum(1 for r in rows if r["positive"])
        strong = sum(1 for r in rows if r["strong_cell"])
        mean_g = float(np.mean([r["G_pixel"] for r in rows]))
        supported = positive >= 3 and any(r["positive"] and r["model"] == "MRAD" for r in rows) and any(r["positive"] and r["model"] == "AnomalyCLIP" for r in rows) and mean_g >= 0.40 and not any(r["G_pixel"] < -0.30 for r in rows)
        strong_support = positive == 4 and strong >= 3 and mean_g >= 0.75
        dataset_rows.append({"target": target, "positive_cells": positive, "strong_cells": strong, "mean_G_pixel": mean_g, "SUPPORTED": supported, "STRONG_NEW_DATASET_GENERALIZATION": strong_support})
    write_csv(BATCH / "DATASET_OUTCOMES.csv", dataset_rows)

    robustness = []
    for target in selected:
        for model in ["MRAD", "AnomalyCLIP"]:
            rows = [r for r in outcomes if r["target"] == target and r["model"] == model]
            if len(rows) == 2:
                robustness.append({"target": target, "model": model, "MVTec_source_G": next(r["G_pixel"] for r in rows if r["source"] == "MVTec"), "VisA_source_G": next(r["G_pixel"] for r in rows if r["source"] == "VisA"), "both_positive": all(r["positive"] for r in rows)})
    write_csv(BATCH / "SOURCE_SYSTEM_ROBUSTNESS.csv", robustness)

    historical = [
        {"target": "VisA", "system": "M-MVTec", "baseline_P-AUROC": 95.8084, "calibrated_P-AUROC": 95.8173, "baseline_PRO": 87.9651, "calibrated_PRO": 90.4225, "G_pixel": 1.2331},
        {"target": "MVTec", "system": "M-VisA", "baseline_P-AUROC": 92.8762, "calibrated_P-AUROC": 92.8752, "baseline_PRO": 86.8162, "calibrated_PRO": 91.6972, "G_pixel": 2.4400},
        {"target": "VisA", "system": "A-MVTec", "baseline_P-AUROC": 95.5003, "calibrated_P-AUROC": 95.5172, "baseline_PRO": 86.7295, "calibrated_PRO": 88.6072, "G_pixel": 0.9473},
        {"target": "MVTec", "system": "A-VisA", "baseline_P-AUROC": 91.0730, "calibrated_P-AUROC": 91.2235, "baseline_PRO": 81.3821, "calibrated_PRO": 86.5827, "G_pixel": 2.6756},
    ]
    matrix = historical[:]
    for r in raw_metrics:
        if r["method"] != "T0_OFFICIAL":
            continue
        m1 = next(x for x in raw_metrics if x["system"] == r["system"] and x["target"] == r["target"] and x["method"] == "T1_SOURCE_NORMAL_PERCENTILE")
        matrix.append({"target": r["target"], "system": r["system"], "baseline_P-AUROC": r["P-AUROC"], "calibrated_P-AUROC": m1["P-AUROC"], "baseline_PRO": r["PRO"], "calibrated_PRO": m1["PRO"], "G_pixel": 0.5 * ((m1["P-AUROC"] - r["P-AUROC"]) + (m1["PRO"] - r["PRO"]))})
    write_csv(BATCH / "CROSS_DATASET_MATRIX.csv", matrix)

    if all(r["STRONG_NEW_DATASET_GENERALIZATION"] for r in dataset_rows):
        outcome = "XXVIII_A_STRONG_MULTI_DATASET_GENERALIZATION"
    elif all(r["SUPPORTED"] for r in dataset_rows):
        outcome = "XXVIII_A_MULTI_DATASET_GENERALIZATION"
    elif any(r["SUPPORTED"] for r in dataset_rows):
        outcome = "XXVIII_A_MIXED_DATASET_GENERALIZATION"
    else:
        outcome = "XXVIII_A_NO_BROAD_DATASET_GENERALIZATION"

    report = ["# Phase XXVIII-A Multi-Dataset Generalization", "", f"Outcome: `{outcome}`", "", "No training, target calibration, target CDF, or method redesign was performed. New target data came from `E:\\WorkSpace\\dataset`.", "", "## Dataset Outcomes", "", "| Target | Positive | Strong | mean G | Supported |", "|---|---:|---:|---:|---|"]
    for r in dataset_rows:
        report.append(f"| {r['target']} | {r['positive_cells']}/4 | {r['strong_cells']}/4 | {r['mean_G_pixel']:+.4f} | {r['SUPPORTED']} |")
    report.extend(["", "## Cell Deltas", "", "| System | Target | dP | dPRO | G | Positive | Strong |", "|---|---|---:|---:|---:|---|---|"])
    for r in outcomes:
        report.append(f"| {r['system']} | {r['target']} | {r['dP-AUROC']:+.4f} | {r['dPRO']:+.4f} | {r['G_pixel']:+.4f} | {r['positive']} | {r['strong_cell']} |")
    report.append("")
    report.append("STOP: Batch A finished; no additional downloaded datasets were evaluated.")
    (BATCH / "XXVIII_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(BATCH / "OUTCOME.json", {"phase": "XXVIII-A", "outcome": outcome, "selected_datasets": selected, "dataset_outcomes": dataset_rows})
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXVIII Multi-Dataset Generalization\n\n")
        fp.write(f"- outcome: `{outcome}`\n")
        fp.write(f"- selected target datasets: `{', '.join(selected)}` from `E:\\WorkSpace\\dataset`.\n")
        for r in dataset_rows:
            fp.write(f"- {r['target']}: positive {r['positive_cells']}/4, strong {r['strong_cells']}/4, mean G {r['mean_G_pixel']:+.4f}, supported {r['SUPPORTED']}.\n")
    return outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--systems", default="M-MVTec,M-VisA,A-MVTec,A-VisA")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, selected = run_audit()
    art = artifact_audit()
    if args.audit_only:
        write_json(BATCH / "OUTCOME.json", {"phase": "XXVIII-0", "outcome": "XXVIII_0_DATASET_SELECTION_FROZEN", "selected_datasets": selected})
        print("XXVIII_0_DATASET_SELECTION_FROZEN", selected)
        return
    if not art["artifacts_ok"]:
        write_json(BATCH / "OUTCOME.json", {"phase": "XXVIII", "outcome": "XXVIII_ARTIFACT_MISMATCH", "selected_datasets": selected, "artifact_audit": art})
        raise SystemExit("XXVIII_ARTIFACT_MISMATCH")
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    cell_results = []
    for target in selected:
        for system in systems:
            cell_results.append(run_cell(system, target, args.device))
    all_systems = ["M-MVTec", "M-VisA", "A-MVTec", "A-VisA"]
    all_cell_paths = [CELL_DIR / f"{system}_{target}.json" for target in selected for system in all_systems]
    if all(p.exists() for p in all_cell_paths):
        all_results = [json.loads(p.read_text(encoding="utf-8")) for p in all_cell_paths]
        outcome = summarize(selected, all_results)
        print(outcome)
    else:
        missing = [str(p.name) for p in all_cell_paths if not p.exists()]
        payload = {"phase": "XXVIII-A", "outcome": "XXVIII_A_PARTIAL_CELLS_COMPLETE", "selected_datasets": selected, "missing_cells": missing}
        write_json(BATCH / "OUTCOME.json", payload)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
