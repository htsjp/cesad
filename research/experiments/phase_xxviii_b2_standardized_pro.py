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
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import auc, average_precision_score, roc_auc_score

import packaging as _packaging
import packaging.version as _packaging_version
import pkg_resources as _pkg_resources

_packaging.version = _packaging_version
if not hasattr(_pkg_resources, "packaging"):
    _pkg_resources.packaging = _packaging


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

XXVI_OFFICIAL = ROOT.parents[0] / "reference_repos" / "phase_VI_A" / "AnomalyCLIP"
IMPORT_MODE = os.environ.get("XXVIII_IMPORT_MODE", "project").strip().lower()
if IMPORT_MODE == "official":
    sys.path.insert(0, str(XXVI_OFFICIAL))
    import AnomalyCLIP_lib  # noqa: E402
    from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402
    sys.path = [p for p in sys.path if p != str(XXVI_OFFICIAL)]
else:
    import AnomalyCLIP_lib  # noqa: E402
    AnomalyCLIP_PromptLearner = None

_anomaly_utils_spec = importlib.util.spec_from_file_location("official_anomalyclip_utils", XXVI_OFFICIAL / "utils.py")
_anomaly_utils = importlib.util.module_from_spec(_anomaly_utils_spec)
assert _anomaly_utils_spec.loader is not None
_anomaly_utils_spec.loader.exec_module(_anomaly_utils)
anomaly_get_transform = _anomaly_utils.get_transform

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import to_device_cache
from experiments.phase_xxiii_b_dual_carrier_local import IMAGE_SIZE, SIGMA, apply_cdf, official_raw_map_score, smooth_full
from models.mlp import average_neighbor
from mrad import compute_patch_socre
from phase_xa.official_task_forward import image_memory_scores, normalized_image_feature, visual_outputs
from phase_xa.teacher_memory_builder import build_memory_from_indices
from utils.metrics import cal_pro_score
from utils.transforms import get_transform as mrad_get_transform


DATA_ROOT = Path(r"E:\WorkSpace\dataset")
CACHE_ROOT = Path(r"E:\WorkSpace\phase_xxviii_eval_cache")
OUT = ROOT / "results" / "phase_xxviii_multidataset" / "batch_b2_standardized_pro"
AUDIT_A = ROOT / "results" / "phase_xxviii_multidataset" / "audit"
XXIV = ROOT / "results" / "phase_xxiv_pixel_calibration" / "final_target"
XXVI = ROOT / "results" / "phase_xxvi_cross_model" / "anomalyclip"
MAX_FPR = 0.3
SYSTEMS = ["M-MVTec", "M-VisA", "A-MVTec", "A-VisA"]
TARGETS = ["MPDD", "InsPLAD_AD"]


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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=jdefault) + "\n", encoding="utf-8")


def jdefault(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(type(obj).__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_hash(path: Path = ROOT) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not-a-git-repo"


def class_info(name: str, classes: list[str]) -> tuple[list[str], dict[str, int]]:
    known = {
        "MPDD": ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"],
        "BTAD": ["01", "02", "03"],
        "DTD": ["Woven_001", "Woven_127", "Woven_104", "Stratified_154", "Blotchy_099", "Woven_068", "Woven_125", "Marbled_078", "Perforated_037", "Mesh_114", "Fibrous_183", "Matted_069"],
        "mvtec": ["carpet", "bottle", "hazelnut", "leather", "cable", "capsule", "grid", "pill", "transistor", "metal_nut", "screw", "toothbrush", "zipper", "tile", "wood"],
        "visa": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
    }
    obj = known.get(name, classes)
    return obj, {k: i for i, k in enumerate(obj)}


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
                r = dict(row)
                r.setdefault("specie_name", r.get("type", ""))
                self.data_all.append(r)
        self.obj_list, self.class_name_map_class_id = class_info(dataset_name, self.cls_names)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, idx):
        row = self.data_all[idx]
        img = Image.open(self.root / row["img_path"]).convert("RGB")
        anomaly = int(row["anomaly"])
        if anomaly:
            mask_path = self.root / row.get("mask_path", "")
            mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
            mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
        else:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        img_t = self.transform(img) if self.transform is not None else img
        mask_t = self.target_transform(mask) if self.target_transform is not None else mask
        cls = row["cls_name"]
        return {"img": img_t, "img_mask": mask_t, "cls_name": cls, "anomaly": anomaly, "img_path": str(self.root / row["img_path"]), "cls_id": self.class_name_map_class_id.get(cls, 0)}


class InsPLADDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path | str, transform, target_transform):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.obj_list = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_name_map_class_id = {k: i for i, k in enumerate(self.obj_list)}
        self.data_all = []
        for cls in self.obj_list:
            cat = self.root / cls
            for sub, anomaly in [("good", 0), ("anomaly", 1)]:
                for img in sorted((cat / "test" / sub).glob("*")):
                    if img.is_file():
                        mask = cat / "ground_truth" / "anomaly" / f"{img.stem}_mask.png" if anomaly else Path("")
                        self.data_all.append({"img": img, "mask": mask, "cls_name": cls, "anomaly": anomaly})

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, idx):
        row = self.data_all[idx]
        img = Image.open(row["img"]).convert("RGB")
        if int(row["anomaly"]):
            mask_arr = np.array(Image.open(row["mask"]).convert("L")) > 0
            mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
        else:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        img_t = self.transform(img) if self.transform is not None else img
        mask_t = self.target_transform(mask) if self.target_transform is not None else mask
        cls = row["cls_name"]
        return {"img": img_t, "img_mask": mask_t, "cls_name": cls, "anomaly": int(row["anomaly"]), "img_path": str(row["img"]), "cls_id": self.class_name_map_class_id[cls]}


def load_dataset_for(model: str, dataset: str, path: Path | str, mode: str = "test"):
    if model == "AnomalyCLIP":
        class Args:
            image_size = IMAGE_SIZE
        transform, target_transform = anomaly_get_transform(Args())
    else:
        transform, target_transform = mrad_get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    if dataset == "InsPLAD_AD":
        return InsPLADDataset(path, transform, target_transform)
    return MetaDataset(path, transform, target_transform, dataset, mode)


def source_spec(system: str) -> dict:
    if system == "M-MVTec":
        return {"model": "MRAD", "source": "MVTec", "dataset": "mvtec", "path": ROOT / "data" / "mvtec", "checkpoint": ROOT / "checkpoints" / "test_on_visa.pth", "cdf": XXIV / "D1_PIXEL_CDF.npz"}
    if system == "M-VisA":
        return {"model": "MRAD", "source": "VisA", "dataset": "visa", "path": ROOT / "data" / "VISA", "checkpoint": ROOT / "checkpoints" / "test_on_mvtec.pth", "cdf": XXIV / "D2_PIXEL_CDF.npz"}
    if system == "A-MVTec":
        return {"model": "AnomalyCLIP", "source": "MVTec", "dataset": "mvtec", "path": ROOT / "data" / "mvtec", "checkpoint": XXVI_OFFICIAL / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth", "cdf": XXVI / "D1_ANOMALYCLIP_PIXEL_CDF.npz"}
    if system == "A-VisA":
        return {"model": "AnomalyCLIP", "source": "VisA", "dataset": "visa", "path": ROOT / "data" / "VISA", "checkpoint": XXVI_OFFICIAL / "checkpoints" / "9_12_4_multiscale" / "epoch_15.pth", "cdf": XXVI / "D2_ANOMALYCLIP_PIXEL_CDF.npz"}
    raise KeyError(system)


def target_spec(target: str) -> dict:
    if target == "MPDD":
        return {"dataset": "MPDD", "path": DATA_ROOT / "MPDD"}
    if target == "InsPLAD_AD":
        return {"dataset": "InsPLAD_AD", "path": DATA_ROOT / "InsPLAD_AD"}
    raise KeyError(target)


def all_indices(dataset) -> list[int]:
    return list(range(len(dataset)))


def mask_array(item) -> np.ndarray:
    return (item["img_mask"].detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8)


def regions_by_image(masks: np.ndarray):
    for i, m in enumerate(masks):
        for region in measure.regionprops(measure.label(m)):
            yield i, region.coords[:, 0], region.coords[:, 1], region.area


def standardized_pro_curve(masks: np.ndarray, amaps: np.ndarray, max_step: int = 200, max_fpr: float = MAX_FPR) -> tuple[float | None, list[dict], dict]:
    masks = np.asarray(masks)
    amaps = np.asarray(amaps, dtype=np.float32)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if amaps.ndim == 4:
        amaps = amaps.squeeze(1)
    min_th, max_th = float(np.min(amaps)), float(np.max(amaps))
    meta = {"min_th": min_th, "max_th": max_th, "raw_points": 0, "unique_fpr_points": 0, "points_le_max_fpr": 0, "boundary_inserted": False, "defined": True, "undefined_reason": ""}
    if not np.isfinite(min_th) or not np.isfinite(max_th):
        meta.update({"defined": False, "undefined_reason": "non_finite_scores"})
        return None, [], meta
    if max_th <= min_th:
        meta.update({"defined": False, "undefined_reason": "constant_scores"})
        return None, [], meta
    delta = (max_th - min_th) / max_step
    regions = list(regions_by_image(masks))
    if not regions:
        meta.update({"defined": False, "undefined_reason": "no_gt_regions"})
        return None, [], meta
    inv = 1 - masks
    denom = inv.sum()
    if denom <= 0:
        meta.update({"defined": False, "undefined_reason": "no_negative_pixels"})
        return None, [], meta
    pts = []
    binary = np.zeros_like(amaps, dtype=bool)
    for th in np.arange(min_th, max_th, delta):
        binary[amaps <= th], binary[amaps > th] = 0, 1
        overlaps = [float(binary[i, rr, cc].sum() / area) for i, rr, cc, area in regions]
        fpr = float(np.logical_and(inv, binary).sum() / denom)
        pro = float(np.mean(overlaps))
        if np.isfinite(fpr) and np.isfinite(pro):
            pts.append((fpr, pro, float(th)))
    meta["raw_points"] = len(pts)
    if not pts:
        meta.update({"defined": False, "undefined_reason": "no_valid_curve_points"})
        return None, [], meta
    by_fpr = {0.0: 0.0}
    for fpr, pro, _ in pts:
        key = float(fpr)
        by_fpr[key] = max(by_fpr.get(key, -np.inf), float(pro))
    curve = sorted((k, v) for k, v in by_fpr.items())
    if curve[-1][0] < max_fpr:
        meta.update({"defined": False, "undefined_reason": "curve_does_not_reach_max_fpr"})
        return None, [{"fpr": f, "pro": p} for f, p in curve], meta
    bounded = []
    prev = None
    inserted = False
    for fpr, pro in curve:
        if fpr <= max_fpr:
            bounded.append((fpr, pro))
            prev = (fpr, pro)
            continue
        if prev is None:
            bounded.append((0.0, 0.0))
            bounded.append((max_fpr, pro))
        elif prev[0] < max_fpr:
            ratio = (max_fpr - prev[0]) / (fpr - prev[0]) if fpr > prev[0] else 0.0
            interp = prev[1] + ratio * (pro - prev[1])
            bounded.append((max_fpr, float(interp)))
        inserted = True
        break
    if len(bounded) < 2:
        meta.update({"defined": False, "undefined_reason": "insufficient_bounded_points"})
        return None, [{"fpr": f, "pro": p} for f, p in bounded], meta
    xs = np.asarray([x for x, _ in bounded], dtype=np.float64)
    ys = np.asarray([y for _, y in bounded], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    val = float(auc(xs, ys) / max_fpr)
    meta.update({"unique_fpr_points": len(curve), "points_le_max_fpr": int(np.sum(xs <= max_fpr)), "boundary_inserted": inserted})
    return val, [{"fpr": float(x), "pro": float(y)} for x, y in zip(xs, ys)], meta


def original_pro(masks: np.ndarray, amaps: np.ndarray) -> float:
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if amaps.ndim == 4:
        amaps = amaps.squeeze(1)
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / 200
    regions = list(regions_by_image(masks))
    pros, fprs = [], []
    inv = 1 - masks
    denom = inv.sum()
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = [float(binary_amaps[i, rr, cc].sum() / area) for i, rr, cc, area in regions]
        fp_pixels = np.logical_and(inv, binary_amaps).sum()
        fprs.append(fp_pixels / denom)
        pros.append(np.array(pro).mean())
    pros, fprs = np.array(pros), np.array(fprs)
    idxes = fprs < MAX_FPR
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    return float(auc(fprs, pros[idxes]))


def synthetic_tests() -> list[dict]:
    rng = np.random.default_rng(0)
    masks = np.zeros((4, 32, 32), dtype=np.uint8)
    masks[:, 8:16, 8:16] = 1
    masks_multi = masks.copy()
    masks_multi[:, 22:27, 22:27] = 1
    cases = {
        "perfect_prediction": (masks, masks.astype(np.float32)),
        "perfect_inverse": (masks, 1 - masks.astype(np.float32)),
        "constant_zero": (masks, np.zeros_like(masks, dtype=np.float32)),
        "constant_one": (masks, np.ones_like(masks, dtype=np.float32)),
        "constant_arbitrary": (masks, np.full_like(masks, 0.42, dtype=np.float32)),
        "random_map": (masks, rng.random(masks.shape, dtype=np.float32)),
        "single_small_anomaly_region": (masks, masks.astype(np.float32) + 0.01 * rng.random(masks.shape, dtype=np.float32)),
        "multiple_regions": (masks_multi, masks_multi.astype(np.float32) + 0.01 * rng.random(masks.shape, dtype=np.float32)),
        "sparse_single_pixel_prediction": (masks, (rng.random(masks.shape) > 0.999).astype(np.float32)),
        "prediction_entirely_inside_anomaly": (masks, masks.astype(np.float32) * (rng.random(masks.shape) > 0.5)),
        "prediction_entirely_outside_anomaly": (masks, (1 - masks).astype(np.float32)),
    }
    rows = []
    for name, (mask, amap) in cases.items():
        std, curve, meta = standardized_pro_curve(mask, amap)
        try:
            ref = original_pro(mask, amap)
            ref_status = "defined"
        except Exception as exc:
            ref = np.nan
            ref_status = f"undefined:{type(exc).__name__}"
        rows.append({"case": name, "standardized_PRO": "" if std is None else std * 100, "standardized_defined": std is not None, "reference_PRO": "" if not np.isfinite(ref) else ref * 100, "reference_status": ref_status, "curve_point_count": len(curve), **meta})
    return rows


def audit_pro_implementations() -> None:
    rows = {
        "phase": "XXVIII-B2",
        "implementations": [
            {
                "name": "project utils.metrics.cal_pro_score",
                "file": str(ROOT / "utils" / "metrics.py"),
                "threshold_generation": "np.arange(min_score, max_score, (max-min)/200)",
                "fpr_definition": "FP pixels over inverse mask pixels",
                "region_overlap_definition": "mean per connected GT region overlap",
                "fpr_upper_bound": MAX_FPR,
                "normalization": "normalize retained fprs to [0,1] before auc",
                "duplicate_fpr_handling": "none",
                "degenerate_curve_behavior": "can raise ValueError or produce NaN",
            },
            {
                "name": "XXVIII local fallback",
                "file": str(ROOT / "experiments" / "phase_xxviii_multidataset_generalization.py"),
                "threshold_generation": "same as project reference",
                "duplicate_fpr_handling": "none; returns 0 on insufficient points",
                "degenerate_curve_behavior": "returns 0.0, now judged insufficient for B2",
            },
            {
                "name": "XXVIII-B2 standardized_pro_curve",
                "file": str(ROOT / "experiments" / "phase_xxviii_b2_standardized_pro.py"),
                "threshold_generation": "same reference trajectory where possible",
                "duplicate_fpr_handling": "sort by FPR; keep max PRO at duplicate FPR",
                "boundary_interpolation": "linear interpolation at max_fpr=0.3",
                "degenerate_curve_behavior": "returns PRO_UNDEFINED, never coerces to 0",
            },
        ],
    }
    write_json(OUT / "PRO_IMPLEMENTATION_AUDIT.json", rows)


def load_anomaly_model(checkpoint: Path, device: str):
    if AnomalyCLIP_PromptLearner is None:
        raise RuntimeError("Run AnomalyCLIP generation with XXVIII_IMPORT_MODE=official")
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
            maps.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
    raw = torch.stack(maps).sum(dim=0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return raw, image_score


def cache_cell(system: str, target: str, device: str) -> None:
    spec = source_spec(system)
    if IMPORT_MODE != "official" and spec["model"] == "AnomalyCLIP":
        raise RuntimeError("AnomalyCLIP cells require XXVIII_IMPORT_MODE=official")
    if IMPORT_MODE == "official" and spec["model"] == "MRAD":
        raise RuntimeError("MRAD cells require project import mode")
    target_info = target_spec(target)
    cell_root = CACHE_ROOT / target / spec["model"] / spec["source"]
    (cell_root / "T0_OFFICIAL").mkdir(parents=True, exist_ok=True)
    (cell_root / "T1_SOURCE_NORMAL_PERCENTILE").mkdir(parents=True, exist_ok=True)
    done = cell_root / "DONE.json"
    if done.exists():
        return
    target_data = load_dataset_for(spec["model"], target_info["dataset"], target_info["path"])
    indices = all_indices(target_data)
    cdf = np.load(spec["cdf"])["cdf"].astype(np.float32)
    if spec["model"] == "MRAD":
        source_data = load_dataset_for("MRAD", spec["dataset"], spec["path"])
        model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(str(spec["checkpoint"]), device)
        image_cache, patch_cache = build_memory_from_indices(model, source_data, all_indices(source_data), spec["dataset"], device, adapter=None)
        image_cache, patch_cache = to_device_cache(image_cache, device), to_device_cache(patch_cache, device)
        producer = lambda item: official_raw_map_score(model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item, device)
    else:
        model, text_features = load_anomaly_model(spec["checkpoint"], device)
        producer = lambda item: anomaly_raw_map_score(model, text_features, item, device)
    n = len(indices)
    raw = np.lib.format.open_memmap(cell_root / "pre_gaussian_raw.npy", mode="w+", dtype=np.float16, shape=(n, IMAGE_SIZE, IMAGE_SIZE))
    rank = np.lib.format.open_memmap(cell_root / "pre_gaussian_rank.npy", mode="w+", dtype=np.float16, shape=(n, IMAGE_SIZE, IMAGE_SIZE))
    maps0 = np.lib.format.open_memmap(cell_root / "T0_OFFICIAL" / "final_maps.npy", mode="w+", dtype=np.float16, shape=(n, IMAGE_SIZE, IMAGE_SIZE))
    maps1 = np.lib.format.open_memmap(cell_root / "T1_SOURCE_NORMAL_PERCENTILE" / "final_maps.npy", mode="w+", dtype=np.float16, shape=(n, IMAGE_SIZE, IMAGE_SIZE))
    masks = np.lib.format.open_memmap(cell_root / "gt_masks.npy", mode="w+", dtype=np.uint8, shape=(n, IMAGE_SIZE, IMAGE_SIZE))
    scores = np.zeros((n,), dtype=np.float32)
    manifest = []
    start = time.time()
    for pos, idx in enumerate(indices):
        item = target_data[idx]
        r, s = producer(item)
        rk = apply_cdf(r, cdf)
        raw[pos] = r.astype(np.float16)
        rank[pos] = rk.astype(np.float16)
        maps0[pos] = smooth_full(r).astype(np.float16)
        maps1[pos] = smooth_full(rk).astype(np.float16)
        masks[pos] = mask_array(item).astype(np.uint8)
        scores[pos] = s
        manifest.append({"pos": pos, "idx": int(idx), "category": item["cls_name"], "anomaly": int(item["anomaly"]), "img_path": item["img_path"], "map_shape": [IMAGE_SIZE, IMAGE_SIZE]})
        if pos % 25 == 0:
            print(f"[B2] {system}->{target}: {pos+1}/{n}", flush=True)
            raw.flush(); rank.flush(); maps0.flush(); maps1.flush(); masks.flush()
    raw.flush(); rank.flush(); maps0.flush(); maps1.flush(); masks.flush()
    np.save(cell_root / "scores.npy", scores)
    for method in ["T0_OFFICIAL", "T1_SOURCE_NORMAL_PERCENTILE"]:
        (cell_root / method).mkdir(parents=True, exist_ok=True)
        np.save(cell_root / method / "scores.npy", scores)
        np.save(cell_root / method / "gt_masks.npy", np.asarray(masks))
    write_json(cell_root / "manifest.json", manifest)
    hashes = {
        "pre_gaussian_raw": sha256_file(cell_root / "pre_gaussian_raw.npy"),
        "pre_gaussian_rank": sha256_file(cell_root / "pre_gaussian_rank.npy"),
        "gt_masks": sha256_file(cell_root / "gt_masks.npy"),
        "scores": sha256_file(cell_root / "scores.npy"),
        "T0_final_maps": sha256_file(cell_root / "T0_OFFICIAL" / "final_maps.npy"),
        "T1_final_maps": sha256_file(cell_root / "T1_SOURCE_NORMAL_PERCENTILE" / "final_maps.npy"),
    }
    write_json(done, {"system": system, "target": target, "seconds": time.time() - start, "images": n, "hashes": hashes})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cell_root(system: str, target: str) -> Path:
    spec = source_spec(system)
    return CACHE_ROOT / target / spec["model"] / spec["source"]


def evaluate_cell(system: str, target: str) -> dict:
    spec = source_spec(system)
    root = cell_root(system, target)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    masks = np.load(root / "gt_masks.npy", mmap_mode="r")
    scores = np.load(root / "scores.npy")
    out = {"metrics": [], "per_category": [], "parity": [], "curves": []}
    for method in ["T0_OFFICIAL", "T1_SOURCE_NORMAL_PERCENTILE"]:
        maps = np.load(root / method / "final_maps.npy", mmap_mode="r")
        vals = {"I-AUROC": [], "I-AP": [], "P-AUROC": [], "PRO_std": []}
        for cat in sorted({m["category"] for m in manifest}):
            idxs = [m["pos"] for m in manifest if m["category"] == cat]
            gt = np.asarray([manifest[i]["anomaly"] for i in idxs])
            mm = np.asarray(masks[idxs])
            aa = np.asarray(maps[idxs], dtype=np.float32)
            ss = np.asarray(scores[idxs])
            ia = roc_auc_score(gt, ss) * 100 if len(np.unique(gt)) > 1 else np.nan
            iap = average_precision_score(gt, ss) * 100 if len(np.unique(gt)) > 1 else np.nan
            pa = roc_auc_score(mm.ravel(), aa.ravel()) * 100 if len(np.unique(mm.ravel())) > 1 else np.nan
            pro_std, curve, meta = standardized_pro_curve(mm, aa)
            pro_pp = np.nan if pro_std is None else pro_std * 100
            try:
                pro_orig = original_pro(mm, aa) * 100
                orig_status = "defined"
                absdiff = abs(pro_orig - pro_pp) if np.isfinite(pro_pp) else np.nan
            except Exception as exc:
                pro_orig = np.nan
                orig_status = f"undefined:{type(exc).__name__}"
                absdiff = np.nan
            out["per_category"].append({"model": spec["model"], "source": spec["source"], "system": system, "target": target, "method": method, "category": cat, "I-AUROC": ia, "I-AP": iap, "P-AUROC": pa, "PRO_std": pro_pp, "PRO_original": pro_orig if np.isfinite(pro_orig) else "", "original_status": orig_status})
            out["parity"].append({"system": system, "target": target, "method": method, "category": cat, "PRO_original": pro_orig if np.isfinite(pro_orig) else "", "PRO_standardized": pro_pp if np.isfinite(pro_pp) else "", "abs_diff_pp": absdiff if np.isfinite(absdiff) else "", "original_status": orig_status, **meta})
            if target == "InsPLAD_AD" and system == "M-VisA" and cat == "vari-grip":
                for row in curve:
                    out["curves"].append({"system": system, "target": target, "method": method, "category": cat, **row, "PRO_std": pro_pp, **meta})
            for k, v in [("I-AUROC", ia), ("I-AP", iap), ("P-AUROC", pa), ("PRO_std", pro_pp)]:
                if np.isfinite(v):
                    vals[k].append(v)
        out["metrics"].append({"model": spec["model"], "source": spec["source"], "system": system, "target": target, "method": method, **{k: float(np.mean(v)) if v else np.nan for k, v in vals.items()}})
    return out


def run_evaluate() -> None:
    metrics, percat, parity, curves, manifest_rows = [], [], [], [], []
    for target in TARGETS:
        for system in SYSTEMS:
            root = cell_root(system, target)
            if not (root / "DONE.json").exists():
                raise SystemExit(f"missing cache {system} {target}: {root}")
            res = evaluate_cell(system, target)
            metrics.extend(res["metrics"])
            percat.extend(res["per_category"])
            parity.extend(res["parity"])
            curves.extend(res["curves"])
            done = json.loads((root / "DONE.json").read_text(encoding="utf-8"))
            for key, val in done["hashes"].items():
                manifest_rows.append({"target": target, "system": system, "artifact": key, "path": str(root), "sha256": val, "images": done["images"]})
    write_csv(OUT / "RAW_METRICS.csv", metrics)
    deltas, cell_outcomes = [], []
    for target in TARGETS:
        for system in SYSTEMS:
            m0 = next(r for r in metrics if r["target"] == target and r["system"] == system and r["method"] == "T0_OFFICIAL")
            m1 = next(r for r in metrics if r["target"] == target and r["system"] == system and r["method"] == "T1_SOURCE_NORMAL_PERCENTILE")
            dP = m1["P-AUROC"] - m0["P-AUROC"]
            dPRO = m1["PRO_std"] - m0["PRO_std"]
            G = 0.5 * dP + 0.5 * dPRO
            positive = (((dPRO >= 0.50) or (dP >= 0.20)) and dP >= -0.15 and dPRO >= -0.15 and G > 0)
            strong = dPRO >= 1.00 and dP >= -0.10 and G >= 0.50
            row = {"model": m0["model"], "source": m0["source"], "system": system, "target": target, "dP-AUROC": dP, "dPRO_std": dPRO, "G_pixel_std": G, "positive": positive, "strong_cell": strong, "image_parity_pass": True, "max_abs_image_score_diff": 0.0}
            deltas.append(row); cell_outcomes.append(row)
    write_csv(OUT / "DELTAS.csv", deltas)
    write_csv(OUT / "CELL_OUTCOMES.csv", cell_outcomes)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "MAP_MANIFEST.csv", manifest_rows)
    write_csv(OUT / "VARI_GRIP_PRO_CURVE.csv", curves)
    defined_parity = [r for r in parity if r["original_status"] == "defined" and r["abs_diff_pp"] != "" and np.isfinite(float(r["abs_diff_pp"]))]
    write_csv(OUT / "PRO_REFERENCE_PARITY.csv", parity)
    write_csv(OUT / "PRO_CATEGORY_PARITY.csv", parity)
    mean_abs = float(np.mean([float(r["abs_diff_pp"]) for r in defined_parity])) if defined_parity else np.nan
    p95_abs = float(np.percentile([float(r["abs_diff_pp"]) for r in defined_parity], 95)) if defined_parity else np.nan
    max_abs = float(np.max([float(r["abs_diff_pp"]) for r in defined_parity])) if defined_parity else np.nan
    parity_pass = bool(np.isfinite(mean_abs) and mean_abs <= 0.02 and np.isfinite(p95_abs) and p95_abs <= 0.05)
    protocol = {"source_file": str(ROOT / "experiments" / "phase_xxviii_b2_standardized_pro.py"), "sha256": sha256_file(ROOT / "experiments" / "phase_xxviii_b2_standardized_pro.py"), "max_fpr": MAX_FPR, "threshold_policy": "reference np.arange(min,max,(max-min)/200)", "duplicate_fpr_rule": "keep max PRO for duplicate FPR", "boundary_interpolation": "linear interpolation at max_fpr", "normalization": "divide AUC by max_fpr", "degenerate_behavior": "PRO_UNDEFINED, not zero", "parity_mean_abs_pp": mean_abs, "parity_p95_abs_pp": p95_abs, "parity_max_abs_pp": max_abs, "parity_pass": parity_pass}
    write_json(OUT / "STANDARDIZED_PRO_PROTOCOL.json", protocol)
    (OUT / "STANDARDIZED_PRO_SHA256.txt").write_text(f"{protocol['sha256']}  {protocol['source_file']}\n", encoding="utf-8")
    dataset_rows = []
    for target in TARGETS:
        rows = [r for r in cell_outcomes if r["target"] == target]
        pos = sum(bool(r["positive"]) for r in rows)
        strong = sum(bool(r["strong_cell"]) for r in rows)
        mean_g = float(np.mean([r["G_pixel_std"] for r in rows]))
        supported = pos >= 3 and any(r["positive"] and r["model"] == "MRAD" for r in rows) and any(r["positive"] and r["model"] == "AnomalyCLIP" for r in rows) and mean_g >= 0.40 and not any(r["G_pixel_std"] < -0.30 for r in rows)
        strong_ds = pos == 4 and strong >= 3 and mean_g >= 0.75
        dataset_rows.append({"target": target, "positive_cells": pos, "strong_cells": strong, "mean_G_pixel_std": mean_g, "SUPPORTED": supported, "STRONG_NEW_DATASET_GENERALIZATION": strong_ds})
    write_csv(OUT / "DATASET_OUTCOMES.csv", dataset_rows)
    mpdd = next(r for r in dataset_rows if r["target"] == "MPDD")
    ins = next(r for r in dataset_rows if r["target"] == "InsPLAD_AD")
    if not parity_pass:
        outcome = "XXVIII_B2_EVALUATOR_NOT_PARITY_COMPATIBLE"
    elif not mpdd["SUPPORTED"]:
        outcome = "XXVIII_B2_MPDD_NOT_REPRODUCIBLE"
    elif ins["SUPPORTED"]:
        outcome = "XXVIII_B2_INSPLAD_SUPPORTED"
    else:
        mv = next(r for r in cell_outcomes if r["target"] == "InsPLAD_AD" and r["system"] == "M-VisA")
        neg = []
        for cat in sorted({r["category"] for r in percat if r["target"] == "InsPLAD_AD" and r["system"] == "M-VisA"}):
            c0 = next(r for r in percat if r["target"] == "InsPLAD_AD" and r["system"] == "M-VisA" and r["category"] == cat and r["method"] == "T0_OFFICIAL")
            c1 = next(r for r in percat if r["target"] == "InsPLAD_AD" and r["system"] == "M-VisA" and r["category"] == cat and r["method"] == "T1_SOURCE_NORMAL_PERCENTILE")
            g = 0.5 * ((c1["P-AUROC"] - c0["P-AUROC"]) + (c1["PRO_std"] - c0["PRO_std"]))
            neg.append(max(0.0, -g))
        outcome = "XXVIII_B2_INSPLAD_CATEGORY_SPECIFIC_BOUNDARY" if sum(neg) > 0 and max(neg) / sum(neg) > 0.5 else "XXVIII_B2_INSPLAD_REAL_MIXED_RESULT"
    write_json(OUT / "OUTCOME.json", {"phase": "XXVIII-B2", "outcome": outcome, "parity_pass": parity_pass, "dataset_outcomes": dataset_rows, "parity": protocol})
    report = ["# Phase XXVIII-B2 Standardized PRO Re-Adjudication", "", f"Outcome: `{outcome}`", "", f"PRO parity: mean abs `{mean_abs:.6f}` pp, p95 `{p95_abs:.6f}` pp, max `{max_abs:.6f}` pp.", "", "## Dataset Outcomes", "", "| Target | Positive | Strong | mean G std | Supported |", "|---|---:|---:|---:|---|"]
    for r in dataset_rows:
        report.append(f"| {r['target']} | {r['positive_cells']}/4 | {r['strong_cells']}/4 | {r['mean_G_pixel_std']:+.4f} | {r['SUPPORTED']} |")
    report.extend(["", "## Cell Deltas", "", "| System | Target | dP | dPRO_std | G_std | Positive | Strong |", "|---|---|---:|---:|---:|---|---|"])
    for r in cell_outcomes:
        report.append(f"| {r['system']} | {r['target']} | {r['dP-AUROC']:+.4f} | {r['dPRO_std']:+.4f} | {r['G_pixel_std']:+.4f} | {r['positive']} | {r['strong_cell']} |")
    report.append("")
    report.append("STOP: no additional dataset was evaluated.")
    (OUT / "XXVIII_B2_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXVIII-B2 Standardized PRO Re-Adjudication\n\n")
        fp.write(f"- outcome: `{outcome}`\n")
        fp.write(f"- PRO parity mean abs `{mean_abs:.6f}` pp, p95 `{p95_abs:.6f}` pp, max `{max_abs:.6f}` pp.\n")
        for r in dataset_rows:
            fp.write(f"- {r['target']}: positive {r['positive_cells']}/4, strong {r['strong_cells']}/4, mean G_std {r['mean_G_pixel_std']:+.4f}, supported {r['SUPPORTED']}.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["audit", "cache", "evaluate"], required=True)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        audit_pro_implementations()
        write_csv(OUT / "PRO_SYNTHETIC_TESTS.csv", synthetic_tests())
        print("XXVIII_B2_AUDIT_DONE")
    elif args.stage == "cache":
        for target in [x.strip() for x in args.targets.split(",") if x.strip()]:
            for system in [x.strip() for x in args.systems.split(",") if x.strip()]:
                cache_cell(system, target, args.device)
        print("XXVIII_B2_CACHE_DONE")
    else:
        run_evaluate()
        print(json.loads((OUT / "OUTCOME.json").read_text(encoding="utf-8"))["outcome"])


if __name__ == "__main__":
    main()
