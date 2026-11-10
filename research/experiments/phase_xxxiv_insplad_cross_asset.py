import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from phase_xa.official_task_forward import normalized_image_feature, normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


DATA_ROOT = Path(r"E:\WorkSpace\dataset\InsPLAD_AD")
CACHE = Path(r"E:\WorkSpace\phase_xxxiv_insplad_cache")
OUT = Path("results/phase_xxxiv_insplad_cross_asset")
IMAGE_SIZE = 518
SIDE = 37
PATCHES = SIDE * SIDE
TOPK = 5
TOP_PERCENT = 0.01
PREVIOUS_VERIFIED_BUDGET = 20_000
EPS = 1e-12
CHECKPOINT = Path("checkpoints/test_on_mvtec.pth")


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def cohen_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = math.sqrt(((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / max(1, a.size + b.size - 2))
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else float("nan")


def image_files(path):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(path.glob(ext))
    return sorted(files, key=lambda p: p.name.lower())


def sha1_file(path, max_bytes=1_000_000):
    h = hashlib.sha1()
    with open(path, "rb") as fp:
        h.update(fp.read(max_bytes))
    return h.hexdigest()


def mask_for(asset_dir, image_path):
    gt = asset_dir / "ground_truth" / "anomaly"
    candidates = [
        gt / image_path.name.replace(".jpg", "_mask.png"),
        gt / image_path.name.replace(".jpeg", "_mask.png"),
        gt / image_path.name.replace(".jpg", ".png"),
        gt / image_path.name.replace(".jpeg", ".png"),
    ]
    stem = image_path.stem
    candidates.extend(sorted(gt.glob(stem + "*mask*.png")))
    candidates.extend(sorted(gt.glob(stem + "*.png")))
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_dataset():
    assets = []
    rows = []
    image_rows = []
    duplicate = defaultdict(list)
    for asset_dir in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name):
        asset = asset_dir.name
        train_good = image_files(asset_dir / "train" / "good")
        test_good = image_files(asset_dir / "test" / "good")
        test_anom = image_files(asset_dir / "test" / "anomaly")
        masks = image_files(asset_dir / "ground_truth" / "anomaly")
        dims = defaultdict(int)
        missing_masks = 0
        nonempty_masks = 0
        for split, label, files in [("train", 0, train_good), ("test", 0, test_good), ("test", 1, test_anom)]:
            for p in files:
                with Image.open(p) as im:
                    dims[f"{im.size[0]}x{im.size[1]}"] += 1
                mf = mask_for(asset_dir, p) if label == 1 else None
                if label == 1:
                    missing_masks += int(mf is None)
                    if mf is not None:
                        with Image.open(mf) as m:
                            nonempty_masks += int(np.asarray(m.convert("L")).max() > 0)
                key = p.name.split("_jpg.rf.")[0].replace(" ", "-")
                duplicate[key].append(str(p))
                image_rows.append({"asset": asset, "split": split, "label": label, "path": str(p), "mask_path": str(mf) if mf else "", "provenance_key": key})
        row = {
            "asset": asset,
            "train_normal": len(train_good),
            "test_normal": len(test_good),
            "test_anomaly": len(test_anom),
            "mask_files": len(masks),
            "missing_anomaly_masks": missing_masks,
            "nonempty_anomaly_masks": nonempty_masks,
            "mask_complete": missing_masks == 0 and len(masks) >= len(test_anom),
            "dominant_resolution": max(dims.items(), key=lambda kv: kv[1])[0] if dims else "",
            "resolution_histogram": json.dumps(dict(sorted(dims.items())), ensure_ascii=False),
        }
        rows.append(row)
        assets.append({"name": asset, "dir": asset_dir, "train_good": train_good, "test_good": test_good, "test_anomaly": test_anom})
    duplicate_groups = {k: v for k, v in duplicate.items() if len(v) > 1}
    return assets, rows, image_rows, duplicate_groups


def write_dataset_audit(assets, count_rows, image_rows, duplicate_groups):
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_ok = len(assets) >= 2 and all(r["train_normal"] > 0 and r["test_normal"] > 0 and r["test_anomaly"] > 0 for r in count_rows)
    mask_ok = all(bool(r["mask_complete"]) for r in count_rows)
    audit = {
        "phase": "XXXIV-A",
        "dataset_root": str(DATA_ROOT),
        "asset_count": len(assets),
        "assets": [a["name"] for a in assets],
        "protocol_supported": protocol_ok,
        "pixel_masks_available": mask_ok,
        "image_level_label_source": "folder names: train/good, test/good, test/anomaly",
        "pixel_mask_source": "ground_truth/anomaly matched by anomaly image filename stem",
        "train_test_split_source": "local directory split",
        "asset_crop_status": "appears to be asset crops with square/cropped image files; original provenance retained in filenames",
        "duplicate_or_multi_crop_key_count": len(duplicate_groups),
        "duplicate_or_multi_crop_examples": dict(list(duplicate_groups.items())[:10]),
    }
    (OUT / "DATASET_AUDIT.json").write_text(json.dumps(audit, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "DATASET_COUNTS.csv", count_rows)
    lines = ["# Phase XXXIV-A Dataset Structure", "", f"Root: `{DATA_ROOT}`", "", f"Asset count: {len(assets)}", "", "| Asset | Train normal | Test normal | Test anomaly | Masks | Complete masks | Dominant resolution |", "|---|---:|---:|---:|---:|---|---|"]
    for r in count_rows:
        lines.append(f"| {r['asset']} | {r['train_normal']} | {r['test_normal']} | {r['test_anomaly']} | {r['mask_files']} | {r['mask_complete']} | {r['dominant_resolution']} |")
    lines.extend(["", f"Protocol supported: **{protocol_ok}**.", f"Pixel masks available for anomaly images: **{mask_ok}**.", f"Duplicate/provenance groups: {len(duplicate_groups)}."])
    (OUT / "DATASET_STRUCTURE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def dino_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def extract_clip_one(model, preprocess, path, device):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    out = visual_outputs(model, x, adapter=None, grad=False)
    cls = normalized_image_feature(out).squeeze(0).float().cpu().numpy().astype(np.float32)
    patch = normalized_patch_feature(out).squeeze(0).float().cpu().numpy().astype(np.float32)
    return cls, patch


@torch.no_grad()
def extract_dino_one(model, transform, path, device):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    out = model.forward_features(x)
    cls = F.normalize(out["x_norm_clstoken"].float(), dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
    patch = F.normalize(out["x_norm_patchtokens"].float(), dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
    return cls, patch


def asset_records(asset):
    rows = []
    for p in asset["train_good"]:
        rows.append({"asset": asset["name"], "split": "train", "label": 0, "path": str(p), "mask_path": ""})
    for p in asset["test_good"]:
        rows.append({"asset": asset["name"], "split": "test", "label": 0, "path": str(p), "mask_path": ""})
    for p in asset["test_anomaly"]:
        rows.append({"asset": asset["name"], "split": "test", "label": 1, "path": str(p), "mask_path": str(mask_for(asset["dir"], p) or "")})
    return rows


def cache_asset_features(asset, carrier, device):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"features_{carrier}_{asset['name']}.npz"
    patch_npy = CACHE / f"features_{carrier}_{asset['name']}.patch.npy"
    meta_path = CACHE / f"features_{carrier}_{asset['name']}.json"
    rows = asset_records(asset)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not patch_npy.exists():
            print(f"[XXXIV-A] converting {carrier} {asset['name']} patch cache to mmap .npy", flush=True)
            z = np.load(path)
            np.save(patch_npy, z["patch"])
        meta["patch_npy"] = str(patch_npy)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return patch_npy, meta
    print(f"[XXXIV-A] extracting {carrier} features for {asset['name']} ({len(rows)} images)", flush=True)
    if carrier == "clip":
        preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
        model, *_ = load_components(CHECKPOINT, device)
        extractor = lambda p: extract_clip_one(model, preprocess, p, device)
    else:
        tfm = dino_transform()
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True).to(device).eval()
        extractor = lambda p: extract_dino_one(model, tfm, p, device)
    cls_list, patch_list = [], []
    for i, r in enumerate(rows):
        cls, patch = extractor(r["path"])
        cls_list.append(cls.astype(np.float16))
        patch_list.append(patch.astype(np.float16))
        if (i + 1) % 100 == 0:
            print(f"[XXXIV-A] {carrier} {asset['name']} {i+1}/{len(rows)}", flush=True)
    cls_arr = np.stack(cls_list).astype(np.float16)
    patch_arr = np.stack(patch_list).astype(np.float16)
    np.savez_compressed(path, cls=cls_arr, patch=patch_arr)
    np.save(patch_npy, patch_arr)
    meta = {"asset": asset["name"], "carrier": carrier, "rows": rows, "cls_shape": list(cls_arr.shape), "patch_shape": list(patch_arr.shape)}
    meta["patch_npy"] = str(patch_npy)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del model, cls_arr, patch_arr, cls_list, patch_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path, meta


def norm_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def fps_indices(features, budget):
    x = norm_rows(features)
    n = x.shape[0]
    if n <= budget:
        return np.arange(n, dtype=np.int64)
    centroid = norm_rows(x.mean(axis=0, keepdims=True))[0]
    first = int(np.argmin(x @ centroid))
    selected = [first]
    max_sim = x @ x[first]
    for _ in range(1, budget):
        idx = int(np.argmin(max_sim))
        selected.append(idx)
        max_sim = np.maximum(max_sim, x @ x[idx])
    return np.asarray(selected, dtype=np.int64)


def local_patch_pick(patches, count, mode, rng):
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if mode == "random":
        return np.sort(rng.choice(PATCHES, size=min(count, PATCHES), replace=False)).astype(np.int64)
    return fps_indices(patches.astype(np.float32), min(count, PATCHES))


def allocate_counts(total, image_count):
    if image_count <= 0:
        return []
    base = total // image_count
    rem = total % image_count
    return [base + int(i < rem) for i in range(image_count)]


def build_memory(source_assets, carrier, feature_paths, metas, total_budget, mode, seed=0):
    rng = np.random.default_rng(seed)
    per_asset_target = total_budget // len(source_assets)
    rem = total_budget % len(source_assets)
    parts, entries, comp = [], [], []
    for ai, asset in enumerate(sorted(source_assets)):
        target = per_asset_target + int(ai < rem)
        patch_arr = np.load(feature_paths[(carrier, asset)], mmap_mode="r")
        meta = metas[(carrier, asset)]
        train_positions = [i for i, r in enumerate(meta["rows"]) if r["split"] == "train" and int(r["label"]) == 0]
        counts = allocate_counts(target, len(train_positions))
        asset_parts = []
        for pos, cnt in zip(train_positions, counts):
            patch = patch_arr[pos].astype(np.float32)
            ids = local_patch_pick(patch, cnt, mode, rng)
            if ids.size:
                asset_parts.append(patch[ids])
                for pid in ids:
                    entries.append({"source_asset": asset, "source_image_id": meta["rows"][pos]["path"], "source_position": int(pos), "memory_patch_id": int(pid)})
        if asset_parts:
            arr = np.concatenate(asset_parts, axis=0)
            if arr.shape[0] > target:
                keep = fps_indices(arr, target) if mode == "fps" else np.sort(rng.choice(arr.shape[0], size=target, replace=False))
                arr = arr[keep]
                asset_entries = [entries[-len(np.concatenate(asset_parts, axis=0)):][int(i)] for i in keep]
                entries = entries[: -len(np.concatenate(asset_parts, axis=0))] + asset_entries
            parts.append(arr)
            comp.append({"carrier": carrier, "mode": mode, "source_asset": asset, "target_entries": target, "actual_entries": int(arr.shape[0]), "normal_train_images": len(train_positions), "shortage": int(max(0, target - arr.shape[0]))})
    bank = np.concatenate(parts, axis=0).astype(np.float32) if parts else np.zeros((0, 1), dtype=np.float32)
    bank = norm_rows(bank)
    return bank, entries, comp


def topk_distances(query_patch, bank, k=TOPK, device="cuda"):
    q = torch.tensor(norm_rows(query_patch.astype(np.float32)), dtype=torch.float32, device=device)
    b = torch.tensor(bank, dtype=torch.float32, device=device)
    vals, inds = [], []
    with torch.no_grad():
        for start in range(0, q.shape[0], 512):
            sim = q[start : start + 512] @ b.t()
            v, ix = torch.topk(sim, k=min(k, b.shape[0]), dim=1)
            vals.append((1.0 - v).detach().cpu().numpy())
            inds.append(ix.detach().cpu().numpy())
    return np.concatenate(vals, axis=0), np.concatenate(inds, axis=0)


def image_score_from_patch_scores(scores):
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    k = max(1, int(math.ceil(flat.size * TOP_PERCENT)))
    return float(np.sort(flat)[-k:].mean())


def upsample_patch_scores(scores37, size):
    x = torch.tensor(scores37, dtype=torch.float32).view(1, 1, SIDE, SIDE)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)


def evaluate_target(target, carrier, mode, bank, entries, feature_paths, metas, device, write_raw=False):
    patch_arr = np.load(feature_paths[(carrier, target)], mmap_mode="r")
    meta = metas[(carrier, target)]
    rows = [r for r in meta["rows"] if r["split"] == "test"]
    positions = [i for i, r in enumerate(meta["rows"]) if r["split"] == "test"]
    y_img, s_img = [], []
    y_pix, s_pix = [], []
    support_rows = []
    retrieval_counter = defaultdict(int)
    unique_keys = set()
    raw_writer = None
    raw_fp = None
    if write_raw:
        raw_path = CACHE / f"RETRIEVAL_RAW_{target}_{carrier}_{mode}.csv.gz"
        raw_fp = gzip.open(raw_path, "wt", newline="", encoding="utf-8")
        raw_writer = csv.DictWriter(raw_fp, fieldnames=["target_asset", "carrier", "mode", "target_image", "target_label", "query_patch_id", "rank", "source_asset", "source_image_id", "memory_patch_id", "distance"])
        raw_writer.writeheader()
    for pos, r in zip(positions, rows):
        patch = patch_arr[pos].astype(np.float32)
        dist, idx = topk_distances(patch, bank, TOPK, device)
        patch_score = dist.mean(axis=1).reshape(SIDE, SIDE)
        img_score = image_score_from_patch_scores(patch_score)
        y_img.append(int(r["label"]))
        s_img.append(img_score)
        top1_assets = [entries[int(i)]["source_asset"] for i in idx[:, 0]]
        for a in top1_assets:
            retrieval_counter[a] += 1
        for i in idx.reshape(-1):
            unique_keys.add(int(i))
        top_assets_all = []
        for qid in range(idx.shape[0]):
            for rank in range(idx.shape[1]):
                e = entries[int(idx[qid, rank])]
                top_assets_all.append(e["source_asset"])
                if raw_writer is not None:
                    raw_writer.writerow({"target_asset": target, "carrier": carrier, "mode": mode, "target_image": r["path"], "target_label": int(r["label"]), "query_patch_id": qid, "rank": rank + 1, "source_asset": e["source_asset"], "source_image_id": e["source_image_id"], "memory_patch_id": e["memory_patch_id"], "distance": float(dist[qid, rank])})
        vals, counts = np.unique(top_assets_all, return_counts=True)
        probs = counts / max(1, counts.sum())
        entropy = float(-(probs * np.log(probs + EPS)).sum())
        margin = float(np.mean(dist[:, 1] - dist[:, 0])) if dist.shape[1] > 1 else float("nan")
        support_rows.append({"target_asset": target, "carrier": carrier, "mode": mode, "target_image": r["path"], "label": int(r["label"]), "image_score": img_score, "mean_top5_distance": float(dist.mean()), "median_top5_distance": float(np.median(dist)), "q90_patch_distance": float(np.quantile(patch_score, 0.90)), "top1_margin": margin, "retrieval_entropy": entropy, "dominant_source_asset": vals[int(np.argmax(counts))] if vals.size else "", "dominant_source_asset_fraction": float(counts.max() / counts.sum()) if counts.size else float("nan")})
        if r["mask_path"]:
            with Image.open(r["mask_path"]) as m:
                mask = (np.asarray(m.convert("L")) > 0).astype(np.uint8)
            full = upsample_patch_scores(patch_score, mask.shape)
            y_pix.append(mask.reshape(-1))
            s_pix.append(full.reshape(-1))
        elif int(r["label"]) == 0:
            with Image.open(r["path"]) as im:
                size = (im.size[1], im.size[0])
            mask = np.zeros(size, dtype=np.uint8)
            full = upsample_patch_scores(patch_score, size)
            y_pix.append(mask.reshape(-1))
            s_pix.append(full.reshape(-1))
    if raw_fp is not None:
        raw_fp.close()
    y_img = np.asarray(y_img, dtype=np.int64)
    s_img = np.asarray(s_img, dtype=np.float32)
    metric = {"I-AUROC": safe_auc(y_img, s_img), "I-AP": safe_ap(y_img, s_img), "anomaly_prevalence": float(y_img.mean() * 100.0), "AP_lift": safe_ap(y_img, s_img) - float(y_img.mean() * 100.0)}
    if y_pix and np.unique(np.concatenate(y_pix)).size == 2:
        metric["P-AUROC"] = safe_auc(np.concatenate(y_pix), np.concatenate(s_pix))
        metric["P-AP"] = safe_ap(np.concatenate(y_pix), np.concatenate(s_pix))
    else:
        metric["P-AUROC"] = float("nan")
        metric["P-AP"] = float("nan")
    total_top1 = sum(retrieval_counter.values())
    metric["unique_memory_key_count"] = len(unique_keys)
    metric["dominant_retrieved_source_asset"] = max(retrieval_counter.items(), key=lambda kv: kv[1])[0] if retrieval_counter else ""
    metric["dominant_retrieved_source_fraction"] = max(retrieval_counter.values()) / max(1, total_top1) if retrieval_counter else float("nan")
    return metric, support_rows, retrieval_counter


def g_image(a, b):
    return 0.5 * (float(a["I-AUROC"]) - float(b["I-AUROC"])) + 0.5 * (float(a["I-AP"]) - float(b["I-AP"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    start = time.time()
    assets, count_rows, image_rows, duplicate_groups = collect_dataset()
    audit = write_dataset_audit(assets, count_rows, image_rows, duplicate_groups)
    required_empty = [
        "FEATURE_AUDIT.json", "PATCH_ALIGNMENT.csv", "SPLIT_PROTOCOL.json", "MEMORY_BUDGET.json", "MEMORY_COMPOSITION.csv",
        "RAW_METRICS.csv", "PER_ASSET.csv", "GROUPED_METRICS.csv", "RETRIEVAL_STRUCTURE.csv", "SOURCE_ASSET_REMOVAL.csv",
        "ASSET_IDENTITY_SENSITIVITY.csv", "NORMAL_ANOMALY_SUPPORT.csv", "CARRIER_PREFERENCE.csv", "FPS_VS_RANDOM.csv",
        "PIXEL_METRICS.csv", "XXXIV_A_REPORT.md", "OUTCOME.json",
    ]
    if not audit["protocol_supported"]:
        for name in required_empty:
            p = OUT / name
            if name.endswith(".json"):
                p.write_text(json.dumps({"status": "NOT_AVAILABLE"}, indent=2), encoding="utf-8")
            elif name.endswith(".md"):
                p.write_text("# Phase XXXIV-A\n\nOutcome: XXXIV_A_DATASET_PROTOCOL_UNSUPPORTED\n", encoding="utf-8")
            else:
                p.write_text("status\nNOT_AVAILABLE\n", encoding="utf-8")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_paths, metas = {}, {}
    for carrier in ["clip", "dino"]:
        for asset in assets:
            p, m = cache_asset_features(asset, carrier, device)
            feature_paths[(carrier, asset["name"])] = p
            metas[(carrier, asset["name"])] = m

    feature_audit = {
        "clip": {"backbone": "Official MRAD-CLIP AnomalyCLIP ViT-L/14@336px visual forward", "input_size": IMAGE_SIZE, "patch_grid": [SIDE, SIDE], "checkpoint": str(CHECKPOINT)},
        "dino": {"backbone": "DINOv2 ViT-L/14 via torch.hub facebookresearch/dinov2:dinov2_vitl14", "input_size": IMAGE_SIZE, "patch_grid": [SIDE, SIDE]},
        "cache_root": str(CACHE),
        "device": device,
    }
    (OUT / "FEATURE_AUDIT.json").write_text(json.dumps(feature_audit, indent=2), encoding="utf-8")
    write_csv(OUT / "PATCH_ALIGNMENT.csv", [{"carrier_a": "clip", "carrier_b": "dino", "clip_grid": "37x37", "dino_grid": "37x37", "alignment": "identical normalized 37x37 patch coordinate grid", "max_coordinate_error": 0.0}])

    asset_names = [a["name"] for a in assets]
    split = [{"target_asset": t, "source_assets": [s for s in asset_names if s != t], "forbidden_target_usage": "target normal/anomaly excluded from memory/statistics/selection"} for t in asset_names]
    (OUT / "SPLIT_PROTOCOL.json").write_text(json.dumps({"protocol": "leave-one-asset-out", "folds": split}, indent=2), encoding="utf-8")

    raw_rows, per_asset_rows, retrieval_structure, removal_rows, sensitivity_rows = [], [], [], [], []
    support_all, carrier_pref, fps_random_rows, memory_comp = [], [], [], []
    pixel_rows = []
    memory_budget_rows = []
    metric_lookup = {}
    removal_lookup = defaultdict(list)

    for target in asset_names:
        print(f"[XXXIV-A] target fold {target}", flush=True)
        sources = [s for s in asset_names if s != target]
        eligible = sum(len([r for r in metas[("clip", s)]["rows"] if r["split"] == "train" and int(r["label"]) == 0]) * PATCHES for s in sources)
        budget = min(PREVIOUS_VERIFIED_BUDGET, eligible)
        memory_budget_rows.append({"target_asset": target, "source_assets": "|".join(sources), "previous_verified_budget": PREVIOUS_VERIFIED_BUDGET, "eligible_source_normal_patch_count": eligible, "final_budget": budget})
        for carrier in ["clip", "dino"]:
            for mode, method in [("fps", f"{carrier.upper()}_FPS"), ("random", f"{carrier.upper()}_RANDOM")]:
                print(f"[XXXIV-A] {target} {method}: build memory/evaluate", flush=True)
                bank, entries, comp = build_memory(sources, carrier, feature_paths, metas, budget, mode, seed=0)
                memory_comp.extend([{"target_asset": target, **r} for r in comp])
                metric, support_rows, ret_counter = evaluate_target(target, carrier, mode, bank, entries, feature_paths, metas, device, write_raw=(mode == "fps"))
                row = {"target_asset": target, "carrier": carrier.upper(), "method": method, "memory_mode": mode, "memory_size": len(entries), **metric}
                raw_rows.append(row)
                metric_lookup[(target, carrier, mode)] = row
                support_all.extend(support_rows)
                pixel_rows.append({"target_asset": target, "carrier": carrier.upper(), "method": method, "P-AUROC": metric["P-AUROC"], "P-AP": metric["P-AP"], "status": "AVAILABLE" if np.isfinite(metric["P-AUROC"]) else "NOT_AVAILABLE"})
                vals = np.asarray(list(ret_counter.values()), dtype=np.float64)
                probs = vals / max(1.0, vals.sum())
                retrieval_structure.append({"target_asset": target, "carrier": carrier.upper(), "method": method, "top1_source_asset_frequency": json.dumps(dict(ret_counter), ensure_ascii=False), "topk_source_asset_entropy_mean": float(np.mean([r["retrieval_entropy"] for r in support_rows])), "dominant_source_asset": metric["dominant_retrieved_source_asset"], "dominant_source_asset_fraction": metric["dominant_retrieved_source_fraction"], "unique_memory_key_count": metric["unique_memory_key_count"], "retrieval_gini_proxy": float(1.0 - np.sum(probs ** 2)) if probs.size else float("nan"), "top1_top5_margin_mean": float(np.mean([r["top1_margin"] for r in support_rows])), "normal_retrieval_entropy": float(np.mean([r["retrieval_entropy"] for r in support_rows if int(r["label"]) == 0])), "anomaly_retrieval_entropy": float(np.mean([r["retrieval_entropy"] for r in support_rows if int(r["label"]) == 1])), "normal_anomaly_support_distance_separation": float(np.mean([r["mean_top5_distance"] for r in support_rows if int(r["label"]) == 1]) - np.mean([r["mean_top5_distance"] for r in support_rows if int(r["label"]) == 0]))})

            # Source-asset removal for FPS only.
            full = metric_lookup[(target, carrier, "fps")]
            use_counter_full = None
            for removed in sources:
                print(f"[XXXIV-A] {target} {carrier.upper()}_FPS remove {removed}", flush=True)
                remain = [s for s in sources if s != removed]
                bank, entries, comp = build_memory(remain, carrier, feature_paths, metas, budget, "fps", seed=0)
                metric, _, _ = evaluate_target(target, carrier, "fps", bank, entries, feature_paths, metas, device, write_raw=False)
                drop = 0.5 * (full["I-AUROC"] - metric["I-AUROC"]) + 0.5 * (full["I-AP"] - metric["I-AP"])
                rrow = {"target_asset": target, "carrier": carrier.upper(), "removed_source_asset": removed, "remaining_source_assets": "|".join(remain), "full_I-AUROC": full["I-AUROC"], "full_I-AP": full["I-AP"], "remove_I-AUROC": metric["I-AUROC"], "remove_I-AP": metric["I-AP"], "G_image_drop": drop, "memory_size": len(entries)}
                removal_rows.append(rrow)
                removal_lookup[(target, carrier)].append(rrow)

    for target in asset_names:
        cf, cr = metric_lookup[(target, "clip", "fps")], metric_lookup[(target, "clip", "random")]
        df, dr = metric_lookup[(target, "dino", "fps")], metric_lookup[(target, "dino", "random")]
        best = max([cf, df], key=lambda r: 0.5 * (r["I-AUROC"] + r["I-AP"]))
        per_asset_rows.append({"target_asset": target, "normal_count": int(next(r for r in count_rows if r["asset"] == target)["test_normal"]), "anomaly_count": int(next(r for r in count_rows if r["asset"] == target)["test_anomaly"]), "anomaly_prevalence": cf["anomaly_prevalence"], "CLIP_FPS_I-AUROC": cf["I-AUROC"], "CLIP_FPS_I-AP": cf["I-AP"], "CLIP_RANDOM_I-AUROC": cr["I-AUROC"], "CLIP_RANDOM_I-AP": cr["I-AP"], "DINO_FPS_I-AUROC": df["I-AUROC"], "DINO_FPS_I-AP": df["I-AP"], "DINO_RANDOM_I-AUROC": dr["I-AUROC"], "DINO_RANDOM_I-AP": dr["I-AP"], "best_carrier": best["carrier"], "dominant_retrieved_source_asset": best["dominant_retrieved_source_asset"], "dominant_retrieved_source_fraction": best["dominant_retrieved_source_fraction"], "retrieval_entropy": next(r for r in retrieval_structure if r["target_asset"] == target and r["carrier"] == best["carrier"] and r["method"].endswith("FPS"))["topk_source_asset_entropy_mean"], "source_asset_removal_sensitivity": max([r["G_image_drop"] for r in removal_lookup[(target, best["carrier"].lower())]]) if removal_lookup[(target, best["carrier"].lower())] else float("nan"), "P-AUROC": best["P-AUROC"], "P-AP": best["P-AP"]})
        pref_g = 0.5 * (df["I-AUROC"] - cf["I-AUROC"]) + 0.5 * (df["I-AP"] - cf["I-AP"])
        carrier_pref.append({"target_asset": target, "DINO_minus_CLIP_I-AUROC": df["I-AUROC"] - cf["I-AUROC"], "DINO_minus_CLIP_I-AP": df["I-AP"] - cf["I-AP"], "G_carrier": pref_g, "preferred": "TIE" if abs(pref_g) <= 0.10 else ("DINO" if pref_g > 0 else "CLIP")})
        for carrier, fps, rand in [("CLIP", cf, cr), ("DINO", df, dr)]:
            fps_random_rows.append({"target_asset": target, "carrier": carrier, "FPS_minus_RANDOM_G_image": g_image(fps, rand), "FPS_I-AUROC": fps["I-AUROC"], "RANDOM_I-AUROC": rand["I-AUROC"], "FPS_I-AP": fps["I-AP"], "RANDOM_I-AP": rand["I-AP"], "FPS_better": g_image(fps, rand) > 0})

    for key, rows in removal_lookup.items():
        target, carrier = key
        vals = [r["G_image_drop"] for r in rows]
        full_ret = [r for r in retrieval_structure if r["target_asset"] == target and r["carrier"] == carrier.upper() and r["method"].endswith("FPS")][0]
        freq = json.loads(full_ret["top1_source_asset_frequency"])
        x = [float(freq.get(r["removed_source_asset"], 0.0)) for r in rows]
        rho = float(spearmanr(x, vals).statistic) if len(rows) > 1 and len(set(x)) > 1 and len(set(vals)) > 1 else float("nan")
        sensitive = any(v >= 0.50 for v in vals)
        sensitivity_rows.append({"target_asset": target, "carrier": carrier.upper(), "max_drop": float(np.max(vals)), "min_drop": float(np.min(vals)), "std_drop": float(np.std(vals)), "range_drop": float(np.max(vals) - np.min(vals)), "ASSET_IDENTITY_SENSITIVE": sensitive, "retrieval_use_vs_removal_drop_spearman": rho})

    support_summary = []
    for (target, carrier, mode), group in defaultdict(list, {}).items():
        pass
    for target in asset_names:
        for carrier in ["CLIP", "DINO"]:
            rows = [r for r in support_all if r["target_asset"] == target and r["carrier"] == carrier.lower() and r["mode"] == "fps"]
            normal = [r for r in rows if int(r["label"]) == 0]
            anom = [r for r in rows if int(r["label"]) == 1]
            support_summary.append({"target_asset": target, "carrier": carrier, "normal_mean_top5_distance": float(np.mean([r["mean_top5_distance"] for r in normal])), "anomaly_mean_top5_distance": float(np.mean([r["mean_top5_distance"] for r in anom])), "normal_median_top5_distance": float(np.mean([r["median_top5_distance"] for r in normal])), "anomaly_median_top5_distance": float(np.mean([r["median_top5_distance"] for r in anom])), "normal_q90_patch_distance": float(np.mean([r["q90_patch_distance"] for r in normal])), "anomaly_q90_patch_distance": float(np.mean([r["q90_patch_distance"] for r in anom])), "normal_top1_margin": float(np.mean([r["top1_margin"] for r in normal])), "anomaly_top1_margin": float(np.mean([r["top1_margin"] for r in anom])), "normal_retrieval_entropy": float(np.mean([r["retrieval_entropy"] for r in normal])), "anomaly_retrieval_entropy": float(np.mean([r["retrieval_entropy"] for r in anom])), "cohen_d_mean_top5_distance": cohen_d([r["mean_top5_distance"] for r in normal], [r["mean_top5_distance"] for r in anom]), "ks_mean_top5_distance": float(ks_2samp([r["mean_top5_distance"] for r in normal], [r["mean_top5_distance"] for r in anom]).statistic), "I-AUROC_distance_score": metric_lookup[(target, carrier.lower(), "fps")]["I-AUROC"], "I-AP_distance_score": metric_lookup[(target, carrier.lower(), "fps")]["I-AP"]})

    write_csv(OUT / "MEMORY_BUDGET.json.tmp.csv", memory_budget_rows)
    (OUT / "MEMORY_BUDGET.json").write_text(json.dumps({"previous_verified_budget": PREVIOUS_VERIFIED_BUDGET, "folds": memory_budget_rows}, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "MEMORY_COMPOSITION.csv", memory_comp)
    write_csv(OUT / "RAW_METRICS.csv", raw_rows)
    write_csv(OUT / "PER_ASSET.csv", per_asset_rows)
    write_csv(OUT / "GROUPED_METRICS.csv", raw_rows)
    write_csv(OUT / "RETRIEVAL_STRUCTURE.csv", retrieval_structure)
    write_csv(OUT / "SOURCE_ASSET_REMOVAL.csv", removal_rows)
    write_csv(OUT / "ASSET_IDENTITY_SENSITIVITY.csv", sensitivity_rows)
    write_csv(OUT / "NORMAL_ANOMALY_SUPPORT.csv", support_summary)
    write_csv(OUT / "CARRIER_PREFERENCE.csv", carrier_pref)
    write_csv(OUT / "FPS_VS_RANDOM.csv", fps_random_rows)
    write_csv(OUT / "PIXEL_METRICS.csv", pixel_rows)

    grouped = []
    for carrier in ["CLIP", "DINO"]:
        fps = [r for r in raw_rows if r["carrier"] == carrier and r["memory_mode"] == "fps"]
        rand = [r for r in raw_rows if r["carrier"] == carrier and r["memory_mode"] == "random"]
        grouped.append({"carrier": carrier, "method": "FPS", "mean_I-AUROC": float(np.mean([r["I-AUROC"] for r in fps])), "mean_I-AP": float(np.mean([r["I-AP"] for r in fps])), "mean_AP_lift": float(np.mean([r["AP_lift"] for r in fps])), "asset_fraction_I-AUROC_gt65": float(np.mean([r["I-AUROC"] > 65.0 for r in fps])), "mean_P-AUROC": float(np.mean([r["P-AUROC"] for r in fps])), "mean_P-AP": float(np.mean([r["P-AP"] for r in fps]))})
        grouped.append({"carrier": carrier, "method": "RANDOM", "mean_I-AUROC": float(np.mean([r["I-AUROC"] for r in rand])), "mean_I-AP": float(np.mean([r["I-AP"] for r in rand])), "mean_AP_lift": float(np.mean([r["AP_lift"] for r in rand])), "asset_fraction_I-AUROC_gt65": float(np.mean([r["I-AUROC"] > 65.0 for r in rand])), "mean_P-AUROC": float(np.mean([r["P-AUROC"] for r in rand])), "mean_P-AP": float(np.mean([r["P-AP"] for r in rand]))})
    write_csv(OUT / "GROUPED_METRICS.csv", grouped)

    feasible = {r["carrier"]: r for r in grouped if r["method"] == "FPS" and r["mean_I-AUROC"] >= 75.0 and r["asset_fraction_I-AUROC_gt65"] >= 0.80 and r["mean_AP_lift"] >= 10.0}
    best_carrier = max([r for r in grouped if r["method"] == "FPS"], key=lambda r: r["mean_I-AUROC"])
    best_name = best_carrier["carrier"]
    best_fps_rand = [r for r in fps_random_rows if r["carrier"] == best_name]
    content_value = float(np.mean([r["FPS_minus_RANDOM_G_image"] for r in best_fps_rand]))
    content_win = float(np.mean([r["FPS_better"] for r in best_fps_rand]))
    sens = [r for r in sensitivity_rows if r["carrier"] == best_name]
    identity_sensitive_frac = float(np.mean([r["ASSET_IDENTITY_SENSITIVE"] for r in sens]))
    removal_range_mean = float(np.mean([r["range_drop"] for r in sens]))
    rhos = [r["retrieval_use_vs_removal_drop_spearman"] for r in sens if np.isfinite(r["retrieval_use_vs_removal_drop_spearman"])]
    pooled_rho = float(np.mean(rhos)) if rhos else float("nan")
    identity_gate = identity_sensitive_frac >= 0.60 and removal_range_mean >= 0.50 and np.isfinite(pooled_rho) and abs(pooled_rho) >= 0.30
    content_gate = content_value >= 0.20 and content_win >= 0.60
    carrier_prefs = [r["preferred"] for r in carrier_pref]
    carrier_asset_specific = len(set([p for p in carrier_prefs if p != "TIE"])) > 1 or ("CLIP" in carrier_prefs and "DINO" in carrier_prefs)
    if feasible and identity_gate:
        outcome = "XXXIV_A_POWER_MEMORY_SIGNAL_SUPPORTED"
    elif feasible and (not identity_gate or not content_gate):
        outcome = "XXXIV_A_POWER_SUPPORT_BUT_IDENTITY_WEAK"
    elif carrier_asset_specific:
        outcome = "XXXIV_A_CARRIER_ASSET_SPECIFIC"
    else:
        outcome = "XXXIV_A_MEMORY_GENERALIZATION_WEAK"

    report = ["# Phase XXXIV-A InsPLAD Cross-Asset Zero-Shot Memory Audit", "", f"Outcome: **{outcome}**", "", f"Dataset root: `{DATA_ROOT}`. Assets: {', '.join(asset_names)}.", "", "## Grouped Metrics", "", "| Carrier | Method | mean I-AUROC | mean I-AP | mean AP lift | frac I-AUROC>65 | mean P-AUROC | mean P-AP |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in grouped:
        report.append(f"| {r['carrier']} | {r['method']} | {r['mean_I-AUROC']:.4f} | {r['mean_I-AP']:.4f} | {r['mean_AP_lift']:.4f} | {r['asset_fraction_I-AUROC_gt65']:.4f} | {r['mean_P-AUROC']:.4f} | {r['mean_P-AP']:.4f} |")
    report.extend(["", "## Per Asset", "", "| Target | Best | CLIP FPS I-AUROC/I-AP | DINO FPS I-AUROC/I-AP | Dominant source | Removal sensitivity |", "|---|---|---:|---:|---|---:|"])
    for r in per_asset_rows:
        report.append(f"| {r['target_asset']} | {r['best_carrier']} | {r['CLIP_FPS_I-AUROC']:.4f}/{r['CLIP_FPS_I-AP']:.4f} | {r['DINO_FPS_I-AUROC']:.4f}/{r['DINO_FPS_I-AP']:.4f} | {r['dominant_retrieved_source_asset']} | {r['source_asset_removal_sensitivity']:.4f} |")
    report.extend(["", "## Gates", "", f"- Cross-asset feasible carriers: {sorted(feasible)}.", f"- Best carrier: {best_name}; FPS-Random mean G={content_value:+.4f}; FPS win fraction={content_win:.4f}.", f"- Identity sensitive fraction={identity_sensitive_frac:.4f}; removal range mean={removal_range_mean:.4f}; pooled Spearman={pooled_rho}.", f"- Runtime: {time.time() - start:.1f}s.", "", "Stop: Phase XXXIV-A ended here."])
    (OUT / "XXXIV_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    payload = {"phase": "XXXIV-A", "outcome": outcome, "dataset_root": str(DATA_ROOT), "cache_root": str(CACHE), "asset_count": len(asset_names), "assets": asset_names, "pixel_masks_available": audit["pixel_masks_available"], "grouped": grouped, "best_carrier": best_name, "content_value_mean_G": content_value, "content_win_fraction": content_win, "identity_sensitive_fraction": identity_sensitive_frac, "removal_range_mean": removal_range_mean, "retrieval_use_removal_spearman_mean": pooled_rho, "runtime_sec": time.time() - start}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXXIV-A\n")
        fp.write("- Command: python experiments/phase_xxxiv_insplad_cross_asset.py\n")
        fp.write(f"- Local InsPLAD root: {DATA_ROOT}; assets: {', '.join(asset_names)}.\n")
        fp.write(f"- Split: leave-one-asset-out, {len(asset_names)} folds; source memory uses source train/good only.\n")
        fp.write(f"- Memory budget: {PREVIOUS_VERIFIED_BUDGET} max, balanced source assets; carriers CLIP local and DINOv2 ViT-L/14 local.\n")
        for r in per_asset_rows:
            fp.write(f"- {r['target_asset']}: CLIP FPS {r['CLIP_FPS_I-AUROC']:.4f}/{r['CLIP_FPS_I-AP']:.4f}; DINO FPS {r['DINO_FPS_I-AUROC']:.4f}/{r['DINO_FPS_I-AP']:.4f}; best {r['best_carrier']}.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXXIV_A_REPORT.md"), "runtime_sec": time.time() - start}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
