import argparse
import csv
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
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import fast_cal_pro_score
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from utils.dataset import Dataset, generate_class_info
from utils.transforms import get_transform


OUT = Path("results/phase_frozen_spatial_backbone_compatibility/formal")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
SEED = 0
EPOCHS = 10
LR = 1e-3
RANK = 4
SOURCE_CAP_PER_ORIGIN_CATEGORY = 512
BATCH_SIZE = 8192
TOP_NEG_FRAC = 0.10
IMAGE_SIZE_CLIP_PROTOCOL = 518
RADIO_SIZE = 512
RADIO_SIDE = 32
SIGMA = 4

B0_CURRENT = {
    "MVTec": {"P-AUROC": 90.3345, "P-AP": 32.5154, "G_pixel": 61.4250},
    "BTAD": {"P-AUROC": 93.8786, "P-AP": 51.3092, "G_pixel": 72.5939},
    "ClinicDB": {"P-AUROC": 86.6497, "P-AP": 43.5498, "G_pixel": 65.0998},
}

SOTA = {
    "MoECLIP": {
        "MVTec": {"P-AUROC": 92.5, "P-AP": 45.7, "G_pixel": 69.10},
        "BTAD": {"P-AUROC": 96.8, "P-AP": 50.4, "G_pixel": 73.60},
        "ClinicDB": {"P-AUROC": 89.7, "P-AP": 49.9, "G_pixel": 69.80},
    },
    "VisualAD": {
        "MVTec": {"P-AUROC": 90.8, "P-AP": 41.2, "G_pixel": 66.00},
        "BTAD": {"P-AUROC": 91.1, "P-AP": 43.1, "G_pixel": 67.10},
        "ClinicDB": {"P-AUROC": 85.2, "P-AP": 37.8, "G_pixel": 61.50},
    },
}


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
    return 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"]))


def norm_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pil_to_radio_tensor(path):
    img = Image.open(path).convert("RGB").resize((RADIO_SIZE, RADIO_SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def mask_to_patch_labels(mask_path, image_path, anomaly):
    img = Image.open(image_path).convert("RGB")
    if int(anomaly) == 0 or not mask_path:
        mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
    else:
        mp = Path(mask_path)
        if not mp.exists() or mp.is_dir():
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        else:
            mask_arr = (np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask_arr, mode="L")
    arr = np.asarray(mask.resize((RADIO_SIDE, RADIO_SIDE), Image.BILINEAR), dtype=np.float32) / 255.0
    return (arr.reshape(-1) > 0.5).astype(np.int64)


class RadioFeatureExtractor:
    def __init__(self, device, batch_size=2):
        self.device = device
        self.batch_size = batch_size
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version="radio_v2.5-l",
            progress=False,
            skip_validation=True,
            trust_repo=True,
        ).to(device).eval()

    @torch.no_grad()
    def extract_paths(self, paths):
        feats = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            x = torch.stack([pil_to_radio_tensor(p) for p in batch_paths]).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                _summary, spatial = self.model(x)
            spatial = spatial.float().detach().cpu().numpy().astype(np.float32)
            feats.extend([norm_np(v) for v in spatial])
        return feats


class PatchStateModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        torch.manual_seed(SEED)
        self.a = nn.Linear(dim, RANK, bias=False)
        self.b = nn.Linear(RANK, dim, bias=False)
        self.head = nn.Parameter(torch.randn(2, dim) * 0.02)
        nn.init.normal_(self.a.weight, std=0.02)
        nn.init.zeros_(self.b.weight)

    def adapt(self, x):
        return F.normalize(x + self.b(self.a(x)), dim=-1)

    def logits(self, x):
        z = self.adapt(F.normalize(x, dim=-1))
        h = F.normalize(self.head, dim=-1)
        return z @ h[1] - z @ h[0]


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


def raw_source_rows():
    _name, dataset, indices, *_ = load_full_visa_source_lazy(False)
    rows = []
    for idx in indices:
        d = dataset.data_all[int(idx)]
        image_path = Path(dataset.root) / d["img_path"]
        mask_path = Path(dataset.root) / d["mask_path"] if d.get("mask_path") else None
        rows.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None else "",
            "category": str(d["cls_name"]),
            "anomaly": int(d["anomaly"]),
        })
    return rows


def collect_source_pool(extractor, smoke=False):
    cache = OUT / "cache" / ("radio_source_pool_smoke.npz" if smoke else "radio_source_pool.npz")
    audit_path = OUT / "SOURCE_POOL_AUDIT.csv"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            categories = z["categories"].tolist()
            if categories:
                return {
                    cat: {"x": z[f"{cat}_x"].astype(np.float32), "y": z[f"{cat}_y"].astype(np.float32), "origin": z[f"{cat}_origin"]}
                    for cat in categories
                }
        cache.unlink()
    rows = raw_source_rows()
    if smoke:
        keep = []
        seen = defaultdict(lambda: defaultdict(int))
        for row in rows:
            label = "anomaly" if int(row["anomaly"]) else "normal"
            limit = 8 if label == "anomaly" else 2
            if seen[row["category"]][label] < limit:
                keep.append(row)
                seen[row["category"]][label] += 1
        rows = keep
    rng = np.random.default_rng(SEED)
    parts = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: defaultdict(int))
    audit = []
    for start in range(0, len(rows), extractor.batch_size):
        if start % max(1, extractor.batch_size * 25) == 0:
            print(f"[RADIO] source features {start}/{len(rows)}", flush=True)
        batch = rows[start : start + extractor.batch_size]
        feats = extractor.extract_paths([r["image_path"] for r in batch])
        for row, patch in zip(batch, feats):
            y = mask_to_patch_labels(row["mask_path"], row["image_path"], row["anomaly"])
            cat = row["category"]
            pools = []
            if int(row["anomaly"]) == 1:
                pools.append(("anomaly_mask", np.flatnonzero(y == 1), 1))
                pools.append(("anomaly_mask_out", np.flatnonzero(y == 0), 0))
            else:
                pools.append(("normal_image", np.flatnonzero(y == 0), 0))
            for origin, ids, label in pools:
                room = SOURCE_CAP_PER_ORIGIN_CATEGORY - counts[cat][origin]
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[cat][origin].append(patch[ids].astype(np.float32))
                counts[cat][origin] += int(ids.size)
    payload = {}
    groups = {}
    categories = []
    for cat in sorted(parts):
        xs, ys, origins = [], [], []
        for origin, label in [("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)]:
            arrs = parts[cat].get(origin, [])
            if not arrs:
                continue
            arr = np.concatenate(arrs, axis=0).astype(np.float32)
            xs.append(arr)
            ys.extend([label] * arr.shape[0])
            origins.extend([origin] * arr.shape[0])
        if not xs or len(set(ys)) < 2:
            continue
        x = norm_np(np.concatenate(xs, axis=0).astype(np.float32))
        y = np.asarray(ys, dtype=np.float32)
        origin = np.asarray(origins, dtype=object)
        groups[cat] = {"x": x, "y": y, "origin": origin}
        categories.append(cat)
        payload[f"{cat}_x"] = x.astype(np.float16)
        payload[f"{cat}_y"] = y.astype(np.float32)
        payload[f"{cat}_origin"] = origin
        audit.append({
            "category": cat,
            "samples": int(x.shape[0]),
            "positive": int(y.sum()),
            "negative": int((y == 0).sum()),
            "normal_image": int(np.sum(origin == "normal_image")),
            "anomaly_mask_out": int(np.sum(origin == "anomaly_mask_out")),
        })
    payload["categories"] = np.asarray(categories, dtype=object)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    write_csv(audit_path, audit)
    return groups


def balanced_batch(groups, rng, total_batch):
    names = sorted(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, origins = [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            if ids.size == 0:
                continue
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            origins.extend(data["origin"][choice].tolist())
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], np.asarray(origins, dtype=object)[order]


def init_head_from_centroids(model, groups):
    x = np.concatenate([groups[c]["x"] for c in groups], axis=0)
    y = np.concatenate([groups[c]["y"] for c in groups], axis=0)
    states = []
    for cls in (0, 1):
        states.append(norm_np(x[y == cls].mean(axis=0, keepdims=True))[0])
    with torch.no_grad():
        model.head.copy_(torch.tensor(np.stack(states), dtype=torch.float32, device=model.head.device))


def train_variant(name, mode, groups, device, smoke=False):
    ckpt = OUT / "checkpoints" / f"{name}_{'smoke' if smoke else 'formal'}.pt"
    curve_path = OUT / f"{name}_TRAINING_CURVE.csv"
    if ckpt.exists():
        model = PatchStateModel(1024).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        return model.eval()
    torch.manual_seed(SEED)
    model = PatchStateModel(1024).to(device)
    init_head_from_centroids(model, groups)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    total = sum(len(groups[c]["y"]) for c in groups)
    steps_per_epoch = max(1, math.ceil(total / BATCH_SIZE))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 2)
    curves = []
    selection_rows = []
    for epoch in range(1, EPOCHS + 1):
        losses, hards = [], []
        for step in range(steps_per_epoch):
            xb, yb, origins = balanced_batch(groups, rng, BATCH_SIZE)
            x = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            logits = model.logits(x)
            task_loss, bce, dl = balanced_loss(logits, y)
            extra = torch.zeros((), dtype=torch.float32, device=device)
            if mode == "hard":
                neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
                k = max(1, int(math.ceil(float(neg_idx.numel()) * TOP_NEG_FRAC)))
                selected = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
                extra = F.softplus(logits[selected]).mean()
                sel_np = selected.detach().cpu().numpy()
                selection_rows.append({
                    "variant": name,
                    "epoch": epoch,
                    "step": step + 1,
                    "selected": int(len(sel_np)),
                    "normal_image_ratio": float(np.mean(origins[sel_np] == "normal_image")),
                    "anomaly_mask_out_ratio": float(np.mean(origins[sel_np] == "anomaly_mask_out")),
                    "score_mean": float(logits.detach()[selected].mean().cpu()),
                })
            loss = task_loss + extra
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            hards.append(float(extra.detach().cpu()))
        row = {"variant": name, "epoch": epoch, "loss": float(np.mean(losses)), "hard_loss": float(np.mean(hards))}
        curves.append(row)
        print(f"[RADIO] train {name} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} hard={row['hard_loss']:.4f}", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    write_csv(curve_path, curves)
    write_csv(OUT / f"{name}_HARD_SELECTION.csv", selection_rows)
    return model.eval()


class RawMetaDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_name, mode="test"):
        self.root = Path(root)
        self.dataset_name = dataset_name
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
            if p.exists() and p.is_file():
                return p
        if self.dataset_name == "BTAD" and int(data.get("anomaly", 0)) == 1:
            img_rel = Path(data["img_path"])
            cand_dir = self.root / img_rel.parts[0] / "ground_truth" / "ko"
            for ext in [img_rel.suffix, ".png", ".bmp", ".jpg"]:
                cand = cand_dir / f"{img_rel.stem}{ext}"
                if cand.exists():
                    return cand
        return None

    def __getitem__(self, idx):
        d = self.data_all[idx]
        img_path = self.root / d["img_path"]
        img = Image.open(img_path).convert("RGB")
        anomaly = int(d["anomaly"])
        mask_file = self._resolve_mask(d)
        if anomaly == 1 and mask_file is not None:
            mask = Image.open(mask_file).convert("L")
            mask_available = True
        else:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
            mask_available = anomaly == 0
        mask_np = (np.asarray(mask.resize((IMAGE_SIZE_CLIP_PROTOCOL, IMAGE_SIZE_CLIP_PROTOCOL), Image.NEAREST)) > 0).astype(np.uint8)
        return {
            "img_path": str(img_path),
            "mask": mask_np,
            "cls_name": str(d["cls_name"]),
            "anomaly": anomaly,
            "mask_available": mask_available,
        }


def make_targets():
    return {
        "MVTec": RawMetaDataset(ROOT / "data" / "mvtec", "mvtec"),
        "BTAD": RawMetaDataset(DATA_ROOT / "btad" / "BTech_Dataset_transformed", "BTAD"),
        "ClinicDB": RawMetaDataset(DATA_ROOT / "CVC-ClinicDB", "ClinicDB"),
    }


def top1_score(grid):
    flat = np.asarray(grid, dtype=np.float32).reshape(-1)
    k = max(1, int(flat.size * 0.01))
    return float(np.sort(flat)[-k:].mean())


def upsample_grid(grid):
    x = torch.tensor(grid, dtype=torch.float32).view(1, 1, RADIO_SIDE, RADIO_SIDE)
    full = F.interpolate(x, size=(IMAGE_SIZE_CLIP_PROTOCOL, IMAGE_SIZE_CLIP_PROTOCOL), mode="bilinear", align_corners=False).squeeze().numpy()
    return gaussian_filter(full.astype(np.float32), sigma=SIGMA).astype(np.float32)


def mask_to_radio_grid(mask518):
    mask = Image.fromarray((np.asarray(mask518) > 0).astype(np.uint8) * 255, mode="L")
    small = np.asarray(mask.resize((RADIO_SIDE, RADIO_SIDE), Image.BILINEAR), dtype=np.float32) / 255.0
    return (small.reshape(-1) > 0.5).astype(np.uint8)


@torch.no_grad()
def score_dataset(target_name, dataset, model, extractor, device):
    by_cat = defaultdict(lambda: {"iy": [], "is": [], "py": [], "ps": [], "masks": [], "maps": []})
    for start in range(0, len(dataset), extractor.batch_size):
        if start % max(1, extractor.batch_size * 25) == 0:
            print(f"[RADIO] eval {target_name} {start}/{len(dataset)}", flush=True)
        items = [dataset[i] for i in range(start, min(len(dataset), start + extractor.batch_size))]
        feats = extractor.extract_paths([it["img_path"] for it in items])
        for it, patch in zip(items, feats):
            x = torch.tensor(patch, dtype=torch.float32, device=device)
            logits = model.logits(x).detach().cpu().numpy().reshape(RADIO_SIDE, RADIO_SIDE)
            score = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
            cat = it["cls_name"]
            by_cat[cat]["iy"].append(int(it["anomaly"]))
            by_cat[cat]["is"].append(top1_score(score))
            if it["mask_available"]:
                mask = mask_to_radio_grid(it["mask"])
                pix = score.reshape(-1)
                by_cat[cat]["py"].append(mask)
                by_cat[cat]["ps"].append(pix)
    per_rows = []
    mean_lists = defaultdict(list)
    for cat, vals in sorted(by_cat.items()):
        if vals["py"]:
            py = np.concatenate(vals["py"])
            ps = np.concatenate(vals["ps"])
            p_auc = safe_auc(py, ps)
            p_ap = safe_ap(py, ps)
            pro = float("nan")
        else:
            p_auc = p_ap = pro = float("nan")
        row = {
            "target": target_name,
            "category": cat,
            "I-AUROC": safe_auc(vals["iy"], vals["is"]),
            "I-AP": safe_ap(vals["iy"], vals["is"]),
            "P-AUROC": p_auc,
            "P-AP": p_ap,
            "PRO": pro,
            "mask_images": len(vals["py"]),
        }
        row["G_pixel"] = gp(row) if np.isfinite(p_auc) and np.isfinite(p_ap) else float("nan")
        per_rows.append(row)
        for k in ["I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"]:
            if np.isfinite(row[k]):
                mean_lists[k].append(row[k])
    mean = {
        "target": target_name,
        "I-AUROC": float(np.mean(mean_lists["I-AUROC"])) if mean_lists["I-AUROC"] else float("nan"),
        "I-AP": float(np.mean(mean_lists["I-AP"])) if mean_lists["I-AP"] else float("nan"),
        "P-AUROC": float(np.mean(mean_lists["P-AUROC"])) if mean_lists["P-AUROC"] else float("nan"),
        "P-AP": float(np.mean(mean_lists["P-AP"])) if mean_lists["P-AP"] else float("nan"),
        "PRO": float(np.mean(mean_lists["PRO"])) if mean_lists["PRO"] else float("nan"),
    }
    mean["G_pixel"] = gp(mean)
    return mean, per_rows


def sota_gap_rows(metrics):
    rows = []
    for r in metrics:
        if r["variant"] not in ("B0_CURRENT_CLIP_HARDNORMAL", "E1_RADIO_HARDNORMAL"):
            continue
        for method, by_target in SOTA.items():
            ref = by_target.get(r["target"])
            if not ref:
                continue
            rows.append({
                "variant": r["variant"],
                "external_method": method,
                "target": r["target"],
                "comparability": "REFERENCE_ONLY_DIFFERENT_BACKBONE" if r["variant"] == "B0_CURRENT_CLIP_HARDNORMAL" else "REFERENCE_ONLY_RADIO_BACKBONE",
                "method_P-AUROC": r["P-AUROC"],
                "method_P-AP": r["P-AP"],
                "method_G": r["G_pixel"],
                "external_P-AUROC": ref["P-AUROC"],
                "external_P-AP": ref["P-AP"],
                "external_G": ref["G_pixel"],
                "Delta_P_AUROC": float(r["P-AUROC"]) - ref["P-AUROC"],
                "Delta_P_AP": float(r["P-AP"]) - ref["P-AP"],
                "Delta_G": float(r["G_pixel"]) - ref["G_pixel"],
            })
    return rows


def decide(metrics):
    radio_hn = {r["target"]: r for r in metrics if r["variant"] == "E1_RADIO_HARDNORMAL"}
    radio_sa = {r["target"]: r for r in metrics if r["variant"] == "E1_RADIO_STATEALIGNED"}
    if len(radio_hn) < 3:
        return "RADIO_FORMAL_INCOMPLETE"
    gains_vs_clip = {t: gp(radio_hn[t]) - B0_CURRENT[t]["G_pixel"] for t in radio_hn}
    mean_gain = float(np.mean(list(gains_vs_clip.values())))
    hn_beats_sa = sum(gp(radio_hn[t]) > gp(radio_sa[t]) for t in radio_hn if t in radio_sa)
    mvtec_up2 = gains_vs_clip.get("MVTec", -999) >= 2.0
    clinic_up2 = gains_vs_clip.get("ClinicDB", -999) >= 2.0
    btad_only = gains_vs_clip.get("BTAD", -999) > 1.5 and gains_vs_clip.get("MVTec", 0) <= 0.5 and gains_vs_clip.get("ClinicDB", 0) <= 0.5
    if (mvtec_up2 or clinic_up2) and mean_gain >= 1.5 and hn_beats_sa >= 2:
        return "SPATIAL_BACKBONE_BOTTLENECK_CONFIRMED"
    if mean_gain >= 1.5 and hn_beats_sa < 2:
        return "BACKBONE_STRONG_BUT_HARDNORMAL_NOT_PORTABLE"
    if btad_only:
        return "BACKBONE_DOMAIN_DEPENDENT"
    return "REPRESENTATION_BACKBONE_NOT_PRIMARY_BOTTLENECK"


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    availability = [
        {
            "candidate": "B0 OpenCLIP ViT-L/14-336 compatible",
            "status": "REUSED_EXISTING_FORMAL_BASELINE",
            "input_resolution": 518,
            "patch_grid": "37x37",
            "feature_dim": 1024,
        },
        {
            "candidate": "E1 RADIO radio_v2.5-l",
            "status": "AVAILABLE_RUN",
            "source": "torch.hub NVlabs/RADIO radio_model version radio_v2.5-l; weight cached from nvidia/RADIO radio-v2.5-l_half.pth.tar",
            "input_resolution": 512,
            "patch_grid": "32x32",
            "feature_dim": 1024,
        },
        {
            "candidate": "E2 TIPS",
            "status": "UNAVAILABLE_NOT_RUN",
            "reason": "No local official TIPS loader/weights were found; no substitute backbone added by protocol.",
        },
    ]
    write_csv(OUT / "BACKBONE_AVAILABILITY.csv", availability)
    metrics = []
    per_rows = []
    for target, vals in B0_CURRENT.items():
        metrics.append({
            "backbone": "OpenCLIP_ViT-L14_336_compatible",
            "variant": "B0_CURRENT_CLIP_HARDNORMAL",
            "target": target,
            "P-AUROC": vals["P-AUROC"],
            "P-AP": vals["P-AP"],
            "PRO": float("nan"),
            "G_pixel": vals["G_pixel"],
            "run_status": "reused_existing_formal",
        })
    extractor = RadioFeatureExtractor(device, batch_size=args.batch_size)
    groups = collect_source_pool(extractor, smoke=args.smoke)
    state_model = train_variant("E1_RADIO_STATEALIGNED", "none", groups, device, smoke=args.smoke)
    hard_model = train_variant("E1_RADIO_HARDNORMAL", "hard", groups, device, smoke=args.smoke)
    targets = make_targets()
    if args.smoke:
        for name, ds in list(targets.items()):
            ds.data_all = ds.data_all[: min(len(ds.data_all), 12)]
    for variant, model in [("E1_RADIO_STATEALIGNED", state_model), ("E1_RADIO_HARDNORMAL", hard_model)]:
        for target, ds in targets.items():
            mean, per = score_dataset(target, ds, model, extractor, device)
            metrics.append({
                "backbone": "RADIO_radio_v2.5-l",
                "variant": variant,
                **mean,
                "run_status": "formal" if not args.smoke else "smoke",
            })
            for r in per:
                per_rows.append({"backbone": "RADIO_radio_v2.5-l", "variant": variant, **r})
    for r in metrics:
        if r["variant"] == "E1_RADIO_HARDNORMAL" and r["target"] in B0_CURRENT:
            r["Delta_G_vs_current_CLIP_HardNormal"] = gp(r) - B0_CURRENT[r["target"]]["G_pixel"]
            sa = next((x for x in metrics if x["variant"] == "E1_RADIO_STATEALIGNED" and x["target"] == r["target"]), None)
            r["HardNormal_minus_StateAligned_G"] = gp(r) - gp(sa) if sa else float("nan")
    write_csv(OUT / "RAW_METRICS.csv", metrics)
    write_csv(OUT / "PER_CATEGORY.csv", per_rows)
    write_csv(OUT / "SOTA_GAP.csv", sota_gap_rows(metrics))
    outcome = "SMOKE_COMPLETE" if args.smoke else decide(metrics)
    write_json(OUT / "OUTCOME.json", {
        "phase": "Frozen Spatial Backbone Compatibility Gate",
        "outcome": outcome,
        "runtime_sec": time.time() - start,
        "strict_source_only": True,
        "target_statistics_used": False,
        "tips_run": False,
        "radio_input_resolution": RADIO_SIZE,
        "radio_patch_grid": f"{RADIO_SIDE}x{RADIO_SIDE}",
    })
    report = [
        "# Frozen Spatial Backbone Compatibility Gate",
        "",
        f"Outcome: **{outcome}**",
        "",
        "## Metrics",
        "",
    ]
    for r in metrics:
        report.append(f"- {r['variant']} -> {r['target']}: P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, G {float(r['G_pixel']):.4f}.")
    report.extend([
        "",
        "## Backbone Audit",
        "",
        "- B0 OpenCLIP ViT-L/14-336 compatible path was reused from the existing formal HardNormal baseline.",
        "- RADIO `radio_v2.5-l` is available through the official NVlabs/RADIO torch.hub path. Its native 518-nearest supported square input is 512, producing a 32x32 patch grid with 1024-d features.",
        "- TIPS was not run because no local official public weight/loader was available in the project; no substitute backbone was introduced.",
        "",
        "## External SOTA Position",
        "",
        "- MoECLIP and VisualAD rows are included in `SOTA_GAP.csv` as reference-only comparisons because RADIO changes the backbone from the reported OpenCLIP ViT-L/14-336 setup.",
        "- The current gate should therefore be read as a representation-carrier compatibility test, not a claim of direct SOTA protocol matching.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch-size", type=int, default=2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
