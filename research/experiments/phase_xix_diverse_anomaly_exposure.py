import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, gaussian_filter
from torch.utils.data import Dataset as TorchDataset
from torchvision.utils import save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import load_dataset, to_device_cache
from experiments.phase_xiv_b_source_smoke import (
    BranchModule,
    evaluate_variant,
    losses,
    mv,
    representation_diag,
    write_csv,
)
from phase_x.gradient_smoke_test import hashes_equal, parameter_hashes
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xix_diverse_anomaly_exposure/batch_a")
CKPT = OUT / "checkpoints"
SHEETS = OUT / "synthetic_audit_sheets"
XIV_SOURCE = Path("results/phase_xiv_dsls/source_smoke")
MEAN = torch.tensor(OPENAI_DATASET_MEAN).view(3, 1, 1)
STD = torch.tensor(OPENAI_DATASET_STD).view(3, 1, 1)
VARIANTS = ["XIX1_REAL_ONLY", "XIX2_SYNTH_ONLY", "XIX3_REAL_PLUS_SYNTH"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def meta_cls(dataset, idx):
    return dataset.data_all[idx]["cls_name"]


def meta_anomaly(dataset, idx):
    return int(dataset.data_all[idx]["anomaly"])


def fast_indices_for_categories(dataset, categories):
    cats = set(categories)
    return [idx for idx, row in enumerate(dataset.data_all) if row["cls_name"] in cats]


def denorm(x):
    return (x.detach().cpu() * STD + MEAN).clamp(0.0, 1.0)


def norm01(x):
    return ((x.detach().cpu().clamp(0.0, 1.0) - MEAN) / STD).float()


def irregular_mask(seed, size=518):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(32):
        target = float(np.exp(rng.uniform(np.log(0.005), np.log(0.15))))
        mask = np.zeros((size, size), dtype=bool)
        blobs = int(rng.integers(1, 5))
        for _b in range(blobs):
            cy = float(rng.uniform(0.12 * size, 0.88 * size))
            cx = float(rng.uniform(0.12 * size, 0.88 * size))
            ry = float(rng.uniform(12, 95))
            rx = float(rng.uniform(12, 95))
            angle = float(rng.uniform(0, math.pi))
            ca, sa = math.cos(angle), math.sin(angle)
            x = xx - cx
            y = yy - cy
            xr = ca * x + sa * y
            yr = -sa * x + ca * y
            blob = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
            mask |= blob
        area = float(mask.mean())
        touches = int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())
        if 0.005 <= area <= 0.15 and touches <= 2:
            return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        if area > 0:
            keep = target / max(area, 1e-6)
            if keep < 1.0:
                mask &= rng.random((size, size)) < keep
            area = float(mask.mean())
            if 0.005 <= area <= 0.15:
                return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    side = int(math.sqrt(0.02 * size * size))
    y0 = int(rng.integers(size // 8, size - size // 8 - side))
    x0 = int(rng.integers(size // 8, size - size // 8 - side))
    mask = np.zeros((size, size), dtype=np.float32)
    mask[y0:y0 + side, x0:x0 + side] = 1.0
    return torch.from_numpy(mask).unsqueeze(0)


def local_contrast(x, mask):
    vals = x[:, mask.squeeze(0) > 0.5]
    return float(vals.std().item()) if vals.numel() else 0.0


def boundary_gradient(x, mask):
    m = mask.squeeze(0).numpy() > 0.5
    boundary = binary_dilation(m, iterations=2) ^ m
    gray = x.mean(dim=0).numpy()
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag[boundary].mean()) if boundary.any() else 0.0


def synthesize(base_dataset, normal_indices, spec):
    host = base_dataset[spec["host_idx"]]
    donor = base_dataset[spec["donor_idx"]]
    host_img = denorm(host["img"])
    donor_img = denorm(donor["img"])
    mask = irregular_mask(spec["mask_seed"])
    rng = np.random.default_rng(spec["seed"])
    before_contrast = local_contrast(host_img, mask)
    if spec["generator"] == "G1":
        donor_local = donor_img.clone()
        hvals = host_img[:, mask.squeeze(0) > 0.5]
        dvals = donor_local[:, mask.squeeze(0) > 0.5]
        if hvals.numel() and dvals.numel():
            hmean, hstd = hvals.mean(dim=1, keepdim=True), hvals.std(dim=1, keepdim=True).clamp_min(1e-4)
            dmean, dstd = dvals.mean(dim=1, keepdim=True), dvals.std(dim=1, keepdim=True).clamp_min(1e-4)
            flat = donor_local.reshape(3, -1)
            flat = (flat - dmean) / dstd * hstd + hmean
            donor_local = flat.reshape_as(donor_local).clamp(0.0, 1.0)
        alpha = float(rng.uniform(0.55, 0.90))
        out = host_img * (1.0 - mask) + (alpha * donor_local + (1.0 - alpha) * host_img) * mask
    elif spec["generator"] == "G2":
        shift_y = int(rng.integers(-18, 19))
        shift_x = int(rng.integers(-18, 19))
        warped = torch.roll(host_img, shifts=(shift_y, shift_x), dims=(1, 2))
        if rng.random() < 0.5:
            warped = torch.rot90(warped, k=1, dims=(1, 2))
        out = host_img * (1.0 - mask) + warped * mask
    else:
        out = host_img.clone()
        choice = int(rng.integers(0, 6))
        region = host_img
        if choice == 0:
            region = region.pow(float(rng.uniform(0.55, 1.65)))
        elif choice == 1:
            region = ((region - 0.5) * float(rng.uniform(0.45, 1.85)) + 0.5)
        elif choice == 2:
            region = region + float(rng.uniform(-0.30, 0.30))
        elif choice == 3:
            shift = torch.tensor(rng.uniform(-0.25, 0.25, size=(3,)), dtype=torch.float32).view(3, 1, 1)
            region = region + shift
        elif choice == 4:
            region = torch.from_numpy(gaussian_filter(region.numpy(), sigma=(0.0, 1.2, 1.2))).float()
        else:
            region = region + torch.randn_like(region) * float(rng.uniform(0.03, 0.12))
        out = host_img * (1.0 - mask) + region.clamp(0.0, 1.0) * mask
    unclamped = out
    out = out.clamp(0.0, 1.0)
    transformed = mask > 0.5
    clip_ratio = float(((unclamped <= 0.0) | (unclamped >= 1.0))[:, transformed.squeeze(0)].float().mean().item()) if transformed.any() else 1.0
    item = {
        "img": norm01(out),
        "img_mask": mask.float(),
        "cls_name": host["cls_name"],
        "anomaly": 1,
        "img_path": f"synthetic/{spec['direction']}/{spec['generator']}/{spec['id']}",
        "cls_id": host["cls_id"],
        "synthetic_generator": spec["generator"],
        "synthetic_id": spec["id"],
    }
    qc = {
        "direction": spec["direction"],
        "fold": 0,
        "synthetic_id": spec["id"],
        "generator": spec["generator"],
        "host_idx": spec["host_idx"],
        "donor_idx": spec["donor_idx"],
        "host_category": host["cls_name"],
        "donor_category": donor["cls_name"],
        "mask_area": float(mask.mean().item()),
        "pixel_clipping_ratio": clip_ratio,
        "mean_local_contrast_change": local_contrast(out, mask) - before_contrast,
        "boundary_gradient": boundary_gradient(out, mask),
        "invalid": bool((not torch.isfinite(item["img"]).all()) or clip_ratio > 0.05),
    }
    return item, qc, out


class CombinedDataset(TorchDataset):
    def __init__(self, base_dataset, synth_specs=None):
        self.base_dataset = base_dataset
        self.synth_specs = synth_specs or []

    def __len__(self):
        return len(self.base_dataset) + len(self.synth_specs)

    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            return self.base_dataset[idx]
        item, _, _ = synthesize(self.base_dataset, None, self.synth_specs[idx - len(self.base_dataset)])
        return item


def make_synth_specs(direction, base_dataset, normal_indices, count, seed=0):
    cats = {}
    for idx in normal_indices:
        cats.setdefault(meta_cls(base_dataset, idx), []).append(idx)
    ordered = sorted(normal_indices)
    specs = []
    sid = 0
    for i in range(count * 2):
        gen = ["G1", "G2", "G3"][i % 3]
        host_idx = ordered[i % len(ordered)]
        host_cat = meta_cls(base_dataset, host_idx)
        donor_pool = [v for c, xs in cats.items() if c != host_cat for v in xs] or ordered
        donor_idx = donor_pool[(i * 17 + seed) % len(donor_pool)]
        spec = {
            "id": sid,
            "direction": direction,
            "generator": gen,
            "host_idx": host_idx,
            "donor_idx": donor_idx,
            "seed": 100000 + seed * 1000 + i,
            "mask_seed": 200000 + seed * 1000 + i,
        }
        specs.append(spec)
        sid += 1
        if len(specs) >= count:
            break
    if len(specs) < count:
        raise RuntimeError(f"Only generated {len(specs)} valid synthetic examples for {direction}; need {count}")
    return specs


def build_cache(model, dataset, indices, dataset_name, device, adapter):
    was = adapter.training
    adapter.eval()
    with torch.no_grad():
        image_cache, patch_cache = build_memory_from_indices(model, dataset, indices, dataset_name, device, adapter=adapter)
    if was:
        adapter.train()
    return to_device_cache(image_cache, device), to_device_cache(patch_cache, device)


def train_xix_variant(direction, variant, model, dataset, train_indices, cache_indices, image_proj, patch_proj, dataset_name, device):
    ckpt = CKPT / f"{direction}_fold0_{variant}_seed0.pt"
    adapter = BranchModule("XIV1_SharedAdapter").to(device).train()
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        adapter.load_state_dict(payload["adapter"])
        return adapter.eval(), payload.get("training_rows", []), payload.get("generator_loss_rows", []), {"reused": True, "checkpoint": str(ckpt), "steps": payload.get("steps")}
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    before = parameter_hashes(model)
    training_rows = []
    gen_loss_rows = []
    step = 0
    for epoch in range(3):
        image_cache, patch_cache = build_cache(model, dataset, cache_indices, dataset_name, device, adapter)
        order = list(train_indices)
        rng = random.Random(1000 + epoch)
        rng.shuffle(order)
        for idx in tqdm(order, desc=f"{direction} {variant} epoch {epoch+1}/3"):
            step += 1
            item = dataset[idx]
            opt.zero_grad(set_to_none=True)
            l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="both")
            loss = l["image"] + l["pixel"] + l["pres_img"] + l["pres_patch"]
            loss.backward()
            opt.step()
            generator = item.get("synthetic_generator", "REAL_ANOMALY" if int(item["anomaly"]) else "NORMAL")
            row = {
                "direction": direction,
                "fold": 0,
                "variant": variant,
                "epoch": epoch + 1,
                "step": step,
                "sample_idx": idx,
                "category": item["cls_name"],
                "label": int(item["anomaly"]),
                "generator": generator,
                "loss": float(loss.detach().cpu()),
                "image_loss": float(l["image"].detach().cpu()),
                "pixel_loss": float(l["pixel"].detach().cpu()),
                "pres_img": float(l["pres_img"].detach().cpu()),
                "pres_patch": float(l["pres_patch"].detach().cpu()),
            }
            training_rows.append(row)
            if variant == "XIX3_REAL_PLUS_SYNTH" and generator in {"REAL_ANOMALY", "G1", "G2", "G3"}:
                gen_loss_rows.append(row.copy())
    torch.save({
        "adapter": adapter.state_dict(),
        "variant": variant,
        "direction": direction,
        "seed": 0,
        "steps": step,
        "training_rows": training_rows,
        "generator_loss_rows": gen_loss_rows,
        "official_unchanged": hashes_equal(before, parameter_hashes(model)),
    }, ckpt)
    return adapter.eval(), training_rows, gen_loss_rows, {"reused": False, "checkpoint": str(ckpt), "steps": step, "official_unchanged": hashes_equal(before, parameter_hashes(model))}


def audit_synthetics(direction, base_dataset, specs):
    rows = []
    by_gen = {"G1": [], "G2": [], "G3": []}
    for spec in specs:
        item, qc, image01 = synthesize(base_dataset, None, spec)
        rows.append(qc)
        if len(by_gen[spec["generator"]]) < 16:
            mask_rgb = item["img_mask"].repeat(3, 1, 1)
            by_gen[spec["generator"]].append(torch.cat([denorm(base_dataset[spec["host_idx"]]["img"]), image01, mask_rgb], dim=2))
    SHEETS.mkdir(parents=True, exist_ok=True)
    for gen, imgs in by_gen.items():
        if imgs:
            save_image(torch.stack(imgs), SHEETS / f"{direction}_{gen}_sheet.png", nrow=4)
    return rows


def metric_row(direction, variant, metrics, val_count):
    return {
        "direction": direction,
        "fold": 0,
        "variant": variant,
        "I-AUROC": mv(metrics, "I-AUROC"),
        "I-AP": mv(metrics, "I-AP"),
        "P-AUROC": mv(metrics, "P-AUROC"),
        "PRO": mv(metrics, "PRO"),
        "val_images": val_count,
    }


def grouped_gains(raw_rows):
    rows = []
    for direction in ["D1", "D2"]:
        base = next(r for r in raw_rows if r["direction"] == direction and r["variant"] == "XIX1_REAL_ONLY")
        for r in [x for x in raw_rows if x["direction"] == direction]:
            di = r["I-AUROC"] - base["I-AUROC"]
            dap = r["I-AP"] - base["I-AP"]
            dp = r["P-AUROC"] - base["P-AUROC"]
            dpro = r["PRO"] - base["PRO"]
            gi = 0.5 * (di + dap)
            gp = 0.5 * (dp + dpro)
            rows.append({
                "direction": direction,
                "fold": 0,
                "variant": r["variant"],
                "dI-AUROC": di,
                "dI-AP": dap,
                "dP-AUROC": dp,
                "dPRO": dpro,
                "G_image": gi,
                "G_pixel": gp,
                "G_balanced": 0.5 * (gi + gp),
            })
    return rows


def decide(gains):
    x3 = {r["direction"]: r for r in gains if r["variant"] == "XIX3_REAL_PLUS_SYNTH"}
    def gains_dir(d):
        return x3[d]["G_balanced"] >= 0.15 and (x3[d]["G_image"] >= 0.20 or x3[d]["G_pixel"] >= 0.20) and min(x3[d]["G_image"], x3[d]["G_pixel"]) >= -0.10 and min(x3[d]["dI-AUROC"], x3[d]["dI-AP"], x3[d]["dP-AUROC"], x3[d]["dPRO"]) >= -0.30
    d1, d2 = gains_dir("D1"), gains_dir("D2")
    strong = all(x3[d]["G_balanced"] >= 0.30 and min(x3[d]["dI-AUROC"], x3[d]["dI-AP"], x3[d]["dP-AUROC"], x3[d]["dPRO"]) >= 0 for d in ["D1", "D2"])
    if strong:
        return "XIX_A_STRONG_DUAL_DIRECTION_DIVERSITY_GAIN"
    if d1 and d2:
        return "XIX_E_SOURCE_DIVERSITY_GAIN"
    if d1 and not d2:
        return "XIX_A_D1_ONLY_SYNTHETIC_GAIN"
    if d2 and not d1:
        return "XIX_A_D2_ONLY_SYNTHETIC_GAIN"
    trade = any((x3[d]["G_image"] >= 0.20 and x3[d]["G_pixel"] < -0.10) or (x3[d]["G_pixel"] >= 0.20 and x3[d]["G_image"] < -0.10) for d in ["D1", "D2"])
    if trade:
        return "XIX_A_SYNTHETIC_METRIC_TRADEOFF"
    return "XIX_A_NO_SYNTHETIC_DIVERSITY_GAIN"


def run_direction(direction):
    set_seed(0)
    if direction == "D1":
        dataset_name, data_path, checkpoint = "mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"
    else:
        dataset_name, data_path, checkpoint = "visa", "data/visa", "checkpoints/test_on_mvtec.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
    base_dataset = load_dataset(dataset_name, data_path)
    rot = deterministic_source_folds(dataset_name)["rotations"][0]
    source_indices = fast_indices_for_categories(base_dataset, rot["train_categories"])
    val_indices = fast_indices_for_categories(base_dataset, rot["val_categories"])
    normal_indices = [i for i in source_indices if meta_anomaly(base_dataset, i) == 0]
    real_anom_indices = [i for i in source_indices if meta_anomaly(base_dataset, i) == 1]
    print(f"{direction}: source={len(source_indices)} normal={len(normal_indices)} real_anom={len(real_anom_indices)} val={len(val_indices)}", flush=True)
    synth_specs = make_synth_specs(direction, base_dataset, normal_indices, max(3, len(real_anom_indices)), seed=0)
    print(f"{direction}: synthetic specs={len(synth_specs)}", flush=True)
    audit_rows = audit_synthetics(direction, base_dataset, synth_specs)
    print(f"{direction}: synthetic audit complete", flush=True)
    dataset = CombinedDataset(base_dataset, synth_specs)
    synth_indices = list(range(len(base_dataset), len(base_dataset) + len(synth_specs)))
    plans = {
        "XIX1_REAL_ONLY": {"train": source_indices, "cache": source_indices},
        "XIX2_SYNTH_ONLY": {"train": normal_indices + synth_indices, "cache": normal_indices + synth_indices},
        "XIX3_REAL_PLUS_SYNTH": {"train": source_indices + synth_indices, "cache": source_indices + synth_indices},
    }
    raw, percat, rep, train_rows, gen_loss_rows, diags = [], [], [], [], [], []
    for variant in VARIANTS:
        print(f"{direction}: train/eval {variant}", flush=True)
        if direction == "D1" and variant == "XIX1_REAL_ONLY" and (XIV_SOURCE / "XIV_B_RAW_METRICS.csv").exists():
            reuse_raw, reuse_percat, reuse_rep = load_d1_xix1_reuse()
            raw.extend(reuse_raw)
            percat.extend(reuse_percat)
            rep.extend(reuse_rep)
            diags.append({"direction": direction, "variant": variant, "reused": True, "source": str(XIV_SOURCE)})
            write_csv(OUT / "XIX_A_RAW_METRICS.csv", existing_csv(OUT / "XIX_A_RAW_METRICS.csv", exclude_direction=direction) + raw)
            write_csv(OUT / "XIX_A_PER_CATEGORY.csv", existing_csv(OUT / "XIX_A_PER_CATEGORY.csv", exclude_direction=direction) + percat)
            write_csv(OUT / "XIX_A_REPRESENTATION.csv", existing_csv(OUT / "XIX_A_REPRESENTATION.csv", exclude_direction=direction) + rep)
            print(f"{direction}: reused XIX1 from XIV1", flush=True)
            continue
        adapter, tr, gl, diag = train_xix_variant(direction, variant, model, dataset, plans[variant]["train"], plans[variant]["cache"], image_proj, patch_proj, dataset_name, device)
        train_rows.extend(tr)
        gen_loss_rows.extend(gl)
        diag.update({"direction": direction, "variant": variant})
        diags.append(diag)
        metrics = evaluate_variant(variant, adapter, model, prompt_learner, image_proj, patch_proj, prompt_proj, dataset, plans[variant]["cache"], val_indices, dataset_name, device)
        raw.append(metric_row(direction, variant, metrics, len(val_indices)))
        for cat, vals in metrics["per_class"].items():
            percat.append({
                "direction": direction,
                "fold": 0,
                "variant": variant,
                "category": cat,
                "I-AUROC": vals.get("I-AUROC", float("nan")) * 100.0,
                "P-AUROC": vals.get("P-AUROC", float("nan")) * 100.0,
                "PRO": vals.get("PRO", float("nan")) * 100.0,
            })
        rep.extend(representation_diag(variant, adapter, model, dataset, val_indices, device))
        write_csv(OUT / "XIX_A_RAW_METRICS.csv", existing_csv(OUT / "XIX_A_RAW_METRICS.csv", exclude_direction=direction) + raw)
        write_csv(OUT / "XIX_A_PER_CATEGORY.csv", existing_csv(OUT / "XIX_A_PER_CATEGORY.csv", exclude_direction=direction) + percat)
        write_csv(OUT / "XIX_A_REPRESENTATION.csv", existing_csv(OUT / "XIX_A_REPRESENTATION.csv", exclude_direction=direction) + rep)
    return audit_rows, raw, percat, rep, train_rows, gen_loss_rows, diags


def existing_csv(path, exclude_direction=None):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if exclude_direction is not None:
        rows = [r for r in rows if r.get("direction") != exclude_direction]
    return rows


def load_d1_xix1_reuse():
    raw_rows = []
    with open(XIV_SOURCE / "XIV_B_RAW_METRICS.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["variant"] == "XIV1_SharedAdapter":
                raw_rows.append({
                    "direction": "D1",
                    "fold": 0,
                    "variant": "XIX1_REAL_ONLY",
                    "I-AUROC": float(row["I-AUROC"]),
                    "I-AP": float(row["I-AP"]),
                    "P-AUROC": float(row["P-AUROC"]),
                    "PRO": float(row["PRO"]),
                    "val_images": int(row["val_images"]),
                    "reused_from": str(XIV_SOURCE / "XIV_B_RAW_METRICS.csv"),
                })
    percat = []
    with open(XIV_SOURCE / "XIV_B_PER_CATEGORY.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["variant"] == "XIV1_SharedAdapter":
                row = dict(row)
                row["direction"] = "D1"
                row["fold"] = 0
                row["variant"] = "XIX1_REAL_ONLY"
                row["reused_from"] = str(XIV_SOURCE / "XIV_B_PER_CATEGORY.csv")
                percat.append(row)
    rep = []
    with open(XIV_SOURCE / "XIV_B_REPRESENTATION.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["variant"] == "XIV1_SharedAdapter":
                row = dict(row)
                row["direction"] = "D1"
                row["fold"] = 0
                row["variant"] = "XIX1_REAL_ONLY"
                row["reused_from"] = str(XIV_SOURCE / "XIV_B_REPRESENTATION.csv")
                rep.append(row)
    return raw_rows, percat, rep


def summarize_generator_losses(rows):
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["direction"], r["variant"], r["generator"]), []).append(r)
    for (direction, variant, gen), xs in groups.items():
        out.append({
            "direction": direction,
            "fold": 0,
            "variant": variant,
            "generator": gen,
            "n": len(xs),
            "mean_loss": float(np.mean([float(x["loss"]) for x in xs])),
            "mean_image_loss": float(np.mean([float(x["image_loss"]) for x in xs])),
            "mean_pixel_loss": float(np.mean([float(x["pixel_loss"]) for x in xs])),
        })
    return out


def add_per_category_wtl(percat):
    rows = list(percat)
    for direction in ["D1", "D2"]:
        cats = sorted({r["category"] for r in percat if r["direction"] == direction})
        for metric in ["I-AUROC", "P-AUROC", "PRO"]:
            deltas = []
            for cat in cats:
                base = next((r for r in percat if r["direction"] == direction and r["variant"] == "XIX1_REAL_ONLY" and r["category"] == cat), None)
                x3 = next((r for r in percat if r["direction"] == direction and r["variant"] == "XIX3_REAL_PLUS_SYNTH" and r["category"] == cat), None)
                if base and x3:
                    delta = float(x3[metric]) - float(base[metric])
                    deltas.append(delta)
                    rows.append({"direction": direction, "fold": 0, "variant": "XIX3_MINUS_XIX1", "category": cat, "metric": metric, "delta": delta, "wtl": "W" if delta > 0.05 else "L" if delta < -0.05 else "T"})
            if deltas:
                rows.append({"direction": direction, "fold": 0, "variant": "XIX3_MINUS_XIX1_SUMMARY", "category": "ALL", "metric": metric, "median": float(np.median(deltas)), "worst": float(np.min(deltas)), "best": float(np.max(deltas))})
    return rows


def write_report(raw, gains, outcome):
    lines = [
        "# Phase XIX-A Diverse Anomaly Exposure Batch A",
        "",
        f"Outcome: **{outcome}**",
        "",
        "Scope: D1 MVTec fold0 and D2 VisA fold0, seed0, 3 epochs. Target data was not accessed. XIX4, generator ablations, extra folds, all-source training, and target evaluation were not started.",
        "",
        "## Raw Metrics",
        "",
        "| Direction | Variant | I-AUROC | I-AP | P-AUROC | PRO |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['direction']} | {r['variant']} | {float(r['I-AUROC']):.4f} | {float(r['I-AP']):.4f} | {float(r['P-AUROC']):.4f} | {float(r['PRO']):.4f} |")
    lines.extend(["", "## Gains vs XIX1", "", "| Direction | Variant | dI | dAP | dP | dPRO | G_image | G_pixel | G_balanced |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in gains:
        lines.append(f"| {r['direction']} | {r['variant']} | {float(r['dI-AUROC']):.4f} | {float(r['dI-AP']):.4f} | {float(r['dP-AUROC']):.4f} | {float(r['dPRO']):.4f} | {float(r['G_image']):.4f} | {float(r['G_pixel']):.4f} | {float(r['G_balanced']):.4f} |")
    lines.extend(["", "Mandatory stop: Batch B was not started."])
    (OUT / "XIX_A_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    all_audit, all_raw, all_percat, all_rep, all_train, all_gen, all_diag = [], [], [], [], [], [], []
    try:
        for direction in ["D1", "D2"]:
            audit, raw, percat, rep, train_rows, gen_rows, diags = run_direction(direction)
            all_audit.extend(audit)
            all_raw.extend(raw)
            all_percat.extend(percat)
            all_rep.extend(rep)
            all_train.extend(train_rows)
            all_gen.extend(gen_rows)
            all_diag.extend(diags)
            write_csv(OUT / "SYNTHETIC_GENERATOR_AUDIT.csv", all_audit)
            write_csv(OUT / "XIX_A_TRAINING_ROWS.csv", all_train)
            write_csv(OUT / "XIX_A_GENERATOR_LOSSES.csv", summarize_generator_losses(all_gen))
            write_csv(OUT / "XIX_A_TRAINING_DIAGNOSTICS.csv", all_diag)
        gains = grouped_gains(all_raw)
        outcome = decide(gains)
    except Exception as exc:
        outcome = "XIX_J_IMPLEMENTATION_FAILURE"
        (OUT / "OUTCOME.json").write_text(json.dumps({"phase": "XIX-A", "outcome": outcome, "error": repr(exc), "target_accessed": False, "next": "STOP"}, indent=2), encoding="utf-8")
        raise
    percat_full = add_per_category_wtl(all_percat)
    write_csv(OUT / "SYNTHETIC_GENERATOR_AUDIT.csv", all_audit)
    write_csv(OUT / "XIX_A_RAW_METRICS.csv", all_raw)
    write_csv(OUT / "XIX_A_GROUPED_GAINS.csv", gains)
    write_csv(OUT / "XIX_A_PER_CATEGORY.csv", percat_full)
    write_csv(OUT / "XIX_A_REPRESENTATION.csv", all_rep)
    write_csv(OUT / "XIX_A_GENERATOR_LOSSES.csv", summarize_generator_losses(all_gen))
    write_report(all_raw, gains, outcome)
    payload = {
        "phase": "XIX-A",
        "outcome": outcome,
        "target_accessed": False,
        "trained": True,
        "seed": 0,
        "epochs": 3,
        "directions": ["D1", "D2"],
        "runtime_sec": time.time() - started,
        "next": "STOP. Do not run Batch B unless explicitly requested after reviewing Batch A.",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write("\n\n## Phase XIX-A\n")
        f.write("- Scope: Batch A only; D1 fold0 and D2 fold0; seed0; epochs=3; target access false.\n")
        f.write("- Generators: G1 cross-image local texture transplant, G2 local structural deformation, G3 local appearance corruption; irregular masks 0.5%-15% area; balanced synthetic pool.\n")
        f.write("- Variants: XIX1_REAL_ONLY, XIX2_SYNTH_ONLY, XIX3_REAL_PLUS_SYNTH. XIX4, generator ablations, Batch B, all-source, and target were skipped.\n")
        f.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XIX_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
