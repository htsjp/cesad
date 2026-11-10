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
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import ExternalDataset, load_full_visa_source
from experiments.phase_xlv_medical_pixel_transfer import MedicalDataset
from experiments.phase_xxxiv_insplad_cross_asset import extract_dino_one
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, IMAGE_SIZE, SIDE, norm_np, patch_labels, upsample_map


OUT = Path("results/phase_xlvii_dino_source_memory")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
XLV_A = Path("results/phase_xlv_cross_domain_transfer")
XLV_B = Path("results/phase_xlv_medical_pixel_transfer")
DINO_CACHE = OUT / "feature_cache"
MEMORY_BUDGET_TOTAL = 20_000
MEMORY_BUDGET_PER_CLASS = MEMORY_BUDGET_TOTAL // 2
SEED = 0
SIGMA = 4
CHECKPOINT = Path("checkpoints/test_on_mvtec.pth")


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


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


def dino_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def fps_indices(features, budget):
    x = norm_np(features)
    n = x.shape[0]
    if n <= budget:
        return np.arange(n, dtype=np.int64)
    centroid = norm_np(x.mean(axis=0, keepdims=True))[0]
    first = int(np.argmin(x @ centroid))
    selected = [first]
    max_sim = x @ x[first]
    for _ in range(1, budget):
        idx = int(np.argmin(max_sim))
        selected.append(idx)
        max_sim = np.maximum(max_sim, x @ x[idx])
    return np.asarray(selected, dtype=np.int64)


def source_binary_labels(dataset, indices):
    labs = []
    cats = []
    for idx in indices:
        labs.append(patch_labels(dataset[idx]))
        cats.append(str(dataset[idx]["cls_name"]))
    return np.stack(labs, axis=0).astype(np.int64), np.asarray(cats)


def load_source_features(carrier):
    dataset_name, dataset, indices, _cls, clip_patch, _y, _cats, _rows = load_full_visa_source()
    if carrier == "clip":
        patch = np.asarray(clip_patch, dtype=np.float16)
    else:
        dino = np.load(FEATURE_CACHE / "D2_fold0_dino.npz", allow_pickle=True)
        patch = np.concatenate([dino["train_patch"], dino["held_patch"]], axis=0).astype(np.float16)
    labels, cats = source_binary_labels(dataset, indices)
    return dataset, indices, patch, labels, cats


def candidate_patches_by_category(patch, labels, cats, cls, max_per_category, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for cat in sorted(set(cats.tolist())):
        image_pos = np.flatnonzero(cats == cat)
        coords = []
        for pos in image_pos:
            local = np.flatnonzero(labels[pos] == cls)
            if local.size:
                coords.extend([(pos, int(p)) for p in local.tolist()])
        if not coords:
            out[cat] = np.zeros((0, patch.shape[-1]), dtype=np.float32)
            continue
        if len(coords) > max_per_category:
            take = rng.choice(len(coords), size=max_per_category, replace=False)
            coords = [coords[int(i)] for i in take]
        arr = np.stack([np.asarray(patch[pos, pid], dtype=np.float32) for pos, pid in coords], axis=0)
        out[cat] = norm_np(arr)
    return out


def build_memory(carrier):
    path = OUT / "memory" / f"{carrier}_source_memory.npz"
    stats_path = OUT / "memory" / f"{carrier}_source_memory_stats.json"
    if path.exists() and stats_path.exists():
        m = np.load(path)
        return m["normal"], m["anomaly"], read_json(stats_path)
    _dataset, _indices, patch, labels, cats = load_source_features(carrier)
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = sorted(set(cats.tolist()))
    per_cat_budget = max(1, MEMORY_BUDGET_PER_CLASS // len(categories))
    stats = {
        "carrier": carrier.upper(),
        "source": "Full VisA",
        "strategy": "category-balanced FPS coreset, separate normal/anomaly memory",
        "memory_budget_total": MEMORY_BUDGET_TOTAL,
        "memory_budget_normal": MEMORY_BUDGET_PER_CLASS,
        "memory_budget_anomaly": MEMORY_BUDGET_PER_CLASS,
        "per_category": [],
    }
    memories = {}
    for cls, name in [(0, "normal"), (1, "anomaly")]:
        max_candidates = 4096 if cls == 0 else 20000
        candidates = candidate_patches_by_category(patch, labels, cats, cls, max_candidates, seed=SEED + cls)
        parts = []
        remainder_pool = []
        for cat in categories:
            x = candidates[cat]
            if x.shape[0] == 0:
                take = np.zeros((0, patch.shape[-1]), dtype=np.float32)
            else:
                ids = fps_indices(x, min(per_cat_budget, x.shape[0]))
                take = x[ids]
                if x.shape[0] > take.shape[0]:
                    remainder_pool.append(x)
            parts.append(take)
            stats["per_category"].append({
                "carrier": carrier.upper(),
                "class": name,
                "source_category": cat,
                "candidate_patches": int(x.shape[0]),
                "selected_initial": int(take.shape[0]),
            })
        bank = np.concatenate([p for p in parts if p.size], axis=0) if any(p.size for p in parts) else np.zeros((0, patch.shape[-1]), dtype=np.float32)
        if bank.shape[0] < MEMORY_BUDGET_PER_CLASS and remainder_pool:
            rem = np.concatenate(remainder_pool, axis=0)
            ids = fps_indices(rem, min(MEMORY_BUDGET_PER_CLASS - bank.shape[0], rem.shape[0]))
            bank = np.concatenate([bank, rem[ids]], axis=0)
        if bank.shape[0] > MEMORY_BUDGET_PER_CLASS:
            bank = bank[fps_indices(bank, MEMORY_BUDGET_PER_CLASS)]
        memories[name] = norm_np(bank.astype(np.float32))
        stats[f"{name}_memory_vectors"] = int(memories[name].shape[0])
        stats[f"{name}_feature_dim"] = int(memories[name].shape[1])
    np.savez_compressed(path, normal=memories["normal"].astype(np.float32), anomaly=memories["anomaly"].astype(np.float32))
    write_json(stats_path, stats)
    return memories["normal"], memories["anomaly"], stats


@torch.no_grad()
def ensure_dino_target_cache(target, dataset, device):
    path = DINO_CACHE / f"{target}_dino.npz"
    if path.exists():
        return path
    DINO_CACHE.mkdir(parents=True, exist_ok=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True).to(device).eval()
    tfm = dino_transform()
    patch_list, rows = [], []
    for i in range(len(dataset)):
        if i % 50 == 0:
            print(f"[XLVII] extract DINO {target} {i}/{len(dataset)}", flush=True)
        item = dataset[i]
        _cls, patch = extract_dino_one(model, tfm, item["img_path"], device)
        patch_list.append(patch.astype(np.float16))
        rows.append({
            "position": i,
            "category": str(item["cls_name"]),
            "label": int(item["anomaly"]),
            "mask_available": bool(item["mask_available"]),
            "img_path": item["img_path"],
        })
    np.savez_compressed(path, patch=np.stack(patch_list).astype(np.float16), rows=np.asarray(rows, dtype=object))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def clip_target_cache(target):
    if target == "BTAD":
        return XLV_A / "feature_cache" / "BTAD_clip.npz"
    return XLV_B / "feature_cache" / "CVC-ClinicDB_clip.npz"


def nearest_diff_maps(patch, mem_n, mem_a, device, chunk=4096):
    maps = np.zeros((patch.shape[0], SIDE, SIDE), dtype=np.float16)
    mn = torch.tensor(norm_np(mem_n), dtype=torch.float32, device=device)
    ma = torch.tensor(norm_np(mem_a), dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(patch.shape[0]):
            q_np = norm_np(np.asarray(patch[i], dtype=np.float32))
            scores = []
            for start in range(0, q_np.shape[0], chunk):
                q = torch.tensor(q_np[start:start + chunk], dtype=torch.float32, device=device)
                sim_n = (q @ mn.t()).max(dim=1).values
                sim_a = (q @ ma.t()).max(dim=1).values
                scores.append((sim_a - sim_n).detach().cpu().numpy())
            maps[i] = np.concatenate(scores, axis=0).reshape(SIDE, SIDE).astype(np.float16)
    return maps


def top1_percent_score(map37):
    flat = np.asarray(map37, dtype=np.float32).reshape(-1)
    k = max(1, int(flat.size * 0.01))
    return float(np.sort(flat)[-k:].mean())


def evaluate_maps(target, dataset, rows, maps):
    by_cat = defaultdict(lambda: {"py": [], "ps": []})
    py_all, ps_all = [], []
    for i, row in enumerate(rows):
        item = dataset[i]
        if not bool(row["mask_available"]):
            continue
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
        full = upsample_map(np.asarray(maps[i], dtype=np.float32)).reshape(-1)
        py_all.append(mask)
        ps_all.append(full)
        by_cat[str(row["category"])]["py"].append(mask)
        by_cat[str(row["category"])]["ps"].append(full)
    py = np.concatenate(py_all)
    ps = np.concatenate(ps_all)
    metric = {
        "target": target,
        "P-AUROC": safe_auc(py, ps),
        "P-AP": safe_ap(py, ps),
        "mask_images": len(py_all),
    }
    metric["G_pixel"] = gp(metric)
    per = []
    for cat, vals in sorted(by_cat.items()):
        cy = np.concatenate(vals["py"])
        cs = np.concatenate(vals["ps"])
        row = {"target": target, "category": cat, "P-AUROC": safe_auc(cy, cs), "P-AP": safe_ap(cy, cs), "mask_images": len(vals["py"])}
        row["G_pixel"] = gp(row)
        per.append(row)
    return metric, per


def run_variant(carrier, target, dataset, cache_path, mem_n, mem_a, device):
    map_path = OUT / "map_cache" / f"{target}_{carrier}_maps37.npy"
    if map_path.exists():
        maps = np.load(map_path, mmap_mode="r")
        c = np.load(cache_path, allow_pickle=True)
        rows = [dict(r) for r in c["rows"]]
    else:
        c = np.load(cache_path, allow_pickle=True)
        patch = c["patch"]
        rows = [dict(r) for r in c["rows"]]
        maps_arr = nearest_diff_maps(patch, mem_n, mem_a, device)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(map_path, maps_arr)
        maps = np.load(map_path, mmap_mode="r")
    metric, per = evaluate_maps(target, dataset, rows, maps)
    metric.update({"variant": f"{carrier.upper()}_SOURCE_MEMORY", "carrier": carrier.upper(), "memory_score": "top1 cosine sim_A - top1 cosine sim_N", "map_cache": str(map_path)})
    for r in per:
        r.update({"variant": f"{carrier.upper()}_SOURCE_MEMORY", "carrier": carrier.upper()})
    return metric, per


def read_existing_references():
    rows = []
    for path in [XLV_A / "PILOT_METRICS.csv", XLV_B / "RAW_METRICS.csv"]:
        with open(path, newline="", encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                target = r["target"]
                if target not in {"BTAD", "CVC-ClinicDB"}:
                    continue
                variant = r["variant"]
                if variant == "V1_STATE_ALIGNED":
                    name = "STATE_ALIGNED_REFERENCE"
                elif variant == "V2_INSTANCE_MEMORY":
                    name = "INSTANCE_MEMORY_REFERENCE"
                else:
                    continue
                rows.append({
                    "target": target,
                    "variant": name,
                    "carrier": "REFERENCE",
                    "P-AUROC": float(r["P-AUROC"]),
                    "P-AP": float(r["P-AP"]),
                    "G_pixel": 0.5 * (float(r["P-AUROC"]) + float(r["P-AP"])),
                    "source": str(path),
                })
    return rows


def compare_rows(raw):
    by = {(r["target"], r["variant"]): r for r in raw}
    dino_clip = []
    dino_state = []
    for target in ["BTAD", "CVC-ClinicDB"]:
        c = by[(target, "CLIP_SOURCE_MEMORY")]
        d = by[(target, "DINO_SOURCE_MEMORY")]
        s = by[(target, "STATE_ALIGNED_REFERENCE")]
        dino_clip.append({
            "target": target,
            "DINO_minus_CLIP_P-AUROC": float(d["P-AUROC"]) - float(c["P-AUROC"]),
            "DINO_minus_CLIP_P-AP": float(d["P-AP"]) - float(c["P-AP"]),
            "DINO_minus_CLIP_G_pixel": gp(d) - gp(c),
        })
        dino_state.append({
            "target": target,
            "DINO_minus_StateAligned_P-AUROC": float(d["P-AUROC"]) - float(s["P-AUROC"]),
            "DINO_minus_StateAligned_P-AP": float(d["P-AP"]) - float(s["P-AP"]),
            "DINO_minus_StateAligned_G_pixel": gp(d) - gp(s),
            "within_2pp_of_StateAligned": bool(gp(d) >= gp(s) - 2.0),
        })
    return dino_clip, dino_state


def determine_outcome(dino_clip, dino_state):
    btad_dc = next(r for r in dino_clip if r["target"] == "BTAD")
    clinic_dc = next(r for r in dino_clip if r["target"] == "CVC-ClinicDB")
    btad_ds = next(r for r in dino_state if r["target"] == "BTAD")
    cont = (btad_dc["DINO_minus_CLIP_G_pixel"] >= 1.0 or btad_ds["within_2pp_of_StateAligned"]) and clinic_dc["DINO_minus_CLIP_G_pixel"] >= -2.0
    if cont:
        return "XLVII_DINO_SOURCE_MEMORY_CONTINUE_VALUE"
    if btad_dc["DINO_minus_CLIP_G_pixel"] < 0 and clinic_dc["DINO_minus_CLIP_G_pixel"] < -2.0 and not btad_ds["within_2pp_of_StateAligned"]:
        return "XLVII_DINO_SOURCE_MEMORY_STOP_BOTH_TARGETS_WEAK"
    return "XLVII_DINO_SOURCE_MEMORY_NO_CONTINUATION"


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    targets = {
        "BTAD": ExternalDataset(DATA_ROOT / "btad" / "BTech_Dataset_transformed", "BTAD", mode="test"),
        "CVC-ClinicDB": MedicalDataset(DATA_ROOT / "CVC-ClinicDB", mode="test"),
    }
    target_cache = {
        "BTAD": {
            "clip": clip_target_cache("BTAD"),
            "dino": ensure_dino_target_cache("BTAD", targets["BTAD"], device),
        },
        "CVC-ClinicDB": {
            "clip": clip_target_cache("CVC-ClinicDB"),
            "dino": ensure_dino_target_cache("CVC-ClinicDB", targets["CVC-ClinicDB"], device),
        },
    }

    memories, stats_rows = {}, []
    for carrier in ["clip", "dino"]:
        mn, ma, stats = build_memory(carrier)
        memories[carrier] = (mn, ma)
        stats_rows.append({
            "carrier": carrier.upper(),
            "source": "Full VisA",
            "normal_memory_vectors": int(mn.shape[0]),
            "anomaly_memory_vectors": int(ma.shape[0]),
            "feature_dim": int(mn.shape[1]),
            "budget_total": MEMORY_BUDGET_TOTAL,
            "strategy": stats["strategy"],
        })
        for r in stats["per_category"]:
            rr = dict(r)
            rr["source"] = "Full VisA"
            stats_rows.append(rr)

    raw, per_btad = [], []
    for target, dataset in targets.items():
        for carrier in ["clip", "dino"]:
            metric, per = run_variant(carrier, target, dataset, target_cache[target][carrier], *memories[carrier], device)
            raw.append(metric)
            if target == "BTAD":
                per_btad.extend(per)
    raw.extend(read_existing_references())
    dino_clip, dino_state = compare_rows(raw)
    outcome = determine_outcome(dino_clip, dino_state)
    btad_dc = next(r for r in dino_clip if r["target"] == "BTAD")
    clinic_dc = next(r for r in dino_clip if r["target"] == "CVC-ClinicDB")
    btad_ds = next(r for r in dino_state if r["target"] == "BTAD")
    btad_continuation_gate = bool(
        btad_dc["DINO_minus_CLIP_G_pixel"] >= 1.0 or btad_ds["within_2pp_of_StateAligned"]
    )
    clinic_noncollapse_gate = bool(clinic_dc["DINO_minus_CLIP_G_pixel"] >= -2.0)

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "MEMORY_STATS.csv", stats_rows)
    write_csv(OUT / "PER_CATEGORY_BTAD.csv", per_btad)
    write_csv(OUT / "DINO_VS_CLIP_MEMORY.csv", dino_clip)
    write_csv(OUT / "DINO_VS_STATEALIGNED.csv", dino_state)
    write_json(OUT / "OUTCOME.json", {
        "phase": "XLVII",
        "outcome": outcome,
        "runtime_sec": time.time() - start,
        "source": "Full VisA",
        "targets": list(targets),
        "target_data_used_for_memory_or_tuning": False,
        "text_used": False,
        "training_used": False,
        "dynamic_memory": False,
        "metric_learning": False,
        "memory_budget_total": MEMORY_BUDGET_TOTAL,
        "continuation_rule": {
            "BTAD_DINO_minus_CLIP_G_ge_1pp_or_DINO_within_StateAligned_minus_2pp": btad_continuation_gate,
            "ClinicDB_DINO_minus_CLIP_G_ge_minus_2pp": clinic_noncollapse_gate,
            "overall_continue": bool(btad_continuation_gate and clinic_noncollapse_gate),
        },
        "dino_vs_clip": dino_clip,
        "dino_vs_statealigned": dino_state,
    })

    lines = ["# Phase XLVII DINO Source Memory Zero-Shot", "", f"Outcome: **{outcome}**", "", "## Raw Metrics", ""]
    for r in raw:
        lines.append(f"- {r['target']} {r['variant']}: P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, G {float(r['G_pixel']):.4f}.")
    lines.extend(["", "## DINO vs CLIP", ""])
    for r in dino_clip:
        lines.append(f"- {r['target']}: DINO-CLIP G {r['DINO_minus_CLIP_G_pixel']:+.4f}, P-AUROC {r['DINO_minus_CLIP_P-AUROC']:+.4f}, P-AP {r['DINO_minus_CLIP_P-AP']:+.4f}.")
    lines.extend(["", "## DINO vs StateAligned", ""])
    for r in dino_state:
        relation = "within -2pp" if r["within_2pp_of_StateAligned"] else "clearly behind"
        lines.append(f"- {r['target']}: DINO-StateAligned G {r['DINO_minus_StateAligned_G_pixel']:+.4f} ({relation}).")
    lines.extend([
        "",
        "## Answers",
        "",
        "1. DINO Source Memory is judged against the identical CLIP Source Memory control in `DINO_VS_CLIP_MEMORY.csv`.",
        "2. Cross-dataset validity requires BTAD continuation plus no ClinicDB collapse.",
        "3. StateAligned and InstanceMemory are references read from XLV outputs; no retraining was run.",
        "",
        f"Runtime: {time.time() - start:.1f}s.",
        "",
        "Stop: no fusion, top-k search, DINO layer search, augmentation, or target-tuned variant was started.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XLVII DINO Source Memory Zero-Shot\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- DINO vs CLIP: {json.dumps(dino_clip, default=json_default)}.\n")
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
