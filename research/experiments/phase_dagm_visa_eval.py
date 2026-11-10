import argparse
import csv
import gc
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_visa_source
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    CHECKPOINT,
    VARIANTS as LZP_VARIANTS,
    evaluate_maps as lzp_evaluate_maps,
    load_components,
    quadrant_crops,
    score_scale_row,
    score_feature_grid,
    torch_resize,
)
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE
from experiments.standard_zsad_metrics import summarize_standard
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = ROOT / "results" / "paper_assets" / "dagm_visa_eval"
CACHE = OUT / "cache"
TARGET = "DAGM"
SOURCE = "Full VisA"
DAGM_EVAL_SIDE = 74


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


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def resize_mask_grid(mask, side):
    import torch.nn.functional as F

    m = torch.tensor(np.asarray(mask, dtype=np.float32)).view(1, 1, *mask.shape)
    y = F.interpolate(m, size=(side, side), mode="nearest").view(side, side).numpy()
    return (y > 0.5).astype(np.uint8)


def resize_score_grid(score, side):
    arr = np.asarray(score, dtype=np.float32)
    if arr.shape == (side, side):
        return arr
    return torch_resize(arr, (side, side), align_corners=True).astype(np.float32)


def evaluate_maps_grid(source, target, maps, metas, variants=None, side=DAGM_EVAL_SIDE):
    variants = tuple(variants or maps.keys())
    raw_rows, cat_rows = [], []
    for variant in variants:
        by_cat = {}
        all_iy, all_is, all_py, all_ps, all_masks, all_scores = [], [], [], [], [], []
        for i, meta in enumerate(metas):
            cat = str(meta["category"])
            label = int(meta["label"])
            smap = resize_score_grid(maps[variant][i], side)
            mask = resize_mask_grid(meta["mask"], side)
            iscore = top1_image_score(smap)
            d = by_cat.setdefault(cat, {"iy": [], "is": [], "py": [], "ps": [], "masks": [], "scores": []})
            d["iy"].append(label)
            d["is"].append(iscore)
            d["py"].append(mask.reshape(-1))
            d["ps"].append(smap.reshape(-1))
            d["masks"].append(mask)
            d["scores"].append(smap)
            all_iy.append(label)
            all_is.append(iscore)
            all_py.append(mask.reshape(-1))
            all_ps.append(smap.reshape(-1))
            all_masks.append(mask)
            all_scores.append(smap)
        for cat, vals in sorted(by_cat.items()):
            py = np.concatenate(vals["py"]) if vals["py"] else np.asarray([], dtype=np.uint8)
            ps = np.concatenate(vals["ps"]) if vals["ps"] else np.asarray([], dtype=np.float32)
            metric = summarize_standard(vals["iy"], vals["is"], py, ps, vals["masks"], vals["scores"])
            cat_rows.append({
                "source": source,
                "dataset": target,
                "category": cat,
                "variant": variant,
                "protocol": "DAGM_GRID",
                "score_resolution": f"{side}x{side}",
                **{f"M_FR_{k.replace('-', '_')}": v for k, v in metric.items()},
            })
        py_all = np.concatenate(all_py) if all_py else np.asarray([], dtype=np.uint8)
        ps_all = np.concatenate(all_ps) if all_ps else np.asarray([], dtype=np.float32)
        metric = summarize_standard(all_iy, all_is, py_all, ps_all, all_masks, all_scores)
        raw_rows.append({
            "source": source,
            "dataset": target,
            "variant": variant,
            "aggregation": "DATASET_GLOBAL",
            "pixel_eval_resolution": f"DAGM_GRID_{side}x{side}",
            "score_resolution": f"{side}x{side}",
            **{f"M_FR_{k.replace('-', '_')}": v for k, v in metric.items()},
        })
    return raw_rows, cat_rows


class DAGMTargetDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"DAGM root not found: {self.root}")
        self.items = []
        for cls in sorted([p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("Class")], key=lambda p: int(p.name[5:])):
            test = cls / "Test"
            label_dir = test / "Label"
            for img_path in sorted(test.glob("*.PNG")):
                mask_path = label_dir / f"{img_path.stem}_label.PNG"
                label = 1 if mask_path.exists() else 0
                self.items.append({
                    "cls_name": cls.name,
                    "img_path": str(img_path),
                    "mask_path": str(mask_path) if mask_path.exists() else "",
                    "anomaly": label,
                })
        if not self.items:
            raise RuntimeError(f"No DAGM test images found under {self.root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        rec = self.items[int(index)]
        img = Image.open(rec["img_path"]).convert("RGB")
        w, h = img.size
        if rec["mask_path"]:
            mask = (np.asarray(Image.open(rec["mask_path"]).convert("L")) > 0).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        return {
            "img_path": rec["img_path"],
            "img_mask": torch.from_numpy(mask),
            "cls_name": rec["cls_name"],
            "anomaly": int(rec["anomaly"]),
            "cls_id": int(rec["cls_name"][5:]) - 1,
        }


def dagm_audit(dataset):
    rows = []
    by_cls = {}
    for rec in dataset.items:
        d = by_cls.setdefault(rec["cls_name"], {"images": 0, "anomaly": 0})
        d["images"] += 1
        d["anomaly"] += int(rec["anomaly"])
    for cls, d in sorted(by_cls.items(), key=lambda x: int(x[0][5:])):
        rows.append({
            "dataset": TARGET,
            "category": cls,
            "test_images": d["images"],
            "test_anomaly": d["anomaly"],
            "test_normal": d["images"] - d["anomaly"],
            "mask_available": d["anomaly"],
        })
    return rows


def dagm_cell(dataset, patch=None):
    refs = list(range(len(dataset)))
    return ("dagm_visa_source_holdout", TARGET, dataset, refs, patch, "source_dataset")


@torch.no_grad()
def build_global_patch_cache(dataset, clip_model, preprocess, device, batch_size=8, force=False, smoke=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    n_total = min(24, len(dataset)) if smoke else len(dataset)
    suffix = "_smoke" if smoke else ""
    cache_path = CACHE / f"dagm_global_patch_float16{suffix}.npy"
    meta_path = CACHE / f"dagm_global_patch_meta{suffix}.json"
    if cache_path.exists() and meta_path.exists() and not force:
        arr = np.load(cache_path, mmap_mode="r")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("n_images", -1)) == n_total:
            print(f"[DAGM] reuse global patch cache {cache_path}", flush=True)
            return arr
    n = n_total
    patches = np.lib.format.open_memmap(cache_path, dtype=np.float16, mode="w+", shape=(n, 37 * 37, 1024))
    batch, posns = [], []
    for pos in range(n):
        if pos % 100 == 0:
            print(f"[DAGM] global CLIP {pos}/{n}", flush=True)
        img = Image.open(dataset.items[pos]["img_path"]).convert("RGB")
        batch.append(preprocess(img))
        posns.append(pos)
        if len(batch) == batch_size or pos == n - 1:
            images = torch.stack(batch, dim=0).to(device)
            out = visual_outputs(clip_model, images, adapter=None, grad=False)
            z = normalized_patch_feature(out).float().detach().cpu().numpy().astype(np.float16)
            for i, p in enumerate(posns):
                patches[p] = z[i]
            patches.flush()
            batch, posns = [], []
            del images, out, z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    meta_path.write_text(json.dumps({"n_images": n, "shape": list(patches.shape), "dtype": "float16", "smoke": bool(smoke)}, indent=2), encoding="utf-8")
    return np.load(cache_path, mmap_mode="r")


def run_ours(dataset, args, device):
    import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp

    raw_path = OUT / "RAW_METRICS_ours.csv"
    per_path = OUT / "PER_CATEGORY_ours.csv"
    if raw_path.exists() and not args.force:
        rows = read_csv(raw_path)
        if any(r.get("source") == SOURCE and r.get("dataset") == TARGET for r in rows):
            print("[DAGM][Ours] skip existing", flush=True)
            return rows
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    model_info = load_visa_source(device)
    patch = build_global_patch_cache(dataset, clip_model, preprocess, device, args.batch_size, args.force_cache, args.smoke)
    refs = list(range(min(24, len(dataset)) if args.smoke else len(dataset)))
    old_variants = lzp.VARIANTS
    lzp.VARIANTS = LZP_VARIANTS
    try:
        maps, metas, scale = build_ours_maps_batched(model_info, clip_model, preprocess, patch, refs, dataset, device, args.batch_size)
        raw, percat = evaluate_maps_grid(SOURCE, TARGET, maps, metas, LZP_VARIANTS)
    finally:
        lzp.VARIANTS = old_variants
    write_csv(raw_path, raw)
    write_csv(per_path, percat)
    write_csv(OUT / "SCORE_SCALE_ours.csv", scale)
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw


@torch.no_grad()
def build_ours_maps_batched(model_info, clip_model, preprocess, patch, refs, dataset, device, crop_batch_size):
    maps = {v: [] for v in LZP_VARIANTS}
    metas = []
    scale_rows = []
    n = len(refs)
    crop_batch_size = max(4, int(crop_batch_size))
    pending, pending_meta = [], []
    local74_by_pos = {}

    def flush():
        if not pending:
            return
        batch = torch.stack(pending, dim=0).to(device)
        out = visual_outputs(clip_model, batch, adapter=None, grad=False)
        patches = normalized_patch_feature(out).float().detach().cpu().numpy().astype(np.float32)
        for z, (pos, crop_idx) in zip(patches, pending_meta):
            q = score_feature_grid(model_info, z, device)
            arr = local74_by_pos.setdefault(pos, np.zeros((37 * 2, 37 * 2), dtype=np.float32))
            if crop_idx == 0:
                arr[:37, :37] = q
            elif crop_idx == 1:
                arr[:37, 37:] = q
            elif crop_idx == 2:
                arr[37:, :37] = q
            else:
                arr[37:, 37:] = q
        pending.clear()
        pending_meta.clear()
        del batch, out, patches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for out_pos, ref in enumerate(refs):
        if out_pos % 25 == 0:
            print(f"[DAGM][OursFast] {out_pos}/{n}", flush=True)
        item = dataset[int(ref)]
        cat = str(item["cls_name"])
        label = int(item["anomaly"])
        path = str(item["img_path"])
        gt = item["img_mask"].numpy().astype(np.uint8)
        metas.append({"category": cat, "label": label, "mask": gt, "img_path": path})
        for ci, crop in enumerate(quadrant_crops(path)):
            pending.append(preprocess(crop))
            pending_meta.append((out_pos, ci))
            if len(pending) >= crop_batch_size:
                flush()
    flush()

    for out_pos, ref in enumerate(refs):
        global37 = score_feature_grid(model_info, patch[out_pos], device)
        local74 = local74_by_pos[out_pos]
        global74 = torch_resize(global37, (37 * 2, 37 * 2), align_corners=True)
        final74 = 0.5 * global74 + 0.5 * local74
        local_down37 = torch_resize(local74, (37, 37), align_corners=True)
        e3 = 0.5 * global37 + 0.5 * local_down37
        maps["B0_HardNormal"].append(global37.astype(np.float16))
        maps["E1_LocalZoomOnly"].append(local74.astype(np.float16))
        maps["E2_GlobalLocalZoomPyramid"].append(final74.astype(np.float16))
        maps["E3_ReencodingResolutionControl"].append(e3.astype(np.float16))
        scale_rows.append(score_scale_row(model_info["source"], TARGET, metas[out_pos]["category"], out_pos, global37, local74, final74))
    return maps, metas, scale_rows


def patch_protocol(module, dataset):
    def _protocol_cells(source_name, device, smoke):
        if source_name != SOURCE:
            return [], []
        cell = dagm_cell(dataset, patch=None)
        return [cell], []
    module.protocol_cells = _protocol_cells


def purge_official_conflicts(prefixes=("AnomalyCLIP_lib", "prompt_ensemble", "utils", "FAPrompt")):
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            del sys.modules[name]


def run_anomalyclip(dataset, args, device):
    raw_path = OUT / "RAW_METRICS_anomalyclip.csv"
    per_path = OUT / "PER_CATEGORY_anomalyclip.csv"
    if raw_path.exists() and not args.force:
        rows = read_csv(raw_path)
        if any(r.get("source") == SOURCE and r.get("dataset") == TARGET for r in rows):
            print("[DAGM][AnomalyCLIP] skip existing", flush=True)
            return rows
    purge_official_conflicts(("AnomalyCLIP_lib", "prompt_ensemble", "utils"))
    import experiments.phase_anomalyclip_same_protocol_audit as ac
    ac = importlib.reload(ac)
    if not ac.WEIGHTS[SOURCE].exists():
        return []
    patch_protocol(ac, dataset)
    model, prompt, text_features, ac_args = ac.load_model(SOURCE, device, smoke=args.smoke)
    ac_args.test_batch_size = args.batch_size
    maps, metas = ac.build_maps(SOURCE, TARGET, model, text_features, ac_args, device, args.smoke)
    raw, percat = evaluate_maps_grid(SOURCE, TARGET, maps, metas, (ac.VARIANT,))
    write_csv(raw_path, raw)
    write_csv(per_path, percat)
    del model, prompt, text_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw


def run_faprompt(dataset, args, device):
    raw_path = OUT / "RAW_METRICS_faprompt.csv"
    per_path = OUT / "PER_CATEGORY_faprompt.csv"
    if raw_path.exists() and not args.force:
        rows = read_csv(raw_path)
        if rows:
            print("[DAGM][FAPrompt] skip existing", flush=True)
            return rows
    purge_official_conflicts(("AnomalyCLIP_lib", "prompt_ensemble", "utils", "FAPrompt"))
    import experiments.phase_paper_baseline_close as fap
    fap = importlib.reload(fap)
    ckpt = fap.FAPROMPT_CHECKPOINTS[SOURCE]
    if not ckpt.exists():
        return []
    model, prompt = fap.load_faprompt(device, ckpt)
    fa_args = argparse.Namespace(image_size=518)
    preprocess, _ = fap.fa_get_transform(fa_args)
    maps, metas = [], []
    n = min(24, len(dataset)) if args.smoke else len(dataset)
    for pos in range(n):
        if pos % 50 == 0:
            print(f"[DAGM][FAPrompt] {pos}/{n}", flush=True)
        item = dataset[pos]
        img = Image.open(item["img_path"]).convert("RGB")
        amap, _score = fap.faprompt_map(model, prompt, preprocess(img).unsqueeze(0).to(device), device)
        maps.append(amap.astype(np.float16))
        metas.append({"category": item["cls_name"], "label": int(item["anomaly"]), "mask": item["img_mask"].numpy().astype(np.uint8), "img_path": item["img_path"]})
    raw_rows, percat = evaluate_maps_grid(SOURCE, TARGET, {"FAPrompt": maps}, metas, ("FAPrompt",))
    raw = raw_rows[0]
    raw_rows = [{**raw, "source": SOURCE, "dataset": TARGET, "method": "FAPrompt", "variant": "FAPrompt"}]
    write_csv(raw_path, raw_rows)
    write_csv(per_path, percat)
    del model, prompt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_rows


def run_visualad(dataset, args, device):
    import experiments.phase_visualad_exact_protocol_audit as va

    raw_path = OUT / "RAW_METRICS_visualad.csv"
    per_path = OUT / "PER_CATEGORY_visualad.csv"
    att_path = OUT / "ATTENTION_visualad.csv"
    ck_path = OUT / "CHECKSUM_visualad.csv"
    if raw_path.exists() and not args.force:
        rows = read_csv(raw_path)
        if any(r.get("source") == SOURCE and r.get("dataset") == TARGET for r in rows):
            print("[DAGM][VisualAD] skip existing", flush=True)
            return rows
    official = va.ensure_official_imports()
    model, layer_transforms, cross_attn, checkpoint = va.load_visualad_bundle(SOURCE, "E1c_FullVisualAD", official, torch.device(device))
    preprocess = va.official_transform(official)
    n = min(24, len(dataset)) if args.smoke else len(dataset)
    maps = {"E1c_FullVisualAD": []}
    metas, att = [], []
    checksum_before = va.frozen_backbone_checksum(model)
    batch, batch_pos = [], []
    for pos in range(n):
        if pos % 50 == 0:
            print(f"[DAGM][VisualAD] {pos}/{n}", flush=True)
        item = dataset[pos]
        batch.append(preprocess(Image.open(item["img_path"]).convert("RGB")))
        batch_pos.append(pos)
        metas.append({"category": item["cls_name"], "label": int(item["anomaly"]), "mask": item["img_mask"].numpy().astype(np.uint8), "img_path": item["img_path"]})
        if len(batch) == args.batch_size or pos == n - 1:
            images = torch.stack(batch, dim=0).to(device)
            final, stat_rows = va.infer_batch(model, layer_transforms, cross_attn, images)
            for bi, image_pos in enumerate(batch_pos):
                maps["E1c_FullVisualAD"].append(gaussian_filter(final[bi], sigma=va.SIGMA).astype(np.float16))
            for sr in stat_rows:
                sr.update({"source": SOURCE, "dataset": TARGET, "variant": "E1c_FullVisualAD", "image_index": ",".join(str(x) for x in batch_pos)})
                att.append(sr)
            batch, batch_pos = [], []
            del images, final
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    checksum_after = va.frozen_backbone_checksum(model)
    ck = [{
        "source": SOURCE,
        "dataset": TARGET,
        "variant": "E1c_FullVisualAD",
        "checkpoint": str(va.checkpoint_path(SOURCE)),
        "frozen_backbone_checksum_before": checksum_before,
        "frozen_backbone_checksum_after": checksum_after,
        "unchanged": checksum_before == checksum_after,
        "trainable_parameter_count_from_checkpoint": va.trainable_count(model, layer_transforms, cross_attn, "E1c_FullVisualAD"),
        "features_list": ",".join(str(x) for x in checkpoint.get("features_list", va.FEATURES_LIST)),
        "image_size": checkpoint.get("image_size", va.IMAGE_SIZE),
    }]
    raw, percat = evaluate_maps_grid(SOURCE, TARGET, maps, metas, ("E1c_FullVisualAD",))
    write_csv(raw_path, raw)
    write_csv(per_path, percat)
    write_csv(att_path, att)
    write_csv(ck_path, ck)
    del model, layer_transforms, cross_attn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw


def run_fbclip(dataset, args, device):
    import experiments.phase_fbclip_pretrained_headroom_audit as fb

    raw_path = OUT / "RAW_METRICS_fbclip.csv"
    per_path = OUT / "PER_CATEGORY_fbclip.csv"
    if raw_path.exists() and not args.force:
        rows = read_csv(raw_path)
        if any(r.get("source") == SOURCE and r.get("dataset") == TARGET for r in rows):
            print("[DAGM][FB-CLIP] skip existing", flush=True)
            return rows
    if not fb.WEIGHTS[SOURCE].exists():
        return []
    patch_protocol(fb, dataset)
    model, prompt, fb_args, _info = fb.load_fb_model(SOURCE, device)
    fb_args.test_batch_size = args.batch_size
    maps, metas = fb.build_maps_for_pair(SOURCE, TARGET, model, prompt, fb_args, device, args.smoke)
    raw, percat = evaluate_maps_grid(SOURCE, TARGET, maps, metas, (fb.VARIANT,))
    write_csv(raw_path, raw)
    write_csv(per_path, percat)
    del model, prompt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw


def summarize_all():
    specs = [
        ("Ours", "RAW_METRICS_ours.csv", "E2_GlobalLocalZoomPyramid"),
        ("AnomalyCLIP", "RAW_METRICS_anomalyclip.csv", "AnomalyCLIP_OfficialCheckpoint"),
        ("FAPrompt", "RAW_METRICS_faprompt.csv", "FAPrompt"),
        ("VisualAD", "RAW_METRICS_visualad.csv", "E1c_FullVisualAD"),
        ("FB-CLIP", "RAW_METRICS_fbclip.csv", "E1_Official_FBCLIP_Pretrained"),
    ]
    rows = []
    for method, filename, variant in specs:
        for r in read_csv(OUT / filename):
            if r.get("variant") != variant:
                continue
            rows.append({
                "method": method,
                "source": r.get("source", SOURCE),
                "dataset": r.get("dataset", TARGET),
                "variant": variant,
                "I_AUROC": fnum(r, "M_FR_I_AUROC"),
                "I_AP": fnum(r, "M_FR_I_AP"),
                "I_F1max": fnum(r, "M_FR_I_F1max"),
                "P_AUROC": fnum(r, "M_FR_P_AUROC"),
                "P_AP": fnum(r, "M_FR_P_AP"),
                "P_F1max": fnum(r, "M_FR_P_F1max"),
                "AUPRO": fnum(r, "M_FR_PRO"),
            })
    write_csv(OUT / "DAGM_VISA_COMPARISON.csv", rows)
    return rows


def merge_big_table(dagm_rows):
    src = ROOT / "results" / "paper_assets" / "metric_extension" / "BIG_MODEL_COMPARISON.csv"
    rows = read_csv(src)
    out = list(rows)
    for r in dagm_rows:
        out.append({
            "method": r["method"],
            "source": SOURCE,
            "target": TARGET,
            "variant": r["variant"],
            "I_AUROC": r["I_AUROC"],
            "I_AP": r["I_AP"],
            "P_AUROC": r["P_AUROC"],
            "P_AP": r["P_AP"],
            "P_F1MAX": r["P_F1max"],
            "AUPRO": r["AUPRO"],
            "strict_comparable": "YES",
            "result_source": "local DAGM VisA-source unified evaluator",
        })
    write_csv(OUT / "BIG_MODEL_COMPARISON_WITH_DAGM.csv", out)


def write_report(rows, audit, runtime, requested):
    lines = [
        "# DAGM VisA-Source Holdout Evaluation",
        "",
        f"Runtime: {runtime:.1f}s",
        "",
        "Target: `DAGM` from `E:/WorkSpace/dataset/DAGM_KaggleUpload`.",
        "Source assets: `Full VisA` checkpoints/cache only. No training or target tuning was performed.",
        "",
        "## Dataset Audit",
        "",
        "| Category | Test images | Anomaly | Normal |",
        "|---|---:|---:|---:|",
    ]
    for r in audit:
        lines.append(f"| {r['category']} | {r['test_images']} | {r['test_anomaly']} | {r['test_normal']} |")
    lines += ["", "## Unified Metrics", "", "| Method | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {fnum(r,'I_AUROC'):.4f} | {fnum(r,'I_AP'):.4f} | {fnum(r,'P_AUROC'):.4f} | {fnum(r,'P_AP'):.4f} | {fnum(r,'P_F1max'):.4f} | {fnum(r,'AUPRO'):.4f} |")
    lines += [
        "",
        "## Notes",
        "",
        f"- Requested models: `{', '.join(requested)}`.",
        f"- DAGM rows use a memory-safe `{DAGM_EVAL_SIDE}x{DAGM_EVAL_SIDE}` grid evaluator (`pixel_eval_resolution=DAGM_GRID_{DAGM_EVAL_SIDE}x{DAGM_EVAL_SIDE}`), not the paper full-resolution M_FR evaluator.",
        "- MRAD is not evaluated here: the local same-protocol MRAD wrapper is dataset-specific and previously documented as unsupported outside MVTec/VisA/BTAD/MPDD without adding a new loader.",
        "- `BIG_MODEL_COMPARISON_WITH_DAGM.csv` appends the local DAGM rows to the current metric-extension big comparison artifact.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dagm-root", default="E:/WorkSpace/dataset/DAGM_KaggleUpload")
    p.add_argument("--models", nargs="+", default=["ours", "anomalyclip", "faprompt", "visualad", "fbclip"], choices=["ours", "anomalyclip", "faprompt", "visualad", "fbclip"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-cache", action="store_true")
    args = p.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    dataset = DAGMTargetDataset(args.dagm_root)
    audit = dagm_audit(dataset)
    write_csv(OUT / "DAGM_DATASET_AUDIT.csv", audit)
    runners = {
        "ours": run_ours,
        "anomalyclip": run_anomalyclip,
        "faprompt": run_faprompt,
        "visualad": run_visualad,
        "fbclip": run_fbclip,
    }
    for name in args.models:
        print(f"[DAGM] run {name}", flush=True)
        runners[name](dataset, args, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows = summarize_all()
    merge_big_table(rows)
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {"outcome": "DAGM_VISA_SOURCE_EVALUATION_COMPLETE", "models": args.models, "rows": len(rows), "runtime_sec": runtime})
    write_report(rows, audit, runtime, args.models)
    print(json.dumps({"outcome": "DAGM_VISA_SOURCE_EVALUATION_COMPLETE", "rows": len(rows), "runtime_sec": runtime}, indent=2), flush=True)


if __name__ == "__main__":
    main()
