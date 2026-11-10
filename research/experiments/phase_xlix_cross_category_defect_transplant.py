import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import load_model
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, EPOCHS, LR, SEED, WEIGHT_DECAY, balanced_loss
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_xliii_anchor_specificity import low_rank_rows
from experiments.phase_xlv_cross_domain_transfer import evaluate_external, gp, load_full_visa_source
from experiments.phase_xlv_cross_domain_transfer import ExternalDataset
from experiments.phase_xlv_medical_pixel_transfer import MedicalDataset, eval_scores
from experiments.phase_xlviii_dino_memory_distillation import e0_rows, metric_row, safe_float, score_v1_cached
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY, SIDE, norm_np, patch_labels
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_xlix_cross_category_defect_transplant")
XLV_A = Path("results/phase_xlv_cross_domain_transfer")
XLV_B = Path("results/phase_xlv_medical_pixel_transfer")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
ALPHA = 0.85


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


def raw_records(dataset, indices):
    rows = []
    root = Path(dataset.root)
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        meta = dataset.data_all[idx]
        rows.append({
            "pos": pos,
            "idx": int(idx),
            "category": str(item["cls_name"]),
            "anomaly": int(item["anomaly"]),
            "img_path": str(root / meta["img_path"]),
            "mask_path": str(root / meta.get("mask_path", "")) if meta.get("mask_path", "") else "",
        })
    return rows


def pil_resize_rgb(path):
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)


def mask_resize(path):
    arr = np.asarray(Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def transplant_image(donor, recipient, rng):
    donor_img = np.asarray(pil_resize_rgb(donor["img_path"]), dtype=np.float32)
    rec_img = np.asarray(pil_resize_rgb(recipient["img_path"]), dtype=np.float32)
    donor_mask_multi = mask_resize(donor["mask_path"])
    donor_mask = donor_mask_multi > 0
    ys, xs = np.where(donor_mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop_img = donor_img[y0:y1, x0:x1]
    crop_mask = donor_mask[y0:y1, x0:x1]
    crop_multi = donor_mask_multi[y0:y1, x0:x1]
    h, w = crop_mask.shape
    if h <= 0 or w <= 0 or h > IMAGE_SIZE or w > IMAGE_SIZE:
        return None
    px = int(rng.integers(0, IMAGE_SIZE - w + 1))
    py = int(rng.integers(0, IMAGE_SIZE - h + 1))
    out = rec_img.copy()
    alpha = (crop_mask.astype(np.float32) * ALPHA)[..., None]
    roi = out[py:py + h, px:px + w]
    out[py:py + h, px:px + w] = roi * (1.0 - alpha) + crop_img * alpha
    out_mask_multi = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    out_mask_multi[py:py + h, px:px + w] = np.where(crop_mask, crop_multi, 0).astype(np.uint8)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB"), out_mask_multi, {
        "paste_x": px,
        "paste_y": py,
        "crop_w": w,
        "crop_h": h,
        "mask_pixels": int(crop_mask.sum()),
        "mask_ids": json.dumps([int(x) for x in sorted(np.unique(crop_multi[crop_multi > 0]).tolist())]),
    }


def patch_labels_from_mask(mask_multi):
    mask = torch.tensor((mask_multi > 0).astype(np.float32)).view(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    small = F.interpolate(mask, size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape(-1)
    return (small > 0.5).numpy().astype(np.int64)


@torch.no_grad()
def extract_patch(model, transform, image_pil, device):
    img = transform(image_pil).unsqueeze(0).to(device)
    out = visual_outputs(model, img, adapter=None, grad=False)
    return normalized_patch_feature(out).squeeze(0).float().cpu().numpy().astype(np.float32)


def build_manifest(kind, records):
    path = OUT / f"{kind}_TRANSPLANT_MANIFEST.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fp:
            return [r for r in csv.DictReader(fp)]
    rng = np.random.default_rng(SEED if kind == "E1_CROSS_CATEGORY" else SEED + 1000)
    normals_by_cat = defaultdict(list)
    anomalies = []
    for r in records:
        if r["anomaly"] == 0:
            normals_by_cat[r["category"]].append(r)
        elif r["mask_path"] and Path(r["mask_path"]).exists():
            anomalies.append(r)
    rows = []
    cats = sorted(normals_by_cat)
    for donor in anomalies:
        if kind == "E1_CROSS_CATEGORY":
            pool_cats = [c for c in cats if c != donor["category"] and normals_by_cat[c]]
        else:
            pool_cats = [donor["category"]] if normals_by_cat[donor["category"]] else []
        if not pool_cats:
            continue
        rec_cat = str(rng.choice(pool_cats))
        recipient = normals_by_cat[rec_cat][int(rng.integers(0, len(normals_by_cat[rec_cat])))]
        rows.append({
            "synthetic_id": len(rows),
            "kind": kind,
            "donor_idx": donor["idx"],
            "donor_category": donor["category"],
            "donor_img_path": donor["img_path"],
            "donor_mask_path": donor["mask_path"],
            "recipient_idx": recipient["idx"],
            "recipient_category": recipient["category"],
            "recipient_img_path": recipient["img_path"],
            "cross_category": donor["category"] != recipient["category"],
            "alpha": ALPHA,
            "blending": "fixed_alpha_bbox_blend",
        })
    write_csv(path, rows)
    return rows


def real_training_sample(dataset, indices, patch):
    cache_path = OUT / "cache" / "real_patch_sample.npz"
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=True)
    x, y, cats = [], [], []
    rng = np.random.default_rng(SEED)
    by_key = defaultdict(list)
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        labs = patch_labels(item)
        for cls in [0, 1]:
            ids = np.flatnonzero(labs == cls)
            key = (str(item["cls_name"]), cls)
            room = PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY - len(by_key[key])
            if room <= 0 or ids.size == 0:
                continue
            if ids.size > room:
                ids = rng.choice(ids, size=room, replace=False)
            by_key[key].extend((pos, int(pid), str(item["cls_name"]), cls) for pid in ids.tolist())
    for vals in by_key.values():
        for pos, pid, cat, cls in vals:
            x.append(patch[pos, pid].astype(np.float32))
            y.append(cls)
            cats.append(cat)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=norm_np(np.stack(x)).astype(np.float32), y=np.asarray(y, dtype=np.int64), cats=np.asarray(cats))
    return np.load(cache_path, allow_pickle=True)


def synthetic_training_sample(kind, records, model, transform, device):
    cache_path = OUT / "cache" / f"{kind}_patch_sample.npz"
    audit_path = OUT / f"{kind}_AUGMENTATION_AUDIT.csv"
    if cache_path.exists() and audit_path.exists():
        return np.load(cache_path, allow_pickle=True), build_manifest(kind, records)
    manifest = build_manifest(kind, records)
    rng = np.random.default_rng(SEED + (1 if kind == "E1_CROSS_CATEGORY" else 2))
    by_key = defaultdict(list)
    audit_rows = []
    for k, row in enumerate(manifest):
        donor = {"img_path": row["donor_img_path"], "mask_path": row["donor_mask_path"]}
        recipient = {"img_path": row["recipient_img_path"]}
        made = transplant_image(donor, recipient, rng)
        if made is None:
            audit_rows.append({**row, "status": "SKIPPED_EMPTY_OR_INVALID_MASK"})
            continue
        image_pil, mask_multi, geom = made
        patch = extract_patch(model, transform, image_pil, device)
        labs = patch_labels_from_mask(mask_multi)
        for cls in [0, 1]:
            ids = np.flatnonzero(labs == cls)
            key = (row["recipient_category"], cls)
            room = PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY - len(by_key[key])
            if room <= 0 or ids.size == 0:
                continue
            if ids.size > room:
                ids = rng.choice(ids, size=room, replace=False)
            by_key[key].extend((patch[int(pid)].astype(np.float32), cls, row["recipient_category"]) for pid in ids.tolist())
        audit_rows.append({**row, **geom, "status": "COMPLETE", "patch_anomaly_count": int(labs.sum()), "patch_normal_count": int((labs == 0).sum())})
        if (k + 1) % 100 == 0:
            print(f"[XLIX] {kind} synthetic feature extraction {k + 1}/{len(manifest)}", flush=True)
    xs, ys, cats = [], [], []
    for vals in by_key.values():
        for feat, cls, cat in vals:
            xs.append(feat)
            ys.append(cls)
            cats.append(cat)
    if not xs:
        raise SystemExit(f"{kind}_NO_SYNTHETIC_PATCHES")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=norm_np(np.stack(xs)).astype(np.float32), y=np.asarray(ys, dtype=np.int64), cats=np.asarray(cats))
    write_csv(audit_path, audit_rows)
    return np.load(cache_path, allow_pickle=True), manifest


def train_variant(variant, x_np, y_np, states, device):
    ckpt = OUT / "checkpoints" / f"{variant}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        model = AdapterToState(x_np.shape[1], states).to(device).eval()
        model.load_state_dict(payload["model"], strict=True)
        return model, payload.get("curves", [])
    torch.manual_seed(SEED)
    model = AdapterToState(x_np.shape[1], states).to(device)
    x_all = norm_np(x_np)
    y_all = y_np.astype(np.float32)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all))
        vals = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            x = torch.tensor(x_all[ids], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[ids], dtype=torch.float32, device=device)
            opt.zero_grad()
            logits, _ = model.logits(x)
            loss, bce, dice = balanced_loss(logits, y)
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(bce.cpu()), float(dice.cpu())))
        arr = np.asarray(vals, dtype=np.float64)
        row = {"variant": variant, "epoch": epoch, "loss": float(arr[:, 0].mean()), "bce": float(arr[:, 1].mean()), "dice": float(arr[:, 2].mean())}
        curves.append(row)
        print(f"[XLIX] {variant} epoch {epoch}/{EPOCHS} loss {row['loss']:.4f}", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "curves": curves}, ckpt)
    return model.eval(), curves


def evaluate_btad(adapter, device):
    dataset = ExternalDataset(DATA_ROOT / "btad" / "BTech_Dataset_transformed", "BTAD", mode="test")
    cache = np.load(XLV_A / "feature_cache" / "BTAD_clip.npz", allow_pickle=True)
    image_scores, maps = score_v1_cached(cache["patch"], adapter, device)
    return evaluate_external(dataset, [dict(r) for r in cache["rows"]], image_scores, maps)[0]


def evaluate_clinicdb(adapter, device):
    dataset = MedicalDataset(DATA_ROOT / "CVC-ClinicDB", mode="test")
    cache = np.load(XLV_B / "feature_cache" / "CVC-ClinicDB_clip.npz", allow_pickle=True)
    image_scores, maps = score_v1_cached(cache["patch"], adapter, device)
    return eval_scores(dataset, [dict(r) for r in cache["rows"]], image_scores, maps)[0]


def comparison_rows(rows):
    by = {(r["target"], r["variant"]): r for r in rows if "status" not in r}
    out = []
    for target in ["BTAD", "CVC-ClinicDB"]:
        e0 = by[(target, "E0_STATE_ALIGNED")]
        for variant in ["E1_CROSS_CATEGORY_TRANSPLANT", "E2_SAME_CATEGORY_TRANSPLANT"]:
            if (target, variant) not in by:
                continue
            e = by[(target, variant)]
            out.append({
                "target": target,
                "comparison": f"{variant}_minus_E0",
                "delta_P-AUROC": safe_float(e["P-AUROC"]) - safe_float(e0["P-AUROC"]),
                "delta_P-AP": safe_float(e["P-AP"]) - safe_float(e0["P-AP"]),
                "delta_G_pixel": gp(e) - gp(e0),
            })
        if (target, "E2_SAME_CATEGORY_TRANSPLANT") in by:
            e1 = by[(target, "E1_CROSS_CATEGORY_TRANSPLANT")]
            e2 = by[(target, "E2_SAME_CATEGORY_TRANSPLANT")]
            out.append({
                "target": target,
                "comparison": "E1_minus_E2",
                "delta_P-AUROC": safe_float(e1["P-AUROC"]) - safe_float(e2["P-AUROC"]),
                "delta_P-AP": safe_float(e1["P-AP"]) - safe_float(e2["P-AP"]),
                "delta_G_pixel": gp(e1) - gp(e2),
            })
    return out


def decide(rows, controls_run):
    by = {(r["target"], r["variant"]): r for r in rows if "status" not in r}
    e0b, e1b = by[("BTAD", "E0_STATE_ALIGNED")], by[("BTAD", "E1_CROSS_CATEGORY_TRANSPLANT")]
    e0c, e1c = by[("CVC-ClinicDB", "E0_STATE_ALIGNED")], by[("CVC-ClinicDB", "E1_CROSS_CATEGORY_TRANSPLANT")]
    btad_g = gp(e1b) - gp(e0b)
    btad_ap = safe_float(e1b["P-AP"]) - safe_float(e0b["P-AP"])
    clinic_g = gp(e1c) - gp(e0c)
    if btad_g < 0.5:
        return "XLIX_CROSS_CATEGORY_TRANSPLANT_NO_SIGNAL"
    if btad_g >= 0.5 and clinic_g <= -1.0:
        return "XLIX_TRANSPLANT_OVER_INDUSTRIALIZED_DOMAIN_SPECIFIC"
    if not (btad_g >= 1.0 and btad_ap > 0.0 and clinic_g >= -0.5):
        return "XLIX_CROSS_CATEGORY_TRANSPLANT_PARTIAL_NO_CONTROL"
    if not controls_run:
        return "XLIX_E2_CONTROL_REQUIRED_BUT_NOT_RUN"
    e2b = by[("BTAD", "E2_SAME_CATEGORY_TRANSPLANT")]
    e2c = by[("CVC-ClinicDB", "E2_SAME_CATEGORY_TRANSPLANT")]
    e1_minus_e2 = min(gp(e1b) - gp(e2b), gp(e1c) - gp(e2c))
    return "XLIX_CROSS_CATEGORY_DECOUPLING_SUPPORTED" if e1_minus_e2 > 0.5 else "XLIX_AUGMENTATION_GAIN_NOT_CROSS_CATEGORY_SPECIFIC"


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, dataset, indices, _cls, patch, _y_img, _cat_img, _rows = load_full_visa_source()
    records = raw_records(dataset, indices)
    states = np.load(XLV_A / "source_model" / "full_visa_states.npz")["patch_states"].astype(np.float32)
    real = real_training_sample(dataset, indices, patch)
    transform, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    model = load_model({"image_size": IMAGE_SIZE, "dpam_layer": 24}, device)
    write_json(OUT / "TRAINING_PROTOCOL.json", {
        "phase": "XLIX",
        "baseline": "XLV Full VisA StateAligned",
        "changed_variable": "source training data only",
        "source": "Full VisA",
        "official_multi_class_masks_used_for_source_transplant": True,
        "text": False,
        "dino": False,
        "memory": False,
        "new_loss": False,
        "adapter_changed": False,
        "target_tuning": False,
        "alpha": ALPHA,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    })

    synth_e1, manifest_e1 = synthetic_training_sample("E1_CROSS_CATEGORY", records, model, transform, device)
    x_e1 = np.concatenate([real["x"], synth_e1["x"]], axis=0)
    y_e1 = np.concatenate([real["y"], synth_e1["y"]], axis=0)
    e1_model, e1_curves = train_variant("E1_CROSS_CATEGORY_TRANSPLANT", x_e1, y_e1, states, device)
    raw = e0_rows()
    for target, fn in [("BTAD", evaluate_btad), ("CVC-ClinicDB", evaluate_clinicdb)]:
        raw.append(metric_row(target, "E1_CROSS_CATEGORY_TRANSPLANT", fn(e1_model, device)))
    by = {(r["target"], r["variant"]): r for r in raw}
    gate = {
        "BTAD_E1_minus_E0_G_pixel": gp(by[("BTAD", "E1_CROSS_CATEGORY_TRANSPLANT")]) - gp(by[("BTAD", "E0_STATE_ALIGNED")]),
        "BTAD_E1_minus_E0_P-AP": safe_float(by[("BTAD", "E1_CROSS_CATEGORY_TRANSPLANT")]["P-AP"]) - safe_float(by[("BTAD", "E0_STATE_ALIGNED")]["P-AP"]),
        "ClinicDB_E1_minus_E0_G_pixel": gp(by[("CVC-ClinicDB", "E1_CROSS_CATEGORY_TRANSPLANT")]) - gp(by[("CVC-ClinicDB", "E0_STATE_ALIGNED")]),
    }
    gate["E1_continuation_pass"] = bool(gate["BTAD_E1_minus_E0_G_pixel"] >= 1.0 and gate["BTAD_E1_minus_E0_P-AP"] > 0.0 and gate["ClinicDB_E1_minus_E0_G_pixel"] >= -0.5)
    gate["E1_no_signal_stop"] = bool(gate["BTAD_E1_minus_E0_G_pixel"] < 0.5)
    gate["E1_domain_specific_stop"] = bool(gate["BTAD_E1_minus_E0_G_pixel"] >= 0.5 and gate["ClinicDB_E1_minus_E0_G_pixel"] <= -1.0)
    write_json(OUT / "E1_GATE.json", gate)

    curves = list(e1_curves)
    controls_run = False
    models_for_geom = {"E1_CROSS_CATEGORY_TRANSPLANT": e1_model}
    if gate["E1_continuation_pass"]:
        synth_e2, _manifest_e2 = synthetic_training_sample("E2_SAME_CATEGORY", records, model, transform, device)
        x_e2 = np.concatenate([real["x"], synth_e2["x"]], axis=0)
        y_e2 = np.concatenate([real["y"], synth_e2["y"]], axis=0)
        e2_model, e2_curves = train_variant("E2_SAME_CATEGORY_TRANSPLANT", x_e2, y_e2, states, device)
        curves.extend(e2_curves)
        models_for_geom["E2_SAME_CATEGORY_TRANSPLANT"] = e2_model
        for target, fn in [("BTAD", evaluate_btad), ("CVC-ClinicDB", evaluate_clinicdb)]:
            raw.append(metric_row(target, "E2_SAME_CATEGORY_TRANSPLANT", fn(e2_model, device)))
        controls_run = True
    else:
        for target in ["BTAD", "CVC-ClinicDB"]:
            raw.append({"target": target, "variant": "E2_SAME_CATEGORY_TRANSPLANT", "status": "NOT_RUN_E1_GATE_FAILED"})

    outcome = decide(raw, controls_run)
    comparisons = comparison_rows(raw)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)
    write_csv(OUT / "CAUSAL_COMPARISON.csv", comparisons)
    geom = []
    for variant, m in models_for_geom.items():
        geom.extend(low_rank_rows({variant: m}, "XLIX", 0))
    write_csv(OUT / "LOW_RANK_UPDATE_GEOMETRY.csv", geom)
    write_csv(OUT / "SOURCE_SAMPLE_AUDIT.csv", [{
        "real_sampled_patches": int(real["x"].shape[0]),
        "E1_synthetic_manifest_images": int(len(manifest_e1)),
        "E1_synthetic_sampled_patches": int(synth_e1["x"].shape[0]),
        "E1_total_training_patches": int(x_e1.shape[0]),
        "E2_control_run": controls_run,
    }])
    write_json(OUT / "OUTCOME.json", {"phase": "XLIX", "outcome": outcome, "controls_run": controls_run, "E1_gate": gate, "runtime_sec": time.time() - start})
    report = ["# Phase XLIX Cross-Category Real Defect Transplantation", "", f"Outcome: **{outcome}**", "", "## E1 Gate", ""]
    report.append(f"- BTAD E1-E0 G: {gate['BTAD_E1_minus_E0_G_pixel']:+.4f}; P-AP: {gate['BTAD_E1_minus_E0_P-AP']:+.4f}.")
    report.append(f"- ClinicDB E1-E0 G: {gate['ClinicDB_E1_minus_E0_G_pixel']:+.4f}.")
    report.extend(["", "## Metrics", ""])
    for r in raw:
        if "status" in r:
            report.append(f"- {r['target']} {r['variant']}: {r['status']}.")
        else:
            report.append(f"- {r['target']} {r['variant']}: P-AUROC {safe_float(r['P-AUROC']):.4f}, P-AP {safe_float(r['P-AP']):.4f}, G {gp(r):.4f}.")
    report.extend([
        "",
        "Only source training data was changed. No text, DINO, Memory, new loss, adapter redesign, or target tuning was used.",
        f"Runtime: {time.time() - start:.1f}s.",
        "",
        "Stop: no blending/K/loss/model search and no further experiment was started.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
