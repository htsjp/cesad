import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_hard_normal_cross_source_mvtec import OUT as MVTEC_HN_OUT
from experiments.phase_hard_normal_cross_source_mvtec import target_cells as mvtec_source_target_cells
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, fnum, load_base, logits_from_z
from experiments.phase_standard_zsad_benchmark_infrastructure import DEFAULT_DATA_ROOT, build_registry
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np
from experiments.standard_zsad_metrics import evaluate_zsad_standard
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_median_spatial_consensus_generalization/formal")
DATA_ROOT = DEFAULT_DATA_ROOT
HOLDOUTS = {
    "MPDD": (DATA_ROOT / "MPDD", "industrial", "mpdd"),
    "DTD-Synthetic": (DATA_ROOT / "DTD-Synthetic" / "DTD-Synthetic" / "DTD-Synthetic", "texture", "dtd"),
    "CVC-ColonDB": (DATA_ROOT / "CVC-ColonDB", "medical", "medical"),
    "Kvasir": (DATA_ROOT / "Kvasir", "medical", "medical"),
}
VARIANTS = ("B0_OriginalScore", "E1_NeighborMedian")


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


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def median3x3(m):
    return ndimage.median_filter(np.asarray(m, dtype=np.float32), size=3, mode="nearest").astype(np.float32)


class MetaDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="test"):
        self.root = Path(root)
        self.preprocess, self.target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))[mode]
        self.data_all = []
        for cls in sorted(meta):
            self.data_all.extend(meta[cls])

    def __len__(self):
        return len(self.data_all)

    def resolve_mask(self, row):
        mask_path = row.get("mask_path", "")
        if mask_path and (self.root / mask_path).exists():
            return self.root / mask_path
        img_rel = Path(row.get("img_path", ""))
        if int(row.get("anomaly", 0)) and len(img_rel.parts) >= 3 and "test" in img_rel.parts:
            parts = list(img_rel.parts)
            ti = parts.index("test")
            parts[ti] = "ground_truth"
            stem = Path(parts[-1]).stem
            parts[-1] = f"{stem}_mask.png"
            cand = self.root.joinpath(*parts)
            if cand.exists():
                return cand
        return None

    def __getitem__(self, idx):
        row = self.data_all[idx]
        img = Image.open(self.root / row["img_path"]).convert("RGB")
        anomaly = int(row.get("anomaly", 0))
        resolved_mask = self.resolve_mask(row)
        if resolved_mask is not None:
            mask = Image.open(resolved_mask).convert("L")
            mask_available = True
        else:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
            mask_available = anomaly == 0
        return {
            "img": self.preprocess(img),
            "img_mask": self.target_transform(mask),
            "cls_name": str(row.get("cls_name", "unknown")),
            "anomaly": anomaly,
            "img_path": str(self.root / row["img_path"]),
            "mask_available": bool(mask_available),
        }


def audit_meta_dataset(name, root):
    meta_path = root / "meta.json"
    row = {
        "dataset": name,
        "path": str(root),
        "meta_exists": meta_path.exists(),
        "official_split_available": False,
        "image_label_available": False,
        "pixel_mask_available": False,
        "num_categories": 0,
        "num_train": 0,
        "num_test": 0,
        "num_normal_test": 0,
        "num_anomaly_test": 0,
        "mask_entries": 0,
        "valid_masks": 0,
        "image_mask_size_consistent": True,
        "status": "NOT_READY",
        "notes": "",
    }
    if not meta_path.exists():
        row["notes"] = "Missing meta.json."
        return row
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    row["official_split_available"] = "test" in meta
    cats = meta.get("test", {})
    row["num_categories"] = len(cats)
    row["num_train"] = sum(len(v) for v in meta.get("train", {}).values())
    row["num_test"] = sum(len(v) for v in cats.values())
    bad_pairs = []
    for _cat, items in cats.items():
        for x in items:
            anom = int(x.get("anomaly", 0))
            row["num_anomaly_test"] += int(anom == 1)
            row["num_normal_test"] += int(anom == 0)
            row["image_label_available"] = True
            mp = x.get("mask_path", "")
            if mp:
                row["mask_entries"] += 1
                ip = root / x["img_path"]
                maskp = root / mp
                if not maskp.exists() and int(x.get("anomaly", 0)):
                    img_rel = Path(x.get("img_path", ""))
                    if len(img_rel.parts) >= 3 and "test" in img_rel.parts:
                        parts = list(img_rel.parts)
                        parts[parts.index("test")] = "ground_truth"
                        parts[-1] = f"{Path(parts[-1]).stem}_mask.png"
                        cand = root.joinpath(*parts)
                        if cand.exists():
                            maskp = cand
                if maskp.exists() and ip.exists():
                    row["valid_masks"] += 1
                    with Image.open(ip) as im, Image.open(maskp) as mm:
                        if im.size != mm.size:
                            row["image_mask_size_consistent"] = False
                            bad_pairs.append(f"{x['img_path']}::{mp}")
                else:
                    bad_pairs.append(f"{x.get('img_path','')}::{mp}")
    row["pixel_mask_available"] = row["valid_masks"] > 0
    ready = bool(row["official_split_available"] and row["image_label_available"] and row["pixel_mask_available"] and row["valid_masks"] == row["mask_entries"] and row["image_mask_size_consistent"])
    row["status"] = "READY" if ready else "NOT_READY"
    notes = []
    if row["num_normal_test"] == 0:
        notes.append("No normal test images; image-level AUROC/AP/F1 may be NA, pixel metrics remain valid.")
    if bad_pairs:
        notes.append(f"{len(bad_pairs)} image-mask inconsistencies.")
    if not notes:
        notes.append("Pixel masks, labels, split, and image-mask size consistency validated.")
    row["notes"] = " ".join(notes)
    return row


@torch.no_grad()
def ensure_patch_cache(name, dataset, device, smoke=False):
    suffix = "_smoke" if smoke else ""
    cache_path = OUT / "feature_cache" / f"{name}_clip{suffix}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return [dict(r) for r in data["rows"]], data["patch"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model, *_ = load_components(str(CHECKPOINT), device)
    rows, patches = [], []
    n = min(24, len(dataset)) if smoke else len(dataset)
    for i in range(n):
        if i % 50 == 0:
            print(f"[MedianConsensus] extract {name} {i}/{n}", flush=True)
        item = dataset[i]
        out = visual_outputs(model, item["img"].unsqueeze(0).to(device), adapter=None, grad=False)
        patch = normalized_patch_feature(out).squeeze(0).float().cpu().numpy().astype(np.float16)
        patches.append(patch)
        rows.append({"position": i, "category": item["cls_name"], "label": int(item["anomaly"]), "mask_available": bool(item["mask_available"]), "img_path": item["img_path"]})
    np.savez_compressed(cache_path, patch=np.stack(patches).astype(np.float16), rows=np.asarray(rows, dtype=object))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data = np.load(cache_path, allow_pickle=True)
    return [dict(r) for r in data["rows"]], data["patch"]


@torch.no_grad()
def score_mvtec_source_cells(device, smoke):
    payload = np.load(MVTEC_HN_OUT / "contrast_vectors.npz")
    midpoint = payload["midpoint"].astype(np.float32)
    alpha = float(payload["alpha"])
    v = unit_np(payload["E1"].astype(np.float32))
    m_t = torch.tensor(midpoint, dtype=torch.float32, device=device)
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in mvtec_source_target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        maps = {k: np.zeros((n, SIDE, SIDE), dtype=np.float16) for k in VARIANTS}
        image = {k: np.zeros((n,), dtype=np.float32) for k in VARIANTS}
        print(f"[MedianConsensus] MVTec-source {target} n={n}", flush=True)
        for i in range(n):
            z = norm_np(np.asarray(patch[i], dtype=np.float32).reshape(SIDE * SIDE, 1024))
            logits = contrast_logits(torch.tensor(z, dtype=torch.float32, device=device), m_t, alpha, v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            med = median3x3(logits)
            for variant, smap in (("B0_OriginalScore", logits), ("E1_NeighborMedian", med)):
                maps[variant][i] = smap.astype(np.float16)
                image[variant][i] = top1_image_score(smap)
        for variant in VARIANTS:
            metric, per = evaluate_zsad_standard(cell_id, target, dataset, idx_use, kind, image[variant], maps[variant], SIDE)
            metric.update({"dataset": target, "source": "Full MVTec", "variant": variant, "evaluation_group": "cross_source_replication"})
            raw.append(metric)
            for cat, vals in per.items():
                percat.append({"dataset": target, "source": "Full MVTec", "category": cat, "variant": variant, **vals})
    return raw, percat


@torch.no_grad()
def score_visa_source_holdouts(ready_audits, device, smoke):
    q_pc1, align_model, m_np, alpha, v_np = load_base(device)
    m_t = torch.tensor(m_np, dtype=torch.float32, device=device)
    v_t = torch.tensor(v_np, dtype=torch.float32, device=device)
    raw, percat = [], []
    for audit in ready_audits:
        name = audit["dataset"]
        root, _domain, _kind = HOLDOUTS[name]
        dataset = MetaDataset(root, mode="test")
        rows, patch = ensure_patch_cache(name, dataset, device, smoke)
        n = min(24, len(rows)) if smoke else len(rows)
        maps = {k: np.zeros((n, SIDE, SIDE), dtype=np.float16) for k in VARIANTS}
        image = {k: np.zeros((n,), dtype=np.float32) for k in VARIANTS}
        print(f"[MedianConsensus] VisA-source holdout {name} n={n}", flush=True)
        for i in range(n):
            z = aligned_grid(patch[i], q_pc1, align_model, device).unsqueeze(0)
            logits = logits_from_z(z, m_t, alpha, v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            med = median3x3(logits)
            for variant, smap in (("B0_OriginalScore", logits), ("E1_NeighborMedian", med)):
                maps[variant][i] = smap.astype(np.float16)
                image[variant][i] = top1_image_score(smap)
        idx = rows[:n]
        for variant in VARIANTS:
            metric, per = evaluate_zsad_standard(f"FullVisA_to_{name}", name, dataset, idx, "external", image[variant], maps[variant], SIDE)
            metric.update({"dataset": name, "source": "Full VisA", "variant": variant, "evaluation_group": "holdout_confirmation"})
            raw.append(metric)
            for cat, vals in per.items():
                percat.append({"dataset": name, "source": "Full VisA", "category": cat, "variant": variant, **vals})
    return raw, percat


def deltas(rows):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in rows}
    out = []
    for source, dataset, _variant in sorted(by):
        if _variant != "E1_NeighborMedian":
            continue
        b0 = by[(source, dataset, "B0_OriginalScore")]
        e1 = by[(source, dataset, "E1_NeighborMedian")]
        out.append({
            "source": source,
            "dataset": dataset,
            "Delta_P-AUROC": fnum(e1, "P-AUROC") - fnum(b0, "P-AUROC"),
            "Delta_P-AP": fnum(e1, "P-AP") - fnum(b0, "P-AP"),
            "Delta_P-F1max": fnum(e1, "P-F1max") - fnum(b0, "P-F1max"),
            "Delta_PRO": fnum(e1, "PRO") - fnum(b0, "PRO"),
            "Delta_G": fnum(e1, "G_pixel") - fnum(b0, "G_pixel"),
            "AP_up": fnum(e1, "P-AP") > fnum(b0, "P-AP"),
            "PRO_up": fnum(e1, "PRO") > fnum(b0, "PRO"),
        })
    return out


def decide(cross_rows, holdout_rows, updated):
    cd = deltas(cross_rows)
    ap = np.asarray([r["Delta_P-AP"] for r in cd], dtype=np.float32)
    gg = np.asarray([r["Delta_G"] for r in cd], dtype=np.float32)
    cross_ok = int(np.sum(ap > 0)) >= 2 and float(ap.mean()) >= 1.0 and float(gg.mean()) >= 0.7
    ready_holdouts = [r for r in updated if r["dataset"] in HOLDOUTS and r["status"] == "READY"]
    hd = deltas(holdout_rows)
    if len(ready_holdouts) >= 3 and hd:
        hap = np.asarray([r["Delta_P-AP"] for r in hd], dtype=np.float32)
        hg = np.asarray([r["Delta_G"] for r in hd], dtype=np.float32)
        hap_f = hap[np.isfinite(hap)]
        hg_f = hg[np.isfinite(hg)]
        holdout_ok = hap_f.size >= 3 and int(np.sum(hap_f > 0)) >= 2 and float(np.median(hap_f)) > 0 and hg_f.size >= 3 and float(hg_f.mean()) > 0
    else:
        holdout_ok = False
    if cross_ok and holdout_ok:
        outcome = "MEDIAN_SPATIAL_CONSENSUS_CROSS_SOURCE_REPLICATED__MEDIAN_SPATIAL_CONSENSUS_HOLDOUT_CONFIRMED"
    elif cross_ok:
        outcome = "MEDIAN_SPATIAL_CONSENSUS_CROSS_SOURCE_REPLICATED"
    elif holdout_ok:
        outcome = "MEDIAN_SPATIAL_CONSENSUS_HOLDOUT_CONFIRMED_ONLY"
    else:
        outcome = "MEDIAN_SPATIAL_CONSENSUS_GENERALIZATION_NOT_CONFIRMED"
    stats = {
        "cross_source_targets": len(cd),
        "cross_source_ap_improved": int(np.sum(ap > 0)) if ap.size else 0,
        "cross_source_mean_P_AP_gain": float(ap.mean()) if ap.size else float("nan"),
        "cross_source_mean_G_gain": float(gg.mean()) if gg.size else float("nan"),
        "new_ready_targets": len(ready_holdouts),
        "holdout_targets": len(hd),
        "holdout_ap_improved": int(np.sum(np.asarray([r["Delta_P-AP"] for r in hd], dtype=np.float32) > 0)) if hd else 0,
        "holdout_median_P_AP_delta": float(np.nanmedian([r["Delta_P-AP"] for r in hd])) if hd else float("nan"),
        "holdout_mean_G_delta": float(np.nanmean([r["Delta_G"] for r in hd])) if hd else float("nan"),
        "AP_and_PRO_up": int(sum(1 for r in cd + hd if r["AP_up"] and r["PRO_up"])),
        "AP_up_PRO_down": int(sum(1 for r in cd + hd if r["AP_up"] and not r["PRO_up"])),
    }
    return outcome, stats, cd, hd


def sota_gap(all_rows):
    refs = {
        ("Full VisA", "MVTec"): ("MoECLIP CVPR 2026", 92.5, 45.7, 69.10, "DIRECT_COMPARABLE"),
        ("Full VisA", "BTAD"): ("MoECLIP CVPR 2026", 96.8, 50.4, 73.60, "DIRECT_COMPARABLE"),
        ("Full VisA", "CVC-ClinicDB"): ("MoECLIP CVPR 2026", 89.7, 49.9, 69.80, "DIRECT_COMPARABLE"),
        ("Full MVTec", "VisA"): ("MoECLIP CVPR 2026", 95.6, 26.1, 60.85, "DIRECT_COMPARABLE"),
    }
    rows = []
    for r in all_rows:
        key = (r.get("source"), r.get("dataset"))
        if key in refs:
            ref, pa, pap, g, comp = refs[key]
            rows.append({
                "source": r["source"],
                "dataset": r["dataset"],
                "variant": r["variant"],
                "reference": ref,
                "comparison_type": comp,
                "current_P-AUROC": r["P-AUROC"],
                "current_P-AP": r["P-AP"],
                "current_G": r["G_pixel"],
                "reference_P-AUROC": pa,
                "reference_P-AP": pap,
                "reference_G": g,
                "Delta_P-AUROC": fnum(r, "P-AUROC") - pa,
                "Delta_P-AP": fnum(r, "P-AP") - pap,
                "Delta_G": fnum(r, "G_pixel") - g,
            })
    return rows


def write_report(cross_rows, holdout_rows, cross_delta, holdout_delta, outcome, stats, runtime):
    lines = [
        "# Parameter-Free Median Consensus - Cross-Source and Holdout Validation",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "No training was run. The median definition is fixed 3x3 on the anomaly logit score map.",
        "",
        "## Cross-Source",
        "",
        "| Target | Delta P-AUROC | Delta P-AP | Delta P-F1max | Delta PRO | Delta G |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in cross_delta:
        lines.append(f"| {r['dataset']} | {r['Delta_P-AUROC']:+.4f} | {r['Delta_P-AP']:+.4f} | {r['Delta_P-F1max']:+.4f} | {r['Delta_PRO']:+.4f} | {r['Delta_G']:+.4f} |")
    lines.extend([
        "",
        "## Holdout",
        "",
        "| Target | Delta P-AUROC | Delta P-AP | Delta P-F1max | Delta PRO | Delta G |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in holdout_delta:
        lines.append(f"| {r['dataset']} | {r['Delta_P-AUROC']:+.4f} | {r['Delta_P-AP']:+.4f} | {r['Delta_P-F1max']:+.4f} | {r['Delta_PRO']:+.4f} | {r['Delta_G']:+.4f} |")
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Cross-source AP improved: {stats['cross_source_ap_improved']}/{stats['cross_source_targets']}",
        f"- Cross-source mean P-AP / G gain: {stats['cross_source_mean_P_AP_gain']:+.4f} / {stats['cross_source_mean_G_gain']:+.4f}",
        f"- New READY holdout targets: {stats['new_ready_targets']}",
        f"- Holdout AP improved: {stats['holdout_ap_improved']}/{stats['holdout_targets']}",
        f"- Holdout median P-AP delta / mean G delta: {stats['holdout_median_P_AP_delta']:+.4f} / {stats['holdout_mean_G_delta']:+.4f}",
        f"- AP and PRO both up: {stats['AP_and_PRO_up']}",
        f"- AP up but PRO down: {stats['AP_up_PRO_down']}",
        "",
        "Thin or small-defect sensitivity: PRO can fall when median removes isolated high-score fragments near tiny components; the AP/PRO split is preserved in the CSV rather than collapsed into G.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MedianConsensus] device={device}", flush=True)
    audits = [audit_meta_dataset(name, root) for name, (root, _domain, _kind) in HOLDOUTS.items()]
    registry = build_registry(DATA_ROOT)
    for row in registry:
        for audit in audits:
            if row["dataset"] == audit["dataset"] and audit["status"] == "READY":
                row["path"] = audit["path"]
                row["status"] = "READY"
                row["current_loader_available"] = True
                row["evaluation_level"] = "PIXEL_AND_IMAGE"
                row["num_normal"] = audit["num_normal_test"]
                row["num_anomaly"] = audit["num_anomaly_test"]
                row["num_categories"] = audit["num_categories"]
                row["normal_available"] = audit["num_normal_test"] > 0
                row["anomaly_available"] = audit["num_anomaly_test"] > 0
                row["pixel_mask_available"] = True
                row["image_label_available"] = True
                row["official_split_available"] = True
                row["notes"] = audit["notes"]
    cross_raw, cross_per = score_mvtec_source_cells(device, args.smoke)
    ready = [r for r in audits if r["status"] == "READY"]
    hold_raw, hold_per = score_visa_source_holdouts(ready, device, args.smoke)
    outcome, stats, cross_delta, hold_delta = decide(cross_raw, hold_raw, registry)
    all_rows = cross_raw + hold_raw
    runtime = time.time() - start
    write_csv(OUT / "CROSS_SOURCE_METRICS.csv", cross_raw)
    write_csv(OUT / "HOLDOUT_METRICS.csv", hold_raw)
    write_csv(OUT / "DATASET_REGISTRY_UPDATED.csv", registry)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", cross_per + hold_per)
    write_csv(OUT / "SOTA_GAP.csv", sota_gap(all_rows))
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_statistics": False,
        "target_training": False,
        "no_training": True,
        "median_kernel": "fixed_3x3",
        "holdout_audit": audits,
    })
    write_report(cross_raw, hold_raw, cross_delta, hold_delta, outcome, stats, runtime)
    print(f"[MedianConsensus] outcome={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
