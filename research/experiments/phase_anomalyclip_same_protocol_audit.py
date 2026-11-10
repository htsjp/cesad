import argparse
import csv
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
import packaging
import packaging.version
import pkg_resources

ROOT = Path(__file__).resolve().parents[1]
AC_ROOT = ROOT / "_external" / "AnomalyCLIP_official"
if str(AC_ROOT) not in sys.path:
    sys.path.insert(0, str(AC_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

if not hasattr(pkg_resources, "packaging"):
    pkg_resources.packaging = packaging

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner

_ac_utils_spec = importlib.util.spec_from_file_location("anomalyclip_official_utils", AC_ROOT / "utils.py")
_ac_utils = importlib.util.module_from_spec(_ac_utils_spec)
_ac_utils_spec.loader.exec_module(_ac_utils)
ac_get_transform = _ac_utils.get_transform

if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
if str(AC_ROOT) in sys.path:
    sys.path.remove(str(AC_ROOT))
sys.path.insert(1, str(AC_ROOT))
if "utils" in sys.modules and not hasattr(sys.modules["utils"], "__path__"):
    del sys.modules["utils"]

from experiments.phase_final_cross_dataset_benchmark import protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import evaluate_maps, fnum, item_at, item_meta, mask_np
import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE


OUT = ROOT / "results" / "phase_anomalyclip_same_protocol_audit" / "formal"
VARIANT = "AnomalyCLIP_OfficialCheckpoint"
E0_VARIANT = "E0_CanonicalTwoState_NTR_FixedLRP"
E0_METRICS = ROOT / "results" / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "FULL_RES_METRICS.csv"
E0_PERCAT = ROOT / "results" / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "PER_CATEGORY_FULL_RES.csv"
WEIGHTS = {
    "Full VisA": AC_ROOT / "checkpoints" / "9_12_4_multiscale" / "epoch_15.pth",
    "Full MVTec": AC_ROOT / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth",
}
TRANSFERS = (
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
)


def parse_pairs(pair_args):
    if not pair_args:
        return TRANSFERS
    pairs = []
    for item in pair_args:
        if "->" not in item:
            raise ValueError(f"pair must be Source->Target, got {item}")
        source, target = item.split("->", 1)
        pairs.append((source.strip(), target.strip()))
    return tuple(pairs)


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def official_args(smoke=False):
    return SimpleNamespace(
        depth=9,
        n_ctx=12,
        t_n_ctx=4,
        feature_map_layer=[0],
        features_list=[24],
        image_size=IMAGE_SIZE,
        sigma=4,
        test_batch_size=1 if smoke else 4,
        metrics="image-pixel-level",
        seed=111,
    )


def checkpoint_audit():
    rows = []
    for source, path in WEIGHTS.items():
        status = "READY" if path.exists() and path.stat().st_size > 1_000_000 else "MISSING"
        rows.append(
            {
                "source": source,
                "checkpoint": str(path),
                "status": status,
                "official_repository": "zqhang/AnomalyCLIP",
                "backbone": "ViT-L/14@336px",
                "image_size": IMAGE_SIZE,
                "features_list": "24",
                "DPAM_layer": 20,
                "sigma": 4,
                "notes": "official test.sh-compatible settings; no retraining; maps evaluated by project full-resolution evaluator",
            }
        )
    return rows


def load_model(source, device, smoke=False):
    args = official_args(smoke)
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    prompt = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    ckpt = torch.load(WEIGHTS[source], map_location="cpu", weights_only=False)
    prompt.load_state_dict(ckpt["prompt_learner"])
    prompt.to(device).eval()
    model.to(device).eval()
    model.visual.DAPM_replace(DPAM_layer=20)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in prompt.parameters():
        p.requires_grad_(False)
    prompts, tokenized_prompts, compound_prompts_text = prompt(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, prompt, text_features.detach(), args


@torch.no_grad()
def predict_batch(model, text_features, args, images):
    _image_features, patch_features = model.encode_image(images, args.features_list, DPAM_layer=20)
    maps = []
    for patch_feature in patch_features:
        patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
        similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
        anomaly_map = (similarity_map[..., 1] + 1.0 - similarity_map[..., 0]) / 2.0
        maps.append(anomaly_map)
    anomaly_map = torch.stack(maps).sum(dim=0)
    out = []
    for i in anomaly_map.detach().cpu().numpy():
        out.append(gaussian_filter(i.astype(np.float32), sigma=args.sigma).astype(np.float32))
    return out


def build_maps(source, target, model, text_features, args, device, smoke):
    preprocess, _target_transform = ac_get_transform(args)
    chosen = None
    cells, _audits = protocol_cells(source, device, smoke)
    for cell_id, tname, dataset, idx_or_rows, _patch, kind in cells:
        if tname == target:
            chosen = (cell_id, dataset, idx_or_rows, kind)
            break
    if chosen is None:
        raise RuntimeError(f"missing protocol cell: {source}->{target}")
    _cell_id, dataset, refs, kind = chosen
    n = min(24, len(refs)) if smoke else len(refs)
    maps, metas = {VARIANT: []}, []
    batch, bmeta = [], []
    for pos, ref in enumerate(refs[:n]):
        if pos % 50 == 0:
            print(f"[AnomalyCLIP] {source}->{target} {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        img = Image.open(path).convert("RGB")
        batch.append(preprocess(img))
        bmeta.append((cat, label, path, mask_np(item)))
        if len(batch) == args.test_batch_size or pos == n - 1:
            preds = predict_batch(model, text_features, args, torch.stack(batch).to(device))
            for pred, meta in zip(preds, bmeta):
                maps[VARIANT].append(pred)
                metas.append({"category": meta[0], "label": meta[1], "mask": meta[3], "img_path": meta[2]})
            batch, bmeta = [], []
    return maps, metas


def evaluate_pair(source, target, model, text_features, args, device, smoke):
    old = lzp.VARIANTS
    lzp.VARIANTS = (VARIANT,)
    try:
        maps, metas = build_maps(source, target, model, text_features, args, device, smoke)
        raw, percat = evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old
    return raw, percat


def load_e0():
    rows = []
    if E0_METRICS.exists():
        for r in read_csv(E0_METRICS):
            if r.get("variant") == "E2_GlobalLocalZoomPyramid" and (r.get("source"), r.get("dataset")) in set(TRANSFERS):
                rr = dict(r)
                rr["variant"] = E0_VARIANT
                rows.append(rr)
    return rows


def compare(e0_rows, ac_rows):
    by0 = {(r["source"], r["dataset"]): r for r in e0_rows}
    by1 = {(r["source"], r["dataset"]): r for r in ac_rows}
    rows = []
    for source, target in TRANSFERS:
        b, a = by0.get((source, target)), by1.get((source, target))
        if not b or not a:
            continue
        row = {"source": source, "dataset": target, "transfer": f"{source}->{target}"}
        for met in ("P_AUROC", "P_AP", "P_F1max", "PRO"):
            row[f"E0_M_FR_{met}"] = fnum(b, f"M_FR_{met}")
            row[f"AnomalyCLIP_M_FR_{met}"] = fnum(a, f"M_FR_{met}")
            row[f"Delta_M_FR_{met}"] = fnum(a, f"M_FR_{met}") - fnum(b, f"M_FR_{met}")
        rows.append(row)
    return rows


def write_report(comp, audit, runtime):
    vals = np.asarray([fnum(r, "Delta_M_FR_P_AP") for r in comp], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    lines = [
        "# Phase AnomalyCLIP Same-Protocol Audit",
        "",
        "Official repository: `zqhang/AnomalyCLIP`. Existing official-style checkpoints were used without retraining. Raw anomaly maps were evaluated by the project full-resolution evaluator.",
        "",
        "## Checkpoint Audit",
        "",
        "| Source | Status | Checkpoint |",
        "|---|---|---|",
    ]
    for r in audit:
        lines.append(f"| {r['source']} | {r['status']} | `{r['checkpoint']}` |")
    lines += ["", "## Strict Comparison", "", "| Transfer | E0 P-AP | AnomalyCLIP P-AP | Delta | P-AUROC | PRO |", "|---|---:|---:|---:|---:|---:|"]
    for r in comp:
        lines.append(f"| {r['transfer']} | {fnum(r,'E0_M_FR_P_AP'):.4f} | {fnum(r,'AnomalyCLIP_M_FR_P_AP'):.4f} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} | {fnum(r,'AnomalyCLIP_M_FR_P_AUROC'):.4f} | {fnum(r,'AnomalyCLIP_M_FR_PRO'):.4f} |")
    lines += [
        "",
        f"- Mean Delta P-AP: `{float(vals.mean()) if vals.size else float('nan'):+.4f}`.",
        f"- Median Delta P-AP: `{float(np.median(vals)) if vals.size else float('nan'):+.4f}`.",
        f"- Improved: `{int(np.sum(vals > 0))}/{int(vals.size)}`.",
        f"- Runtime: `{runtime:.1f}s`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    global TRANSFERS
    TRANSFERS = parse_pairs(args.pairs)
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    audit = checkpoint_audit()
    write_csv(OUT / "CHECKPOINT_AUDIT.csv", audit)
    ready_sources = {r["source"] for r in audit if r["status"] == "READY"}
    e0_rows = load_e0()
    partial = OUT / "RAW_ANOMALYCLIP.partial.csv"
    per_partial = OUT / "PER_CATEGORY_STANDARD_METRICS.partial.csv"
    ac_rows = read_csv(partial) if partial.exists() and not args.force else []
    percat = read_csv(per_partial) if per_partial.exists() and not args.force else []
    done = {(r.get("source"), r.get("dataset")) for r in ac_rows if r.get("variant") == VARIANT}
    for source in ("Full MVTec", "Full VisA"):
        if source not in ready_sources:
            continue
        model, prompt, text_features, ac_args = load_model(source, device, smoke=args.smoke)
        for s, target in TRANSFERS:
            if s != source:
                continue
            if (source, target) in done and not args.force:
                print(f"[AnomalyCLIP] skip completed {source}->{target}", flush=True)
                continue
            try:
                rr, cc = evaluate_pair(source, target, model, text_features, ac_args, device, args.smoke)
            except RuntimeError as exc:
                print(f"[AnomalyCLIP] skip unsupported {source}->{target}: {exc}", flush=True)
                continue
            ac_rows.extend(rr)
            percat.extend(cc)
            write_csv(partial, ac_rows)
            write_csv(per_partial, percat)
        del model, prompt, text_features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    comp = compare(e0_rows, ac_rows)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", e0_rows + ac_rows)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat)
    write_csv(OUT / "COMPARISON.csv", comp)
    vals = np.asarray([fnum(r, "Delta_M_FR_P_AP") for r in comp], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    write_json(OUT / "OUTCOME.json", {"outcome": "ANOMALYCLIP_SAME_PROTOCOL_COMPLETE", "mean_delta_pap": float(vals.mean()) if vals.size else float("nan"), "median_delta_pap": float(np.median(vals)) if vals.size else float("nan"), "improved": int(np.sum(vals > 0)), "total": int(vals.size), "runtime_sec": runtime})
    write_report(comp, audit, runtime)
    print(f"[AnomalyCLIP] complete n={len(comp)} mean_delta={float(vals.mean()) if vals.size else float('nan'):.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pairs", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
