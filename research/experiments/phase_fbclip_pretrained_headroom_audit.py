import argparse
import csv
import gc
import json
import sys
import time
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
import packaging
import packaging.version
import pkg_resources

ROOT = Path(__file__).resolve().parents[1]
FB_ROOT = ROOT / "_external" / "FB-CLIP"
if str(FB_ROOT) not in sys.path:
    sys.path.insert(0, str(FB_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not hasattr(pkg_resources, "packaging"):
    pkg_resources.packaging = packaging

import FBCLIP_lib
from prompt_ensemble import FBCLIP_PromptLearner

_fb_utils_spec = importlib.util.spec_from_file_location("fbclip_official_utils", FB_ROOT / "utils.py")
_fb_utils = importlib.util.module_from_spec(_fb_utils_spec)
_fb_utils_spec.loader.exec_module(_fb_utils)
fb_get_transform = _fb_utils.get_transform

from experiments.phase_final_cross_dataset_benchmark import protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import evaluate_maps, fnum, item_at, item_meta, mask_np
import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE


OUT = Path("results/phase_fbclip_pretrained_headroom_audit/formal")
WEIGHTS = {
    "Full MVTec": FB_ROOT / "official_weights" / "train_on_mvtec" / "mvtec_epoch_1_model.pth",
    "Full VisA": FB_ROOT / "official_weights" / "train_on_visa" / "visa_epoch_2_model.pth",
}
PILOT = [
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "MVTec"),
]


def parse_pairs(pair_args):
    if not pair_args:
        return PILOT
    pairs = []
    for item in pair_args:
        if "->" not in item:
            raise ValueError(f"pair must be Source->Target, got {item}")
        source, target = item.split("->", 1)
        pairs.append((source.strip(), target.strip()))
    return pairs
DATASET_TO_FB = {"MVTec": "mvtec", "VisA": "visa", "BTAD": "btad", "MPDD": "mpdd"}
E0_METRICS = ROOT / "results" / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "FULL_RES_METRICS.csv"
VARIANT = "E1_Official_FBCLIP_Pretrained"


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
        feature_map_layer=[5, 11, 17, 24],
        features_list=[5, 11, 17, 24],
        feature_layers=[1, 6, 12, 18, 24],
        image_size=IMAGE_SIZE,
        sigma=4,
        use_gaussian_filter=True,
        test_batch_size=1 if smoke else 4,
        visualize_samples=0,
        seed=111,
    )


def load_checkpoint_audit():
    rows = []
    for source, path in WEIGHTS.items():
        if not path.exists():
            rows.append({"source": source, "checkpoint": str(path), "status": "MISSING"})
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        trainable = ckpt.get("model_trainable_params", {})
        rows.append({
            "source": source,
            "checkpoint": str(path),
            "status": "READY",
            "checkpoint_source_domain": ckpt.get("source_domain", ""),
            "epoch": ckpt.get("epoch", ""),
            "loss": ckpt.get("loss", ""),
            "trainable_param_tensors": len(trainable) if isinstance(trainable, dict) else 0,
            "first_trainable_tensors": ";".join(list(trainable)[:12]) if isinstance(trainable, dict) else "",
            "backbone": "ViT-L/14@336px",
            "image_size": IMAGE_SIZE,
            "feature_layers": "1,6,12,18,24",
            "prompt": "object-agnostic: object without defect / object with defect",
            "gaussian_filter": "enabled, sigma=4, official test.sh",
            "mvfbe_fg_bg": "enabled via model.FB_params + FB_encode/Enhance_fg_bg_token",
            "background_suppression": "enabled in encode_all_image background feature residual path",
            "scr": "compute_semantic_consistency_loss returned during inference; no optimizer step in this audit",
        })
    return rows


def load_fb_model(source, device):
    args = official_args()
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = FBCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params, download_root=str(ROOT / ".cache" / "clip"))
    for p in model.parameters():
        p.requires_grad_(False)
    prompt = FBCLIP_PromptLearner(model.to("cpu"), params).to(device)
    model.to(device)
    model.FB_params(args=args, device=device)
    ckpt = torch.load(WEIGHTS[source], map_location=device, weights_only=False)
    prompt.load_state_dict(ckpt["prompt_learner"])
    loaded = 0
    trainable = ckpt.get("model_trainable_params", {})
    if isinstance(trainable, dict):
        params_by_name = dict(model.named_parameters())
        for name, value in trainable.items():
            if name in params_by_name:
                params_by_name[name].data.copy_(value.to(device=device, dtype=params_by_name[name].dtype))
                loaded += 1
    model.eval()
    prompt.eval()
    return model, prompt, args, {"checkpoint_source_domain": ckpt.get("source_domain", ""), "loaded_trainable_tensors": loaded}


@torch.no_grad()
def fb_predict_batch(model, prompt, args, images):
    prompts, tokenized, compound = prompt(cls_id=None)
    _labels, masks, _consistency = model.FB_encode(images, args=args, prompts=prompts, tokenized_prompts=tokenized, compound_prompts_text=compound)
    masks = F.interpolate(masks.detach().float().cpu(), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    out = []
    for i in range(masks.shape[0]):
        arr = masks[i, 0].numpy().astype(np.float32)
        if args.use_gaussian_filter:
            arr = gaussian_filter(arr, sigma=args.sigma).astype(np.float32)
        out.append(arr)
    return out


def load_e0_rows():
    rows = []
    if not E0_METRICS.exists():
        return rows
    for r in read_csv(E0_METRICS):
        if r.get("variant") != "E2_GlobalLocalZoomPyramid":
            continue
        if (r.get("source"), r.get("dataset")) not in set(PILOT):
            continue
        out = dict(r)
        out["variant"] = "E0_CanonicalTwoState_NTR_FixedLRP"
        rows.append(out)
    return rows


def build_maps_for_pair(source, target, model, prompt, args, device, smoke):
    preprocess, _target_transform = fb_get_transform(args)
    cells, _audits = protocol_cells(source, device, smoke)
    chosen = None
    for cell_id, tname, dataset, idx_or_rows, _patch, kind in cells:
        if tname == target:
            chosen = (cell_id, dataset, idx_or_rows, kind)
            break
    if chosen is None:
        raise RuntimeError(f"target cell missing: {source} -> {target}")
    _cell_id, dataset, idx_or_rows, kind = chosen
    n = min(24, len(idx_or_rows)) if smoke else len(idx_or_rows)
    refs = idx_or_rows[:n]
    maps = {VARIANT: []}
    metas = []
    batch, batch_meta = [], []
    for pos, ref in enumerate(refs):
        if pos % 25 == 0:
            print(f"[FBCLIP] {source} -> {target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        img = Image.open(path).convert("RGB")
        batch.append(preprocess(img))
        batch_meta.append((cat, label, path, mask_np(item)))
        if len(batch) == args.test_batch_size or pos == n - 1:
            images = torch.stack(batch, dim=0).to(device)
            preds = fb_predict_batch(model, prompt, args, images)
            for pred, meta in zip(preds, batch_meta):
                maps[VARIANT].append(pred)
                metas.append({"category": meta[0], "label": meta[1], "mask": meta[3], "img_path": meta[2]})
            batch, batch_meta = [], []
            del images
    return maps, metas


def evaluate_fb(source, target, model, prompt, args, device, smoke):
    old_variants = lzp.VARIANTS
    lzp.VARIANTS = (VARIANT,)
    try:
        maps, metas = build_maps_for_pair(source, target, model, prompt, args, device, smoke)
        raw, percat = evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old_variants
    return raw, percat


def compare_rows(e0_rows, fb_rows):
    by0 = {(r["source"], r["dataset"]): r for r in e0_rows}
    by1 = {(r["source"], r["dataset"]): r for r in fb_rows}
    rows = []
    for source, target in PILOT:
        b = by0.get((source, target))
        f = by1.get((source, target))
        if not b or not f:
            continue
        row = {"source": source, "dataset": target, "transfer": f"{source}->{target}"}
        for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
            row[f"E0_M_FR_{met}"] = fnum(b, f"M_FR_{met}")
            row[f"FBCLIP_M_FR_{met}"] = fnum(f, f"M_FR_{met}")
            row[f"Delta_M_FR_{met}"] = fnum(f, f"M_FR_{met}") - fnum(b, f"M_FR_{met}")
        rows.append(row)
    return rows


def diagnostics_from_percat(e0_percat, fb_percat):
    by0 = {(r["source"], r["dataset"], r["category"], r["protocol"]): r for r in e0_percat if r.get("variant") == "E2_GlobalLocalZoomPyramid"}
    rows = []
    for r in fb_percat:
        if r.get("variant") != VARIANT or r.get("protocol") != "M":
            continue
        b = by0.get((r["source"], r["dataset"], r["category"], "M"))
        if not b:
            continue
        rows.append({
            "source": r["source"],
            "dataset": r["dataset"],
            "category": r["category"],
            "E0_P_AP": fnum(b, "M_FR_P_AP"),
            "FBCLIP_P_AP": fnum(r, "M_FR_P_AP"),
            "Delta_P_AP": fnum(r, "M_FR_P_AP") - fnum(b, "M_FR_P_AP"),
            "E0_PRO": fnum(b, "M_FR_PRO"),
            "FBCLIP_PRO": fnum(r, "M_FR_PRO"),
            "Delta_PRO": fnum(r, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            "note": "category-level diagnostic; morphology subgroup map-level rescue requires saved E0/FB maps and is not used for tuning",
        })
    return rows


def decide(comp):
    vals = np.asarray([fnum(r, "Delta_M_FR_P_AP") for r in comp], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    improved = int(np.sum(vals > 0))
    mean = float(vals.mean()) if vals.size else float("nan")
    median = float(np.median(vals)) if vals.size else float("nan")
    non_drop = int(np.sum(vals >= -0.2))
    by = {(r["source"], r["dataset"]): fnum(r, "Delta_M_FR_P_AP") for r in comp}
    visa_gain = by.get(("Full MVTec", "VisA"), float("nan"))
    btad = by.get(("Full MVTec", "BTAD"), float("nan"))
    mpdd = by.get(("Full VisA", "MPDD"), float("nan"))
    if improved >= 3 and mean >= 2.0:
        outcome = "FBCLIP_HAS_STRONG_CROSS_DATASET_HEADROOM"
    elif improved >= 2 and mean > 0:
        outcome = "FBCLIP_FOREGROUND_BACKGROUND_MODELING_HAS_TRANSFERABLE_VALUE"
    elif np.isfinite(visa_gain) and visa_gain > 1.0 and ((np.isfinite(btad) and btad < -0.5) or (np.isfinite(mpdd) and mpdd < -0.5)):
        outcome = "FINE_GRAINED_SPECIALIZATION_VS_TRANSFER_STABILITY_TRADEOFF"
    elif np.nanmax(vals) > 1.0 if vals.size else False:
        outcome = "FBCLIP_ONLY_IMPROVES_FINE_GRAINED_TARGETS"
    else:
        outcome = "FBCLIP_DOES_NOT_OUTPERFORM_CURRENT_CORE"
    return outcome, {"mean_delta_P_AP": mean, "median_delta_P_AP": median, "improved": improved, "non_drop_ge_minus_0_2": non_drop, "evaluations": int(vals.size)}


def write_report(outcome, stats, audit, comp, runtime):
    lines = [
        "# Phase FB-CLIP Pretrained Headroom Audit",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Official repository: `Xi-Mu-Yu/FB-CLIP`. Official Drive checkpoints were used without training or target tuning. The wrapper keeps official inference settings from `test.sh`: ViT-L/14@336px, image size 518, depth 9, n_ctx 12, t_n_ctx 4, feature layers 1/6/12/18/24, Gaussian smoothing sigma 4.",
        "",
        "## Checkpoint Audit",
        "",
        "| Source | Checkpoint Domain | Epoch | Loaded Trainable Tensors | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for r in audit:
        lines.append(f"| {r['source']} | {r.get('checkpoint_source_domain','')} | {r.get('epoch','')} | {r.get('trainable_param_tensors','')} | MVFBE/foreground-background path enabled; SCR is inference-only here. |")
    lines += [
        "",
        "## Main Comparison",
        "",
        "| Transfer | E0 P-AP | FB-CLIP P-AP | Delta P-AP | E0 P-AUROC | FB P-AUROC | FB PRO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comp:
        lines.append(f"| {r['transfer']} | {fnum(r,'E0_M_FR_P_AP'):.4f} | {fnum(r,'FBCLIP_M_FR_P_AP'):.4f} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} | {fnum(r,'E0_M_FR_P_AUROC'):.4f} | {fnum(r,'FBCLIP_M_FR_P_AUROC'):.4f} | {fnum(r,'FBCLIP_M_FR_PRO'):.4f} |")
    lines += [
        "",
        "## Summary",
        "",
        f"- mean Delta P-AP: {stats['mean_delta_P_AP']:+.4f}",
        f"- median Delta P-AP: {stats['median_delta_P_AP']:+.4f}",
        f"- improved/non-drop: {stats['improved']}/{stats['evaluations']} improved, {stats['non_drop_ge_minus_0_2']}/{stats['evaluations']} non-drop >= -0.2.",
        "",
        "## Questions",
        "",
        "Q1: Answer from `Delta_M_FR_P_AP`; FB-CLIP must beat E0 after our full-resolution evaluator, not by paper numbers.",
        "Q2: If positive, category diagnostics are used to decide whether gains look like foreground/background FP suppression or FN rescue. No target tuning is performed.",
        "Q3: A VisA-only gain with BTAD/MPDD drops is labeled as the fine-grained specialization vs transfer stability tradeoff.",
        "Q4: Stable multi-target gain would support structured foreground/background representation as missing headroom.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    global PILOT
    PILOT = parse_pairs(args.pairs)
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audit = load_checkpoint_audit()
    write_csv(OUT / "CHECKPOINT_AUDIT.csv", audit)
    if any(r["status"] != "READY" for r in audit):
        write_json(OUT / "OUTCOME.json", {"outcome": "FBCLIP_CHECKPOINT_MISSING", "audit": audit})
        write_report("FBCLIP_CHECKPOINT_MISSING", {"mean_delta_P_AP": float("nan"), "median_delta_P_AP": float("nan"), "improved": 0, "non_drop_ge_minus_0_2": 0, "evaluations": 0}, audit, [], time.time() - start)
        return
    e0_rows = load_e0_rows()
    e0_percat = []
    e0_percat_path = E0_METRICS.with_name("PER_CATEGORY_FULL_RES.csv")
    if e0_percat_path.exists():
        e0_percat = read_csv(e0_percat_path)
    raw_fb, per_fb = [], []
    by_source = {}
    for source, target in PILOT:
        by_source.setdefault(source, []).append(target)
    for source, targets in by_source.items():
        print(f"[FBCLIP] load {source}", flush=True)
        model, prompt, fb_args, load_info = load_fb_model(source, device)
        if args.smoke:
            fb_args.test_batch_size = 1
        for target in targets:
            rr, cc = evaluate_fb(source, target, model, prompt, fb_args, device, args.smoke)
            raw_fb.extend(rr)
            per_fb.extend(cc)
            write_csv(OUT / "RAW_FBCLIP.partial.csv", raw_fb)
        del model, prompt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    comp = compare_rows(e0_rows, raw_fb)
    cat_diag = diagnostics_from_percat(e0_percat, per_fb)
    outcome, stats = decide(comp)
    runtime = time.time() - start
    raw = e0_rows + raw_fb
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", e0_percat + per_fb)
    write_csv(OUT / "COMPARISON.csv", comp)
    write_csv(OUT / "FAILURE_TYPE_DIAGNOSTIC.csv", cat_diag)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "stats": stats, "runtime_sec": runtime, "official_repo": str(FB_ROOT), "official_weights": {k: str(v) for k, v in WEIGHTS.items()}, "no_training": True, "no_target_tuning": True})
    write_report(outcome, stats, audit, comp, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pairs", nargs="*", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
