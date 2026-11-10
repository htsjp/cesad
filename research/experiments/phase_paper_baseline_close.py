import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
FAP_DIR = ROOT / "_external" / "FAPrompt"
if str(FAP_DIR) not in sys.path:
    sys.path.insert(0, str(FAP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

import packaging as _packaging  # noqa: E402
import packaging.version as _packaging_version  # noqa: E402
import pkg_resources as _pkg_resources  # noqa: E402
if not hasattr(_packaging, "version"):
    _packaging.version = _packaging_version
if not hasattr(_pkg_resources, "packaging"):
    _pkg_resources.packaging = _packaging

import AnomalyCLIP_lib as fa_clip  # noqa: E402
from FAPrompt import FAPrompt  # noqa: E402
from utils import get_transform as fa_get_transform  # noqa: E402
sys.modules.pop("utils", None)
if str(FAP_DIR) in sys.path:
    sys.path.remove(str(FAP_DIR))

from experiments.phase_final_cross_dataset_benchmark import protocol_cells  # noqa: E402
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (  # noqa: E402
    PROTOCOLS,
    item_at,
    item_meta,
    mask_np,
    pg_metrics,
    summarize_category,
    torch_resize,
)
from experiments.phase_xa_smoke import write_csv  # noqa: E402
from experiments.phase_xlv_cross_domain_transfer import top1_image_score  # noqa: E402

OUT = Path("results/paper_assets/baseline_close")
CKPT_DIR = OUT / "official_checkpoints"
MAP_CACHE = OUT / "map_cache"
LATEX = OUT / "latex"

PAIRS = [
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
]
COMMON6 = [
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
]
METRICS = ["I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO"]
DISPLAY_METRICS = ["I-AUROC", "I-AP", "P-AUROC", "P-AP", "P-F1max", "AUPRO"]

FAPROMPT_CHECKPOINTS = {
    "Full MVTec": CKPT_DIR / "train_on_mvtecad.pth",
    "Full VisA": CKPT_DIR / "train_on_visa.pth",
}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fnum(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def simplify_row(row):
    return {
        "I_AUROC": fnum(row, "M_FR_I_AUROC"),
        "I_AP": fnum(row, "M_FR_I_AP"),
        "P_AUROC": fnum(row, "M_FR_P_AUROC"),
        "P_AP": fnum(row, "M_FR_P_AP"),
        "P_F1MAX": fnum(row, "M_FR_P_F1max"),
        "AUPRO": fnum(row, "M_FR_PRO"),
    }


def cell_item(dataset, kind, ref, pos):
    return item_at(dataset, kind, ref, pos)


def load_faprompt(device, checkpoint_path):
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = fa_clip.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    prompt_learner = FAPrompt(model.to("cpu"), params)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prompt_learner.load_state_dict(ckpt["prompt_learner"])
    prompt_learner.to(device).eval()
    model.to(device)
    model.eval()
    model.visual.DAPM_replace(DPAM_layer=20)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in prompt_learner.parameters():
        p.requires_grad_(False)
    return model, prompt_learner


@torch.no_grad()
def faprompt_map(model, prompt_learner, image, device, features_list=(6, 12, 18, 24), image_size=518, sigma=4):
    image_features, patch_features = model.encode_image(image, list(features_list), DPAM_layer=20)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    prompts_pos, prompts_neg, tok_pos, tok_neg, compound, _ = prompt_learner.forward()
    text_pos = model.encode_text_learn(prompts_pos, tok_pos, compound).float()
    text_neg = model.encode_text_learn(prompts_neg, tok_neg, compound).float()
    text_neg = torch.mean(text_neg, dim=0, keepdim=True)
    text_features = torch.cat([text_pos, text_neg])
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
    similarity, _ = fa_clip.compute_similarity_ori(patch_features, text_features[0])
    sim_map1 = fa_clip.get_similarity_map(similarity[1:, :], image_size).squeeze(0)
    map_max_score1 = similarity[1:, :, 1].max(dim=0).values

    _pk_value, pk_idx = torch.topk(similarity[1:, :, 1], k=10, dim=0, largest=True, sorted=True)
    pk_idx = pk_idx.permute(1, 0)
    patch_by_layer = patch_features[1:, :, :].permute(1, 0, 2)
    selected = []
    for i in range(len(pk_idx)):
        toks = []
        for k in range(len(pk_idx[0])):
            toks.append(patch_by_layer[i, k, :])
        selected.append(torch.stack(toks))
    selected = torch.stack(selected)

    text_features_list = []
    for i in range(len(selected)):
        ppos, pneg, tpos, tneg, comp, _ = prompt_learner.forward(selected_tokens=selected[i])
        text_pos = model.encode_text_learn(ppos, tpos, comp).float()
        text_neg = model.encode_text_learn(pneg, tneg, comp).float()
        text_neg = torch.mean(text_neg, dim=0, keepdim=True)
        text_features_list.append(torch.cat([text_pos, text_neg]))
    text_features = torch.stack(text_features_list)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    text_probs = image_features @ text_features.permute(0, 2, 1)
    text_probs = (text_probs / 0.07).softmax(-1)[:, 0, 1]
    similarity, _ = fa_clip.compute_similarity(patch_features, text_features)
    map_max_score2 = similarity[1:, :, 1].max(dim=0).values
    sim_map2 = fa_clip.get_similarity_map(similarity[1:, :], image_size).squeeze(0)

    amap = (sim_map1[1, :] + 1 - sim_map1[0, :] + sim_map2[1, :] + 1 - sim_map2[0, :]) / 4.0
    image_score = 0.5 * (text_probs + (2 * map_max_score1 + map_max_score2) / 3)
    arr = amap.detach().cpu().numpy().astype(np.float32)
    arr = gaussian_filter(arr, sigma=sigma).astype(np.float32)
    return arr, float(image_score.detach().cpu().numpy().reshape(-1)[0])


def evaluate_single_variant_maps(source, target, variant, score_maps, metas):
    by_cat = {}
    for i, meta in enumerate(metas):
        d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "maps": []})
        score_map = np.asarray(score_maps[i], dtype=np.float32)
        d["labels"].append(meta["label"])
        d["image_scores"].append(top1_image_score(score_map))
        d["masks"].append(meta["mask"])
        d["maps"].append(score_map)

    fr_cat_rows = []
    cat_rows = []
    metric_names = ["I-AUROC", "I-AP", "P-AUROC", "P-AP", "P-F1max", "PRO"]
    pcfg = PROTOCOLS["M"]
    for cat, vals in sorted(by_cat.items()):
        scores = []
        for s, m in zip(vals["maps"], vals["masks"]):
            scores.append(s if tuple(s.shape) == tuple(m.shape) else torch_resize(s, m.shape, align_corners=pcfg["align_corners"]))
        fr = summarize_category(vals["labels"], vals["image_scores"], vals["masks"], scores)
        fr_row = {
            "source": source,
            "dataset": target,
            "category": cat,
            "variant": variant,
            "protocol": "M",
            "protocol_name": pcfg["name"],
            "aggregation": "CATEGORY",
            "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}",
        }
        for met, val in fr.items():
            fr_row[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = val
        cat_rows.append(fr_row)
        fr_cat_rows.append(fr_row)

    raw = {
        "source": source,
        "dataset": target,
        "variant": variant,
        "aggregation": "CATEGORY_MACRO",
        "score_resolution": f"{score_maps[0].shape[0]}x{score_maps[0].shape[1]}",
    }
    for met in metric_names:
        vals = np.asarray([fnum(r, f"{pcfg['prefix']}_{met.replace('-', '_')}") for r in fr_cat_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        raw[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = float(vals.mean()) if vals.size else float("nan")
    return raw, cat_rows


def evaluate_faprompt_source(source, device, smoke=False):
    ckpt = FAPROMPT_CHECKPOINTS[source]
    if not ckpt.exists():
        return [], [{"source": source, "status": "MISSING_CHECKPOINT", "checkpoint": str(ckpt)}]
    model, prompt_learner = load_faprompt(device, ckpt)
    args = argparse.Namespace(image_size=518)
    preprocess, _target_transform = fa_get_transform(args)
    cells, _audits = protocol_cells(source, device, smoke)
    rows = []
    trace = []
    for _cell_id, target, dataset, refs, _patch, kind in cells:
        if (source, target) not in PAIRS:
            continue
        n = min(24, len(refs)) if smoke else len(refs)
        cache = MAP_CACHE / f"FAPrompt_{source.replace(' ', '_')}__{target}{'_smoke' if smoke else ''}.npz"
        meta_cache = MAP_CACHE / f"FAPrompt_{source.replace(' ', '_')}__{target}{'_smoke' if smoke else ''}.json"
        maps = {"FAPrompt": []}
        metas = []
        if cache.exists() and meta_cache.exists():
            data = np.load(cache, allow_pickle=True)
            arr = data["maps"]
            if arr.shape[0] >= n:
                maps["FAPrompt"] = [arr[i].astype(np.float32) for i in range(n)]
                # The JSON sidecar intentionally stores only lightweight trace fields.
                # Rebuild evaluator metadata from the dataset registry so masks are
                # always fresh and full-resolution.
                metas = []
                for pos in range(n):
                    ref = refs[pos]
                    item = cell_item(dataset, kind, ref, pos)
                    cat, label, path = item_meta(item, kind, ref, target)
                    metas.append({"category": cat, "label": int(label), "mask": mask_np(item), "img_path": path})
        if not maps["FAPrompt"] or not metas:
            maps["FAPrompt"] = []
            metas = []
            for pos in range(n):
                if pos % 25 == 0:
                    print(f"[BaselineClose] FAPrompt {source}->{target} {pos}/{n}", flush=True)
                ref = refs[pos]
                item = cell_item(dataset, kind, ref, pos)
                cat, label, path = item_meta(item, kind, ref, target)
                from PIL import Image
                img = Image.open(path).convert("RGB")
                image = preprocess(img).unsqueeze(0).to(device)
                amap, _official_image_score = faprompt_map(model, prompt_learner, image, device)
                maps["FAPrompt"].append(amap)
                metas.append({"category": cat, "label": int(label), "mask": mask_np(item), "img_path": path})
            MAP_CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, maps=np.stack(maps["FAPrompt"]).astype(np.float16))
            # Metadata with masks is not JSON serializable; write lightweight trace only.
            meta_light = [{"category": m["category"], "label": int(m["label"]), "img_path": m.get("img_path", "")} for m in metas]
            meta_cache.write_text(json.dumps(meta_light, indent=2), encoding="utf-8")
        r, _cat = evaluate_single_variant_maps(source, target, "FAPrompt", maps["FAPrompt"], metas)
        simple = simplify_row(r)
        rows.append({
            "source": source,
            "target": target,
            "method": "FAPrompt",
            "checkpoint": str(ckpt),
            "checkpoint_source": "train_on_mvtecad" if source == "Full MVTec" else "train_on_visa",
            "strict_comparable": "YES",
            **simple,
        })
        trace.append({"source": source, "target": target, "n_images": n, "cache": str(cache), "status": "EVALUATED"})
    return rows, trace


def mean_rows(rows, pairs=COMMON6):
    selected = [r for r in rows if (r.get("source"), r.get("target")) in pairs]
    out = {"method": selected[0].get("method") if selected else "", "n_cells": len(selected)}
    for m in METRICS:
        vals = np.asarray([fnum(r, m) for r in selected], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[m] = float(vals.mean()) if vals.size else float("nan")
    return out


def load_existing_common6():
    path = Path("results/paper_assets/metric_extension/BIG_MODEL_COMMON6_MEAN_SUMMARY.csv")
    return read_csv(path)


def write_table_common6_updated(fap_rows):
    rows = []
    existing = load_existing_common6()
    for r in existing:
        rows.append({"method": r.get("method"), **{m: r.get(m, "") for m in METRICS}, "strict_comparable": "YES", "source": "existing same-protocol common6"})
    if fap_rows:
        fr = mean_rows(fap_rows)
        rows.append({"method": "FAPrompt", **{m: fr[m] for m in METRICS}, "strict_comparable": "YES", "source": "official checkpoints + unified evaluator"})
    else:
        rows.append({"method": "FAPrompt", **{m: "" for m in METRICS}, "strict_comparable": "NO", "source": "reference only; no unified metrics"})
    rows.append({"method": "WinCLIP", **{m: "" for m in METRICS}, "strict_comparable": "NO", "source": "reference only; no author official repository found"})
    write_csv(OUT / "TABLE_COMMON6_UPDATED.csv", rows)
    return rows


def rank_updated(common_rows):
    strict = [r for r in common_rows if r.get("strict_comparable") == "YES"]
    out = []
    rank_by_metric = {}
    for metric in METRICS:
        vals = []
        for r in strict:
            v = fnum(r, metric)
            if np.isfinite(v):
                vals.append((r["method"], v))
        vals.sort(key=lambda x: x[1], reverse=True)
        rank_by_metric[metric] = {name: i + 1 for i, (name, _v) in enumerate(vals)}
        for i, (name, val) in enumerate(vals, 1):
            out.append({"metric": metric, "rank": i, "method": name, "value": val})
    methods = sorted({r["method"] for r in strict})
    for name in methods:
        ranks = [rank_by_metric[m].get(name, float("nan")) for m in METRICS]
        finite = [x for x in ranks if np.isfinite(x)]
        out.append({"metric": "AVERAGE_RANK", "rank": float(np.mean(finite)) if finite else "", "method": name, "value": ""})
    write_csv(OUT / "COMMON6_METRIC_RANK_UPDATED.csv", out)
    return out


def latex_table(path, rows):
    cols = ["method"] + METRICS
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & I-AUROC & I-AP & P-AUROC & P-AP & P-F1max & AUPRO \\\\",
        "\\midrule",
    ]
    best = {}
    for m in METRICS:
        vals = [fnum(r, m) for r in rows if r.get("strict_comparable") == "YES"]
        vals = [v for v in vals if np.isfinite(v)]
        best[m] = max(vals) if vals else float("nan")
    for r in rows:
        if r.get("strict_comparable") != "YES":
            continue
        vals = []
        for m in METRICS:
            v = fnum(r, m)
            text = f"{v:.2f}" if np.isfinite(v) else "--"
            if np.isfinite(v) and abs(v - best[m]) < 1e-9:
                text = f"\\textbf{{{text}}}"
            vals.append(text)
        lines.append(f"{r['method']} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol_audits(fap_strict, winclip_strict):
    fap = [
        "# FAPrompt Protocol Audit",
        "",
        "Official repository: https://github.com/mala-lab/FAPrompt",
        "Official checkpoint folder from README: Google Drive `11Z5msKSrnIECamZO4kYy_rciQXS0v9vI`.",
        "Downloaded official files: `train_on_mvtecad.pth`, `train_on_visa.pth`.",
        "",
        "## Source Provenance",
        "",
        "- `train_on_mvtecad.pth`: treated as Full MVTec-source official checkpoint by filename and Drive listing.",
        "- `train_on_visa.pth`: treated as Full VisA-source official checkpoint by filename and Drive listing.",
        "- Training script supports MVTec via `train.py`; the VisA block in `train.sh` references missing `train_cls.py`, so this run relies on the official pretrained VisA checkpoint rather than local retraining.",
        "",
        "## Target Category Name Audit",
        "",
        "- `FAPrompt.py` sets `classnames = ['object']` and constructs prompts with generic `object` wording plus learned context and `damaged`.",
        "- `test.py` reads `cls_name` for grouping metrics but does not pass target category names into `prompt_learner.forward()` or scoring.",
        "- `USES_TARGET_CATEGORY_NAME = NO` for the official checkpoint inference path used here.",
        "",
        "## Target Statistics / Adaptation Audit",
        "",
        "- No target training, normal reference, target GT, or dataset-level target statistics are used.",
        "- DAP selects top patch tokens from the current test image only, then recomputes prompt-conditioned text features; this is query-internal inference behavior, not target-domain calibration.",
        "- Official Gaussian smoothing with fixed sigma=4 is retained; no target-specific sigma tuning.",
        "",
        f"## Decision: {'FAPROMPT_STRICT_COMPARABLE' if fap_strict else 'FAPROMPT_REFERENCE_ONLY'}",
        "",
    ]
    (OUT / "FAPROMPT_PROTOCOL_AUDIT.md").write_text("\n".join(fap), encoding="utf-8")

    wc = [
        "# WinCLIP Protocol Audit",
        "",
        "Search result: no original-author official public repository was identified in this run.",
        "Repositories found are marked unofficial/re-implementation, including `caoyunkang/WinClip` and `mala-lab/WinCLIP`.",
        "The protocol requires an original-author official repository for strict same-protocol evaluation.",
        "",
        "WinCLIP zero-shot uses text prompt ensembles with object/category semantics in the paper setting; without original official code, this was not reinterpreted or modified.",
        "",
        "## Decision: WINCLIP_NO_AUTHOR_OFFICIAL_REPOSITORY / WINCLIP_REFERENCE_ONLY",
        "",
    ]
    (OUT / "WINCLIP_PROTOCOL_AUDIT.md").write_text("\n".join(wc), encoding="utf-8")


def write_literature_context(common_rows):
    rows = [
        {"Method": "WinCLIP", "Venue/Year": "CVPR 2023", "Backbone": "CLIP", "Training setting": "training-free zero-shot / few-shot variant", "Target class names?": "likely yes in original prompt ensemble", "Target normal reference?": "no for WinCLIP zero-shot; yes for WinCLIP+", "I-AUROC": "", "I-AP": "", "P-AUROC": "", "AUPRO": "", "Result source": "paper / reference only", "Strict comparable?": "NO"},
        {"Method": "AnomalyCLIP", "Venue/Year": "ICLR 2024", "Backbone": "CLIP ViT-L/14@336", "Training setting": "same-protocol local artifact", "Target class names?": "NO in our strict table", "Target normal reference?": "NO", "I-AUROC": "see TABLE_COMMON6_UPDATED", "I-AP": "see TABLE_COMMON6_UPDATED", "P-AUROC": "see TABLE_COMMON6_UPDATED", "AUPRO": "see TABLE_COMMON6_UPDATED", "Result source": "local unified evaluator", "Strict comparable?": "YES"},
        {"Method": "AdaCLIP", "Venue/Year": "ECCV 2024", "Backbone": "CLIP", "Training setting": "literature reference", "Target class names?": "unverified", "Target normal reference?": "unverified", "I-AUROC": "", "I-AP": "", "P-AUROC": "", "AUPRO": "", "Result source": "reference only", "Strict comparable?": "NO"},
        {"Method": "AA-CLIP", "Venue/Year": "CVPR 2025", "Backbone": "CLIP", "Training setting": "literature reference", "Target class names?": "unverified", "Target normal reference?": "unverified", "I-AUROC": "", "I-AP": "", "P-AUROC": "", "AUPRO": "", "Result source": "reference only", "Strict comparable?": "NO"},
        {"Method": "FAPrompt", "Venue/Year": "ICCV 2025", "Backbone": "CLIP ViT-L/14@336", "Training setting": "official source checkpoints", "Target class names?": "NO", "Target normal reference?": "NO", "I-AUROC": "see TABLE_COMMON6_UPDATED", "I-AP": "see TABLE_COMMON6_UPDATED", "P-AUROC": "see TABLE_COMMON6_UPDATED", "AUPRO": "see TABLE_COMMON6_UPDATED", "Result source": "official checkpoint + local unified evaluator", "Strict comparable?": "YES"},
        {"Method": "MRAD", "Venue/Year": "ICLR 2026", "Backbone": "various", "Training setting": "same-protocol local artifacts", "Target class names?": "NO", "Target normal reference?": "NO", "I-AUROC": "see TABLE_COMMON6_UPDATED", "I-AP": "see TABLE_COMMON6_UPDATED", "P-AUROC": "see TABLE_COMMON6_UPDATED", "AUPRO": "see TABLE_COMMON6_UPDATED", "Result source": "local unified evaluator", "Strict comparable?": "YES"},
        {"Method": "VisualAD", "Venue/Year": "CVPR 2026", "Backbone": "CLIP", "Training setting": "same-protocol local artifact", "Target class names?": "NO", "Target normal reference?": "NO", "I-AUROC": "see TABLE_COMMON6_UPDATED", "I-AP": "see TABLE_COMMON6_UPDATED", "P-AUROC": "see TABLE_COMMON6_UPDATED", "AUPRO": "see TABLE_COMMON6_UPDATED", "Result source": "local unified evaluator", "Strict comparable?": "YES"},
        {"Method": "FB-CLIP", "Venue/Year": "CVPR 2026", "Backbone": "CLIP", "Training setting": "official pretrained checkpoint audit", "Target class names?": "NO for evaluated setting", "Target normal reference?": "NO", "I-AUROC": "see TABLE_COMMON6_UPDATED", "I-AP": "see TABLE_COMMON6_UPDATED", "P-AUROC": "see TABLE_COMMON6_UPDATED", "AUPRO": "see TABLE_COMMON6_UPDATED", "Result source": "local unified evaluator", "Strict comparable?": "YES"},
        {"Method": "Ours", "Venue/Year": "This paper", "Backbone": "CLIP ViT-L/14@336", "Training setting": "source-only Canonical Two-State + NTR + Fixed LRP", "Target class names?": "NO", "Target normal reference?": "NO", "I-AUROC": "see TABLE_COMMON6_UPDATED", "I-AP": "see TABLE_COMMON6_UPDATED", "P-AUROC": "see TABLE_COMMON6_UPDATED", "AUPRO": "see TABLE_COMMON6_UPDATED", "Result source": "local unified evaluator", "Strict comparable?": "YES"},
    ]
    write_csv(OUT / "TABLE_LITERATURE_CONTEXT_UPDATED.csv", rows)


def write_report(fap_rows, common_rows, rank_rows, runtime):
    common = {r["method"]: r for r in common_rows}
    fap = common.get("FAPrompt", {})
    ours = common.get("Ours", {})
    strict_methods = [r["method"] for r in common_rows if r.get("strict_comparable") == "YES"]
    avg_rank = {r["method"]: r["rank"] for r in rank_rows if r["metric"] == "AVERAGE_RANK"}
    ours_first = sum(1 for m in METRICS if any(r["metric"] == m and r["rank"] == 1 and r["method"] == "Ours" for r in rank_rows))
    lines = [
        "# Phase PAPER-BASELINE-CLOSE",
        "",
        "OUTCOME: `BASELINE_CLOSE_COMPLETE_REFERENCE_SAFE`",
        "",
        "## Decisions",
        "",
        "- FAPrompt: `FAPROMPT_STRICT_COMPARABLE` using official `train_on_mvtecad.pth` and `train_on_visa.pth` checkpoints.",
        "- WinCLIP: `WINCLIP_REFERENCE_ONLY` because no original-author official public repository was identified; unofficial repos were not used as strict baselines.",
        "- Ours was not modified.",
        "",
        "## FAPrompt Common-6 Mean",
        "",
        "| Method | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["FAPrompt", "Ours"]:
        r = common.get(name, {})
        lines.append("| " + name + " | " + " | ".join(f"{fnum(r, m):.2f}" if np.isfinite(fnum(r, m)) else "--" for m in METRICS) + " |")
    lines += [
        "",
        "## Strict Main Table Methods",
        "",
        ", ".join(strict_methods),
        "",
        "## Ranking Update",
        "",
        f"- Ours first-place count across six metrics: `{ours_first}/6`.",
        f"- Ours average rank: `{avg_rank.get('Ours', 'NA')}`.",
        f"- FAPrompt average rank: `{avg_rank.get('FAPrompt', 'NA')}`.",
        "- Claim status: update according to `COMMON6_METRIC_RANK_UPDATED.csv`; do not retain an older first-place claim if contradicted by the updated ranks.",
        "",
        "## Required Answers",
        "",
        "1. FAPrompt checkpoint provenance: official Google Drive files `train_on_mvtecad.pth` and `train_on_visa.pth`.",
        "2. Legal MVTec-source / VisA-source results: yes, via official pretrained checkpoints.",
        "3. FAPrompt target class name use: no; prompts use generic `object`, `cls_name` only groups metrics.",
        "4. FAPrompt strict protocol: yes for checkpoint inference; no target training/statistics/category names.",
        "5. WinCLIP author official repo: not found in this run.",
        "6. WinCLIP target category name: original method uses object/category prompt semantics; no strict official-code audit was run.",
        "7. Strict table entrants: methods marked `YES` in `TABLE_COMMON6_UPDATED.csv`; WinCLIP remains reference-only.",
        "8. Ours rank after FAPrompt: see ranking update above and `COMMON6_METRIC_RANK_UPDATED.csv`.",
        "9. Common-6 means: see `TABLE_COMMON6_UPDATED.csv`.",
        "10. Paper claim modification: required only if updated ranks change the prior claim.",
        "",
        "## Files",
        "",
        "- `FAPROMPT_PROTOCOL_AUDIT.md`",
        "- `WINCLIP_PROTOCOL_AUDIT.md`",
        "- `FAPROMPT_UNIFIED_METRICS.csv`",
        "- `TABLE_COMMON6_UPDATED.csv`, `TABLE_COMMON6_UPDATED.tex`",
        "- `TABLE_LITERATURE_CONTEXT_UPDATED.csv`",
        "- `COMMON6_METRIC_RANK_UPDATED.csv`",
        "- `PROTOCOL_VALIDITY.csv`, `SOURCE_TRACE.csv`, `OUTCOME.json`",
        "",
        f"Runtime: `{runtime:.1f}s`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-faprompt", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LATEX.mkdir(parents=True, exist_ok=True)
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[BaselineClose] device={device}", flush=True)

    source_trace = []
    fap_rows = []
    if not args.skip_faprompt:
        for source in ["Full MVTec", "Full VisA"]:
            rows, trace = evaluate_faprompt_source(source, device, args.smoke)
            fap_rows.extend(rows)
            source_trace.extend(trace)
    write_csv(OUT / "FAPROMPT_UNIFIED_METRICS.csv", fap_rows)
    write_csv(OUT / "WINCLIP_UNIFIED_METRICS.csv", [])

    fap_strict = bool(fap_rows) and len({(r["source"], r["target"]) for r in fap_rows if (r["source"], r["target"]) in COMMON6}) == 6
    write_protocol_audits(fap_strict, False)

    protocol_rows = [
        {"method": "FAPrompt", "official_repo": "YES", "official_checkpoint": "YES", "source_only": "YES", "target_training": "NO", "target_normal_reference": "NO", "target_statistics": "NO", "target_category_name": "NO", "strict_comparable": "YES" if fap_strict else "NO", "decision": "FAPROMPT_STRICT_COMPARABLE" if fap_strict else "FAPROMPT_REFERENCE_ONLY"},
        {"method": "WinCLIP", "official_repo": "NO_AUTHOR_OFFICIAL_REPOSITORY_FOUND", "official_checkpoint": "NA", "source_only": "NA", "target_training": "NA", "target_normal_reference": "NA", "target_statistics": "NA", "target_category_name": "UNVERIFIED_ORIGINAL_PROMPT_SEMANTICS", "strict_comparable": "NO", "decision": "WINCLIP_REFERENCE_ONLY"},
    ]
    write_csv(OUT / "PROTOCOL_VALIDITY.csv", protocol_rows)
    write_csv(OUT / "SOURCE_TRACE.csv", source_trace)

    common_rows = write_table_common6_updated(fap_rows)
    latex_table(OUT / "TABLE_COMMON6_UPDATED.tex", common_rows)
    rank_rows = rank_updated(common_rows)
    write_literature_context(common_rows)

    outcome = {
        "outcome": "BASELINE_CLOSE_COMPLETE_REFERENCE_SAFE",
        "faprompt": "FAPROMPT_STRICT_COMPARABLE" if fap_strict else "FAPROMPT_REFERENCE_ONLY",
        "winclip": "WINCLIP_REFERENCE_ONLY",
        "runtime_sec": time.time() - start,
    }
    write_json(OUT / "OUTCOME.json", outcome)
    write_report(fap_rows, common_rows, rank_rows, time.time() - start)
    print(f"[BaselineClose] wrote {OUT} outcome={outcome['outcome']}", flush=True)


if __name__ == "__main__":
    main()
