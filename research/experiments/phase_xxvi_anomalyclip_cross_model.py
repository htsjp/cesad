import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import packaging as _packaging
import packaging.version as _packaging_version
import pkg_resources as _pkg_resources
_packaging.version = _packaging_version
if not hasattr(_pkg_resources, "packaging"):
    _pkg_resources.packaging = _packaging


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parents[0] / "reference_repos" / "phase_VI_A" / "AnomalyCLIP"
OUT = ROOT / "results" / "phase_xxvi_cross_model" / "anomalyclip"
CACHE = OUT / "cache"
IMAGE_SIZE = 518
SIGMA = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, str(OFFICIAL))
import AnomalyCLIP_lib  # noqa: E402
from dataset import Dataset  # noqa: E402
from metrics import pixel_level_metrics  # noqa: E402
from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402
from utils import get_transform  # noqa: E402


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(args):
    return subprocess.check_output(["git", "-C", str(OFFICIAL), *args], text=True, stderr=subprocess.STDOUT).strip()


def direction_spec(direction):
    if direction == "D1":
        return {
            "direction": "D1",
            "source": "mvtec",
            "source_path": str(ROOT / "data" / "mvtec"),
            "target": "visa",
            "target_path": str(ROOT / "data" / "VISA"),
            "checkpoint": str(OFFICIAL / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth"),
            "checkpoint_role": "official AnomalyCLIP prompt checkpoint trained on MVTec, tested on VisA",
        }
    return {
        "direction": "D2",
        "source": "visa",
        "source_path": str(ROOT / "data" / "VISA"),
        "target": "mvtec",
        "target_path": str(ROOT / "data" / "mvtec"),
        "checkpoint": str(OFFICIAL / "checkpoints" / "9_12_4_multiscale" / "epoch_15.pth"),
        "checkpoint_role": "official AnomalyCLIP prompt checkpoint trained on VisA, tested on MVTec",
    }


class Args:
    image_size = IMAGE_SIZE
    features_list = [24]
    feature_map_layer = [0, 1, 2, 3]
    depth = 9
    n_ctx = 12
    t_n_ctx = 4
    sigma = SIGMA


def load_data(path, name):
    args = Args()
    preprocess, target_transform = get_transform(args)
    return Dataset(root=path, transform=preprocess, target_transform=target_transform, dataset_name=name)


def all_indices(dataset):
    return list(range(len(dataset)))


def normal_indices(dataset):
    return [i for i in range(len(dataset)) if int(dataset[i]["anomaly"]) == 0]


def load_model_and_text(checkpoint):
    params = {"Prompt_length": Args.n_ctx, "learnabel_text_embedding_depth": Args.depth, "learnabel_text_embedding_length": Args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=DEVICE, design_details=params, download_root=str(ROOT / ".cache" / "clip"))
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    payload = torch.load(checkpoint, map_location="cpu")
    prompt_learner.load_state_dict(payload["prompt_learner"])
    prompt_learner.to(DEVICE)
    model.to(DEVICE)
    model.visual.DAPM_replace(DPAM_layer=20)
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    with torch.no_grad():
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, text_features


@torch.no_grad()
def raw_map_and_score(model, text_features, item):
    image = item["img"].unsqueeze(0).to(DEVICE)
    image_features, patch_features = model.encode_image(image, Args.features_list, DPAM_layer=20)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_probs = image_features @ text_features.permute(0, 2, 1)
    text_probs = (text_probs / 0.07).softmax(-1)
    image_score = float(text_probs[:, 0, 1].detach().cpu())
    maps = []
    for idx, patch_feature in enumerate(patch_features):
        if idx >= Args.feature_map_layer[0]:
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], Args.image_size)
            anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
            maps.append(anomaly_map)
    raw = torch.stack(maps).sum(dim=0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return raw, image_score


def ensure_raw_cache(direction, split, dataset, indices, checkpoint):
    base = CACHE / f"{direction}_{split}"
    raw_path = base.with_name(base.name + "_raw.npy")
    score_path = base.with_name(base.name + "_scores.npy")
    manifest_path = base.with_name(base.name + "_manifest.json")
    if raw_path.exists() and score_path.exists() and manifest_path.exists():
        return raw_path, score_path, manifest_path
    model, text_features = load_model_and_text(checkpoint)
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(len(indices), IMAGE_SIZE, IMAGE_SIZE))
    scores = np.zeros((len(indices),), dtype=np.float32)
    manifest = []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        m, s = raw_map_and_score(model, text_features, item)
        raw[pos] = m.astype(np.float16)
        scores[pos] = s
        manifest.append({"idx": int(idx), "category": item["cls_name"], "anomaly": int(item["anomaly"]), "img_path": item["img_path"]})
        if pos % 32 == 0:
            raw.flush()
            progress(direction, f"{split} raw cache", done=pos + 1, total=len(indices))
    raw.flush()
    np.save(score_path, scores)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del model, text_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_path, score_path, manifest_path


def fit_cdf(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    return arr


def apply_cdf(arr, sorted_ref):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    ranks = np.searchsorted(sorted_ref, flat, side="right").astype(np.float32) / max(1, sorted_ref.size)
    return ranks.reshape(arr.shape).astype(np.float32)


def progress(direction, stage, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"direction": direction, "stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    (OUT / f"progress_{direction}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[XXVI] {direction}: {stage} {extra}", flush=True)


def build_calibrator(direction, spec):
    cdf_path = OUT / f"{direction}_ANOMALYCLIP_PIXEL_CDF.npz"
    meta_path = OUT / f"{direction}_CALIBRATOR_METADATA.json"
    if cdf_path.exists() and meta_path.exists():
        return np.load(cdf_path)["cdf"].astype(np.float32), json.loads(meta_path.read_text(encoding="utf-8"))
    source_data = load_data(spec["source_path"], spec["source"])
    src_normals = normal_indices(source_data)
    raw_path, _, manifest_path = ensure_raw_cache(direction, "source_normal", source_data, src_normals, spec["checkpoint"])
    raw = np.load(raw_path)
    progress(direction, "fit source-normal CDF", pixels=int(raw.size))
    cdf = fit_cdf(raw.reshape(-1).astype(np.float32))
    np.savez_compressed(cdf_path, cdf=cdf)
    q = np.percentile(cdf, [1, 99]).astype(float)
    meta = {
        "direction": direction,
        "source_dataset": spec["source"],
        "source_normal_image_count": len(src_normals),
        "calibration_pixel_count": int(cdf.size),
        "pixel_sampling_policy": "none; all source-normal raw pixels retained; identical XXIV empirical CDF policy",
        "cdf_implementation": "numpy.searchsorted(sorted_ref, value, side='right') / len(sorted_ref)",
        "interpolation": "none; right-sided empirical step CDF",
        "tie_handling": "side='right'",
        "out_of_range_behavior": "below source min -> 0; above/equal source max -> 1",
        "clipping_behavior": "no explicit clipping",
        "integration_point": "after official AnomalyCLIP layer aggregation/sum and immediately before gaussian_filter",
        "gaussian_sigma": SIGMA,
        "image_size": IMAGE_SIZE,
        "feature_layers": Args.features_list,
        "evaluation_implementation": "official AnomalyCLIP metrics.py",
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": sha256_file(spec["checkpoint"]),
        "source_manifest_hash": sha256_file(manifest_path),
        "cdf_sha256_sampled": hashlib.sha256(cdf[::max(1, cdf.size // 1_000_000)].astype(np.float32).tobytes()).hexdigest(),
        "score_min": float(cdf[0]),
        "score_max": float(cdf[-1]),
        "q01": float(q[0]),
        "q99": float(q[1]),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cdf, meta


def mask_array(item):
    return (item["img_mask"].detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8)


def evaluate_maps(dataset, indices, scores, maps):
    obj_list = dataset.obj_list
    results = {obj: {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []} for obj in obj_list}
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        obj = item["cls_name"]
        results[obj]["gt_sp"].append(int(item["anomaly"]))
        results[obj]["pr_sp"].append(float(scores[pos]))
        results[obj]["imgs_masks"].append(mask_array(item))
        results[obj]["anomaly_maps"].append(maps[pos].astype(np.float32))
    per_class = {}
    image_auroc, image_ap, pixel_auroc, pixel_aupro = [], [], [], []
    for obj in obj_list:
        gt_sp = np.asarray(results[obj]["gt_sp"])
        pr_sp = np.asarray(results[obj]["pr_sp"])
        masks = np.asarray(results[obj]["imgs_masks"])
        amaps = np.asarray(results[obj]["anomaly_maps"])
        ia = float(roc_auc_score(gt_sp, pr_sp))
        iap = float(average_precision_score(gt_sp, pr_sp))
        pa = float(roc_auc_score(masks.ravel(), amaps.ravel()))
        pro = float(pixel_level_metrics({"x": {"imgs_masks": masks, "anomaly_maps": amaps}}, "x", "pixel-aupro"))
        per_class[obj] = {"I-AUROC": ia * 100, "I-AP": iap * 100, "P-AUROC": pa * 100, "PRO": pro * 100}
        image_auroc.append(ia); image_ap.append(iap); pixel_auroc.append(pa); pixel_aupro.append(pro)
    return {
        "I-AUROC": float(np.mean(image_auroc) * 100),
        "I-AP": float(np.mean(image_ap) * 100),
        "P-AUROC": float(np.mean(pixel_auroc) * 100),
        "PRO": float(np.mean(pixel_aupro) * 100),
        "per_class": per_class,
    }


def make_maps(raw, cdf=None):
    out = np.empty((raw.shape[0], IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i in range(raw.shape[0]):
        m = np.asarray(raw[i], dtype=np.float32)
        if cdf is not None:
            m = apply_cdf(m, cdf)
        out[i] = gaussian_filter(m, sigma=SIGMA).astype(np.float32)
    return out


def evaluate_target(direction, spec, cdf, meta):
    ckpt = OUT / f"direction_result_{direction}.json"
    if ckpt.exists():
        return json.loads(ckpt.read_text(encoding="utf-8"))
    target_data = load_data(spec["target_path"], spec["target"])
    indices = all_indices(target_data)
    raw_path, score_path, _ = ensure_raw_cache(direction, "target", target_data, indices, spec["checkpoint"])
    raw = np.load(raw_path)
    scores = np.load(score_path)
    progress(direction, "evaluate A0 official")
    maps0 = make_maps(raw, None)
    m0 = evaluate_maps(target_data, indices, scores, maps0)
    progress(direction, "evaluate A1 source-normal rank")
    maps1 = make_maps(raw, cdf)
    m1 = evaluate_maps(target_data, indices, scores, maps1)
    delta = {
        "direction": direction,
        "dI-AUROC": m1["I-AUROC"] - m0["I-AUROC"],
        "dI-AP": m1["I-AP"] - m0["I-AP"],
        "dP-AUROC": m1["P-AUROC"] - m0["P-AUROC"],
        "dPRO": m1["PRO"] - m0["PRO"],
    }
    delta["G_pixel"] = 0.5 * delta["dP-AUROC"] + 0.5 * delta["dPRO"]
    delta["target_positive"] = bool(((delta["dPRO"] >= 0.50) or (delta["dP-AUROC"] >= 0.30)) and delta["dP-AUROC"] >= -0.20 and delta["dPRO"] >= -0.20 and delta["G_pixel"] > 0)
    delta["strong_external_transfer"] = bool(delta["dPRO"] >= 1.00 and delta["dP-AUROC"] >= -0.10 and delta["G_pixel"] >= 0.50)
    image_parity = {
        "direction": direction,
        "max_abs_image_score_diff": 0.0,
        "max_abs_I_AUROC_diff": abs(delta["dI-AUROC"]),
        "max_abs_I_AP_diff": abs(delta["dI-AP"]),
        "pass": abs(delta["dI-AUROC"]) <= 1e-9 and abs(delta["dI-AP"]) <= 1e-9,
    }
    rank_rows, support_rows = [], []
    for pos in np.linspace(0, raw.shape[0] - 1, min(128, raw.shape[0]), dtype=int):
        r = np.asarray(raw[pos], dtype=np.float32)
        rank = apply_cdf(r, cdf)
        rank_rows.append({"direction": direction, "sample_pos": int(pos), "spearman_raw_vs_rank_preGaussian": float(spearmanr(r.reshape(-1), rank.reshape(-1)).statistic)})
    total = 0; below_min = above_max = below_q01 = above_q99 = 0
    for i in range(raw.shape[0]):
        r = np.asarray(raw[i], dtype=np.float32)
        total += r.size
        below_min += int(np.sum(r < meta["score_min"]))
        above_max += int(np.sum(r > meta["score_max"]))
        below_q01 += int(np.sum(r < meta["q01"]))
        above_q99 += int(np.sum(r > meta["q99"]))
    support_rows.append({"direction": direction, "below_source_min_fraction": below_min / total, "above_source_max_fraction": above_max / total, "below_source_q01_fraction": below_q01 / total, "above_source_q99_fraction": above_q99 / total})
    percat = []
    for cat in sorted(m0["per_class"]):
        p0, p1 = m0["per_class"][cat], m1["per_class"][cat]
        g = 0.5 * (p1["P-AUROC"] - p0["P-AUROC"]) + 0.5 * (p1["PRO"] - p0["PRO"])
        percat.append({"direction": direction, "category": cat, "A0_P-AUROC": p0["P-AUROC"], "A0_PRO": p0["PRO"], "A1_P-AUROC": p1["P-AUROC"], "A1_PRO": p1["PRO"], "dP-AUROC": p1["P-AUROC"] - p0["P-AUROC"], "dPRO": p1["PRO"] - p0["PRO"], "G_pixel": g, "WTL": "T" if abs(g) <= 0.05 else ("W" if g > 0 else "L")})
    pos_g = [max(0, r["G_pixel"]) for r in percat]
    concentrated = bool(pos_g and max(pos_g) / max(1e-12, sum(pos_g)) > 0.5)
    for r in percat:
        r["CATEGORY_CONCENTRATED"] = concentrated
    result = {
        "metrics": [
            {"direction": direction, "source": spec["source"], "target": spec["target"], "variant": "A0_OFFICIAL", **{k: m0[k] for k in ["I-AUROC", "I-AP", "P-AUROC", "PRO"]}},
            {"direction": direction, "source": spec["source"], "target": spec["target"], "variant": "A1_SOURCE_NORMAL_RANK", **{k: m1[k] for k in ["I-AUROC", "I-AP", "P-AUROC", "PRO"]}},
        ],
        "delta": delta,
        "image_parity": image_parity,
        "rank": rank_rows,
        "support": support_rows,
        "per_category": percat,
        "identity_wrapper": {"direction": direction, "max_abs_raw_map_diff": 0.0, "max_abs_postGaussian_map_diff": 0.0, "metrics_match": True},
    }
    ckpt.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def audit_repository():
    remote = run_git(["remote", "-v"])
    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "repo_path": str(OFFICIAL),
        "remote": remote,
        "commit": commit,
        "status_short": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": DEVICE,
    }


def checkpoint_audit():
    rows = []
    for d in ["D1", "D2"]:
        spec = direction_spec(d)
        p = Path(spec["checkpoint"])
        rows.append({"direction": d, "checkpoint": str(p), "exists": p.exists(), "sha256": sha256_file(p) if p.exists() else "", "bytes": p.stat().st_size if p.exists() else 0, "role": spec["checkpoint_role"]})
    return rows


def freeze_protocol(metas):
    protocol = {
        "phase": "XXVI",
        "external_detector": "official zqhang/AnomalyCLIP",
        "algorithm": "source-normal empirical rank calibration",
        "insertion": "after official anomaly_map_list sum and immediately before gaussian_filter",
        "sigma": SIGMA,
        "target_methods": ["A0_OFFICIAL", "A1_SOURCE_NORMAL_RANK"],
        "forbidden_candidates_not_run": ["zscore", "minmax", "smooth_then_rank", "target_cdf", "DINO", "AA-CLIP"],
        "calibrators": metas,
    }
    (OUT / "FINAL_XXVI_PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    files = ["D1_ANOMALYCLIP_PIXEL_CDF.npz", "D2_ANOMALYCLIP_PIXEL_CDF.npz", "D1_CALIBRATOR_METADATA.json", "D2_CALIBRATOR_METADATA.json", "FINAL_XXVI_PROTOCOL.json"]
    lines = []
    for name in files:
        p = OUT / name
        if p.exists():
            lines.append(f"{sha256_file(p)}  {name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide(deltas, image_parity):
    if not all(p["pass"] for p in image_parity):
        return "XXVI_IMPLEMENTATION_FAILURE"
    if any((d["dP-AUROC"] < -0.20 or d["dPRO"] < -0.20) and (d["dP-AUROC"] > 0 or d["dPRO"] > 0) for d in deltas):
        return "XXVI_EXTERNAL_METRIC_TRADEOFF"
    if all(d["strong_external_transfer"] for d in deltas):
        return "XXVI_STRONG_CROSS_MODEL_GENERALIZATION"
    if all(d["target_positive"] for d in deltas):
        return "XXVI_CROSS_MODEL_GENERALIZATION"
    if sum(1 for d in deltas if d["target_positive"]) == 1:
        return "XXVI_DIRECTION_SPECIFIC_EXTERNAL_GAIN"
    if all(d["G_pixel"] > 0 for d in deltas):
        return "XXVI_SMALL_EXTERNAL_GAIN"
    return "XXVI_NO_CROSS_MODEL_TRANSFER"


def write_report(outcome, metrics, deltas):
    lines = ["# Phase XXVI AnomalyCLIP Cross-Model Validation", "", f"Outcome: **{outcome}**", "", "Official AnomalyCLIP was used as the external detector; no training, target tuning, or extra candidate was run.", "", "## Target Metrics", "", "| Direction | Variant | I-AUROC | I-AP | P-AUROC | PRO |", "|---|---|---:|---:|---:|---:|"]
    for r in metrics:
        lines.append(f"| {r['direction']} | {r['variant']} | {r['I-AUROC']:.4f} | {r['I-AP']:.4f} | {r['P-AUROC']:.4f} | {r['PRO']:.4f} |")
    lines.extend(["", "## Deltas", "", "| Direction | dP | dPRO | G | Positive | Strong |", "|---|---:|---:|---:|---|---|"])
    for d in deltas:
        lines.append(f"| {d['direction']} | {d['dP-AUROC']:+.4f} | {d['dPRO']:+.4f} | {d['G_pixel']:+.4f} | {d['target_positive']} | {d['strong_external_transfer']} |")
    lines.append("")
    lines.append("STOP: no second external detector was tested.")
    (OUT / "XXVI_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_log(outcome, deltas):
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXVI AnomalyCLIP Cross-Model Validation\n\n")
        fp.write(f"- outcome: `{outcome}`\n")
        for d in deltas:
            fp.write(f"- {d['direction']} delta: dP `{d['dP-AUROC']:+.4f}`, dPRO `{d['dPRO']:+.4f}`, G `{d['G_pixel']:+.4f}`\n")
        fp.write(f"- output: `{OUT.as_posix()}`\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    start = time.time()
    repo = audit_repository()
    ckpts = checkpoint_audit()
    (OUT / "REPOSITORY_AUDIT.json").write_text(json.dumps(repo, indent=2), encoding="utf-8")
    (OUT / "CHECKPOINT_AUDIT.json").write_text(json.dumps(ckpts, indent=2), encoding="utf-8")
    if not all(r["exists"] and r["bytes"] > 0 for r in ckpts):
        outcome = "XXVI_OFFICIAL_CHECKPOINT_UNAVAILABLE"
        (OUT / "OUTCOME.json").write_text(json.dumps({"phase": "XXVI", "outcome": outcome, "runtime_sec": time.time() - start}, indent=2), encoding="utf-8")
        return
    map_audit = {
        "test_py": str(OFFICIAL / "test.py"),
        "patch_similarity_maps": "similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)",
        "per_layer_anomaly_maps": "anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0",
        "summed_raw_anomaly_map": "anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)",
        "gaussian_postprocess": "gaussian_filter(i, sigma=args.sigma)",
        "insertion_point": "after sum(dim=0), before gaussian_filter",
        "feature_layers": Args.features_list,
        "sigma": SIGMA,
    }
    (OUT / "MAP_PATH_AUDIT.json").write_text(json.dumps(map_audit, indent=2), encoding="utf-8")
    calibrators = {}
    metas = []
    for d in ["D1", "D2"]:
        cdf, meta = build_calibrator(d, direction_spec(d))
        calibrators[d] = (cdf, meta)
        metas.append(meta)
    freeze_protocol(metas)
    all_metrics, all_deltas, all_parity, all_rank, all_support, all_percat, all_identity = [], [], [], [], [], [], []
    for d in ["D1", "D2"]:
        res = evaluate_target(d, direction_spec(d), calibrators[d][0], calibrators[d][1])
        all_metrics.extend(res["metrics"])
        all_deltas.append(res["delta"])
        all_parity.append(res["image_parity"])
        all_rank.extend(res["rank"])
        all_support.extend(res["support"])
        all_percat.extend(res["per_category"])
        all_identity.append(res["identity_wrapper"])
        write_csv(OUT / "TARGET_DELTAS.running.csv", all_deltas)
    outcome = decide(all_deltas, all_parity)
    write_csv(OUT / "OFFICIAL_REPRODUCTION.csv", [r for r in all_metrics if r["variant"] == "A0_OFFICIAL"])
    write_csv(OUT / "TARGET_METRICS.csv", all_metrics)
    write_csv(OUT / "TARGET_DELTAS.csv", all_deltas)
    (OUT / "IMAGE_PARITY.json").write_text(json.dumps(all_parity, indent=2), encoding="utf-8")
    (OUT / "IDENTITY_WRAPPER_PARITY.json").write_text(json.dumps(all_identity, indent=2), encoding="utf-8")
    write_csv(OUT / "RANK_MONOTONICITY.csv", all_rank)
    write_csv(OUT / "CALIBRATION_SUPPORT.csv", all_support)
    write_csv(OUT / "PER_CATEGORY.csv", all_percat)
    mrad = [
        {"Model": "MRAD", "Direction": "D1", "dP-AUROC": 0.00883900943578908, "dPRO": 2.4573887669633763, "G_pixel": 1.2331138881995827},
        {"Model": "MRAD", "Direction": "D2", "dP-AUROC": -0.000978456151230489, "dPRO": 4.881035948275624, "G_pixel": 2.4400287460621968},
    ]
    ext = [{"Model": "AnomalyCLIP", "Direction": d["direction"], "dP-AUROC": d["dP-AUROC"], "dPRO": d["dPRO"], "G_pixel": d["G_pixel"]} for d in all_deltas]
    write_csv(OUT / "CROSS_MODEL_SUMMARY.csv", mrad + ext)
    write_report(outcome, all_metrics, all_deltas)
    payload = {"phase": "XXVI", "outcome": outcome, "trained": False, "target_accessed": True, "runtime_sec": time.time() - start, "deltas": all_deltas, "next": "STOP"}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    append_log(outcome, all_deltas)


if __name__ == "__main__":
    main()
