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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import load_dataset, write_csv
from phase_xa.official_task_forward import visual_outputs
from phase_xa.source_folds import deterministic_source_folds
from phase_xi.config import GEOMETRY_CONFIG
from phase_xi.diagnostics import cohen_d
from phase_xi.residual_extractor import l2_normalize, remove_subspace, residual
from phase_xi.shared_anomaly_basis import pca_basis, random_orthonormal


OUT = Path("results/phase_xi_aads/geometry_audit")
BRANCHES = ["global", "local"]
REPRS = ["absolute", "centered", "normal_removed"]
RANKS = [1, 2, 4, 8, 16]


def f(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value, digits=4):
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def direction_cfg(direction):
    if direction == "D1":
        return "mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"
    return "visa", "data/VISA", "checkpoints/test_on_mvtec.pth"


def cap_indices_by_category(dataset, categories, cap):
    selected = []
    counts = defaultdict(int)
    for idx in range(len(dataset)):
        item = dataset[idx]
        cat = item["cls_name"]
        if cat not in categories or counts[cat] >= cap:
            continue
        selected.append(idx)
        counts[cat] += 1
    return selected, dict(counts)


def extract_block18_features(model, image):
    out = visual_outputs(model, image, adapter=None, grad=False, snapshot_mode="true_clone")
    block18 = out["snapshots"]["true_block18"]
    if not isinstance(block18, list) or len(block18) != 2:
        raise RuntimeError("DAPM block18 did not return two branches")
    local = block18[0].permute(1, 0, 2).float()
    global_branch = block18[1].permute(1, 0, 2).float()
    global_cls = global_branch[:, 0, :]
    local_patches = local[:, 1:, :]
    global_cls = global_cls / global_cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    local_patches = local_patches / local_patches.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return global_cls.detach().cpu().numpy()[0], local_patches.detach().cpu().numpy()[0]


def patch_labels(item, n_tokens):
    side = int(round(n_tokens ** 0.5))
    mask = item["img_mask"].float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    small = F.interpolate(mask, size=(side, side), mode="bilinear", align_corners=False).reshape(-1)
    return (small > 0.5).numpy().astype(np.int64)


def append_capped(bucket, category, array, cap):
    cur = len(bucket[category])
    if cur >= cap:
        return
    take = min(cap - cur, len(array))
    if take > 0:
        bucket[category].extend(array[:take])


def extract_direction_cache(direction, device):
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / f"{direction}_fold0_source_features.npz"
    meta_path = OUT / f"{direction}_fold0_source_features_meta.json"
    if cache_path.exists() and meta_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return data, meta

    dataset_name, data_path, checkpoint = direction_cfg(direction)
    model, *_ = load_components(checkpoint, device)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][0]
    all_categories = set(rotation["train_categories"]) | set(rotation["val_categories"])
    indices, counts = cap_indices_by_category(dataset, all_categories, GEOMETRY_CONFIG["max_images_per_category"])

    global_x, global_y, global_cat, global_split = [], [], [], []
    local_x, local_y, local_cat, local_split = [], [], [], []
    style_g0, style_g1, style_cat, style_y = [], [], [], []
    local_caps = {
        "normal": defaultdict(list),
        "anomaly": defaultdict(list),
    }
    local_meta = []
    style_per_cat = defaultdict(int)
    start = time.time()
    for idx in tqdm(indices, desc=f"extract XI-A {direction} source fold0"):
        item = dataset[idx]
        cat = item["cls_name"]
        split = "train" if cat in rotation["train_categories"] else "val"
        image = item["img"].unsqueeze(0).to(device)
        label = int(item["anomaly"])
        with torch.no_grad():
            g, lp = extract_block18_features(model, image)
        global_x.append(g)
        global_y.append(label)
        global_cat.append(cat)
        global_split.append(split)

        labels = patch_labels(item, lp.shape[0])
        normal_patches = lp[labels == 0]
        anomaly_patches = lp[labels == 1]
        append_capped(local_caps["normal"], cat, normal_patches, GEOMETRY_CONFIG["max_normal_patches_per_category"])
        append_capped(local_caps["anomaly"], cat, anomaly_patches, GEOMETRY_CONFIG["max_anomaly_patches_per_category"])
        local_meta.append({"idx": idx, "category": cat, "split": split, "image_label": label, "normal_patches": int(len(normal_patches)), "anomaly_patches": int(len(anomaly_patches))})

        if style_per_cat[cat] < 8:
            styled = torch.clamp(image * 1.06 + 0.015, -3.0, 3.0)
            with torch.no_grad():
                gs, _ = extract_block18_features(model, styled)
            style_g0.append(g)
            style_g1.append(gs)
            style_cat.append(cat)
            style_y.append(label)
            style_per_cat[cat] += 1

    for kind, label in [("normal", 0), ("anomaly", 1)]:
        for cat, values in local_caps[kind].items():
            split = "train" if cat in rotation["train_categories"] else "val"
            for value in values:
                local_x.append(value)
                local_y.append(label)
                local_cat.append(cat)
                local_split.append(split)

    np.savez_compressed(
        cache_path,
        global_x=np.asarray(global_x, dtype=np.float32),
        global_y=np.asarray(global_y, dtype=np.int64),
        global_cat=np.asarray(global_cat),
        global_split=np.asarray(global_split),
        local_x=np.asarray(local_x, dtype=np.float32),
        local_y=np.asarray(local_y, dtype=np.int64),
        local_cat=np.asarray(local_cat),
        local_split=np.asarray(local_split),
        style_g0=np.asarray(style_g0, dtype=np.float32),
        style_g1=np.asarray(style_g1, dtype=np.float32),
        style_cat=np.asarray(style_cat),
        style_y=np.asarray(style_y, dtype=np.int64),
    )
    meta = {
        "direction": direction,
        "dataset": dataset_name,
        "fold": 0,
        "train_categories": rotation["train_categories"],
        "val_categories": rotation["val_categories"],
        "image_counts": counts,
        "local_image_meta": local_meta,
        "seconds": time.time() - start,
        "target_accessed": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return np.load(cache_path, allow_pickle=True), meta


def stats_by_category(x, y, cats, split, wanted_split):
    normal = defaultdict(list)
    for value, label, cat, sp in zip(x, y, cats, split):
        if sp == wanted_split and int(label) == 0:
            normal[str(cat)].append(value)
    stats = {}
    for cat, values in normal.items():
        arr = np.asarray(values, dtype=np.float32)
        mu = arr.mean(axis=0)
        basis, eig, evr = pca_basis(arr - mu, GEOMETRY_CONFIG["normal_rank"])
        stats[cat] = {"mu": mu, "basis": basis, "eig": eig, "evr": evr}
    return stats


def representation(x, cats, stats, kind):
    out = []
    for value, cat in zip(x, cats):
        st = stats.get(str(cat))
        if kind == "absolute" or st is None:
            out.append(value)
        elif kind == "centered":
            out.append(l2_normalize(value - st["mu"]))
        else:
            out.append(residual(value, st["mu"], st["basis"]))
    return np.asarray(out, dtype=np.float32)


def fit_auc(train_x, train_y, val_x, val_y):
    if len(set(train_y)) < 2 or len(set(val_y)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=GEOMETRY_CONFIG["probe_max_iter"], class_weight="balanced", solver="liblinear")
    clf.fit(train_x, train_y)
    score = clf.decision_function(val_x)
    return float(roc_auc_score(val_y, score) * 100.0)


def category_accuracy(x, cats):
    cats = np.asarray(cats)
    unique = sorted(set(cats.tolist()))
    if len(unique) < 2 or len(x) < 10:
        return float("nan")
    idx = np.arange(len(x))
    train_idx, test_idx = train_test_split(idx, test_size=0.35, random_state=0, stratify=cats)
    clf = LogisticRegression(max_iter=GEOMETRY_CONFIG["probe_max_iter"], class_weight="balanced")
    clf.fit(x[train_idx], cats[train_idx])
    pred = clf.predict(x[test_idx])
    return float(accuracy_score(cats[test_idx], pred) * 100.0)


def balanced_anomaly_residuals(x, y, cats, split, stats, wanted_split, max_per_cat=512):
    by_cat = defaultdict(list)
    for value, label, cat, sp in zip(x, y, cats, split):
        if sp == wanted_split and int(label) == 1 and str(cat) in stats:
            by_cat[str(cat)].append(residual(value, stats[str(cat)]["mu"], stats[str(cat)]["basis"]))
    if not by_cat:
        return np.zeros((0, x.shape[1]), dtype=np.float32), {}
    n = min(max(1, min(len(v) for v in by_cat.values())), max_per_cat)
    rows, counts = [], {}
    for cat, values in sorted(by_cat.items()):
        take = np.asarray(values[:n], dtype=np.float32)
        rows.append(take)
        counts[cat] = len(take)
    return np.concatenate(rows, axis=0), counts


def basis_energy(x, basis):
    coeff = x @ basis.T
    return np.sum(coeff ** 2, axis=1) / np.maximum(np.sum(x ** 2, axis=1), 1e-8)


def principal_similarity(b1, b2):
    if b1.size == 0 or b2.size == 0:
        return float("nan")
    return float(np.linalg.svd(b1 @ b2.T, compute_uv=False).mean())


def audit_direction(direction, data, meta):
    rows_linear, rows_cat, rows_pca, rows_stability, rows_random, rows_style, rows_stats = [], [], [], [], [], [], []
    rng = np.random.default_rng(GEOMETRY_CONFIG["random_seed"])
    branch_data = {
        "global": (data["global_x"], data["global_y"], data["global_cat"], data["global_split"]),
        "local": (data["local_x"], data["local_y"], data["local_cat"], data["local_split"]),
    }
    basis_store = {}
    for branch, (x, y, cats, split) in branch_data.items():
        train_stats = stats_by_category(x, y, cats, split, "train")
        val_stats = stats_by_category(x, y, cats, split, "val")
        merged_stats = {**train_stats, **val_stats}
        train_mask = split == "train"
        val_mask = split == "val"

        train_residuals, contrib = balanced_anomaly_residuals(x, y, cats, split, merged_stats, "train")
        basis, eig, evr = pca_basis(train_residuals, GEOMETRY_CONFIG["anomaly_rank"])
        basis_store[branch] = basis
        abs_basis, abs_eig, abs_evr = pca_basis(l2_normalize(x[(split == "train") & (y == 1)]), GEOMETRY_CONFIG["anomaly_rank"])
        random_basis = random_orthonormal(x.shape[1], GEOMETRY_CONFIG["anomaly_rank"], rng)

        for rank in RANKS:
            rows_pca.append({
                "direction": direction,
                "branch": branch,
                "representation": "normal_removed",
                "rank": rank,
                "explained_variance": float(np.sum(evr[:rank])),
                "n": len(train_residuals),
            })
            rows_pca.append({
                "direction": direction,
                "branch": branch,
                "representation": "absolute_anomaly",
                "rank": rank,
                "explained_variance": float(np.sum(abs_evr[:rank])),
                "n": int(((split == "train") & (y == 1)).sum()),
            })

        reps = {}
        for kind in REPRS:
            rep = representation(x, cats, merged_stats, kind)
            reps[kind] = rep
            auc = fit_auc(rep[train_mask], y[train_mask], rep[val_mask], y[val_mask])
            cat_acc = category_accuracy(rep[train_mask & (y == 1)], cats[train_mask & (y == 1)])
            rows_linear.append({"direction": direction, "branch": branch, "representation": kind, "anomaly_vs_normal_auroc": auc})
            rows_cat.append({"direction": direction, "branch": branch, "representation": kind, "category_accuracy": cat_acc})
            rows_stats.append({
                "direction": direction,
                "branch": branch,
                "representation": kind,
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "train_anomaly": int((train_mask & (y == 1)).sum()),
                "val_anomaly": int((val_mask & (y == 1)).sum()),
                "categories_with_normal_stats": len(merged_stats),
                "anomaly_basis_contribution": json.dumps(contrib, sort_keys=True),
            })

        val_res = reps["normal_removed"][val_mask]
        val_y = y[val_mask]
        true_e = basis_energy(val_res, basis)
        rand_e = basis_energy(val_res, random_basis)
        for name, energy in [("true_anomaly_basis", true_e), ("random_basis", rand_e)]:
            normal_e = energy[val_y == 0]
            anomaly_e = energy[val_y == 1]
            rows_random.append({
                "direction": direction,
                "branch": branch,
                "basis": name,
                "mean_E_anomaly": float(np.mean(anomaly_e)) if len(anomaly_e) else float("nan"),
                "mean_E_normal": float(np.mean(normal_e)) if len(normal_e) else float("nan"),
                "cohen_d": cohen_d(anomaly_e, normal_e) if len(anomaly_e) and len(normal_e) else float("nan"),
            })

        train_cats = sorted(set(cats[train_mask].tolist()))
        subset_bases = []
        for fold_id in range(3):
            subset = {cat for i, cat in enumerate(train_cats) if i % 3 != fold_id}
            sub_mask = np.asarray([(sp == "train" and cat in subset) for cat, sp in zip(cats, split)])
            sub_res, _ = balanced_anomaly_residuals(x[sub_mask], y[sub_mask], cats[sub_mask], np.asarray(["train"] * int(sub_mask.sum())), merged_stats, "train")
            b, _, _ = pca_basis(sub_res, GEOMETRY_CONFIG["anomaly_rank"])
            subset_bases.append(b)
        sims = []
        for i in range(len(subset_bases)):
            for j in range(i + 1, len(subset_bases)):
                sims.append(principal_similarity(subset_bases[i], subset_bases[j]))
        rows_stability.append({
            "direction": direction,
            "branch": branch,
            "rank": GEOMETRY_CONFIG["anomaly_rank"],
            "mean_principal_similarity": float(np.nanmean(sims)),
            "pairwise": json.dumps(sims),
        })

        if branch == "global":
            style0 = data["style_g0"]
            style1 = data["style_g1"]
            style_cats = data["style_cat"]
            style_rep0 = representation(style0, style_cats, merged_stats, "normal_removed")
            style_rep1 = representation(style1, style_cats, merged_stats, "normal_removed")
            res_coeff0 = style_rep0 @ basis.T
            res_coeff1 = style_rep1 @ basis.T
            abs_coeff0 = l2_normalize(style0) @ abs_basis.T
            abs_coeff1 = l2_normalize(style1) @ abs_basis.T
            res_cos = np.sum(l2_normalize(res_coeff0) * l2_normalize(res_coeff1), axis=1)
            abs_cos = np.sum(l2_normalize(abs_coeff0) * l2_normalize(abs_coeff1), axis=1)
            rows_style.append({
                "direction": direction,
                "branch": branch,
                "residual_coeff_cosine": float(np.mean(res_cos)),
                "absolute_coeff_cosine": float(np.mean(abs_cos)),
                "delta_residual_minus_absolute": float(np.mean(res_cos) - np.mean(abs_cos)),
                "pairs": len(res_cos),
            })
    return rows_stats, rows_linear, rows_cat, rows_pca, rows_stability, rows_style, rows_random


def decide(linear_rows, cat_rows, stability_rows, random_rows):
    outcomes = {}
    for direction in ["D1", "D2"]:
        supported = False
        for branch in BRANCHES:
            abs_auc = next((r for r in linear_rows if r["direction"] == direction and r["branch"] == branch and r["representation"] == "absolute"), None)
            abs_cat = next((r for r in cat_rows if r["direction"] == direction and r["branch"] == branch and r["representation"] == "absolute"), None)
            for rep in ["centered", "normal_removed"]:
                auc = next((r for r in linear_rows if r["direction"] == direction and r["branch"] == branch and r["representation"] == rep), None)
                cat = next((r for r in cat_rows if r["direction"] == direction and r["branch"] == branch and r["representation"] == rep), None)
                stab = next((r for r in stability_rows if r["direction"] == direction and r["branch"] == branch), None)
                if not all([abs_auc, abs_cat, auc, cat, stab]):
                    continue
                cat_drop = f(abs_cat["category_accuracy"]) - f(cat["category_accuracy"])
                auc_delta = f(auc["anomaly_vs_normal_auroc"]) - f(abs_auc["anomaly_vs_normal_auroc"])
                stable = f(stab["mean_principal_similarity"]) >= 0.70
                if stable and ((cat_drop >= 10.0 and auc_delta >= -0.50) or (auc_delta >= 0.50 and cat_drop >= 0.0)):
                    supported = True
        true_effect = max(
            [f(r["cohen_d"]) for r in random_rows if r["direction"] == direction and r["basis"] == "true_anomaly_basis"] or [float("nan")]
        )
        rand_effect = max(
            [f(r["cohen_d"]) for r in random_rows if r["direction"] == direction and r["basis"] == "random_basis"] or [float("nan")]
        )
        outcomes[direction] = {
            "residual_supported": supported,
            "true_basis_best_cohen_d": true_effect,
            "random_basis_best_cohen_d": rand_effect,
            "basis_specific": true_effect >= 0.30 and true_effect > rand_effect + 0.05,
        }
    pass_all = all(v["residual_supported"] for v in outcomes.values())
    useful_basis = any(v["basis_specific"] for v in outcomes.values())
    label = "PASS_TO_PHASE_XI_B" if pass_all and useful_basis else "STOP_RESIDUAL_DIRECTION_ROUTE"
    return label, outcomes


def write_report(label, outcomes, stats, linear_rows, cat_rows, pca_rows, stability_rows, style_rows, random_rows):
    lines = [
        "# Phase XI-A Geometry Audit",
        "",
        f"Outcome: **{label}**",
        "",
        "Target data was not accessed. This audit used frozen official CLIP features on source fold0 only.",
        "",
        "## Direction Outcomes",
        "",
        "| Direction | Residual Supported | Basis Specific | True Best d | Random Best d |",
        "|---|---:|---:|---:|---:|",
    ]
    for direction, payload in outcomes.items():
        lines.append(
            f"| {direction} | {payload['residual_supported']} | {payload['basis_specific']} | "
            f"{fmt(payload['true_basis_best_cohen_d'])} | {fmt(payload['random_basis_best_cohen_d'])} |"
        )
    lines.extend(["", "## Linear Probe AUROC", "", "| Direction | Branch | Representation | AUROC |", "|---|---|---|---:|"])
    for row in linear_rows:
        lines.append(f"| {row['direction']} | {row['branch']} | {row['representation']} | {fmt(row['anomaly_vs_normal_auroc'])} |")
    lines.extend(["", "## Category Probe Accuracy", "", "| Direction | Branch | Representation | Accuracy |", "|---|---|---|---:|"])
    for row in cat_rows:
        lines.append(f"| {row['direction']} | {row['branch']} | {row['representation']} | {fmt(row['category_accuracy'])} |")
    lines.extend(["", "## Rank-8 Subspace Stability", "", "| Direction | Branch | Mean Principal Similarity |", "|---|---|---:|"])
    for row in stability_rows:
        lines.append(f"| {row['direction']} | {row['branch']} | {fmt(row['mean_principal_similarity'])} |")
    lines.extend(["", "## Random Basis Control", "", "| Direction | Branch | Basis | E anomaly | E normal | Cohen d |", "|---|---|---|---:|---:|---:|"])
    for row in random_rows:
        lines.append(f"| {row['direction']} | {row['branch']} | {row['basis']} | {fmt(row['mean_E_anomaly'])} | {fmt(row['mean_E_normal'])} | {fmt(row['cohen_d'])} |")
    lines.extend(["", "## Style Consistency", "", "| Direction | Branch | Residual coeff cos | Absolute coeff cos | Delta | Pairs |", "|---|---|---:|---:|---:|---:|"])
    for row in style_rows:
        lines.append(f"| {row['direction']} | {row['branch']} | {fmt(row['residual_coeff_cosine'])} | {fmt(row['absolute_coeff_cosine'])} | {fmt(row['delta_residual_minus_absolute'])} | {row['pairs']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- PASS requires residual invariance support in both D1 and D2 plus a true anomaly basis that separates anomaly/normal energy better than random in at least one direction.",
        "- If the outcome is STOP, Phase XI-B adapter training must not start.",
    ])
    (OUT / "GEOMETRY_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    all_stats, all_linear, all_cat, all_pca, all_stability, all_style, all_random = [], [], [], [], [], [], []
    costs = []
    for direction in ["D1", "D2"]:
        start = time.time()
        data, meta = extract_direction_cache(direction, device)
        rows = audit_direction(direction, data, meta)
        for dest, src in zip([all_stats, all_linear, all_cat, all_pca, all_stability, all_style, all_random], rows):
            dest.extend(src)
        costs.append({"direction": direction, "seconds": time.time() - start, "target_accessed": False})
    label, outcomes = decide(all_linear, all_cat, all_stability, all_random)
    write_csv(OUT / "residual_statistics.csv", all_stats)
    write_csv(OUT / "linear_probe_metrics.csv", all_linear)
    write_csv(OUT / "category_probe_metrics.csv", all_cat)
    write_csv(OUT / "pca_energy.csv", all_pca)
    write_csv(OUT / "subspace_stability.csv", all_stability)
    write_csv(OUT / "style_consistency.csv", all_style)
    write_csv(OUT / "random_basis_control.csv", all_random)
    write_report(label, outcomes, all_stats, all_linear, all_cat, all_pca, all_stability, all_style, all_random)
    (OUT / "OUTCOME.json").write_text(json.dumps({
        "outcome": label,
        "direction_outcomes": outcomes,
        "costs": costs,
        "target_accessed": False,
        "next": "Run Phase XI-B only if outcome is PASS_TO_PHASE_XI_B; not launched automatically.",
    }, indent=2), encoding="utf-8")
    with Path("EXPERIMENT_LOG.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Phase XI-A Geometry Audit\n\n"
            "- Command: `python -u experiments\\phase_xi_geometry_audit.py`\n"
            "- Source folds: D1 source MVTec fold0; D2 source VisA fold0.\n"
            "- Feature layer: block18 true DAPM branches.\n"
            "- Basis rank: 8 anomaly, 8 normal.\n"
            f"- Outcome: `{label}`\n"
            "- Target data accessed: `False`\n"
        )
    print(json.dumps({"outcome": label, "report": str(OUT / "GEOMETRY_AUDIT_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
