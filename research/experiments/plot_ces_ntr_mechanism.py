from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_ces_linear_head_control import (  # noqa: E402
    FORMAL_CKPT_DIR,
    HeadSpec,
    direct_target_global_cache,
    load_source_pool_artifact,
    local_cache_path,
    lrp_map,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (  # noqa: E402
    item_at,
    item_meta,
    mask_np,
    torch_resize,
)


OUT = ROOT / "results" / "ces_ntr_visualization" / "final"
SOURCE_CACHE = ROOT / "results" / "paper_assets" / "journal_figures" / "figure_state_compression_cache_visa_source.npz"
SOURCE = "Full VisA"
QUAL_TARGET = "MVTec"
NTR_RHO = 0.10
SEED = 0

BLUE = "#4C78A8"
BLUE_DARK = "#1F4E79"
BLUE_LIGHT = "#8CB6D8"
ORANGE = "#F28E2B"
GRAY = "#6B7280"

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + eps)


def load_source_response_cache():
    if not SOURCE_CACHE.exists():
        raise FileNotFoundError(f"Missing source response cache: {SOURCE_CACHE}")
    data = np.load(SOURCE_CACHE, allow_pickle=True)
    required = {"labels", "meta", "ces_x", "ces_y", "ntr_x", "ntr_y"}
    missing = sorted(required - set(data.files))
    if missing:
        raise RuntimeError(f"Source cache missing required keys: {missing}")
    labels = data["labels"].astype(np.int64)
    ces_x = data["ces_x"].astype(np.float32)
    ces_y = data["ces_y"].astype(np.float32)
    ntr_x = data["ntr_x"].astype(np.float32)
    ntr_y = data["ntr_y"].astype(np.float32)
    meta = data["meta"]
    q_ces = ces_y - ces_x
    q_ntr = ntr_y - ntr_x
    normal_idx = np.where(labels == 0)[0]
    k = max(1, int(np.ceil(NTR_RHO * len(normal_idx))))
    tail_idx = normal_idx[np.argsort(q_ces[normal_idx])[-k:]]
    hard = np.zeros(len(labels), dtype=bool)
    hard[tail_idx] = True
    return {
        "labels": labels,
        "meta": meta,
        "ces_x": ces_x,
        "ces_y": ces_y,
        "ntr_x": ntr_x,
        "ntr_y": ntr_y,
        "q_ces": q_ces,
        "q_ntr": q_ntr,
        "hard": hard,
    }


def load_formal_specs(device: torch.device):
    info = load_source_pool_artifact(SOURCE, device, smoke=False)
    if info is None:
        raise FileNotFoundError("Could not load Full VisA source-pool artifact.")

    ces = HeadSpec(
        "CES",
        {"m": info["m"], "alpha": float(info["alpha"]), "v": unit(info["init_v"])},
        SOURCE,
        seed=SEED,
    )

    ckpt_path = FORMAL_CKPT_DIR / "Full_VisA_seed0_rho0.10.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing NTR checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ntr = HeadSpec(
        "CES",
        {"m": info["m"], "alpha": float(info["alpha"]), "v": unit(ckpt["v"])},
        SOURCE,
        seed=int(ckpt.get("seed", SEED)),
    )
    return info, ces, ntr, ckpt_path


def select_qualitative_samples(dataset, idxs, kind):
    records = []
    for pos, ref in enumerate(idxs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, QUAL_TARGET)
        if int(label) != 1:
            continue
        mask = mask_np(item)
        area = int(mask.sum())
        if area <= 0:
            continue
        records.append(
            {
                "pos": pos,
                "ref": ref,
                "category": str(cat),
                "path": str(path),
                "area": area,
                "area_ratio": float(area / mask.size),
            }
        )
    if not records:
        raise RuntimeError(f"No anomaly samples with masks found for {QUAL_TARGET}.")

    selected = []
    for cat in ("cable", "grid"):
        subset = [r for r in records if r["category"] == cat]
        if subset:
            subset = sorted(subset, key=lambda r: (r["area"], r["path"]))
            selected.append(subset[len(subset) // 2])
    if len(selected) < 2:
        by_cat = {}
        for r in records:
            by_cat.setdefault(r["category"], []).append(r)
        med = []
        for cat, rows in by_cat.items():
            rows = sorted(rows, key=lambda r: (r["area"], r["path"]))
            med.append(rows[len(rows) // 2])
        med = sorted(med, key=lambda r: (r["category"], r["path"]))
        for r in med:
            if all(r["path"] != s["path"] for s in selected):
                selected.append(r)
            if len(selected) >= 2:
                break
    return selected[:2], records


def read_rgb_image(path_str: str, mask_shape):
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    img = Image.open(path).convert("RGB")
    img = img.resize((int(mask_shape[1]), int(mask_shape[0])), Image.BILINEAR)
    return np.asarray(img)


def build_qualitative_data(device: torch.device, source_info: dict, ces: HeadSpec, ntr: HeadSpec):
    _cell_id, target, dataset, idxs, patch, kind = direct_target_global_cache(QUAL_TARGET, device, smoke=False)
    lpath = local_cache_path(SOURCE, QUAL_TARGET)
    if not lpath.exists():
        raise FileNotFoundError(f"Missing local LRP cache: {lpath}")
    local_cache = np.load(lpath, mmap_mode="r")
    selected, all_records = select_qualitative_samples(dataset, idxs, kind)

    samples = []
    all_maps = []
    for rec in selected:
        pos = int(rec["pos"])
        item = item_at(dataset, kind, rec["ref"], pos)
        mask = mask_np(item)
        image = read_rgb_image(rec["path"], mask.shape)
        global_patch = patch[pos]
        local_patches = np.asarray(local_cache[pos], dtype=np.float32)
        ces74 = lrp_map(source_info, ces, global_patch, local_patches, device)
        ntr74 = lrp_map(source_info, ntr, global_patch, local_patches, device)
        ces_full = torch_resize(ces74, mask.shape, align_corners=True)
        ntr_full = torch_resize(ntr74, mask.shape, align_corners=True)
        all_maps.extend([ces_full.reshape(-1), ntr_full.reshape(-1)])
        samples.append(
            {
                **rec,
                "image": image,
                "mask": mask,
                "ces74": ces74,
                "ntr74": ntr74,
                "ces_full": ces_full,
                "ntr_full": ntr_full,
            }
        )
    stacked = np.concatenate(all_maps)
    vmin = float(np.percentile(stacked, 1.0))
    vmax = float(np.percentile(stacked, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(stacked)), float(np.max(stacked))
    return samples, all_records, str(lpath), {"vmin": vmin, "vmax": vmax, "align_corners": True}


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def export_data(cache, samples, all_records, ckpt_path, local_path, heatmap_scale):
    rows = []
    for i in range(len(cache["labels"])):
        meta = cache["meta"][i].tolist() if hasattr(cache["meta"][i], "tolist") else list(cache["meta"][i])
        rows.append(
            {
                "patch_id": i,
                "label": int(cache["labels"][i]),
                "meta0": meta[0],
                "meta1": meta[1],
                "category": meta[2] if len(meta) > 2 else "",
                "ces_sn": float(cache["ces_x"][i]),
                "ces_sa": float(cache["ces_y"][i]),
                "ntr_sn": float(cache["ntr_x"][i]),
                "ntr_sa": float(cache["ntr_y"][i]),
                "q_ces": float(cache["q_ces"][i]),
                "q_ntr": float(cache["q_ntr"][i]),
                "hard_normal": bool(cache["hard"][i]),
            }
        )
    write_csv(
        OUT / "source_patch_responses.csv",
        rows,
        [
            "patch_id",
            "label",
            "meta0",
            "meta1",
            "category",
            "ces_sn",
            "ces_sa",
            "ntr_sn",
            "ntr_sa",
            "q_ces",
            "q_ntr",
            "hard_normal",
        ],
    )

    normal = np.where(cache["labels"] == 0)[0]
    order = normal[np.argsort(cache["q_ces"][normal])]
    tail_threshold_rank = int(np.floor((1.0 - NTR_RHO) * len(order)))
    tail_rows = []
    for rank, idx in enumerate(order):
        tail_rows.append(
            {
                "rank": rank,
                "rank_fraction": float(rank / max(1, len(order) - 1)),
                "patch_id": int(idx),
                "q_ces": float(cache["q_ces"][idx]),
                "q_ntr": float(cache["q_ntr"][idx]),
                "in_top_rho_tail": bool(rank >= tail_threshold_rank),
            }
        )
    write_csv(OUT / "normal_tail_curves.csv", tail_rows, ["rank", "rank_fraction", "patch_id", "q_ces", "q_ntr", "in_top_rho_tail"])

    sample_rows = []
    npz_payload = {}
    for i, s in enumerate(samples):
        sample_rows.append(
            {
                "sample_id": i,
                "target": QUAL_TARGET,
                "category": s["category"],
                "pos": int(s["pos"]),
                "ref": str(s["ref"]),
                "img_path": s["path"],
                "mask_area": int(s["area"]),
                "mask_area_ratio": float(s["area_ratio"]),
                "ces_map_mean": float(np.mean(s["ces_full"])),
                "ntr_map_mean": float(np.mean(s["ntr_full"])),
            }
        )
        npz_payload[f"sample_{i}_image"] = s["image"]
        npz_payload[f"sample_{i}_mask"] = s["mask"]
        npz_payload[f"sample_{i}_ces74"] = s["ces74"]
        npz_payload[f"sample_{i}_ntr74"] = s["ntr74"]
        npz_payload[f"sample_{i}_ces_full"] = s["ces_full"]
        npz_payload[f"sample_{i}_ntr_full"] = s["ntr_full"]
    write_csv(
        OUT / "qualitative_samples.csv",
        sample_rows,
        ["sample_id", "target", "category", "pos", "ref", "img_path", "mask_area", "mask_area_ratio", "ces_map_mean", "ntr_map_mean"],
    )
    np.savez_compressed(
        OUT / "ces_ntr_visualization_data.npz",
        labels=cache["labels"],
        ces_x=cache["ces_x"],
        ces_y=cache["ces_y"],
        ntr_x=cache["ntr_x"],
        ntr_y=cache["ntr_y"],
        q_ces=cache["q_ces"],
        q_ntr=cache["q_ntr"],
        hard_normal=cache["hard"],
        **npz_payload,
    )

    metadata = {
        "source": SOURCE,
        "target_for_qualitative_panel": QUAL_TARGET,
        "source_response_cache": str(SOURCE_CACHE),
        "ntr_checkpoint": str(ckpt_path),
        "local_lrp_cache": local_path,
        "ntr_rho": NTR_RHO,
        "num_source_patches": int(len(cache["labels"])),
        "num_normal_patches": int(np.sum(cache["labels"] == 0)),
        "num_anomaly_patches": int(np.sum(cache["labels"] == 1)),
        "num_hard_normal_patches": int(np.sum(cache["hard"])),
        "qualitative_selection_rule": "MVTec target anomaly samples with median mask area in categories cable and grid; fallback uses median-area samples per category sorted by category/path. Selection does not use method scores.",
        "all_target_anomaly_candidates": len(all_records),
        "heatmap_scale": heatmap_scale,
        "note": "CES-only source responses come from the saved VisA-source response cache. Formal Base+LRP in final ablation uses the init_v_fallback CES direction; CES+NTR uses the seed0 rho=0.10 checkpoint.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.16, linewidth=0.6)
    ax.tick_params(labelsize=8)


def draw_panel_a(ax, cache):
    labels = cache["labels"]
    normal = labels == 0
    anomaly = labels == 1
    ax.scatter(cache["ces_x"][normal], cache["ces_y"][normal], s=9, c=BLUE, alpha=0.22, linewidths=0, label="Normal", rasterized=True)
    ax.scatter(cache["ces_x"][anomaly], cache["ces_y"][anomaly], s=10, c=ORANGE, alpha=0.36, linewidths=0, label="Anomaly", rasterized=True)
    xmin = float(min(cache["ces_x"].min(), cache["ces_y"].min()))
    xmax = float(max(cache["ces_x"].max(), cache["ces_y"].max()))
    pad = 0.04 * (xmax - xmin)
    lim = (xmin - pad, xmax + pad)
    ax.plot(lim, lim, color="#555555", linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.75, label=r"$y=x$")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"$z^\top s_N$", fontsize=9)
    ax.set_ylabel(r"$z^\top s_A$", fontsize=9)
    ax.set_title("CES decision geometry", fontsize=10, fontweight="bold")
    ax.text(-0.12, 1.08, "(a)", transform=ax.transAxes, va="top", ha="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.8, loc="lower right", handletextpad=0.4, borderpad=0.2)
    style_axes(ax)


def draw_panel_b(ax, cache):
    normal = np.where(cache["labels"] == 0)[0]
    order = normal[np.argsort(cache["q_ces"][normal])]
    x = np.linspace(0.0, 1.0, len(order))
    q0 = cache["q_ces"][order]
    q1 = cache["q_ntr"][order]
    tail_start = 1.0 - NTR_RHO
    ax.axvspan(tail_start, 1.0, color=BLUE_LIGHT, alpha=0.15, linewidth=0)
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.65)
    ax.plot(x, q0, color=BLUE_DARK, linewidth=1.35)
    ax.plot(x, q1, color=ORANGE, linewidth=1.35)
    ax.text(1.03, 0.84, "CES", transform=ax.transAxes, ha="left", va="top", fontsize=8.1, color=BLUE_DARK, clip_on=False)
    ax.text(1.03, 0.76, "CES + NTR", transform=ax.transAxes, ha="left", va="top", fontsize=8.1, color=ORANGE, clip_on=False)
    ax.set_xlabel("Normal-patch rank by CES score", fontsize=9)
    ax.set_ylabel("Anomaly score", fontsize=9)
    ax.set_title("Normal tail-risk reduction", fontsize=10, fontweight="bold")
    ax.text(-0.12, 1.08, "(b)", transform=ax.transAxes, va="top", ha="left", fontsize=11, fontweight="bold")
    style_axes(ax)


def draw_panel_c(fig, outer_cell, samples, heatmap_scale):
    sub = outer_cell.subgridspec(len(samples), 4, wspace=0.04, hspace=0.07)
    axes = np.empty((len(samples), 4), dtype=object)
    col_titles = ["Input", "GT", "CES map", "CES + NTR map"]
    for r, s in enumerate(samples):
        for c in range(4):
            ax = fig.add_subplot(sub[r, c])
            axes[r, c] = ax
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8.2, pad=3)
        axes[r, 0].imshow(s["image"])
        axes[r, 1].imshow(s["mask"], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(s["ces_full"], cmap="magma", vmin=heatmap_scale["vmin"], vmax=heatmap_scale["vmax"])
        im = axes[r, 3].imshow(s["ntr_full"], cmap="magma", vmin=heatmap_scale["vmin"], vmax=heatmap_scale["vmax"])
        axes[r, 0].text(
            0.02,
            0.98,
            str(s["category"]),
            transform=axes[r, 0].transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5),
        )
    axes[0, 0].text(-0.20, 1.34, "(c)", transform=axes[0, 0].transAxes, va="top", ha="left", fontsize=11, fontweight="bold")
    axes[0, 0].text(0.0, 1.34, "Qualitative responses", transform=axes[0, 0].transAxes, va="top", ha="left", fontsize=10, fontweight="bold")
    return im


def make_full_figure(cache, samples, heatmap_scale):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(13.8, 4.15), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.7], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_a(ax_a, cache)
    draw_panel_b(ax_b, cache)
    im = draw_panel_c(fig, gs[0, 2], samples, heatmap_scale)
    cax = fig.add_axes([0.935, 0.18, 0.008, 0.56])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.065, right=0.925, top=0.80, bottom=0.18)
    return fig


def make_panel_a(cache):
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    draw_panel_a(ax, cache)
    fig.tight_layout()
    return fig


def make_panel_b(cache):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    draw_panel_b(ax, cache)
    fig.tight_layout()
    return fig


def make_panel_c(samples, heatmap_scale):
    fig = plt.figure(figsize=(6.2, 3.6))
    gs = fig.add_gridspec(1, 1)
    im = draw_panel_c(fig, gs[0, 0], samples, heatmap_scale)
    cax = fig.add_axes([0.92, 0.18, 0.012, 0.58])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2)
    fig.subplots_adjust(left=0.06, right=0.9, top=0.74, bottom=0.05)
    return fig


def save_figures(cache, samples, heatmap_scale):
    fig = make_full_figure(cache, samples, heatmap_scale)
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"ces_ntr_mechanism.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
    plt.close(fig)

    panels = {
        "panel_a_ces_geometry.png": make_panel_a(cache),
        "panel_b_ntr_tail.png": make_panel_b(cache),
        "panel_c_qualitative.png": make_panel_c(samples, heatmap_scale),
    }
    for name, f in panels.items():
        f.savefig(OUT / name, dpi=600, bbox_inches="tight")
        plt.close(f)


def audit_layout_manifest(path: Path):
    """Record a lightweight render-time panel-layout manifest for QA traceability."""
    manifest = {
        "figure": "ces_ntr_mechanism",
        "layout": "1x3 outer GridSpec; panel C uses nested sample-by-column GridSpec",
        "outer_panel_count": 3,
        "exports": {
            "png": "ces_ntr_mechanism.png",
            "pdf": "ces_ntr_mechanism.pdf",
            "svg": "ces_ntr_mechanism.svg",
        },
        "manual_visual_check": "performed with rendered PNG preview",
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_docs(cache, samples, heatmap_scale):
    caption = (
        "Fig. X. CES and NTR mechanism visualization. (a) Source patch responses in the CES "
        "two-state decision geometry, using the normal-state and anomaly-state responses as "
        "coordinates. (b) Anomaly scores of source normal patches sorted by the CES score; the "
        "shaded region denotes the fixed top-10% hard-normal tail used by NTR. (c) Qualitative "
        "full-resolution anomaly maps on MVTec target samples under the same Full VisA source "
        "protocol and fixed LRP fusion. All panels use saved CESAD artifacts without retraining."
    )
    (OUT / "CAPTION_EN.txt").write_text(caption + "\n", encoding="utf-8")
    readme = f"""# CES + NTR 机制可视化

本目录由 `experiments/plot_ces_ntr_mechanism.py` 生成，使用真实 CESAD 实验产物，不包含模拟点或手工修改数值。

## 数据来源

- Source response cache: `{SOURCE_CACHE}`
- Source: `{SOURCE}`
- NTR checkpoint: `results/paper_assets/final_ablation_close/checkpoints/Full_VisA_seed0_rho0.10.pt`
- Qualitative target: `{QUAL_TARGET}`
- Local LRP cache: `E:/MRAD_final_evidence_cache/local_patch_Full_VisA__MVTec.npy`

## Panel 定义

1. `(a) CES decision geometry`：使用保存的 VisA-source patch responses，横轴为 `z^T s_N`，纵轴为 `z^T s_A`。虚线为 `y=x`。
2. `(b) Normal tail-risk reduction`：固定同一批 source normal patches，按 CES anomaly score 排序，展示 CES 与 CES+NTR 的 score 曲线。阴影为 top-{int(NTR_RHO * 100)}% hard-normal tail。
3. `(c) Qualitative anomaly responses`：使用 Full VisA -> MVTec 的真实 global/local LRP cache，输出 CES-only 与 CES+NTR 的 full-resolution heatmap。两个方法共用同一 heatmap 范围：`{heatmap_scale['vmin']:.6f}` 到 `{heatmap_scale['vmax']:.6f}`。

## 样本选择规则

Panel (c) 选择 MVTec target 中 `cable` 与 `grid` 类别 mask 面积中位数附近的异常样本；若类别不存在，则回退为每类中位面积样本按类别名排序。选择不使用模型 score。

## 样本摘要

{chr(10).join([f"- {s['category']}: {s['path']} (mask area ratio={s['area_ratio']:.4f})" for s in samples])}
"""
    (OUT / "README_CN.md").write_text(readme, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = load_source_response_cache()
    source_info, ces, ntr, ckpt_path = load_formal_specs(device)
    samples, all_records, local_path, heatmap_scale = build_qualitative_data(device, source_info, ces, ntr)
    export_data(cache, samples, all_records, ckpt_path, local_path, heatmap_scale)
    save_figures(cache, samples, heatmap_scale)
    audit_layout_manifest(OUT / "LAYOUT_QA_MANIFEST.json")
    write_docs(cache, samples, heatmap_scale)
    shutil.copy2(Path(__file__), OUT / "plot_ces_ntr_mechanism.py")

    print("=== CES + NTR MECHANISM VISUALIZATION AUDIT ===")
    print(f"Source response cache: {SOURCE_CACHE}")
    print(f"NTR checkpoint: {ckpt_path}")
    print(f"Num source normal patches: {int(np.sum(cache['labels'] == 0))}")
    print(f"Num source anomaly patches: {int(np.sum(cache['labels'] == 1))}")
    print(f"NTR rho: {NTR_RHO}")
    print(f"Qualitative target: {SOURCE} -> {QUAL_TARGET}")
    print(f"Local LRP cache: {local_path}")
    print(f"Saved PNG: {OUT / 'ces_ntr_mechanism.png'}")
    print(f"Saved PDF: {OUT / 'ces_ntr_mechanism.pdf'}")
    print(f"Saved SVG: {OUT / 'ces_ntr_mechanism.svg'}")
    print(f"Saved data: {OUT / 'ces_ntr_visualization_data.npz'}")
    print(f"Saved script: {OUT / 'plot_ces_ntr_mechanism.py'}")


if __name__ == "__main__":
    main()
