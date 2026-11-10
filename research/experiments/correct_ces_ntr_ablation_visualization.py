from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_ces_linear_head_control import (  # noqa: E402
    HeadSpec,
    collect_mvtec_pool_direct,
    collect_visa_pool_direct,
    get_target_cell,
    local_cache_path,
    load_source_pool_artifact,
    lrp_map,
    score_patch_grid,
)
from experiments.phase_paper_ready_final_validation import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    LR,
    SEEDS,
    balanced_batch,
    collect_mvtec_pool,
    collect_visa_pool,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (  # noqa: E402
    item_at,
    item_meta,
    mask_np,
    summarize_category,
    torch_resize,
)
from experiments.phase_private_residual_contrast_memory import contrast_logits  # noqa: E402
from experiments.phase_xlv_cross_domain_transfer import top1_image_score  # noqa: E402
from experiments.phase_xlii_patch_specialized_adaptation import balanced_loss  # noqa: E402


OUT = ROOT / "results" / "ces_ntr_visualization" / "corrected"
CKPT_DIR = OUT / "checkpoints"
FIGURE_SOURCE_CACHE = ROOT / "results" / "paper_assets" / "journal_figures" / "figure_state_compression_cache_visa_source.npz"
OLD_FINAL = ROOT / "results" / "ces_ntr_visualization" / "final"
FINAL_ABLATION = ROOT / "results" / "paper_assets" / "final_ablation_close"
PAPER_READY_CKPT = ROOT / "results" / "phase_paper_ready_final_validation" / "formal" / "checkpoints"
FINAL_CLOSE_CKPT = FINAL_ABLATION / "checkpoints"

PAIRS = (
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "VisA"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MVTec"),
)
VARIANTS = ("Trained_CES_Core", "Trained_CES_NTR", "Trained_CES_LRP", "Trained_CES_NTR_LRP")
SUMMARY_METRICS = ("I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO")
NTR_RHO = 0.10
BLUE = "#4C78A8"
BLUE_DARK = "#1F4E79"
ORANGE = "#F28E2B"
BLUE_LIGHT = "#8CB6D8"
SEED = 0

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False


def unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + eps)


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def source_info_for(source: str, seed: int, device: torch.device) -> dict:
    if source == "Full VisA":
        try:
            return collect_visa_pool(seed, device, smoke=False)
        except FileNotFoundError as exc:
            print(f"[Corrected] VisA formal source cache missing; trying preprocessed source artifact: {exc}", flush=True)
            pooled = load_source_pool_artifact(source, device, smoke=False)
            if pooled is not None:
                return pooled
            print("[Corrected] VisA source artifact missing; rebuilding frozen source pool directly.", flush=True)
            return collect_visa_pool_direct(seed, device, smoke=False)
    if source == "Full MVTec":
        try:
            return collect_mvtec_pool(seed, smoke=False)
        except FileNotFoundError as exc:
            print(f"[Corrected] MVTec formal source cache missing; trying preprocessed source artifact: {exc}", flush=True)
            pooled = load_source_pool_artifact(source, device, smoke=False)
            if pooled is not None:
                return pooled
            print("[Corrected] MVTec source artifact missing; rebuilding frozen source pool directly.", flush=True)
            return collect_mvtec_pool_direct(seed, device, smoke=False)
    raise ValueError(source)


def paper_ready_statealigned_path(source: str, seed: int) -> Path:
    return PAPER_READY_CKPT / f"{source.replace(' ', '_')}_seed{seed}_statealigned.pt"


def paper_ready_ntr_path(source: str, seed: int) -> Path:
    if seed == 0:
        return FINAL_CLOSE_CKPT / f"{source.replace(' ', '_')}_seed0_rho0.10.pt"
    return PAPER_READY_CKPT / f"{source.replace(' ', '_')}_seed{seed}_ntr.pt"


def corrected_statealigned_path(source: str, seed: int) -> Path:
    return CKPT_DIR / f"{source.replace(' ', '_')}_seed{seed}_statealigned.pt"


def train_statealigned_if_needed(source_info: dict, seed: int, device: torch.device) -> tuple[np.ndarray, str, str, list[dict]]:
    existing = paper_ready_statealigned_path(source_info["source"], seed)
    if existing.exists():
        ckpt = torch.load(existing, map_location="cpu", weights_only=False)
        return ckpt["v"].astype(np.float32), str(existing), "REUSED_EXISTING_PAPER_READY", []

    out = corrected_statealigned_path(source_info["source"], seed)
    if out.exists():
        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        return ckpt["v"].astype(np.float32), str(out), "REUSED_EXISTING_CORRECTED", []

    print(f"[Corrected] train missing no-NTR statealigned {source_info['source']} seed={seed}", flush=True)
    rng = np.random.default_rng(seed)
    groups = source_info["groups"]
    total_samples = sum(len(g["y"]) for g in groups.values())
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    param = torch.nn.Parameter(torch.tensor(source_info["init_v"], dtype=torch.float32, device=device))
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([param], lr=LR, weight_decay=0.0)
    logs = []
    for epoch in range(1, EPOCHS + 1):
        losses = []
        bces = []
        dices = []
        for _step in range(steps_per_epoch):
            xb, yb = balanced_batch(groups, rng)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(param, dim=-1)
            logits = contrast_logits(z, m, source_info["alpha"], v)
            task, bce, dice = balanced_loss(logits, y)
            loss = task
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            bces.append(float(bce.detach().cpu()))
            dices.append(float(dice.detach().cpu()))
        row = {
            "source": source_info["source"],
            "seed": seed,
            "epoch": epoch,
            "mode": "statealigned",
            "loss": float(np.mean(losses)),
            "bce": float(np.mean(bces)),
            "dice": float(np.mean(dices)),
            "ntr": 0.0,
        }
        logs.append(row)
        print(f"[Corrected] {source_info['source']} seed={seed} epoch={epoch}/{EPOCHS} loss={row['loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(param, dim=-1).detach().cpu().numpy().astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "seed": seed, "mode": "statealigned", "source": source_info["source"], "trained_by": "correct_ces_ntr_ablation_visualization"}, out)
    return v, str(out), "TRAINED_CORRECTED", logs


def load_ntr_v(source: str, seed: int) -> tuple[np.ndarray, str, str]:
    path = paper_ready_ntr_path(source, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing NTR checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["v"].astype(np.float32), str(path), "REUSED_EXISTING_NTR"


def simplify_row(row: dict, variant: str, source: str, target: str, seed: int, ckpt: str, cache: str, status: str) -> dict:
    out = {
        "variant": variant,
        "transfer": f"{source} -> {target}",
        "source": source,
        "target": target,
        "seed": seed,
        "checkpoint": ckpt,
        "cache": cache,
        "formal_status": status,
    }
    mapping = {
        "I_AUROC": ("M_FR_I_AUROC", "I-AUROC", "I_AUROC"),
        "I_AP": ("M_FR_I_AP", "I-AP", "I_AP"),
        "P_AUROC": ("M_FR_P_AUROC", "P-AUROC", "P_AUROC"),
        "P_AP": ("M_FR_P_AP", "P-AP", "P_AP"),
        "P_F1MAX": ("M_FR_P_F1max", "M_FR_P_F1MAX", "P-F1max", "P_F1MAX"),
        "AUPRO": ("M_FR_PRO", "M_FR_AUPRO", "PRO", "AUPRO"),
    }
    for dst, candidates in mapping.items():
        val = ""
        for c in candidates:
            if c in row:
                val = row[c]
                break
        out[dst] = float(val) if val != "" else float("nan")
    return out


def evaluate_maps_global(source: str, target: str, maps_by_method: dict[str, list[np.ndarray]], metas: list[dict]) -> list[dict]:
    rows = []
    labels = [int(m["label"]) for m in metas]
    masks = [m["mask"] for m in metas]
    for variant, maps in maps_by_method.items():
        image_scores = [top1_image_score(m) for m in maps]
        full_scores = [torch_resize(s, m.shape, align_corners=True) for s, m in zip(maps, masks)]
        metrics = summarize_category(labels, image_scores, masks, full_scores)
        row = {
            "source": source,
            "dataset": target,
            "variant": variant,
            "aggregation": "DATASET_GLOBAL",
            "protocol": "M_FR",
            "score_resolution": f"{maps[0].shape[0]}x{maps[0].shape[1]}",
        }
        for met, val in metrics.items():
            row[f"M_FR_{met.replace('-', '_')}"] = val
        rows.append(row)
    return rows


def evaluate_core_heads(source: str, target: str, source_info: dict, specs: dict[str, tuple[HeadSpec, str, str]], device: torch.device) -> list[dict]:
    _cell_id, _target, dataset, idxs, patch, kind = get_target_cell(source, target, device, smoke=False)
    maps_by_method = {name: [] for name in specs}
    metas = []
    for pos, ref in enumerate(idxs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, _path = item_meta(item, kind, ref, target)
        metas.append({"category": cat, "label": int(label), "mask": mask_np(item)})
        global_patch = patch[pos]
        for method, (spec, _ckpt, _status) in specs.items():
            maps_by_method[method].append(score_patch_grid(source_info, spec, global_patch, device))
    return evaluate_maps_global(source, target, maps_by_method, metas)


def evaluate_lrp_heads_global(
    source: str,
    target: str,
    source_info: dict,
    specs: dict[str, HeadSpec],
    device: torch.device,
) -> tuple[list[dict], str]:
    _cell_id, _target, dataset, idxs, patch, kind = get_target_cell(source, target, device, smoke=False)
    lpath = local_cache_path(source, target)
    if not lpath.exists():
        raise FileNotFoundError(f"Missing local LRP cache: {lpath}")
    local_cache = np.load(lpath, mmap_mode="r")
    n = len(idxs)
    if local_cache.shape[0] < n:
        raise RuntimeError(f"Local cache too short for {source}->{target}: {local_cache.shape[0]} < {n}")
    maps_by_method = {name: [] for name in specs}
    metas = []
    for pos, ref in enumerate(idxs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, _path = item_meta(item, kind, ref, target)
        metas.append({"category": cat, "label": int(label), "mask": mask_np(item)})
        global_patch = patch[pos]
        local_patches = np.asarray(local_cache[pos], dtype=np.float32)
        for method, spec in specs.items():
            maps_by_method[method].append(lrp_map(source_info, spec, global_patch, local_patches, device))
    return evaluate_maps_global(source, target, maps_by_method, metas), str(lpath)


def evaluate_all(device: torch.device):
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "CORRECTED_ABLATION_RAW.csv"
    cfg_path = OUT / "CHECKPOINT_CONFIG_COMPARISON.csv"
    log_path = OUT / "TRAINING_LOG_STATEALIGNED_SEED0.csv"
    all_rows = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
    cfg_rows = pd.read_csv(cfg_path).to_dict("records") if cfg_path.exists() else []
    train_logs = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    done = {(r["source"], int(r["seed"]), r["target"], r["variant"]) for r in all_rows}
    cfg_done = {(r["source"], int(r["seed"]), r["variant"]) for r in cfg_rows} if cfg_rows else set()
    source_infos: dict[tuple[str, int], dict] = {}
    for source in ("Full MVTec", "Full VisA"):
        for seed in SEEDS:
            info = source_info_for(source, seed, device)
            source_infos[(source, seed)] = info
            state_v, state_ckpt, state_status, logs = train_statealigned_if_needed(info, seed, device)
            train_logs.extend(logs)
            ntr_v, ntr_ckpt, ntr_status = load_ntr_v(source, seed)
            cos_state_init = float(np.dot(unit(info["init_v"]), unit(state_v)))
            cos_ntr_init = float(np.dot(unit(info["init_v"]), unit(ntr_v)))
            cos_state_ntr = float(np.dot(unit(state_v), unit(ntr_v)))
            for cfg_row in (
                {
                    "source": source,
                    "seed": seed,
                    "variant": "Trained_CES",
                    "checkpoint": state_ckpt,
                    "status": state_status,
                    "uses_base_loss": "YES",
                    "uses_ntr": "NO",
                    "epochs": EPOCHS,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "cos_to_init_v": cos_state_init,
                    "cos_to_ntr_v": cos_state_ntr,
                },
                {
                    "source": source,
                    "seed": seed,
                    "variant": "Trained_CES_NTR",
                    "checkpoint": ntr_ckpt,
                    "status": ntr_status,
                    "uses_base_loss": "YES",
                    "uses_ntr": "YES",
                    "epochs": EPOCHS,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "cos_to_init_v": cos_ntr_init,
                    "cos_to_ntr_v": 1.0,
                },
            ):
                if (source, seed, cfg_row["variant"]) not in cfg_done:
                    cfg_rows.append(cfg_row)
                    cfg_done.add((source, seed, cfg_row["variant"]))
            write_csv(cfg_path, cfg_rows)
            if logs:
                write_csv(log_path, train_logs)

            pairs = [target for s, target in PAIRS if s == source]
            for target in pairs:
                core_specs = {
                    "Trained_CES_Core": (HeadSpec("CES", {"m": info["m"], "alpha": info["alpha"], "v": state_v}, source, seed=seed), state_ckpt, state_status),
                    "Trained_CES_NTR": (HeadSpec("CES", {"m": info["m"], "alpha": info["alpha"], "v": ntr_v}, source, seed=seed), ntr_ckpt, ntr_status),
                }
                missing_core = {k: v for k, v in core_specs.items() if (source, seed, target, k) not in done}
                if missing_core:
                    core_rows = evaluate_core_heads(source, target, info, missing_core, device)
                    for row in core_rows:
                        spec = core_specs[row["variant"]]
                        out_row = simplify_row(row, row["variant"], source, target, seed, spec[1], "target_patch_cache", spec[2])
                        all_rows.append(out_row)
                        done.add((source, seed, target, row["variant"]))
                    write_csv(raw_path, all_rows)
                else:
                    print(f"[Corrected] skip completed core {source} seed={seed} -> {target}", flush=True)

                lrp_specs = {
                    "Trained_CES_LRP": HeadSpec("CES", {"m": info["m"], "alpha": info["alpha"], "v": state_v}, source, seed=seed),
                    "Trained_CES_NTR_LRP": HeadSpec("CES", {"m": info["m"], "alpha": info["alpha"], "v": ntr_v}, source, seed=seed),
                }
                missing_lrp = {k: v for k, v in lrp_specs.items() if (source, seed, target, k) not in done}
                if missing_lrp:
                    lrp_rows, lpath = evaluate_lrp_heads_global(source, target, info, missing_lrp, device)
                    for row in lrp_rows:
                        if row["variant"] == "Trained_CES_LRP":
                            ckpt, status = state_ckpt, state_status
                        else:
                            ckpt, status = ntr_ckpt, ntr_status
                        all_rows.append(simplify_row(row, row["variant"], source, target, seed, ckpt, lpath, status))
                        done.add((source, seed, target, row["variant"]))
                    write_csv(raw_path, all_rows)
                else:
                    print(f"[Corrected] skip completed LRP {source} seed={seed} -> {target}", flush=True)
    return all_rows, cfg_rows, train_logs, source_infos


def build_summary(rows: list[dict]) -> list[dict]:
    out = []
    for variant in VARIANTS:
        part = [r for r in rows if r["variant"] == variant]
        row = {"variant": variant, "num_cells": len(part)}
        for m in SUMMARY_METRICS:
            vals = np.asarray([float(r[m]) for r in part], dtype=np.float64)
            row[f"{m}_cell_mean"] = float(vals.mean()) if vals.size else float("nan")
            row[f"{m}_cell_std"] = float(vals.std(ddof=0)) if vals.size else float("nan")
            # Match the previous table's convention: average per-transfer seed means and average per-transfer seed stds.
            transfer_means = []
            transfer_stds = []
            for source, target in PAIRS:
                arr = np.asarray([float(r[m]) for r in part if r["source"] == source and r["target"] == target], dtype=np.float64)
                if arr.size:
                    transfer_means.append(float(arr.mean()))
                    transfer_stds.append(float(arr.std(ddof=0)))
            row[f"{m}_mean"] = float(np.mean(transfer_means)) if transfer_means else float("nan")
            row[f"{m}_std"] = float(np.mean(transfer_stds)) if transfer_stds else float("nan")
            row[m] = f"{row[f'{m}_mean']:.2f} ± {row[f'{m}_std']:.2f}" if np.isfinite(row[f"{m}_mean"]) else "NA"
        out.append(row)
    return out


def per_transfer_effect(rows: list[dict]) -> list[dict]:
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["variant"], r["source"], r["target"]), []).append(r)
    for source, target in PAIRS:
        row = {"source": source, "target": target, "transfer": f"{source} -> {target}"}
        means = {}
        for variant in VARIANTS:
            means[variant] = {}
            for m in ("P_AP", "P_F1MAX", "AUPRO"):
                vals = [float(r[m]) for r in by.get((variant, source, target), [])]
                means[variant][m] = float(np.mean(vals)) if vals else float("nan")
                row[f"{variant}_{m}"] = means[variant][m]
        for m in ("P_AP", "P_F1MAX", "AUPRO"):
            row[f"Delta_NTR_{m}"] = means["Trained_CES_NTR"][m] - means["Trained_CES_Core"][m]
            row[f"Delta_LRP_base_{m}"] = means["Trained_CES_LRP"][m] - means["Trained_CES_Core"][m]
            row[f"Delta_LRP_after_NTR_{m}"] = means["Trained_CES_NTR_LRP"][m] - means["Trained_CES_NTR"][m]
            row[f"Delta_NTR_with_LRP_{m}"] = means["Trained_CES_NTR_LRP"][m] - means["Trained_CES_LRP"][m]
            row[f"Delta_Final_{m}"] = means["Trained_CES_NTR_LRP"][m] - means["Trained_CES_Core"][m]
        out.append(row)
    return out


def state_pair(m: np.ndarray, alpha: float, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_a = unit(np.asarray(m, dtype=np.float32) + float(alpha) * unit(v))
    s_n = unit(np.asarray(m, dtype=np.float32) - float(alpha) * unit(v))
    return s_n, s_a


def source_response_table(source_info: dict, state_v: np.ndarray, ntr_v: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    data = np.load(FIGURE_SOURCE_CACHE, allow_pickle=True)
    z = data["z"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    meta = data["meta"]
    s_n0, s_a0 = state_pair(source_info["m"], source_info["alpha"], state_v)
    s_n1, s_a1 = state_pair(source_info["m"], source_info["alpha"], ntr_v)
    ces_x = z @ s_n0
    ces_y = z @ s_a0
    ntr_x = z @ s_n1
    ntr_y = z @ s_a1
    q_ces = ces_y - ces_x
    q_ntr = ntr_y - ntr_x
    normal = np.where(labels == 0)[0]
    k = max(1, int(np.ceil(NTR_RHO * len(normal))))
    hard_idx = normal[np.argsort(q_ces[normal])[-k:]]
    hard = np.zeros(len(labels), dtype=bool)
    hard[hard_idx] = True
    rows = []
    for i in range(len(labels)):
        m = meta[i].tolist() if hasattr(meta[i], "tolist") else list(meta[i])
        rows.append(
            {
                "patch_id": i,
                "label": int(labels[i]),
                "category": m[2] if len(m) > 2 else "",
                "ces_sn": float(ces_x[i]),
                "ces_sa": float(ces_y[i]),
                "ntr_sn": float(ntr_x[i]),
                "ntr_sa": float(ntr_y[i]),
                "q_ces": float(q_ces[i]),
                "q_ntr": float(q_ntr[i]),
                "hard_normal": bool(hard[i]),
            }
        )
    return pd.DataFrame(rows), z


def score_stats(df: pd.DataFrame) -> list[dict]:
    labels = df["label"].to_numpy(int)
    hard = df["hard_normal"].astype(bool).to_numpy()
    rows = []
    for col, name in [("q_ces", "Trained_CES"), ("q_ntr", "Trained_CES_NTR")]:
        s = df[col].to_numpy(float)
        normal = labels == 0
        anomaly = labels == 1
        rows.append(
            {
                "variant": name,
                "mean_normal_score": float(s[normal].mean()),
                "mean_anomalous_score": float(s[anomaly].mean()),
                "mean_high_risk_normal_score": float(s[hard].mean()),
                "anomaly_normal_mean_gap": float(s[anomaly].mean() - s[normal].mean()),
                "anomaly_high_risk_normal_mean_gap": float(s[anomaly].mean() - s[hard].mean()),
                "source_patch_AUROC": float(roc_auc_score(labels, s)),
                "source_patch_AP": float(average_precision_score(labels, s)),
            }
        )
    delta = {"variant": "Delta_NTR_minus_CES"}
    for key in rows[0]:
        if key != "variant":
            delta[key] = rows[1][key] - rows[0][key]
    rows.append(delta)
    return rows


def qualitative_maps(source_info: dict, state_v: np.ndarray, ntr_v: np.ndarray, device: torch.device):
    # Reuse the original selected Cable/Grid sample IDs so panel C is exactly comparable.
    old_q = pd.read_csv(OLD_FINAL / "qualitative_samples.csv")
    old_meta = json.loads((OLD_FINAL / "metadata.json").read_text(encoding="utf-8"))
    from experiments.phase_ces_linear_head_control import direct_target_global_cache, local_cache_path, lrp_map
    from PIL import Image

    _cell, target, dataset, idxs, patch, kind = direct_target_global_cache("MVTec", device, smoke=False)
    local = np.load(local_cache_path("Full VisA", "MVTec"), mmap_mode="r")
    specs = {
        "Trained_CES": HeadSpec("CES", {"m": source_info["m"], "alpha": source_info["alpha"], "v": state_v}, "Full VisA", seed=0),
        "Trained_CES_NTR": HeadSpec("CES", {"m": source_info["m"], "alpha": source_info["alpha"], "v": ntr_v}, "Full VisA", seed=0),
    }
    samples = []
    all_maps = []
    for _, r in old_q.drop_duplicates("sample_id").iterrows():
        pos = int(r["pos"])
        ref = idxs[pos]
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, "MVTec")
        mask = mask_np(item)
        img_path = Path(path)
        if not img_path.is_absolute():
            img_path = ROOT / img_path
        image = np.asarray(Image.open(img_path).convert("RGB").resize((mask.shape[1], mask.shape[0])))
        maps = {}
        for name, spec in specs.items():
            m74 = lrp_map(source_info, spec, patch[pos], np.asarray(local[pos], dtype=np.float32), device)
            full = torch_resize(m74, mask.shape, align_corners=True)
            maps[name] = full.astype(np.float32)
            all_maps.append(full.reshape(-1))
        samples.append({"pos": pos, "category": cat, "path": path, "mask": mask, "image": image, "maps": maps})
    flat = np.concatenate(all_maps)
    scale = {"vmin": float(np.percentile(flat, 1.0)), "vmax": float(np.percentile(flat, 99.0)), "align_corners": True}
    rows = []
    for sid, sample in enumerate(samples):
        y = sample["mask"].reshape(-1).astype(int)
        for name, arr in sample["maps"].items():
            scores = arr.reshape(-1)
            rows.append(
                {
                    "sample_id": sid,
                    "source": "Full VisA",
                    "target": "MVTec",
                    "category": sample["category"],
                    "img_path": sample["path"],
                    "variant": name,
                    "pixel_AP": float(average_precision_score(y, scores)),
                    "normal_background_mean_response": float(scores[~y.astype(bool)].mean()),
                    "gt_anomaly_mean_response": float(scores[y.astype(bool)].mean()),
                    "heatmap_vmin": scale["vmin"],
                    "heatmap_vmax": scale["vmax"],
                    "same_color_range": True,
                    "per_map_minmax_normalized": False,
                }
            )
    return samples, rows, scale


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.16, linewidth=0.6)
    ax.tick_params(labelsize=8)


def draw_corrected_figure(resp: pd.DataFrame, samples: list[dict], scale: dict):
    fig = plt.figure(figsize=(13.8, 4.15), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.7], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    labels = resp["label"].to_numpy(int)
    normal = labels == 0
    anomaly = labels == 1
    ax_a.scatter(resp.loc[normal, "ces_sn"], resp.loc[normal, "ces_sa"], s=9, c=BLUE, alpha=0.22, linewidths=0, label="Normal", rasterized=True)
    ax_a.scatter(resp.loc[anomaly, "ces_sn"], resp.loc[anomaly, "ces_sa"], s=10, c=ORANGE, alpha=0.36, linewidths=0, label="Anomaly", rasterized=True)
    xmin = float(min(resp["ces_sn"].min(), resp["ces_sa"].min()))
    xmax = float(max(resp["ces_sn"].max(), resp["ces_sa"].max()))
    pad = 0.04 * (xmax - xmin)
    lim = (xmin - pad, xmax + pad)
    ax_a.plot(lim, lim, color="#555555", linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.75, label=r"$y=x$")
    ax_a.set_xlim(lim)
    ax_a.set_ylim(lim)
    ax_a.set_xlabel(r"$z^\top s_N$", fontsize=9)
    ax_a.set_ylabel(r"$z^\top s_A$", fontsize=9)
    ax_a.set_title("Trained CES geometry", fontsize=10, fontweight="bold")
    ax_a.text(-0.12, 1.08, "(a)", transform=ax_a.transAxes, va="top", ha="left", fontsize=11, fontweight="bold")
    style_axes(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    normal_ids = resp.index[normal].to_numpy()
    order = normal_ids[np.argsort(resp.loc[normal_ids, "q_ces"].to_numpy())]
    x = np.linspace(0.0, 1.0, len(order))
    q0 = resp.loc[order, "q_ces"].to_numpy(float)
    q1 = resp.loc[order, "q_ntr"].to_numpy(float)
    ax_b.axvspan(1.0 - NTR_RHO, 1.0, color=BLUE_LIGHT, alpha=0.15, linewidth=0)
    ax_b.axhline(0, color="#666666", linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.65)
    ax_b.plot(x, q0, color=BLUE_DARK, linewidth=1.35)
    ax_b.plot(x, q1, color=ORANGE, linewidth=1.35)
    ax_b.text(1.03, 0.84, "Trained CES", transform=ax_b.transAxes, ha="left", va="top", fontsize=8.1, color=BLUE_DARK, clip_on=False)
    ax_b.text(1.03, 0.76, "CES + NTR", transform=ax_b.transAxes, ha="left", va="top", fontsize=8.1, color=ORANGE, clip_on=False)
    ax_b.set_xlabel("Normal-patch rank by trained CES score", fontsize=9)
    ax_b.set_ylabel("Anomaly score", fontsize=9)
    ax_b.set_title("Normal tail-risk reduction", fontsize=10, fontweight="bold")
    ax_b.text(-0.12, 1.08, "(b)", transform=ax_b.transAxes, va="top", ha="left", fontsize=11, fontweight="bold")
    style_axes(ax_b)

    sub = gs[0, 2].subgridspec(len(samples), 4, wspace=0.04, hspace=0.07)
    axes = np.empty((len(samples), 4), dtype=object)
    col_titles = ["Input", "GT", "Trained CES", "CES + NTR"]
    for r, sample in enumerate(samples):
        for c in range(4):
            ax = fig.add_subplot(sub[r, c])
            axes[r, c] = ax
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8.2, pad=3)
        axes[r, 0].imshow(sample["image"])
        axes[r, 1].imshow(sample["mask"], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(sample["maps"]["Trained_CES"], cmap="magma", vmin=scale["vmin"], vmax=scale["vmax"])
        im = axes[r, 3].imshow(sample["maps"]["Trained_CES_NTR"], cmap="magma", vmin=scale["vmin"], vmax=scale["vmax"])
        axes[r, 0].text(0.02, 0.98, str(sample["category"]), transform=axes[r, 0].transAxes, ha="left", va="top", fontsize=7.5, color="white", bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5))
    axes[0, 0].text(-0.20, 1.34, "(c)", transform=axes[0, 0].transAxes, va="top", ha="left", fontsize=11, fontweight="bold")
    axes[0, 0].text(0.0, 1.34, "Qualitative responses", transform=axes[0, 0].transAxes, va="top", ha="left", fontsize=10, fontweight="bold")
    cax = fig.add_axes([0.935, 0.18, 0.008, 0.56])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.065, right=0.925, top=0.80, bottom=0.18)
    return fig


def write_report(rows, summary, per_effect, cfg, score_stats_rows, qual_rows, start_time):
    old = pd.read_csv(FINAL_ABLATION / "FACTORIAL_ABLATION.csv")
    old_summary = pd.read_csv(FINAL_ABLATION / "FACTORIAL_ABLATION_SUMMARY.csv")
    old_focus = old[(old["source"] == "Full VisA") & (old["target"] == "MVTec") & (old["seed"] == 0) & (old["variant"].isin(["Base+LRP", "Base+NTR+LRP"]))]
    old_text = old_focus[["variant", "P_AP", "checkpoint", "cache"]].to_markdown(index=False)
    summary_text = pd.DataFrame(summary)[["variant", "I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO"]].to_markdown(index=False)
    per_text = pd.DataFrame(per_effect).to_markdown(index=False)
    cfg_text = pd.DataFrame(cfg).to_markdown(index=False)
    score_text = pd.DataFrame(score_stats_rows).to_markdown(index=False)
    qual_text = pd.DataFrame(qual_rows).to_markdown(index=False)
    duration = time.time() - start_time

    # Determine if NTR remains positive in corrected fair comparison.
    by_variant = {r["variant"]: r for r in summary}
    d_ntr = by_variant["Trained_CES_NTR"]["P_AP_mean"] - by_variant["Trained_CES_Core"]["P_AP_mean"]
    d_ntr_lrp = by_variant["Trained_CES_NTR_LRP"]["P_AP_mean"] - by_variant["Trained_CES_LRP"]["P_AP_mean"]
    conclusion = "positive" if d_ntr > 0 and d_ntr_lrp > 0 else "mixed_or_negative"

    report = f"""# CESAD NTR 消融修复与可视化重新验证

## Scope

This corrected run fixes the previous `init_v_fallback` issue. The no-NTR CES side now uses trained `statealigned` checkpoints. Missing seed0 no-NTR checkpoints were trained into this corrected output directory using the same base CES loss code path as `phase_paper_ready_final_validation.py`.

No old evidence was overwritten. Outputs are under:

`{OUT}`

## 1. Checkpoint Audit

{cfg_text}

Key finding:

- Existing seed1/2 no-NTR checkpoints were reused from `results/phase_paper_ready_final_validation/formal/checkpoints/*_statealigned.pt`.
- Seed0 no-NTR checkpoints were missing and were trained into `results/ces_ntr_visualization/corrected/checkpoints/`.
- Existing seed0 NTR checkpoints were reused from `results/paper_assets/final_ablation_close/checkpoints/*_rho0.10.pt`.
- Existing seed1/2 NTR checkpoints were reused from `results/phase_paper_ready_final_validation/formal/checkpoints/*_ntr.pt`.

## 2. Old Problematic Reference

The previous figure and close ablation used `init_v_fallback` for `Base+LRP`:

{old_text}

This was not a trained no-NTR checkpoint and should not be described as a strict NTR-only ablation.

## 3. Corrected 2x2 Component Ablation

Metrics are M_FR full-resolution metrics on the matched four transfers:

- Full MVTec -> MPDD
- Full MVTec -> VisA
- Full VisA -> BTAD
- Full VisA -> MVTec

Summary:

{summary_text}

Per-transfer effects:

{per_text}

NTR main effect on P-AP without LRP: `{d_ntr:+.4f}`.

NTR main effect on P-AP with LRP: `{d_ntr_lrp:+.4f}`.

Corrected NTR conclusion: `{conclusion}`.

## 4. Corrected Source Patch Statistics for Visualization

{score_text}

## 5. Corrected Qualitative Sample Check

{qual_text}

The corrected panel (c) uses the same Cable/Grid samples as the previous figure, the same Full VisA -> MVTec protocol, the same LRP function, and one shared color range.

## 6. Answers to Required Questions

### Q1. 已有 no-NTR 权重能否直接复用？

Partially. Seed1/2 `*_statealigned.pt` checkpoints existed and were reused. Seed0 no-NTR checkpoints were missing, so they were trained in this corrected directory using the same base CES loss path and no NTR branch.

### Q2. 新旧组件消融数值分别是多少？

Old seed0 Full VisA->MVTec Base+LRP/Base+NTR+LRP are shown above. The corrected full 4-transfer multiseed table is in `CORRECTED_ABLATION_SUMMARY.csv` and summarized in Section 3.

### Q3. NTR 在公平对照下是否仍然提高目标域性能？

Corrected mean P-AP delta without LRP is `{d_ntr:+.4f}` and with LRP is `{d_ntr_lrp:+.4f}`. Therefore the corrected conclusion is `{conclusion}`.

### Q4. 图 (b)、(c) 是否可以作为 NTR 独立作用的证据？

Yes, after correction, because the CES side uses a trained no-NTR checkpoint and the NTR side uses the matched trained NTR checkpoint. The figure should still be worded as source-tail suppression plus illustrative target maps, not as a standalone proof without the corrected table.

### Q5. 论文中哪些句子、表格或图注需要同步修改？

1. Replace the old `init_v_fallback` component ablation values with the corrected table.
2. Update the mechanism figure and caption to say `trained CES` rather than `CES-only fallback`.
3. Remove or qualify any sentence that used the old Base+LRP vs Base+NTR+LRP numbers as a strict NTR-only ablation.
4. Mention that the corrected no-NTR baseline uses `statealigned` checkpoints trained with the base CES loss and NTR disabled.

Runtime: `{duration:.1f}s`.
"""
    (OUT / "CORRECTION_REPORT.md").write_text(report, encoding="utf-8")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, cfg, logs, source_infos = evaluate_all(device)
    summary = build_summary(rows)
    per_effect = per_transfer_effect(rows)

    # Visualization uses Full VisA seed0.
    visa0 = source_infos[("Full VisA", 0)]
    state0_path = corrected_statealigned_path("Full VisA", 0)
    if state0_path.exists():
        state0 = torch.load(state0_path, map_location="cpu", weights_only=False)["v"].astype(np.float32)
    else:
        state0 = torch.load(paper_ready_statealigned_path("Full VisA", 0), map_location="cpu", weights_only=False)["v"].astype(np.float32)
    ntr0 = torch.load(paper_ready_ntr_path("Full VisA", 0), map_location="cpu", weights_only=False)["v"].astype(np.float32)
    resp, _z = source_response_table(visa0, state0, ntr0)
    score_rows = score_stats(resp)
    samples, qual_rows, scale = qualitative_maps(visa0, state0, ntr0, device)

    write_csv(OUT / "CORRECTED_ABLATION_RAW.csv", rows)
    write_csv(OUT / "CORRECTED_ABLATION_SUMMARY.csv", summary)
    write_csv(OUT / "CORRECTED_PER_TRANSFER_EFFECT.csv", per_effect)
    write_csv(OUT / "CHECKPOINT_CONFIG_COMPARISON.csv", cfg)
    write_csv(OUT / "TRAINING_LOG_STATEALIGNED_SEED0.csv", logs)
    write_csv(OUT / "NTR_SCORE_STATISTICS.csv", score_rows)
    write_csv(OUT / "QUALITATIVE_SAMPLE_CHECK.csv", qual_rows)
    resp.to_csv(OUT / "corrected_source_patch_responses.csv", index=False)

    np.savez_compressed(
        OUT / "corrected_visualization_data.npz",
        labels=resp["label"].to_numpy(),
        q_ces=resp["q_ces"].to_numpy(),
        q_ntr=resp["q_ntr"].to_numpy(),
        hard_normal=resp["hard_normal"].to_numpy(),
        **{f"sample_{i}_{key}": val for i, s in enumerate(samples) for key, val in [("image", s["image"]), ("mask", s["mask"]), ("ces", s["maps"]["Trained_CES"]), ("ntr", s["maps"]["Trained_CES_NTR"])]},
    )

    fig = draw_corrected_figure(resp, samples, scale)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(OUT / f"ces_ntr_mechanism_corrected.{ext}", **kwargs)
    plt.close(fig)
    shutil.copy2(Path(__file__), OUT / "correct_ces_ntr_ablation_visualization.py")
    (OUT / "metadata.json").write_text(json.dumps({"source": "Full VisA", "target": "MVTec", "ntr_rho": NTR_RHO, "heatmap_scale": scale}, indent=2, default=json_default), encoding="utf-8")
    write_report(rows, summary, per_effect, cfg, score_rows, qual_rows, start)
    print("=== CORRECTED CES/NTR ABLATION + VISUALIZATION COMPLETE ===")
    print(f"Output: {OUT}")
    print(f"Summary: {OUT / 'CORRECTED_ABLATION_SUMMARY.csv'}")
    print(f"Figure: {OUT / 'ces_ntr_mechanism_corrected.png'}")
    print(f"Report: {OUT / 'CORRECTION_REPORT.md'}")


if __name__ == "__main__":
    main()
