from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_paper_ready_final_validation import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    LR,
    PATCH_SAMPLE_PER_CLASS_CATEGORY,
    SIDE,
    TOP_NEG_FRAC,
    balanced_batch,
    collect_mvtec_pool,
    collect_visa_pool,
    protocol_cells,
)
from experiments.phase_final_cross_dataset_benchmark import ready_holdout_cells  # noqa: E402
from experiments.phase_defect_query_transfer_pilot import light_fold  # noqa: E402
from experiments.phase_liv_memory_support_modes import external_cell  # noqa: E402
from experiments.phase_x0a_mandatory_gates import load_components  # noqa: E402
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT  # noqa: E402
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid  # noqa: E402
from experiments.phase_xlii_patch_specialized_adaptation import balanced_loss  # noqa: E402
from experiments.phase_xxxviii_visual_state_representation import patch_labels  # noqa: E402
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs  # noqa: E402
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (  # noqa: E402
    evaluate_maps,
    item_at,
    item_meta,
    mask_np,
    torch_resize,
)
from experiments.phase_xxxviii_visual_state_representation import norm_np  # noqa: E402


OUT = ROOT / "results" / "phase_ces_linear_head_control"
CKPT_DIR = OUT / "checkpoints"
FORMAL_CLOSE = ROOT / "results" / "paper_assets" / "final_ablation_close"
FORMAL_CKPT_DIR = FORMAL_CLOSE / "checkpoints"
LOCAL_CACHE_DIR = Path("E:/MRAD_final_evidence_cache")

PAIRS = [
    ("Full MVTec", "VisA"),
    ("Full MVTec", "MPDD"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MVTec"),
]

METHODS = ["LinearHead", "NormalizedLinearHead", "CES"]
SEED = 0
RHO = 0.10
WEIGHT_DECAY = 0.0
PREFLIGHT_TOL_PAP = 0.05
MFR_KEYS = {
    "I_AUROC": "M_FR_I_AUROC",
    "I_AP": "M_FR_I_AP",
    "P_AUROC": "M_FR_P_AUROC",
    "P_AP": "M_FR_P_AP",
    "P_F1MAX": "M_FR_P_F1max",
    "AUPRO": "M_FR_PRO",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


def pair_slug(source: str, target: str) -> str:
    return f"{source.replace(' ', '_')}__{target.replace(' ', '_')}"


def unit_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + eps)


def state_pair_from_m_alpha_v(m: np.ndarray, alpha: float, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s_a = unit_np(m + float(alpha) * v)
    s_n = unit_np(m - float(alpha) * v)
    return s_n.astype(np.float32), s_a.astype(np.float32)


def ces_weight_np(m: np.ndarray, alpha: float, v: np.ndarray) -> np.ndarray:
    s_n, s_a = state_pair_from_m_alpha_v(m, alpha, v)
    return (s_a - s_n).astype(np.float32)


@dataclass
class HeadSpec:
    method: str
    params: Dict[str, np.ndarray | float]
    source: str
    seed: int = SEED

    def weight(self) -> np.ndarray:
        if self.method == "LinearHead":
            return np.asarray(self.params["w"], dtype=np.float32)
        if self.method == "NormalizedLinearHead":
            u = unit_np(np.asarray(self.params["u"], dtype=np.float32))
            return float(self.params["c0"]) * u
        if self.method == "CES":
            return ces_weight_np(
                np.asarray(self.params["m"], dtype=np.float32),
                float(self.params["alpha"]),
                unit_np(np.asarray(self.params["v"], dtype=np.float32)),
            )
        raise ValueError(self.method)

    def bias(self) -> float:
        if self.method == "LinearHead":
            return float(self.params.get("b", 0.0))
        return 0.0

    def logits_np(self, z: np.ndarray) -> np.ndarray:
        return z.astype(np.float32) @ self.weight().astype(np.float32) + self.bias()


class LinearHead(torch.nn.Module):
    def __init__(self, w0: np.ndarray):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.w + self.b

    def to_spec(self, source: str) -> HeadSpec:
        return HeadSpec("LinearHead", {"w": self.w.detach().cpu().numpy(), "b": float(self.b.detach().cpu())}, source)


class NormalizedLinearHead(torch.nn.Module):
    def __init__(self, w0: np.ndarray, c0: float):
        super().__init__()
        self.u = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.register_buffer("c0", torch.tensor(float(c0), dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        u = F.normalize(self.u, dim=0)
        return self.c0 * (z @ u)

    def to_spec(self, source: str) -> HeadSpec:
        return HeadSpec(
            "NormalizedLinearHead",
            {"u": self.u.detach().cpu().numpy(), "c0": float(self.c0.detach().cpu())},
            source,
        )


class CESHead(torch.nn.Module):
    def __init__(self, m: np.ndarray, alpha: float, v0: np.ndarray):
        super().__init__()
        self.register_buffer("m", torch.tensor(m, dtype=torch.float32))
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        self.v = torch.nn.Parameter(torch.tensor(v0, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = F.normalize(self.v, dim=0)
        s_a = F.normalize(self.m + self.alpha * v, dim=0)
        s_n = F.normalize(self.m - self.alpha * v, dim=0)
        return z @ (s_a - s_n)

    def to_spec(self, source: str) -> HeadSpec:
        return HeadSpec(
            "CES",
            {
                "m": self.m.detach().cpu().numpy(),
                "alpha": float(self.alpha.detach().cpu()),
                "v": F.normalize(self.v.detach(), dim=0).cpu().numpy(),
            },
            source,
        )


def source_key(source: str) -> str:
    return source.replace("Full ", "").lower()


def load_source_info(source: str, device: torch.device, smoke: bool):
    if source == "Full MVTec":
        try:
            return collect_mvtec_pool(SEED, smoke)
        except FileNotFoundError as exc:
            print(f"[source] formal MVTec cache missing; trying source-pool artifact: {exc}", flush=True)
            pooled = load_source_pool_artifact(source, device, smoke)
            if pooled is not None:
                return pooled
            print("[source] source-pool artifact missing; falling back to direct frozen CLIP extraction", flush=True)
            return collect_mvtec_pool_direct(SEED, device, smoke)
    if source == "Full VisA":
        try:
            return collect_visa_pool(SEED, device, smoke)
        except FileNotFoundError as exc:
            print(f"[source] formal VisA cache missing; trying source-pool artifact: {exc}", flush=True)
            pooled = load_source_pool_artifact(source, device, smoke)
            if pooled is not None:
                return pooled
            print("[source] source-pool artifact missing; falling back to direct frozen CLIP extraction", flush=True)
            return collect_visa_pool_direct(SEED, device, smoke)
    raise ValueError(source)


def load_source_pool_artifact(source: str, device: torch.device, smoke: bool) -> dict | None:
    suffix = "smoke" if smoke else "formal"
    path = ROOT / "results" / "phase_cross_layer_robust_state_learning" / "formal" / f"{source.replace(' ', '_')}_SOURCE_LAYER_POOL_{suffix}.npz"
    if not path.exists():
        return None
    print(f"[source] reusing preprocessed L24 source pool: {path}", flush=True)
    data = np.load(path, allow_pickle=True)
    groups = {}
    for cat in data["categories"].tolist():
        groups[str(cat)] = {
            "x": data[f"{cat}_L24"].astype(np.float32),
            "y": data[f"{cat}_y"].astype(np.float32),
        }
    if source == "Full MVTec":
        payload_path = ROOT / "results" / "phase_hard_normal_cross_source_mvtec" / "formal" / "contrast_vectors.npz"
        payload = np.load(payload_path)
        return {
            "source": source,
            "use_align": False,
            "q_pc1": None,
            "align_model": None,
            "m": payload["midpoint"].astype(np.float32),
            "alpha": float(payload["alpha"]),
            "init_v": unit_np(payload["init"].astype(np.float32)).astype(np.float32),
            "groups": groups,
            "source_feature_path": str(path),
        }

    from experiments.phase_hard_normal_contrast import PC1_DEBIAS_OUT, load_payload
    from experiments.phase_state_pair_decomposition import decompose
    from experiments.phase_statealigned_state_alignment_swap import make_model

    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t).astype(np.float32)
    align_model = make_model(states_c, corrupt_payload, device).eval()
    for p in align_model.parameters():
        p.requires_grad_(False)
    return {
        "source": source,
        "use_align": True,
        "q_pc1": q_pc1,
        "align_model": align_model,
        "m": m_c.astype(np.float32),
        "alpha": alpha,
        "init_v": init_v,
        "groups": groups,
        "source_feature_path": str(path),
    }


def collect_mvtec_pool_direct(seed: int, device: torch.device, smoke: bool) -> dict:
    """Rebuild the same sampled MVTec source pool when the historical npz cache was removed.

    This is frozen-feature extraction only; it does not train or tune any model. The sampling
    and initialization logic mirrors collect_mvtec_pool from phase_paper_ready_final_validation.
    """

    _dataset_name, dataset, train_indices, held_indices = light_fold("D1", 0)
    indices = list(train_indices) + list(held_indices)
    if smoke:
        by_cat = {}
        for pos, idx in enumerate(indices):
            item = dataset.data_all[int(idx)]
            cat = str(item["cls_name"])
            bucket = by_cat.setdefault(cat, {0: [], 1: []})
            lab = int(item["anomaly"])
            if len(bucket[lab]) < 4:
                bucket[lab].append((pos, idx))
        chosen = []
        for cat in sorted(by_cat)[:6]:
            chosen.extend(by_cat[cat][0])
            chosen.extend(by_cat[cat][1])
        indices = [idx for _p, idx in chosen]

    model, *_ = load_components(CHECKPOINT, device)
    cats = sorted({str(dataset.data_all[int(idx)]["cls_name"]) for idx in indices})
    slots = {cat: [p for p, idx in enumerate(indices) if str(dataset.data_all[int(idx)]["cls_name"]) == cat] for cat in cats}
    rng = np.random.default_rng(seed)
    groups = {}
    for cat, positions in slots.items():
        print(f"[source-direct] MVTec {cat} ({len(positions)} images)", flush=True)
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            idx = int(indices[pos])
            item = dataset[idx]
            y = patch_labels(item).astype(np.int64)
            with torch.no_grad():
                image = item["img"].unsqueeze(0).to(device)
                patch_feat = normalized_patch_feature(visual_outputs(model, image, grad=False)).squeeze(0)
                z = patch_feat.detach().cpu().numpy().astype(np.float32)
            z = norm_np(z.reshape(SIDE * SIDE, 1024))
            pools = [("normal_image", np.flatnonzero(y == 0))] if int(item["anomaly"]) == 0 else [
                ("anomaly_mask_out", np.flatnonzero(y == 0)),
                ("anomaly_mask", np.flatnonzero(y == 1)),
            ]
            for origin, ids in pools:
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(x.shape[0] for x in parts[origin])
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin].append(z[ids])
            if all(sum(x.shape[0] for x in parts[k]) >= PATCH_SAMPLE_PER_CLASS_CATEGORY for k in parts):
                break
        xs, ys = [], []
        for origin, cls in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if parts[origin]:
                arr = np.concatenate(parts[origin], axis=0).astype(np.float32)
                xs.append(arr)
                ys.extend([cls] * arr.shape[0])
        if xs and len(set(ys)) == 2:
            groups[cat] = {"x": np.concatenate(xs, axis=0), "y": np.asarray(ys, dtype=np.float32)}
    neg = np.concatenate([g["x"][g["y"] < 0.5] for g in groups.values()], axis=0)
    pos = np.concatenate([g["x"][g["y"] >= 0.5] for g in groups.values()], axis=0)
    neg_mu = norm_np(neg.mean(axis=0, keepdims=True))[0]
    pos_mu = norm_np(pos.mean(axis=0, keepdims=True))[0]
    m = 0.5 * (pos_mu + neg_mu)
    d = 0.5 * (pos_mu - neg_mu)
    alpha = float(np.linalg.norm(d))
    init_v = unit_np(d)
    return {
        "source": "Full MVTec",
        "use_align": False,
        "m": m.astype(np.float32),
        "alpha": alpha,
        "init_v": init_v.astype(np.float32),
        "groups": groups,
        "source_feature_path": "direct_frozen_clip_extraction_missing_D1_fold0_cache",
    }


def collect_visa_pool_direct(seed: int, device: torch.device, smoke: bool) -> dict:
    """Rebuild the VisA source pool with the formal PC1 removal and frozen rank-4 adapter."""

    from experiments.phase_hard_normal_contrast import PC1_DEBIAS_OUT, load_payload
    from experiments.phase_state_pair_decomposition import decompose
    from experiments.phase_statealigned_pc1_debias import remove_pc1
    from experiments.phase_statealigned_state_alignment_swap import make_model

    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    align_model = make_model(states_c, corrupt_payload, device).eval()
    for p in align_model.parameters():
        p.requires_grad_(False)

    _dataset_name, dataset, train_indices, held_indices = light_fold("D2", 0)
    indices = list(train_indices) + list(held_indices)
    if smoke:
        chosen = []
        seen = {}
        for pos, idx in enumerate(indices):
            row = dataset.data_all[int(idx)]
            cat = str(row["cls_name"])
            lab = int(row["anomaly"])
            bucket = seen.setdefault(cat, {0: 0, 1: 0})
            if bucket[lab] < 2:
                chosen.append((pos, idx))
                bucket[lab] += 1
            if len(chosen) >= 48:
                break
        indices = [idx for _p, idx in chosen]

    model, *_ = load_components(CHECKPOINT, device)
    cats = sorted({str(dataset.data_all[int(idx)]["cls_name"]) for idx in indices})
    slots = {cat: [p for p, idx in enumerate(indices) if str(dataset.data_all[int(idx)]["cls_name"]) == cat] for cat in cats}
    rng = np.random.default_rng(seed)
    groups = {}
    for cat, positions in slots.items():
        print(f"[source-direct] VisA {cat} ({len(positions)} images)", flush=True)
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            idx = int(indices[pos])
            item = dataset[idx]
            y = patch_labels(item).astype(np.int64)
            with torch.no_grad():
                image = item["img"].unsqueeze(0).to(device)
                patch_feat = normalized_patch_feature(visual_outputs(model, image, grad=False)).squeeze(0)
                raw = patch_feat.detach().cpu().numpy().astype(np.float32)
                z0 = remove_pc1(raw, q_pc1)
                z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
                z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
            pools = [("normal_image", np.flatnonzero(y == 0))] if y.max() == 0 else [
                ("anomaly_mask_out", np.flatnonzero(y == 0)),
                ("anomaly_mask", np.flatnonzero(y == 1)),
            ]
            for origin, ids in pools:
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(x.shape[0] for x in parts[origin])
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin].append(z[ids])
            if all(sum(x.shape[0] for x in parts[k]) >= PATCH_SAMPLE_PER_CLASS_CATEGORY for k in parts):
                break
        xs, ys = [], []
        for origin, cls in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if parts[origin]:
                arr = np.concatenate(parts[origin], axis=0).astype(np.float32)
                xs.append(arr)
                ys.extend([cls] * arr.shape[0])
        if xs and len(set(ys)) == 2:
            groups[cat] = {"x": np.concatenate(xs, axis=0), "y": np.asarray(ys, dtype=np.float32)}

    return {
        "source": "Full VisA",
        "use_align": True,
        "q_pc1": q_pc1,
        "align_model": align_model,
        "m": m_c.astype(np.float32),
        "alpha": alpha,
        "init_v": init_v.astype(np.float32),
        "groups": groups,
        "source_feature_path": "direct_frozen_clip_extraction_missing_D2_fold0_cache",
    }


def initial_geometry(source_info: dict) -> Dict[str, np.ndarray | float]:
    m = np.asarray(source_info["m"], dtype=np.float32)
    alpha = float(source_info["alpha"])
    init_v = unit_np(np.asarray(source_info["init_v"], dtype=np.float32))
    s_n0, s_a0 = state_pair_from_m_alpha_v(m, alpha, init_v)
    w0 = (s_a0 - s_n0).astype(np.float32)
    c0 = float(np.linalg.norm(w0))
    return {"m": m, "alpha": alpha, "init_v": init_v, "s_n0": s_n0, "s_a0": s_a0, "w0": w0, "c0": c0}


def make_batch_plan(source_info: dict, epochs: int = EPOCHS) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    groups = source_info["groups"]
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, int(math.ceil(total / BATCH_SIZE)))
    plan: List[Tuple[np.ndarray, np.ndarray]] = []
    for _epoch in range(epochs):
        for _ in range(steps):
            plan.append(balanced_batch(groups, rng))
    return plan


def ntr_loss_from_logits(logits: torch.Tensor, yb: torch.Tensor, rho: float = RHO) -> torch.Tensor:
    neg = logits[yb < 0.5]
    if neg.numel() == 0:
        return logits.new_tensor(0.0)
    k = max(1, int(math.ceil(rho * int(neg.numel()))))
    vals, idx = torch.topk(neg.detach(), k)
    del vals
    return F.softplus(neg[idx]).mean()


def train_head(method: str, source: str, source_info: dict, init: dict, batch_plan: List[Tuple[np.ndarray, np.ndarray]], device: torch.device) -> Tuple[HeadSpec, List[dict]]:
    if method == "LinearHead":
        module: torch.nn.Module = LinearHead(init["w0"]).to(device)
    elif method == "NormalizedLinearHead":
        module = NormalizedLinearHead(init["w0"], float(init["c0"])).to(device)
    elif method == "CES":
        module = CESHead(init["m"], float(init["alpha"]), init["init_v"]).to(device)
    else:
        raise ValueError(method)

    opt = torch.optim.AdamW(module.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    log: List[dict] = []
    steps_per_epoch = max(1, len(batch_plan) // EPOCHS)
    for step, (xb, yb_np) in enumerate(batch_plan):
        z = torch.tensor(xb, dtype=torch.float32, device=device)
        yb = torch.tensor(yb_np, dtype=torch.float32, device=device)
        logits = module(z)
        base, bce, dice = balanced_loss(logits, yb)
        ntr = ntr_loss_from_logits(logits, yb, RHO)
        loss = base + ntr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            log.append(
                {
                    "source": source,
                    "method": method,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "base_loss": float(base.detach().cpu()),
                    "bce": float(bce.detach().cpu()),
                    "dice": float(dice.detach().cpu()),
                    "ntr": float(ntr.detach().cpu()),
                }
            )
    return module.to_spec(source), log


def load_formal_ces(source: str, init: dict) -> HeadSpec:
    ckpt_path = FORMAL_CKPT_DIR / f"{source.replace(' ', '_')}_seed0_rho0.10.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return HeadSpec("CES", {"m": init["m"], "alpha": float(init["alpha"]), "v": np.asarray(ckpt["v"], dtype=np.float32)}, source)


def save_head_spec(spec: HeadSpec) -> str:
    path = CKPT_DIR / f"{spec.source.replace(' ', '_')}_{spec.method}_seed0.pt"
    torch.save({"method": spec.method, "source": spec.source, "seed": spec.seed, **spec.params}, path)
    return str(path)


def source_arrays(source_info: dict) -> Tuple[np.ndarray, np.ndarray]:
    x_parts, y_parts = [], []
    for g in source_info["groups"].values():
        x_parts.append(g["x"])
        y_parts.append(g["y"].astype(np.float32))
    return np.concatenate(x_parts, axis=0).astype(np.float32), np.concatenate(y_parts, axis=0).astype(np.float32)


def source_fit(source: str, source_info: dict, specs: Dict[str, HeadSpec]) -> List[dict]:
    x, y = source_arrays(source_info)
    rows = []
    for method, spec in specs.items():
        logits = spec.logits_np(x)
        try:
            auroc = 100.0 * roc_auc_score(y, logits)
        except ValueError:
            auroc = float("nan")
        ap = 100.0 * average_precision_score(y, logits)
        neg = logits[y < 0.5]
        pos = logits[y >= 0.5]
        k = max(1, int(math.ceil(RHO * int(len(neg)))))
        ntr = float(np.logaddexp(0.0, np.partition(neg, len(neg) - k)[-k:]).mean()) if len(neg) else 0.0
        rows.append(
            {
                "source": source,
                "method": method,
                "source_p_auroc": auroc,
                "source_p_ap": ap,
                "mean_normal_logit": float(neg.mean()) if len(neg) else float("nan"),
                "mean_anomaly_logit": float(pos.mean()) if len(pos) else float("nan"),
                "ntr_tail_softplus": ntr,
                "num_patches": int(len(y)),
                "num_anomaly": int(y.sum()),
                "num_normal": int((1 - y).sum()),
            }
        )
    return rows


def preprocess_patch_grid(source_info: dict, patch_np: np.ndarray, device: torch.device) -> torch.Tensor:
    if source_info.get("use_align", False):
        return aligned_grid(patch_np, source_info["q_pc1"], source_info["align_model"], device)
    z = norm_np(patch_np.reshape(-1, patch_np.shape[-1]).astype(np.float32))
    return torch.tensor(z, dtype=torch.float32, device=device)


def score_patch_grid(source_info: dict, spec: HeadSpec, patch_np: np.ndarray, device: torch.device) -> np.ndarray:
    z = preprocess_patch_grid(source_info, patch_np, device)
    w = torch.tensor(spec.weight(), dtype=torch.float32, device=device)
    logits = z @ w + float(spec.bias())
    return logits.detach().cpu().numpy().reshape(37, 37).astype(np.float32)


def stitch_local_maps(local_maps: np.ndarray) -> np.ndarray:
    canvas = np.zeros((74, 74), dtype=np.float32)
    canvas[:37, :37] = local_maps[0]
    canvas[:37, 37:] = local_maps[1]
    canvas[37:, :37] = local_maps[2]
    canvas[37:, 37:] = local_maps[3]
    return canvas


def lrp_map(source_info: dict, spec: HeadSpec, global_patch: np.ndarray, local_patches: np.ndarray, device: torch.device) -> np.ndarray:
    g37 = score_patch_grid(source_info, spec, global_patch, device)
    local_maps = np.stack([score_patch_grid(source_info, spec, local_patches[i], device) for i in range(4)], axis=0)
    local74 = stitch_local_maps(local_maps)
    g74 = torch_resize(g37, (74, 74), align_corners=True)
    return (0.5 * g74 + 0.5 * local74).astype(np.float32)


def local_cache_path(source: str, target: str) -> Path:
    return LOCAL_CACHE_DIR / f"local_patch_{pair_slug(source, target)}.npy"


def direct_target_global_cache(target: str, device: torch.device, smoke: bool):
    direction = "D1" if target == "MVTec" else "D2"
    _dataset_name, dataset, train_indices, held_indices = light_fold(direction, 0)
    indices = list(train_indices) + list(held_indices)
    if smoke:
        chosen = []
        for pos, idx in enumerate(indices):
            if int(dataset.data_all[int(idx)]["anomaly"]) == 1:
                chosen.append((pos, idx))
            if len(chosen) >= 24:
                break
        indices = [idx for _p, idx in chosen]
    cache_path = (OUT / "cache" / f"global_patch_{target}_smoke.npy") if smoke else (LOCAL_CACHE_DIR / f"global_patch_{target}.npy")
    shape = (len(indices), SIDE * SIDE, 1024)
    if cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape == shape:
            return target, target, dataset, indices, arr, "source_dataset"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[target-direct] extracting global frozen CLIP patches for {target}: {shape}", flush=True)
    arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16, shape=shape)
    model, *_ = load_components(CHECKPOINT, device)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i % 100 == 0:
                print(f"[target-direct] {target} {i}/{len(indices)}", flush=True)
            item = dataset[int(idx)]
            image = item["img"].unsqueeze(0).to(device)
            patch_feat = normalized_patch_feature(visual_outputs(model, image, grad=False)).squeeze(0)
            arr[i] = patch_feat.detach().cpu().numpy().astype(np.float16)
    del arr
    return target, target, dataset, indices, np.load(cache_path, mmap_mode="r"), "source_dataset"


def get_target_cell(source: str, target: str, device: torch.device, smoke: bool):
    if target in {"MVTec", "VisA"}:
        return direct_target_global_cache(target, device, smoke)
    if target == "BTAD":
        return external_cell("BTAD", smoke)
    if target == "MPDD":
        cells, _audits = ready_holdout_cells(device, smoke)
        for cell in cells:
            if cell[1] == "MPDD":
                return cell
        raise RuntimeError("MPDD holdout cell is not READY")
    cells, _ = protocol_cells(source, device, smoke)
    selected = [c for c in cells if c[1] == target]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one protocol cell for {source}->{target}, got {len(selected)}")
    return selected[0]


def evaluate_lrp_heads(
    source: str,
    target: str,
    source_info: dict,
    specs: Dict[str, HeadSpec],
    device: torch.device,
    smoke: bool = False,
) -> Tuple[List[dict], str]:
    _cell_id, _target, dataset, idxs, patch, kind = get_target_cell(source, target, device, smoke)

    lpath = local_cache_path(source, target)
    if not lpath.exists():
        raise FileNotFoundError(f"Missing local LRP cache: {lpath}")
    local_cache = np.load(lpath, mmap_mode="r")
    n = len(idxs)
    if local_cache.shape[0] < n:
        raise RuntimeError(f"Local cache too short for {source}->{target}: {local_cache.shape[0]} < {n}")

    maps_by_method: Dict[str, List[np.ndarray]] = {m: [] for m in specs}
    metas = []
    for pos, ref in enumerate(idxs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        metas.append({"category": cat, "label": label, "img_path": path, "mask": mask_np(item)})
        global_patch = patch[pos]
        local_patches = np.asarray(local_cache[pos], dtype=np.float32)
        for method, spec in specs.items():
            maps_by_method[method].append(lrp_map(source_info, spec, global_patch, local_patches, device))

    import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp

    old_variants = lzp.VARIANTS
    try:
        lzp.VARIANTS = tuple(maps_by_method.keys())
        rows, _cat_rows = evaluate_maps(source, target, maps_by_method, metas)
    finally:
        lzp.VARIANTS = old_variants
    return rows, str(lpath)


def read_formal_reference() -> Dict[Tuple[str, str, str], float]:
    path = FORMAL_CLOSE / "RHO_SENSITIVITY.csv"
    ref: Dict[Tuple[str, str, str], float] = {}
    if not path.exists():
        return ref
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("variant") != "Base+NTR+LRP":
                continue
            if str(r.get("rho")) not in {"0.1", "0.10"}:
                continue
            if str(r.get("seed")) != "0":
                continue
            if "metric" in r and "value" in r:
                key = (r["source"], r["target"], r["metric"])
                ref[key] = float(r["value"])
            else:
                for metric in ["I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO"]:
                    if metric in r and r[metric] != "":
                        ref[(r["source"], r["target"], metric)] = float(r[metric])
    return ref


def write_csv(path: Path, rows: List[dict], fieldnames: List[str] | None = None) -> None:
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def flatten_metric_rows(rows: List[dict], cache_path: str, ckpt_map: Dict[str, str]) -> List[dict]:
    out = []
    for r in rows:
        method = r.get("variant") or r.get("method")
        target = r.get("target", r.get("dataset", ""))
        row = {
            "source": r["source"],
            "target": target,
            "method": method,
            "seed": SEED,
            "checkpoint": ckpt_map.get(method, ""),
            "local_cache": cache_path,
            "evaluator": "unified_full_resolution_M_FR",
        }
        for short, key in MFR_KEYS.items():
            row[short] = r.get(key, np.nan)
        out.append(row)
    return out


def initialization_equivalence(source: str, source_info: dict, init: dict) -> List[dict]:
    xb, _ = balanced_batch(source_info["groups"], np.random.default_rng(SEED))
    w0 = np.asarray(init["w0"], dtype=np.float32)
    c0 = float(init["c0"])
    u0 = unit_np(w0)
    ces_w = ces_weight_np(init["m"], init["alpha"], init["init_v"])
    logits = {
        "LinearHead": xb @ w0,
        "NormalizedLinearHead": c0 * (xb @ u0),
        "CES": xb @ ces_w,
    }
    rows = []
    pairs = [("LinearHead", "NormalizedLinearHead"), ("LinearHead", "CES"), ("NormalizedLinearHead", "CES")]
    for a, b in pairs:
        diff = np.abs(logits[a] - logits[b])
        rows.append(
            {
                "source": source,
                "head_a": a,
                "head_b": b,
                "num_patches": int(len(xb)),
                "mean_abs_logit_error": float(diff.mean()),
                "max_abs_logit_error": float(diff.max()),
            }
        )
    return rows


def geometry_rows(source: str, init: dict, specs: Dict[str, HeadSpec], ckpt_map: Dict[str, str]) -> List[dict]:
    w0 = np.asarray(init["w0"], dtype=np.float32)
    rows = []
    for method, spec in specs.items():
        w = spec.weight()
        cos = float(np.dot(unit_np(w0), unit_np(w)))
        angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        rows.append(
            {
                "source": source,
                "method": method,
                "checkpoint": ckpt_map.get(method, ""),
                "initial_w_norm": float(np.linalg.norm(w0)),
                "final_w_norm": float(np.linalg.norm(w)),
                "bias": float(spec.bias()),
                "cos_final_initial_w": cos,
                "angle_final_initial_w_deg": angle,
                "c0": float(init["c0"]),
                "m_norm": float(np.linalg.norm(init["m"])),
                "alpha": float(init["alpha"]),
            }
        )
    return rows


def aggregate_summary(raw_rows: List[dict]) -> List[dict]:
    rows = []
    by_method: Dict[str, List[dict]] = {}
    for r in raw_rows:
        by_method.setdefault(r["method"], []).append(r)
    ces_pap = {r["source"] + "->" + r["target"]: float(r["P_AP"]) for r in by_method.get("CES", [])}
    for method, vals in by_method.items():
        pap = np.array([float(v["P_AP"]) for v in vals], dtype=np.float64)
        auroc = np.array([float(v["P_AUROC"]) for v in vals], dtype=np.float64)
        f1 = np.array([float(v["P_F1MAX"]) for v in vals], dtype=np.float64)
        pro = np.array([float(v["AUPRO"]) for v in vals], dtype=np.float64)
        deltas = []
        for v in vals:
            key = v["source"] + "->" + v["target"]
            if key in ces_pap:
                deltas.append(float(v["P_AP"]) - ces_pap[key])
        rows.append(
            {
                "method": method,
                "num_transfers": len(vals),
                "mean_P_AP": float(pap.mean()) if len(pap) else float("nan"),
                "median_P_AP": float(np.median(pap)) if len(pap) else float("nan"),
                "mean_P_AUROC": float(auroc.mean()) if len(auroc) else float("nan"),
                "mean_P_F1MAX": float(f1.mean()) if len(f1) else float("nan"),
                "mean_AUPRO": float(pro.mean()) if len(pro) else float("nan"),
                "mean_delta_P_AP_vs_CES": float(np.mean(deltas)) if deltas else 0.0,
                "improved_vs_CES": int(sum(d > 0 for d in deltas)) if deltas else 0,
                "non_drop_ge_minus_0p2_vs_CES": int(sum(d >= -0.2 for d in deltas)) if deltas else 0,
            }
        )
    return rows


def report_preflight(pre_rows: List[dict], passed: bool) -> None:
    lines = [
        "# PREFLIGHT_REPRODUCTION",
        "",
        f"Formal checkpoint source: `{FORMAL_CKPT_DIR}`",
        f"Reference metric source: `{FORMAL_CLOSE / 'RHO_SENSITIVITY.csv'}`",
        f"Tolerance: |delta P-AP| <= {PREFLIGHT_TOL_PAP}",
        "",
        "| source | target | reference P-AP | reproduced P-AP | delta | passed |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in pre_rows:
        lines.append(
            f"| {r['source']} | {r['target']} | {r['reference_p_ap']:.6f} | {r['reproduced_p_ap']:.6f} | {r['delta_p_ap']:.6f} | {r['passed']} |"
        )
    lines += ["", f"Overall: {'PASSED' if passed else 'FAILED'}", ""]
    (OUT / "PREFLIGHT_REPRODUCTION.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(outcome: str, summary: List[dict], raw_rows: List[dict]) -> None:
    lines = [
        "# CES vs Linear Head Parameterization Control",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "This phase compares a free linear head, a normalized linear head, and the formal CES parameterization under the same frozen source features, NTR policy, fixed LRP inference, and unified full-resolution evaluator.",
        "",
        "## Summary",
        "",
        "| method | mean P-AP | delta vs CES | improved vs CES | non-drop vs CES |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['method']} | {r['mean_P_AP']:.4f} | {r['mean_delta_P_AP_vs_CES']:.4f} | {r['improved_vs_CES']}/{r['num_transfers']} | {r['non_drop_ge_minus_0p2_vs_CES']}/{r['num_transfers']} |"
        )
    lines += [
        "",
        "## Per-Transfer P-AP",
        "",
        "| source | target | method | P-AP | P-AUROC | P-F1max | AUPRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted(raw_rows, key=lambda x: (x["source"], x["target"], x["method"])):
        lines.append(
            f"| {r['source']} | {r['target']} | {r['method']} | {float(r['P_AP']):.4f} | {float(r['P_AUROC']):.4f} | {float(r['P_F1MAX']):.4f} | {float(r['AUPRO']):.4f} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `HEAD_CONTROL_CONFIG.json`",
        "- `HEAD_CONTROL_RAW_METRICS.csv`",
        "- `HEAD_CONTROL_SUMMARY.csv`",
        "- `HEAD_INITIALIZATION_EQUIVALENCE.csv`",
        "- `SOURCE_FIT_COMPARISON.csv`",
        "- `HEAD_GEOMETRY_DIAGNOSTICS.csv`",
        "",
    ]
    (OUT / "HEAD_CONTROL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def determine_outcome(summary: List[dict]) -> str:
    means = {r["method"]: float(r["mean_P_AP"]) for r in summary}
    if "CES" not in means:
        return "PROTOCOL_VALIDATION_FAILED"
    linear_best = max(means.get("LinearHead", -1e9), means.get("NormalizedLinearHead", -1e9))
    ces = means["CES"]
    if linear_best > ces + 0.5:
        return "LINEAR_HEAD_OUTPERFORMS_CES"
    if abs(linear_best - ces) <= 0.5:
        return "LINEAR_HEADS_PERFORM_SIMILARLY"
    return "CES_PARAMETERIZATION_ADVANTAGE_SUPPORTED"


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config = {
        "phase": "CES-LHC",
        "output": str(OUT),
        "seed": SEED,
        "rho": RHO,
        "ntr_weight": 1.0,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "patch_batch": BATCH_SIZE,
        "transfers": [{"source": s, "target": t} for s, t in PAIRS],
        "formal_checkpoint_dir": str(FORMAL_CKPT_DIR),
        "local_cache_dir": str(LOCAL_CACHE_DIR),
        "device": str(device),
        "smoke": bool(args.smoke),
    }
    (OUT / "HEAD_CONTROL_CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    ref = read_formal_reference()
    all_raw: List[dict] = []
    all_init_eq: List[dict] = []
    all_source_fit: List[dict] = []
    all_geom: List[dict] = []
    train_log: List[dict] = []
    preflight_rows: List[dict] = []

    source_cache: Dict[str, dict] = {}
    trained_specs: Dict[str, Dict[str, HeadSpec]] = {}
    ckpt_paths: Dict[str, Dict[str, str]] = {}

    for source in ["Full MVTec", "Full VisA"]:
        print(f"[source] loading {source}", flush=True)
        source_info = load_source_info(source, device, args.smoke)
        source_cache[source] = source_info
        init = initial_geometry(source_info)
        all_init_eq.extend(initialization_equivalence(source, source_info, init))

        plan = make_batch_plan(source_info, EPOCHS)
        specs: Dict[str, HeadSpec] = {}
        cpaths: Dict[str, str] = {}
        for method in ["LinearHead", "NormalizedLinearHead"]:
            print(f"[train] {source} {method}", flush=True)
            spec, log = train_head(method, source, source_info, init, plan, device)
            specs[method] = spec
            train_log.extend(log)
            cpaths[method] = save_head_spec(spec)

        print(f"[reuse] {source} formal CES", flush=True)
        specs["CES"] = load_formal_ces(source, init)
        cpaths["CES"] = save_head_spec(specs["CES"])
        trained_specs[source] = specs
        ckpt_paths[source] = cpaths
        all_source_fit.extend(source_fit(source, source_info, specs))
        all_geom.extend(geometry_rows(source, init, specs, cpaths))

    write_csv(OUT / "HEAD_INITIALIZATION_EQUIVALENCE.csv", all_init_eq)
    write_csv(OUT / "SOURCE_FIT_COMPARISON.csv", all_source_fit)
    write_csv(OUT / "HEAD_GEOMETRY_DIAGNOSTICS.csv", all_geom)
    write_csv(OUT / "TRAINING_LOG.csv", train_log)

    print("[preflight] reproducing formal CES+NTR+LRP rows", flush=True)
    preflight_ok = True
    for source, target in PAIRS:
        rows, cache_path = evaluate_lrp_heads(source, target, source_cache[source], {"CES": trained_specs[source]["CES"]}, device, args.smoke)
        flat = flatten_metric_rows(rows, cache_path, ckpt_paths[source])
        pap = float(flat[0]["P_AP"])
        reference = ref.get((source, target, "P_AP"), float("nan"))
        delta = pap - reference if np.isfinite(reference) else float("nan")
        passed = bool(args.smoke or (np.isfinite(delta) and abs(delta) <= PREFLIGHT_TOL_PAP))
        preflight_ok = preflight_ok and passed
        preflight_rows.append(
            {
                "source": source,
                "target": target,
                "reference_p_ap": reference,
                "reproduced_p_ap": pap,
                "delta_p_ap": delta,
                "passed": passed,
            }
        )
    report_preflight(preflight_rows, preflight_ok)
    write_csv(OUT / "PREFLIGHT_REPRODUCTION.csv", preflight_rows)
    if not preflight_ok:
        outcome = "PROTOCOL_VALIDATION_FAILED"
        (OUT / "OUTCOME.json").write_text(json.dumps({"outcome": outcome, "preflight_passed": False}, indent=2), encoding="utf-8")
        write_report(outcome, [], [])
        print(f"[stop] {outcome}", flush=True)
        return

    print("[eval] full-resolution LRP for all heads", flush=True)
    for source, target in PAIRS:
        rows, cache_path = evaluate_lrp_heads(source, target, source_cache[source], trained_specs[source], device, args.smoke)
        all_raw.extend(flatten_metric_rows(rows, cache_path, ckpt_paths[source]))

    summary = aggregate_summary(all_raw)
    outcome = determine_outcome(summary)
    write_csv(OUT / "HEAD_CONTROL_RAW_METRICS.csv", all_raw)
    write_csv(OUT / "HEAD_CONTROL_SUMMARY.csv", summary)
    (OUT / "OUTCOME.json").write_text(json.dumps({"outcome": outcome, "preflight_passed": True}, indent=2), encoding="utf-8")
    write_report(outcome, summary, all_raw)
    print(f"[done] {outcome}", flush=True)
    print(f"[out] {OUT}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
