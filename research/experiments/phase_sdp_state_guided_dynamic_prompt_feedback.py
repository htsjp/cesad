import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import tokenize_texts
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_learnable_text_semantic_compatibility import PromptPair, train_prompt
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    item_at,
    item_meta,
    mask_np,
    torch_resize,
)
import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, LR, SIDE, norm_np, patch_labels
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_sdp_state_guided_dynamic_prompt_feedback/formal")
TEXT_OUT = Path("results/phase_learnable_text_semantic_compatibility/formal")
EPOCHS = 10
TOP_NEG_FRAC = 0.10
TEXT_ALPHA = 0.25
STATE_BETA = 0.05
TRUST_DEG = 5.0
PILOT = [
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
]


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


def fnum(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


@torch.no_grad()
def load_clip_text_model(device):
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=None, download_root=str(ROOT / ".cache" / "clip"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    visual_proj = model.visual.proj.detach().float().cpu().numpy().astype(np.float32)
    logit_scale = float(model.logit_scale.detach().exp().cpu())
    return model, visual_proj, logit_scale


@torch.no_grad()
def load_visual_forward_model(device):
    model, _prompt_learner, _image_proj, _patch_proj, _prompt_proj = load_components(CHECKPOINT, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def baseline_states(model_info, device):
    m = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v = torch.tensor(model_info["v1"], dtype=torch.float32, device=device)
    alpha = float(model_info["alpha"])
    s_a = F.normalize(m + alpha * v, dim=-1)
    s_n = F.normalize(m - alpha * v, dim=-1)
    return s_n, s_a


@torch.no_grad()
def state_z(model_info, patch_np, device):
    if model_info["use_align"]:
        return aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).reshape(SIDE * SIDE, -1)
    z = norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    return torch.tensor(z, dtype=torch.float32, device=device)


@torch.no_grad()
def state_logits_from_z(model_info, z, device, v_override=None, s_a_override=None):
    m = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    if s_a_override is not None:
        s_n, _s_a = baseline_states(model_info, device)
        return z @ s_a_override - z @ s_n
    v = torch.tensor(model_info["v1"] if v_override is None else v_override, dtype=torch.float32, device=device)
    return contrast_logits(z, m, float(model_info["alpha"]), v)


def text_z_np(raw_patch, visual_proj):
    raw = norm_np(np.asarray(raw_patch, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    return norm_np(raw @ visual_proj).astype(np.float32)


def text_logits_np(raw_patch, visual_proj, anchors, logit_scale):
    x = text_z_np(raw_patch, visual_proj)
    a = norm_np(np.asarray(anchors, dtype=np.float32))
    return (logit_scale * (x @ a.T)[:, 1] - logit_scale * (x @ a.T)[:, 0]).astype(np.float32)


def source_dataset(source_name, smoke):
    if source_name == "Full VisA":
        name, dataset, indices, _cls, patch, _y_img, cat_img = load_full_visa_source_lazy(smoke)
        labels_by_pos, _corrupt, _audit = source_label_maps(smoke)
        labels = [np.asarray(labels_by_pos.get(i, np.zeros((SIDE * SIDE,), dtype=np.int64)), dtype=np.float32) for i in range(len(indices))]
        cats = [str(c) for c in cat_img]
        return dataset, indices, patch, labels, cats
    _name, dataset, indices, patch = load_full_mvtec_source(smoke)
    labels, cats = [], []
    for idx in indices:
        item = dataset[int(idx)]
        labels.append(patch_labels(item).astype(np.float32))
        cats.append(str(item["cls_name"]))
    return dataset, indices, patch, labels, cats


def source_text_arrays(source_name, visual_proj, smoke):
    cache = OUT / f"{source_name.replace(' ', '_')}_TEXT_SOURCE_{'smoke' if smoke else 'formal'}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"].astype(np.float32), z["y"].astype(np.float32)
    _dataset, _indices, patch, labels, cats = source_dataset(source_name, smoke)
    rng = np.random.default_rng(SEED)
    xs, ys = [], []
    for cat in sorted(set(cats)):
        pos = [i for i, c in enumerate(cats) if c == cat]
        pools = {0: [], 1: []}
        for i in pos:
            y = labels[i].reshape(-1)
            x = text_z_np(patch[i], visual_proj)
            for cls in (0, 1):
                ids = np.flatnonzero(y == cls)
                if ids.size == 0:
                    continue
                room = 768 - sum(part.shape[0] for part in pools[cls])
                if room <= 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                pools[cls].append(x[ids])
        for cls in (0, 1):
            if pools[cls]:
                arr = np.concatenate(pools[cls], axis=0).astype(np.float32)
                xs.append(arr)
                ys.append(np.full((arr.shape[0],), cls, dtype=np.float32))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def load_or_train_static_prompt(source_name, clip_model, visual_proj, logit_scale, device, smoke):
    if source_name == "Full VisA" and not smoke:
        ckpt = TEXT_OUT / "P1_learnable_prompt_formal.pt"
        if ckpt.exists():
            prompt = PromptPair(clip_model, "semantic").to(device)
            payload = torch.load(ckpt, map_location=device, weights_only=False)
            with torch.no_grad():
                prompt.ctx_n.copy_(payload["ctx_n"].to(device=device, dtype=prompt.ctx_n.dtype))
                prompt.ctx_a.copy_(payload["ctx_a"].to(device=device, dtype=prompt.ctx_a.dtype))
            return prompt.eval(), payload.get("curves", []), True
    x, y = source_text_arrays(source_name, visual_proj, smoke)
    import experiments.phase_learnable_text_semantic_compatibility as text_phase

    old_out = text_phase.OUT
    try:
        text_phase.OUT = OUT / "static_prompt_cache" / source_name.replace(" ", "_")
        text_phase.OUT.mkdir(parents=True, exist_ok=True)
        return train_prompt("learnable", clip_model, x, y, logit_scale, smoke, device)
    finally:
        text_phase.OUT = old_out


class DynamicPromptFeedback(nn.Module):
    def __init__(self, text_dim, state_dim, visual_proj_np, t_n0, t_a0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(text_dim * 4, 128), nn.GELU(), nn.Linear(128, text_dim))
        self.register_buffer("visual_back", torch.tensor(visual_proj_np.T, dtype=torch.float32))
        self.register_buffer("t_n0", torch.tensor(t_n0, dtype=torch.float32))
        self.register_buffer("t_a0", torch.tensor(t_a0, dtype=torch.float32))
        self.state_dim = state_dim

    def residual_direction(self, z_state, q0):
        p = torch.sigmoid(q0.detach())
        w_a = p / p.sum().clamp_min(1e-6)
        amb = (p * (1.0 - p))
        w_h = amb / amb.sum().clamp_min(1e-6)
        z_text = F.normalize(z_state @ self.visual_back.T, dim=-1)
        c_a = torch.sum(w_a[:, None] * z_text, dim=0)
        c_h = torch.sum(w_h[:, None] * z_text, dim=0)
        x = torch.cat([c_a, c_h, torch.abs(c_a - c_h), c_a * c_h], dim=0)
        delta = self.net(x)
        t_a = F.normalize(self.t_a0 + TEXT_ALPHA * delta, dim=0)
        t_n = F.normalize(self.t_n0, dim=0)
        d_t = F.normalize(t_a - t_n, dim=0)
        r = F.normalize(d_t @ self.visual_back, dim=0)
        return r, delta, t_a


def trusted_s_a(s_a, r):
    cand = F.normalize(s_a + STATE_BETA * r, dim=0)
    cosv = torch.clamp(torch.sum(cand * s_a), -1.0, 1.0)
    max_angle = math.radians(TRUST_DEG)
    angle = torch.acos(cosv)
    if float(angle.detach().cpu()) <= max_angle:
        return cand
    u = F.normalize(cand - torch.sum(cand * s_a) * s_a, dim=0)
    return F.normalize(math.cos(max_angle) * s_a + math.sin(max_angle) * u, dim=0)


def train_dynamic(source_name, model_info, prompt_anchors, visual_proj, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{source_name.replace(' ', '_')}_E2_dynamic_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        mod = DynamicPromptFeedback(visual_proj.shape[1], 1024, visual_proj, prompt_anchors[0], prompt_anchors[1]).to(device)
        mod.load_state_dict(payload["state_dict"])
        return mod.eval(), payload.get("curves", []), True
    _dataset, _indices, patch, labels, cats = source_dataset(source_name, smoke)
    n = len(labels)
    steps = max(1, int(math.ceil(n / max(1, BATCH_SIZE // 8))))
    if smoke:
        steps = min(steps, 3)
    s_n, s_a = baseline_states(model_info, device)
    mod = DynamicPromptFeedback(visual_proj.shape[1], 1024, visual_proj, prompt_anchors[0], prompt_anchors[1]).to(device)
    opt = torch.optim.AdamW(mod.parameters(), lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _step in range(steps):
            i = int(rng.integers(0, n))
            z = state_z(model_info, patch[i], device)
            y = torch.tensor(labels[i].reshape(-1), dtype=torch.float32, device=device)
            q0 = state_logits_from_z(model_info, z, device)
            r, delta, t_a = mod.residual_direction(z, q0)
            s_ap = trusted_s_a(s_a, r)
            q2 = z @ s_ap - z @ s_n
            bce = F.binary_cross_entropy_with_logits(q2, y)
            neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            if neg.numel():
                k = max(1, int(math.ceil(float(neg.numel()) * TOP_NEG_FRAC)))
                hard = neg[torch.topk(q2.detach()[neg], k=k, largest=True).indices]
                ntr = F.softplus(q2[hard]).mean()
            else:
                ntr = torch.zeros((), device=device)
            loss = bce + ntr
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append([float(loss.detach().cpu()), float(bce.detach().cpu()), float(ntr.detach().cpu()), float(delta.detach().norm().cpu()), float((t_a @ mod.t_a0).detach().cpu())])
        arr = np.asarray(vals, dtype=np.float32)
        row = {
            "source": source_name,
            "variant": "E2_StateGuidedDynamicPrompt",
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "state_loss": float(arr[:, 1].mean()),
            "tail_loss": float(arr[:, 2].mean()),
            "delta_text_norm": float(arr[:, 3].mean()),
            "cos_tA_dynamic_tA0": float(arr[:, 4].mean()),
        }
        curves.append(row)
        print(f"[SDP] {source_name} E2 epoch {epoch}/{EPOCHS} loss={row['loss']:.4f}", flush=True)
    torch.save({"state_dict": mod.state_dict(), "curves": curves}, ckpt)
    return mod.eval(), curves, False


@torch.no_grad()
def score_dynamic_grid(model_info, z, dyn, device):
    s_n, s_a = baseline_states(model_info, device)
    q0 = state_logits_from_z(model_info, z, device)
    r, delta, t_a = dyn.residual_direction(z, q0)
    s_ap = trusted_s_a(s_a, r)
    q2 = z @ s_ap - z @ s_n
    angle = float(torch.acos(torch.clamp(torch.sum(s_ap * s_a), -1, 1)).cpu() * 180.0 / math.pi)
    return q2.detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32), {
        "angle_sA_prime_sA": angle,
        "delta_text_norm": float(delta.norm().cpu()),
        "cos_tA_dynamic_tA0": float((t_a @ dyn.t_a0).cpu()),
    }


@torch.no_grad()
def local_maps(model_info, clip_model, preprocess, img_path, visual_proj, anchors, dyn, device):
    crops = lzp.quadrant_crops(img_path)
    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patches = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)
    maps_text, maps_dyn, diag = [], [], []
    for i in range(4):
        z = state_z(model_info, patches[i], device)
        maps_dyn.append(score_dynamic_grid(model_info, z, dyn, device)[0])
        maps_text.append(text_logits_np(patches[i], visual_proj, anchors, 1.0).reshape(SIDE, SIDE))
    s_text = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    s_dyn = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    s_text[:SIDE, :SIDE], s_text[:SIDE, SIDE:], s_text[SIDE:, :SIDE], s_text[SIDE:, SIDE:] = maps_text
    s_dyn[:SIDE, :SIDE], s_dyn[:SIDE, SIDE:], s_dyn[SIDE:, :SIDE], s_dyn[SIDE:, SIDE:] = maps_dyn
    return s_text, s_dyn, diag


def build_eval_maps(source_name, target, dataset, refs, patch, kind, model_info, clip_model, preprocess, visual_proj, anchors, dyn, device, smoke):
    n = min(24, len(refs)) if smoke else len(refs)
    refs = refs[:n]
    variants = ["E0_Baseline_LRP", "E1_StaticLearnedPrompt_TextOnly", "E2_StateGuidedDynamicPrompt"]
    maps = {v: [] for v in variants}
    metas, diag = [], []
    for pos, ref in enumerate(refs):
        if pos % 25 == 0:
            print(f"[SDP] eval maps {source_name} -> {target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        z = state_z(model_info, patch[pos], device)
        q0 = state_logits_from_z(model_info, z, device).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
        qt = text_logits_np(patch[pos], visual_proj, anchors, 1.0).reshape(SIDE, SIDE)
        qd, drow = score_dynamic_grid(model_info, z, dyn, device)
        local_t, local_d, _ = local_maps(model_info, clip_model, preprocess, path, visual_proj, anchors, dyn, device)
        local_base = lzp.local_zoom_map(model_info, clip_model, preprocess, path, device)
        q0_74 = 0.5 * torch_resize(q0, (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * local_base
        qt_74 = 0.5 * torch_resize(qt, (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * local_t
        qd_74 = 0.5 * torch_resize(qd, (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * local_d
        maps["E0_Baseline_LRP"].append(q0_74.astype(np.float32))
        maps["E1_StaticLearnedPrompt_TextOnly"].append(qt_74.astype(np.float32))
        maps["E2_StateGuidedDynamicPrompt"].append(qd_74.astype(np.float32))
        mask = mask_np(item)
        metas.append({"category": cat, "label": label, "mask": mask, "img_path": path})
        gm = torch_resize(mask.astype(np.float32), qd_74.shape, align_corners=True) > 0.5
        b0 = q0_74.reshape(-1)
        e2 = qd_74.reshape(-1)
        y = gm.reshape(-1)
        an = b0[y].astype(np.float32) if np.any(y) else np.asarray([], dtype=np.float32)
        no = b0[~y].astype(np.float32) if np.any(~y) else np.asarray([], dtype=np.float32)
        if an.size and no.size:
            fn = y & (b0 < np.percentile(an, 20))
            fp = (~y) & (b0 > np.percentile(no, 95))
            tn = (~y) & (b0 <= np.percentile(no, 50))
            diag.append({
                "source": source_name,
                "dataset": target,
                "category": cat,
                "image_index": pos,
                **drow,
                "fn_e2_minus_e0_mean": float(np.mean(e2[fn.reshape(-1)] - b0[fn.reshape(-1)])) if np.any(fn) else float("nan"),
                "fp_e2_minus_e0_mean": float(np.mean(e2[fp.reshape(-1)] - b0[fp.reshape(-1)])) if np.any(fp) else float("nan"),
                "tn_e2_minus_e0_mean": float(np.mean(e2[tn.reshape(-1)] - b0[tn.reshape(-1)])) if np.any(tn) else float("nan"),
                "fn_count": int(np.sum(fn)),
                "fp_count": int(np.sum(fp)),
            })
    return maps, metas, diag


def evaluate_pilot(models, visual_model, visual_proj, prompts, dynamics, smoke):
    old_variants = lzp.VARIANTS
    variants = ("E0_Baseline_LRP", "E1_StaticLearnedPrompt_TextOnly", "E2_StateGuidedDynamicPrompt")
    lzp.VARIANTS = variants
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    raw, percat, diag = [], [], []
    try:
        for source_name, target in PILOT:
            model_info = models[source_name]
            cells, _audits = protocol_cells(source_name, "cuda" if torch.cuda.is_available() else "cpu", smoke)
            for cell_id, tname, dataset, idx_or_rows, patch, kind in cells:
                if tname != target:
                    continue
                maps, metas, d = build_eval_maps(source_name, target, dataset, idx_or_rows, patch, kind, model_info, visual_model, preprocess, visual_proj, prompts[source_name], dynamics[source_name], "cuda" if torch.cuda.is_available() else "cpu", smoke)
                rr, cc = lzp.evaluate_maps(source_name, target, maps, metas)
                raw.extend(rr)
                percat.extend(cc)
                diag.extend(d)
    finally:
        lzp.VARIANTS = old_variants
    return raw, percat, diag


def summarize(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in PILOT:
        b = by.get((source, target, "E0_Baseline_LRP"))
        if not b:
            continue
        row = {"source": source, "dataset": target, "baseline_P_AP": fnum(b, "M_FR_P_AP")}
        for variant in ("E1_StaticLearnedPrompt_TextOnly", "E2_StateGuidedDynamicPrompt"):
            r = by.get((source, target, variant))
            if not r:
                continue
            row[f"{variant}_P_AP"] = fnum(r, "M_FR_P_AP")
            row[f"{variant}_Delta_P_AP"] = fnum(r, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            row[f"{variant}_P_AUROC"] = fnum(r, "M_FR_P_AUROC")
            row[f"{variant}_P_F1max"] = fnum(r, "M_FR_P_F1max")
            row[f"{variant}_PRO"] = fnum(r, "M_FR_PRO")
        rows.append(row)
    return rows


def prompt_geometry(models, prompts, visual_proj):
    rows = []
    for source, anchors in prompts.items():
        t_n, t_a = anchors
        d_t = unit_np(t_a - t_n)
        d_s = unit_np(models[source]["v1"] @ visual_proj)
        cosv = float(np.clip(d_s @ d_t, -1.0, 1.0))
        d_parallel = cosv * d_s
        d_perp = d_t - d_parallel
        rows.append({
            "source": source,
            "cos_d_s_d_t": cosv,
            "angle_deg": float(np.degrees(np.arccos(cosv))),
            "dt_perp_over_dt": float(np.linalg.norm(d_perp) / max(np.linalg.norm(d_t), 1e-12)),
            "cos_tN_tA": float(t_n @ t_a / max(np.linalg.norm(t_n) * np.linalg.norm(t_a), 1e-12)),
        })
    return rows


def decide(summary):
    e2 = np.asarray([fnum(r, "E2_StateGuidedDynamicPrompt_Delta_P_AP") for r in summary], dtype=np.float64)
    e1 = np.asarray([fnum(r, "E1_StaticLearnedPrompt_TextOnly_Delta_P_AP") for r in summary], dtype=np.float64)
    e2 = e2[np.isfinite(e2)]
    e1 = e1[np.isfinite(e1)]
    if e2.size and int(np.sum(e2 > 0)) >= 2 and float(e2.mean()) >= 1.0 and float(np.mean(e2 - e1[: e2.size])) >= 0.5:
        outcome = "STATE_GUIDED_DYNAMIC_PROMPT_HAS_TRANSFERABLE_VALUE"
    elif e2.size and int(np.sum(e2 > 0)) >= 2 and float(e2.mean()) > 0:
        outcome = "QUERY_CONDITIONED_SEMANTICS_OUTPERFORMS_STATIC_TEXT"
    elif e2.size and np.nanmax(e2) > 1.0 and float(e2.mean()) <= 0:
        outcome = "DYNAMIC_PROMPT_IS_TARGET_SENSITIVE"
    else:
        outcome = "STATE_GUIDED_DYNAMIC_PROMPT_HAS_NO_COMPETITIVE_VALUE"
    return outcome, {
        "E2_mean_delta_P_AP": float(e2.mean()) if e2.size else float("nan"),
        "E2_median_delta_P_AP": float(np.median(e2)) if e2.size else float("nan"),
        "E2_improved": int(np.sum(e2 > 0)) if e2.size else 0,
        "E2_non_drop_ge_minus_0_2": int(np.sum(e2 >= -0.2)) if e2.size else 0,
        "E1_mean_delta_P_AP": float(e1.mean()) if e1.size else float("nan"),
    }


def write_report(outcome, stats, summary, geom, diag, runtime):
    lines = [
        "# Phase SDP: State-Guided Dynamic Prompt Feedback",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is the frozen Canonical Two-State + NTR + Fixed LRP path. E1 reuses/trains object-agnostic AnomalyCLIP-style static normal/abnormal prompts. E2 uses a FAPrompt-DAP-inspired image-conditioned abnormal prompt residual, but feeds it back only through a query-specific anomaly state `s_A'`; no visual/text score addition is used.",
        "",
        "## Prompt Geometry",
        "",
        "| Source | cos(d_s,d_t) | angle | ||d_t_perp||/||d_t|| | cos(tN,tA) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in geom:
        lines.append(f"| {r['source']} | {r['cos_d_s_d_t']:.4f} | {r['angle_deg']:.2f} | {r['dt_perp_over_dt']:.4f} | {r['cos_tN_tA']:.4f} |")
    lines += [
        "",
        "## Pilot Results",
        "",
        "| Source | Target | E0 P-AP | E1 P-AP | E1 Delta | E2 P-AP | E2 Delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['source']} | {r['dataset']} | {fnum(r,'baseline_P_AP'):.4f} | {fnum(r,'E1_StaticLearnedPrompt_TextOnly_P_AP'):.4f} | {fnum(r,'E1_StaticLearnedPrompt_TextOnly_Delta_P_AP'):+.4f} | {fnum(r,'E2_StateGuidedDynamicPrompt_P_AP'):.4f} | {fnum(r,'E2_StateGuidedDynamicPrompt_Delta_P_AP'):+.4f} |")
    lines += [
        "",
        "## Gate Summary",
        "",
        f"- E2 mean Delta P-AP: {stats['E2_mean_delta_P_AP']:+.4f}",
        f"- E2 median Delta P-AP: {stats['E2_median_delta_P_AP']:+.4f}",
        f"- E2 improved/non-drop: {stats['E2_improved']}/3 improved, {stats['E2_non_drop_ge_minus_0_2']}/3 non-drop >= -0.2.",
        f"- E1 static text-only mean Delta P-AP vs E0: {stats['E1_mean_delta_P_AP']:+.4f}",
        "",
        "## Error-Rescue Diagnostic",
    ]
    if diag:
        fn = np.asarray([fnum(r, "fn_e2_minus_e0_mean") for r in diag], dtype=np.float64)
        fp = np.asarray([fnum(r, "fp_e2_minus_e0_mean") for r in diag], dtype=np.float64)
        ang = np.asarray([fnum(r, "angle_sA_prime_sA") for r in diag], dtype=np.float64)
        lines += [
            f"- Mean E2 amplification on E0 false-negative anomaly pixels: {np.nanmean(fn):+.6f}.",
            f"- Mean E2 amplification on E0 false-positive normal pixels: {np.nanmean(fp):+.6f}.",
            f"- Query state angle mean/P95: {np.nanmean(ang):.4f} / {np.nanpercentile(ang,95):.4f} degrees; trust region cap = {TRUST_DEG:.1f} degrees.",
        ]
    else:
        lines.append("- No diagnostic rows were produced.")
    lines += [
        "",
        "## Questions",
        "",
        "Q1: Static prompt semantic signal is evaluated by E1 text-only. If E1 stays below E0, it is not competitive as a standalone route.",
        "Q2: Dynamic prompt feedback is evaluated by E2-E0 and E2-E1. Stable positive E2 deltas are required before adding tail-aware E3.",
        "Q3: Value source is considered query-conditioned semantics only if E2 improves over both E0 and E1 without exceeding the state trust region.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, visual_proj, logit_scale = load_clip_text_model(device)
    visual_model = load_visual_forward_model(device)
    models = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    prompts, curves = {}, []
    for source in ("Full VisA", "Full MVTec"):
        p, c, reused = load_or_train_static_prompt(source, clip_model, visual_proj, logit_scale, device, args.smoke)
        with torch.no_grad():
            prompts[source] = p().detach().cpu().numpy().astype(np.float32)
        for row in c:
            rr = dict(row)
            rr["source"] = source
            rr["reused"] = bool(reused)
            curves.append(rr)
    dynamics, dyn_curves = {}, []
    for source in ("Full VisA", "Full MVTec"):
        d, c, reused = train_dynamic(source, models[source], prompts[source], visual_proj, device, args.smoke)
        dynamics[source] = d
        for row in c:
            rr = dict(row)
            rr["reused"] = bool(reused)
            dyn_curves.append(rr)
    raw, percat, diag = evaluate_pilot(models, visual_model, visual_proj, prompts, dynamics, args.smoke)
    summary = summarize(raw)
    geom = prompt_geometry(models, prompts, visual_proj)
    outcome, stats = decide(summary)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat)
    write_csv(OUT / "CONTROL_COMPARISON.csv", summary)
    write_csv(OUT / "PROMPT_GEOMETRY.csv", geom)
    write_csv(OUT / "TRAINING_CURVES.csv", curves + dyn_curves)
    write_csv(OUT / "ERROR_RESCUE_DIAGNOSTIC.csv", diag)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "stats": stats,
        "runtime_sec": runtime,
        "pilot": PILOT,
        "E3_tail_aware_dynamic_prompt": "NOT_RUN_FIRST_PASS_E2_GATE_NOT_EVALUATED_IN_SCRIPT",
        "forbidden_controls": {
            "target_training": False,
            "target_statistics": False,
            "score_fusion": False,
            "dynamic_normal_prompt": False,
            "clip_backbone_trainable": False,
        },
    })
    write_report(outcome, stats, summary, geom, diag, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
