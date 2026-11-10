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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import tokenize_texts
from experiments.phase_frozen_semantic_anchor_fusion import (
    EPS,
    NORMAL_PROMPTS,
    ANOMALY_PROMPTS,
    MOECLIP,
    VISUALAD,
    corr,
    load_b0_rows,
    load_hardnormal_model,
    target_cells,
    target_item,
)
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np, patch_labels


OUT = Path("results/phase_learnable_text_semantic_compatibility/formal")
EPOCHS = 10
N_CTX = 8
CHUNK = 128


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


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


class PromptPair(nn.Module):
    def __init__(self, clip_model, mode="semantic"):
        super().__init__()
        self.clip_model = clip_model
        dtype = clip_model.token_embedding.weight.dtype
        width = clip_model.token_embedding.weight.shape[1]
        self.ctx_n = nn.Parameter(torch.empty(N_CTX, width, dtype=dtype))
        self.ctx_a = nn.Parameter(torch.empty(N_CTX, width, dtype=dtype))
        nn.init.normal_(self.ctx_n, std=0.02)
        nn.init.normal_(self.ctx_a, std=0.02)
        self.mode = mode
        with torch.no_grad():
            if mode == "semantic":
                emb_n, tok_n = self._make_suffix("normal object")
                emb_a, tok_a = self._make_suffix("abnormal object")
            elif mode == "random":
                emb_n, tok_n = self._make_random_suffix(11)
                emb_a, tok_a = self._make_random_suffix(29)
            else:
                raise ValueError(mode)
        self.register_buffer("prefix", clip_model.token_embedding(tokenize_texts([""] ).to(next(clip_model.parameters()).device))[0, :1].detach().clone().to(dtype))
        self.register_buffer("suffix_n", emb_n.detach().clone().to(dtype))
        self.register_buffer("suffix_a", emb_a.detach().clone().to(dtype))
        self.register_buffer("token_n", tok_n.detach().clone())
        self.register_buffer("token_a", tok_a.detach().clone())

    def _make_suffix(self, text):
        device = next(self.clip_model.parameters()).device
        tok = tokenize_texts([text]).to(device)
        ids = tok[0]
        nz = torch.nonzero(ids, as_tuple=False).flatten()
        suffix_ids = ids[nz[1:]]
        suffix = self.clip_model.token_embedding(suffix_ids).type(self.clip_model.dtype)
        full = torch.zeros((77,), dtype=tok.dtype, device=device)
        full[0] = ids[0]
        full[1 + N_CTX : 1 + N_CTX + len(suffix_ids)] = suffix_ids
        return suffix, full

    def _make_random_suffix(self, seed):
        device = next(self.clip_model.parameters()).device
        width = self.clip_model.token_embedding.weight.shape[1]
        rng = np.random.default_rng(seed)
        rand = torch.tensor(rng.normal(scale=0.02, size=(1, width)), dtype=self.clip_model.dtype, device=device)
        eot_tok = tokenize_texts([""] ).to(device)[0, 1]
        eot = self.clip_model.token_embedding(eot_tok.view(1)).type(self.clip_model.dtype)
        suffix = torch.cat([rand, eot], dim=0)
        full = torch.zeros((77,), dtype=torch.int, device=device)
        full[0] = tokenize_texts([""] ).to(device)[0, 0]
        full[1 + N_CTX] = 1
        full[2 + N_CTX] = eot_tok
        return suffix, full

    def _prompt(self, ctx, suffix):
        pad_len = 77 - 1 - N_CTX - suffix.shape[0]
        pad = torch.zeros((pad_len, suffix.shape[1]), dtype=suffix.dtype, device=suffix.device)
        return torch.cat([self.prefix, ctx, suffix, pad], dim=0)

    def forward(self):
        prompts = torch.stack([self._prompt(self.ctx_n, self.suffix_n), self._prompt(self.ctx_a, self.suffix_a)], dim=0)
        tokens = torch.stack([self.token_n, self.token_a], dim=0).long()
        text = self.clip_model.encode_text_learn(prompts, tokens).float()
        return F.normalize(text, dim=-1)


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
def frozen_text_anchors(model, device):
    prompts = ["normal object", "abnormal object"]
    tok = tokenize_texts(prompts).long().to(device)
    text = F.normalize(model.encode_text(tok).float(), dim=-1)
    return text.detach().cpu().numpy().astype(np.float32)


def source_arrays(q_pc1, visual_proj, smoke):
    cache = OUT / ("SOURCE_PATCH_TEXT_SPACE_v2_smoke.npz" if smoke else "SOURCE_PATCH_TEXT_SPACE_v2.npz")
    if cache.exists():
        z = np.load(cache)
        return z["text_x"].astype(np.float32), z["state_x"].astype(np.float32), z["y"].astype(np.float32)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, smoke)
    rng = np.random.default_rng(SEED)
    xs_t, xs_s, ys = [], [], []
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    for _cat, positions in slots.items():
        pools = {0: [], 1: []}
        for pos in positions:
            y_full = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
            raw = np.asarray(patch[int(pos)], dtype=np.float32).reshape(-1, patch.shape[-1])
            text_x = norm_np(norm_np(raw) @ visual_proj)
            state_x = norm_np(remove_pc1(raw, q_pc1).reshape(-1, raw.shape[-1]))
            for cls_id in (0, 1):
                ids = np.flatnonzero(y_full == cls_id)
                if ids.size == 0:
                    continue
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(part[0].shape[0] for part in pools[cls_id])
                if room <= 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                pools[cls_id].append((text_x[ids], state_x[ids], np.full((ids.size,), cls_id, dtype=np.float32)))
        for cls_id in (0, 1):
            for tx, sx, yy in pools[cls_id]:
                xs_t.append(tx.astype(np.float16))
                xs_s.append(sx.astype(np.float16))
                ys.append(yy.astype(np.float32))
    text_x = np.concatenate(xs_t, axis=0)
    state_x = np.concatenate(xs_s, axis=0)
    y = np.concatenate(ys, axis=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, text_x=text_x, state_x=state_x, y=y)
    return text_x.astype(np.float32), state_x.astype(np.float32), y.astype(np.float32)


def train_prompt(mode, clip_model, text_x, y, logit_scale, smoke, device):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"P1_{mode}_prompt_{suffix}.pt"
    if ckpt.exists():
        prompt = PromptPair(clip_model, "semantic" if mode == "learnable" else "random").to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        with torch.no_grad():
            prompt.ctx_n.copy_(payload["ctx_n"].to(device=device, dtype=prompt.ctx_n.dtype))
            prompt.ctx_a.copy_(payload["ctx_a"].to(device=device, dtype=prompt.ctx_a.dtype))
        return prompt.eval(), payload["curves"], True
    prompt = PromptPair(clip_model, "semantic" if mode == "learnable" else "random").to(device)
    opt = torch.optim.AdamW(prompt.parameters(), lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    x = text_x
    labels = y
    pos = np.flatnonzero(labels > 0.5)
    neg = np.flatnonzero(labels < 0.5)
    steps = max(1, int(math.ceil(len(labels) / BATCH_SIZE)))
    if smoke:
        steps = min(steps, 3)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _step in range(steps):
            half = BATCH_SIZE // 2
            ids_p = rng.choice(pos, size=half, replace=pos.size < half)
            ids_n = rng.choice(neg, size=BATCH_SIZE - half, replace=neg.size < BATCH_SIZE - half)
            ids = np.concatenate([ids_p, ids_n])
            rng.shuffle(ids)
            xb = torch.tensor(x[ids], dtype=torch.float32, device=device)
            yb = torch.tensor(labels[ids], dtype=torch.float32, device=device)
            text = prompt()
            logits = logit_scale * (xb @ text.T)
            loss = F.cross_entropy(logits, yb.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append(float(loss.detach().cpu()))
        row = {"prompt": mode, "epoch": epoch, "loss": float(np.mean(vals))}
        curves.append(row)
        print(f"[TextGate] {mode} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f}", flush=True)
    torch.save({"ctx_n": prompt.ctx_n.detach().cpu(), "ctx_a": prompt.ctx_a.detach().cpu(), "curves": curves}, ckpt)
    return prompt.eval(), curves, False


@torch.no_grad()
def score_state(state_model, raw_patch, q_pc1, device):
    raw_state = norm_np(remove_pc1(np.asarray(raw_patch, dtype=np.float32), q_pc1).reshape(-1, raw_patch.shape[-1]))
    x = torch.tensor(raw_state, dtype=torch.float32, device=device)
    logits, _ = state_model.logits(x)
    return logits.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def score_text(raw_patch, visual_proj, anchors, logit_scale, device):
    raw = norm_np(np.asarray(raw_patch, dtype=np.float32).reshape(-1, raw_patch.shape[-1]))
    x = norm_np(raw @ visual_proj)
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    t = torch.tensor(anchors, dtype=torch.float32, device=device)
    logits = logit_scale * (xt @ F.normalize(t, dim=-1).T)
    return (logits[:, 1] - logits[:, 0]).detach().cpu().numpy().astype(np.float32)


def calibration(state_model, q_pc1, visual_proj, anchors_by_name, logit_scale, device, smoke):
    cache = OUT / ("SOURCE_CALIBRATION_smoke.npz" if smoke else "SOURCE_CALIBRATION.npz")
    if cache.exists():
        z = np.load(cache)
        return {k: float(z[k]) for k in z.files}
    _name, _dataset, _indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    n = min(48, patch.shape[0]) if smoke else patch.shape[0]
    vals = {"visual": []}
    for key in anchors_by_name:
        vals[key] = []
    for i in range(n):
        vals["visual"].append(score_state(state_model, patch[i], q_pc1, device))
        for key, anchors in anchors_by_name.items():
            vals[key].append(score_text(patch[i], visual_proj, anchors, logit_scale, device))
    out = {}
    for key, parts in vals.items():
        arr = np.concatenate(parts).astype(np.float32)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(max(arr.std(), EPS))
    np.savez_compressed(cache, **out)
    return out


def zscore(x, mean, std):
    return ((np.asarray(x, dtype=np.float32) - float(mean)) / max(float(std), EPS)).astype(np.float32)


def conditional_auc(y, visual, text, bins=20):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    visual = np.asarray(visual, dtype=np.float32).reshape(-1)
    text = np.asarray(text, dtype=np.float32).reshape(-1)
    qs = np.quantile(visual, np.linspace(0, 1, bins + 1))
    aucs = []
    for i in range(bins):
        mask = (visual >= qs[i]) & (visual <= qs[i + 1] if i == bins - 1 else visual < qs[i + 1])
        if np.unique(y[mask]).size == 2:
            aucs.append(safe_auc(y[mask], text[mask]))
    return float(np.mean(aucs)) if aucs else float("nan"), float(np.std(aucs)) if aucs else float("nan")


def resize_binary_mask(item):
    return patch_labels(item).astype(np.uint8)


@torch.no_grad()
def evaluate_variants(state_model, q_pc1, visual_proj, anchors, stats, logit_scale, device, smoke):
    raw, percat, comp, err = [], [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        variants = ["P0_FrozenText", "P1_LearnableText", "P2_VisualLearnableTextFusion", "P3_VisualRandomSemanticPrompt"]
        maps = {v: np.zeros((n, SIDE, SIDE), dtype=np.float16) for v in variants}
        images = {v: np.zeros((n,), dtype=np.float32) for v in variants}
        all_v, all_t, all_gt = [], [], []
        for i in range(n):
            l_v = score_state(state_model, patch[i], q_pc1, device).reshape(SIDE, SIDE)
            l_frozen = score_text(patch[i], visual_proj, anchors["frozen"], logit_scale, device).reshape(SIDE, SIDE)
            l_learn = score_text(patch[i], visual_proj, anchors["learnable"], logit_scale, device).reshape(SIDE, SIDE)
            l_rand = score_text(patch[i], visual_proj, anchors["random"], logit_scale, device).reshape(SIDE, SIDE)
            block = {
                "P0_FrozenText": zscore(l_frozen, stats["frozen_mean"], stats["frozen_std"]),
                "P1_LearnableText": zscore(l_learn, stats["learnable_mean"], stats["learnable_std"]),
                "P2_VisualLearnableTextFusion": zscore(l_v, stats["visual_mean"], stats["visual_std"]) + zscore(l_learn, stats["learnable_mean"], stats["learnable_std"]),
                "P3_VisualRandomSemanticPrompt": zscore(l_v, stats["visual_mean"], stats["visual_std"]) + zscore(l_rand, stats["random_mean"], stats["random_std"]),
            }
            for name, arr in block.items():
                maps[name][i] = arr.astype(np.float16)
                images[name][i] = top1_image_score(arr)
            all_v.append(l_v.reshape(-1))
            all_t.append(l_learn.reshape(-1))
            item = target_item(dataset, idx_use[i], kind, i)
            all_gt.append(resize_binary_mask(item).reshape(-1))
        vf = np.concatenate(all_v)
        tf = np.concatenate(all_t)
        gt = np.concatenate(all_gt).astype(np.uint8)
        ca_mean, ca_std = conditional_auc(gt, vf, tf)
        comp.extend([
            {"dataset": cell_id, "split": "target", "stat": "corr_lV_lT", "value": corr(vf, tf)},
            {"dataset": cell_id, "split": "target", "stat": "text_conditional_P-AUROC_mean", "value": ca_mean},
            {"dataset": cell_id, "split": "target", "stat": "text_conditional_P-AUROC_std", "value": ca_std},
        ])
        normal_scores = vf[gt == 0]
        anomaly_scores = vf[gt == 1]
        fp = (gt == 0) & (vf > np.percentile(normal_scores, 95))
        fn = (gt == 1) & (vf < np.percentile(anomaly_scores, 20)) if anomaly_scores.size else np.zeros_like(gt, dtype=bool)
        err.extend([
            {"dataset": cell_id, "region": "B0_false_positive", "text_score_mean": float(tf[fp].mean()) if np.any(fp) else float("nan"), "count": int(np.sum(fp))},
            {"dataset": cell_id, "region": "B0_false_negative", "text_score_mean": float(tf[fn].mean()) if np.any(fn) else float("nan"), "count": int(np.sum(fn))},
        ])
        for variant in variants:
            print(f"[TextGate] eval {variant} on {cell_id}", flush=True)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, images[variant], maps[variant])
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat, comp, err


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        p0 = by[(ds, "P0_FrozenText")]
        p1 = by[(ds, "P1_LearnableText")]
        p2 = by[(ds, "P2_VisualLearnableTextFusion")]
        p3 = by[(ds, "P3_VisualRandomSemanticPrompt")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "P0_G": float(p0["G_pixel"]),
            "P1_G": float(p1["G_pixel"]),
            "P2_G": float(p2["G_pixel"]),
            "P3_G": float(p3["G_pixel"]),
            "P1_minus_P0_G": float(p1["G_pixel"]) - float(p0["G_pixel"]),
            "P2_minus_B0_G": float(p2["G_pixel"]) - float(b0["G_pixel"]),
            "P2_minus_P3_G": float(p2["G_pixel"]) - float(p3["G_pixel"]),
            "P2_minus_B0_P-AUROC": float(p2["P-AUROC"]) - float(b0["P-AUROC"]),
            "P2_minus_B0_P-AP": float(p2["P-AP"]) - float(b0["P-AP"]),
        })
    return rows


def decide(summary):
    p2_b0 = np.asarray([r["P2_minus_B0_G"] for r in summary], dtype=np.float32)
    p2_p3 = np.asarray([r["P2_minus_P3_G"] for r in summary], dtype=np.float32)
    p1_p0 = np.asarray([r["P1_minus_P0_G"] for r in summary], dtype=np.float32)
    by = {r["dataset"]: r for r in summary}
    mt_or_clinic = max(by["MVTec"]["P2_minus_B0_G"], by["CVC-ClinicDB"]["P2_minus_B0_G"])
    btad_ok = by["BTAD"]["P2_minus_B0_G"] >= -1.0
    if int(np.sum(p2_b0 > 0)) >= 2 and float(p2_b0.mean()) >= 1.0 and mt_or_clinic >= 1.5 and float(p2_p3.mean()) >= 0.5 and btad_ok:
        outcome = "LEARNABLE_TEXT_SEMANTICS_COMPLEMENT_VISUAL_STATE"
    elif float(p1_p0.mean()) > 1.0 and float(p2_b0.mean()) <= 0:
        outcome = "TEXT_PROMPT_HAS_SIGNAL_BUT_NOT_ADDITIVE"
    elif abs(float(p2_p3.mean())) < 0.5:
        outcome = "PROMPT_GAIN_NOT_LANGUAGE_SEMANTIC_SPECIFIC"
    elif abs(float(p1_p0.mean())) < 0.5:
        outcome = "LEARNABLE_TEXT_PROMPT_NO_NEW_SIGNAL"
    else:
        outcome = "TEXT_VISUAL_COMPLEMENTARITY_WEAK"
    return outcome, {
        "targets_improved_P2_over_B0": int(np.sum(p2_b0 > 0)),
        "mean_P2_minus_B0_G": float(p2_b0.mean()),
        "mean_P2_minus_P3_G": float(p2_p3.mean()),
        "mean_P1_minus_P0_G": float(p1_p0.mean()),
        "MVTec_P2_minus_B0_G": float(by["MVTec"]["P2_minus_B0_G"]),
        "BTAD_P2_minus_B0_G": float(by["BTAD"]["P2_minus_B0_G"]),
        "ClinicDB_P2_minus_B0_G": float(by["CVC-ClinicDB"]["P2_minus_B0_G"]),
        "BTAD_drop_ok": bool(btad_ok),
    }


def prompt_geometry(anchors, init_anchors, visual_v_text_space):
    rows = []
    for name in ("frozen", "learnable", "random"):
        t = anchors[name]
        d = unit_np(t[1] - t[0])
        rows.append({
            "prompt": name,
            "cos_tN_tA": float(t[0] @ t[1] / max(np.linalg.norm(t[0]) * np.linalg.norm(t[1]), EPS)),
            "cos_prompt_contrast_visual_contrast": float(d @ visual_v_text_space / max(np.linalg.norm(d) * np.linalg.norm(visual_v_text_space), EPS)) if d.shape == visual_v_text_space.shape else float("nan"),
            "contrast_norm": float(np.linalg.norm(t[1] - t[0])),
        })
    d0 = unit_np(init_anchors[1] - init_anchors[0])
    d1 = unit_np(anchors["learnable"][1] - anchors["learnable"][0])
    rows.append({"prompt": "learnable_change", "cos_tN_tA": float("nan"), "cos_prompt_contrast_visual_contrast": float(d0 @ d1), "contrast_norm": float(np.linalg.norm(d1 - d0))})
    return rows


def write_sota(raw):
    rows = []
    for row in raw:
        if row["variant"] != "P2_VisualLearnableTextFusion" or row["dataset"] not in MOECLIP:
            continue
        for ref_name, table, status in (("MoECLIP OpenCLIP ViT-L/14-336 518x518", MOECLIP, "REFERENCE_ONLY unless exact protocol/backbone match is confirmed"), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD, "REFERENCE_ONLY_OR_PROTOCOL_CAUTION")):
            ref = table[row["dataset"]]
            rows.append({
                "dataset": row["dataset"],
                "variant": row["variant"],
                "current_P-AUROC": float(row["P-AUROC"]),
                "current_P-AP": float(row["P-AP"]),
                "current_G": float(row["G_pixel"]),
                "reference": ref_name,
                "comparison_status": status,
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": float(row["P-AUROC"]) - ref["P-AUROC"],
                "Delta_P_AP": float(row["P-AP"]) - ref["P-AP"],
                "Delta_G": float(row["G_pixel"]) - ref["G_pixel"],
            })
    return rows


def write_report(outcome, stats, summary, runtime):
    lines = [
        "# Learnable Text-Semantic Compatibility Gate",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA -> MVTec / BTAD / CVC-ClinicDB. Visual HardNormal is frozen/reused. CLIP visual and text encoders are frozen; P1/P3 train only 8+8 context-token embeddings on source binary patch masks.",
        "",
        "| Dataset | B0 Visual | P0 FrozenText | P1 LearnableText | P2 Fusion | P3 RandomSemanticFusion | P2-B0 | P2-P3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['P0_G']:.4f} | {r['P1_G']:.4f} | {r['P2_G']:.4f} | {r['P3_G']:.4f} | {r['P2_minus_B0_G']:+.4f} | {r['P2_minus_P3_G']:+.4f} |")
    lines.extend([
        "",
        "## Gate",
        f"- targets improved P2 over B0: {stats['targets_improved_P2_over_B0']}/3",
        f"- mean P2-B0 G: {stats['mean_P2_minus_B0_G']:+.4f}",
        f"- mean P2-P3 G: {stats['mean_P2_minus_P3_G']:+.4f}",
        f"- mean P1-P0 G: {stats['mean_P1_minus_P0_G']:+.4f}",
        f"- BTAD drop ok: {stats['BTAD_drop_ok']}",
        "",
        "## External SOTA Position",
        "`SOTA_GAP.csv` reports MoECLIP and VisualAD references. Treat them as reference-only unless exact backbone, preprocessing, and evaluator parity are confirmed; the formal decision here is internal B0/P0/P1/P2/P3 under one local protocol.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TextGate] device={device}", flush=True)
    q_pc1, state_model = load_hardnormal_model(device)
    clip_model, visual_proj, logit_scale = load_clip_text_model(device)
    frozen = frozen_text_anchors(clip_model, device)
    text_x, _state_x, y = source_arrays(q_pc1, visual_proj, args.smoke)
    p1, curves1, reused1 = train_prompt("learnable", clip_model, text_x, y, logit_scale, args.smoke, device)
    p3, curves3, reused3 = train_prompt("random", clip_model, text_x, y, logit_scale, args.smoke, device)
    with torch.no_grad():
        learnable = p1().detach().cpu().numpy().astype(np.float32)
        random = p3().detach().cpu().numpy().astype(np.float32)
    anchors = {"frozen": frozen, "learnable": learnable, "random": random}
    stats = calibration(state_model, q_pc1, visual_proj, anchors, logit_scale, device, args.smoke)
    raw_new, per_new, comp, err = evaluate_variants(state_model, q_pc1, visual_proj, anchors, stats, logit_scale, device, args.smoke)
    raw_b0, per_b0 = load_b0_rows()
    if args.smoke:
        raw_b0 = raw_b0[:3]
        per_b0 = per_b0[:3]
    raw = raw_b0 + raw_new
    percat = per_b0 + per_new
    summary = summarize(raw)
    outcome, decision_stats = decide(summary)
    visual_v = unit_np(np.load("results/phase_hard_normal_contrast/formal/contrast_vectors.npz")["E1"].astype(np.float32))
    visual_v_text = unit_np(visual_v @ visual_proj)
    prompt_rows = prompt_geometry(anchors, frozen, visual_v_text)
    sota = write_sota(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "PROMPT_GEOMETRY.csv", prompt_rows)
    write_csv(OUT / "COMPLEMENTARITY.csv", comp + err)
    write_csv(OUT / "CONTROL_COMPARISON.csv", summary)
    write_csv(OUT / "TRAINING_CURVES.csv", curves1 + curves3)
    write_csv(OUT / "SOURCE_CALIBRATION.csv", [{"stat": k, "value": v} for k, v in stats.items()])
    write_csv(OUT / "SOTA_GAP.csv", sota)
    np.savez_compressed(OUT / "prompt_anchors.npz", frozen=frozen, learnable=learnable, random=random)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_category_vocabulary": False,
        "visual_encoder_frozen": True,
        "text_encoder_frozen": True,
        "trainable_parameters": "normal/abnormal prompt context tokens only",
        "n_ctx": N_CTX,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "temperature": logit_scale,
        "prompt_checkpoints_reused": {"P1": bool(reused1), "P3": bool(reused3)},
    })
    write_report(outcome, decision_stats, summary, runtime)
    print(json.dumps({"outcome": outcome, **decision_stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
