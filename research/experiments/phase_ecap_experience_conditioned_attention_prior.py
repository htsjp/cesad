import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_fova_field_of_view_amplification_audit import MAP_CACHE as FOVA_MAP_CACHE, slug as pair_slug
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np, quadrant_crops, torch_resize
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_sgpr import evaluate_maps
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, fnum
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np, patch_labels
from models.mlp import average_neighbor
from phase_x.gradient_visual_forward import _safe_resblock_forwards
from phase_xa.official_task_forward import visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_ecap_experience_conditioned_attention_prior/formal")
VARIANTS = ("E0_FixedLRP", "E1_FrozenAttentionReadout", "E2_ExperienceConditionedAttention")
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "DTD-Synthetic"),
)
ATTN_LAYER = 18
EPOCHS = 3
MAX_SOURCE_IMAGES = 240
SMOKE_SOURCE_IMAGES = 8
LR = 0.001
B_MAX = 1.0
TRUST_WEIGHT = 1.0
REL_FUSION_WEIGHT = 0.1


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite_mean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def checksum_module(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class PairwiseBiasNet(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phi):
        return B_MAX * torch.tanh(self.net(phi).squeeze(-1))


def source_dataset_refs(source, smoke):
    if source == "Full VisA":
        _name, dataset, indices, _cls, _patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    else:
        _name, dataset, indices, _patch = load_full_mvtec_source(False)
    by = {}
    for idx in indices:
        item = dataset[int(idx)]
        cat = str(item.get("cls_name", source))
        lab = int(item.get("anomaly", 0))
        by.setdefault((cat, lab), []).append(int(idx))
    rng = np.random.default_rng(0)
    limit = SMOKE_SOURCE_IMAGES if smoke else MAX_SOURCE_IMAGES
    chosen = []
    keys = sorted(by)
    while len(chosen) < limit and keys:
        progressed = False
        for key in keys:
            bucket = by[key]
            if not bucket:
                continue
            pos = int(rng.integers(0, len(bucket)))
            chosen.append(bucket.pop(pos))
            progressed = True
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    return dataset, chosen


def source_image_item(dataset, idx):
    return dataset[int(idx)]


def image_from_item(item, device):
    return item["img"].unsqueeze(0).to(device)


def canonicalize(model_info, z):
    z = F.normalize(z.float(), dim=-1)
    if not model_info["use_align"]:
        return z
    q = torch.tensor(model_info["q_pc1"], dtype=torch.float32, device=z.device)
    flat = z.reshape(-1, z.shape[-1])
    flat = flat - (flat @ q).unsqueeze(-1) * q.unsqueeze(0)
    flat = F.normalize(flat, dim=-1)
    flat = model_info["align_model"].adapt(flat)
    return F.normalize(flat, dim=-1).reshape_as(z)


def state_score(model_info, z):
    m = torch.tensor(model_info["m"], dtype=torch.float32, device=z.device)
    v = torch.tensor(model_info["v1"], dtype=torch.float32, device=z.device)
    return contrast_logits(z.reshape(-1, z.shape[-1]), m, float(model_info["alpha"]), v).reshape(z.shape[:-1])


def visual_attention_readout(clip_model, image, model_info, bias_net=None, return_attention=False):
    visual = clip_model.visual
    x = visual.conv1(image.to(next(visual.parameters()).device).type(clip_model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    side = int((visual.positional_embedding.shape[0] - 1) ** 0.5)
    new_side = int((x.shape[1] - 1) ** 0.5)
    pos = visual.positional_embedding
    if side != new_side:
        pos_patch = pos[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
        pos_patch = F.interpolate(pos_patch, (new_side, new_side), mode="bilinear", align_corners=False)
        pos_patch = pos_patch.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
        pos = torch.cat([pos[:1, :], pos_patch[0]], dim=0)
    x = x + pos.to(x.dtype)
    x = visual.ln_pre(x).permute(1, 0, 2)

    rel = None
    attn_diag = {}
    final_token = None
    with _safe_resblock_forwards(visual):
        for idx, block in enumerate(visual.transformer.resblocks, start=1):
            current = x[0] if isinstance(x, list) else x
            if idx == ATTN_LAYER:
                h = block.ln_1(current)
                attn_mod = block.attn
                if hasattr(attn_mod, "qkv"):
                    hn = h.transpose(0, 1)
                    bsz, tokens, channels = hn.shape
                    heads = int(attn_mod.num_heads)
                    qkv = attn_mod.qkv(hn).reshape(bsz, tokens, 3, heads, channels // heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    logits_h = (q @ k.transpose(-2, -1)) * float(attn_mod.scale)
                    v_concat = v.transpose(1, 2).reshape(bsz, tokens, channels)
                else:
                    hn = h
                    tokens, bsz, channels = hn.shape
                    heads = int(attn_mod.num_heads)
                    qkv = F.linear(hn, attn_mod.in_proj_weight, attn_mod.in_proj_bias)
                    qkv = qkv.reshape(tokens, bsz, 3, heads, channels // heads).permute(2, 1, 3, 0, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    logits_h = (q @ k.transpose(-2, -1)) / math.sqrt(channels // heads)
                    v_concat = v.transpose(1, 2).reshape(bsz, tokens, channels)
                logits = logits_h.mean(dim=1)[:, 1:, 1:].float()
                v_patch = v_concat[:, 1:, :].float()
                a0 = torch.softmax(logits, dim=-1)
                q_for_phi = None
                z_for_phi = None
                if bias_net is not None:
                    # The bias is conditioned on final scorer-space features. They
                    # are filled after the transformer pass; store logits/V now.
                    rel = {"logits": logits, "a0": a0, "v_patch": v_patch}
                else:
                    ctx = a0 @ v_patch
                    rel_feat = F.normalize(visual.ln_post(ctx).float(), dim=-1)
                    rel_feat = canonicalize(model_info, rel_feat)
                    d_s = torch.tensor(model_info["v1"], dtype=torch.float32, device=rel_feat.device)
                    q_rel = (rel_feat * F.normalize(d_s, dim=-1)).sum(dim=-1).reshape(-1)
                    ent = -(a0 * torch.log(a0.clamp_min(1e-8))).sum(dim=-1)
                    attn_diag = {
                        "attention_top1_probability": float(a0.max(dim=-1).values.mean().detach().cpu()),
                        "attention_entropy": float(ent.mean().detach().cpu()),
                        "kl_to_frozen": 0.0,
                        "bias_abs_mean": 0.0,
                    }
                    rel = {"q_rel": q_rel, "diag": attn_diag}
            x = block(x, ffn=False)
            final_token = x[0] if isinstance(x, list) else x

    token_nld = final_token.permute(1, 0, 2)
    z = F.normalize(visual.ln_post(token_nld).float()[:, 1:, :], dim=-1)
    z = average_neighbor(z)
    z = canonicalize(model_info, z).reshape(-1, z.shape[-1])
    q_state = state_score(model_info, z).reshape(-1).detach()

    if bias_net is not None:
        pack = rel
        logits = pack["logits"].squeeze(0)
        a0 = pack["a0"].squeeze(0)
        v_patch = pack["v_patch"].squeeze(0)
        n = logits.shape[0]
        side_now = int(round(math.sqrt(n)))
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, side_now, device=logits.device),
            torch.linspace(0, 1, side_now, device=logits.device),
            indexing="ij",
        )
        coord = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        dz = z @ z.T
        qi = q_state[:, None]
        qj = q_state[None, :]
        dx = coord[:, 0:1] - coord[:, 0:1].T
        dy = coord[:, 1:2] - coord[:, 1:2].T
        phi = torch.stack([qi.expand(n, n), qj.expand(n, n), torch.abs(qi - qj), dz, dx, dy], dim=-1)
        bias = bias_net(phi.reshape(-1, 6)).reshape(n, n)
        aexp = torch.softmax(logits + bias, dim=-1)
        ctx = aexp @ v_patch
        rel_feat = F.normalize(visual.ln_post(ctx.unsqueeze(0)).float(), dim=-1).squeeze(0)
        rel_feat = canonicalize(model_info, rel_feat)
        d_s = torch.tensor(model_info["v1"], dtype=torch.float32, device=rel_feat.device)
        q_rel = (rel_feat * F.normalize(d_s, dim=-1)).sum(dim=-1).reshape(-1)
        kl = torch.sum(aexp * (torch.log(aexp.clamp_min(1e-8)) - torch.log(a0.clamp_min(1e-8))), dim=-1)
        ent = -(aexp * torch.log(aexp.clamp_min(1e-8))).sum(dim=-1)
        attn_diag = {
            "attention_top1_probability": float(aexp.max(dim=-1).values.mean().detach().cpu()),
            "attention_entropy": float(ent.mean().detach().cpu()),
            "kl_to_frozen": float(kl.mean().detach().cpu()),
            "bias_abs_mean": float(bias.abs().mean().detach().cpu()),
        }
        rel = {"q_rel": q_rel, "diag": attn_diag, "a0": a0, "aexp": aexp, "q_state": q_state, "z": z}
    if return_attention:
        return rel, z, q_state
    return rel["q_rel"].reshape(SIDE, SIDE).detach().cpu().numpy().astype(np.float32), rel["diag"]


def relation_loss_for_item(clip_model, item, model_info, bias_net, device):
    pack = visual_attention_pack(clip_model, image_from_item(item, device), model_info)
    rel = relation_from_pack(clip_model, pack, model_info, bias_net=bias_net, return_attention=True)
    y = torch.tensor(patch_labels(item).reshape(-1), dtype=torch.long, device=device)
    a0 = rel["a0"]
    aexp = rel["aexp"]
    same = y[:, None].eq(y[None, :])
    same.fill_diagonal_(False)
    valid = same.any(dim=-1)
    same_mass = (aexp * same.float()).sum(dim=-1).clamp_min(1e-8)
    rel_loss = -torch.log(same_mass[valid]).mean() if valid.any() else torch.zeros((), dtype=torch.float32, device=device)
    kl = torch.sum(aexp * (torch.log(aexp.clamp_min(1e-8)) - torch.log(a0.clamp_min(1e-8))), dim=-1).mean()
    return rel_loss + TRUST_WEIGHT * kl, rel_loss.detach(), kl.detach(), rel["diag"]


@torch.no_grad()
def visual_attention_pack(clip_model, image, model_info):
    visual = clip_model.visual
    x = visual.conv1(image.to(next(visual.parameters()).device).type(clip_model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    side = int((visual.positional_embedding.shape[0] - 1) ** 0.5)
    new_side = int((x.shape[1] - 1) ** 0.5)
    pos = visual.positional_embedding
    if side != new_side:
        pos_patch = pos[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
        pos_patch = F.interpolate(pos_patch, (new_side, new_side), mode="bilinear", align_corners=False)
        pos_patch = pos_patch.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
        pos = torch.cat([pos[:1, :], pos_patch[0]], dim=0)
    x = x + pos.to(x.dtype)
    x = visual.ln_pre(x).permute(1, 0, 2)
    logits = a0 = v_patch = None
    final_token = None
    with _safe_resblock_forwards(visual):
        for idx, block in enumerate(visual.transformer.resblocks, start=1):
            current = x[0] if isinstance(x, list) else x
            if idx == ATTN_LAYER:
                h = block.ln_1(current)
                attn_mod = block.attn
                if hasattr(attn_mod, "qkv"):
                    hn = h.transpose(0, 1)
                    bsz, tokens, channels = hn.shape
                    heads = int(attn_mod.num_heads)
                    qkv = attn_mod.qkv(hn).reshape(bsz, tokens, 3, heads, channels // heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    logits_h = (q @ k.transpose(-2, -1)) * float(attn_mod.scale)
                    v_concat = v.transpose(1, 2).reshape(bsz, tokens, channels)
                else:
                    hn = h
                    tokens, bsz, channels = hn.shape
                    heads = int(attn_mod.num_heads)
                    qkv = F.linear(hn, attn_mod.in_proj_weight, attn_mod.in_proj_bias)
                    qkv = qkv.reshape(tokens, bsz, 3, heads, channels // heads).permute(2, 1, 3, 0, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    logits_h = (q @ k.transpose(-2, -1)) / math.sqrt(channels // heads)
                    v_concat = v.transpose(1, 2).reshape(bsz, tokens, channels)
                logits = logits_h.mean(dim=1)[:, 1:, 1:].float().squeeze(0).detach()
                a0 = torch.softmax(logits, dim=-1).detach()
                v_patch = v_concat[:, 1:, :].float().squeeze(0).detach()
            x = block(x, ffn=False)
            final_token = x[0] if isinstance(x, list) else x
    token_nld = final_token.permute(1, 0, 2)
    z = F.normalize(visual.ln_post(token_nld).float()[:, 1:, :], dim=-1)
    z = average_neighbor(z)
    z = canonicalize(model_info, z).reshape(-1, z.shape[-1]).detach()
    q_state = state_score(model_info, z).reshape(-1).detach()
    if logits is None:
        raise RuntimeError(f"attention layer {ATTN_LAYER} was not captured")
    return {"logits": logits, "a0": a0, "v_patch": v_patch, "z": z, "q_state": q_state}


def relation_from_pack(clip_model, pack, model_info, bias_net=None, return_attention=False):
    visual = clip_model.visual
    logits = pack["logits"]
    a0 = pack["a0"]
    v_patch = pack["v_patch"]
    z = pack["z"]
    q_state = pack["q_state"]
    if bias_net is None:
        attn = a0
        bias = torch.zeros_like(a0)
    else:
        n = logits.shape[0]
        side_now = int(round(math.sqrt(n)))
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, side_now, device=logits.device),
            torch.linspace(0, 1, side_now, device=logits.device),
            indexing="ij",
        )
        coord = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        dz = z @ z.T
        qi = q_state[:, None]
        qj = q_state[None, :]
        dx = coord[:, 0:1] - coord[:, 0:1].T
        dy = coord[:, 1:2] - coord[:, 1:2].T
        phi = torch.stack([qi.expand(n, n), qj.expand(n, n), torch.abs(qi - qj), dz, dx, dy], dim=-1)
        bias = bias_net(phi.reshape(-1, 6)).reshape(n, n)
        attn = torch.softmax(logits + bias, dim=-1)
    ctx = attn @ v_patch
    rel_feat = F.normalize(visual.ln_post(ctx.unsqueeze(0)).float(), dim=-1).squeeze(0)
    rel_feat = canonicalize(model_info, rel_feat)
    d_s = torch.tensor(model_info["v1"], dtype=torch.float32, device=rel_feat.device)
    q_rel = (rel_feat * F.normalize(d_s, dim=-1)).sum(dim=-1).reshape(-1)
    kl = torch.sum(attn * (torch.log(attn.clamp_min(1e-8)) - torch.log(a0.clamp_min(1e-8))), dim=-1)
    ent = -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1)
    diag = {
        "attention_top1_probability": float(attn.max(dim=-1).values.mean().detach().cpu()),
        "attention_entropy": float(ent.mean().detach().cpu()),
        "kl_to_frozen": float(kl.mean().detach().cpu()),
        "bias_abs_mean": float(bias.abs().mean().detach().cpu()),
    }
    out = {"q_rel": q_rel, "diag": diag, "a0": a0, "aexp": attn, "q_state": q_state, "z": z}
    return out if return_attention else (q_rel.reshape(SIDE, SIDE).detach().cpu().numpy().astype(np.float32), diag)


def train_bias(source, model_info, clip_model, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{source.replace(' ', '_')}_E2_bias_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        net = PairwiseBiasNet().to(device)
        net.load_state_dict(payload["state_dict"], strict=True)
        return net.eval(), payload["calibration"], payload["training_log"], payload["attention_audit"], True
    dataset, refs = source_dataset_refs(source, smoke)
    net = PairwiseBiasNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    logs, audit = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step, idx in enumerate(refs):
            item = source_image_item(dataset, idx)
            opt.zero_grad()
            loss, rel_loss, kl, diag = relation_loss_for_item(clip_model, item, model_info, net, device)
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(rel_loss.cpu()), float(kl.cpu())))
            audit.append({"source": source, "variant": "E2_ExperienceConditionedAttention", "epoch": epoch, "step": step + 1, **diag})
        arr = np.asarray(vals, dtype=np.float32)
        row = {"source": source, "variant": "E2_ExperienceConditionedAttention", "epoch": epoch, "loss": float(arr[:, 0].mean()), "relation_loss": float(arr[:, 1].mean()), "kl": float(arr[:, 2].mean())}
        logs.append(row)
        print(f"[ECAP] {source} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} kl={row['kl']:.4f}", flush=True)
    cal = calibrate_relation(source, model_info, clip_model, net, device, smoke, refs[: min(len(refs), 48 if not smoke else 4)])
    torch.save({"state_dict": net.state_dict(), "calibration": cal, "training_log": logs, "attention_audit": audit}, ckpt)
    return net.eval(), cal, logs, audit, False


@torch.no_grad()
def calibrate_relation(source, model_info, clip_model, net, device, smoke, refs=None):
    dataset, all_refs = source_dataset_refs(source, smoke)
    refs = refs or all_refs[: min(len(all_refs), 48 if not smoke else 4)]
    vals = {"E1_FrozenAttentionReadout": [], "E2_ExperienceConditionedAttention": []}
    for idx in refs:
        item = source_image_item(dataset, idx)
        image = image_from_item(item, device)
        pack = visual_attention_pack(clip_model, image, model_info)
        q1, _ = relation_from_pack(clip_model, pack, model_info, bias_net=None)
        q2, _ = relation_from_pack(clip_model, pack, model_info, bias_net=net)
        vals["E1_FrozenAttentionReadout"].append(q1.reshape(-1))
        vals["E2_ExperienceConditionedAttention"].append(q2.reshape(-1))
    cal = {}
    for variant, parts in vals.items():
        x = np.concatenate(parts, axis=0).astype(np.float32)
        cal[variant] = {"mean": float(np.mean(x)), "std": float(np.std(x) + 1e-6), "samples": int(x.size)}
    return cal


@torch.no_grad()
def relation_lrp_maps_both(model_info, clip_model, preprocess, img_path, device, bias_net):
    item_img = preprocess_image_file(preprocess, img_path, device)
    pack = visual_attention_pack(clip_model, item_img, model_info)
    g1, dg1 = relation_from_pack(clip_model, pack, model_info, bias_net=None)
    g2, dg2 = relation_from_pack(clip_model, pack, model_info, bias_net=bias_net)
    crops = quadrant_crops(img_path)
    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    rels1, rels2, diags1, diags2 = [], [], [], []
    for b in range(batch.shape[0]):
        pack_b = visual_attention_pack(clip_model, batch[b:b + 1], model_info)
        r1, d1 = relation_from_pack(clip_model, pack_b, model_info, bias_net=None)
        r2, d2 = relation_from_pack(clip_model, pack_b, model_info, bias_net=bias_net)
        rels1.append(r1); rels2.append(r2); diags1.append(d1); diags2.append(d2)

    def stitch(rels):
        local = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
        local[:SIDE, :SIDE] = rels[0]
        local[:SIDE, SIDE:] = rels[1]
        local[SIDE:, :SIDE] = rels[2]
        local[SIDE:, SIDE:] = rels[3]
        return local

    out1 = 0.5 * torch_resize(g1, (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * stitch(rels1)
    out2 = 0.5 * torch_resize(g2, (SIDE * 2, SIDE * 2), align_corners=True) + 0.5 * stitch(rels2)

    def diag_mean(first, rest):
        diag = {}
        all_diag = [first] + rest
        for key in ("attention_top1_probability", "attention_entropy", "kl_to_frozen", "bias_abs_mean"):
            diag[key] = float(np.mean([d[key] for d in all_diag]))
        return diag

    return out1.astype(np.float32), diag_mean(dg1, diags1), out2.astype(np.float32), diag_mean(dg2, diags2)


def preprocess_image_file(preprocess, path, device):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    img.close()
    return x


def load_e0_maps(source, target, n):
    path = FOVA_MAP_CACHE / f"{pair_slug(source, target)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Fixed LRP/FOVA map cache for {source}->{target}: {path}")
    data = np.load(path)
    q = np.asarray(data["q_lrp"][:n], dtype=np.float32)
    if q.ndim != 3 or q.shape[1:] != (SIDE * 2, SIDE * 2):
        raise ValueError(f"unexpected q_lrp shape {q.shape} in {path}")
    return q


def pair_meta(source, target, dataset, refs, kind):
    metas, rows = [], []
    for pos, ref in enumerate(refs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        mask = mask_np(item)
        morph = morphology(mask)
        row = {"source": source, "dataset": target, "category": cat, "image_index": pos, "label": int(label), "img_path": path, **morph}
        rows.append(row)
        metas.append({"category": cat, "label": int(label), "mask": mask, "img_path": path})
    assign_groups(rows)
    for meta, row in zip(metas, rows):
        meta.update({k: row[k] for k in ("group_small", "group_large", "group_thin", "group_thick")})
    return metas, rows


def morphology(mask):
    mask = np.asarray(mask, dtype=bool)
    pix = int(mask.sum())
    h, w = mask.shape
    if pix == 0:
        return {"num_anomaly_patches": 0, "thinness": 0.0}
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = mask & (~eroded)
    gm = torch.tensor(mask.astype(np.float32)).view(1, 1, h, w)
    patch = F.interpolate(gm, size=(SIDE, SIDE), mode="nearest").view(SIDE, SIDE).numpy() > 0.5
    return {"num_anomaly_patches": int(patch.sum()), "thinness": float(boundary.sum() / max(pix, 1))}


def assign_groups(rows):
    pos = [r for r in rows if int(r["label"]) == 1 and fnum(r, "num_anomaly_patches") > 0]
    patch_vals = [fnum(r, "num_anomaly_patches") for r in pos]
    thin_vals = [fnum(r, "thinness") for r in pos]
    small_thr = float(np.percentile(patch_vals, 25)) if patch_vals else float("nan")
    large_thr = float(np.percentile(patch_vals, 75)) if patch_vals else float("nan")
    thin_thr = float(np.percentile(thin_vals, 75)) if thin_vals else float("nan")
    for r in rows:
        p = fnum(r, "num_anomaly_patches")
        t = fnum(r, "thinness")
        r["group_small"] = bool(np.isfinite(small_thr) and p > 0 and p <= small_thr)
        r["group_large"] = bool(np.isfinite(large_thr) and p >= large_thr)
        r["group_thin"] = bool(np.isfinite(thin_thr) and p > 0 and t >= thin_thr)
        r["group_thick"] = bool(np.isfinite(thin_thr) and p > 0 and t < thin_thr)


def load_pair(source, target, device, smoke):
    cells, _audits = protocol_cells(source, device, smoke)
    for _cell_id, dst, dataset, refs, _patch, kind in cells:
        if dst == target:
            return dataset, refs, kind
    raise KeyError(f"missing target cell {source}->{target}")


def build_maps(source, target, model_info, clip_model, preprocess, bias_net, calibration, dataset, refs, kind, device, smoke):
    n = min(24, len(refs)) if smoke else len(refs)
    refs_use = refs[:n]
    metas, meta_rows = pair_meta(source, target, dataset, refs_use, kind)
    e0 = load_e0_maps(source, target, n)
    maps = {v: [] for v in VARIANTS}
    attn_rows = []
    for i, meta in enumerate(metas):
        if i % 25 == 0:
            print(f"[ECAP] {source}->{target} image {i}/{n}", flush=True)
        maps["E0_FixedLRP"].append(e0[i].astype(np.float32))
        r1, d1, r2, d2 = relation_lrp_maps_both(model_info, clip_model, preprocess, meta["img_path"], device, bias_net)
        for variant, rel, diag in (("E1_FrozenAttentionReadout", r1, d1), ("E2_ExperienceConditionedAttention", r2, d2)):
            cal = calibration[variant]
            zrel = (rel - cal["mean"]) / max(cal["std"], 1e-6)
            maps[variant].append((e0[i] + REL_FUSION_WEIGHT * zrel).astype(np.float32))
            attn_rows.append({"source": source, "dataset": target, "variant": variant, "category": meta["category"], "image_index": i, **diag})
    return refs_use, maps, metas, meta_rows, attn_rows


def summarize_metrics(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in VARIANTS[1:]:
        vals, improved, nondrop = [], 0, 0
        for source, target in PILOT_PAIRS:
            b = by.get((source, target, VARIANTS[0]))
            e = by.get((source, target, variant))
            if not (b and e):
                continue
            d = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            vals.append(d)
            improved += int(np.isfinite(d) and d > 0)
            nondrop += int(np.isfinite(d) and d >= -0.2)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "E_minus_E0_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "E_minus_E0_M_FR_P_AP": d,
                "E_minus_E0_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "E_minus_E0_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
        rows.append({"source": "SUMMARY", "dataset": "PILOT", "variant": variant, "mean_delta_M_FR_P_AP": finite_mean(vals), "median_delta_M_FR_P_AP": finite_median(vals), "improved": improved, "non_drop_ge_minus_0_2": nondrop, "total": len(vals)})
    return rows


def morphology_diagnostics(cat_rows, meta_rows):
    cat_group = {}
    for r in meta_rows:
        key = (r["source"], r["dataset"], r["category"])
        d = cat_group.setdefault(key, {g: [] for g in ("small", "large", "thin", "thick")})
        for g in d:
            d[g].append(bool(r.get(f"group_{g}", False)))
    group_by = {k: {g: bool(np.mean(v) >= 0.5) if v else False for g, v in val.items()} for k, val in cat_group.items()}
    by = {(r["source"], r["dataset"], r["category"], r["variant"], r.get("protocol")): r for r in cat_rows}
    rows = []
    for key, groups in group_by.items():
        source, target, cat = key
        base = by.get((source, target, cat, VARIANTS[0], "M"))
        if not base:
            continue
        for variant in VARIANTS[1:]:
            e = by.get((source, target, cat, variant, "M"))
            if not e:
                continue
            row = {"source": source, "dataset": target, "category": cat, "variant": variant, **{f"group_{g}": v for g, v in groups.items()}}
            row["Delta_M_FR_P_AP"] = fnum(e, "M_FR_P_AP") - fnum(base, "M_FR_P_AP")
            rows.append(row)
    for g in ("small", "large", "thin", "thick"):
        for variant in VARIANTS[1:]:
            vals = [fnum(r, "Delta_M_FR_P_AP") for r in rows if r["variant"] == variant and r.get(f"group_{g}") is True]
            rows.append({"source": "SUMMARY", "dataset": "PILOT", "category": "ALL", "variant": variant, "group": g, "mean_delta_M_FR_P_AP": finite_mean(vals), "median_delta_M_FR_P_AP": finite_median(vals), "improved": sum(1 for v in vals if np.isfinite(v) and v > 0), "total": sum(1 for v in vals if np.isfinite(v))})
    return rows


def attention_collapse(attn_rows, checksum_ok):
    e2 = [r for r in attn_rows if r["variant"] == "E2_ExperienceConditionedAttention"]
    top1 = np.asarray([fnum(r, "attention_top1_probability") for r in e2], dtype=np.float64)
    kl = np.asarray([fnum(r, "kl_to_frozen") for r in e2], dtype=np.float64)
    return (top1.size and np.nanmean(top1) >= 0.95) or (kl.size and np.nanmean(kl) > 2.0) or (not checksum_ok)


def decide(summary, morph_rows, attn_rows, checksum_ok):
    s = {r["variant"]: r for r in summary if r["source"] == "SUMMARY"}
    e1 = s.get("E1_FrozenAttentionReadout", {})
    e2 = s.get("E2_ExperienceConditionedAttention", {})
    e2_mean = fnum(e2, "mean_delta_M_FR_P_AP")
    e2_imp = int(float(e2.get("improved", 0) or 0))
    mv = [r for r in summary if r.get("source") == "Full MVTec" and r.get("dataset") == "VisA" and r.get("variant") == "E2_ExperienceConditionedAttention"]
    mp = [r for r in summary if r.get("source") == "Full VisA" and r.get("dataset") == "MPDD" and r.get("variant") == "E2_ExperienceConditionedAttention"]
    focus_gain = max([fnum(r, "E_minus_E0_M_FR_P_AP") for r in mv + mp], default=float("nan"))
    small = [r for r in morph_rows if r.get("source") == "SUMMARY" and r.get("variant") == "E2_ExperienceConditionedAttention" and r.get("group") == "small"]
    thin = [r for r in morph_rows if r.get("source") == "SUMMARY" and r.get("variant") == "E2_ExperienceConditionedAttention" and r.get("group") == "thin"]
    small_gain = fnum(small[0], "mean_delta_M_FR_P_AP") if small else float("nan")
    thin_gain = fnum(thin[0], "mean_delta_M_FR_P_AP") if thin else float("nan")
    collapsed = attention_collapse(attn_rows, checksum_ok)
    stats = {
        "E1_mean_delta_P_AP": fnum(e1, "mean_delta_M_FR_P_AP"),
        "E1_improved": int(float(e1.get("improved", 0) or 0)),
        "E2_mean_delta_P_AP": e2_mean,
        "E2_improved": e2_imp,
        "E2_focus_best_delta_P_AP": focus_gain,
        "E2_small_mean_delta_P_AP": small_gain,
        "E2_thin_mean_delta_P_AP": thin_gain,
        "attention_collapse": bool(collapsed),
        "frozen_checksum_ok": bool(checksum_ok),
    }
    if e2_mean >= 1.0 and e2_imp >= 3 and focus_gain > 0.5 and small_gain > 0 and thin_gain > 0 and not collapsed:
        return "EXPERIENCE_ATTENTION_PRIOR_SIGNIFICANTLY_IMPROVES_TRANSFER", stats
    if e2_mean > 0 and e2_imp >= 2 and not collapsed:
        return "EXPERIENCE_RELATION_LEARNING_HAS_WEAK_VALUE", stats
    if fnum(e1, "mean_delta_M_FR_P_AP") > 0.5 and fnum(e2, "mean_delta_M_FR_P_AP") <= fnum(e1, "mean_delta_M_FR_P_AP"):
        return "FROZEN_ATTENTION_IS_USEFUL_BUT_LEARNED_BIAS_IS_NOT", stats
    return "ATTENTION_RELATION_PRIOR_HAS_NO_COMPETITIVE_VALUE", stats


def write_report(raw, summary, morph, attn_rows, outcome, stats, checksum_before, checksum_after, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Phase ECAP: Experience-Conditioned Attention Prior",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        f"Fixed layer: {ATTN_LAYER}. Bias net: Linear(6,16)-GELU-Linear(16,1), b_max={B_MAX}, KL weight={TRUST_WEIGHT}, fusion weight={REL_FUSION_WEIGHT}. CLIP/Two-State/NTR/Fixed LRP are unchanged.",
        "",
        f"Frozen CLIP checksum unchanged: `{checksum_before == checksum_after}`.",
        "",
        "## Pilot Metrics",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for source, target in PILOT_PAIRS:
        for variant in VARIANTS:
            r = by.get((source, target, variant))
            if r:
                lines.append(f"| {source} | {target} | {variant} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    lines.extend(["", "## P-AP Delta", "", "| Source | Target | Variant | Delta P-AP |", "|---|---|---|---:|"])
    for r in summary:
        if r["source"] != "SUMMARY":
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'E_minus_E0_M_FR_P_AP'):+.4f} |")
    lines.extend(["", "## Summary", "", "| Variant | mean Delta P-AP | median | improved | non-drop |", "|---|---:|---:|---:|---:|"])
    for r in summary:
        if r["source"] == "SUMMARY":
            lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {r.get('improved')}/{r.get('total')} | {r.get('non_drop_ge_minus_0_2')}/{r.get('total')} |")
    lines.extend(["", "## Morphology", "", "| Group | Variant | mean Delta P-AP | improved |", "|---|---|---:|---:|"])
    for r in morph:
        if r["source"] == "SUMMARY":
            lines.append(f"| {r['group']} | {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {r.get('improved')}/{r.get('total')} |")
    e2 = [r for r in attn_rows if r["variant"] == "E2_ExperienceConditionedAttention"]
    lines.extend([
        "",
        "## Attention Diagnostics",
        "",
        f"- E2 top1 attention probability mean: {finite_mean([fnum(r,'attention_top1_probability') for r in e2]):.4f}",
        f"- E2 attention entropy mean: {finite_mean([fnum(r,'attention_entropy') for r in e2]):.4f}",
        f"- E2 KL(Aexp||A0) mean: {finite_mean([fnum(r,'kl_to_frozen') for r in e2]):.4f}",
        f"- E2 bias |B| mean: {finite_mean([fnum(r,'bias_abs_mean') for r in e2]):.4f}",
        "",
        f"Decision stats: `{json.dumps(stats, default=json_default)}`",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ECAP] device={device}", flush=True)
    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    checksum_before = checksum_module(clip_model.visual)
    models = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    for info in models.values():
        if info.get("align_model") is not None:
            for p in info["align_model"].parameters():
                p.requires_grad_(False)
            info["align_model"].eval()

    bias_by_source, cal_by_source = {}, {}
    train_rows, attn_rows = [], []
    for source, info in models.items():
        print(f"[ECAP] train source relation prior {source}", flush=True)
        net, cal, logs, audit, reused = train_bias(source, info, clip_model, device, args.smoke)
        bias_by_source[source] = net
        cal_by_source[source] = cal
        train_rows.extend(logs)
        attn_rows.extend({**r, "checkpoint_reused": bool(reused)} for r in audit)

    raw, cats, meta_rows_all = [], [], []
    for source, target in PILOT_PAIRS:
        dataset, refs, kind = load_pair(source, target, device, args.smoke)
        print(f"[ECAP] evaluate {source}->{target}", flush=True)
        refs_use, maps, metas, meta_rows, ar = build_maps(source, target, models[source], clip_model, preprocess, bias_by_source[source], cal_by_source[source], dataset, refs, kind, device, args.smoke)
        rr, cc = evaluate_maps(source, target, refs_use, dataset, kind, maps, metas, variants=VARIANTS)
        raw.extend(rr)
        cats.extend(cc)
        meta_rows_all.extend(meta_rows)
        attn_rows.extend(ar)
        if not args.smoke:
            write_csv(OUT / "RAW_METRICS.partial.csv", raw)
            write_csv(OUT / "ATTENTION_DIAGNOSTICS.partial.csv", attn_rows)
    checksum_after = checksum_module(clip_model.visual)
    checksum_ok = checksum_before == checksum_after
    summary = summarize_metrics(raw)
    morph = morphology_diagnostics(cats, meta_rows_all)
    outcome, stats = decide(summary, morph, attn_rows, checksum_ok)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", summary)
    write_csv(OUT / "MORPHOLOGY_DIAGNOSTICS.csv", morph)
    write_csv(OUT / "ATTENTION_DIAGNOSTICS.csv", attn_rows)
    write_csv(OUT / "TRAINING_LOG.csv", train_rows)
    write_csv(OUT / "MORPHOLOGY_GROUPS_USED.csv", meta_rows_all)
    write_json(OUT / "CHECKSUM.json", {"clip_visual_before": checksum_before, "clip_visual_after": checksum_after, "unchanged": checksum_ok})
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": stats, "runtime_sec": runtime, "device": device, "no_target_tuning": True, "clip_frozen": checksum_ok, "attention_layer": ATTN_LAYER, "b_max": B_MAX, "fusion_weight": REL_FUSION_WEIGHT, "epochs": EPOCHS, "max_source_images": SMOKE_SOURCE_IMAGES if args.smoke else MAX_SOURCE_IMAGES, "variants": VARIANTS})
    write_report(raw, summary, morph, attn_rows, outcome, stats, checksum_before, checksum_after, runtime)
    print(f"[ECAP] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
