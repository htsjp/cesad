import math
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F

from models.mlp import average_neighbor

from .hooks import dpam_value_path, multihead_value_path


def _prepare_visual_tokens(model, images: torch.Tensor, device: str) -> torch.Tensor:
    visual = model.visual
    x = visual.conv1(images.to(device).type(model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    side = int((visual.positional_embedding.shape[0] - 1) ** 0.5)
    new_side = int((x.shape[1] - 1) ** 0.5)
    if side != new_side:
        new_pos = visual.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
        new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode="bilinear")
        new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
        visual.positional_embedding.data = torch.cat([visual.positional_embedding[:1, :], new_pos[0]], 0)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    return x.permute(1, 0, 2)


def _project_tokens(model, tokens_bld: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    visual = model.visual
    projected = visual.ln_post(tokens_bld) @ visual.proj
    projected = projected.float()
    if normalize:
        projected = F.normalize(projected, dim=-1)
    return projected


def _finite_normalized(x: torch.Tensor) -> bool:
    if not torch.isfinite(x).all():
        return False
    norms = x.norm(dim=-1)
    return bool(torch.allclose(norms, torch.ones_like(norms), atol=2e-3, rtol=2e-3))


@torch.no_grad()
def extract_value_representations(
    model,
    images: torch.Tensor,
    layers: Sequence[int],
    device: str,
    smooth: bool = True,
    include_cls: bool = True,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Mirror MRAD's visual forward while collecting T, V, AV, and PAV.

    All returned patch features are projected to CLIP output dimension so retrieval
    uses the same dimensional interface as existing FPS_K4 MRAD audits.
    """
    wanted = set(int(x) for x in layers)
    x = _prepare_visual_tokens(model, images, device)
    outputs: Dict[int, Dict[str, torch.Tensor]] = {}
    current = x

    for idx, block in enumerate(model.visual.transformer.resblocks, start=1):
        if isinstance(current, list):
            value_source = current[1]
            active_input = current[0]
        else:
            value_source = current
            active_input = current
        ln_value_source = block.ln_1(value_source)
        attn = block.attn
        if hasattr(attn, "qkv"):
            value_data = dpam_value_path(attn, ln_value_source)
        else:
            value_data = multihead_value_path(attn, ln_value_source)

        current = block(current, ffn=False)
        block_out = current[0] if isinstance(current, list) else current

        if idx in wanted:
            layer_data = {
                "T": _project_tokens(model, block_out.permute(1, 0, 2), normalize=False),
                "V": _project_tokens(model, value_data["raw_v"], normalize=False),
                "AV": _project_tokens(model, value_data["av"], normalize=False),
                "PAV": _project_tokens(model, value_data["pav"], normalize=False),
            }
            if smooth:
                for key in list(layer_data):
                    cls = layer_data[key][:, :1, :]
                    patches = average_neighbor(layer_data[key][:, 1:, :])
                    layer_data[key] = torch.cat([cls, patches], dim=1)
            for key in list(layer_data):
                layer_data[key] = F.normalize(layer_data[key], dim=-1)
            if include_cls:
                layer_data["CLS"] = F.normalize((block_out[0, :, :] @ model.visual.proj).float(), dim=-1)
            outputs[idx] = layer_data

    missing = wanted.difference(outputs)
    if missing:
        raise RuntimeError(f"Requested layers were not reached: {sorted(missing)}")
    return outputs


def patch_features_from_outputs(outputs: Dict[int, Dict[str, torch.Tensor]], layer: int, rep: str) -> torch.Tensor:
    return outputs[int(layer)][rep][:, 1:, :]


def cls_from_outputs(outputs: Dict[int, Dict[str, torch.Tensor]], layer: int = 24) -> torch.Tensor:
    return outputs[int(layer)]["CLS"]


@torch.no_grad()
def extract_head_value_representations(
    model,
    images: torch.Tensor,
    layers: Sequence[int],
    device: str,
    smooth: bool = True,
    include_cls: bool = True,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Collect raw value heads before attention aggregation.

    `V_heads_hidden` has shape [B, N+1, H, Dh] in the model hidden width.
    Weighted heads should be concatenated first and then passed through the same
    ln_post/proj path as the existing raw V representation.
    """
    wanted = set(int(x) for x in layers)
    x = _prepare_visual_tokens(model, images, device)
    outputs: Dict[int, Dict[str, torch.Tensor]] = {}
    current = x

    for idx, block in enumerate(model.visual.transformer.resblocks, start=1):
        value_source = current[1] if isinstance(current, list) else current
        value_data = None
        if idx in wanted:
            ln_value_source = block.ln_1(value_source)
            attn = block.attn
            value_data = dpam_value_path(attn, ln_value_source) if hasattr(attn, "qkv") else multihead_value_path(attn, ln_value_source)
        current = block(current, ffn=False)
        block_out = current[0] if isinstance(current, list) else current

        if idx in wanted:
            raw_hidden = value_data["raw_v"]
            raw_heads = value_data["raw_v_heads"].permute(0, 2, 1, 3).contiguous()
            layer_data = {
                "T": _project_tokens(model, block_out.permute(1, 0, 2), normalize=False),
                "V": _project_tokens(model, raw_hidden, normalize=False),
                "V_heads_hidden": raw_heads.float(),
            }
            if smooth:
                cls = layer_data["T"][:, :1, :]
                layer_data["T"] = torch.cat([cls, average_neighbor(layer_data["T"][:, 1:, :])], dim=1)
                cls_v = layer_data["V"][:, :1, :]
                layer_data["V"] = torch.cat([cls_v, average_neighbor(layer_data["V"][:, 1:, :])], dim=1)
            layer_data["T"] = F.normalize(layer_data["T"], dim=-1)
            layer_data["V"] = F.normalize(layer_data["V"], dim=-1)
            if include_cls:
                layer_data["CLS"] = F.normalize((block_out[0, :, :] @ model.visual.proj).float(), dim=-1)
            outputs[idx] = layer_data

    missing = wanted.difference(outputs)
    if missing:
        raise RuntimeError(f"Requested layers were not reached: {sorted(missing)}")
    return outputs


def project_hidden_value_heads(model, heads_bnhd: torch.Tensor, weights_bnh: torch.Tensor = None, smooth: bool = True, has_cls: bool = True) -> torch.Tensor:
    """Scale hidden value heads, concatenate, and project to CLIP output width."""
    x = heads_bnhd.float()
    if weights_bnh is not None:
        x = x * weights_bnh.unsqueeze(-1).float()
    batch, tokens, heads, head_dim = x.shape
    concat = x.reshape(batch, tokens, heads * head_dim)
    projected = _project_tokens(model, concat, normalize=False)
    if smooth and has_cls:
        cls = projected[:, :1, :]
        projected = torch.cat([cls, average_neighbor(projected[:, 1:, :])], dim=1)
    elif smooth:
        projected = average_neighbor(projected)
    return F.normalize(projected, dim=-1)
