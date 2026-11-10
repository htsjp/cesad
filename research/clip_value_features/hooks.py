import math
from typing import Dict, Iterable

import torch
import torch.nn.functional as F


def split_qkv_from_multihead(attn, x_lbd: torch.Tensor):
    qkv = F.linear(x_lbd, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    return q, k, v


def heads_from_lbd(x_lbd: torch.Tensor, num_heads: int) -> torch.Tensor:
    length, batch, channels = x_lbd.shape
    head_dim = channels // num_heads
    return x_lbd.permute(1, 0, 2).reshape(batch, length, num_heads, head_dim).permute(0, 2, 1, 3)


def concat_heads(x_bhld: torch.Tensor) -> torch.Tensor:
    batch, heads, length, head_dim = x_bhld.shape
    return x_bhld.permute(0, 2, 1, 3).reshape(batch, length, heads * head_dim)


def attention_from_qk(q_bhld: torch.Tensor, k_bhld: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.softmax((q_bhld @ k_bhld.transpose(-2, -1)) * scale, dim=-1)


def multihead_value_path(attn, x_lbd: torch.Tensor) -> Dict[str, torch.Tensor]:
    q, k, v = split_qkv_from_multihead(attn, x_lbd)
    heads = int(attn.num_heads)
    head_dim = q.shape[-1] // heads
    qh = heads_from_lbd(q, heads)
    kh = heads_from_lbd(k, heads)
    vh = heads_from_lbd(v, heads)
    weights = attention_from_qk(qh, kh, 1.0 / math.sqrt(head_dim))
    av = concat_heads(weights @ vh)
    pav = attn.out_proj(av)
    return {
        "raw_v": concat_heads(vh),
        "raw_v_heads": vh,
        "av": av,
        "pav": pav,
        "weights": weights.detach(),
    }


def dpam_value_path(attn, x_lbd: torch.Tensor) -> Dict[str, torch.Tensor]:
    x_bld = x_lbd.transpose(0, 1)
    batch, length, channels = x_bld.shape
    heads = int(attn.num_heads)
    head_dim = channels // heads
    qkv = attn.qkv(x_bld).reshape(batch, length, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    # MRAD's replaced branch sets q=k=v for the active value-value path.
    weights = attention_from_qk(v, v, float(attn.scale))
    av = concat_heads(weights @ v)
    pav = attn.proj_drop(attn.proj(av))
    return {
        "raw_v": concat_heads(v),
        "raw_v_heads": v,
        "av": av,
        "pav": pav,
        "weights": weights.detach(),
    }


def remove_handles(handles: Iterable):
    for handle in handles:
        handle.remove()
