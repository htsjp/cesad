from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer


_TOKENIZER = SimpleTokenizer()


def tokenize(text, context_length=77):
    if isinstance(text, str):
        text = [text]
    sot = _TOKENIZER.encoder["<|startoftext|>"]
    eot = _TOKENIZER.encoder["<|endoftext|>"]
    out = torch.zeros(len(text), context_length, dtype=torch.long)
    for i, t in enumerate(text):
        toks = [sot] + _TOKENIZER.encode(t) + [eot]
        if len(toks) > context_length:
            toks = toks[:context_length]
            toks[-1] = eot
        out[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
    return out


class ObjectPromptLearner(nn.Module):
    """Minimal AnomalyCLIP-style object prompt with optional abnormal ctx bias."""

    def __init__(self, clip_model, n_ctx=4):
        super().__init__()
        self.n_ctx = int(n_ctx)
        dtype = clip_model.transformer.get_cast_dtype()
        width = int(clip_model.ln_final.weight.shape[0])
        self.text_width = width
        self.ctx_normal = nn.Parameter(torch.empty(self.n_ctx, width, dtype=dtype))
        self.ctx_abnormal = nn.Parameter(torch.empty(self.n_ctx, width, dtype=dtype))
        nn.init.normal_(self.ctx_normal, std=0.02)
        nn.init.normal_(self.ctx_abnormal, std=0.02)
        prefix = " ".join(["X"] * self.n_ctx)
        normal_text = f"{prefix} object."
        abnormal_text = f"{prefix} damaged object."
        tokenized = tokenize([normal_text, abnormal_text])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized.to(next(clip_model.parameters()).device)).type(dtype).cpu()
        self.register_buffer("tokenized_prompts", tokenized)
        self.register_buffer("prefix", embedding[:, :1, :])
        self.register_buffer("suffix", embedding[:, 1 + self.n_ctx :, :])

    def prompts(self, bias: Optional[torch.Tensor] = None, symmetric=False):
        if bias is None:
            bias = torch.zeros(self.text_width, device=self.ctx_normal.device, dtype=self.ctx_normal.dtype)
        bias = bias.to(device=self.ctx_normal.device, dtype=self.ctx_normal.dtype)
        if symmetric:
            ctx_n = self.ctx_normal - 0.5 * bias[None, :]
            ctx_a = self.ctx_abnormal + 0.5 * bias[None, :]
        else:
            ctx_n = self.ctx_normal
            ctx_a = self.ctx_abnormal + bias[None, :]
        ctx = torch.stack([ctx_n, ctx_a], dim=0)
        return torch.cat([self.prefix.to(ctx.device), ctx, self.suffix.to(ctx.device)], dim=1)

    def encode(self, clip_model, bias: Optional[torch.Tensor] = None, symmetric=False):
        prompts = self.prompts(bias=bias, symmetric=symmetric)
        tokenized = self.tokenized_prompts.to(prompts.device)
        first_block = getattr(getattr(clip_model, "transformer", None), "resblocks", [None])[0]
        if hasattr(first_block, "compound_prompt_nctx"):
            text = clip_model.encode_text_learn(prompts.unsqueeze(0), tokenized, []).float().squeeze(0)
        else:
            text = clip_model.encode_text_learn(prompts, tokenized, None).float()
        return F.normalize(text, dim=-1)

    def static_direction(self, clip_model):
        text = self.encode(clip_model, None)
        return F.normalize(text[1] - text[0], dim=-1), text


def freeze_clip(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def trainable_parameter_count(module):
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
