import torch
import torch.nn.functional as F

from memory_conditioned_prompt.base_prompt import ObjectPromptLearner, tokenize


NORMAL_STATES = ["{}", "flawless {}", "{} without defect", "{} without damage"]
ABNORMAL_STATES = ["damaged {}", "broken {}", "{} with defect", "{} with damage"]
PROMPT_TEMPLATES = ["a photo of a {}.", "a good photo of the {}.", "there is a {} in the scene."]
NEUTRAL_STATES = ["flawless {}", "{} without defect"]


@torch.no_grad()
def encode_prompts(clip_model, prompts, device):
    tokenized = tokenize(prompts).to(device)
    first_block = getattr(getattr(clip_model, "transformer", None), "resblocks", [None])[0]
    if hasattr(first_block, "compound_prompt_nctx"):
        dtype = clip_model.transformer.get_cast_dtype()
        prompt_embeddings = clip_model.token_embedding(tokenized).type(dtype)
        text = clip_model.encode_text_learn(prompt_embeddings.unsqueeze(0), tokenized, []).float().squeeze(0)
    else:
        text = clip_model.encode_text(tokenized).float()
    return F.normalize(text, dim=-1)


@torch.no_grad()
def static_anchors(clip_model, device):
    normal = [t.format(s.format("object")) for s in NORMAL_STATES for t in PROMPT_TEMPLATES]
    abnormal = [t.format(s.format("object")) for s in ABNORMAL_STATES for t in PROMPT_TEMPLATES]
    tn = F.normalize(encode_prompts(clip_model, normal, device).mean(dim=0), dim=0)
    ta = F.normalize(encode_prompts(clip_model, abnormal, device).mean(dim=0), dim=0)
    d = F.normalize(ta - tn, dim=0)
    return {"normal": tn, "abnormal": ta, "direction": d, "normal_prompts": normal, "abnormal_prompts": abnormal}


@torch.no_grad()
def neutral_direction(clip_model, device):
    prompts = [PROMPT_TEMPLATES[0].format(s.format("object")) for s in NEUTRAL_STATES]
    text = encode_prompts(clip_model, prompts, device)
    return F.normalize(text[1] - text[0], dim=0)


def learned_prompt_module(clip_model, n_ctx):
    return ObjectPromptLearner(clip_model, n_ctx=n_ctx)
