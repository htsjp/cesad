import torch

from models.mlp import average_neighbor
from phase_x.gradient_visual_forward import forward_visual_with_module


def visual_outputs(model, image, adapter=None, grad=True, snapshot_mode=None):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return forward_visual_with_module(
            model.visual,
            image.type(model.dtype),
            insertion_layer=18,
            inserted_module=adapter,
            return_layers=[6, 12, 18, 24],
            dpam_layer=24,
            ori_patch=False,
            proj_use=True,
            ffn=False,
            snapshot_mode=snapshot_mode,
        )


def normalized_image_feature(outputs):
    image_feature = outputs[0]
    return image_feature / image_feature.norm(dim=-1, keepdim=True)


def normalized_patch_feature(outputs):
    patch_feature = average_neighbor(outputs[1][3])
    return patch_feature / patch_feature.norm(dim=-1, keepdim=True)


def normalized_patch_projection(outputs):
    patch_projection = average_neighbor(outputs[3][3])
    return patch_projection / patch_projection.norm(dim=-1, keepdim=True)


def image_memory_scores(image_feature, image_keys, image_values, image_proj):
    query, keys = image_proj(image_feature, image_keys)
    logits = query @ keys.t()
    posterior = torch.softmax(logits, dim=-1) @ image_values.float()
    return posterior, logits, query, keys


def patch_memory_scores(patch_feature, patch_keys, patch_values, patch_proj):
    query = patch_proj(patch_feature, 0)
    keys = patch_proj(patch_keys, 1)
    logits = query @ keys.t()
    posterior = torch.softmax(logits, dim=-1) @ patch_values.float()
    return posterior, logits, query, keys
