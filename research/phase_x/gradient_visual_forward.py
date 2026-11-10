from contextlib import contextmanager
import types

import torch


def _safe_resblock_forward(self, x, whole=False, ffn=False):
    if isinstance(self.attn, torch.nn.Module) and self.attn.__class__.__name__ == "Attention":
        if isinstance(x, list):
            if not ffn:
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res
                x_ori = x_ori + x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x = x + x_res
                return [x, x_ori]
            x, x_ori_1 = x
            x_res = self.attention(self.ln_1(x_ori_1))
            x_res, x_ori_res = x_res
            x_ori = x_ori_1 + x_ori_res
            x_ori = x_ori + self.mlp(self.ln_2(x_ori))
            x = x + x_res
            x = x_res + x_ori_1
            x = x + self.mlp(self.ln_2(x))
            return [x, x_ori]
        x_res = self.attention(self.ln_1(x))
        if isinstance(x_res, list):
            x_res, x_ori_res = x_res
            x_ori = x + x_ori_res
            x_ori = x_ori + self.mlp(self.ln_2(x_ori))
            x = x + x_res
            return [x, x_ori]
    else:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
    return x


@contextmanager
def _safe_resblock_forwards(visual_module):
    originals = []
    for block in visual_module.transformer.resblocks:
        originals.append((block, block.forward))
        block.forward = types.MethodType(_safe_resblock_forward, block)
    try:
        yield
    finally:
        for block, forward in originals:
            block.forward = forward


def _apply_inserted_module(output, inserted_module):
    if inserted_module is None:
        return output
    if isinstance(output, list):
        if getattr(inserted_module, "accepts_list_output", False):
            return inserted_module(output)
        return [_apply_inserted_module(item, inserted_module) for item in output]
    # Official transformer tokens are LND. The public insertion API is NLD.
    adapted = inserted_module(output.permute(1, 0, 2))
    return adapted.permute(1, 0, 2)


@contextmanager
def _block_insertion_hook(visual_module, insertion_layer, inserted_module):
    handle = None
    if inserted_module is not None:
        block = visual_module.transformer.resblocks[insertion_layer - 1]

        def hook(_module, _inputs, output):
            return _apply_inserted_module(output, inserted_module)

        handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def forward_visual_with_module(
    visual_module,
    image,
    insertion_layer=18,
    inserted_module=None,
    return_layers=None,
    dpam_layer=24,
    ori_patch=False,
    proj_use=True,
    ffn=False,
    snapshot_mode=None,
):
    if return_layers is None:
        return_layers = [6, 12, 18, 24]
    wrapped = getattr(type(visual_module).forward, "__wrapped__", None)
    if wrapped is None:
        raise RuntimeError("VisionTransformer.forward.__wrapped__ is unavailable")
    for param in visual_module.parameters():
        param.requires_grad_(False)
    snapshots = {}
    handles = []

    def make_snapshot_hook(layer_idx):
        def hook(_module, _inputs, output):
            if isinstance(output, list):
                snapshots[f"true_block{layer_idx}"] = [item.detach().clone() for item in output]
            else:
                snapshots[f"true_block{layer_idx}"] = output.detach().clone()
        return hook

    if snapshot_mode == "true_clone":
        for layer_idx in return_layers:
            handles.append(visual_module.transformer.resblocks[layer_idx - 1].register_forward_hook(make_snapshot_hook(layer_idx)))

    with _safe_resblock_forwards(visual_module):
        with _block_insertion_hook(visual_module, insertion_layer, inserted_module):
            with torch.enable_grad():
                result = wrapped(
                    visual_module,
                    image,
                    return_layers,
                    ori_patch=ori_patch,
                    proj_use=proj_use,
                    DPAM_layer=dpam_layer,
                    ffn=ffn,
                )
    for handle in handles:
        handle.remove()
    if snapshot_mode == "true_clone":
        return {"outputs": result, "snapshots": snapshots}
    return result


def freeze_official_modules(*modules):
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(False)
