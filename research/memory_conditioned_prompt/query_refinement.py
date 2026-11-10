import torch.nn.functional as F


def refine_query(q0, direction, patch_gate, lambda_text):
    return F.normalize(q0.float() + float(lambda_text) * patch_gate[:, None].float() * direction.float()[None, :], dim=-1)

