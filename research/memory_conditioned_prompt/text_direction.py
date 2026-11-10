import torch.nn.functional as F


def dynamic_direction(text_features):
    return F.normalize(text_features[1] - text_features[0], dim=-1)

