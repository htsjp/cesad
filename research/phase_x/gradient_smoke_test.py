import hashlib

import torch


def parameter_hashes(module):
    hashes = {}
    for name, param in module.named_parameters():
        data = param.detach().cpu().contiguous().numpy().tobytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def hashes_equal(before, after):
    return before == after
