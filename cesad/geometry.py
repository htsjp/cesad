"""Load saved CES geometry without substituting an untrained fallback."""

import numpy as np
import torch
from .heads import CESHead


def model_state(checkpoint):
    if checkpoint.get("format") == "cesad_head_v1":
        return checkpoint["model"]
    if "raw_model" in checkpoint:
        return checkpoint["raw_model"]
    params = checkpoint.get("params", {})
    if "raw_model" in params:
        return params["raw_model"]
    if "initial_geometry" in checkpoint and "v" in params:
        g = checkpoint["initial_geometry"]
        return dict(
            m=torch.as_tensor(g["m"]),
            alpha=torch.as_tensor(g["alpha"]),
            v=torch.as_tensor(params["v"]),
        )
    raise ValueError(
        "Unsupported checkpoint schema; no initialization fallback is allowed"
    )


def load_head(path, device="cpu"):
    # Only load trusted locally obtained research checkpoints.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = model_state(saved)
    m = np.asarray(state["m"], np.float32)
    v = np.asarray(state["v"], np.float32)
    head = CESHead(m, float(state["alpha"]), v)
    head.load_state_dict(
        {k: torch.as_tensor(value, dtype=torch.float32) for k, value in state.items()},
        strict=True,
    )
    return head.to(device), saved
