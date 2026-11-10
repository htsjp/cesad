import numpy as np
import torch
from scipy import sparse


def scipy_csr_to_torch(mat: sparse.csr_matrix, device: str):
    coo = mat.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def propagate(weights: torch.Tensor, transition_t: torch.Tensor) -> torch.Tensor:
    # weights are row distributions. torch sparse mm is left-multiply, so use P^T.
    original_shape = weights.shape
    flat = weights.reshape(-1, original_shape[-1])
    out = torch.sparse.mm(transition_t, flat.T.contiguous()).T.contiguous()
    return out.reshape(original_shape)


def residual_fusion(w0: torch.Tensor, w1: torch.Tensor, lambda_graph: float) -> torch.Tensor:
    lam = float(lambda_graph)
    return (1.0 - lam) * w0 + lam * w1


def two_step(weights: torch.Tensor, transition_t: torch.Tensor) -> torch.Tensor:
    return propagate(propagate(weights, transition_t), transition_t)


def fully_mixed(weights: torch.Tensor) -> torch.Tensor:
    return torch.full_like(weights, 1.0 / max(1, weights.shape[-1]))
