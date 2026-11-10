import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from shared_virtual_geometry.factorized_similarity import coefficients_from_geometry, reconstruct_keys
from shared_virtual_geometry.geometry_parameterization import SharedGeometry


def geometry_separation_loss(geoms):
    losses = []
    for g in geoms:
        d = torch.cdist(g, g)
        vals = d[d > 0]
        if vals.numel():
            losses.append(-vals.mean())
    return torch.stack(losses).mean() if losses else geoms.sum() * 0.0


def retrieval_logits_with_self_mask(query, keys, values, query_indices=None, temperature=1.0):
    sim = query @ keys.T
    if query_indices is not None:
        query_indices = query_indices.to(query.device)
        valid = query_indices < keys.shape[0]
        if bool(valid.any()):
            rows = torch.arange(query.shape[0], device=query.device)[valid]
            sim[rows, query_indices[valid]] = -1e4
    weights = F.softmax(sim / temperature, dim=-1)
    return weights @ values.float()


def train_shared_geometry(memory, backend, cfg, device, label_specific=True, seed=0, init="square", desc="train geometry"):
    module = SharedGeometry(label_specific=label_specific, seed=seed, init=init).to(device)
    opt = torch.optim.AdamW(
        module.parameters(),
        lr=float(cfg.get("geometry_lr", 0.03)),
        weight_decay=float(cfg.get("geometry_weight_decay", 0.0)),
    )
    queries = F.normalize(memory["patch_keys"].float(), dim=-1).to(device)
    targets = memory["patch_values"].argmax(dim=-1).long().to(device)
    values = memory["patch_values"].float().to(device)
    labels = torch.as_tensor(backend["labels"], device=device).long()
    scales = backend["scales"].float().to(device)
    mu_hat = backend["mu_hat"].float().to(device)
    basis = backend["basis"].float().to(device)
    batch_size = int(cfg.get("geometry_batch_size", 256))
    epochs = int(cfg.get("epochs", 12))
    lambda_sep = float(cfg.get("lambda_geometry_separation", 0.01))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    curves = []
    best_loss = math.inf
    best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    patience = int(cfg.get("patience", 4))
    bad = 0
    for epoch in range(epochs):
        perm = torch.randperm(queries.shape[0], device=device)
        losses = []
        for start in tqdm(range(0, queries.shape[0], batch_size), desc=f"{desc} epoch {epoch + 1}/{epochs}", leave=False):
            idx = perm[start : start + batch_size]
            geoms = module.geometries()
            coeff = coefficients_from_geometry(scales, labels, geoms)
            keys = reconstruct_keys(mu_hat, basis, coeff, normalize=True)
            logits = retrieval_logits_with_self_mask(queries[idx], keys, values, query_indices=idx)
            loss = F.cross_entropy(logits.clamp_min(1e-8).log(), targets[idx])
            loss = loss + lambda_sep * geometry_separation_loss(geoms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(sum(losses) / max(1, len(losses)))
        curves.append({"epoch": epoch + 1, "loss": mean_loss})
        if mean_loss < best_loss - 1e-6:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    module.load_state_dict(best_state)
    return curves, module
