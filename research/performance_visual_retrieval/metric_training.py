import torch
import torch.nn.functional as F
from tqdm import tqdm


def retrieval_logits(query, keys, values, temperature=1.0, self_indices=None):
    sim = query @ keys.T
    if self_indices is not None:
        idx = self_indices.to(sim.device)
        valid = idx < keys.shape[0]
        if bool(valid.any()):
            rows = torch.arange(idx.numel(), device=sim.device)[valid]
            sim[rows, idx[valid]] = -1e4
    w = F.softmax(sim / float(temperature), dim=-1)
    return w @ values.float(), sim, w


def train_metric_adapter(adapter, memory, cfg, device, desc="train metric"):
    adapter = adapter.to(device)
    opt = torch.optim.AdamW(
        [p for p in adapter.parameters() if p.requires_grad],
        lr=float(cfg.get("metric_lr", 5e-4)),
        weight_decay=float(cfg.get("metric_weight_decay", 1e-4)),
    )
    q0 = memory["patch_keys"].float().to(device)
    values = memory["patch_values"].float().to(device)
    labels = values.argmax(dim=-1).long()
    batch = int(cfg.get("metric_batch_size", 256))
    epochs = int(cfg.get("epochs", 10))
    patience = int(cfg.get("patience", 3))
    best_loss = float("inf")
    best = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    curves, bad = [], 0
    for epoch in range(epochs):
        perm = torch.randperm(q0.shape[0], device=device)
        losses = []
        for start in tqdm(range(0, q0.shape[0], batch), desc=f"{desc} epoch {epoch + 1}/{epochs}", leave=False):
            idx = perm[start : start + batch]
            keys = adapter.normalized(q0)
            query = adapter.normalized(q0[idx])
            logits, _, _ = retrieval_logits(query, keys, values, self_indices=idx, temperature=cfg.get("retrieval_temperature", 1.0))
            loss = F.cross_entropy(logits.clamp_min(1e-8).log(), labels[idx])
            loss = loss + adapter.regularization(cfg.get("lambda_identity", 1e-3))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], float(cfg.get("grad_clip", 1.0)))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / max(1, len(losses))
        curves.append({"epoch": epoch + 1, "loss": mean_loss})
        if mean_loss < best_loss - 1e-6:
            best_loss = mean_loss
            best = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    adapter.load_state_dict(best)
    return curves, adapter

