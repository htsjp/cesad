import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def balanced_indices(labels, seed):
    labels = labels.long().cpu()
    gen = torch.Generator().manual_seed(int(seed))
    idx0 = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)
    idx1 = torch.nonzero(labels == 1, as_tuple=False).squeeze(1)
    n = int(min(idx0.numel(), idx1.numel()))
    if n == 0:
        return torch.arange(labels.numel())
    idx = torch.cat([idx0[torch.randperm(idx0.numel(), generator=gen)[:n]], idx1[torch.randperm(idx1.numel(), generator=gen)[:n]]])
    return idx[torch.randperm(idx.numel(), generator=gen)]


def train_source_prompt(clip_model, prompt, prototypes, labels, static_anchor, cfg, device, seed):
    epochs = int(cfg.get("prompt_epochs", 30))
    patience = int(cfg.get("prompt_patience", 5))
    lr = float(cfg.get("prompt_lr", 1e-3))
    wd = float(cfg.get("prompt_weight_decay", 1e-4))
    tau = float(cfg.get("tau_text", 10.0))
    lam = float(cfg.get("lambda_anchor", 1e-3))
    batch = int(cfg.get("prompt_batch_size", 256))
    opt = torch.optim.AdamW(prompt.parameters(), lr=lr, weight_decay=wd)
    x = F.normalize(prototypes.float().to(device), dim=-1)
    y = labels.long().to(device)
    idx_all = balanced_indices(y.detach().cpu(), seed).to(device)
    best, wait, curves = float("inf"), 0, []
    for epoch in range(epochs):
        losses = []
        prompt.train()
        perm = idx_all[torch.randperm(idx_all.numel(), device=device)]
        for start in tqdm(range(0, perm.numel(), batch), desc=f"train text anchor epoch {epoch+1}/{epochs}"):
            idx = perm[start : start + batch]
            text = prompt.encode(clip_model)
            logits = tau * (x[idx] @ text.T)
            anchor = (1 - F.cosine_similarity(text[0:1], static_anchor["normal"][None, :].to(device))).mean()
            anchor = anchor + (1 - F.cosine_similarity(text[1:2], static_anchor["abnormal"][None, :].to(device))).mean()
            loss = F.cross_entropy(logits, y[idx]) + lam * anchor
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean = float(np.mean(losses)) if losses else float("nan")
        curves.append({"epoch": epoch + 1, "loss": mean, "lambda_anchor": lam, "n_ctx": prompt.n_ctx})
        if mean + 1e-8 < best:
            best, wait = mean, 0
        else:
            wait += 1
        if wait >= patience:
            break
    prompt.eval()
    with torch.no_grad():
        text = prompt.encode(clip_model)
    return {
        "normal": text[0].detach(),
        "abnormal": text[1].detach(),
        "direction": F.normalize(text[1] - text[0], dim=0).detach(),
        "curves": curves,
    }

