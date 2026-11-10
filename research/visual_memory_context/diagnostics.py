import torch


def summarize_adapter(q0, q1, delta, gate=None, attn=None):
    row = {
        "q_movement": float((q1 - q0).norm(dim=-1).mean().detach().cpu()),
        "cosine_q1_q0": float((q1 * q0).sum(dim=-1).mean().detach().cpu()),
        "delta_norm": float(delta.norm(dim=-1).mean().detach().cpu()) if delta is not None else 0.0,
    }
    if gate is not None:
        row["gate_mean"] = float(gate.mean().detach().cpu())
        row["gate_min"] = float(gate.min().detach().cpu())
        row["gate_max"] = float(gate.max().detach().cpu())
    if attn is not None:
        vals = attn.mean(dim=0).detach().cpu().tolist()
        for i, v in enumerate(vals):
            row[f"attn_token_{i}"] = float(v)
    return row


def singular_spectrum(module):
    rows = []
    for name, p in module.named_parameters():
        if p.ndim == 2:
            with torch.no_grad():
                s = torch.linalg.svdvals(p.float().detach().cpu())
            rows.append({"parameter": name, "top_singular": float(s[0]), "rank_energy": float((s > 1e-6).float().sum())})
    return rows

