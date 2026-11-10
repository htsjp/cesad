import torch
import torch.nn.functional as F


def representation_row(name, mu, mu_hat, basis, stats, d_text=None):
    device = basis.device
    mu = mu.to(device)
    mu_hat = mu_hat.to(device)
    centered = stats["centered"].float().to(device)
    delta_v = stats["delta_v"].float().to(device)
    total = centered.pow(2).sum().clamp_min(1e-12)
    rec = mu_hat.float() - stats["mu0"][None, :].float().to(device)
    mse = (mu.float() - mu_hat.float()).pow(2).mean()
    cos = F.cosine_similarity(F.normalize(mu.float(), dim=-1), F.normalize(mu_hat.float(), dim=-1), dim=-1)
    row = {
        "variant": name,
        "centroid_reconstruction_mse": float(mse.item()),
        "centroid_mean_cosine": float(cos.mean().item()),
        "worst_centroid_cosine": float(cos.min().item()),
        "retained_covariance_energy": float(rec.pow(2).sum().div(total).item()),
        "projection_delta_v": float((basis.T @ delta_v).norm().item()),
        "normal_anomaly_centroid_distance": stats["centroid_distance"],
    }
    if d_text is not None:
        d_text = d_text.float().to(device)
        row["projection_d_text"] = float((basis.T @ d_text).norm().item())
        row["cosine_delta_v_d_text"] = float(F.cosine_similarity(delta_v, d_text, dim=0).item())
    return row


def text_anchor_row(direction, anchor_name, anchors, stats):
    device = anchors["direction"].device
    delta_v = stats["delta_v"].float().to(device)
    mu_normal = stats["mu_normal"].float().to(device)
    mu_abnormal = stats["mu_abnormal"].float().to(device)
    return {
        "direction": direction,
        "anchor": anchor_name,
        "text_anchor_separation": float((anchors["abnormal"] - anchors["normal"]).norm().item()),
        "cosine_delta_v_d_text": float(F.cosine_similarity(delta_v, anchors["direction"].float(), dim=0).item()),
        "text_to_normal_centroid_cosine": float(F.cosine_similarity(anchors["normal"].float(), mu_normal, dim=0).item()),
        "text_to_anomaly_centroid_cosine": float(F.cosine_similarity(anchors["abnormal"].float(), mu_abnormal, dim=0).item()),
    }
