import torch

from static_memory_reliability.prior_standardization import standardize_within_label


def raw_prior(stats, family, lambda_h=0.5, lambda_s=0.25, eps=1e-6):
    D = stats["D"].float()
    H = stats["H"].float()
    sigma = stats["sigma"].float()
    if family == "P0_Uniform":
        return torch.zeros_like(D)
    if family == "P1_DiscriminationOnly":
        return D
    if family == "P2_StabilityAdjustedDiscrimination":
        return stats["T"].float()
    if family == "P3_HubSuppressionOnly":
        return -H
    if family == "P4_DiscriminationMinusHubness":
        return D - float(lambda_h) * H
    if family == "P5_StableDiscriminationMinusHubness":
        return D - float(lambda_h) * H - float(lambda_s) * torch.log(sigma + eps)
    if family == "P6_CorrectCoveragePrior":
        return torch.log(stats["coverage"].float() + eps)
    if family == "P7_ScalePrior":
        return torch.log(stats["scale"].float() + eps)
    if family == "P8_DensityPrior":
        return stats["density"].float()
    raise ValueError(f"Unknown prior family: {family}")


def make_prior(stats, labels, family, alpha=0.0, lambda_h=0.5, lambda_s=0.25, centered=True):
    raw = raw_prior(stats, family, lambda_h=lambda_h, lambda_s=lambda_s)
    if centered:
        prior, standardization = standardize_within_label(raw, labels)
    else:
        prior = raw.clamp(-3, 3)
        standardization = {"non_centered": True}
    return {
        "family": family,
        "alpha": float(alpha),
        "lambda_h": float(lambda_h) if lambda_h is not None else "",
        "lambda_s": float(lambda_s) if lambda_s is not None else "",
        "prior": prior,
        "standardization": standardization,
    }

