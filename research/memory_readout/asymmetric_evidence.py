from .kernel_evidence import topk_kernel_evidence


def asymmetric_evidence_logit(query, normal_keys, anomaly_keys, k, tau, alpha_a, alpha_n):
    en, vals_n, idx_n = topk_kernel_evidence(query @ normal_keys.T, k, tau)
    ea, vals_a, idx_a = topk_kernel_evidence(query @ anomaly_keys.T, k, tau)
    return float(alpha_a) * ea - float(alpha_n) * en, en, ea, vals_n, vals_a, idx_n, idx_a
