import torch
import torch.nn.functional as F

from memory_graph.sparse_propagation import fully_mixed, propagate, residual_fusion, two_step


def original_weights(query: torch.Tensor, memory: dict, tau_original: float):
    sim = query @ memory["patch_keys"].T
    w0 = F.softmax(sim / float(tau_original), dim=-1)
    return w0, w0 @ memory["patch_values"].float(), sim


def graph_residual_probs(query: torch.Tensor, memory: dict, transition_t, spec: dict):
    w0, probs0, sim = original_weights(query, memory, spec.get("tau_original", 1.0))
    family = spec["family"]
    if family == "v0":
        return probs0, {"w0": w0, "w": w0, "sim": sim}
    if family == "fully_mixed":
        w_graph = fully_mixed(w0)
    elif family == "two_step":
        w_graph = two_step(w0, transition_t)
    elif family == "dual_residual":
        w1 = propagate(w0, transition_t)
        w2 = propagate(w1, transition_t)
        l1 = float(spec["lambda1"])
        l2 = float(spec["lambda2"])
        w = (1.0 - l1 - l2) * w0 + l1 * w1 + l2 * w2
        return w @ memory["patch_values"].float(), {"w0": w0, "w": w, "w1": w1, "w2": w2, "sim": sim}
    else:
        w_graph = propagate(w0, transition_t)
    if family == "replacement":
        w = w_graph
    else:
        w = residual_fusion(w0, w_graph, spec.get("lambda_graph", 0.1))
    return w @ memory["patch_values"].float(), {"w0": w0, "w": w, "w_graph": w_graph, "sim": sim}

