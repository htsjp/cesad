from pathlib import Path


def write_retrieval_path_audit(path):
    lines = [
        "# Phase VII-B Retrieval Path Audit",
        "",
        "Canonical feature path:",
        "",
        "- `experiments/cross_dataset_ultra_compression_audit.py::build_source_memory` extracts layers 12/18/24 with `extract_value_representations`.",
        "- Patch query/key feature is `base_residual_memory.canonical::canonical_fuse(T24, V12, V18, alpha=0.5)`.",
        "- Canonical formula string is recorded by `base_residual_memory.canonical::FORMULA_NAME`.",
        "",
        "Original retrieval path:",
        "",
        "- `experiments/mrad_audit_utils.py::retrieval_logits(query, keys, values, temperature=1.0)`.",
        "- Similarity: `sim = torch.matmul(query, keys.T)`.",
        "- Temperature: `weights = F.softmax(sim / temperature, dim=-1)`.",
        "- Support/mask: no Top-k truncation and no mask in the verified path; support is the full memory bank.",
        "- Label aggregation: `logits = torch.matmul(weights, values.float())`.",
        "",
        "Image/patch score fusion:",
        "",
        "- `experiments/cross_dataset_ultra_compression_audit.py::evaluate_memory_variant`.",
        "- Image logits use the unchanged image-memory branch.",
        "- Patch logits are converted to anomaly maps by `AnomalyCLIP_lib.get_similarity_map(...)[..., 1]`.",
        "- Image score is `(1 - fusion_k) * topk_patch_mean + fusion_k * image_logits[:, 1]`.",
        "",
        "Phase VII-B insertion point:",
        "",
        "- New path: `static_memory_reliability.logit_calibration::calibrated_retrieval_logits`.",
        "- It first calculates the unmodified similarity `sim = query @ keys.T`.",
        "- The original support is full-bank support from unmodified similarity; no entry is added or removed.",
        "- It then applies `scaled = sim / temperature + alpha * s_logit * b_entry`.",
        "- `b_entry` is one scalar per region, repeated exactly across the four virtual entries.",
        "- The unchanged softmax and label-value aggregation are applied after this additive calibration.",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

