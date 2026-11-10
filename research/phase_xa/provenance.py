def image_provenance(dataset_name, item, idx, memory_level, patch_region_index=None, label=None):
    return {
        "source_dataset": dataset_name,
        "category": item["cls_name"],
        "sample_id": int(idx),
        "image_path": item["img_path"],
        "memory_level": memory_level,
        "patch_region_index": patch_region_index,
        "label": int(label if label is not None else item["anomaly"]),
        "source_augmentation_id": "none",
    }


def provenance_aligned(keys, values, provenance):
    return keys.shape[0] == values.shape[0] == len(provenance)
