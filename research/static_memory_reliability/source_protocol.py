from text_anchored_memory.source_protocol import clone_filtered_loader, split_categories


def source_protocol_record(seed, direction, source, target, train_cats, val_cats):
    return {
        "seed": int(seed),
        "direction": direction,
        "source": source,
        "target": target,
        "source_train_categories": list(train_cats),
        "source_validation_categories": list(val_cats),
        "memory_construction": "compact memory is constructed from source training categories during selection; all source categories after selection",
        "reliability_population": "source training queries only during selection; all source queries for final target evaluation",
        "query_self_exclusion": "all memory entries from the same source image are masked while calculating source reliability statistics",
        "balancing": "patches are averaged within image, then images within category/query-label, then categories equally",
        "target_label_access_audit": "target labels are used only for final metric and post-hoc diagnostic calculation",
    }

