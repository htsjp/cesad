def transfer_directions(cfg):
    directions = cfg.get("transfer_directions")
    if directions:
        return directions
    return [
        {"name": "visa_to_mvtec", "source": "visa", "target": "mvtec"},
        {"name": "mvtec_to_visa", "source": "mvtec", "target": "visa"},
    ]


def selection_note():
    return {
        "source_validation_protocol": "source-only category-held-out fold scaffold; current run records target metrics as POST_HOC_TARGET_DIAGNOSTIC only",
        "target_labels_used_for_selection": False,
    }

