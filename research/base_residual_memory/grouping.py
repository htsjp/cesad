from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch

from .canonical import tensor_checksum


def load_head_state(path: str) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def recover_groups(entries: List[Dict], memory: Dict[str, torch.Tensor]) -> Tuple[List[Dict], Dict]:
    by_region = defaultdict(list)
    for i, entry in enumerate(entries):
        e = dict(entry)
        e["_position"] = i
        by_region[int(e["region_id"])].append(e)

    groups = []
    bad_size = []
    bad_label = []
    order_positions = []
    for gid in sorted(by_region):
        group = sorted(by_region[gid], key=lambda x: int(x["memory_id"]))
        labels = {int(x["label"]) for x in group}
        if len(group) != 4:
            bad_size.append(gid)
        if len(labels) != 1:
            bad_label.append(gid)
        positions = [int(x["_position"]) for x in group]
        order_positions.extend(positions)
        groups.append(
            {
                "region_id": gid,
                "source_id": group[0].get("source_id"),
                "region_type": group[0].get("region_type"),
                "label": int(group[0].get("label")),
                "memory_ids": [int(x["memory_id"]) for x in group],
                "positions": positions,
                "patch_indices": [int(x.get("patch_index", -1)) for x in group],
                "patch_yx": [[int(x.get("patch_y", -1)), int(x.get("patch_x", -1))] for x in group],
            }
        )

    labels_from_entries = [int(e["label"]) for e in sorted(entries, key=lambda x: int(x["memory_id"]))]
    labels_from_memory = memory["patch_values"].argmax(dim=-1).tolist()
    checks = {
        "total_prototypes": len(entries),
        "expected_total_prototypes": 12476,
        "group_count": len(groups),
        "all_groups_size_four": len(bad_size) == 0,
        "bad_size_group_ids": bad_size[:20],
        "all_group_labels_consistent": len(bad_label) == 0,
        "bad_label_group_ids": bad_label[:20],
        "every_prototype_once": sorted(order_positions) == list(range(len(entries))),
        "group_order_reconstructs_memory_order": order_positions == list(range(len(entries))),
        "labels_match_memory": labels_from_entries == [int(x) for x in labels_from_memory],
        "f_base_checksum": tensor_checksum(memory["F_base"]),
        "patch_values_checksum": tensor_checksum(memory["patch_values"]),
        "region_size_histogram": dict(sorted(Counter(len(v) for v in by_region.values()).items())),
    }
    checks["status"] = "OK" if (
        checks["total_prototypes"] == checks["expected_total_prototypes"]
        and checks["all_groups_size_four"]
        and checks["all_group_labels_consistent"]
        and checks["every_prototype_once"]
        and checks["group_order_reconstructs_memory_order"]
        and checks["labels_match_memory"]
    ) else "GROUP_METADATA_UNAVAILABLE"
    return groups, checks

