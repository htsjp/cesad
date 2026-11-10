import copy
import numpy as np
import torch


def split_categories(dataset, seed, val_fraction=0.25):
    cats = sorted({x["cls_name"] for x in dataset.data_all}) or list(dataset.obj_list)
    if len(cats) < 2:
        return cats, cats
    rng = np.random.default_rng(int(seed))
    perm = list(rng.permutation(cats))
    n_val = max(1, int(round(len(perm) * float(val_fraction))))
    return sorted(perm[n_val:]), sorted(perm[:n_val])


def clone_filtered_loader(loader, categories, batch_size=1, shuffle=False):
    ds = copy.deepcopy(loader.dataset)
    categories = set(categories)
    ds.data_all = [x for x in ds.data_all if x["cls_name"] in categories]
    ds.length = len(ds.data_all)
    ds.cls_names = sorted(categories)
    ds.obj_list = [x for x in ds.obj_list if x in categories]
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def source_protocol_record(seed, direction, source, target, train_cats, val_cats):
    return {
        "seed": int(seed),
        "direction": direction,
        "source": source,
        "target": target,
        "source_train_categories": list(train_cats),
        "source_validation_categories": list(val_cats),
        "source_memory_construction": "source train categories for folds; all source categories after selection",
        "source_query_self_exclusion": "category-held-out validation makes image-level self-exclusion unnecessary for held-out queries",
        "target_label_access_audit": "target labels used only by metric evaluation",
    }

