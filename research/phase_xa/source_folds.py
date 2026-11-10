from utils.dataset import generate_class_info


def deterministic_source_folds(dataset_name, folds=3):
    classes, _ = generate_class_info(dataset_name)
    sorted_classes = sorted(classes)
    buckets = [[] for _ in range(folds)]
    for idx, cls_name in enumerate(sorted_classes):
        buckets[idx % folds].append(cls_name)
    rotations = []
    for fold_id in range(folds):
        val = buckets[fold_id]
        train = [c for i, bucket in enumerate(buckets) if i != fold_id for c in bucket]
        rotations.append({"fold": fold_id, "train_categories": train, "val_categories": val})
    return {"dataset": dataset_name, "folds": buckets, "rotations": rotations}
