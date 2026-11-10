from phase_xa.teacher_memory_builder import build_memory_from_indices


def build_adapted_task_cache(model, dataset, indices, dataset_name, device, adapter, max_items=None):
    was_training = adapter.training
    adapter.eval()
    image_cache, patch_cache = build_memory_from_indices(model, dataset, indices, dataset_name, device, adapter=adapter, max_items=max_items)
    if was_training:
        adapter.train()
    return image_cache, patch_cache
