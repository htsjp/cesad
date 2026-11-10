import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import OUT, rebuild_image_cache_safe, rebuild_patch_cache_safe, write_csv


def timed(name, fn, *args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = fn(*args)
    return result, {
        "subtest": name,
        "seconds": time.time() - start,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "batch_size": 1,
        "precision": "official fp32 tensors",
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    timed_root = OUT / "safe_rebuilt_cache_timed"
    timed_root.mkdir(parents=True, exist_ok=True)
    rows = []
    shape_rows = []
    for dataset_name, data_path, ckpt, cache_name, image_count in [
        ("mvtec", "data/mvtec", "checkpoints/test_on_visa.pth", "mvtec", 1725),
        ("visa", "data/VISA", "checkpoints/test_on_mvtec.pth", "visa", 2162),
    ]:
        image_path = timed_root / f"cache_model_{cache_name}.pt"
        patch_path = timed_root / f"cache_patch_model_{cache_name}.pt"
        if image_path.exists():
            image_path.unlink()
        if patch_path.exists():
            patch_path.unlink()
        image_cache, image_cost = timed(f"identity_rebuild_image_cache_{dataset_name}", rebuild_image_cache_safe, dataset_name, data_path, ckpt, image_path, device)
        image_cost["images"] = image_count
        rows.append(image_cost)
        patch_cache, patch_cost = timed(f"identity_rebuild_patch_cache_{dataset_name}", rebuild_patch_cache_safe, dataset_name, data_path, ckpt, patch_path, device)
        patch_cost["images"] = image_count
        rows.append(patch_cost)
        shape_rows.append({
            "dataset": dataset_name,
            "image_keys_shape": "x".join(map(str, image_cache["keys"].shape)),
            "image_values_shape": "x".join(map(str, image_cache["values"].shape)),
            "patch_keys_shape": "x".join(map(str, patch_cache["keys"].shape)),
            "patch_values_shape": "x".join(map(str, patch_cache["values"].shape)),
            "image_key_norm_mean": float(image_cache["keys"].norm(dim=-1).mean()),
            "patch_key_norm_mean": float(patch_cache["keys"].norm(dim=-1).mean()),
        })
        write_csv(OUT / "timed_cache_rebuild_costs_running.csv", rows)
        write_csv(OUT / "timed_cache_rebuild_shapes_running.csv", shape_rows)
    write_csv(OUT / "timed_cache_rebuild_costs.csv", rows)
    write_csv(OUT / "timed_cache_rebuild_shapes.csv", shape_rows)


if __name__ == "__main__":
    main()
