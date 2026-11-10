import torch
import torch.nn.functional as F

from models.mlp import average_neighbor
from phase_xa.official_task_forward import normalized_image_feature, visual_outputs
from phase_xa.provenance import image_provenance, provenance_aligned


def build_memory_from_indices(model, dataset, indices, dataset_name, device, adapter=None, max_items=None):
    image_keys, image_values, image_prov = [], [], []
    patch_keys, patch_values, patch_prov = [], [], []
    use_indices = indices if max_items is None else indices[:max_items]
    for idx in use_indices:
        item = dataset[idx]
        image = item["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = visual_outputs(model, image, adapter=adapter, grad=False)
            image_feature = normalized_image_feature(outputs).detach().cpu()
            image_keys.append(image_feature)
            image_values.append(torch.tensor([int(item["anomaly"])], dtype=torch.long))
            image_prov.append(image_provenance(dataset_name, item, idx, "image"))

            patch = average_neighbor(outputs[1][3]).reshape(37, 37, -1)
            mask = item["img_mask"].squeeze().to(device)
            mask = (mask > 0.5).float()
            small = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(37, 37), mode="bilinear", align_corners=False).squeeze()
            for label, region_mask in [(1, small > 0.5), (0, small <= 0.5)]:
                if region_mask.sum().item() == 0:
                    continue
                feature = patch[region_mask].mean(dim=0, keepdim=True)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                patch_keys.append(feature.detach().cpu())
                patch_values.append(torch.tensor([label], dtype=torch.long))
                patch_prov.append(image_provenance(dataset_name, item, idx, "patch", patch_region_index=len(patch_prov), label=label))
    image_labels = torch.cat(image_values).long()
    patch_labels = torch.cat(patch_values).long()
    image_cache = {
        "keys": torch.cat(image_keys, dim=0),
        "values": F.one_hot(image_labels, num_classes=2).float(),
        "provenance": image_prov,
    }
    patch_cache = {
        "keys": torch.cat(patch_keys, dim=0),
        "values": F.one_hot(patch_labels, num_classes=2).float(),
        "provenance": patch_prov,
    }
    if not provenance_aligned(image_cache["keys"], image_cache["values"], image_cache["provenance"]):
        raise RuntimeError("image cache provenance alignment failed")
    if not provenance_aligned(patch_cache["keys"], patch_cache["values"], patch_cache["provenance"]):
        raise RuntimeError("patch cache provenance alignment failed")
    return image_cache, patch_cache
