from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


def variant_name(spec: Dict) -> str:
    family = spec["family"]
    if family == "single":
        return f"{spec['rep']}{spec['layers'][0]}"
    if family == "mean":
        return f"{spec['rep']}{'_'.join(map(str, spec['layers']))}_Mean"
    if family == "anchor":
        return f"T24_plus_{spec['source']}{'_'.join(map(str, spec['layers']))}_alpha{spec['alpha']}"
    if family == "random":
        return f"Random_{spec['rep']}{'_'.join(map(str, spec['layers']))}_Mean"
    return spec.get("name", family)


def build_variant_specs(cfg: Dict) -> List[Dict]:
    layers = [int(x) for x in cfg.get("layers", [6, 12, 18, 24])]
    specs = [{"family": "single", "rep": "T", "layers": [24], "name": "T24_Baseline"}]
    for rep in ("T", "V", "AV", "PAV"):
        for layer in layers:
            spec = {"family": "single", "rep": rep, "layers": [layer]}
            if rep == "T" and layer == 24:
                spec["name"] = "T24_Baseline"
            specs.append(spec)
    combos = [[18, 24], [12, 18, 24], [6, 12, 18, 24]]
    for rep in ("T", "V", "AV"):
        for combo in combos:
            specs.append({"family": "mean", "rep": rep, "layers": combo})
    for source, combos_for_source in {
        "V": [[18], [18, 24], [12, 18]],
        "AV": [[18], [18, 24], [12, 18]],
    }.items():
        for combo in combos_for_source:
            for alpha in cfg.get("alpha_values", [0.1, 0.25, 0.5]):
                specs.append({"family": "anchor", "source": source, "layers": combo, "alpha": float(alpha)})
    random_layers = cfg.get("random_layers", [[4, 10], [4, 10, 16], [4, 10, 16, 22]])
    for rep in ("T", "V", "AV"):
        for combo in random_layers:
            specs.append({"family": "random", "rep": rep, "layers": [int(x) for x in combo]})

    seen = set()
    unique = []
    for spec in specs:
        name = spec.get("name", variant_name(spec))
        spec["name"] = name
        if name not in seen:
            seen.add(name)
            unique.append(spec)
    if cfg.get("run_mode", "primary") == "smoke":
        keep = {"T24_Baseline", "T6", "V6", "AV6", "PAV6", "T18_24_Mean", "V18_24_Mean", "AV18_24_Mean", "T24_plus_V18_alpha0.25"}
        unique = [x for x in unique if x["name"] in keep]
    return unique


def _mean_features(outputs: Dict[int, Dict[str, torch.Tensor]], layers: Sequence[int], rep: str) -> torch.Tensor:
    parts = [F.normalize(outputs[int(layer)][rep][:, 1:, :].float(), dim=-1) for layer in layers]
    return F.normalize(torch.stack(parts, dim=0).mean(dim=0), dim=-1)


def fuse_patch_features(outputs: Dict[int, Dict[str, torch.Tensor]], spec: Dict) -> torch.Tensor:
    family = spec["family"]
    if family == "single":
        return F.normalize(outputs[int(spec["layers"][0])][spec["rep"]][:, 1:, :].float(), dim=-1)
    if family in {"mean", "random"}:
        return _mean_features(outputs, spec["layers"], spec["rep"])
    if family == "anchor":
        anchor = F.normalize(outputs[24]["T"][:, 1:, :].float(), dim=-1)
        value = _mean_features(outputs, spec["layers"], spec["source"])
        return F.normalize(anchor + float(spec["alpha"]) * value, dim=-1)
    raise ValueError(f"Unknown variant family: {family}")


def required_layers(specs: Iterable[Dict]) -> List[int]:
    layers = {24}
    for spec in specs:
        layers.update(int(x) for x in spec.get("layers", []))
    return sorted(layers)
