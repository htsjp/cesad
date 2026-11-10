import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import evaluate_results, init_results
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import load_dataset, to_device_cache
from models.mlp import average_neighbor
from mrad import compute_patch_socre
from phase_x.gradient_smoke_test import hashes_equal, parameter_hashes
from phase_xa.official_task_forward import (
    image_memory_scores,
    normalized_image_feature,
    normalized_patch_feature,
    patch_memory_scores,
    visual_outputs,
)
from phase_xa.shared_adapter import SharedBlock18Adapter, trainable_parameter_count
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xiv_dsls/source_smoke")
CKPT = OUT / "checkpoints"
VARIANTS = ["XIV1_SharedAdapter", "XIV2_SemanticOnly", "XIV3_LocalOnly", "XIV5_DSLS"]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class BottleneckAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck=32, max_gate=0.20):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        initial = 0.05 / max_gate
        self.gate_raw = nn.Parameter(torch.tensor(math.log(initial / (1.0 - initial)), dtype=torch.float32))
        self.max_gate = float(max_gate)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.last = {}

    @property
    def gate(self):
        return self.max_gate * torch.sigmoid(self.gate_raw)

    def forward(self, x):
        y = self.up(self.activation(self.down(self.pre_norm(x))))
        out = x + self.gate.to(dtype=x.dtype, device=x.device) * y
        self.last = {"input": x, "output": out, "residual": y}
        return out


class BranchModule(nn.Module):
    accepts_list_output = True

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.mode = "both"
        if variant == "XIV1_SharedAdapter":
            self.accepts_list_output = False
            self.shared = SharedBlock18Adapter(dim=1024, bottleneck=64)
        elif variant == "XIV2_SemanticOnly":
            self.semantic = BottleneckAdapter(bottleneck=32)
        elif variant == "XIV3_LocalOnly":
            self.local = BottleneckAdapter(bottleneck=32)
        elif variant == "XIV5_DSLS":
            self.semantic = BottleneckAdapter(bottleneck=32)
            self.local = BottleneckAdapter(bottleneck=32)
        else:
            raise ValueError(variant)
        self.last = {}

    def trainable_modules(self):
        mods = []
        if hasattr(self, "shared"):
            mods.append(("shared", self.shared))
        if hasattr(self, "semantic"):
            mods.append(("semantic", self.semantic))
        if hasattr(self, "local"):
            mods.append(("local", self.local))
        return mods

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, output):
        if hasattr(self, "shared"):
            return self.shared(output)
        if not isinstance(output, list) or len(output) != 2:
            raise RuntimeError("BranchModule expected DAPM two-branch list at block18")
        branch0, branch1 = output
        self.last = {}
        if hasattr(self, "local") and self.mode in {"both", "local"}:
            nld = branch0.permute(1, 0, 2)
            y = self.local(nld).permute(1, 0, 2)
            self.last["local"] = self.local.last
            branch0 = y
        if hasattr(self, "semantic") and self.mode in {"both", "semantic"}:
            nld = branch1.permute(1, 0, 2)
            y = self.semantic(nld).permute(1, 0, 2)
            self.last["semantic"] = self.semantic.last
            branch1 = y
        return [branch0, branch1]


def params_of(module, which=None):
    if which is None:
        return [p for p in module.parameters() if p.requires_grad]
    return [p for n, p in module.named_parameters() if p.requires_grad and n.startswith(which + ".")]


def all_indices_for_categories(dataset, categories):
    cats = set(categories)
    return [idx for idx in range(len(dataset)) if dataset[idx]["cls_name"] in cats]


def patch_mask(item, device, side):
    mask = item["img_mask"].squeeze().to(device)
    mask = (mask > 0.5).float()
    return F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(side, side), mode="bilinear", align_corners=False).reshape(1, -1)


def build_cache(model, dataset, indices, dataset_name, device, adapter):
    was = adapter.training
    adapter.eval()
    if hasattr(adapter, "set_mode"):
        old = adapter.mode
        adapter.set_mode("both")
    with torch.no_grad():
        image_cache, patch_cache = build_memory_from_indices(model, dataset, indices, dataset_name, device, adapter=adapter)
    if hasattr(adapter, "set_mode"):
        adapter.set_mode(old)
    if was:
        adapter.train()
    return to_device_cache(image_cache, device), to_device_cache(patch_cache, device)


def losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="both"):
    image = item["img"].unsqueeze(0).to(device)
    if hasattr(adapter, "set_mode"):
        adapter.set_mode(mode)
    teacher = None
    if mode == "both":
        with torch.no_grad():
            teacher = visual_outputs(model, image, adapter=None, grad=False)
    out = visual_outputs(model, image, adapter=adapter, grad=True)
    img = normalized_image_feature(out)
    patch = normalized_patch_feature(out)
    image_post, _, _, _ = image_memory_scores(img, image_cache["keys"], image_cache["values"], image_proj)
    patch_post, _, _, _ = patch_memory_scores(patch, patch_cache["keys"], patch_cache["values"], patch_proj)
    target = torch.tensor([int(item["anomaly"])], device=device)
    l_image = F.nll_loss(torch.log(image_post.clamp_min(1e-6)), target)
    side = int(math.sqrt(patch_post.shape[1]))
    l_pixel = F.binary_cross_entropy(patch_post[..., 1].clamp(1e-6, 1 - 1e-6), patch_mask(item, device, side))
    l_pres_img = torch.tensor(0.0, device=device)
    l_pres_patch = torch.tensor(0.0, device=device)
    if teacher is not None:
        t_img = normalized_image_feature(teacher)
        t_patch = normalized_patch_feature(teacher)
        l_pres_img = 1.0 - F.cosine_similarity(img, t_img.detach(), dim=-1).mean()
        l_pres_patch = 1.0 - F.cosine_similarity(patch.reshape(-1, patch.shape[-1]), t_patch.detach().reshape(-1, t_patch.shape[-1]), dim=-1).mean()
    l_pres_branch_sem = torch.tensor(0.0, device=device)
    l_pres_branch_loc = torch.tensor(0.0, device=device)
    if hasattr(adapter, "last"):
        if "semantic" in adapter.last:
            a = adapter.last["semantic"]
            l_pres_branch_sem = 1.0 - F.cosine_similarity(a["output"].reshape(-1, a["output"].shape[-1]), a["input"].detach().reshape(-1, a["input"].shape[-1]), dim=-1).mean()
        if "local" in adapter.last:
            a = adapter.last["local"]
            l_pres_branch_loc = 1.0 - F.cosine_similarity(a["output"].reshape(-1, a["output"].shape[-1]), a["input"].detach().reshape(-1, a["input"].shape[-1]), dim=-1).mean()
    return {
        "image": l_image,
        "pixel": l_pixel,
        "pres_img": l_pres_img,
        "pres_patch": l_pres_patch,
        "pres_sem": l_pres_branch_sem + l_pres_img,
        "pres_local": l_pres_branch_loc + l_pres_patch,
        "image_post": image_post,
        "patch_post": patch_post,
    }


def grad_vec(grads, params):
    parts = []
    for g, p in zip(grads, params):
        parts.append(torch.zeros_like(p, dtype=torch.float32).reshape(-1) if g is None else g.detach().float().reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(1)


def cos_or_none(a, b):
    na = float(a.norm().cpu())
    nb = float(b.norm().cpu())
    if na < 1e-12 or nb < 1e-12:
        return None, na, nb
    return float((a @ b / (a.norm() * b.norm())).cpu()), na, nb


def group_name(name):
    if name.endswith("gate_raw"):
        return "gate"
    if ".pre_norm." in name or name.startswith("pre_norm."):
        return "LayerNorm"
    if name.endswith("down.weight"):
        return "down.weight"
    if name.endswith("down.bias"):
        return "down.bias"
    if name.endswith("up.weight"):
        return "up.weight"
    if name.endswith("up.bias"):
        return "up.bias"
    return name


def gradient_snapshot(model, adapter, dataset, diag_indices, image_proj, patch_proj, image_cache, patch_cache, device, tag, global_step):
    rows = []
    params = params_of(adapter)
    named = [(n, p) for n, p in adapter.named_parameters() if p.requires_grad]
    for b, idx in enumerate(diag_indices):
        item = dataset[idx]
        l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="both")
        gi = torch.autograd.grad(l["image"], params, retain_graph=False, allow_unused=True)
        l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="both")
        gp = torch.autograd.grad(l["pixel"], params, retain_graph=False, allow_unused=True)
        vi, vp = grad_vec(gi, params), grad_vec(gp, params)
        c, ni, npix = cos_or_none(vi, vp)
        rows.append({"timepoint": tag, "global_step": global_step, "batch_id": b, "sample_idx": idx, "parameter_group": "total", "cos_grad": c, "image_grad_norm": ni, "pixel_grad_norm": npix})
        for group in sorted(set(group_name(n) for n, _ in named)):
            sel = [i for i, (n, _) in enumerate(named) if group_name(n) == group]
            vi_g = grad_vec([gi[i] for i in sel], [named[i][1] for i in sel])
            vp_g = grad_vec([gp[i] for i in sel], [named[i][1] for i in sel])
            cg, nig, npg = cos_or_none(vi_g, vp_g)
            rows.append({"timepoint": tag, "global_step": global_step, "batch_id": b, "sample_idx": idx, "parameter_group": group, "cos_grad": cg, "image_grad_norm": nig, "pixel_grad_norm": npg})
    return rows


def summarize_timeline(rows):
    out = []
    for tp in ["T0", "T1", "T2", "T3", "T4"]:
        vals = [r["cos_grad"] for r in rows if r["timepoint"] == tp and r["parameter_group"] == "total" and r["cos_grad"] is not None]
        arr = np.asarray(vals, dtype=np.float64)
        out.append({
            "timepoint": tp,
            "mean_cos": float(arr.mean()) if arr.size else float("nan"),
            "std_cos": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "median_cos": float(np.median(arr)) if arr.size else float("nan"),
            "frac_negative": float((arr < 0).mean()) if arr.size else float("nan"),
            "frac_lt_minus_0p1": float((arr < -0.1).mean()) if arr.size else float("nan"),
            "batches": int(arr.size),
        })
    return out


def interference_label(summary):
    by = {r["timepoint"]: r for r in summary}
    t1t4 = [by[t] for t in ["T1", "T2", "T3", "T4"]]
    persistent = sum(r["mean_cos"] <= 0.10 and r["frac_negative"] >= 0.30 for r in t1t4)
    if persistent >= 3:
        return "PERSISTENT_INTERFERENCE"
    if by["T0"]["mean_cos"] <= 0.10 and by["T0"]["frac_negative"] >= 0.30 and all(by[t]["mean_cos"] > 0.20 and by[t]["frac_negative"] < 0.20 for t in ["T2", "T3", "T4"]):
        return "TRANSIENT_INITIALIZATION_INTERFERENCE"
    return "MIXED_INTERFERENCE"


def connectivity_audit(model, dataset, idx, image_proj, patch_proj, image_cache, patch_cache, device):
    adapter = BranchModule("XIV5_DSLS").to(device)
    item = dataset[idx]
    l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="semantic")
    sem_params, loc_params = params_of(adapter, "semantic"), params_of(adapter, "local")
    g_sem = grad_vec(torch.autograd.grad(l["image"], sem_params, retain_graph=False, allow_unused=True), sem_params)
    l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="semantic")
    g_loc_from_image = grad_vec(torch.autograd.grad(l["image"], loc_params, retain_graph=False, allow_unused=True), loc_params)
    l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="local")
    g_loc = grad_vec(torch.autograd.grad(l["pixel"], loc_params, retain_graph=False, allow_unused=True), loc_params)
    l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="local")
    g_sem_from_pixel = grad_vec(torch.autograd.grad(l["pixel"], sem_params, retain_graph=False, allow_unused=True), sem_params)
    result = {
        "grad_L_image_to_SemanticAdapter": float(g_sem.norm().cpu()),
        "grad_L_image_to_LocalAdapter": float(g_loc_from_image.norm().cpu()),
        "grad_L_pixel_to_LocalAdapter": float(g_loc.norm().cpu()),
        "grad_L_pixel_to_SemanticAdapter": float(g_sem_from_pixel.norm().cpu()),
    }
    result["pass"] = result["grad_L_image_to_SemanticAdapter"] > 0 and result["grad_L_pixel_to_LocalAdapter"] > 0 and result["grad_L_image_to_LocalAdapter"] < 1e-12 and result["grad_L_pixel_to_SemanticAdapter"] < 1e-12
    return result


def train_variant(variant, model, dataset, train_indices, diag_indices, image_proj, patch_proj, dataset_name, device):
    adapter = BranchModule(variant).to(device).train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    before = parameter_hashes(model)
    training_rows, timeline_rows = [], []
    total_steps = len(train_indices) * 3
    marks = {0: "T0", max(1, total_steps // 4): "T1", max(1, total_steps // 2): "T2", max(1, (3 * total_steps) // 4): "T3", total_steps: "T4"}
    step = 0
    image_cache, patch_cache = build_cache(model, dataset, train_indices, dataset_name, device, adapter)
    if variant == "XIV1_SharedAdapter":
        timeline_rows.extend(gradient_snapshot(model, adapter, dataset, diag_indices, image_proj, patch_proj, image_cache, patch_cache, device, "T0", 0))
    prev_state = {n: p.detach().clone() for n, p in adapter.named_parameters() if p.requires_grad}
    for epoch in range(3):
        image_cache, patch_cache = build_cache(model, dataset, train_indices, dataset_name, device, adapter)
        for idx in tqdm(train_indices, desc=f"{variant} epoch {epoch+1}/3"):
            step += 1
            item = dataset[idx]
            opt.zero_grad(set_to_none=True)
            if variant == "XIV1_SharedAdapter":
                l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="both")
                loss = l["image"] + l["pixel"] + l["pres_img"] + l["pres_patch"]
            elif variant == "XIV2_SemanticOnly":
                l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="semantic")
                loss = l["image"] + l["pres_sem"]
            elif variant == "XIV3_LocalOnly":
                l = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="local")
                loss = l["pixel"] + l["pres_local"]
            else:
                l_sem = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="semantic")
                l_loc = losses(model, adapter, item, image_proj, patch_proj, image_cache, patch_cache, device, mode="local")
                loss = l_sem["image"] + l_sem["pres_sem"] + l_loc["pixel"] + l_loc["pres_local"]
                l = {"image": l_sem["image"], "pixel": l_loc["pixel"], "pres_sem": l_sem["pres_sem"], "pres_local": l_loc["pres_local"]}
            loss.backward()
            opt.step()
            upd_sem = upd_loc = 0.0
            for n, p in adapter.named_parameters():
                if not p.requires_grad:
                    continue
                delta = float((p.detach() - prev_state[n]).float().norm().cpu())
                if n.startswith("semantic."):
                    upd_sem += delta ** 2
                if n.startswith("local."):
                    upd_loc += delta ** 2
                prev_state[n] = p.detach().clone()
            training_rows.append({
                "variant": variant,
                "epoch": epoch,
                "step": step,
                "sample_idx": idx,
                "category": item["cls_name"],
                "label": int(item["anomaly"]),
                "loss": float(loss.detach().cpu()),
                "image_loss": float(l["image"].detach().cpu()),
                "pixel_loss": float(l["pixel"].detach().cpu()),
                "semantic_update_norm": math.sqrt(upd_sem),
                "local_update_norm": math.sqrt(upd_loc),
            })
            if variant == "XIV1_SharedAdapter" and step in marks and marks[step] != "T0":
                timeline_rows.extend(gradient_snapshot(model, adapter, dataset, diag_indices, image_proj, patch_proj, image_cache, patch_cache, device, marks[step], step))
    if variant == "XIV1_SharedAdapter" and not any(r["timepoint"] == "T4" for r in timeline_rows):
        timeline_rows.extend(gradient_snapshot(model, adapter, dataset, diag_indices, image_proj, patch_proj, image_cache, patch_cache, device, "T4", step))
    ckpt = CKPT / f"D1_fold0_{variant}_seed0.pt"
    torch.save({"adapter": adapter.state_dict(), "variant": variant, "seed": 0}, ckpt)
    return adapter.eval(), training_rows, timeline_rows, {"official_unchanged": hashes_equal(before, parameter_hashes(model)), "checkpoint": str(ckpt), "steps": step}


def append_result(results, cls_name, item, image_score, anomaly_map):
    mask = (item["img_mask"].detach().cpu() > 0.5).float()
    results[cls_name]["imgs_masks"].append(mask)
    results[cls_name]["gt_sp"].append(int(item["anomaly"]))
    results[cls_name]["pr_sp"].append(float(image_score))
    results[cls_name]["anomaly_maps"].append(anomaly_map.detach().cpu())


def evaluate_variant(variant, adapter, model, prompt_learner, image_proj, patch_proj, prompt_proj, dataset, train_indices, val_indices, dataset_name, device):
    image_cache, patch_cache = build_cache(model, dataset, train_indices, dataset_name, device, adapter)
    obj_list = sorted({dataset[i]["cls_name"] for i in val_indices})
    results = init_results(obj_list)
    for idx in tqdm(val_indices, desc=f"eval {variant}"):
        item = dataset[idx]
        image = item["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(adapter, "set_mode"):
                adapter.set_mode("both")
            outputs = visual_outputs(model, image, adapter=adapter, grad=False)
            image_feature = normalized_image_feature(outputs)
            image_post, _, _, _ = image_memory_scores(image_feature, image_cache["keys"], image_cache["values"], image_proj)
            cls_memory = image_post[:, 1]
            patch_projection = average_neighbor(outputs[3][3])
            patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
            patch_bias = average_neighbor(outputs[1][3])
            patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)
            _, patch_bias_tokens, _, _ = compute_patch_socre(patch_bias, patch_cache["keys"], patch_cache["values"], device=device, proj=patch_proj, need_mask=False, patch_projection=patch_projection, gt_mask=item["img_mask"].unsqueeze(0), is_mradft=False, use_proj=True)
            bias = prompt_proj(patch_bias_tokens[:, 0, :], patch_bias_tokens[:, 1, :])
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            maps = []
            for pidx, patch_feature in enumerate(outputs[3]):
                if pidx >= 3:
                    patch_feature = average_neighbor(patch_feature)
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    sim = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                    smap = AnomalyCLIP_lib.get_similarity_map(sim, 518)
                    maps.append((smap[..., 1] + 1 - smap[..., 0]) / 2.0)
            amap = torch.stack(maps)
            amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=4)) for x in amap.detach().cpu()], dim=0).sum(dim=0)
            topk = torch.topk(amap.flatten(), max(1, int(amap.numel() * 0.01))).values.mean()
            final_score = 0.3 * topk.to(device) + 0.7 * cls_memory
            append_result(results, item["cls_name"], item, final_score.detach().cpu(), amap)
    return evaluate_results(results, obj_list, compute_pro=True)


def mv(metrics, key):
    v = float(metrics["mean"].get(key, float("nan")))
    return v * 100.0 if np.isfinite(v) and v <= 1.5 else v


def representation_diag(variant, adapter, model, dataset, indices, device):
    rows = []
    cos_b0 = []
    cos_b1 = []
    cos_cls = []
    cos_patch = []
    for idx in indices[:32]:
        item = dataset[idx]
        image = item["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            base = visual_outputs(model, image, adapter=None, grad=False, snapshot_mode="true_clone")
            if hasattr(adapter, "set_mode"):
                adapter.set_mode("both")
            out = visual_outputs(model, image, adapter=adapter, grad=False, snapshot_mode="true_clone")
            b = base["snapshots"]["true_block18"]
            a = out["snapshots"]["true_block18"]
            cos_b0.append(float(F.cosine_similarity(a[0].reshape(-1, a[0].shape[-1]).float(), b[0].reshape(-1, b[0].shape[-1]).float(), dim=-1).mean().cpu()))
            cos_b1.append(float(F.cosine_similarity(a[1].reshape(-1, a[1].shape[-1]).float(), b[1].reshape(-1, b[1].shape[-1]).float(), dim=-1).mean().cpu()))
            cls_a = normalized_image_feature(out["outputs"]).float()
            cls_b = normalized_image_feature(base["outputs"]).float()
            pa = normalized_patch_feature(out["outputs"]).float()
            pb = normalized_patch_feature(base["outputs"]).float()
            cos_cls.append(float(F.cosine_similarity(cls_a, cls_b, dim=-1).mean().cpu()))
            cos_patch.append(float(F.cosine_similarity(pa.reshape(-1, pa.shape[-1]), pb.reshape(-1, pb.shape[-1]), dim=-1).mean().cpu()))
    for name, mod in adapter.trainable_modules():
        residual_norm = float(mod.last.get("residual", torch.zeros(1)).detach().float().norm().cpu()) if hasattr(mod, "last") else 0.0
        rows.append({
            "variant": variant,
            "adapter": name,
            "block18_branch0_cosine": float(np.mean(cos_b0)),
            "block18_branch1_cosine": float(np.mean(cos_b1)),
            "final_cls_cosine": float(np.mean(cos_cls)),
            "final_patch_cosine": float(np.mean(cos_patch)),
            "adapter_residual_norm": residual_norm,
            "adapter_gate": float(mod.gate.detach().cpu()) if hasattr(mod, "gate") else float(adapter.shared.gate.detach().cpu()),
            "trainable_parameters": trainable_parameter_count(mod),
        })
    return rows


def decide(raw_rows, rep_rows, timeline_summary):
    base = next(r for r in raw_rows if r["variant"] == "XIV1_SharedAdapter")
    grouped = []
    for r in raw_rows:
        di = r["I-AUROC"] - base["I-AUROC"]
        dap = r["I-AP"] - base["I-AP"]
        dp = r["P-AUROC"] - base["P-AUROC"]
        dpro = r["PRO"] - base["PRO"]
        gi = 0.5 * di + 0.5 * dap
        gp = 0.5 * dp + 0.5 * dpro
        gb = 0.5 * gi + 0.5 * gp
        grouped.append({"variant": r["variant"], "dI-AUROC": di, "dI-AP": dap, "dP-AUROC": dp, "dPRO": dpro, "G_image": gi, "G_pixel": gp, "G_balanced": gb})
    g = {r["variant"]: r for r in grouped}
    clip_ok = all(float(r[k]) >= 0.90 for r in rep_rows for k in ["block18_branch0_cosine", "block18_branch1_cosine", "final_cls_cosine", "final_patch_cosine"])
    major_drop_ok = all(min(r["dI-AUROC"], r["dI-AP"], r["dP-AUROC"], r["dPRO"]) >= -0.30 for r in grouped)
    xiv5 = g["XIV5_DSLS"]
    p1 = g["XIV2_SemanticOnly"]["G_image"] >= 0.15
    p2 = g["XIV3_LocalOnly"]["G_pixel"] >= 0.15
    p3 = (xiv5["G_image"] >= 0.20 and xiv5["G_pixel"] >= -0.10) or (xiv5["G_pixel"] >= 0.20 and xiv5["G_image"] >= -0.10)
    if p1 and p2 and p3 and xiv5["G_balanced"] >= 0.15 and clip_ok and major_drop_ok:
        outcome = "B1_PASS_TO_XIV_C"
    elif xiv5["G_balanced"] >= 0.20:
        outcome = "B2_DECOUPLING_GAIN_WITHOUT_CLEAN_DECOMPOSITION"
    elif p1 and p2:
        outcome = "B3_SPECIALISTS_WORK_BUT_COMBINATION_FAILS"
    else:
        sec = interference_label(timeline_summary)
        outcome = "B5_TRANSIENT_CONFLICT_NO_TASK_VALUE" if sec == "TRANSIENT_INITIALIZATION_INTERFERENCE" else "B4_NO_SEMANTIC_LOCAL_DECOMPOSITION"
    return outcome, grouped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_name, data_path, checkpoint = "mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
    dataset = load_dataset(dataset_name, data_path)
    rot = deterministic_source_folds(dataset_name)["rotations"][0]
    train_indices = all_indices_for_categories(dataset, rot["train_categories"])
    val_indices = all_indices_for_categories(dataset, rot["val_categories"])
    diag_indices = train_indices[:16]
    base_cache_i, base_cache_p = build_cache(model, dataset, train_indices, dataset_name, device, BranchModule("XIV1_SharedAdapter").to(device))
    routing = {"parameter_counts": {}, "connectivity": connectivity_audit(model, dataset, diag_indices[0], image_proj, patch_proj, base_cache_i, base_cache_p, device)}
    for v in VARIANTS:
        routing["parameter_counts"][v] = trainable_parameter_count(BranchModule(v))
    c = routing["parameter_counts"]
    routing["xiv5_vs_xiv1_param_difference"] = {"absolute": c["XIV5_DSLS"] - c["XIV1_SharedAdapter"], "percent": 100.0 * (c["XIV5_DSLS"] - c["XIV1_SharedAdapter"]) / c["XIV1_SharedAdapter"]}
    if not routing["connectivity"]["pass"]:
        payload = {"primary_outcome": "B6_IMPLEMENTATION_FAILURE", "reason": "XIV5 gradient routing failed", **routing}
        (OUT / "XIV_B_ROUTING_AUDIT.json").write_text(json.dumps(routing, indent=2), encoding="utf-8")
        (OUT / "XIV_B_OUTCOME.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit("B6_IMPLEMENTATION_FAILURE")
    adapters, all_train, all_timeline, train_diag = {}, [], [], {}
    for v in VARIANTS:
        adapter, train_rows, timeline_rows, diag = train_variant(v, model, dataset, train_indices, diag_indices, image_proj, patch_proj, dataset_name, device)
        adapters[v] = adapter
        all_train.extend(train_rows)
        for r in timeline_rows:
            r["variant"] = v
        all_timeline.extend(timeline_rows)
        train_diag[v] = diag
        write_csv(OUT / "XIV_B_TRAINING_GRADIENTS.csv", all_train)
        write_csv(OUT / "XIV_B_GRADIENT_TIMELINE.csv", all_timeline)
    raw, percat, repr_rows = [], [], []
    for v, adapter in adapters.items():
        metrics = evaluate_variant(v, adapter, model, prompt_learner, image_proj, patch_proj, prompt_proj, dataset, train_indices, val_indices, dataset_name, device)
        raw.append({"variant": v, "I-AUROC": mv(metrics, "I-AUROC"), "I-AP": mv(metrics, "I-AP"), "P-AUROC": mv(metrics, "P-AUROC"), "PRO": mv(metrics, "PRO"), "val_images": len(val_indices)})
        for cat, vals in metrics["per_class"].items():
            percat.append({"variant": v, "category": cat, "I-AUROC": vals.get("I-AUROC", float("nan")) * 100.0, "P-AUROC": vals.get("P-AUROC", float("nan")) * 100.0, "PRO": vals.get("PRO", float("nan")) * 100.0})
        repr_rows.extend(representation_diag(v, adapter, model, dataset, val_indices, device))
        write_csv(OUT / "XIV_B_RAW_METRICS.csv", raw)
        write_csv(OUT / "XIV_B_PER_CATEGORY.csv", percat)
        write_csv(OUT / "XIV_B_REPRESENTATION.csv", repr_rows)
    timeline_summary = summarize_timeline(all_timeline)
    secondary = interference_label(timeline_summary)
    primary, grouped = decide(raw, repr_rows, timeline_summary)
    write_csv(OUT / "XIV_B_GROUPED_GAINS.csv", grouped)
    routing["training"] = train_diag
    routing["secondary_interference_label"] = secondary
    (OUT / "XIV_B_ROUTING_AUDIT.json").write_text(json.dumps(routing, indent=2), encoding="utf-8")
    outcome = {"primary_outcome": primary, "secondary_interference_label": secondary, "routing": routing, "target_accessed": False, "next": "STOP. Do not run XIV-C in this invocation."}
    (OUT / "XIV_B_OUTCOME.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    lines = ["# Phase XIV-B Source Smoke", "", f"Primary outcome: **{primary}**", f"Secondary interference label: **{secondary}**", "", "No D2 training and no target access were performed.", "", "## Raw Metrics", "", "| Variant | I-AUROC | I-AP | P-AUROC | PRO |", "|---|---:|---:|---:|---:|"]
    for r in raw:
        lines.append(f"| {r['variant']} | {r['I-AUROC']:.4f} | {r['I-AP']:.4f} | {r['P-AUROC']:.4f} | {r['PRO']:.4f} |")
    lines.extend(["", "## Grouped Gains vs XIV1", "", "| Variant | dI | dAP | dP | dPRO | G_image | G_pixel | G_balanced |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in grouped:
        lines.append(f"| {r['variant']} | {r['dI-AUROC']:.4f} | {r['dI-AP']:.4f} | {r['dP-AUROC']:.4f} | {r['dPRO']:.4f} | {r['G_image']:.4f} | {r['G_pixel']:.4f} | {r['G_balanced']:.4f} |")
    lines.extend(["", "## Gradient Timeline", "", "| T | Mean Cos | Frac < 0 | Frac < -0.1 |", "|---|---:|---:|---:|"])
    for r in timeline_summary:
        lines.append(f"| {r['timepoint']} | {r['mean_cos']:.4f} | {r['frac_negative']:.3f} | {r['frac_lt_minus_0p1']:.3f} |")
    lines.extend(["", "Mandatory stop: XIV-C was not started."])
    (OUT / "XIV_B_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log = ROOT / "EXPERIMENT_LOG.md"
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n\n## Phase XIV-B\n")
        f.write(f"- Variants: {', '.join(VARIANTS)}; D1 fold0 seed0; epochs=3.\n")
        f.write(f"- Parameter counts: {routing['parameter_counts']}.\n")
        f.write(f"- Routing proof: {routing['connectivity']}.\n")
        f.write(f"- Outcome: {primary}; secondary: {secondary}.\n")
    print(json.dumps({"primary_outcome": primary, "secondary": secondary, "report": str(OUT / "XIV_B_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
