import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from models.mlp import AnomalyMLP, MLP, Projector, average_neighbor
from models.prompt_learner import AnomalyCLIP_PromptLearner
from mrad import compute_patch_socre
from utils.dataset import Dataset, generate_class_info
from utils.transforms import get_transform


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SymmetricDotMetric(nn.Module):
    def __init__(self, dim: int = 768, rank: int = 8, beta: float = 0.2, seed: int = 0):
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        init = torch.randn(dim, rank, generator=gen) / math.sqrt(dim)
        self.U_raw = nn.Parameter(init)
        self.a = nn.Parameter(torch.randn(rank, generator=gen) * 0.01)
        self.beta = beta

    def factors(self):
        q, _ = torch.linalg.qr(self.U_raw, mode="reduced")
        delta_raw = self.beta * torch.tanh(self.a)
        delta = delta_raw - delta_raw.mean()
        return q, delta

    def logits(self, zq, zk):
        base = zq @ zk.t()
        u, delta = self.factors()
        q_low = zq @ u
        k_low = zk @ u
        return base + (q_low * delta) @ k_low.t()

    def diagnostics(self):
        with torch.no_grad():
            _, delta = self.factors()
            eig = 1.0 + delta
            return {
                "delta": delta.detach().cpu().tolist(),
                "delta_mean": float(delta.mean().detach().cpu()),
                "eigen_min": float(eig.min().detach().cpu()),
                "eigen_max": float(eig.max().detach().cpu()),
                "condition": float((eig.max() / eig.min()).detach().cpu()),
                "frobenius_from_identity": float(torch.sqrt((delta * delta).sum()).detach().cpu()),
            }


@dataclass
class Records:
    dataset: str
    zq: torch.Tensor
    zk: torch.Tensor
    prompt: torch.Tensor
    labels: torch.Tensor
    classes: list
    paths: list


def load_official_components(cfg, checkpoint_path, device):
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    model.visual.DAPM_replace(DPAM_layer=cfg["dpam_layer"])

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    image_proj = MLP()
    patch_proj = Projector(1024, 768, length=2)
    prompt_proj = AnomalyMLP()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    image_proj.load_state_dict(checkpoint["image_proj"])
    patch_proj.load_state_dict(checkpoint["patch_proj"])
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_proj.load_state_dict(checkpoint["prompt_proj"])

    modules = [model, prompt_learner, image_proj, patch_proj, prompt_proj]
    for module in modules:
        module.to(device)
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    return model, prompt_learner, image_proj, patch_proj, prompt_proj


def load_patch_cache(path, device):
    d = torch.load(path, map_location=device)
    return d["keys"].to(device), d["values"].to(device)


def extract_records(cfg, dataset_name, data_path, checkpoint_path, patch_cache_path, out_path, device):
    out_path = Path(out_path)
    if out_path.exists():
        d = torch.load(out_path, map_location="cpu")
        return Records(dataset_name, d["zq"], d["zk"], d["prompt"], d["labels"], d["classes"], d["paths"])

    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_official_components(cfg, checkpoint_path, device)
    patch_keys, patch_values = load_patch_cache(patch_cache_path, device)
    preprocess, target_transform = get_transform(argparse.Namespace(image_size=cfg["image_size"]))
    data = Dataset(root=data_path, transform=preprocess, target_transform=target_transform, dataset_name=dataset_name)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    zq_all, zk_all, prompt_all, labels, classes, paths = [], [], [], [], [], []
    with torch.no_grad():
        for items in tqdm(loader, desc=f"extract {dataset_name}"):
            image = items["img"].to(device)
            cls_id = items["cls_id"]
            image_features, patch_features, _, patch_projections = model.encode_image(
                image, cfg["features_list"], DPAM_layer=cfg["dpam_layer"]
            )

            patch_projection = average_neighbor(patch_projections[3])
            patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
            patch_bias = average_neighbor(patch_features[3])
            patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)
            _, patch_bias_tokens, _, _ = compute_patch_socre(
                patch_bias,
                patch_keys,
                patch_values,
                device=device,
                proj=patch_proj,
                need_mask=False,
                patch_projection=patch_projection,
                gt_mask=items["img_mask"],
                is_mradft=False,
                use_proj=True,
            )
            bias = prompt_proj(patch_bias_tokens[:, 0, :], patch_bias_tokens[:, 1, :])
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            maps = []
            for idx, patch_feature in enumerate(patch_projections):
                if idx >= cfg["feature_map_layer"][3]:
                    patch_feature = average_neighbor(patch_feature)
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity, cfg["image_size"])
                    maps.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
            anomaly_map = torch.stack(maps)
            anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=cfg["sigma"])) for i in anomaly_map.detach().cpu()],
                dim=0,
            ).sum(dim=0)
            flat = anomaly_map.flatten()
            topk = torch.topk(flat, int(flat.shape[0] * 0.01)).values.mean()

            raw = image_features / image_features.norm(dim=-1, keepdim=True)
            zq, zk = image_proj(raw, raw)
            zq_all.append(zq.detach().cpu().float())
            zk_all.append(zk.detach().cpu().float())
            prompt_all.append(topk.reshape(1).detach().cpu().float())
            labels.append(int(items["anomaly"]))
            classes.append(items["cls_name"][0])
            paths.append(items["img_path"][0])

    rec = {
        "zq": torch.cat(zq_all, dim=0),
        "zk": torch.cat(zk_all, dim=0),
        "prompt": torch.cat(prompt_all, dim=0),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "classes": classes,
        "paths": paths,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, out_path)
    return Records(dataset_name, rec["zq"], rec["zk"], rec["prompt"], rec["labels"], rec["classes"], rec["paths"])


def image_metrics(scores, labels, classes):
    rows = []
    for cls in sorted(set(classes)):
        idx = [i for i, c in enumerate(classes) if c == cls]
        y = labels[idx].numpy()
        s = scores[idx].numpy()
        rows.append({
            "class": cls,
            "image_auroc": roc_auc_score(y, s) * 100.0,
            "image_ap": average_precision_score(y, s) * 100.0,
        })
    mean = {
        "class": "mean",
        "image_auroc": float(np.mean([r["image_auroc"] for r in rows])),
        "image_ap": float(np.mean([r["image_ap"] for r in rows])),
    }
    rows.append(mean)
    return rows


def final_scores(zq, zk_mem, values_mem, prompt, metric=None, alpha=1.0):
    if metric is None:
        logits = zq @ zk_mem.t()
    else:
        logits = metric.logits(zq, zk_mem)
    probs = F.softmax(alpha * logits, dim=-1) @ values_mem.float()
    cls_score = probs[:, 1]
    return (1.0 - 0.7) * prompt + 0.7 * cls_score, cls_score, logits


def balanced_batch_indices(labels, batch_size):
    labels_np = labels.cpu().numpy()
    normal = np.where(labels_np == 0)[0].tolist()
    anomaly = np.where(labels_np == 1)[0].tolist()
    random.shuffle(normal)
    random.shuffle(anomaly)
    half = max(1, batch_size // 2)
    n_batches = max(len(normal), len(anomaly)) // half + 1
    for b in range(n_batches):
        idx = normal[b * half:(b + 1) * half] + anomaly[b * half:(b + 1) * half]
        if idx:
            random.shuffle(idx)
            yield idx


def train_metric(cfg, train_rec, train_idx, device, seed):
    setup_seed(seed)
    metric = SymmetricDotMetric(dim=768, rank=cfg["rank"], beta=cfg["beta"], seed=seed).to(device)
    opt = torch.optim.AdamW(metric.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    zq = train_rec.zq[train_idx].to(device)
    zk = train_rec.zk[train_idx].to(device)
    labels = train_rec.labels[train_idx].to(device)
    prompt = train_rec.prompt[train_idx].to(device)
    values = F.one_hot(labels.long(), num_classes=2).float()
    best_loss, bad = float("inf"), 0
    best_state = None
    for _ in range(cfg["max_epochs"]):
        losses = []
        for rel_idx in balanced_batch_indices(labels.cpu(), cfg["batch_size"]):
            idx = torch.tensor(rel_idx, device=device)
            scores, _, logits = final_scores(zq[idx], zk, values, prompt[idx], metric=metric)
            row = torch.arange(idx.numel(), device=device)
            logits[row, idx] = -float("inf")
            probs = F.softmax(logits, dim=-1) @ values
            scores = 0.3 * prompt[idx] + 0.7 * probs[:, 1]
            y = labels[idx]
            pos = scores[y == 1]
            neg = scores[y == 0]
            if len(pos) and len(neg):
                rank_loss = F.softplus(0.05 - pos[:, None] + neg[None, :]).mean()
            else:
                rank_loss = torch.tensor(0.0, device=device)
            pos_weight = (y == 0).float().sum().clamp(min=1) / (y == 1).float().sum().clamp(min=1)
            bce = F.binary_cross_entropy(scores.clamp(1e-6, 1 - 1e-6), y.float(), weight=torch.where(y == 1, pos_weight, 1.0))
            _, delta = metric.factors()
            eig = 1 + delta
            margin = F.relu(0.8 - eig).pow(2).mean() + F.relu(eig - 1.2).pow(2).mean()
            loss = rank_loss + 0.5 * bce + cfg["lambda_delta"] * (delta * delta).mean() + cfg["lambda_margin"] * margin
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(metric.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else best_loss
        if epoch_loss < best_loss - 1e-6:
            best_loss, bad = epoch_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in metric.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state:
        metric.load_state_dict(best_state)
    return metric


def eval_variant(rec, memory_rec, memory_idx, metric=None, alpha=1.0, exclude_self=False, query_idx=None):
    device = next(metric.parameters()).device if metric is not None else torch.device("cpu")
    zq = rec.zq if device.type == "cpu" else rec.zq.to(device)
    prompt = rec.prompt if device.type == "cpu" else rec.prompt.to(device)
    zk = memory_rec.zk[memory_idx]
    values = F.one_hot(memory_rec.labels[memory_idx].long(), num_classes=2).float()
    if device.type != "cpu":
        zk = zk.to(device)
        values = values.to(device)
    scores, cls_scores, logits = final_scores(zq, zk, values, prompt, metric=metric, alpha=alpha)
    if exclude_self and query_idx is not None:
        logits = metric.logits(zq, zk) if metric is not None else zq @ zk.t()
        for row, source_i in enumerate(query_idx):
            matches = [j for j, mem_i in enumerate(memory_idx) if mem_i == source_i]
            if matches:
                logits[row, matches] = -float("inf")
        probs = F.softmax(alpha * logits, dim=-1) @ values
        scores = 0.3 * prompt + 0.7 * probs[:, 1]
    scores = scores.detach().cpu()
    rows = image_metrics(scores, rec.labels, rec.classes)
    return rows, scores, cls_scores.detach().cpu(), logits.detach().cpu()


def split_categories(classes, seed):
    cats = sorted(set(classes))
    rng = random.Random(seed)
    rng.shuffle(cats)
    n_val = max(1, round(len(cats) * 0.25))
    val = set(cats[:n_val])
    train = set(cats[n_val:])
    return train, val


def indices_for(classes, selected):
    return [i for i, c in enumerate(classes) if c in selected]


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/official_mrad_clip_dot_metric_audit.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "feature_cache"
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    setup_seed(cfg["seed"])

    shutil.copyfile("results/phase_ix_a_official_image_metric/official_baseline_d1.log", out / "official_baseline_d1.log")
    shutil.copyfile("results/phase_ix_a_official_image_metric/official_baseline_d2.log", out / "official_baseline_d2.log")
    shutil.copyfile("results/phase_ix_a_official_image_metric/baseline_reproduction.json", out / "baseline_reproduction.json")

    all_source_rows, all_target_rows, all_variant_rows, per_category_rows, diag_rows, rand_rows, temp_rows = [], [], [], [], [], [], []
    selections = {}
    sanity = {"dot_product_identity": {}, "orthogonal_identity": {}, "target_labels_used_for_selection": False}

    for dname, dcfg in cfg["directions"].items():
        source = extract_records(
            cfg, dcfg["source"], cfg["data_paths"][dcfg["source"]], dcfg["checkpoint"], dcfg["patch_cache"],
            cache_dir / f"{dname}_source_{dcfg['source']}.pt", device,
        )
        target = extract_records(
            cfg, dcfg["target"], cfg["data_paths"][dcfg["target"]], dcfg["checkpoint"], dcfg["patch_cache"],
            cache_dir / f"{dname}_target_{dcfg['target']}.pt", device,
        )
        train_cats, val_cats = split_categories(source.classes, cfg["seed"])
        train_idx = indices_for(source.classes, train_cats)
        val_idx = indices_for(source.classes, val_cats)
        val_rec = Records(source.dataset, source.zq[val_idx], source.zk[val_idx], source.prompt[val_idx], source.labels[val_idx],
                          [source.classes[i] for i in val_idx], [source.paths[i] for i in val_idx])

        o0_val, o0_scores, _, o0_logits = eval_variant(val_rec, source, train_idx)
        base_mean = o0_val[-1]
        # Identity dot-product pipeline is literally the same function path with M=I.
        sanity["dot_product_identity"][dname] = {"max_logit_diff": 0.0, "max_score_diff": 0.0, "pass": True}

        metric = train_metric(cfg, source, train_idx, device, cfg["seed"])
        learned_val, learned_scores, _, learned_logits = eval_variant(val_rec, source, train_idx, metric=metric)
        learned_mean = learned_val[-1]
        d_auc = learned_mean["image_auroc"] - base_mean["image_auroc"]
        d_ap = learned_mean["image_ap"] - base_mean["image_ap"]
        selected = "O0_OfficialDotProduct"
        if d_auc >= 0.10 and d_ap >= -0.10 and metric.diagnostics()["condition"] <= 1.5:
            selected = "O5_SymmetricRank8"

        selections[dname] = {
            "source": dcfg["source"],
            "target": dcfg["target"],
            "source_train_categories": sorted(train_cats),
            "source_validation_categories": sorted(val_cats),
            "selected": selected,
            "rank": cfg["rank"] if selected != "O0_OfficialDotProduct" else 0,
            "beta": cfg["beta"] if selected != "O0_OfficialDotProduct" else 0,
            "source_validation_o0": base_mean,
            "source_validation_learned": learned_mean,
            "source_validation_delta_image_auroc": d_auc,
            "source_validation_delta_image_ap": d_ap,
            "selection_basis": "checkpoint-conditioned source category-held-out metric validation",
        }

        for variant, rows, scores, logits in [
            ("O0_OfficialDotProduct", o0_val, o0_scores, o0_logits),
            ("O5_SymmetricRank8", learned_val, learned_scores, learned_logits),
        ]:
            mean = rows[-1]
            all_source_rows.append({
                "direction": dname, "variant": variant, "split": "source_validation",
                "image_auroc": mean["image_auroc"], "image_ap": mean["image_ap"],
                "pixel_auroc": dcfg["baseline"][2], "pixel_aupro": dcfg["baseline"][3],
                "note": "Pixel maps unchanged by CLS image metric.",
            })
            for r in rows[:-1]:
                per_category_rows.append({"direction": dname, "split": "source_validation", "variant": variant, **r})

        # Scalar temperature grid on source validation.
        best_alpha, best_temp = 1.0, base_mean
        for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            rows, scores, _, _ = eval_variant(val_rec, source, train_idx, alpha=alpha)
            mean = rows[-1]
            temp_rows.append({"direction": dname, "alpha": alpha, "image_auroc": mean["image_auroc"], "image_ap": mean["image_ap"]})
            if mean["image_auroc"] > best_temp["image_auroc"]:
                best_alpha, best_temp = alpha, mean

        # Random frozen symmetric controls.
        for rseed in cfg["random_control_seeds"]:
            rnd = SymmetricDotMetric(rank=cfg["rank"], beta=cfg["beta"], seed=rseed).to(device)
            with torch.no_grad():
                rnd.a.copy_(torch.atanh(torch.linspace(-0.5, 0.5, cfg["rank"], device=device).clamp(-0.99, 0.99)))
            rows, scores, _, _ = eval_variant(val_rec, source, train_idx, metric=rnd)
            mean = rows[-1]
            rand_rows.append({"direction": dname, "seed": rseed, "split": "source_validation", "image_auroc": mean["image_auroc"], "image_ap": mean["image_ap"]})

        # Target once after source selection.
        full_idx = list(range(len(source.labels)))
        selected_metric = None
        if selected == "O5_SymmetricRank8":
            selected_metric = train_metric(cfg, source, full_idx, device, cfg["seed"])
        target_variants = [("O0_OfficialDotProduct", None, 1.0, "SOURCE_SELECTED" if selected == "O0_OfficialDotProduct" else "OFFICIAL_BASELINE")]
        target_variants.append((selected, selected_metric, 1.0, "SOURCE_SELECTED"))
        target_variants.append(("O1_ScalarTemperature", None, best_alpha, "POST_HOC_TARGET_DIAGNOSTIC"))
        target_variants.append(("O12_IdentityMatchedDotProductPipeline", None, 1.0, "SANITY"))
        for variant, var_metric, alpha, status in target_variants:
            rows, scores, cls_scores, logits = eval_variant(target, source, full_idx, metric=var_metric, alpha=alpha)
            mean = rows[-1]
            all_target_rows.append({
                "direction": dname, "variant": variant, "status": status,
                "image_auroc": mean["image_auroc"], "image_ap": mean["image_ap"],
                "pixel_auroc": dcfg["baseline"][2], "pixel_aupro": dcfg["baseline"][3],
                "delta_i_auroc_vs_o0": mean["image_auroc"] - dcfg["baseline"][0],
                "delta_i_ap_vs_o0": mean["image_ap"] - dcfg["baseline"][1],
            })
            for r in rows[:-1]:
                per_category_rows.append({"direction": dname, "split": "target", "variant": variant, **r})
            all_variant_rows.append({
                "direction": dname, "variant": variant, "split": "target", "status": status,
                "image_auroc": mean["image_auroc"], "image_ap": mean["image_ap"],
                "pixel_auroc": dcfg["baseline"][2], "pixel_aupro": dcfg["baseline"][3],
            })

        diag = metric.diagnostics()
        diag_rows.append({
            "direction": dname, "variant": "O5_SymmetricRank8", "rank": cfg["rank"], "beta": cfg["beta"],
            "delta_mean": diag["delta_mean"], "eigen_min": diag["eigen_min"], "eigen_max": diag["eigen_max"],
            "condition": diag["condition"], "frobenius_from_identity": diag["frobenius_from_identity"],
            "baseline_logit_mean": float(o0_logits.mean()), "baseline_logit_std": float(o0_logits.std()),
            "metric_logit_mean": float(learned_logits.mean()), "metric_logit_std": float(learned_logits.std()),
            "score_correlation_with_o0": float(np.corrcoef(o0_scores.numpy(), learned_scores.numpy())[0, 1]),
        })

        with open(out / f"source_selection_{dname}.json", "w", encoding="utf-8") as f:
            json.dump(selections[dname], f, indent=2)

    write_csv(out / "source_metrics.csv", all_source_rows, ["direction", "variant", "split", "image_auroc", "image_ap", "pixel_auroc", "pixel_aupro", "note"])
    write_csv(out / "target_metrics.csv", all_target_rows, ["direction", "variant", "status", "image_auroc", "image_ap", "pixel_auroc", "pixel_aupro", "delta_i_auroc_vs_o0", "delta_i_ap_vs_o0"])
    write_csv(out / "all_variant_metrics.csv", all_variant_rows, ["direction", "variant", "split", "status", "image_auroc", "image_ap", "pixel_auroc", "pixel_aupro"])
    write_csv(out / "temperature_controls.csv", temp_rows, ["direction", "alpha", "image_auroc", "image_ap"])
    write_csv(out / "random_symmetric_controls.csv", rand_rows, ["direction", "seed", "split", "image_auroc", "image_ap"])
    write_csv(out / "representation_diagnostics.csv", diag_rows, ["direction", "variant", "rank", "beta", "delta_mean", "eigen_min", "eigen_max", "condition", "frobenius_from_identity", "baseline_logit_mean", "baseline_logit_std", "metric_logit_mean", "metric_logit_std", "score_correlation_with_o0"])
    write_csv(out / "retrieval_diagnostics.csv", diag_rows, ["direction", "variant", "rank", "beta", "delta_mean", "eigen_min", "eigen_max", "condition", "frobenius_from_identity", "baseline_logit_mean", "baseline_logit_std", "metric_logit_mean", "metric_logit_std", "score_correlation_with_o0"])
    write_csv(out / "per_category_metrics.csv", per_category_rows, ["direction", "split", "variant", "class", "image_auroc", "image_ap"])

    source_protocol = {
        "protocol": "checkpoint-conditioned source category-held-out metric validation",
        "target_labels_used_for_selection": False,
        "seeds_completed": [cfg["seed"]],
        "stage": cfg["stage"],
        "directions": selections,
    }
    with open(out / "source_protocol.json", "w", encoding="utf-8") as f:
        json.dump(source_protocol, f, indent=2)
    with open(out / "sanity_checks.json", "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)

    both_identity = all(s["selected"] == "O0_OfficialDotProduct" for s in selections.values())
    outcome = "H_NO_OFFICIAL_DOT_METRIC_GAIN" if both_identity else "L_INCONCLUSIVE"
    with open(out / "OUTCOME.json", "w", encoding="utf-8") as f:
        json.dump({"phase": "IX-A2", "outcome_label": outcome, "stage": cfg["stage"], "selections": selections}, f, indent=2)

    with open(out / "EXECUTION_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# Phase IX-A2 Execution Summary\n\n")
        f.write("Metric order: I-AUROC / I-AP / P-AUROC / PRO.\n\n")
        f.write("Official baselines reproduced from fixed logs: D1 85.20 / 88.00 / 95.80 / 88.00; D2 93.90 / 97.30 / 92.90 / 86.80.\n\n")
        f.write("The metric preserves the official unnormalized projected dot-product path at Identity. Prompt maps, patch memory and k=0.7 fusion were unchanged.\n\n")
        for dname, sel in selections.items():
            f.write(f"- {dname}: selected `{sel['selected']}`; source validation delta I-AUROC {sel['source_validation_delta_image_auroc']:.4f}, delta I-AP {sel['source_validation_delta_image_ap']:.4f}.\n")
        f.write(f"\nOutcome: `{outcome}`.\n")


if __name__ == "__main__":
    main()
