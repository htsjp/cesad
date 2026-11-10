import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import load_dataset, write_csv
from phase_xa.official_task_forward import visual_outputs
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xiii_counterfactual/audit")
LAYERS = [12, 18, 24]
MAX_OPERATORS_PER_DIRECTION = 48
SEED = 0


def direction_cfg(direction):
    if direction == "D1":
        return "mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"
    return "visa", "data/VISA", "checkpoints/test_on_mvtec.pth"


def f(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value, nd=4):
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{nd}f}"


def l2(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-8 else float("nan")


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_ci(values, n=1000, seed=0):
    values = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bbox_from_mask(mask):
    m = (mask > 0.5).detach().cpu().numpy().squeeze()
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def tensor_blur(x, sigma=5):
    arr = x.detach().cpu().numpy()
    out = np.stack([gaussian_filter(arr[c], sigma=sigma) for c in range(arr.shape[0])], axis=0)
    return torch.from_numpy(out).to(x.device, dtype=x.dtype)


def feather_mask(mask, sigma=1.0):
    arr = mask.detach().cpu().numpy().astype(np.float32)
    soft = gaussian_filter(arr, sigma=sigma)
    soft = np.clip(soft, 0.0, 1.0)
    return torch.from_numpy(soft).to(mask.device, dtype=mask.dtype)


def style_match(content, host_region, mask):
    out = content.clone()
    m = mask > 0.05
    for c in range(content.shape[0]):
        vals = content[c][m]
        host_vals = host_region[c][m]
        if vals.numel() < 2 or host_vals.numel() < 2:
            continue
        out[c] = (content[c] - vals.mean()) / vals.std().clamp_min(1e-6) * host_vals.std().clamp_min(1e-6) + host_vals.mean()
    return out


def phase_shuffle(content, rng):
    arr = content.detach().cpu().numpy()
    out = []
    for c in range(arr.shape[0]):
        fft = np.fft.fft2(arr[c])
        mag = np.abs(fft)
        phase = rng.uniform(-np.pi, np.pi, size=fft.shape)
        out.append(np.real(np.fft.ifft2(mag * np.exp(1j * phase))))
    return torch.from_numpy(np.stack(out, axis=0)).to(content.device, dtype=content.dtype)


def crop_resize(tensor, bbox, mask_tensor=None, max_side=160):
    y0, y1, x0, x1 = bbox
    crop = tensor[:, y0:y1, x0:x1]
    if mask_tensor is None:
        mask = torch.ones((1, crop.shape[1], crop.shape[2]), device=tensor.device, dtype=tensor.dtype)
    else:
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask = mask_tensor[:, y0:y1, x0:x1].to(tensor.device, dtype=tensor.dtype)
    h, w = crop.shape[-2:]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
        crop = F.interpolate(crop.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(nh, nw), mode="nearest").squeeze(0)
    return crop, mask


def place_location(host_shape, patch_shape, op_id):
    _, h, w = host_shape
    ph, pw = patch_shape[-2:]
    if ph >= h or pw >= w:
        return None
    rng = random.Random(op_id)
    y = rng.randint(8, max(8, h - ph - 8))
    x = rng.randint(8, max(8, w - pw - 8))
    return y, x


def apply_operator(host, op, family, control, seed):
    host = host.clone()
    content = op["patch"].clone()
    mask = op["mask"].clone()
    loc = place_location(host.shape, content.shape, seed)
    if loc is None:
        return None, None, {"valid": False, "reason": "does_not_fit"}
    y, x = loc
    ph, pw = content.shape[-2:]
    region = host[:, y:y + ph, x:x + pw]
    hard = mask[0] > 0.5
    soft = feather_mask(mask[0], sigma=1.0)
    if control == "random_content":
        rng = torch.Generator(device=host.device).manual_seed(seed)
        content = torch.randn(content.shape, generator=rng, device=host.device, dtype=host.dtype) * region.std() + region.mean()
    elif control == "phase_shuffled":
        content = phase_shuffle(content, np.random.default_rng(seed))
    elif control == "mask_only_mild":
        blended = region * (1.0 + 0.12 * soft.unsqueeze(0))
        new_region = region * (1.0 - soft.unsqueeze(0)) + blended * soft.unsqueeze(0)
        host[:, y:y + ph, x:x + pw] = torch.clamp(new_region, -3.0, 3.0)
        out_mask = torch.zeros((host.shape[-2], host.shape[-1]), device=host.device)
        out_mask[y:y + ph, x:x + pw] = mask[0]
        clipped = float(((new_region <= -3.0) | (new_region >= 3.0)).float().mean().cpu())
        return host, out_mask, {"valid": True, "clipped_fraction": clipped, "y": y, "x": x, "h": ph, "w": pw}

    if family == "A":
        content = style_match(content, region, hard)
        alpha = 0.80
        blended = region * (1.0 - alpha * soft.unsqueeze(0)) + content * (alpha * soft.unsqueeze(0))
    else:
        low = tensor_blur(content, sigma=5)
        residual = content - low
        blended = region + residual * mask * 1.0
    host[:, y:y + ph, x:x + pw] = torch.clamp(blended, -3.0, 3.0)
    out_mask = torch.zeros((host.shape[-2], host.shape[-1]), device=host.device)
    out_mask[y:y + ph, x:x + pw] = mask[0]
    clipped = float(((blended <= -3.0) | (blended >= 3.0)).float().mean().cpu())
    return host, out_mask, {"valid": True, "clipped_fraction": clipped, "y": y, "x": x, "h": ph, "w": pw}


def layer_tokens(snapshot):
    if isinstance(snapshot, list):
        global_branch = snapshot[1].permute(1, 0, 2).float()[0]
        local_branch = snapshot[0].permute(1, 0, 2).float()[0]
    else:
        global_branch = snapshot.permute(1, 0, 2).float()[0]
        local_branch = global_branch
    return global_branch[0], local_branch[1:]


def response(model, image, mask):
    with torch.no_grad():
        out = visual_outputs(model, image.unsqueeze(0), adapter=None, grad=False, snapshot_mode="true_clone")
    feats = {}
    maps = {}
    profile = []
    for layer in LAYERS:
        cls, patches = layer_tokens(out["snapshots"][f"true_block{layer}"])
        cls = cls.detach().cpu().numpy()
        patches = patches.detach().cpu().numpy()
        feats[layer] = {"cls": cls, "patches": patches}
    return feats


def diff_response(clean, cf, mask):
    channels = {}
    profile = []
    roi_maps = {}
    inside_ratios = {}
    for layer in LAYERS:
        dg = cf[layer]["cls"] - clean[layer]["cls"]
        channels[f"global_l{layer}"] = dg.astype(np.float32)
        profile.append(float(np.linalg.norm(dg)))
        dp = np.linalg.norm(cf[layer]["patches"] - clean[layer]["patches"], axis=1)
        side = int(round(math.sqrt(len(dp))))
        m = F.interpolate(mask[None, None].float(), size=(side, side), mode="nearest").squeeze().cpu().numpy() > 0.5
        dmap = dp.reshape(side, side)
        inside = dmap[m]
        outside = dmap[~m]
        profile.append(float(np.mean(inside)) if len(inside) else 0.0)
        inside_ratios[layer] = float((np.mean(inside) if len(inside) else 0.0) / ((np.mean(outside) if len(outside) else 0.0) + 1e-8))
        ys, xs = np.where(m)
        if len(xs):
            roi = dmap[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        else:
            roi = dmap
        roi_t = torch.from_numpy(roi).float()[None, None]
        roi32 = F.interpolate(roi_t, size=(32, 32), mode="bilinear", align_corners=False).squeeze().numpy()
        roi32 = roi32 - roi32.mean()
        roi32 = roi32 / (np.linalg.norm(roi32) + 1e-8)
        roi_maps[layer] = roi32.astype(np.float32)
    profile = np.asarray(profile, dtype=np.float32)
    return {"channels": channels, "profile": profile, "roi_maps": roi_maps, "inside_ratios": inside_ratios, "score_delta": float(profile[[0, 2, 4]].mean() + profile[[1, 3, 5]].mean())}


def valid_anomaly(item):
    bbox = bbox_from_mask(item["img_mask"])
    if bbox is None:
        return False
    area = float((item["img_mask"] > 0.5).float().mean())
    return 0.001 <= area <= 0.30


def build_operators(dataset, categories, device):
    by_cat = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        cat = item["cls_name"]
        if cat in categories and int(item["anomaly"]) == 1 and valid_anomaly(item):
            by_cat[cat].append(idx)
    ops = []
    per_cat = max(1, MAX_OPERATORS_PER_DIRECTION // max(1, len(by_cat)))
    op_id = 0
    for cat in sorted(by_cat):
        for idx in by_cat[cat][:per_cat]:
            item = dataset[idx]
            bbox = bbox_from_mask(item["img_mask"])
            patch, mask = crop_resize(item["img"].to(device), bbox, item["img_mask"].to(device))
            area = float(mask.mean().cpu())
            if not (0.001 <= area <= 0.30):
                continue
            ops.append({"operator_id": f"op{op_id:04d}", "donor_idx": idx, "donor_category": cat, "patch": patch, "mask": mask, "mask_area": area})
            op_id += 1
    return ops[:MAX_OPERATORS_PER_DIRECTION]


def normal_hosts(dataset, categories):
    by_cat = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item["cls_name"] in categories and int(item["anomaly"]) == 0:
            by_cat[item["cls_name"]].append(idx)
    return by_cat


def choose_hosts(hosts, donor_cat, op_num):
    cats = [c for c in sorted(hosts) if c != donor_cat and hosts[c]]
    if len(cats) < 2:
        return None
    c1 = cats[op_num % len(cats)]
    c2 = cats[(op_num + 1) % len(cats)]
    return c1, hosts[c1][op_num % len(hosts[c1])], c2, hosts[c2][op_num % len(hosts[c2])]


def audit_direction(direction, device):
    dataset_name, data_path, checkpoint = direction_cfg(direction)
    model, *_ = load_components(checkpoint, device)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][0]
    categories = set(rotation["train_categories"])
    ops = build_operators(dataset, categories, device)
    hosts = normal_hosts(dataset, categories)
    clean_cache = {}
    cf_cache = {}
    operator_rows, quality_rows = [], []
    pair_records = []
    controls = ["real", "random_content", "phase_shuffled", "mask_only_mild"]
    families = ["A", "B"]
    for op_num, op in enumerate(tqdm(ops, desc=f"XIII-A {direction} operators")):
        chosen = choose_hosts(hosts, op["donor_category"], op_num)
        if not chosen:
            continue
        hcat_a, hidx_a, hcat_b, hidx_b = chosen
        operator_rows.append({
            "direction": direction,
            "operator_id": op["operator_id"],
            "donor_idx": op["donor_idx"],
            "donor_category": op["donor_category"],
            "host_a_idx": hidx_a,
            "host_a_category": hcat_a,
            "host_b_idx": hidx_b,
            "host_b_category": hcat_b,
            "mask_area": op["mask_area"],
        })
        for family in families:
            host_responses = {}
            for host_tag, hidx, hcat in [("A", hidx_a, hcat_a), ("B", hidx_b, hcat_b)]:
                host = dataset[hidx]["img"].to(device)
                if hidx not in clean_cache:
                    clean_cache[hidx] = response(model, host, None)
                for control in controls:
                    cf, mask, q = apply_operator(host, op, family, control if control != "real" else "real", seed=op_num * 100 + (0 if host_tag == "A" else 1))
                    q.update({"direction": direction, "operator_id": op["operator_id"], "family": family, "control": control, "host": host_tag, "host_category": hcat})
                    quality_rows.append(q)
                    if not q.get("valid") or q.get("clipped_fraction", 1.0) > 0.05:
                        continue
                    key = (family, control, host_tag)
                    cf_cache_key = (hidx, op["operator_id"], family, control)
                    if cf_cache_key not in cf_cache:
                        cf_cache[cf_cache_key] = diff_response(clean_cache[hidx], response(model, cf, mask), mask)
                    host_responses[key] = cf_cache[cf_cache_key]
            if ("A" not in [hcat_a, hcat_b]):
                pass
            for control in controls:
                ka = (family, control, "A")
                kb = (family, control, "B")
                if ka in host_responses and kb in host_responses:
                    pair_records.append({
                        "direction": direction,
                        "operator_id": op["operator_id"],
                        "family": family,
                        "control": control,
                        "donor_category": op["donor_category"],
                        "host_a_category": hcat_a,
                        "host_b_category": hcat_b,
                        "response_a": host_responses[ka],
                        "response_b": host_responses[kb],
                    })
    return operator_rows, quality_rows, pair_records


def summarize_pairs(records):
    rows_sim, rows_score, rows_top, rows_loc, rows_cat, rows_ctrl = [], [], [], [], [], []
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["direction"], r["family"], r["control"])].append(r)

    for key, rows in by_key.items():
        direction, family, control = key
        real_rows = rows
        wrong_rows = rows[1:] + rows[:1]
        for layer in LAYERS:
            vals, wrong, tops, wrong_tops = [], [], [], []
            loc = []
            for r, wr in zip(real_rows, wrong_rows):
                ca = r["response_a"]["channels"][f"global_l{layer}"]
                cb = r["response_b"]["channels"][f"global_l{layer}"]
                wb = wr["response_b"]["channels"][f"global_l{layer}"]
                vals.append(cosine(ca, cb))
                wrong.append(cosine(ca, wb))
                ta = r["response_a"]["roi_maps"][layer]
                tb = r["response_b"]["roi_maps"][layer]
                tw = wr["response_b"]["roi_maps"][layer]
                tops.append(cosine(ta, tb))
                wrong_tops.append(cosine(ta, tw))
                try:
                    ssim_val = float(ssim(ta, tb, data_range=float(tb.max() - tb.min() + 1e-8)))
                except Exception:
                    ssim_val = float("nan")
                loc.append({
                    "direction": direction, "family": family, "control": control, "operator_id": r["operator_id"],
                    "layer": layer, "inside_outside_a": r["response_a"]["inside_ratios"][layer],
                    "inside_outside_b": r["response_b"]["inside_ratios"][layer], "topology_ssim": ssim_val,
                })
            delta = np.asarray(vals) - np.asarray(wrong)
            lo, hi = bootstrap_ci(delta)
            rows_sim.append({
                "direction": direction, "family": family, "control": control, "channel": f"global_l{layer}",
                "matched_mean": float(np.nanmean(vals)), "wrong_mean": float(np.nanmean(wrong)),
                "delta_mean": float(np.nanmean(delta)), "delta_std": float(np.nanstd(delta)),
                "delta_median": float(np.nanmedian(delta)), "ci95_low": lo, "ci95_high": hi, "n": len(delta),
            })
            tdelta = np.asarray(tops) - np.asarray(wrong_tops)
            tlo, thi = bootstrap_ci(tdelta)
            rows_top.append({
                "direction": direction, "family": family, "control": control, "layer": layer,
                "matched_cosine": float(np.nanmean(tops)), "wrong_cosine": float(np.nanmean(wrong_tops)),
                "delta_mean": float(np.nanmean(tdelta)), "ci95_low": tlo, "ci95_high": thi, "n": len(tdelta),
            })
            rows_loc.extend(loc)

        profiles_a = np.stack([r["response_a"]["profile"] for r in real_rows], axis=0)
        profiles_b = np.stack([r["response_b"]["profile"] for r in real_rows], axis=0)
        profiles_w = np.stack([r["response_b"]["profile"] for r in wrong_rows], axis=0)
        pc = [cosine(a, b) for a, b in zip(profiles_a, profiles_b)]
        pw = [cosine(a, b) for a, b in zip(profiles_a, profiles_w)]
        delta = np.asarray(pc) - np.asarray(pw)
        lo, hi = bootstrap_ci(delta)
        rows_sim.append({
            "direction": direction, "family": family, "control": control, "channel": "profile_cosine",
            "matched_mean": float(np.nanmean(pc)), "wrong_mean": float(np.nanmean(pw)),
            "delta_mean": float(np.nanmean(delta)), "delta_std": float(np.nanstd(delta)),
            "delta_median": float(np.nanmedian(delta)), "ci95_low": lo, "ci95_high": hi, "n": len(delta),
        })
        scores_a = [r["response_a"]["score_delta"] for r in real_rows]
        scores_b = [r["response_b"]["score_delta"] for r in real_rows]
        scores_w = [r["response_b"]["score_delta"] for r in wrong_rows]
        rho = float(spearmanr(scores_a, scores_b).correlation) if len(scores_a) >= 3 else float("nan")
        wrong_rho = float(spearmanr(scores_a, scores_w).correlation) if len(scores_a) >= 3 else float("nan")
        rows_score.append({
            "direction": direction, "family": family, "control": control,
            "pearson": pearson(scores_a, scores_b), "spearman": rho,
            "wrong_pearson": pearson(scores_a, scores_w), "wrong_spearman": wrong_rho,
            "spearman_advantage": rho - wrong_rho if math.isfinite(rho) and math.isfinite(wrong_rho) else float("nan"),
            "n": len(scores_a),
        })
        by_cat = defaultdict(list)
        for r, wr in zip(real_rows, wrong_rows):
            d = cosine(r["response_a"]["profile"], r["response_b"]["profile"]) - cosine(r["response_a"]["profile"], wr["response_b"]["profile"])
            by_cat[r["donor_category"]].append(d)
        for cat, vals in by_cat.items():
            rows_cat.append({"direction": direction, "family": family, "control": control, "category": cat, "mean_profile_advantage": float(np.mean(vals)), "positive": float(np.mean(np.asarray(vals) > 0)), "n": len(vals)})

    # Real-vs-control comparison on primary profile and topology deltas.
    index = {(r["direction"], r["family"], r["control"], r["channel"]): r for r in rows_sim}
    for direction in sorted({r["direction"] for r in rows_sim}):
        for family in ["A", "B"]:
            real = index.get((direction, family, "real", "profile_cosine"))
            if not real:
                continue
            for control in ["random_content", "phase_shuffled", "mask_only_mild"]:
                ctrl = index.get((direction, family, control, "profile_cosine"))
                if ctrl:
                    rows_ctrl.append({
                        "direction": direction, "family": family, "metric": "profile_delta",
                        "control": control, "real_delta": real["delta_mean"], "control_delta": ctrl["delta_mean"],
                        "real_minus_control": real["delta_mean"] - ctrl["delta_mean"],
                    })
    return rows_sim, rows_score, rows_top, rows_loc, rows_cat, rows_ctrl


def decide(rows_sim, rows_score, rows_top, rows_cat, rows_ctrl):
    outcomes = {}
    for direction in ["D1", "D2"]:
        passed_channels = []
        for r in rows_sim:
            if r["direction"] == direction and r["control"] == "real" and r["delta_mean"] >= 0.08 and r["ci95_low"] > 0:
                passed_channels.append((r["family"], r["channel"], r["delta_mean"]))
        score_ok = False
        for r in rows_score:
            if r["direction"] == direction and r["control"] == "real" and r["spearman"] >= 0.40 and r["spearman_advantage"] >= 0.15:
                score_ok = True
        top_ok = any(r["direction"] == direction and r["control"] == "real" and r["delta_mean"] >= 0.08 for r in rows_top)
        random_ok = any(r["direction"] == direction and r["control"] == "random_content" and c["real_minus_control"] >= 0.05 for c in rows_ctrl if c["direction"] == direction and c["control"] == "random_content")
        cats = [r for r in rows_cat if r["direction"] == direction and r["control"] == "real"]
        cat_ok = False
        if cats:
            cat_ok = sum(1 for r in cats if r["mean_profile_advantage"] > 0) / len(cats) >= 0.50
        outcomes[direction] = {
            "passed_channel_count": len(passed_channels),
            "passed_channels": passed_channels,
            "score_ok": score_ok,
            "topology_ok": top_ok,
            "random_control_ok": random_ok,
            "category_distribution_ok": cat_ok,
            "pass": len(passed_channels) >= 2 and score_ok and top_ok and random_ok and cat_ok,
        }
    if outcomes["D1"]["pass"] and outcomes["D2"]["pass"]:
        label = "PASS_TO_PHASE_XIII_B"
    elif outcomes["D1"]["pass"] or outcomes["D2"]["pass"]:
        label = "XIII_A_DIRECTION_SPECIFIC_RELATION"
    else:
        any_random_explains = any(not o["random_control_ok"] and o["passed_channel_count"] >= 2 for o in outcomes.values())
        label = "XIII_A_RANDOM_PERTURBATION_EXPLAINS_EFFECT" if any_random_explains else "XIII_A_NO_RELATIONAL_STRUCTURE"
    return label, outcomes


def write_report(label, outcomes, rows_sim, rows_score, rows_top, rows_cat, rows_ctrl):
    lines = [
        "# Phase XIII-A Counterfactual Relation Audit",
        "",
        f"Outcome: **{label}**",
        "",
        "No adapters were trained. Target data was not accessed. Operators were generated from source fold0 only.",
        "",
        "## Direction Gates",
        "",
        "| Direction | Channel Count | Score OK | Topology OK | Random Control OK | Category Distributed | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for direction, o in outcomes.items():
        lines.append(f"| {direction} | {o['passed_channel_count']} | {o['score_ok']} | {o['topology_ok']} | {o['random_control_ok']} | {o['category_distribution_ok']} | {o['pass']} |")
    lines.extend(["", "## Matched vs Wrong Similarity", "", "| Direction | Family | Control | Channel | Matched | Wrong | Delta | CI Low | CI High | N |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in rows_sim:
        lines.append(f"| {r['direction']} | {r['family']} | {r['control']} | {r['channel']} | {fmt(r['matched_mean'])} | {fmt(r['wrong_mean'])} | {fmt(r['delta_mean'])} | {fmt(r['ci95_low'])} | {fmt(r['ci95_high'])} | {r['n']} |")
    lines.extend(["", "## Score Response Consistency", "", "| Direction | Family | Control | Spearman | Wrong Spearman | Advantage | Pearson | N |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in rows_score:
        lines.append(f"| {r['direction']} | {r['family']} | {r['control']} | {fmt(r['spearman'])} | {fmt(r['wrong_spearman'])} | {fmt(r['spearman_advantage'])} | {fmt(r['pearson'])} | {r['n']} |")
    lines.extend(["", "## Topology Similarity", "", "| Direction | Family | Control | Layer | Matched | Wrong | Delta | N |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in rows_top:
        lines.append(f"| {r['direction']} | {r['family']} | {r['control']} | {r['layer']} | {fmt(r['matched_cosine'])} | {fmt(r['wrong_cosine'])} | {fmt(r['delta_mean'])} | {r['n']} |")
    lines.extend(["", "## Control Comparison", "", "| Direction | Family | Metric | Control | Real Delta | Control Delta | Real-Control |", "|---|---|---|---|---:|---:|---:|"])
    for r in rows_ctrl:
        lines.append(f"| {r['direction']} | {r['family']} | {r['metric']} | {r['control']} | {fmt(r['real_delta'])} | {fmt(r['control_delta'])} | {fmt(r['real_minus_control'])} |")
    lines.extend([
        "",
        "## Required Answers",
        "",
        "Q1. The same real anomaly transformation is considered supported only where the direction gate passes; see Direction Gates and Matched vs Wrong tables.",
        "Q2. The strongest layers are the channels with the largest positive real-control delta in matched_wrong_similarity.csv.",
        "Q3. Global responses are reported for layers 12/18/24; local topology is reported separately by layer.",
        "Q4. Real defect content must exceed random content by >=0.05 on a primary metric; see control_comparison.csv.",
        "Q5. Category distribution is judged by per_category_relation.csv; at least 50% positive categories are required.",
        "Q6. Operator B vs A can be read from the family rows; B is preferred only if it preserves relation while reducing donor appearance dependence.",
        "",
        "Mandatory stop: Phase XIII-B training was not started.",
    ])
    (OUT / "COUNTERFACTUAL_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    all_ops, all_quality, all_records = [], [], []
    costs = []
    start_all = time.time()
    for direction in ["D1", "D2"]:
        start = time.time()
        ops, quality, records = audit_direction(direction, device)
        all_ops.extend(ops)
        all_quality.extend(quality)
        all_records.extend(records)
        costs.append({"direction": direction, "seconds": time.time() - start, "pairs": len(records), "target_accessed": False})
    rows_sim, rows_score, rows_top, rows_loc, rows_cat, rows_ctrl = summarize_pairs(all_records)
    label, outcomes = decide(rows_sim, rows_score, rows_top, rows_cat, rows_ctrl)
    write_csv(OUT / "operator_inventory.csv", all_ops)
    write_csv(OUT / "operator_quality.csv", all_quality)
    write_csv(OUT / "matched_wrong_similarity.csv", rows_sim)
    write_csv(OUT / "score_response_consistency.csv", rows_score)
    write_csv(OUT / "topology_similarity.csv", rows_top)
    write_csv(OUT / "localization_effect.csv", rows_loc)
    write_csv(OUT / "per_category_relation.csv", rows_cat)
    write_csv(OUT / "control_comparison.csv", rows_ctrl)
    write_report(label, outcomes, rows_sim, rows_score, rows_top, rows_cat, rows_ctrl)
    outcome = {
        "outcome": label,
        "direction_outcomes": outcomes,
        "costs": costs,
        "seconds_total": time.time() - start_all,
        "target_accessed": False,
        "next": "Do not start XIII-B automatically.",
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    with Path("EXPERIMENT_LOG.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Phase XIII-A Counterfactual Relation Audit\n\n"
            "- Command: `python -u experiments\\phase_xiii_counterfactual_audit.py`\n"
            "- Source folds: D1 source MVTec fold0; D2 source VisA fold0.\n"
            "- Operators: A style-matched transplant; B defect residual transplant; controls random/phase/mild/wrong-pair.\n"
            "- Training runs: `0`\n"
            "- Target data accessed: `False`\n"
            f"- Outcome: `{label}`\n"
        )
    print(json.dumps({"outcome": label, "report": str(OUT / "COUNTERFACTUAL_AUDIT_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
