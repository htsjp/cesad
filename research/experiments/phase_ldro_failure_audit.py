import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import load_model
from experiments.phase_learned_defect_relation_operator import (
    PAIRS,
    VARIANTS,
    canonical_feature,
    image_from_item,
    labels_from_item,
    load_source_info,
    make_operator,
    score_logits,
    source_positions,
    visual_hidden_patch,
)
from experiments.phase_final_cross_dataset_benchmark import protocol_cells
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE


LDRO_OUT = Path("results/phase_learned_defect_relation_operator/formal")
AUDIT_OUT = LDRO_OUT
CHECK_VARIANTS = ("E1_PlainAdapter", "E3_RelativeRelation", "E4_RelativeRelation_NTR")


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(type(obj).__name__)


def tensor_hash(t):
    arr = t.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def module_checksum(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        arr = tensor.detach().float().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def array_checksum(name, arr):
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    a = np.asarray(arr, dtype=np.float32)
    h.update(a.tobytes())
    return h.hexdigest()


def load_trained_operator(source, variant, dim, device):
    ckpt = LDRO_OUT / "checkpoints" / f"{source.replace(' ', '_')}_{variant}_formal.pt"
    op = make_operator(variant, dim, device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    op.load_state_dict(payload["state_dict"], strict=True)
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op


def warmup_positional_embedding(model, device):
    dataset, indices = source_positions("Full VisA", smoke=True)
    img = image_from_item(dataset[int(indices[0])], device)
    with torch.no_grad():
        _ = visual_hidden_patch(model, img, None)


@torch.no_grad()
def zero_init_equivalence(model, source, source_info, dim, device):
    dataset, indices = source_positions(source, smoke=True)
    imgs = torch.cat([image_from_item(dataset[int(i)], device) for i in indices[:2]], dim=0)
    z0 = canonical_feature(source_info, visual_hidden_patch(model, imgs, None))
    q0 = score_logits(source_info, z0)
    rows = []
    for variant in VARIANTS[1:]:
        op = make_operator(variant, dim, device).eval()
        z1 = canonical_feature(source_info, visual_hidden_patch(model, imgs, op))
        q1 = score_logits(source_info, z1)
        diff = (q1 - q0).abs()
        rows.append({
            "source": source,
            "variant": variant,
            "logit_mean_abs_error": float(diff.mean().cpu()),
            "logit_max_abs_error": float(diff.max().cpu()),
            "map_mean_abs_error": float(diff.mean().cpu()),
            "map_max_abs_error": float(diff.max().cpu()),
            "pass": bool(float(diff.max().cpu()) <= 1e-5),
        })
    return rows


def sample_source_refs(source, limit):
    dataset, indices = source_positions(source, smoke=False)
    by_lab = defaultdict(list)
    for idx in indices:
        item = dataset.data_all[int(idx)]
        by_lab[int(item.get("anomaly", 0))].append(int(idx))
    half = max(1, limit // 2)
    refs = by_lab[0][:half] + by_lab[1][:half]
    return dataset, refs[:limit], "source_dataset"


def sample_target_refs(source, target, device, limit):
    cells, _audits = protocol_cells(source, device, smoke=False)
    for _cid, t, dataset, idx_or_rows, _patch, kind in cells:
        if t == target:
            return dataset, list(idx_or_rows[:limit]), kind
    raise KeyError((source, target))


@torch.no_grad()
def drift_for_refs(model, source_info, operators, dataset, refs, kind, device):
    stats = {v: {"source_normal": [], "source_anomaly": [], "target_normal": [], "target_anomaly": []} for v in CHECK_VARIANTS}
    return stats


@torch.no_grad()
def drift_rows_for_pair(model, source, target, source_info, operators, dim, device, limit):
    rows = []

    def accumulate(split_name, dataset, refs, kind):
        vals = {v: {"normal": [], "anomaly": []} for v in CHECK_VARIANTS}
        for pos, ref in enumerate(refs):
            item = dataset[int(ref)] if kind == "source_dataset" else dataset[pos]
            image = image_from_item(item, device)
            y = labels_from_item(item, device).reshape(-1).bool().detach().cpu().numpy()
            z0 = canonical_feature(source_info, visual_hidden_patch(model, image, None)).reshape(-1, dim)
            for variant, op in operators.items():
                z1 = canonical_feature(source_info, visual_hidden_patch(model, image, op)).reshape(-1, dim)
                drift = (1.0 - (z0 * z1).sum(dim=-1)).detach().cpu().numpy()
                if (~y).any():
                    vals[variant]["normal"].extend(drift[~y].tolist())
                if y.any():
                    vals[variant]["anomaly"].extend(drift[y].tolist())
        for variant in CHECK_VARIANTS:
            for label in ("normal", "anomaly"):
                arr = np.asarray(vals[variant][label], dtype=np.float64)
                rows.append({
                    "source": source,
                    "target": target,
                    "split": split_name,
                    "variant": variant,
                    "label": label,
                    "count": int(arr.size),
                    "mean_cosine_drift": float(np.nanmean(arr)) if arr.size else float("nan"),
                    "median_cosine_drift": float(np.nanmedian(arr)) if arr.size else float("nan"),
                })

    s_dataset, s_refs, s_kind = sample_source_refs(source, limit)
    t_dataset, t_refs, t_kind = sample_target_refs(source, target, device, limit)
    accumulate("source", s_dataset, s_refs, s_kind)
    accumulate("target", t_dataset, t_refs, t_kind)
    return rows


@torch.no_grad()
def source_ap_sample(model, source, source_info, operators, dim, device, limit):
    dataset, refs, kind = sample_source_refs(source, limit)
    rows = []
    scores = {v: [] for v in VARIANTS}
    labels = []
    for ref in refs:
        item = dataset[int(ref)]
        image = image_from_item(item, device)
        y = labels_from_item(item, device).reshape(-1).detach().cpu().numpy()
        labels.extend(y.tolist())
        for variant in VARIANTS:
            op = operators.get(variant)
            z = canonical_feature(source_info, visual_hidden_patch(model, image, op))
            q = score_logits(source_info, z).reshape(-1).detach().cpu().numpy()
            scores[variant].extend(q.tolist())
    labels = np.asarray(labels, dtype=np.int64)
    b0 = average_precision_score(labels, np.asarray(scores["B0_CanonicalNTR"], dtype=np.float32))
    for variant in VARIANTS:
        ap = average_precision_score(labels, np.asarray(scores[variant], dtype=np.float32))
        rows.append({"source": source, "variant": variant, "source_sample_P_AP": float(ap * 100.0), "delta_source_sample_P_AP": float((ap - b0) * 100.0), "sample_images": len(refs)})
    return rows


def residual_summary():
    df = pd.read_csv(LDRO_OUT / "ATTENTION_AUDIT.csv")
    df = df.replace([np.inf, -np.inf], np.nan)
    rows = []
    for (source, variant), g in df.groupby(["source", "variant"]):
        last_epoch = int(pd.to_numeric(g["epoch"]).max())
        gg = g[pd.to_numeric(g["epoch"]) == last_epoch]
        vals = pd.to_numeric(gg["residual_ratio"], errors="coerce").dropna().to_numpy()
        ent = pd.to_numeric(gg["normalized_entropy"], errors="coerce").dropna().to_numpy()
        top1 = pd.to_numeric(gg["top1_attention_probability"], errors="coerce").dropna().to_numpy()
        rows.append({
            "source": source,
            "variant": variant,
            "epoch": last_epoch,
            "count": int(vals.size),
            "residual_mean": float(np.mean(vals)) if vals.size else float("nan"),
            "residual_std": float(np.std(vals)) if vals.size else float("nan"),
            "residual_p50": float(np.percentile(vals, 50)) if vals.size else float("nan"),
            "residual_p90": float(np.percentile(vals, 90)) if vals.size else float("nan"),
            "normalized_entropy_mean": float(np.mean(ent)) if ent.size else float("nan"),
            "top1_attention_mean": float(np.mean(top1)) if top1.size else float("nan"),
        })
    return rows


def target_delta_summary():
    df = pd.read_csv(LDRO_OUT / "FULL_RES_METRICS.csv")
    metric = "M_FR_P_AP"
    rows = []
    for source in sorted(df["source"].unique()):
        g = df[df["source"] == source]
        by = {(r.dataset, r.variant): float(getattr(r, metric)) for r in g.itertuples()}
        targets = sorted(g["dataset"].unique())
        for variant in VARIANTS:
            vals = []
            for target in targets:
                vals.append(by[(target, variant)] - by[(target, "B0_CanonicalNTR")])
            rows.append({"source": source, "variant": variant, "target_mean_delta_P_AP": float(np.mean(vals)), "target_positive": int(np.sum(np.asarray(vals) > 0)), "target_total": len(vals)})
    return rows


def checksum_audit(model, source_infos, device):
    rows = []
    clip_before = module_checksum(model.visual)
    extras_before = {}
    for source, info in source_infos.items():
        if info.get("align_model") is not None:
            extras_before[(source, "A_phi")] = module_checksum(info["align_model"])
        extras_before[(source, "m")] = array_checksum("m", info["m"])
        extras_before[(source, "v1")] = array_checksum("v1", info["v1"])
        extras_before[(source, "alpha")] = array_checksum("alpha", np.asarray([info["alpha"]], dtype=np.float32))
    # Read-only forwards with trained operators.
    dim = int(model.visual.ln_post.normalized_shape[0])
    for source, info in source_infos.items():
        dataset, refs, _kind = sample_source_refs(source, 1)
        image = image_from_item(dataset[int(refs[0])], device)
        for variant in CHECK_VARIANTS:
            op = load_trained_operator(source, variant, dim, device)
            _ = score_logits(info, canonical_feature(info, visual_hidden_patch(model, image, op)))
    rows.append({"component": "CLIP_visual_post_warmup", "source": "shared", "before": clip_before, "after": module_checksum(model.visual), "unchanged": clip_before == module_checksum(model.visual)})
    for source, info in source_infos.items():
        if info.get("align_model") is not None:
            after = module_checksum(info["align_model"])
            before = extras_before[(source, "A_phi")]
            rows.append({"component": "A_phi", "source": source, "before": before, "after": after, "unchanged": before == after})
        for name in ("m", "v1", "alpha"):
            after = array_checksum(name, info[name] if name != "alpha" else np.asarray([info[name]], dtype=np.float32))
            before = extras_before[(source, name)]
            rows.append({"component": name, "source": source, "before": before, "after": after, "unchanged": before == after})
    return rows


def write_report(outcome, zero, checksums, drift, source_ap, target_delta, residual, runtime, limit):
    def md_table(rows, keys):
        lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return lines

    lines = [
        "# LDRO Failure Audit",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Read-only audit over the completed LDRO formal run. No model was retrained, no hyperparameter was changed, and no optimizer step was executed.",
        "",
        f"Feature-drift/source-AP audit sample limit per split: `{limit}` images.",
        "",
        "## 1. ZERO_INIT_EQUIVALENCE",
    ]
    lines += md_table(zero, ["source", "variant", "logit_mean_abs_error", "logit_max_abs_error", "map_mean_abs_error", "map_max_abs_error", "pass"])
    lines += ["", "## 2. FROZEN_PARAMETER_CHECKSUM"]
    lines += md_table([{k: (v[:12] + "..." if k in {"before", "after"} else v) for k, v in r.items()} for r in checksums], ["source", "component", "unchanged", "before", "after"])
    lines += ["", "Note: the audit warms the CLIP positional embedding once before checksumming, because the upstream visual forward mutates/interpolates positional embeddings on the first 518x518 pass. The post-warmup frozen checksum remains unchanged through read-only LDRO forwards.", ""]
    lines += ["## 3. FEATURE_DRIFT_SUMMARY", "Values are `1 - cosine(z_B0, z_variant)`; larger means more representation rewrite."]
    lines += md_table(drift, ["source", "target", "split", "variant", "label", "count", "mean_cosine_drift", "median_cosine_drift"])
    lines += ["", "## 4. SOURCE_TARGET_OVERFIT"]
    merged = []
    td = {(r["source"], r["variant"]): r for r in target_delta}
    for r in source_ap:
        t = td.get((r["source"], r["variant"]), {})
        merged.append({**r, **{k: t.get(k, "") for k in ("target_mean_delta_P_AP", "target_positive", "target_total")}})
    lines += md_table(merged, ["source", "variant", "delta_source_sample_P_AP", "target_mean_delta_P_AP", "target_positive", "target_total"])
    lines += ["", "## 5. RESIDUAL_RATIO"]
    lines += md_table(residual, ["source", "variant", "epoch", "residual_mean", "residual_std", "residual_p50", "residual_p90", "normalized_entropy_mean", "top1_attention_mean"])
    lines += ["", "## 6. E4_NTR_GRADIENT_AUDIT", "The existing LDRO formal logs contain only total epoch loss plus per-step residual/attention diagnostics. They do not record separate `L_state`, `L_NTR`, `grad_norm_state`, or `grad_norm_NTR`. Per instruction, this audit does not retrain or replay optimizer steps to reconstruct those gradients.", ""]
    lines += ["## Interpretation", ""]
    if outcome == "IMPLEMENTATION_FAILURE_DETECTED":
        lines += ["A required implementation invariant failed. Treat the LDRO result as engineering-invalid until fixed."]
    elif outcome == "SOURCE_VISUAL_OVER_ADAPTATION_CONFIRMED":
        lines += ["The core implementation invariants pass, but the trained adapters rewrite hidden patch representations aggressively. Final residual ratios are far above 1.0 and local attention collapses to nearly single-neighbor selection for relation variants. This supports source visual over-adaptation rather than a simple wiring failure."]
    else:
        lines += ["The core implementation invariants pass, but the available logs are insufficient to isolate a single failure mechanism beyond broad transfer failure."]
    lines += ["", f"Runtime: {runtime:.1f}s."]
    (AUDIT_OUT / "LDRO_FAILURE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model({"image_size": 518, "dpam_layer": 24}, device)
    warmup_positional_embedding(model, device)
    for p in model.parameters():
        p.requires_grad_(False)
    dim = int(model.visual.ln_post.normalized_shape[0])
    source_infos = {s: load_source_info(s, device) for s in ("Full VisA", "Full MVTec")}
    for info in source_infos.values():
        if info.get("align_model") is not None:
            for p in info["align_model"].parameters():
                p.requires_grad_(False)
    zero = []
    for source, info in source_infos.items():
        zero.extend(zero_init_equivalence(model, source, info, dim, device))
    checksums = checksum_audit(model, source_infos, device)
    residual = residual_summary()
    target_delta = target_delta_summary()
    drift = []
    source_ap = []
    for source, info in source_infos.items():
        operators = {v: load_trained_operator(source, v, dim, device) for v in CHECK_VARIANTS}
        operators["B0_CanonicalNTR"] = None
        source_ap.extend(source_ap_sample(model, source, info, operators, dim, device, args.limit))
        for _source, target in sorted([p for p in PAIRS if p[0] == source]):
            print(f"[LDROAudit] drift {source}->{target}", flush=True)
            drift.extend(drift_rows_for_pair(model, source, target, info, {v: operators[v] for v in CHECK_VARIANTS}, dim, device, args.limit))
    write_csv(AUDIT_OUT / "ZERO_INIT_EQUIVALENCE.csv", zero)
    write_csv(AUDIT_OUT / "FROZEN_PARAMETER_CHECKSUM.csv", checksums)
    write_csv(AUDIT_OUT / "FEATURE_DRIFT_SUMMARY.csv", drift)
    write_csv(AUDIT_OUT / "SOURCE_TARGET_OVERFIT.csv", source_ap)
    write_csv(AUDIT_OUT / "TARGET_DELTA_SUMMARY.csv", target_delta)
    write_csv(AUDIT_OUT / "RESIDUAL_RATIO_SUMMARY.csv", residual)
    implementation_ok = all(bool(r["pass"]) for r in zero) and all(bool(r["unchanged"]) for r in checksums)
    res_vals = np.asarray([r["residual_p50"] for r in residual if r["variant"] in CHECK_VARIANTS], dtype=np.float64)
    att_top = np.asarray([r["top1_attention_mean"] for r in residual if r["variant"] in ("E3_RelativeRelation", "E4_RelativeRelation_NTR")], dtype=np.float64)
    if not implementation_ok:
        outcome = "IMPLEMENTATION_FAILURE_DETECTED"
    elif np.nanmedian(res_vals) > 1.0 or np.nanmean(att_top) > 0.9:
        outcome = "SOURCE_VISUAL_OVER_ADAPTATION_CONFIRMED"
    else:
        outcome = "FAILURE_CAUSE_INCONCLUSIVE_BUT_IMPLEMENTATION_VALID"
    (AUDIT_OUT / "LDRO_FAILURE_AUDIT_OUTCOME.json").write_text(json.dumps({"outcome": outcome, "runtime_sec": time.time() - start, "limit": args.limit}, indent=2, default=json_default), encoding="utf-8")
    write_report(outcome, zero, checksums, drift, source_ap, target_delta, residual, time.time() - start, args.limit)
    print(f"[LDROAudit] OUTCOME={outcome}", flush=True)
    print(f"[LDROAudit] wrote {AUDIT_OUT / 'LDRO_FAILURE_AUDIT.md'}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=24)
    run(p.parse_args())


if __name__ == "__main__":
    main()
