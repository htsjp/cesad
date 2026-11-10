import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import load_model
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_morphology_balanced_source_training import source_dataset
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import evaluate_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, patch_labels
from models.mlp import average_neighbor
from phase_x.gradient_visual_forward import forward_visual_with_module
from phase_x.local_relation_operator import LocalRelationOperator, PlainTokenAdapter


PAIRS = {
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
}
VARIANTS = (
    "B0_CanonicalNTR",
    "E1_PlainAdapter",
    "E2_AbsoluteLocalAttention",
    "E3_RelativeRelation",
    "E4_RelativeRelation_NTR",
)
EPOCHS = 10
LR = 0.001
TOP_NEG_FRAC = 0.10
INSERTION_LAYER = 20
RETURN_LAYERS = [24]
IMAGE_BATCH_SIZE = 1


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def fnum(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return default


def out_dir(args):
    return Path("results/phase_learned_defect_relation_operator") / ("smoke" if args.smoke else "formal")


def item_at(dataset, ref, kind, pos):
    return dataset[int(ref)] if kind == "source_dataset" else dataset[pos]


def meta_from_item(item, ref, kind, target):
    if kind == "source_dataset":
        return {
            "category": str(item.get("cls_name", target)),
            "label": int(item.get("anomaly", 0)),
            "mask": (np.squeeze(item["img_mask"].detach().cpu().numpy()) > 0.5).astype(np.uint8),
        }
    row = dict(ref)
    return {
        "category": str(row.get("category", item.get("cls_name", target))),
        "label": int(row.get("label", item.get("anomaly", 0))),
        "mask": (np.squeeze(item["img_mask"].detach().cpu().numpy()) > 0.5).astype(np.uint8),
    }


def image_from_item(item, device):
    return item["img"].unsqueeze(0).to(device)


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def load_source_info(source, device):
    info = load_visa_source(device) if source == "Full VisA" else load_mvtec_source()
    if info.get("align_model") is not None:
        freeze(info["align_model"])
    return info


def make_operator(variant, dim, device):
    torch.manual_seed(SEED)
    if variant == "E1_PlainAdapter":
        # Hidden=65 keeps the plain capacity control within ~1% of the local
        # relation operators for ViT-L hidden dim 1024.
        return PlainTokenAdapter(dim=dim, hidden=65).to(device)
    if variant == "E2_AbsoluteLocalAttention":
        return LocalRelationOperator(dim=dim, side=SIDE, hidden=32, relation_dim=32, mode="absolute").to(device)
    if variant in {"E3_RelativeRelation", "E4_RelativeRelation_NTR"}:
        return LocalRelationOperator(dim=dim, side=SIDE, hidden=32, relation_dim=32, mode="relative").to(device)
    if variant == "B0_CanonicalNTR":
        return None
    raise ValueError(variant)


def visual_hidden_patch(model, image, operator=None):
    outputs = forward_visual_with_module(
        model.visual,
        image.type(model.dtype),
        insertion_layer=INSERTION_LAYER,
        inserted_module=operator,
        return_layers=RETURN_LAYERS,
        dpam_layer=24,
        ori_patch=False,
        proj_use=True,
        ffn=False,
    )
    patch = average_neighbor(outputs[1][-1]).float()
    return F.normalize(patch, dim=-1).reshape(image.shape[0], SIDE, SIDE, -1)


def canonical_feature(source_info, z):
    if not source_info["use_align"]:
        return z
    q = torch.tensor(source_info["q_pc1"], dtype=torch.float32, device=z.device)
    flat = z.reshape(-1, z.shape[-1])
    flat = flat - (flat @ q).unsqueeze(-1) * q.unsqueeze(0)
    flat = F.normalize(flat, dim=-1)
    aligned = source_info["align_model"].adapt(flat)
    return F.normalize(aligned, dim=-1).reshape_as(z)


def score_logits(source_info, z):
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=z.device)
    v = torch.tensor(source_info["v1"], dtype=torch.float32, device=z.device)
    return contrast_logits(z.reshape(-1, z.shape[-1]), m, float(source_info["alpha"]), v).reshape(z.shape[0], SIDE, SIDE)


def labels_from_item(item, device):
    return torch.tensor(patch_labels(item).reshape(SIDE, SIDE), dtype=torch.float32, device=device)


def top_hard_negative(logits, y):
    neg = torch.nonzero(y.reshape(-1) < 0.5, as_tuple=False).flatten()
    k = max(1, int(np.ceil(float(len(neg)) * TOP_NEG_FRAC)))
    return neg[torch.topk(logits.reshape(-1).detach()[neg], k=k, largest=True).indices]


def normal_bce(logits_flat, index):
    return F.binary_cross_entropy_with_logits(logits_flat[index], torch.zeros_like(logits_flat[index]))


def source_positions(source_name, smoke):
    dataset, indices, _patch = source_dataset(source_name, smoke=False)
    by_cat = defaultdict(lambda: {0: [], 1: []})
    for pos, idx in enumerate(indices):
        item = dataset.data_all[int(idx)]
        by_cat[str(item["cls_name"])][int(item["anomaly"])].append(int(idx))
    if smoke:
        chosen = []
        for cat in sorted(by_cat)[:3]:
            chosen.extend(by_cat[cat][0][:1])
            chosen.extend(by_cat[cat][1][:1])
        return dataset, chosen
    chosen = []
    for cat in sorted(by_cat):
        chosen.extend(by_cat[cat][0])
        chosen.extend(by_cat[cat][1])
    return dataset, chosen


def sample_batch(indices, dataset, rng, smoke, batch_images):
    if smoke:
        n = min(2, len(indices), int(batch_images))
    else:
        n = min(int(batch_images), len(indices))
    chosen = rng.choice(len(indices), size=n, replace=len(indices) < n)
    return [dataset[int(indices[i])] for i in chosen]


def train_variant(out, source_name, variant, model, source_info, dim, device, smoke, batch_images):
    if variant == "B0_CanonicalNTR":
        return None, [], [], True
    suffix = "smoke" if smoke else "formal"
    ckpt = out / "checkpoints" / f"{source_name.replace(' ', '_')}_{variant}_{suffix}.pt"
    if ckpt.exists() and not smoke:
        op = make_operator(variant, dim, device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        op.load_state_dict(payload["state_dict"], strict=True)
        return op.eval(), payload.get("training_log", []), payload.get("attention_audit", []), True
    print(f"[LDRO] train {source_name} {variant}", flush=True)
    op = make_operator(variant, dim, device)
    opt = torch.optim.AdamW(op.parameters(), lr=LR, weight_decay=0.0)
    dataset, indices = source_positions(source_name, smoke)
    rng = np.random.default_rng(SEED)
    steps_per_epoch = 2 if smoke else max(1, int(np.ceil(len(indices) / float(batch_images))))
    epochs = 1 if smoke else EPOCHS
    log, attn_rows = [], []
    for epoch in range(1, epochs + 1):
        losses = []
        for step in range(steps_per_epoch):
            items = sample_batch(indices, dataset, rng, smoke, batch_images)
            imgs = torch.cat([image_from_item(it, device) for it in items], dim=0)
            ys = torch.stack([labels_from_item(it, device) for it in items], dim=0)
            z = canonical_feature(source_info, visual_hidden_patch(model, imgs, op))
            logits = score_logits(source_info, z)
            loss_task, _bce, _dice = balanced_loss(logits.reshape(-1), ys.reshape(-1))
            loss = loss_task
            if variant == "E4_RelativeRelation_NTR":
                hard = top_hard_negative(logits, ys)
                loss = loss + normal_bce(logits.reshape(-1), hard)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if torch.cuda.is_available() and step % 25 == 0:
                torch.cuda.empty_cache()
            losses.append(float(loss.detach().cpu()))
            stats = getattr(op, "last_stats", {})
            if stats:
                def stat(name):
                    val = stats.get(name)
                    if torch.is_tensor(val):
                        return float(val.float().mean().detach().cpu())
                    return float(val) if val == val else float("nan")
                attn_rows.append({
                    "source": source_name,
                    "variant": variant,
                    "epoch": epoch,
                    "step": step,
                    "mean_entropy": stat("mean_entropy"),
                    "normalized_entropy": stat("normalized_entropy"),
                    "top1_attention_probability": stat("top1_attention_probability"),
                    "effective_neighbor_count": stat("effective_neighbor_count"),
                    "residual_ratio": stat("residual_ratio"),
                })
        log.append({"source": source_name, "variant": variant, "epoch": epoch, "loss": float(np.mean(losses)), "lr": LR})
        print(f"[LDRO] {source_name} {variant} epoch={epoch}/{epochs} loss={log[-1]['loss']:.4f}", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": op.state_dict(), "training_log": log, "attention_audit": attn_rows}, ckpt)
    return op.eval(), log, attn_rows, False


@torch.no_grad()
def score_dataset(out, source_name, target, dataset, idx_or_rows, kind, model, source_info, operators, device, smoke):
    n = min(8, len(idx_or_rows)) if smoke else len(idx_or_rows)
    maps = {v: [] for v in VARIANTS}
    image_scores = {v: [] for v in VARIANTS}
    metas = []
    for i in range(n):
        ref = idx_or_rows[i]
        item = item_at(dataset, ref, kind, i)
        metas.append(meta_from_item(item, ref, kind, target))
        image = image_from_item(item, device)
        for variant in VARIANTS:
            z = canonical_feature(source_info, visual_hidden_patch(model, image, operators.get(variant)))
            smap = score_logits(source_info, z).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            maps[variant].append(smap)
            image_scores[variant].append(top1_image_score(smap))
        if i % 25 == 0:
            print(f"[LDRO] eval {source_name}->{target} image {i}/{n}", flush=True)
    old = tuple(getattr(__import__("experiments.phase_parameter_free_local_zoom_pyramid_fullres", fromlist=["VARIANTS"]), "VARIANTS"))
    import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
    try:
        lzp.VARIANTS = VARIANTS
        return evaluate_maps(source_name, target, {k: maps[k] for k in VARIANTS}, metas)
    finally:
        lzp.VARIANTS = old


def parameter_audit(source, operators):
    rows = []
    for variant, op in operators.items():
        trainable = 0 if op is None else sum(p.numel() for p in op.parameters() if p.requires_grad)
        total = 0 if op is None else sum(p.numel() for p in op.parameters())
        rows.append({"source": source, "variant": variant, "trainable_parameters": trainable, "total_parameters": total, "trainable_ratio": float(trainable / total) if total else 0.0})
    return rows


def block_audit(model):
    n = len(model.visual.transformer.resblocks)
    return [
        "# Relation Insertion Audit",
        "",
        f"Visual transformer blocks: {n}.",
        "Block numbering in this script is 1-based for protocol reporting and 0-based inside Python lists.",
        f"`INSERTION_LAYER={INSERTION_LAYER}` registers a forward hook on `resblocks[{INSERTION_LAYER - 1}]`, so LDRO is applied after block {INSERTION_LAYER} and before blocks {INSERTION_LAYER + 1}-{n}.",
        "For a 24-block ViT this is exactly before the final four blocks.",
        "If the visual path returns `[active, original]` DPAM outputs, LDRO modifies only the active patch path and leaves the original/CLS path unchanged.",
        "Local neighborhood: fixed 3x3 legal neighbors, center excluded, no fake padding token.",
    ]


def zero_init_parity(model, source_info, dim, device):
    dataset, indices = source_positions(source_info["source"], smoke=True)
    item = dataset[int(indices[0])]
    img = image_from_item(item, device)
    with torch.no_grad():
        z0 = canonical_feature(source_info, visual_hidden_patch(model, img, None))
        rows = []
        for variant in ("E1_PlainAdapter", "E2_AbsoluteLocalAttention", "E3_RelativeRelation", "E4_RelativeRelation_NTR"):
            op = make_operator(variant, dim, device).eval()
            z1 = canonical_feature(source_info, visual_hidden_patch(model, img, op))
            rows.append({"source": source_info["source"], "variant": variant, "max_abs_feature_error": float((z1 - z0).abs().max().cpu()), "mean_abs_feature_error": float((z1 - z0).abs().mean().cpu())})
    return rows


def b0_reference_provenance(out):
    text = "\n".join([
        "# B0 Reference Provenance",
        "",
        "B0 is reconstructed from the current frozen Canonical Two-State + NTR state geometry loaded by `phase_final_cross_dataset_benchmark.load_visa_source/load_mvtec_source`.",
        "The script does not use the older mixed historical registry as a formal reproduction reference.",
        "Full-run B0 comparison should be made against current full-resolution `M_FR` benchmark rows.",
        "Smoke mode only checks execution/parity on tiny samples and is not a metric reproduction run.",
        "",
    ])
    (out / "B0_REFERENCE_PROVENANCE.md").write_text(text, encoding="utf-8")


def run(args):
    if not args.smoke and not args.full:
        raise SystemExit("Refusing to run by default. Use --smoke for wiring test or --full for the long formal experiment.")
    start = time.time()
    out = out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = load_model({"image_size": 518, "dpam_layer": 24}, device)
    freeze(model)
    dim = int(model.visual.ln_post.normalized_shape[0])
    (out / "RELATION_INSERTION_AUDIT.md").write_text("\n".join(block_audit(model)) + "\n", encoding="utf-8")
    b0_reference_provenance(out)

    raw, percat, params, train_log, attn, parity = [], [], [], [], [], []
    sources = ("Full VisA", "Full MVTec") if args.full else ("Full VisA",)
    for source in sources:
        source_info = load_source_info(source, device)
        parity.extend(zero_init_parity(model, source_info, dim, device))
        operators = {"B0_CanonicalNTR": None}
        for variant in VARIANTS[1:]:
            op, log, att, _cached = train_variant(out, source, variant, model, source_info, dim, device, args.smoke, args.batch_images)
            operators[variant] = op
            train_log.extend(log)
            attn.extend(att)
        params.extend(parameter_audit(source, operators))
        cells, _audits = protocol_cells(source, device, args.smoke)
        for _cid, target, dataset, idx_or_rows, _patch, kind in cells:
            if (source, target) not in PAIRS:
                continue
            rr, pp = score_dataset(out, source, target, dataset, idx_or_rows, kind, model, source_info, operators, device, args.smoke)
            raw.extend(rr)
            percat.extend(pp)
            write_csv(out / "RAW_METRICS.partial.csv", raw)
            write_csv(out / "PER_CATEGORY_FULL_RES.partial.csv", percat)
            if args.smoke:
                break

    write_csv(out / "B0_REPRODUCTION.csv", [r for r in raw if r.get("variant") == "B0_CanonicalNTR"])
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "FULL_RES_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY_FULL_RES.csv", percat)
    write_csv(out / "PARAMETER_AUDIT.csv", params)
    write_csv(out / "TRAINING_LOG.csv", train_log)
    write_csv(out / "ATTENTION_AUDIT.csv", attn)
    write_csv(out / "FEATURE_DRIFT.csv", parity)
    write_csv(out / "MORPHOLOGY_DELTA_ANALYSIS.csv", [{"status": "NOT_COMPUTED_IN_SMOKE" if args.smoke else "TODO_POSTHOC"}])
    write_csv(out / "LRP_COMPLEMENTARITY.csv", [{"status": "NOT_RUN_PRIMARY_GATE_NOT_EVALUATED"}])
    max_parity = max([fnum(r, "max_abs_feature_error") for r in parity], default=float("nan"))
    outcome = "LDRO_SMOKE_TEST_PASSED" if args.smoke and np.isfinite(max_parity) and max_parity < 1e-5 and raw else "LDRO_FORMAL_RUN_COMPLETE"
    write_json(out / "OUTCOME.json", {"outcome": outcome, "smoke": bool(args.smoke), "full": bool(args.full), "runtime_sec": time.time() - start, "device": device, "image_batch_size": int(args.batch_images), "max_zero_init_feature_error": max_parity})
    report = [
        "# Learned Defect Relation Operator",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        f"Mode: `{'smoke' if args.smoke else 'formal'}`.",
        f"Runtime: {time.time() - start:.1f}s.",
        f"Image micro-batch: `{int(args.batch_images)}`.",
        "",
        f"Zero-init max feature error: `{max_parity:.8g}`.",
        "",
        "Run full formal experiment with:",
        "",
        "```powershell",
        "python experiments\\phase_learned_defect_relation_operator.py --full",
        "```",
        "",
        "Smoke mode intentionally evaluates only a tiny subset and is not a performance result.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[LDRO] OUTCOME={outcome}", flush=True)
    print(f"[LDRO] wrote {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--batch-images", type=int, default=IMAGE_BATCH_SIZE, help="Engineering image micro-batch for LDRO visual backprop; default keeps 16GB GPUs safe.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
