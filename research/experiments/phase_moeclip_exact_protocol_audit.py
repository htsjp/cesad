import argparse
import csv
import hashlib
import importlib
import json
import logging
import math
import os
import sys
import time
import types
from pathlib import Path
from contextlib import contextmanager

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "_external" / "MoECLIP"
OUT = Path("results/phase_moeclip_exact_protocol_audit/formal")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
IMG_SIZE = 518
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "MVTec"),
)
SOURCE_TO_OFFICIAL = {"Full MVTec": "MVTec", "Full VisA": "VisA"}
BASELINE_VARIANT = "E0_Core_NTR_FixedLRP"
FULL_VARIANT = "E1_OfficialMoECLIP"
OFFICIAL_REPO_URL = "https://github.com/CoCoRessa/MoECLIP"

ABLATIONS = {
    "B1_SingleLoRA": {"experts": 1, "topk": 1, "use_fofs": False, "use_paa": False, "balance": 0.0, "etf": 0.0},
    "B2_MoE_RoutingExperts": {"experts": 4, "topk": 2, "use_fofs": False, "use_paa": False, "balance": 0.0, "etf": 0.0},
    "B3_MoE_FOFS": {"experts": 4, "topk": 2, "use_fofs": True, "use_paa": False, "balance": 0.0, "etf": 0.0},
    FULL_VARIANT: {"experts": 4, "topk": 2, "use_fofs": True, "use_paa": True, "balance": 0.01, "etf": 0.01},
}


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


@contextmanager
def pushd(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def official_get_dataset(official, *args, **kwargs):
    with pushd(OFFICIAL_ROOT):
        return official["get_dataset"](*args, **kwargs)


def fnum(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def official_imports():
    if not OFFICIAL_ROOT.exists():
        raise FileNotFoundError(OFFICIAL_ROOT)
    sys.modules.setdefault("ipdb", types.SimpleNamespace(set_trace=lambda *args, **kwargs: None))
    try:
        import pkg_resources
        import packaging
        if not hasattr(pkg_resources, "packaging"):
            pkg_resources.packaging = packaging
    except Exception:
        pass
    sys.path.insert(0, str(OFFICIAL_ROOT))
    model_clip = importlib.import_module("model.clip")
    moe_adapter = importlib.import_module("model.moe_adapter")
    forward_utils = importlib.import_module("forward_utils")
    dataset_mod = importlib.import_module("dataset")
    constants = importlib.import_module("dataset.constants")
    utils = importlib.import_module("utils")
    patch_official_dataset_paths(dataset_mod, constants)
    return {
        "create_model": model_clip.create_model,
        "MoECLIP": moe_adapter.MoECLIP,
        "get_single_text": forward_utils.get_adapted_single_class_text_embedding,
        "calculate_similarity_map": forward_utils.calculate_similarity_map,
        "calculate_seg_loss": forward_utils.calculate_seg_loss,
        "get_dataset": dataset_mod.get_dataset,
        "DOMAINS": dataset_mod.DOMAINS,
        "setup_seed": utils.setup_seed,
    }


def patch_official_dataset_paths(dataset_mod, constants):
    paths = {
        "MVTec": str(DATA_ROOT / "mvtec"),
        "VisA": str(DATA_ROOT / "VISA"),
        "BTAD": str(DATA_ROOT / "btad" / "BTech_Dataset_transformed"),
        "DTD-Synthetic": str(DATA_ROOT / "DTD-Synthetic" / "DTD-Synthetic" / "DTD-Synthetic"),
    }
    constants.DATA_PATH.update(paths)
    dataset_mod.DATA_PATH.update(paths)


def sha256_params(named_params):
    h = hashlib.sha256()
    for name, param in named_params:
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def clip_checksum(model):
    return sha256_params((f"clipmodel.{n}", p) for n, p in model.clipmodel.named_parameters())


def trainable_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def total_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def make_model(official, variant, device, relu=False):
    cfg = ABLATIONS[variant]
    clip_model = official["create_model"](
        model_name="ViT-L-14-336",
        img_size=IMG_SIZE,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = official["MoECLIP"](
        clip_model=clip_model,
        use_paa=cfg["use_paa"],
        seg_proj_sharing_strategy="shared",
        image_adapt_weight=0.1,
        moe_r=8,
        use_fofs=cfg["use_fofs"],
        moe_lora_alpha=16,
        moe_num_experts=cfg["experts"],
        moe_top_k=cfg["topk"],
        moe_layers=[5, 11, 17, 23],
        relu=relu,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.text_adapter.parameters():
        p.requires_grad = True
    image_params = []
    if cfg["use_fofs"]:
        for name, p in model.image_adapter.named_parameters():
            if "lora_A" in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
                image_params.append(p)
    else:
        for p in model.image_adapter.parameters():
            p.requires_grad = True
            image_params.append(p)
    return model


def optimizer_for(model, lr):
    params = [{"params": model.text_adapter.parameters()}]
    image_params = [p for p in model.image_adapter.parameters() if p.requires_grad]
    if image_params:
        params.append({"params": image_params})
    return torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))


def train_variant(official, source, variant, args, device):
    suffix = "smoke" if args.smoke else "formal"
    save_dir = OUT / "checkpoints" / suffix / source.replace(" ", "_") / variant
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_dir / "moe_last.pth"
    model = make_model(official, variant, device)
    if ckpt.exists() and not args.force_retrain:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model.text_adapter.load_state_dict(payload["text_adapter"], strict=True)
        model.image_adapter.load_state_dict(payload["image_adapter"], strict=True)
        model.eval()
        rows = read_csv(save_dir / "TRAINING_CURVE.csv") if (save_dir / "TRAINING_CURVE.csv").exists() else []
        return model, rows, True

    official["setup_seed"](111)
    before = clip_checksum(model)
    opt = optimizer_for(model, args.lr)
    _, train_dataset = official_get_dataset(official, SOURCE_TO_OFFICIAL[source], IMG_SIZE, "full_shot", -1, "train", logging.getLogger("moeclip_audit"))
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    rows = []
    cfg = ABLATIONS[variant]
    # Official train.py leaves the MoECLIP wrapper in eval mode before calling
    # train_adapter. This keeps CLIP patch_dropout disabled; enabling train()
    # makes PAA receive a non-square number of tokens.
    model.eval()
    for epoch in range(1, args.epoch + 1):
        vals = []
        for step, batch in enumerate(loader):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            class_names = batch["class_name"]
            text_by_class = {}
            for class_name in set(class_names):
                text_by_class[class_name] = official["get_single_text"](model, SOURCE_TO_OFFICIAL[source], class_name, device)
            text_feature = torch.stack([text_by_class[c] for c in class_names], dim=0)
            patch_features, det_feature, aux_loss, special_loss = model(image)
            det_feature = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature, text_feature)[:, 0]
            loss = F.cross_entropy(cls_preds, label)
            for feat in patch_features:
                pred = official["calculate_similarity_map"](feat, text_feature, IMG_SIZE)
                loss = loss + official["calculate_seg_loss"](pred, mask)
            loss = loss + cfg["balance"] * aux_loss + cfg["etf"] * special_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(aux_loss.detach().cpu()), float(special_loss.detach().cpu())))
            if args.smoke and step >= 1:
                break
        arr = np.asarray(vals, dtype=np.float64)
        row = {
            "source": source,
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "balance_aux": float(arr[:, 1].mean()),
            "etf_loss": float(arr[:, 2].mean()),
            "trainable_params": trainable_count(model),
            "total_params": total_count(model),
        }
        rows.append(row)
        print(f"[MoECLIP] train {source} {variant} epoch={epoch}/{args.epoch} loss={row['loss']:.4f}", flush=True)
        if args.smoke:
            break
    torch.save({"epoch": args.epoch, "text_adapter": model.text_adapter.state_dict(), "image_adapter": model.image_adapter.state_dict()}, ckpt)
    write_csv(save_dir / "TRAINING_CURVE.csv", rows)
    after = clip_checksum(model)
    write_json(save_dir / "CHECKSUM.json", {"clip_before": before, "clip_after": after, "clip_unchanged": before == after})
    model.eval()
    return model, rows, False


def mrad_imports():
    official_path = str(OFFICIAL_ROOT)
    if official_path in sys.path:
        sys.path.remove(official_path)
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))
    mod = sys.modules.get("utils")
    if mod is not None and getattr(mod, "__file__", "") and str(OFFICIAL_ROOT) in str(getattr(mod, "__file__", "")):
        del sys.modules["utils"]
    from experiments.phase_final_cross_dataset_benchmark import protocol_cells
    from experiments.phase_parameter_free_local_zoom_pyramid_fullres import evaluate_maps, item_at, item_meta, mask_np
    import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
    return protocol_cells, evaluate_maps, item_at, item_meta, mask_np, lzp


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])


def resolve_path(item, ref):
    candidates = []
    if isinstance(item, dict) and item.get("img_path"):
        candidates.append(Path(str(item["img_path"])))
    if isinstance(ref, dict) and ref.get("img_path"):
        candidates.append(Path(str(ref["img_path"])))
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(candidates)


@torch.no_grad()
def moe_maps_for_target(official, model, source, target, device, smoke):
    protocol_cells, _evaluate_maps, item_at, item_meta, mask_np, _lzp = mrad_imports()
    cells, _audits = protocol_cells(source, device, smoke)
    cell = next((c for c in cells if c[1] == target), None)
    if cell is None:
        raise RuntimeError(f"No MRAD protocol cell for {source}->{target}")
    _cid, _target, dataset, refs, _patch, kind = cell
    n = min(8, len(refs)) if smoke else len(refs)
    transform = eval_transform()
    text = official["get_single_text"](model, "MVTec", "object", device)
    maps, metas = [], []
    for pos, ref in enumerate(refs[:n]):
        if pos % 25 == 0:
            print(f"[MoECLIP] eval {source}->{target} image={pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, _p = item_meta(item, kind, ref, target)
        path = resolve_path(item, ref)
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        patch_features, _det_feature, _aux, _special = model(image)
        preds = []
        for feat in patch_features:
            pred = official["calculate_similarity_map"](feat, text.unsqueeze(0), IMG_SIZE, test=True, domain="Industrial")
            preds.append(pred)
        score = torch.cat(preds, dim=1).sum(1)[0].detach().cpu().numpy().astype(np.float32)
        maps.append(score)
        metas.append({"category": cat, "label": label, "mask": mask_np(item), "img_path": path})
    return maps, metas


def evaluate_variant(official, model, source, target, variant, device, smoke):
    cache = OUT / "maps" / f"{source.replace(' ', '_')}_to_{target}_{variant}_{'smoke' if smoke else 'formal'}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        maps = [x.astype(np.float32) for x in data["maps"]]
        metas = data["metas"].tolist()
    else:
        maps, metas = moe_maps_for_target(official, model, source, target, device, smoke)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, maps=np.stack(maps).astype(np.float16), metas=np.asarray(metas, dtype=object))
    _protocol_cells, evaluate_maps, _item_at, _item_meta, _mask_np, lzp = mrad_imports()
    old = tuple(lzp.VARIANTS)
    try:
        lzp.VARIANTS = (variant,)
        raw, percat = evaluate_maps(source, target, {variant: maps}, metas)
    finally:
        lzp.VARIANTS = old
    return raw, percat


def baseline_rows():
    path = ROOT / "results" / "phase_parameter_free_local_zoom_pyramid_fullres" / "formal" / "RAW_METRICS.csv"
    rows = read_csv(path)
    out = []
    for r in rows:
        if r.get("variant") == "E2_GlobalLocalZoomPyramid":
            nr = dict(r)
            nr["variant"] = BASELINE_VARIANT
            out.append(nr)
    return out


def merge_baseline(raw):
    base = {(r["source"], r["dataset"]): r for r in baseline_rows()}
    out = []
    for source, target in PILOT_PAIRS:
        if (source, target) in base:
            out.append(base[(source, target)])
    out.extend(raw)
    return out


def deltas(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    variants = sorted({r["variant"] for r in raw if r["variant"] != BASELINE_VARIANT})
    for source, target in PILOT_PAIRS:
        b = by.get((source, target, BASELINE_VARIANT))
        if not b:
            continue
        for variant in variants:
            e = by.get((source, target, variant))
            if not e:
                continue
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP"),
                "Delta_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
    return rows


def summarize_delta(delta_rows):
    rows = []
    for variant in sorted({r["variant"] for r in delta_rows}):
        vals = np.asarray([float(r["Delta_M_FR_P_AP"]) for r in delta_rows if r["variant"] == variant], dtype=np.float64)
        rows.append({
            "variant": variant,
            "mean_delta_p_ap": float(vals.mean()) if vals.size else float("nan"),
            "median_delta_p_ap": float(np.median(vals)) if vals.size else float("nan"),
            "improved": int(np.sum(vals > 0)),
            "non_drop_ge_neg_0p2": int(np.sum(vals >= -0.2)),
            "total": int(vals.size),
        })
    return rows


def router_diagnostics(model, source, variant, device, official, smoke):
    official_dataset = SOURCE_TO_OFFICIAL[source]
    _, ds = official_get_dataset(official, official_dataset, IMG_SIZE, "full_shot", -1, "train", logging.getLogger("moeclip_diag"))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    handles = []
    captures = {}
    for li, moe in enumerate(model.image_adapter["moe_adapters"]):
        def make_hook(idx):
            def hook(_module, inputs):
                captures[idx] = inputs[0].detach()
            return hook
        handles.append(moe.register_forward_pre_hook(make_hook(li)))
    transform_count = min(24, len(ds)) if smoke else min(128, len(ds))
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= transform_count:
                break
            captures.clear()
            image = batch["image"].to(device)
            _ = model(image)
            for li, hidden in captures.items():
                moe = model.image_adapter["moe_adapters"][li]
                flat = hidden.reshape(-1, hidden.shape[-1])
                probs = F.softmax(moe.gate(flat), dim=-1).float()
                k = moe.config.top_k_
                weights, selected = torch.topk(probs, k, dim=-1)
                top1 = selected[:, 0].detach().cpu().numpy()
                ent = (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).detach().cpu().numpy()
                for ex in range(moe.config.num_experts_):
                    rows.append({
                        "source": source,
                        "variant": variant,
                        "layer_index_0based": [5, 11, 17, 23][li],
                        "expert": ex,
                        "top1_fraction": float(np.mean(top1 == ex)),
                        "mean_prob": float(probs[:, ex].mean().detach().cpu()),
                        "routing_entropy_mean": float(np.mean(ent)),
                    })
    for h in handles:
        h.remove()
    return rows


def official_audit_rows():
    return [
        {"item": "official_repo", "value": OFFICIAL_REPO_URL},
        {"item": "commit", "value": os.popen(f'git -C "{OFFICIAL_ROOT}" rev-parse HEAD').read().strip()},
        {"item": "backbone", "value": "ViT-L-14-336, official local OpenCLIP weight ViT-L-14-336px.pt"},
        {"item": "input_resolution", "value": "518x518"},
        {"item": "moe_layers", "value": "0-based [5,11,17,23], corresponding to transformer layers 6/12/18/24"},
        {"item": "experts", "value": "4 for full MoECLIP"},
        {"item": "top_k", "value": "2 for full MoECLIP"},
        {"item": "lora", "value": "rank=8 alpha=16 dropout=0.05"},
        {"item": "moe_residual_weight", "value": "image_adapt_weight=0.1"},
        {"item": "fofs", "value": "fixed orthogonal lora_A chunks per expert; lora_A frozen under use_fofs"},
        {"item": "etf_loss", "value": "expert output equiangular MSE, lambda=0.01 in full variant"},
        {"item": "balance_loss", "value": "router load CV^2, lambda=0.01 in full variant"},
        {"item": "paa", "value": "patch average aggregation scales [1,3,5], shared seg projection by layer"},
        {"item": "prompt_scoring", "value": "official text normal/abnormal prompts; strict target eval uses class-agnostic object prompt to avoid target category names"},
        {"item": "optimizer", "value": "Adam lr=5e-5 betas=(0.5,0.999), official default epoch=20"},
    ]


def decide(summary):
    full = next((r for r in summary if r["variant"] == FULL_VARIANT), None)
    if not full:
        return "MOECLIP_AUDIT_INCOMPLETE"
    if full["mean_delta_p_ap"] >= 1.0 and full["improved"] >= 3:
        return "MOECLIP_PATCH_SPECIALIZATION_HAS_STRONG_HEADROOM"
    if full["improved"] >= 2 and full["mean_delta_p_ap"] > 0:
        return "MOECLIP_COMPONENT_HAS_TRANSFERABLE_VALUE"
    mtv = None
    for r in read_csv(OUT / "PILOT_DELTAS.csv") if (OUT / "PILOT_DELTAS.csv").exists() else []:
        if r.get("variant") == FULL_VARIANT and r.get("source") == "Full MVTec" and r.get("dataset") == "VisA":
            mtv = float(r["Delta_M_FR_P_AP"])
    if mtv is not None and mtv >= 1.0 and full["mean_delta_p_ap"] < 0:
        return "MOECLIP_SPECIALIZATION_IS_TARGET_MORPHOLOGY_SENSITIVE"
    return "MOECLIP_DOES_NOT_OUTPERFORM_CURRENT_CORE"


def write_report(raw, delta_rows, summary, train_rows, router_rows, outcome, runtime, variants):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Phase MoECLIP Exact-Protocol Audit",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        f"Runtime: {runtime:.1f}s",
        "",
        f"Official source: [CoCoRessa/MoECLIP]({OFFICIAL_REPO_URL}), commit `{official_audit_rows()[1]['value']}`.",
        "",
        "Protocol note: the official model and losses are used, but target evaluation is routed through the local full-resolution MRAD evaluator. Because the strict protocol forbids target category names, target text scoring uses the official `object` fallback prompt instead of target class names.",
        "",
        "## Official Implementation Audit",
        "",
    ]
    for r in official_audit_rows():
        lines.append(f"- {r['item']}: {r['value']}")
    lines += ["", "## Pilot Results", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO | Delta P-AP |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for source, target in PILOT_PAIRS:
        b = by.get((source, target, BASELINE_VARIANT))
        for variant in [BASELINE_VARIANT] + list(variants):
            r = by.get((source, target, variant))
            if not r:
                continue
            delta = 0.0 if variant == BASELINE_VARIANT or not b else fnum(r, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            lines.append(f"| {source} | {target} | {variant} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} | {delta:+.4f} |")
    lines += ["", "## Delta Summary", "", "| Variant | Mean Delta P-AP | Median | Improved | Non-drop >= -0.2 |", "|---|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['variant']} | {s['mean_delta_p_ap']:+.4f} | {s['median_delta_p_ap']:+.4f} | {s['improved']}/{s['total']} | {s['non_drop_ge_neg_0p2']}/{s['total']} |")
    lines += ["", "## Diagnostics", "", f"- Training rows: {len(train_rows)}.", f"- Router diagnostic rows: {len(router_rows)}.", "- Frozen CLIP base checksum is stored per checkpoint in `checkpoints/*/*/CHECKSUM.json`.", "- Ablation variants are official-code approximations: single expert LoRA, MoE without FOFS/PAA losses, MoE+FOFS, and full MoECLIP."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    official = official_imports()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    variants = [FULL_VARIANT] if not args.with_ablations else ["B1_SingleLoRA", "B2_MoE_RoutingExperts", "B3_MoE_FOFS", FULL_VARIANT]
    raw_eval, percat_eval, train_rows, router_rows = [], [], [], []
    for source in ("Full MVTec", "Full VisA"):
        needed_targets = [target for src, target in PILOT_PAIRS if src == source]
        for variant in variants:
            model, curves, cached = train_variant(official, source, variant, args, device)
            train_rows.extend(curves)
            router_rows.extend(router_diagnostics(model, source, variant, device, official, args.smoke))
            for target in needed_targets:
                rr, cc = evaluate_variant(official, model, source, target, variant, device, args.smoke)
                raw_eval.extend(rr)
                percat_eval.extend(cc)
                write_csv(OUT / "RAW_METRICS.partial.csv", merge_baseline(raw_eval))
    raw = merge_baseline(raw_eval)
    delta_rows = deltas(raw)
    summary = summarize_delta(delta_rows)
    write_csv(OUT / "OFFICIAL_IMPLEMENTATION_AUDIT.csv", official_audit_rows())
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat_eval)
    write_csv(OUT / "PILOT_DELTAS.csv", delta_rows)
    write_csv(OUT / "SUMMARY.csv", summary)
    write_csv(OUT / "TRAINING_CURVES.csv", train_rows)
    write_csv(OUT / "ROUTER_DIAGNOSTICS.csv", router_rows)
    outcome = decide(summary)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "summary": summary, "variants": variants, "smoke": args.smoke})
    write_report(raw, delta_rows, summary, train_rows, router_rows, outcome, time.time() - t0, variants)
    print(f"[MoECLIP] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.00005)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--with-ablations", action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
