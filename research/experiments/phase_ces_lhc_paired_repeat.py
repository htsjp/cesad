"""New paired CES/linear control; cached frozen features, no backbone training.

All primary runs have identical realized index streams and LR. Any conditional
LR diagnostic is selected using a fixed source-only probe BEFORE target access.
Historical four-transfer loader-GT P-AP only; never edits existing paper tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import phase_ces_linear_head_control as hist
from experiments import phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_defect_query_transfer_pilot import light_fold
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_xxxviii_visual_state_representation import norm_np, patch_labels
from experiments.phase_xlii_patch_specialized_adaptation import balanced_loss

DEFAULT_OUT = ROOT / "results/phase_ces_lhc_paired_repeat"
PAYLOAD = ROOT / "results/phase_statealigned_pc1_debias/formal"
METHODS = ("LinearHead", "NormalizedLinearHead", "CES")
SEEDS = (0, 1, 2)
PAIRS = tuple(hist.PAIRS)
LR = .001
RHO = .10
EPOCHS = 10
LR_GRID = (.0003, .001, .003)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_hash(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def json_hash(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def write_json(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def write_csv(p, rows):
    if not rows:
        return
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def status(out, phase, **details):
    write_json(out / "STATUS.json", dict(phase=phase, updated=time.strftime("%Y-%m-%d %H:%M:%S"), **details))
    print(f"[{phase}] {details}", flush=True)


def load_transform(source, device):
    if source == "Full MVTec":
        return dict(use_align=False, q_pc1=None, align_model=None)
    e1 = torch.load(PAYLOAD / "E1_PC1Removed-TrueMask_formal.pt", map_location="cpu", weights_only=False)
    e2 = torch.load(PAYLOAD / "E2_PC1Removed-CorruptedMask_formal.pt", map_location="cpu", weights_only=False)
    pc1 = hist.unit_np(np.load(PAYLOAD / "q_pc1_statealigned1024.npy").astype(np.float32))
    model = make_model(np.asarray(e2["patch_states"], np.float32), e2, device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    st, sc = np.asarray(e1["patch_states"], np.float32), np.asarray(e2["patch_states"], np.float32)
    d = .5 * (st[1] - st[0])
    return dict(use_align=True, q_pc1=pc1, align_model=model,
                m=.5 * (sc[0] + sc[1]), alpha=float(np.linalg.norm(d)), init_v=hist.unit_np(d))


def transformed(z, transform, device):
    z = np.asarray(z, dtype=np.float32).reshape(-1, 1024)
    if transform["use_align"]:
        z = norm_np(remove_pc1(z, transform["q_pc1"]))
        with torch.no_grad():
            return transform["align_model"].adapt(torch.tensor(z, device=device)).float()
    return torch.tensor(norm_np(z), device=device)


def source_dataset(source):
    return light_fold("D1" if source == "Full MVTec" else "D2", 0)


def mask_only(ds, idx):
    row = ds.data_all[int(idx)]
    if int(row["anomaly"]) == 0:
        return np.zeros(1369, dtype=np.int64)
    p = Path(ds.root) / row["mask_path"]
    if not p.is_file():
        raise FileNotFoundError(f"Source mask missing: {p}")
    m = Image.fromarray((np.asarray(Image.open(p).convert("L")) > 0).astype(np.uint8) * 255, mode="L")
    return patch_labels({"img_mask": ds.target_transform(m)})


def source_arrays(direction):
    base = ROOT / "results/phase_memory_support_modes/array_cache"
    paths = []
    for k in ("train_patch", "held_patch"):
        found = list(base.glob(f"{direction}_fold0_clip_*_{k}.npy"))
        if len(found) != 1:
            raise RuntimeError(f"Ambiguous/missing source array {direction}/{k}: {found}")
        paths.append(found[0])
    return [np.load(p, mmap_mode="r") for p in paths], paths


def balanced_indices(y, cat, rng):
    names = sorted(np.unique(cat).tolist())
    per = max(1, max(2, 8192 // len(names)) // 2)
    ids = []
    for name in names:
        for cls in (0, 1):
            pool = np.flatnonzero((cat == name) & (y == cls))
            if not len(pool):
                raise ValueError(f"Missing class in {name}")
            ids.append(rng.choice(pool, per, replace=len(pool) < per))
    ix = np.concatenate(ids)
    return ix[rng.permutation(len(ix))].astype(np.int32)


def prepare_source(out, source, seed, transform, device, smoke):
    slug = source.replace(" ", "_") + f"_seed{seed}"
    directory = out / "source_pools" / slug
    done = directory / "COMPLETE.json"
    if done.exists():
        meta = json.loads(done.read_text(encoding="utf-8"))
        z = np.load(directory / "features.npy", mmap_mode="r")
        y = np.load(directory / "labels.npy")
        batches = np.load(directory / "batch_indices.npy")
        probe = np.load(directory / "probe_indices.npy")
        for key, arr in [("feature_array_hash", z), ("labels_hash", y), ("batch_stream_hash", batches)]:
            assert digest(arr) == meta[key], f"Changed saved source pool: {key}"
        return z, y, batches, probe, meta
    directory.mkdir(parents=True, exist_ok=True)
    _, ds, tr, held = source_dataset(source)
    indices = tr + held
    direction = "D1" if source == "Full MVTec" else "D2"
    arrays, paths = source_arrays(direction)
    assert len(arrays[0]) == len(tr) and len(arrays[1]) == len(held)
    names = sorted({r["cls_name"] for r in ds.data_all})
    if smoke:
        names = names[:2]
    labels = {p: mask_only(ds, idx) for p, idx in enumerate(indices) if ds.data_all[idx]["cls_name"] in names}
    rng = np.random.default_rng(seed)
    records, features = [], []
    for name in names:
        parts = {k: [] for k in ("normal_image", "anomaly_mask_out", "anomaly_mask")}
        for pos, idx in enumerate(indices):
            row = ds.data_all[idx]
            if row["cls_name"] != name:
                continue
            y = labels[pos]
            normal = int(row["anomaly"]) == 0 if direction == "D1" else y.max() == 0
            pools = [("normal_image", 0, np.flatnonzero(y == 0))] if normal else [
                ("anomaly_mask_out", 0, np.flatnonzero(y == 0)), ("anomaly_mask", 1, np.flatnonzero(y == 1))]
            for origin, cls, ids in pools:
                room = (64 if smoke else 512) - len(parts[origin])
                if room <= 0 or not len(ids):
                    continue
                if len(ids) > room:
                    ids = rng.choice(ids, room, replace=False)
                for pid in ids:
                    parts[origin].append(dict(source=source, seed=seed, category=name,
                        source_position=pos, dataset_index=int(idx), patch_flat_index=int(pid), label=cls,
                        origin=origin, image=str(Path(ds.root) / row["img_path"]), mask=str(Path(ds.root) / row["mask_path"])))
        items = sum(parts.values(), [])
        if len({r["label"] for r in items}) < 2:
            continue
        records.extend(items)
    for i, rec in enumerate(records):
        pos = rec["source_position"]
        raw = arrays[0][pos] if pos < len(tr) else arrays[1][pos - len(tr)]
        features.append(np.asarray(raw[rec["patch_flat_index"]], np.float32))
        rec["pool_index"] = i
    raw = np.stack(features)
    chunks = []
    for i in range(0, len(raw), 8192):
        chunks.append(transformed(raw[i:i+8192], transform, device).cpu().numpy())
    z = np.concatenate(chunks).astype(np.float32)
    y = np.array([r["label"] for r in records], np.float32)
    cat = np.array([r["category"] for r in records])
    if transform["use_align"]:
        geom = {k: transform[k] for k in ("m", "alpha", "init_v")}
    else:
        sn, sa = hist.unit_np(z[y < .5].mean(0)), hist.unit_np(z[y > .5].mean(0))
        d = .5 * (sa - sn)
        geom = dict(m=.5 * (sa + sn), alpha=float(np.linalg.norm(d)), init_v=hist.unit_np(d))
    init = hist.initial_geometry(geom)
    steps = 2 if smoke else EPOCHS * max(1, math.ceil(len(y) / 8192))
    brng = np.random.default_rng(seed)
    batches = np.stack([balanced_indices(y, cat, brng) for _ in range(steps)])
    probe = balanced_indices(y, cat, np.random.default_rng(1000000 + seed))
    for name, value in [("features", z), ("labels", y), ("batch_indices", batches), ("probe_indices", probe)]:
        np.save(directory / (name + ".npy"), value)
    np.savez(directory / "initial_geometry.npz", **init)
    write_csv(directory / "PATCH_MANIFEST.csv", records)
    write_csv(directory / "SOURCE_IMAGE_MANIFEST.csv", [dict(source_position=p, dataset_index=int(idx), **ds.data_all[idx]) for p, idx in enumerate(indices)])
    meta = dict(source=source, seed=seed, n_patches=len(y), n_categories=len(np.unique(cat)),
        actual_batch=batches.shape[1], updates=len(batches), epochs=1 if smoke else EPOCHS,
        feature_array_hash=digest(z), labels_hash=digest(y), batch_stream_hash=digest(batches),
        probe_indices_hash=digest(probe), manifest_hash=file_hash(directory / "PATCH_MANIFEST.csv"),
        initial_geometry_hash=file_hash(directory / "initial_geometry.npz"),
        feature_dtype=str(z.dtype), feature_files=[str(p) for p in paths],
        feature_file_shapes=[list(a.shape) for a in arrays],
        feature_preprocessing="VisA: PC1 remove -> L2 -> frozen E2 residual -> L2; MVTec: L2",
        init={k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in init.items()})
    write_json(done, meta)
    status(out, "SOURCE_READY", source=source, seed=seed, patches=len(y), updates=len(batches), batch=batches.shape[1])
    return z, y, batches, probe, meta


def init_from_meta(meta):
    return {k: np.asarray(v, np.float32) if isinstance(v, list) else v for k,v in meta["init"].items()}


def new_module(method, init, device):
    if method == "LinearHead":
        return hist.LinearHead(init["w0"]).to(device)
    if method == "NormalizedLinearHead":
        return hist.NormalizedLinearHead(init["w0"], init["c0"]).to(device)
    return hist.CESHead(init["m"], init["alpha"], init["init_v"]).to(device)


def objective(logits, y):
    base, bce, dice = balanced_loss(logits, y)
    ntr = hist.ntr_loss_from_logits(logits, y, RHO)
    return base + ntr, bce, dice, ntr


def source_probe(module, z, y, probe, device):
    with torch.no_grad():
        q = module(torch.tensor(np.asarray(z[probe]), device=device))
        loss, bce, dice, ntr = objective(q, torch.tensor(y[probe], device=device))
    return dict(probe_loss=float(loss), probe_bce=float(bce), probe_dice=float(dice), probe_ntr=float(ntr),
                probe_mean_normal=float(q[torch.tensor(y[probe] < .5, device=device)].mean()),
                probe_mean_anomaly=float(q[torch.tensor(y[probe] > .5, device=device)].mean()))


def equivalence(z, meta, device):
    init = init_from_meta(meta)
    modules = {m: new_module(m, init, device) for m in METHODS}
    maxima = {m: 0. for m in METHODS}
    for i in range(0, len(z), 8192):
        zz = torch.tensor(np.asarray(z[i:i+8192]), device=device)
        with torch.no_grad():
            ref = modules["CES"](zz)
            for method, module in modules.items():
                maxima[method] = max(maxima[method], float((module(zz)-ref).abs().max()))
    assert max(maxima.values()) < 1e-6, maxima
    return [dict(source=meta["source"], seed=meta["seed"], method=m, max_abs_initial_logit_error=v,
                 num_patches=len(z), initial_raw_parameter_norm=float(torch.sqrt(sum((p.detach()**2).sum() for p in modules[m].parameters()))),
                 initial_effective_weight_norm=init["c0"], initial_bias=0.) for m,v in maxima.items()]


def train_one(out, method, source, seed, z, y, batches, probe, meta, lr, run_type, config_hash, device):
    slug = f"{source.replace(' ', '_')}_{method}_seed{seed}_{run_type}_lr{lr:.4g}"
    path = out / "checkpoints" / (slug + ".pt")
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["config_hash"] == config_hash and saved["batch_stream_hash"] == meta["batch_stream_hash"]
        assert saved["realized_updates"] == len(batches)
        return saved, path
    init = init_from_meta(meta)
    torch.manual_seed(seed)
    module = new_module(method, init, device)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0., betas=(.9,.999), eps=1e-8, amsgrad=False)
    initial_probe = source_probe(module, z, y, probe, device)
    log = []
    for step, ids in enumerate(batches):
        x = torch.tensor(np.asarray(z[ids]), device=device)
        labels = torch.tensor(y[ids], device=device)
        loss, bce, dice, ntr = objective(module(x), labels)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.sqrt(sum((p.grad**2).sum() for p in module.parameters())))
        opt.step()
        assert all(torch.isfinite(p).all() for p in module.parameters())
        spec = module.to_spec(source)
        rawnorm = float(torch.sqrt(sum((p.detach()**2).sum() for p in module.parameters())))
        counters = [int(v["step"].item()) for v in opt.state.values()]
        assert counters and all(v == step+1 for v in counters)
        row = dict(source=source, method=method, seed=seed, run_type=run_type, lr=lr,
            step=step+1, epoch=(step // max(1,len(batches)//EPOCHS))+1, actual_batch=len(ids),
            batch_ids_hash=digest(ids), loss=float(loss), bce=float(bce), dice=float(dice), ntr=float(ntr),
            grad_norm=grad, raw_parameter_norm=rawnorm, effective_weight_norm=float(np.linalg.norm(spec.weight())),
            bias=spec.bias(), optimizer_step_min=min(counters), optimizer_step_max=max(counters),
            config_hash=config_hash, pool_hash=meta["feature_array_hash"], **source_probe(module,z,y,probe,device))
        log.append(row)
    spec = module.to_spec(source)
    # Selection uses the fixed source probe only, before any target evaluation.
    lookback = min(6, len(log)-1)
    relative_drop = ((log[-1-lookback]["probe_loss"] - log[-1]["probe_loss"]) /
                     max(abs(log[-1-lookback]["probe_loss"]),1e-12))
    saved = dict(method=method, source=source, seed=seed, run_type=run_type, lr=lr,
        params=spec.params, raw_model_state=module.state_dict(), optimizer_state=opt.state_dict(),
        config_hash=config_hash, batch_stream_hash=meta["batch_stream_hash"], feature_array_hash=meta["feature_array_hash"],
        initial_geometry=init, initial_probe=initial_probe, final_probe=log[-1]["probe_loss"],
        recent_relative_source_loss_drop=relative_drop,
        diagnostic_trigger=bool(relative_drop > .005 and log[-1]["grad_norm"] > 1e-6),
        realized_updates=len(log), logs=log, checkpoint_selection="LAST_UPDATE_NO_TARGET_ACCESS")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    write_csv(out / "training_logs" / (slug + ".csv"), log)
    status(out, "TRAINED", source=source, method=method, seed=seed, lr=lr, updates=len(log), probe_loss=saved["final_probe"])
    return saved, path


def preflight(out, smoke):
    needed = [Path(p) for p in [PAYLOAD / "E1_PC1Removed-TrueMask_formal.pt", PAYLOAD / "E2_PC1Removed-CorruptedMask_formal.pt", PAYLOAD / "q_pc1_statealigned1024.npy"]]
    needed += [hist.local_cache_path(*p) for p in PAIRS]
    needed += [hist.LOCAL_CACHE_DIR / f"global_patch_{t}.npy" for t in ("MVTec", "VisA")]
    needed += [ROOT / "results/phase_xlv_cross_domain_transfer/feature_cache/BTAD_clip.npz",
               ROOT / "results/phase_median_spatial_consensus_generalization/formal/feature_cache/MPDD_clip.npz"]
    for direction in ("D1", "D2"):
        _, paths = source_arrays(direction)
        needed.extend(paths)
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    rows = []
    for p in needed:
        row = dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        if p.suffix == ".npy":
            a = np.load(p,mmap_mode="r")
            row.update(shape=str(a.shape), dtype=str(a.dtype))
        if p in needed[:3]:
            row["sha256"] = file_hash(p)
        rows.append(row)
    write_csv(out / "INPUT_PREFLIGHT.csv", rows)
    for source,target in PAIRS:
        arr = np.load(hist.local_cache_path(source,target),mmap_mode="r")
        expected = dict(VisA=2162, MPDD=458, BTAD=741, MVTec=1725)[target]
        assert arr.shape == (expected,4,1369,1024)
    config = dict(phase="CES-LHC paired repeat", seeds=[0] if smoke else list(SEEDS), methods=list(METHODS),
        pairs=list(PAIRS), primary_lr=LR, adam_betas=[.9,.999], adam_eps=1e-8, weight_decay=0.,
        rho=RHO, ntr_weight=1., epochs=EPOCHS, nominal_batch=8192, source_cap=512,
        budget="10*ceil(n_source_pool/8192), same realized batch indices for all heads",
        checkpoint_selection="last_update", lrp="global37->74 + four quadrants37 stitched74; .5/.5; no smoothing",
        evaluator="P-AP ONLY; historical lzp.safe_ap after lzp.torch_resize(align_corners=True) to historical loader GT; category macro",
        image_metrics="not requested by StepB and not recomputed", training_dtype="float32; no autocast; TF32 disabled",
        lr_diagnostic=dict(grid=LR_GRID, trigger="source fixed-probe objective relative drop >.005 over last6 updates AND final batch grad_norm>1e-6",
            selection="minimum last-update fixed-source-probe BCE+Dice+NTR; smaller LR breaks exact ties; same budget",
            report="all source runs; primary matched-LR results unchanged; source-selected controls separately evaluated"),
        smoke=smoke, code_hash=file_hash(__file__),
        historical_head_code_hash=file_hash(hist.__file__), historical_metric_code_hash=file_hash(lzp.__file__),
        balanced_loss_code_hash=hashlib.sha256(inspect.getsource(balanced_loss).encode()).hexdigest(),
        software=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version))
    p = out / "LOCKED_CONFIG.json"
    if p.exists():
        previous = json.loads(p.read_text(encoding="utf-8"))
        current = json.loads(json.dumps(config))
        if previous["code_hash"] != current["code_hash"]:
            # Resume-only serialization fix: preserve the training config/hash.
            # Authenticate its old script and require every non-preflight AST
            # node (including all fitting/scoring functions) to be identical.
            import ast
            snapshot = out.parent / "code_provenance/phase_ces_lhc_paired_repeat_at_training.py"
            assert snapshot.exists() and file_hash(snapshot) == previous["code_hash"]
            def non_preflight_ast(path):
                tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
                return [ast.dump(n, include_attributes=False) for n in tree.body
                        if not (isinstance(n, ast.FunctionDef) and n.name == "preflight")]
            assert non_preflight_ast(snapshot) == non_preflight_ast(__file__), "Scientific code changed; new run required"
            a = {k:v for k,v in previous.items() if k != "code_hash"}
            b = {k:v for k,v in current.items() if k != "code_hash"}
            assert a == b, "Scientific config changed; new run required"
            write_json(out / "RESUME_CODE_PROVENANCE.json", dict(training_script=str(snapshot),
                training_code_hash=previous["code_hash"], runtime_code_hash=current["code_hash"],
                only_preflight_AST_changed=True, checkpoints_and_training_config_unchanged=True,
                reason="canonical JSON list/tuple comparison at resume"))
        else:
            assert previous == current, "Locked config changed; use new output directory"
        config = previous
    else:
        write_json(p,config)
    h = json_hash(config)
    status(out,"PREFLIGHT_PASSED",config_hash=h,smoke=smoke)
    return config,h


def extract_existing_npz(cache, key, dest):
    if not dest.exists():
        import shutil
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix(".tmp")
        with zipfile.ZipFile(cache) as z, z.open(key+".npy") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src,dst,length=16*1024*1024)
        tmp.replace(dest)
    return np.load(dest,mmap_mode="r")


def target_cell(out,target):
    if target in ("MVTec","VisA"):
        _,ds,tr,held = light_fold("D1" if target=="MVTec" else "D2",0)
        refs=tr+held
        patches=np.load(hist.LOCAL_CACHE_DIR/f"global_patch_{target}.npy",mmap_mode="r")
        return ds,refs,patches,"source_dataset"
    if target=="BTAD":
        from experiments.phase_xlv_cross_domain_transfer import ExternalDataset
        from experiments.phase_li_asymmetric_state import DATA_ROOT
        cache=ROOT/"results/phase_xlv_cross_domain_transfer/feature_cache/BTAD_clip.npz"
        ds=ExternalDataset(DATA_ROOT/"btad/BTech_Dataset_transformed","BTAD",mode="test")
        z=np.load(cache,allow_pickle=True)
        refs=[dict(r) for r in z["rows"]]
        candidates=list((ROOT/"results/phase_memory_support_modes/array_cache").glob("BTAD_clip_*_patch.npy"))
        assert len(candidates)==1
        return ds,refs,np.load(candidates[0],mmap_mode="r"),"external"
    from experiments.phase_median_spatial_consensus_generalization import MetaDataset,HOLDOUTS
    ds=MetaDataset(HOLDOUTS["MPDD"][0],mode="test")
    cache=ROOT/"results/phase_median_spatial_consensus_generalization/formal/feature_cache/MPDD_clip.npz"
    with np.load(cache,allow_pickle=True) as z:
        refs=[dict(r) for r in z["rows"]]
    patches=extract_existing_npz(cache,"patch",out.parent/"shared_input_cache/MPDD_global_patch.npy")
    return ds,refs,patches,"external"


def packed_spec(saved):
    return hist.HeadSpec(saved["method"], saved["params"], saved["source"],saved["seed"])


def predict_pair(out,source,target,entries,transform,device,smoke):
    slug=hist.pair_slug(source,target)
    dest=out/"predictions"/slug
    dest.mkdir(parents=True,exist_ok=True)
    complete=dest/"COMPLETE.json"
    signatures=[dict(model_id=e["id"], checkpoint_hash=file_hash(e["path"])) for e in entries]
    if complete.exists():
        old=json.loads(complete.read_text(encoding="utf-8"))
        assert old["models"]==signatures
        return dest
    ds,refs,patch,kind=target_cell(out,target)
    local=np.load(hist.local_cache_path(source,target),mmap_mode="r")
    assert len(refs)==len(patch)==len(local)
    if smoke:
        selected=[]
        counts={}
        for pos,ref in enumerate(refs):
            row=ds.data_all[int(ref)] if kind=="source_dataset" else dict(ref)
            cat=str(row.get("cls_name",row.get("category",target)))
            lab=int(row.get("anomaly",row.get("label",0)))
            if cat not in counts and len(counts)>=1:
                continue
            bucket=counts.setdefault(cat,{0:0,1:0})
            if bucket[lab]<2:
                selected.append(pos);bucket[lab]+=1
    else:
        selected=list(range(len(refs)))
    n=len(selected)
    specs=[packed_spec(e["saved"]) for e in entries]
    weights=torch.tensor(np.stack([s.weight() for s in specs],axis=1),dtype=torch.float32,device=device)
    biases=torch.tensor([s.bias() for s in specs],dtype=torch.float32,device=device)
    maps=np.lib.format.open_memmap(dest/"maps74.npy",mode="w+",dtype=np.float32,shape=(n,len(entries),74,74))
    masks=np.lib.format.open_memmap(dest/"masks518.npy",mode="w+",dtype=np.uint8,shape=(n,518,518))
    metas=[]
    input_hasher=hashlib.sha256()
    t=time.time()
    max_map_equiv=0.
    for j,pos in enumerate(selected):
        item=lzp.item_at(ds,kind,refs[pos],pos)
        cat,label,path=lzp.item_meta(item,kind,refs[pos],target)
        if kind!="source_dataset":
            rp=dict(refs[pos])
            assert Path(path).resolve()==Path(item["img_path"]).resolve(),f"Target cache-order mismatch {pos}"
            assert str(rp.get("category",cat))==str(item.get("cls_name",cat))
        mask=lzp.mask_np(item)
        assert mask.shape==(518,518)
        masks[j]=mask
        metas.append(dict(position=j,original_cache_position=pos,category=cat,label=label,image=path))
        g=np.asarray(patch[pos])
        loc=np.asarray(local[pos])
        input_hasher.update(g.tobytes());input_hasher.update(loc.tobytes())
        raw=np.concatenate([g[None],loc],axis=0).astype(np.float32)
        with torch.no_grad():
            z=transformed(raw,transform,device)
            q=(z@weights+biases).view(5,37,37,len(entries)).permute(3,0,1,2)
            gl=F.interpolate(q[:,0:1],size=(74,74),mode="bilinear",align_corners=True)[:,0]
            ll=torch.zeros_like(gl)
            ll[:,:37,:37]=q[:,1];ll[:,:37,37:]=q[:,2]
            ll[:,37:,:37]=q[:,3];ll[:,37:,37:]=q[:,4]
            fused=(.5*gl+.5*ll).cpu().numpy()
        maps[j]=fused
        if j==0:
            # Actual real-feature path vs historical per-head scorer/fusion.
            for k in (0,len(specs)-1):
                old=hist.lrp_map(transform,specs[k],g,loc,device)
                err=float(np.max(abs(old-fused[k])))
                max_map_equiv=max(max_map_equiv,err)
                assert err<2e-6,(source,target,err)
        if j%50==0 or j==n-1:
            maps.flush();masks.flush()
            status(out,"PREDICTING_CACHED_FEATURES",source=source,target=target,images=j+1,total=n,
                seconds=round(time.time()-t,1),heads=len(entries))
    maps.flush();masks.flush()
    del maps,masks
    write_csv(dest/"TARGET_MANIFEST.csv",metas)
    write_json(complete,dict(source=source,target=target,n_images=n,models=signatures,
        input_feature_payload_sha256=input_hasher.hexdigest(),hash_scope="ordered selected global+local raw feature payloads, not npy header",
        map_dtype="float32",mask_policy="historical loader518",map_max_error_vs_historical=max_map_equiv,
        manifest_hash=file_hash(dest/"TARGET_MANIFEST.csv"),mask_hash=file_hash(dest/"masks518.npy"),
        maps_hash=file_hash(dest/"maps74.npy")))
    return dest


def evaluate_pair(out,source,target,entries,dest,smoke):
    maps=np.load(dest/"maps74.npy",mmap_mode="r")
    masks=np.load(dest/"masks518.npy",mmap_mode="r")
    metas=read_csv(dest/"TARGET_MANIFEST.csv")
    results=[]; cats=[]
    for head_index,entry in enumerate(entries):
        path=out/"evaluations"/(hist.pair_slug(source,target)+"__"+entry["id"]+".json")
        if path.exists():
            saved=json.loads(path.read_text(encoding="utf-8"))
            assert saved["checkpoint_hash"]==file_hash(entry["path"])
            results.append(saved["row"]);cats.extend(saved["categories"]);continue
        rows=[]
        for cat in sorted({r["category"] for r in metas}):
            ids=[i for i,r in enumerate(metas) if r["category"]==cat]
            yy=np.concatenate([np.asarray(masks[i]).reshape(-1) for i in ids])
            ss=np.concatenate([lzp.torch_resize(maps[i,head_index],masks[i].shape,align_corners=True).reshape(-1) for i in ids])
            pap=lzp.safe_ap(yy,ss)
            rows.append(dict(source=source,target=target,model_id=entry["id"],method=entry["saved"]["method"],
                seed=entry["saved"]["seed"],run_type=entry["role"],category=cat,P_AP=pap,n_images=len(ids),
                n_pixels=len(yy),n_positive=int(yy.sum())))
            del yy,ss
            status(out,"EVALUATING_P_AP",source=source,target=target,model=entry["id"],category=cat,P_AP=round(pap,6))
        values=[r["P_AP"] for r in rows if np.isfinite(r["P_AP"])]
        assert values and (smoke or len(values)==len(rows)),"Category validity changed"
        row=dict(source=source,target=target,method=entry["saved"]["method"],seed=entry["saved"]["seed"],
            run_type=entry["role"],lr=entry["saved"]["lr"],model_id=entry["id"],P_AP=float(np.mean(values)),
            valid_categories=len(values),total_categories=len(rows),checkpoint=str(entry["path"]),
            checkpoint_hash=file_hash(entry["path"]),map_cache=str(dest/"maps74.npy"),map_head_index=head_index,
            evaluator="historical_lzp.safe_ap / loader518 / bilinear align_corners=True / category-macro",
            GT_cache=str(dest/"masks518.npy"),config_hash=entry["saved"]["config_hash"],
            pool_hash=entry["saved"]["feature_array_hash"],batch_stream_hash=entry["saved"]["batch_stream_hash"])
        write_json(path,dict(row=row,categories=rows,checkpoint_hash=row["checkpoint_hash"]))
        results.append(row);cats.extend(rows)
        write_csv(out/"PAIRED_RAW_METRICS.partial.csv",results)
    return results,cats


def summaries(out,raw):
    primary=[r for r in raw if r["run_type"]=="PRIMARY_MATCHED_LR"]
    paired=[]
    for source,target in PAIRS:
        for seed in sorted({int(r["seed"]) for r in primary}):
            vals={r["method"]:float(r["P_AP"]) for r in primary if r["source"]==source and r["target"]==target and int(r["seed"])==seed}
            if set(vals)!=set(METHODS):continue
            paired.append(dict(source=source,target=target,seed=seed,**vals,
                CES_minus_Linear=vals["CES"]-vals["LinearHead"],CES_minus_Normalized=vals["CES"]-vals["NormalizedLinearHead"]))
    write_csv(out/"PAIRED_DELTAS.csv",paired)
    seed_means=[]
    for seed in sorted({r["seed"] for r in paired}):
        rr=[r for r in paired if r["seed"]==seed]
        if len(rr)!=4:continue
        seed_means.append(dict(seed=seed,**{k:float(np.mean([r[k] for r in rr])) for k in (*METHODS,"CES_minus_Linear","CES_minus_Normalized")}))
    write_csv(out/"SEED_LEVEL_MEANS.csv",seed_means)
    rows=[]
    for source,target in [*PAIRS,("FOUR_TRANSFER_MEAN","ALL")]:
        rr=seed_means if target=="ALL" else [r for r in paired if r["source"]==source and r["target"]==target]
        for k in (*METHODS,"CES_minus_Linear","CES_minus_Normalized"):
            a=np.asarray([r[k] for r in rr],float)
            if len(a):rows.append(dict(source=source,target=target,measure=k,n_seeds=len(a),mean=float(a.mean()),
                sample_sd=float(a.std(ddof=1)) if len(a)>1 else None,positive_seeds=int((a>0).sum()) if "minus" in k else "N/A"))
    write_csv(out/"PAIRED_SUMMARY.csv",rows)
    lines=["# Paired CES decision-head control","","New experiment, separate from frozen manuscript tables.",
        "All primary heads were trained anew on identical source features/initial effective logits/realized batch stream, at LR0.001. No historical CES checkpoint is reused as a primary repeat.",
        "Historical four-transfer loader-GT category-macro P-AP, not the native-GT six-target main comparison. SD is ddof=1 across seed-level values; four-transfer mean SD uses seed-level four-transfer means.","",
        "| Transfer | Seed | Linear | Normalized | CES | CES-Linear | CES-Normalized |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['source']} -> {r['target']} | {r['seed']} | {r['LinearHead']:.6f} | {r['NormalizedLinearHead']:.6f} | {r['CES']:.6f} | {r['CES_minus_Linear']:+.6f} | {r['CES_minus_Normalized']:+.6f} |")
    lines += ["","Source optimization and conditional source-only LR diagnostics are retained in SOURCE_OPTIMIZATION.csv, SOURCE_SELECTED_LR.csv and all per-step training logs. Primary matched-LR results and source-selected-LR sensitivity controls are separate, not substituted.",
        "No target P-AP decided diagnostic triggers, LR choices or checkpoint selection. Every model uses the last update. Same initialized logits do not imply matched gradients; raw norms and gradient norms are recorded."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(args):
    os.chdir(ROOT)
    out=Path(args.output).resolve()/("smoke" if args.smoke else "formal")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    config,ch=preflight(out,args.smoke)
    if args.stage=="preflight":return
    source_entries={}; transforms={}; eq=[]; trainlogs=[]; optrows=[]; selections=[]
    for source in ("Full MVTec","Full VisA"):
        tr=load_transform(source,device);transforms[source]=tr
        frozen_before={k:digest(v.detach().cpu().numpy()) for k,v in tr["align_model"].state_dict().items()} if tr["use_align"] else {}
        entries=[]
        for seed in ([0] if args.smoke else SEEDS):
            z,y,batches,probe,meta=prepare_source(out,source,seed,tr,device,args.smoke)
            eq.extend(equivalence(z,meta,device))
            for method in METHODS:
                saved,path=train_one(out,method,source,seed,z,y,batches,probe,meta,LR,"primary",ch,device)
                primary=dict(saved=saved,path=path,id=f"{method}_seed{seed}_primary",role="PRIMARY_MATCHED_LR")
                entries.append(primary);trainlogs.extend(saved["logs"])
                candidates=[(saved,path)]
                if method!="CES" and saved["diagnostic_trigger"] and not args.smoke:
                    for lr in LR_GRID:
                        if lr==LR:continue
                        sd,cp=train_one(out,method,source,seed,z,y,batches,probe,meta,lr,"source_lr_diagnostic",ch,device)
                        candidates.append((sd,cp));trainlogs.extend(sd["logs"])
                for sd,cp in candidates:
                    optrows.append(dict(source=source,seed=seed,method=method,lr=sd["lr"],run_type=sd["run_type"],
                        final_source_probe_loss=sd["final_probe"],recent_relative_loss_drop=sd["recent_relative_source_loss_drop"],
                        primary_diagnostic_trigger=saved["diagnostic_trigger"],updates=sd["realized_updates"],checkpoint=str(cp)))
                if method!="CES":
                    winner,wp=min(candidates,key=lambda a:(a[0]["final_probe"],a[0]["lr"]))
                    selections.append(dict(source=source,seed=seed,method=method,selected_lr=winner["lr"],
                        source_probe_loss=winner["final_probe"],checkpoint=str(wp),selection="source-only before any target evaluation"))
                    if wp!=path:
                        entries.append(dict(saved=winner,path=wp,id=f"{method}_seed{seed}_source_selected",role="SOURCE_SELECTED_LR_DIAGNOSTIC"))
            del z
        if tr["use_align"]:
            frozen_after={k:digest(v.detach().cpu().numpy()) for k,v in tr["align_model"].state_dict().items()}
            assert frozen_before==frozen_after
        source_entries[source]=entries
        write_json(out/f"{source.replace(' ','_')}_FROZEN_TRANSFORM_CHECK.json",dict(before=frozen_before,unchanged=True))
    write_csv(out/"INITIALIZATION_EQUIVALENCE.csv",eq)
    write_csv(out/"TRAINING_STEPS.csv",trainlogs)
    write_csv(out/"SOURCE_OPTIMIZATION.csv",optrows)
    write_csv(out/"SOURCE_SELECTED_LR.csv",selections)
    # A source-only seal exists before target data are accessed below.
    write_json(out/"SOURCE_PHASE_COMPLETE.json",dict(config_hash=ch,primary_checkpoints=18 if not args.smoke else 6,
        lr_choices_hash=file_hash(out/"SOURCE_SELECTED_LR.csv"),time=time.strftime("%Y-%m-%d %H:%M:%S"),
        target_results_accessed_for_selection=False))
    if args.stage=="train":return
    raw=[];percat=[]
    for source,target in PAIRS:
        entries=source_entries[source]
        dest=predict_pair(out,source,target,entries,transforms[source],device,args.smoke)
        rr,cc=evaluate_pair(out,source,target,entries,dest,args.smoke)
        raw.extend(rr);percat.extend(cc)
        write_csv(out/"PAIRED_RAW_METRICS.csv",raw);write_csv(out/"PER_CATEGORY_P_AP.csv",percat)
        summaries(out,raw)
    primary=[r for r in raw if r["run_type"]=="PRIMARY_MATCHED_LR"]
    assert len(primary)==(12 if args.smoke else 36)
    status(out,"SMOKE_COMPLETE" if args.smoke else "PAIRED_REPEAT_COMPLETE",primary_cells=len(primary),total_cells=len(raw))
    write_json(out/"OUTCOME.json",dict(status="SMOKE_COMPLETE" if args.smoke else "PAIRED_REPEAT_COMPLETE",
        primary_cells=len(primary),all_cells=len(raw),config_hash=ch,not_substituted_into_frozen_tables=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default=str(DEFAULT_OUT))
    p.add_argument("--device",default="cuda")
    p.add_argument("--stage",choices=("preflight","train","all"),default="all")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    try:run(args)
    except Exception as e:
        out=Path(args.output).resolve()/("smoke" if args.smoke else "formal")
        status(out,"FAILED",exception=repr(e))
        raise
