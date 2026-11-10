"""Final-head training from explicitly paired, already prepared source pools."""

from pathlib import Path
import numpy as np
import torch
from .config import asset, identity, save_json
from .geometry import load_head
from .heads import balanced_loss, ntr_loss_from_logits


def run(cfg, output, device, dry_run=False):
    paths = {
        k: asset(cfg, k)
        for k in ("features", "labels", "batch_indices", "geometry_checkpoint")
    }
    x = np.load(paths["features"], mmap_mode="r")
    y = np.load(paths["labels"], mmap_mode="r")
    batches = np.load(paths["batch_indices"], mmap_mode="r")
    updates = int(cfg["updates"])
    if updates != 30 or x.ndim != 2 or x.shape[1] != 1024 or y.shape != (len(x),):
        raise ValueError("Wrong source pool/budget")
    if (
        batches.ndim != 2
        or len(batches) != updates
        or batches.min() < 0
        or batches.max() >= len(x)
    ):
        raise ValueError("Invalid actual batch plan")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Expected true binary source labels")
    if cfg["ntr"] and not 0 < float(cfg["rho"]) <= 1:
        raise ValueError("Enabled NTR requires 0 < rho <= 1; use ntr:false to disable")
    head, saved = load_head(paths["geometry_checkpoint"], device)
    if int(saved.get("step", saved.get("update", 0))) != 0:
        raise ValueError("Training must start from the initialization endpoint")
    if saved.get("source") not in (None, cfg["source"]):
        raise ValueError("Source checkpoint/config mismatch")
    inputs = {key: identity(path) for key, path in paths.items()}
    print(
        f"Source preflight: {len(x)} prepared patches,30 updates, batch={batches.shape[1]}"
    )
    if dry_run:
        return
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be empty; no silent overwrite/resume")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    initial_m = head.m.detach().clone()
    initial_alpha = head.alpha.detach().clone()
    logs = []

    def checkpoint(step):
        torch.save(
            dict(
                format="cesad_head_v1",
                source=cfg["source"],
                seed=int(cfg["seed"]),
                step=step,
                model=head.state_dict(),
                optimizer=optimizer.state_dict(),
                source_inputs=inputs,
                actual_batch_indices=np.asarray(batches[:step]).copy(),
                losses=logs,
            ),
            output / f"update_{step:03d}.pt",
        )

    save_json(output / "config.json", cfg)
    save_json(output / "source_inputs.json", inputs)
    checkpoint(0)
    for step, ids in enumerate(batches, 1):
        xx = torch.from_numpy(np.asarray(x[ids], np.float32)).to(device)
        yy = torch.from_numpy(np.asarray(y[ids], np.float32)).to(device)
        q = head(xx)
        base, bce, dice = balanced_loss(q, yy)
        ntr = (
            ntr_loss_from_logits(q, yy, float(cfg["rho"]))
            if cfg["ntr"]
            else q.new_zeros(())
        )
        loss = base + ntr
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite source loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(head.v).all():
            raise RuntimeError("Nonfinite source direction")
        assert torch.equal(initial_m, head.m) and torch.equal(initial_alpha, head.alpha)
        logs.append(
            dict(
                step=step,
                loss=float(loss.detach()),
                bce=float(bce),
                dice=float(dice),
                ntr=float(ntr.detach()),
            )
        )
        checkpoint(step)
        print(logs[-1], flush=True)
    save_json(output / "training_log.json", logs)
