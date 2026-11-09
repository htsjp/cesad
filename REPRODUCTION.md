# Reproduction and asset requirements

## Environment

The audited source build used Python3.13.5, PyTorch2.9.1+cu130,
torchvision0.24.1+cu130, CUDA runtime13.0 and FP32 source/score computation.
`environment-record.json` records currently installed relevant package versions.
The upstream MRAD requirements are retained separately and are not a lockfile
for the newer CESAD runs. Do not install both requirement sets blindly.

## Read the core without datasets

```bash
python -m unittest -v test_core.py
```

This is a small mathematical smoke test, not training or a paper benchmark.

## Assets that are NOT uploaded

- Actual CLIP base weights (`ViT-L-14-336px.pt`). The ancillary
  `test_on_mvtec.pth` is NOT the complete visual base weight.
- Source datasets and exact image/mask manifests; paired physical patch plans.
- Certified FP32 patch/CLS features, normal PC1, E1/E2 auxiliary states,
  frozen E2 adapter, trained CES endpoints and source/target seal records.
- Original target datasets, released native masks and cache identity manifests.
- Third-party baseline repositories/checkpoints, obtainable separately under
  their own official release and license conditions.

## Gated original commands

From the extracted project, inspect help before any data operation:

```bash
python experiments/rebuild_traceable_visa_source.py --help
python experiments/source_build_traceable_visa.py --help
python experiments/evaluate_traceable_visa_targets.py --help
```

The source sequence is preflight -> fresh feature certification -> source
preparation approval -> source-build -> source seal -> target approval -> run.
Full dataset and checkpoint paths in historical scripts must be mapped explicitly.
Approvals bind code/input hashes; changing a path/code invalidates that identity.
Do not disable the gates, fabricate approval files or silently use old assets.

No training, feature extraction or target inference was performed to prepare
this repository. Existing experiment CSVs and manuscript files are not changed.
