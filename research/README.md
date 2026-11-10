# sjp_ces - CESAD private research source archive

CESAD: Compact Experience States for Strict Cross-Dataset Zero-Shot Anomaly Detection.

This repository is a **private source-only backup**, not a self-contained pretrained
release. It preserves the research directory structure and distinguishes certified
source reconstruction from historical experiments. No model was trained for this upload.

## Start here

- `cesad_core.py`: exact source-extracted CES, LinearHead, NormalizedLinearHead,
  NTR and BCE/Dice implementations, readable without the experiment import graph.
- `METHOD_IMPLEMENTATION.md`: mathematical definitions and actual preparation order.
- `CODE_MAP.md`: canonical entry points, dependencies and historical-branch boundaries.
- `REPRODUCTION.md`: environment, external assets and safe reproduction requirements.
- `sjp_ces_source.zip`: complete code-only snapshot with original relative imports.
- `SOURCE_MANIFEST.csv`: every selected original source path, byte size and SHA256.
- `CORE_EXTRACTION_TRACE.json`: source file/line/hash for every extracted core symbol.

## Restore the full project

```bash
python -m zipfile -e sjp_ces_source.zip .
cd sjp_ces_source
```

The archive contains `experiments/`, the frozen visual implementation, loaders,
model helpers, tests, configurations and scientific plotting scripts. Keep these
relative paths intact. The hundreds of historical experiments are **not** all
components of the final CESAD method.

## Final method

Frozen visual features -> source-specific fixed preparation -> constrained CES
direction learning with BCE + Dice + NTR -> fixed global/local LRP.

The final scorer uses two normalized 1024-dimensional states. Only one
1024-dimensional raw direction is optimized during the final CES stage; the
construction center, separation scale and source feature transform are frozen.
PC1 and the residual adapter are additional frozen source assets, not counted as
the compact scorer's optimized parameters.

## Strict protocol

MVTec -> VisA; VisA -> MVTec, BTAD, MPDD, DTD-Synthetic and DAGM.
No target training, target normal library, target statistics or target-driven
checkpoint selection is introduced by this package. Ground truth is for evaluation.
Pixel results use native paired GT and category-macro aggregation. Image results
use a 74x74 score grid and top-1% mean (55 positions), category-macro aggregation.

## Important limitations

Historical source scripts contain machine-specific D:/E: paths and review gates.
This is an auditable archive, not a claim that a fresh clone can reproduce paper
results without datasets, weights, source manifests and explicit asset mapping.
Do not remove certification gates or silently substitute older caches.
The directly computed initialization used for figures is not a trained no-NTR model.

## Excluded deliberately

Datasets/masks, checkpoints, feature banks, anomaly maps, optimizer files, cached
responses, manuscript documents, logs, credentials and downloaded baseline repos.
Keep the repository **Private** until provenance, licenses and a public asset
distribution plan have been reviewed. Existing third-party notices are preserved.
