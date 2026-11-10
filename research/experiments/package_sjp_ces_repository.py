"""Create a source-only private backup without changing original research assets."""
from pathlib import Path
import ast
import csv
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parent/'sjp_ces'
SOURCE=DEST/'source'
DELIVERY=DEST/'github_upload'
BLOCKED={'.git','.cache','cache','checkpoints','data','results','logs','reports','figures',
         '_external','external_sources','__pycache__','.venv','venv','.idea'}
ALLOWED={'.py','.yaml','.yml','.sh','.toml','.ini','.cfg'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(path,names):
    text=path.read_text(encoding='utf-8-sig');lines=text.splitlines()
    nodes={n.name:n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    pieces=[];records=[]
    for name in names:
        node=nodes[name]
        start=min([node.lineno,*[d.lineno for d in getattr(node,'decorator_list',[])]])
        code='\n'.join(lines[start-1:node.end_lineno])
        pieces.append(code)
        records.append(dict(file=str(path.relative_to(ROOT)),symbol=name,line=start,
                            original_file_sha256=sha(path),segment_sha256=hashlib.sha256(code.encode()).hexdigest()))
    return '\n\n\n'.join(pieces),records


def main():
    for p in (DEST,SOURCE,DELIVERY):p.mkdir(parents=True,exist_ok=True)
    files=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(ROOT)
        if any(part in BLOCKED for part in rel.parts):continue
        if path.suffix.lower() not in ALLOWED:continue
        if path.stat().st_size>2_000_000:raise RuntimeError(f'Unexpected large source file: {rel}')
        if path.is_symlink():raise RuntimeError(f'No symlink upload: {rel}')
        files.append(path)
    for special in ('AnomalyCLIP_lib/bpe_simple_vocab_16e6.txt.gz',):files.append(ROOT/special)
    patterns=[('github_token',r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{35,})\b'),
              ('openai_key',r'\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b'),
              ('aws_key',r'\bAKIA[0-9A-Z]{16}\b'),
              ('private_key',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
    findings=[];inventory=[];syntax=[]
    for path in files:
        rel=path.relative_to(ROOT)
        if path.suffix.lower() in ALLOWED:
            text=path.read_text(encoding='utf-8-sig')
            for kind,pattern in patterns:
                if re.search(pattern,text):findings.append(dict(path=str(rel),kind=kind))
            if path.suffix=='.py':
                try:ast.parse(text,filename=str(rel))
                except SyntaxError as exc:syntax.append(dict(path=str(rel),line=exc.lineno,error=exc.msg))
        target=SOURCE/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        assert sha(path)==sha(target)
        inventory.append(dict(path=rel.as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    if findings:raise RuntimeError('Secret-pattern matches (values withheld): '+json.dumps(findings))
    # Preserve upstream attribution without granting a new blanket code license.
    (SOURCE/'docs').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'README.md',SOURCE/'docs/UPSTREAM_MRAD_README.md')
    shutil.copy2(ROOT/'LICENSE',SOURCE/'docs/UPSTREAM_MRAD_LICENSE.txt')
    shutil.copy2(ROOT/'requirements.txt',SOURCE/'docs/UPSTREAM_REQUIREMENTS.txt')

    head,headtrace=extract(ROOT/'experiments/phase_ces_linear_head_control.py',
        ['unit_np','state_pair_from_m_alpha_v','ces_weight_np','HeadSpec','LinearHead',
         'NormalizedLinearHead','CESHead','initial_geometry','ntr_loss_from_logits'])
    loss,losstrace=extract(ROOT/'experiments/phase_xlii_patch_specialized_adaptation.py',
        ['dice_loss','balanced_loss'])
    core='''"""Exact source-extracted CES/NTR heads and losses for inspection.

No encoder, target data, asset fallback or training is run at import time.
Canonical historical implementations remain inside sjp_ces_source.zip.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math
import numpy as np
import torch
import torch.nn.functional as F
SEED=0
RHO=0.10

'''+head+'\n\n\n'+loss+'\n'
    (DELIVERY/'cesad_core.py').write_text(core,encoding='utf-8')
    ast.parse(core)
    (DELIVERY/'CORE_EXTRACTION_TRACE.json').write_text(json.dumps(headtrace+losstrace,indent=2),encoding='utf-8')
    readme='''# sjp_ces - CESAD private research source archive

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
'''
    method='''# Method implementation guide

## CES scoring

For normalized patch features z, v=normalize(v_raw),
s_A=normalize(m+alpha*v), s_N=normalize(m-alpha*v), q=z^T(s_A-s_N).
There is no independent bias in CES. m and alpha are buffers; v_raw is trainable.
Do not add a plotting-only temperature, sigmoid or score offset.

## Source initialization and preparation

MVTec's direct path normalizes frozen patch features, averages the normal and
anomaly patch pools separately, normalizes those means, then sets
m=(s_A+s_N)/2, d=(s_A-s_N)/2, alpha=||d||, v0=normalize(d).
Historical formal MVTec supervision and nearest-mask matched controls are
different documented runs; do not merge their metrics or checkpoints.

For the certified VisA path, raw source image/patch identities are bound before
training. Source-normal PC1 is removed and features are L2-normalized. True-mask
and circularly shifted-mask auxiliary state pairs are optimized at Stage I.
The rank-4 residual adapter is trained with the shifted supervision at Stage II:
normalize(x+B(A(x))), A:1024->4, B:4->1024, no bias or activation.
The frozen E2 adapter is applied after PC1 removal/normalization, then normalized.

Final VisA CES geometry uses Stage-I true-mask states for d and alpha, and the
Stage-I shifted-mask midpoint for m. E2's trained adapter is frozen. Final
CES/NTR uses **true physical source masks**, not shifted labels. The certified
pipeline preserves update0 and update30, with update30 as the trained endpoint.

## NTR and base loss

Inside each training batch, select the ceil(rho*n_normal) largest detached
normal logits, rho=0.10, then minimize mean softplus of those live logits.
This is batch-normal top-k mean, recomputed each batch, not category retrieval.
The final loss adds NTR to balanced BCE and sigmoid Dice, with the archived
training code controlling the NTR-enabled/disabled branch. Calling the extracted
NTR helper with rho=0 does not disable it (the helper clamps k to at least one).

Balanced BCE weights positives and negatives to contribute equally in aggregate.
Dice is 1-(2*sum(sigmoid(q)*y)+1)/(sum(sigmoid(q))+sum(y)+1).
Refer to the exact extracted functions rather than replacing their reductions.

## Linear controls

LinearHead trains w and an independent zero-initialized scalar bias.
NormalizedLinearHead trains u, uses frozen c0, and scores c0*z^T normalize(u).
The control initialization is constructed from CES's initial effective weight;
do not reinterpret these as independently randomly initialized baselines.

## LRP and external evaluation

Fixed LRP uses one global and four nonoverlapping 2x2 local views, each encoded
at518x518 with the same frozen visual path. Four37x37 local logits form a74x74
map; resize the global map to74x74 and fuse0.5 global +0.5 local.
External native-GT resizing is float32 bilinear align_corners=True. The frozen
unified evaluator preserves exact score ties for AP/AUROC/F1. AUPRO uses the
archived200-threshold policy and FPR cap0.30. Image scoring uses top55 of74x74
before native-resolution pixel resizing. No per-image min-max normalization is
used for metric evaluation; visual display normalization is not a model change.
'''
    code_map='''# Code map

All paths below are relative to the extracted `sjp_ces_source/` directory.

| Role | Source |
|---|---|
| Visual outputs and feature normalization | phase_xa/official_task_forward.py |
| Attention-aware visual forward | phase_x/gradient_visual_forward.py |
| Fixed1/3/5 neighborhood aggregation | models/mlp.py: average_neighbor |
| Frozen modified CLIP implementation | AnomalyCLIP_lib/ |
| Physical source preflight/certification | experiments/rebuild_traceable_visa_source.py |
| Fresh FP32 source extraction | experiments/ (see manifest names containing traceable/source/certification) |
| Certified source build and30-update CES | experiments/source_build_traceable_visa.py |
| Certified paired endpoint target evaluation | experiments/evaluate_traceable_visa_targets.py |
| CES/linear heads and NTR helper | experiments/phase_ces_linear_head_control.py |
| Exact BCE/Dice helper | experiments/phase_xlii_patch_specialized_adaptation.py |
| Residual adapter definition | experiments/phase_xliii_adapter_head_decomposition.py |
| Circular mask shift definition | experiments/phase_spatial_supervision_transfer_audit.py |
| Frozen final evaluator | experiments/final_unified_evaluation.py |
| Repaired component source-pool comparison | experiments/repair_component_ablation_original_cache.py |
| Aligned head/capacity controls | experiments/repair_linear_capacity_aligned.py |
| Six-metric ablation completion | experiments/evaluate_aligned_ablation_six_metrics.py |
| MVTec exact-VisA-style source control | experiments/audit_repair_mvtec_exact_visa_recipe.py |
| Cache-only recipe metric completion | experiments/complete_mvtec_recipe_six_metrics.py |
| Figure scripts | visualization/ces_initialization_3d/ |

Keep dependencies in their original locations. Files named `phase_*` include
historical/exploratory models, not a recommendation to run every experiment.
Old loader/carrier paths cannot retrospectively certify historical pairing.
Current certified preparation is identified by explicit manifests, asset hashes
and source/target approval gates, not only by filenames.

`cesad_core.py` at repository root is a mechanically extracted reading/testing
surface. Its exact symbol provenance is in CORE_EXTRACTION_TRACE.json. The
historical modules inside the archive remain the authoritative experiment code.
'''
    reproduce='''# Reproduction and asset requirements

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
'''
    notice='''# Third-party and publication notes

This research workspace was developed on an MRAD-based project and includes a
modified CLIP/AnomalyCLIP-style visual implementation. It is not pure unmodified
OpenAI CLIP. Preserve upstream notices and cite inherited components separately
from CES/NTR/LRP contributions.

The original MRAD README, requirements and MIT notice are preserved under
`sjp_ces_source/docs/UPSTREAM_*` in the archive. No new blanket license is
assigned to this private research backup. Review per-component rights before
public release or redistribution of datasets/weights.

External baseline repositories are deliberately excluded. This package does not
claim ownership of AnomalyCLIP, MRAD, CLIP or downloaded third-party code.
Local figure scripts may rely on non-distributed plotting QA helpers; they are
preserved for provenance and are not advertised as standalone installations.
'''
    gitignore='''# Source-only repository: no training assets or personal credentials
__pycache__/
*.py[cod]
.env
.env.*
!.env.example
.venv/
venv/
.idea/
.vscode/
data/
datasets/
checkpoints/
cache/
.cache/
results/
logs/
*.pt
*.pth
*.ckpt
*.npy
*.npz
*.h5
*.hdf5
*.pem
*.key
*.docx
*.log
'''
    test='''"""Synthetic mathematical smoke tests only; not experimental evidence."""
import unittest
import numpy as np
import torch
from cesad_core import CESHead, LinearHead, NormalizedLinearHead, ntr_loss_from_logits

class CoreTests(unittest.TestCase):
    def test_matched_logits_and_frozen_geometry(self):
        rng=np.random.default_rng(0)
        sn=rng.normal(size=1024).astype(np.float32);sn/=np.linalg.norm(sn)
        sa=rng.normal(size=1024).astype(np.float32);sa/=np.linalg.norm(sa)
        m=(sa+sn)/2;d=(sa-sn)/2;a=float(np.linalg.norm(d));v=d/a
        head=CESHead(m,a,v)
        z=torch.nn.functional.normalize(torch.from_numpy(rng.normal(size=(16,1024)).astype(np.float32)),dim=-1)
        w=torch.nn.functional.normalize(head.m+head.alpha*torch.nn.functional.normalize(head.v,dim=0),dim=0)-torch.nn.functional.normalize(head.m-head.alpha*torch.nn.functional.normalize(head.v,dim=0),dim=0)
        linear=LinearHead(w.detach().numpy());normalized=NormalizedLinearHead(w.detach().numpy(),float(w.norm()))
        self.assertLess(float((head(z)-linear(z)).abs().max().detach()),1e-6)
        self.assertLess(float((head(z)-normalized(z)).abs().max().detach()),1e-6)
        self.assertEqual(sum(p.numel() for p in head.parameters()),1024)
        self.assertFalse(head.m.requires_grad);self.assertFalse(head.alpha.requires_grad)
    def test_ntr_real_top10_percent(self):
        scores=torch.arange(20,dtype=torch.float32,requires_grad=True)
        labels=torch.zeros(20)
        loss=ntr_loss_from_logits(scores,labels,.1)
        self.assertTrue(torch.equal(loss,torch.nn.functional.softplus(scores[-2:]).mean()))
        loss.backward();self.assertEqual(int((scores.grad!=0).sum()),2)

if __name__=='__main__':unittest.main()
'''
    for name,text in {'README.md':readme,'METHOD_IMPLEMENTATION.md':method,'CODE_MAP.md':code_map,
        'REPRODUCTION.md':reproduce,'THIRD_PARTY_NOTICES.md':notice,'.gitignore':gitignore,
        'test_core.py':test,'requirements-core.txt':'numpy>=1.26\ntorch>=2.0\n'}.items():
        (DELIVERY/name).write_text(text,encoding='utf-8')
    environment={}
    for package in ('torch','torchvision','numpy','scipy','scikit-learn','scikit-image',
                    'pandas','Pillow','matplotlib','ftfy','regex','timm','tqdm','PyYAML'):
        try:environment[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:environment[package]='NOT_INSTALLED'
    (DELIVERY/'environment-record.json').write_text(json.dumps(dict(python=sys.version,packages=environment),indent=2),encoding='utf-8')
    with (DELIVERY/'SOURCE_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(inventory)
    scan=dict(secret_pattern_matches=findings,syntax_findings=syntax,selected_source_files=len(inventory),
        excluded_directories=sorted(BLOCKED),core='exact AST source-segment extraction; no semantic rewriting',
        originals_changed=False)
    (DELIVERY/'SOURCE_CHECKS.json').write_text(json.dumps(scan,indent=2),encoding='utf-8')
    for name in ('README.md','METHOD_IMPLEMENTATION.md','CODE_MAP.md','REPRODUCTION.md','THIRD_PARTY_NOTICES.md'):
        shutil.copy2(DELIVERY/name,SOURCE/name)
    archive=DELIVERY/'sjp_ces_source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(SOURCE.rglob('*')):
            if path.is_file():z.write(path,Path('sjp_ces_source')/path.relative_to(SOURCE))
    print(json.dumps(dict(local_project=str(DEST),upload_folder=str(DELIVERY),
        source_files=len(inventory),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
        syntax_findings=syntax,secret_matches=len(findings)),indent=2))


if __name__=='__main__':main()
