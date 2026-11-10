# Method implementation guide

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
