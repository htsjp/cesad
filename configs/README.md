# Configuration files

These are explicit path templates, not bundled trained assets. Edit paths to
your local approved source pools, checkpoints and target manifest. Relative paths
are resolved from the YAML location. No target hyperparameter selection occurs.

Training consumes prepared source features and true source labels. The supplied
batch plan must contain the actual30 update batches; it is not regenerated from a
seed name. Source preparation is a separate gated operation.

Inference chooses the correct source checkpoint and source transform. Frozen
ViT weights and the ancillary MRAD checkpoint are different required assets.
`cached_features` is optional, but when used its row/view order must match the
target manifest. No fallback search through historical directories is performed.
