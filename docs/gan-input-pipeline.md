# F1 training-only GAN input construction

Phase F1 implements data preparation only. There is no generator, discriminator,
GAN loss, training loop, detector training, or detector evaluation in this phase.

## Data boundary

Every source record must have both `official_split=train` and
`development_split=train`. Defect templates come only from defective training
images and their real binary masks. Backgrounds come only from normal training
images. Manifest construction filters before opening images, and both metadata
loading and online sampling repeat the hard split checks. Any validation or
official-test record raises an error. Validation predictions and known validation
failure cases are not inputs.

The metadata records training-only manifest and split SHA-256 values. It also
records zero validation rows, zero official-test rows, zero validation predictions,
and zero materialized generated images. Online sampling validates these counts.

## Geometry and templates

Inputs use native-aspect 256-pixel-wide by 512-pixel-high windows. Complete images
are never resized. Ordinary connected components receive centered windows with
available context. Long components use deterministic 50%-overlap windows; partial
components are retained with their coverage fraction. Border contact and positive
pixel counts are recorded. Edge padding has a separate false valid-region mask and
cannot become valid defect content. Normal patches must meet the configured native
valid-pixel fraction.

Online transforms are deliberately conservative: horizontal/vertical flips,
scale in `[0.9, 1.1]`, and translation wholly inside the target valid region. RGB
content uses bilinear interpolation while masks use nearest-neighbour interpolation
and are re-binarized. Minimum positive-pixel and retained-area checks reject unsafe
transforms. An optional robust median/MAD boundary-ring match has clamped gain and
offset. Feathering affects only the coarse composite; pixels outside its allowed
support remain bit-exact with the original normal background.

RGB GAN tensors use `[-1,1]`, independently of detector normalization. Condition,
support, and valid-region masks remain exactly `{0,1}`. The future generator output
can therefore use tanh-compatible scaling without changing mask semantics.

## Determinism and storage

The sample seed is a SHA-256-derived function of base seed, sample index, training
manifest hash, and split hash. Provenance contains source/background IDs, component
and window geometry, transform decisions, coverage/border state, colour parameters,
patch size, hashes, and pipeline version. Equal seed plus metadata reproduces all
tensors and provenance exactly.

Generated inputs stay in memory. The manifest command writes only JSON/Markdown
metadata, while the visualization command writes one small, fixed-seed contact
sheet rather than a bulk patch dataset.
