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

### F1.1 normal-background eligibility audit

The F1 threshold of `0.9` accepted only 583 of 1,772 development-training normal
images (32.90%). This was an acquisition-width filter: all audited normal heights
are 602–660 pixels, while native widths are 184–241 pixels and therefore below the
256-pixel patch width.

For a native image of height `h` and width `w`, the maximum achievable native
fraction is `min(1,h/512) * min(1,w/256)`. Development-training normal-image
fractions have minimum 0.71875, p01 0.8046875, p05 0.8671875, median 0.89453125,
p95 0.9140625, and maximum 0.94140625. The selected threshold is therefore
`0.71875` (`184/256`): it is the strict audited geometric minimum and includes all
1,772 legitimate normal training images while still marking every reflected
padding pixel invalid. The complete width histogram, quantiles, source hashes, and
threshold comparisons are frozen in
`reports/gan_input_design/normal_valid_fraction_audit.json`.

The expected rebuilt manifest has 209 defective training images, 235 connected
components, 232 accepted components, and three components rejected for fewer than
eight positive pixels. It has 232 full windows, zero partial windows, and zero
components requiring overlapping windows. Individual connected components have a
maximum width of 231 and height of 423, so each accepted component fits within
256×512. Earlier image-level defect bounding boxes could exceed a patch because
they spanned spatially separated components or requested context; that does not
make an individual component partial. Desired context is reduced at native borders
when unavailable, without cropping the component. Expected normal-background
counts are 1,772 accepted, zero rejected, and 3,544 available vertical windows.

Online transforms are deliberately conservative: horizontal/vertical flips,
scale in `[0.9, 1.1]`, and translation wholly inside the target valid region. RGB
content uses bilinear interpolation while masks use nearest-neighbour interpolation
and are re-binarized. Minimum positive-pixel and retained-area checks reject unsafe
transforms. An optional robust median/MAD boundary-ring match has clamped gain and
offset. Feathering affects only the coarse composite; pixels outside its allowed
support remain bit-exact with the original normal background. Both the transformed
defect mask and feathered support are intersected with the target native-valid mask;
reflected padding may be visible as background context but cannot receive defect
content.

RGB GAN tensors use `[-1,1]`, independently of detector normalization. Condition,
support, and valid-region masks remain exactly `{0,1}`. The online sample returns
the valid-region mask as a first-class tensor for the future GAN and discriminator.
The future generator output can therefore use tanh-compatible scaling without
changing mask semantics.

## Determinism and storage

The sample seed is a SHA-256-derived function of base seed, sample index, training
manifest hash, and split hash. Provenance contains source/background IDs, component
and window geometry, transform decisions, coverage/border state, colour parameters,
patch size, hashes, and pipeline version. Equal seed plus metadata reproduces all
tensors and provenance exactly.

Generated inputs stay in memory. The manifest command writes only JSON/Markdown
metadata, while the visualization command writes one small, fixed-seed contact
sheet rather than a bulk patch dataset.
