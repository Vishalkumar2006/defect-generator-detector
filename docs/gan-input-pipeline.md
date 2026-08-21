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

### F1.2 censored-border geometry

Native-border contact is represented explicitly as independent `top`, `bottom`,
`left`, and `right` flags; the former boolean is derived from whether any side is
active. Horizontal flips exchange left/right and vertical flips exchange
top/bottom. Scaling retains the transformed side set, including multiple contacts.

A border-censored component is placed only on matching native edges. The selected
normal-background window must contain every required image edge, and the transformed
mask is aligned exactly to the corresponding native-valid boundary. Corner contacts
remain corners. A template requiring two opposite edges is rejected when no single
256×512 window can contain both. Incompatible component/background geometry is
reported as `no_compatible_target_background_or_window` with its exact underlying
reasons; it is never repaired by moving the defect into the interior.

Non-border components retain the configured eight-pixel native-valid margin on all
sides, preventing accidental border contact. Returned provenance contains source,
post-flip source, target-window native, and actual target contact sides. Every sample
also reports accidental contact violations, support pixels outside validity, and the
maximum composite difference outside support; all three invariants must be zero.

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

## Hash definitions

- `source_manifest_sha256` is the former `manifest_sha256`. It hashes canonical,
  source-order JSON for development-training rows only, using sample ID, official
  split, development split, image path, mask path, defect label, and image SHA-256.
- `split_sha256` hashes compact JSON for the source-order list of
  `[sample_id, has_defect]` pairs after the same training-only filter. Image paths,
  configuration, and generated template/background records are intentionally not
  part of this split-assignment identity.
- `gan_manifest_content_sha256` hashes sorted-key, compact JSON for the complete
  generated GAN manifest—including pipeline version, full configuration, templates,
  backgrounds, rejection records, data-boundary counters, source hash, and split
  hash—excluding only `gan_manifest_content_sha256` itself. Equal content reproduces
  the hash; any meaningful field change changes it. Online loading verifies it.

## Category-aware visual checks

The visualization command accepts `all`, `border`, `non-border`, `small-thin`,
`large`, and `narrow-background`. Each sheet includes the native-valid boundary,
source/transformed/target contact sides, all input/composite masks and images,
absolute difference, a defect zoom, and numerical invariant checks. A sibling
`.accounting.json` reports successful target contacts, compatibility-index
exclusions, actual retries, non-border placements, accidental contacts, and support
outside validity.

## F1.3 compatibility-indexed sampling

The F1.2 `border.accounting.json` value 1,113 did not describe a metadata prefilter.
It combined 1,111 actual normal-background candidates that the online dataset
loaded and passed to placement before six successful samples, plus two terminal
visualization-category index searches that exhausted all 1,772 backgrounds. Thus
the counter mixed candidate attempts with category-search failures: each terminal
entry counted as one even though it hid 1,772 attempted backgrounds. The audit
therefore implies 4,655 incompatible runtime background attempts in total
(`1,111 + 2×1,772`), not 1,113. The old online path was genuinely inefficient. It
shuffled every background and discovered incompatibility only after loading the
image and attempting placement.

F1.3 builds an in-memory metadata index grouped by native `(height,width)`. For the
selected template and deterministic transform attempt, it calculates the exact
nearest-neighbour mask bounding box, scaled source-content footprint, transformed
contact sides, feather radius, and eight-pixel non-border margin. Each native-size
group is checked once for required edge-window availability and exact horizontal
and vertical placement bounds. The sampler then chooses directly from the resulting
compatible background indices and loads only that selected image.

Candidates removed by this metadata query are reported as
`candidates_excluded_by_compatibility_index`; they are not runtime placement
failures or retries. Separate counters cover empty pools, deterministic transform
retries, and any unexpected placement retry after indexing. Provenance records pool
size, exclusions, attempts, template/background identity, sampling class, and the
successful side combination. Empty pools never change a border template into a
non-border template or remove required contact sides.

Border/non-border selection is explicit under `sampling`. The default
`border_fraction_mode=empirical` samples the two classes at their proportions in
the currently selected template set, then samples uniformly within the selected
class. An optional fixed fraction can be declared before sampling. Compatibility
difficulty therefore cannot silently suppress the intended border share.

`scripts/audit_gan_sampling.py` performs a requested deterministic training-only
online audit and atomically writes JSON and Markdown. It reports success rate,
runtime, attempt quantiles, failures by reason and side combination, index
exclusions, actual retries, empty pools, template/background utilization, contact
invariants, and border-distribution drift. It writes no generated images and never
constructs validation or official-test datasets.

## F1.4 native-edge symmetry and feasible transforms

The retained F1.3 audit (`reports/gan_inputs/sampling_audit.json` and `.md`) is the
diagnostic baseline: 983/1,000 samples succeeded, all 17 terminal failures involved
`left`, `left+right`, `top+left`, or `bottom+left`, while right-side placement was
common. This was a coordinate-origin bug, not evidence that left-censored defects
were intrinsically incompatible.

Short native windows had been copied at tensor x=0 with padding only on the right.
A right-contact source crop therefore retained reflected context to its right; an
horizontal flip moved that context before the now-left-contact mask. Aligning that
mask to x=0 required a negative content origin, so the index declared the state
empty. The equivalent right state had a non-negative origin and passed.

F1.4 uses symmetric reflection padding. For native width `Wn` in patch width `Wp`,
the inclusive valid interval is
`[(Wp-Wn)//2, (Wp-Wn)//2 + Wn - 1]`; height follows the same rule. Placement and
compatibility share those inclusive bounds. Crop origins and window offsets remain
zero-based, while maximum coordinates are inclusive. A left contact equals the
valid interval's minimum and a right contact equals its maximum. Neither is
defined by tensor x=0 or x=`Wp-1` when padding exists. Horizontal flips still swap
left/right exactly; top/bottom are unchanged. `left+right` states require the
transformed mask extent to equal a compatible target native-valid width and retain
both equalities.

Before online selection, scale space is partitioned at every point where rounded
tensor width or height changes. Each flip/scale interval is checked for retained
area, patch fit, contact constraints, the eight-pixel non-border margin, and a
non-empty compatible background pool. Sampling preserves the configured flip and
continuous-scale weights but chooses only among feasible intervals, then selects a
background directly from that interval's pool. Known-impossible states are index
exclusions, not online retries. Given the manifest, seed, and sample index, template,
transform, background, window, and provenance remain deterministic.

The F1.4 audit adds expected and observed target-side counts, terminal failures by
transformed side, templates with no feasible transform/pool, and exact comparisons
between each sampled transform and its horizontally mirrored counterpart. New
audit outputs use `sampling_audit_f1_4.json` and `.md`, preserving the failed F1.3
files unchanged.
