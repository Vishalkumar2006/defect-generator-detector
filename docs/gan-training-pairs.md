# G1.3 deterministic GAN training pairs

Phase G1.3 defines the data bridge between the F1.4 online compositor and a future
GAN trainer. It does not define an optimizer, training step, schedule, checkpoint,
or final loss weights, and it never uses detector validation or official-test data.

## Sample contract

`GANTrainingPairDataset` returns a frozen `GANTrainingSample` with these fields:

| Field | Shape | Dtype and range | Meaning |
|---|---:|---|---|
| `composite_image` | `[3,512,256]` | `float32`, `[-1,1]` | F1.4 coarse composite on an accepted normal background |
| `generator_mask` | `[1,512,256]` | `float32`, `[0,1]` | Fractional transformed feather support used for conditioning |
| `transformed_defect_alpha` | `[1,512,256]` | `float32`, `[0,1]` | Source binary defect mask transformed with the real-view sampling grid |
| `fake_discriminator_mask` | `[1,512,256]` | `float32`, `{0,1}` | G1.2 canonicalization of `generator_mask` |
| `real_image` | `[3,512,256]` | `float32`, `[-1,1]` | Genuine defective source window under the shared spatial transform |
| `real_discriminator_mask` | `[1,512,256]` | `float32`, `{0,1}` | Canonical real mask; bit-exact equal to the fake mask |
| `fake_valid_mask` | `[1,512,256]` | `float32`, `{0,1}` | Native-valid target-background pixels |
| `real_valid_mask` | `[1,512,256]` | `float32`, `{0,1}` | Native-valid source pixels after the shared transform |
| `real_valid_coverage` | `[1,512,256]` | `float32`, `[0,1]` | Continuous native-source coverage from the real-view sampling grid |
| `metadata` | n/a | JSON-compatible mapping | IDs, split, transform, placement, contacts, dimensions, padding, seed, and hashes |

The real and fake images are discriminator examples, not paired reconstruction
targets. In particular, this bridge does not define or calculate an image-space
real-versus-fake L1 loss.

## One transform for both branches

F1.4 remains the authority for template selection, feasible flip and scale,
compatibility indexing, target-window selection, and placement. Its compositor has
a backward-compatible detailed return mode. In that mode the already-selected
crop, flips, scale, and translation are also applied to the genuine defective
source window and source native-valid mask. The bridge does not resample placement
or reconstruct those parameters independently.

The F1 feathered conditioning mask feeds both discriminator-mask branches. Each
copy is canonicalized with the G1.2 threshold, and retrieval raises immediately
unless the two discriminator masks are bit-exact equal and non-empty. The source
binary defect mask is also transformed with the exact real-view grid as an
ordering check; it is not substituted for the generator's F1 conditioning mask.

## Determinism

The per-epoch sampler seed is a SHA-256 derivation of the base seed, bridge split,
effective epoch, and GAN-manifest content hash. F1.4 then derives every item from
that seed and the sample index with a local NumPy generator. Item retrieval does
not mutate Python, NumPy, or Torch global RNG state and is independent of DataLoader
worker scheduling.

`set_epoch(epoch)` changes deterministic training pairs. The monitor split always
uses effective epoch zero, so its samples remain fixed across epochs.

## Internal development-training split

`create_internal_gan_split` creates a deterministic 90/10 train/monitor partition
using seed 42 by default. Defect templates are grouped by defective source sample
ID, and normal backgrounds are grouped by background source ID. Neither kind of
source can cross the split. A deterministic greedy stratifier preserves
border/non-border and common contact combinations where feasible and reports rare
combinations that cannot appear in both partitions.

This monitor partition is only for GAN stability and overfitting checks. After all
hyperparameters are frozen, a later final model may be retrained on all
development-training data. That later training policy is outside G1.3.

## Continuous native-valid handling

The fake validity mask comes from the selected normal target window. The real
validity coverage starts from the binary source-window native-valid mask and is
sampled with the exact grid, bilinear interpolation, zero exterior, coordinate
convention, flips, scale, crop, and placement used for the transformed source
defect mask. Reflection padding remains visible RGB context but contributes zero
validity. The float coverage is retained, and every sample asserts pointwise that
transformed defect alpha is no greater than native-valid coverage (tolerance
`1e-6`).

The binary real validity mask uses `coverage > 1e-6`, an explicit exception to
the usual 0.5 canonical threshold. The stratified real-data audit showed that a
0.5 cutoff still excluded valid canonical boundary pixels because the established
F1 nearest-resized canonical mask and the continuous source grid have different
subpixel phase; an affected boundary pixel had native coverage 0.329. A strictly
positive cutoff includes only pixels with actual native-source contribution,
continues to reject pure reflection padding, and avoids repairing validity by
OR-ing in either the defect or its support.

## Aligned discriminator views

`prepare_aligned_discriminator_views` forms the joint mask as binary real validity
AND binary fake validity. It preserves each branch bit-for-bit inside the joint
region and replaces both branches with exactly `0.0` outside it. The discriminator
therefore sees aligned RGB views plus the shared canonical discriminator mask;
the G1.1 architecture and channel count are unchanged. The original training-pair
tensors are not mutated, and the generator still receives the original composite
and conditioning mask. Canonical containment in real, fake, and joint validity is
strict. Feather support and the 12-pixel refinement halo may extend beyond joint
validity and are reported separately rather than relabelled as native.

## Audits

The audit checks support containment, localized-logit interaction with padding,
real/fake valid-fraction asymmetry, contact distributions, utilization, split
disjointness, mask synchronization, and replay determinism. It writes only a JSON
report, Markdown report, and one representative contact sheet—not a generated
training dataset:

```powershell
python scripts/audit_gan_training_pairs.py --config configs/gan_training_pairs.json
```

The G1.3b corrective audit covers all ten target-contact combinations, verifies
continuous alpha/coverage ordering and strict canonical containment, prepares
aligned views, and checks padding equality and adversarial-gradient gating:

```powershell
python scripts/audit_gan_validity_alignment.py --config configs/gan_validity_alignment.json
```

The command is hard-guarded by the F1 manifest validation: every row must have
official split `train` and development split `train`, templates must be defective,
backgrounds must be normal, content hashing must verify, and recorded
validation/official-test access must be zero.
