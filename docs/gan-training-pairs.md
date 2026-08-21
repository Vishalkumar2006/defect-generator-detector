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
| `fake_discriminator_mask` | `[1,512,256]` | `float32`, `{0,1}` | G1.2 canonicalization of `generator_mask` |
| `real_image` | `[3,512,256]` | `float32`, `[-1,1]` | Genuine defective source window under the shared spatial transform |
| `real_discriminator_mask` | `[1,512,256]` | `float32`, `{0,1}` | Canonical real mask; bit-exact equal to the fake mask |
| `fake_valid_mask` | `[1,512,256]` | `float32`, `{0,1}` | Native-valid target-background pixels |
| `real_valid_mask` | `[1,512,256]` | `float32`, `{0,1}` | Native-valid source pixels after the shared transform |
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

The fractional mask emitted by that same operation feeds both branches. Each copy
is canonicalized with the G1.2 threshold, and retrieval raises immediately unless
the two discriminator masks are bit-exact equal and non-empty.

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

## Native-valid handling and audit

The fake validity mask comes from the selected normal target window. The real
validity mask is the source-window validity mask under the shared spatial
transform. Reflection padding remains visible context but is explicitly marked
non-native. Border and multi-side defects remain in the pool.

The audit checks support containment, localized-logit interaction with padding,
real/fake valid-fraction asymmetry, contact distributions, utilization, split
disjointness, mask synchronization, and replay determinism. It writes only a JSON
report, Markdown report, and one representative contact sheet—not a generated
training dataset:

```powershell
python scripts/audit_gan_training_pairs.py --config configs/gan_training_pairs.json
```

The command is hard-guarded by the F1 manifest validation: every row must have
official split `train` and development split `train`, templates must be defective,
backgrounds must be normal, content hashing must verify, and recorded
validation/official-test access must be zero.
