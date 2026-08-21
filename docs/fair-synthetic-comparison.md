# Frozen E1.2 BF16 reference and future synthetic-data comparison

E1.2 is the frozen stabilized BF16 real-only reference experiment. Its architecture, valid-region
preprocessing, loss, optimizer, scheduler, early stopping, sampler, augmentation,
seed, optimizer-attempt budget, checkpoint selection, and validation finalization
must not be retuned after seeing a synthetic-data result.

## Paired protocol

The future real-only and real-plus-synthetic runs must use the same:

- GroupNorm U-Net initialization and seed;
- native-geometry 256 x 672 canvas, image normalization, image/mask padding, and
  valid-region loss and metrics;
- synchronized training-only horizontal and vertical flips;
- AdamW optimizer at initial learning rate 0.0003, learning-rate schedule,
  physical batch size, BF16/no-GradScaler policy, maximum gradient norm 1.0,
  early stopping, and best-checkpoint rule;
- number and ordering of attempted optimizer updates, including the same behavior
  for a skipped numerical attempt;
- validation set, threshold candidates, fixed diagnostic IDs, and report metrics.

The synthetic condition changes one variable only: a predeclared fraction of the
defective branch of the deterministic weighted replacement sampler is replaced by
synthetic defective samples. It does not add optimizer attempts. Normal-branch
sampling remains real and unchanged. The replacement fraction and synthetic pool
are fixed before either paired run begins, and the data-source decision for each
defective draw is reproducible from the shared seed.

Synthetic images and masks must obey the same tensor contract, canvas geometry,
normalization, augmentation, binary-mask requirement, and valid-region semantics
as real samples. GAN training, GAN model selection, synthetic generation, and
synthetic quality filtering may use development-training data only. A GAN or its
outputs must never enter development validation or official test data.

## Evaluation isolation

Both paired runs select checkpoints by minimum development-validation combined
BCE plus Dice loss. Each reloads its best checkpoint and performs exactly one
validation threshold sweep from 0.05 through 0.95. Development validation is not
used to tune the synthetic replacement fraction or regenerate synthetic data.

The official test split remains untouched until both configurations, checkpoints,
and selected validation thresholds are frozen. The only final test comparison is:

1. frozen E1.2 stabilized BF16 real-only baseline;
2. frozen real-plus-synthetic condition.

No GAN, ablation, failed run, or intermediate checkpoint is evaluated on the
official test split. Any divergence in attempted optimizer updates invalidates the
pair and requires both sides to be rerun under an identical attempt schedule.
