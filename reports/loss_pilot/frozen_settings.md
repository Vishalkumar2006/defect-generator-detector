# Frozen settings for the validation-only loss pilot

The only experimental variable is `loss.pos_weight`, screened at 1, 5, 10, and 20. Every candidate reinitializes from seed 42 and uses the same data order where deterministic PyTorch execution permits it.

| Setting | Frozen value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Scheduler | none |
| BCE / Dice coefficients | 1.0 / 1.0 |
| Physical batch size | 4 |
| Mixed precision | enabled |
| Sampler | deterministic weighted replacement; target 50% defective |
| Normalization | GroupNorm, at most 8 groups |
| Base channels | 32 |
| Canvas | 256 × 672 |
| Padding | symmetric reflection for images; constant zero for masks |
| Detector normalization | Phase C development-training-only mean and standard deviation |
| Augmentation | disabled |
| Seed | 42 |
| Workers / pinned memory | 0 / enabled |
| Training / validation rows | 1,981 / 350 |
| Official test rows loaded | 0 |

The scheduler is explicitly `none` because the stable smoke implementation used no scheduler. The eight-epoch maximum is a phase-required duration applied equally to every candidate, not a candidate-specific hyperparameter change.
