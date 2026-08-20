# D1 positive-weight pilot

> **PROVISIONAL — AMP AUDIT REQUIRED:** D1 automatically retried non-finite fp16 forwards in fp32. The historical metrics below are unchanged and describe the runs that actually executed, but the candidates followed different hybrid-precision paths. No positive weight is final until the manual numerical audit is reviewed.

Validation-only screening on all 1,981 development-training and 350 validation images. The official test dataset was not constructed or evaluated.

| pw | best ep | val loss | Dice@.5 | def Dice@.5 | P@.5 | R@.5 | normal FP@.5 | global t | global Dice | global normal FP | def t | best def Dice | def normal FP | train sec | total sec | VRAM GiB | failures | stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7 | 0.0828 | 0.5569 | 0.5410 | 0.8567 | 0.4125 | 0.0032 | 0.15 | 0.5619 | 0.0032 | 0.6 | 0.5417 | 0.0032 | 4332.5 | 4361.8 | 5.37 | 1349 | obvious validation divergence; substantial AMP numerical fallback activity |
| 5 | 8 | 0.0756 | 0.7744 | 0.6012 | 0.7517 | 0.7985 | 0.0000 | 0.8 | 0.7887 | 0.0000 | 0.9 | 0.6236 | 0.0000 | 3267.9 | 3311.6 | 3.27 | 1272 | large validation-loss regime shift; substantial AMP numerical fallback activity |
| 10 | 7 | 0.1527 | 0.7401 | 0.6092 | 0.6531 | 0.8538 | 0.0319 | 0.65 | 0.7634 | 0.0319 | 0.9 | 0.6456 | 0.0192 | 3559.6 | 3609.2 | 3.27 | 2056 | obvious validation divergence; substantial AMP numerical fallback activity |
| 20 | 8 | 0.9578 | 0.5615 | 0.7368 | 0.4300 | 0.8089 | 0.5751 | 0.85 | 0.6803 | 0.3450 | 0.2 | 0.7547 | 0.7604 | 3354.2 | 3403.9 | 3.27 | 1505 | no obvious instability; substantial AMP numerical fallback activity |

The CSV and JSON contain the complete threshold-0.5 metrics, both independently selected threshold operating points, runtime, memory, failures, and normal-image behavior.

## Threshold trade-offs

- `pos_weight=1`: global-Dice threshold 0.15 and defective-image-Dice threshold 0.6 are different.
- `pos_weight=5`: global-Dice threshold 0.8 and defective-image-Dice threshold 0.9 are different.
- `pos_weight=10`: global-Dice threshold 0.65 and defective-image-Dice threshold 0.9 are different.
- `pos_weight=20`: global-Dice threshold 0.85 and defective-image-Dice threshold 0.2 are different.

## Stability

- `pos_weight=1`: obvious validation divergence; substantial AMP numerical fallback activity (last/best validation-loss ratio 3.644).
- `pos_weight=5`: large validation-loss regime shift; substantial AMP numerical fallback activity (last/best validation-loss ratio 1.000).
- `pos_weight=10`: obvious validation divergence; substantial AMP numerical fallback activity (last/best validation-loss ratio 1.328).
- `pos_weight=20`: no obvious instability; substantial AMP numerical fallback activity (last/best validation-loss ratio 1.000).

## Recommendation

Provisional screening preference `pos_weight=5`: best overall validation balance: highest swept global Dice, strong precision/recall, zero normal-image false-positive rate, fastest runtime, and lowest failure count; still provisional because AMP fallback activity and missed defects remain

This phase does not authorize official-test evaluation, final baseline training, augmentation, synthetic data, or GAN work.
