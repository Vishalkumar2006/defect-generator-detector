# Baseline CUDA smoke test (non-final)

This is a bounded plumbing check, not a trained or selected baseline.

- Status: **PASS - NON-FINAL SMOKE ONLY**
- Epochs: 2
- Physical batch size: 4
- Model parameters: 28,285,569
- Parameters updated: True
- Finite gradients: True
- Checkpoint round trip: True

## Losses

- Training: [1.6734803020954132, 1.2885996997356415]
- Validation: [1.3364052772521973, 1.2570661306381226]
- First-batch unweighted BCE + Dice: 1.584525
- First-batch capped-weight BCE + Dice: 1.730163

## Final smoke validation metrics (non-final)

- pixel_dice: 0.12269291615398137
- pixel_iou: 0.0653558052434457
- pixel_precision: 0.09994272623138603
- pixel_recall: 0.15885298133818843
- pixel_true_positive_pixels: 1396.0
- pixel_false_positive_pixels: 12572.0
- pixel_false_negative_pixels: 7392.0
- defective_pixel_dice: 0.238143978164449
- defective_pixel_iou: 0.1351665375677769
- defective_pixel_precision: 0.47547683923705725
- defective_pixel_recall: 0.15885298133818843
- defective_pixel_true_positive_pixels: 1396.0
- defective_pixel_false_positive_pixels: 1540.0
- defective_pixel_false_negative_pixels: 7392.0
- image_precision: 0.2
- image_recall: 1.0
- image_f1: 0.33333333333333337
- image_threshold: 0.9
