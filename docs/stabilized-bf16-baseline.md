# Stabilized BF16 final real-only baseline

The final real-only reference configuration is
`configs/final_real_baseline_bf16.json`, with experiment identity
`final_real_baseline_bf16_seed42`. It starts from seed 42 in new report and
checkpoint directories and cannot load or overwrite the historical failed FP16
run.

Only four optimization settings differ from the failed configuration:

- BF16 autocast instead of FP16;
- AdamW learning rate 0.0003 instead of 0.001;
- maximum gradient norm 1.0 instead of no clipping;
- no GradScaler, because BF16 does not require loss scaling.

BCE and Dice remain float32. Gradient norms are recorded before and after clipping,
and clipping occurs after backward and before the single allowed optimizer update.
All architecture, preprocessing, loss, sampling, augmentation, scheduling, early
stopping, checkpoint selection, threshold finalization, and test-isolation rules
remain unchanged.

The user-run 512-attempt development-training-only BF16 numerical probe completed
512 optimizer updates with no skips, non-finite losses/gradients, retries, or
invariant violations. It loaded neither validation nor official-test data. This
probe supports executing the stabilized configuration; it is not a trained or
evaluated final model.

The future real-plus-synthetic detector must reuse the initialization seed,
initialization hash, optimizer-attempt budget, and complete detector protocol
recorded by this run.

## Frozen result

The successful run completed all 5,952 attempted optimizer updates without a skip
or numerical anomaly. The best checkpoint is epoch 11, with validation loss
0.1107619467. Its SHA-256 is
`6a2127fad5fca66108de38226b050b9ef7d09025c4528a5bf48285ffaabfd277`
and its size is 339,558,727 bytes. Checkpoint binaries remain ignored.

At the fixed comparison threshold 0.5, validation global Dice is 0.7777003, IoU
is 0.6362599, precision/recall are 0.8547696/0.7133793, mean defective-image
Dice is 0.7519212, and normal-image false-positive rate is 0.0415335.

The single validation sweep selected 0.05 as the global-Dice-optimal threshold.
At 0.05, global Dice is 0.7982509, precision/recall are
0.7755540/0.8223163, and normal-image false-positive rate is 0.0638978.
Threshold 0.5 remains the fixed comparison threshold, and future real-plus-synthetic
results must report both. The real-only threshold must not be refined below 0.05.
Synthetic conditions must use the same one-time validation selection procedure,
and thresholds must never be retuned on official-test data.
