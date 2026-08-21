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
