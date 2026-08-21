# Historical failed FP16 baseline

`configs/final_real_baseline.json` and `reports/final_real_baseline/` describe the
historical E1 FP16 experiment. They are retained for numerical-failure analysis;
they are not the final real-only reference.

The run used AdamW at learning rate 0.001, FP16 autocast with GradScaler, and no
gradient clipping. It completed three epochs and failed during epoch 4 validation.
Epoch 1 batch 338 and epoch 3 batch 14 each produced an infinite unscaled gradient
norm. In both cases the forward losses and logits were finite, the optimizer update
was skipped, and GradScaler reduced its scale.

The persisted best checkpoint represents epoch 2 and the last checkpoint represents
epoch 3. CPU inspection found every model and floating optimizer-state tensor finite.
A read-only validation precision diagnostic completed both checkpoints independently
under FP16, BF16, and FP32, with finite logits/losses and unchanged parameter hashes.
This locates the terminal failure in the unsaved epoch-4 optimization state rather
than in either persisted checkpoint.

The run must not be resumed: its two infinite-gradient events and terminal validation
failure violate the numerical-stability requirements for the reference experiment.
It must not be used for publication or as the real-only comparator. Its checkpoints
remain diagnostic artifacts only and stay excluded from Git.

The sanitized evidence record is
`reports/final_real_baseline/failure_summary.json`. Raw precision-diagnostic output
is kept locally because it contains machine-specific absolute paths.
