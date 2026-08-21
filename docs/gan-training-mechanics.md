# G1.4 GAN training mechanics and loss-scale calibration

Phase G1.4 implements an auditable trainer and executes one discriminator update
and one generator update. It is a mechanics and scale-measurement phase, not a GAN
training run. The learning rates, R1 settings, clipping limits, and loss weights in
`configs/gan_one_step.json` are explicitly provisional.

## Update isolation

The discriminator update generates a fresh fake under `torch.no_grad`, aligns
real and fake views with the G1.3b joint-valid mask, reuses one canonical condition
mask and one localization-weight tensor, and applies localized hinge loss. Lazy R1
uses the convention `(global_step + 1) % r1_interval == 0`, so the ordinary
zero-indexed one-step audit does not apply it. A scheduled update uses
`raw_r1 * gamma * interval / 2`.

The generator update clears both optimizers, temporarily disables discriminator
parameter gradients, reruns the generator, and sends the aligned refined view
through the updated discriminator. Its provisional aggregate contains localized
adversarial, support-normalized change, boundary seam, and masked residual-TV
losses. Discriminator parameter-gradient state is restored afterward. Parameter
hashes are captured before the discriminator update, after it, and after the
generator update; mutable spectral-normalization buffers are intentionally not
used as parameter-isolation evidence.

Both updates reject non-finite logits, losses, or unscaled gradients before an
optimizer call. Gradient norms are measured before and after clipping. CUDA BF16
uses autocast without a scaler, CUDA FP16 requires `GradScaler`, and CPU falls back
to FP32. G1.4 does not skip, retry, or reduce the learning rate after a numerical
failure.

## Calibration rule

Eight deterministic internal-training batches are evaluated without optimizer
updates. Each raw loss and its unit-coefficient parameter-gradient norm are
reported. Suggested generator coefficients set adversarial to one and use:

`median(adversarial gradient norm) / median(component gradient norm)`

Suggestions are clamped to `[0.001, 1000]`, are not written back to configuration,
and are not used by the one-step mechanics test. They are provisional measurements,
not selected training hyperparameters.

The completed audit suggested adversarial `1.0`, change `2.35898`, seam `2.89869`,
and total variation `3.75910`. The actual mechanics step retained unit coefficients
to expose rather than hide scale differences.

## Bounded audit

The audit reads development-training data only, performs a fixed monitor forward
before and after the updates, materializes no generated dataset, and saves no
checkpoint:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_gan_one_step.py --config .\configs\gan_one_step.json
```

Its JSON and Markdown reports are written atomically under
`reports/gan_training/one_step/`. The verified run used CUDA BF16, passed every
mechanics invariant, executed exactly one optimizer step per model, and loaded
zero validation or official-test rows.
