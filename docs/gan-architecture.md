# G1.5a identity-initialized residual GAN architecture

Phase G1.1 defines network architecture and architecture-level validation only.
It does not add GAN losses, optimizers, dataloaders, checkpoint management, or a
training loop. It preserves the F1 online input pipeline unchanged.

## Generator

`MaskedResidualGenerator` receives a normalized RGB coarse composite and its
one-channel defect mask. It concatenates them into four channels and uses:

- a reflection-padded 7 x 7, 4-to-32 stem;
- three stride-two convolution stages ending at 256 channels;
- four 256-channel residual blocks;
- three bilinear resize-convolution stages with U-Net skips; and
- a reflection-padded 7 x 7 unconstrained RGB residual head whose final
  convolution is initialized with exactly zero weights and bias.

Every normalized layer uses GroupNorm and the model contains no BatchNorm,
transposed convolution, or latent random vector. The configured model has
5,896,739 trainable parameters.

The binary support is the positive defect mask dilated by 12 pixels. For input
`x`, unconstrained head output `raw`, and configured maximum delta `d = 0.25`,
the range-aware residual is:

```
r = tanh(raw)
positive_cap = min(d, 1 - x)
negative_cap = min(d, x + 1)
delta = where(r >= 0, r * positive_cap, r * negative_cap)
candidate = x + delta
refined = where(support, candidate, x)
```

There is no hard clamp in the differentiable output path. The directional caps
guarantee `refined` remains in `[-1, 1]` and `abs(delta) <= d`. The returned
`raw_residual` is the unconstrained head output; `applied_residual` is the actual,
support-masked delta used by change, seam, TV, and monitoring code. The final
`torch.where` preserves pixels outside support bit-exactly.

At initialization, non-empty and empty masks both produce an exact identity:
raw and applied residuals are zero. On the first adversarial backward, the output
head receives gradient while its zero weights may block gradient to earlier
layers. Once the head takes a non-zero update, subsequent backward passes reach
the earlier generator. This staged activation is expected and is audited rather
than misclassified as a dead network.

## Discriminator

`MaskConditionedPatchDiscriminator` concatenates RGB and mask channels and applies
five 4 x 4 convolutions with channel progression 4, 32, 64, 128, 256, 1 and strides
2, 2, 2, 1, 1. Every convolution uses spectral normalization. The first block has
no feature normalization; later hidden blocks use GroupNorm and all hidden blocks
use LeakyReLU(0.2). The output is raw patch logits without sigmoid. At 512 x 256,
the logit shape is 1 x 1 x 62 x 30. The configured model has 695,649 trainable
parameters.

## Validation and audit

Both networks reject invalid ranks, channels, batch/spatial alignment, non-floating
images, non-finite values, image values outside [-1,1], mask values outside [0,1],
and dimensions not divisible by eight. No input is resized implicitly.

`scripts/audit_gan_architecture.py` constructs deterministic synthetic production,
zero-mask, and border-mask inputs. It checks shape, normalized range, exact locality,
forward/backward finiteness, gradient flow, parameter counts, and discriminator
logit shape. It atomically writes JSON and Markdown under
`reports/gan_architecture/` and returns a non-zero exit code if an invariant fails.
It instantiates no dataset and records zero validation rows, official-test rows,
materialized images, and training steps. `scripts/audit_gan_identity.py` adds a
two-sample, development-training-only corrective audit covering exact identity,
one head-only update, the second-backward gradient stage, range telemetry, and
read-only hashes of the preserved failed-smoke checkpoints.
