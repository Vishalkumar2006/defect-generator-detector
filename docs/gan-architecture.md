# G1.1 mask-conditioned residual GAN architecture

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
- a reflection-padded 7 x 7 head with `tanh` for a bounded RGB residual.

Every normalized layer uses GroupNorm and the model contains no BatchNorm,
transposed convolution, or latent random vector. The configured model has
5,896,739 trainable parameters.

The binary support is the positive defect mask dilated by 12 pixels. The candidate
is `clamp(composite + 0.25 * residual, -1, 1)`. A final `torch.where` copies the
original composite outside support, making those pixels bit-exact. Empty masks
therefore return the input exactly. Native-edge, corner, and opposite-side masks
use the same operation and require no coordinate special case.

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
materialized images, and training steps.
