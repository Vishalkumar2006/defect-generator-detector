# G1.2 localized GAN objectives

Phase G1.2 defines loss primitives only. It adds no GAN dataset, optimizer,
training loop, checkpointing, schedule, or paired target. The original coarse
composite is used only to regularize the magnitude and boundary of the generator's
change; it is not treated as a ground-truth refined image.

## Canonical discriminator conditioning

Both real and synthetic discriminator masks use
`canonicalize_discriminator_mask`. With configurable threshold `t=0.5`, the
canonical value is `1[m >= t]`, returned as floating values exactly equal to zero
or one. The generator continues to receive its original fractional mask.

Canonicalization removes representation differences but cannot remove a geometric
augmentation side channel. A later data pipeline must apply statistically
equivalent flips, scales, crops, boundary placement, and other geometric
augmentations to real and synthetic discriminator masks.

## Localized PatchGAN weights and hinge losses

The canonical mask is dilated by radius 35 input pixels, approximately half of a
70 x 70 PatchGAN receptive field, then projected to the logit grid with adaptive
max pooling. This produces binary weights `w` without a uniform background floor.
Every sample must contain at least one active weight. For any elementwise quantity
`x`, the localized mean is:

`weighted_mean(x,w) = sum(x*w) / sum(w)`.

The raw-logit discriminator components are:

- `L_D_real = weighted_mean(relu(1-real_logits), real_weights)`;
- `L_D_fake = weighted_mean(relu(1+fake_logits), fake_weights)`; and
- `L_D = L_D_real + L_D_fake`.

The generator adversarial component is:

`L_G_adv = -weighted_mean(fake_logits, fake_weights)`.

For a discriminator step, the generated image must be detached before it is passed
to the discriminator. Detaching logits inside the loss would incorrectly remove
the fake contribution to discriminator gradients. For a generator step, the
discriminator parameters are frozen but its operations remain differentiable so
the adversarial gradient reaches the generator.

## Local regularizers

Let `S` be the binary generator support, `R` the refined image, `C` the coarse
composite, and `K=3` RGB channels.

The support-normalized change magnitude is:

`L_change = sum(|R-C| * S) / (K * sum(S))`.

This returns zero for identical inputs and ignores all changes outside support.

The seam ring is the inner width-three boundary of `S`, computed by erosion-like
pooling without wraparound. With ring `B`:

`L_boundary = sum(|R-C| * B) / (K * sum(B))`.

Border, corner, and opposite-side support are handled by the same bounded pooling
operation. Empty support or ring reductions return differentiable finite zero.

For RGB residual `E`, masked total variation counts horizontal and vertical pixel
pairs only when both endpoints belong to `S`:

`L_TV = sum_active_pairs(|E[p]-E[q]|) / (K * active_pair_count)`.

## Aggregation and R1

`aggregate_generator_losses` computes

`L_G = wa*L_G_adv + wc*L_change + wb*L_boundary + wt*L_TV`.

The checked-in unit coefficients are deliberately marked provisional. They make
the interface and audit executable but do not claim suitable training balance. A
future smoke audit must measure relative scales before weights are frozen.

The optional R1 helper differentiates the localized weighted discriminator score
with respect to a real image and returns the batch mean of the summed squared image
gradient. It is unscaled and unscheduled; a later trainer may apply it lazily.

`scripts/audit_gan_losses.py` uses deterministic synthetic 512 x 256 tensors for
central, border, corner, and left+right masks. It reports active-logit fractions,
every unweighted loss component, zero-weight invariance, R1, and parameter/element
gradient coverage. It atomically writes JSON and Markdown under
`reports/gan_losses/`, loads no dataset rows, and records zero training steps.
