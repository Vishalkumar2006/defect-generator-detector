# G1.1 GAN architecture audit

- Status: **PASS**
- Architecture: `g1_1_masked_residual_gan_v1`
- Input shape: `[1, 3, 512, 256]`
- Generator trainable parameters: 5896739
- Discriminator trainable parameters: 695649
- Discriminator logit shape: `[1, 1, 62, 30]`
- Maximum change outside support: 0.0
- Border maximum change outside support: 0.0
- Zero-mask maximum change: 0.0
- Validation rows loaded: 0
- Official-test rows loaded: 0
- Training steps: 0

## Invariants

- PASS: `production_output_shape_matches`
- PASS: `discriminator_logits_nonempty`
- PASS: `output_in_normalized_range`
- PASS: `maximum_change_outside_support_is_zero`
- PASS: `border_change_outside_support_is_zero`
- PASS: `zero_mask_change_is_zero`
- PASS: `forward_tensors_finite`
- PASS: `backward_gradients_finite`
- PASS: `generator_has_nonzero_gradients`
