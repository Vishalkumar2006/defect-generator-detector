# G1.2 localized GAN loss audit

- Status: **PASS**
- Loss version: `g1_2_localized_gan_losses_v1`
- Input shape: `[1, 3, 512, 256]`
- Localized R1, unscaled: 0.07006777077913284
- Generator gradient tensors: 56 / 56
- Generator finite/non-zero gradient elements: 5896739 / 5896739 / 5896739
- Discriminator gradient tensors: 16 / 16
- Discriminator finite/non-zero gradient elements: 695649 / 695611 / 695649
- Validation rows loaded: 0
- Official-test rows loaded: 0
- Training steps: 0

## Invariants

- PASS: `canonical_real_fake_masks_identical`
- PASS: `all_case_losses_finite`
- PASS: `zero_weight_logits_have_zero_influence`
- PASS: `discriminator_step_has_no_generator_gradients`
- PASS: `generator_step_has_no_discriminator_parameter_gradients`
- PASS: `generator_gradients_finite`
- PASS: `generator_has_nonzero_gradients`
- PASS: `discriminator_gradients_finite`
- PASS: `discriminator_has_nonzero_gradients`
- PASS: `r1_finite`

## Mask cases and unweighted components

### central

- Active logits: 120 / 1860 (0.064516)
- `discriminator_real_hinge`: 1.0428834
- `discriminator_fake_hinge`: 0.94602406
- `discriminator_total`: 1.98890746
- `generator_adversarial`: 0.0539759435
- `support_normalized_change`: 0.0692230165
- `boundary_seam`: 0.0342125371
- `masked_total_variation`: 0.0747203827
- `provisional_aggregated_generator`: 0.232131869

### border_left

- Active logits: 50 / 1860 (0.026882)
- `discriminator_real_hinge`: 0.807132423
- `discriminator_fake_hinge`: 1.12397885
- `discriminator_total`: 1.93111134
- `generator_adversarial`: -0.12397889
- `support_normalized_change`: 0.144542798
- `boundary_seam`: 0.119212598
- `masked_total_variation`: 0.0715554729
- `provisional_aggregated_generator`: 0.211331964

### corner_top_left

- Active logits: 36 / 1860 (0.019355)
- `discriminator_real_hinge`: 0.28132382
- `discriminator_fake_hinge`: 1.77724314
- `discriminator_total`: 2.05856705
- `generator_adversarial`: -0.777243316
- `support_normalized_change`: 0.189285934
- `boundary_seam`: 0.145163208
- `masked_total_variation`: 0.059659414
- `provisional_aggregated_generator`: -0.383134753

### left_right

- Active logits: 300 / 1860 (0.161290)
- `discriminator_real_hinge`: 1.33111894
- `discriminator_fake_hinge`: 0.675008357
- `discriminator_total`: 2.00612736
- `generator_adversarial`: 0.341747791
- `support_normalized_change`: 0.10937164
- `boundary_seam`: 0.0738013908
- `masked_total_variation`: 0.0669039488
- `provisional_aggregated_generator`: 0.59182477
