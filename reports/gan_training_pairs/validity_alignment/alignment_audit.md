# G1.3b discriminator-view validity alignment audit

- Status: **PASS**
- Requested pairs: 128
- Runtime seconds: 263.5732166000089
- Sampling rate: 0.48563356190416385
- Corrective alignment executed: True
- Pre-fix canonical containment: `{"fake_valid": {"full_containment_count": 128, "full_containment_rate": 1.0}, "joint_valid": {"full_containment_count": 110, "full_containment_rate": 0.859375}, "real_valid": {"full_containment_count": 110, "full_containment_rate": 0.859375}}`
- Alpha-versus-coverage: `{"maximum_violation": 0.0, "tolerance": 1e-06, "violation_pixels": 0, "violation_samples": 0}`
- Canonical/support inside real_valid: canonical minimum 1.0, rate 1.0; support minimum 0.7226645435244161, mean 0.9096896873243187
- Canonical/support inside fake_valid: canonical minimum 1.0, rate 1.0; support minimum 0.697452229299363, mean 0.9067336417695508
- Canonical/support inside joint_valid: canonical minimum 1.0, rate 1.0; support minimum 0.697452229299363, mean 0.9036787920870658
- Original validity asymmetry: {'minimum': 0.00042724609375, 'mean': 0.17293846607208252, 'maximum': 0.6317520141601562}
- Aligned validity asymmetry: {'minimum': 0.0, 'mean': 0.0, 'maximum': 0.0}
- Maximum difference outside joint validity: 0.0
- Maximum native-valid mutation: 0.0
- Padding-only equality rate: 1.0
- Generator gradient coverage: {'canonical_defect_pixels': {'minimum': 1.0, 'mean': 1.0, 'maximum': 1.0}, 'joint_valid_support_pixels': {'minimum': 1.0, 'mean': 1.0, 'maximum': 1.0}, 'support_pixels_outside_joint_validity': {'minimum': 0.0, 'mean': 0.0, 'maximum': 0.0}, 'outside_joint_support_pixel_count': 112533}
- Support warnings: `["88 samples below 95.00% joint support containment", "mean joint support containment 0.903679 below 0.990000"]`
- Validation rows loaded: 0
- Official-test rows loaded: 0
- Training steps: 0
- Materialized generated training images: 0

## Target-contact strata

`{"bottom": 13, "bottom+left": 13, "bottom+right": 12, "left": 13, "left+right": 12, "none": 13, "right": 13, "top": 13, "top+left": 13, "top+right": 13}`

## Invariants

- PASS: `zero_alpha_coverage_violations`
- PASS: `real_canonical_containment_complete`
- PASS: `fake_canonical_containment_complete`
- PASS: `joint_canonical_containment_complete`
- PASS: `canonical_masks_equal`
- PASS: `aligned_padding_bit_exact_equal`
- PASS: `padding_only_branches_bit_exact_equal`
- PASS: `aligned_validity_asymmetry_zero`
- PASS: `native_valid_pixels_unchanged`
- PASS: `canonical_gradient_coverage_complete`
- PASS: `joint_support_gradient_coverage_complete`
- PASS: `outside_joint_support_gradients_zero`
- PASS: `invalid_fake_gradients_zero`
- PASS: `all_contact_combinations_present`
