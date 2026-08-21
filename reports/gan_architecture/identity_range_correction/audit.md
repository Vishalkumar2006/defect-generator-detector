# G1.5a identity/range corrective audit

- Status: **PASS**
- Architecture: `g1_5a_identity_range_aware_residual_gan_v1`
- Residual semantics: `g1_5a_directional_range_aware_residual_v1`
- Initial mean/max support change: 0.0 / 0.0
- Initial raw/applied maximum: 0.0 / 0.0
- Initial output-range violations: 0
- After-step mean/max support change: 0.0060091144405305386 / 0.025789260864257812
- After-step output-range violations: 0
- After-step directional-cap / tanh saturation: 0.0 / 0.0
- First backward head / earlier nonzero tensors: 2 / 0
- Second backward head / earlier nonzero tensors: 2 / 54
- Failed checkpoints unchanged: True
- Validation / official-test rows loaded: 0 / 0
- Materialized generated images: 0

## Invariants

- PASS: `synthetic_initial_identity`
- PASS: `real_initial_identity`
- PASS: `initial_raw_residual_zero`
- PASS: `initial_applied_residual_zero`
- PASS: `initial_regularizers_zero`
- PASS: `initial_output_range_violations_zero`
- PASS: `first_backward_reaches_output_head`
- PASS: `first_backward_earlier_layers_are_staged_zero`
- PASS: `one_step_output_range_violations_zero`
- PASS: `one_step_residual_nonzero`
- PASS: `second_backward_reaches_earlier_layers`
- PASS: `outside_support_remains_exact`
- PASS: `failed_checkpoints_unchanged`
- PASS: `preserved_checkpoint_lacks_new_residual_semantics`
