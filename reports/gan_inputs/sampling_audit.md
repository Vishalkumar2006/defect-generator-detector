# GAN online-sampling audit

- Pipeline: `f1_3_gan_inputs_v1`
- Requested/successful/failed: 1000 / 983 / 17
- Success rate: 98.3000%
- Runtime: 94.3726 seconds
- Samples/second: 10.416159545572102
- Attempts mean/P95/P99/max: 1.2300 / 3.0000 / 4.0000 / 4
- Candidates excluded by compatibility index: 451941
- Actual transform/placement retries: 230
- Actual placement retries after indexing: 0
- Empty compatibility pools: 243
- Template utilization: 229 / 232
- Background utilization: 756 / 1772
- Border fraction empirical/target/observed/drift: 0.413793 / 0.413793 / 0.397762 / 0.016031
- Accidental contact violations: 0
- Support pixels outside valid: 0
- Materialized generated images: 0
- Validation rows loaded: 0
- Official-test rows loaded: 0

## Failures by reason

- `gan_sampling_attempts_exhausted:empty_compatibility_pool:bottom+left,empty_compatibility_pool:top+left`: 3
- `gan_sampling_attempts_exhausted:empty_compatibility_pool:left`: 8
- `gan_sampling_attempts_exhausted:empty_compatibility_pool:left+right`: 3
- `gan_sampling_attempts_exhausted:empty_compatibility_pool:left+right,transformed_template_exceeds_patch`: 3

## Successful placements by side combination

- `bottom`: 53
- `bottom+right`: 49
- `left+right`: 12
- `none`: 592
- `right`: 183
- `top`: 48
- `top+right`: 46
