# G1.3 deterministic GAN training-pair audit

- Status: **PASS**
- Requested samples: 256
- Runtime seconds: 327.7575624000019
- Samples/second: 0.7810651205892618
- Train/monitor defect sources: 188 / 21
- Train/monitor backgrounds: 1595 / 177
- Real/fake canonical-mask equality rate: 1.0
- Transform-metadata equality rate: 1.0
- Empty masks: 0
- Minimum support inside fake validity: 1.0
- Mean fake localized logits affected by padding: 0.3009551816212479
- Mean real localized logits affected by padding: 0.3879126401152462
- Mean real/fake valid-fraction asymmetry: 0.29806220531463623
- Contact sheet: `reports/gan_training_pairs/contact_sheet.png`
- Validation rows loaded: 0
- Official-test rows loaded: 0
- Training steps: 0
- Materialized generated training images: 0
- Templates used/available: 139 / 203
- Backgrounds used/available: 241 / 1595
- Target side combinations: `{"bottom": 7, "bottom+left": 3, "bottom+right": 11, "left": 27, "left+right": 6, "none": 162, "right": 22, "top": 11, "top+left": 2, "top+right": 5}`

## Invariants

- PASS: `defect_source_ids_disjoint`
- PASS: `background_ids_disjoint`
- PASS: `tensor_shapes`
- PASS: `tensor_dtypes`
- PASS: `tensor_ranges`
- PASS: `tensor_finiteness`
- PASS: `repeat_hashes_equal`
- PASS: `epoch_change_changes_training_pair`
- PASS: `monitor_hash_stable_across_epochs`
- PASS: `canonical_mask_equality_rate_is_one`
- PASS: `transform_metadata_equality_rate_is_one`
- PASS: `empty_mask_count_is_zero`
- PASS: `all_generator_support_inside_fake_valid`

## Representation warnings

- None
