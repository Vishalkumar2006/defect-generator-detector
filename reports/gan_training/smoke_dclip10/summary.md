# G1.5 gated 200-step GAN smoke

- Status: **PASS**
- Warmup steps/status: 10 / accepted
- 20-step gate passed: True
- Joint D/G steps: 200 / 200
- Early stop: False (None)
- D/G clipping fractions: 0.3190 / 0.0700
- R1 events: 13 at [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208]
- Locality/invalid-gradient violations: 0 / 0
- Monitor sheets: 11
- Runtime seconds: 147.034
- Peak allocated/reserved VRAM: 1326761472 / 1600126976
- Validation rows: 0
- Official-test rows: 0
- Materialized training images: 0

## Terminal joint update

- Joint step: 200
- Discriminator total hinge: 1.7369096279144287
- Generator losses: `{"adversarial": 1.358778953552246, "boundary": 0.0019038834143429995, "change": 0.00767997233197093, "total": 1.3684735298156738, "total_variation": 0.0011079020332545042}`
- D/G pre-clip gradient norms: 23.258211135864258 / 0.44798898696899414
- Output-range violations: 0
- Would-have-clamped fraction (deprecated additive rule): 0.0027858680114150047
- Directional-cap / tanh saturation: 0.0 / 0.0
- Mean/max support change: 0.00767997233197093 / 0.045166015625
- Canonical gradient pixel coverage: 1542 / 1542 (1.0)
- Canonical gradient active/total/non-finite RGB components: 4626 / 4626 / 0
- Invalid adversarial gradient / outside-support change: 0.0 / 0.0

## Frozen-detector retention

- Composite/refined/genuine-real inside probability: 0.4739814663854694 / 0.5538277688070333 / 0.5817895872002866
- Composite/refined/genuine-real Dice: 0.5527691670826503 / 0.5891958049365452 / 0.5612687702689853
- Retention gate: `{"consecutive_below_stop": 0, "inside_retention_ratio": 1.1684587016249033, "stop": false, "warning": false}`

## Warnings

- `fixed_monitor_mean_support_containment_below_99_percent`
- `fixed_monitor_support_containment_below_95_percent`
- `strongly_unequal_contact_side_sampling`

All optimization settings remain provisional; this smoke is not final training.
