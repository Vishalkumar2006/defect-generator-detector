# G1.5 gated 200-step GAN smoke

- Status: **PASS**
- Warmup steps/status: 10 / accepted
- 20-step gate passed: True
- Joint D/G steps: 200 / 200
- Early stop: False (None)
- D/G clipping fractions: 0.7048 / 0.1000
- R1 events: 13 at [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208]
- Locality/invalid-gradient violations: 0 / 0
- Monitor sheets: 11
- Runtime seconds: 148.095
- Peak allocated/reserved VRAM: 1326761472 / 1600126976
- Validation rows: 0
- Official-test rows: 0
- Materialized training images: 0

## Terminal joint update

- Joint step: 200
- Discriminator total hinge: 1.814286231994629
- Generator losses: `{"adversarial": -0.7200412750244141, "boundary": 0.0029964346904307604, "change": 0.007407996337860823, "total": -0.7095339298248291, "total_variation": 0.0010289100464433432}`
- D/G pre-clip gradient norms: 91.8863525390625 / 0.8212974667549133
- Output-range violations: 0
- Would-have-clamped fraction (deprecated additive rule): 0.004643113352358341
- Directional-cap / tanh saturation: 0.0 / 0.0
- Mean/max support change: 0.007407996337860823 / 0.07421875
- Canonical gradient coverage: 1.0
- Invalid adversarial gradient / outside-support change: 0.0 / 0.0

## Frozen-detector retention

- Composite/refined/genuine-real inside probability: 0.4739814663854694 / 0.885163426399231 / 0.5817895872002866
- Composite/refined/genuine-real Dice: 0.5527691670826503 / 0.6913069060870579 / 0.5612687702689853
- Retention gate: `{"consecutive_below_stop": 0, "inside_retention_ratio": 1.8675064093737463, "stop": false, "warning": false}`

## Warnings

- `discriminator_gradient_clipping_above_50_percent`
- `fixed_monitor_mean_support_containment_below_99_percent`
- `fixed_monitor_support_containment_below_95_percent`
- `strongly_unequal_contact_side_sampling`

All optimization settings remain provisional; this smoke is not final training.
