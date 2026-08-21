# G1.5 gated 200-step GAN smoke

- Status: **STOPPED**
- Warmup steps/status: 10 / accepted
- 20-step gate passed: True
- Joint D/G steps: 197 / 197
- Early stop: True (incomplete_canonical_adversarial_gradient)
- D/G clipping fractions: 0.3092 / 0.0711
- R1 events: 12 at [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192]
- Locality/invalid-gradient violations: 0 / 0
- Monitor sheets: 10
- Runtime seconds: 146.291
- Peak allocated/reserved VRAM: 1326761472 / 1600126976
- Validation rows: 0
- Official-test rows: 0
- Materialized training images: 0

## Terminal joint update

- Joint step: 197
- Discriminator total hinge: 1.8290431499481201
- Generator losses: `{"adversarial": 0.2750255763530731, "boundary": 0.002536806743592024, "change": 0.008562790229916573, "total": 0.2862246632575989, "total_variation": 0.0009949102532118559}`
- D/G pre-clip gradient norms: 6.799462795257568 / 1.234806776046753
- Output-range violations: 0
- Would-have-clamped fraction (deprecated additive rule): 0.0
- Directional-cap / tanh saturation: 0.0 / 0.0
- Mean/max support change: 0.008562790229916573 / 0.0634765625
- Canonical gradient coverage: 0.9999846644583487
- Invalid adversarial gradient / outside-support change: 0.0 / 0.0

## Frozen-detector retention

- Composite/refined/genuine-real inside probability: 0.4739814663854694 / 0.5605029028146029 / 0.5817895872002866
- Composite/refined/genuine-real Dice: 0.5527691670826503 / 0.6107146314212254 / 0.5612687702689853
- Retention gate: `{"consecutive_below_stop": 0, "inside_retention_ratio": 1.182541813478355, "stop": false, "warning": false}`

## Warnings

- `fixed_monitor_mean_support_containment_below_99_percent`
- `fixed_monitor_support_containment_below_95_percent`
- `strongly_unequal_contact_side_sampling`

All optimization settings remain provisional; this smoke is not final training.
