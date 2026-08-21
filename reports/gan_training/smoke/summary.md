# G1.5 gated 200-step GAN smoke

- Status: **STOPPED**
- Warmup steps/status: 10 / accepted
- 20-step gate passed: False
- Joint D/G steps: 1 / 1
- Early stop: True (clamp_saturation_above_limit)
- D/G clipping fractions: 0.9091 / 1.0000
- R1 events: 0 at []
- Locality/invalid-gradient violations: 0 / 0
- Monitor sheets: 2
- Runtime seconds: 22.306
- Peak allocated/reserved VRAM: 1271938048 / 1595932672
- Validation rows: 0
- Official-test rows: 0
- Materialized training images: 0

## Warmup gate

- Monitor real-minus-fake margin at step 0: `0.0080123`
- Monitor real-minus-fake margin at step 5: `0.0293423`
- Monitor real-minus-fake margin at step 10: `0.0209570`
- Accepted warmup length: 10 discriminator-only updates

## Terminal joint update

- Discriminator hinge: `1.9219294`
- Generator adversarial/change/seam/TV/total: `-0.3958836 / 0.1364426 / 0.0969601 / 0.0735260 / -0.1551283`
- D/G pre-clip gradient norms: `64.0623 / 43.9843`
- Clamp saturation: `0.0734871` (stop limit `0.05`)
- Mean/max support change: `0.1364426 / 0.25`
- Canonical gradient coverage: `1.0`
- Invalid-region adversarial gradient: `0.0`
- Outside-support change: `0.0`

The clamp stop was confirmed against the pre-clamp candidate: `1554 / 21201`
support-channel pixels (`0.07330`) exceeded `[-1,1]`; none were already saturated
in the composite.

## Frozen-detector retention

- Composite/refined/genuine-real inside probability: `0.47398 / 0.96229 / 0.58179`
- Composite/refined/genuine-real Dice @ 0.5: `0.55277 / 0.51817 / 0.56127`
- Refined inside-probability retention ratio: `2.0303`
- Detector retention warning/stop: `False / False`

R1 events were zero because the stop occurred after 11 total discriminator updates,
before the first scheduled update at 16. The 20-step gate was not reached and the
run was not resumed.

## Warnings

- `discriminator_gradient_clipping_above_50_percent`
- `fixed_monitor_mean_support_containment_below_99_percent`
- `fixed_monitor_support_containment_below_95_percent`
- `generator_gradient_clipping_above_50_percent`
- `strongly_unequal_contact_side_sampling`

All optimization settings remain provisional; this smoke is not final training.
