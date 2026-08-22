# G2.1 sustained 2,000-update GAN training

- Status: **PASS**
- Warmup steps/status: 10 / accepted
- 20-step gate passed: True
- Joint D/G steps: 2000 / 2000
- Early stop: False (None)
- D/G clipping fractions: 0.9169 / 0.0220
- R1 events: 125 at [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008, 1024, 1040, 1056, 1072, 1088, 1104, 1120, 1136, 1152, 1168, 1184, 1200, 1216, 1232, 1248, 1264, 1280, 1296, 1312, 1328, 1344, 1360, 1376, 1392, 1408, 1424, 1440, 1456, 1472, 1488, 1504, 1520, 1536, 1552, 1568, 1584, 1600, 1616, 1632, 1648, 1664, 1680, 1696, 1712, 1728, 1744, 1760, 1776, 1792, 1808, 1824, 1840, 1856, 1872, 1888, 1904, 1920, 1936, 1952, 1968, 1984, 2000]
- Locality/invalid-gradient violations: 0 / 0
- Monitor sheets: 7
- Runtime seconds: 1354.893
- Peak allocated/reserved VRAM: 1333703168 / 1642070016
- Validation rows: 0
- Official-test rows: 0
- Materialized training images: 0
- Stratified monitor audits/pairs: 5 / 128
- Step-200 selected-smoke replay match: True
- Best checkpoint created: False

## Terminal joint update

- Joint step: 2000
- Discriminator total hinge: 1.0447169542312622
- Generator losses: `{"adversarial": 0.5308176875114441, "boundary": 0.01220294926315546, "change": 0.010494773276150227, "total": 0.5543364882469177, "total_variation": 0.008210658095777035}`
- D/G pre-clip gradient norms: 24.03338623046875 / 0.5704735517501831
- Output-range violations: 0
- Would-have-clamped fraction (deprecated additive rule): 0.0
- Directional-cap / tanh saturation: 0.0 / 0.0
- Mean/max support change: 0.010494774207472801 / 0.2392578125
- Boundary residual mass/enrichment: 0.20126573741436005 / 1.1627641605917336
- Canonical gradient pixel coverage: 1538 / 1538 (1.0)
- Canonical gradient active/total/non-finite RGB components: 4614 / 4614 / 0
- Invalid adversarial gradient / outside-support change: 0.0 / 0.0

## Frozen-detector retention

- Composite/refined/genuine-real inside probability: 0.4739814663854694 / 0.6108477679788068 / 0.5817895872002866
- Composite/refined/genuine-real Dice: 0.5527691670826503 / 0.6060976045472282 / 0.5612687702689853
- Retention gate: `{"consecutive_below_stop": 0, "inside_retention_ratio": 1.2887587623141994, "stop": false, "warning": false}`

## Warnings

- `discriminator_gradient_clipping_above_50_percent`
- `fixed_monitor_mean_support_containment_below_99_percent`
- `fixed_monitor_support_containment_below_95_percent`
- `strongly_unequal_contact_side_sampling`

Numbered checkpoints are recovery milestones; no visually or detector-confidence-selected best checkpoint was created.
