# G1.4 one-step GAN mechanics audit

- Status: **PASS**
- Precision: `bf16` on `cuda`
- Runtime seconds: 26.23335889998998
- Peak CUDA memory bytes: 1568063488
- Discriminator optimizer steps: 1
- Generator optimizer steps: 1
- Total training batches optimized: 1
- D changed only D parameters: True
- G changed only G parameters: True
- Canonical native coverage minimum / median: 0.49310302734375 / 1.0
- Canonical coverage fraction below 0.5: 0.001028221987715453
- Canonical defect gradient coverage: 1.0
- Maximum invalid fake gradient: 0.0
- Generator locality after step: True
- Validation rows loaded: 0
- Official-test rows loaded: 0
- Monitor optimizer steps: 0
- Materialized training images: 0

## Calibration losses

- `discriminator_real_hinge`: min 1.4607095, median 1.5696276, max 1.7325716, finite 8/8
- `discriminator_fake_hinge`: min 0.42147487, median 0.57003993, max 0.65513152, finite 8/8
- `discriminator_total_hinge`: min 2.0876086, median 2.1391965, max 2.2215562, finite 8/8
- `unscaled_r1`: min 0.012478453, median 0.022312433, max 0.038899623, finite 8/8
- `generator_adversarial`: min 0.37815213, median 0.48929752, max 0.64201462, finite 8/8
- `change`: min 0.080889732, median 0.11929237, max 0.14843228, finite 8/8
- `seam`: min 0.050250385, median 0.075640149, max 0.10795527, finite 8/8
- `total_variation`: min 0.041324779, median 0.053640367, max 0.056931309, finite 8/8

## Unit-coefficient generator gradients

- `adversarial`: median 12.217177, relative to adversarial 1.0, zero batches 0
- `change`: median 5.1790135, relative to adversarial 0.4239124410577802, zero batches 0
- `seam`: median 4.2147276, relative to adversarial 0.3449837474930017, zero batches 0
- `total_variation`: median 3.2500277, relative to adversarial 0.2660211563217104, zero batches 0

## Suggested provisional coefficients

`{"adversarial": 1.0, "change": 2.358977711304533, "seam": 2.8986872780732544, "total_variation": 3.7590995160950977}`

Suggestions were not written to configuration and were not used for the mechanics step.

## One-step losses

- Discriminator: `{"fake_hinge": 0.47754549980163574, "raw_r1": 0.0, "real_hinge": 1.857008457183838, "scaled_r1": 0.0, "total": 2.3345539569854736, "total_hinge": 2.3345539569854736}`
- Generator: `{"adversarial": -2.9775824546813965, "boundary": 0.0869678184390068, "change": 0.12327779829502106, "total": -2.7107315063476562, "total_variation": 0.05660529062151909}`
