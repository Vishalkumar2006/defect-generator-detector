# G1.6 discriminator-update ablation

- Status: **PASS**
- Baseline safety invariants: True
- Candidate safety invariants: False
- Candidate terminal status/step/reason: STOPPED / 197 / `incomplete_canonical_adversarial_gradient`
- Candidate terminal active/total canonical gradients: {'active': 65207, 'total': 65208, 'fraction': 0.9999846644583487}
- Baseline D clipping overall/final-60: 0.7047619047619048 / 0.9833333333333333
- Candidate D clipping overall/final-60: 0.30917874396135264 / 0.4666666666666667
- Baseline/candidate final-60 mean margin: 0.09689742812576393 / 0.05643698312342167
- Baseline/candidate detector L2 distance: 0.22863142177389997 / 0.15757842026591404
- Stratified monitor samples: 28
- Validation / official-test rows: 0 / 0
- Baseline artifacts unchanged: True
- Recommendation: **baseline** (`neither_clearly_dominates_retain_baseline`)

## Configuration differences

`{"checkpoint_directory": {"baseline": "checkpoints/gan_smoke", "candidate": "checkpoints/gan_smoke_dclip10"}, "discriminator_gradient_clip_max_norm": {"baseline": 5.0, "candidate": 10.0}, "discriminator_learning_rate": {"baseline": 5e-05, "candidate": 2.5e-05}, "report_directory": {"baseline": "reports/gan_training/smoke", "candidate": "reports/gan_training/smoke_dclip10"}}`

## Blinded visual sheets

- `reports/gan_training/g1_6_dclip10_ablation/blinded_sheets/sheet_01.png`
- `reports/gan_training/g1_6_dclip10_ablation/blinded_sheets/sheet_02.png`
- `reports/gan_training/g1_6_dclip10_ablation/blinded_sheets/sheet_03.png`
- `reports/gan_training/g1_6_dclip10_ablation/blinded_sheets/sheet_04.png`
