# G1.6 selected configuration

- Selected: `configs/gan_smoke_dclip10.json`
- Generator/discriminator learning rate: `1e-4` / `2.5e-5`
- Generator/discriminator gradient clip: `5` / `10`
- All other settings: unchanged from G1.5b
- Completed D-clip-10 smoke: 200 discriminator and 200 generator joint updates
- Baseline and candidate artifacts: preserved in separate directories
- Best checkpoint created: no

The earlier step-197 candidate stop in `comparison.md` is retained as diagnostic
evidence of the corrected scalar-channel gradient gate. The deterministically
resumed candidate completed after the gate began measuring RGB vectors per
canonical pixel. Selection is based on the completed safety contract, reduced
discriminator clipping, and an explicit controlled-continuation decision. The
fixed seven-image and blinded panels are qualitative QA only and do not define a
best checkpoint.

