# G1.6 single discriminator-update ablation

G1.6 compared the completed G1.5b baseline with exactly one controlled
discriminator update ablation. The candidate changed only discriminator gradient
clipping from 5 to 10 and discriminator learning rate from `5e-5` to `2.5e-5`;
its report and checkpoint directories were isolated. All generator settings,
data identities, seed, BF16 precision, warmup, losses, and smoke limits remained
unchanged.

The candidate passed warmup and the 20-step continuation gate, then stopped at
joint step 197 on the exact canonical-gradient safety check. The terminal update
had 65,207 active canonical-gradient pixels out of 65,208, so this was not the
former floating-point coverage false positive. The stopped state was preserved
and was not resumed.

The higher clip/lower learning-rate candidate reduced discriminator clipping from
70.48% to 30.92% overall and from 98.33% to 46.67% over each run's final 60
completed joint steps. Its final-60 mean real-minus-fake margin was lower,
`0.05644` versus `0.09690`. On a deterministic 28-sample internal-monitor audit,
the candidate's detector-statistic L2 distance from genuine real was lower,
`0.15758` versus `0.22863`.

Four blinded sheets covered four unique samples from each of the seven existing
geometry and morphology strata. Before reveal, the baseline was preferred: the
candidate more often produced an edge-dominant, ring-like residual in amplified
difference views. Both candidates retained localized changes without rectangular
crop seams in the reviewed sheets.

Because the candidate failed a safety invariant, did not complete 200 steps, had
a weaker final-60 discriminator margin, and did not win blinded visual review, it
does not clearly dominate despite improving clipping and detector-statistic
distance. G1.6 therefore retains the G1.5b baseline and authorizes no additional
sweep or automatic run.

The machine-readable comparison, baseline integrity hashes, detector statistics,
provenance, visual review, and blinded sheets are under
`reports/gan_training/g1_6_dclip10_ablation/`. No validation or official-test row
was loaded, and no generated training dataset was materialized.
