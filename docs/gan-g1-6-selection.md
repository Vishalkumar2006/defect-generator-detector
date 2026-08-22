# G1.6 discriminator-update selection

G1.6 selects `configs/gan_smoke_dclip10.json` as the configuration to replay in
the first sustained GAN run. The selected optimizer and clipping values are:

- generator learning rate: `1e-4`;
- discriminator learning rate: `2.5e-5`;
- generator gradient maximum norm: `5`;
- discriminator gradient maximum norm: `10`.

Every other architecture, loss, optimizer, data, precision, seed, warmup, and
safety setting remains identical to the completed G1.5b smoke configuration.
The baseline and D-clip-10 smoke reports and recovery checkpoints remain separate
and are preserved unchanged as experiment evidence.

The original automated ablation report is retained as historical diagnostic
evidence. It was written when the D-clip-10 run stopped at joint step 197 because
the old canonical-gradient gate counted nonzero RGB scalar components. After the
gate was corrected to count finite, nonzero RGB vectors per canonical pixel, the
same checkpoint resumed deterministically and completed 200/200 joint updates.
The completed run has zero output-range, locality, invalid-gradient, and
non-finite canonical-gradient violations.

The D-clip-10 configuration is selected for controlled continuation because it
completed the smoke safety contract while materially reducing discriminator
clipping relative to the baseline. This is a configuration selection, not a
claim that any visually inspected sample is a best model. The seven-image fixed
panel and the blinded sheets are qualitative safety evidence only. No `best.pt`
checkpoint is created or inferred from detector confidence or from those panels;
the sustained run starts from fresh deterministic identity initialization.

