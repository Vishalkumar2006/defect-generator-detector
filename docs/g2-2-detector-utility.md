# G2.2 equal-budget detector utility experiment

G2.2 treats downstream segmentation utility as the checkpoint-selection test. It
does not update the GAN. The sustained G2.1 checkpoints at joint updates 1,000 and
1,500 are loaded in evaluation mode, with gradients disabled, and their file hashes
are checked before and after synthetic materialization.

## Paired synthetic data

`scripts/build_g2_2_synthetic_manifests.py` samples the development-training GAN
bridge once per synthetic identity. The same coarse composite, template,
background, condition mask, valid region, spatial transform, placement and seed is
sent to both frozen generators. The paired-manifest check excludes only checkpoint
identity and the rendered output image. Any other mismatch is fatal.

Every source record must have both `official_split=train` and
`development_split=train`. Template and normal-background provenance is checked
independently. The manifest records zero detector-validation and zero official-test
sources. Reflection padding remains visible context, but the rendered image is
restored exactly to the coarse background outside the native-valid region. Masks
and valid regions are shared bit-for-bit between checkpoints, use no resizing, and
are padded vertically to the detector's 256 x 672 canvas.

## Equal training budget

The bounded pilot uses seed 42 and trains three fresh GroupNorm U-Nets from an
identical initialization. Each arm receives exactly 2,000 successful AdamW updates,
batch size four, constant learning rate 0.0003, weight decay 0.0001, BF16 forward,
maximum gradient norm 1, the same synchronized flips, and the frozen real-baseline
BCE plus soft-Dice objective. A skipped update is a hard failure.

The synthetic arms contain exactly one synthetic sample per batch (25%) and three
real samples. Those three real identities are shared between both synthetic arms
and the control. The control uses a fourth deterministic real draw. Explicit
sample-level schedules are saved with content hashes; there is no probabilistic
source mixer, early stopping, adaptive scheduler, or validation during training.

## Evaluation and gate

The pilot evaluates detector validation exactly once per arm at the precommitted
0.5 threshold. It reports global Dice, IoU, pixel precision, pixel recall, normal
image false-positive rate, and defective-image strata by training-derived
mask-pixel tertiles and native border contact. The size cutoffs are computed only
from development-training metadata.

A synthetic arm is a meaningful pilot winner only if all of these precommitted
conditions hold relative to real-only:

- global Dice improves by at least 0.01;
- global IoU improves by at least 0.005;
- normal-image false-positive rate worsens by no more than 0.02;
- pixel precision worsens by no more than 0.01; and
- pixel recall worsens by no more than 0.01.

If neither arm passes, G2.2 stops: there is no longer GAN run, additional detector
seed, or official-test evaluation. If an arm passes, only the better passing arm's
detector configuration may be confirmed against real-only over seeds 42, 43 and
44. Official test remains forbidden until that three-seed confirmation succeeds,
after which exactly one precommitted seed-42 final evaluation is allowed.

## Commands

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_g2_2_synthetic_manifests.py --config .\configs\g2_2_detector_utility.json
.\.venv\Scripts\python.exe .\scripts\train_g2_2_detector_utility.py --config .\configs\g2_2_detector_utility.json --mode pilot
```

The first command materializes only the bounded paired train-only detector inputs.
The second command performs the one-seed pilot and cannot construct the official
test split.

## Recorded result

The seed-42 pilot completed all three 2,000-update arms with zero skipped updates.
Checkpoint 1,500 passed the precommitted pilot gate; checkpoint 1,000 was rejected
despite its higher global Dice because pixel recall regressed by 0.0710.

Checkpoint 1,500 was therefore confirmed against real-only over seeds 42, 43 and
44. It improved global Dice in all three seeds, with mean Dice and IoU gains of
0.0843 and 0.0769, and reduced normal-image false-positive rate by 0.2119 on
average. The confirmation nevertheless failed because mean pixel recall regressed
by 0.0572, beyond the allowed 0.01. Seed 43 alone regressed recall by 0.2070.

The terminal decision is `stop_not_confirmed`. No additional GAN updates or
detector variants were run, and the official test split was not constructed or
evaluated. The detailed pilot and confirmation evidence is under `reports/g2_2/`.
