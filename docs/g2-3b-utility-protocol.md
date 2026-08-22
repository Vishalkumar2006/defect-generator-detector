# G2.3B mature-budget, prevalence-controlled detector-utility protocol

**Status: PRECOMMITTED PROTOCOL. No G2.3B training has been executed.** This
document, `configs/g2_3b_utility_confirmation.json`, and
`reports/g2_3b/plan/precommitted_plan.json` were written and frozen before any
detector was trained.

G2.2 remains permanently `stop_not_confirmed`. G2.3A is post-hoc diagnostic
evidence only. Neither a G2.3B PASS nor a G2.3B FAIL reopens, reinterprets, or
alters either of them.

## Hypothesis

> Does frozen GAN checkpoint 1,500 provide detector utility **beyond the effect of
> increasing defective-sample prevalence**, when detectors are trained to a mature
> budget under a common training protocol?

G2.3A established two facts that this experiment is built around:

1. **A class-prevalence confound.** Every G2.2 synthetic arm trained at 62.5%
   effective defective samples while its real-only control trained at 50.0%,
   because each synthetic sample is defective and displaced a class-balanced real
   draw. No G2.2 comparison could separate synthetic *content* from defect
   *prevalence*.
2. **Underconvergence and control instability.** 2,000 updates is 4.03 baseline
   epochs, 33.6% of the accepted mature budget. Control Dice ranged 0.406–0.623
   across seeds (s.d. 0.109) and control PR-AUC s.d. was 0.111 — roughly 2.5×
   the synthetic arms' dispersion. Threshold-0.5 comparisons were additionally
   calibration-sensitive: all six G2.2 checkpoints preferred a Dice threshold
   above 0.5, while the converged baseline preferred 0.05.

G2.3B removes the confound by construction, trains to the mature budget, and
selects an operating threshold by one precommitted rule.

## The three arms

All three are built by one scheduling framework and differ **only** in their
precommitted per-batch source composition.

| Arm | Name | Normal-real | Defective-real | Synthetic | Effective defective |
|---|---|---:|---:|---:|---:|
| A | `standard_real` | 50.0% | 50.0% | 0% | **0.500** |
| B | `prevalence_matched_real` | 37.5% | 62.5% | 0% | **0.625** |
| C | `gan_1500` | 37.5% | 37.5% | 25.0% | **0.625** |

Every arm repeats one two-batch (eight-slot) unit, the smallest unit in which
50.0%, 37.5%, 62.5%, and 25.0% are all exactly representable at batch size four:

```
A  [N D N D]  [N D N D]
B  [N D N D]  [N D D D]
C  [N D N S]  [N D D S]
```

Per epoch (496 updates × batch 4 = 1,984 slots), and over the full 12-epoch
budget (23,808 slots):

| Arm | Normal-real / epoch | Defective-real / epoch | Synthetic / epoch | Normal total | Defective-real total | Synthetic total |
|---|---:|---:|---:|---:|---:|---:|
| A | 992 | 992 | 0 | 11,904 | 11,904 | 0 |
| B | 744 | 1,240 | 0 | 8,928 | 14,880 | 0 |
| C | 744 | 744 | 496 | 8,928 | 8,928 | 5,952 |

Arm C carries exactly one synthetic sample in every four-sample batch, preserving
G2.2's synthetic fraction exactly. Arm B's extra defective exposure is **real
defective images only** — it never touches synthetic data. B and C are
slot-for-slot identical except at batch position 3, where B takes the next real
defective draw and C takes the next synthetic sample.

## Why C versus B is the primary comparison

B and C carry **identical effective defect prevalence (0.625)** and identical
budgets, initialization, loss, optimizer, augmentation, scheduler, and
checkpoint-selection rule. The only difference is what fills 25% of the training
stream: real defective images (B) or frozen checkpoint-1,500 synthetic defective
composites (C). A positive C−B effect is therefore attributable to synthetic
*content*, which is exactly what G2.2 could not establish.

A versus B is secondary and ungated. It measures the effect of raising effective
defect prevalence from 0.500 to 0.625 using real data alone — the magnitude of the
confound that contaminated every G2.2 comparison. It is reported, never gated.

## Sampling fairness

Fairness is enforced structurally, not inferred from percentages.

- **One common framework.** All three arms come from `build_arm_schedule`. The arm
  chooses only how many draws of each class it consumes and where those slots sit
  in a batch.
- **Arm-independent draw streams.** For each `(seed, epoch, class)` one stream is
  built to the longest length any arm needs (992 normal, 1,240 defective, 496
  synthetic per epoch). The stream key is
  `experiment_version:seed:epoch:class` and contains **no arm**, so no arm can
  perturb another's draws. Every arm consumes a **prefix** of that identical
  stream: C's 744 defective-real draws are the first 744 of the 1,240 B consumes,
  and the first 744 of the 992 A consumes. Positional alignment inside an epoch is
  not preserved — cursors advance at different rates — but the draw law and the
  identity prefix are identical.
- **Replacement behaviour matches the accepted baseline.** The historical BF16
  baseline used `WeightedRandomSampler` with weights `0.5/positives` and
  `0.5/negatives` and `replacement=True`, i.e. uniform-with-replacement *inside*
  each class. G2.3B draws uniform-with-replacement inside each class, including
  the synthetic pool, so all four pools obey the same law.
- **Identical augmentation.** `SynchronizedRandomFlips(0.5, 0.5, seed)` keyed by
  `(seed, epoch, sample_id)`. The same identity at the same epoch receives the
  same flip in every arm, which is also the baseline's semantics.
- **Exposure.** The plan verifies that all 209 development-training defective
  identities appear in every arm and every seed, and that 1,754–1,770 of the 1,772
  normal identities do.
- **Identical initialization within a seed.** Each arm builds its U-Net after
  `configure_reproducibility(seed)`, and the runner refuses to continue unless all
  three arms report the same `initialization_sha256`.

## Mature-budget rationale

The accepted stabilized BF16 baseline (`configs/final_real_baseline_bf16.json`,
`reports/final_real_baseline_bf16_seed42/`) ran **496 optimizer updates per epoch
× 12 epochs = 5,952 successful updates**, with zero skips and zero numerical
anomalies, and produced validation Dice 0.7777 / normal-image FPR 0.0415 at
threshold 0.5. That is the repository's only evidence of a converged detector, so
5,952 is adopted exactly — not approximated.

G2.3B deliberately does **not** extend the G2.2 constant-LR 2,000-update
protocol. It reproduces the accepted baseline's mature semantics.

### What matches the accepted baseline

| Component | Value |
|---|---|
| Architecture | GroupNorm U-Net, 3→1 channels, base width 32 |
| Canvas / padding | 256×672, reflection-padded image, zero-padded mask |
| Loss region | native `valid_region` pixels only |
| Loss | valid-region BCE-with-logits (`pos_weight` 5) + soft Dice, weights 1/1, float32 |
| Precision | BF16 forward, no GradScaler, no automatic fp32 retry |
| Optimizer | AdamW, LR `3e-4`, weight decay `1e-4` |
| Gradient control | clip 1.0, at most one optimizer update per attempted batch |
| LR schedule | `ReduceLROnPlateau` on validation total loss, factor 0.5, patience 2, min LR `1e-5` |
| Augmentation | synchronized H/V flips p=0.5, keyed by seed/epoch/sample id |
| Checkpoint selection | minimum validation total loss over all 12 epochs |
| Budget | 496 updates/epoch × 12 epochs = 5,952 |
| Threshold sweep | executed once, only after training completes and the best checkpoint reloads |
| Sweep objective and tie-break | maximum global Dice, then mean defective-image Dice, then pixel precision, then smallest threshold |
| Real-sampling law | uniform with replacement inside each class |
| Arm A defect prevalence | 50.0%, the baseline sampler's `target_defective_fraction` |

### Unavoidable differences, stated explicitly

1. **1,984 slots per epoch instead of 1,981.** The baseline drew
   `num_samples = len(dataset) = 1981`, giving 496 batches of which the last held
   1 sample. G2.3B uses 1,984 slots so every batch is exactly 4 and all three
   composition patterns are exactly representable. The optimizer-update count per
   epoch and in total is unchanged.
2. **Exact rather than stochastic class composition.** The baseline's sampler
   realized 48.4–51.1% defective per epoch; G2.3B's arm A is exactly 50.0%. This
   is a tightening required by "arms differ only in precommitted composition".
3. **Early stopping is monitor-only.** The baseline configured patience 4 but
   never triggered it. Acting on it would break the equal-budget invariant, so
   G2.3B records `early_stopping_would_have_triggered` per epoch and never acts.
4. **Threshold grid 0.01–0.99 step 0.01 (99 points)** instead of the baseline's
   0.05–0.95 step 0.05 (19 points). The finer grid is a strict superset of every
   baseline candidate. It is widened because G2.3A found immature checkpoints
   whose best-Dice threshold exceeded 0.95, so the baseline grid could clip.
5. **The realized LR trajectory may differ per arm.** The *rule* is identical, but
   `ReduceLROnPlateau` is data-dependent, so arms may halve LR at different
   epochs. This is inherent to reproducing the accepted protocol; per-epoch LR is
   recorded for every arm.
6. **No `WeightedRandomSampler` object.** The schedule is explicit and content-
   hashed. The within-class law is the same; the exact identity sequence a
   `WeightedRandomSampler` would have produced is not reproduced.
7. **25% of arm C's slots are 256×672 synthetic composites**, not native KSDD2
   captures. That is the treatment under test.

### How G2.3B differs from G2.2

| Aspect | G2.2 | G2.3B |
|---|---|---|
| Budget | 2,000 updates | 5,952 updates (mature) |
| LR schedule | constant `3e-4` | `ReduceLROnPlateau` (baseline rule) |
| Epoch structure | none, one flat stream | 12 epochs with per-epoch validation |
| Reported model | unselected last iterate | best epoch by validation total loss |
| Primary control | real-only at 50.0% defective | prevalence-matched at 62.5% defective |
| Prevalence confound | present (0.500 vs 0.625) | removed in the primary comparison |
| Operating point | fixed 0.5 | precommitted selected threshold; 0.5 secondary only |
| Threshold-independent evidence | none | pixel PR-AUC, gated |
| Gate criteria | 5 | 8 (all 5 unchanged, 3 added) |
| Seeds | 42, 43, 44 | 45, 46, 47 (fresh) |
| Schedules | per-arm balanced streams | one shared per-class stream, arms consume prefixes |

## Fresh-seed rationale

Seeds 45, 46, and 47 have never been used by any phase of this project. Seeds 42,
43, and 44 are burned: they carry completed G2.2 arms, a terminal
`stop_not_confirmed`, and a full G2.3A post-hoc analysis whose findings were used
to design this protocol. Reusing them would make G2.3B a re-test of data that
informed its own design. Fresh seeds keep the confirmation genuinely
out-of-sample with respect to every G2.3A observation.

## Threshold-selection rule

Precommitted before training, identical for every arm and every seed, and applied
to **development validation only**.

1. Train to the full 5,952-update budget.
2. Reload the best checkpoint (minimum validation total loss).
3. Sweep the fixed grid `{0.01, 0.02, …, 0.99}` — 99 points, built in integer
   hundredths, containing `0.5`, strictly inside `(0, 1)`.
4. Select the threshold maximizing validation **global Dice**, breaking ties by:
   maximum mean defective-image Dice → maximum pixel precision → **smallest
   threshold**. Grid thresholds are unique, so this is a total order and no tie
   can reach an implementation-defined choice.

The runner refuses to proceed if the sweep did not use the precommitted grid.

### Reported metrics

Threshold-independent, for every arm and seed:

- **pixel PR-AUC**, step-rule average precision over the G2.3A probability grid
  (uniform 0.0005 probability lattice ∪ 0.02-step logit lattice over [−20, 20]).

At the precommitted selected operating threshold:

- global Dice, global IoU, pixel precision, pixel recall;
- normal-image false-positive rate;
- image-level defect recall;
- border and non-border Dice **and** recall;
- small, medium, and large Dice **and** recall (development-training mask-pixel
  tertiles: small ≤ 1,261; medium 1,262–3,671; large > 3,671 — the frozen G2.2
  cutoffs).

At threshold 0.5: the same block, labelled
`at_fixed_threshold_0_5_secondary_continuity_only`. It is continuity evidence
against G2.2 and G2.3A and is **never** a gate input.

## Exact numerical confirmation gate

Frozen in `configs/g2_3b_utility_confirmation.json` before any training.
Primary comparison: **`gan_1500` minus `prevalence_matched_real`**, over seeds
45/46/47, evaluated at each arm's precommitted selected operating threshold.

All eight criteria must hold:

| # | Criterion | Threshold | Source |
|---|---|---|---|
| 1 | mean global Dice gain | `≥ +0.01` | unchanged from G2.2 |
| 2 | mean global IoU gain | `≥ +0.005` | unchanged from G2.2 |
| 3 | mean pixel recall delta | `≥ −0.01` | unchanged from G2.2 |
| 4 | mean pixel precision delta | `≥ −0.01` | unchanged from G2.2 |
| 5 | mean normal-image FPR delta | `≤ +0.02` | unchanged from G2.2 |
| 6 | seeds with positive Dice gain | `≥ 2 of 3` | unchanged from G2.2 |
| 7 | mean pixel PR-AUC gain | `≥ +0.01` | **added** |
| 8 | seeds with positive PR-AUC gain | `≥ 2 of 3` | **added** |

### Justification of every value

Criteria 1–6 are carried over **numerically unchanged** from the frozen G2.2
selection rules in `configs/g2_2_detector_utility.json`. Nothing was relaxed. In
particular the `−0.01` recall tolerance — the criterion G2.2 actually failed — is
deliberately kept at its original strictness, even though G2.3A showed that
threshold-0.5 recall is calibration-sensitive; the fix applied here is a better
operating point and a prevalence-matched control, not a looser bound. A test
(`test_gate_thresholds_are_not_weaker_than_the_frozen_g2_2_rules`) asserts that no
carried-over value is weaker than G2.2's.

Criteria 7 and 8 are **added, never substituted**, so the gate can only be harder
to pass than G2.2's was. G2.3A showed that threshold-0.5 comparisons alone can
misattribute a calibration shift, so a threshold-independent criterion is
required. The `0.01` magnitude mirrors the G2.2 minimum Dice gain because both are
bounded-`[0,1]` aggregate quality measures; for scale, G2.3A measured per-seed
PR-AUC deltas of `+0.104 / −0.029 / +0.050` at the immature budget, so `0.01` is
well inside the measurable range and is not a token bar.

**No gate value was chosen from any G2.3B result. No G2.3B result exists.**

## Leakage restrictions

- The official held-out KSDD2 split is **inaccessible**. `assert_permitted_split`
  admits `train` and `validation` only; the reserved split raises
  `OfficialTestAccessError`. `scripts/train_g2_3b_utility.py` offers modes
  `plan`, `train`, `confirm` and nothing else, and a test asserts the script
  contains no quoted reserved-split literal at all.
- `access_policy.official_test_allowed` and
  `access_policy.official_test_allowed_after_confirmation` are both `false`, and
  the config loader refuses to run if either is flipped. A future evaluation on
  the reserved split requires a **separate authorization** granted only after a
  precommitted G2.3B confirmation PASS; G2.3B itself never grants it, and the
  confirmation summary records
  `official_test_authorized_by_this_decision: false` in both outcomes.
- Training reads development-training rows only. Threshold selection, model
  selection, and all reported metrics read development validation only.
- The GAN is never loaded, updated, or run. Synthetic data is never regenerated.
  `verify_frozen_synthetic_identity` re-checks, before every plan and every
  training run: the frozen `joint_1500.pt` file hash
  (`5af1c6aa…4fdf1c81`), the checkpoint-1,500 manifest content hash
  (`9eba21b4…fed7a4cd62ca91`), the pairing report content hash
  (`540a4637…ee078768f491ed33`), all 1,536 per-row image/mask/valid-region file
  hashes, that all 512 rows carry a positive in-valid-region defect pixel, and
  that every row and every template/background source declares
  `official_split=train` and `development_split=train`.
- `checkpoint_1000` and `checkpoint_2000` are refused by name.
- G2.2 and G2.3A artifacts are read-only; G2.3B writes only under
  `reports/g2_3b/` and `checkpoints/g2_3b/`.
- A skipped optimizer update is fatal, and every arm is checked to have executed
  exactly 5,952 updates with 0 skips before it may be evaluated.

## What PASS and FAIL will mean

**PASS** (all eight criteria hold): frozen GAN checkpoint 1,500 provides detector
utility beyond the effect of increased defective-sample prevalence, at a mature
training budget, under a precommitted operating point, on three fresh seeds.
Decision string `confirmed_gan_1500_utility_beyond_prevalence`. This is the
*only* condition under which a separate future authorization may consider a single
precommitted evaluation on the reserved split. A PASS does not itself grant that
access, does not authorize further GAN training, and does not revisit G2.2.

**FAIL** (any criterion fails): decision string `stop_not_confirmed_g2_3b`. There
is no authorized synthetic detector configuration and no reserved-split access.
A FAIL is not an invitation to relax the gate, add seeds, retune the threshold,
change the synthetic fraction, substitute another GAN checkpoint, or train the GAN
longer. The secondary A-versus-B result is still reported either way, because the
magnitude of the pure prevalence effect is useful evidence regardless of outcome.

Neither outcome changes G2.2's terminal `stop_not_confirmed`.

## Artifacts

| Path | Content |
|---|---|
| `configs/g2_3b_utility_confirmation.json` | Frozen protocol, composition, budget, threshold rule, and gate |
| `src/defectgen/training/g2_3b_protocol.py` | Guards, budget arithmetic, common scheduling framework, datasets, threshold rule, gate |
| `scripts/train_g2_3b_utility.py` | `plan` / `train` / `confirm` runner |
| `reports/g2_3b/plan/precommitted_plan.json` | Frozen plan: schedule hashes, composition audit, synthetic identity, gate |
| `reports/g2_3b/plan/seed*/…_schedule.json` | Expanded 23,808-slot schedules (bulk local artifact, Git-ignored; hashes are in the plan) |
| `tests/test_g2_3b_protocol.py` | 63 focused tests |

## Commands

From the repository root in Windows PowerShell:

```powershell
# Freeze the protocol. Trains nothing; CPU only.
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode plan

# Not yet authorized to run. One seed, three arms, 5,952 updates each.
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode train --seed 45
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode train --seed 46
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode train --seed 47

# Apply the frozen gate after all three seeds complete.
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode confirm

.\.venv\Scripts\python.exe -m pytest .\tests\test_g2_3b_protocol.py
```

Only `--mode plan` has been executed. **No G2.3B detector training has been run.**
