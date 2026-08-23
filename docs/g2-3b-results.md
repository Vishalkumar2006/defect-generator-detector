# G2.3B mature-budget, prevalence-controlled detector-utility result

**Status: EXECUTED AND DECIDED.** All nine arms trained to the full precommitted
budget and the frozen eight-criterion gate was applied exactly once.

```text
stop_not_confirmed_g2_3b
```

The protocol, gate, threshold rule, seeds, arms, and budget were frozen before
any G2.3B detector was trained (`docs/g2-3b-utility-protocol.md`, commit
`6e566a5`). Nothing in this document changed any of them. No gate value was
chosen, relaxed, reordered, or reinterpreted after observing a result.

G2.2 remains permanently `stop_not_confirmed`. G2.3A remains post-hoc diagnostic
evidence only. This FAIL changes neither.

## Decision

The primary comparison is `gan_1500` minus `prevalence_matched_real` over seeds
45, 46, and 47, evaluated at each arm's precommitted selected operating
threshold. **Three of the eight criteria passed and five failed**, so the gate
is not satisfied.

| # | Criterion | Required | Observed | Outcome |
|---|---|---:|---:|---|
| 1 | mean global Dice gain | `>= +0.01` | `-0.000196` | **FAIL** |
| 2 | mean global IoU gain | `>= +0.005` | `-0.000675` | **FAIL** |
| 3 | mean pixel recall delta | `>= -0.01` | `+0.022999` | PASS |
| 4 | mean pixel precision delta | `>= -0.01` | `-0.026376` | **FAIL** |
| 5 | mean normal-image FPR delta | `<= +0.02` | `+0.028754` | **FAIL** |
| 6 | seeds with positive Dice gain | `>= 2 of 3` | `2` | PASS |
| 7 | mean pixel PR-AUC gain | `>= +0.01` | `-0.002609` | **FAIL** |
| 8 | seeds with positive PR-AUC gain | `>= 2 of 3` | `2` | PASS |

Criterion 3 — the pixel-recall tolerance that G2.2 actually failed — passed
here, and comfortably. The prevalence-matched control removed the confound that
made G2.2's recall regression uninterpretable. What replaced it is a different
and clearer finding: once the control carries the same 0.625 effective defect
prevalence, **synthetic content produces no aggregate quality gain at all**. The
mean Dice and IoU gains are approximately zero rather than negative-but-large,
and the threshold-independent PR-AUC gain is likewise approximately zero.

## Execution integrity

Verified before the gate was applied, for all nine arms:

| Property | Result |
|---|---|
| Arms completed | 9 / 9 (seeds 45, 46, 47 x three arms) |
| Optimizer updates per arm | exactly `5,952` |
| Attempted batches per arm | exactly `5,952` (one successful update per batch) |
| Skipped updates | `0` in every arm |
| Epochs | `12 / 12` per arm, exactly `496` updates each |
| Schedule identity | all nine `schedule_sha256` match `reports/g2_3b/plan/precommitted_plan.json` |
| Configuration identity | one `config_sha256` across all nine arms |
| Initialization | identical within each seed; distinct across seeds |
| Evaluation split | `validation` in every arm |
| Official held-out samples loaded | `0` in every arm |
| GAN optimizer updates | `0` |
| Synthetic samples regenerated | `0` |
| Frozen `joint_1500.pt` hash | re-verified identical to the recorded value |
| Resume events | none; every arm ran start-to-finish in one pass |
| Test suite | `448 passed` |

Total training wall clock was about 14 hours 7 minutes.

## Per-arm composition, as executed

Every epoch of every arm matched the frozen composition exactly, with no drift:

| Arm | Normal-real | Defective-real | Synthetic | Effective defective |
|---|---:|---:|---:|---:|
| `standard_real` | 992 | 992 | 0 | `0.500` |
| `prevalence_matched_real` | 744 | 1,240 | 0 | `0.625` |
| `gan_1500` | 744 | 744 | 496 | `0.625` |

## Selected operating thresholds and best epochs

Selected by the frozen rule on development validation only: 99-point grid
`0.01..0.99`, maximum global Dice, tie-broken by mean defective-image Dice, then
pixel precision, then the smallest threshold. Model selection is minimum
validation total loss over all 12 epochs.

| Seed | Arm | Selected threshold | Best epoch |
|---:|---|---:|---:|
| 45 | `standard_real` | 0.16 | 12 |
| 45 | `prevalence_matched_real` | 0.02 | 10 |
| 45 | `gan_1500` | 0.11 | 12 |
| 46 | `standard_real` | 0.14 | 8 |
| 46 | `prevalence_matched_real` | 0.17 | 9 |
| 46 | `gan_1500` | 0.13 | 12 |
| 47 | `standard_real` | 0.54 | 8 |
| 47 | `prevalence_matched_real` | 0.07 | 12 |
| 47 | `gan_1500` | 0.13 | 12 |

Eight of the nine mature checkpoints prefer a threshold **below** 0.5, matching
the accepted converged BF16 baseline's preference for `0.05` and reversing the
G2.3A observation that all six immature G2.2 checkpoints preferred a threshold
above 0.5. The mature budget did resolve the calibration pathology that G2.3A
identified. It did not produce a synthetic-content benefit.

## Metrics at the selected operating threshold

| Seed | Arm | Dice | IoU | Precision | Recall | Normal FPR | Image recall | PR-AUC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 45 | `standard_real` | 0.7071 | 0.5469 | 0.7050 | 0.7091 | 0.0288 | 0.9730 | 0.8065 |
| 45 | `prevalence_matched_real` | 0.7498 | 0.5998 | 0.7170 | 0.7859 | 0.0064 | 0.9459 | 0.8258 |
| 45 | `gan_1500` | 0.7671 | 0.6222 | 0.6845 | 0.8724 | 0.1086 | 0.9730 | 0.8441 |
| 46 | `standard_real` | 0.7332 | 0.5788 | 0.7152 | 0.7521 | 0.0351 | 0.9189 | 0.7734 |
| 46 | `prevalence_matched_real` | 0.6839 | 0.5197 | 0.7874 | 0.6045 | 0.0447 | 0.9459 | 0.7519 |
| 46 | `gan_1500` | 0.7031 | 0.5421 | 0.7447 | 0.6659 | 0.0351 | 0.9459 | 0.7552 |
| 47 | `standard_real` | 0.8113 | 0.6825 | 0.7778 | 0.8478 | 0.0256 | 0.9730 | 0.8226 |
| 47 | `prevalence_matched_real` | 0.7618 | 0.6153 | 0.7026 | 0.8319 | 0.0415 | 0.9730 | 0.8092 |
| 47 | `gan_1500` | 0.7248 | 0.5683 | 0.6985 | 0.7531 | 0.0351 | 0.9459 | 0.7798 |

All nine arms reach a materially more converged state than any G2.2 arm did.
For reference, the accepted real-only BF16 baseline records validation Dice
`0.7777` at threshold 0.5 and `0.7983` at its swept `0.05`, and the best Dice any
G2.2 checkpoint achieved at any threshold was `0.6729`.

## Per-seed primary deltas: `gan_1500` minus `prevalence_matched_real`

| Seed | Dice gain | IoU gain | Precision delta | Recall delta | Normal-FPR delta | PR-AUC gain |
|---:|---:|---:|---:|---:|---:|---:|
| 45 | +0.017308 | +0.022460 | -0.032406 | +0.086529 | **+0.102236** | +0.018274 |
| 46 | +0.019141 | +0.022429 | -0.042661 | +0.061332 | -0.009585 | +0.003292 |
| 47 | **-0.037037** | **-0.046913** | -0.004060 | -0.078865 | -0.006390 | **-0.029391** |
| **Mean** | **-0.000196** | **-0.000675** | **-0.026376** | **+0.022999** | **+0.028754** | **-0.002609** |

The result is not a uniform deficit. Seeds 45 and 46 show small positive Dice,
IoU, and PR-AUC gains; seed 47 reverses all three by a larger margin, so the
means land essentially at zero. Criterion 5 fails almost entirely on seed 45,
whose `gan_1500` arm selected a low threshold (0.11) that bought a large recall
gain at the cost of a `+0.1022` normal-image false-positive-rate regression;
the other two seeds' FPR deltas are slightly negative.

Seed dispersion remains the dominant effect. With three seeds and per-seed Dice
gains spanning `-0.037` to `+0.019`, this experiment cannot distinguish a small
true effect from zero. It can, however, exclude the large positive effect the
gate required.

## Secondary comparison: `standard_real` minus `prevalence_matched_real`

Ungated, reported for both outcomes. This isolates the pure defect-prevalence
effect using real data only — the confound that contaminated every G2.2
comparison.

| Seed | Dice gain | IoU gain | Precision delta | Recall delta | Normal-FPR delta | PR-AUC gain |
|---:|---:|---:|---:|---:|---:|---:|
| 45 | -0.042768 | -0.052918 | -0.011953 | -0.076732 | +0.022364 | -0.019274 |
| 46 | +0.049240 | +0.059070 | -0.072163 | +0.147552 | -0.009585 | +0.021531 |
| 47 | +0.049473 | +0.067225 | +0.075212 | +0.015846 | -0.015974 | +0.013486 |
| **Mean** | **+0.018649** | **+0.024459** | **-0.002968** | **+0.028889** | **-0.001065** | **+0.005247** |

At the mature budget, raising effective defect prevalence from 0.500 to 0.625
using real defective images alone produces a mean Dice change of `+0.0186` in
the *opposite* direction from the G2.2-era assumption: the 50/50 arm is on
average slightly better than the 62.5% arm, not worse. The per-seed signs are
inconsistent (`-0.043`, `+0.049`, `+0.049`), so the honest reading is that the
pure prevalence effect at this budget is small and not consistently signed.

That matters for interpreting G2.2: the confound G2.3A proved was real, but its
magnitude at a mature budget is not large enough to have been the sole cause of
G2.2's failure, and its direction is not stable across seeds.

## Stratified results at the selected threshold

Mean Dice difference across the three seeds, by stratum. Size cutoffs are the
frozen G2.2 development-training tertiles (small `<= 1,261`; medium
`1,262..3,671`; large `> 3,671`).

| Comparison | Border | Non-border | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|
| `gan_1500` - `prevalence_matched_real` | +0.0227 | +0.0015 | -0.0078 | -0.0244 | +0.0374 |
| `standard_real` - `prevalence_matched_real` | +0.0396 | -0.0058 | -0.0155 | -0.0024 | +0.0435 |

G2.3A found the synthetic arm improved small-defect PR-AUC in all three
immature seeds. That pattern does **not** survive prevalence matching at the
mature budget: the mean small-defect Dice difference is now slightly negative
(`-0.0078`), and the strata where `gan_1500` leads its prevalence-matched
control — border and large — are also the strata where the plain 50/50 real arm
leads that same control by a similar or larger margin. The stratum structure
therefore tracks the control's operating point rather than synthetic content.

Full per-stratum Dice and recall for every seed and arm are recorded in
`reports/g2_3b/confirmation_summary.json`.

## Threshold-0.5 continuity, secondary only

Recorded as continuity evidence against G2.2 and G2.3A. **Never a gate input.**

| Seed | `standard_real` | `prevalence_matched_real` | `gan_1500` |
|---:|---:|---:|---:|
| 45 | 0.6981 / 0.0288 | 0.6914 / 0.0064 | 0.7612 / 0.1054 |
| 46 | 0.7214 / 0.0288 | 0.6814 / 0.0415 | 0.7002 / 0.0319 |
| 47 | 0.8110 / 0.0256 | 0.6942 / 0.0288 | 0.6870 / 0.0256 |

Values are global Dice / normal-image FPR. Normal-image FPR at threshold 0.5 is
`0.0064`-`0.1054` here against `0.597`-`0.927` for the G2.2 controls, which is
independent confirmation that these detectors are converged in a way no G2.2 arm
was.

## What this result does and does not license

**It does not authorize** relaxing the gate, adding seeds, retuning any
threshold, changing the synthetic fraction, substituting GAN checkpoint 1,000 or
2,000, training the GAN longer, regenerating synthetic data, or re-opening G2.2.
`stop_not_confirmed_g2_3b` is terminal for the authorized experiment on the same
terms `stop_not_confirmed` was terminal for G2.2.

**It does not authorize any access to the official held-out KSDD2 split.** The
confirmation records `official_test_access_count: 0` and
`official_test_authorized_by_this_decision: false`, and
`configs/g2_3b_utility_confirmation.json` sets both
`access_policy.official_test_allowed` and
`access_policy.official_test_allowed_after_confirmation` to `false`. A PASS
would not have granted that access either; only a separate explicit
authorization could, and a FAIL forecloses it.

**What it does establish**, and this is the point of having run it: G2.2's
comparison was uninterpretable because synthetic content and defect prevalence
moved together. Separating them at a mature budget, on three fresh seeds, under
a precommitted operating point, shows no aggregate benefit from frozen GAN
checkpoint 1,500's synthetic content — on threshold-dependent quality and on
threshold-independent PR-AUC alike. The negative result is now attributable in a
way G2.2's never was.

## Artifacts

| Path | Content |
|---|---|
| `reports/g2_3b/confirmation_summary.json` | Machine-readable confirmation: gate, per-seed metrics, deltas, decision |
| `reports/g2_3b/seed<SEED>/<ARM>.json` | Per-arm report: budget, identity, thresholds, metrics, strata |
| `reports/g2_3b/seed<SEED>/seed_summary.json` | Per-seed rollup across the three arms |
| `reports/g2_3b/seed<SEED>/<ARM>_epochs.json` | Per-epoch progress log (Git-ignored) |
| `reports/g2_3b/plan/precommitted_plan.json` | Frozen plan and schedule hashes, reproduced byte-identically after training |
| `checkpoints/g2_3b/seed<SEED>/<ARM>_{best,last}.pt` | Durable detector checkpoints (Git-ignored) |
| `configs/g2_3b_utility_confirmation.json` | Frozen configuration, gate, and access policy |
| `docs/g2-3b-utility-protocol.md` | The precommitted protocol this result executed |

## Commands

From the repository root in Windows PowerShell:

```powershell
# Reproduce the frozen plan; writes the same bytes. CPU only, trains nothing.
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode plan

# Re-apply the frozen gate to the nine completed arms. Trains nothing.
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode confirm

.\.venv\Scripts\python.exe -m pytest
```

Training is complete and must not be rerun. The runner offers `plan`, `train`,
and `confirm` only; there is no official-held-out-split mode.
