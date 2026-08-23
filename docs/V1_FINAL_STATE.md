# Version 1 final state — `defect-generator-detector`

**This is the authoritative status document for the project.** It supersedes
`docs/PROJECT_STATE.md`, which remains preserved verbatim as the accurate
handoff *through G2.2* and is still correct for everything it covers. Where the
two overlap, they agree; where this document goes further, it is because work
happened after `bc36ed4`.

Frozen at commit `7c533d6` (`result: apply the frozen G2.3B gate --
stop_not_confirmed_g2_3b`), 2026-08-23. The commit that adds this document is
documentation-only. It does not supersede, reopen, or reinterpret any
experimental decision.

---

## A. Final terminal status

```text
stop_not_confirmed_g2_3b
```

| Property | Value |
|---|---|
| V1 scientific status | `stop_not_confirmed_g2_3b` — terminal |
| Experimental phase | **Complete.** All authorized V1 experiments have run. |
| G2.2 status | `stop_not_confirmed` — unchanged and permanent |
| G2.3A status | Post-hoc diagnostic evidence only — unchanged |
| G2.3B status | Executed in full; frozen gate applied once; **FAIL** |
| Official held-out KSDD2 test split | **Never constructed, inspected, counted, or evaluated.** Access remains unauthorized. |
| Version 2 | **Not started.** No V2 code, config, or experiment exists. |
| Authorized synthetic detector configuration | **None.** |
| Authorized final detector checkpoint for release | **None.** |

There is no accepted synthetic-augmentation configuration in V1, and no basis in
the recorded evidence for training the GAN longer, adding seeds, retuning any
threshold, or changing the synthetic fraction.

---

## B. Complete commit ledger

Every subject below was read from Git, not from memory.

| Commit | Date | Subject | Phase established |
|---|---|---|---|
| `2d5f217` | 2026-08-20 | chore: establish verified KSDD2 data pipeline | Extraction, audit, manifest, deterministic development split, geometry, leakage rules |
| `39f3eff` | 2026-08-20 | feat: add CUDA-verified segmentation baseline scaffold | Full-image segmentation scaffold |
| `9750826` | 2026-08-20 | experiment: add provisional class-weight validation pilot | Class-weight pilot |
| `1548203` | 2026-08-20 | fix: make mixed-precision training numerically auditable | Auditable mixed-precision step accounting |
| `ff2c8c8` | 2026-08-20 | feat: add frozen real-only baseline protocol | Frozen real-only protocol |
| `ec68d29` | 2026-08-21 | fix: add baseline numerical failure diagnostics | FP16 failure diagnostics |
| `ed89a0f` | 2026-08-21 | fix: stabilize final baseline with bf16 and gradient clipping | Stabilized BF16 configuration |
| `5c77b9a` | 2026-08-21 | experiment: freeze stabilized real-only baseline | **Accepted real-only reference** |
| `e5f204e` | 2026-08-21 | feat: add deterministic GAN input pipeline | F1 training-only GAN inputs |
| `9156bb4` | 2026-08-21 | fix: remove GAN normal-background width bias | F1.1 |
| `9fd3d52` | 2026-08-21 | fix: preserve border geometry in GAN inputs | F1.2 border censoring, manifest hashing |
| `dbf5903` | 2026-08-21 | perf: index compatible GAN placements | F1.3 compatibility index |
| `4eea741` | 2026-08-21 | fix: restore symmetric GAN border sampling | F1.4 symmetric placement |
| `aeaeea5` | 2026-08-21 | feat: add mask-conditioned residual GAN architecture | G1.1 |
| `96ab048` | 2026-08-21 | feat: add localized GAN objectives | G1.2 |
| `7872293` | 2026-08-21 | feat: add deterministic GAN training-pair bridge | G1.3 |
| `00bb747` | 2026-08-21 | fix: align GAN discriminator validity | G1.3a/G1.3b |
| `991ddfb` | 2026-08-21 | feat: add auditable GAN one-step trainer | G1.4 |
| `5ed2f65` | 2026-08-22 | feat: add gated GAN smoke training | G1.5 |
| `babe4b2` | 2026-08-22 | fix: initialize GAN residual output safely | G1.5a identity/range-aware output |
| `006156e` | 2026-08-22 | perf: audit GAN discriminator clipping ablation | G1.6 ablation |
| `bdb9085` | 2026-08-22 | fix: finalize G1.6 discriminator selection | D-clip-10 selection |
| `1b69785` | 2026-08-22 | feat: add sustained GAN training | **G2.1** 2,000-update GAN run |
| `a41be83` | 2026-08-22 | feat: evaluate downstream GAN utility | **G2.2** — terminal `stop_not_confirmed` |
| `bc36ed4` | 2026-08-22 | docs: record project state through G2.2 | `docs/PROJECT_STATE.md` |
| `e1171a1` | 2026-08-22 | diagnose: add G2.3A post-hoc validation-only diagnostic | **G2.3A** diagnostic |
| `6e566a5` | 2026-08-22 | feat: precommit the G2.3B prevalence-controlled utility protocol | **G2.3B protocol frozen before training** |
| `ee51b1f` | 2026-08-23 | docs: add G2.3B active-run handoff state | Active-run handoff (now completed; see §K) |
| `b531031` | 2026-08-23 | feat: harden G2.3B execution durability without touching the protocol | Restart safety; no scientific surface changed |
| `7c533d6` | 2026-08-23 | result: apply the frozen G2.3B gate -- stop_not_confirmed_g2_3b | **V1 terminal result** |

Branch: `main`. HEAD at freeze: `7c533d6`.

---

## C. G2.3A final diagnostic conclusions

Full detail: `docs/g2-3-diagnostic.md`; machine-readable evidence under
`reports/g2_3/diagnostic/`. G2.3A trained nothing — 0 GAN optimizer updates, 0
detector optimizer updates, 0 synthetic samples regenerated, 6 validation-only
forward passes over already-trained G2.2 checkpoints. It reproduced every
recorded G2.2 threshold-0.5 metric with a maximum absolute delta of exactly
`0.0`, so it measured the same models through the same path.

Its four findings, in the order the evidence supports them:

1. **Class-prevalence confound — established as fact.** Recomputed from the
   saved 8,000-row schedules: every G2.2 synthetic arm trained at **0.625**
   effective defective fraction while its real-only control trained at **0.500**,
   because each synthetic sample is defective and displaced a class-balanced real
   draw. No G2.2 comparison could separate synthetic *content* from defect
   *prevalence*.
2. **Detector underconvergence and control instability — strongly supported.**
   2,000 updates is 4.032 baseline epochs, **33.6%** of the accepted mature
   budget, landing near baseline epoch 4. Every arm's training loss was still
   falling. Control Dice ranged 0.406–0.623 (s.d. `0.1086`); control normal-image
   FPR ranged 0.597–0.927 against the accepted baseline's `0.0415`. Recall
   variance was ~4.5× larger in controls than in arms.
3. **Calibration / operating-point shift — strongly supported.** All six G2.2
   checkpoints preferred a best-Dice threshold **above** 0.5 (several above
   0.95), while the converged baseline preferred `0.05`. The control's recall was
   reachable by rethresholding the arm in every seed, including seed 43.
4. **Genuine representation degradation — partially supported, seed-43 only, and
   not GAN-attributable.** After matching recall, seed 43's arm trailed its own
   control on Dice by `0.0887` and on PR-AUC by `0.0287`. But that control was
   the strongest of the three controls and that arm the strongest of the three
   arms, so the deficit tracks control-side variance rather than seed-43 GAN
   pathology. Seeds 42 and 44 show the opposite sign on every frontier measure.

**Synthetic-mask and provenance audit** (hard check on all 512 local masks):
512/512 read; every sample carries at least one positive *valid* defect pixel;
every mask's support lies entirely inside `valid_region` (0 outside pixels);
shapes aligned for all 512; masks and valid regions shared bit-for-bit between
the two checkpoints; 512/512 rows declare `official_split=train` **and**
`development_split=train`; 617 distinct template/background source identities,
all in development train; detector-validation source overlap **0**; official-test
rows read, counted, or inspected **0**. Positive valid defect pixels:
min 23 / median 1,615.5 / max 32,426.

**No official-test access.** `reports/g2_3/diagnostic/diagnostic_summary.json`
records `official_test_access_count = 0`, and
`scripts/run_g2_3_diagnostic.py` contains no literal reserved-split string.

G2.3A explicitly did **not** authorize G2.3B or anything else; G2.3B was a
separate, explicitly authorized, precommitted experiment.

---

## D. G2.3B frozen protocol

Frozen before any G2.3B training at commit `6e566a5`. Authoritative sources:
`docs/g2-3b-utility-protocol.md`, `configs/g2_3b_utility_confirmation.json`,
`reports/g2_3b/plan/precommitted_plan.json`.

**Hypothesis.** Does frozen GAN checkpoint 1,500 provide detector utility
*beyond the effect of increasing defective-sample prevalence*, when detectors are
trained to a mature budget under a common protocol?

### The three arms

| Arm | Name | Normal-real | Defective-real | Synthetic | Effective defective |
|---|---|---:|---:|---:|---:|
| A | `standard_real` | 50.0% | 50.0% | 0% | `0.500` |
| B | `prevalence_matched_real` | 37.5% | 62.5% | 0% | `0.625` |
| C | `gan_1500` | 37.5% | 37.5% | 25.0% | `0.625` |

Repeating two-batch (eight-slot) units at batch size 4:

```
A  [N D N D]  [N D N D]
B  [N D N D]  [N D D D]
C  [N D N S]  [N D D S]
```

B and C are slot-for-slot identical except at batch position 3, where B takes the
next real defective draw and C takes the next synthetic sample. Per-epoch source
counts, verified identical in all 12 epochs of all 9 executed arms:

| Arm | Normal-real | Defective-real | Synthetic |
|---|---:|---:|---:|
| A | 992 | 992 | 0 |
| B | 744 | 1,240 | 0 |
| C | 744 | 744 | 496 |

### Immutable parameters

- **Seeds 45, 46, 47** — fresh; never used by any earlier phase. Seeds 42/43/44
  are burned (they carry G2.2 arms and the G2.3A analysis that informed this
  design), so reusing them would have made G2.3B a re-test of its own design data.
- **5,952 successful optimizer updates per arm** = 496 updates/epoch × 12 epochs,
  batch size 4, 1,984 slots/epoch. **Zero skipped updates; a skip is fatal.**
- **Mature BF16 semantics reproduced from the accepted baseline**
  (`configs/final_real_baseline_bf16.json`): GroupNorm U-Net 3→1, base width 32;
  256×672 canvas with reflection-padded image and zero-padded mask; loss on native
  `valid_region` pixels only; valid-region BCE-with-logits (`pos_weight` 5) + soft
  Dice, weights 1/1, float32; BF16 forward, no GradScaler, no fp32 retry; AdamW
  LR `3e-4`, weight decay `1e-4`; gradient clip 1.0; `ReduceLROnPlateau` on
  validation total loss, factor 0.5, patience 2, min LR `1e-5`; synchronized H/V
  flips p=0.5 keyed by seed/epoch/sample id; checkpoint selection = minimum
  validation total loss over all 12 epochs.
- **Early stopping is monitor-only.** `early_stopping_would_have_triggered` is
  recorded per epoch and never acted on, preserving the equal-budget invariant.

### Sampling design

- One common framework (`build_arm_schedule`); the arm chooses only how many
  draws of each class it consumes and where those slots sit in a batch.
- **Arm-independent draw streams.** For each `(seed, epoch, class)` one stream is
  built to the longest length any arm needs. The stream key is
  `experiment_version:seed:epoch:class` and contains **no arm**, so no arm can
  perturb another's draws; every arm consumes a *prefix* of the identical stream.
- Uniform-with-replacement inside each class, matching the accepted baseline's
  `WeightedRandomSampler` law; the synthetic pool obeys the same law.
- Identical initialization within a seed, enforced: the runner refuses to
  continue unless all three arms report the same `initialization_sha256`.

### Frozen checkpoint-1,500 identity (re-verified at every run)

| Item | SHA-256 |
|---|---|
| `checkpoints/gan_training_2000/joint_1500.pt` (whole file, 79,262,487 bytes) | `5af1c6aafabcc0444117aa43209dcab168e57f4489259728e8f9066a4fdf1c81` |
| `reports/g2_2/synthetic_manifests/checkpoint_1500.json` (content) | `9eba21b4347dcdafafd9d0f90dd06b297cb58b2f7ee58f1887fed7a4cd62ca91` |
| `reports/g2_2/synthetic_manifests/pairing_report.json` (content) | `540a4637936c25ae9fd3678732bbc9d81e75f066e584e6d3ee078768f491ed33` |

Also re-verified at every run: all 1,536 per-row image/mask/valid-region file hashes,
that all 512 rows carry a positive in-valid-region defect pixel, and train-only
provenance on every row and every template/background source. **Checkpoints 1,000
and 2,000 are refused by name.** The GAN is never loaded for training, updated,
or run; synthetic data is never regenerated.

### Threshold-selection rule

Precommitted, identical for every arm and seed, applied to **development
validation only**: train the full budget → reload the best checkpoint → sweep the
fixed 99-point grid `{0.01 … 0.99}` (built in integer hundredths, contains `0.5`,
strictly inside `(0,1)`) → select maximum validation **global Dice**, tie-broken
by mean defective-image Dice → pixel precision → **smallest threshold**. The
runner refuses to proceed if the sweep did not use the precommitted grid.

### The eight-criterion gate

Primary comparison **`gan_1500` minus `prevalence_matched_real`** over seeds
45/46/47, at each arm's precommitted selected operating threshold. All eight must
hold:

| # | Criterion | Threshold | Provenance |
|---|---|---|---|
| 1 | mean global Dice gain | `>= +0.01` | unchanged from G2.2 |
| 2 | mean global IoU gain | `>= +0.005` | unchanged from G2.2 |
| 3 | mean pixel recall delta | `>= -0.01` | unchanged from G2.2 |
| 4 | mean pixel precision delta | `>= -0.01` | unchanged from G2.2 |
| 5 | mean normal-image FPR delta | `<= +0.02` | unchanged from G2.2 |
| 6 | seeds with positive Dice gain | `>= 2 of 3` | unchanged from G2.2 |
| 7 | mean pixel PR-AUC gain | `>= +0.01` | **added** |
| 8 | seeds with positive PR-AUC gain | `>= 2 of 3` | **added** |

Criteria 1–6 are carried over numerically unchanged from
`configs/g2_2_detector_utility.json` — including the `-0.01` recall tolerance
G2.2 actually failed, deliberately kept at original strictness. Criteria 7–8 were
**added, never substituted**, so the gate can only be harder to pass than G2.2's.
A test (`test_gate_thresholds_are_not_weaker_than_the_frozen_g2_2_rules`) asserts
no carried-over value is weaker.

**Secondary, ungated:** `standard_real` minus `prevalence_matched_real` —
the pure defect-prevalence effect using real data only. Reported in both
outcomes; never a gate input.

---

## E. G2.3B final results

Authoritative machine-readable source: **`reports/g2_3b/confirmation_summary.json`**.
Narrative: `docs/g2-3b-results.md`. Per-arm detail:
`reports/g2_3b/seed<SEED>/<ARM>.json`; per-seed rollups
`reports/g2_3b/seed<SEED>/seed_summary.json`.

### Outcome

```text
confirmed: false
decision:  stop_not_confirmed_g2_3b
```

**Three criteria passed, five failed.**

| # | Criterion | Required | Observed | Outcome |
|---|---|---:|---:|---|
| 1 | mean global Dice gain | `>= +0.01` | `-0.000196` | **FAIL** |
| 2 | mean global IoU gain | `>= +0.005` | `-0.000675` | **FAIL** |
| 3 | mean pixel recall delta | `>= -0.01` | `+0.022999` | PASS |
| 4 | mean pixel precision delta | `>= -0.01` | `-0.026376` | **FAIL** |
| 5 | mean normal-image FPR delta | `<= +0.02` | `+0.028754` | **FAIL** |
| 6 | positive-Dice seeds | `>= 2 of 3` | `2` | PASS |
| 7 | mean pixel PR-AUC gain | `>= +0.01` | `-0.002608` | **FAIL** |
| 8 | positive-PR-AUC seeds | `>= 2 of 3` | `2` | PASS |

Exact recorded mean deltas (`aggregate.mean_deltas`):

```json
{
  "global_dice_gain":      -0.00019599518935083568,
  "global_iou_gain":       -0.0006748178865427038,
  "pixel_precision_delta": -0.02637562744415752,
  "pixel_recall_delta":     0.022998687664041968,
  "normal_fpr_delta":       0.02875399361022364,
  "pixel_pr_auc_gain":     -0.0026084855173092634
}
```

### Selected thresholds and best epochs

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

Eight of nine mature checkpoints prefer a threshold **below** 0.5, matching the
accepted converged baseline's preference for `0.05` and reversing G2.3A's
observation that all six immature G2.2 checkpoints preferred a threshold above
0.5. The mature budget resolved the calibration pathology G2.3A identified.

### Per-seed primary comparison, `gan_1500` − `prevalence_matched_real`

| Seed | Dice gain | IoU gain | Precision delta | Recall delta | Normal-FPR delta | PR-AUC gain |
|---:|---:|---:|---:|---:|---:|---:|
| 45 | +0.017308 | +0.022460 | -0.032406 | +0.086529 | **+0.102236** | +0.018274 |
| 46 | +0.019141 | +0.022429 | -0.042661 | +0.061332 | -0.009585 | +0.003292 |
| 47 | **-0.037037** | **-0.046913** | -0.004060 | -0.078865 | -0.006390 | **-0.029391** |

Not a uniform deficit. Seeds 45 and 46 show small positive Dice, IoU, and PR-AUC
gains; seed 47 reverses all three by a larger margin, so the means land at
approximately zero. Criterion 5 fails almost entirely on seed 45, whose
`gan_1500` arm selected a low threshold (0.11) that bought a large recall gain at
the cost of a `+0.1022` normal-image false-positive-rate regression; the other
two seeds' FPR deltas are slightly negative.

### Per-arm metrics at the selected operating threshold

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

For scale: the accepted real-only BF16 baseline records validation Dice `0.7777`
at threshold 0.5 and `0.7983` at its swept `0.05`; the best Dice any G2.2
checkpoint reached at *any* threshold was `0.6729`. All nine G2.3B arms are
materially more converged than any G2.2 arm.

### Secondary, ungated: `standard_real` − `prevalence_matched_real`

| Seed | Dice gain | IoU gain | Precision delta | Recall delta | Normal-FPR delta | PR-AUC gain |
|---:|---:|---:|---:|---:|---:|---:|
| 45 | -0.042768 | -0.052918 | -0.011953 | -0.076732 | +0.022364 | -0.019274 |
| 46 | +0.049240 | +0.059070 | -0.072163 | +0.147552 | -0.009585 | +0.021531 |
| 47 | +0.049473 | +0.067225 | +0.075212 | +0.015846 | -0.015974 | +0.013486 |
| **Mean** | **+0.018649** | **+0.024459** | **-0.002968** | **+0.028889** | **-0.001065** | **+0.005247** |

At a mature budget, raising effective defect prevalence from 0.500 to 0.625 using
real defective images alone produces a mean Dice change of `+0.0186` in the
*opposite* direction from the G2.2-era assumption — the 50/50 arm is on average
slightly better than the 62.5% arm. Per-seed signs are inconsistent
(`-0.043`, `+0.049`, `+0.049`), so the honest reading is that the pure prevalence
effect at this budget is small and not consistently signed. The confound G2.3A
proved was real, but its mature-budget magnitude is not large enough to have been
the sole cause of G2.2's failure, and its direction is not stable across seeds.

### Stratified findings

Mean Dice difference across the three seeds. Size cutoffs are the frozen G2.2
development-training mask-pixel tertiles: small `<= 1,261`; medium
`1,262..3,671`; large `> 3,671`.

| Comparison | Border | Non-border | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|
| `gan_1500` − `prevalence_matched_real` | +0.0227 | +0.0015 | -0.0078 | -0.0244 | +0.0374 |
| `standard_real` − `prevalence_matched_real` | +0.0396 | -0.0058 | -0.0155 | -0.0024 | +0.0435 |

G2.3A found the synthetic arm improved small-defect PR-AUC in all three immature
seeds. **That pattern does not survive prevalence matching at the mature
budget:** mean small-defect Dice difference is now slightly negative. The strata
where `gan_1500` leads its prevalence-matched control — border and large — are
also strata where the plain 50/50 real arm leads that same control by a similar or
larger margin, so the stratum structure tracks the control's operating point
rather than synthetic content. Full per-stratum Dice and recall for every seed
and arm: `reports/g2_3b/confirmation_summary.json`.

### Threshold-0.5 continuity (secondary only, never a gate input)

Global Dice / normal-image FPR:

| Seed | `standard_real` | `prevalence_matched_real` | `gan_1500` |
|---:|---:|---:|---:|
| 45 | 0.6981 / 0.0288 | 0.6914 / 0.0064 | 0.7612 / 0.1054 |
| 46 | 0.7214 / 0.0288 | 0.6814 / 0.0415 | 0.7002 / 0.0319 |
| 47 | 0.8110 / 0.0256 | 0.6942 / 0.0288 | 0.6870 / 0.0256 |

Normal-image FPR at threshold 0.5 is `0.0064`–`0.1054` here against `0.597`–`0.927`
for the G2.2 controls — independent confirmation of convergence.

### Integrity evidence

| Property | Result |
|---|---|
| Arms completed | 9 / 9 |
| Optimizer updates per arm | exactly `5,952` |
| Attempted batches per arm | exactly `5,952` (one successful update per batch) |
| Skipped updates | `0` in every arm |
| Epochs | `12 / 12` per arm, exactly `496` updates each |
| Per-epoch composition | matched the frozen design in all 12 epochs of all 9 arms |
| Schedule identity | all nine `schedule_sha256` match `reports/g2_3b/plan/precommitted_plan.json` |
| Configuration identity | one `config_sha256` `1f9db6dab65d42b164e996b45087a3e5e4d195c79d8f8a538da2a640048ef549` across all nine |
| Initialization | identical within each seed (`a91dbe89…`, `5b996309…`, `0ad6972e…`); distinct across seeds |
| Evaluation split | `validation` in every arm |
| Official held-out samples loaded | `0` in every arm |
| GAN optimizer updates | `0` |
| Synthetic samples regenerated | `0` |
| Frozen `joint_1500.pt` | re-hashed identical at every run |
| Resume events | none; every arm ran start-to-finish in one pass |
| Plan reproducibility | `--mode plan` re-run after the durability commit reproduced `precommitted_plan.json` **byte-identically**, along with all nine expanded schedules |
| Test suite | `448 passed` |

Training wall clock: 2026-08-22T15:26:07Z → 2026-08-23T05:33:07Z, about
**14 h 07 m** for nine arms.

---

## F. Final scientific interpretation

Deliberately factual and conservative. Read the distinctions carefully; they are
the point of the whole G2.3 sequence.

### What was demonstrated

- A deterministic, leakage-controlled pipeline for **GAN-based industrial-defect
  synthesis and downstream detector evaluation** was implemented end to end and
  is reproducible: audited dataset extraction, a development split that never
  touches the official held-out split, training-only template/background
  construction, indexed placement compatibility, a mask-conditioned residual GAN
  with localized objectives, auditable update mechanics, sustained GAN training
  with numbered checkpoints, paired synthetic materialization, and equal-budget
  downstream detector experiments under precommitted gates.
- **Experimental hygiene held throughout.** Content hashes, precommitted
  protocols, frozen gates, fresh seeds, explicit schedules, and hard leakage
  guards were used consistently, and the guards were exercised rather than
  assumed. The official held-out split was never touched at any point in V1.
- **G2.2's evidence was confounded**, and G2.3A established exactly how: unequal
  effective defect prevalence (0.500 vs 0.625), detector underconvergence at
  33.6% of the mature budget, control instability, and a calibration /
  operating-point shift at the fixed threshold 0.5.
- **G2.3B corrected all of those defects by construction** — prevalence-matched
  real control, mature 5,952-update budget, three fresh seeds, precommitted
  operating-point selection, and an added threshold-independent PR-AUC criterion —
  and did so under a protocol frozen before any G2.3B detector was trained.
- Under that corrected experiment, **frozen GAN checkpoint 1,500 did not
  demonstrate robust incremental downstream segmentation utility beyond matched
  real-defect exposure.** Mean Dice and IoU gains were approximately zero, mean
  PR-AUC gain was approximately zero, and precision and normal-image FPR
  regressed past their precommitted tolerances.

### What was NOT demonstrated

- **It was not demonstrated that the GAN improved detector performance.** No
  result in this repository supports that claim, at any budget, under any
  comparison.
- It was **not** demonstrated that the synthetic images are visually invalid,
  unrealistic, or malformed. V1 measured *downstream detector utility*, not
  generator quality. The 512 materialized samples passed every structural audit:
  valid-region-contained support, positive defect pixels, aligned shapes,
  train-only provenance. **"Utility not confirmed" is not "images are bad."**
- It was not demonstrated that synthetic augmentation is useless in general, nor
  that a different generator, objective, or sampling ratio would also fail.
- It was not demonstrated that checkpoint 1,000 or 2,000 would behave
  differently. Checkpoint 2,000 was never evaluated for utility; checkpoint 1,000
  failed the G2.2 recall gate and was never carried into G2.3B.
- Nothing was demonstrated about official-held-out-split behaviour. Every number
  in V1 is development-validation only.

### What remains uncertain

- **Whether a small true effect exists.** With three seeds and per-seed Dice
  gains spanning `-0.037` to `+0.019`, this experiment cannot distinguish a small
  positive effect from zero. It can and does exclude the large positive effect
  the precommitted gate required.
- **How much of the outcome is seed dispersion.** Seed 47 alone reverses the sign
  of the mean on Dice, IoU, and PR-AUC.
- Whether a different synthetic fraction, a different GAN checkpoint, a longer GAN
  run, or a different generator formulation would change the result. **None of
  these was tested, and none is authorized by this evidence.**
- Whether the generator produces genuinely novel defect *information* as opposed
  to plausible refinement of real-derived geometry. V1 measured this only
  indirectly, through detector utility. See §G.

### Why V1 stopped

The precommitted gate was designed before training precisely so the stopping
decision could not be negotiated afterwards. Five of eight criteria failed. Under
the rules agreed in advance, that is `stop_not_confirmed_g2_3b`, and the correct
action is to stop and record the negative result rather than to search for a
configuration that passes.

**A FAIL is not an invitation to relax the gate, add seeds, retune the threshold,
change the synthetic fraction, substitute another GAN checkpoint, or train the
GAN longer.** V1 must not be retrospectively tuned until it passes. The value of
this result is that it is attributable in a way G2.2's never was.

---

## G. V1 limitations discovered

The first two are **measured facts** from this repository's evidence. The
remainder are labelled explicitly.

1. **Measured.** Synthetic augmentation from frozen checkpoint 1,500 did not add
   robust detector utility beyond matched real-defect exposure at a mature
   budget, on three fresh seeds, under a precommitted operating point — on
   threshold-dependent quality and threshold-independent PR-AUC alike (§E).
2. **Measured.** Adversarial realism is not equivalent to downstream detector
   utility. G2.1 completed 2,000 sustained joint updates with no output-range,
   locality, invalid-gradient, or non-finite-gradient violations, and its
   checkpoints produced structurally valid composites — yet the downstream
   utility gate still failed. Generator-side health does not predict detector-side
   benefit.
3. **Measured (design fact, visible in the F1 manifest).** Defect geometry derives
   heavily from a **limited real defect/template library**: 209 defective
   development-training images yielding 235 connected components, of which 232
   were accepted, 96 border-touching. Every synthetic defect's shape originates in
   that pool, transformed by scale in `[0.9, 1.1]` and H/V flips.
4. **Interpretation, not measured.** The current residual, mask-conditioned
   synthesis formulation appears to emphasize *refinement and local realism* more
   than generating fundamentally novel defect information. The architecture
   supports this reading — a zero-initialized residual head that starts at exact
   identity, a maximum directional residual magnitude of `0.25`, `torch.where`
   preserving pixels outside the dilated support bit-exactly, and losses that are
   localized to the support and its inner boundary — but V1 ran no experiment that
   isolates "novel information" from "plausible refinement", so this is an
   architectural reading of the negative result, not a measurement of it.
5. **Interpretation, not measured.** V1 has limited explicit stochastic
   appearance/shape diversity. There is no latent noise input, no style vector,
   and no explicit diversity objective; variation comes from template choice,
   background choice, placement, scale, and flips. No diversity metric was
   computed, so the *magnitude* of this limitation is unquantified.
6. **Established by absence of evidence.** Synthetic-ratio tuning or additional
   GAN training steps are **not justified** by the existing evidence. G2.2 tested
   one ratio at one immature budget; G2.3B tested the same ratio at a mature
   budget with a proper control. No experiment in V1 varied the ratio or the GAN
   step count against a prevalence-matched control, so any claim that "more steps"
   or "a different ratio" would help is unsupported speculation, and acting on it
   would be exactly the retrospective tuning this project has refused throughout.

---

## H. Future / Version 2 — not yet implemented

**Nothing in this section exists. No V2 code, config, experiment, result, or
branch has been created.** This records the research direction under discussion
at the time V1 was frozen so it is not lost. It is not a plan of record, it is
not authorized, and it must not be read as a claim about anything that was built.

Direction under consideration:

- Generate genuinely **novel defect geometry** rather than only transforming
  real-derived masks.
- Introduce explicit **latent / style diversity** into the generator.
- Potentially separate **shape and appearance latents** so geometry and texture
  can vary independently.
- Target synthetic generation toward **detector weaknesses**, using training-only
  or out-of-fold evidence — never development validation, never the official
  held-out split.
- Add explicit **diversity-aware and utility-aware objectives** rather than
  relying on adversarial realism as a proxy for usefulness.
- Consider a **conditional diffusion / inpainting** approach later if a GAN-based
  V2 proves insufficient.

Any V2 experiment requires its own precommitted protocol, its own frozen gate,
and its own explicit authorization. It must not reuse V1's seeds for
confirmation, and it must not alter V1's recorded results.

---

## I. Exact reproducibility state

### Environment assumptions

| Component | Value at freeze |
|---|---|
| OS | Windows 11, PowerShell primary shell |
| Python | 3.14.2 (`.venv` at repository root) |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| BF16 support | required — training refuses a CPU fallback |

Reproducing G2.3B training requires CUDA with BF16. Exact bitwise reproduction on
different hardware is **not** claimed; schedule content hashes, composition, and
budgets are hardware-independent and will reproduce.

### Authoritative configurations

| Path | SHA-256 (raw file bytes) |
|---|---|
| `configs/final_real_baseline_bf16.json` | `fc1950be8d6c4950d52d6f46bc0e7d8f5df464e19c9f6f784ca6074bbabddff3` |
| `configs/gan_inputs.json` | `8e39d5c65c9bd4f141048db5df613902d4f086ca678c2402225c70c06f5d9437` |
| `configs/gan_architecture.json` | `62983c2d00cf77a000662f35085c80e53bf74df08c9f26dbfc5d901b5c45bf29` |
| `configs/gan_losses.json` | `4cbd09a3bfac9da6bcf2cb811d57b11e90ac01a131442ff5d2d849e7b2452082` |
| `configs/gan_training_2000.json` | `65a3d45886a44106aff200f761559144533fa40279d002b1dc49e7c2986586a6` |
| `configs/g2_2_detector_utility.json` | `87b8605b8586715ec4e90f320f8ab82cc1e85342c9a0f4d717327bf74cf68d02` |
| `configs/g2_3_diagnostic.json` | `be7b669a6258f6cadaaaf1a9ecb59ba895c7e4be9e9bf9f687e4a6ac014cabc2` |
| `configs/g2_3b_utility_confirmation.json` | `a2b20fad50ab4a36fa8d23db5361fe1ff5c57fdc550fa808d031fcab1d344173` |

Canonical **content** hashes recorded inside run artifacts (these are what the
runners compare, and they differ from raw-file hashes by design):

| Item | Canonical SHA-256 |
|---|---|
| `configs/gan_training_2000.json` (G2.1 identity) | `e801f22ba6c869c57b5816855fd25eb1c84fea9cff20564380c29c601022ff7c` |
| `configs/g2_3b_utility_confirmation.json` (G2.3B identity) | `1f9db6dab65d42b164e996b45087a3e5e4d195c79d8f8a538da2a640048ef549` |
| `reports/g2_3b/plan/precommitted_plan.json` (`plan_content_sha256`) | `d6918353e1515ee84bbac3cbab40743603042f5e70239e9545ff36cff3d6e7ac` |

### Data and split manifests

| Item | Value |
|---|---|
| `data/metadata/ksdd2_split_seed42.csv` | SHA-256 `024495c9673a7096c79f342cce58ad6dd5e7434951b9b61053e926ab7c8c9f07`, 616,772 bytes, 3,335 rows |
| Official train / development train | 1,981 rows (`train`,`train`) |
| Official train / development validation | 350 rows (`train`,`validation`) |
| Official held-out | 1,004 rows (`test`,`test`) — **never loaded** |
| GAN manifest content SHA-256 | `bf3cdad05f402cfdd785e1c88254687db2d9d3ded30f6fef419e49d0cfc18c38` |
| Split SHA-256 | `096ea5adee3aa08ac590decc7cc663dbb7d889764a0f645332c64d5ba15d2b9e` |

Loaders must consume the manifest; directory globbing is not an acceptable source
of sample identities. The nonconforming duplicates `train/10301 (copy).png` and
`train/10301_GT (copy).png` must never be added to any manifest or loader, and
must never be deleted.

### Important checkpoint hashes (whole file)

| Checkpoint | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/final_real_baseline_bf16_seed42/best.pt` | 339,558,727 | `6a2127fad5fca66108de38226b050b9ef7d09025c4528a5bf48285ffaabfd277` |
| `checkpoints/gan_training_2000/joint_1000.pt` | 79,262,487 | `801f60860f5f4d011c87f415090d96acbd93a13372feb8e0d8cac9881c50ae38` |
| `checkpoints/gan_training_2000/joint_1500.pt` | 79,262,487 | `5af1c6aafabcc0444117aa43209dcab168e57f4489259728e8f9066a4fdf1c81` |
| `checkpoints/gan_training_2000/joint_2000.pt` = `last.pt` | 79,262,487 | `82cfe4a70470ac2c7ffb5d4ecaf4f7357d63b698ed43a9b83e3dc297a0766f5f` |

All four re-verified at freeze time. G2.1 step-200 generator/discriminator
parameter and optimizer hashes matched the selected D-clip-10 smoke reference
exactly; those values are recorded in `docs/PROJECT_STATE.md`.

### Synthetic manifest hashes

| Item | SHA-256 |
|---|---|
| Pairing report content | `540a4637936c25ae9fd3678732bbc9d81e75f066e584e6d3ee078768f491ed33` |
| Checkpoint-1,000 synthetic manifest | `72e7e2862c47351c7f236136e5d829a2b456a440cf962c299f57780aa6974de3` |
| Checkpoint-1,500 synthetic manifest | `9eba21b4347dcdafafd9d0f90dd06b297cb58b2f7ee58f1887fed7a4cd62ca91` |

### Tests

`448 passed` at freeze, across 25 test files. Phase-specific suites:

| Suite | Tests | Scope |
|---|---:|---|
| `tests/test_g2_3b_protocol.py` | 63 | composition, budgets, initialization equality, deterministic schedules, synthetic identity, validation-only threshold selection, official-split refusal, gate arithmetic |
| `tests/test_g2_3b_durability.py` | 43 | atomic writes, run identity, resume compatibility, completed-epoch/arm protection, state round-trip, LR restoration, plan-unchanged assertions |
| `tests/test_g2_3_diagnostic.py` | 44 | G2.3A guards, histograms, curve/PR-AUC math, composition and mask primitives |

### Key report paths

| Path | Content |
|---|---|
| `reports/data_audit/summary.json` | Official dataset count/integrity audit |
| `reports/preprocessing/bbox_summary.json` | Native geometry and mask statistics |
| `reports/final_real_baseline_bf16_seed42/` | Accepted real-only reference evidence |
| `reports/gan_training/g2_1_2000/summary.json` | G2.1 sustained run summary |
| `reports/g2_2/synthetic_manifests/pairing_report.json` | G2.2 paired materialization audit |
| `reports/g2_2/pilot_seed42/pilot_summary.json` | G2.2 seed-42 pilot |
| `reports/g2_2/confirmation/confirmation_summary.json` | G2.2 terminal `stop_not_confirmed` |
| `reports/g2_3/diagnostic/diagnostic_summary.json` | G2.3A machine-readable summary |
| `reports/g2_3b/plan/precommitted_plan.json` | G2.3B frozen plan and schedule hashes |
| `reports/g2_3b/seed<SEED>/<ARM>.json` | G2.3B per-arm reports (9) |
| `reports/g2_3b/seed<SEED>/seed_summary.json` | G2.3B per-seed rollups (3) |
| `reports/g2_3b/confirmation_summary.json` | **G2.3B terminal result** |

### Safe inspection commands

These do not authorize a new experiment:

```powershell
git log --oneline
git show --stat 7c533d6
git status --short
Get-Content .\reports\g2_3b\confirmation_summary.json
Get-Content .\reports\g2_2\confirmation\confirmation_summary.json
Get-Content .\reports\g2_3\diagnostic\diagnostic_summary.json
Get-FileHash .\checkpoints\gan_training_2000\joint_1500.pt -Algorithm SHA256
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode plan
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode confirm
.\.venv\Scripts\python.exe -m pytest
```

`--mode plan` rewrites `precommitted_plan.json` with identical bytes and
`--mode confirm` recomputes the gate from the nine completed arms; both are
deterministic, train nothing, and were verified to be non-destructive.

### Commands that MUST NOT be run without new explicit authorization

Each of these would extend or alter a terminated experiment:

```powershell
# G2.1 fresh or resume -- checkpoints are frozen.
.\.venv\Scripts\python.exe .\scripts\train_gan.py --config .\configs\gan_training_2000.json
.\.venv\Scripts\python.exe .\scripts\train_gan.py --config .\configs\gan_training_2000.json --resume

# G2.2 materialization / pilot / confirmation -- completed and terminal.
.\.venv\Scripts\python.exe .\scripts\build_g2_2_synthetic_manifests.py --config .\configs\g2_2_detector_utility.json
.\.venv\Scripts\python.exe .\scripts\train_g2_2_detector_utility.py --config .\configs\g2_2_detector_utility.json --mode pilot
.\.venv\Scripts\python.exe .\scripts\train_g2_2_detector_utility.py --config .\configs\g2_2_detector_utility.json --mode confirmation

# G2.3B training -- all nine arms are complete; rerunning is forbidden.
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode train --seed 45
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode train --seed 46
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode train --seed 47
```

**There is no official-held-out-split mode.** `scripts/train_g2_3b_utility.py`
offers `plan`, `train`, and `confirm` and nothing else;
`assert_permitted_split` admits `train` and `validation` only and raises
`OfficialTestAccessError` otherwise; a test asserts the script contains no quoted
reserved-split literal. `access_policy.official_test_allowed` and
`access_policy.official_test_allowed_after_confirmation` are both `false` and the
config loader refuses to run if either is flipped. The G2.2-era
`--mode official-test` route on `train_g2_2_detector_utility.py` remains
forbidden.

---

## J. Local / non-Git artifacts

Large artifacts are intentionally Git-ignored (see `.gitignore`). Their absence
from `git status` does **not** mean they are disposable. At freeze the working
tree holds:

| Path | Size | Content | Needed for exact reproduction? |
|---|---:|---|---|
| `data/` | ~2.0 GB | Raw/extracted/processed KSDD2 (only `data/metadata/ksdd2_split_seed42.csv` is tracked) | **Yes** — required for any retraining or re-materialization |
| `checkpoints/` | ~17 GB | 111 `.pt` files: baselines, GAN smoke, G2.1 numbered, G2.2 detectors, all 18 G2.3B checkpoints | **Yes** for exact reuse of frozen weights; the G2.1 `joint_1500.pt` is required by any G2.3B re-verification |
| `data/synthetic/` | ~319 MB | 2,048 files — 512 materialized samples each for `checkpoint_1000`, `checkpoint_1500`, and `common` masks/valid regions | **Yes** — G2.3B arm C reads these directly and re-hashes all 1,536 per-row files |
| `reports/g2_3b/plan/seed*/` | ~48 MB | Nine expanded 23,808-slot schedules | No — regenerable byte-identically by `--mode plan`; their hashes are tracked in the plan |
| `reports/g2_3b/**/*_epochs.json` | small | Per-epoch progress logs | No — superseded by the tracked per-arm reports |
| `reports/g2_3/diagnostic/threshold_curves/` | bulk | Six 3,999-point curves | No — hashes recorded in `threshold_calibration.json` |
| `reports/g2_3/diagnostic/synthetic_mask_records.json` | bulk | Per-mask records for 512 masks | No — compact summary is tracked |
| `reports/g2_2/**/*_schedule.json`, `*_progress.json`, `synthetic_manifests/checkpoint_*.json` | bulk | G2.2 expanded schedules, progress, full 512-row manifests | Partly — manifests are needed for exact re-materialization; content hashes are tracked |

**Code-level reproduction** (rerunning the pipeline from scratch on a fresh
machine) needs: this repository, the KSDD2 dataset, and a CUDA+BF16 GPU.

**Exact reproduction** (reusing the identical frozen weights and synthetic
pixels, and re-verifying recorded hashes) additionally needs the local
`checkpoints/`, `data/`, and `data/synthetic/` trees, which are **not** in Git
and exist only on this machine.

### Untracked artifacts that must never be staged or deleted

Pre-existing, unrelated to any G2.3B commit:

- `reports/g2_2.zip`
- `reports/gan_inputs/manifest.json`
- `reports/gan_inputs/sampling_audit_f1_4.json`
- `reports/gan_inputs/sampling_audit_f1_4.md`
- `reports/gan_inputs/summary.json`
- `reports/gan_inputs/summary.md`
- `reports/gan_inputs/visualizations/`

Also never stage the duplicate KSDD2 files `train/10301 (copy).png` and
`train/10301_GT (copy).png`, and never delete them.

### Note for a public GitHub release

A fresh agent preparing this repository for release should be aware that:

- No dataset, checkpoint, or synthetic image is in Git; a public clone reproduces
  **code and compact reports only**.
- `reports/gan_inputs/*` and `reports/g2_2.zip` are untracked local artifacts. They
  are not part of the project's committed history and should be reviewed
  separately rather than swept into a release commit.
- `reports/final_real_baseline/precision_diagnostic/` is Git-ignored because it
  contains local absolute paths; sanitized findings are tracked.
- No credential, token, or private path is expected in tracked files, but a
  release preparation pass should verify that independently.

---

## K. Obsolete active-run state

`docs/G2_3B_ACTIVE_RUN_STATE.md` was written mid-execution, on 2026-08-23, while
G2.3B was still running. **That run has since completed.** The document is
retained unmodified apart from a completion banner at its head, because its
operational record — the durability design, the resume procedure, and the
process-independence analysis — is genuine project history worth keeping.

Read it as a historical operations log, **not** as a description of a running
experiment. Where it says "Completed arms: 2 of 9" or "Current arm: seed 45
`gan_1500`", those statements were accurate when written and are now superseded
by §E of this document and by `reports/g2_3b/confirmation_summary.json`.

---

## L. Resume boundary

**Version 1 is frozen at the final G2.3B result, commit `7c533d6`.**

If V1 is ever resumed:

1. Read this document and `docs/g2-3b-results.md` first, then
   `docs/PROJECT_STATE.md` for pre-G2.2 detail.
2. Verify the recorded hashes still hold before touching any frozen artifact.
3. Create a **new, explicitly authorized experiment** with its own precommitted
   protocol and its own frozen gate. Do not alter, reinterpret, or re-run a
   historical result to obtain a different outcome.
4. Never construct, inspect, count, or evaluate the official held-out split
   without a separate explicit authorization. The current configuration forbids it
   unconditionally, and the G2.3B FAIL forecloses rather than opens that route.

**Version 2 must branch conceptually from this frozen state and must not rewrite
V1.** V1's commits, reports, configs, gates, and recorded results are immutable
history. A V2 experiment may cite them, must not modify them, and must not reuse
seeds 42–47 for its own confirmation.

---

## Non-negotiable handoff constraints

Carried forward from `docs/PROJECT_STATE.md` and extended by G2.3A/G2.3B. All
remain in force.

- Do not resume, extend, fine-tune, or update any G2.1 GAN checkpoint.
- Do not rerun G2.1, G2.2, G2.3A, or any G2.3B arm.
- Do not run another GAN or detector hyperparameter sweep to work around a
  terminal decision.
- Do not reinterpret G2.2 as confirmed, or checkpoint 1,500 as a pilot winner
  that "really" passed.
- Do not select checkpoint 1,000 on its Dice; it failed the precommitted recall
  constraint. Do not substitute checkpoint 2,000; it was never evaluated for
  utility.
- Do not create a `best` GAN checkpoint or alias from monitor confidence or
  visual panels.
- Do not weaken, reorder, or re-derive the eight-criterion G2.3B gate after the
  fact, and do not add seeds or retune thresholds to change its outcome.
- Do not construct, inspect images from, tune on, or evaluate the official
  held-out split.
- Do not use development validation in template extraction, GAN training,
  synthetic source sampling, or detector training.
- Do not change the dataset split, masks, geometry, thresholds, seeds, hashes, or
  recorded reports retroactively.
- Do not stage or delete the unrelated local artifacts listed in §J.
