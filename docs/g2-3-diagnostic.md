# G2.3A post-hoc validation-only diagnostic

**POST-HOC DIAGNOSTIC.** Every number, threshold sweep, matched operating point,
and stratum comparison in this document and under `reports/g2_3/diagnostic/` is
diagnostic evidence produced after the fact. None of it is a gate, a selection, a
retuned threshold, or a new experimental arm.

**The G2.2 terminal decision is unchanged and remains `stop_not_confirmed`.**
G2.3A did not rescue, reinterpret, or reopen G2.2. Its purpose is to explain why
the precommitted three-seed confirmation failed and to leave clean evidence for
designing a future experiment.

## What G2.3A did and did not do

| Action | Count |
|---|---:|
| GAN optimizer updates | 0 |
| Detector optimizer updates | 0 |
| Synthetic samples regenerated | 0 |
| G2.2 pilot/confirmation reruns | 0 |
| G2.2 thresholds, seeds, gates, reports, checkpoints, configs modified | 0 |
| G2.1 checkpoint 2,000 evaluations | 0 |
| Checkpoint 1,000 selections | 0 |
| Official-test rows constructed, counted, inspected, or evaluated | 0 |
| Validation-only forward passes over already-trained checkpoints | 6 |

The diagnostic reads six already-trained G2.2 detector checkpoints (seeds 42, 43,
44 × `real_only`, `checkpoint_1500`), the frozen G2.2 schedules and synthetic
manifests, the 512 locally materialized synthetic masks, and the historical BF16
baseline reports. It writes only into `reports/g2_3/diagnostic/`.

## Leakage and access guards

`assert_validation_only_split` accepts `validation` and nothing else. Asking for
`test` raises `OfficialTestAccessError`; asking for `train` raises `ValueError`.
`scripts/run_g2_3_diagnostic.py` contains no literal `"test"` split string at all,
and a test asserts that.

Synthetic provenance is proven **positively**: every template and background
identity is shown to be a member of `development_split=train`. Because the
development splits partition the official train split and official test maps only
to `development_split=test`, train membership already excludes both validation and
official test. No official-test row is read, listed, or counted. The validation
intersection is checked independently as redundant evidence and is empty.

## Method

Deterministic validation-only inference replays the exact G2.2 evaluation path:
manifest-only `KSDD2FullImageDataset` validation split (350 images), batch size 4,
no shuffle, `model.eval()`, `torch.inference_mode()`, BF16 autocast, float32
sigmoid, metrics on native `valid_region` pixels only.

Threshold curves are exact at every grid point, not sampled. For each image the
valid-pixel probabilities are bucketed into survivor histograms, so
`TP(t)`/`FP(t)`/`FN(t)` at a grid point are exact integer counts. The grid has
3,999 strictly increasing points: a uniform probability lattice of step `0.0005`
unioned with a uniform logit lattice of step `0.02` over `[-20, 20]` for tail
resolution. `0.5` is always a grid point. Normal-image false-positive rate is
exact at any threshold from the 313 per-image maximum probabilities.

Pixel PR-AUC uses the step rule `sum((R_k - R_{k+1}) * P_k)` (average precision),
not trapezoidal interpolation, which is optimistic across precision jumps.

**Verification.** For all six checkpoints the diagnostic reproduced the recorded
G2.2 threshold-0.5 global Dice, IoU, precision, recall, and normal-image FPR with
a maximum absolute delta of exactly `0.0`. Repeated forward passes over the first
batch were bitwise identical for all six. The diagnostic therefore measures the
same models through the same path that produced the frozen G2.2 numbers.

---

## Question 1 — detector convergence and control instability

### Where 2,000 updates sits on the historical baseline trajectory

The accepted BF16 real-only baseline ran 496 optimizer updates per epoch for 12
epochs (5,952 total). **2,000 updates is 4.032 baseline epochs, i.e. 33.6% of the
accepted baseline budget.**

| Point | Updates | LR | Train loss | Val Dice @0.5 | Val precision @0.5 | Val recall @0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline epoch 4 | 1,984 | 3e-4 | 0.790137 | 0.634924 | 0.682107 | 0.593845 |
| Baseline epoch 5 | 2,480 | 3e-4 | 0.613775 | 0.634734 | 0.586971 | 0.690958 |
| Baseline epoch 11 (accepted) | 5,456 | 1.5e-4 | 0.246286 | **0.777700** | 0.854770 | 0.713379 |

Both regimes still ran at LR `3e-4` at the 2,000-update mark — the baseline's
`ReduceLROnPlateau` halving happened only after epoch 9 — so the learning-rate
schedule does not explain the gap *at* update 2,000. It does explain part of the
distance still to travel: the baseline's jump from 0.653 (epoch 10) to 0.778
(epoch 11) coincides with the LR halving, and G2.2's constant-LR arms never
received it.

### Are the G2.2 controls underconverged?

Yes, on every available indicator.

| Indicator | Result |
|---|---|
| Every control Dice below the baseline's 2,000-update-equivalent epoch (0.6349) | true |
| Every control Dice below the accepted baseline (0.7777) | true |
| Lowest control normal-image FPR | 0.597444 (accepted baseline: 0.041534) |
| Training loss still falling at update 2,000 in every arm | true |
| Best achievable Dice of any G2.2 checkpoint at *any* threshold | 0.672931 |
| Accepted baseline Dice at threshold 0.5 / at 0.05 | 0.777700 / 0.798251 |

No G2.2 checkpoint reaches the accepted baseline's Dice at any threshold, so the
gap is a capability gap, not only a threshold placement gap.

Control metrics at threshold 0.5 across the three seeds:

| Metric | seed 42 | seed 43 | seed 44 | s.d. | range |
|---|---:|---:|---:|---:|---:|
| Global Dice | 0.406363 | 0.623450 | 0.517701 | 0.108556 | 0.217087 |
| Pixel precision | 0.318976 | 0.522620 | 0.455827 | 0.103811 | 0.203644 |
| Pixel recall | 0.559698 | 0.772487 | 0.599009 | 0.113225 | 0.212789 |
| Normal-image FPR | 0.926518 | 0.597444 | 0.645367 | 0.177779 | 0.329073 |

For scale, the single well-behaved historical baseline run oscillated between
0.6261 and 0.8008 Dice over epochs 3–10 (s.d. 0.0585) on its own trajectory. A
last-iterate reading at a fixed budget in this regime is a noisy statistic even
within one run.

### Does the real-sampling policy change the interpretation?

Not materially for effective defect prevalence. The historical baseline used a
deterministic weighted sampler with `target_defective_fraction = 0.5` and observed
per-epoch defective fractions of 0.4836–0.5114 (mean 0.4973). The G2.2 real-only
control draws exactly 4,000 normal and 4,000 defective real slots — 50.0%. Both
regimes trained the control at ~50% defective real samples.

Two policy differences that do matter, and are not about prevalence:

1. **Checkpoint selection.** The baseline's 0.7777 is a validation-loss-selected
   best epoch out of 12. Every G2.2 arm reports its *unselected last iterate* at
   update 2,000. This is not a like-for-like statistic.
2. **Epoch structure and LR schedule.** The baseline had `ReduceLROnPlateau`,
   early stopping, and 12 passes; G2.2 had a flat 2,000-update constant-LR budget.

### Were training losses still improving near update 2,000?

Yes in all seven saved arms, but only coarsely resolvable. Only the run mean and
the final-100-update mean survive in `*_progress.json` — no per-update or
per-epoch curve was ever recorded.

| Arm | Run mean loss | Final-100 mean | Delta |
|---|---:|---:|---:|
| seed42 real_only | 0.865635 | 0.766899 | −0.098736 |
| seed42 checkpoint_1000 | 0.768237 | 0.643836 | −0.124401 |
| seed42 checkpoint_1500 | 0.792507 | 0.642185 | −0.150323 |
| seed43 real_only | 0.870827 | 0.789541 | −0.081287 |
| seed43 checkpoint_1500 | 0.802705 | 0.704586 | −0.098119 |
| seed44 real_only | 0.868127 | 0.778903 | −0.089224 |
| seed44 checkpoint_1500 | 0.786694 | 0.630154 | −0.156540 |

The control final-100 losses (0.767–0.790) bracket the baseline's epoch-4 training
loss (0.790), which is independent corroboration that 2,000 updates sits near
baseline epoch 4.

### Is seed 43 abnormal GAN behaviour or unusually strong real-only behaviour?

**Unusually strong real-only behaviour.** The evidence is one-sided:

| Quantity | seed 42 | seed 43 | seed 44 | s.d. |
|---|---:|---:|---:|---:|
| real_only recall @0.5 | 0.559698 | **0.772487** | 0.599009 | 0.113225 |
| checkpoint_1500 recall @0.5 | 0.579934 | 0.565479 | 0.614140 | 0.024990 |
| real_only pixel PR-AUC | 0.502874 | **0.721202** | 0.579923 | 0.110730 |
| checkpoint_1500 pixel PR-AUC | 0.606494 | **0.692547** | 0.629677 | 0.044529 |

Seed 43's *control* is the strongest of the three controls by a wide margin, and
its recall sits 0.1931 above the mean of the other two controls. Seed 43's *arm*
is also the strongest of the three arms — its recall is only 0.0316 below the mean
of the other two arms. The synthetic arms are roughly 4.5× more stable than the
controls (recall s.d. 0.0250 vs 0.1132; PR-AUC s.d. 0.0445 vs 0.1107). The
−0.2070 seed-43 recall delta is therefore produced mainly by control-side variance,
not by seed-43 GAN pathology.

### What these artifacts cannot establish

- No G2.2 arm was trained past 2,000 updates. Nothing here shows that a longer
  budget would have changed the confirmation outcome. **No causal claim is made.**
- Per-update and per-epoch validation curves were never recorded for G2.2 arms;
  the loss trend near update 2,000 is only two-point evidence.
- Per-epoch normal-image FPR was never recorded for the historical baseline, so
  only its final accepted value can be compared.

---

## Question 2 — threshold and calibration effect

All sweeps below are **POST-HOC DIAGNOSTIC**. Threshold 0.5 remains the frozen
G2.2 comparison threshold; nothing here retunes it, and none of these thresholds
is ever applied to the official test split.

### Calibration of the fixed 0.5 threshold

| Checkpoint | Best-Dice threshold (diagnostic only) | Dice there |
|---|---:|---:|
| seed42 real_only | 0.99748 | 0.548359 |
| seed42 checkpoint_1500 | 0.97250 | 0.600536 |
| seed43 real_only | 0.95500 | 0.672931 |
| seed43 checkpoint_1500 | 0.58600 | 0.641477 |
| seed44 real_only | 0.98433 | 0.585466 |
| seed44 checkpoint_1500 | 0.97450 | 0.634965 |

**All six G2.2 checkpoints prefer a threshold above 0.5**, several of them far
above. The accepted, converged baseline preferred `0.05` — below 0.5. Under a
`pos_weight = 5` BCE-plus-Dice objective, an underconverged model over-predicts,
so 0.5 falls on the steep, high-FPR shoulder of its curve. Small differences in
where an arm's shoulder lands then translate into large differences in
threshold-0.5 recall, precision, and normal-image FPR. Logit distributions confirm
the shift: e.g. seed 43 median defect-pixel logit is 4.94 for the control and 2.54
for the arm, while median background logits are −8.12 and −8.43 — the arm is
uniformly more conservative on defect pixels.

### Per-seed matched operating points

**A — both models at threshold 0.5** (reproduces the frozen G2.2 numbers exactly)

| Seed | Model | Dice | IoU | Precision | Recall | Normal FPR |
|---:|---|---:|---:|---:|---:|---:|
| 42 | real_only | 0.406363 | 0.254991 | 0.318976 | 0.559698 | 0.926518 |
| 42 | checkpoint_1500 | 0.572083 | 0.400642 | 0.564442 | 0.579934 | 0.738019 |
| 43 | real_only | 0.623450 | 0.452908 | 0.522620 | 0.772487 | 0.597444 |
| 43 | checkpoint_1500 | 0.640538 | 0.471171 | 0.738574 | 0.565479 | 0.303514 |
| 44 | real_only | 0.517701 | 0.349255 | 0.455827 | 0.599009 | 0.645367 |
| 44 | checkpoint_1500 | 0.587661 | 0.416091 | 0.563371 | 0.614140 | 0.492013 |

**B — checkpoint_1500 rethresholded to match the control's threshold-0.5 recall**

Feasible for all three seeds (recall gaps 2.6e-05, 7.2e-04, 6.6e-06).

| Seed | Threshold | Recall | Dice | IoU | Precision | Normal FPR | ΔDice | ΔPrecision | ΔFPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.76300 | 0.559672 | 0.587523 | 0.415953 | 0.618292 | 0.680511 | **+0.181161** | +0.299316 | −0.246006 |
| 43 | 0.01015 | 0.773209 | 0.534744 | 0.364949 | 0.408698 | 0.690096 | **−0.088706** | −0.113922 | +0.092652 |
| 44 | 0.69650 | 0.599016 | 0.603654 | 0.432310 | 0.608365 | 0.437700 | **+0.085953** | +0.152537 | −0.207668 |

**C — checkpoint_1500 rethresholded to match the control's threshold-0.5
normal-image FPR**

Feasible for all three seeds (gaps 3.2e-03, 0.0, 0.0; one image is 1/313 = 0.0032).

| Seed | Threshold | Normal FPR | Dice | IoU | Precision | Recall | ΔDice | ΔPrecision | ΔRecall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.03450 | 0.923323 | 0.479359 | 0.315235 | 0.378054 | 0.654829 | +0.072996 | +0.059078 | **+0.095131** |
| 43 | 0.01799 | 0.597444 | 0.567050 | 0.395722 | 0.462230 | 0.733353 | −0.056400 | −0.060390 | **−0.039134** |
| 44 | 0.13250 | 0.645367 | 0.532892 | 0.363227 | 0.447914 | 0.657664 | +0.015192 | −0.007913 | **+0.058655** |

### Pixel precision-recall frontiers

| Seed | real_only PR-AUC | checkpoint_1500 PR-AUC | Delta | Verdict |
|---:|---:|---:|---:|---|
| 42 | 0.502874 | 0.606494 | +0.103619 | operating-point shift, not frontier degradation |
| 43 | 0.721202 | 0.692547 | **−0.028655** | frontier not uniformly better |
| 44 | 0.579923 | 0.629677 | +0.049754 | operating-point shift, not frontier degradation |

### Is seed 43's recall regression calibration or a genuinely worse frontier?

**Both, in that order of magnitude — and neither reads as GAN-induced degradation.**

- The regression is **not a capability ceiling.** The control's 0.7725 recall is
  fully reachable by rethresholding checkpoint_1500 to `0.01015`, which achieves
  0.7732. The raw −0.2070 figure at threshold 0.5 is an operating-point artifact
  of the fixed threshold, not evidence that the arm cannot recall those pixels.
- But **a residual genuine deficit remains** against that particular control:
  after matching recall, the arm is behind on Dice by 0.0887 and on precision by
  0.1139; after matching normal FPR it is still behind on recall by 0.0391; and its
  PR-AUC is 0.0287 lower. Those are frontier facts, not threshold facts.
- The residual is nevertheless **control-side, not arm-side.** Seed 43's control
  has the highest PR-AUC of any of the six checkpoints and seed 43's arm has the
  highest PR-AUC of any arm. The arm did not get worse on seed 43; the control got
  unusually good.

### Stratum comparisons (POST-HOC DIAGNOSTIC)

Pixel PR-AUC, `real_only` / `checkpoint_1500`:

| Seed | Border | Non-border | Small | Medium | Large |
|---:|---|---|---|---|---|
| 42 | 0.5808 / 0.7539 | 0.9164 / 0.9193 | 0.6016 / 0.6445 | 0.8455 / 0.8715 | 0.6218 / 0.7881 |
| 43 | 0.8096 / 0.7918 | 0.9112 / 0.9121 | 0.6778 / 0.7134 | 0.8262 / 0.8376 | 0.8573 / 0.8274 |
| 44 | 0.7391 / 0.7599 | 0.9151 / 0.9175 | 0.5984 / 0.6981 | 0.9151 / 0.9070 | 0.7779 / 0.7868 |

The synthetic arm improves stratum PR-AUC in **all three seeds** only for
**small defects** (+0.043, +0.036, +0.100) and **non-border defects** (+0.003,
+0.001, +0.002; small but consistently signed). Border, medium, and large are
seed-dependent and dominated by seed 42's large gains and seed 43's small losses.

Seed 43 per-stratum Dice, control at 0.5 versus the arm rethresholded to that
stratum's control recall, is the sharpest view of the residual deficit:

| Stratum | Control Dice @0.5 | Arm Dice at matched stratum recall |
|---|---:|---:|
| Border | 0.7657 | 0.7277 |
| Non-border | 0.7664 | 0.7613 |
| Small | 0.5815 | **0.6226** |
| Medium | 0.7224 | 0.7080 |
| Large | 0.7938 | 0.7489 |

Even on seed 43 the arm is ahead on small defects after recall matching; its
deficit is concentrated in border and large defects.

Full per-threshold curves for every checkpoint are in
`reports/g2_3/diagnostic/threshold_curves/*.csv` (3,999 rows each, hashed).

---

## Question 3 — training-composition confound

### Exact schedule composition

Recomputed from the saved 8,000-row schedules against the manifest training-row
labels. Every saved schedule's recomputed content hash matches its recorded hash,
and every schedule matches the composition the implementation is designed to
produce.

| Seed | Arm | Normal real | Defective real | Synthetic | Effective defective | Effective normal | Effective defective fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | real_only | 4,000 | 4,000 | 0 | 4,000 | 4,000 | 0.500 |
| 42 | checkpoint_1000 | 3,000 | 3,000 | 2,000 | 5,000 | 3,000 | **0.625** |
| 42 | checkpoint_1500 | 3,000 | 3,000 | 2,000 | 5,000 | 3,000 | **0.625** |
| 43 | real_only | 4,000 | 4,000 | 0 | 4,000 | 4,000 | 0.500 |
| 43 | checkpoint_1500 | 3,000 | 3,000 | 2,000 | 5,000 | 3,000 | **0.625** |
| 44 | real_only | 4,000 | 4,000 | 0 | 4,000 | 4,000 | 0.500 |
| 44 | checkpoint_1500 | 3,000 | 3,000 | 2,000 | 5,000 | 3,000 | **0.625** |

That is exactly the hypothesized layout, verified rather than assumed:

```
real_only:      50.0% normal real + 50.0% defective real            = 50.0% defective
synthetic arm:  37.5% normal real + 37.5% defective real
                                  + 25.0% synthetic defective       = 62.5% defective
```

### Synthetic mask integrity — hard check on all 512 local masks

| Check | Result |
|---|---|
| Masks read from disk | 512 / 512 |
| Every sample has at least one positive **valid** defect pixel | **true** |
| Every mask's support lies entirely inside `valid_region` | **true** (0 outside pixels) |
| Image / mask / valid-region shapes aligned | true for all 512 |
| Masks and valid regions shared bit-for-bit between the two checkpoints | true |
| Rows declaring `official_split=train` and `development_split=train` | 512 / 512 |
| Distinct template/background source identities | 617, all in development train |
| Detector-validation source overlap | 0 |
| Official-test rows read, counted, or inspected | 0 |
| Min / median / max positive valid defect pixels | 23 / 1,615.5 / 32,426 |

### Classification

**CONFIRMED: this is a class-prevalence confound.** G2.2 simultaneously changed
two variables in the same comparison — the *content* of 25% of the training stream
(real → GAN-refined synthetic) and the *effective defect prevalence* of the stream
(50.0% → 62.5%, a delta of +0.125). Because every synthetic sample is defective and
synthetic samples displace a balanced real draw rather than a defective one, no
G2.2 comparison — pilot or confirmation, checkpoint 1,000 or 1,500 — can attribute
its measured effect to synthetic image content alone.

This is a direct, mechanical explanation for the *direction* of the observed
deltas: the arms shifted toward higher precision, much lower normal-image FPR, and
lower threshold-0.5 recall, which is what a higher-defect-prevalence, more
confident training stream produces at a fixed threshold.

---

## Attribution summary

The evidence favours a **combination**, with the components ordered by strength:

1. **Class-prevalence confounding — established as fact.** Verified from the
   schedules and all 512 masks. 50.0% vs 62.5% effective defective. This alone
   makes the G2.2 comparison non-attributable to synthetic content.
2. **Detector underconvergence and control instability — strongly supported.**
   2,000 updates is 33.6% of the accepted budget and lands at baseline epoch ~4.
   Every arm's loss was still falling. Control Dice ranged 0.406–0.623 (s.d.
   0.109); control normal FPR ranged 0.597–0.927 against the accepted baseline's
   0.042. Recall variance is 4.5× larger in the controls than in the arms.
3. **Calibration / operating-point shift — strongly supported.** All six
   checkpoints prefer a threshold well above 0.5 while the converged baseline
   prefers 0.05. The control's recall is reachable by rethresholding the arm in
   every seed, including seed 43. At matched recall the arm gains Dice in seeds 42
   and 44 (+0.181, +0.086).
4. **Genuine representation degradation — partially supported, seed-43 only and
   not GAN-attributable.** Seed 43's arm is behind its own control after matching
   recall (Dice −0.089) and on PR-AUC (−0.029). But that arm is the best of the
   three arms and that control is the best of the three controls, so the deficit
   tracks control variance rather than a seed-43 GAN failure. Seeds 42 and 44 show
   the opposite sign on every frontier measure.

## What remains uncertain

- **Whether a longer budget would change the outcome.** Unknowable from these
  artifacts. No arm ran past 2,000 updates and G2.3A trained nothing.
- **Whether removing the prevalence confound alone would flip the recall result.**
  The confound is proven; its magnitude of effect is not measurable without a
  prevalence-matched arm, which does not exist.
- **How much of seed 43's residual frontier deficit is real signal.** With n = 3
  seeds and control s.d. of 0.111 in PR-AUC, a single −0.029 delta is well inside
  control-side noise. It cannot be resolved as either real or noise here.
- **Whether the fine-grained loss trend near update 2,000 was still steep.** Only
  a two-point comparison (run mean vs final-100 mean) survives.
- **Whether checkpoint 1,500 is a good generator.** G2.3A measured downstream
  detectors, not generator quality, and the confound means even that measurement
  is not attributable to synthetic content.
- **All validation-only.** Nothing here says anything about official-test
  behaviour, and nothing here authorizes measuring it.

## Artifacts

| Path | Content |
|---|---|
| `reports/g2_3/diagnostic/convergence_audit.json` | Q1 budget mapping, baseline trajectory, dispersion, loss progress, evidence limits |
| `reports/g2_3/diagnostic/threshold_calibration.json` | Q2 per-checkpoint identities/hashes, reproduction check, PR-AUC, distributions, strata at 0.5, best-Dice threshold |
| `reports/g2_3/diagnostic/matched_operating_points.json` | Q2 comparisons A/B/C per seed, stratum-matched recall, frontier verdicts |
| `reports/g2_3/diagnostic/threshold_curves/*.csv` | Six full 3,999-point threshold curves, each hashed (bulk local artifact, Git-ignored; the hashes are recorded in `threshold_calibration.json`) |
| `reports/g2_3/diagnostic/schedule_composition_audit.json` | Q3 exact per-arm counts, expected-vs-observed, prevalence classification |
| `reports/g2_3/diagnostic/synthetic_mask_integrity.json` | Q3 compact hard-check summary, provenance proof, and the per-record content hash |
| `reports/g2_3/diagnostic/synthetic_mask_records.json` | Q3 per-mask records for all 512 masks (bulk local artifact, Git-ignored) |
| `reports/g2_3/diagnostic/diagnostic_summary.json` | Concise machine-readable summary and honoured-constraints block |
| `configs/g2_3_diagnostic.json` | Frozen diagnostic configuration and access policy |
| `src/defectgen/training/g2_3_diagnostic.py` | Guards, histograms, curve/PR-AUC math, composition and mask primitives |
| `scripts/run_g2_3_diagnostic.py` | Stage runner |
| `tests/test_g2_3_diagnostic.py` | 44 focused tests |

## Commands

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_g2_3_diagnostic.py --stage convergence
.\.venv\Scripts\python.exe .\scripts\run_g2_3_diagnostic.py --stage masks
.\.venv\Scripts\python.exe .\scripts\run_g2_3_diagnostic.py --stage schedules
.\.venv\Scripts\python.exe .\scripts\run_g2_3_diagnostic.py --stage thresholds
.\.venv\Scripts\python.exe .\scripts\run_g2_3_diagnostic.py --stage summary
.\.venv\Scripts\python.exe -m pytest .\tests\test_g2_3_diagnostic.py
```

`--stage all` runs everything in order. No stage can train, update a checkpoint,
regenerate synthetic data, or construct the official test split.

## Explicit non-authorizations

G2.3A does **not** authorize, design, or imply approval for: a G2.3B experiment, a
longer GAN run, a longer detector budget, a prevalence-matched rerun, additional
seeds, a different synthetic fraction, evaluating G2.1 checkpoint 2,000, selecting
checkpoint 1,000, retuning the G2.2 threshold, or any official-test access. The
G2.2 decision stands at `stop_not_confirmed`.
