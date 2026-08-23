> **SUPERSEDED AS THE CURRENT STATUS — see [`docs/V1_FINAL_STATE.md`](V1_FINAL_STATE.md).**
>
> This document remains the accurate, unmodified handoff **through G2.2**
> (commit `a41be83`) and everything it records is still correct. It is not the
> current project status: G2.3A (`e1171a1`), the precommitted G2.3B protocol
> (`6e566a5`), and the terminal G2.3B result (`7c533d6`) all came afterwards.
>
> The Version 1 terminal status is `stop_not_confirmed_g2_3b`, recorded in
> [`docs/V1_FINAL_STATE.md`](V1_FINAL_STATE.md),
> [`docs/g2-3b-results.md`](g2-3b-results.md), and
> `reports/g2_3b/confirmation_summary.json`. Read the final-state document first.

# Project state through G2.2

This is the factual handoff state of `defect-generator-detector` through commit
`a41be83` (`feat: evaluate downstream GAN utility`). The commit that adds this
document is documentation-only; it does not supersede or reopen any experimental
decision recorded at `a41be83`.

## Terminal status

The project has completed the bounded G2.2 downstream detector-utility experiment.
The terminal decision is:

```text
stop_not_confirmed
```

GAN checkpoint 1,500 passed the one-seed pilot but failed the precommitted
three-seed confirmation because mean pixel recall regressed by `0.0572135`, beyond
the permitted `0.01`. GAN checkpoint 1,000 failed the pilot recall gate. There is
therefore no accepted synthetic detector configuration, no authorized final
detector checkpoint, and no basis for training the GAN longer.

The official KSDD2 test split has never been constructed or evaluated by the GAN,
G2.1, or G2.2 workflows. G2.2 records `official_test_access_count = 0`; no
`reports/g2_2/official_test/evaluation.json` exists. Do not run the official-test
mode: the failed confirmation does not authorize it.

## Non-negotiable handoff constraints

- Do not resume, extend, fine-tune, or otherwise update any G2.1 GAN checkpoint.
- Do not rerun G2.1, G2.2 materialization, the G2.2 pilot, or its confirmation.
- Do not run another GAN or detector hyperparameter sweep to work around G2.2.
- Do not reinterpret checkpoint 1,500 as confirmed. It was only the pilot winner.
- Do not select checkpoint 1,000 based on its larger seed-42 Dice; it failed the
  precommitted recall constraint.
- Do not create a `best` GAN checkpoint or alias from monitor confidence or visual
  panels. G2.1 checkpoints are numbered recovery/audit milestones only.
- Do not construct, inspect images from, tune on, or evaluate the official test.
- Do not use development validation in template extraction, GAN training,
  synthetic source sampling, or detector training.
- Do not change the dataset split, masks, geometry, thresholds, seeds, hashes, or
  recorded reports retroactively.
- Do not add the nonconforming duplicate files `train/10301 (copy).png` and
  `train/10301_GT (copy).png` to any manifest or loader. Do not delete them either.
- Do not stage or rewrite unrelated local F1 artifacts when making later changes.

## Phase and commit ledger

The important commits, in chronological order, are:

| Commit | Phase/state established |
|---|---|
| `2d5f217` | Verified KSDD2 extraction, audit, manifest, deterministic development split, geometry analysis, and leakage rules. |
| `39f3eff` | CUDA-verified full-image segmentation baseline scaffold. |
| `9750826` | Provisional class-weight validation pilot. |
| `1548203` | Numerically auditable mixed-precision step accounting. |
| `ff2c8c8` | Frozen real-only baseline protocol. |
| `ec68d29` | Baseline numerical failure diagnostics. |
| `ed89a0f` | Stabilized BF16 baseline configuration. |
| `5c77b9a` | Successful stabilized real-only baseline frozen as the reference. |
| `e5f204e` | F1 deterministic training-only GAN input pipeline. |
| `9156bb4` | F1.1 native-width bias removed. |
| `9fd3d52` | F1.2 source-border censoring geometry and manifest content hashing. |
| `dbf5903` | F1.3 deterministic placement compatibility index and sampling audit. |
| `4eea741` | F1.4 symmetric native-edge placement and feasible-scale selection. |
| `aeaeea5` | G1.1 mask-conditioned residual generator and discriminator architecture. |
| `96ab048` | G1.2 localized adversarial and regularization losses. |
| `7872293` | G1.3 deterministic GAN training-pair bridge. |
| `00bb747` | G1.3a/G1.3b aligned real/fake discriminator validity semantics. |
| `991ddfb` | G1.4 auditable one-step GAN trainer and calibrated mechanics. |
| `5ed2f65` | G1.5 gated GAN smoke runner. |
| `babe4b2` | G1.5a identity-initialized, range-aware residual output. |
| `006156e` | G1.6 single discriminator-update ablation and audit. |
| `bdb9085` | Corrected RGB-vector gradient gate outcome and final D-clip-10 selection. |
| `1b69785` | G2.1 deterministic 2,000-update sustained GAN training. |
| `a41be83` | G2.2 equal-budget downstream detector utility experiment and terminal stop. |

### Completed phase summary

- **Dataset phases A-C:** safe extraction, audited official manifest,
  official-test-preserving development split, duplicate exclusion, native geometry
  analysis, valid-region padding, and a CUDA detector scaffold.
- **Detector phases D/E:** class-weight and numerical-stability work, a preserved
  failed FP16 run, then the successful frozen BF16 real-only reference.
- **F1-F1.4:** deterministic development-training-only template/background
  construction, native-width eligibility correction, explicit border-contact
  semantics, canonical manifest hashing, compatibility indexing, and symmetric
  left/right feasible placement.
- **G1.1-G1.4:** residual GAN architecture, localized losses, deterministic
  training-pair bridge, aligned discriminator validity, and auditable update
  mechanics.
- **G1.5-G1.6:** gated smoke training, identity/range correction, exact canonical
  RGB-vector gradient coverage, one controlled discriminator ablation, and frozen
  selection of the D-clip-10 settings.
- **G2.1:** one sustained 2,000-joint-update GAN run from fresh identity
  initialization, with numbered checkpoints and no validation/test access.
- **G2.2:** paired checkpoint-1,000/checkpoint-1,500 synthetic materialization,
  equal-budget seed-42 detector pilot, three-seed confirmation of the only pilot
  winner, and terminal `stop_not_confirmed`.

## Dataset and split contract

KSDD2 remains a binary normal/defective dataset. Authoritative masks are never
modified or replaced by feathered/derived masks.

| Partition | Defective | Normal | Total | Permitted use |
|---|---:|---:|---:|---|
| Official train | 246 | 2,085 | 2,331 | Sole source of development train/validation. |
| Development train | 209 | 1,772 | 1,981 | Detector training; GAN templates/backgrounds; GAN internal train/monitor split. |
| Development validation | 37 | 313 | 350 | Detector validation and checkpoint/configuration evaluation only. |
| Official test | 110 | 894 | 1,004 | Untouched; not authorized after failed G2.2 confirmation. |

Binding split rules:

- Only the official training split was divided into development train and
  validation. Official test maps only to `development_split=test`.
- All loaders must consume `data/metadata/ksdd2_split_seed42.csv` or another
  explicitly validated manifest; directory globbing is not an acceptable source
  of sample identities.
- GAN templates and backgrounds require both `official_split=train` and
  `development_split=train`. Templates must be defective and backgrounds normal.
- The GAN's internal 90/10 train/monitor partition is grouped by source identity;
  template and background source IDs cannot cross its boundary. It is not detector
  validation.
- Synthetic detector rows may contain only development-training source provenance.
  G2.2 paired manifests record zero detector-validation and zero official-test
  source rows.
- Full detector images retain native aspect ratio on a `256 x 672` canvas.
  Images use symmetric reflection padding, labels use zero padding, and loss and
  metrics operate only on the binary native `valid_region`.

## Frozen detector decisions

### Rejected historical FP16 baseline

`configs/final_real_baseline.json` and `reports/final_real_baseline/` are diagnostic
only. The run used FP16, AdamW LR `0.001`, and no clipping; it encountered two
infinite-gradient events and failed during epoch-4 validation. Its epoch-2/3
checkpoints are finite but must not be resumed or used as a comparator.

### Accepted real-only reference

`configs/final_real_baseline_bf16.json` is the frozen detector reference:

- seed 42 GroupNorm U-Net;
- `256 x 672` native-geometry canvas;
- BF16 forward with float32 BCE and Dice;
- AdamW LR `3e-4`, weight decay `1e-4`;
- gradient clip `1.0`, no GradScaler;
- synchronized horizontal/vertical flips;
- deterministic weighted real sampling;
- valid-native pixels only.

The run completed `5,952/5,952` optimizer updates with no skips or numerical
anomalies. The accepted epoch-11 checkpoint is
`checkpoints/final_real_baseline_bf16_seed42/best.pt`, size `339,558,727` bytes,
whole-file SHA-256
`6a2127fad5fca66108de38226b050b9ef7d09025c4528a5bf48285ffaabfd277`.

At fixed threshold `0.5`, its validation Dice/IoU/precision/recall are
`0.7777003 / 0.6362599 / 0.8547696 / 0.7133793`; normal-image FPR is `0.0415335`.
The one permitted validation sweep selected `0.05` for global Dice (`0.7982509`).
Threshold `0.5` remains the fixed comparison threshold. Neither threshold may be
retuned on official test.

## Frozen F1 GAN-input decisions

The authoritative configuration is `configs/gan_inputs.json` with pipeline
version `f1_4_gan_inputs_v1`:

- patch geometry `256 x 512`, no resizing or distortion;
- native-valid background fraction minimum `0.71875` (`184/256`), accepting all
  1,772 legitimate normal development-training images in the audited metadata;
- symmetric reflection padding may remain visible context but is never native
  valid, label content, or defect support;
- context margin 24, overlap fraction 0.5, minimum 8 positive pixels, component
  coverage minimum 0.05;
- scale range `[0.9, 1.1]`, horizontal/vertical flip probability 0.5;
- feather radius 5 and non-border native margin 8;
- empirical border/non-border template distribution;
- feasible transformed states and compatible native windows are indexed before
  selection rather than found through repeated incompatible random placement.

The audited F1 manifest state has 209 defective development-training images, 235
connected components, 232 accepted components, 3 rejected components, 96
border-touching accepted templates, and zero partial template windows. Individual
connected components fit `256 x 512`; older complete image-level defect bounding
boxes could fail added-context fit without implying an oversized component.

Border censoring is explicit per top/bottom/left/right side. Flips swap the
appropriate sides, corner and multiple-side contacts remain multiple contacts,
and target placement must touch corresponding native-valid—not tensor—edges.
Non-border templates retain the eight-pixel margin. Support outside validity and
accidental contacts are hard failures.

The canonical hashes have distinct meanings:

- `source_manifest_sha256`: original audited source metadata identity;
- `split_sha256`: deterministic selected split identity;
- `gan_manifest_content_sha256`: canonical complete GAN manifest content,
  excluding only its own self-referential field.

The G2.1/G2.2 input identity is:

- GAN manifest content SHA-256:
  `bf3cdad05f402cfdd785e1c88254687db2d9d3ded30f6fef419e49d0cfc18c38`
- split SHA-256:
  `096ea5adee3aa08ac590decc7cc663dbb7d889764a0f645332c64d5ba15d2b9e`

## Frozen G1 architecture, loss, and training semantics

The generator in `configs/gan_architecture.json` is a four-channel
RGB-plus-mask, three-downsample-stage residual U-Net with base width 32, four
residual blocks, GroupNorm, a 12-pixel support dilation, and maximum directional
residual magnitude `0.25`. Its final residual head is zero-initialized, so fresh
initialization is exact identity. Range-aware directional caps keep output in
`[-1,1]`; `torch.where` preserves pixels outside support bit-exactly.

The discriminator is a spectral-normalized, mask-conditioned PatchGAN with base
width 32 and raw logits. Real and fake conditioning masks use the same canonical
threshold `0.5`. Discriminator views use the intersection of transformed real and
fake native validity and are exactly zero outside it.

G1.2 uses localized raw-logit hinge losses plus support-normalized change,
inner-boundary seam, masked total variation, and lazy R1. The generator does not
have a paired refined-image reconstruction target. Canonical adversarial-gradient
coverage is per pixel: all RGB gradient components must be finite and at least one
must be nonzero. Any non-finite canonical component is a separate hard failure;
invalid-region adversarial gradient must remain exactly zero.

## Checkpoint decisions before G2.2

| Artifact | Disposition | Reason |
|---|---|---|
| Historical FP16 detector checkpoints | Rejected/diagnostic only | Two infinite-gradient events and terminal epoch-4 validation failure. |
| Stabilized BF16 detector epoch 11 | Accepted and frozen | Completed 5,952 updates cleanly; lowest frozen validation loss. |
| Original pre-range-aware G1.5 smoke | Rejected/diagnostic only | Genuine clamp saturation `0.07349 > 0.05` after first joint update. |
| Completed G1.5b smoke baseline | Valid smoke reference, not final GAN | Passed 200 updates; used only as ablation/replay reference. |
| D-clip-10 smoke at initial step 197 stop | Historical interrupted evidence | Old scalar/channel interpretation reported one inactive scalar coverage element. |
| Corrected D-clip-10 step 200 | Selected configuration reference | RGB-vector pixel gate passed; lower D LR and higher D clip selected at `bdb9085`. |
| G2.1 numbered checkpoints | Frozen recovery/audit milestones | No detector-confidence or visual `best` selection was allowed. |
| G2.1 step 1,000 | Rejected by G2.2 pilot | High seed-42 Dice but pixel recall regressed `0.0710171`, exceeding `0.01`. |
| G2.1 step 1,500 | Pilot winner, then rejected terminally | Passed seed-42 pilot; failed three-seed mean recall confirmation. |
| G2.1 step 2,000 | Not evaluated for G2.2 utility | G2.2 was explicitly limited to steps 1,000 and 1,500. Do not substitute it. |

## G2.1 sustained GAN state

G2.1 is frozen at commit `1b69785`. It started from fresh seed-42 identity
initialization rather than loading smoke weights and replayed the selected G1.6
optimization exactly:

| Setting | Frozen value |
|---|---:|
| Joint generator updates | 2,000 |
| Joint discriminator updates | 2,000 |
| Discriminator warmup updates | 10 |
| Generator LR | `1e-4` |
| Discriminator LR | `2.5e-5` |
| Generator gradient clip | `5` |
| Discriminator gradient clip | `10` |
| Adam betas | `[0.0, 0.9]` |
| Weight decay | `0` |
| Precision | BF16 |
| Batch size | 2 |
| R1 | gamma 1, every 16 updates |
| Numbered checkpoint interval | 100 joint updates |

Configuration SHA-256 is
`e801f22ba6c869c57b5816855fd25eb1c84fea9cff20564380c29c601022ff7c`.
At step 200, generator/discriminator parameter and optimizer hashes exactly matched
the selected D-clip-10 smoke reference:

| State | SHA-256 |
|---|---|
| Generator parameters | `92ab734f5af71a39761b6132622c4dcb15454dc4735ab19c18b3774bc753f23f` |
| Discriminator parameters | `c11977b5be2cfdbd354d93a0caae1ca9eeb8c32cf0069c6f43d2b8c24b0cece5` |
| Generator optimizer | `6ca5d78398cd2cb122a5202f7f231ed4a477977066770f45258cfd85823895ef` |
| Discriminator optimizer | `add5d3206d99f77dbde78b5582a9ba3ed6640dbd445a9ccc990a719f06003fcd` |

Whole-file checkpoint hashes used or preserved by G2.2 are:

| Checkpoint | SHA-256 |
|---|---|
| `joint_1000.pt` | `801f60860f5f4d011c87f415090d96acbd93a13372feb8e0d8cac9881c50ae38` |
| `joint_1500.pt` | `5af1c6aafabcc0444117aa43209dcab168e57f4489259728e8f9066a4fdf1c81` |
| `joint_2000.pt` | `82cfe4a70470ac2c7ffb5d4ecaf4f7357d63b698ed43a9b83e3dc297a0766f5f` |
| `last.pt` | `82cfe4a70470ac2c7ffb5d4ecaf4f7357d63b698ed43a9b83e3dc297a0766f5f` |

G2.1 status is `PASS`: 2,000 generator updates, 2,010 discriminator optimizer
updates including warmup, no early stop, no output-range/locality/invalid-gradient/
non-finite-canonical-gradient violations, no validation/test rows, and no
materialized training-image dataset. Generator clipping occurred on `2.2%` of
joint updates; discriminator clipping remained high at `91.6915%` and is recorded
as a warning, not authorization for another sweep.

The run was resumed once from durable step 500 after a transient Windows atomic
file-replacement sharing violation occurred after step 566. Stale metrics were
trimmed and steps 501 onward replayed deterministically without a configuration
change. Seven fixed monitor sheets and 128-pair stratified monitors are safety
evidence only. No `best.pt` was created.

## Complete G2.2 protocol

G2.2 is frozen at commit `a41be83` and configured by
`configs/g2_2_detector_utility.json`.

### Paired synthetic materialization

- Compared only G2.1 `joint_1000.pt` and `joint_1500.pt`.
- Materialized 512 deterministic development-training-only samples.
- Each pair held template, background, coarse composite, masks, valid region,
  placement, transform, and seed fixed. Only frozen generator checkpoint identity
  and rendered image were allowed to differ.
- Generators ran in evaluation/inference mode with gradients disabled; no GAN
  optimizer was constructed and G2.2 records zero GAN optimizer updates.
- Pixels outside native validity were restored to the coarse background before
  detector materialization. Masks and valid regions were shared bit-for-bit.
- Checkpoint whole-file hashes were equal before and after materialization.
- Detector-validation source rows: 0. Official-test source rows: 0.
- Pairing report content SHA-256:
  `540a4637936c25ae9fd3678732bbc9d81e75f066e584e6d3ee078768f491ed33`.
- Checkpoint-1,000 synthetic manifest SHA-256:
  `72e7e2862c47351c7f236136e5d829a2b456a440cf962c299f57780aa6974de3`.
- Checkpoint-1,500 synthetic manifest SHA-256:
  `9eba21b4347dcdafafd9d0f90dd06b297cb58b2f7ee58f1887fed7a4cd62ca91`.

The full 512-row manifests, expanded detector schedules, generated images, and
detector checkpoints are intentionally Git-ignored local artifacts. Compact
pairing, arm, pilot, and confirmation reports are committed.

### Equal detector budget

The seed-42 pilot trained three fresh detector arms:

1. real-only control;
2. 75% real plus 25% checkpoint-1,000 synthetic;
3. 75% real plus 25% checkpoint-1,500 synthetic.

Every final arm used a GroupNorm U-Net, batch size 4, exactly 2,000 successful
AdamW updates, constant LR `3e-4`, weight decay `1e-4`, BF16, gradient clip 1,
threshold 0.5, identical loss/schedule/augmentation policy, and zero skipped
updates. Synthetic batches contained exactly three real samples and one synthetic
sample. The three real identities were shared across control and synthetic arms;
the control used a fourth deterministic real draw. Within each seed, model
initialization hashes matched exactly.

An initial seed-42 real-only execution was stopped at update 1,200 before any
checkpoint or validation result while progress buffering/runtime was diagnosed.
It was discarded. The reported control was restarted from fresh initialization
and completed all 2,000 updates; the discarded attempt is not an experimental arm.

The fixed pilot threshold was `0.5`. The meaningful-win gate required all of:

- global Dice gain at least `0.01`;
- global IoU gain at least `0.005`;
- normal-image FPR regression no more than `0.02`;
- precision regression no more than `0.01`;
- recall regression no more than `0.01`.

Defect-size cutoffs were derived only from development-training mask-pixel
tertiles: small `<=1261`, medium `1262..3671`, large `>3671`.

### Seed-42 pilot results

| Arm | Dice | IoU | Precision | Recall | Normal-image FPR | Decision |
|---|---:|---:|---:|---:|---:|---|
| Real-only | 0.406363 | 0.254991 | 0.318976 | 0.559698 | 0.926518 | Control |
| Checkpoint 1,000 synthetic | 0.604969 | 0.433660 | 0.793883 | 0.488681 | 0.210863 | Rejected: recall delta `-0.071017` |
| Checkpoint 1,500 synthetic | 0.572083 | 0.400642 | 0.564442 | 0.579934 | 0.738019 | Passed; sole confirmation candidate |

Seed-42 Dice by requested stratum:

| Arm | Border | Non-border | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|
| Real-only | 0.589084 | 0.816243 | 0.451993 | 0.785888 | 0.601589 |
| Checkpoint 1,000 | 0.536560 | 0.848352 | 0.640219 | 0.811591 | 0.535817 |
| Checkpoint 1,500 | 0.624323 | 0.825991 | 0.567929 | 0.794696 | 0.631669 |

Checkpoint 1,000 must not be selected by looking only at Dice, precision, FPR, or
small-defect performance. It failed the precommitted recall requirement.

### Three-seed confirmation

Only checkpoint 1,500 was compared with real-only over seeds 42, 43, and 44. Seed
42 reused the completed pilot arms; seeds 43 and 44 trained fresh paired arms with
the same exact 2,000-update budget.

| Seed | Dice gain | IoU gain | Precision delta | Recall delta | Normal-FPR delta | Per-seed gate |
|---:|---:|---:|---:|---:|---:|---|
| 42 | +0.165720 | +0.145651 | +0.245466 | +0.020236 | -0.188498 | Pass |
| 43 | +0.017088 | +0.018263 | +0.215954 | **-0.207008** | -0.293930 | Fail |
| 44 | +0.069960 | +0.066835 | +0.107543 | +0.015131 | -0.153355 | Pass |
| Mean | +0.084256 | +0.076917 | +0.189654 | **-0.057213** | -0.211928 | **Not confirmed** |

Dice improved in all three seeds, but confirmation required the multimetric mean
gate as well as positive Dice in at least two seeds. The mean recall regression
failed. The committed confirmation fields are:

```json
{
  "confirmed": false,
  "decision": "stop_not_confirmed",
  "positive_dice_seeds": 3,
  "official_test_access_count": 0,
  "gan_optimizer_updates": 0
}
```

This decision is terminal for the authorized experiment. It is not a request to
change the threshold, relax the gate, add seeds, test checkpoint 2,000, alter the
synthetic fraction, or train the GAN longer.

## Reproducibility and leakage safeguards

- Global Python/NumPy/Torch seeding and deterministic-algorithm controls are used.
- GAN online samples derive local seeds from base seed, split, epoch, manifest
  content hash, and sample index; worker scheduling does not change identities.
- GAN train/monitor source sets are grouped and disjoint.
- Placement compatibility is deterministic and index-based, including contacts,
  valid boundaries, support extents, scale, and non-border margin.
- Manifests use canonical JSON content hashes; meaningful content changes alter
  the hash and unchanged inputs reproduce it.
- G2.1 checkpoints contain models, optimizers, progress, data position, RNG state,
  configuration identity, and manifest/split identities. CPU tests verify exact
  resume behavior.
- G2.1 step-200 model and optimizer hashes match the selected smoke run exactly.
- G2.2 paired rows are checked field-for-field before rendering; only checkpoint
  identity/image output is excluded from the paired-source signature.
- G2.2 freezes GAN parameters, hashes checkpoint files before/after, and restores
  invalid-region pixels to the coarse background.
- G2.2 detector schedules are explicit, content-hashed, exact-budget, and use one
  successful update per attempted batch. A skip is fatal.
- Validation uses one fixed threshold and never alters training schedules.
- Official test construction exists only behind a separate confirmation gate and
  a one-evaluation sentinel. The failed confirmation makes that route forbidden.
- The complete test suite at `a41be83` passed: `298 passed`.

## Important paths

### Data and metadata

- `data/metadata/ksdd2_split_seed42.csv` — authoritative development split.
- `reports/data_audit/summary.json` — official dataset count/integrity audit.
- `reports/preprocessing/bbox_summary.json` — native geometry and mask statistics.
- `reports/gan_input_design/normal_valid_fraction_audit.json` — F1.1 width audit.
- `reports/gan_inputs/manifest.json` — local generated F1.4 GAN manifest; currently
  untracked, but its content identity is recorded in G2.1 checkpoints.

### Frozen configurations

- `configs/final_real_baseline_bf16.json`
- `configs/gan_inputs.json`
- `configs/gan_architecture.json`
- `configs/gan_losses.json`
- `configs/gan_training_pairs.json`
- `configs/gan_one_step.json`
- `configs/gan_smoke_dclip10.json`
- `configs/gan_training_2000.json`
- `configs/g2_2_detector_utility.json`

### Code entry points

- `scripts/train_final_real_baseline.py`
- `scripts/audit_gan_sampling.py`
- `scripts/audit_gan_training_pairs.py`
- `scripts/audit_gan_validity_alignment.py`
- `scripts/train_gan_smoke.py`
- `scripts/train_gan.py`
- `scripts/build_g2_2_synthetic_manifests.py`
- `scripts/train_g2_2_detector_utility.py`
- `src/defectgen/gan/` — F1/G1 data, geometry, hashing, compatibility, and pairs.
- `src/defectgen/models/gan.py` — frozen GAN architecture.
- `src/defectgen/training/gan_trainer.py` — audited GAN update mechanics.
- `src/defectgen/training/g2_2_utility.py` — G2.2 schedules, leakage checks, and gates.

### Evidence and checkpoints

- `reports/final_real_baseline_bf16_seed42/`
- `checkpoints/final_real_baseline_bf16_seed42/best.pt` — ignored binary.
- `reports/gan_training/smoke/` — completed G1.5b reference evidence.
- `reports/gan_training/smoke_dclip10/` — corrected selected smoke evidence.
- `reports/gan_training/g2_1_2000/summary.json`
- `reports/gan_training/g2_1_2000/step_0200_replay_verification.json`
- `reports/gan_training/g2_1_2000/execution_notes.json`
- `reports/gan_training/g2_1_2000/visual_review.json`
- `checkpoints/gan_training_2000/` — frozen ignored recovery binaries.
- `reports/g2_2/synthetic_manifests/pairing_report.json`
- `reports/g2_2/pilot_seed42/pilot_summary.json`
- `reports/g2_2/confirmation/confirmation_summary.json`
- `checkpoints/g2_2/` — ignored detector binaries; no checkpoint is authorized for
  official-test evaluation after `stop_not_confirmed`.

### Detailed design records

- `docs/design-decisions.md`
- `docs/stabilized-bf16-baseline.md`
- `docs/gan-input-pipeline.md`
- `docs/gan-architecture.md`
- `docs/gan-losses.md`
- `docs/gan-training-pairs.md`
- `docs/gan-training-mechanics.md`
- `docs/gan-smoke.md`
- `docs/gan-g1-6-selection.md`
- `docs/gan-sustained-training.md`
- `docs/g2-2-detector-utility.md`

## Commands and execution boundaries

The following are safe inspection/maintenance commands. They do not authorize a
new experiment:

```powershell
git show --stat a41be83
git status --short
Get-Content .\reports\g2_2\pilot_seed42\pilot_summary.json
Get-Content .\reports\g2_2\confirmation\confirmation_summary.json
Get-FileHash .\checkpoints\gan_training_2000\joint_1000.pt -Algorithm SHA256
Get-FileHash .\checkpoints\gan_training_2000\joint_1500.pt -Algorithm SHA256
.\.venv\Scripts\python.exe -m pytest
```

These historical entry points identify how existing artifacts were produced.
**Do not execute them without a new, explicit authorization that also addresses
the terminal G2.2 stop:**

```powershell
# G2.1 fresh/resume -- both forbidden; checkpoints are frozen.
.\.venv\Scripts\python.exe .\scripts\train_gan.py --config .\configs\gan_training_2000.json
.\.venv\Scripts\python.exe .\scripts\train_gan.py --config .\configs\gan_training_2000.json --resume

# G2.2 materialization/pilot/confirmation -- completed; do not rerun.
.\.venv\Scripts\python.exe .\scripts\build_g2_2_synthetic_manifests.py --config .\configs\g2_2_detector_utility.json
.\.venv\Scripts\python.exe .\scripts\train_g2_2_detector_utility.py --config .\configs\g2_2_detector_utility.json --mode pilot
.\.venv\Scripts\python.exe .\scripts\train_g2_2_detector_utility.py --config .\configs\g2_2_detector_utility.json --mode confirmation

# Explicitly forbidden: confirmation failed and official test is untouched.
.\.venv\Scripts\python.exe .\scripts\train_g2_2_detector_utility.py --config .\configs\g2_2_detector_utility.json --mode official-test
```

## Worktree and local-artifact note

At `a41be83`, committed G2.2 implementation and compact results were clean. The
worktree contained pre-existing untracked local F1 manifest/audit/visualization
artifacts under `reports/gan_inputs/`; they are not part of this handoff document's
commit and must not be accidentally staged, deleted, or rewritten.

Large data, generated images, full G2.2 paired manifests, expanded schedules, and
model checkpoints are intentionally ignored by Git. Their absence from `git status`
does not mean they are disposable. Preserve the local checkpoint/report directories
and verify hashes before any authorized read-only use.
