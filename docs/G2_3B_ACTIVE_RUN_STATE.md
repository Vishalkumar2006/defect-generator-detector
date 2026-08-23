> **COMPLETED — THIS RUN IS NO LONGER ACTIVE. Retained as a historical
> operations log.**
>
> The G2.3B run this document describes finished on 2026-08-23T05:33:07Z. All
> nine arms completed; the frozen gate was applied once; the outcome is
> `stop_not_confirmed_g2_3b`.
>
> Statements below such as "Completed arms: 2 of 9" and "Current arm: seed 45
> `gan_1500`" were accurate when written mid-execution and are now superseded.
> For current status see [`docs/V1_FINAL_STATE.md`](V1_FINAL_STATE.md); for the
> result see [`docs/g2-3b-results.md`](g2-3b-results.md) and
> `reports/g2_3b/confirmation_summary.json`.
>
> **Do not follow the resume procedure in section 6.** Training is complete and
> must not be rerun. The document is kept because its durability design, resume
> semantics, and process-independence analysis are genuine project history.

# G2.3B active-run handoff state

Factual handoff for a fresh conversation while the G2.3B experiment is executing.
This document is descriptive only. It changes no protocol, no configuration, no
training code, no result, and no official-split state.

**Written at:** 2026-08-23, during G2.3B execution.

---

## 1. Authoritative sources

Reread these before any substantial work. They outrank conversational memory.

| Path | Role |
|---|---|
| `docs/PROJECT_STATE.md` | Authoritative project handoff through G2.2 |
| `docs/g2-3-diagnostic.md` | G2.3A post-hoc diagnostic findings |
| `docs/g2-3b-utility-protocol.md` | Frozen G2.3B protocol, gate, threshold rule |
| `configs/g2_3b_utility_confirmation.json` | Frozen G2.3B configuration |
| `reports/g2_3b/plan/precommitted_plan.json` | Frozen plan and schedule hashes |
| `configs/g2_2_detector_utility.json` | Frozen G2.2 configuration and selection rules |
| `configs/final_real_baseline_bf16.json` | Accepted mature BF16 detector reference |

### Commit ledger

| Commit | Content |
|---|---|
| `a41be83` | G2.2 downstream detector utility, terminal `stop_not_confirmed` |
| `bc36ed4` | `docs/PROJECT_STATE.md` |
| `e1171a1` | G2.3A post-hoc validation-only diagnostic |
| `6e566a5` | G2.3B precommitted protocol and scaffolding |

---

## 2. Frozen experimental facts

- **G2.2 is permanently `stop_not_confirmed`.** It is never to be reinterpreted
  as a pass.
- **G2.3A** is post-hoc diagnostic evidence only. It confirmed class-prevalence
  confounding in G2.2, strong detector underconvergence and control instability,
  strong calibration/operating-point effects, and only a partial genuine
  representation deficit concentrated in one seed. It touched no official data.
- **G2.3B primary comparison:** `gan_1500` versus `prevalence_matched_real`.
- **G2.3B secondary comparison (ungated):** `standard_real` versus
  `prevalence_matched_real`.

### Arms

| Arm | Normal-real | Defective-real | Synthetic | Effective defective |
|---|---:|---:|---:|---:|
| `standard_real` | 50.0% | 50.0% | 0% | 0.500 |
| `prevalence_matched_real` | 37.5% | 62.5% | 0% | 0.625 |
| `gan_1500` | 37.5% | 37.5% | 25.0% | 0.625 |

### Immutable parameters

- Fresh seeds are exactly **45, 46, 47**.
- Each arm receives exactly **5,952 successful optimizer updates over 12 epochs**
  (496 updates/epoch, batch size 4, 1,984 sample slots/epoch), with **zero
  skipped updates**; a skip is fatal.
- Mature BF16 detector training semantics must not be changed.
- The frozen threshold-selection rule must not be changed: a fixed 99-point grid
  from 0.01 to 0.99 in steps of 0.01, maximum validation global Dice, tie-broken
  by mean defective-image Dice, then pixel precision, then the smallest
  threshold; development validation only.
- The **eight-criterion confirmation gate is precommitted** and must not be
  weakened, reordered, or modified after observing any result.
- **Checkpoint 1500 is the only permitted GAN checkpoint.** Checkpoints 1000 and
  2000 must never be substituted.
- **No GAN retraining and no synthetic regeneration are authorized.**
- **The official held-out split remains completely sealed.** It must not be
  constructed, inspected, counted, or evaluated during G2.3B.

---

## 3. Operational run state

Only operational progress is recorded here. Intermediate scientific performance
metrics are deliberately **not** preserved or interpreted, because intermediate
results must never influence the experiment.

- Total scope: **3 seeds x 3 arms = 9 arms.**
- **Completed arms: 2 of 9** — seed 45 `standard_real`, seed 45
  `prevalence_matched_real`.
- **Current arm:** seed 45 `gan_1500`.
- **Not yet started:** all of seed 46 and seed 47.
- Approximate wall-clock cost is about 90 minutes per arm.

Arm completion is determined solely by the presence of a verified
`reports/g2_3b/seed<SEED>/<ARM>.json` report.

---

## 4. Paths

### Results and progress

| Path | Content |
|---|---|
| `reports/g2_3b/plan/precommitted_plan.json` | Frozen plan, tracked |
| `reports/g2_3b/seed<SEED>/<ARM>.json` | Completed-arm report; presence means the arm is finished |
| `reports/g2_3b/seed<SEED>/<ARM>_epochs.json` | Per-epoch progress log, Git-ignored |
| `reports/g2_3b/seed<SEED>/seed_summary.json` | Written after all three arms of a seed finish |
| `reports/g2_3b/confirmation_summary.json` | Written by `--mode confirm`, does not exist yet |

### Durable checkpoints (Git-ignored)

| Path | Content |
|---|---|
| `checkpoints/g2_3b/seed<SEED>/<ARM>_last.pt` | Latest durable epoch; the resume point |
| `checkpoints/g2_3b/seed<SEED>/<ARM>_best.pt` | Best epoch by validation total loss |

### Execution driver and log

The driver script and the run log live in the session scratchpad directory:

```
<scratchpad>/run_g2_3b.sh          driver: seeds 45, 46, 47 in sequence
<tasks>/b7o312etr.output           combined stdout/stderr of the run
```

If the scratchpad is unavailable in a new session, the driver is trivially
reconstructed from the resume procedure in section 6.

### Frozen synthetic inputs (read-only, never regenerate)

| Path | Identity |
|---|---|
| `checkpoints/gan_training_2000/joint_1500.pt` | `5af1c6aafabcc0444117aa43209dcab168e57f4489259728e8f9066a4fdf1c81` |
| `reports/g2_2/synthetic_manifests/checkpoint_1500.json` | content `9eba21b4347dcdafafd9d0f90dd06b297cb58b2f7ee58f1887fed7a4cd62ca91` |
| `reports/g2_2/synthetic_manifests/pairing_report.json` | content `540a4637936c25ae9fd3678732bbc9d81e75f066e584e6d3ee078768f491ed33` |
| `data/synthetic/g2_2/checkpoint_1500/`, `data/synthetic/g2_2/common/` | 512 materialized synthetic samples |

---

## 5. Durability work present but uncommitted

The following durability hardening exists in the working tree and is **not yet
committed**. It was completed and verified before the run began. It must be
committed later as part of the completed-G2.3B commit, not as part of this
documentation-only handoff.

Uncommitted paths:

- `scripts/train_g2_3b_utility.py` (modified)
- `src/defectgen/training/g2_3b_protocol.py` (modified)
- `tests/test_g2_3b_durability.py` (new, untracked)

What the durability work provides:

- Atomic checkpoint writes via temporary file, `fsync`, and `os.replace`, so a
  failed or interrupted write can never replace a good checkpoint.
- Atomic JSON report writes with no temporary residue.
- **Automatic mandatory resume.** Durable state is either identity-compatible and
  resumed from, or it is a fatal error. There is no code path that silently
  discards completed epochs and restarts an arm.
- Run-identity verification across restarts covering experiment version, arm,
  seed, schedule hash, initialization hash, config hash, batch size, updates per
  epoch, maximum epochs, and total optimizer updates.
- Completed-arm reuse verification: a finished arm is reused only after its
  identity, exact budget, zero-skip record, and validation-only evaluation are
  all confirmed.
- Report overwrite protection: a completed arm report is never overwritten.
- Durable optimizer-counter checks tying recorded updates to completed epochs.
- RNG state persistence and restoration.
- Optimizer and LR-scheduler restoration checks, including a guard for the
  `ReduceLROnPlateau` behaviour where the learning rate lives in optimizer state
  rather than scheduler state.

Verification already performed before the run:

- A fresh rerun of a reduced-budget smoke experiment was bitwise identical to its
  reference.
- A hard `kill -9` mid-arm followed by resume produced a **bitwise identical**
  result to the uninterrupted run.
- Tampered durable state was refused with the mismatched field named.
- Re-running a completed seed reused all three arms after verification and
  retrained nothing.
- Re-running `--mode plan` reproduced `precommitted_plan.json` byte-identically,
  confirming the durability work altered no scientific decision.

---

## 6. Resume procedure

Resume is automatic and safe. Completed epochs and completed arms are never
rerun or overwritten. Re-running the driver, or a single seed, continues from the
latest durable state.

Whole run, seeds in sequence:

```powershell
foreach ($seed in 45,46,47) {
  .\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py `
      --mode train --seed $seed --resume
  if ($LASTEXITCODE -ne 0) { break }
}
```

Single seed:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode train --seed 45 --resume
```

Notes:

- The `--resume` flag is explicit intent only; resume happens whenever compatible
  durable state exists.
- If a resume attempt reports an identity mismatch, **stop and investigate**. Do
  not delete durable state to force progress.
- Training requires CUDA and BF16 support and refuses a CPU fallback.

---

## 7. Integrity requirements

Every arm must satisfy all of the following before its result may be used:

- exactly 5,952 executed optimizer updates;
- zero skipped optimizer updates;
- all three arms of a seed share one identical model-initialization hash;
- each arm's schedule content hash matches the frozen precommitted plan;
- evaluation split is development validation, with zero official held-out samples
  loaded;
- the frozen synthetic identity re-verifies at every run: GAN checkpoint file
  hash, manifest content hash, pairing report content hash, all 1,536 per-row
  image/mask/valid-region file hashes, all 512 rows carrying a positive
  in-valid-region defect pixel, and train-only provenance on every row and every
  template and background source.

**Intermediate results must never be used to change the experiment.** Do not
inspect partial metrics to make experimental decisions, and do not stop, alter,
extend, or reorder the protocol based on any individual seed or arm. Complete all
nine arms unless a genuine execution or integrity failure occurs.

---

## 8. Tests

The full suite passed before the run: **448 tests**.

- `tests/test_g2_3b_protocol.py` — 63 tests: composition, budgets, initialization
  equality, deterministic schedules, synthetic identity, validation-only
  threshold selection, official-split refusal, gate arithmetic.
- `tests/test_g2_3b_durability.py` — 43 tests: atomic writes, run identity,
  resume compatibility, completed-epoch and completed-arm protection, state
  round-trip, LR restoration, plan-unchanged assertions.

Run with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

---

## 9. Files that must never be staged

These are pre-existing, unrelated, and untracked. They are not part of any G2.3B
commit and must not be added, deleted, or rewritten:

- `reports/g2_2.zip`
- `reports/gan_inputs/manifest.json`
- `reports/gan_inputs/sampling_audit_f1_4.json`
- `reports/gan_inputs/sampling_audit_f1_4.md`
- `reports/gan_inputs/summary.json`
- `reports/gan_inputs/summary.md`
- `reports/gan_inputs/visualizations/`

Also never stage the duplicate KSDD2 files `train/10301 (copy).png` and
`train/10301_GT (copy).png`, and never delete them.

---

## 10. Authorized post-training sequence

Only after **all nine arms** have completed:

1. Perform the frozen threshold-selection procedure. Each arm already applies it
   at the end of its own training; `--mode confirm` consumes those results.
2. Calculate the precommitted metrics: pixel PR-AUC as threshold-independent
   evidence; and at the selected operating threshold the global Dice, global IoU,
   pixel precision, pixel recall, normal-image false-positive rate, image-level
   defect recall, border and non-border Dice and recall, and small, medium and
   large Dice and recall. Threshold 0.5 is reported as secondary continuity
   evidence only and is never a gate input.
3. Apply the frozen eight-criterion `gan_1500` versus `prevalence_matched_real`
   confirmation gate over seeds 45, 46 and 47:

   | # | Criterion | Threshold |
   |---|---|---|
   | 1 | mean global Dice gain | `>= +0.01` |
   | 2 | mean global IoU gain | `>= +0.005` |
   | 3 | mean pixel recall delta | `>= -0.01` |
   | 4 | mean pixel precision delta | `>= -0.01` |
   | 5 | mean normal-image FPR delta | `<= +0.02` |
   | 6 | seeds with positive Dice gain | `>= 2 of 3` |
   | 7 | mean pixel PR-AUC gain | `>= +0.01` |
   | 8 | seeds with positive PR-AUC gain | `>= 2 of 3` |

   Command:

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\train_g2_3b_utility.py --mode confirm
   ```

4. Report `standard_real` versus `prevalence_matched_real` as **secondary
   evidence only**. It is ungated and quantifies the pure defect-prevalence
   effect.
5. Generate the G2.3B machine-readable summary and documentation.
6. Run the full test suite.
7. Commit the completed G2.3B code, reports and documentation in **one isolated
   commit**, excluding the unrelated untracked artifacts listed in section 9.
8. **STOP** and report: PASS or FAIL of the primary confirmation; metrics for
   every seed and arm; C-versus-B deltas per seed and their means; all eight gate
   outcomes; the A-versus-B secondary comparison; selected thresholds and best
   epochs; optimizer, learning-rate, update-count and resume integrity; tests;
   commits; and Git status.

### Official held-out split

**Even if G2.3B passes, do not access the official held-out split.** A pass does
not authorize it. Any evaluation there requires a separate, explicit
authorization after review, and the current configuration forbids that access
unconditionally.

---

## 11. Process independence of the running experiment

Determined by read-only inspection only. No process was signalled, stopped,
restarted, or detached.

Observed process chain at the time of writing:

```
ai-assistant.exe (session)
  ...
  bash  (driver root)     -> parent shell had already exited
    bash  run_g2_3b.sh
      python train_g2_3b_utility.py --mode train --seed 45 --resume
        python (worker)
```

Findings:

- The shell that originally launched the driver chain **has already exited**, so
  the training tree currently has **no live ancestor process**. On Windows an
  orphaned process is not reparented, so OS-level parent-child inheritance alone
  would not terminate it when the session ends.
- Every inspected process reports Windows **job-object membership**, but so does
  an unrelated ad-hoc shell spawned purely for this check. Job membership is
  therefore ambient in this environment and does **not** establish that the run
  belongs to a session-scoped job with kill-on-close semantics. Whether the run
  shares a job with the session process could not be determined by read-only
  means.
- Independently of operating-system parentage, the run was started as a
  harness-tracked background task. Harness-level cleanup when a session ends is
  plausible and cannot be ruled out from inside the session.

**Conclusion.** Survival of the running process across a session close **cannot
be guaranteed**. Treat the run as potentially session-coupled.

**This does not put the experiment at risk.** The run is fully restartable and
was empirically verified to resume bitwise identically after a hard kill. If the
process does not survive, a new conversation simply re-runs the driver or the
per-seed command in section 6; execution continues from the latest durable epoch,
no completed epoch or arm is repeated, and no scientific result is affected.

**Recommended practice:** after starting a new conversation, first check whether
the run is still alive; if it is not, resume with section 6 before doing anything
else. Never delete durable state to force progress.
