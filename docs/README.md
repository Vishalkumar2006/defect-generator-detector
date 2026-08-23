# Documentation index

This project produced a lot of technical documentation. You almost certainly do
not need to read all of it.

Every document is labelled with a status:

| Label | Meaning |
|---|---|
| **AUTHORITATIVE** | Currently binding. Read this if you read nothing else. |
| **FINAL** | A terminal experimental result. Not superseded, not reopened. |
| **DIAGNOSTIC** | Post-hoc analysis explaining *why* something happened. Trained nothing. |
| **HISTORICAL** | Accurate for the phase it describes; later work has moved past it. |
| **REFERENCE** | Standing design constraints and contracts. |

---

## Start here

| If you are… | Read, in order |
|---|---|
| **A recruiter or reviewer** (5 min) | The root [`README.md`](../README.md), then [§A and §F of `V1_FINAL_STATE.md`](V1_FINAL_STATE.md) |
| **An ML engineer evaluating the work** (30 min) | [`TECHNICAL_PROJECT_REPORT.md`](TECHNICAL_PROJECT_REPORT.md) — it consolidates everything below |
| **A researcher checking the method** | [`g2-3b-utility-protocol.md`](g2-3b-utility-protocol.md) → [`g2-3-diagnostic.md`](g2-3-diagnostic.md) → [`g2-2-detector-utility.md`](g2-2-detector-utility.md) |
| **A developer who wants to run it** | [`dataset-setup.md`](dataset-setup.md) → [`design-decisions.md`](design-decisions.md) → root README "Quick start" |
| **Someone resuming the project** | [`V1_FINAL_STATE.md`](V1_FINAL_STATE.md) §L "Resume boundary" — read it before touching anything |

---

## 1. Final V1 findings

**Read these first.** They carry the project's terminal conclusion.

| Document | Status | What it covers |
|---|---|---|
| [`TECHNICAL_PROJECT_REPORT.md`](TECHNICAL_PROJECT_REPORT.md) | **AUTHORITATIVE** | Comprehensive verification-grade technical report: architecture, losses, every experiment, consolidated quantitative results, challenges, and a claim-by-claim verification appendix. Written for an engineer or reviewer who needs the whole project without reading the source. Start here for a technical evaluation. |
| [`V1_FINAL_STATE.md`](V1_FINAL_STATE.md) | **AUTHORITATIVE** | The single source of truth for V1. Terminal status, complete commit ledger, frozen protocol, final results, scientific interpretation, limitations, reproducibility state, hashes, and the resume boundary. Supersedes `PROJECT_STATE.md` where they overlap. |
| [`g2-3b-results.md`](g2-3b-results.md) | **FINAL** | The narrative of the terminal result: the eight-criterion gate, what passed, what failed, per-seed and stratified breakdowns, and why the project stopped. |
| [`g2-3b-utility-protocol.md`](g2-3b-utility-protocol.md) | **FINAL** | The protocol as it was frozen *before* any G2.3B detector was trained — arms, seeds, budget, sampling law, threshold rule, and gate. Read this to confirm the gate was not chosen after seeing results. |

---

## 2. Downstream evaluation and diagnosis

How the utility question was actually asked, got a confounded answer, and was
then asked properly.

| Document | Status | What it covers |
|---|---|---|
| [`g2-2-detector-utility.md`](g2-2-detector-utility.md) | **HISTORICAL** | G2.2, the first downstream utility experiment. Terminal decision `stop_not_confirmed`. Its evidence was later shown to be confounded — see the diagnostic. |
| [`g2-3-diagnostic.md`](g2-3-diagnostic.md) | **DIAGNOSTIC** | G2.3A. Trained nothing; 6 validation-only forward passes over already-trained G2.2 checkpoints. Establishes the class-prevalence confound as fact and documents detector underconvergence, control instability, and calibration shift. This is the most instructive document in the repository. |
| [`fair-synthetic-comparison.md`](fair-synthetic-comparison.md) | **REFERENCE** | The standing rules for any real-versus-synthetic comparison: equal budgets, matched controls, precommitted gates. |

---

## 3. GAN architecture and training

The generator side, in the order it was built.

| Document | Status | Phase | What it covers |
|---|---|---|---|
| [`gan-input-pipeline.md`](gan-input-pipeline.md) | **REFERENCE** | F1–F1.4 | Training-only template/background construction, border-contact censoring, the deterministic placement compatibility index, and the provenance contract. The leakage rules live here. |
| [`gan-architecture.md`](gan-architecture.md) | **REFERENCE** | G1.1 | Mask-conditioned residual generator with exact outside-support copying; spectrally normalized mask-conditioned PatchGAN discriminator. |
| [`gan-losses.md`](gan-losses.md) | **REFERENCE** | G1.2 | Localized raw-logit hinge objectives plus change, seam, TV, and R1 primitives. |
| [`gan-training-pairs.md`](gan-training-pairs.md) | **REFERENCE** | G1.3–G1.3b | The deterministic data bridge and aligned real/fake discriminator views. |
| [`gan-training-mechanics.md`](gan-training-mechanics.md) | **HISTORICAL** | G1.4 | The bounded one-step mechanics harness: scale calibration, lazy-R1 convention, numerical guards. |
| [`gan-smoke.md`](gan-smoke.md) | **HISTORICAL** | G1.5 | Gated smoke training. Records a run that correctly *stopped itself* on a clamp-saturation gate. |
| [`gan-g1-6-ablation.md`](gan-g1-6-ablation.md) | **HISTORICAL** | G1.6 | The discriminator-clipping ablation. |
| [`gan-g1-6-selection.md`](gan-g1-6-selection.md) | **HISTORICAL** | G1.6 | Selection of the D-clip-10 configuration, and the explicit refusal to manufacture a "best" checkpoint from visual panels. |
| [`gan-sustained-training.md`](gan-sustained-training.md) | **HISTORICAL** | G2.1 | The one authorized 2,000-update sustained run, with numbered recovery checkpoints instead of confidence-selected state. |

---

## 4. Baseline detector

The real-only reference every synthetic arm is measured against.

| Document | Status | What it covers |
|---|---|---|
| [`stabilized-bf16-baseline.md`](stabilized-bf16-baseline.md) | **REFERENCE** | The **accepted** real-only baseline (E1.2). BF16, AdamW LR 3e-4, gradient clip 1.0, no GradScaler. Validation Dice `0.7777` at threshold 0.5. This is the comparator. |
| [`numerical-stability.md`](numerical-stability.md) | **REFERENCE** | Auditable mixed-precision step accounting and the class-weight pilot. |
| [`failed-fp16-baseline.md`](failed-fp16-baseline.md) | **HISTORICAL** | The FP16 baseline (E1) that failed on genuine infinite-gradient events during epoch-4 validation. Retained for diagnostics; **must not** be resumed or used as a comparator. |

---

## 5. Data and preprocessing

| Document | Status | What it covers |
|---|---|---|
| [`dataset-setup.md`](dataset-setup.md) | **AUTHORITATIVE** | How to obtain KSDD2 legitimately, where to put it, how to verify it, and the licence obligations you take on. Start here to run anything. |
| [`design-decisions.md`](design-decisions.md) | **REFERENCE** | Binding project constraints: dataset integrity, official-test sealing, native geometry, the 256 × 672 canvas, mask handling, and the GAN direction. Changing one of these requires a documented experiment. |

---

## 6. Superseded and operational records

Kept because they are genuine project history, not because they are current.

| Document | Status | What it covers |
|---|---|---|
| [`PROJECT_STATE.md`](PROJECT_STATE.md) | **HISTORICAL** | The accurate handoff state *through G2.2*, preserved verbatim. Still correct for everything it covers; superseded by `V1_FINAL_STATE.md` where they overlap. Useful for pre-G2.2 detail such as the G2.1 step-200 hashes. |
| [`G2_3B_ACTIVE_RUN_STATE.md`](G2_3B_ACTIVE_RUN_STATE.md) | **HISTORICAL** | Written mid-execution while G2.3B was still running. That run has since completed. Read it as an operations log — durability design, resume procedure, process-independence analysis — **not** as a description of a running experiment. |
| [`public-release-notes.md`](public-release-notes.md) | **REFERENCE** | Exactly what changed when this repository was prepared for public release, including the authorship audit and the history rewrite that removed AI-assistant attribution (§6), with its original→new SHA mapping, and the confirmation that no V1 scientific result was altered. Packaging, licensing, and documentation only. |

---

## 7. Future work

| Topic | Status |
|---|---|
| Version 2 | **Planned — not implemented.** No V2 code, config, experiment, result, or branch exists. The direction under consideration is recorded in [`V1_FINAL_STATE.md` §H](V1_FINAL_STATE.md), explicitly as a record of discussion rather than a plan of record. |

---

## Regenerating dataset-derived figures

Figures that embed KSDD2 pixels are not tracked in Git (see
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)). Once you have the
dataset locally, rebuild them:

| Artifact | Command |
|---|---|
| Dataset audit contact sheets | `scripts/visualize_ksdd2.py` |
| Preprocessing patch grids | `scripts/visualize_preprocessing.py` |
| GAN composite previews | `scripts/visualize_gan_inputs.py` |
| A single prediction overlay | `scripts/demo_segment_image.py` |

The V1 result figure in `assets/` carries no dataset pixels and *is* tracked;
regenerate it with `scripts/plot_v1_result_summary.py`.
