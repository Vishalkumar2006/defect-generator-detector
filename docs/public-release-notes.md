# Public-release notes — v1.0.0

This document records exactly what was changed when the repository was prepared
for public release, so that every difference from the frozen V1 state at commit
`a794625` is auditable.

**Status: REFERENCE.** It describes packaging and documentation only.

---

## The one binding claim

**No V1 scientific result was altered.** No metric, threshold, seed, hash, count,
schedule, gate value, decision, or recorded experimental outcome was changed,
recomputed, reinterpreted, or removed. The terminal decision remains:

```text
stop_not_confirmed_g2_3b
```

Verified at release time:

| Check | Result |
|---|---|
| `reports/g2_3b/confirmation_summary.json` decision | `stop_not_confirmed_g2_3b` — unchanged |
| G2.2 terminal decision | `stop_not_confirmed` — unchanged |
| `official_test_access_count` across all reports | `0` — unchanged |
| `reports/g2_2/official_test/` | does not exist — unchanged |
| Test suite | `448 passed` — matches the count recorded at freeze |
| V2 code, config, or experiment | none created |

---

## 1. Files added

| Path | Purpose |
|---|---|
| `LICENSE` | MIT, scoped explicitly to this repository's own work |
| `THIRD_PARTY_NOTICES.md` | KSDD2 attribution and CC BY-NC-SA 4.0 terms; dependency and architecture attribution |
| `pyproject.toml` | Optional editable install. Deliberately omits `torch` so it cannot overwrite a CUDA build |
| `docs/README.md` | Documentation navigation layer with per-document status labels |
| `docs/dataset-setup.md` | Legitimate KSDD2 acquisition, expected layout, verification, licence obligations |
| `docs/public-release-notes.md` | This file |
| `scripts/demo_segment_image.py` | Read-only single-image inference demo; trains nothing |
| `scripts/plot_v1_result_summary.py` | Regenerates the README figure from the authoritative summary JSON |
| `assets/v1_result_summary.png` | The headline result figure. Contains measured deltas only — no dataset pixels |

## 2. Files rewritten

| Path | Change |
|---|---|
| `README.md` | Replaced entirely. The previous README was a chronological append-only phase log that ended at G2.2 and never stated the final result. The new one leads with the negative outcome. |
| `.gitignore` | Extended with a public-release policy section (see §4). Existing rules were left untouched. |

## 3. Historical reports: path sanitization

Two tracked report files contained the author's local Windows absolute path,
including a personal username. Only the **path strings** were rewritten:

| File | Change | Occurrences |
|---|---|---:|
| `reports/g2_3/diagnostic/threshold_calibration.json` | `C:/Users/<user>/Projects/defect-generator-detector/` prefix removed, leaving repo-relative paths | 6 |
| `reports/environment/pytorch_environment.json` | Same prefix replaced with the literal `<repo-root>` | 2 |

**Nothing else in either file was touched.** No metric, hash, threshold, count,
or timestamp changed. Both files still parse as valid JSON, and
`tests/test_g2_3_diagnostic.py::test_diagnostic_threshold_metrics_reproduce_the_recorded_g2_2_values`
— which reads `threshold_calibration.json` — still passes.

These `path` fields are informational provenance labels. No recorded content hash
covers them, no runner compares them, and no assertion reads them.

## 4. Dataset-derived figures removed from tracking

**48 PNG files (53.9 MB) were untracked** with `git rm --cached`. They remain on
the author's disk; only their tracking was removed.

Every one of them reproduces KSDD2 image or mask pixels and is therefore an
*adaptation* of CC BY-NC-SA 4.0 material:

| Category | Count |
|---|---:|
| Dataset audit contact sheets (`reports/data_audit/`) | 5 |
| Preprocessing patch grids (`reports/preprocessing/`) | 3 |
| GAN composite contact sheets (`reports/gan_training/*/contact_sheets/`) | 28 |
| G1.6 blinded comparison sheets | 4 |
| GAN training-pair contact sheet | 1 |
| Detector prediction overlays and fixed-validation diagnostic panels | 7 |

`.gitignore` now blocks all of them by pattern, along with `data/raw/**`, any
image file under `data/`, and `reports/demo/`.

**21 PNG files were retained** — loss curves, gradient-norm traces, logit traces,
threshold sweeps, and comparison plots. These plot measured numbers and contain
no dataset pixels.

Each removed figure is regenerable locally once you hold KSDD2; the commands are
listed in `docs/README.md` under "Regenerating dataset-derived figures".

### Known limitation

These files remain reachable in **Git history**, in the nine commits that
originally introduced them (`2d5f217`, `39f3eff`, `9750826`, `5c77b9a`,
`7872293`, `5ed2f65`, `006156e`, `bdb9085`, `1b69785`). Purging them would
require rewriting every commit in the repository, which would invalidate the
commit ledger in `docs/V1_FINAL_STATE.md` §B, the `git.commit` provenance fields
recorded inside tracked report JSONs, and the frozen result commit `a794625`
itself. **The scientific record was given priority**, and history was left
intact.

This is a considered trade-off, not an oversight, and it was explicitly approved
before the repository was pushed.

Because those figures are therefore **distributed** with this repository, they
carry a formal licence notice rather than being treated as an accident.
CC BY-NC-SA 4.0 §2(a)(1)(B) permits producing, reproducing, and Sharing Adapted
Material for NonCommercial purposes, subject to the §3 conditions. All three are
satisfied:

| Condition | Where it is satisfied |
|---|---|
| Attribution — creator, notice, licence link | `LICENSE` scope section, `THIRD_PARTY_NOTICES.md` §2, README "Licence" |
| Indication that the material was modified | Stated explicitly: the figures are cropped, composited, tiled, colour-overlaid, and/or annotated adaptations |
| ShareAlike — adaptation under the same licence | The figures are declared **CC BY-NC-SA 4.0**, explicitly excluded from the MIT `LICENSE` |

Untracking them at `HEAD` is a conservative measure *beyond* what the licence
requires; the notice is what makes the historical copies compliant.

## 5. What is intentionally absent from Git

| Excluded | Approximate size | Why |
|---|---:|---|
| KSDD2 archive, extracted and processed images | ~2.0 GB | Third-party CC BY-NC-SA 4.0 data; not redistributed |
| `checkpoints/` — 111 `.pt` files | ~17 GB | Far beyond normal Git use; hashes and paths are documented instead |
| `data/synthetic/` — 2,048 files | ~319 MB | Adaptations of KSDD2 material |
| Expanded per-slot schedules, per-epoch progress logs, threshold curves | ~48 MB+ | Bulk reproducible artifacts; their content hashes are tracked |
| `reports/g2_2.zip`, `reports/gan_inputs/*` local artifacts | ~14 MB | Pre-existing untracked local artifacts; never part of committed history |

Identity of the absent binaries is preserved by recorded SHA-256 hashes, byte
sizes, and expected paths in `docs/V1_FINAL_STATE.md` §I and §J, so a holder of
the originals can verify them exactly.

---

## What was explicitly not done

- No experiment was re-run, resumed, extended, or re-tuned.
- No GAN or detector checkpoint was created, modified, or deleted.
- No gate criterion was relaxed, reordered, re-derived, or re-applied.
- The official held-out KSDD2 split was not constructed, inspected, counted, or
  evaluated.
- No Version 2 work was started.
- No experimental commit was removed, reordered, or squashed.
- No Git history was rewritten **to remove the KSDD2-derived figures**; they
  remain in the nine commits that introduced them (see §4). A separate,
  message-only rewrite did occur — see §6.
- The unrelated local artifacts listed in `V1_FINAL_STATE.md` §J were neither
  staged nor deleted.


---

## 6. History rewrite: removing AI-assistant attribution

After the initial public release, the commit history was rewritten **once**, so
that contributor attribution shows only the author's own Git identity and the
public record carries no vendor-specific AI product names.

### What the authorship audit found first

| Question | Finding |
|---|---|
| Distinct commit **authors** | **1** — `Vishalkumar2006 <vishku2004@gmail.com>` |
| Distinct **committers** | **1** — the same identity |
| An AI assistant as Git author or committer | **Never — not in a single commit** |
| AI attribution in tag objects | None |
| How it appeared | **Only** as a `Co-Authored-By:` message trailer, one in each of 11 commits |
| Other co-author trailers | None — no human attribution existed to preserve |

Because the attribution lived only in message trailers, no change to Git author
or committer metadata was needed or made.

### What the rewrite changed

A single combined `git filter-branch` pass over `bc36ed4..HEAD`:

1. **Message filter** — deleted the AI co-author trailers, and replaced the
   remaining vendor-specific prose with neutral wording (*AI assistant*, *AI
   coding assistant*).
2. **Tree filter** — in `G2_3B_ACTIVE_RUN_STATE.md` only, generalized the
   assistant's binary name in the recorded process chain to
   `ai-assistant.exe`. The operational facts are unchanged: the root of the
   observed chain was the interactive assistant session, and it had already
   exited, leaving the training tree with no live ancestor process.

Scope and preservation:

- **11 commits rewritten**, from the G2.3A diagnostic through the final release
  commit; **25 commits untouched**, keeping their original SHAs — every commit
  up to and including `bc36ed4`.
- Verified preserved for all 11: **author name/email, author date, committer
  name/email, committer date, subject line, and parent relationships.**
- Nothing was squashed, removed, or reordered, and no experimental commit was
  touched. Chronology is intact, so the visible fact that the G2.3B protocol was
  committed **before** its training and results remains verifiable from the log.
- The only file content altered anywhere was the one process-tree line above.
  The `reports/`, `configs/`, `src/`, `tests/`, and `data/` trees are
  **byte-identical** to their pre-rewrite state, verified by Git tree hash.

### SHA reference repair

The rewrite changed 11 SHAs, so 20 documentation references were updated in a
follow-up commit — a commit cannot contain its own SHA:

| File | References repaired |
|---|---:|
| `V1_FINAL_STATE.md` | 10 |
| `PROJECT_STATE.md` | 3 |
| `README.md` | 2 |
| `G2_3B_ACTIVE_RUN_STATE.md` | 2 |
| `public-release-notes.md` | 2 |
| `g2-3b-results.md` | 1 |

The one `git.commit` field stored in a report
(`reports/final_real_baseline_bf16_seed42/`, recording `ed89a0f`) needed **no**
repair, because that commit predates the rewrite range. No test asserts a Git
commit ID.

### What was NOT touched

Only Git commit SHAs were substituted in documentation. Verified mechanically:

- **Zero** 64-character hashes appear in any changed line.
- All **3,460 distinct SHA-256 values** across the tracked tree are unchanged in
  both value and occurrence count — checkpoint, config, manifest, schedule,
  split, and report-content hashes alike.
- Test suite: **448 passed**, unchanged.
- Terminal decision: still `stop_not_confirmed_g2_3b`; official-test access
  still `0`; GAN optimizer updates still `0`.

### Backup

The complete pre-rewrite history is preserved and was verified by cloning it
before any rewrite began:

```
defect-generator-detector-backup-20260823/
├── defect-generator-detector-preRewrite.bundle   all refs, verified complete
├── sha-mapping-old-to-new.json                   machine-readable mapping
├── sha-mapping-old-to-new.tsv                    same, tabular
├── refs-snapshot.txt
└── commit-log-snapshot.txt
```

Plus in-repo backup refs `refs/backup/pre-coauthor-rewrite/{main,v1.0.0}`.

**Future commits do not add AI co-author trailers.**
