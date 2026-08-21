# Defect Generator & Detector

This project will explore data-driven surface-defect generation and detection. Day 1 is deliberately limited to trustworthy KSDD2 ingestion: safe extraction, deterministic indexing, auditing, and visual verification. No model has been trained yet.

## Dataset

The project uses the [Kolektor Surface-Defect Dataset 2 (KSDD2)](https://www.vicos.si/resources/kolektorsdd2/), released under the [CC BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/). The dataset contains an official train/test split and is initially treated as binary (normal or defective); this project does not invent defect-type labels.

Please cite the dataset's paper:

> Jakob Božič, Domen Tabernik, and Danijel Skočaj. “Mixed supervision for surface-defect detection: from weakly to fully supervised learning.” *Computers in Industry* 129 (2021), article 103459. [https://doi.org/10.1016/j.compind.2021.103459](https://doi.org/10.1016/j.compind.2021.103459).

The original archive, extracted images, and any future generated datasets are intentionally excluded from Git and will not be committed to this repository.

## Windows PowerShell setup

Place the unmodified archive at `data\raw\KolektorSDD2.zip`, then run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\scripts\extract_ksdd2.py
.\.venv\Scripts\python.exe .\scripts\audit_ksdd2.py
.\.venv\Scripts\python.exe .\scripts\visualize_ksdd2.py
.\.venv\Scripts\python.exe .\scripts\create_development_split.py
.\.venv\Scripts\python.exe .\scripts\analyze_defect_geometry.py
.\.venv\Scripts\python.exe .\scripts\visualize_preprocessing.py
.\.venv\Scripts\python.exe -m pytest
```

Calling the virtual-environment Python directly avoids PowerShell activation-policy issues. Extraction is idempotent only for a verified complete extraction and never overwrites an existing unverified destination. Audit artifacts are written to `reports\data_audit\`.

### CUDA PyTorch baseline environment

The verified Windows environment uses the stable CUDA 12.8 wheels selected for the installed NVIDIA driver. Install them explicitly so pip cannot silently choose a CPU-only build:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe .\scripts\verify_pytorch_environment.py
.\.venv\Scripts\python.exe .\scripts\analyze_full_image_data.py
.\.venv\Scripts\python.exe .\scripts\probe_gpu_memory.py
.\.venv\Scripts\python.exe .\scripts\train_baseline_smoke.py
```

The verification command exits with an error if CUDA is unavailable; it does not accept a silent CPU fallback. The smoke command is bounded to at most two epochs on a deterministic real-only subset and does not evaluate the official test set.

## Current scope

Day 1, Phases A–C provide repository scaffolding, safe extraction, an official-split-preserving audit manifest, a deterministic manifest-only development split, geometry analysis, mask-aware patches, and a CUDA-verified full-image real-only U-Net smoke pipeline. The binding design constraints are recorded in [`docs\design-decisions.md`](docs/design-decisions.md). Full baseline training, inpainting, template extraction, GAN training, and synthetic data generation remain outside this phase.

The provisional D1 class-weight pilot and its corrected numerical-step accounting are documented in [`docs\numerical-stability.md`](docs/numerical-stability.md). Numerical probes and future training commands are executed manually; implementation tasks do not automatically run them.

## Historical FP16 baseline (E1)

The experiment configured by `configs/final_real_baseline.json` failed during
epoch 4 validation after two genuine infinite-gradient events. Its finite epoch-2
and epoch-3 checkpoints and small reports are retained only for diagnostics. Do
not resume it or use it as the final real-only comparator. See
`docs/failed-fp16-baseline.md`.

The controlled future real-versus-synthetic protocol is documented in
`docs/fair-synthetic-comparison.md`.

## Stabilized BF16 final baseline (E1.2)

The replacement final real-only reference is frozen in
`configs/final_real_baseline_bf16.json`. It uses BF16, AdamW learning rate 0.0003,
maximum gradient norm 1.0, and no GradScaler under the isolated identity
`final_real_baseline_bf16_seed42`. Its implementation and evidence are described
in `docs/stabilized-bf16-baseline.md`.

The successful reference selected epoch 11 by validation loss and completed
5,952/5,952 optimizer updates with no numerical anomalies. Its fixed comparison
threshold is 0.5; its one-time validation Dice-optimal threshold is 0.05. Both
thresholds are frozen for future reporting and official-test data remains untouched.

## Training-only GAN inputs (F1)

`configs/gan_inputs.json` and `src/defectgen/gan/` define a deterministic online
256 x 512 template/compositing pipeline. It accepts development-training defects
and normal backgrounds only, hard-rejects validation or official-test records,
keeps masks binary, and does not materialize a generated dataset. This phase has no
GAN model or training implementation. The rules and provenance contract are in
`docs/gan-input-pipeline.md`.

F1.1 sets normal-background eligibility to the audited native-valid fraction
`0.71875` rather than `0.9`. This removes native-width selection bias while keeping
all reflected padding invalid for defect placement. The training-only distribution
is recorded in `reports/gan_input_design/normal_valid_fraction_audit.json`.

F1.2 preserves the censoring semantics of native-border defects: explicit contact
sides are transformed with flips and must match target native edges, while
non-border defects retain an eight-pixel margin. It also distinguishes source and
split identities from the canonical `gan_manifest_content_sha256` and adds
category-aware visual audits with placement accounting.

F1.3 replaces background-by-background placement retries with a deterministic
native-geometry compatibility index. Border sampling defaults explicitly to the
empirical template fraction, index exclusions are distinguished from actual
placement retries, and `scripts/audit_gan_sampling.py` provides a bounded
training-only performance and distribution audit without materializing images.

F1.4 centers reflection padding around narrow native windows and evaluates
placement against the inclusive native-valid bounds instead of tensor edges. It
indexes feasible flip and rounded-scale intervals before sampling, preserving
left/right and multi-side contacts without impossible-state retries. The audit now
reports expected versus observed target-side counts and horizontal-counterpart
pool symmetry. The failed F1.3 1,000-sample audit remains in
`reports/gan_inputs/sampling_audit.{json,md}` as diagnostic evidence; new defaults
write `sampling_audit_f1_4.{json,md}`.

G1.1 adds architecture definitions only: a mask-conditioned residual generator
with exact outside-support copying and a spectrally normalized mask-conditioned
PatchGAN discriminator. The synthetic architecture audit uses a 512 x 256 input,
loads no dataset rows, performs no training step, and writes its invariant report
under `reports/gan_architecture/`. Architecture details are documented in
`docs/gan-architecture.md`.

G1.2 adds localized, raw-logit hinge objectives and independent change, seam, TV,
and R1 primitives without adding a dataset or trainer. A shared canonical mask
policy prevents real/fake conditioning-format differences, while dilated mask
projection prevents unrelated PatchGAN background logits from dominating. The
synthetic audit and mathematical definitions are under `reports/gan_losses/` and
`docs/gan-losses.md`.

G1.3 adds the deterministic data bridge between the F1.4 online sampler and a
future GAN trainer. It creates synchronized real/fake mask-conditioned pairs,
keeps native-valid masks on both branches, and uses a grouped development-training
train/monitor split without accessing detector validation or official test data.
The contract and audit procedure are documented in `docs/gan-training-pairs.md`.

G1.3b propagates continuous real native-valid coverage with the exact transformed
defect grid, then constructs discriminator views on the intersection of real and
fake native validity. Both branches are exactly zero outside that intersection;
inside it their original pixels and fake-branch gradients are preserved. Strict
canonical-mask containment is separated from informational feather-and-halo
support containment, with the correction and audit policy documented in
`docs/gan-training-pairs.md`.

G1.4 adds a bounded, precision-aware GAN mechanics harness. It calibrates raw loss
and unit-gradient scales over eight deterministic internal-training batches, then
executes exactly one isolated discriminator update and one isolated generator
update using the G1.3b aligned views. Its provisional settings, lazy-R1 convention,
numerical guards, and audit command are documented in
`docs/gan-training-mechanics.md`.
