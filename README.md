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
