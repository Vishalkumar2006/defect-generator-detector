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

## Current scope

Day 1, Phases A and B provide repository scaffolding, safe extraction, an official-split-preserving audit manifest, a deterministic manifest-only development split, defect-geometry analysis, mask-aware rectangular patch extraction, reproducible visual checks, and focused tests. The binding design constraints are recorded in [`docs\design-decisions.md`](docs/design-decisions.md). PyTorch, inpainting, template extraction, model code, training, and synthetic data generation remain outside this phase.
