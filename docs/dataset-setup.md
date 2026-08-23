# Dataset setup — KSDD2

This project trains and evaluates on the **Kolektor Surface-Defect Dataset 2
(KSDD2)**. The dataset is **not** distributed with this repository and is not
covered by its MIT licence. You must obtain it yourself from the official
source, under its own terms.

Read [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before you start.

---

## 1. Licence — your obligations

KSDD2 is released by the Visual Cognitive Systems Laboratory, University of
Ljubljana, under
[**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/):

- **Attribution** — credit the authors and link the licence when you use or
  publish anything derived from it.
- **NonCommercial** — commercial use is not permitted under this licence.
- **ShareAlike** — adaptations (composites, crops, overlays, synthetic images
  built from it) must be shared under CC BY-NC-SA 4.0.

By downloading KSDD2 you accept those terms. This repository grants you no
rights in the dataset.

Cite the dataset paper in any published work:

> Jakob Božič, Domen Tabernik, and Danijel Skočaj. "Mixed supervision for
> surface-defect detection: from weakly to fully supervised learning."
> *Computers in Industry* 129 (2021), article 103459.
> <https://doi.org/10.1016/j.compind.2021.103459>

---

## 2. Obtain the archive

1. Go to the official dataset page: <https://www.vicos.si/resources/kolektorsdd2/>
2. Download the full dataset archive (`KolektorSDD2.zip`) from that page.
3. Do **not** re-host, mirror, or commit the archive.

Place the **unmodified** archive at the repository-relative path:

```
data/raw/KolektorSDD2.zip
```

`.gitignore` already blocks `data/raw/**`, so it cannot be staged by accident.

---

## 3. Extract

Extraction is performed by a script, not by hand, so that integrity is checked:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe .\scripts\extract_ksdd2.py
```

```bash
# Linux / macOS
./.venv/bin/python ./scripts/extract_ksdd2.py
```

The extractor is idempotent only for a **verified complete** extraction; it
never overwrites an existing unverified destination.

---

## 4. Expected directory structure

After extraction the tree must look like this:

```
data/
├── raw/
│   └── KolektorSDD2.zip              # you provide this; never committed
├── extracted/
│   └── KolektorSDD2/
│       ├── train/                    # 2,331 images + matching *_GT.png masks
│       │   ├── 10000.png
│       │   ├── 10000_GT.png
│       │   └── ...
│       └── test/                     # 1,004 images + matching *_GT.png masks
│           ├── 20000.png
│           ├── 20000_GT.png
│           └── ...
├── processed/                        # derived artifacts; never committed
├── synthetic/                        # generated composites; never committed
└── metadata/
    └── ksdd2_split_seed42.csv        # tracked: this project's own split
```

Only `data/metadata/ksdd2_split_seed42.csv` and the `.gitkeep` placeholders are
tracked in Git. Everything else under `data/` is local-only.

---

## 5. Verify

Run the audit. It checks counts, dimensions, image/mask pairing, and content
hashes against the recorded expectations:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_ksdd2.py
```

The audit writes `reports/data_audit/summary.json`. A correct extraction
produces `"status": "PASS"` and exactly these counts:

| Official split | Defective | Normal | Total |
|---|---:|---:|---:|
| `train` | 246 | 2,085 | **2,331** |
| `test` | 110 | 894 | **1,004** |
| **Total** | **356** | **2,979** | **3,335** |

Total physical masks: **3,335**. Images are tall, narrow strips of roughly
230 × 630 px; the maximum native size across the dataset is 241 × 665, which is
what fixes this project's 256 × 672 model canvas.

### Known nonconforming files

The archive ships two exact duplicates:

```
train/10301 (copy).png
train/10301_GT (copy).png
```

They are **excluded from every manifest, split, and loader**, and they must
**never be deleted or modified** — the audit expects them to exist. This is a
frozen project constraint; see [`design-decisions.md`](design-decisions.md).

---

## 6. Build the development split

The official `test` split is sealed and is never used for any development
decision. Only the official `train` split is subdivided:

```powershell
.\.venv\Scripts\python.exe .\scripts\create_development_split.py
```

This writes `data/metadata/ksdd2_split_seed42.csv`. The file is already tracked,
so a correct regeneration reproduces it **byte-identically**:

| Property | Expected value |
|---|---|
| SHA-256 | `024495c9673a7096c79f342cce58ad6dd5e7434951b9b61053e926ab7c8c9f07` |
| Size | 616,772 bytes |
| Rows | 3,335 |

Resulting partitions:

| `official_split` | `development_split` | Rows | Role |
|---|---|---:|---|
| `train` | `train` | 1,981 | All model training, template extraction, GAN sources |
| `train` | `validation` | 350 | Every metric reported in V1 |
| `test` | `test` | 1,004 | **Official held-out — never loaded in V1** |

Verify the hash after regenerating:

```powershell
Get-FileHash .\data\metadata\ksdd2_split_seed42.csv -Algorithm SHA256
```

```bash
sha256sum data/metadata/ksdd2_split_seed42.csv
```

If the hash differs, stop — your extraction does not match the audited dataset,
and no downstream result will be comparable to the recorded ones.

---

## 7. Regenerating the dataset-derived figures

Figures that embed KSDD2 pixels are intentionally **not** in Git (contact
sheets, prediction overlays, patch example grids, blinded comparison sheets).
Once you have the dataset locally, rebuild them:

```powershell
.\.venv\Scripts\python.exe .\scripts\visualize_ksdd2.py          # audit contact sheets
.\.venv\Scripts\python.exe .\scripts\visualize_preprocessing.py  # patch example grids
.\.venv\Scripts\python.exe .\scripts\visualize_gan_inputs.py     # composite previews
```

They are written under `reports/` and are ignored by Git. If you publish any of
them, remember the ShareAlike obligation: they are adaptations of KSDD2 and must
carry CC BY-NC-SA 4.0 attribution.

---

## 8. Loader contract

Two rules are enforced in code and covered by tests:

1. **Manifest-only sample discovery.** Every loader consumes the validated
   split manifest. Unrestricted directory globbing is not an acceptable source
   of sample identities.
2. **Split gating.** In the G2.3A and G2.3B code paths,
   `assert_permitted_split` admits development `train` and `validation` only and
   raises `OfficialTestAccessError` for anything else;
   `scripts/train_g2_3b_utility.py` offers `plan`, `train`, and `confirm` and has
   no official-test mode at all.

   One older route does exist and is worth stating plainly:
   `scripts/train_g2_2_detector_utility.py --mode official-test`. It is
   **hard-gated** — it refuses to run unless the G2.2 three-seed confirmation
   recorded `authorize_single_official_test`. G2.2 recorded
   `stop_not_confirmed`, so the gate is shut and the route is inert. It was
   never taken: `official_test_access_count` is `0` in every recorded report,
   and `reports/g2_2/official_test/` does not exist. Do not run it.
