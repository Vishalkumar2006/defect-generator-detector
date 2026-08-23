# Third-party notices

This file records third-party works this project depends on, and the terms
under which they are licensed. **Nothing in this file is covered by the
repository's own MIT `LICENSE`.**

---

## 1. Licensing separation — read this first

This repository contains two legally distinct categories of material:

| Category | What it is | Licence | Distributed here? |
|---|---|---|---|
| **This project's own work** | `src/`, `scripts/`, `tests/`, `configs/`, `docs/`, `assets/`, and the numeric report files under `reports/` | **MIT** (see [`LICENSE`](LICENSE)) | Yes |
| **KSDD2 itself** | The industrial surface images and their ground-truth masks | **CC BY-NC-SA 4.0**, held by its own authors | **No — you must obtain it yourself** |
| **KSDD2-derived figures** | 48 figures reproducing KSDD2 pixels, present in Git history only | **CC BY-NC-SA 4.0** — *not* MIT | Yes, as **Adapted Material** — see [§2 notice](#ksdd2-derived-figures-present-in-git-history--cc-by-nc-sa-40) |

**The MIT licence of this repository does not apply to KSDD2 or to any
KSDD2-derived figure, does not grant you any rights in either, and must not be
read as relicensing them.** Both remain under CC BY-NC-SA 4.0, and its terms are
set by the dataset's authors, not by this project.

---

## 2. Kolektor Surface-Defect Dataset 2 (KSDD2)

### Attribution

> Jakob Božič, Domen Tabernik, and Danijel Skočaj.
> "Mixed supervision for surface-defect detection: from weakly to fully
> supervised learning."
> *Computers in Industry* 129 (2021), article 103459.
> <https://doi.org/10.1016/j.compind.2021.103459>

- Dataset home: <https://www.vicos.si/resources/kolektorsdd2/>
- Publisher: Visual Cognitive Systems Laboratory, University of Ljubljana
- Licence: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### What CC BY-NC-SA 4.0 requires of you

If you obtain and use KSDD2, you are bound by its licence, not by this
repository's. In summary — the full legal text at the link above governs:

- **BY — Attribution.** You must give appropriate credit to the dataset
  authors, provide a link to the licence, and indicate if you made changes.
  You may do so in any reasonable manner, but not in any way that suggests the
  licensor endorses you or your use.
- **NC — NonCommercial.** You may **not** use the material for commercial
  purposes. This restriction covers KSDD2 itself, adaptations of it, and — by
  extension — any workflow of yours that redistributes KSDD2-derived material
  commercially. If your intended use is commercial, do not use this dataset
  without first obtaining separate permission from its authors.
- **SA — ShareAlike.** If you remix, transform, or build upon the material,
  you must distribute *your* contributions to that adapted material under the
  **same licence** (CC BY-NC-SA 4.0) as the original.

The ShareAlike obligation attaches to **adaptations of the licensed material**
— for example, a figure that reproduces KSDD2 pixels, a cropped patch, a
prediction overlay drawn on a KSDD2 image, or a synthetic image composited from
KSDD2 backgrounds. It does **not** attach to independently authored source code
that merely *reads* the dataset, which is why this project's code can be MIT
while the dataset stays CC BY-NC-SA 4.0.

### What this repository deliberately does NOT redistribute

To keep the separation clean, and to avoid redistributing third-party data
merely to make a repository self-contained, **none of the following is tracked
in Git**:

- the KSDD2 archive (`data/raw/KolektorSDD2.zip`);
- extracted or processed KSDD2 images and masks (`data/extracted/`,
  `data/processed/`);
- any report figure that embeds KSDD2 image or mask pixels — contact sheets,
  blinded comparison sheets, prediction overlays, patch example grids, and
  fixed-validation diagnostic panels;
- generated synthetic images (`data/synthetic/`), which are composited from
  KSDD2 backgrounds and KSDD2-derived defect templates and are therefore
  adaptations of the licensed material;
- model checkpoints (`checkpoints/`).

`.gitignore` enforces every one of these exclusions. All of these artifacts are
**regenerable locally** once you have obtained KSDD2 yourself — see
[`docs/dataset-setup.md`](docs/dataset-setup.md) for acquisition and
[`docs/README.md`](docs/README.md) for the scripts that rebuild each one.

> **"Not tracked" refers to the current tree.** 48 such figures were untracked
> rather than purged and remain reachable in Git history. They *are* distributed
> with this repository, under CC BY-NC-SA 4.0 — see the next section, which is
> their formal licence notice.

### KSDD2-derived figures present in Git history — CC BY-NC-SA 4.0

**This section is the formal licence notice for adapted material distributed by
this repository.**

The current tree (`HEAD`) tracks **no** KSDD2 pixels. However, **48 figure files
remain reachable in Git history**, in the nine commits that originally introduced
them (`2d5f217`, `39f3eff`, `9750826`, `5c77b9a`, `7872293`, `5ed2f65`,
`006156e`, `bdb9085`, `1b69785`). They were untracked at `HEAD` rather than
purged, because rewriting history would have invalidated the frozen V1 commit
ledger. They are therefore **distributed with this repository**, and the
following applies to them:

| Property | Statement |
|---|---|
| **What they are** | Contact sheets, patch-example grids, blinded comparison sheets, prediction overlays, and fixed-validation diagnostic panels that reproduce KSDD2 image and/or mask pixels |
| **Status under the licence** | **Adapted Material** derived from KSDD2 — cropped, composited, tiled, downscaled, colour-overlaid, and/or annotated |
| **Modified?** | **Yes.** Every one of these figures is a modification of the original KSDD2 material. None reproduces the dataset unaltered |
| **Licence** | **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** — the same licence as the source material, as ShareAlike requires |
| **Covered by this repo's MIT `LICENSE`?** | **No. Explicitly excluded.** The MIT licence does not apply to them, does not relicense them, and grants no rights in them |
| **Original creators** | Jakob Božič, Domen Tabernik, and Danijel Skočaj; Visual Cognitive Systems Laboratory, University of Ljubljana |
| **Source** | <https://www.vicos.si/resources/kolektorsdd2/> |
| **Licence text** | <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode> |

Redistribution of these adaptations is **permitted by the licence itself**:
CC BY-NC-SA 4.0 §2(a)(1)(B) grants the right to produce, reproduce, and Share
Adapted Material for NonCommercial purposes, subject to the §3 conditions —
attribution, an indication of modification, and ShareAlike. Those three
conditions are satisfied by this notice.

**If you reuse any of these figures**, you take on the same obligations: credit
the KSDD2 authors, link the licence, state that the material was modified, use it
non-commercially only, and license your own adaptation under CC BY-NC-SA 4.0.

### What this repository does contain

Only material that carries **no KSDD2 pixels**:

- **Numeric reports** under `reports/` — metrics, hashes, counts, schedules,
  audits, and decisions. These are measurements and statistics produced by this
  project's code, not reproductions of the dataset.
- **Metric plots** under `reports/` — loss curves, gradient-norm traces, logit
  traces, and threshold sweeps. These plot numbers, not images.
- **`data/metadata/ksdd2_split_seed42.csv`** and `reports/data_audit/manifest.csv`
  — this project's own deterministic development split and audit manifest. Their
  columns are KSDD2 **filenames, relative paths, a defect boolean, and SHA-256
  content hashes**. They contain no image data. A hash identifies a file but
  cannot reconstruct it, so these files do not reproduce the dataset or make it
  recoverable without a legitimate copy of KSDD2.

If you believe any tracked file nevertheless constitutes a redistribution of
KSDD2 beyond what CC BY-NC-SA 4.0 permits, please open an issue and it will be
removed.

---

## 3. Software dependencies

| Package | Licence | Use |
|---|---|---|
| [PyTorch](https://pytorch.org/) (`torch`, `torchvision`) | BSD-3-Clause | Models, training, CUDA/BF16 execution |
| [NumPy](https://numpy.org/) | BSD-3-Clause | Array and numeric primitives |
| [Pillow](https://python-pillow.org/) | MIT-CMU | Image I/O |
| [pandas](https://pandas.pydata.org/) | BSD-3-Clause | Manifest and report tabulation |
| [Matplotlib](https://matplotlib.org/) | PSF-based (Matplotlib licence) | Plots and figures |
| [pytest](https://pytest.org/) | MIT | Test suite |

Each dependency remains under its own licence. This project neither vendors nor
relicenses any of them.

---

## 4. Model architectures

The U-Net segmentation detector and the mask-conditioned residual generator with
a spectrally normalized PatchGAN discriminator are **original implementations in
this repository**, written from the published descriptions of these
architectures. No third-party model code was copied, and **no pretrained weights
from any external source were used** — every checkpoint in this project was
trained from random initialization on KSDD2 development-training data only.

Conceptual references (methods, not code):

- Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical
  Image Segmentation.* MICCAI 2015.
- Isola, Zhu, Zhou, Efros. *Image-to-Image Translation with Conditional
  Adversarial Networks* (PatchGAN discriminator). CVPR 2017.
- Miyato, Kataoka, Koyama, Yoshida. *Spectral Normalization for Generative
  Adversarial Networks.* ICLR 2018.
- Mescheder, Geiger, Nowozin. *Which Training Methods for GANs do actually
  Converge?* (R1 gradient penalty). ICML 2018.
