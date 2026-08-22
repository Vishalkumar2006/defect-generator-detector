# G2.1 sustained GAN training

G2.1 is one deterministic 2,000-joint-update continuation of the selected G1.6
configuration. It starts from fresh seed-42 identity initialization; it does not
load the selected smoke model as a starting point. The generator/discriminator
learning rates are `1e-4`/`2.5e-5`, their gradient clips are `5`/`10`, and all
other architecture, loss, optimizer, data, precision, warmup, and safety settings
replay `configs/gan_smoke_dclip10.json` exactly.

The run writes reports to `reports/gan_training/g2_1_2000` and recovery
checkpoints to `checkpoints/gan_training_2000`. Numbered checkpoints are written
every 100 joint updates. They are recovery and audit milestones, not candidates
chosen by detector confidence. No `best.pt` link is created.

The fixed seven-category panel is evaluated every 100 joint updates, in addition
to the existing warmup gates. Visual sheets are restricted to steps 0, 100, 250,
500, 1,000, 1,500, and 2,000. A deterministic, unique 128-pair internal-monitor
panel is evaluated at steps 0, 500, 1,000, 1,500, and 2,000. Both panels come only
from the grouped development-training monitor partition; validation and official
test datasets are never constructed.

Every update retains the existing finite loss/parameter/optimizer, logit,
output-range, exact locality, invalid-gradient, and canonical RGB-vector gradient
gates. Reports include 100-update clipping and margin windows, residual mass in
the loss-defined inner support boundary, boundary enrichment, saturation,
locality, and detector-statistic L2/mean-absolute distance from transformed
genuine real images.

At joint step 200 the runner hashes generator and discriminator parameters and
both optimizer states. All four hashes must exactly equal the independently
completed selected D-clip-10 smoke checkpoint or the sustained run stops. This
verifies fresh initialization plus the first 200 updates are an exact replay,
without promoting the smoke checkpoint into a training initializer.

Fresh run:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_gan.py --config .\configs\gan_training_2000.json
```

Exact resume from the sustained run's own `last.pt`:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_gan.py --config .\configs\gan_training_2000.json --resume
```

