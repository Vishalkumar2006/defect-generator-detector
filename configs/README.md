# Experiment configurations

`final_real_baseline.json` is retained unchanged as the historical failed FP16
experiment configuration. It failed numerical-stability requirements during epoch
4 validation and must not be resumed or used as the final reference. See
`docs/failed-fp16-baseline.md` for its disposition and preserved evidence.

`final_real_baseline_bf16.json` is the stabilized final reference configuration.
It uses the isolated identity `final_real_baseline_bf16_seed42`, BF16 autocast,
learning rate 0.0003, maximum gradient norm 1.0, and no GradScaler. Start it only
as a fresh seed-42 run:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_final_real_baseline.py --config .\configs\final_real_baseline_bf16.json
```

`gan_architecture.json` defines the architecture-only G1.1 mask-conditioned
residual GAN. It does not define losses, optimization, dataloading, checkpointing,
or training. Run its synthetic CPU audit with:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_gan_architecture.py --config .\configs\gan_architecture.json
```

`gan_losses.json` defines the independently testable G1.2 localized objectives.
Its unit aggregation coefficients are explicitly provisional; they are not final
training hyperparameters and must be revisited after a future training-smoke audit
measures component scales.

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_gan_losses.py --architecture-config .\configs\gan_architecture.json --loss-config .\configs\gan_losses.json
```

`gan_one_step.json` defines the explicitly provisional G1.4 mechanics audit. It
does not define final GAN hyperparameters or a training run. The command performs
eight no-update calibration batches followed by exactly one optimizer step per
model and writes no checkpoint:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_gan_one_step.py --config .\configs\gan_one_step.json
```

`gan_smoke.json` records the explicitly provisional G1.5 CUDA BF16 smoke settings,
including discriminator warmup, the 20-to-200 stage gate, frozen-detector warnings,
and hard numerical/semantic stops. The preserved run stopped correctly after its
first joint update; do not resume it without a new authorized phase. Checkpoint
binaries under `checkpoints/gan_smoke/` remain ignored by git.

`gan_smoke_dclip10.json` is the G1.6 configuration selected for controlled
continuation. It keeps the generator learning rate and clip at `1e-4` and `5`,
uses discriminator learning rate `2.5e-5` and clip `10`, and otherwise preserves
the G1.5b settings exactly. Both smoke runs remain historical evidence; the
selection does not create a visually chosen `best` checkpoint. See
`docs/gan-g1-6-selection.md`.

`gan_training_2000.json` defines the single G2.1 sustained run. It replays the
selected learning rates, clips, losses, precision, seed, data bridge, warmup, and
safety gates from fresh initialization for 2,000 joint updates. It adds only the
longer recovery, fixed-monitor, visual, stratified-audit, and rolling-report
schedules described in `docs/gan-sustained-training.md`.
