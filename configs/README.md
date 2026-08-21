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
