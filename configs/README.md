# Experiment configurations

`final_real_baseline.json` is the frozen E1 real-only reference. Its training
entry point validates all critical values and rejects modifications rather than
silently creating a different experiment. Checkpoint resume additionally requires
the complete frozen configuration, including manifest and split names, to match.

Run it manually only when a full real-data experiment is intended:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_final_real_baseline.py --config .\configs\final_real_baseline.json
```

Use `--resume` only with that exact configuration and the matching ignored
`checkpoints/final_real_baseline/last.pt` state.
