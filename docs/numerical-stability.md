# Numerical-stability and optimizer-step accounting

## D1 audit verdict

The D1 implementation did **not** use the return value of `GradScaler.step(optimizer)` to decide whether AdamW ran. It also did not execute two optimizer updates for one batch.

The historical D1 count named `failure_count` combined different events:

- `nonfinite_gradient_steps` counted batches whose gradients were non-finite after `scaler.unscale_(optimizer)`. Those batches cleared gradients, updated the scaler, and executed zero optimizer steps.
- `amp_training_fallback_batches` counted fp16 forwards with a non-finite loss that were automatically rerun in fp32 before any backward or optimizer step. If the retry was finite, that batch executed one update using the fp32 forward graph.
- `amp_validation_fallback_batches` counted analogous fp32 validation retries; no optimizer was involved.

Consequently, the large totals were not caused by interpreting AdamW's normal `None` return as a skipped update. They are a mixture of genuine non-finite unscaled gradients and automatic fp32 forward retries. D1 did not record scale-before/scale-after values, so its historical gradient events cannot be classified retrospectively as confirmed GradScaler overflow scale drops.

| D1 weight | Non-finite gradients | Training fp32 retries | Validation fp32 retries | Historical aggregate |
|---:|---:|---:|---:|---:|
| 1 | 4 | 1,212 | 133 | 1,349 |
| 5 | 3 | 1,078 | 191 | 1,272 |
| 10 | 3 | 1,667 | 386 | 2,056 |
| 20 | 3 | 1,277 | 225 | 1,505 |

Only 13 of the 6,182 aggregated events were direct non-finite-gradient detections. The other 6,169 entries were training or validation forward retries, so the old aggregate must not be described as an AMP-overflow count. The `pw=1` process also retained the earlier fallback graph-allocation behavior while later processes used an allocation cleanup; that explains its higher recorded peak memory but did not add a second optimizer update.

The stored D1 metrics remain accurate for the hybrid-precision runs that actually executed. They do not establish a clean `pos_weight`-only fp16 comparison because each candidate used a different number of fp32 retries. The `pos_weight=5` preference therefore remains provisional.

## Why `scaler.step()` returning `None` proves nothing

`GradScaler.step(optimizer)` returns the underlying optimizer's return value when it invokes that optimizer. AdamW's ordinary successful `optimizer.step()` returns `None`. Thus both a successful AdamW update and a GradScaler-skipped call may present `None` to the caller.

Actual optimizer execution is now detected with an optimizer post-step hook. Each attempted batch snapshots that counter before the step call and calculates its delta afterward. A delta of one means one actual update; zero means no update; a delta above one is an invariant violation and aborts the command.

## Counter definitions

- `nonfinite_forward_loss`: attempted batches whose float32 BCE, Dice, or total loss is non-finite. They execute no backward and no optimizer update.
- `nonfinite_gradient`: batches with any non-finite gradient after GradScaler unscaling, or after ordinary backward for bf16/fp32.
- `amp_overflow_scale_drop`: fp16 attempts for which `scale_after < scale_before`. This is recorded separately from gradient finiteness.
- `optimizer_step_executed`: actual optimizer post-step-hook executions.
- `optimizer_step_skipped`: attempted batches that execute zero optimizer updates.
- `fp32_retry_attempted`: automatic retry attempts. This must remain zero in the corrected engine.
- `fp32_retry_executed`: automatic fp32 retry forwards that ran. This must remain zero in the corrected engine.

An unchanged GradScaler scale is not treated as proof of success. Success requires an actual post-step-hook count. Likewise, a scale drop cannot coexist with an optimizer update for the same batch.

## Batch invariant and precision sequence

Every batch starts with `optimizer.zero_grad(set_to_none=True)`. Model forward uses the selected precision mode, while logits, targets, valid-region masks, BCE, Dice, and their reductions are explicitly float32. In fp16 mode the engine scales backward, unscales before gradient checks or clipping, calls `scaler.step()` exactly once, calls `scaler.update()` exactly once, and measures actual optimizer execution. Bf16 and fp32 do not use GradScaler.

A successful batch must execute exactly one optimizer update. A non-finite loss or gradient must execute zero. More than one update aborts immediately.

Automatic fp32 batch retry is deliberately disabled: silently retrying only unstable candidates changes their effective precision and compromises controlled comparison. A failed attempt remains a failed attempt and is reported for an explicit follow-up decision.

## Normalization and data boundaries

The detector uses development-training-only RGB channel standardization from `configs\baseline.json`. This detector normalization is unrelated to any later GAN input/output normalization. GAN normalization must be specified and audited separately for its architecture (for example, it must not be inferred merely from a future `tanh` output).

The numerical probe constructs only the development-training dataset. It never constructs validation or official-test datasets and saves no evaluation-ready model. The official test split remains untouched.

## Execution policy

Numerical and future training commands are run manually by the user. Automation in implementation/audit tasks is limited to static checks and tiny synthetic unit tests. Run only the requested initial profile first:

```powershell
.\.venv\Scripts\python.exe .\scripts\probe_numerical_stability.py --profile current_fp16 --steps 128
```

Inspect `reports\numerical_stability\current_fp16\summary.json`, `batch_telemetry.csv`, and `anomalies.json` before selecting another profile.

The later user-run stabilized BF16 profile used learning rate 0.0003, maximum
gradient norm 1.0, no GradScaler, and 512 attempts. It passed with 512 actual
updates and no numerical events. Its compact evidence is retained under
`reports/numerical_stability/bf16/`; the per-batch CSV remains ignored.
