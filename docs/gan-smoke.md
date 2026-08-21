# G1.5 gated GAN smoke training

G1.5 is a bounded stability and semantic-preservation smoke test. Its CUDA BF16
optimizer settings are provisional and do not establish final GAN hyperparameters.
It uses the G1.3 internal training split for updates, fixed internal-monitor samples
for evaluation, and the frozen real-only detector only as a detached metric.

## Gates and resume contract

The runner begins with ten discriminator-only updates. It evaluates the fixed
monitor batch at steps 0, 5, and 10 and proceeds only when the mean real logit is
greater than the mean fake logit. A non-positive margin extends warmup to at most
20 updates; it never triggers automatic learning-rate changes.

After accepted warmup, the same model and optimizer state enters a 20-joint-step
micro-smoke. Only a clean micro-smoke may continue to 200 total joint steps. Every
update retains G1.3b aligned discriminator views and G1.4 finite-gradient,
parameter-isolation, and exact-locality mechanics. Lazy R1 uses
`(global_step + 1) % 16 == 0` and a scheduled contribution of
`raw_r1 * gamma * interval / 2`. With the G1.4 median gradient ratio, gamma 1 gives
an estimated scheduled ratio of `0.05543 * 0.5 * 16 = 0.443`.

Atomic checkpoints contain both models and optimizers, optional scaler state,
configuration and hashes, manifest/split hashes, fixed monitor identities,
operation counters, data epoch and position, RNG state, and the last completed
operation. G1.5a checkpoints additionally carry the architecture and residual-
semantics versions. Loading rejects missing or mismatched semantics before model
state restoration, and the generator state contains a persistent semantics marker,
so the preserved G1.5 checkpoints cannot be resumed accidentally. Numbered and `last.pt` checkpoints live in the ignored
`checkpoints/gan_smoke/` directory. There is no "best GAN" checkpoint. CPU tests
verify uninterrupted and checkpoint/resumed updates produce identical parameter
hashes without repeating or skipping a step.

## Frozen detector evaluation

The finalized BF16 real-only detector checkpoint is loaded in evaluation mode with
all parameters frozen. GAN images are detached and converted from `[-1,1]` into
the detector's development-training normalization. No detector validation or
official-test image is constructed. The evaluator reports inside/outside
probability, contrast, Dice, IoU, and positive-sample rate for the initial
composite, refined composite, and transformed genuine image. These values never
enter a GAN loss.

## Preserved smoke outcome

The executed run accepted the discriminator warmup after ten updates: the fixed
monitor real-minus-fake margin was `0.00801` at step 0, `0.02934` at step 5, and
`0.02096` at step 10. It then stopped immediately after the first joint D/G update
because the generator clamp-saturation fraction was `0.07349`, above the mandatory
`0.05` limit. A read-only check against the pre-clamp candidate confirmed this was
genuine: 1,554 of 21,201 support-channel pixels (`0.07330`) exceeded `[-1,1]`,
while the corresponding composite support contained no already-saturated pixels.

The stopped update retained 100% canonical adversarial-gradient coverage, zero
invalid-region adversarial gradient, zero outside-support change, and support mean
change `0.13644`. The frozen detector did not trigger a retention warning before
the stop: refined inside-mask probability was `0.96229` versus composite `0.47398`;
refined Dice was `0.51817` versus composite `0.55277`. R1 was not reached because
only eleven total discriminator updates were executed, fewer than the first
scheduled event at update 16.

The 20-step gate therefore did not pass and the run was not continued. G1.5a
preserves this result as diagnostic evidence while replacing clamp monitoring in
future runs with a strict zero output-range-violation invariant. Future telemetry
also reports the old additive rule's would-have-clamped fraction, directional-cap
saturation, `tanh(raw)` saturation, and actual applied-residual magnitude. Do not
resume or tune around the preserved stop without a separately authorized phase.
Compact reports and fixed sheets are under `reports/gan_training/smoke/`; ignored
recovery checkpoints remain under `checkpoints/gan_smoke/`.
