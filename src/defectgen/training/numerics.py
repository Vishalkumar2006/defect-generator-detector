"""Precision-aware, at-most-one-update numerical training primitives."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch


PRECISION_MODES = {"fp16", "bf16", "fp32"}


@dataclass
class NumericalCounters:
    attempted_batches: int = 0
    nonfinite_forward_loss: int = 0
    nonfinite_gradient: int = 0
    amp_overflow_scale_drop: int = 0
    optimizer_step_executed: int = 0
    optimizer_step_skipped: int = 0
    fp32_retry_attempted: int = 0
    fp32_retry_executed: int = 0


@dataclass
class BatchNumericalTelemetry:
    attempt: int
    precision_mode: str
    bce_loss: float
    dice_loss: float
    total_loss: float
    loss_dtype: str
    maximum_absolute_logit: float
    unscaled_gradient_norm: float | None
    scale_before: float | None
    scale_after: float | None
    optimizer_step_executed: bool
    optimizer_updates_this_batch: int
    skipped_reason: str | None
    nonfinite_forward_loss: bool
    nonfinite_gradient: bool
    amp_overflow_scale_drop: bool
    fp32_retry_attempted: bool = False
    fp32_retry_executed: bool = False

    @property
    def is_anomaly(self) -> bool:
        return bool(
            self.skipped_reason
            or self.nonfinite_forward_loss
            or self.nonfinite_gradient
            or self.amp_overflow_scale_drop
            or self.fp32_retry_attempted
            or self.fp32_retry_executed
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OptimizerStepCounter:
    """Count actual optimizer updates via the optimizer's post-step hook."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.total = 0
        self._handle = optimizer.register_step_post_hook(self._after_step)

    def _after_step(self, optimizer, args, kwargs) -> None:  # noqa: ARG002
        self.total += 1

    def close(self) -> None:
        self._handle.remove()


def _finite_gradients(parameters) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)


def _gradient_norm(parameters) -> float:
    gradients = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    squared = torch.stack([torch.sum(gradient * gradient) for gradient in gradients]).sum()
    return float(torch.sqrt(squared).item())


def precision_autocast(device_type: str, precision_mode: str):
    if precision_mode == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision_mode == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype)


class NumericalStepController:
    """Execute one attempted batch with an invariant of at most one update."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        precision_mode: str,
        scaler: Any | None = None,
        gradient_clip_max_norm: float | None = None,
        automatic_fp32_retry: bool = False,
    ) -> None:
        if precision_mode not in PRECISION_MODES:
            raise ValueError(f"precision_mode must be one of {sorted(PRECISION_MODES)}")
        if precision_mode == "fp16" and scaler is None:
            raise ValueError("fp16 requires a GradScaler-compatible scaler")
        if precision_mode != "fp16" and scaler is not None:
            raise ValueError("bf16 and fp32 must not use GradScaler")
        if gradient_clip_max_norm is not None and gradient_clip_max_norm <= 0:
            raise ValueError("gradient_clip_max_norm must be positive")
        if automatic_fp32_retry:
            raise ValueError("Automatic fp32 retry is intentionally unsupported; failed batches must remain skipped")
        self.optimizer = optimizer
        self.precision_mode = precision_mode
        self.scaler = scaler
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.automatic_fp32_retry = False
        self.counters = NumericalCounters()
        self.step_counter = OptimizerStepCounter(optimizer)

    def close(self) -> None:
        self.step_counter.close()

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "precision_mode": self.precision_mode,
            "gradient_clip_max_norm": self.gradient_clip_max_norm,
            "automatic_fp32_retry": self.automatic_fp32_retry,
            "counters": asdict(self.counters),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "precision_mode": self.precision_mode,
            "gradient_clip_max_norm": self.gradient_clip_max_norm,
            "automatic_fp32_retry": self.automatic_fp32_retry,
        }
        actual = {key: state[key] for key in expected}
        if actual != expected:
            raise ValueError("Numerical telemetry configuration does not match the checkpoint")
        self.counters = NumericalCounters(**state["counters"])

    def run_batch(self, model, inputs, targets, valid_region, criterion) -> BatchNumericalTelemetry:
        """Run one batch; losses are always evaluated in float32."""
        self.counters.attempted_batches += 1
        attempt = self.counters.attempted_batches
        self.optimizer.zero_grad(set_to_none=True)
        with precision_autocast(inputs.device.type, self.precision_mode):
            logits = model(inputs)
        # Explicit casts outside autocast make BCE, Dice, and total loss fp32.
        components = criterion.components(logits.float(), targets.float(), valid_region.float())
        maximum_logit = float(logits.detach().float().abs().max().item())
        values = {key: float(value.detach().item()) for key, value in components.items()}
        finite_loss = logits.shape == targets.shape and all(torch.isfinite(value) for value in components.values())
        if not finite_loss:
            self.counters.nonfinite_forward_loss += 1
            self.counters.optimizer_step_skipped += 1
            self.optimizer.zero_grad(set_to_none=True)
            return BatchNumericalTelemetry(
                attempt=attempt,
                precision_mode=self.precision_mode,
                bce_loss=values["bce"],
                dice_loss=values["dice"],
                total_loss=values["total"],
                loss_dtype=str(components["total"].dtype),
                maximum_absolute_logit=maximum_logit,
                unscaled_gradient_norm=None,
                scale_before=float(self.scaler.get_scale()) if self.scaler is not None else None,
                scale_after=float(self.scaler.get_scale()) if self.scaler is not None else None,
                optimizer_step_executed=False,
                optimizer_updates_this_batch=0,
                skipped_reason="nonfinite_forward_loss",
                nonfinite_forward_loss=True,
                nonfinite_gradient=False,
                amp_overflow_scale_drop=False,
            )
        return self._backward_and_step(model, components, maximum_logit, values, attempt)

    def _backward_and_step(self, model, components, maximum_logit, values, attempt):
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        scale_before = float(self.scaler.get_scale()) if self.scaler is not None else None
        before_updates = self.step_counter.total
        if self.precision_mode == "fp16":
            self.scaler.scale(components["total"]).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            components["total"].backward()

        gradient_norm = _gradient_norm(parameters)
        gradients_finite = _finite_gradients(parameters)
        if not gradients_finite:
            self.counters.nonfinite_gradient += 1
        elif self.gradient_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_max_norm)

        if self.precision_mode == "fp16":
            # Exactly one scaler.step call. The return value is intentionally ignored:
            # AdamW returns None even when this successfully invokes optimizer.step().
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif gradients_finite:
            self.optimizer.step()

        scale_after = float(self.scaler.get_scale()) if self.scaler is not None else None
        updates = self.step_counter.total - before_updates
        scale_drop = bool(scale_before is not None and scale_after < scale_before)
        if scale_drop:
            self.counters.amp_overflow_scale_drop += 1
        if updates > 1:
            raise RuntimeError("A batch executed more than one optimizer update")
        if updates == 1 and (not gradients_finite or scale_drop):
            raise RuntimeError("Optimizer updated despite a detected numerical failure")
        if updates == 0:
            self.counters.optimizer_step_skipped += 1
            if gradients_finite and not scale_drop:
                raise RuntimeError("Finite batch was skipped without an AMP scale drop")
            reason = "nonfinite_gradient" if not gradients_finite else "amp_overflow_scale_drop"
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.counters.optimizer_step_executed += 1
            reason = None

        return BatchNumericalTelemetry(
            attempt=attempt,
            precision_mode=self.precision_mode,
            bce_loss=values["bce"],
            dice_loss=values["dice"],
            total_loss=values["total"],
            loss_dtype=str(components["total"].dtype),
            maximum_absolute_logit=maximum_logit,
            unscaled_gradient_norm=gradient_norm,
            scale_before=scale_before,
            scale_after=scale_after,
            optimizer_step_executed=updates == 1,
            optimizer_updates_this_batch=updates,
            skipped_reason=reason,
            nonfinite_forward_loss=False,
            nonfinite_gradient=not gradients_finite,
            amp_overflow_scale_drop=scale_drop,
        )
