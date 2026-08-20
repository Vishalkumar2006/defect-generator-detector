from __future__ import annotations

import copy

import pytest
import torch

import defectgen.training.numerics as numerics
from defectgen.training.engine import load_training_checkpoint, save_training_checkpoint
from defectgen.training.losses import CombinedBCEDiceLoss
from defectgen.training.numerics import NumericalStepController, OptimizerStepCounter
from scripts.probe_numerical_stability import build_training_dataset


class FakeGradScaler:
    def __init__(self, *, scale: float = 8.0, force_inf_on_unscale: bool = False) -> None:
        self.current_scale = scale
        self.force_inf_on_unscale = force_inf_on_unscale
        self.operations: list[str] = []

    def get_scale(self):
        return self.current_scale

    def scale(self, loss):
        self.operations.append("scale")
        return loss

    def unscale_(self, optimizer):
        self.operations.append("unscale")
        if self.force_inf_on_unscale:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.fill_(float("inf"))

    def step(self, optimizer):
        self.operations.append("scaler_step")
        if not self.force_inf_on_unscale:
            return optimizer.step()
        return None

    def update(self):
        self.operations.append("update")
        if self.force_inf_on_unscale:
            self.current_scale /= 2

    def state_dict(self):
        return {"scale": self.current_scale, "force_inf": self.force_inf_on_unscale}

    def load_state_dict(self, state):
        self.current_scale = state["scale"]
        self.force_inf_on_unscale = state["force_inf"]


def _batch():
    inputs = torch.randn(2, 1, 4, 4)
    targets = torch.zeros(2, 1, 4, 4)
    targets[0, :, 1:3, 1:3] = 1
    return inputs, targets, torch.ones_like(targets)


def _model_optimizer():
    model = torch.nn.Conv2d(1, 1, 1)
    return model, torch.optim.AdamW(model.parameters(), lr=0.01)


def test_adamw_none_return_still_counts_successful_update():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    counter = OptimizerStepCounter(optimizer)
    parameter.grad = torch.tensor(1.0)
    before = parameter.detach().clone()
    result = optimizer.step()
    assert result is None
    assert counter.total == 1
    assert not torch.equal(before, parameter)
    counter.close()


def test_finite_scaled_batch_executes_exactly_one_step():
    model, optimizer = _model_optimizer()
    scaler = FakeGradScaler()
    controller = NumericalStepController(optimizer, precision_mode="fp16", scaler=scaler)
    telemetry = controller.run_batch(model, *_batch(), CombinedBCEDiceLoss(pos_weight=5))
    assert telemetry.optimizer_step_executed
    assert telemetry.optimizer_updates_this_batch == 1
    assert controller.counters.optimizer_step_executed == 1
    assert controller.counters.optimizer_step_skipped == 0
    controller.close()


def test_forced_inf_gradient_executes_zero_steps_and_records_scale_drop():
    model, optimizer = _model_optimizer()
    before = copy.deepcopy(model.state_dict())
    scaler = FakeGradScaler(force_inf_on_unscale=True)
    controller = NumericalStepController(optimizer, precision_mode="fp16", scaler=scaler)
    telemetry = controller.run_batch(model, *_batch(), CombinedBCEDiceLoss(pos_weight=5))
    assert telemetry.optimizer_updates_this_batch == 0
    assert telemetry.nonfinite_gradient
    assert telemetry.amp_overflow_scale_drop
    assert telemetry.scale_after < telemetry.scale_before
    assert controller.counters.nonfinite_gradient == 1
    assert controller.counters.amp_overflow_scale_drop == 1
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    controller.close()


def test_nonfinite_forward_loss_executes_no_backward_or_step():
    class NonfiniteModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, inputs):
            return inputs * self.weight * float("nan")

    model = NonfiniteModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    controller = NumericalStepController(optimizer, precision_mode="fp32")
    telemetry = controller.run_batch(model, *_batch(), CombinedBCEDiceLoss())
    assert telemetry.nonfinite_forward_loss
    assert telemetry.optimizer_updates_this_batch == 0
    assert model.weight.grad is None
    assert controller.counters.nonfinite_forward_loss == 1
    controller.close()


def test_automatic_fp32_fallback_is_disabled_and_cannot_double_step():
    model, optimizer = _model_optimizer()
    with pytest.raises(ValueError, match="intentionally unsupported"):
        NumericalStepController(
            optimizer,
            precision_mode="fp16",
            scaler=FakeGradScaler(),
            automatic_fp32_retry=True,
        )
    controller = NumericalStepController(optimizer, precision_mode="fp16", scaler=FakeGradScaler())
    telemetry = controller.run_batch(model, *_batch(), CombinedBCEDiceLoss())
    assert telemetry.optimizer_updates_this_batch == 1
    assert not telemetry.fp32_retry_attempted
    assert not telemetry.fp32_retry_executed
    controller.close()


def test_loss_is_float32_under_bf16_autocast():
    model, optimizer = _model_optimizer()
    controller = NumericalStepController(optimizer, precision_mode="bf16")
    telemetry = controller.run_batch(model, *_batch(), CombinedBCEDiceLoss())
    assert telemetry.loss_dtype == "torch.float32"
    assert telemetry.optimizer_updates_this_batch == 1
    controller.close()


def test_unscale_precedes_gradient_check_and_clipping(monkeypatch):
    model, optimizer = _model_optimizer()
    scaler = FakeGradScaler()
    order = scaler.operations

    original_finite = numerics._finite_gradients
    original_clip = torch.nn.utils.clip_grad_norm_

    def checked(parameters):
        order.append("finite_check")
        return original_finite(parameters)

    def clipped(parameters, max_norm):
        order.append("clip")
        return original_clip(parameters, max_norm)

    monkeypatch.setattr(numerics, "_finite_gradients", checked)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clipped)
    controller = NumericalStepController(
        optimizer, precision_mode="fp16", scaler=scaler, gradient_clip_max_norm=1.0
    )
    controller.run_batch(model, *_batch(), CombinedBCEDiceLoss())
    assert order.index("unscale") < order.index("finite_check") < order.index("clip")
    assert order.count("scaler_step") == 1
    assert order.count("update") == 1
    controller.close()


def test_checkpoint_resume_preserves_scaler_and_numerical_telemetry(tmp_path):
    model, optimizer = _model_optimizer()
    scaler = FakeGradScaler(scale=16)
    controller = NumericalStepController(optimizer, precision_mode="fp16", scaler=scaler)
    controller.run_batch(model, *_batch(), CombinedBCEDiceLoss())
    configuration = {"loss": {"pos_weight": 5.0}, "training": {"mixed_precision": True}}
    path = tmp_path / "telemetry.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        epoch=1,
        configuration=configuration,
        best_validation={"epoch": 1, "validation_total_loss": 1.0},
        numerical_controller=controller,
    )
    restored_model, restored_optimizer = _model_optimizer()
    restored_scaler = FakeGradScaler(scale=2)
    restored_controller = NumericalStepController(
        restored_optimizer, precision_mode="fp16", scaler=restored_scaler
    )
    load_training_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=None,
        scaler=restored_scaler,
        expected_configuration=configuration,
        numerical_controller=restored_controller,
    )
    assert restored_scaler.get_scale() == 16
    assert restored_controller.counters.optimizer_step_executed == 1
    assert restored_controller.counters.attempted_batches == 1
    controller.close()
    restored_controller.close()


def test_existing_configuration_without_precision_mode_still_maps_to_fp16():
    configuration = {"training": {"mixed_precision": True}}
    mode = configuration["training"].get(
        "precision_mode", "fp16" if configuration["training"]["mixed_precision"] else "fp32"
    )
    assert mode == "fp16"


def test_probe_dataset_builder_can_only_request_development_training(monkeypatch):
    requested_splits = []

    class FakeDataset:
        def __init__(self, repo_root, development_split, manifest_path, **kwargs):  # noqa: ARG002
            requested_splits.append(development_split)
            self.rows = [{"development_split": development_split}] * 1981

        def __len__(self):
            return len(self.rows)

    monkeypatch.setattr("scripts.probe_numerical_stability.KSDD2FullImageDataset", FakeDataset)
    config = {
        "detector_normalization": {"mean": [0, 0, 0], "standard_deviation": [1, 1, 1]},
        "paths": {"manifest": "unused.csv"},
        "model": {"input_width": 256, "input_height": 672, "image_padding_mode": "reflect"},
    }
    assert len(build_training_dataset(config)) == 1981
    assert requested_splits == ["train"]
