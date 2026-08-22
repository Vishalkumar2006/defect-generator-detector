from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from defectgen.gan.training_pairs import GANTrainingSample
from defectgen.models import MaskConditionedPatchDiscriminator, MaskedResidualGenerator
from defectgen.training.gan_losses import load_gan_loss_config
from defectgen.training.gan_trainer import (
    GANOneStepTrainer,
    GANTrainingNumericalError,
    calibrate_gan_loss_scales,
    canonical_adversarial_gradient_telemetry,
    collate_gan_training_samples,
    load_gan_trainer_config,
    parameter_hash,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides):
    selected = load_gan_trainer_config(REPO_ROOT / "configs" / "gan_one_step.json")
    return replace(selected, **overrides)


def _models():
    return (
        MaskedResidualGenerator(
            base_channels=8,
            residual_blocks=1,
            group_norm_groups=4,
            support_dilation_radius=3,
        ),
        MaskConditionedPatchDiscriminator(
            base_channels=8,
            group_norm_groups=4,
            use_spectral_norm=True,
        ),
    )


def _sample(index: int, *, split: str = "train") -> GANTrainingSample:
    generator = torch.Generator().manual_seed(100 + index)
    composite = torch.rand(3, 64, 64, generator=generator) * 1.6 - 0.8
    real = torch.rand(3, 64, 64, generator=generator) * 1.6 - 0.8
    mask = torch.zeros(1, 64, 64)
    left = 25 + index
    mask[:, 27:37, left : left + 8] = 1
    valid = torch.zeros(1, 64, 64)
    valid[:, :, 4:60] = 1
    return GANTrainingSample(
        composite_image=composite,
        generator_mask=mask,
        transformed_defect_alpha=mask.clone(),
        fake_discriminator_mask=mask.clone(),
        real_image=real,
        real_discriminator_mask=mask.clone(),
        fake_valid_mask=valid.clone(),
        real_valid_mask=valid.clone(),
        real_valid_coverage=valid.clone(),
        metadata={
            "split": split,
            "sample_index": index,
            "template_id": f"synthetic:{index}",
            "target_contact_sides": {
                "top": False,
                "bottom": False,
                "left": False,
                "right": False,
            },
        },
    )


def _batch(*, split: str = "train"):
    return collate_gan_training_samples(
        [_sample(0, split=split), _sample(1, split=split)]
    )


def _trainer(*, config=None, device="cpu"):
    generator, discriminator = _models()
    return GANOneStepTrainer(
        generator,
        discriminator,
        config or _config(),
        load_gan_loss_config(REPO_ROOT / "configs" / "gan_losses.json"),
        device=device,
    )


def test_configuration_and_batch_collation_contract() -> None:
    config = _config()
    assert config.trainer_version == "g1_4_gan_one_step_v1"
    assert config.batch_size == 2 and config.loss_coefficients_provisional
    assert config.generator_optimizer.betas == (0.0, 0.9)
    batch = _batch()
    assert batch.composite_image.shape == (2, 3, 64, 64)
    assert batch.generator_mask.shape == (2, 1, 64, 64)
    assert len(batch.metadata) == 2 and batch.batch_size == 2
    with pytest.raises(ValueError, match="empty"):
        collate_gan_training_samples([])
    with pytest.raises(ValueError, match="one recognized split"):
        collate_gan_training_samples([_sample(0), _sample(1, split="monitor")])


def test_discriminator_step_changes_only_discriminator_and_detaches_fake() -> None:
    torch.manual_seed(1)
    trainer = _trainer()
    generator_before = parameter_hash(trainer.generator)
    discriminator_before = parameter_hash(trainer.discriminator)
    result = trainer.discriminator_step(_batch(), global_step=0)
    assert parameter_hash(trainer.generator) == generator_before
    assert parameter_hash(trainer.discriminator) != discriminator_before
    assert result["fake_tensor_detached"]
    assert not result["generator_gradients_constructed"]
    assert result["real_fake_masks_identical"]
    assert result["real_fake_localization_weights_identical"]
    assert result["aligned_views_used"]
    assert result["optimizer_state_finite"]
    assert result["gradient_clipping"]["pre_clipping_norm"] > 0
    assert (
        result["gradient_clipping"]["post_clipping_norm"]
        <= result["gradient_clipping"]["maximum_norm"] + 1e-5
    )


def test_generator_step_changes_only_generator_and_preserves_locality_and_gradients() -> None:
    torch.manual_seed(2)
    trainer = _trainer()
    trainer.discriminator_step(_batch(), global_step=0)
    generator_before = parameter_hash(trainer.generator)
    discriminator_before = parameter_hash(trainer.discriminator)
    result = trainer.generator_step(_batch())
    assert parameter_hash(trainer.generator) != generator_before
    assert parameter_hash(trainer.discriminator) == discriminator_before
    assert result["generator_graph_recomputed"]
    assert not result["discriminator_gradients_constructed"]
    assert result["discriminator_requires_grad_restored"]
    assert result["generator_locality_before_step"]
    assert result["generator_locality_after_step"]
    assert result["canonical_defect_gradient_coverage"] == 1.0
    assert result["canonical_defect_gradient_total_pixel_count"] > 0
    assert result["canonical_defect_gradient_active_pixel_count"] == result[
        "canonical_defect_gradient_total_pixel_count"
    ]
    assert result["canonical_defect_gradient_active_count"] == result[
        "canonical_defect_gradient_active_pixel_count"
    ]
    assert result["canonical_defect_gradient_total_count"] == result[
        "canonical_defect_gradient_total_pixel_count"
    ]
    assert result["canonical_defect_gradient_total_channel_count"] == (
        result["canonical_defect_gradient_total_pixel_count"] * 3
    )
    assert result["canonical_defect_gradient_nonfinite_channel_count"] == 0
    assert result["maximum_invalid_fake_pixel_gradient"] == 0.0
    assert result["output_range_violation_count"] == 0
    assert result["clamp_saturation_fraction"] == 0.0
    assert result["clamp_saturation_deprecated"]
    assert result["mean_absolute_applied_residual"] == result[
        "mean_absolute_residual_inside_support"
    ]
    assert result["optimizer_state_finite"]
    assert all(torch.isfinite(torch.tensor(value)) for value in result["losses"].values())


def test_canonical_gradient_pixel_allows_one_zero_rgb_component() -> None:
    gradient = torch.tensor([[[[0.0]], [[2.0]], [[-3.0]]]])
    telemetry = canonical_adversarial_gradient_telemetry(
        gradient, torch.ones(1, 1, 1, 1)
    )

    assert telemetry["canonical_defect_gradient_active_pixel_count"] == 1
    assert telemetry["canonical_defect_gradient_total_pixel_count"] == 1
    assert telemetry["canonical_defect_gradient_active_channel_count"] == 2
    assert telemetry["canonical_defect_gradient_total_channel_count"] == 3
    assert telemetry["canonical_defect_gradient_nonfinite_channel_count"] == 0
    assert telemetry["canonical_defect_gradient_coverage"] == 1.0
    assert isinstance(telemetry["canonical_defect_gradient_active_pixel_count"], int)
    assert isinstance(telemetry["canonical_defect_gradient_total_pixel_count"], int)


def test_canonical_gradient_pixel_rejects_an_all_zero_rgb_vector() -> None:
    gradient = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]]
    )
    telemetry = canonical_adversarial_gradient_telemetry(
        gradient, torch.ones(1, 1, 1, 2)
    )

    assert telemetry["canonical_defect_gradient_active_pixel_count"] == 1
    assert telemetry["canonical_defect_gradient_total_pixel_count"] == 2
    assert telemetry["canonical_defect_gradient_active_channel_count"] == 1
    assert telemetry["canonical_defect_gradient_coverage"] == 0.5


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_canonical_gradient_pixel_rejects_any_nonfinite_component(
    nonfinite: float,
) -> None:
    gradient = torch.tensor([[[[1.0]], [[nonfinite]], [[2.0]]]])
    telemetry = canonical_adversarial_gradient_telemetry(
        gradient, torch.ones(1, 1, 1, 1)
    )

    assert telemetry["canonical_defect_gradient_active_pixel_count"] == 0
    assert telemetry["canonical_defect_gradient_total_pixel_count"] == 1
    assert telemetry["canonical_defect_gradient_nonfinite_channel_count"] == 1
    assert telemetry["canonical_defect_gradient_coverage"] == 0.0


def test_generator_graph_is_detached_then_recomputed_for_the_two_steps() -> None:
    torch.manual_seed(21)
    trainer = _trainer()
    forward_requires_grad: list[bool] = []
    handle = trainer.generator.register_forward_hook(
        lambda _module, _inputs, output: forward_requires_grad.append(
            output.refined_image.requires_grad
        )
    )
    try:
        trainer.discriminator_step(_batch(), global_step=0)
        assert forward_requires_grad == [False]
        trainer.generator_step(_batch())
    finally:
        handle.remove()
    # The G step performs a fresh differentiable forward, followed by a
    # no-grad locality verification after its optimizer update.
    assert forward_requires_grad == [False, True, False]


def test_zero_mask_generator_behavior_remains_bit_exact_after_update() -> None:
    torch.manual_seed(3)
    trainer = _trainer()
    trainer.discriminator_step(_batch(), global_step=0)
    trainer.generator_step(_batch())
    image = torch.rand(1, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        output = trainer.generator(image, torch.zeros(1, 1, 64, 64))
    assert torch.equal(output.refined_image, image)
    assert not output.support_mask.any()


def test_r1_schedule_and_lazy_scaling_without_sixteen_updates() -> None:
    torch.manual_seed(4)
    trainer = _trainer()
    assert not trainer.r1_is_scheduled(0)
    assert not trainer.r1_is_scheduled(14)
    assert trainer.r1_is_scheduled(15)
    result = trainer.discriminator_step(_batch(), global_step=15)
    assert result["r1_scheduled"]
    expected = (
        result["losses"]["raw_r1"]
        * result["r1_gamma"]
        * result["r1_interval"]
        / 2
    )
    assert result["losses"]["scaled_r1"] == pytest.approx(expected)


def test_monitor_batch_is_never_optimized_and_forward_has_no_step() -> None:
    trainer = _trainer()
    with pytest.raises(ValueError, match="train split"):
        trainer.discriminator_step(_batch(split="monitor"), global_step=0)
    with pytest.raises(ValueError, match="train split"):
        trainer.generator_step(_batch(split="monitor"))
    result = trainer.monitor_forward(_batch(split="monitor"))
    assert result["optimizer_steps"] == 0


def test_fp32_deterministic_repeatability() -> None:
    batch = _batch()
    results = []
    for _ in range(2):
        torch.manual_seed(5)
        trainer = _trainer()
        discriminator = trainer.discriminator_step(batch, global_step=0)
        generator = trainer.generator_step(batch)
        results.append(
            (
                discriminator["losses"],
                generator["losses"],
                parameter_hash(trainer.generator),
                parameter_hash(trainer.discriminator),
            )
        )
    assert results[0] == results[1]


def test_identity_calibration_is_finite_and_reports_expected_staged_zeros() -> None:
    torch.manual_seed(6)
    trainer = _trainer(config=_config(deterministic_audit_batches=2))
    generator_before = parameter_hash(trainer.generator)
    discriminator_before = parameter_hash(trainer.discriminator)
    report = calibrate_gan_loss_scales(trainer, [_batch(), _batch()])
    assert report["parameters_unchanged"]
    assert parameter_hash(trainer.generator) == generator_before
    assert parameter_hash(trainer.discriminator) == discriminator_before
    for statistics in report["raw_loss_distributions"].values():
        assert statistics["finite_count"] == 2 and statistics["nonfinite_count"] == 0
    gradients = report["generator_unit_gradient_scales"]
    assert gradients["adversarial"]["zero_gradient_count"] == 0
    assert gradients["adversarial"]["unit_gradient_norm"]["minimum"] > 0
    for name in ("change", "seam", "total_variation"):
        assert gradients[name]["zero_gradient_count"] == 2
        assert gradients[name]["unit_gradient_norm"]["maximum"] == 0
    assert not report["suggestions_written_to_configuration"]
    assert not report["suggestions_used_for_one_step"]


class _NaNDiscriminator(MaskConditionedPatchDiscriminator):
    def forward(self, image, defect_mask):
        return super().forward(image, defect_mask) * torch.tensor(float("nan"))


def test_nonfinite_loss_prevents_parameter_mutation() -> None:
    torch.manual_seed(7)
    generator, _ = _models()
    discriminator = _NaNDiscriminator(
        base_channels=8, group_norm_groups=4, use_spectral_norm=True
    )
    trainer = GANOneStepTrainer(
        generator,
        discriminator,
        _config(),
        load_gan_loss_config(REPO_ROOT / "configs" / "gan_losses.json"),
        device="cpu",
    )
    before = parameter_hash(trainer.discriminator)
    with pytest.raises((ValueError, GANTrainingNumericalError), match="finite|Non-finite"):
        trainer.discriminator_step(_batch(), global_step=0)
    assert parameter_hash(trainer.discriminator) == before


def test_nonfinite_gradient_prevents_parameter_mutation() -> None:
    torch.manual_seed(8)
    trainer = _trainer()
    parameter = next(trainer.discriminator.parameters())
    handle = parameter.register_hook(lambda gradient: gradient * float("nan"))
    before = parameter_hash(trainer.discriminator)
    try:
        with pytest.raises(GANTrainingNumericalError, match="non-finite gradient"):
            trainer.discriminator_step(_batch(), global_step=0)
    finally:
        handle.remove()
    assert parameter_hash(trainer.discriminator) == before


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA bf16 is not supported",
)
def test_bf16_cuda_autocast_mechanics() -> None:
    torch.manual_seed(9)
    trainer = _trainer(device="cuda")
    assert trainer.precision == "bf16"
    assert trainer.generator_scaler is None and trainer.discriminator_scaler is None
    result = trainer.discriminator_step(_batch(), global_step=0)
    assert all(torch.isfinite(torch.tensor(value)) for value in result["losses"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fp16_cuda_uses_grad_scalers() -> None:
    torch.manual_seed(10)
    trainer = _trainer(config=_config(cuda_precision="fp16"), device="cuda")
    assert trainer.precision == "fp16"
    assert trainer.generator_scaler is not None and trainer.discriminator_scaler is not None
    before = parameter_hash(trainer.discriminator)
    try:
        result = trainer.discriminator_step(_batch(), global_step=0)
    except GANTrainingNumericalError as error:
        # A high initial scale may overflow fp16. G1.4 must fail explicitly
        # without mutating parameters, not silently skip/retry the batch.
        assert "non-finite gradient" in str(error)
        assert parameter_hash(trainer.discriminator) == before
    else:
        assert result["gradient_clipping"]["scale_before"] is not None
