from __future__ import annotations

from pathlib import Path

import pytest
import torch

from defectgen.models import (
    MaskConditionedPatchDiscriminator,
    MaskedResidualGenerator,
    build_gan_models,
    load_gan_architecture_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _generator(**overrides) -> MaskedResidualGenerator:
    settings = {
        "base_channels": 8,
        "downsample_stages": 3,
        "residual_blocks": 1,
        "group_norm_groups": 4,
        "support_dilation_radius": 3,
        "residual_scale": 0.25,
    }
    settings.update(overrides)
    return MaskedResidualGenerator(**settings)


def _discriminator() -> MaskConditionedPatchDiscriminator:
    return MaskConditionedPatchDiscriminator(
        base_channels=8, group_norm_groups=4, use_spectral_norm=True
    )


def test_production_input_shape_and_configured_architecture() -> None:
    config = load_gan_architecture_config(REPO_ROOT / "configs" / "gan_architecture.json")
    generator, discriminator = build_gan_models(config)
    assert generator.support_dilation_radius == 12
    assert generator.residual_scale == 0.25
    composite = torch.zeros(1, 3, 512, 256)
    mask = torch.zeros(1, 1, 512, 256, dtype=torch.bool)
    mask[:, :, 250:256, 120:128] = True
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        result = generator(composite, mask)
        logits = discriminator(result.refined_image, mask)
    assert result.refined_image.shape == composite.shape
    assert result.raw_residual.shape == composite.shape
    assert result.support_mask.shape == mask.shape
    assert torch.isfinite(result.refined_image).all()
    assert result.refined_image.min() >= -1 and result.refined_image.max() <= 1
    assert logits.ndim == 4 and logits.shape[:2] == (1, 1) and logits.numel() > 0
    assert torch.isfinite(logits).all()


def test_generator_is_exact_outside_support_and_for_zero_mask() -> None:
    torch.manual_seed(3)
    model = _generator().eval()
    composite = torch.rand(1, 3, 64, 64) * 2 - 1
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 30:34, 30:34] = 1
    with torch.no_grad():
        result = model(composite, mask)
        zero = model(composite, torch.zeros_like(mask))
    outside = ~result.support_mask.expand_as(composite)
    assert torch.equal(result.refined_image[outside], composite[outside])
    assert torch.equal(zero.refined_image, composite)
    assert not zero.support_mask.any()


def test_nonzero_generator_gradients_reach_parameters() -> None:
    torch.manual_seed(4)
    model = _generator()
    composite = torch.zeros(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 24:40, 24:40] = 1
    result = model(composite, mask)
    result.refined_image.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool((gradient != 0).any()) for gradient in gradients)


def test_discriminator_forward_backward_is_finite_and_returns_raw_patch_logits() -> None:
    torch.manual_seed(5)
    model = _discriminator()
    image = torch.rand(2, 3, 64, 64) * 2 - 1
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:30, 18:35] = 1
    logits = model(image, mask)
    assert logits.shape[0:2] == (2, 1) and logits.shape[-2] > 0 and logits.shape[-1] > 0
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    "mask_slice",
    [
        (slice(0, 4), slice(20, 30)),
        (slice(60, 64), slice(20, 30)),
        (slice(20, 30), slice(0, 4)),
        (slice(20, 30), slice(60, 64)),
        (slice(0, 4), slice(0, 4)),
        (slice(60, 64), slice(60, 64)),
        (slice(28, 34), slice(0, 64)),
    ],
)
def test_generator_supports_border_corner_and_left_right_masks(mask_slice) -> None:
    model = _generator().eval()
    composite = torch.zeros(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    mask[:, :, mask_slice[0], mask_slice[1]] = True
    with torch.no_grad():
        result = model(composite, mask)
    assert result.refined_image.shape == composite.shape
    assert torch.isfinite(result.refined_image).all()
    assert bool(result.support_mask[mask].all())
    outside = ~result.support_mask.expand_as(composite)
    assert torch.equal(result.refined_image[outside], composite[outside])


def test_state_dict_round_trip_preserves_eval_outputs(tmp_path: Path) -> None:
    torch.manual_seed(6)
    generator = _generator().eval()
    discriminator = _discriminator().eval()
    image = torch.zeros(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 28:36, 28:36] = 1
    with torch.no_grad():
        expected_generator = generator(image, mask).refined_image
        expected_discriminator = discriminator(image, mask)
    path = tmp_path / "gan_architecture.pt"
    torch.save(
        {"generator": generator.state_dict(), "discriminator": discriminator.state_dict()}, path
    )
    restored_generator = _generator().eval()
    restored_discriminator = _discriminator().eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored_generator.load_state_dict(state["generator"])
    restored_discriminator.load_state_dict(state["discriminator"])
    with torch.no_grad():
        actual_generator = restored_generator(image, mask).refined_image
        actual_discriminator = restored_discriminator(image, mask)
    assert torch.equal(expected_generator, actual_generator)
    assert torch.equal(expected_discriminator, actual_discriminator)


@pytest.mark.parametrize(
    ("image", "mask", "message"),
    [
        (torch.zeros(3, 64, 64), torch.zeros(1, 1, 64, 64), "rank 4"),
        (torch.zeros(1, 2, 64, 64), torch.zeros(1, 1, 64, 64), "3 channels"),
        (torch.zeros(1, 3, 64, 64), torch.zeros(1, 2, 64, 64), "1 channel"),
        (torch.zeros(1, 3, 64, 64), torch.zeros(1, 1, 32, 64), "matching"),
        (torch.zeros(1, 3, 62, 64), torch.zeros(1, 1, 62, 64), "divisible by 8"),
        (torch.full((1, 3, 64, 64), 1.1), torch.zeros(1, 1, 64, 64), "[-1, 1]"),
        (torch.zeros(1, 3, 64, 64), torch.full((1, 1, 64, 64), -0.1), "[0, 1]"),
        (
            torch.full((1, 3, 64, 64), float("nan")),
            torch.zeros(1, 1, 64, 64),
            "finite",
        ),
        (
            torch.zeros(1, 3, 64, 64),
            torch.full((1, 1, 64, 64), float("inf")),
            "finite",
        ),
    ],
)
def test_invalid_generator_inputs_are_rejected(image, mask, message) -> None:
    with pytest.raises(ValueError, match=message):
        _generator()(image, mask)


def test_invalid_discriminator_inputs_are_rejected() -> None:
    model = _discriminator()
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.zeros(1, 4, 64, 64), torch.zeros(1, 1, 64, 64))
    with pytest.raises(ValueError, match="at least 24"):
        model(torch.zeros(1, 3, 16, 16), torch.zeros(1, 1, 16, 16))
