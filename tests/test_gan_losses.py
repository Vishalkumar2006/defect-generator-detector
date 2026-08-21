from __future__ import annotations

from pathlib import Path

import pytest
import torch

from defectgen.models import MaskConditionedPatchDiscriminator, MaskedResidualGenerator
from defectgen.training.gan_losses import (
    GeneratorLossWeights,
    aggregate_generator_losses,
    boundary_seam_loss,
    canonicalize_discriminator_mask,
    inner_boundary_ring,
    load_gan_loss_config,
    localized_discriminator_hinge_loss,
    localized_generator_adversarial_loss,
    localized_r1_gradient_penalty,
    localized_weighted_mean,
    masked_total_variation_loss,
    patch_logit_localization_weights,
    support_normalized_change_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _generator() -> MaskedResidualGenerator:
    return MaskedResidualGenerator(
        base_channels=8,
        residual_blocks=1,
        group_norm_groups=4,
        support_dilation_radius=3,
    )


def _discriminator() -> MaskConditionedPatchDiscriminator:
    return MaskConditionedPatchDiscriminator(
        base_channels=8, group_norm_groups=4, use_spectral_norm=True
    )


def test_canonical_mask_is_binary_and_uses_inclusive_threshold() -> None:
    mask = torch.tensor([[[[0.0, 0.49, 0.5, 1.0]]]])
    canonical = canonicalize_discriminator_mask(mask, threshold=0.5)
    assert canonical.dtype == torch.float32
    assert torch.equal(canonical, torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]]))
    assert set(canonical.unique().tolist()) == {0.0, 1.0}
    assert torch.equal(
        canonicalize_discriminator_mask(mask.bool()), mask.bool().float()
    )


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (torch.zeros(1, 16, 16), "rank 4"),
        (torch.zeros(1, 2, 16, 16), "1 channel"),
        (torch.zeros(1, 1, 16, 16, dtype=torch.int64), "bool or floating"),
        (torch.full((1, 1, 16, 16), -0.1), "[0, 1]"),
        (torch.full((1, 1, 16, 16), 1.1), "[0, 1]"),
        (torch.full((1, 1, 16, 16), float("nan")), "finite"),
        (torch.full((1, 1, 16, 16), float("inf")), "finite"),
    ],
)
def test_invalid_canonical_masks_are_rejected(mask, message) -> None:
    with pytest.raises(ValueError, match=message):
        canonicalize_discriminator_mask(mask)


def test_patch_weights_match_logits_and_leave_far_background_zero() -> None:
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 30:34, 30:34] = 1
    logits = torch.zeros(1, 1, 8, 8)
    weights = patch_logit_localization_weights(mask, logits, localization_radius=0)
    assert weights.shape == logits.shape
    assert torch.isfinite(weights).all() and weights.min() >= 0
    assert weights[0, 0, 0, 0] == 0
    assert weights[0, 0, -1, -1] == 0
    assert weights.sum() > 0


def test_localized_mean_ignores_zero_weight_logits() -> None:
    values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    weights = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    expected = localized_weighted_mean(values, weights)
    changed = values.clone()
    changed[weights == 0] = 1_000_000
    assert torch.equal(localized_weighted_mean(changed, weights), expected)
    assert expected == 2.0


def test_hand_calculated_localized_hinge_losses_and_generator_sign() -> None:
    real_logits = torch.tensor([[[[2.0, 0.0, -1.0]]]])
    fake_logits = torch.tensor([[[[-2.0, 0.0, 5.0]]]])
    weights = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    discriminator = localized_discriminator_hinge_loss(
        real_logits, fake_logits, weights, weights
    )
    generator = localized_generator_adversarial_loss(fake_logits, weights)
    assert discriminator.real == 0.5
    assert discriminator.fake == 0.5
    assert discriminator.total == 1.0
    assert generator.adversarial == 1.0
    assert localized_generator_adversarial_loss(
        torch.tensor([[[[2.0]]]]), torch.ones(1, 1, 1, 1)
    ).total == -2.0


def test_discriminator_step_detaches_generator_but_updates_discriminator() -> None:
    torch.manual_seed(10)
    generator = _generator()
    discriminator = _discriminator()
    composite = torch.zeros(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 28:36, 28:36] = 1
    canonical = canonicalize_discriminator_mask(mask)
    generated = generator(composite, mask)
    real_logits = discriminator(composite, canonical)
    fake_logits = discriminator(generated.refined_image.detach(), canonical)
    real_weights = patch_logit_localization_weights(
        canonical, real_logits, localization_radius=3
    )
    fake_weights = patch_logit_localization_weights(
        canonical, fake_logits, localization_radius=3
    )
    loss = localized_discriminator_hinge_loss(
        real_logits, fake_logits, real_weights, fake_weights
    ).total
    loss.backward()
    assert all(parameter.grad is None for parameter in generator.parameters())
    discriminator_gradients = [
        parameter.grad for parameter in discriminator.parameters() if parameter.grad is not None
    ]
    assert discriminator_gradients
    assert all(torch.isfinite(gradient).all() for gradient in discriminator_gradients)


def test_generator_adversarial_loss_reaches_generator_through_discriminator() -> None:
    torch.manual_seed(11)
    generator = _generator()
    discriminator = _discriminator()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    composite = torch.zeros(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 28:36, 28:36] = 1
    generated = generator(composite, mask)
    logits = discriminator(
        generated.refined_image, canonicalize_discriminator_mask(mask)
    )
    weights = patch_logit_localization_weights(mask, logits, localization_radius=3)
    localized_generator_adversarial_loss(logits, weights).total.backward()
    gradients = [parameter.grad for parameter in generator.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool((gradient != 0).any()) for gradient in gradients)


def test_change_loss_is_support_normalized_and_ignores_background() -> None:
    composite = torch.zeros(1, 3, 16, 16)
    support = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    support[:, :, 4:12, 4:12] = True
    assert support_normalized_change_loss(composite, composite, support) == 0
    changed = composite.clone()
    changed[:, :, 6:8, 6:8] = 0.5
    baseline = support_normalized_change_loss(changed, composite, support)
    changed[:, :, :2, :2] = 1.0
    assert torch.equal(support_normalized_change_loss(changed, composite, support), baseline)


def test_boundary_loss_detects_only_inner_ring_changes() -> None:
    composite = torch.zeros(1, 3, 24, 24)
    support = torch.zeros(1, 1, 24, 24, dtype=torch.bool)
    support[:, :, 4:20, 4:20] = True
    ring = inner_boundary_ring(support, width=2)
    assert ring.any() and not ring[:, :, 10:14, 10:14].any()
    boundary_changed = composite.clone()
    boundary_changed[ring.expand_as(composite)] = 0.5
    assert boundary_seam_loss(
        boundary_changed, composite, support, boundary_width=2
    ) == 0.5
    interior_changed = composite.clone()
    interior_changed[:, :, 10:14, 10:14] = 1
    assert boundary_seam_loss(interior_changed, composite, support, boundary_width=2) == 0
    assert boundary_seam_loss(composite, composite, support, boundary_width=2) == 0


@pytest.mark.parametrize(
    "mask_slice",
    [
        (slice(0, 5), slice(10, 20)),
        (slice(27, 32), slice(10, 20)),
        (slice(10, 20), slice(0, 5)),
        (slice(10, 20), slice(27, 32)),
        (slice(0, 5), slice(0, 5)),
        (slice(27, 32), slice(27, 32)),
        (slice(14, 18), slice(0, 32)),
    ],
)
def test_boundary_and_tv_losses_are_finite_for_border_combinations(mask_slice) -> None:
    support = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    support[:, :, mask_slice[0], mask_slice[1]] = True
    composite = torch.zeros(1, 3, 32, 32)
    refined = composite.clone()
    refined[support.expand_as(refined)] = 0.2
    assert torch.isfinite(boundary_seam_loss(refined, composite, support))
    assert torch.isfinite(masked_total_variation_loss(refined, support))
    assert not bool((inner_boundary_ring(support) < 0).any())


def test_total_variation_is_zero_for_constant_residual() -> None:
    residual = torch.full((1, 3, 16, 16), 0.25)
    support = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    assert masked_total_variation_loss(residual, support) == 0
    assert masked_total_variation_loss(torch.zeros_like(residual), support) == 0


def test_all_zero_localization_is_explicitly_rejected() -> None:
    with pytest.raises(ValueError, match="at least one active"):
        patch_logit_localization_weights(
            torch.zeros(1, 1, 64, 64), torch.zeros(1, 1, 8, 8)
        )
    with pytest.raises(ValueError, match="positive total weight"):
        localized_weighted_mean(torch.zeros(1), torch.zeros(1))


def test_r1_penalty_and_backward_gradients_are_finite() -> None:
    torch.manual_seed(12)
    discriminator = _discriminator()
    image = torch.zeros(1, 3, 64, 64, requires_grad=True)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 28:36, 28:36] = 1
    penalty = localized_r1_gradient_penalty(
        discriminator, image, mask, localization_radius=3
    )
    assert penalty.ndim == 0 and torch.isfinite(penalty) and penalty >= 0
    penalty.backward()
    gradients = [
        parameter.grad for parameter in discriminator.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_config_and_provisional_aggregation_are_finite() -> None:
    config = load_gan_loss_config(REPO_ROOT / "configs" / "gan_losses.json")
    assert config.aggregation_weights_provisional
    scalar = torch.tensor(1.0, requires_grad=True)
    components = aggregate_generator_losses(
        adversarial=scalar,
        change=2 * scalar,
        boundary=3 * scalar,
        total_variation=4 * scalar,
        weights=config.generator_loss_weights,
    )
    assert components.total == 10
    components.total.backward()
    assert scalar.grad == 10
    with pytest.raises(ValueError, match="not all zero"):
        GeneratorLossWeights(0, 0, 0, 0).validate()
