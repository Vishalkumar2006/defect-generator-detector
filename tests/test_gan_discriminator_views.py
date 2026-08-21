from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from defectgen.gan.discriminator_views import (
    DISCRIMINATOR_INVALID_FILL_VALUE,
    prepare_aligned_discriminator_views,
    prepare_training_sample_discriminator_views,
)
from defectgen.gan.geometry import ContactSides
from defectgen.gan.pipeline import construct_coarse_gan_input
from defectgen.gan.training_pairs import GANTrainingSample
from defectgen.models.gan import MaskConditionedPatchDiscriminator
from defectgen.training.gan_losses import (
    canonicalize_discriminator_mask,
    localized_generator_adversarial_loss,
    patch_logit_localization_weights,
)


HEIGHT = WIDTH = 64


def _provenance(source_id: str, contacts: ContactSides) -> dict:
    return {
        "normal_background_sample_id": "normal",
        "source_defect_sample_id": source_id,
        "connected_component_id": 0,
        "source_mask_bounding_box": {"x_min": 8, "y_min": 8, "x_max": 55, "y_max": 55},
        "source_window_coordinates": {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT},
        "target_window_coordinates": {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT},
        "partial_component": False,
        "coverage_fraction": 1.0,
        "touches_native_border": contacts.any,
        "source_contact_sides": contacts.to_dict(),
        "target_window_native_contact_sides": ContactSides(True, True, True, True).to_dict(),
        "minimum_positive_pixels": 4,
        "source_manifest_sha256": "a" * 64,
        "gan_manifest_content_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "pipeline_version": "test_g1_3b",
    }


def _continuous_validity_sample(
    source_id: str,
    contacts: ContactSides,
    *,
    scale: float,
    horizontal_flip: bool = False,
) -> dict:
    source_valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
    source_valid[:, 8:56] = True
    mask = np.zeros_like(source_valid)
    if contacts.left and contacts.right:
        mask[28:36, 8:56] = True
    else:
        top = 0 if contacts.top else 24
        left = 8 if contacts.left else (50 if contacts.right else 49)
        mask[top : top + 8, left : left + 6] = True
    source = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)
    source[mask] = [230, 45, 20]
    background = np.full_like(source, 110)
    target_valid = np.zeros_like(source_valid)
    target_valid[:, 8:56] = True
    return construct_coarse_gan_input(
        source,
        mask,
        background,
        target_valid,
        seed=42,
        transform_settings={
            "horizontal_flip_probability": float(horizontal_flip),
            "vertical_flip_probability": 0.0,
            "minimum_scale": scale,
            "maximum_scale": scale,
            "minimum_retained_area_fraction": 0.70,
            "feather_radius": 3,
            "non_border_native_margin": 4,
        },
        colour_settings={
            "enabled": False,
            "boundary_radius": 4,
            "minimum_gain": 0.8,
            "maximum_gain": 1.2,
            "maximum_absolute_offset": 0.15,
        },
        provenance_base=_provenance(source_id, contacts),
        transform_parameters={
            "horizontal_flip": horizontal_flip,
            "vertical_flip": False,
            "scale": scale,
        },
        source_valid_region=source_valid,
        include_training_details=True,
    )


def _images() -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.linspace(-0.9, 0.9, HEIGHT * WIDTH).reshape(1, 1, HEIGHT, WIDTH)
    real = values.repeat(1, 3, 1, 1)
    fake = torch.flip(real, dims=(-1,))
    return real, fake


def _case_masks(case: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    real_valid = torch.ones(1, 1, HEIGHT, WIDTH)
    fake_valid = torch.ones_like(real_valid)
    support = torch.zeros_like(real_valid)
    regions = {
        "central": (slice(26, 38), slice(26, 38)),
        "left": (slice(24, 40), slice(0, 6)),
        "right": (slice(24, 40), slice(WIDTH - 6, WIDTH)),
        "top": (slice(0, 6), slice(24, 40)),
        "bottom": (slice(HEIGHT - 6, HEIGHT), slice(24, 40)),
        "corner": (slice(0, 7), slice(0, 7)),
        "left+right": (slice(28, 36), slice(0, WIDTH)),
    }
    support[:, :, regions[case][0], regions[case][1]] = 1
    if case not in {"left", "corner", "left+right"}:
        real_valid[:, :, :, :4] = 0
    if case not in {"right", "left+right"}:
        fake_valid[:, :, :, -5:] = 0
    if case not in {"top", "corner"}:
        real_valid[:, :, :3, :] = 0
    if case != "bottom":
        fake_valid[:, :, -3:, :] = 0
    return real_valid, fake_valid, support


@pytest.mark.parametrize(
    "case", ["central", "left", "right", "top", "bottom", "corner", "left+right"]
)
def test_aligned_views_preserve_joint_pixels_and_all_contact_geometries(case: str) -> None:
    real, fake = _images()
    real_valid, fake_valid, support = _case_masks(case)
    result = prepare_aligned_discriminator_views(
        real,
        fake,
        real_valid,
        fake_valid,
        support,
        generator_support_mask=support,
    )
    expected_joint = real_valid.bool() & fake_valid.bool()
    assert torch.equal(result.joint_valid_mask.bool(), expected_joint)
    expanded = expected_joint.expand_as(real)
    assert torch.equal(result.real_discriminator_view[expanded], real[expanded])
    assert torch.equal(result.fake_discriminator_view[expanded], fake[expanded])
    assert torch.equal(
        result.real_discriminator_view[~expanded],
        result.fake_discriminator_view[~expanded],
    )
    assert bool(
        (
            result.real_discriminator_view[~expanded]
            == DISCRIMINATOR_INVALID_FILL_VALUE
        ).all()
    )
    assert torch.equal(result.discriminator_mask, support)


def test_fractional_validity_uses_inclusive_canonical_threshold() -> None:
    real, fake = _images()
    real_valid = torch.full((1, 1, HEIGHT, WIDTH), 0.49)
    fake_valid = torch.full_like(real_valid, 0.90)
    real_valid[:, :, 10:20, 10:20] = 0.50
    support = torch.zeros_like(real_valid)
    support[:, :, 12:18, 12:18] = 0.75
    result = prepare_aligned_discriminator_views(
        real,
        fake,
        real_valid,
        fake_valid,
        support,
        generator_support_mask=support,
    )
    expected = torch.zeros_like(real_valid, dtype=torch.bool)
    expected[:, :, 10:20, 10:20] = True
    assert torch.equal(result.joint_valid_mask.bool(), expected)
    assert set(result.discriminator_mask.unique().tolist()) <= {0.0, 1.0}


def test_support_halo_outside_validity_is_allowed_when_canonical_mask_is_contained() -> None:
    real, fake = _images()
    real_valid = fake_valid = torch.ones(1, 1, HEIGHT, WIDTH)
    real_valid = real_valid.clone()
    real_valid[:, :, :, :4] = 0
    canonical = torch.zeros_like(real_valid)
    canonical[:, :, 20:24, 8:12] = 1
    support = canonical.clone()
    support[:, :, 20:24, :2] = 1
    result = prepare_aligned_discriminator_views(
        real,
        fake,
        real_valid,
        fake_valid,
        canonical,
        generator_support_mask=support,
    )
    assert result.joint_valid_mask.shape == real_valid.shape


def test_canonical_mask_outside_joint_validity_is_rejected() -> None:
    real, fake = _images()
    real_valid = fake_valid = torch.ones(1, 1, HEIGHT, WIDTH)
    real_valid = real_valid.clone()
    real_valid[:, :, :, :4] = 0
    canonical = torch.zeros_like(real_valid)
    canonical[:, :, 20:24, :2] = 1
    with pytest.raises(
        ValueError, match="canonical discriminator mask lies outside joint validity"
    ):
        prepare_aligned_discriminator_views(
            real, fake, real_valid, fake_valid, canonical
        )


def test_generator_adversarial_gradient_survives_support_and_stops_in_padding() -> None:
    _, fake_base = _images()
    fake = fake_base.clone().requires_grad_(True)
    real = torch.zeros_like(fake)
    real_valid, fake_valid, support = _case_masks("central")
    result = prepare_aligned_discriminator_views(
        real,
        fake,
        real_valid,
        fake_valid,
        support,
        generator_support_mask=support,
    )
    discriminator = MaskConditionedPatchDiscriminator(
        base_channels=8, group_norm_groups=4, use_spectral_norm=True
    ).eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    logits = discriminator(
        result.fake_discriminator_view, result.discriminator_mask
    )
    weights = patch_logit_localization_weights(
        result.discriminator_mask, logits, localization_radius=3
    )
    localized_generator_adversarial_loss(logits, weights).total.backward()
    assert fake.grad is not None and bool(torch.isfinite(fake.grad).all())
    expanded_support = support.bool().expand_as(fake)
    assert bool((fake.grad[expanded_support] != 0).any())
    invalid = ~result.joint_valid_mask.bool().expand_as(fake)
    assert torch.equal(fake.grad[invalid], torch.zeros_like(fake.grad[invalid]))


def test_preparation_is_deterministic_and_does_not_mutate_inputs() -> None:
    real, fake = _images()
    real_valid, fake_valid, support = _case_masks("central")
    inputs = (real, fake, real_valid, fake_valid, support)
    originals = tuple(tensor.clone() for tensor in inputs)
    first = prepare_aligned_discriminator_views(
        *inputs, generator_support_mask=support
    )
    second = prepare_aligned_discriminator_views(
        *inputs, generator_support_mask=support
    )
    for name in (
        "real_discriminator_view",
        "fake_discriminator_view",
        "joint_valid_mask",
        "discriminator_mask",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert all(torch.equal(tensor, original) for tensor, original in zip(inputs, originals))


def test_training_sample_adapter_preserves_sample_and_checks_mask_equality() -> None:
    real, fake = (tensor[0] for tensor in _images())
    real_valid, fake_valid, support = (tensor[0] for tensor in _case_masks("central"))
    sample = GANTrainingSample(
        composite_image=fake,
        generator_mask=support,
        transformed_defect_alpha=support.clone(),
        fake_discriminator_mask=support.clone(),
        real_image=real,
        real_discriminator_mask=support.clone(),
        fake_valid_mask=fake_valid,
        real_valid_mask=real_valid,
        real_valid_coverage=real_valid.clone(),
        metadata={"sample": "unchanged"},
    )
    snapshots = {
        name: getattr(sample, name).clone()
        for name in (
            "composite_image",
            "generator_mask",
            "transformed_defect_alpha",
            "fake_discriminator_mask",
            "real_image",
            "real_discriminator_mask",
            "fake_valid_mask",
            "real_valid_mask",
            "real_valid_coverage",
        )
    }
    metadata = copy.deepcopy(sample.metadata)
    result = prepare_training_sample_discriminator_views(sample)
    assert result.real_discriminator_view.shape == (1, 3, HEIGHT, WIDTH)
    assert all(torch.equal(getattr(sample, name), value) for name, value in snapshots.items())
    assert sample.metadata == metadata
    divergent = GANTrainingSample(
        **{
            **sample.__dict__,
            "real_discriminator_mask": torch.zeros_like(sample.real_discriminator_mask),
        }
    )
    with pytest.raises(ValueError, match="real/fake discriminator masks differ"):
        prepare_training_sample_discriminator_views(divergent)


@pytest.mark.parametrize(
    ("source_id", "contacts", "horizontal_flip", "expected"),
    [
        ("train-10382", ContactSides(left=True), False, ContactSides(left=True)),
        ("train-10382", ContactSides(left=True), True, ContactSides(right=True)),
        ("train-11775", ContactSides(right=True), False, ContactSides(right=True)),
        ("train-11775", ContactSides(right=True), True, ContactSides(left=True)),
        (
            "train-11214",
            ContactSides(left=True, right=True),
            True,
            ContactSides(left=True, right=True),
        ),
        (
            "corner",
            ContactSides(top=True, left=True),
            True,
            ContactSides(top=True, right=True),
        ),
    ],
)
def test_continuous_real_validity_regressions_preserve_border_contacts(
    source_id: str,
    contacts: ContactSides,
    horizontal_flip: bool,
    expected: ContactSides,
) -> None:
    scale = 1.0 if contacts.left and contacts.right else 0.953
    sample = _continuous_validity_sample(
        source_id, contacts, scale=scale, horizontal_flip=horizontal_flip
    )
    details = sample["training_details"]
    alpha = details["transformed_defect_alpha"]
    coverage = details["transformed_real_valid_coverage"]
    real_valid = details["transformed_real_valid_region"]
    canonical = canonicalize_discriminator_mask(alpha.unsqueeze(0))[0]
    assert details["alpha_coverage_maximum_violation"] <= 1e-6
    assert bool((alpha <= coverage + 1e-6).all())
    assert not bool((canonical.bool() & ~real_valid.bool()).any())
    assert sample["provenance"]["target_contact_sides"] == expected.to_dict()
    transform = details["shared_spatial_transform"]
    assert transform["continuous_mask_validity_transform"] == {
        "grid": "shared",
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False,
        "validity_threshold": 1e-6,
        "threshold_policy": "strictly_greater_than",
    }


@pytest.mark.parametrize("scale", [0.9, 0.953, 1.1])
def test_continuous_scales_share_defect_and_validity_interpolation(scale: float) -> None:
    sample = _continuous_validity_sample(
        "train-11775", ContactSides(right=True), scale=scale
    )
    details = sample["training_details"]
    alpha = details["transformed_defect_alpha"]
    coverage = details["transformed_real_valid_coverage"]
    assert bool((alpha <= coverage + 1e-6).all())
    if scale != 1.0:
        assert bool(((coverage > 0) & (coverage < 1)).any())


def test_source_defect_must_be_subset_of_source_native_validity() -> None:
    source_valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
    source_valid[:, 8:56] = True
    mask = np.zeros_like(source_valid)
    mask[20:28, 4:10] = True
    source = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Source defect mask entered non-native"):
        construct_coarse_gan_input(
            source,
            mask,
            np.full_like(source, 100),
            np.ones_like(mask),
            seed=42,
            transform_settings={
                "horizontal_flip_probability": 0.0,
                "vertical_flip_probability": 0.0,
                "minimum_scale": 1.0,
                "maximum_scale": 1.0,
                "minimum_retained_area_fraction": 0.7,
                "feather_radius": 3,
                "non_border_native_margin": 4,
            },
            colour_settings={
                "enabled": False,
                "boundary_radius": 4,
                "minimum_gain": 0.8,
                "maximum_gain": 1.2,
                "maximum_absolute_offset": 0.15,
            },
            provenance_base=_provenance("invalid", ContactSides()),
            transform_parameters={
                "horizontal_flip": False,
                "vertical_flip": False,
                "scale": 1.0,
            },
            source_valid_region=source_valid,
            include_training_details=True,
        )


def test_support_only_non_border_regression_is_informational() -> None:
    sample = _continuous_validity_sample(
        "train-10352", ContactSides(), scale=0.939, horizontal_flip=True
    )
    details = sample["training_details"]
    real_valid = details["transformed_real_valid_region"].bool()
    canonical = canonicalize_discriminator_mask(
        details["transformed_defect_alpha"].unsqueeze(0)
    )[0].bool()
    generator_support = sample["feathered_support"].bool()
    assert not bool((canonical & ~real_valid).any())
    assert bool((generator_support & ~real_valid).any())


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda values: values.update(real_image=torch.zeros(3, HEIGHT, WIDTH)), "shape"),
        (
            lambda values: values.update(fake_image=torch.zeros(1, 3, 32, WIDTH)),
            "identical shapes",
        ),
        (lambda values: values.update(real_image=torch.zeros(1, 3, HEIGHT, WIDTH, dtype=torch.int64)), "floating"),
        (lambda values: values["real_image"].fill_(float("nan")), "finite"),
        (lambda values: values["fake_image"].fill_(1.1), r"\[-1,1\]"),
        (lambda values: values.update(real_valid_mask=torch.ones(1, 2, HEIGHT, WIDTH)), "shape"),
        (lambda values: values["fake_valid_mask"].fill_(float("inf")), "finite"),
        (lambda values: values["discriminator_mask"].fill_(-0.1), r"\[0,1\]"),
    ],
)
def test_invalid_inputs_are_rejected(mutator, message: str) -> None:
    real, fake = _images()
    valid = torch.ones(1, 1, HEIGHT, WIDTH)
    support = torch.zeros_like(valid)
    support[:, :, 28:36, 28:36] = 1
    values = {
        "real_image": real,
        "fake_image": fake,
        "real_valid_mask": valid.clone(),
        "fake_valid_mask": valid.clone(),
        "discriminator_mask": support,
    }
    mutator(values)
    with pytest.raises(ValueError, match=message):
        prepare_aligned_discriminator_views(**values)
