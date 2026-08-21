"""Mask-conditioned residual generator and PatchGAN architecture definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


ARCHITECTURE_VERSION = "g1_5a_identity_range_aware_residual_gan_v1"
RESIDUAL_SEMANTICS_VERSION = "g1_5a_directional_range_aware_residual_v1"


@dataclass(frozen=True)
class GANArchitectureConfig:
    architecture_version: str
    image_height: int
    image_width: int
    generator_base_channels: int
    generator_downsample_stages: int
    generator_residual_blocks: int
    group_norm_groups: int
    support_dilation_radius: int
    residual_scale: float
    discriminator_base_channels: int
    discriminator_spectral_norm: bool

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GANArchitectureConfig":
        required = tuple(cls.__dataclass_fields__)
        missing = [field for field in required if field not in values]
        if missing:
            raise ValueError(f"GAN architecture config is missing: {', '.join(missing)}")
        config = cls(**{field: values[field] for field in required})
        config.validate()
        return config

    def validate(self) -> None:
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"architecture_version must be {ARCHITECTURE_VERSION!r}"
            )
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Configured image dimensions must be positive")
        if self.image_height % 8 or self.image_width % 8:
            raise ValueError("Configured image height and width must be divisible by 8")
        if self.generator_base_channels <= 0 or self.discriminator_base_channels <= 0:
            raise ValueError("Base channel counts must be positive")
        if self.generator_downsample_stages != 3:
            raise ValueError("g1_1 requires exactly three generator downsample stages")
        if self.generator_residual_blocks <= 0:
            raise ValueError("generator_residual_blocks must be positive")
        if self.group_norm_groups <= 0:
            raise ValueError("group_norm_groups must be positive")
        generator_widths = [
            self.generator_base_channels * 2**stage
            for stage in range(self.generator_downsample_stages + 1)
        ]
        discriminator_widths = [
            self.discriminator_base_channels * 2**stage for stage in range(4)
        ]
        invalid_widths = [
            width
            for width in generator_widths + discriminator_widths[1:]
            if width % self.group_norm_groups
        ]
        if invalid_widths:
            raise ValueError("group_norm_groups must divide every normalized channel count")
        if self.support_dilation_radius < 0:
            raise ValueError("support_dilation_radius must be non-negative")
        if not 0 < self.residual_scale <= 1:
            raise ValueError("residual_scale must be in (0, 1]")
        if not self.discriminator_spectral_norm:
            raise ValueError("g1_1 requires discriminator spectral normalization")


def load_gan_architecture_config(path: Path | str) -> GANArchitectureConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("GAN architecture config must contain a JSON object")
    return GANArchitectureConfig.from_dict(values)


def _validate_conditioned_inputs(
    image: torch.Tensor,
    defect_mask: torch.Tensor,
    *,
    image_name: str,
) -> None:
    if image.ndim != 4:
        raise ValueError(f"{image_name} must have rank 4 [B,3,H,W]")
    if defect_mask.ndim != 4:
        raise ValueError("defect_mask must have rank 4 [B,1,H,W]")
    if image.shape[1] != 3:
        raise ValueError(f"{image_name} must have exactly 3 channels")
    if defect_mask.shape[1] != 1:
        raise ValueError("defect_mask must have exactly 1 channel")
    if image.shape[0] != defect_mask.shape[0] or image.shape[-2:] != defect_mask.shape[-2:]:
        raise ValueError(f"{image_name} and defect_mask must have matching batch and spatial dimensions")
    if image.shape[0] <= 0 or image.shape[-2] <= 0 or image.shape[-1] <= 0:
        raise ValueError("Input batch and spatial dimensions must be positive")
    if not image.is_floating_point():
        raise ValueError(f"{image_name} must be floating point")
    if defect_mask.dtype != torch.bool and not defect_mask.is_floating_point():
        raise ValueError("defect_mask must be bool or floating point")
    if image.shape[-2] % 8 or image.shape[-1] % 8:
        raise ValueError("Input height and width must be divisible by 8")
    if not bool(torch.isfinite(image).all()):
        raise ValueError(f"{image_name} must contain only finite values")
    if bool((image < -1).any()) or bool((image > 1).any()):
        raise ValueError(f"{image_name} values must be in [-1, 1]")
    if defect_mask.dtype != torch.bool:
        if not bool(torch.isfinite(defect_mask).all()):
            raise ValueError("defect_mask must contain only finite values")
        if bool((defect_mask < 0).any()) or bool((defect_mask > 1).any()):
            raise ValueError("defect_mask values must be in [0, 1]")


def _normalization(groups: int, channels: int) -> nn.GroupNorm:
    if channels % groups:
        raise ValueError("group_norm_groups must divide every normalized channel count")
    return nn.GroupNorm(groups, channels)


class _DownsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, groups: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, bias=False),
            _normalization(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            _normalization(groups, channels),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            _normalization(groups, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class _UpsampleSkipBlock(nn.Module):
    def __init__(
        self, input_channels: int, skip_channels: int, output_channels: int, groups: int
    ) -> None:
        super().__init__()
        combined_channels = input_channels + skip_channels
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(combined_channels, output_channels, kernel_size=3, bias=False),
            _normalization(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=False),
            _normalization(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((inputs, skip), dim=1))


@dataclass(frozen=True)
class MaskedResidualGeneratorOutput:
    refined_image: torch.Tensor
    raw_residual: torch.Tensor
    applied_residual: torch.Tensor
    support_mask: torch.Tensor


def range_aware_residual(
    image: torch.Tensor,
    raw_residual: torch.Tensor,
    maximum_absolute_delta: float,
) -> torch.Tensor:
    """Map an unconstrained residual to the image's available directional range."""
    if image.shape != raw_residual.shape:
        raise ValueError("image and raw_residual must have identical shapes")
    if not 0 < maximum_absolute_delta <= 1:
        raise ValueError("maximum_absolute_delta must be in (0, 1]")
    direction = torch.tanh(raw_residual)
    configured_cap = torch.full_like(image, maximum_absolute_delta)
    positive_cap = torch.minimum(configured_cap, 1.0 - image)
    negative_cap = torch.minimum(configured_cap, image + 1.0)
    return torch.where(
        direction >= 0,
        direction * positive_cap,
        direction * negative_cap,
    )


class MaskedResidualGenerator(nn.Module):
    """U-Net residual refiner that is bit-exact outside dilated defect support."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        downsample_stages: int = 3,
        residual_blocks: int = 4,
        group_norm_groups: int = 8,
        support_dilation_radius: int = 12,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if downsample_stages != 3:
            raise ValueError("MaskedResidualGenerator requires exactly 3 downsample stages")
        if residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive")
        if support_dilation_radius < 0:
            raise ValueError("support_dilation_radius must be non-negative")
        if not 0 < residual_scale <= 1:
            raise ValueError("residual_scale must be in (0, 1]")
        widths = [base_channels * 2**stage for stage in range(downsample_stages + 1)]
        for width in widths:
            _normalization(group_norm_groups, width)
        self.support_dilation_radius = int(support_dilation_radius)
        self.residual_scale = float(residual_scale)
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, widths[0], kernel_size=7, bias=False),
            _normalization(group_norm_groups, widths[0]),
            nn.SiLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                _DownsampleBlock(widths[index], widths[index + 1], group_norm_groups)
                for index in range(downsample_stages)
            ]
        )
        self.residual_blocks = nn.Sequential(
            *[
                _ResidualBlock(widths[-1], group_norm_groups)
                for _ in range(residual_blocks)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                _UpsampleSkipBlock(
                    widths[index + 1], widths[index], widths[index], group_norm_groups
                )
                for index in reversed(range(downsample_stages))
            ]
        )
        self.output_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(widths[0], 3, kernel_size=7),
        )
        final_convolution = self.output_head[-1]
        if not isinstance(final_convolution, nn.Conv2d):
            raise RuntimeError("Generator output head must end with a convolution")
        nn.init.zeros_(final_convolution.weight)
        nn.init.zeros_(final_convolution.bias)
        self.register_buffer(
            "_residual_semantics_marker",
            torch.tensor([1], dtype=torch.int32),
            persistent=True,
        )

    def _support_mask(self, defect_mask: torch.Tensor) -> torch.Tensor:
        binary = defect_mask.bool()
        radius = self.support_dilation_radius
        if radius == 0:
            return binary
        return F.max_pool2d(
            binary.float(), kernel_size=2 * radius + 1, stride=1, padding=radius
        ).bool()

    def forward(
        self, composite_image: torch.Tensor, defect_mask: torch.Tensor
    ) -> MaskedResidualGeneratorOutput:
        _validate_conditioned_inputs(
            composite_image, defect_mask, image_name="composite_image"
        )
        mask = defect_mask.to(device=composite_image.device, dtype=composite_image.dtype)
        features = [self.stem(torch.cat((composite_image, mask), dim=1))]
        for block in self.down_blocks:
            features.append(block(features[-1]))
        decoded = self.residual_blocks(features[-1])
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            decoded = block(decoded, skip)
        raw_residual = self.output_head(decoded)
        support = self._support_mask(defect_mask).to(device=composite_image.device)
        directional_residual = range_aware_residual(
            composite_image,
            raw_residual,
            self.residual_scale,
        )
        expanded_support = support.expand_as(composite_image)
        applied_residual = torch.where(
            expanded_support,
            directional_residual,
            torch.zeros_like(directional_residual),
        )
        candidate = composite_image + directional_residual
        refined = torch.where(expanded_support, candidate, composite_image)
        return MaskedResidualGeneratorOutput(
            refined_image=refined,
            raw_residual=raw_residual,
            applied_residual=applied_residual,
            support_mask=support,
        )


class MaskConditionedPatchDiscriminator(nn.Module):
    """Spectrally normalized conditional PatchGAN returning raw patch logits."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        group_norm_groups: int = 8,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if not use_spectral_norm:
            raise ValueError("g1_1 discriminator requires spectral normalization")

        def convolution(
            input_channels: int, output_channels: int, stride: int
        ) -> nn.Conv2d:
            layer = nn.Conv2d(
                input_channels, output_channels, kernel_size=4, stride=stride, padding=1
            )
            return spectral_norm(layer)

        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers: list[nn.Module] = [convolution(4, widths[0], 2), nn.LeakyReLU(0.2, True)]
        for input_channels, output_channels, stride in (
            (widths[0], widths[1], 2),
            (widths[1], widths[2], 2),
            (widths[2], widths[3], 1),
        ):
            layers.extend(
                [
                    convolution(input_channels, output_channels, stride),
                    _normalization(group_norm_groups, output_channels),
                    nn.LeakyReLU(0.2, True),
                ]
            )
        layers.append(convolution(widths[3], 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, defect_mask: torch.Tensor) -> torch.Tensor:
        _validate_conditioned_inputs(image, defect_mask, image_name="image")
        if image.shape[-2] < 24 or image.shape[-1] < 24:
            raise ValueError("Discriminator input height and width must each be at least 24")
        mask = defect_mask.to(device=image.device, dtype=image.dtype)
        return self.layers(torch.cat((image, mask), dim=1))


def build_gan_models(
    config: GANArchitectureConfig,
) -> tuple[MaskedResidualGenerator, MaskConditionedPatchDiscriminator]:
    config.validate()
    generator = MaskedResidualGenerator(
        base_channels=config.generator_base_channels,
        downsample_stages=config.generator_downsample_stages,
        residual_blocks=config.generator_residual_blocks,
        group_norm_groups=config.group_norm_groups,
        support_dilation_radius=config.support_dilation_radius,
        residual_scale=config.residual_scale,
    )
    discriminator = MaskConditionedPatchDiscriminator(
        base_channels=config.discriminator_base_channels,
        group_norm_groups=config.group_norm_groups,
        use_spectral_norm=config.discriminator_spectral_norm,
    )
    return generator, discriminator
