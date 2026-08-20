"""A GroupNorm U-Net with learned downsampling and resize-convolution decoding."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.block = ConvBlock(output_channels, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(self.downsample(inputs))


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.block = ConvBlock(output_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        inputs = self.project(inputs)
        return self.block(torch.cat((inputs, skip), dim=1))


class UNet(nn.Module):
    """Five-level full-resolution U-Net returning one-channel logits."""

    def __init__(self, input_channels: int = 3, output_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16, base_channels * 16]
        self.stem = ConvBlock(input_channels, widths[0])
        self.down_blocks = nn.ModuleList(
            [DownBlock(widths[index], widths[index + 1]) for index in range(len(widths) - 1)]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(widths[index + 1], widths[index], widths[index])
                for index in reversed(range(len(widths) - 1))
            ]
        )
        self.output = nn.Conv2d(widths[0], output_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = [self.stem(inputs)]
        for block in self.down_blocks:
            features.append(block(features[-1]))
        decoded = features[-1]
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            decoded = block(decoded, skip)
        return self.output(decoded)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

