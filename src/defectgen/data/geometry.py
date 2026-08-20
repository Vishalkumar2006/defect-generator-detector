"""Binary-mask geometry measurements used for KSDD2 patch design."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


PATCH_CANDIDATES = ((256, 256), (256, 384), (256, 512))
CONTEXT_MARGINS = (0, 16, 32)


@dataclass(frozen=True)
class BoundingBox:
    """Inclusive pixel-coordinate bounding box."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def height(self) -> int:
        return self.y_max - self.y_min + 1


def bounding_box(mask: np.ndarray) -> BoundingBox | None:
    if mask.ndim != 2:
        raise ValueError(f"Mask must be two-dimensional, got {mask.shape}")
    y_coordinates, x_coordinates = np.nonzero(mask)
    if len(x_coordinates) == 0:
        return None
    return BoundingBox(
        x_min=int(x_coordinates.min()),
        y_min=int(y_coordinates.min()),
        x_max=int(x_coordinates.max()),
        y_max=int(y_coordinates.max()),
    )


def connected_component_count(mask: np.ndarray) -> int:
    """Count 8-connected foreground components in a binary mask."""
    if mask.ndim != 2:
        raise ValueError(f"Mask must be two-dimensional, got {mask.shape}")
    foreground = mask.astype(bool, copy=False)
    visited = np.zeros(foreground.shape, dtype=bool)
    height, width = foreground.shape
    components = 0
    for initial_y, initial_x in np.argwhere(foreground):
        y = int(initial_y)
        x = int(initial_x)
        if visited[y, x]:
            continue
        components += 1
        visited[y, x] = True
        queue = deque([(y, x)])
        while queue:
            current_y, current_x = queue.popleft()
            for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                    if foreground[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
    return components


def touches_border(mask: np.ndarray) -> bool:
    if mask.ndim != 2 or not mask.size:
        raise ValueError(f"Mask must be a non-empty two-dimensional array, got {mask.shape}")
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def bbox_fits_patch(box: BoundingBox, patch_size: tuple[int, int], context_margin: int = 0) -> bool:
    """Return whether the complete box and symmetric context fit width × height."""
    patch_width, patch_height = patch_size
    if patch_width <= 0 or patch_height <= 0 or context_margin < 0:
        raise ValueError("Patch dimensions must be positive and context margin non-negative")
    return box.width + 2 * context_margin <= patch_width and box.height + 2 * context_margin <= patch_height

