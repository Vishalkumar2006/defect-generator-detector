"""Connected-component-aware 256x512 native-window planning."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from defectgen.data.geometry import BoundingBox, bounding_box
from defectgen.data.patches import Padding, reflection_pad


@dataclass(frozen=True)
class ContactSides:
    top: bool = False
    bottom: bool = False
    left: bool = False
    right: bool = False

    @property
    def any(self) -> bool:
        return self.top or self.bottom or self.left or self.right

    @property
    def combination(self) -> str:
        active = [
            side for side in ("top", "bottom", "left", "right") if getattr(self, side)
        ]
        return "+".join(active) if active else "none"

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def transformed(self, *, horizontal_flip: bool, vertical_flip: bool) -> "ContactSides":
        return ContactSides(
            top=self.bottom if vertical_flip else self.top,
            bottom=self.top if vertical_flip else self.bottom,
            left=self.right if horizontal_flip else self.left,
            right=self.left if horizontal_flip else self.right,
        )

    @classmethod
    def from_dict(cls, value: dict[str, bool]) -> "ContactSides":
        return cls(**{side: bool(value.get(side, False)) for side in ("top", "bottom", "left", "right")})


@dataclass(frozen=True)
class MaskComponent:
    component_id: int
    mask: np.ndarray
    bounding_box: BoundingBox
    positive_pixels: int
    contact_sides: ContactSides

    @property
    def touches_native_border(self) -> bool:
        return self.contact_sides.any


@dataclass(frozen=True)
class ComponentWindow:
    top: int
    left: int
    width: int
    height: int
    partial_component: bool
    coverage_fraction: float
    positive_pixels: int
    source_contact_sides: ContactSides

    @property
    def touches_native_border(self) -> bool:
        return self.source_contact_sides.any

    def to_dict(self) -> dict:
        value = asdict(self)
        value["touches_native_border"] = self.touches_native_border
        return value


@dataclass(frozen=True)
class WindowPlan:
    windows: tuple[ComponentWindow, ...]
    rejected_reasons: tuple[str, ...]


def connected_components(mask: np.ndarray) -> list[MaskComponent]:
    if mask.ndim != 2:
        raise ValueError("Connected-component mask must be two-dimensional")
    binary = mask.astype(bool, copy=False)
    visited = np.zeros_like(binary)
    components: list[MaskComponent] = []
    height, width = binary.shape
    for initial_y, initial_x in np.argwhere(binary):
        y, x = int(initial_y), int(initial_x)
        if visited[y, x]:
            continue
        component_mask = np.zeros_like(binary)
        queue = deque([(y, x)])
        visited[y, x] = True
        while queue:
            current_y, current_x = queue.popleft()
            component_mask[current_y, current_x] = True
            for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                    if binary[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
        box = bounding_box(component_mask)
        assert box is not None
        components.append(
            MaskComponent(
                component_id=len(components),
                mask=component_mask,
                bounding_box=box,
                positive_pixels=int(component_mask.sum()),
                contact_sides=ContactSides(
                    top=bool(component_mask[0].any()),
                    bottom=bool(component_mask[-1].any()),
                    left=bool(component_mask[:, 0].any()),
                    right=bool(component_mask[:, -1].any()),
                ),
            )
        )
    return components


def _axis_starts(
    length: int,
    patch: int,
    minimum: int,
    maximum: int,
    context: int,
    overlap_fraction: float,
) -> list[int]:
    if length <= patch:
        return [0]
    max_start = length - patch
    desired_min = max(0, minimum - context)
    desired_max = min(max_start, maximum + context - patch + 1)
    if maximum - minimum + 1 + 2 * context <= patch:
        center = (minimum + maximum) // 2
        return [min(max(center - patch // 2, 0), max_start)]
    first = min(desired_min, max_start)
    last = max(first, desired_max)
    step = max(1, int(round(patch * (1.0 - overlap_fraction))))
    starts = list(range(first, last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    return starts


def plan_component_windows(
    component: MaskComponent,
    image_shape: tuple[int, int],
    *,
    patch_size: tuple[int, int] = (256, 512),
    context_margin: int = 24,
    overlap_fraction: float = 0.5,
    minimum_positive_pixels: int = 8,
    minimum_component_coverage: float = 0.05,
) -> WindowPlan:
    patch_width, patch_height = patch_size
    height, width = image_shape
    if patch_width <= 0 or patch_height <= 0 or height <= 0 or width <= 0:
        raise ValueError("Patch and image dimensions must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0,1)")
    if minimum_positive_pixels <= 0 or not 0 <= minimum_component_coverage <= 1:
        raise ValueError("Positive-pixel minimum must be positive and coverage must be in [0,1]")
    if component.positive_pixels < minimum_positive_pixels:
        return WindowPlan((), ("component_below_minimum_positive_pixels",))
    box = component.bounding_box
    lefts = _axis_starts(
        width, patch_width, box.x_min, box.x_max, context_margin, overlap_fraction
    )
    tops = _axis_starts(
        height, patch_height, box.y_min, box.y_max, context_margin, overlap_fraction
    )
    windows: list[ComponentWindow] = []
    rejected: list[str] = []
    for top in tops:
        for left in lefts:
            selected = component.mask[top : min(top + patch_height, height), left : min(left + patch_width, width)]
            positive = int(selected.sum())
            coverage = positive / component.positive_pixels
            if positive < minimum_positive_pixels:
                rejected.append("window_below_minimum_positive_pixels")
                continue
            if coverage < minimum_component_coverage:
                rejected.append("window_below_minimum_component_coverage")
                continue
            windows.append(
                ComponentWindow(
                    top=top,
                    left=left,
                    width=patch_width,
                    height=patch_height,
                    partial_component=positive < component.positive_pixels,
                    coverage_fraction=coverage,
                    positive_pixels=positive,
                    source_contact_sides=component.contact_sides,
                )
            )
    if not windows:
        rejected.append("no_usable_component_window")
    unique_rejections = tuple(dict.fromkeys(rejected))
    return WindowPlan(tuple(windows), unique_rejections)


def deterministic_component_windows(
    mask: np.ndarray,
    component_id: int,
    **kwargs,
) -> tuple[ComponentWindow, ...]:
    components = connected_components(mask)
    if not 0 <= component_id < len(components):
        raise ValueError(f"Unknown component ID: {component_id}")
    return plan_component_windows(components[component_id], mask.shape, **kwargs).windows


def extract_native_window(
    image: np.ndarray,
    mask: np.ndarray,
    window: ComponentWindow | tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("Expected aligned HxWx3 image and HxW mask")
    if isinstance(window, ComponentWindow):
        top, left, width, height = window.top, window.left, window.width, window.height
    else:
        top, left, width, height = window
    if top < 0 or left < 0 or width <= 0 or height <= 0:
        raise ValueError("Window coordinates and dimensions are invalid")
    source = image[top : min(top + height, len(image)), left : min(left + width, image.shape[1])]
    source_mask = mask[top : min(top + height, len(mask)), left : min(left + width, mask.shape[1])]
    output_image = np.zeros((height, width, 3), dtype=image.dtype)
    output_mask = np.zeros((height, width), dtype=bool)
    valid = np.zeros((height, width), dtype=bool)
    source_height, source_width = source.shape[:2]
    pad_top = (height - source_height) // 2
    pad_left = (width - source_width) // 2
    output_image[
        pad_top : pad_top + source_height, pad_left : pad_left + source_width
    ] = source
    output_mask[
        pad_top : pad_top + source_height, pad_left : pad_left + source_width
    ] = source_mask.astype(bool)
    valid[pad_top : pad_top + source_height, pad_left : pad_left + source_width] = True
    if source_height and source_width and (source_height < height or source_width < width):
        pad_bottom = height - source_height - pad_top
        pad_right = width - source_width - pad_left
        output_image = reflection_pad(
            source,
            Padding(
                left=pad_left,
                top=pad_top,
                right=pad_right,
                bottom=pad_bottom,
            ),
        )
    return output_image, output_mask, valid
