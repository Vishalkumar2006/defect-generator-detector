"""Auditable G1.4 one-step GAN training mechanics and scale calibration."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import nn

from defectgen.gan.discriminator_views import prepare_aligned_discriminator_views
from defectgen.gan.training_pairs import GANTrainingSample
from defectgen.models.gan import (
    MaskConditionedPatchDiscriminator,
    MaskedResidualGenerator,
)
from defectgen.training.gan_losses import (
    GANLossConfig,
    GeneratorLossWeights,
    aggregate_generator_losses,
    boundary_seam_loss,
    localized_discriminator_hinge_loss,
    localized_generator_adversarial_loss,
    localized_r1_gradient_penalty,
    masked_total_variation_loss,
    patch_logit_localization_weights,
    support_normalized_change_loss,
)


TRAINER_VERSION = "g1_4_gan_one_step_v1"
PRECISION_MODES = {"fp32", "bf16", "fp16"}


@dataclass(frozen=True)
class GANOptimizerConfig:
    name: str
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GANOptimizerConfig":
        normalized = dict(values)
        normalized["betas"] = tuple(values.get("betas", ()))
        try:
            result = cls(**normalized)
        except TypeError as error:
            raise ValueError(f"Invalid optimizer configuration: {error}") from error
        result.validate()
        return result

    def validate(self) -> None:
        if self.name != "Adam":
            raise ValueError("G1.4 requires the Adam optimizer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Optimizer learning_rate must be finite and positive")
        if len(self.betas) != 2 or not all(0 <= value < 1 for value in self.betas):
            raise ValueError("Optimizer betas must contain two values in [0,1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Optimizer weight_decay must be finite and non-negative")


@dataclass(frozen=True)
class GANTrainerConfig:
    trainer_version: str
    training_pair_config_path: str
    architecture_config_path: str
    loss_config_path: str
    report_directory: str
    seed: int
    batch_size: int
    generator_optimizer: GANOptimizerConfig
    discriminator_optimizer: GANOptimizerConfig
    generator_gradient_clip_max_norm: float
    discriminator_gradient_clip_max_norm: float
    cuda_precision: str
    cpu_precision: str
    r1_gamma: float
    r1_interval: int
    deterministic_audit_batches: int
    loss_coefficients_provisional: bool
    one_step_generator_loss_weights: GeneratorLossWeights
    suggestion_coefficient_minimum: float
    suggestion_coefficient_maximum: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GANTrainerConfig":
        required = tuple(field.name for field in fields(cls))
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"GAN trainer config is missing: {', '.join(missing)}")
        normalized = dict(values)
        normalized["generator_optimizer"] = GANOptimizerConfig.from_dict(
            values["generator_optimizer"]
        )
        normalized["discriminator_optimizer"] = GANOptimizerConfig.from_dict(
            values["discriminator_optimizer"]
        )
        try:
            normalized["one_step_generator_loss_weights"] = GeneratorLossWeights(
                **values["one_step_generator_loss_weights"]
            )
        except TypeError as error:
            raise ValueError(f"Invalid one-step loss weights: {error}") from error
        config = cls(**{name: normalized[name] for name in required})
        config.validate()
        return config

    def validate(self) -> None:
        if self.trainer_version != TRAINER_VERSION:
            raise ValueError(f"trainer_version must be {TRAINER_VERSION!r}")
        if self.seed < 0 or self.batch_size <= 0:
            raise ValueError("seed must be non-negative and batch_size must be positive")
        self.generator_optimizer.validate()
        self.discriminator_optimizer.validate()
        for name, value in (
            ("generator_gradient_clip_max_norm", self.generator_gradient_clip_max_norm),
            (
                "discriminator_gradient_clip_max_norm",
                self.discriminator_gradient_clip_max_norm,
            ),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cuda_precision not in PRECISION_MODES:
            raise ValueError("cuda_precision must be fp32, bf16, or fp16")
        if self.cpu_precision != "fp32":
            raise ValueError("G1.4 CPU fallback must use fp32")
        if not math.isfinite(self.r1_gamma) or self.r1_gamma < 0:
            raise ValueError("r1_gamma must be finite and non-negative")
        if self.r1_interval <= 0 or self.deterministic_audit_batches <= 0:
            raise ValueError("R1 interval and audit batch count must be positive")
        if self.loss_coefficients_provisional is not True:
            raise ValueError("G1.4 loss coefficients must remain explicitly provisional")
        self.one_step_generator_loss_weights.validate()
        if not 0 < self.suggestion_coefficient_minimum <= self.suggestion_coefficient_maximum:
            raise ValueError("Invalid coefficient suggestion clamp")


def load_gan_trainer_config(path: Path | str) -> GANTrainerConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("GAN trainer config must contain a JSON object")
    return GANTrainerConfig.from_dict(values)


@dataclass(frozen=True)
class GANTrainingBatch:
    composite_image: torch.Tensor
    generator_mask: torch.Tensor
    transformed_defect_alpha: torch.Tensor
    fake_discriminator_mask: torch.Tensor
    real_image: torch.Tensor
    real_discriminator_mask: torch.Tensor
    fake_valid_mask: torch.Tensor
    real_valid_mask: torch.Tensor
    real_valid_coverage: torch.Tensor
    metadata: tuple[dict[str, Any], ...]

    @property
    def batch_size(self) -> int:
        return int(self.composite_image.shape[0])

    @property
    def batch_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{item['split']}:{item['sample_index']}:{item['template_id']}"
            for item in self.metadata
        )

    def to(self, device: torch.device | str) -> "GANTrainingBatch":
        selected = torch.device(device)
        values = {
            field.name: getattr(self, field.name).to(selected, non_blocking=True)
            for field in fields(self)
            if field.name != "metadata"
        }
        return GANTrainingBatch(**values, metadata=self.metadata)


def collate_gan_training_samples(
    samples: Sequence[GANTrainingSample],
) -> GANTrainingBatch:
    if not samples:
        raise ValueError("Cannot collate an empty GAN training batch")
    tensor_names = tuple(
        field.name for field in fields(GANTrainingSample) if field.name != "metadata"
    )
    values = {
        name: torch.stack([getattr(sample, name) for sample in samples], dim=0)
        for name in tensor_names
    }
    splits = {sample.metadata.get("split") for sample in samples}
    if len(splits) != 1 or splits.pop() not in {"train", "monitor"}:
        raise ValueError("A GAN batch must contain exactly one recognized split")
    if not torch.equal(values["real_discriminator_mask"], values["fake_discriminator_mask"]):
        raise ValueError("Collated real/fake discriminator masks differ")
    return GANTrainingBatch(
        **values,
        metadata=tuple(dict(sample.metadata) for sample in samples),
    )


class GANTrainingNumericalError(RuntimeError):
    """Raised before an optimizer mutation when a G1.4 numerical guard fails."""


def resolve_precision(config: GANTrainerConfig, device: torch.device) -> str:
    if device.type == "cpu":
        return config.cpu_precision
    if device.type != "cuda":
        raise ValueError(f"Unsupported GAN trainer device type: {device.type}")
    selected = config.cuda_precision
    if selected == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured CUDA bf16 is not explicitly supported")
    return selected


def precision_autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def parameter_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def module_state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor_value in sorted(module.state_dict().items()):
        tensor = tensor_value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tree_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_tree_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_tree_finite(item) for item in value)
    return True


def optimizer_state_is_finite(optimizer: torch.optim.Optimizer) -> bool:
    return _tree_finite(optimizer.state_dict())


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def _autograd_gradient_norm(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    if not squares:
        return 0.0
    norm = torch.stack(squares).sum().sqrt()
    return float(norm)


def _loss_values(
    components: dict[str, torch.Tensor],
    *,
    batch_ids: tuple[str, ...] | None = None,
    label: str = "GAN",
) -> dict[str, float]:
    values = {name: float(value.detach().float()) for name, value in components.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise GANTrainingNumericalError(
            f"{label} non-finite forward loss components={values}; "
            f"batch_ids={batch_ids}"
        )
    return values


def _logit_statistics(logits: torch.Tensor) -> dict[str, float]:
    value = logits.detach().float()
    return {
        "mean": float(value.mean()),
        "standard_deviation": float(value.std(unbiased=False)),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
    }


def _weighted_sign_accuracy(
    logits: torch.Tensor, weights: torch.Tensor, *, positive: bool
) -> float:
    active = weights > 0
    correct = logits > 0 if positive else logits < 0
    return float(correct[active].float().mean())


def canonical_adversarial_gradient_telemetry(
    adversarial_image_gradient: torch.Tensor,
    canonical_mask: torch.Tensor,
) -> dict[str, int | float]:
    """Measure canonical gradient coverage using finite RGB vectors per pixel."""
    if adversarial_image_gradient.ndim != 4 or adversarial_image_gradient.shape[1] != 3:
        raise ValueError("adversarial_image_gradient must have shape [B, 3, H, W]")
    if canonical_mask.ndim != 4 or canonical_mask.shape[1] != 1:
        raise ValueError("canonical_mask must have shape [B, 1, H, W]")
    if (
        adversarial_image_gradient.shape[0] != canonical_mask.shape[0]
        or adversarial_image_gradient.shape[2:] != canonical_mask.shape[2:]
    ):
        raise ValueError("adversarial_image_gradient and canonical_mask must align")

    canonical_vectors = adversarial_image_gradient.permute(0, 2, 3, 1)[
        canonical_mask[:, 0].bool()
    ]
    finite_components = torch.isfinite(canonical_vectors)
    active_components = finite_components & (canonical_vectors != 0)
    finite_vectors = finite_components.all(dim=1)
    active_vectors = finite_vectors & active_components.any(dim=1)

    active_pixel_count = int(active_vectors.sum().item())
    total_pixel_count = int(canonical_vectors.shape[0])
    active_channel_count = int(active_components.sum().item())
    total_channel_count = int(canonical_vectors.numel())
    nonfinite_channel_count = int((~finite_components).sum().item())
    coverage = (
        active_pixel_count / total_pixel_count if total_pixel_count else 0.0
    )
    return {
        # The unqualified counts retain their established API names, but now
        # intentionally count RGB vectors (pixels), not scalar components.
        "canonical_defect_gradient_active_count": active_pixel_count,
        "canonical_defect_gradient_total_count": total_pixel_count,
        "canonical_defect_gradient_active_pixel_count": active_pixel_count,
        "canonical_defect_gradient_total_pixel_count": total_pixel_count,
        "canonical_defect_gradient_active_channel_count": active_channel_count,
        "canonical_defect_gradient_total_channel_count": total_channel_count,
        "canonical_defect_gradient_nonfinite_channel_count": (
            nonfinite_channel_count
        ),
        "canonical_defect_gradient_coverage": coverage,
    }


@contextmanager
def _temporarily_disable_parameter_gradients(module: nn.Module) -> Iterator[None]:
    parameters = list(module.parameters())
    states = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(parameters, states):
            parameter.requires_grad_(state)


@contextmanager
def _temporary_eval(*modules: nn.Module) -> Iterator[None]:
    states = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, state in zip(modules, states):
            module.train(state)


class GANOneStepTrainer:
    """Execute isolated discriminator and generator updates with audit telemetry."""

    def __init__(
        self,
        generator: MaskedResidualGenerator,
        discriminator: MaskConditionedPatchDiscriminator,
        config: GANTrainerConfig,
        loss_config: GANLossConfig,
        *,
        device: torch.device | str,
    ) -> None:
        config.validate()
        loss_config.validate()
        self.config = config
        self.loss_config = loss_config
        self.device = torch.device(device)
        self.precision = resolve_precision(config, self.device)
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.generator_optimizer = self._build_optimizer(
            self.generator.parameters(), config.generator_optimizer
        )
        self.discriminator_optimizer = self._build_optimizer(
            self.discriminator.parameters(), config.discriminator_optimizer
        )
        self.generator_scaler = self._build_scaler()
        self.discriminator_scaler = self._build_scaler()
        self.generator_optimizer_steps = 0
        self.discriminator_optimizer_steps = 0

    def _build_optimizer(
        self, parameters: Iterable[nn.Parameter], selected: GANOptimizerConfig
    ) -> torch.optim.Adam:
        return torch.optim.Adam(
            parameters,
            lr=selected.learning_rate,
            betas=selected.betas,
            weight_decay=selected.weight_decay,
        )

    def _build_scaler(self):
        if self.precision != "fp16":
            return None
        if self.device.type != "cuda":
            raise RuntimeError("fp16 requires CUDA GradScaler")
        return torch.amp.GradScaler("cuda")

    def _require_train_batch(self, batch: GANTrainingBatch) -> GANTrainingBatch:
        if any(item.get("split") != "train" for item in batch.metadata):
            raise ValueError("Optimizer steps require only the internal GAN train split")
        return batch.to(self.device)

    def r1_is_scheduled(self, global_step: int) -> bool:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        return (global_step + 1) % self.config.r1_interval == 0

    def _backward_clip_step(
        self,
        *,
        loss: torch.Tensor,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler,
        maximum_norm: float,
        batch_ids: tuple[str, ...],
        label: str,
    ) -> dict[str, Any]:
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not bool(torch.isfinite(loss.detach().float())):
            optimizer.zero_grad(set_to_none=True)
            raise GANTrainingNumericalError(
                f"{label} non-finite loss before backward; batch_ids={batch_ids}"
            )
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            optimizer.zero_grad(set_to_none=True)
            raise GANTrainingNumericalError(
                f"{label} non-finite gradient before optimizer step; batch_ids={batch_ids}"
            )
        pre = _gradient_norm(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, maximum_norm)
        post = _gradient_norm(parameters)
        if not math.isfinite(post):
            optimizer.zero_grad(set_to_none=True)
            raise GANTrainingNumericalError(
                f"{label} non-finite post-clipping gradient; batch_ids={batch_ids}"
            )
        scale_before = float(scaler.get_scale()) if scaler is not None else None
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        scale_after = float(scaler.get_scale()) if scaler is not None else None
        if scaler is not None and scale_after is not None and scale_after < scale_before:
            raise GANTrainingNumericalError(
                f"{label} GradScaler overflow despite finite pre-step gradients"
            )
        return {
            "pre_clipping_norm": pre,
            "post_clipping_norm": post,
            "maximum_norm": maximum_norm,
            "scale_before": scale_before,
            "scale_after": scale_after,
        }

    def discriminator_step(
        self,
        batch: GANTrainingBatch,
        *,
        global_step: int,
        verify_parameter_isolation: bool = True,
    ) -> dict[str, Any]:
        selected = self._require_train_batch(batch)
        generator_parameters_before = (
            parameter_hash(self.generator) if verify_parameter_isolation else None
        )
        generator_state_before = (
            module_state_hash(self.generator) if verify_parameter_isolation else None
        )
        discriminator_parameters_before = (
            parameter_hash(self.discriminator) if verify_parameter_isolation else None
        )
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), precision_autocast(self.device, self.precision):
            generated = self.generator(selected.composite_image, selected.generator_mask)
        fake_detached = not generated.refined_image.requires_grad
        aligned = prepare_aligned_discriminator_views(
            selected.real_image,
            generated.refined_image.detach(),
            selected.real_valid_mask,
            selected.fake_valid_mask,
            selected.fake_discriminator_mask,
            generator_support_mask=generated.support_mask.float(),
            discriminator_mask_threshold=self.loss_config.canonical_mask_threshold,
        )
        with precision_autocast(self.device, self.precision):
            real_logits = self.discriminator(
                aligned.real_discriminator_view, aligned.discriminator_mask
            )
            fake_logits = self.discriminator(
                aligned.fake_discriminator_view, aligned.discriminator_mask
            )
        if not bool(torch.isfinite(real_logits).all()) or not bool(
            torch.isfinite(fake_logits).all()
        ):
            raise GANTrainingNumericalError(
                "discriminator non-finite logits before loss; "
                f"batch_ids={selected.batch_ids}"
            )
        real_weights = patch_logit_localization_weights(
            aligned.discriminator_mask,
            real_logits.float(),
            localization_radius=self.loss_config.localization_radius,
            mask_threshold=self.loss_config.canonical_mask_threshold,
        )
        fake_weights = real_weights
        hinge = localized_discriminator_hinge_loss(
            real_logits.float(), fake_logits.float(), real_weights, fake_weights
        )
        scheduled = self.r1_is_scheduled(global_step)
        raw_r1 = torch.zeros((), device=self.device, dtype=torch.float32)
        if scheduled:
            with torch.autocast(device_type=self.device.type, enabled=False):
                raw_r1 = localized_r1_gradient_penalty(
                    self.discriminator,
                    aligned.real_discriminator_view.float(),
                    aligned.discriminator_mask.float(),
                    localization_radius=self.loss_config.localization_radius,
                    mask_threshold=self.loss_config.canonical_mask_threshold,
                ).float()
        lazy_multiplier = self.config.r1_gamma * self.config.r1_interval / 2.0
        scaled_r1 = raw_r1 * lazy_multiplier if scheduled else raw_r1
        total = hinge.total.float() + scaled_r1
        values = _loss_values(
            {
                "real_hinge": hinge.real.float(),
                "fake_hinge": hinge.fake.float(),
                "total_hinge": hinge.total.float(),
                "raw_r1": raw_r1,
                "scaled_r1": scaled_r1,
                "total": total,
            },
            batch_ids=selected.batch_ids,
            label="discriminator",
        )
        clipping = self._backward_clip_step(
            loss=total,
            module=self.discriminator,
            optimizer=self.discriminator_optimizer,
            scaler=self.discriminator_scaler,
            maximum_norm=self.config.discriminator_gradient_clip_max_norm,
            batch_ids=selected.batch_ids,
            label="discriminator",
        )
        self.discriminator_optimizer_steps += 1
        generator_parameters_after = (
            parameter_hash(self.generator) if verify_parameter_isolation else None
        )
        generator_state_after = (
            module_state_hash(self.generator) if verify_parameter_isolation else None
        )
        discriminator_parameters_after = (
            parameter_hash(self.discriminator) if verify_parameter_isolation else None
        )
        if verify_parameter_isolation and (
            generator_parameters_before != generator_parameters_after
            or generator_state_before != generator_state_after
        ):
            raise RuntimeError("Discriminator step mutated generator parameters or buffers")
        if verify_parameter_isolation and discriminator_parameters_before == discriminator_parameters_after:
            raise RuntimeError("Discriminator optimizer step changed no parameters")
        if any(parameter.grad is not None for parameter in self.generator.parameters()):
            raise RuntimeError("Generator gradients were constructed during discriminator step")
        return {
            "batch_ids": selected.batch_ids,
            "losses": values,
            "real_logits": _logit_statistics(real_logits),
            "fake_logits": _logit_statistics(fake_logits),
            "real_minus_fake_logit_margin": float(
                real_logits.detach().float().mean() - fake_logits.detach().float().mean()
            ),
            "real_active_logit_sign_accuracy": _weighted_sign_accuracy(
                real_logits.float(), real_weights, positive=True
            ),
            "fake_active_logit_sign_accuracy": _weighted_sign_accuracy(
                fake_logits.float(), fake_weights, positive=False
            ),
            "gradient_clipping": clipping,
            "gradient_clipping_applied": (
                clipping["pre_clipping_norm"]
                > self.config.discriminator_gradient_clip_max_norm
            ),
            "r1_scheduled": scheduled,
            "r1_gamma": self.config.r1_gamma,
            "r1_interval": self.config.r1_interval,
            "r1_schedule_convention": "(global_step + 1) % r1_interval == 0",
            "fake_tensor_detached": fake_detached,
            "real_fake_masks_identical": torch.equal(
                selected.real_discriminator_mask, selected.fake_discriminator_mask
            ),
            "real_fake_localization_weights_identical": torch.equal(
                real_weights, fake_weights
            ),
            "aligned_views_used": True,
            "generator_parameter_hash_before": generator_parameters_before,
            "generator_parameter_hash_after": generator_parameters_after,
            "discriminator_parameter_hash_before": discriminator_parameters_before,
            "discriminator_parameter_hash_after": discriminator_parameters_after,
            "generator_parameters_changed": False if verify_parameter_isolation else None,
            "discriminator_parameters_changed": True if verify_parameter_isolation else None,
            "parameter_isolation_hashes_computed": verify_parameter_isolation,
            "generator_gradients_constructed": False,
            "optimizer_state_finite": optimizer_state_is_finite(
                self.discriminator_optimizer
            ),
        }

    def generator_step(
        self, batch: GANTrainingBatch, *, verify_parameter_isolation: bool = True
    ) -> dict[str, Any]:
        selected = self._require_train_batch(batch)
        generator_parameters_before = (
            parameter_hash(self.generator) if verify_parameter_isolation else None
        )
        discriminator_parameters_before = (
            parameter_hash(self.discriminator) if verify_parameter_isolation else None
        )
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_states_before = [
            parameter.requires_grad for parameter in self.discriminator.parameters()
        ]
        try:
            with _temporarily_disable_parameter_gradients(self.discriminator):
                with precision_autocast(self.device, self.precision):
                    generated = self.generator(
                        selected.composite_image, selected.generator_mask
                    )
                generated.refined_image.retain_grad()
                aligned = prepare_aligned_discriminator_views(
                    selected.real_image,
                    generated.refined_image,
                    selected.real_valid_mask,
                    selected.fake_valid_mask,
                    selected.fake_discriminator_mask,
                    generator_support_mask=generated.support_mask.float(),
                    discriminator_mask_threshold=self.loss_config.canonical_mask_threshold,
                )
                with precision_autocast(self.device, self.precision):
                    fake_logits = self.discriminator(
                        aligned.fake_discriminator_view, aligned.discriminator_mask
                    )
                if not bool(torch.isfinite(fake_logits).all()):
                    raise GANTrainingNumericalError(
                        "generator non-finite discriminator logits before loss; "
                        f"batch_ids={selected.batch_ids}"
                    )
                weights = patch_logit_localization_weights(
                    aligned.discriminator_mask,
                    fake_logits.float(),
                    localization_radius=self.loss_config.localization_radius,
                    mask_threshold=self.loss_config.canonical_mask_threshold,
                )
                adversarial = localized_generator_adversarial_loss(
                    fake_logits.float(), weights
                ).total
                change = support_normalized_change_loss(
                    generated.refined_image.float(),
                    selected.composite_image.float(),
                    generated.support_mask,
                )
                boundary = boundary_seam_loss(
                    generated.refined_image.float(),
                    selected.composite_image.float(),
                    generated.support_mask,
                    boundary_width=self.loss_config.boundary_ring_width,
                )
                total_variation = masked_total_variation_loss(
                    generated.applied_residual.float(), generated.support_mask
                )
                aggregated = aggregate_generator_losses(
                    adversarial=adversarial.float(),
                    change=change.float(),
                    boundary=boundary.float(),
                    total_variation=total_variation.float(),
                    weights=self.config.one_step_generator_loss_weights,
                )
                values = _loss_values(
                    {
                        "adversarial": adversarial.float(),
                        "change": change.float(),
                        "boundary": boundary.float(),
                        "total_variation": total_variation.float(),
                        "total": aggregated.total.float(),
                    },
                    batch_ids=selected.batch_ids,
                    label="generator",
                )
                adversarial_image_gradient = torch.autograd.grad(
                    adversarial,
                    generated.refined_image,
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                outside_support = ~generated.support_mask.expand_as(
                    generated.refined_image
                )
                locality_before = torch.equal(
                    generated.refined_image[outside_support],
                    selected.composite_image[outside_support],
                )
                clipping = self._backward_clip_step(
                    loss=aggregated.total,
                    module=self.generator,
                    optimizer=self.generator_optimizer,
                    scaler=self.generator_scaler,
                    maximum_norm=self.config.generator_gradient_clip_max_norm,
                    batch_ids=selected.batch_ids,
                    label="generator",
                )
                total_image_gradient = generated.refined_image.grad
                if total_image_gradient is None:
                    raise RuntimeError("Generator adversarial image gradient was not retained")
                canonical_mask = aligned.discriminator_mask.bool()
                canonical = canonical_mask.expand_as(adversarial_image_gradient)
                invalid = (~aligned.joint_valid_mask.bool()).expand_as(
                    adversarial_image_gradient
                )
                canonical_gradient_telemetry = (
                    canonical_adversarial_gradient_telemetry(
                        adversarial_image_gradient,
                        canonical_mask,
                    )
                )
                maximum_invalid_gradient = float(
                    adversarial_image_gradient[invalid].detach().float().abs().max()
                    if bool(invalid.any())
                    else 0.0
                )
                absolute_change = (
                    generated.refined_image.detach().float()
                    - selected.composite_image.detach().float()
                ).abs()
                canonical_pixels = aligned.discriminator_mask.bool().expand_as(
                    absolute_change
                )
                support_pixels = generated.support_mask.expand_as(absolute_change)
                mean_canonical_change = float(absolute_change[canonical_pixels].mean())
                mean_support_change = float(absolute_change[support_pixels].mean())
                applied = generated.applied_residual.detach().float()
                raw_direction = torch.tanh(generated.raw_residual.detach().float())
                composite_float = selected.composite_image.detach().float()
                configured_cap = torch.full_like(
                    composite_float, self.generator.residual_scale
                )
                positive_cap = torch.minimum(configured_cap, 1.0 - composite_float)
                negative_cap = torch.minimum(configured_cap, composite_float + 1.0)
                directional_cap = torch.where(
                    raw_direction >= 0, positive_cap, negative_cap
                )
                old_additive = (
                    composite_float
                    + self.generator.residual_scale * raw_direction
                )
                maximum_residual = float(applied.abs().max())
                mean_residual = float(applied.abs()[support_pixels].mean())
                output_range_violation_count = int(
                    ((generated.refined_image.detach().float() < -1.0)
                    | (generated.refined_image.detach().float() > 1.0)).sum()
                )
                would_have_clamped = float(
                    ((old_additive < -1.0) | (old_additive > 1.0))
                    [support_pixels]
                    .float()
                    .mean()
                )
                positive_directional_cap = directional_cap > 0
                cap_audit_pixels = support_pixels & positive_directional_cap
                directional_cap_saturation = float(
                    (applied.abs() >= 0.99 * directional_cap)
                    [cap_audit_pixels]
                    .float()
                    .mean()
                    if bool(cap_audit_pixels.any())
                    else 0.0
                )
                tanh_saturation = float(
                    (raw_direction.abs() >= 0.99)[support_pixels].float().mean()
                )
                exact_outside_support_change = float(
                    absolute_change[outside_support].max()
                    if bool(outside_support.any())
                    else 0.0
                )
        finally:
            restored = [
                parameter.requires_grad for parameter in self.discriminator.parameters()
            ]
        if restored != discriminator_states_before:
            raise RuntimeError("Discriminator requires_grad state was not restored")
        self.generator_optimizer_steps += 1
        generator_parameters_after = (
            parameter_hash(self.generator) if verify_parameter_isolation else None
        )
        discriminator_parameters_after = (
            parameter_hash(self.discriminator) if verify_parameter_isolation else None
        )
        if verify_parameter_isolation and generator_parameters_before == generator_parameters_after:
            raise RuntimeError("Generator optimizer step changed no parameters")
        if verify_parameter_isolation and discriminator_parameters_before != discriminator_parameters_after:
            raise RuntimeError("Generator step mutated discriminator parameters")
        if any(parameter.grad is not None for parameter in self.discriminator.parameters()):
            raise RuntimeError("Discriminator parameter gradients exist after generator step")
        with torch.no_grad(), precision_autocast(self.device, self.precision):
            after = self.generator(selected.composite_image, selected.generator_mask)
        outside_after = ~after.support_mask.expand_as(after.refined_image)
        locality_after = torch.equal(
            after.refined_image[outside_after], selected.composite_image[outside_after]
        )
        if not locality_before or not locality_after:
            raise RuntimeError("Generator exact locality failed around its optimizer step")
        return {
            "batch_ids": selected.batch_ids,
            "losses": values,
            "fake_logits": _logit_statistics(fake_logits),
            "gradient_clipping": clipping,
            "gradient_clipping_applied": (
                clipping["pre_clipping_norm"]
                > self.config.generator_gradient_clip_max_norm
            ),
            "mean_absolute_residual_inside_support": mean_residual,
            "maximum_absolute_residual": maximum_residual,
            "mean_absolute_applied_residual": mean_residual,
            "maximum_absolute_applied_residual": maximum_residual,
            "mean_change_inside_canonical_defect": mean_canonical_change,
            "mean_change_inside_support_halo": mean_support_change,
            "output_range_violation_count": output_range_violation_count,
            "would_have_clamped_fraction_old_additive": would_have_clamped,
            "directional_cap_saturation_fraction": directional_cap_saturation,
            "clamp_saturation_fraction": 0.0,
            "clamp_saturation_deprecated": True,
            "tanh_raw_residual_saturation_fraction": tanh_saturation,
            "tanh_residual_saturation_fraction": tanh_saturation,
            "exact_outside_support_change": exact_outside_support_change,
            **canonical_gradient_telemetry,
            "maximum_invalid_fake_pixel_gradient": maximum_invalid_gradient,
            "generator_locality_before_step": locality_before,
            "generator_locality_after_step": locality_after,
            "generator_graph_recomputed": True,
            "discriminator_gradients_constructed": False,
            "discriminator_requires_grad_restored": True,
            "generator_parameter_hash_before": generator_parameters_before,
            "generator_parameter_hash_after": generator_parameters_after,
            "discriminator_parameter_hash_before": discriminator_parameters_before,
            "discriminator_parameter_hash_after": discriminator_parameters_after,
            "generator_parameters_changed": True if verify_parameter_isolation else None,
            "discriminator_parameters_changed": False if verify_parameter_isolation else None,
            "parameter_isolation_hashes_computed": verify_parameter_isolation,
            "optimizer_state_finite": optimizer_state_is_finite(
                self.generator_optimizer
            ),
        }

    def monitor_forward(self, batch: GANTrainingBatch) -> dict[str, Any]:
        if any(item.get("split") != "monitor" for item in batch.metadata):
            raise ValueError("Monitor forward requires only the internal monitor split")
        selected = batch.to(self.device)
        with _temporary_eval(self.generator, self.discriminator), torch.no_grad():
            with precision_autocast(self.device, self.precision):
                generated = self.generator(
                    selected.composite_image, selected.generator_mask
                )
                aligned = prepare_aligned_discriminator_views(
                    selected.real_image,
                    generated.refined_image,
                    selected.real_valid_mask,
                    selected.fake_valid_mask,
                    selected.fake_discriminator_mask,
                    generator_support_mask=generated.support_mask.float(),
                    discriminator_mask_threshold=self.loss_config.canonical_mask_threshold,
                )
                real_logits = self.discriminator(
                    aligned.real_discriminator_view, aligned.discriminator_mask
                )
                fake_logits = self.discriminator(
                    aligned.fake_discriminator_view, aligned.discriminator_mask
                )
            weights = patch_logit_localization_weights(
                aligned.discriminator_mask,
                real_logits.float(),
                localization_radius=self.loss_config.localization_radius,
                mask_threshold=self.loss_config.canonical_mask_threshold,
            )
            hinge = localized_discriminator_hinge_loss(
                real_logits.float(), fake_logits.float(), weights, weights
            )
        return {
            "batch_ids": selected.batch_ids,
            "real_logits": _logit_statistics(real_logits),
            "fake_logits": _logit_statistics(fake_logits),
            "real_minus_fake_logit_margin": float(
                real_logits.detach().float().mean() - fake_logits.detach().float().mean()
            ),
            "real_hinge": float(hinge.real.detach()),
            "fake_hinge": float(hinge.fake.detach()),
            "real_active_logit_sign_accuracy": _weighted_sign_accuracy(
                real_logits.float(), weights, positive=True
            ),
            "fake_active_logit_sign_accuracy": _weighted_sign_accuracy(
                fake_logits.float(), weights, positive=False
            ),
            "optimizer_steps": 0,
        }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    selected = array[finite]
    if not len(selected):
        return {
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "maximum": None,
            "finite_count": 0,
            "nonfinite_count": int(len(array)),
        }
    return {
        "minimum": float(selected.min()),
        "p10": float(np.percentile(selected, 10)),
        "median": float(np.median(selected)),
        "p90": float(np.percentile(selected, 90)),
        "maximum": float(selected.max()),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
    }


def calibrate_gan_loss_scales(
    trainer: GANOneStepTrainer,
    batches: Sequence[GANTrainingBatch],
    *,
    progress=None,
) -> dict[str, Any]:
    """Measure eight-batch raw losses and unit-coefficient gradients without updates."""

    if len(batches) != trainer.config.deterministic_audit_batches:
        raise ValueError("Calibration batch count disagrees with G1.4 configuration")
    raw: dict[str, list[float]] = {
        name: []
        for name in (
            "discriminator_real_hinge",
            "discriminator_fake_hinge",
            "discriminator_total_hinge",
            "unscaled_r1",
            "generator_adversarial",
            "change",
            "seam",
            "total_variation",
        )
    }
    generator_norms = {
        name: [] for name in ("adversarial", "change", "seam", "total_variation")
    }
    discriminator_norms = {"hinge": [], "unscaled_r1": []}
    generator_hash_before = parameter_hash(trainer.generator)
    discriminator_hash_before = parameter_hash(trainer.discriminator)
    generator_parameters = tuple(trainer.generator.parameters())
    discriminator_parameters = tuple(trainer.discriminator.parameters())
    with _temporary_eval(trainer.generator, trainer.discriminator):
        for batch_index, cpu_batch in enumerate(batches, start=1):
            batch = trainer._require_train_batch(cpu_batch)
            trainer.generator_optimizer.zero_grad(set_to_none=True)
            trainer.discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), precision_autocast(trainer.device, trainer.precision):
                generated_for_d = trainer.generator(
                    batch.composite_image, batch.generator_mask
                )
            aligned_d = prepare_aligned_discriminator_views(
                batch.real_image,
                generated_for_d.refined_image.detach(),
                batch.real_valid_mask,
                batch.fake_valid_mask,
                batch.fake_discriminator_mask,
                generator_support_mask=generated_for_d.support_mask.float(),
                discriminator_mask_threshold=trainer.loss_config.canonical_mask_threshold,
            )
            with precision_autocast(trainer.device, trainer.precision):
                real_logits = trainer.discriminator(
                    aligned_d.real_discriminator_view, aligned_d.discriminator_mask
                )
                fake_logits = trainer.discriminator(
                    aligned_d.fake_discriminator_view, aligned_d.discriminator_mask
                )
            weights = patch_logit_localization_weights(
                aligned_d.discriminator_mask,
                real_logits.float(),
                localization_radius=trainer.loss_config.localization_radius,
                mask_threshold=trainer.loss_config.canonical_mask_threshold,
            )
            hinge = localized_discriminator_hinge_loss(
                real_logits.float(), fake_logits.float(), weights, weights
            )
            raw["discriminator_real_hinge"].append(float(hinge.real.detach()))
            raw["discriminator_fake_hinge"].append(float(hinge.fake.detach()))
            raw["discriminator_total_hinge"].append(float(hinge.total.detach()))
            discriminator_norms["hinge"].append(
                _autograd_gradient_norm(
                    hinge.total, discriminator_parameters, retain_graph=False
                )
            )
            trainer.discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=trainer.device.type, enabled=False):
                raw_r1 = localized_r1_gradient_penalty(
                    trainer.discriminator,
                    aligned_d.real_discriminator_view.float(),
                    aligned_d.discriminator_mask.float(),
                    localization_radius=trainer.loss_config.localization_radius,
                    mask_threshold=trainer.loss_config.canonical_mask_threshold,
                ).float()
            raw["unscaled_r1"].append(float(raw_r1.detach()))
            discriminator_norms["unscaled_r1"].append(
                _autograd_gradient_norm(
                    raw_r1, discriminator_parameters, retain_graph=False
                )
            )
            with _temporarily_disable_parameter_gradients(trainer.discriminator):
                with precision_autocast(trainer.device, trainer.precision):
                    generated = trainer.generator(
                        batch.composite_image, batch.generator_mask
                    )
                aligned = prepare_aligned_discriminator_views(
                    batch.real_image,
                    generated.refined_image,
                    batch.real_valid_mask,
                    batch.fake_valid_mask,
                    batch.fake_discriminator_mask,
                    generator_support_mask=generated.support_mask.float(),
                    discriminator_mask_threshold=trainer.loss_config.canonical_mask_threshold,
                )
                with precision_autocast(trainer.device, trainer.precision):
                    fake_for_g = trainer.discriminator(
                        aligned.fake_discriminator_view, aligned.discriminator_mask
                    )
                g_weights = patch_logit_localization_weights(
                    aligned.discriminator_mask,
                    fake_for_g.float(),
                    localization_radius=trainer.loss_config.localization_radius,
                    mask_threshold=trainer.loss_config.canonical_mask_threshold,
                )
                components = {
                    "adversarial": localized_generator_adversarial_loss(
                        fake_for_g.float(), g_weights
                    ).total,
                    "change": support_normalized_change_loss(
                        generated.refined_image.float(),
                        batch.composite_image.float(),
                        generated.support_mask,
                    ),
                    "seam": boundary_seam_loss(
                        generated.refined_image.float(),
                        batch.composite_image.float(),
                        generated.support_mask,
                        boundary_width=trainer.loss_config.boundary_ring_width,
                    ),
                    "total_variation": masked_total_variation_loss(
                        generated.applied_residual.float(), generated.support_mask
                    ),
                }
                for position, (name, value) in enumerate(components.items()):
                    raw_name = "generator_adversarial" if name == "adversarial" else name
                    raw[raw_name].append(float(value.detach()))
                    generator_norms[name].append(
                        _autograd_gradient_norm(
                            value,
                            generator_parameters,
                            retain_graph=position < len(components) - 1,
                        )
                    )
            if progress is not None:
                progress(batch_index, len(batches))
    if parameter_hash(trainer.generator) != generator_hash_before:
        raise RuntimeError("Calibration mutated generator parameters")
    if parameter_hash(trainer.discriminator) != discriminator_hash_before:
        raise RuntimeError("Calibration mutated discriminator parameters")
    if any(not math.isfinite(value) for values in raw.values() for value in values):
        raise GANTrainingNumericalError("Calibration produced non-finite loss values")
    if any(
        not math.isfinite(value)
        for values in generator_norms.values()
        for value in values
    ):
        raise GANTrainingNumericalError("Calibration produced non-finite generator gradients")
    adversarial_median = float(np.median(generator_norms["adversarial"]))
    minimum = trainer.config.suggestion_coefficient_minimum
    maximum = trainer.config.suggestion_coefficient_maximum
    suggestions: dict[str, float | None] = {"adversarial": 1.0}
    ratios: dict[str, dict[str, Any]] = {}
    for name, values in generator_norms.items():
        median = float(np.median(values))
        ratio = median / adversarial_median if adversarial_median > 0 else None
        ratios[name] = {
            "unit_gradient_norm": _distribution(values),
            "median_relative_to_adversarial": ratio,
            "zero_gradient_count": sum(value == 0 for value in values),
        }
        if name != "adversarial":
            unbounded = adversarial_median / median if median > 0 else None
            suggestions[name] = (
                None if unbounded is None else float(np.clip(unbounded, minimum, maximum))
            )
    r1_median = float(np.median(discriminator_norms["unscaled_r1"]))
    hinge_median = float(np.median(discriminator_norms["hinge"]))
    return {
        "batch_count": len(batches),
        "raw_loss_distributions": {
            name: _distribution(values) for name, values in raw.items()
        },
        "generator_unit_gradient_scales": ratios,
        "discriminator_unit_gradient_scales": {
            "hinge": _distribution(discriminator_norms["hinge"]),
            "unscaled_r1": _distribution(discriminator_norms["unscaled_r1"]),
            "r1_median_relative_to_hinge": (
                r1_median / hinge_median if hinge_median > 0 else None
            ),
        },
        "suggested_provisional_generator_coefficients": suggestions,
        "suggestion_rule": (
            "Set adversarial=1 and scale each other component by median adversarial "
            "unit-gradient norm divided by its own median unit-gradient norm"
        ),
        "suggestion_clamp": {"minimum": minimum, "maximum": maximum},
        "suggestions_written_to_configuration": False,
        "suggestions_used_for_one_step": False,
        "parameters_unchanged": True,
    }


def config_as_dict(config: GANTrainerConfig) -> dict[str, Any]:
    return asdict(config)
