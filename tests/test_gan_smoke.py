from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from defectgen.gan.training_pairs import GANTrainingSample
from defectgen.models import MaskConditionedPatchDiscriminator, MaskedResidualGenerator
from defectgen.training.gan_losses import load_gan_loss_config
from defectgen.training.gan_smoke import (
    AtomicJSONLLog,
    DetectorRetentionGate,
    FrozenDetectorEvaluator,
    MONITOR_CATEGORIES,
    SmokeCheckpointIdentity,
    SmokeProgress,
    canonical_configuration_hash,
    load_gan_smoke_config,
    load_smoke_checkpoint,
    save_smoke_checkpoint,
    select_fixed_monitor_samples,
    stage_one_allows_continuation,
    warmup_gate_decision,
)
from defectgen.training.gan_trainer import (
    GANOneStepTrainer,
    collate_gan_training_samples,
    load_gan_trainer_config,
    parameter_hash,
)
from scripts.train_gan_smoke import _training_gate


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample(
    index: int,
    *,
    split: str = "train",
    contacts: dict[str, bool] | None = None,
    size: int = 8,
) -> GANTrainingSample:
    generator = torch.Generator().manual_seed(900 + index)
    composite = torch.rand(3, 64, 64, generator=generator) * 1.6 - 0.8
    real = torch.rand(3, 64, 64, generator=generator) * 1.6 - 0.8
    mask = torch.zeros(1, 64, 64)
    mask[:, 28 : 28 + size, 28 : 28 + size] = 1
    valid = torch.ones(1, 64, 64)
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
            "template_id": f"smoke:{index}",
            "normal_background_sample_id": f"normal:{index}",
            "target_contact_sides": contacts
            or {"top": False, "bottom": False, "left": False, "right": False},
        },
    )


def _batch(index: int = 0, *, split: str = "train"):
    return collate_gan_training_samples(
        [_sample(index, split=split), _sample(index + 1, split=split)]
    )


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


def _trainer():
    smoke = load_gan_smoke_config(REPO_ROOT / "configs" / "gan_smoke.json")
    base = load_gan_trainer_config(REPO_ROOT / "configs" / "gan_one_step.json")
    generator, discriminator = _models()
    return GANOneStepTrainer(
        generator,
        discriminator,
        smoke.trainer_config(base),
        load_gan_loss_config(REPO_ROOT / "configs" / "gan_losses.json"),
        device="cpu",
    )


def test_provisional_configuration_is_exact_and_does_not_use_suggestions() -> None:
    config = load_gan_smoke_config(REPO_ROOT / "configs" / "gan_smoke.json")
    assert config.provisional_configuration
    assert config.generator_learning_rate == 0.0001
    assert config.discriminator_learning_rate == 0.00005
    assert config.generator_loss_weights.total_variation == 0.1
    assert config.r1_gamma == 1 and config.r1_interval == 16
    assert config.initial_discriminator_warmup_steps == 10
    assert config.micro_smoke_joint_steps == 20
    assert config.full_smoke_joint_steps == 200


def test_warmup_gate_and_stage_one_gate_behaviour() -> None:
    assert warmup_gate_decision(completed_steps=5, monitor_margin=1) == "continue"
    assert warmup_gate_decision(completed_steps=10, monitor_margin=0.01) == "accepted"
    assert warmup_gate_decision(completed_steps=10, monitor_margin=0) == "continue"
    assert warmup_gate_decision(completed_steps=20, monitor_margin=0) == "failed"
    assert stage_one_allows_continuation(
        completed_joint_steps=20, target=20, early_stop_reason=None
    )
    assert not stage_one_allows_continuation(
        completed_joint_steps=20, target=20, early_stop_reason="finite_stop"
    )
    assert not stage_one_allows_continuation(
        completed_joint_steps=19, target=20, early_stop_reason=None
    )


def test_smoke_warmup_mutates_only_d_and_r1_uses_gamma_one() -> None:
    torch.manual_seed(31)
    trainer = _trainer()
    generator_before = parameter_hash(trainer.generator)
    discriminator_before = parameter_hash(trainer.discriminator)
    result = trainer.discriminator_step(_batch(), global_step=15)
    assert parameter_hash(trainer.generator) == generator_before
    assert parameter_hash(trainer.discriminator) != discriminator_before
    assert result["r1_scheduled"] and result["r1_gamma"] == 1
    assert result["losses"]["scaled_r1"] == pytest.approx(
        result["losses"]["raw_r1"] * 1 * 16 / 2
    )
    assert trainer.generator_optimizer_steps == 0
    assert trainer.discriminator_optimizer_steps == 1


class _TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, inputs):
        return self.layer(inputs)


def test_detector_evaluator_is_frozen_detached_and_reports_metrics() -> None:
    detector = _TinyDetector()
    evaluator = FrozenDetectorEvaluator(
        detector,
        mean=(0, 0, 0),
        standard_deviation=(1, 1, 1),
        device="cpu",
    )
    images = torch.zeros(2, 3, 32, 32, requires_grad=True)
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 10:20, 10:20] = 1
    metrics, probabilities = evaluator.metrics(images, mask, torch.ones_like(mask))
    assert all(not parameter.requires_grad for parameter in detector.parameters())
    assert not probabilities.requires_grad
    assert images.grad is None
    assert set(metrics) == {
        "mean_probability_inside_mask",
        "mean_probability_outside_mask",
        "inside_outside_probability_contrast",
        "dice_at_0_5",
        "iou_at_0_5",
        "samples_with_any_predicted_positive_fraction",
    }
    assert all(parameter.grad is None for parameter in detector.parameters())


def test_detector_retention_requires_two_consecutive_stop_failures() -> None:
    gate = DetectorRetentionGate()
    first = gate.update(0.8, 0.39)
    second = gate.update(0.8, 0.39)
    assert first["warning"] and not first["stop"]
    assert second["stop"]
    recovered = gate.update(0.8, 0.8)
    assert recovered["consecutive_below_stop"] == 0 and not recovered["stop"]


def test_finite_and_locality_stop_gates() -> None:
    trainer = _trainer()
    config = load_gan_smoke_config(REPO_ROOT / "configs" / "gan_smoke.json")
    discriminator = {
        "real_logits": {"minimum": -1.0, "maximum": 1.0},
        "fake_logits": {"minimum": -1.0, "maximum": 1.0},
    }
    generator = {
        "exact_outside_support_change": 0.0,
        "maximum_invalid_fake_pixel_gradient": 0.0,
        "canonical_defect_gradient_coverage": 1.0,
        "clamp_saturation_fraction": 0.0,
        "output_range_violation_count": 0,
        "mean_absolute_residual_inside_support": 0.1,
        "fake_logits": {"minimum": -1.0, "maximum": 1.0},
    }
    assert _training_gate(trainer, config, discriminator, generator) is None
    changed = dict(generator, exact_outside_support_change=1e-7)
    assert _training_gate(trainer, config, discriminator, changed) == "outside_support_change"
    nonfinite = dict(generator, mean_absolute_residual_inside_support=float("nan"))
    assert _training_gate(trainer, config, discriminator, nonfinite) == "nonfinite_training_metric"
    out_of_range = dict(generator, output_range_violation_count=1)
    assert _training_gate(trainer, config, discriminator, out_of_range) == "output_range_violation"


def test_fixed_monitor_categories_and_identities_are_deterministic() -> None:
    contacts = [
        {"top": False, "bottom": False, "left": False, "right": False},
        {"top": True, "bottom": False, "left": False, "right": False},
        {"top": False, "bottom": False, "left": True, "right": False},
        {"top": True, "bottom": False, "left": True, "right": False},
        {"top": False, "bottom": False, "left": True, "right": True},
    ]
    samples = [
        _sample(index, split="monitor", contacts=value, size=2 + index * 2)
        for index, value in enumerate(contacts)
    ]
    first = select_fixed_monitor_samples(samples)
    second = select_fixed_monitor_samples(samples)
    assert tuple(first) == MONITOR_CATEGORIES
    assert [sample.metadata["template_id"] for sample in first.values()] == [
        sample.metadata["template_id"] for sample in second.values()
    ]


def test_atomic_jsonl_metric_logging(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    log = AtomicJSONLLog(path)
    log.append({"kind": "warmup", "step": 1})
    log.append({"kind": "joint", "step": 1})
    assert [json.loads(line) for line in path.read_text().splitlines()] == log.records
    assert not path.with_suffix(".jsonl.tmp").exists()


def _identity(configuration: dict) -> SmokeCheckpointIdentity:
    return SmokeCheckpointIdentity(
        canonical_configuration_hash(configuration),
        "manifest",
        "split",
        ("monitor:0", "monitor:1"),
    )


def test_atomic_checkpoint_and_incompatible_hash_rejection(tmp_path: Path) -> None:
    torch.manual_seed(32)
    trainer = _trainer()
    configuration = {"phase": "test", "seed": 42}
    identity = _identity(configuration)
    path = tmp_path / "last.pt"
    save_smoke_checkpoint(
        path,
        trainer=trainer,
        progress=SmokeProgress(),
        identity=identity,
        configuration=configuration,
    )
    assert path.is_file() and not path.with_suffix(".pt.tmp").exists()
    restored = _trainer()
    with pytest.raises(ValueError, match="manifest, split, or monitor"):
        load_smoke_checkpoint(
            path,
            trainer=restored,
            expected_identity=SmokeCheckpointIdentity(
                identity.configuration_sha256,
                "changed",
                identity.split_sha256,
                identity.fixed_monitor_sample_ids,
            ),
            expected_configuration=configuration,
        )
    with pytest.raises(ValueError, match="configuration"):
        load_smoke_checkpoint(
            path,
            trainer=restored,
            expected_identity=identity,
            expected_configuration={"phase": "changed"},
        )


def test_old_smoke_checkpoint_without_residual_semantics_is_rejected_unchanged(
    tmp_path: Path,
) -> None:
    trainer = _trainer()
    configuration = {"phase": "legacy", "seed": 42}
    identity = _identity(configuration)
    current = tmp_path / "current.pt"
    legacy = tmp_path / "failed_g1_5.pt"
    save_smoke_checkpoint(
        current,
        trainer=trainer,
        progress=SmokeProgress(),
        identity=identity,
        configuration=configuration,
    )
    payload = torch.load(current, map_location="cpu", weights_only=False)
    payload.pop("residual_semantics_version")
    payload.pop("architecture_version")
    torch.save(payload, legacy)
    before = legacy.read_bytes()
    with pytest.raises(ValueError, match="residual-semantics"):
        load_smoke_checkpoint(
            legacy,
            trainer=_trainer(),
            expected_identity=identity,
            expected_configuration=configuration,
        )
    assert legacy.is_file() and legacy.read_bytes() == before


def _joint_step(trainer: GANOneStepTrainer, batch) -> None:
    trainer.discriminator_step(
        batch, global_step=trainer.discriminator_optimizer_steps
    )
    trainer.generator_step(batch)


def test_uninterrupted_and_resumed_cpu_training_are_identical(tmp_path: Path) -> None:
    batches = [_batch(0), _batch(2)]
    torch.manual_seed(33)
    uninterrupted = _trainer()
    for batch in batches:
        _joint_step(uninterrupted, batch)

    torch.manual_seed(33)
    interrupted = _trainer()
    _joint_step(interrupted, batches[0])
    progress = SmokeProgress(
        joint_discriminator_steps=1,
        joint_generator_steps=1,
        data_epoch=0,
        batch_position=1,
        last_completed_operation="joint_1_g",
        warmup_gate_status="accepted",
    )
    configuration = {"phase": "resume-equivalence", "seed": 42}
    identity = _identity(configuration)
    path = tmp_path / "joint_001.pt"
    save_smoke_checkpoint(
        path,
        trainer=interrupted,
        progress=progress,
        identity=identity,
        configuration=configuration,
    )
    resumed = _trainer()
    restored_progress = load_smoke_checkpoint(
        path,
        trainer=resumed,
        expected_identity=identity,
        expected_configuration=configuration,
    )
    assert asdict(restored_progress) == asdict(progress)
    assert resumed.generator_optimizer_steps == 1
    assert resumed.discriminator_optimizer_steps == 1
    _joint_step(resumed, batches[1])
    assert resumed.generator_optimizer_steps == 2
    assert resumed.discriminator_optimizer_steps == 2
    assert parameter_hash(resumed.generator) == parameter_hash(uninterrupted.generator)
    assert parameter_hash(resumed.discriminator) == parameter_hash(
        uninterrupted.discriminator
    )


def test_monitor_data_cannot_be_optimized() -> None:
    trainer = _trainer()
    monitor = _batch(split="monitor")
    with pytest.raises(ValueError, match="train split"):
        trainer.discriminator_step(monitor, global_step=0)
    with pytest.raises(ValueError, match="train split"):
        trainer.generator_step(monitor)
    before = (trainer.generator_optimizer_steps, trainer.discriminator_optimizer_steps)
    trainer.monitor_forward(monitor)
    assert before == (
        trainer.generator_optimizer_steps,
        trainer.discriminator_optimizer_steps,
    )
