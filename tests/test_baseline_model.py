from __future__ import annotations

import torch

from defectgen.models import UNet
from defectgen.training.losses import CombinedBCEDiceLoss, masked_soft_dice_loss


def test_model_output_shape_on_complete_canvas() -> None:
    model = UNet(base_channels=4)
    inputs = torch.randn(1, 3, 672, 256)
    with torch.no_grad():
        output = model(inputs)
    assert output.shape == (1, 1, 672, 256)


def test_loss_and_gradients_are_finite_with_empty_mask() -> None:
    model = UNet(base_channels=4)
    inputs = torch.randn(2, 3, 64, 64)
    targets = torch.zeros(2, 1, 64, 64)
    valid = torch.ones_like(targets)
    logits = model(inputs)
    loss = CombinedBCEDiceLoss(pos_weight=10.0)(logits, targets, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_padded_predictions_do_not_affect_loss() -> None:
    targets = torch.zeros(1, 1, 8, 8)
    targets[:, :, 3:5, 3:5] = 1
    valid = torch.zeros_like(targets)
    valid[:, :, 2:6, 2:6] = 1
    first = torch.zeros_like(targets)
    second = first.clone()
    second[valid == 0] = 1000
    criterion = CombinedBCEDiceLoss(pos_weight=5.0)
    assert torch.equal(criterion(first, targets, valid), criterion(second, targets, valid))


def test_empty_mask_dice_has_no_nan() -> None:
    logits = torch.full((2, 1, 8, 8), -10.0)
    targets = torch.zeros_like(logits)
    valid = torch.ones_like(logits)
    loss = masked_soft_dice_loss(logits, targets, valid)
    assert torch.isfinite(loss)


def test_checkpoint_round_trip_cpu(tmp_path) -> None:
    model = UNet(base_channels=4).eval()
    inputs = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(inputs)
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, path)
    restored = UNet(base_channels=4).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    with torch.no_grad():
        actual = restored(inputs)
    assert torch.equal(expected, actual)


def test_cuda_forward_optional() -> None:
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is unavailable")
    model = UNet(base_channels=4).cuda().eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(torch.randn(1, 3, 64, 64, device="cuda"))
    assert output.shape == (1, 1, 64, 64)
    assert torch.isfinite(output).all()

