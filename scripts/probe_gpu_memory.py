"""Probe real physical mixed-precision batch size without hiding non-OOM errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from defectgen.models import UNet, count_parameters  # noqa: E402
from defectgen.training.losses import CombinedBCEDiceLoss  # noqa: E402
from defectgen.training.reproducibility import configure_reproducibility  # noqa: E402


def probe(config: dict) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; memory probe refuses CPU fallback")
    configure_reproducibility(config["seed"], deterministic=True, warn_only=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.empty(1, device=device)  # initialize the CUDA context before memory-stat resets
    width = config["model"]["input_width"]
    height = config["model"]["input_height"]
    mixed_precision = bool(config["training"]["mixed_precision"])
    attempts = []
    for batch_size in (4, 2, 1):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        model = UNet(base_channels=config["model"]["base_channels"]).to(device)
        criterion = CombinedBCEDiceLoss(
            config["loss"]["bce_weight"], config["loss"]["dice_weight"], config["loss"]["pos_weight"]
        )
        try:
            inputs = torch.randn(batch_size, 3, height, width, device=device)
            targets = torch.zeros(batch_size, 1, height, width, device=device)
            targets[:, :, height // 3 : height // 3 + 16, width // 3 : width // 3 + 8] = 1
            valid = torch.ones_like(targets)
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                logits = model(inputs)
                loss = criterion(logits, targets, valid)
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError("Memory probe produced a non-finite loss")
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError("Memory probe produced non-finite gradients")
            torch.cuda.synchronize()
            return {
                "status": "PASS",
                "device": torch.cuda.get_device_name(0),
                "successful_physical_batch_size": batch_size,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "input_shape": [batch_size, 3, height, width],
                "mixed_precision": mixed_precision,
                "model_parameter_count": count_parameters(model),
                "attempts": [*attempts, {"batch_size": batch_size, "result": "PASS"}],
            }
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch_size": batch_size, "result": "CUDA out of memory"})
            del model
            torch.cuda.empty_cache()
            continue
    raise RuntimeError(f"CUDA out of memory at physical batch sizes 4, 2, and 1: {attempts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "baseline.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "baseline_smoke" / "memory_probe.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = probe(config)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f'GPU memory probe PASS: batch={report["successful_physical_batch_size"]}, allocated={report["peak_allocated_bytes"] / 2**20:.1f} MiB, reserved={report["peak_reserved_bytes"] / 2**20:.1f} MiB')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
