"""Verify CUDA PyTorch and save a reproducible Windows environment report."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_COMMAND = (
    r".\.venv\Scripts\python.exe -m pip install torch torchvision "
    r"--index-url https://download.pytorch.org/whl/cu128"
)


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _system_ram_bytes() -> int:
    status = MemoryStatusEx()
    status.length = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.total_physical)


def build_report() -> dict[str, object]:
    smi = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True).stdout
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    gpu_name, driver, memory_mib, reported_capability = (value.strip() for value in query.split(","))
    cuda_match = re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", smi)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing silent CPU fallback")
    device = torch.device("cuda:0")
    left = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    operation_result = float((left @ left.T).sum().item())
    torch.cuda.synchronize()
    return {
        "status": "PASS",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "install_command": INSTALL_COMMAND,
        "python": {
            "version": sys.version,
            "architecture": platform.architecture()[0],
            "executable": sys.executable,
            "pip_version": subprocess.run(
                [sys.executable, "-m", "pip", "--version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
        },
        "system": {"platform": platform.platform(), "total_ram_bytes": _system_ram_bytes()},
        "nvidia": {
            "gpu_name": gpu_name,
            "driver_version": driver,
            "driver_reported_cuda_version": cuda_match.group(1) if cuda_match else None,
            "memory_mib": int(memory_mib),
            "nvidia_smi_compute_capability": reported_capability,
        },
        "pytorch": {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_tensor_operation_result": operation_result,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "environment" / "pytorch_environment.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report()
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"CUDA PyTorch environment PASS -> {output}")
    print(f'{report["pytorch"]["torch_version"]}; {report["pytorch"]["device_name"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

