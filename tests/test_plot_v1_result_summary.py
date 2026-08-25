from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.plot_v1_result_summary import REPO_ROOT, SUMMARY, display_path


def test_display_path_is_repository_relative_for_paths_inside_the_repo() -> None:
    assert display_path(REPO_ROOT / "assets" / "v1_result_summary.png") == "assets/v1_result_summary.png"
    assert display_path(SUMMARY) == "reports/g2_3b/confirmation_summary.json"


def test_display_path_accepts_paths_outside_the_repo(tmp_path: Path) -> None:
    """Regression: CI writes the figure to a temporary directory outside the repo.

    ``Path.relative_to`` raises ``ValueError`` there, which used to abort the
    script after the figure had already been written.
    """
    outside = tmp_path / "v1_result_summary.png"
    assert not str(outside.resolve()).startswith(str(REPO_ROOT))
    assert display_path(outside) == outside.resolve().as_posix()


def test_display_path_resolves_relative_paths_against_the_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    assert display_path(Path("assets/v1_result_summary.png")) == "assets/v1_result_summary.png"


@pytest.mark.skipif(not SUMMARY.is_file(), reason="G2.3B confirmation summary is absent")
def test_figure_regenerates_to_a_destination_outside_the_repository(tmp_path: Path) -> None:
    """End-to-end reproduction of the failing CI step, in a temporary directory."""
    output = tmp_path / "nested" / "v1_result_summary.png"
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "plot_v1_result_summary.py"), "--output", str(output)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert output.stat().st_size > 0
    assert output.resolve().as_posix() in completed.stdout


@pytest.mark.skipif(not SUMMARY.is_file(), reason="G2.3B confirmation summary is absent")
def test_the_plotted_summary_is_the_frozen_not_confirmed_result() -> None:
    """The figure must never be able to depict a different decision than the JSON."""
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["decision"] == "stop_not_confirmed_g2_3b"
    assert summary["confirmed"] is False
    assert summary["official_test_access_count"] == 0
