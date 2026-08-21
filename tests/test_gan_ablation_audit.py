from __future__ import annotations

from scripts.audit_gan_smoke_ablation import clipping_statistics, recommendation


def _record(step: int, *, discriminator_clipped: bool, generator_clipped: bool):
    return {
        "kind": "joint",
        "joint_step": step,
        "discriminator": {
            "gradient_clipping_applied": discriminator_clipped,
            "gradient_clipping": {"pre_clipping_norm": 12.0},
        },
        "generator": {
            "gradient_clipping_applied": generator_clipped,
            "gradient_clipping": {"pre_clipping_norm": 2.0},
        },
    }


def test_clipping_statistics_separate_overall_and_final_60() -> None:
    records = [
        _record(
            step,
            discriminator_clipped=step <= 140,
            generator_clipped=step % 2 == 0,
        )
        for step in range(1, 201)
    ]
    result = clipping_statistics(records)
    assert result["overall"]["discriminator"]["update_count"] == 200
    assert result["overall"]["discriminator"]["clipped_fraction"] == 0.7
    assert result["final_60_joint_steps"]["discriminator"]["update_count"] == 60
    assert result["final_60_joint_steps"]["discriminator"]["clipped_fraction"] == 0
    assert result["final_60_joint_steps"]["generator"]["clipped_fraction"] == 0.5


def _comparison(clipping: float, final60: float, distance: float, margin: float):
    return {
        "safety_invariants": {"safe": True},
        "clipping": {
            "overall": {"discriminator": {"clipped_fraction": clipping}},
            "final_60_joint_steps": {
                "discriminator": {"clipped_fraction": final60}
            },
        },
        "detector_distance": {"l2": distance},
        "logits": {
            "final_60_joint_steps": {
                "real_minus_fake_margin": {"mean": margin}
            }
        },
    }


def test_ambiguous_ablation_retains_baseline() -> None:
    baseline = _comparison(0.70, 0.65, 0.10, 0.05)
    candidate = _comparison(0.65, 0.60, 0.09, 0.05)
    result = recommendation(
        baseline,
        candidate,
        {"preference": "tie", "safety_concerns": []},
    )
    assert result["selected_configuration"] == "baseline"
    assert result["reason"] == "neither_clearly_dominates_retain_baseline"


def test_clear_safe_candidate_dominance_is_allowed() -> None:
    baseline = _comparison(0.70, 0.65, 0.10, 0.05)
    candidate = _comparison(0.45, 0.40, 0.09, 0.04)
    result = recommendation(
        baseline,
        candidate,
        {"preference": "candidate", "safety_concerns": []},
    )
    assert result["selected_configuration"] == "candidate"
    assert result["candidate_clearly_dominates"]
