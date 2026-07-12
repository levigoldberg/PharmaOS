from __future__ import annotations

from types import SimpleNamespace

from pharma_os.validators import (
    assign_human_gate,
    generate_confidence_flags,
    validate_numeric_provenance,
    validate_source_coverage,
)


def test_source_coverage_fails_missing_source_id() -> None:
    claim = SimpleNamespace(
        claim_id="claim-1",
        claim_text="Trial has recruiting status.",
        source_ids=(),
        provenance="test",
        confidence=0.5,
        confidence_level="low",
    )

    result = validate_source_coverage(
        target_id="output-1",
        claims=(claim,),
        source_ids=set(),
        run_id="RUN",
    )

    assert result.status == "failed"
    assert result.gate_reason


def test_numeric_provenance_fails_high_risk_numeric_claim_without_source() -> None:
    claim = SimpleNamespace(
        claim_id="claim-1",
        claim_text="Enrollment was 12 patients.",
        source_ids=(),
        provenance="test",
        confidence=0.5,
        confidence_level="low",
    )

    result = validate_numeric_provenance(target_id="output-1", claims=(claim,), run_id="RUN")

    assert result.status == "failed"


def test_human_gate_created_for_high_risk_language() -> None:
    gate = assign_human_gate(
        run_id="RUN",
        workflow_name="trial_intelligence",
        validation_results=(),
        output_text="This is a go/no-go recommendation.",
    )

    assert gate is not None
    assert gate.decision == "needs_human_review"


def test_confidence_flags_are_selective_ranked_and_gate_aware() -> None:
    gate = assign_human_gate(
        run_id="RUN",
        workflow_name="due_diligence",
        validation_results=(),
        output_text="This requires an investment recommendation review.",
    )
    flags = generate_confidence_flags(
        run_id="RUN",
        validation_results=(),
        risk_flags=(
            SimpleNamespace(flag_id="low-noise", severity="low", reason="Low-value missing detail."),
            SimpleNamespace(flag_id="high-gap", severity="high", reason="Commercial model is non-calculable."),
            SimpleNamespace(flag_id="duplicate-gap", severity="medium", reason="Commercial model is non-calculable."),
        ),
        human_gate=gate,
    )

    assert [flag.severity for flag in flags] == ["high", "medium"]
    assert [flag.reason for flag in flags] == [
        "Commercial model is non-calculable.",
        gate.gate_reason,
    ]
