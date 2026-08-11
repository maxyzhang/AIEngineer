from evaluation.escalation import EscalationPolicy


def test_escalates_when_recovery_failed_and_confidence_low():
    policy = EscalationPolicy(escalation_threshold=0.65)

    decision = policy.evaluate(
        confidence=0.40,
        recovered=False,
        stop_reason="max_retries_reached",
    )

    assert decision.escalate is True
    assert decision.severity == "high"
    assert decision.reason == "max_retries_reached"
    assert decision.confidence == 0.40


def test_does_not_escalate_when_confidence_high():
    policy = EscalationPolicy(escalation_threshold=0.65)

    decision = policy.evaluate(
        confidence=0.85,
        recovered=False,
        stop_reason="confidence_threshold_met",
    )

    assert decision.escalate is False
    assert decision.severity == "none"


def test_does_not_escalate_when_recovery_succeeded():
    policy = EscalationPolicy(escalation_threshold=0.65)

    decision = policy.evaluate(
        confidence=0.60,
        recovered=True,
        stop_reason="recovered",
    )

    assert decision.escalate is False


def test_threshold_boundary_does_not_escalate():
    policy = EscalationPolicy(escalation_threshold=0.65)

    decision = policy.evaluate(
        confidence=0.65,
        recovered=False,
        stop_reason="threshold_boundary",
    )

    assert decision.escalate is False

def test_low_confidence_requires_human_review():
    policy = EscalationPolicy(escalation_threshold=0.65)

    decision = policy.evaluate(
        confidence=0.42,
        recovered=False,
        stop_reason="max_retries_reached",
    )

    human_review_required = decision.escalate

    assert human_review_required is True
    assert decision.severity == "high"


def test_high_confidence_does_not_require_human_review():
    policy = EscalationPolicy(escalation_threshold=0.65)

    decision = policy.evaluate(
        confidence=0.88,
        recovered=True,
        stop_reason="confidence_recovered",
    )

    human_review_required = decision.escalate

    assert human_review_required is False
    assert decision.severity == "none"