from evaluation.recovery import ConfidenceRecovery
from agent_loop import estimate_confidence


def test_confidence_already_high():
    recovery = ConfidenceRecovery(
        confidence_threshold=0.65,
        max_retries=2,
    )

    result = recovery.recover(
        query="What MongoDB version are we using?",
        confidence=0.90,
        retrieve_fn=lambda **kwargs: [],
        evaluate_fn=lambda **kwargs: 0.0,
    )

    assert result.confidence == 0.90
    assert result.attempts == 0
    assert result.recovered is True
    assert result.stop_reason == "confidence_threshold_met"


def test_confidence_recovers_after_retry():
    recovery = ConfidenceRecovery(
        confidence_threshold=0.65,
        max_retries=2,
    )

    def retrieve_fn(query, top_k):
        return [
            "Production MongoDB version is 6.0.24"
        ]

    def evaluate_fn(query, evidence):
        return 0.82

    result = recovery.recover(
        query="What MongoDB version are we using?",
        confidence=0.40,
        retrieve_fn=retrieve_fn,
        evaluate_fn=evaluate_fn,
    )

    assert result.confidence == 0.82
    assert result.attempts == 1
    assert result.recovered is True
    assert result.stop_reason == "confidence_recovered"


def test_no_evidence_found():
    recovery = ConfidenceRecovery(
        confidence_threshold=0.65,
        max_retries=2,
    )

    def retrieve_fn(query, top_k):
        return []

    result = recovery.recover(
        query="Unknown production configuration",
        confidence=0.30,
        retrieve_fn=retrieve_fn,
        evaluate_fn=lambda **kwargs: 0.30,
    )

    assert result.confidence == 0.30
    assert result.attempts == 1
    assert result.recovered is False
    assert result.stop_reason == "no_evidence"


def test_max_retries_reached():
    recovery = ConfidenceRecovery(
        confidence_threshold=0.65,
        max_retries=2,
    )

    counter = {"attempt": 0}

    def retrieve_fn(query, top_k):
        counter["attempt"] += 1
        return [
            f"Weak evidence attempt {counter['attempt']}"
        ]

    def evaluate_fn(query, evidence):
        return 0.50

    result = recovery.recover(
        query="Unknown configuration",
        confidence=0.30,
        retrieve_fn=retrieve_fn,
        evaluate_fn=evaluate_fn,
    )

    assert result.confidence == 0.50
    assert result.attempts == 2
    assert result.recovered is False
    assert result.stop_reason == "max_retries_reached"


def test_top_k_increases_on_retry():
    recovery = ConfidenceRecovery(
        confidence_threshold=0.65,
        max_retries=2,
    )

    top_k_values = []

    def retrieve_fn(query, top_k):
        top_k_values.append(top_k)
        return [f"evidence top_k={top_k}"]

    def evaluate_fn(query, evidence):
        return 0.50

    recovery.recover(
        query="test query",
        confidence=0.20,
        retrieve_fn=retrieve_fn,
        evaluate_fn=evaluate_fn,
    )

    assert top_k_values == [5, 8]

def test_estimate_confidence_high_when_pass_and_supported():
    history = """
Observation:
source 1

Observation:
source 2
"""

    answer = (
        "This answer is grounded in multiple retrieved observations "
        "and contains enough detail to be considered complete."
    )

    confidence = estimate_confidence(
        answer=answer,
        history=history,
        review="PASS",
    )

    assert confidence >= 0.85


def test_estimate_confidence_low_when_retry_requested():
    history = """
Observation:
weak evidence
"""

    confidence = estimate_confidence(
        answer="Short answer.",
        history=history,
        review="RETRY: search for more evidence",
    )

    assert confidence < 0.65


def test_pass_can_still_be_low_confidence():
    confidence = estimate_confidence(
        answer="Too short.",
        history="",
        review="PASS",
    )

    assert confidence < 0.85


def test_confidence_is_bounded_between_zero_and_one():
    confidence = estimate_confidence(
        answer="A" * 500,
        history="Observation:\n" * 20,
        review="PASS",
    )

    assert 0.0 <= confidence <= 1.0