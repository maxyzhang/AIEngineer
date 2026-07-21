from evaluation.rag_evaluator import (
    calculate_answer_completeness,
    calculate_answer_groundedness,
    calculate_citation_coverage,
    calculate_context_relevance,
    classify_hallucination_risk,
    evaluate_rag_response,
    find_unsupported_claims,
    split_claims,
    token_overlap_score,
)

def test_split_claims_returns_sentences() -> None:
    claims = split_claims(
        "Pressure is 5,000 psi. Human review is required."
    )

    assert claims == [
        "Pressure is 5,000 psi.",
        "Human review is required.",
    ]


def test_token_overlap_score_detects_support() -> None:
    score = token_overlap_score(
        "The maximum pressure is 5,000 psi.",
        "Maximum pressure is 5,000 psi.",
    )

    assert score > 0.70

def test_context_relevance_is_high_for_relevant_context() -> None:
    score = calculate_context_relevance(
        question="What is the maximum operating pressure?",
        contexts=[
            "The maximum operating pressure is 5,000 psi."
        ],
    )

    assert score >= 0.60


def test_context_relevance_is_low_for_irrelevant_context() -> None:
    score = calculate_context_relevance(
        question="What is the maximum operating pressure?",
        contexts=[
            "The report describes employee training schedules."
        ],
    )

    assert score < 0.40

def test_grounded_answer_has_high_groundedness() -> None:
    score = calculate_answer_groundedness(
        answer="The maximum operating pressure is 5,000 psi.",
        contexts=[
            "The maximum operating pressure is 5,000 psi."
        ],
    )

    assert score == 1.0


def test_unsupported_claim_is_detected() -> None:
    unsupported = find_unsupported_claims(
        answer=(
            "The maximum pressure is 5,000 psi. "
            "The anomaly was caused by pump failure."
        ),
        contexts=[
            "The maximum pressure is 5,000 psi."
        ],
    )

    assert len(unsupported) == 1
    assert "pump failure" in unsupported[0].lower() 

def test_complete_answer_addresses_question_terms() -> None:
    score = calculate_answer_completeness(
        question="What is the maximum operating pressure?",
        answer="The maximum operating pressure is 5,000 psi.",
    )

    assert score >= 0.60

def test_citation_coverage_detects_source_reference() -> None:
    score = calculate_citation_coverage(
        answer=(
            "The maximum pressure is 5,000 psi "
            "[source: well_design.pdf]."
        ),
    )

    assert score == 1.0


def test_uncited_answer_has_zero_citation_coverage() -> None:
    score = calculate_citation_coverage(
        answer="The maximum pressure is 5,000 psi."
    )

    assert score == 0.0

def test_low_hallucination_risk() -> None:
    risk = classify_hallucination_risk(
        groundedness=1.0,
        unsupported_claim_count=0,
    )

    assert risk == "low"


def test_high_hallucination_risk() -> None:
    risk = classify_hallucination_risk(
        groundedness=0.25,
        unsupported_claim_count=3,
    )

    assert risk == "high"

def test_evaluate_rag_response_returns_structured_result() -> None:
    result = evaluate_rag_response(
        question="What is the maximum operating pressure?",
        answer=(
            "The maximum operating pressure is 5,000 psi "
            "[source: well_design.pdf]."
        ),
        contexts=[
            "The maximum operating pressure is 5,000 psi."
        ],
    )

    assert result.context_relevance >= 0.60
    assert result.answer_groundedness == 1.0
    assert result.citation_coverage == 1.0
    assert result.hallucination_risk == "low"
    assert result.unsupported_claims == []
    assert result.overall_score > 0.70


def test_evaluate_rag_response_flags_unsupported_claim() -> None:
    result = evaluate_rag_response(
        question="What caused the pressure anomaly?",
        answer="The anomaly was caused by pump failure.",
        contexts=[
            "A pressure anomaly was observed at 14:30."
        ],
    )

    assert result.hallucination_risk == "high"
    assert len(result.unsupported_claims) == 1
    assert "pump failure" in result.unsupported_claims[0].lower()
