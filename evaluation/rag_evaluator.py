from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class RAGEvaluationResult:
    """
    Structured quality assessment for one RAG response.
    """

    context_relevance: float
    answer_groundedness: float
    answer_completeness: float
    citation_coverage: float
    hallucination_risk: str
    unsupported_claims: list[str]
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic token comparison.
    """

    return " ".join(text.lower().split())


def tokenize(text: str) -> set[str]:
    """
    Convert text into a set of normalized alphanumeric tokens.
    """

    normalized = normalize_text(text)

    return {
        token
        for token in _TOKEN_PATTERN.findall(normalized)
        if len(token) > 1
    }


def split_claims(answer: str) -> list[str]:
    """
    Split an answer into non-empty sentence-like claims.
    """

    normalized = answer.strip()

    if not normalized:
        return []

    return [
        sentence.strip()
        for sentence in _SENTENCE_PATTERN.split(normalized)
        if sentence.strip()
    ]


def token_overlap_score(
    source_text: str,
    target_text: str,
) -> float:
    """
    Measure how much of the target text is supported by source tokens.

    A score of 1.0 means every target token appears in the source.
    """

    source_tokens = tokenize(source_text)
    target_tokens = tokenize(target_text)

    if not target_tokens:
        return 0.0

    overlap = source_tokens.intersection(target_tokens)

    return round(
        len(overlap) / len(target_tokens),
        4,
    )

def calculate_context_relevance(
    *,
    question: str,
    contexts: list[str],
) -> float:
    """
    Estimate whether retrieved contexts are relevant to the question.
    """

    if not contexts:
        return 0.0

    combined_context = "\n".join(contexts)

    return token_overlap_score(
        combined_context,
        question,
    )

def find_unsupported_claims(
    *,
    answer: str,
    contexts: list[str],
    support_threshold: float = 0.35,
) -> list[str]:
    """
    Return answer claims that have insufficient lexical support
    in the retrieved context.
    """

    combined_context = "\n".join(contexts)
    unsupported: list[str] = []

    for claim in split_claims(answer):
        score = token_overlap_score(
            combined_context,
            claim,
        )

        if score < support_threshold:
            unsupported.append(claim)

    return unsupported


def calculate_answer_groundedness(
    *,
    answer: str,
    contexts: list[str],
    support_threshold: float = 0.35,
) -> float:
    """
    Estimate the percentage of answer claims supported by context.
    """

    claims = split_claims(answer)

    if not claims:
        return 0.0

    unsupported = find_unsupported_claims(
        answer=answer,
        contexts=contexts,
        support_threshold=support_threshold,
    )

    supported_count = len(claims) - len(unsupported)

    return round(
        supported_count / len(claims),
        4,
    )

def calculate_answer_completeness(
    *,
    question: str,
    answer: str,
) -> float:
    """
    Estimate whether the answer addresses the important terms
    in the question.
    """

    return token_overlap_score(
        answer,
        question,
    )

_CITATION_PATTERN = re.compile(
    r"\[source:\s*[^\]]+\]",
    re.IGNORECASE,
)

def calculate_citation_coverage(
    *,
    answer: str,
) -> float:
    """
    Estimate the percentage of answer claims containing citations.
    """

    claims = split_claims(answer)

    if not claims:
        return 0.0

    cited_claims = sum(
        1
        for claim in claims
        if _CITATION_PATTERN.search(claim)
    )

    return round(
        cited_claims / len(claims),
        4,
    )

def classify_hallucination_risk(
    *,
    groundedness: float,
    unsupported_claim_count: int,
) -> str:
    """
    Classify hallucination risk using deterministic thresholds.
    """

    if groundedness >= 0.85 and unsupported_claim_count == 0:
        return "low"

    if groundedness >= 0.60:
        return "medium"

    return "high"

def calculate_overall_score(
    *,
    context_relevance: float,
    groundedness: float,
    completeness: float,
    citation_coverage: float,
) -> float:
    """
    Compute a weighted RAG quality score.

    Groundedness receives the highest weight because unsupported
    factual claims are the greatest trust risk.
    """

    score = (
        context_relevance * 0.20
        + groundedness * 0.40
        + completeness * 0.20
        + citation_coverage * 0.20
    )

    return round(score, 4)

def evaluate_rag_response(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    support_threshold: float = 0.35,
) -> RAGEvaluationResult:
    """
    Evaluate one RAG response using deterministic quality metrics.
    """

    context_relevance = calculate_context_relevance(
        question=question,
        contexts=contexts,
    )

    unsupported_claims = find_unsupported_claims(
        answer=answer,
        contexts=contexts,
        support_threshold=support_threshold,
    )

    groundedness = calculate_answer_groundedness(
        answer=answer,
        contexts=contexts,
        support_threshold=support_threshold,
    )

    completeness = calculate_answer_completeness(
        question=question,
        answer=answer,
    )

    citation_coverage = calculate_citation_coverage(
        answer=answer,
    )

    hallucination_risk = classify_hallucination_risk(
        groundedness=groundedness,
        unsupported_claim_count=len(unsupported_claims),
    )

    overall_score = calculate_overall_score(
        context_relevance=context_relevance,
        groundedness=groundedness,
        completeness=completeness,
        citation_coverage=citation_coverage,
    )

    return RAGEvaluationResult(
        context_relevance=context_relevance,
        answer_groundedness=groundedness,
        answer_completeness=completeness,
        citation_coverage=citation_coverage,
        hallucination_risk=hallucination_risk,
        unsupported_claims=unsupported_claims,
        overall_score=overall_score,
    )
