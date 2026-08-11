from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RecoveryResult:
    confidence: float
    attempts: int
    recovered: bool
    stop_reason: str
    evidence: Any = None


class ConfidenceRecovery:
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        max_retries: int = 2,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

    def recover(
        self,
        query: str,
        confidence: float,
        retrieve_fn: Callable,
        evaluate_fn: Callable,
    ) -> RecoveryResult:
        """
        Retry retrieval when confidence is below the threshold.

        Args:
            query:
                Original user query.

            confidence:
                Initial confidence score between 0.0 and 1.0.

            retrieve_fn:
                Function used to retrieve additional evidence.
                Expected signature:

                    retrieve_fn(query=query, top_k=top_k)

            evaluate_fn:
                Function used to recalculate confidence.
                Expected signature:

                    evaluate_fn(query=query, evidence=evidence)

        Returns:
            RecoveryResult
        """

        # Confidence is already high enough.
        if confidence >= self.confidence_threshold:
            return RecoveryResult(
                confidence=confidence,
                attempts=0,
                recovered=True,
                stop_reason="confidence_threshold_met",
                evidence=None,
            )

        attempts = 0
        evidence = None

        while (
            confidence < self.confidence_threshold
            and attempts < self.max_retries
        ):
            attempts += 1

            # Increase retrieval depth with each retry.
            top_k = 5 + (attempts - 1) * 3

            evidence = retrieve_fn(
                query=query,
                top_k=top_k,
            )

            # Nothing useful was retrieved.
            if not evidence:
                return RecoveryResult(
                    confidence=confidence,
                    attempts=attempts,
                    recovered=False,
                    stop_reason="no_evidence",
                    evidence=evidence,
                )

            # Recalculate confidence using the new evidence.
            confidence = evaluate_fn(
                query=query,
                evidence=evidence,
            )

            # Confidence recovered successfully.
            if confidence >= self.confidence_threshold:
                return RecoveryResult(
                    confidence=confidence,
                    attempts=attempts,
                    recovered=True,
                    stop_reason="confidence_recovered",
                    evidence=evidence,
                )

        # Maximum retry count reached.
        return RecoveryResult(
            confidence=confidence,
            attempts=attempts,
            recovered=False,
            stop_reason="max_retries_reached",
            evidence=evidence,
        )