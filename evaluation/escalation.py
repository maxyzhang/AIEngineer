from dataclasses import dataclass

@dataclass
class EscalationDecision:
    escalate: bool
    reason: str
    confidence: float
    severity: str


class EscalationPolicy:
    def __init__(
        self,
        escalation_threshold: float = 0.65,
    ):
        self.escalation_threshold = escalation_threshold

    def evaluate(
        self,
        confidence: float,
        recovered: bool,
        stop_reason: str,
    ) -> EscalationDecision:

        if (
            not recovered
            and confidence < self.escalation_threshold
        ):
            return EscalationDecision(
                escalate=True,
                reason=stop_reason,
                confidence=confidence,
                severity="high",
            )

        return EscalationDecision(
            escalate=False,
            reason="confidence_acceptable",
            confidence=confidence,
            severity="none",
        )