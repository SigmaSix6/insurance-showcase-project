"""Domain models for insurance claim adjudication."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any


class Decision(str, Enum):
    APPROVED = "approved"
    PARTIAL = "partial"
    DENIED = "denied"
    NEEDS_REVIEW = "needs_review"


@dataclass
class PolicyParameters:
    """Tunable adjudication knobs — exposed in the UI for demos."""

    # Timing
    max_claim_age_days: int = 90
    policy_effective_date: date = field(default_factory=lambda: date(2024, 1, 1))
    policy_end_date: date | None = None

    # Money
    max_reimbursement: float = 5000.0
    deductible: float = 250.0
    coinsurance_percent: float = 80.0  # plan pays this % after deductible
    min_claim_amount: float = 25.0

    # Coverage
    covered_categories: list[str] = field(
        default_factory=lambda: [
            "outpatient",
            "emergency",
            "diagnostics",
            "pharmacy",
            "preventive",
            "specialist",
        ]
    )
    excluded_keywords: list[str] = field(
        default_factory=lambda: [
            "cosmetic",
            "elective",
            "experimental",
            "experimental treatment",
            "dental veneers",
        ]
    )

    # Quality gates
    require_provider_npi: bool = True
    require_diagnosis_code: bool = True
    min_extraction_confidence: float = 0.55
    auto_approve_threshold: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy_effective_date"] = self.policy_effective_date.isoformat()
        data["policy_end_date"] = (
            self.policy_end_date.isoformat() if self.policy_end_date else None
        )
        return data


@dataclass
class ExtractedClaim:
    """Structured fields pulled from an uploaded insurance document."""

    raw_text: str
    patient_name: str | None = None
    policy_number: str | None = None
    provider_name: str | None = None
    provider_npi: str | None = None
    service_date: date | None = None
    claim_date: date | None = None
    amount_claimed: float | None = None
    diagnosis_code: str | None = None
    procedure_code: str | None = None
    category: str | None = None
    description: str | None = None
    confidence: float = 0.0
    field_confidences: dict[str, float] = field(default_factory=dict)
    extraction_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("service_date", "claim_date"):
            value = getattr(self, key)
            data[key] = value.isoformat() if value else None
        return data


@dataclass
class RuleResult:
    rule_id: str
    name: str
    passed: bool
    severity: str  # "critical" | "warning" | "info"
    message: str
    impact_amount: float = 0.0


@dataclass
class AdjudicationResult:
    decision: Decision
    reimbursable_amount: float
    patient_responsibility: float
    confidence: float
    rules: list[RuleResult] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    claim: ExtractedClaim | None = None
    parameters_snapshot: dict[str, Any] = field(default_factory=dict)
    adjudicated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .replace(tzinfo=None)
        .isoformat(timespec="seconds")
        + "Z"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reimbursable_amount": self.reimbursable_amount,
            "patient_responsibility": self.patient_responsibility,
            "confidence": self.confidence,
            "rules": [asdict(r) for r in self.rules],
            "rationale": self.rationale,
            "claim": self.claim.to_dict() if self.claim else None,
            "parameters_snapshot": self.parameters_snapshot,
            "adjudicated_at": self.adjudicated_at,
        }
