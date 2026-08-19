"""Rule-based claim adjudication engine with tunable policy parameters."""

from __future__ import annotations

import re
from datetime import date

from .models import (
    AdjudicationResult,
    Decision,
    ExtractedClaim,
    PolicyParameters,
    RuleResult,
)


def _keyword_hit(blob: str, keyword: str) -> bool:
    """Match excluded keywords as whole words/phrases, skipping simple negations."""
    escaped = re.escape(keyword.lower())
    pattern = (
        rf"(?<![a-z0-9])(?<!\bno\s)(?<!\bnot\s)(?<!\bwithout\s)"
        rf"{escaped}(?![a-z0-9])"
    )
    return re.search(pattern, blob) is not None


def adjudicate(
    claim: ExtractedClaim,
    params: PolicyParameters,
    *,
    as_of: date | None = None,
) -> AdjudicationResult:
    """Apply policy rules and compute reimbursement / decision."""
    today = as_of or date.today()
    rules: list[RuleResult] = []
    rationale: list[str] = []

    amount = claim.amount_claimed or 0.0
    reimbursable = amount
    patient_pay = 0.0
    critical_failures = 0
    soft_flags = 0

    # --- Extraction quality -------------------------------------------------
    if claim.confidence < params.min_extraction_confidence:
        critical_failures += 1
        rules.append(
            RuleResult(
                rule_id="EXT-001",
                name="Extraction confidence",
                passed=False,
                severity="critical",
                message=(
                    f"Extraction confidence {claim.confidence:.0%} is below "
                    f"minimum {params.min_extraction_confidence:.0%}."
                ),
            )
        )
    else:
        rules.append(
            RuleResult(
                rule_id="EXT-001",
                name="Extraction confidence",
                passed=True,
                severity="info",
                message=f"Extraction confidence {claim.confidence:.0%} meets threshold.",
            )
        )

    # --- Required identity fields -------------------------------------------
    if not claim.patient_name or not claim.policy_number:
        critical_failures += 1
        rules.append(
            RuleResult(
                rule_id="ID-001",
                name="Member identification",
                passed=False,
                severity="critical",
                message="Patient name and/or policy number missing.",
            )
        )
    else:
        rules.append(
            RuleResult(
                rule_id="ID-001",
                name="Member identification",
                passed=True,
                severity="info",
                message="Member identity fields present.",
            )
        )

    if params.require_provider_npi and not claim.provider_npi:
        critical_failures += 1
        rules.append(
            RuleResult(
                rule_id="ID-002",
                name="Provider NPI required",
                passed=False,
                severity="critical",
                message="Valid 10-digit provider NPI is required by current parameters.",
            )
        )
    elif claim.provider_npi:
        rules.append(
            RuleResult(
                rule_id="ID-002",
                name="Provider NPI required",
                passed=True,
                severity="info",
                message=f"Provider NPI {claim.provider_npi} accepted.",
            )
        )

    if params.require_diagnosis_code and not claim.diagnosis_code:
        critical_failures += 1
        rules.append(
            RuleResult(
                rule_id="CLN-001",
                name="Diagnosis code required",
                passed=False,
                severity="critical",
                message="ICD-10 diagnosis code is required.",
            )
        )
    elif claim.diagnosis_code:
        rules.append(
            RuleResult(
                rule_id="CLN-001",
                name="Diagnosis code required",
                passed=True,
                severity="info",
                message=f"Diagnosis {claim.diagnosis_code} present.",
            )
        )

    # --- Timing rules -------------------------------------------------------
    service_date = claim.service_date
    claim_date = claim.claim_date or claim.service_date

    if not service_date:
        critical_failures += 1
        rules.append(
            RuleResult(
                rule_id="TIME-001",
                name="Service date present",
                passed=False,
                severity="critical",
                message="Date of service could not be determined.",
            )
        )
    else:
        if service_date < params.policy_effective_date:
            critical_failures += 1
            rules.append(
                RuleResult(
                    rule_id="TIME-002",
                    name="Policy effective window",
                    passed=False,
                    severity="critical",
                    message=(
                        f"Service date {service_date.isoformat()} is before policy "
                        f"effective date {params.policy_effective_date.isoformat()}."
                    ),
                )
            )
        elif params.policy_end_date and service_date > params.policy_end_date:
            critical_failures += 1
            rules.append(
                RuleResult(
                    rule_id="TIME-002",
                    name="Policy effective window",
                    passed=False,
                    severity="critical",
                    message=(
                        f"Service date {service_date.isoformat()} is after policy "
                        f"end date {params.policy_end_date.isoformat()}."
                    ),
                )
            )
        else:
            rules.append(
                RuleResult(
                    rule_id="TIME-002",
                    name="Policy effective window",
                    passed=True,
                    severity="info",
                    message="Service date falls within the policy period.",
                )
            )

        if claim_date:
            age_days = (claim_date - service_date).days
            # Also allow filing lag measured from today if claim_date == service_date
            filing_lag = (today - service_date).days
            effective_age = max(age_days, filing_lag)
            if effective_age > params.max_claim_age_days:
                critical_failures += 1
                rules.append(
                    RuleResult(
                        rule_id="TIME-003",
                        name="Maximum claim age",
                        passed=False,
                        severity="critical",
                        message=(
                            f"Claim is {effective_age} days after service; "
                            f"maximum allowed is {params.max_claim_age_days} days."
                        ),
                    )
                )
            else:
                rules.append(
                    RuleResult(
                        rule_id="TIME-003",
                        name="Maximum claim age",
                        passed=True,
                        severity="info",
                        message=(
                            f"Claim age {effective_age} days is within "
                            f"{params.max_claim_age_days}-day limit."
                        ),
                    )
                )

    # --- Amount / coverage rules --------------------------------------------
    if amount < params.min_claim_amount:
        critical_failures += 1
        rules.append(
            RuleResult(
                rule_id="AMT-001",
                name="Minimum claim amount",
                passed=False,
                severity="critical",
                message=(
                    f"Claimed ${amount:,.2f} is below minimum "
                    f"${params.min_claim_amount:,.2f}."
                ),
            )
        )
    else:
        rules.append(
            RuleResult(
                rule_id="AMT-001",
                name="Minimum claim amount",
                passed=True,
                severity="info",
                message=f"Claimed amount ${amount:,.2f} meets minimum.",
            )
        )

    blob = " ".join(
        filter(
            None,
            [
                claim.category or "",
                claim.description or "",
                claim.raw_text[:2000],
            ],
        )
    ).lower()

    excluded_hit = next(
        (kw for kw in params.excluded_keywords if _keyword_hit(blob, kw)), None
    )
    if excluded_hit:
        critical_failures += 1
        rules.append(
            RuleResult(
                rule_id="COV-001",
                name="Excluded services",
                passed=False,
                severity="critical",
                message=f"Service matches excluded keyword: '{excluded_hit}'.",
            )
        )
    else:
        rules.append(
            RuleResult(
                rule_id="COV-001",
                name="Excluded services",
                passed=True,
                severity="info",
                message="No excluded-service keywords detected.",
            )
        )

    covered = {c.lower() for c in params.covered_categories}
    if claim.category and claim.category.lower() not in covered:
        soft_flags += 1
        rules.append(
            RuleResult(
                rule_id="COV-002",
                name="Covered category",
                passed=False,
                severity="warning",
                message=(
                    f"Category '{claim.category}' is not in the covered list; "
                    "flagged for review."
                ),
            )
        )
    else:
        rules.append(
            RuleResult(
                rule_id="COV-002",
                name="Covered category",
                passed=True,
                severity="info",
                message=(
                    f"Category '{claim.category or 'unspecified'}' is covered "
                    "or not restricted."
                ),
            )
        )

    # --- Benefit calculation (only meaningful if not hard-denied) -----------
    if critical_failures == 0 and amount > 0:
        after_deductible = max(amount - params.deductible, 0.0)
        patient_pay = min(amount, params.deductible)
        plan_share = after_deductible * (params.coinsurance_percent / 100.0)
        patient_coinsurance = after_deductible - plan_share
        patient_pay += patient_coinsurance
        reimbursable = plan_share

        if reimbursable > params.max_reimbursement:
            clipped = reimbursable - params.max_reimbursement
            reimbursable = params.max_reimbursement
            patient_pay += clipped
            soft_flags += 1
            rules.append(
                RuleResult(
                    rule_id="AMT-002",
                    name="Maximum reimbursement cap",
                    passed=False,
                    severity="warning",
                    message=(
                        f"Plan share capped at ${params.max_reimbursement:,.2f}; "
                        f"${clipped:,.2f} shifted to patient."
                    ),
                    impact_amount=-clipped,
                )
            )
        else:
            rules.append(
                RuleResult(
                    rule_id="AMT-002",
                    name="Maximum reimbursement cap",
                    passed=True,
                    severity="info",
                    message=(
                        f"Reimbursement ${reimbursable:,.2f} is within "
                        f"${params.max_reimbursement:,.2f} cap."
                    ),
                )
            )

        rules.append(
            RuleResult(
                rule_id="AMT-003",
                name="Deductible & coinsurance",
                passed=True,
                severity="info",
                message=(
                    f"Applied ${params.deductible:,.2f} deductible and "
                    f"{params.coinsurance_percent:.0f}% coinsurance."
                ),
                impact_amount=reimbursable,
            )
        )

    # --- Decision -----------------------------------------------------------
    if critical_failures > 0:
        decision = Decision.DENIED
        reimbursable = 0.0
        patient_pay = amount
        rationale.append(
            f"Denied due to {critical_failures} critical rule failure(s)."
        )
    elif soft_flags > 0 and claim.confidence < params.auto_approve_threshold:
        decision = Decision.NEEDS_REVIEW
        rationale.append(
            "Soft policy warnings plus moderate extraction confidence — "
            "route to human review."
        )
    elif reimbursable <= 0:
        decision = Decision.DENIED
        rationale.append("No reimbursable amount after benefits application.")
    elif reimbursable < amount and patient_pay > 0:
        decision = Decision.PARTIAL
        rationale.append(
            f"Partial approval: plan pays ${reimbursable:,.2f}; "
            f"patient responsibility ${patient_pay:,.2f}."
        )
    else:
        decision = Decision.APPROVED
        rationale.append(
            f"Approved for ${reimbursable:,.2f} under current policy parameters."
        )

    if claim.extraction_notes:
        rationale.extend(claim.extraction_notes)

    # Blend rule pass-rate with extraction confidence
    passed = sum(1 for r in rules if r.passed)
    rule_score = passed / len(rules) if rules else 0.0
    confidence = round(0.6 * claim.confidence + 0.4 * rule_score, 3)

    return AdjudicationResult(
        decision=decision,
        reimbursable_amount=round(reimbursable, 2),
        patient_responsibility=round(patient_pay, 2),
        confidence=confidence,
        rules=rules,
        rationale=rationale,
        claim=claim,
        parameters_snapshot=params.to_dict(),
    )
