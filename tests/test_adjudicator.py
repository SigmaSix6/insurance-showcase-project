"""Unit tests for extraction and adjudication."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.adjudicator import adjudicate
from src.extractor import extract_claim_fields, load_document_text
from src.models import Decision, PolicyParameters

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_docs"
AS_OF = date(2026, 7, 14)


def test_extract_approved_sample():
    text = load_document_text(SAMPLES / "sample_approved_outpatient.txt")
    claim = extract_claim_fields(text)
    assert claim.patient_name == "Jordan Hale"
    assert claim.policy_number == "POL-88421A"
    assert claim.amount_claimed == 480.0
    assert claim.service_date == date(2026, 5, 12)
    assert claim.diagnosis_code == "J06.9"
    assert claim.provider_npi == "1678954321"
    assert claim.confidence >= 0.7


def test_adjudicate_approved():
    text = load_document_text(SAMPLES / "sample_approved_outpatient.txt")
    claim = extract_claim_fields(text)
    result = adjudicate(claim, PolicyParameters(), as_of=AS_OF)
    assert result.decision in {Decision.APPROVED, Decision.PARTIAL}
    assert result.reimbursable_amount > 0


def test_adjudicate_late_filing_denied():
    text = load_document_text(SAMPLES / "sample_denied_late_filing.txt")
    claim = extract_claim_fields(text)
    result = adjudicate(claim, PolicyParameters(max_claim_age_days=90), as_of=AS_OF)
    assert result.decision == Decision.DENIED
    assert any(r.rule_id == "TIME-003" and not r.passed for r in result.rules)


def test_adjudicate_excluded_service():
    text = load_document_text(SAMPLES / "sample_denied_excluded.txt")
    claim = extract_claim_fields(text)
    result = adjudicate(claim, PolicyParameters(), as_of=AS_OF)
    assert result.decision == Decision.DENIED
    assert any(r.rule_id == "COV-001" and not r.passed for r in result.rules)


def test_max_reimbursement_cap():
    text = load_document_text(SAMPLES / "sample_partial_over_cap.txt")
    claim = extract_claim_fields(text)
    params = PolicyParameters(max_reimbursement=5000.0, deductible=250.0, coinsurance_percent=80.0)
    result = adjudicate(claim, params, as_of=AS_OF)
    assert result.decision in {Decision.PARTIAL, Decision.NEEDS_REVIEW, Decision.APPROVED}
    assert result.reimbursable_amount <= 5000.0
    assert any(r.rule_id == "AMT-002" for r in result.rules)


def test_parameter_tweak_extends_claim_window():
    text = load_document_text(SAMPLES / "sample_denied_late_filing.txt")
    claim = extract_claim_fields(text)
    denied = adjudicate(claim, PolicyParameters(max_claim_age_days=90), as_of=AS_OF)
    allowed = adjudicate(claim, PolicyParameters(max_claim_age_days=400), as_of=AS_OF)
    assert denied.decision == Decision.DENIED
    assert allowed.decision != Decision.DENIED or allowed.reimbursable_amount >= 0
    # With a long window the late-filing rule should pass
    assert any(r.rule_id == "TIME-003" and r.passed for r in allowed.rules)
