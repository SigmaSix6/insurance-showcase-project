"""Document loading and claim-field extraction.

Uses regex + heuristics with per-field confidence scores so the app works
offline without an API key. Optionally enriches extraction via Gemini when
GEMINI_API_KEY is set.
"""

from __future__ import annotations

import io
import re
import json
from datetime import date, datetime
from pathlib import Path

from .config import get_gemini_api_key, get_gemini_model
from .models import ExtractedClaim

DATE_PATTERNS = [
    r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})",
    r"(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{2,4})",
    r"(?P<d>\d{1,2})-(?P<m>[A-Za-z]{3})-(?P<y>\d{4})",
]

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

CATEGORY_KEYWORDS = {
    "emergency": ["emergency", "er visit", "urgent care", "trauma"],
    "outpatient": ["outpatient", "clinic visit", "office visit"],
    "diagnostics": ["lab", "mri", "ct scan", "x-ray", "ultrasound", "diagnostic"],
    "pharmacy": ["pharmacy", "prescription", "rx ", "medication"],
    "preventive": ["preventive", "annual physical", "immunization", "screening"],
    "specialist": ["specialist", "cardiology", "orthopedic", "dermatology"],
    "cosmetic": ["cosmetic", "aesthetic", "veneers", "botox"],
}


def load_document_text(source: str | Path | bytes, filename: str = "") -> str:
    """Extract plain text from uploaded bytes or a file path."""
    name = filename.lower()
    if isinstance(source, (str, Path)):
        path = Path(source)
        name = path.name.lower()
        data = path.read_bytes()
    else:
        data = source

    if name.endswith(".pdf") or data[:4] == b"%PDF":
        return _extract_pdf(data)
    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")):
        return _extract_image(data)
    # Default: treat as UTF-8 text
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass
    return data.decode("utf-8", errors="replace")


def _extract_image(data: bytes) -> str:
    """Best-effort OCR via pytesseract when available; otherwise return a hint."""
    try:
        from PIL import Image
        import pytesseract

        image = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(image)
    except Exception:
        return (
            "[Image uploaded — OCR unavailable. Install pytesseract + Tesseract OCR, "
            "or upload a PDF/TXT claim form.]"
        )


def _parse_date(value: str) -> date | None:
    value = value.strip()
    for pattern in DATE_PATTERNS:
        match = re.fullmatch(pattern, value, flags=re.IGNORECASE)
        if not match:
            continue
        parts = match.groupdict()
        year = int(parts["y"])
        if year < 100:
            year += 2000
        month_raw = parts["m"]
        if month_raw.isdigit():
            month = int(month_raw)
        else:
            month = MONTHS.get(month_raw.lower()[:3])
            if not month:
                return None
        day = int(parts["d"])
        try:
            return date(year, month, day)
        except ValueError:
            return None
    # Fallback to fromisoformat-ish
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _find_labeled_value(text: str, labels: list[str]) -> tuple[str | None, float]:
    for label in labels:
        pattern = rf"{label}\s*[:\-]\s*(.+)$"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip(), 0.9
    return None, 0.0


def _find_date_near(text: str, labels: list[str]) -> tuple[date | None, float]:
    for label in labels:
        pattern = (
            rf"{label}\s*[:\-]?\s*"
            rf"((?:\d{{4}}-\d{{1,2}}-\d{{1,2}})|"
            rf"(?:\d{{1,2}}/\d{{1,2}}/\d{{2,4}})|"
            rf"(?:\d{{1,2}}-[A-Za-z]{{3}}-\d{{4}}))"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parsed = _parse_date(match.group(1))
            if parsed:
                return parsed, 0.92
    return None, 0.0


def _find_amount(text: str) -> tuple[float | None, float]:
    labeled = re.search(
        r"(?:total\s*(?:amount|charged|billed)?|amount\s*claimed|claim\s*amount|"
        r"billed\s*amount|charges?)\s*[:\-]?\s*\$?\s*([0-9,]+\.?\d{0,2})",
        text,
        flags=re.IGNORECASE,
    )
    if labeled:
        return float(labeled.group(1).replace(",", "")), 0.93

    amounts = re.findall(r"\$\s*([0-9,]+\.\d{2})", text)
    if amounts:
        values = [float(a.replace(",", "")) for a in amounts]
        return max(values), 0.7
    return None, 0.0


def _infer_category(text: str) -> tuple[str | None, float]:
    lower = text.lower()
    labeled, conf = _find_labeled_value(text, ["Category", "Service Category", "Type of Service"])
    if labeled:
        return labeled.lower().split()[0], conf

    best: str | None = None
    best_hits = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > best_hits:
            best = category
            best_hits = hits
    if best and best_hits:
        return best, min(0.55 + 0.1 * best_hits, 0.88)
    return None, 0.0


def extract_claim_fields(text: str) -> ExtractedClaim:
    """Heuristic extraction of claim fields from document text."""
    notes: list[str] = []
    field_conf: dict[str, float] = {}

    patient, c = _find_labeled_value(
        text, ["Patient Name", "Patient", "Member Name", "Insured Name"]
    )
    field_conf["patient_name"] = c

    policy, c = _find_labeled_value(
        text, ["Policy Number", "Policy #", "Member ID", "Policy ID"]
    )
    if not policy:
        match = re.search(r"\b(?:POL|MBR)[- ]?[A-Z0-9]{5,}\b", text, flags=re.IGNORECASE)
        if match:
            policy, c = match.group(0).upper(), 0.75
    field_conf["policy_number"] = c

    provider, c = _find_labeled_value(
        text, ["Provider", "Provider Name", "Facility", "Rendering Provider"]
    )
    field_conf["provider_name"] = c

    npi_match = re.search(r"\bNPI\s*[:#-]?\s*(\d{10})\b", text, flags=re.IGNORECASE)
    if not npi_match:
        npi_match = re.search(r"\b(\d{10})\b", text)
        npi_conf = 0.55 if npi_match else 0.0
    else:
        npi_conf = 0.95
    provider_npi = npi_match.group(1) if npi_match else None
    field_conf["provider_npi"] = npi_conf

    service_date, c = _find_date_near(
        text, ["Date of Service", "Service Date", "DOS", "Visit Date"]
    )
    field_conf["service_date"] = c

    claim_date, c = _find_date_near(
        text, ["Claim Date", "Date Submitted", "Submission Date", "Filing Date"]
    )
    if not claim_date:
        claim_date, c = service_date, field_conf.get("service_date", 0.0) * 0.6
        if claim_date:
            notes.append("Claim date inferred from service date.")
    field_conf["claim_date"] = c

    amount, c = _find_amount(text)
    field_conf["amount_claimed"] = c

    diagnosis, c = _find_labeled_value(
        text, ["Diagnosis Code", "ICD-10", "Diagnosis", "Dx Code"]
    )
    if diagnosis:
        dx_match = re.search(r"[A-Z]\d{2}(?:\.\d{1,4})?", diagnosis.upper())
        diagnosis = dx_match.group(0) if dx_match else diagnosis.split()[0]
    else:
        dx_match = re.search(r"\b([A-Z]\d{2}(?:\.\d{1,4})?)\b", text)
        if dx_match:
            diagnosis, c = dx_match.group(1), 0.7
    field_conf["diagnosis_code"] = c

    procedure, c = _find_labeled_value(
        text, ["Procedure Code", "CPT", "HCPCS", "Procedure"]
    )
    if procedure:
        cpt = re.search(r"\b(\d{5})\b", procedure)
        procedure = cpt.group(1) if cpt else procedure.split()[0]
    else:
        cpt = re.search(r"\bCPT\s*[:#-]?\s*(\d{5})\b", text, flags=re.IGNORECASE)
        if cpt:
            procedure, c = cpt.group(1), 0.9
    field_conf["procedure_code"] = c

    category, c = _infer_category(text)
    field_conf["category"] = c

    description, c = _find_labeled_value(
        text, ["Description", "Service Description", "Narrative", "Notes"]
    )
    field_conf["description"] = c

    weighed = [v for v in field_conf.values() if v > 0]
    overall = sum(weighed) / len(weighed) if weighed else 0.0

    # Soft boost when core money/date/identity fields are strong
    core = [
        field_conf.get("patient_name", 0),
        field_conf.get("amount_claimed", 0),
        field_conf.get("service_date", 0),
        field_conf.get("policy_number", 0),
    ]
    if all(v >= 0.7 for v in core):
        overall = min(overall + 0.08, 0.98)
        notes.append("Core claim fields extracted with high confidence.")

    claim = ExtractedClaim(
        raw_text=text,
        patient_name=patient,
        policy_number=policy,
        provider_name=provider,
        provider_npi=provider_npi,
        service_date=service_date,
        claim_date=claim_date,
        amount_claimed=amount,
        diagnosis_code=diagnosis,
        procedure_code=procedure,
        category=category,
        description=description,
        confidence=round(overall, 3),
        field_confidences={k: round(v, 3) for k, v in field_conf.items()},
        extraction_notes=notes,
    )
    return _maybe_llm_enrich(claim)

def _maybe_llm_enrich(claim: ExtractedClaim) -> ExtractedClaim:
    """Optionally refine extraction with Gemini if a key is available."""
    api_key = get_gemini_api_key()
    if not api_key:
        return claim

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = get_gemini_model()
        prompt = (
            "Extract insurance claim fields as compact JSON with keys: "
            "patient_name, policy_number, provider_name, provider_npi, "
            "service_date (YYYY-MM-DD), claim_date (YYYY-MM-DD), amount_claimed, "
            "diagnosis_code, procedure_code, category, description. "
            "Use null when unknown.\n\nDocument:\n"
            f"{claim.raw_text[:6000]}"
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You extract structured medical claim data. Reply with JSON only.",
                response_mime_type="application/json",
                temperature=0,
            ),
        )

        data = json.loads(response.text or "{}")
        mapping = {
            "patient_name": "patient_name",
            "policy_number": "policy_number",
            "provider_name": "provider_name",
            "provider_npi": "provider_npi",
            "diagnosis_code": "diagnosis_code",
            "procedure_code": "procedure_code",
            "category": "category",
            "description": "description",
        }
        for src, dest in mapping.items():
            value = data.get(src)
            if value and not getattr(claim, dest):
                setattr(claim, dest, str(value))
                claim.field_confidences[dest] = max(
                    claim.field_confidences.get(dest, 0.0), 0.9
                )

        if data.get("amount_claimed") is not None and claim.amount_claimed is None:
            claim.amount_claimed = float(data["amount_claimed"])
            claim.field_confidences["amount_claimed"] = 0.9

        for date_field in ("service_date", "claim_date"):
            raw = data.get(date_field)
            if raw and getattr(claim, date_field) is None:
                parsed = _parse_date(str(raw))
                if parsed:
                    setattr(claim, date_field, parsed)
                    claim.field_confidences[date_field] = 0.9

        claim.extraction_notes.append(f"Fields enriched via Gemini ({model}).")
        weighed = [v for v in claim.field_confidences.values() if v > 0]
        claim.confidence = round(sum(weighed) / len(weighed), 3) if weighed else claim.confidence
    except Exception as exc:  # noqa: BLE001 — optional path must never break demo
        claim.extraction_notes.append(f"LLM enrichment skipped: {exc}")
    return claim