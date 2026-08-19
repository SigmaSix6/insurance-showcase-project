"""
Insurance AI Adjudicator — Streamlit portfolio demo.

Upload an insurance claim document, tune policy parameters, and watch
rule-based (optionally LLM-enriched) adjudication explain its decision.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st

from src.adjudicator import adjudicate
from src.config import load_env, get_gemini_model, gemini_configured
from src.extractor import extract_claim_fields, load_document_text
from src.models import Decision, PolicyParameters

load_env()

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_docs"

DECISION_STYLES = {
    Decision.APPROVED: ("Approved", "#5eead4", "#0f2f2c"),
    Decision.PARTIAL: ("Partial approval", "#fbbf24", "#3a2a0a"),
    Decision.DENIED: ("Denied", "#fca5a5", "#3b1219"),
    Decision.NEEDS_REVIEW: ("Needs review", "#93c5fd", "#13233f"),
}

st.set_page_config(
    page_title="ClaimLens — Insurance AI Adjudicator",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
  --cl-bg: #0b1215;
  --cl-surface: #152025;
  --cl-surface-2: #1c2a30;
  --cl-border: rgba(148, 163, 184, 0.18);
  --cl-text: #e8eef2;
  --cl-muted: #9aafb8;
  --cl-accent: #2dd4bf;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--cl-text);
}
.stApp {
  background:
    radial-gradient(1000px 520px at 8% -12%, rgba(45, 212, 191, 0.12) 0%, transparent 55%),
    radial-gradient(900px 480px at 100% 0%, rgba(56, 120, 180, 0.16) 0%, transparent 52%),
    linear-gradient(180deg, #0b1215 0%, #0e171b 100%);
  color: var(--cl-text);
}
h1, h2, h3, .brand-title {
  font-family: 'Fraunces', Georgia, serif !important;
  letter-spacing: -0.02em;
  color: var(--cl-text) !important;
}
p, label, span, li, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] {
  color: var(--cl-text);
}
.stMarkdown, .stText, [data-testid="stMarkdownContainer"] p {
  color: var(--cl-text) !important;
}
.hero {
  padding: 1.4rem 1.6rem 1.1rem;
  border-radius: 18px;
  background: linear-gradient(135deg, #0d2a2a 0%, #123528 45%, #132840 100%);
  color: #f8fafc;
  margin-bottom: 1.2rem;
  border: 1px solid var(--cl-border);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
}
.hero .brand {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 2.2rem;
  font-weight: 700;
  margin: 0;
  line-height: 1.1;
  color: #f8fafc;
}
.hero .tag {
  margin: 0.55rem 0 0;
  opacity: 0.92;
  font-size: 1.02rem;
  max-width: 48rem;
  color: #d7e6ea;
}
.metric-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin: 0.5rem 0 1.25rem;
}
.metric {
  background: var(--cl-surface);
  border: 1px solid var(--cl-border);
  border-radius: 14px;
  padding: 0.9rem 1rem;
}
.metric .label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--cl-muted);
  margin-bottom: 0.25rem;
}
.metric .value {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.45rem;
  color: var(--cl-text);
}
.decision-banner {
  border-radius: 14px;
  padding: 1rem 1.2rem;
  margin: 0.4rem 0 1rem;
  border: 1px solid transparent;
}
.decision-meta {
  color: var(--cl-muted) !important;
  margin-top: 0.35rem;
}
.rule-ok { color: #5eead4 !important; }
.rule-bad { color: #fca5a5 !important; }
.rule-warn { color: #fbbf24 !important; }
.rule-msg { color: var(--cl-muted) !important; }

section[data-testid="stSidebar"] {
  background: #0e171b !important;
  border-right: 1px solid var(--cl-border);
}
section[data-testid="stSidebar"] * {
  color: var(--cl-text);
}
div[data-testid="stFileUploader"] {
  background: var(--cl-surface);
  border: 1px solid var(--cl-border);
  border-radius: 12px;
  padding: 0.4rem;
}
div[data-testid="stExpander"] {
  background: var(--cl-surface);
  border: 1px solid var(--cl-border);
  border-radius: 12px;
}
[data-testid="stTable"] {
  color: var(--cl-text);
}
[data-testid="stTable"] table {
  color: var(--cl-text);
}
[data-testid="stTable"] th, [data-testid="stTable"] td {
  color: var(--cl-text) !important;
  border-color: var(--cl-border) !important;
}
.stAlert { color: var(--cl-text); }
</style>
""",
    unsafe_allow_html=True,
)


def _default_params() -> PolicyParameters:
    return PolicyParameters()


def render_provider_status() -> None:
    st.sidebar.markdown("### AI provider")
    if gemini_configured():
        st.sidebar.success(f"Gemini connected · `{get_gemini_model()}`")
        st.sidebar.caption(
            "Claim fields are extracted with heuristics, then refined by GPT when useful."
        )
    else:
        st.sidebar.warning("Gemini not configured")
        st.sidebar.caption(
            "Add `GEMINI_API_KEY` to `.env` in the project root to enable Gemini extraction."
        )
    # if openai_configured():
    #     st.sidebar.success(f"OpenAI connected · `{get_openai_model()}`")
    #     st.sidebar.caption(
    #         "Claim fields are extracted with heuristics, then refined by GPT when useful."
    #     )
    # else:
    #     st.sidebar.warning("OpenAI not configured")
    #     st.sidebar.caption(
    #         "Add `OPENAI_API_KEY` to `.env` in the project root to enable GPT extraction."
    #     )


def sidebar_params() -> PolicyParameters:
    render_provider_status()
    st.sidebar.markdown("### Policy parameters")
    st.sidebar.caption("Hover the ⓘ icons for what each setting changes. Re-run adjudication after tweaks.")

    max_age = st.sidebar.slider(
        "Max claim age (days)",
        15,
        365,
        90,
        5,
        help=(
            "Maximum days allowed between the date of service and when the claim is "
            "evaluated. Claims older than this are denied (rule TIME-003)."
        ),
    )
    max_reimb = st.sidebar.number_input(
        "Max reimbursement ($)",
        min_value=100.0,
        max_value=50000.0,
        value=5000.0,
        step=100.0,
        help=(
            "Hard cap on what the plan will pay for a single claim. Anything above "
            "this is shifted to patient responsibility (rule AMT-002)."
        ),
    )
    deductible = st.sidebar.number_input(
        "Deductible ($)",
        min_value=0.0,
        max_value=10000.0,
        value=250.0,
        step=25.0,
        help=(
            "Amount the patient pays first before plan coinsurance applies. "
            "Reduces reimbursable amount dollar-for-dollar up to this value."
        ),
    )
    coinsurance = st.sidebar.slider(
        "Plan coinsurance (%)",
        50,
        100,
        80,
        5,
        help=(
            "Share of the remaining amount (after deductible) that the plan pays. "
            "Example: 80% means the plan pays 80% and the patient pays 20%."
        ),
    )
    min_amount = st.sidebar.number_input(
        "Minimum claim amount ($)",
        min_value=0.0,
        max_value=1000.0,
        value=25.0,
        step=5.0,
        help=(
            "Claims billed below this threshold are denied as too small to process "
            "(rule AMT-001)."
        ),
    )

    st.sidebar.markdown("#### Policy window")
    effective = st.sidebar.date_input(
        "Policy effective date",
        value=date(2024, 1, 1),
        help=(
            "Earliest covered service date. Services before this date fall outside "
            "the policy period and are denied (rule TIME-002)."
        ),
    )
    use_end = st.sidebar.checkbox(
        "Set policy end date",
        value=False,
        help="Enable to enforce a hard end to coverage. Services after that date are denied.",
    )
    end_date = None
    if use_end:
        end_date = st.sidebar.date_input(
            "Policy end date",
            value=date(2026, 12, 31),
            help=(
                "Last covered service date. If set, services after this date fail "
                "the policy window check (rule TIME-002)."
            ),
        )

    st.sidebar.markdown("#### Coverage & quality gates")
    categories = st.sidebar.multiselect(
        "Covered categories",
        options=[
            "outpatient",
            "emergency",
            "diagnostics",
            "pharmacy",
            "preventive",
            "specialist",
            "inpatient",
            "cosmetic",
        ],
        default=[
            "outpatient",
            "emergency",
            "diagnostics",
            "pharmacy",
            "preventive",
            "specialist",
        ],
        help=(
            "Service categories the plan covers. Claims in a category not listed "
            "here are flagged for review (rule COV-002)."
        ),
    )
    excluded_raw = st.sidebar.text_area(
        "Excluded keywords (comma-separated)",
        value="cosmetic, elective, experimental, dental veneers",
        help=(
            "If any of these words/phrases appear in the claim text (and are not "
            "negated), the claim is denied as an excluded service (rule COV-001)."
        ),
    )
    excluded = [p.strip() for p in excluded_raw.split(",") if p.strip()]

    require_npi = st.sidebar.checkbox(
        "Require provider NPI",
        value=True,
        help=(
            "When on, a valid 10-digit provider NPI must be extracted or the claim "
            "is denied (rule ID-002)."
        ),
    )
    require_dx = st.sidebar.checkbox(
        "Require diagnosis code",
        value=True,
        help=(
            "When on, an ICD-10 diagnosis code must be present or the claim is "
            "denied (rule CLN-001)."
        ),
    )
    min_conf = st.sidebar.slider(
        "Min extraction confidence",
        0.2,
        0.95,
        0.55,
        0.05,
        help=(
            "Minimum overall field-extraction confidence required to adjudicate. "
            "Below this, the claim is denied for poor document quality (rule EXT-001)."
        ),
    )
    auto_approve = st.sidebar.slider(
        "Auto-approve confidence",
        0.5,
        0.99,
        0.85,
        0.01,
        help=(
            "If soft warnings exist and extraction confidence is below this value, "
            "the claim is routed to Needs Review instead of auto-approving."
        ),
    )

    return PolicyParameters(
        max_claim_age_days=int(max_age),
        policy_effective_date=effective if isinstance(effective, date) else date(2024, 1, 1),
        policy_end_date=end_date if isinstance(end_date, date) else None,
        max_reimbursement=float(max_reimb),
        deductible=float(deductible),
        coinsurance_percent=float(coinsurance),
        min_claim_amount=float(min_amount),
        covered_categories=categories or ["outpatient"],
        excluded_keywords=excluded,
        require_provider_npi=require_npi,
        require_diagnosis_code=require_dx,
        min_extraction_confidence=float(min_conf),
        auto_approve_threshold=float(auto_approve),
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <p class="brand">Simple Document Adjudicator</p>
          <p class="tag">
            Upload insurance documents, extract claim fields with
            confidence scoring, then adjudicate against tunable policy parameters.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_sample(name: str) -> tuple[str, str]:
    path = SAMPLE_DIR / name
    return path.read_text(encoding="utf-8"), name


def main() -> None:
    render_hero()
    params = sidebar_params()
    default_text = "Select a sample document or upload your own"

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("1. Claim document")
        sample_choice = st.selectbox(
            "Load a sample claim",
            options=[default_text, *sorted(p.name for p in SAMPLE_DIR.glob("*"))],
        )
        uploaded = st.file_uploader(
            "Upload PDF, TXT, or image",
            type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
            help="Sample TXT claims are included for a zero-setup demo.",
        )

        text: str | None = None
        source_name = ""

        if sample_choice != default_text:
            text, source_name = load_sample(sample_choice)
            st.success(f"Loaded sample: {source_name}")
        elif uploaded is not None:
            raw = uploaded.getvalue()
            source_name = uploaded.name
            text = load_document_text(raw, filename=uploaded.name)

        if text:
            with st.expander("Document text", expanded=False):
                st.text(text[:8000])

            if st.button("Adjudicate claim", type="primary", use_container_width=True):
                with st.spinner("Extracting fields and running rules engine…"):
                    claim = extract_claim_fields(text)
                    # Freeze "today" for sample docs so age rules demo cleanly
                    as_of = date(2026, 7, 14) if source_name.startswith("sample_") else date.today()
                    result = adjudicate(claim, params, as_of=as_of)
                    st.session_state["result"] = result
                    st.session_state["source_name"] = source_name
        else:
            st.info("Upload a document or pick a sample to begin.")

    with right:
        st.subheader("2. Adjudication result")
        result = st.session_state.get("result")
        if not result:
            st.markdown(
                "Results appear here after adjudication — decision, dollars, "
                "and a per-rule explanation trail."
            )
            return

        label, fg, bg = DECISION_STYLES[result.decision]
        st.markdown(
            f"""
            <div class="decision-banner" style="background:{bg}; border-color:{fg}55; color:{fg};">
              <strong style="font-size:1.25rem;">{label}</strong>
              <div class="decision-meta">
                Confidence {result.confidence:.0%} · Source: {st.session_state.get('source_name', 'upload')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-strip">
              <div class="metric"><div class="label">Plan pays</div>
                <div class="value">${result.reimbursable_amount:,.2f}</div></div>
              <div class="metric"><div class="label">Patient pays</div>
                <div class="value">${result.patient_responsibility:,.2f}</div></div>
              <div class="metric"><div class="label">Claimed</div>
                <div class="value">${(result.claim.amount_claimed or 0):,.2f}</div></div>
              <div class="metric"><div class="label">Rules passed</div>
                <div class="value">{sum(1 for r in result.rules if r.passed)}/{len(result.rules)}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Extracted fields")
        claim = result.claim
        fields = {
            "Patient": claim.patient_name,
            "Policy": claim.policy_number,
            "Provider": claim.provider_name,
            "NPI": claim.provider_npi,
            "Service date": claim.service_date.isoformat() if claim.service_date else None,
            "Claim date": claim.claim_date.isoformat() if claim.claim_date else None,
            "Amount": f"${claim.amount_claimed:,.2f}" if claim.amount_claimed is not None else None,
            "Diagnosis": claim.diagnosis_code,
            "Procedure": claim.procedure_code,
            "Category": claim.category,
            "Extraction confidence": f"{claim.confidence:.0%}",
        }
        st.table({"Field": list(fields.keys()), "Value": [v or "—" for v in fields.values()]})

        st.markdown("#### Rule trail")
        for rule in result.rules:
            if rule.passed:
                icon, css = "✓", "rule-ok"
            elif rule.severity == "warning":
                icon, css = "!", "rule-warn"
            else:
                icon, css = "✕", "rule-bad"
            st.markdown(
                f"<span class='{css}'><strong>{icon} {rule.rule_id} — {rule.name}</strong></span><br/>"
                f"<span class='rule-msg'>{rule.message}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Rationale")
        for line in result.rationale:
            st.write(f"• {line}")

        payload = result.to_dict()
        st.download_button(
            "Download adjudication JSON",
            data=json.dumps(payload, indent=2),
            file_name=f"adjudication_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
