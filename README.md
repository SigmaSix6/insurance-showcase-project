# ClaimLens — Insurance AI Adjudicator

Portfolio showcase: upload insurance documents, extract structured claim fields, and adjudicate them against **tunable policy parameters**.

![Python](https://img.shields.io/badge/Python-3.10%2B-0f766e)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-1e3a5f)
![License](https://img.shields.io/badge/License-MIT-slategray)

## What it demonstrates

| Capability                | How                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Document intake           | PDF / TXT / image upload                                                                                     |
| AI-style extraction       | Regex + heuristics with per-field confidence; optional Gemini enrichment                                     |
| Adjudication              | Transparent rule engine (timing, coverage, deductible, caps, quality gates)                                  |
| Interactive policy design | Sidebar controls for max claim age, reimbursement cap, deductible, coinsurance, covered categories, and more |
| Explainability            | Per-rule pass/fail trail + JSON export                                                                       |

## Quick start

```bash
cd showcase-project
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The UI uses Streamlit **dark mode** by default (see `.streamlit/config.toml`).

### OpenAI / GPT extraction

Create a `.env` file in the project root:

```env
openai_api_key=sk-your-key-here
# optional
openai_model=gpt-4o-mini
```

The app loads this automatically on startup. Claim documents are parsed with heuristics first, then sent to GPT to fill or refine missing fields.

You can also set `OPENAI_API_KEY` in the shell instead of using `.env`. Without a key, the app runs fully offline using the deterministic extractor.

## Sample claims

| File                                         | Expected demo outcome (default params)      |
| -------------------------------------------- | ------------------------------------------- |
| `sample_docs/sample_approved_outpatient.txt` | Approved / partial after deductible         |
| `sample_docs/sample_denied_late_filing.txt`  | Denied — exceeds max claim age              |
| `sample_docs/sample_denied_excluded.txt`     | Denied — excluded cosmetic/elective service |
| `sample_docs/sample_partial_over_cap.txt`    | Capped at max reimbursement                 |

Try raising **Max claim age** on the late-filing sample and re-running — the decision flips.

## Tunable parameters

- Max claim age (days)
- Policy effective / end dates
- Max reimbursement
- Deductible & plan coinsurance %
- Minimum claim amount
- Covered categories
- Excluded keywords
- Require provider NPI / diagnosis code
- Extraction & auto-approve confidence thresholds

## Project layout

```
app.py                 # Streamlit UI
src/
  models.py            # PolicyParameters, ExtractedClaim, AdjudicationResult
  extractor.py         # Document text + field extraction
  adjudicator.py       # Rules engine
sample_docs/           # Ready-to-demo claim files
tests/                 # pytest coverage for core paths
```

## Tests

```bash
pytest -q
```

## Architecture

```
Upload → Text extract → Field extract (+ optional LLM)
                              ↓
                     PolicyParameters (UI)
                              ↓
                      Rules adjudication
                              ↓
              Decision + amounts + rule trail + JSON
```

## License

MIT — built as a portfolio demonstration of applied AI + domain rules engineering.
