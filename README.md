# Medical Entity Extraction (NER) – AI Medical Transcription Tool

This module extracts **medical entities** from clinical text.

- **Entity labels**: `DRUG`, `DISEASE`, `PROCEDURE`, `ANATOMY`
- **Two engines**:
  1) **Rules (default)** — spaCy `EntityRuler` (case-insensitive). **No extra model downloads** required.
  2) **Optional scispaCy** — prebuilt clinical NER models with optional UMLS linking. *Only needed if you explicitly enable it, (We will not need to enable it for this class guys).*

---

## Quickstart for everyone (rule-based, default)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
