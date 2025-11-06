# Medical Entity Extraction (NER) – AI Medical Transcription Tool

This module extracts **medical entities** from clinical text:
- Labels: `DRUG`, `DISEASE`, `PROCEDURE`, `ANATOMY`
- Two engines:
  1) **Rules (default)**: spaCy `EntityRuler`, case-insensitive, no extra models
  2) **Optional scispaCy**: prebuilt clinical NER models + optional UMLS linking

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
