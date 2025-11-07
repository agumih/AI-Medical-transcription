# Medical Entity Extraction (NER) – AI Medical Transcription Tool

This module extracts **medical entities** from clinical text.

- **Entity labels**: `DRUG`, `DISEASE`, `PROCEDURE`, `ANATOMY`
- **Two engines**:
  1) **Rules (default)** — spaCy `EntityRuler` (case-insensitive). **No extra model downloads** required.
  2) **Optional scispaCy** — prebuilt clinical NER models with optional UMLS linking. *Only needed if you explicitly enable it, (We will not need to enable it for this class guys).*

---

## Quickstart for testing and using the Entity Extractor (rule-based, default)

- First, clone the repo by using: **git clone git@github.com:agumih/AI-Medical-transcription.git**

- Navigate to the root directory by using: **cd AI-Medical-transcription**

- Next, just copy-paste the commands below for the installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


- Next, to run the Entity Extractor rule based module, copy-paste the commands below:

**python extract_entities_cli.py clinical_notes.txt --csv entities.csv**

**python evaluate_ner.py gold.jsonl**
