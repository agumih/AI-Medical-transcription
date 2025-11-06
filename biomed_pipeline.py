# biomed_pipeline.py
from __future__ import annotations
import csv
import functools
from pathlib import Path
from typing import List, Dict, Any
import spacy
from spacy.language import Language

# Optional data directory with CSVs. If missing, we fall back to built-in seeds.
DATA_DIR = Path(__file__).parent / "data"

# ---------- CSV loaders (optional) ----------
def _load_terms_one_col(path: Path, label: str) -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    if not path.exists():
        return patterns
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "pattern" not in r.fieldnames:
            return patterns
        for row in r:
            t = (row.get("pattern") or "").strip()
            if t:
                patterns.append({"label": label, "pattern": t})
    return patterns

def _load_terms_labelled(path: Path) -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    if not path.exists():
        return patterns
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or not {"label", "pattern"}.issubset(set(r.fieldnames)):
            return patterns
        for row in r:
            lab = (row.get("label") or "").strip()
            t = (row.get("pattern") or "").strip()
            if lab and t:
                patterns.append({"label": lab, "pattern": t})
    return patterns

def _seed_patterns() -> List[Dict[str, Any]]:
    drug_patterns = [
        "metoprolol", "aspirin", "ibuprofen", "acetaminophen",
        "amoxicillin", "atorvastatin", "lisinopril", "metformin",
        "albuterol", "omeprazole"
    ]
    disease_patterns = [
        "COPD", "hypertension", "diabetes mellitus", "myocardial infarction",
        "heart failure", "pneumonia", "asthma", "stroke", "sepsis"
    ]
    procedure_patterns = [
        "echocardiogram", "ECG", "EKG", "MRI", "CT scan", "x-ray",
        "laparoscopic cholecystectomy", "colonoscopy", "angiography"
    ]
    anatomy_patterns = [
        "left arm", "right arm", "left ventricle", "right ventricle",
        "left lung", "right lung", "liver", "kidney", "cervical spine"
    ]

    patterns: List[Dict[str, Any]] = []
    patterns += [{"label": "DRUG", "pattern": t} for t in drug_patterns]
    patterns += [{"label": "DISEASE", "pattern": t} for t in disease_patterns]
    patterns += [{"label": "PROCEDURE", "pattern": t} for t in procedure_patterns]
    patterns += [{"label": "ANATOMY", "pattern": t} for t in anatomy_patterns]
    return patterns

def _load_all_patterns() -> List[Dict[str, Any]]:
    # If CSVs exist, use them; otherwise fall back to built-in seeds.
    csv_patterns: List[Dict[str, Any]] = []
    csv_patterns += _load_terms_one_col(DATA_DIR / "drugs.csv", "DRUG")
    csv_patterns += _load_terms_one_col(DATA_DIR / "diseases.csv", "DISEASE")
    csv_patterns += _load_terms_labelled(DATA_DIR / "procedures_anatomy.csv")
    return csv_patterns if csv_patterns else _seed_patterns()

# ---------- Pipeline ----------
@functools.lru_cache(maxsize=1)
def _get_nlp() -> Language:
    nlp = spacy.blank("en")  # tokenizer only
    ruler = nlp.add_pipe(
        "entity_ruler",
        name="clinical_patterns",
        config={
            "overwrite_ents": False,
            # Case-insensitive matching via token attr LOWER (works in spaCy 3.8.x)
            "phrase_matcher_attr": "LOWER"
        },
    )
    ruler.add_patterns(_load_all_patterns())
    return nlp

def build_pipeline() -> Language:
    """Return a cached, ready-to-use rule-based pipeline."""
    return _get_nlp()

def extract_entities(text: str) -> List[Dict[str, Any]]:
    nlp = _get_nlp()
    doc = nlp(text)
    return [
        {"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
        for e in doc.ents
    ]

if __name__ == "__main__":
    sample = "Patient with COPD on metoprolol and aspirin. Echocardiogram performed. Pain in left arm."
    for r in extract_entities(sample):
        print(r)





"""
import csv
import functools
from pathlib import Path
from typing import List, Dict, Any
import spacy
from spacy.language import Language

DATA_DIR = Path(__file__).parent / "data"

def _load_terms_one_col(path: Path, label: str) -> List[Dict[str, Any]]:
    patterns = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "pattern" not in r.fieldnames:
            raise ValueError(f"{path} must have a 'pattern' column")
        for row in r:
            t = (row.get("pattern") or "").strip()
            if t:
                patterns.append({"label": label, "pattern": t})
    return patterns

def _load_terms_labelled(path: Path) -> List[Dict[str, Any]]:
    patterns = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not {"label", "pattern"}.issubset(set(r.fieldnames or [])):
            raise ValueError(f"{path} must have 'label' and 'pattern' columns")
        for row in r:
            lab = (row.get("label") or "").strip()
            t = (row.get("pattern") or "").strip()
            if lab and t:
                patterns.append({"label": lab, "pattern": t})
    return patterns

def _load_all_patterns() -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    patterns += _load_terms_one_col(DATA_DIR / "drugs.csv", "DRUG")
    patterns += _load_terms_one_col(DATA_DIR / "diseases.csv", "DISEASE")
    patterns += _load_terms_labelled(DATA_DIR / "procedures_anatomy.csv")
    return patterns

@functools.lru_cache(maxsize=1)
def _get_nlp() -> Language:
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe(
        "entity_ruler",
        name="clinical_patterns",
        config={"overwrite_ents": False, "phrase_matcher_attr": "LOWER"}
    )
    ruler.add_patterns(_load_all_patterns())
    return nlp

def build_pipeline() -> Language:
    # Kept for API compatibility; returns the cached pipeline
    return _get_nlp()

def extract_entities(text: str) -> List[Dict[str, Any]]:
    nlp = _get_nlp()
    doc = nlp(text)
    return [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents]

if __name__ == "__main__":
    sample = "Patient with COPD on metoprolol and aspirin. Echocardiogram performed. Pain in left arm."
    for r in extract_entities(sample):
        print(r)










import spacy
from typing import List, Dict, Any
from spacy.language import Language
from spacy.pipeline import EntityRuler

def build_pipeline() -> Language:
    
    #Minimal, fast clinical NER using spaCy + EntityRuler only.
    #Compatible with spaCy 3.8.x.
    
    nlp = spacy.blank("en")  # tokenizer only

    # Case-insensitive ruler so we don't need to duplicate lowercase patterns
    ruler = nlp.add_pipe(
        "entity_ruler",
        name="clinical_patterns",
        config={"overwrite_ents": False, "phrase_matcher_attr": "LOWER"}
    )

    drug_patterns = [
        "metoprolol", "aspirin", "ibuprofen", "acetaminophen",
        "amoxicillin", "atorvastatin", "lisinopril", "metformin",
        "albuterol", "omeprazole"
    ]
    disease_patterns = [
        "COPD", "hypertension", "diabetes mellitus", "myocardial infarction",
        "heart failure", "pneumonia", "asthma", "stroke", "sepsis"
    ]
    procedure_patterns = [
        "echocardiogram", "ECG", "EKG", "MRI", "CT scan", "x-ray",
        "laparoscopic cholecystectomy", "colonoscopy", "angiography"
    ]
    anatomy_patterns = [
        "left arm", "right arm", "left ventricle", "right ventricle",
        "left lung", "right lung", "liver", "kidney", "cervical spine"
    ]

    patterns: List[Dict[str, Any]] = []
    patterns += [{"label": "DRUG", "pattern": t} for t in drug_patterns]
    patterns += [{"label": "DISEASE", "pattern": t} for t in disease_patterns]
    patterns += [{"label": "PROCEDURE", "pattern": t} for t in procedure_patterns]
    patterns += [{"label": "ANATOMY", "pattern": t} for t in anatomy_patterns]

    ruler.add_patterns(patterns)
    return nlp




def extract_entities(text: str) -> List[Dict[str, Any]]:
    nlp = build_pipeline()
    doc = nlp(text)
    return [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents]

if __name__ == "__main__":
    sample = "Patient with COPD on metoprolol and aspirin. Echocardiogram performed. Pain in left arm."
    for r in extract_entities(sample):
        print(r)
"""
