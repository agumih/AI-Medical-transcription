# entity_extractor.py  (lazy-load scispaCy models)
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import spacy

# --- Globals are set to None and filled only when actually used ---
_NLP_EMB = None                # tokenizer + embeddings (en_core_sci_lg)
_NER_DISEASE_DRUG = None       # en_ner_bc5cdr_md  (DISEASE, CHEMICAL)
_NER_PROC_ANAT = None          # en_ner_bionlp13cg_md (PROCEDURE-ish, ANATOMY-ish)
_WITH_LINKER = False           # becomes True if UMLS linker loads

_SCISPACY_LOAD_ERROR: Optional[str] = None


def _ensure_models():
    """
    Load scispaCy models lazily. If any model is missing, raise a clear error
    telling the user how to install them (or to stick to the rule-based module).
    """
    global _NLP_EMB, _NER_DISEASE_DRUG, _NER_PROC_ANAT, _WITH_LINKER, _SCISPACY_LOAD_ERROR
    if _NLP_EMB is not None and _NER_DISEASE_DRUG is not None and _NER_PROC_ANAT is not None:
        return

    try:
        _NLP_EMB = spacy.load("en_core_sci_lg")
        _NER_DISEASE_DRUG = spacy.load("en_ner_bc5cdr_md")
        _NER_PROC_ANAT = spacy.load("en_ner_bionlp13cg_md")
    except Exception as e:
        _SCISPACY_LOAD_ERROR = (
            "scispaCy models are not installed. Either:\n"
            "  • Use the rule-based extractor from biomed_pipeline.py (no extra installs), or\n"
            "  • Install the optional models, e.g.:\n"
            "      pip install scispacy\n"
            "      pip install https://github.com/allenai/scispacy/releases/download/v0.5.4/en_core_sci_lg-0.5.4.tar.gz\n"
            "      pip install https://github.com/allenai/scispacy/releases/download/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz\n"
            "      pip install https://github.com/allenai/scispacy/releases/download/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz\n"
            f"\nOriginal error: {e!r}"
        )
        # I will leave globals as None so we can only raise on use.
        raise RuntimeError(_SCISPACY_LOAD_ERROR) from e

    try:
        from scispacy.linking import UmlsEntityLinker
        linker = UmlsEntityLinker(resolve_abbreviations=True, threshold=0.85)
        _NLP_EMB.add_pipe(linker)  # attaches as nlp.get_pipe("umls_linker")
        _WITH_LINKER = True
    except Exception:
        _WITH_LINKER = False  # perfectly fine I hope; linking remains off


@dataclass
class MedicalEntity:
    text: str
    label: str
    start: int
    end: int
    source: str
    kb_id: str | None = None
    kb_name: str | None = None
    kb_score: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Map model labels into a small, stable schema
LABEL_MAP = {
    # bc5cdr
    "DISEASE": "DISEASE",
    "CHEMICAL": "DRUG",  # fold CHEMICAL → DRUG for our schema
    # bionlp13cg (subset)
    "ANATOMICAL_SITE": "ANATOMY",
    "TISSUE": "ANATOMY",
    "ORGAN": "ANATOMY",
    "CELL": "ANATOMY",
    "AMINO_ACID": "CHEMICAL",      # we’ll treat CHEMICAL as DRUG later if needed
    "SIMPLE_CHEMICAL": "CHEMICAL",
    "PROCESS": "PROCEDURE",        # heuristic
    "MULTI-TISSUE_STRUCTURE": "ANATOMY",
    "ORGANISM_SUBDIVISION": "ANATOMY",
    "ORGANISM": "ANATOMY",
}


def _priority(e: MedicalEntity) -> Tuple[int, float, int]:
    """Prefer KB-linked, higher score, longer span for de-duplication."""
    has_kb = 1 if e.kb_id else 0
    kb_score = e.kb_score or 0.0
    length = e.end - e.start
    return (has_kb, kb_score, length)


def _link_span(text: str, start: int, end: int) -> Tuple[str | None, str | None, float | None]:
    if not _WITH_LINKER:
        return None, None, None
    # Build a minimal doc and span to access linker
    doc = _NLP_EMB.make_doc(text)
    span = doc.char_span(start, end, alignment_mode="expand")
    if span is None or not span._.kb_ents:
        return None, None, None
    cui, score = span._.kb_ents[0]
    kb_ent = _NLP_EMB.get_pipe("umls_linker").kb.cui_to_entity.get(cui)
    name = kb_ent.canonical_name if kb_ent else None
    return cui, name, float(score)


class MedicalEntityExtractor:
    """scispaCy-based extractor. Only used if your environment has the optional models installed."""

    def __init__(self, do_linking: bool = True):
        self.do_linking = do_linking  # will be ANDed with _WITH_LINKER after models load

    def extract(self, text: str) -> List[MedicalEntity]:
        # Ensure models are present (or raise a clear error once)
        _ensure_models()

        raw: List[MedicalEntity] = []

        doc1 = _NER_DISEASE_DRUG(text)
        doc2 = _NER_PROC_ANAT(text)

        def collect(doc, src: str):
            for ent in doc.ents:
                mapped = LABEL_MAP.get(ent.label_, ent.label_)
                if mapped not in {"DISEASE", "DRUG", "PROCEDURE", "ANATOMY", "CHEMICAL"}:
                    continue
                kb_id = kb_name = None
                kb_score = None
                if self.do_linking and _WITH_LINKER:
                    kb_id, kb_name, kb_score = _link_span(text, ent.start_char, ent.end_char)
                # Normalize CHEMICAL → DRUG in final output
                final_label = "DRUG" if mapped == "CHEMICAL" else mapped
                raw.append(MedicalEntity(
                    text=ent.text,
                    label=final_label,
                    start=ent.start_char,
                    end=ent.end_char,
                    source=src,
                    kb_id=kb_id, kb_name=kb_name, kb_score=kb_score
                ))

        collect(doc1, "bc5cdr")
        collect(doc2, "bionlp13cg")

        # Merge overlaps: keep best by priority
        raw.sort(key=lambda e: (e.start, e.end))
        merged: List[MedicalEntity] = []
        for e in raw:
            if merged and not (e.start >= merged[-1].end or e.end <= merged[-1].start):
                best = max([merged[-1], e], key=_priority)
                merged[-1] = best
            else:
                merged.append(e)

        return merged


def extract_entities(text: str, link: bool = True) -> List[Dict[str, Any]]:
    """
    Public API for scispaCy extractor.
    If models are missing, raise a friendly error telling the user what to do.
    """
    try:
        extractor = MedicalEntityExtractor(do_linking=link)
        return [e.to_dict() for e in extractor.extract(text)]
    except RuntimeError as e:
        # Re-raise with the helpful message from _ensure_models()
        raise
