# evaluate_ner.py
import json, argparse
from typing import List, Dict, Tuple
from biomed_pipeline import extract_entities as extract_rule

Span = Tuple[int, int, str]

def _to_spans(ents: List[Dict]) -> List[Span]:
    return [(e["start"], e["end"], e["label"]) for e in ents]

def _match(pred: List[Span], gold: List[Span]) -> Tuple[int, int, int]:
    # exact span + label match
    pset, gset = set(pred), set(gold)
    tp = len(pset & gset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    return tp, fp, fn

def _safe_div(a: int, b: int) -> float:
    return (a / b) if b else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold_jsonl", help="JSONL with fields: text, spans:[{start,end,label}]")
    ap.add_argument("--scispacy", action="store_true",
                    help="Evaluate using scispaCy extractor instead of rule-only baseline")
    args = ap.parse_args()

    if args.scispacy:
        from entity_extractor import extract_entities as extract_scispacy
        extractor = lambda txt: extract_scispacy(txt, link=False)
    else:
        extractor = extract_rule

    tp = fp = fn = 0
    with open(args.gold_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            gold_spans = [(s["start"], s["end"], s["label"]) for s in rec.get("spans", [])]
            pred_spans = _to_spans(extractor(text))
            a, b, c = _match(pred_spans, gold_spans)
            tp += a; fp += b; fn += c

    prec = _safe_div(tp, tp + fp)
    rec  = _safe_div(tp, tp + fn)
    f1   = _safe_div(2 * prec * rec, prec + rec)

    print(json.dumps({
        "engine": "scispacy" if args.scispacy else "rules",
        "micro": {"precision": prec, "recall": rec, "f1": f1},
        "counts": {"tp": tp, "fp": fp, "fn": fn}
    }, indent=2))

if __name__ == "__main__":
    main()
