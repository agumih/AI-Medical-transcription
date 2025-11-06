# extract_entities_cli.py
import sys, json, csv, pathlib, argparse
from typing import Iterable, Dict, Any, List
from biomed_pipeline import extract_entities as extract_rule

def _iter_files(root: pathlib.Path, glob: str) -> Iterable[pathlib.Path]:
    if root.is_file():
        yield root
    else:
        yield from sorted(root.rglob(glob))

def _read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _as_rows(file: str, ents: List[Dict[str, Any]]) -> Iterable[List[Any]]:
    for e in ents:
        yield [file, e["text"], e["label"], e["start"], e["end"]]

def extract(text: str, use_scispacy: bool = False) -> List[Dict[str, Any]]:
    if use_scispacy:
        # Heavy models; only import if requested
        from entity_extractor import extract_entities as extract_scispacy
        return extract_scispacy(text, link=False)  # linker optional
    return extract_rule(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="File or directory")
    ap.add_argument("--glob", default="*.txt", help="Glob when path is a directory (default: *.txt)")
    ap.add_argument("--csv", metavar="OUT.csv", help="Write CSV with columns: file,text,label,start,end")
    ap.add_argument("--jsonl", metavar="OUT.jsonl", help="Write JSONL: {file, entities:[...]}")
    ap.add_argument("--scispacy", action="store_true",
                    help="Use scispaCy-based extractor instead of rule-only baseline")
    args = ap.parse_args()

    root = pathlib.Path(args.path)
    files = list(_iter_files(root, args.glob))
    if not files:
        print("No files matched.", file=sys.stderr)
        sys.exit(2)

    all_rows: List[List[Any]] = []
    jl = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None

    # Process each file once
    outputs: List[Dict[str, Any]] = []
    for p in files:
        text = _read_text(p)
        ents = extract(text, use_scispacy=args.scispacy)
        outputs.append({"file": str(p), "entities": ents})
        if jl:
            jl.write(json.dumps(outputs[-1], ensure_ascii=False) + "\n")
        if args.csv:
            all_rows.extend(_as_rows(str(p), ents))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "text", "label", "start", "end"])
            w.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {args.csv}")

    if not args.csv and not args.jsonl:
        print(json.dumps(outputs if len(outputs) > 1 else outputs[0], indent=2, ensure_ascii=False))

    if jl:
        jl.close()

if __name__ == "__main__":
    main()







"""
import sys, json, csv, pathlib, argparse
from typing import Iterable, Dict, Any, List
#from biomed_pipeline import extract_entities
from biomed_pipeline import extract_entities as extract_rule


def _iter_files(root: pathlib.Path, glob: str) -> Iterable[pathlib.Path]:
    if root.is_file():
        yield root
    else:
        yield from sorted(root.rglob(glob))

def _read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _as_rows(file: str, ents: List[Dict[str, Any]]) -> Iterable[List[Any]]:
    for e in ents:
        yield [file, e["text"], e["label"], e["start"], e["end"]]

def extract(text, use_scispacy=False):
    if use_scispacy:
        from entity_extractor import extract_entities as extract_scispaCy
        return extract_scispaCy(text, link=False)  # linker optional
    return extract_rule(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="File or directory")
    ap.add_argument("--glob", default="*.txt", help="Glob when path is a directory (default: *.txt)")
    ap.add_argument("--csv", metavar="OUT.csv", help="Write CSV with columns: file,text,label,start,end")
    ap.add_argument("--jsonl", metavar="OUT.jsonl", help="Write JSONL: {file, entities:[...]}")
    args = ap.parse_args()

    root = pathlib.Path(args.path)
    files = list(_iter_files(root, args.glob))
    if not files:
        print("No files matched.", file=sys.stderr)
        sys.exit(2)

    all_rows: List[List[Any]] = []
    if args.jsonl:
        jl = open(args.jsonl, "w", encoding="utf-8")
    else:
        jl = None

    for p in files:
        text = _read_text(p)
        ents = extract_entities(text)
        if jl:
            jl.write(json.dumps({"file": str(p), "entities": ents}, ensure_ascii=False) + "\n")
        if args.csv:
            all_rows.extend(_as_rows(str(p), ents))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "text", "label", "start", "end"])
            w.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {args.csv}")

    if not args.csv and not args.jsonl:
        # default: print a compact summary to stdout
        for p in files:
            text = _read_text(p)
            ents = extract_entities(text)
            print(json.dumps({"file": str(p), "entities": ents}, indent=2))

    if jl:
        jl.close()

if __name__ == "__main__":
    main()











# extract_entities_cli.py (CSV variant)
import sys, json, pathlib, csv
from biomed_pipeline import extract_entities

def run_one(path: pathlib.Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    ents = extract_entities(text)
    return {"file": str(path), "entities": ents}

def write_csv(rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "text", "label", "start", "end"])
        for row in rows:
            w.writerow([row["file"], row["text"], row["label"], row["start"], row["end"]])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_entities_cli.py <file_or_dir> [--csv out.csv]")
        sys.exit(1)

    target = pathlib.Path(sys.argv[1])
    out_csv = None
    if "--csv" in sys.argv:
        i = sys.argv.index("--csv")
        out_csv = sys.argv[i+1] if i+1 < len(sys.argv) else "entities.csv"

    results = []
    files = [target] if target.is_file() else sorted(target.rglob("*.txt"))
    for p in files:
        for e in run_one(p)["entities"]:
            results.append({"file": str(p), **e})

    if out_csv:
        write_csv(results, out_csv)
        print(f"Wrote {len(results)} rows to {out_csv}")
    else:
        print(json.dumps(results, indent=2))

        






import sys, json, pathlib
from biomed_pipeline import extract_entities

def run_one(path: pathlib.Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    ents = extract_entities(text)
    return {"file": str(path), "entities": ents}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_entities_cli.py <file_or_dir>")
        sys.exit(1)

    target = pathlib.Path(sys.argv[1])
    outputs = []

    if target.is_file():
        outputs.append(run_one(target))
    else:
        for p in sorted(target.rglob("*.txt")):
            outputs.append(run_one(p))

    print(json.dumps(outputs, indent=2))
    







import json, sys, pathlib
from entity_extractor import extract_entities

def run_file(p: pathlib.Path, link: bool = True):
    text = p.read_text(encoding="utf-8", errors="ignore")
    ents = extract_entities(text, link=link)
    print(json.dumps({"file": str(p), "entities": ents}, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_entities_cli.py <file_or_dir> [--no-link]")
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    link = "--no-link" not in sys.argv[2:]
    if path.is_file():
        run_file(path, link)
    else:
        for f in sorted(path.rglob("*.txt")):
            run_file(f, link)
            
            """
