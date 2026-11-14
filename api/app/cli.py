import json
import pathlib
import sys

from .answer import answer_question
from .db import init_db
from .excel import process_workbook
from .ingestion import ingest_document


def main():
    init_db()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ingest":
        for p in sorted(pathlib.Path(sys.argv[2]).rglob("*")):
            if p.suffix.lower() in {".md", ".txt", ".pdf"}:
                print(ingest_document(p.name, p.read_bytes()))
    elif cmd == "ask":
        print(json.dumps(answer_question(" ".join(sys.argv[2:])), indent=2))
    elif cmd == "fill":
        in_path = pathlib.Path(sys.argv[2])
        out_path = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else in_path.with_name(f"completed_{in_path.name}")
        out_bytes, stats = process_workbook(in_path.read_bytes())
        out_path.write_bytes(out_bytes)
        print(f"Saved: {out_path}")
        print(json.dumps(stats, indent=2))
    else:
        print("Usage: python -m app.cli ingest <folder> | ask <question> | fill <questionnaire.xlsx>")


if __name__ == "__main__":
    main()