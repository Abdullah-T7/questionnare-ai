import json
import pathlib
import sys

from .answer import answer_question
from .db import init_db
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
    else:
        print("Usage: python -m app.cli ingest <folder> | ask <question>")


if __name__ == "__main__":
    main()