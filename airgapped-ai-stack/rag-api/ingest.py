"""Bulk-index a folder of documents from inside the rag-api container.

    docker compose run --rm -v /path/to/docs:/docs:ro rag-api python ingest.py /docs
"""

import sys
from pathlib import Path

import rag


def main(root: str) -> int:
    files = sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in rag.SUPPORTED_SUFFIXES)
    if not files:
        print(f"no supported files under {root}")
        return 1
    total = 0
    for p in files:
        source = str(p.relative_to(root))
        try:
            n = rag.ingest(source, p.read_bytes())
            total += n
            print(f"  {n:5d} chunks  {source}")
        except Exception as e:  # noqa: BLE001 - keep going on bad files
            print(f"  FAILED      {source}: {e}")
    print(f"indexed {len(files)} files, {total} chunks into collection '{rag.COLLECTION}'")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/docs"))
