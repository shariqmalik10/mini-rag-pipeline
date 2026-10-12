import argparse
import sys

from rag.pipeline import load_queries
from rag.validate import validate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pipeline artifacts.")
    parser.add_argument("--queries", default="queries.json")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    try:
        query_ids = [q["query_id"] for q in load_queries(args.queries)]
    except ValueError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    errors = validate(args.out, query_ids)
    for error in errors:
        print(f"FAIL {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"OK: all artifacts valid for {len(query_ids)} queries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
