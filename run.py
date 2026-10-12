import argparse
import sys

from rag.ingest import STRATEGIES
from rag.pipeline import PRIMARY_STRATEGY, ValidationFailed, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the mini-RAG pipeline and regenerate artifacts.")
    parser.add_argument("--kb", default="kb")
    parser.add_argument("--queries", default="queries.json")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--strategy", default=PRIMARY_STRATEGY, choices=sorted(STRATEGIES))
    args = parser.parse_args()
    try:
        manifest = run_pipeline(args.kb, args.queries, args.out, args.strategy)
    except (ValueError, FileNotFoundError, ValidationFailed) as exc:
        print(f"pipeline failed: {exc}", file=sys.stderr)
        return 1
    print(" -> ".join(manifest["stage_trace"]))
    print(f"{manifest['documents']} documents, {manifest['chunks']} chunks, {manifest['queries']} queries. Artifacts in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
