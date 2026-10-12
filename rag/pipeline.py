from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Sequence

from .answer import answer
from .evaluate import evaluate_query, summarize
from .index import Index
from .ingest import STRATEGIES, chunk_documents, kb_files, load_documents
from .validate import ARTIFACTS, MIN_RETRIEVED, validate
from .vocab import NEXT_STAGE, Stage

TOP_K = 5
PRIMARY_STRATEGY = "sentence"
MANIFEST = "run_manifest.json"


class StageError(RuntimeError):
    pass


class ValidationFailed(RuntimeError):
    def __init__(self, errors: List[str]):
        super().__init__("validation failed:\n  " + "\n  ".join(errors))
        self.errors = errors


class Run:
    def __init__(self) -> None:
        self.stage = Stage.INIT
        self.trace = [Stage.INIT.value]

    def advance(self, to: Stage) -> None:
        if NEXT_STAGE.get(self.stage) is not to:
            raise StageError(f"cannot move from {self.stage.value} to {to.value}")
        self.stage = to
        self.trace.append(to.value)


def load_queries(path: str | Path) -> List[dict]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read queries ({exc})") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON list of queries")
    queries, seen = [], set()
    for i, item in enumerate(raw):
        where = f"{path}[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where}: expected an object")
        qid, question = item.get("query_id"), item.get("question")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"{where}: 'query_id' must be a non-empty string")
        if qid in seen:
            raise ValueError(f"{where}: duplicate query_id {qid!r}")
        if not isinstance(question, str):
            raise ValueError(f"{where}: 'question' must be a string")
        lists = {}
        for key in ("expected_doc_titles", "ground_truth_answer_points"):
            value = item.get(key) or []
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{where}: {key!r} must be a list of strings")
            lists[key] = value
        seen.add(qid)
        queries.append({"query_id": qid, "question": question, **lists})
    return queries


def _write(out: Path, name: str, data) -> None:
    (out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _retrieval_summary(docs, queries: List[dict], strategy: str) -> dict:
    chunks = chunk_documents(docs, strategy)
    index = Index(chunks)
    titles = [d.title for d in docs]
    results = [index.search(q["question"], TOP_K)[1] for q in queries]
    records = [evaluate_query(q, hits, titles) for q, hits in zip(queries, results)]
    return {
        "strategy": strategy,
        "chunk_count": len(chunks),
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / len(chunks), 1),
        "avg_rank1_chunk_chars": round(sum(len(hits[0].chunk.text) for hits in results) / len(results), 1),
        "summary": summarize(records),
    }


def compare_strategies(docs, queries: List[dict], strategies: Sequence[str]) -> dict:
    runs = [_retrieval_summary(docs, queries, s) for s in strategies]
    result = "; ".join(
        f"{r['strategy']}: top-3 hit rate {r['summary']['top3_hit_rate']}, MRR {r['summary']['mrr_top3']}, "
        f"answer point in rank-1 chunk {r['summary']['answer_point_recall_rank1']}, "
        f"rank-1 chunk {r['avg_rank1_chunk_chars']} chars on average"
        for r in runs
    ) + "."
    return {
        "strategies": runs,
        "result": result,
        "tradeoff": (
            "Sentence chunks hold one fact each, so a citation points at exactly the supporting sentence and an "
            "extractive answer quotes nothing extra. They lose context when a fact spans sentences, which the "
            "pronoun-merge rule only partly recovers. Fixed-size windows keep neighbouring sentences together and "
            "tolerate unpunctuated text, but they cut facts mid-sentence, dilute term scores with unrelated words, "
            "and make quoted answers carry text the question did not ask about."
        ),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_pipeline(kb_dir: str | Path, queries_path: str | Path, out_dir: str | Path,
                 strategy: str = PRIMARY_STRATEGY) -> dict:
    run = Run()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in [*ARTIFACTS.values(), MANIFEST]:
        (out / name).unlink(missing_ok=True)
    queries = load_queries(queries_path)

    docs = load_documents(kb_dir)
    run.advance(Stage.DOCUMENTS_LOADED)

    chunks = chunk_documents(docs, strategy)
    if len(chunks) < MIN_RETRIEVED:
        raise ValueError(f"knowledge base yields {len(chunks)} chunks; at least {MIN_RETRIEVED} are needed per query")
    _write(out, ARTIFACTS["chunks"], [c.to_dict() for c in chunks])
    run.advance(Stage.DOCUMENTS_CHUNKED)

    index = Index(chunks)
    run.advance(Stage.INDEX_BUILT)

    searched = [(q, *index.search(q["question"], TOP_K)) for q in queries]
    _write(out, ARTIFACTS["retrieval"], [
        {"query_id": q["query_id"], "question": q["question"], "top_k": [h.to_dict() for h in hits]}
        for q, _, hits in searched
    ])
    run.advance(Stage.RETRIEVAL_COMPLETE)

    _write(out, ARTIFACTS["answers"], [{"query_id": q["query_id"], **answer(terms, hits)} for q, terms, hits in searched])
    run.advance(Stage.ANSWERS_GENERATED)

    titles = [d.title for d in docs]
    records = [evaluate_query(q, hits, titles) for q, _, hits in searched]
    _write(out, ARTIFACTS["eval"], {"summary": summarize(records), "results": records})
    others = [s for s in STRATEGIES if s != strategy]
    _write(out, ARTIFACTS["chunking_comparison"], compare_strategies(docs, queries, [strategy, *others]))
    run.advance(Stage.EVALUATION_COMPLETE)

    errors = validate(out, [q["query_id"] for q in queries])
    if errors:
        raise ValidationFailed(errors)
    run.advance(Stage.VALIDATION_COMPLETE)

    run.advance(Stage.RESULTS_FINALISED)
    manifest = {
        "stage_trace": run.trace,
        "chunking_strategy": strategy,
        "top_k": TOP_K,
        "llm_used": False,
        "documents": len(docs),
        "chunks": len(chunks),
        "queries": len(queries),
        "kb_sha256": sorted(_sha256(p.read_bytes()) for p in kb_files(kb_dir)),
        "queries_sha256": _sha256(Path(queries_path).read_bytes()),
        "artifacts": sorted(ARTIFACTS.values()),
    }
    _write(out, MANIFEST, manifest)
    return manifest
