from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .answer import CONFLICT_PREFIX, NOT_IN_KB_PREFIX
from .index import analyze, normalize
from .vocab import ANSWER_LABELS, CITATION_RE, RETRIEVAL_STATUSES, AnswerLabel, parse_citation

MIN_RETRIEVED = 3
SUPPORT_MIN_OVERLAP = 0.6
_NEGATION = re.compile(r"\b(?:not|no|never|nor|without|cannot|cant|wont|dont|doesnt|didnt|isnt|arent|wasnt)\b")
SUMMARY_KEYS = ("top3_hit_rate", "total_queries", "hits", "partial_hits", "misses")

ARTIFACTS = {
    "chunks": "chunks.json",
    "retrieval": "retrieval.json",
    "answers": "answers.json",
    "eval": "eval.json",
    "chunking_comparison": "chunking_comparison.json",
    "grounding_check": "grounding_check.json",
}


def _squash(text: str) -> str:
    return " ".join(text.split())


def _support(claim: str, chunk_text: str) -> Tuple[str, float]:
    if _squash(claim) and _squash(claim) in _squash(chunk_text):
        return "exact", 1.0
    claim_terms = set(analyze(claim))
    if not claim_terms:
        return "none", 0.0
    overlap = len(claim_terms & set(analyze(chunk_text))) / len(claim_terms)
    same_polarity = bool(_NEGATION.search(normalize(claim))) == bool(_NEGATION.search(normalize(chunk_text)))
    return ("overlap" if overlap >= SUPPORT_MIN_OVERLAP and same_polarity else "none"), round(overlap, 4)


def grounding_check(retrieval: List[dict], answers: List[dict]) -> dict:
    retrieved = {
        r["query_id"]: {(h["doc_title"], h["chunk_id"]): h["chunk_text"] for h in r["top_k"]}
        for r in retrieval
    }
    results = []
    for ans in answers:
        pool = retrieved.get(ans["query_id"], {})
        text = ans["answer"]
        checks, cursor = [], 0
        for match in CITATION_RE.finditer(text):
            claim = text[cursor:match.start()].strip()
            if claim.startswith(CONFLICT_PREFIX):
                claim = claim[len(CONFLICT_PREFIX):].strip()
            key = (match["title"], match["chunk_id"])
            support, overlap = _support(claim, pool[key]) if key in pool else ("none", 0.0)
            checks.append({
                "citation": match.group(0),
                "claim": claim,
                "in_retrieval": key in pool,
                "support": support,
                "overlap": overlap,
            })
            cursor = match.end()
        trailing = text[cursor:].strip()
        uncited = trailing if checks and trailing and not trailing.startswith(NOT_IN_KB_PREFIX) else ""
        listed = set(ans.get("citations", []))
        in_text = {c["citation"] for c in checks}
        needs_citation = ans["answer_label"] != AnswerLabel.INSUFFICIENT.value
        grounded = (
            (bool(checks) or not needs_citation)
            and all(c["in_retrieval"] and c["support"] != "none" for c in checks)
            and not uncited
            and listed == in_text
        )
        results.append({
            "query_id": ans["query_id"],
            "answer_label": ans["answer_label"],
            "citations": checks,
            "citations_match_answer_text": listed == in_text,
            "uncited_text": uncited,
            "grounded": grounded,
        })
    return {
        "summary": {
            "answers": len(results),
            "grounded": sum(r["grounded"] for r in results),
            "ungrounded_query_ids": [r["query_id"] for r in results if not r["grounded"]],
        },
        "results": results,
    }


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_one_per_query(name: str, records, query_ids: List[str], errors: List[str]) -> Dict[str, dict]:
    if not isinstance(records, list):
        errors.append(f"{name}: expected a list of records")
        return {}
    by_id: Dict[str, dict] = {}
    for rec in records:
        if not isinstance(rec, dict):
            errors.append(f"{name}: record is not an object: {rec!r}")
            continue
        qid = rec.get("query_id")
        if qid in by_id:
            errors.append(f"{name}: duplicate record for {qid}")
        by_id[qid] = rec
    missing = [q for q in query_ids if q not in by_id]
    extra = [q for q in by_id if q not in query_ids]
    if missing:
        errors.append(f"{name}: queries not processed: {missing}")
    if extra:
        errors.append(f"{name}: records for unknown queries: {extra}")
    return by_id


def check_artifacts(out_dir: Path, query_ids: List[str]) -> List[str]:
    errors: List[str] = []
    data = {}
    for key, filename in ARTIFACTS.items():
        path = out_dir / filename
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
            continue
        try:
            data[key] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{filename}: invalid JSON ({exc})")
    if errors:
        return errors

    chunk_text = {}
    for c in data["chunks"]:
        key = (c.get("doc_title"), c.get("chunk_id"))
        if not isinstance(c.get("text"), str) or not c["text"]:
            errors.append(f"chunks.json: {key} has empty text")
        elif c.get("end_char", 0) - c.get("start_char", 0) != len(c["text"]):
            errors.append(f"chunks.json: {key} offsets do not span its text")
        if key in chunk_text:
            errors.append(f"chunks.json: duplicate chunk {key}")
        chunk_text[key] = c.get("text")

    retrieval = _check_one_per_query("retrieval.json", data["retrieval"], query_ids, errors)
    retrieved_keys = {}
    for qid, rec in retrieval.items():
        top_k = rec.get("top_k", [])
        if len(top_k) < MIN_RETRIEVED:
            errors.append(f"retrieval.json: {qid} has {len(top_k)} chunks, needs at least {MIN_RETRIEVED}")
        for i, hit in enumerate(top_k, start=1):
            if hit.get("rank") != i:
                errors.append(f"retrieval.json: {qid} rank {hit.get('rank')} out of order")
            if not _is_number(hit.get("score")):
                errors.append(f"retrieval.json: {qid} rank {i} score is not numeric: {hit.get('score')!r}")
            key = (hit.get("doc_title"), hit.get("chunk_id"))
            if chunk_text.get(key) != hit.get("chunk_text"):
                errors.append(f"retrieval.json: {qid} rank {i} does not match a chunk in chunks.json")
        retrieved_keys[qid] = {(h.get("doc_title"), h.get("chunk_id")) for h in top_k}

    answers = _check_one_per_query("answers.json", data["answers"], query_ids, errors)
    for qid, ans in answers.items():
        label = ans.get("answer_label")
        if label not in ANSWER_LABELS:
            errors.append(f"answers.json: {qid} label {label!r} not in {sorted(ANSWER_LABELS)}")
        citations = ans.get("citations", [])
        if label in (AnswerLabel.GROUNDED.value, AnswerLabel.CONFLICTING.value) and not citations:
            errors.append(f"answers.json: {qid} is {label} but has no citations")
        for citation in citations:
            try:
                key = parse_citation(citation)
            except ValueError as exc:
                errors.append(f"answers.json: {qid} {exc}")
                continue
            if key not in retrieved_keys.get(qid, set()):
                errors.append(f"answers.json: {qid} cites {citation}, which was not retrieved for it")
        retrieved_ids = {cid for _, cid in retrieved_keys.get(qid, set())}
        for cid in ans.get("used_chunk_ids", []):
            if cid not in retrieved_ids:
                errors.append(f"answers.json: {qid} uses chunk {cid}, which was not retrieved for it")

    ev = data["eval"]
    if not isinstance(ev, dict) or "results" not in ev or "summary" not in ev:
        errors.append("eval.json: expected an object with 'results' and 'summary'")
    else:
        for rec in _check_one_per_query("eval.json", ev["results"], query_ids, errors).values():
            if rec.get("retrieval_status") not in RETRIEVAL_STATUSES:
                errors.append(f"eval.json: {rec.get('query_id')} status {rec.get('retrieval_status')!r} not in {sorted(RETRIEVAL_STATUSES)}")
        summary = ev["summary"]
        for key in SUMMARY_KEYS:
            if not _is_number(summary.get(key)):
                errors.append(f"eval.json: summary.{key} missing or not numeric")
        if not errors and summary["hits"] + summary["partial_hits"] + summary["misses"] != summary["total_queries"]:
            errors.append("eval.json: summary counts do not add up to total_queries")

    for rec in data["grounding_check"].get("results", []):
        if not rec.get("grounded"):
            errors.append(f"grounding_check.json: {rec.get('query_id')} answer is not grounded in its cited chunks")
    return errors


def validate(out_dir: str | Path, query_ids: List[str]) -> List[str]:
    out = Path(out_dir)
    try:
        retrieval = json.loads((out / ARTIFACTS["retrieval"]).read_text(encoding="utf-8"))
        answers = json.loads((out / ARTIFACTS["answers"]).read_text(encoding="utf-8"))
        report = grounding_check(retrieval, answers)
    except OSError:
        return check_artifacts(out, query_ids)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return [f"cannot run grounding check: {exc}"]
    (out / ARTIFACTS["grounding_check"]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return check_artifacts(out, query_ids)
