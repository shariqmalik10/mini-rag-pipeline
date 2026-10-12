from __future__ import annotations

import re
from itertools import combinations
from typing import List, Sequence

from .index import Hit, QueryTerm, is_numeric, normalize
from .vocab import AnswerLabel, format_citation

MIN_COVERAGE = 0.5
MIN_GAIN = 0.2
MIN_SUPPORTING_TERMS = 2
SAME_DOC_SCORE_RATIO = 0.75
CONFLICT_SCORE_RATIO = 0.75
MAX_SENTENCES = 3

NOT_IN_KB_PREFIX = "The knowledge base does not mention:"
NO_TERMS = "Insufficient context: the question has no searchable terms."
NO_MATCH = "Insufficient context: the knowledge base does not contain information that answers this question."
CONFLICT_PREFIX = "Conflicting context: the knowledge base gives different answers."

_NUMBER = re.compile(r"\d+(?:[.:]\d+)?")


def _numbers(text: str) -> frozenset:
    return frozenset(_NUMBER.findall(text))


def _same_text(a: Hit, b: Hit) -> bool:
    return " ".join(normalize(a.chunk.text).split()) == " ".join(normalize(b.chunk.text).split())


def _record(label: AnswerLabel, sentences: Sequence[Hit], prefix: str = "", suffix: str = "") -> dict:
    citations = [format_citation(h.chunk.doc_title, h.chunk.chunk_id) for h in sentences]
    body = " ".join(f"{' '.join(h.chunk.text.split())} {c}" for h, c in zip(sentences, citations))
    return {
        "answer_label": label.value,
        "answer": " ".join(part for part in (prefix, body, suffix) if part),
        "citations": citations,
        "used_chunk_ids": [h.chunk.chunk_id for h in sentences],
    }


def answer(terms: List[QueryTerm], hits: List[Hit]) -> dict:
    content = [t for t in terms if not is_numeric(t.term)]
    if not content:
        return _record(AnswerLabel.INSUFFICIENT, [], NO_TERMS)
    total = sum(t.weight for t in content)

    def weight(matched) -> float:
        return sum(t.weight for t in content if t.term in matched) / total

    candidates = [h for h in hits if h.score > 0]
    if not candidates:
        return _record(AnswerLabel.INSUFFICIENT, [], NO_MATCH)
    gated = [h for h in candidates if weight(h.matched) >= MIN_COVERAGE]
    if not gated:
        return _record(AnswerLabel.INSUFFICIENT, [], NO_MATCH)
    first = gated[0]

    strong = [h for h in gated if h.score >= CONFLICT_SCORE_RATIO * first.score]
    for a, b in combinations(strong, 2):
        disagrees = (
            a.chunk.doc_title != b.chunk.doc_title
            and len(a.text_matched & b.text_matched) >= MIN_SUPPORTING_TERMS
            and _numbers(a.chunk.text) and _numbers(b.chunk.text)
            and _numbers(a.chunk.text) != _numbers(b.chunk.text)
        )
        if disagrees:
            return _record(AnswerLabel.CONFLICTING, [a, b], CONFLICT_PREFIX)

    selected = [first]
    covered = set(first.matched)
    for hit in candidates:
        if len(selected) == MAX_SENTENCES:
            break
        if any(hit is s or _same_text(hit, s) for s in selected):
            continue
        adds_coverage = (
            weight(hit.matched - covered) >= MIN_GAIN
            and len({t.term for t in content} & hit.text_matched) >= MIN_SUPPORTING_TERMS
        )
        close_in_same_doc = (
            hit.chunk.doc_title == first.chunk.doc_title
            and hit.score >= SAME_DOC_SCORE_RATIO * first.score
        )
        if adds_coverage or close_in_same_doc:
            selected.append(hit)
            covered |= hit.matched

    absent = [t.surface for t in content if not t.in_kb]
    suffix = f"{NOT_IN_KB_PREFIX} {', '.join(absent)}." if absent else ""
    return _record(AnswerLabel.GROUNDED, selected, suffix=suffix)
