from __future__ import annotations

import re
from enum import Enum
from typing import List, Tuple


class AnswerLabel(str, Enum):
    GROUNDED = "grounded_answer"
    INSUFFICIENT = "insufficient_context"
    CONFLICTING = "conflicting_context"


class RetrievalStatus(str, Enum):
    HIT = "hit"
    PARTIAL_HIT = "partial_hit"
    MISS = "miss"


class Stage(str, Enum):
    INIT = "INIT"
    DOCUMENTS_LOADED = "DOCUMENTS_LOADED"
    DOCUMENTS_CHUNKED = "DOCUMENTS_CHUNKED"
    INDEX_BUILT = "INDEX_BUILT"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    ANSWERS_GENERATED = "ANSWERS_GENERATED"
    EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


NEXT_STAGE = {a: b for a, b in zip(list(Stage), list(Stage)[1:])}

ANSWER_LABELS = frozenset(label.value for label in AnswerLabel)
RETRIEVAL_STATUSES = frozenset(status.value for status in RetrievalStatus)

_CITATION_UNSAFE = str.maketrans({"[": "(", "]": ")", "§": "S"})
CITATION_RE = re.compile(r"\[(?P<title>[^\[\]§]+?) §(?P<chunk_id>[^\[\]§\s]+)\]")


def citation_safe(title: str) -> str:
    return title.translate(_CITATION_UNSAFE)


def format_citation(doc_title: str, chunk_id: str) -> str:
    return f"[{doc_title} §{chunk_id}]"


def parse_citation(citation: str) -> Tuple[str, str]:
    match = CITATION_RE.fullmatch(citation)
    if not match:
        raise ValueError(f"not a citation in [doc_title §chunk_id] format: {citation!r}")
    return match["title"], match["chunk_id"]


def find_citations(text: str) -> List[str]:
    return [m.group(0) for m in CITATION_RE.finditer(text)]
