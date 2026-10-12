from __future__ import annotations

import difflib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import FrozenSet, List, Tuple

from .ingest import Chunk

_STOPWORDS = frozenset("""
a an the and or but if then so of to in on at by for from with without about as into over under after before
is are was were be been being am do does did doing have has had having can could will would shall should may might must
i me my mine we us our you your yours he him his she her it its they them their this that these those there here
what which who whom whose when where why how not no nor cannot cant dont doesnt didnt wont isnt arent wasnt
any some all each every more most other such only own same than too very just also
anything something everything nothing anyone someone everyone
""".split())

_FILLER_WORDS = """
hi hello hey help please thanks thank ok okay sure sorry wondering wonder question questions ask asking asked explain
know knew tell told think thought guess mean means maybe perhaps possibly probably really actually basically
currently recently still already yet ago yesterday today tomorrow tonight now soon later earlier just
need needs want wants like would get got getting make made making take took taken taking give gave go going come
see seen look looking say said use used using work works working worked happen happens happened expect expected
thing things stuff way kind sort lot bit long much many time times true possible able allowed info information
detail details rule rules policy policies someone somebody anybody
"""

_SUFFIXES = (
    ("ification", "ify"), ("ified", "ify"), ("ifies", "ify"), ("ations", ""), ("ation", ""),
    ("ments", ""), ("ment", ""), ("ies", "y"), ("ied", "y"), ("ing", ""), ("als", ""), ("al", ""),
    ("ed", ""), ("es", ""), ("s", ""),
)

_TOKEN = re.compile(r"\d{1,2}:\d{2}|[a-z0-9]+(?:-[a-z0-9]+)*")


def _stem(token: str) -> str:
    if not token.isalpha():
        return token
    for suffix, replacement in _SUFFIXES:
        if suffix == "s" and token.endswith(("ss", "us", "is")):
            continue
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + replacement
            break
    while token.endswith("e") and len(token) > 3:
        token = token[:-1]
    if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "aeious":
        token = token[:-1]
    return token


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'(?:m|re|ve|ll|d)\b", "", text)
    return re.sub(r"'s\b", "", text).replace("'", "")


def analyze(text: str) -> List[str]:
    norm = normalize(text)
    tokens = []
    for raw in _TOKEN.findall(norm):
        parts = [raw] + ([p for p in raw.split("-") if len(p) > 1] if "-" in raw else [])
        tokens.extend(_stem(t) for t in parts if t not in _STOPWORDS)
    return tokens


_FILLER = frozenset(_stem(w) for w in _FILLER_WORDS.split())


def is_numeric(term: str) -> bool:
    return any(ch.isdigit() for ch in term) and not any(ch.isalpha() for ch in term)


@dataclass(frozen=True)
class QueryTerm:
    term: str
    surface: str
    weight: float
    in_kb: bool


@dataclass(frozen=True)
class Hit:
    rank: int
    chunk: Chunk
    score: float
    matched: FrozenSet[str]
    text_matched: FrozenSet[str]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk.chunk_id,
            "doc_title": self.chunk.doc_title,
            "score": round(self.score, 6),
            "chunk_text": self.chunk.text,
        }


def _conflate(vocabulary) -> dict:
    canonical: dict = {}
    for term in sorted(vocabulary, key=lambda t: (len(t), t)):
        root = next((term[:n] for n in (len(term) - 1, len(term) - 2) if n >= 5 and term[:n] in canonical), None)
        canonical[term] = canonical[root] if root else term
    return canonical


class Index:
    K1 = 1.5
    B = 0.75
    TYPO_CUTOFF = 0.85

    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")
        self.chunks = chunks
        raw_counts = [Counter(analyze(f"{c.text}\n{c.doc_title}\n{c.section}")) for c in chunks]
        self.canonical = _conflate(set().union(*raw_counts))
        self.term_counts = [Counter({}) for _ in chunks]
        for counts, raw in zip(self.term_counts, raw_counts):
            for term, n in raw.items():
                counts[self.canonical[term]] += n
        self.text_terms = [frozenset(self.canonical[t] for t in analyze(c.text)) for c in chunks]
        self.body_vocabulary = frozenset().union(*self.text_terms)
        self.lengths = [sum(tc.values()) for tc in self.term_counts]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        self.doc_freq: Counter = Counter()
        for tc in self.term_counts:
            self.doc_freq.update(tc.keys())
        self.vocabulary = sorted(self.doc_freq)

    def idf(self, term: str) -> float:
        n, df = len(self.chunks), max(self.doc_freq.get(term, 0), 1)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query_terms(self, question: str) -> List[QueryTerm]:
        terms: dict = {}
        for surface in _TOKEN.findall(normalize(question)):
            for term in analyze(surface):
                if term in _FILLER and term not in self.body_vocabulary:
                    continue
                if term not in self.doc_freq:
                    if term.isalpha() and len(term) >= 5:
                        close = difflib.get_close_matches(term, self.vocabulary, n=1, cutoff=self.TYPO_CUTOFF)
                        term = close[0] if close else term
                terms.setdefault(self.canonical.get(term, term), surface)
        return [QueryTerm(t, s, self.idf(t), t in self.doc_freq) for t, s in terms.items()]

    def search(self, question: str, k: int) -> Tuple[List[QueryTerm], List[Hit]]:
        terms = self.query_terms(question)
        scored = []
        for i, counts in enumerate(self.term_counts):
            norm = self.K1 * (1 - self.B + self.B * self.lengths[i] / self.avg_length)
            score, matched = 0.0, set()
            for qt in terms:
                tf = counts.get(qt.term, 0)
                if tf:
                    matched.add(qt.term)
                    score += qt.weight * tf * (self.K1 + 1) / (tf + norm)
            scored.append((-round(score, 9), i, score, frozenset(matched)))
        scored.sort()
        hits = [Hit(rank, self.chunks[i], score, matched, matched & self.text_terms[i])
                for rank, (_, i, score, matched) in enumerate(scored[:k], start=1)]
        return terms, hits
