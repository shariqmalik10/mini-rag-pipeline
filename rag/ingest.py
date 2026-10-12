from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .vocab import citation_safe

Span = Tuple[int, int]


@dataclass(frozen=True)
class Document:
    title: str
    section: str
    body: str
    source: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_title: str
    section: str
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        return asdict(self)


_HEADER = re.compile(r"([A-Za-z][A-Za-z0-9 _-]{0,30}?)\s*:\s*(.*)")
_KNOWN_HEADERS = ("title", "section")
MAX_SENTENCE_CHARS = 500

log = logging.getLogger("rag")


def parse_document(raw: str, source: str) -> Document:
    text = raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    fields: Dict[str, str] = {}
    in_header_block = True
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            pos += len(line)
            in_header_block = in_header_block and not fields
            continue
        match = _HEADER.fullmatch(stripped)
        key = match[1].strip().lower() if match else None
        new_known = key in _KNOWN_HEADERS and key not in fields
        if not (match and (new_known or (in_header_block and key not in fields))):
            break
        fields[key] = " ".join(match[2].split())
        pos += len(line)

    title = fields.get("title", "")
    if not title:
        raise ValueError(f"{source}: missing or empty 'Title:' header")
    return Document(title=citation_safe(title), section=fields.get("section", ""), body=text[pos:], source=source)


def kb_files(kb_dir: str | Path) -> List[Path]:
    root = Path(kb_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"knowledge base directory not found: {root}")
    return [p for p in sorted(root.rglob("*"))
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)]


def _decode(data: bytes, source: str) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data:
        raise ValueError(f"{source}: binary file")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        log.warning("%s: not UTF-8, decoded as cp1252", source)
        return data.decode("cp1252", errors="replace")


def load_documents(kb_dir: str | Path) -> List[Document]:
    root = Path(kb_dir)
    docs = []
    for path in kb_files(root):
        rel = str(path.relative_to(root))
        try:
            docs.append(parse_document(_decode(path.read_bytes(), rel), rel))
        except ValueError as exc:
            log.warning("skipping %s", exc)
    if not docs:
        raise ValueError(f"no usable documents found in {root}")
    return sorted(docs, key=lambda d: (d.title, d.section, d.body))


def _trim(body: str, start: int, end: int) -> Span:
    while start < end and body[start].isspace():
        start += 1
    while end > start and body[end - 1].isspace():
        end -= 1
    return start, end


_SENTENCE_BREAK = re.compile(
    r"(?<=[.!?])\s+(?=[^\sa-z])"
    r"|\n[ \t]*\n\s*"
    r"|\n(?=[ \t]*(?:[-*•]|\d+[.)])\s)"
)
_ABBREVIATION_END = re.compile(r"(?:\b(?:e\.g|i\.e|vs|approx|dr|mr|mrs|ms)|\b(?:[A-Za-z]\.)+[A-Za-z])\.$|^\s*\d{1,3}\.$", re.IGNORECASE)
_PRONOUN_LEAD = re.compile(r"(?:this|that|these|those|it|they|such)\b", re.IGNORECASE)


def sentence_spans(body: str) -> List[Span]:
    pieces: List[Span] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(body):
        if body[match.start() - 1] == "." and _ABBREVIATION_END.search(body[start:match.start()]):
            continue
        pieces.append((start, match.start()))
        start = match.end()
    pieces.append((start, len(body)))

    spans: List[Span] = []
    for s, e in (_trim(body, s, e) for s, e in pieces):
        if s == e:
            continue
        leans_on_previous = _PRONOUN_LEAD.match(body, s) and spans and body.count("\n", spans[-1][1], s) < 2
        if leans_on_previous:
            spans[-1] = (spans[-1][0], e)
        else:
            spans.append((s, e))

    bounded: List[Span] = []
    for s, e in spans:
        if e - s <= MAX_SENTENCE_CHARS:
            bounded.append((s, e))
        else:
            bounded.extend((s + a, s + b) for a, b in fixed_spans(body[s:e], size=MAX_SENTENCE_CHARS // 2, overlap=0))
    return bounded


def fixed_spans(body: str, size: int = 160, overlap: int = 40) -> List[Span]:
    spans: List[Span] = []
    start, _ = _trim(body, 0, len(body))
    while start < len(body):
        end = min(start + size, len(body))
        if end < len(body) and not body[end].isspace():
            cut = max(body.rfind(" ", start, end), body.rfind("\n", start, end))
            if cut > start:
                end = cut
        s, e = _trim(body, start, end)
        if s < e:
            spans.append((s, e))
        if end >= len(body):
            break
        nxt = max(end - overlap, start + 1)
        while nxt < end and not body[nxt - 1].isspace():
            nxt += 1
        start, _ = _trim(body, nxt, len(body))
    return spans


STRATEGIES: Dict[str, Callable[[str], List[Span]]] = {
    "sentence": sentence_spans,
    "fixed": fixed_spans,
}


def chunk_documents(docs: List[Document], strategy: str) -> List[Chunk]:
    split = STRATEGIES[strategy]
    numbering: Counter = Counter()
    chunks = []
    for doc in docs:
        for start, end in split(doc.body):
            numbering[doc.title] += 1
            chunks.append(Chunk(
                chunk_id=f"chunk_{numbering[doc.title]}",
                doc_title=doc.title,
                section=doc.section,
                text=doc.body[start:end],
                start_char=start,
                end_char=end,
            ))
    return chunks
