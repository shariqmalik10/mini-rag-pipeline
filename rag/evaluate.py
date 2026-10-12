from __future__ import annotations

from typing import Dict, List

from .index import Hit, analyze
from .vocab import RetrievalStatus, citation_safe

TOP_N = 3
ANSWER_POINT_MIN_OVERLAP = 0.8


def _norm_title(title: str) -> str:
    return " ".join(citation_safe(title).split()).casefold()


def point_supported(point: str, text: str) -> bool:
    point_terms = set(analyze(point))
    if not point_terms:
        return False
    return len(point_terms & set(analyze(text))) / len(point_terms) >= ANSWER_POINT_MIN_OVERLAP


def evaluate_query(query: dict, hits: List[Hit], kb_titles: List[str]) -> dict:
    top = hits[:TOP_N]
    top_titles: List[str] = []
    for hit in top:
        if hit.chunk.doc_title not in top_titles:
            top_titles.append(hit.chunk.doc_title)

    first_rank: Dict[str, int] = {}
    for hit in top:
        first_rank.setdefault(_norm_title(hit.chunk.doc_title), hit.rank)

    expected = query["expected_doc_titles"]
    known = {_norm_title(t) for t in kb_titles}
    found = [t for t in expected if _norm_title(t) in first_rank]
    if not expected:
        status, explanation = RetrievalStatus.MISS, "No expected titles provided, so retrieval cannot be scored"
    else:
        status = (RetrievalStatus.HIT if len(found) == len(expected)
                  else RetrievalStatus.PARTIAL_HIT if found else RetrievalStatus.MISS)
        notes = []
        for title in expected:
            norm = _norm_title(title)
            if norm in first_rank:
                notes.append(f"'{title}' found at rank {first_rank[norm]}")
            elif norm not in known:
                notes.append(f"'{title}' is not a document in the knowledge base")
            else:
                notes.append(f"'{title}' not in top {TOP_N}")
        explanation = "; ".join(notes)

    ranks = [first_rank[_norm_title(t)] for t in found]
    points = query["ground_truth_answer_points"]
    return {
        "query_id": query["query_id"],
        "expected_doc_titles": expected,
        "retrieved_doc_titles_top3": top_titles,
        "retrieval_status": status.value,
        "matched_expected_title": bool(found),
        "explanation": explanation,
        "first_match_rank": min(ranks) if ranks else None,
        "answer_points_in_top3": [any(point_supported(p, h.chunk.text) for h in top) for p in points],
        "answer_points_at_rank1": [bool(top) and point_supported(p, top[0].chunk.text) for p in points],
    }


def summarize(records: List[dict]) -> dict:
    total = len(records)
    count = {s.value: sum(r["retrieval_status"] == s.value for r in records) for s in RetrievalStatus}
    points = [ok for r in records for ok in r["answer_points_in_top3"]]
    points_rank1 = [ok for r in records for ok in r["answer_points_at_rank1"]]
    return {
        "top3_hit_rate": round(count["hit"] / total, 4) if total else 0.0,
        "total_queries": total,
        "hits": count["hit"],
        "partial_hits": count["partial_hit"],
        "misses": count["miss"],
        "mrr_top3": round(sum(1 / r["first_match_rank"] for r in records if r["first_match_rank"]) / total, 4) if total else 0.0,
        "answer_point_recall_top3": round(sum(points) / len(points), 4) if points else None,
        "answer_point_recall_rank1": round(sum(points_rank1) / len(points_rank1), 4) if points_rank1 else None,
    }
