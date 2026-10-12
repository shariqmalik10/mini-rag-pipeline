# Mini-RAG pipeline

This pipeline ingests a plain-text knowledge base from `kb/` and answers the questions in `queries.json` with citation-strict answers. It evaluates retrieval against the ground truth in `queries.json` and validates every artifact it writes. The spec is in [Prompt.md](Prompt.md), and design decisions and edge cases are in [PLAN.md](PLAN.md).

**No LLM is used.** Answers are extractive. Each answer quotes knowledge-base sentences word for word and cites each one, so the pipeline writes no `llm_calls.jsonl`.

## Run

You need Python 3.9 or newer. It uses the standard library only, so there is nothing to install.

```bash
make run        # regenerate artifacts/
make validate   # check artifacts against the spec, write artifacts/grounding_check.json
make test       # unit and end-to-end tests
make serve      # POST /answer on http://127.0.0.1:8000
```

Without `make`:

```bash
python3 run.py --kb kb --queries queries.json --out artifacts
python3 validate.py --queries queries.json --out artifacts
```

The `artifacts/` folder in the repository holds the output of a run on the sample `kb/` and `queries.json`, for reference. `make run` deletes and regenerates every file in it.

`run.py` exits non-zero if any stage fails or validation finds an error. It deletes earlier artifacts first, so a failed run never leaves stale files that would pass validation. Running it again on the same inputs produces byte-identical artifacts.

## Inputs

- `kb/` holds any number of text files, at any depth and with any names. Each file starts with `Title:` and an optional `Section:` header. Headers are case-insensitive and can come in either order. Other `Key: value` lines in the header block are ignored. The body follows the headers. UTF-8, UTF-16 and cp1252 are read. Hidden files are skipped. A file with no title, or a binary file, is skipped with a warning on stderr, so one stray file does not stop the run. `[`, `]` and `§` in a title become `(`, `)` and `S` so citations stay parseable.
- `queries.json` is a list of `{query_id, question, expected_doc_titles, ground_truth_answer_points}`. The last two fields are optional and may be `null`. A record that breaks the schema, such as a missing question or a duplicate id, stops the run with an error naming the record.

## Artifacts

| File | Contents |
| --- | --- |
| `artifacts/chunks.json` | One record per sentence chunk. `start_char` and `end_char` index into the document body after the header, so `body[start_char:end_char] == text`. |
| `artifacts/retrieval.json` | Top 5 chunks per query with BM25 scores. |
| `artifacts/answers.json` | Label, answer, citations, and `used_chunk_ids` per query. |
| `artifacts/eval.json` | Per-query retrieval status and explanation, plus `summary`. |
| `artifacts/grounding_check.json` | For each citation, whether it was retrieved for that query and whether the cited chunk supports the claim before it. |
| `artifacts/chunking_comparison.json` | Sentence and fixed-size chunking compared on the same queries. |
| `artifacts/run_manifest.json` | Stage trace, settings, and input hashes. |

## How it works

The stages run in the order `INIT`, `DOCUMENTS_LOADED`, `DOCUMENTS_CHUNKED`, `INDEX_BUILT`, `RETRIEVAL_COMPLETE`, `ANSWERS_GENERATED`, `EVALUATION_COMPLETE`, `VALIDATION_COMPLETE`, `RESULTS_FINALISED`. `Run.advance` in `rag/pipeline.py` rejects any skipped or repeated stage. Answer generation takes the retrieval results as input, so it cannot run before retrieval.

**Chunking** (`rag/ingest.py`). Each chunk is one sentence. Sentences split at terminal punctuation followed by a capital letter, at blank lines, and at bullets. They do not split inside `17:00`, `1.5`, `e.g.` or `U.S.`, or after a list number such as `1.`. A sentence longer than 500 characters is cut into 250-character windows. A sentence that starts with a pronoun ("This applies...") is joined to the sentence before it. Chunk IDs are `chunk_N` within each document, and documents are ordered by title, so renaming or reordering files changes nothing.

**Retrieval** (`rag/index.py`). Retrieval uses BM25 (k1 1.5, b 0.75), written in the repository. Each chunk is indexed with its document's title and section. The tokenizer lowercases text, expands contractions, strips English function words, and applies a light suffix stemmer (-s, -es, -ed, -ing, -ment, -al, -ation, -ify, trailing e, doubled final consonant). The code holds no vocabulary from the sample KB or queries, and `tests/test_no_hardcoding.py` fails if a sample title, filename, query id, question or answer point appears in the pipeline source. Three rules derive their behaviour from the KB being searched.

- Conversational filler in a question ("yesterday", "currently", "working") is ignored only when no document body uses that word. Questions about a "work permit" KB still match on "work".
- KB terms that differ only by up to two trailing letters, both at least five letters long, are merged. In the sample KB this merges only `withdrawn` into `withdraw`.
- A question word missing from the KB is matched to the closest KB term when they are at least 85% similar, which fixes typos such as "pasword". The candidate terms come from the KB being searched.

**Answering** (`rag/answer.py`). Each content word in the question is weighted by IDF. A word the KB never uses gets the weight of the rarest KB word.

- If no retrieved chunk covers at least 50% of the question's weight, the label is `insufficient_context`.
- If two strong chunks from different documents share at least two question terms but state different numbers, the label is `conflicting_context`, and both are quoted and cited.
- Otherwise the label is `grounded_answer`. The best chunk is quoted first. A further chunk is added when it covers at least 20% more of the question with at least two of its own words, or when it scores close to the best chunk in the same document. At most 3 chunks are quoted. Question words that appear nowhere in the KB are listed after the answer ("The knowledge base does not mention: weekends.").

Ambiguous questions use the existing labels. Several close matches within one document give `grounded_answer` with each sentence cited. Weak matches scattered across documents give `insufficient_context`.

**Evaluation** (`rag/evaluate.py`). The status is `hit` when every expected title appears among the top 3 chunks, `partial_hit` when only some do, and `miss` when none do. `top3_hit_rate` is hits divided by total queries. The summary also reports MRR. It reports how often a ground-truth answer point is supported by a top-3 chunk (`answer_point_recall_top3`) and by the rank-1 chunk (`answer_point_recall_rank1`). Titles are compared ignoring case and spacing.

**Validation** (`rag/validate.py`). The same code runs inside the pipeline and from `validate.py`. It checks every item in the spec's validation list. It also checks that chunk offsets match their text, that retrieved chunks exist in `chunks.json`, that citations listed in the record match the citations in the answer text, and that every claim is supported by its cited chunk. Support means an exact quote, or at least 60% term overlap with the same negation polarity.

## Chunking comparison

On the sample KB both strategies place the expected document in the top 3 for every query. The difference shows in precision. The rank-1 sentence chunk contains the answer point for all 5 queries, against 3 of 5 for 160-character windows. The windows' rank-1 chunk is also about three times longer. Sentence chunks give precise citations and answers with nothing extra. Fixed windows keep neighbouring context but cut facts mid-sentence and drag unrelated text into quoted answers. `artifacts/chunking_comparison.json` recomputes these numbers on every run.

## API

`make serve` starts the API on http://127.0.0.1:8000. `python3 serve.py --kb <dir> --port <n>` serves another KB or port.

```bash
curl -s -X POST http://127.0.0.1:8000/answer -H 'Content-Type: application/json' -d '{"question": "Can I withdraw demo profit?"}'
```

```json
{"answer_label": "grounded_answer", "answer": "Demo profit cannot be withdrawn. [Demo account behaviour §chunk_3]", "citations": ["[Demo account behaviour §chunk_3]"]}
```

Input and output follow the spec: `{"question": string}` in, and `{answer_label, answer, citations}` out. A malformed body returns 400, `GET /answer` returns 405, and an unknown path returns 404. If the KB cannot be loaded, the API returns 503 with the reason. Opening http://127.0.0.1:8000/ in a browser shows a small form that posts to `/answer`. The server checks the KB folder on every request and rebuilds the index when a file is added, changed or removed, so new documents need no restart.

## Tests

`tests/fixtures/alt/` holds a replacement KB. Its files are renamed and nested, its documents are reworded, one file has no section, one query expects two titles, one query has no expected titles, and one expected title is not in the KB. `tests/fixtures/conflict/` holds a KB in which two documents disagree and two documents share an identical sentence. The tests cover the edge cases listed in PLAN.md. The validator tests also confirm that it fails on tampered artifacts, including a citation to a chunk that was not retrieved, an uncited claim, and a negation flip.

## Limits

- Matching is lexical, and the code carries no synonym list. A paraphrase that shares no words with the KB ("get my money out" against "withdrawals") returns `insufficient_context` rather than a guess.
- The pipeline quotes rules and does not reason over them. Asked whether an 8-month-old utility bill is acceptable, it cites "Utility bills older than 6 months are not accepted" and leaves the comparison to the reader.
- Conflict detection only catches differing numbers in closely matching sentences from different documents. It does not catch contradictions phrased in words, or two documents that share a title.
- A question that shares one distinctive word with an unrelated sentence can get that sentence quoted. In a held-out test, "Who founded the company?" quoted a sentence about changing the company name on an invoice.
- Long preambles and multi-part questions dilute coverage, so they abstain more often than short questions.
- The conversational filler list (`_FILLER_WORDS` in `rag/index.py`) is general English, and it is ignored for any word the KB body uses. Most sample questions still depend on it to drop words like "yesterday" and "currently".
- The thresholds (50% coverage, 20% gain, 0.75 score ratios) were set from the sample queries, the edge-case questions in PLAN.md, and off-topic negatives. They were not tuned on expected answers.
