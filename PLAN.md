# Mini-RAG pipeline plan

Status: implemented. See [README.md](README.md) for usage. The spec is [Prompt.md](Prompt.md). Where the build diverged from this plan, the section below says so.

## Decisions

| Area | Choice | Why |
| --- | --- | --- |
| Language | Python 3.9+, standard library only | Runs from a clean checkout with no install step. Local `python3` is 3.9.6. |
| LLM | None. Answers are extractive. | Requested. No `llm_calls.jsonl` is produced and the README says so. |
| Primary chunking | Sentence chunks, one per sentence in the body | Each sample line is one fact, so a sentence chunk is the unit a citation should point at. |
| Comparison chunking | Fixed-size character windows with overlap | Required by item 6. Paragraph chunking collapses each sample doc into one chunk, so it is a weaker contrast. |
| Retrieval | BM25 written by hand, with title and section tokens added to each chunk's index terms | Deterministic, explainable, and no dependencies. The title boost helps short chunks match on topic words. |
| Tokenizer | Lowercase, alphanumeric tokens, English function-word stopwords, light suffix stemmer, conversational filler ignored in questions only when the KB body never uses it, and a typo fallback to the closest KB term | Handles word forms and misspellings with general English rules only. The code holds no vocabulary from the sample KB or queries. |
| Answering | Quote the best supporting chunk verbatim and append its citation | Cannot fabricate. Every claim is a source substring. |
| Abstaining | `insufficient_context` when the best chunk covers too little of the question's weighted content terms. Words the KB never uses carry the highest weight. | A question whose key words never appear in the KB cannot be answered from it, however well its common words match. Threshold set from sample queries plus off-topic negatives, never from sample answers. |
| Ambiguity | Mapped onto the existing labels. Several close matches in one article give `grounded_answer` with each sentence cited. Weak, scattered matches give `insufficient_context`. | The controlled vocabulary already covers it, so no new label. |
| Conflict | `conflicting_context` only when two top chunks from different docs both pass the threshold, score within a margin, and state different numbers for the same query terms | Narrow rule. It may never fire on the samples, which is correct. |
| Stage order | `Stage` enum plus a transition table. A `Run` object refuses any out-of-order `advance`. | Answers take the retrieval result object as input, so they cannot be generated earlier. |
| Determinism | Sorted inputs, ties broken by `(-score, doc_title, chunk index)`, scores rounded to 6 places, no timestamps in artifacts | Two runs produce byte-identical artifacts. |

## Data shapes

```text
Document     { doc_title, section, body, source_path }
Chunk        { chunk_id, doc_title, section, text, start_char, end_char, strategy }
             invariant: document.body[start_char:end_char] == text
ScoredChunk  { rank, chunk_id, doc_title, score, chunk_text }
Retrieval    { query_id, question, top_k: [ScoredChunk] }        top_k length 5, eval reads first 3
Answer       { query_id, answer_label, answer, citations, used_chunk_ids }
EvalRecord   { query_id, expected_doc_titles, retrieved_doc_titles_top3,
               retrieval_status, matched_expected_title, explanation, first_match_rank }
EvalSummary  { top3_hit_rate, total_queries, hits, partial_hits, misses, mrr }
```

Controlled vocabularies live in one module as enums (`AnswerLabel`, `RetrievalStatus`, `Stage`) with one citation formatter and one citation parser regex. Every writer and the validator import them from there.

Proposed definitions, pending confirmation:

- `chunk_id` is `chunk_N`, numbered from 1 within each document. Lookups key on `(doc_title, chunk_id)`. This reproduces the spec's example `[Cash withdrawal processing §chunk_2]` exactly.
- `start_char`/`end_char` index into the document body after the `Title:`/`Section:` header.
- `hit` means every expected title appears in the top 3. `partial_hit` means some but not all do. `miss` means none do. `top3_hit_rate` is hits divided by total.
- `retrieved_doc_titles_top3` holds the distinct titles of the top 3 chunks in rank order, as in the spec example.

## Layout

```text
rag/vocab.py      enums, stage transition table, citation format and parse
rag/ingest.py     Document and Chunk, file discovery, header parsing, chunking strategies (registry dict)
rag/index.py      QueryTerm and Hit, tokenizer, BM25 index, search
rag/answer.py     extractive answer policy and labels
rag/evaluate.py   per-query eval, summary, chunking comparison
rag/validate.py   artifact checks and grounding check
rag/pipeline.py   Run state machine, artifact writes
run.py            python3 run.py [--kb kb --queries queries.json --out artifacts]
validate.py       python3 validate.py [--out artifacts]
serve.py          POST /answer on http.server (stretch)
tests/            unittest suites and a second fixture set
Makefile          run, validate, test, serve, clean
README.md
```

## Edge cases

Every row is a test with an expected answer label, not expected wording. Examples run against the sample KB unless marked as a fixture.

### Query wording

| Case | Example | Expected |
| --- | --- | --- |
| Paraphrase | "How do I get my money out to my bank?" | `insufficient_context`. The code has no synonym list, because one would encode sample-KB vocabulary. A paraphrase sharing no words with the KB abstains rather than guessing. |
| Abbreviation | "I lost my 2FA app, what now?" | `grounded_answer` from the password article, matched on "lost" and "app". `2FA` and `two-factor` are not linked. |
| Typo | "pasword reset link expired", "withdrawl time" | The typo fallback maps each to the closest KB term. `grounded_answer`. |
| Word forms | "withdraw" vs "withdrawn" vs "withdrawals" | One stem. Covered by Q1 and Q5. |
| Special tokens | "17:00", "e-wallet", a curly apostrophe | Times and hyphenated words stay single tokens. Apostrophes are normalized. |

### Cross-topic word collisions

| Case | Example | Expected |
| --- | --- | --- |
| "bank" | "Is a bank statement from 2 months ago OK?" | Top chunk is the bank statements sentence, not bank withdrawals. |
| "withdraw" | Q5, demo profit | Top chunk is the demo article. |
| "verification" | "How do I verify my identity to reset 2FA?" | Top chunk is the password article. |

### Ambiguous questions

| Case | Example | Expected |
| --- | --- | --- |
| Same article, several answers | "How long does a withdrawal take?" | `grounded_answer` citing every close-scoring withdrawal sentence. |
| Spread across articles | "Is there a time limit?" | `insufficient_context`. |
| No content words | "Is it expected?", "help" | `insufficient_context`. |

### Missing or partial context

| Case | Example | Expected |
| --- | --- | --- |
| Off-topic | "What is the capital of France?" | `insufficient_context`. |
| Close but not covered | "How long do card deposits take?" | `insufficient_context`. The key words "card" and "deposits" never appear in the KB. |
| Half covered | "How long do e-wallet withdrawals take on weekends?" | `grounded_answer` quoting the 4-hour sentence, plus a note that the KB does not mention weekends. |
| Wrong premise | "Why do reset links expire after 10 minutes?" | `grounded_answer` quoting the 30-minute sentence. |

### Multi-sentence answers

| Case | Example | Expected |
| --- | --- | --- |
| Two topics | "How long do bank withdrawals take, and can I withdraw demo profit?" | `grounded_answer` citing both articles. |
| Two sentences, one article | "My account is under compliance review. How long will my bank withdrawal take?" | `grounded_answer` citing the compliance and bank sentences. |
| Pronoun-led sentence | Fixture sentence starting "This..." | Chunked together with the sentence before it. |

### KB conflicts (fixture)

| Case | Expected |
| --- | --- |
| A second article says bank withdrawals take 2 to 5 business days | `conflicting_context`, citing both. |
| Two articles with an identical sentence | Not a conflict. |

### Malformed input

| Case | Expected |
| --- | --- |
| Empty or whitespace question | `insufficient_context`, no crash. |
| Prompt-injection text | Ordinary question. There is no LLM to obey it. |
| Duplicate `query_id`, missing `question`, non-list file | Load fails with a clear error. |
| Empty `expected_doc_titles` | Evaluated as `miss` with an explanation saying nothing was expected. |
| Expected title absent from the KB | `miss`, and the explanation says so. Titles compare ignoring case and spacing. |

### KB files

| Case | Expected |
| --- | --- |
| Hidden files such as `.DS_Store` | Skipped. |
| CRLF line endings, UTF-8 BOM | Parsed the same as clean files. |
| Lowercase headers, swapped order, missing `Section:` | Parsed. A missing section is an empty string. |
| Missing `Title:` | Load fails naming the file. |
| `§`, `[` or `]` in a title | Load fails, since the citation format cannot carry it. |
| Two files with the same title | Chunk numbering continues across both, so citations stay unique. |
| Fewer than 3 chunks in the KB | Pipeline fails with a clear error. |
| "17:00", "e.g.", "1.5" inside a sentence | No sentence split. |
| Renamed or shuffled files | Byte-identical artifacts. |

## Steps

Each step ends in a check that must pass before the next starts.

0. **Repo setup.** `git init`, `.gitignore` for `artifacts/` and `__pycache__/`. Check that `git status` shows only source and fixtures.
1. **Vocabulary and models.** Enums, transition table, citation format and parse, dataclasses. Check with unit tests that an illegal stage jump raises and that citation format and parse round-trip.
2. **Ingestion.** Discover every regular file under `kb/` regardless of name. Parse `Title:` and `Section:` tolerantly (case, spacing, either order, missing section). Split the body into sentences with character spans. Write `artifacts/chunks.json`. Check that the offset invariant holds for every chunk and that renaming and shuffling the KB files yields identical chunks.
3. **Index and retrieval.** Tokenizer, BM25 with title boost, top 5 per query. Write `artifacts/retrieval.json`. Check that every query has at least 3 numeric scores, and print the score table for the samples plus a few off-topic questions.
4. **Threshold sketch.** Use the score table from step 3 to set the abstain threshold and conflict margin. Record the chosen values and the reasoning in the README. Check that every sample is answerable and every off-topic question returns `insufficient_context`.
5. **Answer generation.** Extractive answers, labels, citations from retrieved chunks only. Write `artifacts/answers.json`. Check that the answer text minus its citation is a substring of the cited chunk.
6. **Evaluation.** Per-query records and summary. Also report whether each `ground_truth_answer_point` appears in the top 3 chunks, so those fields are used. Write `artifacts/eval.json`. Check against hand-computed expectations for the samples.
7. **Validation and grounding.** One module serves both the pipeline's `VALIDATION_COMPLETE` stage and `python3 validate.py`. It checks every item in the spec's validation list, then writes `artifacts/grounding_check.json` with citation existence and token overlap per citation. Check that it passes on real output and fails on tampered copies (a citation to an unretrieved chunk, a bad label, a query with 2 chunks, a string score, a missing summary).
8. **Chunking comparison.** Run chunk, index, retrieve, and evaluate for both strategies in memory. Write `artifacts/chunking_comparison.json` with both summaries, answer-point recall, and a short tradeoff note. Check that it is present and both strategies cover all queries.
9. **Pipeline wiring.** `Run` advances through every stage in order and writes `artifacts/run_manifest.json` with the stage trace, input file hashes, and strategy. A failed validation exits nonzero before `RESULTS_FINALISED`. Check with `rm -rf artifacts && make run && make validate`, then run again and diff the two artifact sets.
10. **Fixture swap.** A second KB under `tests/fixtures/` with different filenames, reworded docs, an extra document, a multi-title query, and an unanswerable query. Check that the pipeline and validator pass on it unchanged.
11. **API (stretch).** `POST /answer` reuses the built index and the answer policy. Check with curl for an answerable and an unanswerable question.
12. **README.** Run commands, design choices, the no-LLM statement, the thresholds, and known limits. Check by following it from a fresh clone.

## Throughput checkpoint

- **Blocking first steps.** Steps 0 to 2 fix the data shapes everything else imports.
- **Independent workstreams.** n/a. Steps 3 to 8 each consume the previous step's output in a codebase of roughly 600 lines.
- **Shared mutable state.** Only `artifacts/`, written by the single `Run` object.
- **Smallest safe decomposition.** One implementation owner, because every step depends on the one before it.

## Changes found during implementation

- A word the KB never uses was first weighted as if it appeared in zero chunks. In a small KB that let one unknown word ("anything", "cost") veto an otherwise covered question. It now gets the weight of the rarest KB word.
- The first quoted sentence is the highest-ranked chunk that passes the coverage gate, not the chunk with the most coverage. Coverage-first let a sentence matching one rare word outrank the best BM25 match.
- Conflict detection compares every pair of strong candidates. Comparing only against the first sentence missed a direct contradiction in the conflict fixture.
- A further sentence needs at least two question terms in its own text. Without that rule, "an hour" pulled the e-wallet sentence into the reset-link answer.
- The grounding check's overlap fallback requires matching negation. A tampered "Support agents can reveal passwords" otherwise passed against "cannot ... reveal".
- The chunking comparison tied at 1.0 on hit rate, so it also reports answer-point recall at rank 1 and average rank-1 chunk length. Those separate the strategies.
- Data classes live in the module that owns them, so there is no separate `models.py`.
- A hard-coding audit removed the phrase-synonym table (it mapped "cash out", "two-factor" and "e-wallet" onto sample-KB words) and the irregular-verb table (its first entry existed for Q5). A general rule now strips the "n" from "-wn" participles. Conversational filler used to be dropped from the index too, which would have discarded topic words such as "work" or "time" in other KBs. Filler is now ignored only in questions, and only when no document body uses the word.
- The API rebuilds its index when the KB folder changes and serves a test form at `/`.
- An independent stress test on 9 unseen-domain KBs (HR, SaaS, airline, pharmacy, IT, a 40-document campus KB, conflict and chunk-trap cases) and 27 malformed inputs found no crashes and no validation failures. It found answer-quality gaps, which led to these fixes: contraction expansion, -ment and doubled-consonant stemming, a broader filler list, KB-driven merging of term variants (replacing the "-wn" rule), splitting after single-letter words, keeping list numbers with their item, a 500-character sentence cap, tolerated extra header lines, skipping untitled or binary files instead of aborting, cp1252 and UTF-16 decoding, and replacing citation-unsafe title characters. Across the 119 held-out questions, grounded answers went from 66 to 68. Four questions that used to abstain now get correct answers, and two wrong quotes are gone. One answer regressed. "Maximum vitamin C dose" now abstains, because correct sentence splitting separates "Take vitamin C." from "Do not exceed 1000 mg per day.".

## Defaults applied

These were open and now use the proposed defaults. Each is a one-module change if you want the alternative.

1. `chunk_id` is `chunk_N` within each document, looked up by `(doc_title, chunk_id)`.
2. `hit` means every expected title is in the top 3, `partial_hit` some, `miss` none.
3. The stretch API is in scope, on the standard library HTTP server.
