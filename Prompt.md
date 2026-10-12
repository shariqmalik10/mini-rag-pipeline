## BUILD

Build a replayable mini-RAG pipeline that ingests a small product knowledge base from local files, indexes it, answers user questions with citation-strict responses, and evaluates retrieval quality deterministically.

This is not a one-shot chatbot task. The evaluator will run your pipeline from a clean checkout, may replace the knowledge base documents and query file with equivalent fixtures, and will verify that retrieval, ranking, and evaluation are implemented in code rather than delegated entirely to the LLM.

Your solution should show practical AI engineering judgment: document preprocessing, chunking choices, retrieval logic, grounded generation, validation, and artifact logging.

---

## INPUT FILES

Your pipeline must read inputs from disk:

- `kb/` containing plain-text knowledge base documents
- `queries.json`

Use the sample fixtures below for local testing. The evaluator may replace them with equivalent files using the same schema. Your implementation must not depend on exact filenames, document order, or exact wording.

---

## SAMPLE `kb/article_01.txt`

```text
Title: Password reset and account recovery
Section: Login help

If a user forgets their password, they can request a reset email from the login page.
Reset links expire after 30 minutes.
For security reasons, support agents cannot see or manually reveal user passwords.
If two-factor authentication is enabled and the user lost access to their authenticator app, backup codes can be used.
If backup codes are unavailable, the user must complete identity verification before 2FA can be reset.
```

## SAMPLE `kb/article_02.txt`

```text
Title: Cash withdrawal processing
Section: Payments

Withdrawal requests submitted before 17:00 UTC are usually reviewed on the same business day.
Bank withdrawals may take 1 to 3 business days after approval.
E-wallet withdrawals are typically processed within 4 hours after approval.
If an account is under compliance review, withdrawals remain pending until the review is completed.
Users can check the withdrawal status from the cashier history page.
```

## SAMPLE `kb/article_03.txt`

```text
Title: Document verification requirements
Section: Verification

Proof of identity documents must be valid and unexpired.
Proof of address documents must show the user's full name, address, and issue date.
Utility bills older than 6 months are not accepted.
Bank statements older than 3 months are not accepted.
Images must be clear, uncropped, and readable.
```

## SAMPLE `kb/article_04.txt`

```text
Title: Demo account behaviour
Section: Trading

Demo accounts use virtual funds and do not reflect real-market execution exactly.
Prices may be delayed in some environments.
Demo profit cannot be withdrawn.
Users should switch to a real account to trade with real funds.
```

## SAMPLE `queries.json`

```json
[
  {
    "query_id": "Q1",
    "question": "I requested a bank withdrawal yesterday. How long can it take after approval?",
    "expected_doc_titles": ["Cash withdrawal processing"],
    "ground_truth_answer_points": [
      "Bank withdrawals may take 1 to 3 business days after approval"
    ]
  },
  {
    "query_id": "Q2",
    "question": "My reset link is not working after an hour. Is that expected?",
    "expected_doc_titles": ["Password reset and account recovery"],
    "ground_truth_answer_points": [
      "Reset links expire after 30 minutes"
    ]
  },
  {
    "query_id": "Q3",
    "question": "Can support tell me what my password currently is?",
    "expected_doc_titles": ["Password reset and account recovery"],
    "ground_truth_answer_points": [
      "Support agents cannot see or manually reveal user passwords"
    ]
  },
  {
    "query_id": "Q4",
    "question": "Will a utility bill from 8 months ago work for proof of address?",
    "expected_doc_titles": ["Document verification requirements"],
    "ground_truth_answer_points": [
      "Utility bills older than 6 months are not accepted"
    ]
  },
  {
    "query_id": "Q5",
    "question": "Can I withdraw profit made on a demo account?",
    "expected_doc_titles": ["Demo account behaviour"],
    "ground_truth_answer_points": [
      "Demo profit cannot be withdrawn"
    ]
  }
]
```

---

## CONTROLLED VOCABULARIES

Define these vocabularies in code and validate outputs against them.

Allowed answer labels:

```text
grounded_answer
insufficient_context
conflicting_context
```

Allowed retrieval statuses:

```text
hit
partial_hit
miss
```

Allowed citation formats:

```text
[doc_title §chunk_id]
```

---

## PIPELINE STAGES

Your implementation must enforce these stages in code:

```text
INIT
 -> DOCUMENTS_LOADED
 -> DOCUMENTS_CHUNKED
 -> INDEX_BUILT
 -> RETRIEVAL_COMPLETE
 -> ANSWERS_GENERATED
 -> EVALUATION_COMPLETE
 -> VALIDATION_COMPLETE
 -> RESULTS_FINALISED
```

Final answers must not be generated before retrieval completes.

---

## MUST COMPLETE

### 1. Document Ingestion and Chunking

Load all documents from `kb/` and chunk them in code.

For each chunk, store at minimum:

```json
{
  "chunk_id": "string",
  "doc_title": "string",
  "section": "string",
  "text": "string",
  "start_char": 0,
  "end_char": 120
}
```

Requirements:

- Parse `Title:` and `Section:` from each file.
- Chunk in deterministic code.
- Preserve chunk text exactly as sourced from the document body.
- Save output to `artifacts/chunks.json`.

---

### 2. Retrieval Pipeline

For each query, retrieve the top chunks from your index.

You may use embeddings, keyword search, hybrid retrieval, or another local approach, but retrieval must be implemented as runnable code in the repository.

Save output to `artifacts/retrieval.json`.

Each retrieval result must include:

```json
{
  "query_id": "Q1",
  "question": "string",
  "top_k": [
    {
      "rank": 1,
      "chunk_id": "string",
      "doc_title": "string",
      "score": 0.0,
      "chunk_text": "string"
    }
  ]
}
```

Requirements:

- Return at least top 3 chunks per query.
- Scores must be numeric.
- The pipeline must work if document filenames change.

---

### 3. Citation-Strict Answer Generation

For each query, generate an answer grounded only in the retrieved chunks.

If the retrieved context is insufficient, the output must say so rather than inventing facts.

Each answer record must include:

```json
{
  "query_id": "Q1",
  "answer_label": "grounded_answer",
  "answer": "Bank withdrawals may take 1 to 3 business days after approval. [Cash withdrawal processing §chunk_2]",
  "citations": ["[Cash withdrawal processing §chunk_2]"],
  "used_chunk_ids": ["chunk_2"]
}
```

Requirements:

- Use only the allowed answer labels.
- Every factual answer must include at least one citation.
- Citations must refer to retrieved chunks only.
- Do not cite chunks that were not retrieved.
- Save output to `artifacts/answers.json`.

---

### 4. Deterministic Retrieval Evaluation

Evaluate retrieval quality in code using `queries.json` ground truth.

For each query, compute at minimum:

- whether the expected document title appears in top 3
- retrieval status: `hit`, `partial_hit`, or `miss`
- a brief deterministic explanation

Save output to `artifacts/eval.json`.

Each evaluation record must include:

```json
{
  "query_id": "Q1",
  "expected_doc_titles": ["Cash withdrawal processing"],
  "retrieved_doc_titles_top3": ["Cash withdrawal processing", "Demo account behaviour"],
  "retrieval_status": "hit",
  "matched_expected_title": true,
  "explanation": "Expected title found at rank 1"
}
```

Also produce an aggregate summary with at least:

```json
{
  "top3_hit_rate": 0.0,
  "total_queries": 5,
  "hits": 0,
  "partial_hits": 0,
  "misses": 0
}
```

---

## SHOULD ATTEMPT

### 5. Answer Grounding Check

Add a deterministic validator that checks whether each citation in `answers.json` exists in `retrieval.json` and whether cited chunk text contains supporting wording for the answer.

You may use a simple heuristic such as phrase overlap, keyword overlap, or exact substring checks.

Save output to `artifacts/grounding_check.json`.

---

### 6. Chunking Comparison

Run the pipeline with two chunking strategies, for example:

- fixed-size chunks
- paragraph-based chunks

Compare aggregate retrieval performance and explain the tradeoff briefly.

Save output to `artifacts/chunking_comparison.json`.

---

## STRETCH

### 7. Minimal API

Expose a small local API endpoint such as:

```text
POST /answer
```

Input:

```json
{ "question": "string" }
```

Output:

```json
{
  "answer_label": "grounded_answer",
  "answer": "string",
  "citations": ["string"]
}
```

---

## REQUIRED ARTIFACTS

Your repository must produce:

- `kb/`
- `queries.json`
- `artifacts/chunks.json`
- `artifacts/retrieval.json`
- `artifacts/answers.json`
- `artifacts/eval.json`
- `artifacts/grounding_check.json`, if attempted
- `artifacts/chunking_comparison.json`, if attempted
- `llm_calls.jsonl`, if an LLM is used

---

## `llm_calls.jsonl` REQUIREMENTS

If you use an LLM for answer generation, log one JSON object per call.

Each record must include:

```json
{
  "stage": "string",
  "query_id": "string | null",
  "timestamp": "ISO-8601 timestamp",
  "provider": "string",
  "model": "string",
  "prompt_hash": "string",
  "input_artifacts": ["path"],
  "output_artifact": "path"
}
```

There must be separate records for each answer-generation call.

If you do not use an LLM, state that clearly in your README and ensure the pipeline still runs.

---

## VALIDATION REQUIREMENTS

The repository must include a validation command, for example:

```bash
make validate
```

or:

```bash
python validate.py
```

The validation command must check that:

- required artifacts exist
- JSON files are valid
- all queries were processed
- each query has at least 3 retrieved chunks
- retrieval scores are numeric
- answer labels use only the controlled vocabulary
- grounded answers include citations
- citations refer only to retrieved chunks
- evaluation statuses use only the controlled vocabulary
- aggregate evaluation summary is present

---

## EXECUTION REQUIREMENTS

The evaluator will run the pipeline from a clean checkout.

Generated artifacts may be deleted before evaluation.

The evaluator may replace `kb/` documents and `queries.json` with equivalent fixtures using the same schema.

Static precomputed outputs are not sufficient.

Your solution must actually run the pipeline and regenerate artifacts.

---

## TOOLS

Any programming language may be used.

Any local retrieval library, vector database, or lightweight framework may be used.

An LLM is optional. If used, answers must still be grounded in retrieved chunks and validated.

---

## TECHNICAL CONSTRAINTS

- Chunking must be implemented in code.
- Retrieval must be implemented in code.
- Evaluation must be deterministic code over the retrieval outputs.
- Do not hardcode answers from the sample queries.
- Do not depend on exact sample filenames or document order.
- Do not use external proprietary data.
- Do not fabricate facts beyond the retrieved context.
- Keep the solution runnable by another engineer from a clean checkout.
