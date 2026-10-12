import json
import shutil
import tempfile
import unittest
from pathlib import Path

from rag.ingest import chunk_documents, load_documents
from rag.pipeline import Run, StageError, ValidationFailed, load_queries, run_pipeline
from rag.validate import ARTIFACTS, validate
from rag.vocab import Stage
from tests.helpers import FIXTURES, SAMPLE_KB, SAMPLE_QUERIES, TempDirTest

SAMPLE_IDS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


class StageMachineTest(unittest.TestCase):
    def test_full_order_is_accepted(self):
        run = Run()
        for stage in list(Stage)[1:]:
            run.advance(stage)
        self.assertEqual(run.trace, [s.value for s in Stage])

    def test_skipping_or_repeating_a_stage_is_rejected(self):
        run = Run()
        run.advance(Stage.DOCUMENTS_LOADED)
        for illegal in (Stage.INDEX_BUILT, Stage.ANSWERS_GENERATED, Stage.DOCUMENTS_LOADED, Stage.INIT):
            with self.assertRaises(StageError):
                run.advance(illegal)


class LoadQueriesTest(TempDirTest):
    def write(self, data):
        path = self.tmp / "q.json"
        path.write_text(json.dumps(data) if not isinstance(data, str) else data)
        return path

    def test_malformed_query_files_fail_clearly(self):
        cases = {
            "duplicate query_id": [{"query_id": "A", "question": "x"}, {"query_id": "A", "question": "y"}],
            "'question' must be a string": [{"query_id": "A"}],
            "non-empty JSON list": {"query_id": "A", "question": "x"},
            "cannot read queries": "{not json",
            "list of strings": [{"query_id": "A", "question": "x", "expected_doc_titles": "Fees"}],
        }
        for message, data in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                load_queries(self.write(data))

    def test_optional_ground_truth_fields_default_to_empty(self):
        for extra in ({}, {"expected_doc_titles": None, "ground_truth_answer_points": None}):
            [query] = load_queries(self.write([{"query_id": "A", "question": "x", **extra}]))
            self.assertEqual((query["expected_doc_titles"], query["ground_truth_answer_points"]), ([], []))


class EndToEndTest(TempDirTest):
    def read_all(self, out):
        return {name: (out / name).read_bytes() for name in [*ARTIFACTS.values(), "run_manifest.json"]}

    def test_sample_run_is_complete_and_byte_identical_on_rerun(self):
        manifest = run_pipeline(SAMPLE_KB, SAMPLE_QUERIES, self.tmp / "a")
        run_pipeline(SAMPLE_KB, SAMPLE_QUERIES, self.tmp / "b")
        self.assertEqual(manifest["stage_trace"], [s.value for s in Stage])
        self.assertEqual(self.read_all(self.tmp / "a"), self.read_all(self.tmp / "b"))
        summary = json.loads((self.tmp / "a" / "eval.json").read_text())["summary"]
        self.assertEqual((summary["total_queries"], summary["hits"]), (5, 5))

    def test_rerun_into_dirty_dir_replaces_stale_artifacts(self):
        out = self.tmp / "out"
        out.mkdir()
        (out / "answers.json").write_text("stale")
        run_pipeline(SAMPLE_KB, SAMPLE_QUERIES, out)
        self.assertEqual(validate(out, SAMPLE_IDS), [])

    def test_replacement_fixture_runs_and_scores_as_expected(self):
        alt = FIXTURES / "alt"
        run_pipeline(alt / "kb", alt / "queries.json", self.tmp)
        results = {r["query_id"]: r for r in json.loads((self.tmp / "eval.json").read_text())["results"]}
        answers = {a["query_id"]: a["answer_label"] for a in json.loads((self.tmp / "answers.json").read_text())}
        self.assertEqual({q: r["retrieval_status"] for q, r in results.items()},
                         {"A1": "hit", "A2": "hit", "A3": "hit", "A4": "miss", "A5": "hit", "A6": "miss"})
        self.assertIn("not a document in the knowledge base", results["A6"]["explanation"])
        self.assertIn("No expected titles", results["A4"]["explanation"])
        self.assertEqual(answers, {"A1": "grounded_answer", "A2": "grounded_answer", "A3": "grounded_answer",
                                   "A4": "insufficient_context", "A5": "grounded_answer", "A6": "insufficient_context"})

    def test_kb_too_small_for_top_3_fails(self):
        kb = self.tmp / "kb"
        kb.mkdir()
        (kb / "a.txt").write_text("Title: Tiny\n\nOnly one sentence.")
        with self.assertRaisesRegex(ValueError, "at least 3"):
            run_pipeline(kb, SAMPLE_QUERIES, self.tmp / "out")


class ValidatorTest(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.golden = Path(tempfile.mkdtemp())
        run_pipeline(SAMPLE_KB, SAMPLE_QUERIES, cls.golden)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.golden, ignore_errors=True)

    def tampered(self, name, mutate):
        out = self.tmp / f"out{len(list(self.tmp.iterdir()))}"
        shutil.copytree(self.golden, out)
        path = out / name
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))
        return validate(out, SAMPLE_IDS)

    def assertCaught(self, errors, fragment):
        self.assertTrue(any(fragment in e for e in errors), f"{fragment!r} not in {errors}")

    def test_real_output_passes(self):
        self.assertEqual(validate(self.golden, SAMPLE_IDS), [])

    def test_citation_to_unretrieved_chunk_is_caught(self):
        def cite_elsewhere(answers):
            bad = "[Demo account behaviour §chunk_4]"
            answers[0]["citations"] = [bad]
            answers[0]["answer"] = f"Users should switch to a real account to trade with real funds. {bad}"
        errors = self.tampered("answers.json", cite_elsewhere)
        self.assertCaught(errors, "which was not retrieved for it")
        self.assertCaught(errors, "Q1 answer is not grounded")

    def test_uncited_or_unsupported_claims_are_caught(self):
        def add_claim(answers):
            answers[1]["answer"] += " Links can be extended by support."
        self.assertCaught(self.tampered("answers.json", add_claim), "Q2 answer is not grounded")

        def swap_claim(answers):
            citation = answers[2]["citations"][0]
            answers[2]["answer"] = f"Support agents can reveal passwords on request. {citation}"
        self.assertCaught(self.tampered("answers.json", swap_claim), "Q3 answer is not grounded")

    def test_vocabulary_and_shape_violations_are_caught(self):
        cases = [
            ("answers.json", lambda d: d[0].update(answer_label="maybe"), "label 'maybe' not in"),
            ("answers.json", lambda d: d[0].update(citations=[], answer="Bank withdrawals take days."), "has no citations"),
            ("answers.json", lambda d: d[0].update(citations=["Cash withdrawal processing chunk_2"]), "not a citation"),
            ("answers.json", lambda d: d.pop(), "queries not processed: ['Q5']"),
            ("retrieval.json", lambda d: d[0].update(top_k=d[0]["top_k"][:2]), "needs at least 3"),
            ("retrieval.json", lambda d: d[0]["top_k"][0].update(score="0.9"), "score is not numeric"),
            ("retrieval.json", lambda d: d[0]["top_k"][0].update(chunk_text="invented"), "does not match a chunk"),
            ("eval.json", lambda d: d["results"][0].update(retrieval_status="near_hit"), "status 'near_hit' not in"),
            ("eval.json", lambda d: d["summary"].pop("top3_hit_rate"), "summary.top3_hit_rate missing"),
            ("eval.json", lambda d: d.pop("summary"), "'results' and 'summary'"),
        ]
        for name, mutate, fragment in cases:
            with self.subTest(fragment):
                self.assertCaught(self.tampered(name, mutate), fragment)

    def test_missing_and_invalid_artifacts_are_caught(self):
        out = self.tmp / "out"
        shutil.copytree(self.golden, out)
        (out / "chunking_comparison.json").unlink()
        (out / "eval.json").write_text("{broken")
        errors = validate(out, SAMPLE_IDS)
        self.assertCaught(errors, "missing artifact")
        self.assertCaught(errors, "eval.json: invalid JSON")

    def test_failed_validation_stops_before_finalising(self):
        kb = self.tmp / "kb"
        shutil.copytree(SAMPLE_KB, kb)
        queries = self.tmp / "q.json"
        queries.write_text(json.dumps([{"query_id": "X", "question": "demo profit"}]))
        import rag.pipeline as pipeline
        original = pipeline.validate
        pipeline.validate = lambda out, ids: ["forced failure"]
        try:
            with self.assertRaises(ValidationFailed):
                run_pipeline(kb, queries, self.tmp / "out")
        finally:
            pipeline.validate = original
        self.assertFalse((self.tmp / "out" / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
