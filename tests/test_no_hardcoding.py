import json
import unittest

from rag.ingest import kb_files, load_documents
from tests.helpers import ROOT, SAMPLE_KB, SAMPLE_QUERIES

SOURCES = sorted((ROOT / "rag").glob("*.py")) + [ROOT / name for name in ("run.py", "validate.py", "serve.py")]


class NoSampleDataInCodeTest(unittest.TestCase):
    def test_pipeline_source_contains_no_sample_input_data(self):
        queries = json.loads(SAMPLE_QUERIES.read_text())
        needles = {d.title for d in load_documents(SAMPLE_KB)}
        needles |= {p.name for p in kb_files(SAMPLE_KB)} | {p.stem for p in kb_files(SAMPLE_KB)}
        for q in queries:
            needles |= {q["query_id"], q["question"], *q["expected_doc_titles"], *q["ground_truth_answer_points"]}
        for path in SOURCES:
            source = path.read_text().casefold()
            for needle in needles:
                with self.subTest(file=path.name, needle=needle):
                    self.assertNotIn(needle.casefold(), source)


if __name__ == "__main__":
    unittest.main()
