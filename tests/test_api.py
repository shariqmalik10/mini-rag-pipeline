import json
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from serve import LiveIndex, make_handler
from tests.helpers import SAMPLE_KB, TempDirTest


class AnswerApiTest(TempDirTest):
    def setUp(self):
        super().setUp()
        self.kb = self.tmp / "kb"
        shutil.copytree(SAMPLE_KB, self.kb)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(LiveIndex(str(self.kb))))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def request(self, path, body=None, method="POST"):
        data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        try:
            with urllib.request.urlopen(req) as res:
                return res.status, res.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode()

    def test_answer_matches_the_spec_shape(self):
        status, body = self.request("/answer", {"question": "Can I withdraw profit made on a demo account?"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {
            "answer_label": "grounded_answer",
            "answer": "Demo profit cannot be withdrawn. [Demo account behaviour §chunk_3]",
            "citations": ["[Demo account behaviour §chunk_3]"],
        })

    def test_unanswerable_question_returns_insufficient_context(self):
        status, body = self.request("/answer", {"question": "What is the capital of France?"})
        self.assertEqual((status, json.loads(body)["answer_label"], json.loads(body)["citations"]),
                         (200, "insufficient_context", []))

    def test_bad_requests_are_rejected(self):
        self.assertEqual(self.request("/answer", b"not json")[0], 400)
        self.assertEqual(self.request("/answer", {"q": "x"})[0], 400)
        self.assertEqual(self.request("/answer", {"question": 5})[0], 400)
        self.assertEqual(self.request("/answer", method="GET")[0], 405)
        self.assertEqual(self.request("/other", {"question": "x"})[0], 404)

    def test_test_page_is_served(self):
        status, body = self.request("/", method="GET")
        self.assertEqual(status, 200)
        self.assertIn('fetch("/answer"', body)

    def test_new_kb_file_is_answered_without_restart(self):
        question = {"question": "How many vacation days do new employees get?"}
        self.assertEqual(json.loads(self.request("/answer", question)[1])["answer_label"], "insufficient_context")
        (self.kb / "leave.txt").write_text(
            "Title: Annual leave\nSection: HR\n\nNew employees get 20 vacation days per year.\nUnused days expire in March.\n")
        result = json.loads(self.request("/answer", question)[1])
        self.assertEqual(result["answer_label"], "grounded_answer")
        self.assertEqual(result["citations"], ["[Annual leave §chunk_1]"])

    def test_untitled_file_is_skipped_and_missing_kb_returns_503(self):
        (self.kb / "notes.txt").write_text("No title header here.")
        self.assertEqual(self.request("/answer", {"question": "demo profit"})[0], 200)
        shutil.rmtree(self.kb)
        status, body = self.request("/answer", {"question": "demo profit"})
        self.assertEqual(status, 503)
        self.assertIn("knowledge base directory not found", json.loads(body)["error"])
