import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rag.answer import answer
from rag.index import Index
from rag.ingest import chunk_documents, kb_files, load_documents
from rag.pipeline import PRIMARY_STRATEGY, TOP_K

MAX_BODY_BYTES = 64 * 1024

TEST_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini-RAG answer</title>
<style>
  :root { color-scheme: light dark; --bg: #fafaf9; --fg: #1c1917; --muted: #78716c; --line: #d6d3d1; --accent: #2563eb; }
  @media (prefers-color-scheme: dark) { :root { --bg: #1c1917; --fg: #f5f5f4; --muted: #a8a29e; --line: #44403c; --accent: #60a5fa; } }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
  main { max-width: 720px; margin: 0 auto; padding: 32px 16px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p { color: var(--muted); margin: 0 0 20px; }
  textarea { box-sizing: border-box; width: 100%; min-height: 88px; padding: 10px; font: inherit; color: inherit;
             background: transparent; border: 1px solid var(--line); border-radius: 8px; }
  button { margin-top: 10px; padding: 8px 16px; font: inherit; color: #fff; background: var(--accent); border: 0; border-radius: 8px; cursor: pointer; }
  pre { margin-top: 20px; padding: 12px; white-space: pre-wrap; word-break: break-word; border: 1px solid var(--line); border-radius: 8px; }
</style>
</head>
<body>
<main>
  <h1>POST /answer</h1>
  <p>Answers come only from the files in the knowledge base folder. Edits to that folder are picked up on the next question.</p>
  <form id="ask">
    <label for="q">Question</label>
    <textarea id="q" name="question" required></textarea>
    <button type="submit">Ask</button>
  </form>
  <pre id="out" aria-live="polite">Response JSON appears here.</pre>
</main>
<script>
  const form = document.getElementById("ask"), out = document.getElementById("out");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    out.textContent = "Asking...";
    try {
      const res = await fetch("/answer", { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify({ question: document.getElementById("q").value }) });
      out.textContent = `HTTP ${res.status}\\n` + JSON.stringify(await res.json(), null, 2);
    } catch (err) {
      out.textContent = "Request failed: " + err;
    }
  });
</script>
</body>
</html>
"""


class LiveIndex:
    def __init__(self, kb_dir: str) -> None:
        self.kb_dir = Path(kb_dir)
        self._lock = threading.Lock()
        self._signature = None
        self._index = None

    def _current_signature(self) -> tuple:
        return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in kb_files(self.kb_dir))

    def get(self) -> Index:
        with self._lock:
            signature = self._current_signature()
            if signature != self._signature:
                self._index = Index(chunk_documents(load_documents(self.kb_dir), PRIMARY_STRATEGY))
                self._signature = signature
            return self._index


def make_handler(live: LiveIndex):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str, extra_headers: dict = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict, extra_headers: dict = None) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8", extra_headers)

        def do_GET(self) -> None:
            if self.path == "/":
                return self._send(200, TEST_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if self.path == "/answer":
                return self._json(405, {"error": "use POST with {\"question\": string}"}, {"Allow": "POST"})
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/answer":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "invalid Content-Length"})
            if length > MAX_BODY_BYTES:
                return self._json(413, {"error": f"body larger than {MAX_BODY_BYTES} bytes"})
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._json(400, {"error": "body must be JSON"})
            if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
                return self._json(400, {"error": 'body must be {"question": string}'})
            try:
                index = live.get()
            except (ValueError, FileNotFoundError) as exc:
                return self._json(503, {"error": f"knowledge base cannot be loaded: {exc}"})
            result = answer(*index.search(payload["question"], TOP_K))
            self._json(200, {k: result[k] for k in ("answer_label", "answer", "citations")})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve POST /answer over the knowledge base.")
    parser.add_argument("--kb", default="kb")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    live = LiveIndex(args.kb)
    live.get()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(live))
    print(f"Serving POST http://{args.host}:{args.port}/answer (test page at http://{args.host}:{args.port}/)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
