"""Zero-dependency dev server (stdlib only) that mirrors the FastAPI routes.

Use ONLY for offline demos / environments where FastAPI cannot be installed.
Production and Railway use backend/app.py.   Run:  python scripts/dev_server.py [port]
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import service  # noqa: E402


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, (ROOT / "frontend" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/health":
            return self._send(200, {"status": "ok"})
        if u.path == "/api/dashboard":
            ds = parse_qs(u.query).get("dataset", ["extended"])[0]
            return self._send(200 if ds in ("extended", "bundled") else 422,
                              service.dashboard(ds) if ds in ("extended", "bundled") else {"detail": "bad dataset"})
        if u.path == "/api/genai":
            return self._send(200, service.genai_status())
        if u.path == "/api/metrics":
            return self._send(200, service.metrics())
        self._send(404, {"detail": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(422, {"detail": "invalid JSON"})
        if self.path == "/api/run":
            r = service.analyze_text(body.get("text", ""))
        elif self.path == "/api/batch":
            r = service.analyze_batch(body.get("texts", []))
        else:
            return self._send(404, {"detail": "not found"})
        self._send(422 if "error" in r else 200, {"detail": r["error"]} if "error" in r else r)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"dev server on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
