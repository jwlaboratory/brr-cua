"""Tiny collector: receives a data-URL PNG via POST and writes it to data/real/.
Used only to grab real browser-rendered frames for validation. Run on :8780."""
import base64, os, urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = os.path.join(os.path.dirname(__file__), "data", "real")
os.makedirs(OUT, exist_ok=True)


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_POST(self):
        q = urllib.parse.urlparse(self.path).query
        name = urllib.parse.parse_qs(q).get("name", ["frame"])[0]
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode("utf-8")
        b64 = body.split(",", 1)[1] if "," in body else body
        with open(os.path.join(OUT, name + ".png"), "wb") as f:
            f.write(base64.b64decode(b64))
        self.send_response(200); self._cors(); self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("collector on :8780 ->", OUT)
    HTTPServer(("127.0.0.1", 8780), H).serve_forever()
