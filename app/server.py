"""Loopback-only, single-request-at-a-time inference server (also safe for PL)."""
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
from PIL import UnidentifiedImageError
from .engine import fixture_frames, image_frames

MAX_BODY = 6 * 1024 * 1024


def infer_request(model, data):
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    if data.get("fixture") is True:
        if not model.metadata.get("verification_fixture"):
            raise ValueError("Fixture replay requires fixture weights")
        frames = fixture_frames()
    else:
        encoded = data.get("image")
        if not isinstance(encoded, str) or len(encoded) > MAX_BODY:
            raise ValueError("Select an image first")
        seed = data.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
            raise ValueError("Seed must be an integer in 0..4294967295")
        content = base64.b64decode(encoded, validate=True)
        frames = image_frames(io.BytesIO(content), data.get("timesteps", 50), seed)
    result = model.infer(frames)
    result.pop("trace_codes", None)
    return result


def handler_for(model):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="application/json"):
            content = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/api/model":
                self.reply(200, {"metadata": model.metadata, "labels": model.labels})
            elif self.path in ("/", "/ui.js", "/style.css"):
                filename = {"/": "index.html", "/ui.js": "ui.js", "/style.css": "style.css"}[self.path]
                kind = {"/": "text/html; charset=utf-8", "/ui.js": "text/javascript; charset=utf-8", "/style.css": "text/css; charset=utf-8"}[self.path]
                self.reply(200, (Path(__file__).parent / "static" / filename).read_bytes(), kind)
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/infer":
                self.reply(404, {"error": "Not found"})
                return
            # Reject browser cross-origin submissions; never enable CORS here.
            if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
                self.reply(403, {"error": "Cross-origin requests are disabled"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self.reply(413, {"error": "Upload must be between 1 byte and 6 MiB"})
                    return
                if self.headers.get_content_type() != "application/json":
                    self.reply(415, {"error": "Send application/json"})
                    return
                data = json.loads(self.rfile.read(length))
                self.reply(200, infer_request(model, data))
            except (ValueError, KeyError, TypeError, UnidentifiedImageError, OSError) as exc:
                self.reply(400, {"error": str(exc)})
            except (RuntimeError, TimeoutError) as exc:
                self.reply(503, {"error": str(exc)})
    return Handler


def serve(model, port):
    server = HTTPServer(("127.0.0.1", port), handler_for(model))
    print(f"SpikeFace: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if hasattr(model, "close"):
            model.close()
