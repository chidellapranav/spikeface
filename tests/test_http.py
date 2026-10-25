import http.client
from http.server import HTTPServer
import json
import threading
import unittest
from app.engine import Model
from app.server import handler_for


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), handler_for(Model.golden()))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=15)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            conn.close()

    def test_assets_and_model(self):
        for path in ["/", "/style.css", "/ui.js", "/api/model"]:
            status, content, headers = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertGreater(len(content), 0)
            self.assertIn("Content-Security-Policy", headers)
        self.assertEqual(self.request("GET", "/../../etc/passwd")[0], 404)

    def test_fixture_inference(self):
        status, body, _ = self.request("POST", "/api/infer", json.dumps({"fixture": True}), {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["prediction"], 12)

    def test_rejected_requests(self):
        self.assertEqual(self.request("POST", "/api/infer", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/infer", "{broken", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/infer", "{}", {"Content-Type": "application/json", "Origin": "https://example.org"})[0], 403)
        self.assertEqual(self.request("POST", "/api/infer", None, {"Content-Length": "7000000"})[0], 413)


if __name__ == "__main__":
    unittest.main()
