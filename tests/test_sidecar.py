from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from q38_sidecar import Q38HttpServer, SidecarApplication  # noqa: E402


class FakeControl:
    model = "Qwen3.8-Flash-Next-Dual-GPU"


class ModelsEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        app = SidecarApplication(FakeControl(), codec=None, api_key="test-key")  # type: ignore[arg-type]
        self.server = Q38HttpServer(("127.0.0.1", 0), app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def get(self, path: str, *, authorized: bool = True) -> tuple[int, dict[str, object]]:
        headers = {"Authorization": "Bearer test-key"} if authorized else {}
        request = urllib.request.Request(self.base + path, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=2)
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read())
            finally:
                error.close()
        with response:
            return response.status, json.loads(response.read())

    def test_lists_the_configured_model(self) -> None:
        status, body = self.get("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], FakeControl.model)  # type: ignore[index]

    def test_gets_one_model_and_rejects_unknown_model(self) -> None:
        encoded = urllib.parse.quote(FakeControl.model, safe="")
        status, body = self.get(f"/v1/models/{encoded}")
        self.assertEqual(status, 200)
        self.assertEqual(body["id"], FakeControl.model)
        missing_status, missing = self.get("/v1/models/not-installed")
        self.assertEqual(missing_status, 404)
        self.assertEqual(missing["error"]["code"], "model_not_found")  # type: ignore[index]

    def test_models_endpoint_requires_the_api_key(self) -> None:
        status, body = self.get("/v1/models", authorized=False)
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["code"], "unauthorized")  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
