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

from q38_sidecar import (  # noqa: E402
    Q38HttpServer,
    SidecarApplication,
    _function_tools,
    _normalize_messages,
    _parse_tool_output,
    _tool_choice,
    _validate_output_tools,
)
from q38_control_plane import ControlPlaneError  # noqa: E402


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


class ToolProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

    def test_parses_qwen_tool_xml_into_openai_calls(self) -> None:
        text = """check the city
</think>

<tool_call>
<function=get_weather>
<parameter=city>
上海
</parameter>
<parameter=days>
2
</parameter>
</function>
</tool_call><|im_end|>"""
        reasoning, content, calls = _parse_tool_output(text, "chatcmpl-q38-42")
        self.assertEqual(reasoning, "check the city")
        self.assertIsNone(content)
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")  # type: ignore[index]
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),  # type: ignore[index]
            {"city": "上海", "days": 2},
        )

    def test_normalizes_openai_tool_history_for_qwen_template(self) -> None:
        messages = _normalize_messages(
            [
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"上海"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
            ]
        )
        function = messages[1]["tool_calls"][0]["function"]  # type: ignore[index]
        self.assertEqual(function["arguments"], {"city": "上海"})

    def test_tool_choice_filters_or_disables_tools(self) -> None:
        tools = _function_tools(self.tools)
        selected, instruction = _tool_choice("required", tools)
        self.assertEqual(selected, tools)
        self.assertIn("must call", instruction)
        disabled, instruction = _tool_choice("none", tools)
        self.assertEqual(disabled, [])
        self.assertIsNone(instruction)
        forced, instruction = _tool_choice(
            {"type": "function", "function": {"name": "get_weather"}}, tools
        )
        self.assertEqual(len(forced), 1)
        self.assertIn("get_weather", instruction)

    def test_rejects_missing_or_unavailable_model_tool_call(self) -> None:
        tools = _function_tools(self.tools)
        with self.assertRaises(ControlPlaneError) as missing:
            _validate_output_tools([], tools, True)
        self.assertEqual(missing.exception.code, "tool_call_required")
        _reasoning, _content, calls = _parse_tool_output(
            "<tool_call><function=delete_world></function></tool_call>",
            "chatcmpl-q38-43",
        )
        with self.assertRaises(ControlPlaneError) as unavailable:
            _validate_output_tools(calls, tools, False)
        self.assertEqual(unavailable.exception.code, "unknown_tool_call")


if __name__ == "__main__":
    unittest.main()
