from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import api as api_module  # noqa: E402

# The date block of Llama 3.2's chat template, verbatim.
LLAMA_DATE_BLOCK = (
    '{%- if not date_string is defined %}\n'
    '    {%- if strftime_now is defined %}\n'
    '        {%- set date_string = strftime_now("%d %b %Y") %}\n'
    '    {%- else %}\n'
    '        {%- set date_string = "26 Jul 2024" %}\n'
    '    {%- endif %}\n'
    '{%- endif %}\n'
    '{{- "Today Date: " + date_string + "\\n\\n" }}')


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def submitted_body(**kwargs: object) -> dict:
    payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    with patch.object(api_module.request, "urlopen", return_value=_FakeResponse(payload)) as urlopen:
        api_module.api_call(
            api_base="http://127.0.0.1:8000/v1", served_model="model",
            messages=[{"role": "user", "content": "test"}],
            decoding={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 8},
            seed=1, thinking=False, **kwargs)
    return json.loads(urlopen.call_args.args[0].data.decode("utf-8"))


def snapshot(directory: Path, template: str) -> Path:
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template}), encoding="utf-8")
    return directory


class ApiCallTemplateKwargsTest(unittest.TestCase):
    def test_registered_penalties_are_sent_so_generation_config_cannot_fill_them(self) -> None:
        payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        decoding = {"temperature": 0.0, "top_p": 1.0, "top_k": None, "max_new_tokens": 8,
                    "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
        with patch.object(api_module.request, "urlopen", return_value=_FakeResponse(payload)) as urlopen:
            api_module.api_call(api_base="http://127.0.0.1:8000/v1", served_model="model",
                                messages=[{"role": "user", "content": "test"}],
                                decoding=decoding, seed=1, thinking=False)
        body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(body["repetition_penalty"], 1.0)
        self.assertEqual(body["frequency_penalty"], 0.0)
        self.assertEqual(body["presence_penalty"], 0.0)
        self.assertNotIn("top_k", body)

    def test_without_kwargs_the_request_is_unchanged(self) -> None:
        self.assertEqual(submitted_body()["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(submitted_body(chat_template_kwargs=None)["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_registered_kwargs_are_merged(self) -> None:
        body = submitted_body(chat_template_kwargs={"date_string": "26 Jul 2024"})
        self.assertEqual(body["chat_template_kwargs"],
                         {"enable_thinking": False, "date_string": "26 Jul 2024"})

    def test_kwargs_cannot_override_thinking(self) -> None:
        with self.assertRaises(ValueError):
            submitted_body(chat_template_kwargs={"enable_thinking": True})


class ChatTemplateKwargsForTest(unittest.TestCase):
    def test_clock_reading_template_requires_a_pinned_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = snapshot(Path(directory), LLAMA_DATE_BLOCK)
            with self.assertRaises(SystemExit):
                api_module.chat_template_kwargs_for({"server_path": str(path)})
            self.assertEqual(
                api_module.chat_template_kwargs_for(
                    {"server_path": str(path),
                     "chat_template_kwargs": {"date_string": "26 Jul 2024"}}),
                {"date_string": "26 Jul 2024"})

    def test_template_without_clock_needs_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = snapshot(Path(directory), "{{ messages[0]['content'] }}")
            self.assertIsNone(api_module.chat_template_kwargs_for({"server_path": str(path)}))

    def test_absent_snapshot_is_not_read(self) -> None:
        self.assertIsNone(api_module.chat_template_kwargs_for({"server_path": "/nonexistent"}))
        self.assertIsNone(api_module.chat_template_kwargs_for({}))

    def test_thinking_is_not_a_template_kwarg(self) -> None:
        with self.assertRaises(SystemExit):
            api_module.chat_template_kwargs_for({"chat_template_kwargs": {"enable_thinking": False}})


if __name__ == "__main__":
    unittest.main()
