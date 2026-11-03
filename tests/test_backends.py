"""Offline checks for provider routing and credential isolation."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import LLM_CONFIG
from modules.models import get_backend_config, query_model
from modules.prompt_utils import build_prompt, extract_result


class BackendTests(unittest.TestCase):
    def test_default_and_chinese_prompt(self):
        with patch.dict(LLM_CONFIG, {"provider": "qwen"}):
            self.assertEqual(get_backend_config()[1]["model"], "qwen3-max")
        prompt = build_prompt(5, "DATA_MARKER", "SAMPLE_MARKER")
        self.assertIn("DATA_MARKER", prompt)
        self.assertIn("SAMPLE_MARKER", prompt)
        self.assertNotIn("{x}", prompt)
        expressions, memory = extract_result("f(x0) = C*x0\n经验总结：线性关系")
        self.assertEqual(expressions, ["f(x0) = C*x0"])
        self.assertEqual(memory, "线性关系")

    def test_missing_key_and_unknown_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            for provider, key in [("qwen", "DASHSCOPE_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")]:
                with self.subTest(provider=provider):
                    with self.assertRaisesRegex(RuntimeError, key):
                        query_model("prompt", debug=False, provider=provider)
        with self.assertRaisesRegex(ValueError, "Unknown LLM provider"):
            get_backend_config("invalid")

    def test_only_selected_backend_requires_credentials(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])
        for provider, key, model in [
            ("qwen", "DASHSCOPE_API_KEY", "qwen3-max"),
            ("deepseek", "DEEPSEEK_API_KEY", "deepseek-chat"),
            ("openai", "OPENAI_API_KEY", "o4-mini"),
        ]:
            with self.subTest(provider=provider), patch.dict(os.environ, {key: "test-only"}, clear=True):
                with patch("openai.OpenAI") as client:
                    client.return_value.chat.completions.create.return_value = response
                    self.assertEqual(query_model("prompt", debug=False, provider=provider), "answer")
                    self.assertEqual(client.call_args.kwargs["api_key"], "test-only")
                    self.assertEqual(client.return_value.chat.completions.create.call_args.kwargs["model"], model)


if __name__ == "__main__":
    unittest.main()
