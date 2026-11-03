"""Exercise real preprocessing, fitting, state and evaluation with a fake LLM."""

import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main_loop
from config import PROJECT_ROOT, RUN_CONFIG, LLM_CONFIG


class PipelineTests(unittest.TestCase):
    def test_chinese_backends_fit_and_save_separate_runs(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "Data" / "srsd-feynman_easy" / "feynman-i.12.1"
            dataset.mkdir(parents=True)
            for split in ["train", "test"]:
                X = rng.uniform(0.5, 2, size=(40, 2))
                y = 2 * X[:, 0] + 3 * X[:, 1]
                np.save(dataset / f"{split}_points.npy", np.column_stack([X, y]))
            shutil.copy(PROJECT_ROOT / "easy-info.csv", root / "easy-info.csv")
            response = "f(x0, x1) = C*x0 + C*x1\n经验总结：线性组合"
            config = {"group": "Data/srsd-feynman_easy", "dataset": "feynman-i.12.1"}
            for provider in ["qwen", "deepseek"]:
                with self.subTest(provider=provider), patch.object(main_loop, "PROJECT_ROOT", root):
                    with patch.dict(LLM_CONFIG, {"provider": provider}), patch.dict(RUN_CONFIG, {"max_iterations": 0}):
                        with patch.object(main_loop, "query_model", return_value=response) as query:
                            with contextlib.redirect_stdout(io.StringIO()):
                                main_loop.main(config)
                                main_loop.main(config)
                            self.assertEqual(query.call_args.kwargs["provider"], provider)
                            self.assertIn("数据信息", query.call_args.args[0])
                            self.assertNotIn("未在 Excel 中找到", query.call_args.args[0])
                results = list((root / "outputs" / provider).rglob("results_*.txt"))
                self.assertEqual(len(results), 2)
                for result in results:
                    self.assertIn("Test  R² = 1.000000", result.read_text(encoding="utf-8"))
                    self.assertTrue((result.parent / "state.json").exists())
            self.assertFalse((root / "state.json").exists())


if __name__ == "__main__":
    unittest.main()
