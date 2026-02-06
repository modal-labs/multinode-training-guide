import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from configs.arg_tokens import parse_args_file, source_model_args_tokens


class ArgTokenTests(unittest.TestCase):
    def test_parse_args_file_supports_comments(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.args"
            path.write_text("--foo 1\n# comment\n--bar 'two words'\n", encoding="utf-8")
            self.assertEqual(parse_args_file(path), ["--foo", "1", "--bar", "two words"])

    def test_source_model_args_tokens_from_override_root(self):
        root = Path(__file__).resolve().parents[2] / "dev" / "slime" / "scripts" / "models"
        self.assertTrue(root.exists(), f"missing expected model scripts root: {root}")

        with patch.dict(os.environ, {"SLIME_MODEL_ARGS_ROOT": str(root)}):
            tokens = source_model_args_tokens("qwen3-4B-Instruct-2507.sh")

        self.assertIn("--num-layers", tokens)
        self.assertIn("36", tokens)
        self.assertIn("--hidden-size", tokens)

    def test_source_model_args_tokens_missing_script(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"SLIME_MODEL_ARGS_ROOT": td}):
                with self.assertRaises(FileNotFoundError):
                    source_model_args_tokens("missing.sh")


if __name__ == "__main__":
    unittest.main()
