import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from configs.types import SlimExperimentConfig
from modal_train import build_train_tokens, preview_slime_cmd_for_config


class CommandAssemblyTests(unittest.TestCase):
    def test_build_train_tokens_for_thin_config_order(self):
        with tempfile.TemporaryDirectory() as td:
            args_file = Path(td) / "extra.args"
            args_file.write_text("--from-file 3\n", encoding="utf-8")

            cfg = SlimExperimentConfig(
                name="x",
                source_path=Path(td) / "x.toml",
                model_id="org/model",
                model_args_script="model.sh",
                args_files=(args_file,),
                args=("--inline", "4"),
                train_script="slime/train_async.py",
            )

            with patch("modal_train.source_model_args_tokens", return_value=["--model", "1"]):
                tokens, script = build_train_tokens(
                    cfg,
                    model_path="/m",
                    slime_args="--cli 5",
                    slime_args_file=str(args_file),
                )

        self.assertEqual(script, "slime/train_async.py")
        self.assertEqual(
            tokens,
            [
                "--model", "1",
                "--hf-checkpoint", "/m", "--ref-load", "/m",
                "--from-file", "3",
                "--inline", "4",
                "--from-file", "3",
                "--cli", "5",
            ],
        )

    def test_build_train_tokens_with_inline_only(self):
        cfg = SlimExperimentConfig(
            name="x",
            source_path=Path("/tmp/x.toml"),
            model_id="org/model",
            model_args_script="model.sh",
            args=("--legacy", "true"),
            train_script="slime/train.py",
        )
        with patch("modal_train.source_model_args_tokens", return_value=[]):
            tokens, script = build_train_tokens(
                cfg,
                model_path="/m",
                slime_args="--x 1",
            )
        self.assertEqual(script, "slime/train.py")
        self.assertEqual(tokens, ["--hf-checkpoint", "/m", "--ref-load", "/m", "--legacy", "true", "--x", "1"])

    def test_preview_command_uses_model_args(self):
        root = Path(__file__).resolve().parents[2] / "dev" / "slime" / "scripts" / "models"
        with patch.dict(os.environ, {"SLIME_MODEL_ARGS_ROOT": str(root)}):
            cmd = preview_slime_cmd_for_config("qwen-4b", model_path="/tmp/model")
        self.assertIn("python3", cmd)
        self.assertIn("--num-layers", cmd)
        self.assertIn("--hf-checkpoint /tmp/model", cmd)


if __name__ == "__main__":
    unittest.main()
