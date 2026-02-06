import tempfile
from pathlib import Path
import unittest

from configs.toml_loader import discover_toml_configs, resolve_toml_config


class TomlLoaderTests(unittest.TestCase):
    def test_discover_marks_hidden(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()

            (cfg_dir / "_base.toml").write_text("model_id='a/b'\nmodel_args_script='x.sh'\n", encoding="utf-8")
            (cfg_dir / "exp.toml").write_text("model_id='a/b'\nmodel_args_script='x.sh'\n", encoding="utf-8")

            raw = discover_toml_configs(cfg_dir)
            self.assertIn("base", raw)
            self.assertTrue(raw["base"].is_hidden)
            self.assertIn("exp", raw)
            self.assertFalse(raw["exp"].is_hidden)

    def test_resolve_extends_merges_args_and_resolves_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()

            parent = cfg_dir / "_base.toml"
            child = cfg_dir / "child.toml"
            (cfg_dir / "base.args").write_text("--foo 1\n", encoding="utf-8")
            (cfg_dir / "child.args").write_text("--bar 2\n", encoding="utf-8")

            parent.write_text(
                "\n".join(
                    [
                        "model_id = 'org/model'",
                        "model_args_script = 'model.sh'",
                        "args_files = ['base.args']",
                        "args = ['--a', '1']",
                        "model_args_env = { X = '1' }",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            child.write_text(
                "\n".join(
                    [
                        "extends = 'base'",
                        "app_name = 'child-app'",
                        "args_files = ['child.args']",
                        "args = ['--b', '2']",
                        "model_args_env = { Y = '2' }",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            raw = discover_toml_configs(cfg_dir)
            resolved = resolve_toml_config("child", raw)

            self.assertEqual(resolved.model_id, "org/model")
            self.assertEqual(resolved.app_name, "child-app")
            self.assertEqual(resolved.args, ("--a", "1", "--b", "2"))
            self.assertEqual(resolved.model_args_env, {"X": "1", "Y": "2"})
            self.assertEqual(resolved.args_files[0], (cfg_dir / "base.args").resolve())
            self.assertEqual(resolved.args_files[1], (cfg_dir / "child.args").resolve())

    def test_cycle_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()

            (cfg_dir / "a.toml").write_text("extends='b'\nmodel_id='x/y'\nmodel_args_script='m.sh'\n", encoding="utf-8")
            (cfg_dir / "b.toml").write_text("extends='a'\nmodel_id='x/y'\nmodel_args_script='m.sh'\n", encoding="utf-8")

            raw = discover_toml_configs(cfg_dir)
            with self.assertRaises(ValueError):
                resolve_toml_config("a", raw)


if __name__ == "__main__":
    unittest.main()
