import unittest

from configs import get_config, list_configs
from configs.types import SlimExperimentConfig


class RegistryPrecedenceTests(unittest.TestCase):
    def test_get_config_returns_thin_wrapper(self):
        cfg = get_config("qwen-4b")
        self.assertIsInstance(cfg, SlimExperimentConfig)

    def test_lists_only_canonical_configs(self):
        names = list_configs()
        self.assertEqual(names, ["glm-4-7", "glm-4-7-flash", "qwen-4b"])


if __name__ == "__main__":
    unittest.main()
