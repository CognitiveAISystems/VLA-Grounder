from __future__ import annotations

import unittest
from pathlib import Path

from vla_grounder.config import load_config


class BaselineConfigTest(unittest.TestCase):
    def test_baselines_do_not_require_grounder_model_fields(self):
        repository = Path(__file__).resolve().parents[1]
        paths = sorted((repository / "configs" / "eval").glob("*_baseline.yaml"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertFalse(config.grounder.enabled)
                self.assertEqual(set(config.grounder), {"enabled"})


if __name__ == "__main__":
    unittest.main()
