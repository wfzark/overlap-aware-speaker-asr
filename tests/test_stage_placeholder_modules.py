from __future__ import annotations

import importlib
import unittest


PLACEHOLDER_MODULES = [
    "src.evaluate_summary",
    "src.evaluate_speaker",
    "src.evaluate_terms",
    "src.transcribe_funasr",
    "src.rag_retrieve",
]


class StagePlaceholderModulesTest(unittest.TestCase):
    def test_placeholder_modules_are_importable_with_docstrings(self) -> None:
        for module_name in PLACEHOLDER_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module.__doc__)
                self.assertIn("placeholder", module.__doc__.lower())


if __name__ == "__main__":
    unittest.main()
