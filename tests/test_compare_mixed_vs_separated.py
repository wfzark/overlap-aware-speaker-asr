from __future__ import annotations

import unittest

from src.compare_mixed_vs_separated import preview


class CompareMixedVsSeparatedPreviewTest(unittest.TestCase):
    def test_preview_collapses_whitespace(self) -> None:
        self.assertEqual(preview("hello\n\nworld"), "hello world")


if __name__ == "__main__":
    unittest.main()
