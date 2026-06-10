from __future__ import annotations

import unittest

from src.audit_synthetic_benchmark import get_hypothesis_text, get_reference_text, safe_preview


class AuditSyntheticBenchmarkPreviewTest(unittest.TestCase):
    def test_safe_preview_collapses_whitespace(self) -> None:
        self.assertEqual(safe_preview("line one\n\nline two"), "line one line two")


if __name__ == "__main__":
    unittest.main()
