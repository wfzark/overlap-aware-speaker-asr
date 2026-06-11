from __future__ import annotations

import unittest

from src.adaptive_router_v2 import build_synthetic_decisions, read_csv_rows
from src.config import PROJECT_ROOT


class AdaptiveRouterV2BuildSyntheticDecisionsTest(unittest.TestCase):
    def test_build_synthetic_decisions_returns_row_for_manifest_sample(self) -> None:
        manifest_rows = read_csv_rows(PROJECT_ROOT / "results" / "tables" / "synthetic_manifest.csv")
        sample_row = next(
            row for row in manifest_rows if str(row.get("sample_id", "")).strip() == "SyntheticNoOverlap_01"
        )
        decisions = build_synthetic_decisions([sample_row], cleaned_rows={})
        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertEqual(decision["sample_id"], "SyntheticNoOverlap_01")
        self.assertEqual(decision["tier"], "SyntheticNoOverlap")
        self.assertIn(decision["selected_method"], {"mixed_whisper", "separated_whisper", "separated_whisper_cleaned"})
        self.assertTrue(str(decision["decision_rule"]).strip())


if __name__ == "__main__":
    unittest.main()
