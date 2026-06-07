from __future__ import annotations

import unittest

from src.frontier_execution_receipt_fill_execution_handoff import build_handoff_rows


class FrontierExecutionReceiptFillExecutionHandoffTest(unittest.TestCase):
    def test_build_handoff_rows_recommends_execute_when_awaiting(self) -> None:
        rows = build_handoff_rows(
            {
                "meeteval_fill_execution_status": "awaiting_fill",
                "speaker_profile_fill_execution_status": "awaiting_fill",
                "external_staging_fill_execution_status": "awaiting_fill",
            }
        )

        self.assertEqual(rows[0]["frontier_name"], "meeteval_compatibility")
        self.assertIn("Execute the real frontier run", rows[0]["recommended_action"])


if __name__ == "__main__":
    unittest.main()
