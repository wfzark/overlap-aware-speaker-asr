from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_handoff import build_handoff_rows


class FrontierExecutionReceiptQueueHandoffTest(unittest.TestCase):
    def test_build_handoff_rows_recommends_update_when_ready(self) -> None:
        rows = build_handoff_rows(
            {
                "meeteval_readiness_status": "receipt_ready_to_fill",
                "speaker_profile_readiness_status": "receipt_ready_to_fill",
                "external_staging_readiness_status": "receipt_ready_to_fill",
            }
        )

        self.assertEqual(rows[0]["frontier_name"], "meeteval_compatibility")
        self.assertIn("Update execution_status", rows[0]["recommended_action"])


if __name__ == "__main__":
    unittest.main()
