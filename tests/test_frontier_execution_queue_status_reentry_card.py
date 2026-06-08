from __future__ import annotations

import unittest

from src.frontier_execution_queue_status_reentry_card import build_reentry_card_row


class FrontierExecutionQueueStatusReentryCardTest(unittest.TestCase):
    def test_build_reentry_card_row_links_preflight_to_status_rollup(self) -> None:
        row = build_reentry_card_row(
            [
                {
                    "current_first_frontier": "meeteval_compatibility",
                    "receipt_target": "results/figures/frontier_execution_queue_status.md",
                }
            ],
            {"combined_chain_status": "execution_chain_ready"},
        )

        self.assertEqual(row["current_first_frontier"], "meeteval_compatibility")
        self.assertEqual(row["status_rollup_target"], "results/figures/frontier_execution_queue_status.md")
        self.assertEqual(row["combined_chain_status"], "execution_chain_ready")
        self.assertIn("reopen", row["reentry_action"])

    def test_build_reentry_card_row_returns_empty_without_inputs(self) -> None:
        row = build_reentry_card_row([], {})

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
