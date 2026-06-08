from __future__ import annotations

import unittest

from src.frontier_execution_queue_phase_checkpoint_card import build_phase_checkpoint_row


class FrontierExecutionQueuePhaseCheckpointCardTest(unittest.TestCase):
    def test_build_phase_checkpoint_row_uses_runbook_completion_signal(self) -> None:
        row = build_phase_checkpoint_row(
            {
                "recommended_frontier": "meeteval_compatibility",
                "recommended_action": "Fill the execution receipt at results/tables/meeteval_cpwer_execution_receipt.json after final bridge verification.",
                "completion_signal": (
                    "execution queue verification is complete and the target artifact "
                    "results/tables/meeteval_cpwer_execution_receipt.json is ready to open"
                ),
            }
        )

        self.assertEqual(row["checkpoint_frontier"], "meeteval_compatibility")
        self.assertIn("Fill the execution receipt", row["checkpoint_action"])
        self.assertIn("meeteval_cpwer_execution_receipt.json", row["completion_signal"])

    def test_build_phase_checkpoint_row_returns_empty_without_runbook(self) -> None:
        row = build_phase_checkpoint_row({})

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
