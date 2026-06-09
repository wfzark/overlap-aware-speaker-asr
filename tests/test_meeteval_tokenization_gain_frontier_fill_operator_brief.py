from __future__ import annotations

import unittest

from src.meeteval_tokenization_gain_frontier_fill_operator_brief import build_operator_brief_row


class MeetEvalTokenizationGainFrontierFillOperatorBriefTest(unittest.TestCase):
    def test_build_operator_brief_row_gives_plain_language_receipt_action(self) -> None:
        row = build_operator_brief_row(
            {
                "recommended_frontier": "meeteval_compatibility",
                "receipt_target": "results/tables/meeteval_cpwer_execution_receipt.json",
                "bridge_note": "No full MeetEval benchmark completion is claimed by this bridge alone.",
            }
        )

        self.assertEqual(row["operator_frontier"], "meeteval_compatibility")
        self.assertEqual(row["operator_receipt"], "results/tables/meeteval_cpwer_execution_receipt.json")
        self.assertIn("execution_receipt_bridge_checklist", row["operator_evidence"])
        self.assertIn("character-spaced cpWER", row["operator_action"])
        self.assertIn("is claimed", row["operator_note"])

    def test_build_operator_brief_row_empty_when_checklist_missing(self) -> None:
        self.assertEqual(build_operator_brief_row({}), {})


if __name__ == "__main__":
    unittest.main()
