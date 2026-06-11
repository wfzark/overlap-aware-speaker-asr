from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.meeteval_tokenization_gain_frontier_fill_runbook_card import (
    RUNBOOK_COLUMNS,
    build_runbook_card_row,
    write_outputs,
)


class MeetEvalTokenizationGainFrontierFillRunbookCardWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_runbook_card_artifacts(self) -> None:
        row = build_runbook_card_row(
            {
                "handoff_status": "tokenization_gain_frontier_fill_handoff_ready",
                "queue_status": "queue_complete",
                "adapted_and_aligned_count": "3",
                "case_count": "5",
                "handoff_goal": "Fill MeetEval receipt with real cpWER evidence.",
                "handoff_note": "Handoff verified.",
            },
            {
                "recommended_frontier": "meeteval_compatibility",
                "recommended_action": "Run character-spaced cpWER and update receipt.",
                "required_evidence": "results/tables/meeteval_cpwer_execution_receipt.json",
                "completion_signal": "receipt_filled",
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch(
                "src.meeteval_tokenization_gain_frontier_fill_runbook_card.PROJECT_ROOT",
                root,
            ):
                csv_path, json_path, md_path = write_outputs(row)

            for path in (csv_path, json_path, md_path):
                self.assertTrue(path.exists())
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, RUNBOOK_COLUMNS)
                self.assertEqual(
                    list(reader)[0]["runbook_status"],
                    "tokenization_gain_frontier_fill_runbook_ready",
                )
            self.assertIn("Runbook Card", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
