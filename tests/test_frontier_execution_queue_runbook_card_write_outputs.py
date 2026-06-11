from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.frontier_execution_queue_runbook_card import (
    RUNBOOK_COLUMNS,
    build_runbook_card_row,
    write_outputs,
)


class FrontierExecutionQueueRunbookCardWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_csv_json_and_markdown(self) -> None:
        row = build_runbook_card_row(
            {
                "operator_frontier": "speaker_profile",
                "operator_action": "Complete scaffold verification.",
                "operator_evidence": "results/figures/frontier_execution_queue_handoff.md",
                "operator_urgency": "queue_status=queue_in_progress",
            },
            [
                {
                    "frontier_name": "speaker_profile",
                    "expected_outputs": "results/tables/speaker_profile_embedding_trial_execution_receipt.json",
                }
            ],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.frontier_execution_queue_runbook_card.PROJECT_ROOT", root):
                csv_path, json_path, md_path = write_outputs(row)

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, RUNBOOK_COLUMNS)
                self.assertEqual(list(reader)[0]["recommended_frontier"], "speaker_profile")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["recommended_frontier"], "speaker_profile")
            self.assertIn("Frontier Execution Queue Runbook Card", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
