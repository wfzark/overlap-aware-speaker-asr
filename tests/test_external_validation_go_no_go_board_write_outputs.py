from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.external_validation_go_no_go_board import (
    BOARD_COLUMNS,
    SUMMARY_COLUMNS,
    write_outputs,
)


def _sample_board_row() -> dict[str, str]:
    return {
        "checkpoint_name": "license_gate",
        "dataset_name": "AISHELL-4",
        "current_status": "pending_confirmation",
        "blocker": "license_confirmation_pending",
        "go_no_go_state": "no_go",
        "next_action": "Confirm license terms.",
        "evidence_artifact": "results/figures/external_validation_license_gate.md",
    }


def _sample_summary_row() -> dict[str, str]:
    return {
        "scope": "external/sanity-check",
        "dataset_name": "AISHELL-4",
        "checkpoint_count": "1",
        "go_count": "0",
        "no_go_count": "1",
        "overall_state": "no_go",
        "primary_blocker": "license_confirmation_pending",
        "recommended_next_action": "Confirm license terms.",
        "observation": "fixture",
    }


class ExternalValidationGoNoGoBoardWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_board_and_summary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.external_validation_go_no_go_board.PROJECT_ROOT", root):
                outputs = write_outputs([_sample_board_row()], _sample_summary_row())

            for path in outputs:
                self.assertTrue(path.exists())

            board_csv, board_json, summary_csv, summary_json, board_md, summary_md = outputs
            with board_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, BOARD_COLUMNS)
                self.assertEqual(list(reader)[0]["checkpoint_name"], "license_gate")

            with summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SUMMARY_COLUMNS)
                self.assertEqual(list(reader)[0]["overall_state"], "no_go")

            board_payload = json.loads(board_json.read_text(encoding="utf-8"))
            self.assertEqual(board_payload[0]["dataset_name"], "AISHELL-4")
            summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["no_go_count"], "1")

            self.assertIn("External Validation Go-No-Go Board", board_md.read_text(encoding="utf-8"))
            self.assertIn("External Validation Go-No-Go Summary", summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
