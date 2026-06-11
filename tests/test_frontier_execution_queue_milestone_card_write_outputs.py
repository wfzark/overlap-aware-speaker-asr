from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.frontier_execution_queue_milestone_card import (
    MILESTONE_COLUMNS,
    build_milestone_card_row,
    write_outputs,
)


class FrontierExecutionQueueMilestoneCardBuildRowTest(unittest.TestCase):
    def test_build_milestone_card_row_returns_empty_without_completion_summary(self) -> None:
        self.assertEqual(build_milestone_card_row({}, [{"frontier_name": "speaker_profile"}]), {})

    def test_build_milestone_card_row_uses_second_handoff_frontier(self) -> None:
        row = build_milestone_card_row(
            {"total_chain_count": "5"},
            [
                {"frontier_name": "meeteval_compatibility"},
                {"frontier_name": "speaker_profile"},
            ],
        )
        self.assertEqual(row["next_milestone"], "first_execution_queue_checkpoint_complete")
        self.assertEqual(row["remaining_frontier_count"], "4")
        self.assertIn("speaker_profile", row["unlocks"])


if __name__ == "__main__":
    unittest.main()
