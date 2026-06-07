from __future__ import annotations

import unittest
from unittest.mock import patch

from src.meeteval_cpwer_official_execution import (
    build_execution_row,
    build_speaker_text_lists,
    merge_receipt_entry,
    resolve_case_ids,
)
from src.meeteval_cpwer_official_execution_alignment_audit import (
    classify_alignment,
    compute_alignment_delta,
)
from src.meeteval_cpwer_official_execution_completion_summary import (
    build_completion_summary_row,
)


class MeetEvalCpwerOfficialExecutionTest(unittest.TestCase):
    def test_build_execution_row_when_tool_unavailable(self) -> None:
        row = build_execution_row("NoOverlap", "separated_whisper", None, 2, tool_available=False)

        self.assertEqual(row["execution_status"], "official_cpwer_tool_unavailable")
        self.assertEqual(row["cpwer_tool"], "unavailable")

    def test_build_execution_row_when_complete(self) -> None:
        row = build_execution_row("NoOverlap", "separated_whisper", 0.12, 2, tool_available=True)

        self.assertEqual(row["execution_status"], "official_cpwer_narrow_dry_run_complete")
        self.assertEqual(row["official_cpwer"], "0.12")

    def test_build_speaker_text_lists_aligns_by_speaker(self) -> None:
        reference = [
            {"speaker": "SPEAKER_1", "text": "你好"},
            {"speaker": "SPEAKER_2", "text": "世界"},
        ]
        hypothesis = [
            {"speaker": "SPEAKER_1", "text": "你好啊"},
            {"speaker": "SPEAKER_2", "text": "世界"},
        ]
        ref_texts, hyp_texts = build_speaker_text_lists(reference, hypothesis, ["SPEAKER_1", "SPEAKER_2"])

        self.assertEqual(ref_texts, ["你好", "世界"])
        self.assertEqual(hyp_texts, ["你好啊", "世界"])

    def test_merge_receipt_entry_updates_official_fields(self) -> None:
        with patch(
            "src.meeteval_cpwer_official_execution.load_bridge_lite_score",
            return_value="0.15",
        ):
            merged = merge_receipt_entry(
                {"case_id": "NoOverlap", "execution_status": "template_only"},
                {
                    "case_id": "NoOverlap",
                    "execution_status": "official_cpwer_narrow_dry_run_complete",
                    "official_cpwer": "0.12",
                    "cpwer_tool": "meeteval",
                    "result_label": "experimental/frontier",
                    "execution_note": "done",
                },
            )

        self.assertEqual(merged["official_cpwer"], "0.12")
        self.assertEqual(merged["cpwer_bridge_lite"], "0.15")

    def test_resolve_case_ids_all(self) -> None:
        case_ids = resolve_case_ids("all", run_all=False)
        self.assertEqual(len(case_ids), 5)
        self.assertIn("NoOverlap", case_ids)

    def test_resolve_case_ids_run_all_flag(self) -> None:
        case_ids = resolve_case_ids("preferred", run_all=True)
        self.assertEqual(len(case_ids), 5)


class MeetEvalCpwerOfficialExecutionCompletionSummaryTest(unittest.TestCase):
    def test_build_completion_summary_queue_complete(self) -> None:
        rows = [
            {"execution_status": "official_cpwer_narrow_dry_run_complete"},
            {"execution_status": "official_cpwer_narrow_dry_run_complete"},
        ]
        summary = build_completion_summary_row(rows)
        self.assertEqual(summary["queue_status"], "queue_complete")
        self.assertEqual(summary["complete_count"], "2")

    def test_build_completion_summary_tool_blocked(self) -> None:
        rows = [{"execution_status": "official_cpwer_tool_unavailable"}]
        summary = build_completion_summary_row(rows)
        self.assertEqual(summary["queue_status"], "queue_blocked_by_tool")


class MeetEvalCpwerOfficialExecutionAlignmentAuditTest(unittest.TestCase):
    def test_compute_alignment_delta(self) -> None:
        self.assertEqual(compute_alignment_delta("0.12", "0.10"), "0.02")

    def test_classify_alignment_minor_drift(self) -> None:
        self.assertEqual(classify_alignment("0.02"), "minor_drift")

    def test_classify_alignment_aligned(self) -> None:
        self.assertEqual(classify_alignment("0.005"), "aligned")


if __name__ == "__main__":
    unittest.main()
