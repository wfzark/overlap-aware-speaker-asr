from __future__ import annotations

import json
import unittest

from src.export_meeteval_compatibility import (
    build_meeteval_compatibility_lines,
    build_meeteval_compatibility_rows,
    build_meeteval_segment_lines,
)


class MeetEvalCompatibilityTest(unittest.TestCase):
    def test_build_meeteval_compatibility_rows_summarize_reference_and_hypothesis(self) -> None:
        rows = build_meeteval_compatibility_rows(
            case_ids=["NoOverlap"],
            reference_payloads={
                "NoOverlap": {
                    "segments": [
                        {"speaker": "SPEAKER_1", "start": 0.0, "end": 1.0, "text": "alpha"},
                        {"speaker": "SPEAKER_2", "start": 1.0, "end": 2.0, "text": "beta"},
                    ]
                }
            },
            hypothesis_payloads={
                "NoOverlap": {
                    "segments": [
                        {"speaker": "SPEAKER_1", "start": 0.0, "end": 1.0, "text": "alpha"},
                        {"speaker": "SPEAKER_2", "start": 1.0, "end": 2.0, "text": "beta"},
                    ]
                }
            },
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "NoOverlap")
        self.assertEqual(rows[0]["reference_segment_count"], 2)
        self.assertEqual(rows[0]["hypothesis_segment_count"], 2)
        self.assertEqual(rows[0]["speaker_count"], 2)
        self.assertIn("compatibility bridge", rows[0]["observation"])

    def test_build_meeteval_segment_lines_render_jsonl_export(self) -> None:
        lines = build_meeteval_segment_lines(
            case_id="NoOverlap",
            source="reference",
            segments=[
                {"speaker": "SPEAKER_1", "start": 0.0, "end": 1.0, "text": "alpha"},
                {"speaker": "SPEAKER_2", "start": 1.0, "end": 2.0, "text": "beta"},
            ],
        )

        payload = [json.loads(line) for line in lines]

        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["session_id"], "NoOverlap")
        self.assertEqual(payload[0]["speaker"], "SPEAKER_1")
        self.assertEqual(payload[0]["source"], "reference")
        self.assertEqual(payload[1]["text"], "beta")

    def test_build_meeteval_compatibility_lines_render_note(self) -> None:
        lines = build_meeteval_compatibility_lines(
            [
                {
                    "case_id": "NoOverlap",
                    "reference_segment_count": 2,
                    "hypothesis_segment_count": 2,
                    "speaker_count": 2,
                    "reference_export": "results/tables/meeteval_reference_segments.jsonl",
                    "hypothesis_export": "results/tables/meeteval_hypothesis_segments.jsonl",
                    "observation": "Compatibility bridge only; no cpWER claim yet.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# MeetEval Compatibility Note", rendered)
        self.assertIn("NoOverlap", rendered)
        self.assertIn("meeteval_reference_segments.jsonl", rendered)
        self.assertIn("no cpWER claim yet", rendered)


if __name__ == "__main__":
    unittest.main()
