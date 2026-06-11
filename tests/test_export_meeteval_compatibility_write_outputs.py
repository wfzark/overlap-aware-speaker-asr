from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.export_meeteval_compatibility import (
    build_meeteval_compatibility_rows,
    build_meeteval_segment_lines,
    write_outputs,
)


class ExportMeetEvalCompatibilityWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_writes_summary_jsonl_and_dry_run_artifacts(self) -> None:
        segments = [
            {"speaker": "SPEAKER_1", "start": 0.0, "end": 1.0, "text": "alpha"},
            {"speaker": "SPEAKER_2", "start": 1.0, "end": 2.0, "text": "beta"},
        ]
        rows = build_meeteval_compatibility_rows(
            case_ids=["NoOverlap"],
            reference_payloads={"NoOverlap": {"segments": segments}},
            hypothesis_payloads={
                "NoOverlap": {
                    "segments": segments,
                    "hypothesis_source": "separated_whisper",
                }
            },
        )
        reference_lines = build_meeteval_segment_lines("NoOverlap", "reference", segments)
        hypothesis_lines = build_meeteval_segment_lines("NoOverlap", "separated_whisper", segments)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with unittest.mock.patch("src.export_meeteval_compatibility.PROJECT_ROOT", root):
                paths = write_outputs(rows, reference_lines, hypothesis_lines)
                summary_json = paths[1]
                reference_jsonl = paths[2]
                note_md = paths[4]
                checklist_json = paths[17]
                loaded_summary = json.loads(summary_json.read_text(encoding="utf-8"))
                reference_text = reference_jsonl.read_text(encoding="utf-8")
                note_text = note_md.read_text(encoding="utf-8")
                checklist_payload = json.loads(checklist_json.read_text(encoding="utf-8"))
                self.assertEqual(loaded_summary[0]["case_id"], "NoOverlap")
                self.assertIn("NoOverlap", reference_text)
                self.assertIn("compatibility bridge", note_text)
                self.assertGreaterEqual(len(checklist_payload), 1)


if __name__ == "__main__":
    unittest.main()
