from __future__ import annotations

import csv
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.evaluate_error_types import write_outputs


def _error_row(case_id: str, method: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "method": method,
        "reference_length": 20,
        "hypothesis_length": 18,
        "length_ratio": 0.9,
        "substitution_count": 1,
        "deletion_count": 2,
        "insertion_count": 3,
        "edit_distance": 6,
        "cer": 0.3,
        "repetition_count": 1,
        "removed_count_if_cleaned": 0,
        "dominant_error_type": "insertion",
        "observation": f"{case_id} error-type observation",
    }


class EvaluateErrorTypesWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_writes_csv_json_markdown_and_figure_paths(self) -> None:
        rows = [_error_row("NoOverlap", "mixed_whisper")]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with unittest.mock.patch("src.evaluate_error_types.PROJECT_ROOT", root):
                with unittest.mock.patch("src.evaluate_error_types.render_figure") as render_figure:
                    with unittest.mock.patch("src.evaluate_error_types.update_current_summary"):
                        csv_path, json_path, fig_path, md_path = write_outputs(rows)
                        render_figure.assert_called_once()
                        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                            loaded_csv = list(csv.DictReader(handle))
                        loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
                        markdown = md_path.read_text(encoding="utf-8")
        self.assertEqual(loaded_csv[0]["case_id"], "NoOverlap")
        self.assertEqual(loaded_json, rows)
        self.assertIn("# Error Type Summary", markdown)
        self.assertEqual(fig_path.name, "error_type_by_case.png")


if __name__ == "__main__":
    unittest.main()
