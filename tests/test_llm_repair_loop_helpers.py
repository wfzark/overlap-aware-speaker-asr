from __future__ import annotations

import unittest

from src.llm_repair_loop import get_asr_output_text, get_risk_info


class LlmRepairLoopHelpersTest(unittest.TestCase):
    def test_get_asr_output_text_loads_existing_mixed_transcript(self) -> None:
        text = get_asr_output_text("NoOverlap", "mixed_whisper")
        self.assertTrue(text.strip())

    def test_get_asr_output_text_returns_empty_for_unknown_method(self) -> None:
        self.assertEqual(get_asr_output_text("NoOverlap", "unknown_method"), "")

    def test_get_risk_info_returns_string_for_gold_case(self) -> None:
        info = get_risk_info("NoOverlap")
        self.assertIsInstance(info, str)


if __name__ == "__main__":
    unittest.main()
