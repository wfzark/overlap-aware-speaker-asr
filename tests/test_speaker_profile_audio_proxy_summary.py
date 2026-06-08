from __future__ import annotations

import unittest

from src.speaker_profile_audio_proxy_summary import (
    build_audio_proxy_summary_lines,
    build_audio_proxy_summary_row,
)


class SpeakerProfileAudioProxySummaryTest(unittest.TestCase):
    def test_build_audio_proxy_summary_row_marks_weak_signal_when_gap_is_tiny(self) -> None:
        row = build_audio_proxy_summary_row(
            [
                {
                    "case_id": "NoOverlap",
                    "best_audio_alignment": "swapped",
                    "audio_confidence_gap": "0.000012",
                },
                {
                    "case_id": "HeavyOverlap",
                    "best_audio_alignment": "swapped",
                    "audio_confidence_gap": "0.000014",
                },
            ]
        )

        self.assertEqual(row["dominant_alignment"], "swapped_bias")
        self.assertEqual(row["signal_strength"], "weak_near_tie")
        self.assertIn("not yet justify", row["next_action"])

    def test_build_audio_proxy_summary_lines_render_table(self) -> None:
        lines = build_audio_proxy_summary_lines(
            {
                "case_count": "5",
                "dominant_alignment": "swapped_bias",
                "average_confidence_gap": "0.000013",
                "signal_strength": "weak_near_tie",
                "next_action": "Current lightweight audio proxy does not yet justify attribution claims.",
            }
        )

        rendered = "\n".join(lines)
        self.assertIn("# Speaker Profile Audio Proxy Summary", rendered)
        self.assertIn("weak_near_tie", rendered)
        self.assertIn("0.000013", rendered)


if __name__ == "__main__":
    unittest.main()
