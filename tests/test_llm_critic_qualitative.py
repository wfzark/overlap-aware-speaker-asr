from __future__ import annotations

import unittest

from src.llm_correct import (
    build_critic_note_lines,
    build_critic_rows,
)


class LlmCriticQualitativeTest(unittest.TestCase):
    def test_build_critic_rows_explain_risk_and_repair(self) -> None:
        rows = build_critic_rows(
            risk_rows=[
                {
                    "case_id": "HeavyOverlap",
                    "risk_level": "medium",
                    "risk_flags": "length_inflation_risk;method_disagreement_risk",
                    "recommended_action": "repair separated output with cleaned transcript",
                }
            ],
            profile_rows=[
                {
                    "case_id": "HeavyOverlap",
                    "best_profile_alignment": "swapped",
                    "profile_confidence_gap": 0.411129,
                    "hypothesis_source": "separated_whisper_cleaned",
                }
            ],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "HeavyOverlap")
        self.assertIn("length_inflation_risk", rows[0]["risk_explanation"])
        self.assertIn("cleaned transcript", rows[0]["candidate_repair"])
        self.assertIn("swapped", rows[0]["uncertainty_note"])
        self.assertEqual(rows[0]["label"], "qualitative/demo")

    def test_build_critic_note_lines_render_qualitative_warning(self) -> None:
        lines = build_critic_note_lines(
            [
                {
                    "case_id": "HeavyOverlap",
                    "label": "qualitative/demo",
                    "risk_explanation": "length_inflation_risk and method disagreement both suggest unstable separated output.",
                    "candidate_repair": "Try the cleaned transcript first before treating the separated output as final.",
                    "uncertainty_note": "Profile alignment still prefers swapped, so attribution remains uncertain.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# LLM Critic Qualitative Note", rendered)
        self.assertIn("qualitative", rendered)
        self.assertIn("HeavyOverlap", rendered)
        self.assertIn("cleaned transcript", rendered)
        self.assertIn("attribution remains uncertain", rendered)


if __name__ == "__main__":
    unittest.main()
