from __future__ import annotations

import unittest

from src.demo_storyboard import (
    build_demo_storyboard_cards,
    build_demo_storyboard_lines,
)


class DemoStoryboardTest(unittest.TestCase):
    def test_build_demo_storyboard_cards_cover_story_sections(self) -> None:
        cards = build_demo_storyboard_cards(
            {
                "baseline": "Selective separation beats blind separation.",
                "cascade": "router_v2 is the balanced default.",
                "frontier": "Breadth-first artifacts now exist across multiple frontiers.",
            }
        )

        titles = [card["title"] for card in cards]
        self.assertIn("Problem", titles)
        self.assertIn("Pipeline", titles)
        self.assertIn("Findings", titles)
        self.assertIn("Frontier", titles)
        findings = next(card for card in cards if card["title"] == "Findings")
        self.assertIn("router_v2", findings["body"])

    def test_build_demo_storyboard_lines_render_mermaid_and_cards(self) -> None:
        lines = build_demo_storyboard_lines(
            [
                {"title": "Problem", "body": "Overlap-aware ASR should separate selectively."},
                {"title": "Pipeline", "body": "Mixed ASR, separated ASR, routing, and evaluation compose the main flow."},
                {"title": "Findings", "body": "router_v2 is the balanced default and cleaned separation is the robust fallback."},
                {"title": "Frontier", "body": "Compute-aware, MeetEval, speaker profile, and critic bridges now exist."},
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# Demo Storyboard", rendered)
        self.assertIn("```mermaid", rendered)
        self.assertIn("Problem", rendered)
        self.assertIn("router_v2", rendered)
        self.assertIn("critic bridges now exist", rendered)


if __name__ == "__main__":
    unittest.main()
