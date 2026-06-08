from __future__ import annotations

import unittest

from src.frontier_operator_next_action_handoff_packet import build_handoff_packet_rows


class FrontierOperatorNextActionHandoffPacketTest(unittest.TestCase):
    def test_build_handoff_packet_rows_include_runbook_and_operator(self) -> None:
        rows = build_handoff_packet_rows()
        paths = [row["artifact_path"] for row in rows]

        self.assertIn("results/figures/frontier_operator_next_action_operator_brief.md", paths)
        self.assertIn("results/figures/frontier_operator_next_action_runbook_card.md", paths)
        self.assertIn("results/figures/frontier_operator_next_action_frontier_bridge_checklist.md", paths)


if __name__ == "__main__":
    unittest.main()
