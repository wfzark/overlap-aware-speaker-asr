from __future__ import annotations

import unittest

from src.adaptive_router_v2 import load_gold_inputs


class AdaptiveRouterV2LoadInputsTest(unittest.TestCase):
    def test_load_gold_inputs_returns_mixed_separated_cleaned_and_cer_lookup(self) -> None:
        mixed, separated, cleaned, cer_lookup = load_gold_inputs()
        self.assertIn("NoOverlap", mixed)
        self.assertIn("NoOverlap", separated)
        self.assertGreater(len(cer_lookup), 0)
        self.assertTrue(any(method in {"mixed_whisper", "separated_whisper"} for _, method in cer_lookup))


if __name__ == "__main__":
    unittest.main()
