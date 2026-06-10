from __future__ import annotations

import unittest

from src.adaptive_router_v2 import is_unstable


class AdaptiveRouterV2StabilityTest(unittest.TestCase):
    def test_is_unstable_flags_length_inflation(self) -> None:
        self.assertTrue(is_unstable(mixed_len=100, separated_len=150, duplicate_removed_count=0, runtime_ratio=1.0))

    def test_is_unstable_flags_high_duplicate_removal(self) -> None:
        self.assertTrue(is_unstable(mixed_len=100, separated_len=100, duplicate_removed_count=10, runtime_ratio=1.0))

    def test_is_unstable_is_false_for_balanced_outputs(self) -> None:
        self.assertFalse(is_unstable(mixed_len=100, separated_len=110, duplicate_removed_count=2, runtime_ratio=1.2))


if __name__ == "__main__":
    unittest.main()
