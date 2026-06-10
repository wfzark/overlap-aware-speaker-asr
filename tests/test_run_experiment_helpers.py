from __future__ import annotations

import unittest
from unittest.mock import patch

from src.run_experiment import parse_args


class RunExperimentHelpersTest(unittest.TestCase):
    @patch("sys.argv", ["run_experiment", "--stage", "compare"])
    def test_parse_args_accepts_compare_stage(self) -> None:
        args = parse_args()
        self.assertEqual(args.stage, "compare")
        self.assertFalse(args.overwrite)

    @patch("sys.argv", ["run_experiment", "--stage", "separated", "--overwrite"])
    def test_parse_args_accepts_separated_stage_with_overwrite(self) -> None:
        args = parse_args()
        self.assertEqual(args.stage, "separated")
        self.assertTrue(args.overwrite)


if __name__ == "__main__":
    unittest.main()
