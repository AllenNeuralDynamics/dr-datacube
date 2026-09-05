import unittest

import polars as pl

from dr_datacube import (
    _brainwide_ephys_filter,
    _naive_ephys_filter,
    _templeton_ephys_filter,
)


class TestSessionFilters(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = pl.DataFrame(
            {
                "session_id": ["brainwide", "naive", "templeton", "issues"],
                "keywords": [
                    ["prod", "brainwide_survey", "task", "ephys", "ccf"],
                    ["dynamic_routing", "task", "ephys", "ccf", "context naive"],
                    ["prod", "templeton", "task", "ephys", "ccf"],
                    ["prod", "brainwide_survey", "task", "ephys", "ccf", "issues"],
                ],
            }
        )

    def test_brainwide_keyword_filter(self) -> None:
        result = self.sessions.filter(_brainwide_ephys_filter(with_behavior_filter=False))

        self.assertEqual(result["session_id"].to_list(), ["brainwide"])

    def test_naive_keyword_filter(self) -> None:
        result = self.sessions.filter(_naive_ephys_filter(with_behavior_filter=False))

        self.assertEqual(result["session_id"].to_list(), ["naive"])

    def test_templeton_keyword_filter(self) -> None:
        result = self.sessions.filter(_templeton_ephys_filter(with_behavior_filter=False))

        self.assertEqual(result["session_id"].to_list(), ["templeton"])


if __name__ == "__main__":
    unittest.main()
