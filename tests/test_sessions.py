import unittest

import polars as pl

from dr_datacube.sessions import (
    _brainwide_ephys_filter,
    _ensure_id_cols,
    _naive_ephys_filter,
    _name_to_nwb_internal_path,
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


class TestEnsureIdCols(unittest.TestCase):
    def test_extracts_session_and_subject_ids_from_nwb_path(self) -> None:
        result = _ensure_id_cols(
            pl.DataFrame(
                {
                    "_nwb_path": ["s3://bucket/nwb/123456_2024-01-01.nwb"],
                    "value": [1],
                }
            )
        )

        self.assertEqual(result["session_id"].to_list(), ["123456_2024-01-01"])
        self.assertEqual(result["subject_id"].to_list(), ["123456"])

    def test_preserves_existing_id_columns(self) -> None:
        original = pl.DataFrame(
            {
                "session_id": ["123456_2024-01-01"],
                "subject_id": ["123456"],
            }
        )

        self.assertIs(_ensure_id_cols(original), original)

    def test_requires_a_source_for_missing_session_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "_nwb_path"):
            _ensure_id_cols(pl.DataFrame({"value": [1]}))


class TestNwbInternalPaths(unittest.TestCase):
    def test_translates_table_names(self) -> None:
        expected_paths = {
            "performance": "intervals/performance",
            "optotagging": "intervals/optotagging_trials",
            "frametimes_eye_camera": "acquisition/frametimes_eye_camera",
            "running_speed": "processing/behavior/running_speed",
            "electrodes": "general/extracellular_ephys/electrodes",
            "units": "units",
        }

        for name, expected in expected_paths.items():
            with self.subTest(name=name):
                self.assertEqual(_name_to_nwb_internal_path(name), expected)

if __name__ == "__main__":
    unittest.main()
