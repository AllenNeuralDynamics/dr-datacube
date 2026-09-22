import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import polars as pl

import dr_datacube.sessions as sessions_module
from dr_datacube.sessions import (
    _brainwide_ephys_filter,
    _ensure_id_cols,
    _naive_ephys_filter,
    _name_to_nwb_internal_path,
    _templeton_ephys_filter,
    get_lf,
    get_session_ids_from_github,
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


class TestGetSessionIdsFromGithub(unittest.TestCase):
    def setUp(self) -> None:
        self.original_table = sessions_module._GITHUB_SESSION_TABLE
        sessions_module._GITHUB_SESSION_TABLE = None

    def tearDown(self) -> None:
        sessions_module._GITHUB_SESSION_TABLE = self.original_table

    @mock.patch("dr_datacube.sessions.pl.read_csv")
    def test_caches_published_table(self, read_csv: mock.Mock) -> None:
        read_csv.return_value = pl.DataFrame(
            {
                "session_type": ["brainwide", "templeton"],
                "is_behavior_pass": [True, False],
                "session_id": ["b", "t"],
            }
        )

        self.assertEqual(get_session_ids_from_github("brainwide"), ["b"])
        self.assertEqual(
            get_session_ids_from_github("templeton", with_behavior_filter=False),
            ["t"],
        )

        read_csv.assert_called_once_with(sessions_module._GITHUB_SESSION_TABLE_URL)


class TestGetLfUnits(unittest.TestCase):
    session_id = "123456_2024-01-01"

    @staticmethod
    def scanned_lf() -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "session_id": [TestGetLfUnits.session_id],
                "subject_id": ["123456"],
            }
        ).lazy()

    def cache_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            use_cache=True,
            nwb_only=False,
            parquet_dir=Path("/cache/nwb_components/v1/consolidated"),
            storage_options={},
        )

    @mock.patch("dr_datacube.sessions.get_session_ids_from_github")
    @mock.patch("dr_datacube.sessions.npc_session.extract_session_id")
    @mock.patch("dr_datacube.sessions.pl.scan_parquet")
    @mock.patch("dr_datacube.sessions._get_config")
    def test_units_with_session_scans_only_that_parquet(
        self,
        get_config: mock.Mock,
        scan_parquet: mock.Mock,
        extract_session_id: mock.Mock,
        get_session_ids: mock.Mock,
    ) -> None:
        get_config.return_value = self.cache_config()
        scan_parquet.return_value = self.scanned_lf()
        extract_session_id.return_value = self.session_id
        get_session_ids.return_value = [self.session_id]

        with mock.patch.object(Path, "exists", return_value=True):
            get_lf("units", session_id=self.session_id)

        scan_parquet.assert_called_once_with(
            f"/cache/nwb_components/v1/units/{self.session_id}.parquet",
            storage_options={},
        )
    @mock.patch("dr_datacube.sessions.get_session_ids_from_github")
    @mock.patch("dr_datacube.sessions.npc_session.extract_session_id")
    @mock.patch("dr_datacube.sessions.pl.scan_parquet")
    @mock.patch("dr_datacube.sessions._get_config")
    def test_only_in_data_asset_false_allows_other_cached_session(
        self,
        get_config: mock.Mock,
        scan_parquet: mock.Mock,
        extract_session_id: mock.Mock,
        get_session_ids: mock.Mock,
    ) -> None:
        uncatalogued_session_id = "999999_2024-01-01"
        get_config.return_value = self.cache_config()
        scan_parquet.return_value = pl.DataFrame(
            {
                "session_id": [uncatalogued_session_id],
                "subject_id": ["999999"],
            }
        ).lazy()
        extract_session_id.return_value = uncatalogued_session_id

        with mock.patch.object(Path, "exists", return_value=True):
            result = get_lf(
                "units",
                session_id=uncatalogued_session_id,
                only_in_data_asset=False,
            ).collect()

        get_session_ids.assert_not_called()
        self.assertEqual(result["session_id"].to_list(), [uncatalogued_session_id])
        scan_parquet.assert_called_once_with(
            f"/cache/nwb_components/v1/units/{uncatalogued_session_id}.parquet",
            storage_options={},
        )

    @mock.patch("dr_datacube.sessions.get_session_ids_from_github", return_value=[])
    @mock.patch("dr_datacube.sessions.pl.scan_parquet")
    @mock.patch("dr_datacube.sessions._get_config")
    def test_units_without_session_scans_all_units_parquets(
        self,
        get_config: mock.Mock,
        scan_parquet: mock.Mock,
        _get_session_ids: mock.Mock,
    ) -> None:
        get_config.return_value = self.cache_config()
        scan_parquet.return_value = self.scanned_lf()

        with mock.patch.object(Path, "exists", return_value=True):
            get_lf("units")

        scan_parquet.assert_called_once_with(
            "/cache/nwb_components/v1/units/*.parquet",
            storage_options={},
        )

    @mock.patch("dr_datacube.sessions.get_session_ids_from_github", return_value=[])
    @mock.patch("dr_datacube.sessions.lazynwb.scan_nwb")
    @mock.patch("dr_datacube.sessions.list_nwb_sources", return_value=("first.nwb", "second.nwb"))
    @mock.patch("dr_datacube.sessions._get_config")
    def test_units_falls_back_to_nwb_when_parquet_is_unavailable(
        self,
        get_config: mock.Mock,
        _list_nwb_sources: mock.Mock,
        scan_nwb: mock.Mock,
        _get_session_ids: mock.Mock,
    ) -> None:
        get_config.return_value = SimpleNamespace(
            use_cache=False,
            nwb_only=False,
            nwb_dir=Path("/asset/nwb"),
        )
        scan_nwb.return_value = self.scanned_lf()

        with self.assertLogs("dr_datacube.sessions", level="WARNING") as logs:
            get_lf("units")

        scan_nwb.assert_called_once_with(
            ("first.nwb", "second.nwb"),
            "units",
            infer_schema_length=1,
        )
        all_logs = " ".join(logs.output)
        self.assertIn("not available", all_logs)
        self.assertIn("can be slow", all_logs)
        self.assertIn("unit_metrics", all_logs)

    @mock.patch("dr_datacube.sessions.get_session_ids_from_github", return_value=[])
    @mock.patch("dr_datacube.sessions.pl.scan_parquet")
    @mock.patch("dr_datacube.sessions._get_config")
    def test_unit_metrics_uses_consolidated_parquet(
        self,
        get_config: mock.Mock,
        scan_parquet: mock.Mock,
        _get_session_ids: mock.Mock,
    ) -> None:
        get_config.return_value = self.cache_config()
        scan_parquet.return_value = self.scanned_lf()

        with mock.patch.object(Path, "exists", return_value=True):
            get_lf("unit_metrics")

        scan_parquet.assert_called_once_with(
            "/cache/nwb_components/v1/consolidated/units.parquet",
            storage_options={},
        )


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
