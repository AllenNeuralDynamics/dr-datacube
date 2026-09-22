import unittest
from unittest import mock

import polars as pl
from polars.testing import assert_frame_equal

import dr_datacube
from dr_datacube.settings import DatacubeConfig


DEFAULT_DATACUBE_VERSION = DatacubeConfig.model_fields["version"].default
# A relatively small unit table in the current default datacube keeps the four
# remote-source scans suitable for routine CI runs.
UNIT_TEST_SESSION_ID = "714753_2024-07-02"


def identifier_value_counts(
    frame: pl.DataFrame | pl.LazyFrame,
    columns: tuple[str, ...],
) -> dict[str, pl.DataFrame]:
    """Return comparable value counts for identifier columns.

    NWB and parquet can infer different integer/string types for the same ID,
    so IDs are normalized to strings before their values are compared.
    """
    identifiers = (
        frame.lazy()
        .select(pl.col(column).cast(pl.String) for column in columns)
        .collect()
    )
    return {
        column: identifiers.group_by(column).len().sort(column)
        for column in columns
    }


class TestSessionCatalogConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with dr_datacube.config.override(
            version=DEFAULT_DATACUBE_VERSION,
            use_cache=False,
        ) as config:
            # Seed get_session_table with the complete authoritative asset set.
            # Without this patch, get_session_table is initially filtered by the
            # GitHub catalog, making the comparison circular.
            asset_session_ids = (
                pl.read_parquet(
                    (config.asset_dir / "session_table.parquet").as_posix(),
                    columns=["session_id"],
                )["session_id"]
                .sort()
                .to_list()
            )
            with mock.patch(
                "dr_datacube.sessions.get_session_ids_from_github",
                return_value=asset_session_ids,
            ):
                cls.datacube_sessions = dr_datacube.get_session_table(
                    session_type=None,
                    with_behavior_filter=False,
                    only_in_data_asset=True,
                )

    def test_github_ids_match_datacube_session_table(self) -> None:
        for session_type in (None, "brainwide", "naive", "templeton"):
            for with_behavior_filter in (False, True):
                with self.subTest(
                    session_type=session_type,
                    with_behavior_filter=with_behavior_filter,
                ):
                    expected = self.datacube_sessions
                    if session_type is not None:
                        expected = expected.filter(
                            pl.col("session_type") == session_type,
                        )
                    if with_behavior_filter:
                        expected = expected.filter(pl.col("is_behavior_pass"))

                    github_ids = dr_datacube.get_session_ids_from_github(
                        session_type=session_type,
                        with_behavior_filter=with_behavior_filter,
                    )
                    actual = pl.DataFrame(
                        {"session_id": github_ids},
                        schema={"session_id": pl.String},
                    ).with_columns(
                        subject_id=pl.col("session_id").str.split("_").list.first(),
                    )
                    expected_counts = identifier_value_counts(
                        expected,
                        ("session_id", "subject_id"),
                    )
                    actual_counts = identifier_value_counts(
                        actual,
                        ("session_id", "subject_id"),
                    )
                    for column in expected_counts:
                        assert_frame_equal(
                            actual_counts[column],
                            expected_counts[column],
                        )


class TestGetLfDataSourceConsistency(unittest.TestCase):
    def test_asset_cache_parquet_and_nwb_have_same_unit_ids(self) -> None:
        identifiers = ("session_id", "subject_id", "unit_id")

        def get_counts(*, use_cache: bool, nwb: bool) -> dict[str, pl.DataFrame]:
            with dr_datacube.config.override(
                version=DEFAULT_DATACUBE_VERSION,
                use_cache=use_cache,
            ):
                frame = dr_datacube.get_lf(
                    "unit_metrics",
                    session_id=UNIT_TEST_SESSION_ID,
                    nwb=nwb,
                )
                return identifier_value_counts(frame, identifiers)

        # The non-cache datacube asset is authoritative. Keep
        # only_in_data_asset at its default True: False intentionally exposes a
        # larger, advanced-use cache set and is not a consistency condition.
        expected = get_counts(use_cache=False, nwb=False)
        for use_cache, nwb in ((True, False), (False, True), (True, True)):
            with self.subTest(use_cache=use_cache, nwb=nwb):
                actual = get_counts(use_cache=use_cache, nwb=nwb)
                for column in identifiers:
                    assert_frame_equal(actual[column], expected[column])


if __name__ == "__main__":
    unittest.main()
