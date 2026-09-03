# /// script
# dependencies = [
#     "dr-datacube[co]",
# ]
# requires-python = ">=3.11"
#
# [tool.uv.sources]
# dr-datacube = { git = "https://github.com/AllenNeuralDynamics/dr-datacube" }
# ///
"""Write session IDs for each standard ephys preset to assets/session_ids.json."""

from pathlib import Path

import polars as pl

import dr_datacube


def session_table() -> pl.DataFrame:
    df = (
        dr_datacube.get_session_table(
            session_type=None,
            with_behavior_filter=False,
            only_in_data_asset=True,
        )
        .select("subject_id", "session_id", "is_behavior_pass", "session_type")
        .sort("session_id")
    )
    return df


def dump_session_table() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    session_table().write_csv(output_dir / "datacube_sessions.csv")


if __name__ == "__main__":
    dump_session_table()
