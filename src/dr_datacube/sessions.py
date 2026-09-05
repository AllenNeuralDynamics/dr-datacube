from collections.abc import Callable, Collection
from typing import Literal

import polars as pl

from .loading import get_lf
from .settings import _get_config

SessionType = Literal["brainwide", "naive", "templeton"]


def _behavior_summary(block_dprime_threshold: float = 1.0) -> pl.DataFrame:
    return (
        get_lf("performance")
        .with_columns(
            pl.col("n_contingent_rewards").ge(10).alias("is_engaged_block"),
            pl.col("cross_modality_dprime").ge(block_dprime_threshold).alias("is_good_block"),
        )
        .with_columns(
            (pl.col("is_good_block") & pl.col("is_engaged_block")).alias("is_good_engaged_block"),
        )
        .group_by("session_id", "rewarded_modality")
        .agg(
            "is_engaged_block",
            "is_good_block",
            "is_good_engaged_block",
            pl.col("is_engaged_block", "is_good_block", "is_good_engaged_block")
            .sum()
            .name.replace("is_", "n_")
            .name.suffix("s"),
        )
        .group_by("session_id")
        .agg(
            pl.col("n_engaged_blocks").sum(),
            pl.col("n_good_blocks").sum(),
            pl.col("n_good_engaged_blocks").sum(),
            pl.col("n_good_blocks").filter(pl.col("rewarded_modality") == "vis").first().alias("n_good_vis_blocks"),
            pl.col("n_good_blocks").filter(pl.col("rewarded_modality") == "aud").first().alias("n_good_aud_blocks"),
            pl.col("n_good_engaged_blocks")
            .filter(pl.col("rewarded_modality") == "vis")
            .first()
            .alias("n_good_engaged_vis_blocks"),
            pl.col("n_good_engaged_blocks")
            .filter(pl.col("rewarded_modality") == "aud")
            .first()
            .alias("n_good_engaged_aud_blocks"),
        )
    ).collect()


def _brainwide_ephys_filter(with_behavior_filter: bool = True) -> pl.Expr:
    required = (
        "prod",
        "brainwide_survey",
        "task",
        "ephys",
        "ccf",
    )  # good_behavior is incorrect - will be fixed in v0.0.290
    excluded = (
        "issues",
        "context naive",
    )  # context_naive (w underscore) had bug - should be mutually exclusive with bws from >=v0.0.290
    if with_behavior_filter:
        good_behavior_session_ids = (
            _behavior_summary(block_dprime_threshold=1.0).filter(
                pl.col("n_good_engaged_aud_blocks").ge(2),
                pl.col("n_good_engaged_vis_blocks").ge(2),
            )
        )["session_id"].to_list()
    else:
        good_behavior_session_ids = []
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
        pl.col("session_id").is_in(good_behavior_session_ids) if with_behavior_filter else pl.lit(True),
    )


def _naive_ephys_filter(with_behavior_filter: bool = True) -> pl.Expr:
    required = ("dynamic_routing", "task", "ephys", "ccf", "context naive")
    # TODO prod should be included, but is incorrect
    # TODO switch to "context_naive" (w/underscore) when fixed in v0.0.290
    excluded = ("issues", "templeton")  # TODO add "brainwide_survey" when fixed in v0.0.290
    if with_behavior_filter:
        engaged_session_ids = (
            _behavior_summary().filter(
                pl.col("n_engaged_blocks").ge(4),
            )
        )["session_id"].to_list()
    else:
        engaged_session_ids = []
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
        pl.col("session_id").is_in(engaged_session_ids) if with_behavior_filter else pl.lit(True),
    )


def _templeton_ephys_filter(with_behavior_filter: bool = True) -> pl.Expr:
    required = ("prod", "templeton", "task", "ephys", "ccf")
    excluded = ("issues",)
    if with_behavior_filter:
        good_behavior_session_ids = (
            (
                get_lf("performance").filter(
                    pl.col("cross_modality_dprime").is_null(),
                    pl.col("aud_dprime").ge(1.0) | pl.col("vis_dprime").ge(1.0),
                )
            )
            .collect()["session_id"]
            .to_list()
        )
    else:
        good_behavior_session_ids = []
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
        pl.col("session_id").is_in(good_behavior_session_ids) if with_behavior_filter else pl.lit(True),
    )


def filter_functions() -> dict[str, Callable[[bool], pl.Expr]]:
    """Return filter functions keyed by session type."""
    return {
        "brainwide": _brainwide_ephys_filter,
        "naive": _naive_ephys_filter,
        "templeton": _templeton_ephys_filter,
    }


def get_session_table(
    session_type: SessionType | Collection[SessionType] | None = None,
    with_behavior_filter: bool = True,
    only_in_data_asset: bool = True,
) -> pl.DataFrame:
    """Return sessions and their standard session-set classifications.

    ``session_type`` selects brainwide, naive, or Templeton sessions. Passing
    ``None`` returns every session in the datacube. When
    ``with_behavior_filter`` is true, each session type's standard behavioral
    filter is applied. When ``only_in_data_asset`` is true, only sessions in the
    configured Code Ocean data asset are returned.
    """
    config = _get_config()
    session_expr = (
        pl.lit(True)
        if session_type is None
        else (
            pl.col("session_type").is_in(session_type)
            if not isinstance(session_type, str)
            else pl.col("session_type").eq(session_type)
        )
    )
    filtered = (
        get_lf("session")
        .select("session_id", "subject_id", "keywords")
        .with_columns(
            session_type=pl.coalesce(
                *(
                    pl.when(func(with_behavior_filter=with_behavior_filter)).then(pl.lit(name))
                    for name, func in filter_functions().items()
                )
            )
        )
        .with_columns(
            is_behavior_pass=(
                pl.when(pl.col("session_type").eq("brainwide"))
                .then(filter_functions()["brainwide"](with_behavior_filter=True))
                .when(pl.col("session_type").eq("naive"))
                .then(filter_functions()["naive"](with_behavior_filter=True))
                .when(pl.col("session_type").eq("templeton"))
                .then(filter_functions()["templeton"](with_behavior_filter=True))
                .when(pl.col("session_type").is_not_null())
                .then(pl.lit(False))
            )
        )
        .filter(session_expr)
        .join(_behavior_summary().lazy(), on="session_id", how="left")
        .collect()
    )
    if only_in_data_asset:
        session_ids_in_data_asset = (
            pl.read_parquet((config.asset_dir / "session_table.parquet").as_posix(), columns=["session_id"])[
                "session_id"
            ]
            .sort()
            .to_list()
        )
        filtered = filtered.filter(pl.col("session_id").is_in(session_ids_in_data_asset))
    return filtered


def get_session_ids_from_github(
    session_type: SessionType | Collection[SessionType] | None = "brainwide",
    with_behavior_filter: bool = True,
) -> list[str]:
    """Return session IDs from the published table without requiring Code Ocean credentials."""
    if session_type is None:
        filter_expr = pl.lit(True)
    else:
        filter_expr = (
            pl.col("session_type").is_in(session_type)
            if not isinstance(session_type, str)
            else pl.col("session_type").eq(session_type)
        )
    if with_behavior_filter:
        filter_expr = filter_expr & pl.col("is_behavior_pass")
    return (
        pl.read_csv(
            "https://raw.githubusercontent.com/AllenNeuralDynamics/dr-datacube/main/assets/datacube_sessions.csv"
        )
        .filter(filter_expr)["session_id"]
        .sort()
        .to_list()
    )
