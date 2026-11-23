"""Standard behavioral criteria for Dynamic Routing sessions.

These functions operate on a ``performance`` table (one row per task block, as in the NWB
``intervals/performance`` table or the consolidated ``performance.parquet``), so they can be
used both on the datacube and on a single session's performance table, e.g. from
``npc_sessions``. They are the single definition of the behavior filters used by
:func:`dr_datacube.get_session_table`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)

__all__ = [
    "BLOCK_DPRIME_THRESHOLD",
    "MIN_CONTINGENT_REWARDS",
    "behavior_pass_expr",
    "behavior_summary",
    "get_behavior_pass_session_ids",
    "is_behavior_pass",
]

SessionType = Literal["brainwide", "naive", "templeton"]

BLOCK_DPRIME_THRESHOLD = 1.0
"""Minimum d' for a block to count as 'good'."""
MIN_CONTINGENT_REWARDS = 10
"""Minimum number of contingent rewards for a block to count as 'engaged'."""

_SINGLE_SESSION_ID = "__session__"
_DPRIME_COLUMNS = ("cross_modality_dprime", "vis_dprime", "aud_dprime")


def _to_lazyframe(performance: Any) -> pl.LazyFrame:
    if isinstance(performance, pl.LazyFrame):
        lf = performance
    elif isinstance(performance, pl.DataFrame):
        lf = performance.lazy()
    else:  # e.g. pandas.DataFrame from npc_sessions: only convert the columns needed
        columns = [c for c in ("session_id", "rewarded_modality", "n_contingent_rewards", *_DPRIME_COLUMNS) if c in performance.columns]
        lf = pl.from_pandas(performance[columns]).lazy()
    schema = lf.collect_schema()
    if "session_id" not in schema:
        lf = lf.with_columns(session_id=pl.lit(_SINGLE_SESSION_ID))
    # d' columns that don't apply to a session (e.g. cross-modality d' for Templeton) may be absent
    lf = lf.with_columns(pl.lit(None, dtype=pl.Float64).alias(c) for c in _DPRIME_COLUMNS if c not in schema)
    schema = lf.collect_schema()
    # NaN (e.g. from pandas or NWB) means 'not applicable', same as null. Without this,
    # NaN >= threshold is True in polars.
    return lf.with_columns(
        pl.col(c).cast(pl.Float64).fill_nan(None) for c in _DPRIME_COLUMNS if c in schema
    )


def behavior_summary(
    performance: Any,
    block_dprime_threshold: float = BLOCK_DPRIME_THRESHOLD,
    min_contingent_rewards: int = MIN_CONTINGENT_REWARDS,
) -> pl.DataFrame:
    """Return per-session block counts used by the standard behavior filters.

    ``performance`` can be a polars DataFrame/LazyFrame or a pandas DataFrame, with
    one row per block. If it has no ``session_id`` column it's treated as a single
    session.

    A block is *engaged* if it has at least ``min_contingent_rewards`` contingent
    rewards and *good* if its cross-modality d' is at least ``block_dprime_threshold``.
    A *good single-modality* block (Templeton) has no cross-modality d' and a visual or
    auditory d' of at least ``block_dprime_threshold``.
    """
    lf = _to_lazyframe(performance)
    return (
        lf.with_columns(
            pl.col("n_contingent_rewards").ge(min_contingent_rewards).fill_null(False).alias("is_engaged_block"),
            pl.col("cross_modality_dprime").ge(block_dprime_threshold).fill_null(False).alias("is_good_block"),
            (
                pl.col("cross_modality_dprime").is_null()
                & (pl.col("aud_dprime").ge(block_dprime_threshold) | pl.col("vis_dprime").ge(block_dprime_threshold))
            )
            .fill_null(False)
            .alias("is_good_single_modality_block"),
        )
        .with_columns(
            (pl.col("is_good_block") & pl.col("is_engaged_block")).alias("is_good_engaged_block"),
        )
        .group_by("session_id", "rewarded_modality")
        .agg(
            pl.col(
                "is_engaged_block", "is_good_block", "is_good_engaged_block", "is_good_single_modality_block"
            )
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
            pl.col("n_good_single_modality_blocks").sum(),
        )
        .with_columns(pl.selectors.starts_with("n_").cast(pl.UInt32).fill_null(0))
        .sort("session_id")
        .collect()
    )


def behavior_pass_expr(session_type: SessionType) -> pl.Expr:
    """Return the behavior-pass criterion for a session type, as an expression on
    :func:`behavior_summary` output.

    - ``brainwide``: at least 2 good, engaged blocks of each rewarded modality.
    - ``naive``: at least 4 engaged blocks.
    - ``templeton``: at least 1 good single-modality block.
    """
    if session_type == "brainwide":
        return pl.col("n_good_engaged_aud_blocks").ge(2) & pl.col("n_good_engaged_vis_blocks").ge(2)
    if session_type == "naive":
        return pl.col("n_engaged_blocks").ge(4)
    if session_type == "templeton":
        return pl.col("n_good_single_modality_blocks").ge(1)
    raise ValueError(f"Unknown session type: {session_type!r}")


def get_behavior_pass_session_ids(
    performance: Any,
    session_type: SessionType | Collection[SessionType],
) -> list[str]:
    """Return IDs of sessions in ``performance`` that pass the behavior criterion for
    ``session_type`` (any of them, if a collection is passed)."""
    session_types = [session_type] if isinstance(session_type, str) else list(session_type)
    return (
        behavior_summary(performance)
        .filter(pl.any_horizontal(behavior_pass_expr(t) for t in session_types))["session_id"]
        .to_list()
    )


def is_behavior_pass(performance: Any, session_type: SessionType) -> bool:
    """Whether a single session's ``performance`` table passes the behavior criterion for
    ``session_type``.

    >>> import polars as pl
    >>> performance = pl.DataFrame({
    ...     "rewarded_modality": ["vis", "aud", "vis", "aud"],
    ...     "n_contingent_rewards": [20, 20, 20, 9],
    ...     "cross_modality_dprime": [2.0, 2.0, 2.0, 2.0],
    ...     "vis_dprime": [2.0, 2.0, 2.0, 2.0],
    ...     "aud_dprime": [2.0, 2.0, 2.0, 2.0],
    ... })
    >>> is_behavior_pass(performance, "brainwide")  # only 1 engaged aud block
    False
    >>> is_behavior_pass(performance[:3], "naive")  # only 3 engaged blocks
    False
    """
    summary = behavior_summary(performance)
    if summary["session_id"].n_unique() > 1:
        raise ValueError("`performance` contains more than one session: use `get_behavior_pass_session_ids`")
    if summary.is_empty():
        return False
    return bool(summary.select(behavior_pass_expr(session_type).fill_null(False)).item())
