import logging
from collections.abc import Callable, Collection, Iterable
from typing import Literal

import lazynwb
import npc_session
import polars as pl
from polars._typing import FrameType

from .settings import _get_config

logger = logging.getLogger(__name__)

SessionType = Literal["brainwide", "naive", "templeton"]
_GITHUB_SESSION_TABLE_URL = (
    "https://raw.githubusercontent.com/AllenNeuralDynamics/dr-datacube/main/assets/datacube_sessions.csv"
)
_GITHUB_SESSION_TABLE: pl.DataFrame | None = None


def _behavior_summary(block_dprime_threshold: float = 1.0) -> pl.DataFrame:
    return (
        get_lf("performance", session_type=None, with_behavior_filter=False)
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
                get_lf("performance", session_type=None, with_behavior_filter=False).filter(
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
        get_lf("session", session_type=None, with_behavior_filter=False)
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
    """Return session IDs from the published table without requiring Code Ocean credentials.

    The published table is cached in memory for the lifetime of the Python process.
    """
    global _GITHUB_SESSION_TABLE
    if _GITHUB_SESSION_TABLE is None:
        _GITHUB_SESSION_TABLE = pl.read_csv(_GITHUB_SESSION_TABLE_URL)

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
    return _GITHUB_SESSION_TABLE.filter(filter_expr)["session_id"].sort().to_list()


def get_lf(
    name: str,
    session_id: str | None = None,
    nwb: bool = False,
    session_type: SessionType | Collection[SessionType] | None = "brainwide",
    with_behavior_filter: bool = True,
    only_in_data_asset: bool = True,
    **scan_args,
) -> pl.LazyFrame:
    """Return a lazily scanned datacube or NWB table.

    ``units`` always refers to the full table, including spike times and other
    large array-like columns. Use ``unit_metrics`` for the smaller consolidated
    parquet table without spike times. Set ``only_in_data_asset=False`` to
    bypass the published session list, for example when loading another session
    that is present in the cache.
    """

    config = _get_config()

    session_ids = get_session_ids_from_github(session_type, with_behavior_filter)
    if session_id:
        session_id = npc_session.extract_session_id(session_id)
        if session_id not in session_ids and only_in_data_asset:
            raise ValueError(f"{session_id} is not in the session list ({session_type=}, {with_behavior_filter=}).")
        session_ids = [session_id]

    session_filter = pl.col("session_id").is_in(session_ids)
    if name == "units" and not nwb and not config.use_cache:
        logger.warning("Full units parquet files are not available in the datacube asset; fetching from NWB.")
        nwb = True

    if name == "units" and nwb:
        logger.warning(
            "Fetching `units` includes spike times and can be slow. "
            "Setting `infer_schema_length=1` for faster inference: set it to `None` to disable this optimization.\n"
            "Use `get_lf('unit_metrics')` for the consolidated units table without spike times."
        )
        scan_args.setdefault("infer_schema_length", 1)

    if not nwb:
        storage_options = config.storage_options | scan_args.pop("storage_options", {})
        if name == "units":
            path = (
                config.parquet_dir.parent
                / "units"
                / (f"{session_id}.parquet" if session_id is not None else "*.parquet")
            )
            logger.info(f"Fetching full units table from parquet at {path.as_posix()}")
        elif name == "unit_metrics":
            path = config.parquet_dir / "units.parquet"
        else:
            path = config.parquet_dir / f"{name}.parquet"
            logger.info(f"Fetching {name} for consolidated parquet at {path.as_posix()}")
        return (
            pl.scan_parquet(
                path.as_posix(),
                storage_options=storage_options,
                **scan_args,
            )
            .pipe(_ensure_id_cols)
            .filter(session_filter)
        )
    else:
        name = _name_to_nwb_internal_path(name)
        sources = list_nwb_sources(session_ids)
        if not sources:
            raise ValueError(
                f"No NWB sources found for session_ids: {session_ids} ({session_type=}, {with_behavior_filter=})."
            )
        logger.info(f"Fetching {name} for {len(sources)} NWB sources in {config.nwb_dir}")
        lf = lazynwb.scan_nwb(sources, name, **scan_args).pipe(_ensure_id_cols)
        if name == "unit_metrics":
            lf = lf.drop("spike_times", "spike_amplitudes", "waveform_mean", "waveform_std", strict=False)
        return lf


def _name_to_nwb_internal_path(name: str) -> str:
    if any(
        name.startswith(prefix)
        for prefix in ("performance", "optotagging", "rf_mapping", "aud_rf_mapping", "vis_rf_mapping", "epochs")
    ):
        name = f"intervals/{name}"
    if any(name.endswith(suffix) for suffix in ("optotagging", "rf_mapping")):
        name += "_trials"
    if name.startswith("frametimes"):
        name = f"acquisition/{name}"
    if name in (
        "dlc_eye_camera",
        "eye_tracking",
        "facemap_front_camera",
        "facemap_side_camera",
        "lp_front_camera",
        "lp_side_camera",
        "licks",
        "running_speed",
        "rewards",
        "quiescent_interval_violations",
    ):
        name = f"processing/behavior/{name}"
    if name == "electrodes":
        name = "general/extracellular_ephys/electrodes"
    if name == "unit_metrics":
        name = "units"
    return name


def list_nwb_sources(session_id: str | Iterable[str] | None = None) -> tuple[str, ...]:
    """Get all file URIs from data asset(s) or from scratch bucket cache, depending on current config."""
    config = _get_config()
    available_sources = config.nwb_dir.glob("*.nwb*")
    if session_id:
        session_ids = [session_id] if isinstance(session_id, str) else list(session_id)
    else:
        session_ids = []
    if session_ids:
        sources = sorted(path.as_posix() for path in available_sources if path.stem in session_ids)
    else:
        sources = sorted(path.as_posix() for path in available_sources)
    logger.info(f"Found {len(sources)} NWB sources in {config.nwb_dir}")
    return tuple(sources)


def _ensure_id_cols(df: FrameType) -> FrameType:
    schema = df.lazy().collect_schema()  # works if we pass a dataframe or lazyframe
    if "session_id" in schema and "subject_id" in schema:
        logger.debug("DataFrame already has a `session_id` and `subject_id` columns, skipping parsing from `_nwb_path`")
        return df
    if "_nwb_path" not in schema and "session_id" not in schema:
        raise ValueError("Attempted to parse `session_id` from `_nwb_path` column, which doesn't exist in dataframe")
    if "session_id" not in schema:
        df = df.with_columns(
            pl.col("_nwb_path").str.split("/").list.get(-1).str.split(".").list.get(0).alias("session_id")
        )
    if "subject_id" not in schema:
        df = df.with_columns(pl.col("session_id").str.split("_").list.get(0).alias("subject_id"))
    return df
