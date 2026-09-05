import logging

import lazynwb
import npc_session
import polars as pl
from polars._typing import FrameType

from .settings import _get_config

logger = logging.getLogger(__name__)


def get_lf(
    name: str,
    session_id: str | None = None,
    nwb: bool = False,
    **scan_args,
) -> pl.LazyFrame:
    config = _get_config()
    if session_id:
        session_id = npc_session.extract_session_id(session_id)
    if not nwb:
        storage_options = config.storage_options | scan_args.pop("storage_options", {})
        if "units" in name and session_id is not None and not config.use_cache:
            logger.warning(
                "Full units table with spike times, amplitudes and waveforms is not available as parquet in data asset: pass `get_lf(..., nwb=True)`"
            )
        if "units" in name and session_id is not None and config.use_cache:
            logger.info(f"Fetching single session full units table for session_id={session_id}")
            path = config.parquet_dir.parent / "units" / f"{session_id}.parquet"
            session_filter = pl.lit(True)
        else:
            path = config.parquet_dir / f"{name}.parquet"
            logger.info(f"Fetching {name} for consolidated parquet at {path.as_posix()}")
            session_filter = pl.col("session_id").eq(session_id) if session_id is not None else pl.lit(True)
        return (
            pl.scan_parquet(
                path.as_posix(),
                storage_options=storage_options,
                **scan_args,
            )
            .pipe(ensure_id_cols)
            .filter(session_filter)
        )

    name = _name_to_nwb_internal_path(name)
    if session_id is not None:
        sources = (config.nwb_dir / f"{session_id}.nwb").as_posix()
        logger.info(f"Fetching {name} for NWB source {sources}")
    else:
        sources = list_nwb_sources()
        logger.info(f"Fetching {name} for {len(sources)} NWB sources in {config.nwb_dir}")
    return lazynwb.scan_nwb(sources, name, **scan_args).pipe(ensure_id_cols)


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
    return name


def list_nwb_sources() -> tuple[str, ...]:
    """Get all file URIs from data asset(s) or from scratch bucket cache, depending on current config."""
    config = _get_config()
    sources = sorted(path.as_posix() for path in config.nwb_dir.glob("*.nwb*"))
    logger.info(f"Found {len(sources)} NWB sources in {config.nwb_dir}")
    return tuple(sources)


def ensure_id_cols(df: FrameType) -> FrameType:
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
