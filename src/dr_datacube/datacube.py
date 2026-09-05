import contextlib
import functools
import logging
import os
import re
from collections.abc import Callable, Collection, Iterator
from contextvars import ContextVar
from typing import Any, Literal

import lazynwb
import npc_session
import polars as pl
import pydantic
import pydantic_settings
import upath
from polars._typing import FrameType

logger = logging.getLogger(__name__)


def pipeline_data_dir() -> upath.UPath:
    return upath.UPath("/tmp/data")


def capsule_data_dir() -> upath.UPath:
    return upath.UPath("/root/capsule/data")


@functools.cache
def on_codeocean() -> bool:
    with contextlib.suppress(Exception):
        return capsule_data_dir().exists() or is_pipeline()
    return False


@functools.cache
def is_pipeline() -> bool:
    with contextlib.suppress(Exception):
        return pipeline_data_dir().exists() and bool(os.environ.get("AWS_BATCH_JOB_ID"))
    return False


class DatacubeConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="DR_DATACUBE_",
        validate_assignment=True,
    )

    version: str = "v0.0.289"
    disable_asset_streaming: bool = False
    use_cache: bool = False
    anon: bool = False
    storage_options: dict = pydantic.Field(default_factory=lambda: {"region": "us-west-2"})

    @pydantic.field_validator("version")
    @classmethod
    def ensure_version_prefix(cls, value: str) -> str:
        return value if value.startswith("v") else f"v{value}"

    @pydantic.model_validator(mode="after")
    def set_anonymous_storage_options(self) -> "DatacubeConfig":
        if self.anon:
            self.storage_options["skip_signature"] = "true"
        else:
            self.storage_options.pop("skip_signature", None)
        return self

    @property
    def s3_cache_dir(self) -> upath.UPath:
        return upath.UPath("s3://aind-scratch-data/dynamic-routing/cache", anon=self.anon)

    @property
    def asset_dir(self) -> upath.UPath:
        if on_codeocean():
            logger.info("Running on CodeOcean: attempting to use local data asset directory")
            data_dir = pipeline_data_dir() if is_pipeline() else capsule_data_dir()
            datacube_dir = tuple(data_dir.glob("dynamicrouting_datacube*"))
            if not datacube_dir:
                logger.warning(f"Could not find dynamicrouting_datacube data asset in {data_dir}")
            elif len(datacube_dir) > 1:
                choice = next(
                    (d for d in datacube_dir if _asset_name_has_version(d.name, self.version)),
                    None,
                )
                if choice is not None:
                    logger.warning(
                        f"Found multiple dynamicrouting_datacube data assets in {data_dir}, using: {choice} (set `dr_datacube.config.version` to change)"
                    )
                    return choice
                logger.warning(
                    f"Found multiple dynamicrouting_datacube data assets in {data_dir}, but none match the requested version {self.version}. Set `dr_datacube.config.version` to match an available local asset to use it."
                )
                logger.warning("Falling back to streaming assets from S3.")
            elif _asset_name_has_version(datacube_dir[0].name, self.version):
                return datacube_dir[0]
            else:
                logger.warning(
                    f"Found dynamicrouting_datacube data asset in {data_dir}, but it does not match the requested version {self.version}. Set `dr_datacube.config.version` to match an available local asset to use it."
                )

        if self.disable_asset_streaming:
            raise FileNotFoundError(f"No local data asset matching {self.version} is attached and streaming has been disabled. Set `dr_datacube.config.version` to match an available local asset or set `dr_datacube.config.disable_asset_streaming` to False.")
        
        # to avoid dependencies we can hardcode S3 path for well-known assets:
        asset_paths = {
            "v0.0.272": "s3://codeocean-s3datasetsbucket-1u41qdg42ur9/bfaffa07-a179-4f02-9724-4e51451d8ba6",
            "v0.0.288.naive": "s3://codeocean-s3datasetsbucket-1u41qdg42ur9/e8468ee4-9825-4254-b374-fb0b9dab160a",
            "v0.0.289": "s3://codeocean-s3datasetsbucket-1u41qdg42ur9/4491d1c4-400c-4e76-b81a-c437478f188b",
        }
        if self.version in asset_paths:
            return upath.UPath(asset_paths[self.version], anon=self.anon)

        # get S3 dir of datacube asset from CO API:
        # TODO replace with docdb query when possible
        try:
            import aind_session
        except ImportError:
            raise ImportError(
                "aind_session and a CO_API_TOKEN are required to find the datacube data asset on S3. Install as an optional-dependency with `dr-datacube[co]`."
            )
        return upath.UPath(
            aind_session.get_data_asset_source_dir(
                next(
                    d
                    for d in reversed(aind_session.get_data_assets("dynamicrouting_datacube"))
                    if _asset_name_has_version(d.name, self.version)
                ).id
            ),
            anon=self.anon,
        )

    @property
    def nwb_dir(self) -> upath.UPath:
        if self.use_cache:
            return self.s3_cache_dir / "nwb" / self.version
        return self.asset_dir / "nwb"

    @property
    def parquet_dir(self) -> upath.UPath:
        if self.use_cache:
            return self.s3_cache_dir / "nwb_components" / self.version / "consolidated"
        else:
            return self.nwb_dir.parent / "consolidated"

    def override(self, **overrides: Any) -> contextlib.AbstractContextManager["DatacubeConfig"]:
        """Temporarily use a copy of this config without changing this object."""
        base = _get_config() if self is config else self
        return _temporary_config(base, overrides)


config = DatacubeConfig()
_active_config: ContextVar[DatacubeConfig | None] = ContextVar("dr_datacube_config", default=None)


def _get_config() -> DatacubeConfig:
    """Return the config active in the current context, or the global config."""
    active_config = _active_config.get()
    return config if active_config is None else active_config


class _ContextualAnon:
    """Resolve Lazynwb anonymous access from the active datacube config."""

    def __bool__(self) -> bool:
        return _get_config().anon


# Lazynwb resolves this option when it opens a file, including when a LazyFrame is
# collected after ``get_lf`` returns. Keeping the value context-aware avoids sharing
# temporary anonymous-access settings between concurrent tasks.
lazynwb.config.anon = None  # fallback to fsspec storage options
lazynwb.config.fsspec_storage_options["anon"] = _ContextualAnon()


def _asset_name_has_version(asset_name: str, version: str) -> bool:
    """Return whether an asset name contains the complete requested version token."""
    version_token = re.escape(version)
    return re.search(rf"(?<![A-Za-z0-9.]){version_token}(?![A-Za-z0-9.])", asset_name) is not None


@contextlib.contextmanager
def _temporary_config(base: DatacubeConfig, overrides: dict[str, Any]) -> Iterator[DatacubeConfig]:
    temporary_config = DatacubeConfig(**{**base.model_dump(), **overrides})
    token = _active_config.set(temporary_config)
    try:
        yield temporary_config
    finally:
        _active_config.reset(token)


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
    else:

        def _name_to_nwb_internal_path(name: str) -> str:
            if any(
                name.startswith(k)
                for k in ("performance", "optotagging", "rf_mapping", "aud_rf_mapping", "vis_rf_mapping", "epochs")
            ):
                name = f"intervals/{name}"
            if any(name.endswith(k) for k in ("optotagging", "rf_mapping")):
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

        name = _name_to_nwb_internal_path(name)
        try:
            import lazynwb
        except ImportError:
            raise ImportError(
                "lazynwb is required to read NWBs. Install as an optional-dependency with `dr-datacube[nwb]`."
            )
        if session_id is not None:
            sources = (config.nwb_dir / f"{session_id}.nwb").as_posix()
            logger.info(f"Fetching {name} for NWB source {sources}")
        else:
            sources = list_nwb_sources()
            logger.info(f"Fetching {name} for {len(sources)} NWB sources in {config.nwb_dir}")
        return lazynwb.scan_nwb(sources, name, **scan_args).pipe(ensure_id_cols)


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


def behavior_summary(block_dprime_threshold: float = 1.0) -> pl.DataFrame:
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
    ).collect()  # Added return statement


def brainwide_ephys_filter(with_behavior_filter: bool = True) -> pl.Expr:
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
            behavior_summary(block_dprime_threshold=1.0).filter(
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


def naive_ephys_filter(with_behavior_filter: bool = True) -> pl.Expr:
    required = ("dynamic_routing", "task", "ephys", "ccf", "context naive")
    # TODO prod should be included, but is incorrect
    # TODO switch to "context_naive" (w/underscore) when fixed in v0.0.290
    excluded = ("issues", "templeton")  # TODO add "brainwide_survey" when fixed in v0.0.290
    if with_behavior_filter:
        engaged_session_ids = (
            behavior_summary().filter(
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


def templeton_ephys_filter(with_behavior_filter: bool = True) -> pl.Expr:
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
    """Return a dict of filter functions for each session_type. Each function has signature (with_behavior_filter: bool = True) -> pl.Expr."""
    return {
        "brainwide": brainwide_ephys_filter,
        "naive": naive_ephys_filter,
        "templeton": templeton_ephys_filter,
    }


def get_session_table(
    session_type: Literal["brainwide", "naive", "templeton"]
    | Collection[Literal["brainwide", "naive", "templeton"]]
    | None = None,
    with_behavior_filter: bool = True,
    only_in_data_asset: bool = True,
) -> pl.DataFrame:
    """A DataFrame with 'session_id' and 'keywords'.

    Options:
    'brainwide' - standard brainwide survey ephys sessions
    'naive' - context naive dynamic routing ephys sessions.
    'templeton' - Templeton ephys sessions.
    None - all sessions in datacube

    If `with_behavior_filter` is True, a standard behavioral filter for each session type will be applied.

    If only_in_data_asset is True, a further filter will be applied to return only sessions present in the CO data asset. This requires credentials to check CO and S3.
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
        .join(behavior_summary().lazy(), on="session_id", how="left")
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
    session_type: Literal["brainwide", "naive", "templeton"]
    | Collection[Literal["brainwide", "naive", "templeton"]]
    | None = "brainwide",
    with_behavior_filter: bool = True,
) -> list[str]:
    """Return a list of session IDs for the given session type, without requiring credentials to access the CO data asset."""
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


if __name__ == "__main__":
    get_lf("session")
    import doctest

    doctest.testmod()
