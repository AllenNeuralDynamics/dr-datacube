import contextlib
import functools
import logging
import os
import re
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

import lazynwb
import pydantic
import pydantic_settings
import upath


logger = logging.getLogger(__name__)


def _pipeline_data_dir() -> upath.UPath:
    return upath.UPath("/tmp/data")


def _capsule_data_dir() -> upath.UPath:
    return upath.UPath("/root/capsule/data")


@functools.cache
def on_codeocean() -> bool:
    with contextlib.suppress(Exception):
        return _capsule_data_dir().exists() or is_pipeline()
    return False


@functools.cache
def is_pipeline() -> bool:
    with contextlib.suppress(Exception):
        return _pipeline_data_dir().exists() and bool(os.environ.get("AWS_BATCH_JOB_ID"))
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
            data_dir = _pipeline_data_dir() if is_pipeline() else _capsule_data_dir()
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
            raise FileNotFoundError(
                f"No local data asset matching {self.version} is attached and streaming has been disabled. Set `dr_datacube.config.version` to match an available local asset or set `dr_datacube.config.disable_asset_streaming` to False."
            )

        # To avoid dependencies, hardcode S3 paths for well-known assets.
        asset_paths = {
            "v0.0.272": "s3://codeocean-s3datasetsbucket-1u41qdg42ur9/bfaffa07-a179-4f02-9724-4e51451d8ba6",
            "v0.0.288.naive": "s3://codeocean-s3datasetsbucket-1u41qdg42ur9/e8468ee4-9825-4254-b374-fb0b9dab160a",
            "v0.0.289": "s3://codeocean-s3datasetsbucket-1u41qdg42ur9/4491d1c4-400c-4e76-b81a-c437478f188b",
        }
        if self.version in asset_paths:
            return upath.UPath(asset_paths[self.version], anon=self.anon)

        # Get the S3 directory of the datacube asset from the Code Ocean API.
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
