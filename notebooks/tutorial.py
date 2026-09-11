# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dr-datacube",
#     "marimo",
#     "polars",
# ]
#
# [tool.uv.sources]
# dr-datacube = { git = "https://github.com/AllenNeuralDynamics/dr-datacube" }
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import dr_datacube

    return dr_datacube, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dynamic Routing datacube tutorial

    `dr_datacube.get_lf()` returns a Polars `LazyFrame`, so queries can push
    column selection and row filters down to the remote parquet/NWB files. This
    notebook uses the public cache with anonymous S3 access and demonstrates
    the current defaults.

    - By default, `get_lf()` selects behavior-passing brain-wide
    sessions in the published datacube, i.e. **the sessions that should be used for most analyses**.

    - The default data source is the datacube Code Ocean asset, accessed locally when in Code Ocean, or via streaming from S3 when outside. The data asset is created from a less-restrictive "cache" in a public bucket, which is a superset of the asset: this is convenient as it can be used from outside Code Ocean, without AWS credentials.

    - The default datacube version is kept up-to-date, but can be controlled by modifying the global `dr_datacube` config.
    """)
    return


@app.cell
def _(dr_datacube):
    dr_datacube.config.use_cache = True
    dr_datacube.config.anon = True
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Configuration options**

    | Option | Effect |
    |---|---|
    | `version` | Select the datacube version. |
    | `use_cache` | Use the public scratch bucket (`aind-scratch-data`) for both NWB files and parquet tables. |
    | `disable_asset_streaming` | Disable streaming the datacube asset through fsspec/boto; a matching asset must be attached locally in Code Ocean. |
    | `anon` | Use anonymous access: AWS credentials are not required and are not used, even if available. |
    | `nwb_only` | Disable access to parquet tables, so reads use NWB data only. |
    """)
    return


@app.cell(hide_code=True)
def _(dr_datacube, mo):
    mo.md(f"""
    **Data source for this notebook**

    - version: `{dr_datacube.config.version}`
    - public cache: `{dr_datacube.config.use_cache}`
    - anonymous access: `{dr_datacube.config.anon}`

    ---
    - `anon=True` allows data access libraries to operate without AWS S3 credentials.
    - `use_cache=True` directs data access libraries to the `aind-scratch-data` bucket, instead of the Code Ocean datacube asset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic query to demonstrate `polars.LazyFrame` syntax

    Apply `select`, `filter`, and `head` **before** `collect()`.

    `collect()` is the point at which remote data is actually read.
    """)
    return


@app.cell
def _(dr_datacube, pl):
    (
        dr_datacube.get_lf("performance")
        .filter(pl.col("n_contingent_rewards") >= 10)
        .select(
            "session_id",
            "block_index",
            "rewarded_modality",
            "cross_modality_dprime",
        )
        .head(10)
        .collect()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The call above relies on all current `get_lf()` defaults:

    | Option | Default | Effect |
    |---|---:|---|
    | `nwb` | `False` | Read from a parquet table when available (data is the same). |
    | `session_type` | `"brainwide"` | Other options are 'templeton', 'naive', None (all). Can be a list. |
    | `with_behavior_filter` | `True` | Keep sessions that pass the standard behavior criteria for their type. |
    | `only_in_data_asset` | `True` | Disable to access problematic sessions that are cached, but aren't in the datacube. |

    If you need to include other session types or bypass the behavior filter, you can adjust the `session_type` and `with_behavior_filter` parameters accordingly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    Here we demonstrate basic group-by and aggregation in `polars`, to interrogate the ratio of hits/misses in instruction trials:
    """)
    return


@app.cell
def _(dr_datacube, pl):
    trials = dr_datacube.get_lf("trials")
    (
        trials
        .filter(pl.col("is_instruction"))
        .group_by("trial_index_in_block")
        .agg(
            hits=pl.col("is_hit").sum(),
            misses=pl.col("is_miss").sum(),
            total=pl.len(),
        )
        .sort("trial_index_in_block")
        .collect()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    `epochs` is a table of contents for each session with:
    - start/stop times,
    - the name of the script that was run (see https://github.com/samgale/DynamicRoutingTask),
    - other "intervals" tables (i.e. contains `start_time` and `stop_time` columns) that correspond to each epoch
    """)
    return


@app.cell
def _(dr_datacube):
    epochs = dr_datacube.get_lf("epochs").collect()
    epochs
    return (epochs,)


@app.cell
def _(epochs, pl):
    # Check the duration of the Spontaneous epoch across sessions is close to 10 minutes:
    (
        epochs
        .filter(pl.col("script_name") == "Spontaneous")
        .select(
            "session_id",
            "start_time",
            "stop_time",
            duration=pl.col("stop_time") - pl.col("start_time"),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    Opt-in to other cohorts as needed:

    ```python
    # Multiple session types, still behavior-filtered:
    dr_datacube.get_lf(
        "epochs",
        session_type=["brainwide", "templeton"],
    )

    # Every session available in the cache (including those with issues and missing data):
    dr_datacube.get_lf(
        "epochs",
        session_type=None,
        with_behavior_filter=False,
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Unit metadata and spike times

    `units` is the full table, including spike times and other large array-like
    columns, most of which is unnecessary.

    To limit the amount of data read into memory:
    - filter on QC and area predicates,
    - select only the columns needed for the analysis,
    - supply a `session_id` where possible.

    > ℹ️ This is the primary usage for `get_lf()` (and `lazynwb`, which it uses). The schema for the NWB units table mixes small scalar values with large array-like columns, making it poorly-suited for "eager" loading from cloud storage.
    """)
    return


@app.cell
def _(dr_datacube, pl):
    (
        dr_datacube.get_lf("units", session_id='733891_2024-09-18')
        .filter(
            # multiple filters is an AND operation:
            "is_qc_pass", # specifying a boolean column by name implies "keep rows where this value is True"
            pl.col("structure").eq("MOs"), 
        )
        .select("session_id", "unit_id", "location", "ccf_ap", "ccf_ml", "spike_times") # `location` contains layers
        .collect()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `unit_metrics` is also exposed as a table intended for fast, metadata-only queries, without having to explicitly select/drop columns:

    >💡 **for users familiar with the cache: **this is the new name for the consolidated `units.parquet` file, to distinguish it from the full units table.
    """)
    return


@app.cell
def _(dr_datacube):
    (
        dr_datacube.get_lf("unit_metrics")
        .collect()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Temporary configuration

    Use `config.override()` when only one operation needs different access
    settings. Keep `collect()` inside the context.

    ```python
    with dr_datacube.config.override(use_cache=True, version='v0.0.272'):
        old_units = (
            dr_datacube.get_lf("units", session_id=example_session_id)
            .select("unit_id", "spike_times")
            .collect()
        )
    # The previous global configuration is restored here, even on error.
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
