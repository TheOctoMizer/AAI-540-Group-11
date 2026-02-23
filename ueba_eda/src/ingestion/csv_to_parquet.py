import logging
import shutil
from pathlib import Path
from typing import Literal, TypeAlias, Union, cast

import polars as pl

from config.schema import PreprocessConfig

logger = logging.getLogger(__name__)


def csv_to_parquet(
    config: PreprocessConfig | None,
    filename: Union[str, Path],
    partition_enabled: bool = True,
    partition_col: Union[str, None] = "timestamp",
    partition_condition: Union[str, None] = "day",
) -> None:
    """
    Ingest a CERT CSV file into Parquet using Polars.

    Partitioning behavior is controlled via function arguments
    (not config), to avoid over-engineering.
    """
    CompressionType: TypeAlias = Literal[
        "lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"
    ]
    # =========================
    # File & PATH RESOLUTION
    # =========================

    if config is None:
        raise ValueError("Config cannot be None for ingestion.")

    paths = config.paths
    general = config.general
    runtime = config.runtime
    env = config.environment
    pq = config.parquet

    filename = Path(filename)
    input_path = Path(paths.raw_data_dir) / filename
    output_path = Path(paths.bronze_data_dir) / f"{Path(filename).stem}.parquet"

    output_path = Path(output_path)

    if general.overwrite_datasource and output_path.exists():
        if output_path.is_dir():
            logger.info("Deleting existing directory: %s", output_path)
            shutil.rmtree(output_path)

        elif output_path.is_file():
            logger.info("Deleting existing file: %s", output_path)
            output_path.unlink()

    if partition_enabled:
        output_path.mkdir(parents=True, exist_ok=True)

    partition_condition = (partition_condition or "day").casefold()
    # Skip if file exists and overwrite is disabled
    if not general.overwrite_datasource and output_path.exists():
        logger.info(
            "file=%s already exists and overwrite=False. Skipping ingestion.",
            filename,
        )
        return
    # =========================
    # INGESTION PIPELINE START
    # =========================
    logger.info(
        "ingestion started | file=%s | input=%s",
        filename,
        input_path,
    )

    logger.info(
        "output=%s | chunk_size=%d",
        output_path,
        runtime.chunk_size,
    )

    if partition_enabled:
        logger.info(
            "file=%s | partitioning=enabled | partition-on=%s | partition-condition=%s",
            filename,
            partition_col,
            partition_condition,
        )
    else:
        logger.info(
            "file=%s | partitioning=disabled",
            filename,
        )

    # =========================
    # LAZY CSV SCAN
    # =========================
    lf = pl.scan_csv(
        input_path,
        has_header=True,
        infer_schema_length=1_000,
        try_parse_dates=False,
    )

    logger.debug(
        "file=%s | inferred-schema=%s",
        filename,
        lf.collect_schema(),
    )

    rows_processed = 0
    max_rows = env.max_rows
    # =========================
    # BATCH INGESTION
    # =========================

    for batch_id, df in enumerate(
        lf.collect_batches(
            chunk_size=runtime.chunk_size,
            maintain_order=False,
        )
    ):
        # Enforce max_rows
        if max_rows is not None and max_rows > 0:
            remaining = max_rows - rows_processed

            if remaining <= 0:
                logger.info("Reached max_rows=%d. Stopping ingestion.", max_rows)
                break

            if df.height > remaining:
                df = df.head(remaining)

        rows_processed += df.height

        logger.info(
            "file=%s | batch=%d | rows=%d | total=%d",
            filename,
            batch_id,
            df.height,
            rows_processed,
        )

        formats = (
            general.timestamp_formats
            if general.timestamp_formats
            else ["%Y-%m-%d %H:%M:%S"]
        )
        if "date" in df.columns:
            parsed_ts = pl.coalesce(
                [
                    pl.col("date").str.strptime(pl.Datetime, fmt, strict=False)
                    for fmt in formats
                ]
            )

            df = (
                df.with_columns(parsed_ts.alias("timestamp")).drop(
                    "date"
                )  # <-- remove original column
            )

        # =========================
        # PARTITION HANDLING
        # =========================
        if partition_enabled:
            if partition_col not in df.columns:
                logger.error(
                    "file=%s | missing partition column '%s'",
                    filename,
                    partition_col,
                )
                raise KeyError(
                    f"Partition column '{partition_col}' not found in {filename}"
                )

            try:
                partition_condition = partition_condition.casefold()

                if partition_condition == "year":
                    df = df.with_columns(pl.col("timestamp").dt.year().alias("year"))
                    partition_keys = ["year"]

                elif partition_condition == "month":
                    df = df.with_columns(
                        [
                            pl.col("timestamp").dt.year().alias("year"),
                            pl.col("timestamp").dt.month().alias("month"),
                        ]
                    )
                    partition_keys = ["year", "month"]

                elif partition_condition == "day":
                    df = df.with_columns(
                        [
                            pl.col("timestamp").dt.year().alias("year"),
                            pl.col("timestamp").dt.month().alias("month"),
                            pl.col("timestamp").dt.day().alias("day"),
                        ]
                    )
                    partition_keys = ["year", "month", "day"]

                else:
                    logger.warning(
                        "file=%s | unrecognized partition condition '%s' - partitioning skipped",
                        filename,
                        partition_condition,
                    )
                    partition_keys = None

                if partition_keys:
                    logger.info(
                        "file=%s | partition-condition=%s | partition keys=%s",
                        filename,
                        partition_condition,
                        partition_keys,
                    )

            except Exception:
                logger.exception(
                    "file=%s | failed parsing partition column '%s'",
                    filename,
                    partition_col,
                )
                raise

            # create output dir if it doesn't exist only for partitioned writes
            # (non-partitioned writes will create the file directly)
            # output_path.mkdir(parents=True, exist_ok=True)

            df.write_parquet(
                output_path,
                use_pyarrow=True,
                pyarrow_options={"partition_cols": partition_keys},
                compression=cast(CompressionType, pq.compression),
                compression_level=pq.compression_level,
                statistics=pq.statistics,
            )

        else:
            df.write_parquet(
                output_path,
                use_pyarrow=True,
                compression=cast(CompressionType, pq.compression),
                compression_level=pq.compression_level,
                statistics=pq.statistics,
            )

    # =========================
    # PIPELINE END
    # =========================
    logger.info(
        "ingestion completed for file=%s | total_rows=%d",
        filename,
        rows_processed,
    )
