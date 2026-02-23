# src/preprocessing/logon.py

from pathlib import Path
from typing import Union

import polars as pl

from utils.common import parse_timestamp_multi


def preprocess_logon(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    env = config.environment if config else None
    max_rows = env.max_rows if env else None

    # Polars scan
    lf = pl.scan_parquet(
        input_path, n_rows=max_rows if max_rows and max_rows > 0 else None
    )

    # Parse timestamps
    lf = parse_timestamp_multi(
        lf,
        column="timestamp",
        config=config.general if config else None,
    )

    lf = lf.with_columns(
        [
            # Normalize user
            pl.col("user").str.to_lowercase().alias("user"),
            # Normalize PC name
            pl.col("pc").str.to_lowercase().alias("pc"),
            # Basic after-hours flag (row-level)
            ((pl.col("hour") < 8) | (pl.col("hour") > 18))
            .cast(pl.Int8)
            .alias("after_hours_flag"),
            # Weekend indicator (row-level)
            (pl.col("weekday") >= 5).cast(pl.Int8).alias("weekend_flag"),
        ]
    )

    lf.sink_parquet(output_path, compression="snappy")
