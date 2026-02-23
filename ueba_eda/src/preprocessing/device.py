# src/preprocessing/device.py

from pathlib import Path
from typing import Union

import polars as pl

from utils.common import parse_timestamp_multi


def preprocess_device(
    input_path: Union[str, Path], output_path: Union[str, Path], config: None
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

    # Normalize and create row-level flags
    lf = lf.with_columns(
        [
            # Normalize user
            pl.col("user").str.to_lowercase().alias("user"),
            # Normalize activity
            pl.col("activity").str.to_lowercase().alias("activity"),
            # Row-level connect flag
            (pl.col("activity") == "connect").cast(pl.Int8).alias("usb_connect_event"),
            # Row-level disconnect flag
            (pl.col("activity") == "disconnect")
            .cast(pl.Int8)
            .alias("usb_disconnect_event"),
        ]
    )

    # Write silver
    lf.sink_parquet(output_path, compression="snappy")
