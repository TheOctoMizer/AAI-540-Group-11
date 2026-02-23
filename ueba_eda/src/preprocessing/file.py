# src/preprocessing/file.py

from pathlib import Path
from typing import Union

import polars as pl

from utils.common import parse_timestamp_multi


def preprocess_file(
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

    lf = lf.with_columns(
        [
            pl.col("activity").str.to_lowercase().alias("activity"),
            (pl.col("activity") == "copy").cast(pl.Int8).alias("copy_event"),
            (pl.col("activity") == "write").cast(pl.Int8).alias("write_event"),
            (pl.col("activity") == "delete").cast(pl.Int8).alias("delete_event"),
            ((pl.col("hour") < 8) | (pl.col("hour") > 18))
            .cast(pl.Int8)
            .alias("after_hours_flag"),
            pl.col("to_removable_media")
            .fill_null(False)
            .cast(pl.Int8)
            .alias("to_usb_event"),
            pl.col("from_removable_media")
            .fill_null(False)
            .cast(pl.Int8)
            .alias("from_usb_event"),
            (pl.col("content").str.count_matches(" ") + 1).alias(
                "content_keyword_count"
            ),
        ]
    )

    lf.sink_parquet(output_path, compression="snappy")
