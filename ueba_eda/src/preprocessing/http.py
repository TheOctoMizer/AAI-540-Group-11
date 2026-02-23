from pathlib import Path
from typing import Union

import polars as pl

from utils.common import parse_timestamp_multi


def preprocess_http(
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
            pl.col("url").str.extract(r"https?://([^/]+)", 1).alias("domain"),
            pl.col("url").str.len_chars().alias("url_length"),
            pl.col("url")
            .str.to_lowercase()
            .str.contains("login|auth|verify|secure")
            .cast(pl.Int8)
            .alias("suspicious_url_flag"),
        ]
    )

    lf.sink_parquet(output_path, compression="snappy")
