# src/preprocessing/decoy.py

from pathlib import Path
from typing import Union

import polars as pl


def preprocess_decoy_file(
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

    lf = (
        lf.select(["decoy_filename", "pc"])
        .rename({"decoy_filename": "filename"})
        .with_columns(
            [
                pl.col("pc").str.to_lowercase().alias("pc"),
            ]
        )
        .unique()  # remove accidental duplicates
    )

    lf.sink_parquet(output_path, compression="snappy")
