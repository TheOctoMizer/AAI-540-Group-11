# src/feature_engineering/file.py

from pathlib import Path
from typing import Union

import polars as pl

from config.schema import PreprocessConfig


def engineer_file_features(
    input_path: Union[str, Path], config: PreprocessConfig | None
) -> pl.LazyFrame:

    input_path = Path(input_path)
    decoy_file_path = input_path.parent / "decoy_file_preprocessed.parquet"

    env = config.environment if config else None
    max_rows = env.max_rows if env else None

    # Polars scan
    lf = pl.scan_parquet(
        input_path, n_rows=max_rows if max_rows and max_rows > 0 else None
    )

    if decoy_file_path.exists():
        if max_rows is not None and max_rows > 0:  # Use the local variable here too!
            decoy_df = pl.scan_parquet(decoy_file_path, n_rows=max_rows)
        else:
            decoy_df = pl.scan_parquet(decoy_file_path)
    else:
        raise FileNotFoundError(f"Decoy file not found at {decoy_file_path}...")

    decoy_df = decoy_df.with_columns([pl.lit(1).alias("decoy_access_flag")])

    # --------------------------------
    # Join with decoy table
    # --------------------------------
    lf = lf.join(decoy_df.lazy(), on=["filename", "pc"], how="left").with_columns(
        pl.col("decoy_access_flag").fill_null(0).cast(pl.Int8)
    )

    # --------------------------------
    # Daily Aggregation
    # --------------------------------
    daily_features = lf.group_by(["user", "date"]).agg(
        [
            # Activity counts
            pl.sum("copy_event").alias("file_copy_count"),
            pl.sum("write_event").alias("file_write_count"),
            pl.sum("delete_event").alias("file_delete_count"),
            # USB
            pl.sum("to_usb_event").alias("usb_copy_count"),
            pl.sum("from_usb_event").alias("usb_read_count"),
            # Risk behavior
            pl.sum("after_hours_flag").alias("after_hours_file_count"),
            # Volume
            pl.col("filename").approx_n_unique().alias("unique_file_count"),
            # Content complexity
            pl.mean("content_keyword_count").alias("avg_keyword_count"),
            pl.sum("decoy_access_flag").alias("decoy_access_count"),
        ]
    )

    # --------------------------------
    # User Baseline
    # --------------------------------
    user_baseline = daily_features.group_by("user").agg(
        [
            pl.col("file_copy_count").mean().alias("mean_copy"),
            pl.col("file_copy_count").std().alias("std_copy"),
        ]
    )

    # Join baseline
    daily_features = daily_features.join(user_baseline, on="user", how="left")

    daily_features = daily_features.with_columns(
        [
            pl.when((pl.col("std_copy") == 0) | (pl.col("std_copy").is_null()))
            .then(0)
            .otherwise(
                (pl.col("file_copy_count") - pl.col("mean_copy")) / pl.col("std_copy")
            )
            .alias("file_copy_zscore")
        ]
    )

    final_daily_features = daily_features.select(
        [
            "user",
            "date",
            "file_copy_count",
            "file_write_count",
            "file_delete_count",
            "usb_copy_count",
            "usb_read_count",
            "after_hours_file_count",
            "unique_file_count",
            "avg_keyword_count",
            "decoy_access_count",
            "file_copy_zscore",
        ]
    )

    return final_daily_features
