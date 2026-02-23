from pathlib import Path
from typing import Union

import polars as pl

from config.schema import PreprocessConfig


def engineer_device_features(
    input_path: Union[str, Path], config: PreprocessConfig | None
) -> pl.LazyFrame:

    input_path = Path(input_path)

    env = config.environment if config else None
    max_rows = env.max_rows if env else None

    # Polars scan
    lf = pl.scan_parquet(
        input_path, n_rows=max_rows if max_rows and max_rows > 0 else None
    )

    # --------------------------------
    # Daily Aggregation
    # --------------------------------
    daily_features = (
        lf.group_by(["user", "date"])
        .agg(
            [
                pl.col("usb_connect_event").sum().alias("usb_connect_count"),
                pl.col("usb_disconnect_event").sum().alias("usb_disconnect_count"),
            ]
        )
        .with_columns(
            (pl.col("usb_connect_count") != pl.col("usb_disconnect_count"))
            .cast(pl.Int8)
            .alias("incomplete_session_flag")
        )
    )

    # --------------------------------
    # User Baseline
    # --------------------------------
    user_baseline = daily_features.group_by("user").agg(
        [
            pl.col("usb_connect_count").mean().alias("mean_usb"),
            pl.col("usb_connect_count").std().alias("std_usb"),
        ]
    )

    # Join baseline
    daily_features = daily_features.join(user_baseline, on="user", how="left")

    # --------------------------------
    # Z-score
    # --------------------------------
    daily_features = daily_features.with_columns(
        [
            pl.when(pl.col("std_usb") > 0)
            .then(
                (pl.col("usb_connect_count") - pl.col("mean_usb")) / pl.col("std_usb")
            )
            .otherwise(0)
            .alias("usb_connect_zscore")
        ]
    )

    final_daily_features = daily_features.select(
        [
            "user",
            "date",
            "usb_connect_count",
            "usb_disconnect_count",
            "incomplete_session_flag",
            "usb_connect_zscore",
        ]
    )
    return final_daily_features
