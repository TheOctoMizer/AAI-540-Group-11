from pathlib import Path
from typing import Union

import polars as pl

from config.schema import PreprocessConfig


def engineer_logon_features(
    input_path: Union[str, Path], config: PreprocessConfig | None
) -> pl.LazyFrame:

    input_path = Path(input_path)

    env = config.environment if config else None
    max_rows = env.max_rows if env else None

    # Polars scan
    lf = pl.scan_parquet(
        input_path, n_rows=max_rows if max_rows and max_rows > 0 else None
    )

    # Daily aggregation
    daily_features = lf.group_by(["user", "date"]).agg(
        [
            pl.len().alias("logon_count"),
            pl.col("pc").n_unique().alias("unique_pc_count"),
            pl.col("hour").min().alias("first_logon_hour"),
            pl.col("hour").max().alias("last_activity_hour"),
            pl.col("after_hours_flag").sum().alias("after_hours_logon_count"),
            pl.col("weekend_flag").max().alias("weekend_flag"),
        ]
    )

    # User baseline
    user_baseline = daily_features.group_by("user").agg(
        [
            pl.col("logon_count").mean().alias("mean_logon"),
            pl.col("logon_count").std().alias("std_logon"),
        ]
    )

    # Join baseline
    daily_features = daily_features.join(user_baseline, on="user", how="left")

    # --------------------------------
    # Z-score
    # --------------------------------
    daily_features = daily_features.with_columns(
        [
            pl.when(pl.col("std_logon") > 0)
            .then((pl.col("logon_count") - pl.col("mean_logon")) / pl.col("std_logon"))
            .otherwise(0)
            .alias("logon_count_zscore"),
            # Work duration
            (pl.col("last_activity_hour") - pl.col("first_logon_hour")).alias(
                "work_duration_hours"
            ),
        ]
    )

    final_daily_features = daily_features.select(
        [
            "user",
            "date",
            "logon_count",
            "unique_pc_count",
            "first_logon_hour",
            "last_activity_hour",
            "after_hours_logon_count",
            "weekend_flag",
            "logon_count_zscore",
            "work_duration_hours",
        ]
    )

    return final_daily_features
