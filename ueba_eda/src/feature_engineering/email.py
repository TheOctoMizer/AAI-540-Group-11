# src/feature_engineering/email.py

from pathlib import Path
from typing import Union

import polars as pl

from config.schema import PreprocessConfig


def engineer_email_features(
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
    daily_features = lf.group_by(["user", "date"]).agg(
        [
            # Volume
            pl.count().alias("email_count"),
            pl.col("send_flag").sum().alias("send_count"),
            # Time behavior
            pl.col("after_hours_flag").sum().alias("after_hours_email_count"),
            # Recipients
            pl.col("all_recipient_count").sum().alias("total_recipients"),
            pl.col("external_recipient_count").sum().alias("total_external_recipients"),
            pl.col("has_external_recipient").sum().alias("emails_with_external"),
            # Attachments
            pl.col("attachment_count").sum().alias("total_attachments"),
        ]
    )
    daily_features = daily_features.with_columns(
        [
            (pl.col("total_external_recipients") / pl.col("total_recipients"))
            .fill_null(0)
            .alias("external_recipient_ratio"),
            (pl.col("emails_with_external") / pl.col("email_count"))
            .fill_null(0)
            .alias("external_email_ratio"),
            (pl.col("total_attachments") / pl.col("email_count"))
            .fill_null(0)
            .alias("attachment_per_email"),
        ]
    )

    # --------------------------------
    # User Baseline
    # --------------------------------
    user_baseline = daily_features.group_by("user").agg(
        [
            pl.col("email_count").mean().alias("mean_email_count"),
            pl.col("external_email_ratio").mean().alias("mean_external_ratio"),
        ]
    )

    # --------------------------------
    # Z-score
    # --------------------------------
    daily_features = daily_features.join(user_baseline, on="user", how="left")
    daily_features = daily_features.with_columns(
        [
            (pl.col("email_count") - pl.col("mean_email_count")).alias(
                "email_volume_deviation"
            ),
            (pl.col("external_email_ratio") - pl.col("mean_external_ratio")).alias(
                "external_ratio_deviation"
            ),
        ]
    )

    final_daily_features = daily_features.select(
        [
            "user",
            "date",
            "email_count",
            "send_count",
            "after_hours_email_count",
            "external_email_ratio",
            "attachment_per_email",
            "email_volume_deviation",
            "external_ratio_deviation",
        ]
    )
    return final_daily_features
