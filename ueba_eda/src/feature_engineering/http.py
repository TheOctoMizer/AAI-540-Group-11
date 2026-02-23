import logging
import shutil
from pathlib import Path
from typing import Union

import polars as pl

from config.schema import PreprocessConfig

logger = logging.getLogger(__name__)


def engineer_http_features(
    input_path: Union[str, Path],
    cache_dir: Union[str, Path],
    config: PreprocessConfig | None,
) -> pl.LazyFrame:

    input_path = Path(input_path)
    cache_dir = Path(cache_dir)

    env = config.environment if config else None
    max_rows = env.max_rows if env else None

    # Polars scan
    lf = pl.scan_parquet(
        input_path, n_rows=max_rows if max_rows and max_rows > 0 else None
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(cache_dir.glob("user_daily_agg_*.parquet"))

    # --------------------------------------------------
    # Build daily aggregation per partition
    # --------------------------------------------------
    overwrite = (
        config.general.overwrite_features if (config and config.general) else False
    )

    if overwrite or not existing_files:
        if overwrite and cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Computing HTTP daily aggregation per partition...")

        # Get unique partitions (small metadata collect only)
        dates = lf.select(["year", "month", "day"]).unique().collect().to_dicts()

        # Process per partition (true pruning)
        for row in dates:
            y, m, d = row["year"], row["month"], row["day"]

            logger.info(f"Processing HTTP partition={y}-{m:02d}-{d:02d}")

            daily_chunk = (
                lf.filter(
                    (pl.col("year") == y)
                    & (pl.col("month") == m)
                    & (pl.col("day") == d)
                )
                .with_columns(
                    [
                        ((pl.col("hour") < 8) | (pl.col("hour") > 18))
                        .cast(pl.Int8)
                        .alias("after_hours_flag"),
                        (pl.col("weekday") >= 5).cast(pl.Int8).alias("weekend_flag"),
                    ]
                )
                .group_by(["user", "date"])
                .agg(
                    [
                        pl.len().alias("http_request_count"),
                        pl.col("url").approx_n_unique().alias("unique_url_count"),
                        pl.col("domain").approx_n_unique().alias("unique_domain_count"),
                        pl.sum("suspicious_url_flag").alias("suspicious_url_count"),
                        pl.sum("after_hours_flag").alias("after_hours_http_count"),
                        pl.max("weekend_flag").alias("weekend_http_flag"),
                        pl.mean("url_length").alias("mean_url_length"),
                    ]
                )
            )

            daily_chunk.sink_parquet(
                cache_dir / f"user_daily_agg_{y}_{m:02d}_{d:02d}.parquet",
                compression="snappy",
            )

    # --------------------------------------------------
    # Load all daily aggregated partitions lazily
    # --------------------------------------------------

    daily_features = pl.scan_parquet(cache_dir / "user_daily_agg_*.parquet")

    logger.info("Computing HTTP user baseline...")

    user_baseline = daily_features.group_by("user").agg(
        [
            pl.col("http_request_count").mean().alias("mean_http"),
            pl.col("http_request_count").std().alias("std_http"),
            pl.col("unique_domain_count").mean().alias("mean_domain"),
            pl.col("unique_domain_count").std().alias("std_domain"),
        ]
    )

    logger.info("Computing HTTP z-scores...")
    # Join baseline
    daily_features = daily_features.join(user_baseline, on="user", how="left")

    daily_features = daily_features.with_columns(
        [
            pl.when((pl.col("std_http") == 0) | (pl.col("std_http").is_null()))
            .then(0)
            .otherwise(
                (pl.col("http_request_count") - pl.col("mean_http"))
                / pl.col("std_http")
            )
            .alias("http_count_zscore"),
            pl.when((pl.col("std_domain") == 0) | (pl.col("std_domain").is_null()))
            .then(0)
            .otherwise(
                (pl.col("unique_domain_count") - pl.col("mean_domain"))
                / pl.col("std_domain")
            )
            .alias("domain_count_zscore"),
        ]
    )

    logger.info("HTTP feature engineering complete (lazy).")

    final_daily_features = daily_features.select(
        [
            "user",
            "date",
            "http_request_count",
            "unique_url_count",
            "unique_domain_count",
            "suspicious_url_count",
            "after_hours_http_count",
            "weekend_http_flag",
            "mean_url_length",
            "http_count_zscore",
            "domain_count_zscore",
        ]
    )
    return final_daily_features
