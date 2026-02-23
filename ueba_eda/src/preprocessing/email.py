# src/preprocessing/email.py

from pathlib import Path
from typing import Union

import polars as pl

from utils.common import parse_timestamp_multi


def preprocess_email(
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
            (pl.col("activity") == "Send").cast(pl.Int8).alias("send_flag"),
            ((pl.col("hour") < 8) | (pl.col("hour") >= 18))
            .cast(pl.Int8)
            .alias("after_hours_flag"),
            pl.when(pl.col("attachments").is_not_null())
            .then(pl.col("attachments").str.count_matches(";") + 1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("attachment_count"),
        ]
    )

    # recipient flags
    lf = lf.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("to").fill_null(""),
                    pl.col("cc").fill_null(""),
                    pl.col("bcc").fill_null(""),
                ],
                separator=";",
            ).alias("all_recipients")
        ]
    )

    lf = lf.with_columns(
        [pl.col("all_recipients").str.split(";").alias("recipient_list")]
    )

    lf = lf.with_columns(
        [
            pl.col("recipient_list").list.len().alias("all_recipient_count"),
            pl.col("recipient_list")
            .list.eval(pl.element().str.contains("@dtaa.com").not_())
            .list.sum()
            .cast(pl.Int32)
            .alias("external_recipient_count"),
        ]
    )

    lf = lf.with_columns(
        [
            (pl.col("external_recipient_count") > 0)
            .cast(pl.Int8)
            .alias("has_external_recipient")
        ]
    )
    lf.sink_parquet(output_path, compression="snappy")
