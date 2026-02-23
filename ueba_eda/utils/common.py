import polars as pl  # type: ignore

from config.schema import GeneralConfig


def parse_timestamp_multi(
    lf: pl.LazyFrame, column: str, config: None | GeneralConfig
) -> pl.LazyFrame:

    schema = lf.collect_schema()

    # If already datetime, we still want to extract date/hour/weekday
    if schema[column] in (pl.Datetime, pl.Date):
        ts_column = pl.col(column)
    else:
        formats = (
            config.timestamp_formats
            if config and config.timestamp_formats
            else ["%Y-%m-%d %H:%M:%S"]
        )

        ts_column = pl.coalesce(
            [pl.col(column).str.to_datetime(fmt, strict=False) for fmt in formats]
        )

    # Use .alias() to keep the code clean and then extract features
    return (
        lf.with_columns(ts_column.alias("temp_ts"))
        .with_columns(
            [
                pl.col("temp_ts").dt.date().alias("date"),
                pl.col("temp_ts").dt.hour().alias("hour"),
                pl.col("temp_ts").dt.weekday().alias("weekday"),
            ]
        )
        .drop(["temp_ts", column])
    )
