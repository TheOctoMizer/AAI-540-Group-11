import logging
from pathlib import Path
from time import time
from typing import Optional, TypedDict, Union, cast

from src.ingestion.csv_to_parquet import csv_to_parquet

logger = logging.getLogger(__name__)


class DatasetDict(TypedDict):
    filename: str
    partition_enabled: bool
    partition_col: Union[str, None]
    partition_condition: Union[str, None]


def run_ingestion(config) -> None:
    """
    Orchestrates CSV → Parquet ingestion for all CERT datasets.
    """

    logger.info("Starting ingestion pipeline")

    datasets: list[DatasetDict] = [
        {
            "filename": "logon.csv",
            "partition_enabled": True,
            "partition_col": "timestamp",
            "partition_condition": "day",
        },
        {
            "filename": "device.csv",
            "partition_enabled": True,
            "partition_col": "timestamp",
            "partition_condition": "day",
        },
        {
            "filename": "http.csv",
            "partition_enabled": True,
            "partition_col": "timestamp",
            "partition_condition": "day",
        },
        {
            "filename": "file.csv",
            "partition_enabled": True,
            "partition_col": "timestamp",
            "partition_condition": "day",
        },
        {
            "filename": "email.csv",
            "partition_enabled": True,
            "partition_col": "timestamp",
            "partition_condition": "day",
        },
        {
            # static lookup – no partitioning
            "filename": "decoy_file.csv",
            "partition_enabled": False,
            "partition_col": None,
            "partition_condition": None,
        },
    ]

    for ds in datasets:
        filename = ds["filename"]

        logger.info("Starting ingestion for file=%s", filename)

        try:
            time_start = time()
            csv_to_parquet(
                config=config,
                filename=ds["filename"],
                partition_enabled=ds["partition_enabled"],
                partition_col=ds["partition_col"],
                partition_condition=ds["partition_condition"],
            )

            logger.info("Completed ingestion for file=%s", filename)
            time_end = time()

            elapsed = time_end - time_start

            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)

            logger.info(
                "Ingestion time for file=%s: %02dh:%02dm:%05.2fs",
                filename,
                int(hours),
                int(minutes),
                seconds,
            )

        except Exception:
            logger.exception("Ingestion failed for file=%s", filename)
            raise  # fail fast – data pipelines must not continue silently

    logger.info("All ingestion pipelines completed successfully")


if __name__ == "__main__":
    from utils.config_util import load_preprocess_config
    from utils.log_util import setup_logger

    config = load_preprocess_config(Path("config/config.yaml"))

    # Set up logger
    logger = setup_logger()

    logger.info("Starting ingestion pipeline")
    run_ingestion(config)
