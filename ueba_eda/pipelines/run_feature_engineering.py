import logging
import shutil
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, TypedDict

import polars as pl

from src.feature_engineering.device import engineer_device_features
from src.feature_engineering.email import engineer_email_features
from src.feature_engineering.file import engineer_file_features
from src.feature_engineering.http import engineer_http_features
from src.feature_engineering.logon import engineer_logon_features


class FeatureJob(TypedDict):
    name: str
    func: Callable[
        ..., pl.LazyFrame
    ]  # Tells Mypy this is a function returning a LazyFrame
    args: Dict[str, Any]
    output: Path


logger = logging.getLogger(__name__)


def run_feature_engineering(config) -> None:
    logger.info("Starting feature engineering pipeline")

    try:
        time_start = time()
        input_root = Path(config.paths.silver_data_dir)
        output_root = Path(config.paths.gold_data_dir)
        cache_dir = Path(config.paths.gold_data_dir) / "features" / "cache"
        overwrite_flag = config.general.overwrite_features

        feature_jobs: list[FeatureJob] = [
            {
                "name": "device",
                "func": engineer_device_features,
                "args": {
                    "input_path": input_root / "device_preprocessed.parquet",
                    "config": config,
                },
                "output": output_root / "features" / "device_features.parquet",
            },
            {
                "name": "http",
                "func": engineer_http_features,
                "args": {
                    "input_path": input_root / "http_preprocessed.parquet",
                    "cache_dir": cache_dir,
                    "config": config,
                },
                "output": output_root / "features" / "http_features.parquet",
            },
            {
                "name": "logon",
                "func": engineer_logon_features,
                "args": {
                    "input_path": input_root / "logon_preprocessed.parquet",
                    "config": config,
                },
                "output": output_root / "features" / "logon_features.parquet",
            },
            {
                "name": "email",
                "func": engineer_email_features,
                "args": {
                    "input_path": input_root / "email_preprocessed.parquet",
                    "config": config,
                },
                "output": output_root / "features" / "email_features.parquet",
            },
            {
                "name": "file",
                "func": engineer_file_features,
                "args": {
                    "input_path": input_root / "file_preprocessed.parquet",
                    "config": config,
                },
                "output": output_root / "features" / "file_features.parquet",
            },
        ]
        for job in feature_jobs:
            logger.info("Starting Feature Engineering on: %s", job["name"])

            output_path = Path(job["output"])

            # Overwrite logic
            if overwrite_flag and output_path.exists():
                logger.info("Overwriting: %s", output_path)

                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Run feature engineering with unpacked arguments
            features = job["func"](**job["args"])

            # Save
            features.sink_parquet(output_path, compression="snappy")

            # Schema + Sample
            schema = features.collect_schema()

            sample = (
                pl.scan_parquet(output_path)
                .limit(5)
                .collect()
                .glimpse(return_type="string")
            )

            logger.info("Completed: %s", job["name"])
            logger.info("Saved to: %s", output_path)
            logger.info("Schema: %s", schema)
            logger.info("Sample:\n%s", sample)
        time_end = time()

        elapsed = time_end - time_start

        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        logger.info(
            "Feature engineering time taken: %02dh:%02dm:%05.2fs",
            int(hours),
            int(minutes),
            seconds,
        )

    except Exception:
        logger.exception("Feature engineering failed")
        raise  # fail fast – data pipelines must not continue silently

    logger.info("Feature Engineering pipeline completed successfully")


if __name__ == "__main__":
    from utils.config_util import load_preprocess_config
    from utils.log_util import setup_logger

    config = load_preprocess_config(Path("config/config.yaml"))

    # Set up logger
    logger = setup_logger()

    logger.info("Starting feature engineering pipeline")
    run_feature_engineering(config)
