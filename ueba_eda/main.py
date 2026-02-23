from datetime import datetime
from pathlib import Path

from pipelines.run_feature_engineering import run_feature_engineering
from pipelines.run_ingestion import run_ingestion
from pipelines.run_preprocessing import run_preprocessing
from utils.config_util import load_preprocess_config
from utils.log_util import setup_logger

config = load_preprocess_config(Path("config/config.yaml"))

now = datetime.now()
ms = now.strftime("%f")[:-3]
filename = f"ueba_{now.strftime('%Y_%m_%d_%H_%M_%S')}_{ms}.log"

logger = setup_logger(log_file=f"logs/{filename}")


def main():
    logger.info("Project UEBA - User and Entity Behavior Analytics")

    if config.environment.max_rows is not None and config.environment.max_rows > 0:
        logger.info("Row limit enabled | max_rows=%d", config.environment.max_rows)

    logger.info("Pipeline mode=%s", config.environment.mode)
    if config.pipeline.run_ingestion:
        run_ingestion(config)
    if config.pipeline.run_preprocessing:
        run_preprocessing(config)
    if config.pipeline.run_feature_engineering:
        run_feature_engineering(config)


if __name__ == "__main__":
    main()
