import logging
import shutil
from pathlib import Path
from time import time

from src.preprocessing.decoy_file import preprocess_decoy_file
from src.preprocessing.device import preprocess_device
from src.preprocessing.email import preprocess_email
from src.preprocessing.file import preprocess_file
from src.preprocessing.http import preprocess_http
from src.preprocessing.logon import preprocess_logon

logger = logging.getLogger(__name__)

PREPROCESS_MAP = {
    "http": preprocess_http,
    "email": preprocess_email,
    "file": preprocess_file,
    "logon": preprocess_logon,
    "device": preprocess_device,
    "decoy_file": preprocess_decoy_file,
}


def run_preprocessing(config):

    logger.info("Starting preprocessing pipeline")

    input_root = Path(config.paths.bronze_data_dir)
    output_root = Path(config.paths.silver_data_dir)

    overwrite_flag = config.general.overwrite_preprocessing
    time_start = time()
    for name, preprocess_func in PREPROCESS_MAP.items():
        logger.info("Processing: %s", name)

        input_path = input_root / f"{name}.parquet"
        output_path = output_root / f"{name}_preprocessed.parquet"

        # Overwrite logic
        if overwrite_flag and output_path.exists():
            logger.info("Overwriting: %s", output_path)

            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run preprocessing
        preprocess_func(
            input_path=str(input_path), output_path=str(output_path), config=config
        )

        logger.info("Completed preprocessing for: %s", name)
    time_end = time()

    elapsed = time_end - time_start

    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(
        "Preprocessing pipeline time taken: %02d:%02d:%02d", hours, minutes, seconds
    )
    logger.info("Preprocessing pipeline completed successfully")
