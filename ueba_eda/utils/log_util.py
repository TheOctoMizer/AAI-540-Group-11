import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.config_util import load_logger_config


def setup_logger(
    log_file: str | Path = "logs/ueba.log",
) -> logging.Logger:

    try:
        config = load_logger_config(Path("config/logger.yaml"))
    except Exception as e:
        print(f"Error loading logger config: {e}")
        raise
    try:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {e}")
        raise

    logger = logging.getLogger(None)  # get root logger
    logger.setLevel(getattr(logging, config.level.upper()))

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(fmt=config.format, datefmt=config.timestamp_format)

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Rotating file
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False  # keep logs from propagating to root if you want

    return logger  # return the logger so you can use it


if __name__ == "__main__":
    logger = setup_logger(log_file="logs/ueba.log")
    logger.info("Logger has been set up successfully!")
