from pathlib import Path
from typing import Union

import yaml

from config.schema import (
    EnvironmentConfig,
    GeneralConfig,
    LoggerConfig,
    ParquetConfig,
    PathsConfig,
    PipelineConfig,
    PreprocessConfig,
    RuntimeConfig,
)


def load_preprocess_config(path: Union[str, Path] = "") -> PreprocessConfig:
    path = Path(path)

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    return PreprocessConfig(
        paths=PathsConfig(**raw["paths"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        environment=EnvironmentConfig(**raw["environment"]),
        pipeline=PipelineConfig(**raw["pipeline"]),
        general=GeneralConfig(**raw["general"]),
        parquet=ParquetConfig(**raw["parquet"]),
    )


def load_logger_config(path: Union[str, Path] = "") -> LoggerConfig:

    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    return LoggerConfig(**raw["logger"])


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    preprocess_cfg = load_preprocess_config(
        path=PROJECT_ROOT / "config" / "config.yaml"
    )
    print(preprocess_cfg)

    logger_cfg = load_logger_config(path=PROJECT_ROOT / "config" / "logger.yaml")
    print(logger_cfg)
