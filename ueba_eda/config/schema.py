from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ============================================================
# PATH CONFIG
# ============================================================
@dataclass(frozen=True)
class PathsConfig:
    raw_data_dir: Path
    bronze_data_dir: Path
    silver_data_dir: Path
    gold_data_dir: Path


# ============================================================
# RUNTIME CONFIG
# ============================================================
@dataclass(frozen=True)
class RuntimeConfig:
    chunk_size: int = 500_000


# ============================================================
# PIPELINE CONFIG
# ============================================================
@dataclass(frozen=True)
class PipelineConfig:
    run_ingestion: bool = False
    run_preprocessing: bool = False
    run_feature_engineering: bool = False


# ============================================================
# GENERAL CONFIG
# ============================================================
@dataclass(frozen=True)
class GeneralConfig:
    timestamp_formats: Optional[list[str]] = None
    overwrite_datasource: bool = False
    overwrite_preprocessing: bool = False
    overwrite_features: bool = False


# ============================================================
# PARQUET CONFIG
# ============================================================
@dataclass(frozen=True)
class ParquetConfig:
    compression: str = "zstd"
    compression_level: int = 3
    statistics: bool = True


# ============================================================
# LOGGER CONFIG
# ============================================================
@dataclass(frozen=True)
class LoggerConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-7s | %(message)s"
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"


# ============================================================
# ENVIRONMENT CONFIG
# ============================================================
@dataclass(frozen=True)
class EnvironmentConfig:
    mode: str = "prod"  # options: test | prod
    max_rows: int | None = (
        None  # for testing, limit number of ingested rows (overrides chunk_size)
    )


# ============================================================
# ROOT CONFIGS
# ============================================================
@dataclass(frozen=True)
class PreprocessConfig:
    paths: PathsConfig
    runtime: RuntimeConfig
    environment: EnvironmentConfig
    pipeline: PipelineConfig
    general: GeneralConfig
    parquet: ParquetConfig


if __name__ == "__main__":
    # Example usage
    preprocess_cfg = PreprocessConfig(
        paths=PathsConfig(
            raw_data_dir=Path("/path/to/input.csv"),
            bronze_data_dir=Path("/path/to/bronze/"),
            silver_data_dir=Path("/path/to/silver/"),
            gold_data_dir=Path("/path/to/gold/"),
        ),
        environment=EnvironmentConfig(mode="test", max_rows=10),
        runtime=RuntimeConfig(chunk_size=300_000),
        pipeline=PipelineConfig(run_ingestion=False, run_feature_engineering=True),
        general=GeneralConfig(
            timestamp_formats=["%m/%d/%Y %H:%M:%S"],
            overwrite_datasource=True,
            overwrite_preprocessing=True,
            overwrite_features=True,
        ),
        parquet=ParquetConfig(
            compression="zstd",
            compression_level=3,
            statistics=True,
        ),
    )

    print("Preprocess Config:    ")
    print(preprocess_cfg)

    logger_cfg = (
        LoggerConfig(
            level="INFO",
            format="%(asctime)s | %(levelname)-7s | %(message)s",
            timestamp_format="%Y-%m-%d %H:%M:%S",
        ),
    )
    print("\nLogger Config:    ")
    print(logger_cfg)
