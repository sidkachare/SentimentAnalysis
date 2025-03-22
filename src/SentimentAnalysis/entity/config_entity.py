from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    data_url: str
    data_dir: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float


@dataclass
class DataPreprocessingConfig:
    max_features: int
    vectorizer_path: Path


@dataclass
class ModelTrainerConfig:
    model_path: Path
    batch_size: int
    max_seq_length: int
    learning_rate: float
    epochs: int

@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str