from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    data_url: str
    data_dir: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float


