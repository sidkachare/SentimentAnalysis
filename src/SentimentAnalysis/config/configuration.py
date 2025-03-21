import os
from pathlib import Path
from src.SentimentAnalysis.entity.config_entity import DataIngestionConfig
from src.SentimentAnalysis.entity.config_entity import DataPreprocessingConfig

class ConfigurationManager:
    def __init__(self, config_filepath):
        self.config = self.read_config(config_filepath)

    def read_config(self, config_filepath):
        import yaml
        with open(config_filepath, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        data_ingestion_config = DataIngestionConfig(
            data_url=config["data_url"],
            data_dir=Path(config["data_dir"]),
            train_data_path=Path(config["train_data_path"]),
            test_data_path=Path(config["test_data_path"]),
            test_size=config["test_size"]
        )
        return data_ingestion_config
    

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config["data_preprocessing"]
        return DataPreprocessingConfig(
            max_features=config["max_features"],
            vectorizer_path=Path(config["vectorizer_path"])
        )
    
