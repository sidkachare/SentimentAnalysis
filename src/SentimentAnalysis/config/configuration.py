import os
from pathlib import Path
from src.SentimentAnalysis.entity.config_entity import DataIngestionConfig
from src.SentimentAnalysis.entity.config_entity import DataPreprocessingConfig
from src.SentimentAnalysis.entity.config_entity import ModelTrainerConfig
from src.SentimentAnalysis.entity.config_entity import MLflowConfig

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
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config["model_trainer"]
        return ModelTrainerConfig(
            model_path=Path(config["model_path"]),
            batch_size=int(config["batch_size"]),
            max_seq_length=int(config["max_seq_length"]),
            learning_rate=float(config["learning_rate"]),
            epochs=int(config["epochs"])
        )

    def get_mlflow_config(self) -> MLflowConfig:
        config = self.config["mlflow"]
        return MLflowConfig(
            tracking_uri=config["tracking_uri"],
            experiment_name=config["experiment_name"]
        )
