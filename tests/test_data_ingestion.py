import pandas as pd
import pytest
from pathlib import Path
from src.SentimentAnalysis.components.data_ingestion import DataIngestion
from src.SentimentAnalysis.config.configuration import ConfigurationManager

def test_data_ingestion():
    # Initialize configuration
    config_manager = ConfigurationManager("config/config.yaml")
    data_ingestion_config = config_manager.get_data_ingestion_config()

    # Perform data ingestion
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.download_and_extract_data()
    data_ingestion.load_and_split_data()

    # Check if train and test files exist
    train_path = Path(data_ingestion_config.train_data_path)
    test_path = Path(data_ingestion_config.test_data_path)
    assert train_path.exists(), "Train data file not found!"
    assert test_path.exists(), "Test data file not found!"

    # Check if train and test files contain valid data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert len(train_df) > 0, "Train data is empty!"
    assert len(test_df) > 0, "Test data is empty!"

    # Check if the dataset is split correctly
    assert len(train_df) / (len(train_df) + len(test_df)) == pytest.approx(0.8, 0.01), "Train-test split ratio is incorrect!"

if __name__ == "__main__":
    test_data_ingestion()