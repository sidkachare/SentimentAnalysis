import pandas as pd
from src.SentimentAnalysis.components.data_ingestion import DataIngestion
from src.SentimentAnalysis.config.configuration import ConfigurationManager


def main():
    # Initialize configuration
    config_manager = ConfigurationManager("config/config.yaml")
    data_ingestion_config = config_manager.get_data_ingestion_config()

    # Perform data ingestion
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.download_and_extract_data()
    data_ingestion.load_and_split_data()

if __name__ == "__main__":
    main()