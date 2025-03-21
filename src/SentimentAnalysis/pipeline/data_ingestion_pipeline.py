from src.SentimentAnalysis.config.configuration import ConfigurationManager
from src.SentimentAnalysis.components.data_ingestion import DataIngestion


def run_data_ingestion():
    config_manager = ConfigurationManager()
    ingestion_config = config_manager.get_data_ingestion_config()

    ingestion = DataIngestion(ingestion_config)
    raw_data_path = ingestion.initiate_data_ingestion()

    print(f"Data ingestion completed. Raw data saved at: {raw_data_path}")


if __name__ == "__main__":
    run_data_ingestion()