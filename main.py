from src.SentimentAnalysis.components.data_ingestion import DataIngestion
from src.SentimentAnalysis.pipeline.data_preprocessing_pipeline import DataPreprocessingPipeline
from src.SentimentAnalysis.pipeline.model_trainer_pipeline import ModelTrainingPipeline
from src.SentimentAnalysis.config.configuration import ConfigurationManager
from pathlib import Path

def main():
    # Initialize configuration
    config_manager = ConfigurationManager("config/config.yaml")

    # Data Ingestion (only if data doesn't exist)
    data_ingestion_config = config_manager.get_data_ingestion_config()
    if not Path(data_ingestion_config.train_data_path).exists() or not Path(data_ingestion_config.test_data_path).exists():
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_and_extract_data()
        data_ingestion.load_and_split_data()
    else:
        print("Data already exists. Skipping data ingestion.")

    # Data Preprocessing (only if vectorizer doesn't exist)
    data_preprocessing_config = config_manager.get_data_preprocessing_config()
    if not Path(data_preprocessing_config.vectorizer_path).exists():
        preprocessing_pipeline = DataPreprocessingPipeline()
        X_train, X_test = preprocessing_pipeline.run_pipeline()
    else:
        print("Vectorizer already exists. Skipping data preprocessing.")

    # Model Training (only if model doesn't exist)
    model_trainer_config = config_manager.get_model_trainer_config()
    if not Path(model_trainer_config.model_path).exists():
        training_pipeline = ModelTrainingPipeline()
        training_pipeline.run_pipeline()
    else:
        print("Model already exists. Skipping model training.")

if __name__ == "__main__":
    main()