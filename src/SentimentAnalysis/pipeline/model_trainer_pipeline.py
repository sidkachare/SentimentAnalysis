import pandas as pd
from pathlib import Path
from src.SentimentAnalysis.components.model_trainer import ModelTrainer
from src.SentimentAnalysis.logging.logger import logging
from src.SentimentAnalysis.config.configuration import ConfigurationManager

class ModelTrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager("config/config.yaml")
        self.model_trainer_config = self.config_manager.get_model_trainer_config()
        self.mlflow_config = self.config_manager.get_mlflow_config()
        self.data_ingestion_config = self.config_manager.get_data_ingestion_config()

    def run_pipeline(self):
        """
        Run the model training pipeline:
        1. Load data.
        2. Train and evaluate the BERT model.
        3. Log metrics and model to MLflow.
        """
        try:
            # Load data
            train_df = pd.read_csv(self.data_ingestion_config.train_data_path)
            test_df = pd.read_csv(self.data_ingestion_config.test_data_path)

            print("train_df:")
            print(train_df.head())
            print("\ntest_df:")
            print(test_df.head())

            # Initialize ModelTrainer
            model_trainer = ModelTrainer(self.model_trainer_config, self.mlflow_config)
            model_trainer.train_model(train_df, test_df)

            logging.info("Model training pipeline completed successfully.")

        except Exception as e:
            logging.error(f"Error in model training pipeline: {e}")
            raise e

if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.run_pipeline()