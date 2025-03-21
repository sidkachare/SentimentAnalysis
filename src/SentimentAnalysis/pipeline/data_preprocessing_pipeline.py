import pandas as pd
from pathlib import Path
from src.SentimentAnalysis.components.data_preprocessing import DataPreprocessing
from src.SentimentAnalysis.logging.logger import logging
from src.SentimentAnalysis.utils.common import save_pickle
from src.SentimentAnalysis.config.configuration import ConfigurationManager

class DataPreprocessingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager("config/config.yaml")
        self.data_preprocessing_config = self.config_manager.get_data_preprocessing_config()
        self.data_ingestion_config = self.config_manager.get_data_ingestion_config()

    def run_pipeline(self):
        """
        Run the data preprocessing pipeline:
        1. Load data.
        2. Clean and vectorize text.
        3. Save vectorized data.
        """
        try:
            # Load data
            train_df = pd.read_csv(self.data_ingestion_config.train_data_path)
            test_df = pd.read_csv(self.data_ingestion_config.test_data_path)

            # Initialize DataPreprocessing
            data_preprocessing = DataPreprocessing(self.data_preprocessing_config)

            # Vectorize text data
            X_train, X_test = data_preprocessing.vectorize_text(train_df, test_df)

            # Save vectorized data (optional, if needed)
            save_pickle(X_train, Path(self.data_preprocessing_config.vectorizer_path).parent / "X_train.pkl")
            save_pickle(X_test, Path(self.data_preprocessing_config.vectorizer_path).parent / "X_test.pkl")

            logging.info("Data preprocessing pipeline completed successfully.")
            return X_train, X_test

        except Exception as e:
            logging.error(f"Error in data preprocessing pipeline: {e}")
            raise e

if __name__ == "__main__":
    pipeline = DataPreprocessingPipeline()
    pipeline.run_pipeline()