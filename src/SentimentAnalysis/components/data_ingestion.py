import pandas as pd
from sklearn.model_selection import train_test_split
from src.SentimentAnalysis.logging.logger import logging
from src.SentimentAnalysis.utils.common import download_file, extract_tar, save_data
from pathlib import Path

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def download_and_extract_data(self):
        try:
            # Check if data already exists
            if Path(self.config.train_data_path).exists() and Path(self.config.test_data_path).exists():
                logging.info("Data already exists. Skipping download and extraction.")
                return

            # Download dataset
            tar_filepath = Path(self.config.data_dir) / "aclImdb_v1.tar.gz"
            logging.info(f"Downloading dataset from {self.config.data_url}...")
            download_file(self.config.data_url, tar_filepath)
            logging.info("Dataset downloaded successfully.")

            # Extract dataset
            logging.info(f"Extracting dataset to {self.config.data_dir}...")
            extract_tar(tar_filepath, Path(self.config.data_dir))
            logging.info("Dataset extracted successfully.")
        except Exception as e:
            logging.error(f"Error downloading or extracting dataset: {e}")
            raise e

    def load_and_split_data(self):
        try:
            # Load data
            reviews = []
            labels = []
            for label in ["pos", "neg"]:
                folder = Path(self.config.data_dir) / "aclImdb" / "train" / label
                for file in folder.glob("*.txt"):
                    with open(file, "r", encoding="utf-8") as f:
                        reviews.append(f.read())
                        labels.append(1 if label == "pos" else 0)

            df = pd.DataFrame({"review": reviews, "sentiment": labels})

            # Split data
            train_df, test_df = train_test_split(df, test_size=self.config.test_size, random_state=42)
            save_data(train_df, Path(self.config.train_data_path))
            save_data(test_df, Path(self.config.test_data_path))
            logging.info("Data split and saved successfully.")
        except Exception as e:
            logging.error(f"Error loading or splitting data: {e}")
            raise e
