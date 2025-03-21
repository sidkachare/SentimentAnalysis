import pandas as pd
from pathlib import Path
from src.SentimentAnalysis.components.data_preprocessing import DataPreprocessing
from src.SentimentAnalysis.config.configuration import ConfigurationManager

def test_data_preprocessing():
    """
    Test the data preprocessing pipeline:
    1. Check if vectorizer is saved.
    2. Check if X_train and X_test are generated.
    3. Verify text cleaning (stemming).
    """
    # Initialize configuration
    config_manager = ConfigurationManager("config/config.yaml")
    data_preprocessing_config = config_manager.get_data_preprocessing_config()

    # Load data
    train_df = pd.read_csv(Path("artifacts/data/train.csv"))
    test_df = pd.read_csv(Path("artifacts/data/test.csv"))

    # Test data preprocessing
    data_preprocessing = DataPreprocessing(data_preprocessing_config)
    X_train, X_test = data_preprocessing.vectorize_text(train_df, test_df)

    # Check if vectorizer is saved
    assert Path(data_preprocessing_config.vectorizer_path).exists(), "Vectorizer not saved!"

    # Check if X_train and X_test are generated
    assert X_train.shape[0] == len(train_df), "X_train shape mismatch!"
    assert X_test.shape[0] == len(test_df), "X_test shape mismatch!"

    # Verify stemming (new test)
    sample_text = "running quickly"
    cleaned_text = data_preprocessing.clean_text(sample_text)
    assert "run" in cleaned_text, "Stemming not applied correctly!"

if __name__ == "__main__":
    test_data_preprocessing()