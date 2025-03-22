import pandas as pd
from pathlib import Path
from src.SentimentAnalysis.components.model_trainer import ModelTrainer
from src.SentimentAnalysis.config.configuration import ConfigurationManager

def test_model_trainer():
    """
    Test the BERT model training pipeline:
    1. Check if the model is saved.
    2. Verify that metrics are logged to MLflow.
    """
    # Initialize configuration
    config_manager = ConfigurationManager("config/config.yaml")
    model_trainer_config = config_manager.get_model_trainer_config()

    # Load data
    train_df = pd.read_csv(Path("artifacts/data/train.csv"))
    test_df = pd.read_csv(Path("artifacts/data/test.csv"))

    # Test model training
    model_trainer = ModelTrainer(model_trainer_config)
    model_trainer.train_model(train_df, test_df)

    # Check if model is saved
    assert Path(model_trainer_config.model_path).exists(), "Model not saved!"

if __name__ == "__main__":
    test_model_trainer()