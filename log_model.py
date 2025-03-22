import mlflow
import mlflow.pytorch
from transformers import BertForSequenceClassification, BertTokenizer

def log_model_to_mlflow():
    # Set the tracking URI to your local MLflow server
    mlflow.set_tracking_uri("https://127.0.0.1:5000")

    # Start a new MLflow run
    with mlflow.start_run():
        # Log parameters (example)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("epochs", 3)

        # Log metrics (example)
        accuracy = 0.9394  # Replace with actual accuracy
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        model = BertForSequenceClassification.from_pretrained("artifacts/models/bert_model")
        mlflow.pytorch.log_model(model, "bert_model")

        # Optionally, log the tokenizer
        tokenizer = BertTokenizer.from_pretrained("artifacts/models/bert_model")
        tokenizer.save_pretrained("artifacts/models/bert_model")
        mlflow.log_artifact("artifacts/models/bert_model")

if __name__ == "__main__":
    log_model_to_mlflow()