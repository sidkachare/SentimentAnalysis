import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
import pandas as pd
import mlflow
from src.SentimentAnalysis.logging.logger import logging
from src.SentimentAnalysis.utils.common import read_yaml

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BertForSequenceClassification.from_pretrained(self.config.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_path)
        self.model.to(self.device)

    def evaluate_model(self):
        try:
            # Load test data
            test_df = pd.read_csv(os.path.join(self.config.eval_data_dir, "test.csv"))
            test_dataset = Dataset.from_pandas(test_df)

            # Tokenize test data
            test_dataset = test_dataset.map(
                lambda x: self.tokenizer(x["review"], padding="max_length", truncation=True, max_length=self.config.max_seq_length),
                batched=True
            )
            test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "sentiment"])

            # Create DataLoader
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.batch_size)

            # Evaluate the model
            self.model.eval()
            predictions, true_labels = [], []
            for batch in test_loader:
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["sentiment"].to(self.device)

                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            classification_rep = classification_report(true_labels, predictions, target_names=["negative", "positive"])

            # Log metrics to MLflow
            with mlflow.start_run():
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_text(classification_rep, "classification_report.txt")

            # Save metrics to file
            os.makedirs(self.config.eval_results_dir, exist_ok=True)
            with open(os.path.join(self.config.eval_results_dir, self.config.metrics_file), "w") as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write("Classification Report:\n")
                f.write(classification_rep)

            logging.info(f"Evaluation completed. Accuracy: {accuracy}")

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise e