import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import Dataset
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pytorch
from src.SentimentAnalysis.logging.logger import logging

class ModelTrainer:
    def __init__(self, model_trainer_config, mlflow_config):
        self.model_trainer_config = model_trainer_config
        self.mlflow_config = mlflow_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_data(self, dataset):
        return self.tokenizer(
            dataset["review"],
            padding="max_length",
            truncation=True,
            max_length=self.model_trainer_config.max_seq_length,
            return_tensors="pt"  # Ensure the output is PyTorch tensors
        )

    def train_model(self, train_df, test_df):
        try:
            # Start MLflow run
            mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
            mlflow.set_experiment(self.mlflow_config.experiment_name)

            with mlflow.start_run():
                # Convert DataFrame to Hugging Face Dataset
                train_dataset = Dataset.from_pandas(train_df)
                test_dataset = Dataset.from_pandas(test_df)

                # Tokenize data
                train_dataset = train_dataset.map(self.tokenize_data, batched=True)
                test_dataset = test_dataset.map(self.tokenize_data, batched=True)

                # Set format to PyTorch tensors
                train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "sentiment"])
                test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "sentiment"])

                # Create DataLoader
                train_loader = DataLoader(train_dataset, batch_size=self.model_trainer_config.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.model_trainer_config.batch_size)

                # Load pre-trained BERT model and move to GPU
                model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
                model.to(self.device)  # Move model to GPU

                # Define optimizer
                optimizer = AdamW(model.parameters(), lr=self.model_trainer_config.learning_rate)

                # Training loop
                for epoch in range(self.model_trainer_config.epochs):
                    model.train()
                    total_loss = 0
                    for batch in train_loader:
                        optimizer.zero_grad()

                        # Move batch to GPU
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["sentiment"].to(self.device)

                        # Forward pass
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        total_loss += loss.item()

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                    avg_loss = total_loss / len(train_loader)
                    logging.info(f"Epoch {epoch + 1}/{self.model_trainer_config.epochs}, Loss: {avg_loss}")

                # Save the model
                model.save_pretrained(self.model_trainer_config.model_path)
                self.tokenizer.save_pretrained(self.model_trainer_config.model_path)
                logging.info(f"Model saved to {self.model_trainer_config.model_path}")

                # Evaluate the model
                model.eval()
                predictions, true_labels = [], []
                for batch in test_loader:
                    with torch.no_grad():
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["sentiment"].to(self.device)

                        outputs = model(input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)

                        predictions.extend(preds.cpu().numpy())
                        true_labels.extend(labels.cpu().numpy())

                accuracy = accuracy_score(true_labels, predictions)
                logging.info(f"Test Accuracy: {accuracy}")

                # Log metrics to MLflow
                mlflow.log_metric("accuracy", accuracy)

                # Log model to MLflow
                mlflow.pytorch.log_model(model, "bert_model")

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise e