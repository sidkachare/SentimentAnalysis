import os
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from src.SentimentAnalysis.logging.logger import logging

class ModelInferencer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BertForSequenceClassification.from_pretrained(self.config.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_path)
        self.model.to(self.device)

    def infer(self, text):
        try:
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_seq_length)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).item()

            # Map prediction to label
            labels = ["negative", "positive"]
            return labels[preds]

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise e

    def infer_batch(self):
        try:
            # Load inference data
            inference_df = pd.read_csv(os.path.join(self.config.inference_data_dir, "inference_data.csv"))
            results = []

            # Perform inference on each row
            for _, row in inference_df.iterrows():
                text = row["review"]
                prediction = self.infer(text)
                results.append({"review": text, "prediction": prediction})

            # Save results to file
            os.makedirs(self.config.inference_results_dir, exist_ok=True)
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(self.config.inference_results_dir, self.config.inference_file), index=False)

            logging.info("Inference completed. Results saved.")

        except Exception as e:
            logging.error(f"Error during batch inference: {e}")
            raise e