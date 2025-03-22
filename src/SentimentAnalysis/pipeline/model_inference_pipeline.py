from src.SentimentAnalysis.components.model_inference import ModelInferencer
from src.SentimentAnalysis.config.configuration import ConfigurationManager
from src.SentimentAnalysis.logging.logger import logging

class ModelInferencePipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_inference_config()

    def run_pipeline(self):
        try:
            logging.info("Starting model inference pipeline")
            inferencer = ModelInferencer(self.config)
            inferencer.infer_batch()
            logging.info("Model inference pipeline completed")
        except Exception as e:
            logging.error(f"Error in model inference pipeline: {e}")
            raise e