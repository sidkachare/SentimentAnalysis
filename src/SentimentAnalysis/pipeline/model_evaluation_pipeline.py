from src.SentimentAnalysis.components.model_evaluation import ModelEvaluator
from src.SentimentAnalysis.config.configuration import ConfigurationManager
from src.SentimentAnalysis.logging.logger import logging

class ModelEvaluationPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def run_pipeline(self):
        try:
            logging.info("Starting model evaluation pipeline")
            evaluator = ModelEvaluator(self.config)
            evaluator.evaluate_model()
            logging.info("Model evaluation pipeline completed")
        except Exception as e:
            logging.error(f"Error in model evaluation pipeline: {e}")
            raise e