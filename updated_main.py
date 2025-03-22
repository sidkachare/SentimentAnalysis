import os
from src.SentimentAnalysis.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.SentimentAnalysis.pipeline.data_preprocessing_pipeline import DataPreprocessingPipeline
from src.SentimentAnalysis.pipeline.model_trainer_pipeline import ModelTrainingPipeline
from src.SentimentAnalysis.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
from src.SentimentAnalysis.pipeline.model_inference_pipeline import ModelInferencePipeline
from src.SentimentAnalysis.config.configuration import ConfigurationManager
from src.SentimentAnalysis.logging.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Starting pipelines")

        # Flag to control which pipelines to run
        RUN_PREVIOUS_MODULES = False  # Set to False to skip previous modules

        if RUN_PREVIOUS_MODULES:
            # Run data ingestion pipeline
            logging.info("Running data ingestion pipeline")
            data_ingestion_pipeline = DataIngestionPipeline()
            data_ingestion_pipeline.run_pipeline()

            # Run data preprocessing pipeline
            logging.info("Running data preprocessing pipeline")
            data_preprocessing_pipeline = DataPreprocessingPipeline()
            data_preprocessing_pipeline.run_pipeline()

            # Run model training pipeline
            logging.info("Running model training pipeline")
            model_training_pipeline = ModelTrainingPipeline()
            model_training_pipeline.run_pipeline()
        else:
            logging.info("Skipping data ingestion, preprocessing, and model training pipelines")

        # Run model evaluation pipeline
        logging.info("Running model evaluation pipeline")
        eval_pipeline = ModelEvaluationPipeline()
        eval_pipeline.run_pipeline()

        # Run model inference pipeline
        logging.info("Running model inference pipeline")
        inference_pipeline = ModelInferencePipeline()
        inference_pipeline.run_pipeline()

        logging.info("All pipelines completed successfully")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise e