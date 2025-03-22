import pytest
from src.SentimentAnalysis.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

def test_model_evaluation():
    try:
        pipeline = ModelEvaluationPipeline()
        pipeline.run_pipeline()
        assert True
    except Exception as e:
        pytest.fail(f"Model evaluation pipeline failed: {e}")