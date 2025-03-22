import pytest
from src.SentimentAnalysis.pipeline.model_inference_pipeline import ModelInferencePipeline

def test_model_inference():
    try:
        pipeline = ModelInferencePipeline()
        pipeline.run_pipeline()
        assert True
    except Exception as e:
        pytest.fail(f"Model inference pipeline failed: {e}")