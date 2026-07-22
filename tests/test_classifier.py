"""
Unit tests for FlowerClassifier inference engine.
"""

import pytest
import numpy as np
from PIL import Image

from src.classifier import FlowerClassifier
from src.config import CLASS_NAMES


@pytest.fixture
def classifier():
    return FlowerClassifier()


@pytest.fixture
def sample_pil_image():
    # Create a 200x200 RGB synthetic image
    arr = np.ones((200, 200, 3), dtype=np.uint8) * 128
    return Image.fromarray(arr)


def test_classifier_initialization(classifier):
    assert classifier.model is not None
    assert classifier.class_names == CLASS_NAMES


def test_predict_image(classifier, sample_pil_image):
    result = classifier.predict_image(sample_pil_image, top_k=3)
    assert "predicted_class" in result
    assert result["predicted_class"] in CLASS_NAMES
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_k"]) == 3
    assert result["top_k"][0]["confidence"] >= result["top_k"][1]["confidence"]


def test_get_model_summary_dict(classifier):
    summary = classifier.get_model_summary_dict()
    assert "total_layers" in summary
    assert summary["target_classes"] == 5
    assert summary["trainable_params"] > 0
