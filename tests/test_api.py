"""
Integration tests for FastAPI endpoints.
"""

import pytest
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_classes_endpoint():
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 5


def test_predict_endpoint():
    # Create synthetic image bytes
    arr = np.random.randint(0, 256, (180, 180, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    files = {"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence_percentage" in data
