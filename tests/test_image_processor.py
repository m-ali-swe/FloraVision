"""
Unit tests for Digital Image Processing helper functions.
"""

import pytest
import numpy as np
from PIL import Image

from src.image_processor import (
    load_image,
    process_image_for_inference,
    analyze_image_channels,
    compute_edge_map,
    compute_image_stats,
)


@pytest.fixture
def dummy_image():
    """Create a synthetic RGB PIL image for testing."""
    arr = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_load_image(dummy_image):
    loaded = load_image(dummy_image)
    assert isinstance(loaded, Image.Image)
    assert loaded.mode == "RGB"


def test_process_image_for_inference(dummy_image):
    processed = process_image_for_inference(dummy_image, target_size=(180, 180))
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 180, 180, 3)
    assert processed.dtype == np.float32


def test_analyze_image_channels(dummy_image):
    channels = analyze_image_channels(dummy_image)
    assert "red" in channels
    assert "green" in channels
    assert "blue" in channels
    assert "grayscale" in channels
    assert channels["grayscale"].shape == (200, 300)


def test_compute_edge_map(dummy_image):
    channels = analyze_image_channels(dummy_image)
    edge_map = compute_edge_map(channels["grayscale"])
    assert isinstance(edge_map, np.ndarray)
    assert edge_map.shape == (200, 300)


def test_compute_image_stats(dummy_image):
    stats = compute_image_stats(dummy_image)
    assert "dimensions" in stats
    assert stats["dimensions"] == "300 x 200 px"
    assert "mean_intensity" in stats
