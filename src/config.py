"""
Configuration constants and settings for Flower Classifier application.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "flower_model.keras"

# Image Preprocessing Parameters
TARGET_IMAGE_SIZE = (180, 180)
IMAGE_CHANNELS = 3

# Target Flower Classes
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# Color Mapping for Visualization & UI Dashboard
CLASS_COLORS = {
    "daisy": "#F9A825",       # Warm Gold / Yellow Center
    "dandelion": "#FBC02D",   # Bright Dandelion Yellow
    "roses": "#E53935",       # Vibrant Crimson Rose Red
    "sunflowers": "#FB8C00",  # Sunflower Deep Orange
    "tulips": "#8E24AA",      # Deep Tulip Purple
}

# Detailed Class Metadata for Rich UI Display
CLASS_METADATA = {
    "daisy": {
        "scientific_name": "Bellis perennis",
        "family": "Asteraceae",
        "description": "Composite flower head composed of small florets surrounded by white ray florets."
    },
    "dandelion": {
        "scientific_name": "Taraxacum officinale",
        "family": "Asteraceae",
        "description": "Bright yellow flowering herb known for broad rosette leaves and silky seed heads."
    },
    "roses": {
        "scientific_name": "Rosa spp.",
        "family": "Rosaceae",
        "description": "Perennial woody plant featuring layered petals, rich fragrance, and protective thistles."
    },
    "sunflowers": {
        "scientific_name": "Helianthus annuus",
        "family": "Asteraceae",
        "description": "Tall annual plant with prominent dark central disk florets surrounded by vivid yellow petals."
    },
    "tulips": {
        "scientific_name": "Tulipa spp.",
        "family": "Liliaceae",
        "description": "Spring-blooming perennial bulb with symmetrical cup-shaped flowers."
    }
}
