"""
FlowerClassifier Module.
Encapsulates TensorFlow/Keras deep learning model inference, softmax calculations, and top-k probability ranking.
"""

import os
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image

import tensorflow as tf

from src.config import DEFAULT_MODEL_PATH, TARGET_IMAGE_SIZE, CLASS_NAMES, CLASS_METADATA, CLASS_COLORS
from src.image_processor import load_image, process_image_for_inference


class FlowerClassifier:
    """
    Production-ready Flower Classification Engine encapsulating TensorFlow model loading & inference.
    """

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model: Optional[tf.keras.Model] = None
        self.class_names = CLASS_NAMES
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the pre-trained Keras model from disk.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at path: {self.model_path}")
        
        # Load model using TensorFlow Keras
        self.model = tf.keras.models.load_model(str(self.model_path))

    def predict_image(
        self,
        image_input: Union[str, bytes, Image.Image],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predicts the flower species for a single input image.

        Args:
            image_input: File path, bytes, or PIL Image object.
            top_k: Number of top candidate predictions to return.

        Returns:
            Dict containing predicted class, confidence, top-k probabilities, and metadata.
        """
        if self.model is None:
            self._load_model()

        # Load and preprocess image
        pil_image = load_image(image_input)
        batch_input = process_image_for_inference(pil_image, TARGET_IMAGE_SIZE)

        # Run inference
        raw_predictions = self.model.predict(batch_input, verbose=0)
        
        # Apply Softmax activation to convert raw logits into probability distribution
        probabilities = tf.nn.softmax(raw_predictions[0]).numpy()

        # Extract top prediction
        top_idx = int(np.argmax(probabilities))
        top_class = self.class_names[top_idx]
        top_confidence = float(probabilities[top_idx])

        # Extract Top-K predictions
        top_k_indices = np.argsort(probabilities)[::-1][:top_k]
        top_k_results = [
            {
                "class_name": self.class_names[idx],
                "confidence": round(float(probabilities[idx]), 4),
                "percentage": round(float(probabilities[idx]) * 100, 2),
                "color": CLASS_COLORS.get(self.class_names[idx], "#000000"),
            }
            for idx in top_k_indices
        ]

        # Detailed response payload
        return {
            "predicted_class": top_class,
            "confidence": round(top_confidence, 4),
            "confidence_percentage": round(top_confidence * 100, 2),
            "metadata": CLASS_METADATA.get(top_class, {}),
            "all_probabilities": {
                name: round(float(prob), 4)
                for name, prob in zip(self.class_names, probabilities)
            },
            "top_k": top_k_results
        }

    def predict_batch(
        self,
        images: List[Union[str, bytes, Image.Image]],
        top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Predicts flower species for a batch of input images.
        """
        results = []
        for img in images:
            results.append(self.predict_image(img, top_k=top_k))
        return results

    def get_model_summary_dict(self) -> Dict[str, Any]:
        """
        Returns structured architectural metadata about the loaded TensorFlow model.
        """
        if self.model is None:
            self._load_model()

        return {
            "name": self.model.name,
            "total_layers": len(self.model.layers),
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "trainable_params": int(np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),
            "non_trainable_params": int(np.sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])),
            "target_classes": len(self.class_names)
        }
