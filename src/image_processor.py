"""
Digital Image Processing (DIP) and Preprocessing Module.
Handles image loading, format conversions, channel extraction, edge detection, and histogram statistics.
"""

from typing import Tuple, Dict, Any, Union
import numpy as np
from PIL import Image, ImageOps
import io


def load_image(image_input: Union[str, bytes, Image.Image, io.BytesIO]) -> Image.Image:
    """
    Load an image from filepath, raw bytes, BytesIO, or PIL Image into a standard RGB PIL Image.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (str, bytes, io.BytesIO)):
        if isinstance(image_input, bytes):
            image_input = io.BytesIO(image_input)
        img = Image.open(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    return img.convert("RGB")


def process_image_for_inference(
    image: Image.Image,
    target_size: Tuple[int, int] = (180, 180)
) -> np.ndarray:
    """
    Preprocesses PIL image to match Keras model input expectations.
    
    Returns:
        np.ndarray of shape (1, target_size[0], target_size[1], 3) with float32 values.
    """
    resized_img = image.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32)
    # Expand dims for batch dimension: (1, H, W, C)
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array


def analyze_image_channels(image: Image.Image) -> Dict[str, np.ndarray]:
    """
    Extract individual RGB channels and grayscale representation for DIP analysis.
    """
    rgb_array = np.array(image.convert("RGB"))
    
    r_channel = rgb_array[:, :, 0]
    g_channel = rgb_array[:, :, 1]
    b_channel = rgb_array[:, :, 2]
    
    # Grayscale conversion using standard luminance weights
    gray_array = np.dot(rgb_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    return {
        "rgb": rgb_array,
        "red": r_channel,
        "green": g_channel,
        "blue": b_channel,
        "grayscale": gray_array
    }


def compute_edge_map(gray_array: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Simple Digital Image Processing gradient/edge preview (Sobel-like gradient approximation).
    Used for demonstrating DIP concepts without requiring external heavy C-libraries.
    """
    # Simple spatial horizontal and vertical differences
    dx = np.abs(np.diff(gray_array.astype(np.int32), axis=1, append=0))
    dy = np.abs(np.diff(gray_array.astype(np.int32), axis=0, append=0))
    magnitude = np.clip(dx + dy, 0, 255).astype(np.uint8)
    edge_binary = (magnitude > threshold).astype(np.uint8) * 255
    return edge_binary


def compute_image_stats(image: Image.Image) -> Dict[str, Any]:
    """
    Compute basic digital statistics (dimensions, mean, std, aspect ratio, mode).
    """
    arr = np.array(image)
    width, height = image.size
    
    return {
        "dimensions": f"{width} x {height} px",
        "aspect_ratio": round(width / height, 2) if height > 0 else 1.0,
        "mode": image.mode,
        "mean_intensity": float(np.mean(arr)),
        "std_intensity": float(np.std(arr)),
        "min_val": int(np.min(arr)),
        "max_val": int(np.max(arr)),
    }
