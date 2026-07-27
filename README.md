# 🌸 FloraVision — Deep Learning Flower Classification & DIP Suite

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Pytest](https://img.shields.io/badge/Pytest-Automated_Tests-0A9EDC?style=for-the-badge&logo=pytest)](https://docs.pytest.org/)

**FloraVision** is a production-grade **Computer Vision & Digital Image Processing (DIP)** platform powered by **TensorFlow 2.15**, **Keras**, **FastAPI**, **Streamlit**, and **Docker**. 

The system utilizes a custom **12-Layer Convolutional Neural Network (CNN)** (`sequential_3`, 7.97M total parameters) pre-trained on botanical image datasets to classify 5 flower species (*Daisy, Dandelion, Roses, Sunflowers, Tulips*). It provides multi-interface execution via an interactive Streamlit Web Studio, a high-throughput FastAPI REST service, and a CLI utility.

---

## 🔬 CNN Model Architecture & Layer Topology

The deep learning inference engine (`src/classifier.py`) loads a trained Keras model (`flower_model.keras`) with an input shape of `(180, 180, 3)` (RGB). The 12-layer sequential CNN topology extracts spatial features through progressive 2D convolutions and max-pooling filters:

| Layer Index | Layer Name | Type | Layer Specifications / Operations | Output Tensor Shape | Param # |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Layer 0** | `sequential_2` | Data Augmentation | Random horizontal flipping & random rotation | `(None, 180, 180, 3)` | `0` |
| **Layer 1** | `rescaling_3` | Normalization | Pixel intensity rescaling (`1 / 255`) | `(None, 180, 180, 3)` | `0` |
| **Layer 2** | `conv2d_6` | Conv2D | 16 Filters, 3x3 Kernel, ReLU Activation | `(None, 180, 180, 16)` | `448` |
| **Layer 3** | `max_pooling2d_6` | MaxPooling2D | 2x2 Spatial Downsampling Pool | `(None, 90, 90, 16)` | `0` |
| **Layer 4** | `conv2d_7` | Conv2D | 32 Filters, 3x3 Kernel, ReLU Activation | `(None, 90, 90, 32)` | `4,640` |
| **Layer 5** | `max_pooling2d_7` | MaxPooling2D | 2x2 Spatial Downsampling Pool | `(None, 45, 45, 32)` | `0` |
| **Layer 6** | `conv2d_8` | Conv2D | 64 Filters, 3x3 Kernel, ReLU Activation | `(None, 45, 45, 64)` | `18,496` |
| **Layer 7** | `max_pooling2d_8` | MaxPooling2D | 2x2 Spatial Downsampling Pool | `(None, 22, 22, 64)` | `0` |
| **Layer 8** | `dropout` | Dropout | Spatial Dropout Regularization | `(None, 22, 22, 64)` | `0` |
| **Layer 9** | `flatten_2` | Flatten | Multi-Dimensional Tensor Flattening | `(None, 30976)` | `0` |
| **Layer 10** | `dense_4` | Dense | Fully Connected Layer (128 Units, ReLU) | `(None, 128)` | `3,965,056` |
| **Layer 11** | `outputs` | Dense (Softmax) | Output Classification (5 Botanical Classes) | `(None, 5)` | `645` |

> 📊 **Model Metrics**: **Total Parameters**: `7,978,572` (30.44 MB) | **Trainable Parameters**: `3,989,285` (15.22 MB)

---

## 🏗️ High-Level System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                    MULTI-INTERFACE EXECUTION LAYER                     │
│    (Streamlit Web Studio [app.py] / FastAPI [api.py] / CLI [cli.py])   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Raw Image Stream / File Path
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│             DIGITAL IMAGE PROCESSING (src/image_processor.py)           │
│   (load_image() ➔ BILINEAR 180x180 Resample ➔ Tensor (1, 180, 180, 3)) │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Normalized 4D Tensor Batch
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│             TENSORFLOW KERAS ENGINE (src/classifier.py)                 │
│   (FlowerClassifier ➔ model.predict() ➔ tf.nn.softmax ➔ Top-K Rank)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Calculated Probabilities & Metadata
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   BOTANICAL PAYLOAD SYNTHESIS                          │
│   (Top Class & Confidence % / Scientific Metadata / Color Tokens)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Engineering Challenges & Technical Solutions

### 1. Lazy Singleton Model Instantiation & Memory Caching
* **Challenge**: Loading a 47.9 MB `.keras` deep learning model on every HTTP request or Streamlit page re-render incurs massive CPU overhead and memory duplication.
* **Solution**: Implemented a singleton loading pattern inside `api.py` (`get_classifier_instance`) and leveraged Streamlit's `@st.cache_resource` decorator in `app.py`. The model is loaded into RAM once upon application cold-start and reused concurrently across all subsequent inference requests.

### 2. Multi-Input Image Standardization Pipeline
* **Challenge**: Supporting file paths (`str`), raw uploaded byte streams (`bytes`), and active `PIL.Image` objects across different interfaces without causing tensor shape mismatches or memory leaks.
* **Solution**: Engineered `load_image()` in `src/image_processor.py`. It inspects input types dynamically, uses `io.BytesIO` for binary uploads, enforces standard 3-channel `RGB` color conversion, resamples images using bilinear interpolation to `(180, 180)`, and expands tensor dimensions to yield float32 arrays of shape `(1, 180, 180, 3)`.

### 3. Native Spatial Gradient & Color Channel Decomposition (DIP Lab)
* **Challenge**: Providing digital image processing diagnostics (channel decomposition, edge detection) without relying on heavy external C++ libraries like OpenCV.
* **Solution**: Developed lightweight NumPY spatial algorithms in `src/image_processor.py`. `analyze_image_channels()` extracts individual Red, Green, Blue, and Luminance arrays using standard ITU-R 601-2 luminance weights (`0.2989 R + 0.5870 G + 0.1140 B`). `compute_edge_map()` implements a Sobel-like spatial difference gradient using `np.diff` matrix operations to render binary edge maps directly in Streamlit.

---

## 📊 API Reference & Schemas

### Response Payload Structure (`POST /predict`)

```json
{
  "predicted_class": "sunflowers",
  "confidence": 0.9854,
  "confidence_percentage": 98.54,
  "metadata": {
    "scientific_name": "Helianthus annuus",
    "family": "Asteraceae",
    "description": "Tall annual plant with prominent dark central disk florets surrounded by vivid yellow petals."
  },
  "all_probabilities": {
    "daisy": 0.0012,
    "dandelion": 0.0084,
    "roses": 0.0021,
    "sunflowers": 0.9854,
    "tulips": 0.0029
  },
  "top_k": [
    {
      "class_name": "sunflowers",
      "confidence": 0.9854,
      "percentage": 98.54,
      "color": "#FB8C00"
    },
    {
      "class_name": "dandelion",
      "confidence": 0.0084,
      "percentage": 0.84,
      "color": "#FBC02D"
    },
    {
      "class_name": "tulips",
      "confidence": 0.0029,
      "percentage": 0.29,
      "color": "#8E24AA"
    }
  ]
}
```

---

## 📁 Repository Architecture

```
tf-flower-classifier/
├── src/                          # Core Python Packages
│   ├── classifier.py             # FlowerClassifier engine & Softmax Top-K calculation
│   ├── image_processor.py        # DIP algorithms, channel extraction & tensor resizing
│   └── config.py                 # Hyperparameters, botanical metadata & color tokens
├── tests/                        # Automated Pytest Suite
│   ├── test_api.py               # FastAPI route integration tests
│   ├── test_classifier.py        # Model inference unit tests
│   └── test_image_processor.py   # Preprocessing & image transformation tests
├── api.py                        # FastAPI REST API service
├── app.py                        # Streamlit Interactive Web Studio & DIP Lab
├── cli.py                        # Command-line interface inference utility
├── flower_model.keras            # Trained 12-layer Keras CNN model weights (47.9 MB)
├── Dockerfile                    # Container build configuration
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Dependency specifications
└── README.md
```

---

## 🛠️ Local Development & Deployment Setup

### Prerequisites
- **Python**: `3.11+`
- **Pip**: Latest version

---

### 1. Virtual Environment Setup & Dependencies

```bash
# Clone the repository
git clone https://github.com/m-ali-swe/tf-flower-classifier.git
cd tf-flower-classifier

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows (PowerShell):
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

### 2. Running the Interactive Streamlit Web Studio

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### 3. Running the FastAPI REST Service

```bash
uvicorn api:app --reload --port 8000
```
- API Base Endpoint: `http://localhost:8000`
- Swagger OpenAPI Docs: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

---

### 4. Running CLI Inference

```bash
# Predict a single image with top-3 rankings
python cli.py predict path/to/flower.jpg --top-k 3

# View model architecture summary via CLI
python cli.py info
```

---

### 5. Automated Test Suite (`pytest`)

```bash
pytest -v
```

---

### 6. Docker Deployment

```bash
# Build and run containerized application
docker-compose up --build
```
Access Streamlit UI at `http://localhost:8501` and FastAPI at `http://localhost:8000`.
