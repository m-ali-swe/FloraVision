"""
FastAPI REST API Service for Flower Classifier Model.
Exposes endpoints for single/batch predictions, health monitoring, and class metadata.
"""

from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import io

from src.classifier import FlowerClassifier
from src.config import CLASS_NAMES, CLASS_METADATA

# Initialize FastAPI App with OpenAPI metadata
app = FastAPI(
    title="FloraVision Flower Classification REST API",
    description="Production-grade API for deep learning flower classification and digital image processing.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy/Singleton Classifier Instance
_classifier: FlowerClassifier = None


def get_classifier_instance() -> FlowerClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FlowerClassifier()
    return _classifier


# Response Pydantic Schemas for Swagger Docs
class ClassProbability(BaseModel):
    class_name: str
    confidence: float
    percentage: float


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., example="sunflowers")
    confidence: float = Field(..., example=0.9854)
    confidence_percentage: float = Field(..., example=98.54)
    metadata: Dict[str, str]
    all_probabilities: Dict[str, float]
    top_k: List[ClassProbability]


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    supported_classes: List[str]


@app.get("/", tags=["General"])
def read_root():
    """API Index Endpoint"""
    return {
        "title": "FloraVision Flower Classification API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": ["/health", "/classes", "/predict", "/predict-batch"]
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint for container readiness/liveness probes."""
    try:
        classifier = get_classifier_instance()
        model_ready = classifier.model is not None
    except Exception:
        model_ready = False

    return {
        "status": "healthy" if model_ready else "unhealthy",
        "model_loaded": model_ready,
        "supported_classes": CLASS_NAMES
    }


@app.get("/classes", tags=["Metadata"])
def get_supported_classes():
    """Returns supported flower species and botanical descriptions."""
    return {
        "total_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "details": CLASS_METADATA
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (JPG, PNG, WEBP) and returns prediction probabilities.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file content type '{file.content_type}'. Must be an image file."
        )

    try:
        image_bytes = await file.read()
        classifier = get_classifier_instance()
        result = classifier.predict_image(image_bytes, top_k=5)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/predict-batch", tags=["Inference"])
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """
    Accepts multiple uploaded image files and returns a list of classification results.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit exceeded. Maximum 10 files per batch request."
        )

    results = []
    classifier = get_classifier_instance()

    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({"filename": file.filename, "error": "Invalid content type"})
            continue

        try:
            image_bytes = await file.read()
            pred = classifier.predict_image(image_bytes, top_k=3)
            pred["filename"] = file.filename
            results.append(pred)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"batch_size": len(files), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
