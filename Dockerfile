# Use official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing pyc files to disc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and model weights
COPY . .

# Expose Streamlit (8501) and FastAPI (8000) ports
EXPOSE 8501 8000

# Default command launches FastAPI REST API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
