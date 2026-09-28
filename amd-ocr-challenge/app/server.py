import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from PIL import Image
import io

from app.config import DEVICE, MODEL_TYPE
from app.engines.factory import OCREngineFactory

app = FastAPI(title="AMD OCR Challenge Server")

# Global variable for the OCR engine
engine = None

class PredictionResponse(BaseModel):
    """Schema for OCR prediction responses."""
    text: str
    confidence: float

def log_vram(label: str):
    """Logs the current GPU VRAM usage."""
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[{label}] VRAM Allocated: {allocated:.2f} MiB | Max Allocated: {max_allocated:.2f} MiB")
    else:
        print(f"[{label}] VRAM logging skipped (Device: {DEVICE})")

@app.on_event("startup")
async def load_model():
    """
    Initializes the OCR engine based on configuration.
    Ensures weights are loaded into VRAM on startup.
    """
    global engine
    print(f"Initializing OCR engine with model type: {MODEL_TYPE} onto {DEVICE}...")

    log_vram("Pre-Load")

    try:
        engine = OCREngineFactory.create(MODEL_TYPE)
        engine.load()
    except Exception as e:
        print(f"Failed to load OCR engine: {e}")
        # We allow the server to start even if the model fails,
        # but predictions will return an error.
        engine = None

    log_vram("Post-Load")
    print("OCR engine initialized successfully.")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Processes an uploaded image and returns the extracted text using the active OCR engine.
    """
    if engine is None:
        return PredictionResponse(text="Model not loaded", confidence=0.0)

    try:
        # Read image bytes
        contents = await file.read()

        # Use PIL to open image and convert to RGB
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Perform OCR using the selected engine
        text, confidence = engine.predict(image)

        return PredictionResponse(text=text, confidence=float(confidence))

    except Exception as e:
        print(f"Prediction error: {e}")
        return PredictionResponse(text=f"Error: {str(e)}", confidence=0.0)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
