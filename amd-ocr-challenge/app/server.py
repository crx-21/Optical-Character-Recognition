import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from PIL import Image
import io
import numpy as np
import easyocr
from app.config import DEVICE

app = FastAPI(title="AMD OCR Challenge Server")

# Global variable for the OCR reader
reader = None

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
    Loads EasyOCR reader into memory on server startup.
    Ensures weights are loaded into VRAM to avoid per-image latency.
    """
    global reader
    print(f"Loading EasyOCR reader onto {DEVICE}...")

    log_vram("Pre-Load")

    # Initialize reader for English and Simplified Chinese
    # EasyOCR handles GPU automatically if torch.cuda.is_available()
    reader = easyocr.Reader(['en', 'ch_sim'], gpu=(DEVICE == "cuda"))

    log_vram("Post-Load")
    print("OCR reader loaded successfully.")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Processes an uploaded image and returns the extracted text using EasyOCR.

    Args:
        file: The image file to process.

    Returns:
        A PredictionResponse containing the extracted text and a confidence score.
    """
    if reader is None:
        return PredictionResponse(text="Model not loaded", confidence=0.0)

    try:
        # Read image bytes
        contents = await file.read()

        # Use PIL to open image (handles PNG, JPEG, TIFF) and convert to RGB
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Convert PIL image to numpy array for EasyOCR
        image_np = np.array(image)

        # Perform OCR
        # result is a list of tuples: (bbox, text, confidence)
        results = reader.readtext(image_np)

        if not results:
            return PredictionResponse(text="", confidence=0.0)

        # Concatenate text fragments and calculate average confidence
        texts = [res[1] for res in results]
        confidences = [res[2] for res in results]

        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences)

        return PredictionResponse(text=full_text, confidence=float(avg_confidence))

    except Exception as e:
        print(f"Prediction error: {e}")
        return PredictionResponse(text=f"Error: {str(e)}", confidence=0.0)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
