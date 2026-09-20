import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from PIL import Image
import io
from transformers import AutoProcessor, AutoModelForCausalLM
from app.config import MODEL_ID, DEVICE

app = FastAPI(title="AMD OCR Challenge Server")

# Global variables for model and processor
model = None
processor = None

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
    Loads the VLM and processor into memory on server startup.
    Ensures weights are loaded into VRAM to avoid per-image latency.
    """
    global model, processor
    print(f"Loading model {MODEL_ID} onto {DEVICE}...")

    log_vram("Pre-Load")

    # Florence-2 requires trust_remote_code=True
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    model.eval()
    log_vram("Post-Load")
    print("Model loaded successfully.")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Processes an uploaded image and returns the extracted text using Florence-2.

    Args:
        file: The image file to process.

    Returns:
        A PredictionResponse containing the extracted text and a confidence score.
    """
    if model is None or processor is None:
        return PredictionResponse(text="Model not loaded", confidence=0.0)

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Florence-2 OCR task
        prompt = "<OCR>"

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, torch.float16 if DEVICE == "cuda" else torch.float32)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse Florence-2 output
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            pairwise=False
        )

        # For <OCR>, parsed_answer is a dictionary with 'text' key
        text_result = parsed_answer.get("text", "") if isinstance(parsed_answer, dict) else str(parsed_answer)

        return PredictionResponse(text=text_result, confidence=0.95)

    except Exception as e:
        print(f"Prediction error: {e}")
        return PredictionResponse(text=f"Error: {str(e)}", confidence=0.0)

if __name__ == "__main__":
    # This block is kept for local development.
    # In production/Docker, uvicorn is called directly via CMD.
    uvicorn.run(app, host="0.0.0.0", port=8000)
