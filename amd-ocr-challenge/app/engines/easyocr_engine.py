import numpy as np
import easyocr
from PIL import Image
from typing import Tuple
from app.engines.base import BaseOCREngine
from app.config import DEVICE

class EasyOCREngine(BaseOCREngine):
    """EasyOCR implementation of the OCR engine."""

    def __init__(self):
        self.reader = None

    def load(self) -> None:
        """Loads EasyOCR reader into memory."""
        print(f"Loading EasyOCR reader onto {DEVICE}...")
        # Initialize reader for English and Simplified Chinese
        self.reader = easyocr.Reader(['en', 'ch_sim'], gpu=(DEVICE == "cuda"))
        print("EasyOCR reader loaded successfully.")

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """Performs OCR using EasyOCR."""
        if self.reader is None:
            return "Model not loaded", 0.0

        try:
            # Convert PIL image to numpy array for EasyOCR
            image_np = np.array(image)

            # Perform OCR
            # result is a list of tuples: (bbox, text, confidence)
            results = self.reader.readtext(image_np)

            if not results:
                return "", 0.0

            # Concatenate text fragments and calculate average confidence
            texts = [res[1] for res in results]
            confidences = [res[2] for res in results]

            full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences)

            return full_text, float(avg_confidence)

        except Exception as e:
            print(f"EasyOCR prediction error: {e}")
            return f"Error: {str(e)}", 0.0
