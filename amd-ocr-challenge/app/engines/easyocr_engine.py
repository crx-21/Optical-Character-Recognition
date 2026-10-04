import re
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
        """Performs OCR using EasyOCR and selects the most plate-like result."""
        if self.reader is None:
            return "Model not loaded", 0.0

        try:
            # Convert PIL image to numpy array for EasyOCR
            image_np = np.array(image)

            # Perform OCR
            # We disable paragraph merging to get individual text blocks
            results = self.reader.readtext(image_np, paragraph=False)

            if not results:
                return "", 0.0

            # Since the images are plate-only, we want the single most "plate-like" block
            # rather than joining everything (which includes noise/glitches).
            best_text = ""
            best_score = -1.0

            for (bbox, text, confidence) in results:
                # Heuristic for "plate-likeness":
                # 1. Length should be reasonable (usually 3-10 chars)
                # 2. Should contain a mix of alphanumeric characters
                # 3. Higher confidence is better

                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                length_penalty = 1.0 if 4 <= len(clean_text) <= 9 else 0.5

                # Calculate a score based on confidence and length heuristic
                score = confidence * length_penalty

                if score > best_score:
                    best_score = score
                    best_text = text

            return best_text, float(best_score if best_score != -1.0 else 0.0)

        except Exception as e:
            print(f"EasyOCR prediction error: {e}")
            return f"Error: {str(e)}", 0.0
