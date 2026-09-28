from abc import ABC, abstractmethod
from typing import Tuple
from PIL import Image

class BaseOCREngine(ABC):
    """Abstract base class for all OCR engines."""

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights into VRAM.
        This should be called once during server startup.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Perform OCR on the provided image.

        Args:
            image: A PIL Image object in RGB mode.

        Returns:
            A tuple of (extracted_text, confidence_score).
        """
        pass
