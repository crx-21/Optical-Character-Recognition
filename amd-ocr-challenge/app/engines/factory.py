from typing import Dict, Type
from app.engines.base import BaseOCREngine
from app.engines.qwen_vl_engine import QwenVLEngine
from app.config import MODEL_ADAPTER_PATH

class OCREngineFactory:

class OCREngineFactory:
    """Factory to create OCR engines based on configuration."""

    _engines: Dict[str, Type[BaseOCREngine]] = {
        "qwen2.5-vl": QwenVLEngine,
    }

    @classmethod
    def create(cls, model_type: str) -> BaseOCREngine:
        """
        Instantiates the specified OCR engine.

        Args:
            model_type: The key identifying the model in the _engines map.

        Returns:
            An instance of a BaseOCREngine.

        Raises:
            ValueError: If the model_type is not supported.
        """
        engine_class = cls._engines.get(model_type.lower())
        if not engine_class:
            supported = ", ".join(cls._engines.keys())
            raise ValueError(f"Unsupported model type '{model_type}'. Supported types: {supported}")

        return QwenVLEngine(adapter_path=MODEL_ADAPTER_PATH) if engine_class == QwenVLEngine else engine_class()
