import torch
from typing import Tuple
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
from amd_ocr_challenge.app.engines.base import BaseOCREngine

class QwenVLEngine(BaseOCREngine):
    """
    OCR Engine implementation using Qwen-2.5 VL 7B.
    """
    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """
        Load Qwen-2.5 VL model weights into VRAM with 4-bit quantization.
        """
        print(f"Loading Qwen-2.5 VL model from {self.model_name_or_path}...")

        # ROCm compatible 4-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("Qwen-2.5 VL model loaded successfully.")

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Perform OCR using a generative prompt.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        # Construct the prompt for OCR
        # Qwen-2.5 VL uses specific chat templates
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from this image. Output only the extracted text, no conversational filler."},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process image and text
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        # Trim the input tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Confidence fallback as per plan
        confidence = 0.95 if output_text else 0.0

        return output_text, confidence
