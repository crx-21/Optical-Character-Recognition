import torch

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Configuration
MODEL_TYPE = "easyocr"  # Options: "easyocr", "qwen2.5-vl", "paddle", etc.

# Server Settings
SERVER_URL = "http://127.0.0.1:8000/predict"

# Client Retry Settings
RETRY_SETTINGS = {
    "max_retries": 3,
    "retry_delay": 2,
}
