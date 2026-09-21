import torch

# Model and Device Configuration
MODEL_ID = "microsoft/Florence-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Server Settings
SERVER_URL = "http://127.0.0.1:8000/predict"

# Client Retry Settings
RETRY_SETTINGS = {
    "max_retries": 3,
    "retry_delay": 2,
}
