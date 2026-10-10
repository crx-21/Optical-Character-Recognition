# Optical-Character-Recognition
Surveillance systems use AI to read license plates. Autonomous driving systems read speed limit postings and advisories — road work ahead, icy bridge, detour. Both are the same underlying problem: pulling reliable text out of an uncooperative image.

## What I am building
An application that accepts an image and uses a vision model or vision-language model to recognize the characters in it and report what it found.

## Model Details:
Qwen 2.5 VL trained on ROCm 10.0

## 🚀 How to Run the Model

To test the OCR model on your own images, you need to run the **Inference Server** (to load the model into GPU memory) and then use the **Client Script** to send an image for prediction.

### Prerequisites
Ensure you have the virtual environment available at:
`C:\Users\[USER_PROFILE]\Optical-Character-Recognition\amd-ocr-challenge\.venv\Scripts\python.exe`
And that all requirements needed have been installed.
---

### Step 1: Start the Inference Server
The server must be running in the background to handle requests. It loads the VLM model into your GPU VRAM.

1. Open a terminal (PowerShell).
2. Run the following commands:
   ```powershell
   cd C:\Users\[USER_PROFILE]\Optical-Character-Recognition\amd-ocr-challenge
   $env:PYTHONPATH = "C:\Users\[USER_PROFILE]\Optical-Character-Recognition\amd-ocr-challenge"
   & "C:\Users\[USER_PROFILE]\Optical-Character-Recognition\amd-ocr-challenge\.venv\Scripts\python.exe" -m app.server
   ```
3. **Wait** until you see the message: `OCR engine initialized successfully.`
4. **Keep this terminal open.** If you close it, the model is unloaded from the GPU.

---

### Step 2: Run a Prediction on an Image
Now that the server is ready, you can use `app.py` to process any image on your computer.

1. Open a **second terminal** (PowerShell).
2. Run the following command, replacing `path/to/your/image.jpg` with the actual path to your file:
   ```powershell
   cd C:\Users\[USER_PROFILE]\Optical-Character-Recognition\amd-ocr-challenge
   $env:PYTHONPATH = "C:\Users\[USER_PROFILE]\Optical-Character-Recognition\amd-ocr-challenge"
   & "C:\Users\[USER_PROFILE]\Optical-Character-Recognition\amd-ocr-challenge\.venv\Scripts\python.exe" app/app.py --input-image "C:\Users\CrX\Pictures\my_plate.jpg"
   ```

### What happens next?
- The script sends the image to the server.
- The server performs OCR using the Qwen-VL model.
- The script applies **domain normalization** (e.g., stripping "CALIFORNIA" from US plates).
- The final result is printed to the terminal and saved as a JSON file in `app/output/`.

---

### 🛠 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **`ModuleNotFoundError`** | Ensure you ran the `$env:PYTHONPATH` command in **both** terminals. |
| **`ConnectionError`** | Make sure the server in Terminal 1 is actually running and hasn't crashed. |
| **`Out of Memory (OOM)`** | Close other GPU-heavy apps and restart the server. |
| **Wrong Text Output** | Check `app/config.py` to ensure `MODEL_ADAPTER_PATH` points to the correct fine-tuned weights. |
