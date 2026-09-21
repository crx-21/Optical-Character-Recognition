# CLAUDE.md - AMD AI Academy OCR Challenge

## Project Overview
This repository contains an end-to-end Optical Character Recognition (OCR) pipeline built for the **LabLab x AMD AI Academy Challenge (Mini-Challenge 2)**. The application receives an image path, extracts characters using a Vision-Language Model (VLM), and writes a JSON output file.

Target Domains: US License Plates, Chinese License Plates, Speed Limit Signs, Advisory Speed Plaques, Stop Signs, and Work Zone Signs under heavy noise, glare, blur, and off-axis skew.

---

## Architectural Principles & Daemon Design
1. **Daemon Pattern:** To avoid per-image model loading latency (30s timeout per image), `app/server.py` runs a persistent FastAPI server loading VLM weights into VRAM on boot.
2. **Harness Entrypoint:** `app/app.py` is invoked per image by the evaluation harness. It must remain lightweight, sending an HTTP request to `http://127.0.0.1:8000/predict` and saving output JSON to `/app/output/`.
3. **Domain Normalization Rules (`postprocess.py`):**
   - **US Plates:** Extract ONLY the core alphanumeric registration. Strip state names/slogans (e.g., `CALIFORNIA`, `NEW YORK`, `EXCELSIOR`).
   - **Chinese Plates:** KEEP the leading Chinese province character & letter prefix (e.g., `京A` or `沪B`).
   - **Road Signs:** Retain full wording (e.g., `SPEED LIMIT 65`, `STOP`, `ROAD WORK AHEAD`).
   - **Advisory Speed Plaques:** Digits only (e.g., `35`). Do not append units (e.g., "MPH").
   - **Multi-line Text:** Read top-to-bottom, joining lines with a single space.

---

## Hard Rules & Constraints (DO NOT VIOLATE)
- **Base Image:** MUST start from `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`.
- **Layer Identity:** NEVER use `--squash`, `buildah` flattening, or `FROM scratch` multistage copies. The harness checks lower layer hashes against the official image.
- **Image Size Limit:** Total uncompressed image size must be strictly `< 60 GiB`.
- **VRAM Constraints:** Peak GPU VRAM must stay between `1 GiB` and `48 GiB` (margin allowed: 1%).
- **Execution Paths:**
  - Mandatory script: `/app/app.py`
  - Dependencies file: `/app/requirements.txt`
  - Model weights directory: `/models`
  - Output directory: `/app/output/`
- **Output JSON Format (`/app/output/<image_name>_output.json`):**
  ```json
  {
    "text": "7ABC123",
    "confidence": 0.95
  }

No Secrets: NEVER embed API keys, cloud tokens, or credentials in the code or container.


### Development & Test Commands:
1.Run Local Daemon 
python3 app/server.py

2.Run Test Harness Execution
python3 app/app.py --input-image app/input/sample.png

3.Verify Docker Image Size & Layers 
docker build -t amd-ocr-solution .
docker history --no-trunc --format '{{.Size}}' amd-ocr-solution

4.Monitor VRAM Usage 
watch -n1 'rocm-smi --showmeminfo vram'