# Base image for ROCm 10.0 on Ubuntu 26.04
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn requests pillow opencv-python-headless python-multipart tqdm scipy && \
    pip install --no-cache-dir easyocr --no-deps

# Copy application code
COPY app/ ./app/

# Setup mandatory directories
RUN mkdir -p /app/output /models

# Expose server port
EXPOSE 8000

# Start the server using uvicorn directly for better process management
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
