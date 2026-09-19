# Must use the exact mandated ROCm base image
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# Set environment variables for ROCm / PyTorch execution
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code
COPY app/ /app/

# Make sure scripts and output directory exist
RUN mkdir -p /app/output /models

# Start background server daemon on container boot
CMD ["python3", "/app/server.py"]