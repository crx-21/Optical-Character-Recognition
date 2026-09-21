#!/bin/bash
# Start the OCR server in the background
echo "Starting OCR server..."
nohup python3 app/server.py > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
until curl -s http://127.0.0.1:8000/docs > /dev/null; do
  sleep 2
done
echo "Server is up!"

# Process all images in the input folder
INPUT_DIR="app/input"
OUTPUT_DIR="app/output"
mkdir -p $OUTPUT_DIR

if [ -z "$(ls -A $INPUT_DIR 2>/dev/null)" ]; then
  echo "No images found in $INPUT_DIR"
else
  for img in $INPUT_DIR/*; do
    echo "Processing $img..."
    python3 app/app.py --input-image "$img"
  done
fi

# Cleanup
echo "Shutting down server..."
kill $SERVER_PID
echo "Done."
