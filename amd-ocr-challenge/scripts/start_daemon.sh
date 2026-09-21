#!/bin/bash
# Start the OCR server as a daemon
echo "Starting OCR server..."
nohup python3 app/server.py > server.log 2>&1 &
echo "Server is running in background. Log: server.log"
