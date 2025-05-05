#!/bin/bash
set -e

# Check if the config file exists at the expected path
CONFIG_PATH="/app/config.yaml" # Default path, adjust if needed
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[Startup] Error: Configuration file not found at $CONFIG_PATH!"
     exit 1
fi

echo "[Startup] Starting FailTale Flask server (run_server.py)..."
# Execute your Flask server script using python3
# Pass the config file path expected by your script
# Ensure run_server.py uses paths relative to /app or absolute paths
python3 run_server.py --config "$CONFIG_PATH"

echo "Flask server exited."
