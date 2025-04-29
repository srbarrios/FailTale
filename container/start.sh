#!/bin/bash
set -e

# Use maximum available threads for Ollama
export OLLAMA_NUM_THREADS=$(nproc)

echo "[Startup] Starting Ollama service in background..."
# Start ollama serve in the background
ollama serve &
# Capture the Process ID (PID) of the background process
OLLAMA_PID=$!
echo "[Startup] Ollama service PID: $OLLAMA_PID"

# Wait for Ollama to initialize (adjust sleep time if necessary)
echo "[Startup] Waiting 5 seconds for Ollama to initialize..."
sleep 5
echo "[Startup] Ollama service appears to be running."

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
