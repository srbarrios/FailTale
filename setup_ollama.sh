#!/bin/bash
# This script sets up the Ollama environment by installing the necessary dependencies and pulling the required models.
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
ollama pull nomic-embed-text
